"""Shared sign recognition, grammar correction, and TTS for web + desktop."""
from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tensorflow.keras.models import load_model

import params as P
from utils.sequence_features import add_velocity

STATIC_MODEL_PATH = Path("models/sign_model.pkl")
STATIC_LE_PATH = Path("models/label_encoder.pkl")
DYNAMIC_MODEL_PATH = Path("models/dynamic_model.keras")
DYNAMIC_LE_PATH = Path("models/dynamic_label_encoder.pkl")

load_dotenv()

# Re-export for callers that imported names from this module
SEQUENCE_LENGTH = P.SEQUENCE_LENGTH
CONFIRM_SECONDS = P.CONFIRM_SECONDS
COOLDOWN_SECONDS = P.COOLDOWN_SECONDS
DYNAMIC_THRESHOLD = P.DYNAMIC_THRESHOLD
STATIC_THRESHOLD = P.STATIC_THRESHOLD

BROWSER_VOICE = {
    "id": "browser",
    "name": "System Voice",
    "gender": "Free · works offline",
    "provider": "browser",
}

# Premade library voice IDs — blocked on ElevenLabs free API tier (402)
_LIBRARY_VOICE_IDS = {
    "9BWtsMINqrJLrRacOk9x",
    "onwK4e9ZLuTAKqWWVeF4",
    "21m00Tcm4TlvDq8ikWAM",
    "IKne3meq5aSn9XLyUdCD",
    "JBFqnCBsd6RMkjVDRZzb",
}

_GLOSS_RE = re.compile(r"^[\s\"'`“”‘’]+|[\s\"'`“”‘’]+$")


def _env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def _format_gloss(words: list[str]) -> str:
    """HELLO THANK_YOU → Hello, Thank You"""
    parts = []
    for w in words:
        pretty = w.replace("_", " ").strip()
        pretty = " ".join(t.capitalize() for t in pretty.split())
        if pretty:
            parts.append(pretty)
    return ", ".join(parts)


def _clean_llm_sentence(text: str, fallback: str) -> str:
    text = (text or "").strip()
    if not text:
        return fallback
    # Drop common wrappers: quotes, "Sentence:", markdown fences
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^(corrected|natural|sentence|output)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = _GLOSS_RE.sub("", text).strip()
    # Keep first line only
    text = text.splitlines()[0].strip()
    return text or fallback


def correct_sentence(words: list[str]) -> dict:
    raw = " ".join(words)
    gloss = _format_gloss(words)
    key = _env("DEEPSEEK_API_KEY")
    if not key:
        return {
            "raw": raw,
            "corrected": gloss.replace(", ", " "),
            "grammar_active": False,
            "error": "DEEPSEEK_API_KEY not set in .env",
        }

    client = OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
        timeout=P.DEEPSEEK_TIMEOUT_S,
    )
    try:
        kwargs: dict = {
            "model": P.DEEPSEEK_MODEL,
            "max_tokens": P.DEEPSEEK_MAX_TOKENS,
            "temperature": P.DEEPSEEK_TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You convert ASL/sign glosses into one short, natural English "
                        "sentence for accessibility (hospitals, transit, daily talk).\n"
                        "Rules:\n"
                        "- Use ONLY the meaning of the given signs; do not invent facts.\n"
                        "- Fix grammar, articles, pronouns, and punctuation.\n"
                        "- Keep it concise (one sentence).\n"
                        "- Return ONLY the sentence — no quotes, labels, or explanation.\n"
                        "Examples:\n"
                        "Signs: Hello, Help → Hello, I need help.\n"
                        "Signs: Water, Please → Can I have some water, please?\n"
                        "Signs: Thank You → Thank you.\n"
                        "Signs: Pain, Help → I am in pain. Please help me."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Signs: {gloss}\nSentence:",
                },
            ],
        }
        # V4 Flash enables thinking by default — disable for fast grammar output
        if getattr(P, "DEEPSEEK_DISABLE_THINKING", True):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**kwargs)
        corrected = _clean_llm_sentence(
            response.choices[0].message.content or "",
            gloss.replace(", ", " "),
        )
        return {"raw": raw, "corrected": corrected, "grammar_active": True}
    except Exception as exc:
        return {
            "raw": raw,
            "corrected": gloss.replace(", ", " "),
            "grammar_active": False,
            "error": str(exc),
        }


def get_recognition_params() -> dict:
    """Public snapshot of tunable parameters (for UI / debugging)."""
    return {
        "sequence_length": P.SEQUENCE_LENGTH,
        "dynamic_feature_dim": getattr(P, "DYNAMIC_FEATURE_DIM", 126),
        "confirm_seconds": P.CONFIRM_SECONDS,
        "cooldown_seconds": P.COOLDOWN_SECONDS,
        "gap_tolerance_seconds": P.GAP_TOLERANCE_SECONDS,
        "static_threshold": P.STATIC_THRESHOLD,
        "dynamic_threshold": P.DYNAMIC_THRESHOLD,
        "auto_motion_margin": P.AUTO_MOTION_MARGIN,
        "ema_alpha": P.EMA_ALPHA,
        "ema_lock_threshold": P.EMA_LOCK_THRESHOLD,
        "ema_switch_margin": P.EMA_SWITCH_MARGIN,
        "smooth_window": P.SMOOTH_WINDOW,
        "smooth_min_votes": P.SMOOTH_MIN_VOTES,
        "stable_frames": P.STABLE_FRAMES,
        "landmark_hold_seconds": P.LANDMARK_HOLD_SECONDS,
        "mediapipe": {
            "max_hands": P.MP_MAX_HANDS,
            "detection_confidence": P.MP_DET_CONF,
            "tracking_confidence": P.MP_TRACK_CONF,
            "presence_confidence": P.MP_PRESENCE_CONF,
        },
        "deepseek": {
            "model": P.DEEPSEEK_MODEL,
            "max_tokens": P.DEEPSEEK_MAX_TOKENS,
            "temperature": P.DEEPSEEK_TEMPERATURE,
            "timeout_s": P.DEEPSEEK_TIMEOUT_S,
            "api_key_configured": bool(_env("DEEPSEEK_API_KEY")),
        },
    }


def _is_paid_plan_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "402" in text
        or "paid_plan_required" in text
        or "payment_required" in text
        or "cannot use library voices" in text
    )


def _resolve_elevenlabs_voice() -> dict | None:
    """One ElevenLabs voice that works on free tier (account/cloned, not library)."""
    key = _env("ELEVENLABS_API_KEY")
    if not key:
        return None

    custom_id = _env("ELEVENLABS_VOICE_ID")
    if custom_id:
        voice_name = _env("ELEVENLABS_VOICE_NAME") or "ElevenLabs"
        return {
            "id": custom_id,
            "name": voice_name,
            "gender": "ElevenLabs · free tier",
            "provider": "elevenlabs",
        }

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=key)
        for voice in client.voices.get_all().voices:
            if voice.voice_id in _LIBRARY_VOICE_IDS:
                continue
            labels = voice.labels or {}
            gender = (
                labels.get("gender") or labels.get("accent") or "AI voice"
                if isinstance(labels, dict)
                else "AI voice"
            )
            return {
                "id": voice.voice_id,
                "name": voice.name,
                "gender": str(gender),
                "provider": "elevenlabs",
            }
    except Exception:
        pass

    return None


def get_available_voices() -> list[dict]:
    """Exactly two options: system TTS + one ElevenLabs voice."""
    voices: list[dict] = [dict(BROWSER_VOICE)]
    el = _resolve_elevenlabs_voice()
    if el:
        voices.append(el)
    return voices


def synthesize_speech(text: str, voice_id: str) -> bytes:
    if voice_id == "browser":
        raise ValueError("browser")

    key = _env("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )
    return b"".join(audio)


def speak_text(text: str, voice_id: str) -> dict:
    """Return ElevenLabs audio or signal browser fallback for free-tier limits."""
    if voice_id == "browser":
        return {
            "provider": "browser",
            "text": text,
            "warning": None,
        }

    try:
        audio = synthesize_speech(text, voice_id)
        return {
            "provider": "elevenlabs",
            "audio_base64": __import__("base64").b64encode(audio).decode("ascii"),
            "mime": "audio/mpeg",
            "warning": None,
        }
    except Exception as exc:
        if _is_paid_plan_error(exc):
            return {
                "provider": "browser",
                "text": text,
                "warning": (
                    "ElevenLabs free plan cannot use premade library voices via API. "
                    "Using your browser voice instead. Add ELEVENLABS_VOICE_ID in .env "
                    "with a voice from your ElevenLabs account, or upgrade your plan."
                ),
            }
        raise


class SignEngine:
    def __init__(self) -> None:
        self.static_model = None
        self.static_le = None
        self.dynamic_model = None
        self.dynamic_le = None
        self.dynamic_feat_dim = 63
        self._model_mtimes: dict[str, float] = {}
        self.mode = "STATIC"
        self._load_models(force=True)
        self.reset_session()

    def _file_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _load_models(self, force: bool = False) -> bool:
        """Load (or reload) models from disk. Returns True if anything changed."""
        static_m = self._file_mtime(STATIC_MODEL_PATH)
        le_m = self._file_mtime(STATIC_LE_PATH)
        dyn_m = self._file_mtime(DYNAMIC_MODEL_PATH)
        dyn_le_m = self._file_mtime(DYNAMIC_LE_PATH)
        stamp = {
            "static": static_m,
            "static_le": le_m,
            "dynamic": dyn_m,
            "dynamic_le": dyn_le_m,
        }
        if not force and stamp == self._model_mtimes:
            return False

        self.static_model = joblib.load(STATIC_MODEL_PATH)
        self.static_le = joblib.load(STATIC_LE_PATH)
        n_static = len(self.static_le.classes_)
        print(f"📦 Static model loaded: {n_static} signs → {list(self.static_le.classes_)}")

        self.dynamic_model = None
        self.dynamic_le = None
        self.dynamic_feat_dim = 63
        try:
            if DYNAMIC_MODEL_PATH.is_file():
                self.dynamic_model = load_model(DYNAMIC_MODEL_PATH)
                self.dynamic_le = joblib.load(DYNAMIC_LE_PATH)
                self.dynamic_feat_dim = self._read_dynamic_feat_dim()
                print(
                    f"📦 Dynamic LSTM: (T, {self.dynamic_feat_dim}) "
                    f"signs={list(self.dynamic_le.classes_)}"
                )
        except Exception as e:
            print(f"⚠️  Dynamic model not loaded ({e}). STATIC mode still works.")

        self._model_mtimes = stamp
        return True

    def maybe_reload_models(self) -> bool:
        """Hot-reload if train_model.py wrote newer pickles (fixes stale API vs main.py)."""
        changed = self._load_models(force=False)
        if changed:
            # Keep mode; clear buffers so old EMA labels don't stick
            keep = self.mode
            self.reset_session()
            self.mode = keep
            print(f"🔄 Models reloaded from disk (mode={self.mode})")
        return changed

    def _read_dynamic_feat_dim(self) -> int:
        """Match live features to whatever the saved Keras model was trained with."""
        try:
            shape = self.dynamic_model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            dim = int(shape[-1])
            if dim in (63, 126):
                return dim
        except Exception:
            pass
        return int(getattr(P, "DYNAMIC_FEATURE_DIM", 126))

    def reset_session(self) -> None:
        keep_mode = getattr(self, "mode", "STATIC")
        self.sequence: deque = deque(maxlen=P.SEQUENCE_LENGTH)
        self.sentence: list[str] = []
        self.buffer_label: str | None = None
        self.buffer_start: float | None = None
        self.last_spoken_time = 0.0
        self.last_seen_time = 0.0
        self.mode = keep_mode
        self.ema_scores: dict[str, float] = {}
        self.locked_label: str | None = None
        self.stable_count = 0
        self.stable_label: str | None = None
        self.last_features: list[float] | None = None
        self.last_features_time = 0.0

    def vocabulary(self) -> dict:
        static = (
            [str(c) for c in self.static_le.classes_]
            if self.static_le is not None
            else []
        )
        dynamic = (
            [str(c) for c in self.dynamic_le.classes_]
            if self.dynamic_le is not None
            else []
        )
        return {
            "static_signs": static,
            "dynamic_signs": dynamic,
            "all_signs": sorted(set(static) | set(dynamic)),
        }

    def _pick_auto(
        self,
        static_pred: str,
        static_conf: float,
        dynamic_pred: str | None,
        dynamic_conf: float,
    ) -> tuple[str | None, float, str]:
        static_ok = static_conf >= P.STATIC_THRESHOLD
        dynamic_ok = bool(dynamic_pred) and dynamic_conf >= P.DYNAMIC_THRESHOLD

        # Dynamic LSTM only knows a small motion set. If static predicts a sign
        # outside that set (STOP, EAT, …), never let motion override it.
        dyn_vocab = (
            {str(c) for c in self.dynamic_le.classes_}
            if self.dynamic_le is not None
            else set()
        )
        static_outside_motion = static_ok and static_pred not in dyn_vocab
        if static_outside_motion:
            return static_pred, static_conf, "STATIC"

        if dynamic_ok and static_ok:
            if dynamic_conf >= static_conf + P.AUTO_MOTION_MARGIN:
                return dynamic_pred, dynamic_conf, "MOTION"
            if static_conf >= dynamic_conf + P.AUTO_MOTION_MARGIN:
                return static_pred, static_conf, "STATIC"
            if dynamic_conf >= static_conf:
                return dynamic_pred, dynamic_conf, "MOTION"
            return static_pred, static_conf, "STATIC"

        if dynamic_ok:
            return dynamic_pred, dynamic_conf, "MOTION"
        if static_ok:
            return static_pred, static_conf, "STATIC"
        return None, 0.0, ""

    def _update_ema(self, label: str | None, conf: float) -> tuple[str | None, float]:
        """Confidence-weighted EMA. Prevents majority-vote race with hold timer."""
        alpha = P.EMA_ALPHA
        # Decay all tracked labels each frame
        for k in list(self.ema_scores.keys()):
            self.ema_scores[k] *= 1.0 - alpha
            if self.ema_scores[k] < 0.02:
                del self.ema_scores[k]

        if label is not None and conf > 0:
            prev = self.ema_scores.get(label, 0.0)
            self.ema_scores[label] = alpha * conf + (1.0 - alpha) * prev

        if not self.ema_scores:
            self.locked_label = None
            return None, 0.0

        best_label, best_score = max(self.ema_scores.items(), key=lambda kv: kv[1])
        if best_score < P.EMA_LOCK_THRESHOLD:
            self.locked_label = None
            return None, best_score

        # Sticky lock: only switch if challenger clearly beats current
        if self.locked_label and self.locked_label != best_label:
            current = self.ema_scores.get(self.locked_label, 0.0)
            if best_score < current + P.EMA_SWITCH_MARGIN:
                return self.locked_label, current

        self.locked_label = best_label
        return best_label, best_score

    def _confirm_logic(
        self, final_pred: str | None, now: float
    ) -> tuple[float, str | None]:
        hold_progress = 0.0
        confirmed = None

        if final_pred:
            self.last_seen_time = now
            if final_pred == self.stable_label:
                self.stable_count += 1
            else:
                self.stable_label = final_pred
                self.stable_count = 1

            if self.stable_count >= P.STABLE_FRAMES:
                if final_pred == self.buffer_label and self.buffer_start is not None:
                    held = now - self.buffer_start
                    hold_progress = min(held / P.CONFIRM_SECONDS, 1.0)
                    if (
                        held >= P.CONFIRM_SECONDS
                        and (now - self.last_spoken_time) > P.COOLDOWN_SECONDS
                    ):
                        self.sentence.append(final_pred)
                        confirmed = final_pred
                        self.last_spoken_time = now
                        self.buffer_label = None
                        self.buffer_start = None
                        self.stable_count = 0
                        self.stable_label = None
                        self.ema_scores.clear()
                        self.locked_label = None
                elif self.buffer_label != final_pred:
                    # Only reset hold when EMA stickiness already accepted a new label
                    self.buffer_label = final_pred
                    self.buffer_start = now
        else:
            if (
                self.buffer_label is not None
                and self.last_seen_time
                and (now - self.last_seen_time) <= P.GAP_TOLERANCE_SECONDS
            ):
                if self.buffer_start is not None:
                    held = now - self.buffer_start
                    hold_progress = min(held / P.CONFIRM_SECONDS, 1.0)
            else:
                self.buffer_label = None
                self.buffer_start = None
                self.stable_count = 0
                self.stable_label = None

        return hold_progress, confirmed

    def process_gap(self, now: float) -> dict:
        """No fresh landmarks — optionally reuse last vector, else soft-hold only."""
        if (
            self.last_features is not None
            and (now - self.last_features_time) <= P.LANDMARK_HOLD_SECONDS
        ):
            return self.process_frame(self.last_features, now, held_frame=True)

        hold_progress, confirmed = self._confirm_logic(None, now)
        return {
            "static_pred": None,
            "static_conf": 0.0,
            "dynamic_pred": None,
            "dynamic_conf": 0.0,
            "final_pred": self.buffer_label if hold_progress > 0 else None,
            "final_conf": 0.0,
            "final_source": "",
            "hold_progress": hold_progress,
            "confirmed": confirmed,
            "sentence": self.sentence.copy(),
            "sequence_len": len(self.sequence),
            "sequence_max": P.SEQUENCE_LENGTH,
            "mode": self.mode,
            "stable_count": self.stable_count,
            "stable_needed": P.STABLE_FRAMES,
            "hand_held": False,
        }

    def process_frame(
        self,
        features: list[float],
        now: float,
        held_frame: bool = False,
    ) -> dict:
        self.maybe_reload_models()

        if not held_frame:
            self.last_features = list(features)
            self.last_features_time = now

        static_proba = self.static_model.predict_proba([features])[0]
        static_conf = float(static_proba.max())
        static_pred = str(self.static_le.classes_[static_proba.argmax()])

        dynamic_pred = None
        dynamic_conf = 0.0
        # Buffer stores pose-only (63). Older LSTMs expect 63; newer expect 126 (+velocity).
        pose = list(features[:63]) if len(features) >= 63 else list(features)
        self.sequence.append(pose)

        # Skip LSTM work in STATIC mode (faster) or when no dynamic model is loaded
        run_dynamic = (
            self.dynamic_model is not None
            and self.dynamic_le is not None
            and self.mode in ("AUTO", "DYNAMIC")
        )
        if run_dynamic and len(self.sequence) == P.SEQUENCE_LENGTH:
            pose_seq = np.array(self.sequence, dtype=np.float32)  # (T, 63)
            if self.dynamic_feat_dim >= 126:
                feat_seq = add_velocity(pose_seq)  # (T, 126)
            else:
                feat_seq = pose_seq  # legacy model
            seq_array = feat_seq[np.newaxis, ...]
            dyn_proba = self.dynamic_model.predict(seq_array, verbose=0)[0]
            dynamic_conf = float(dyn_proba.max())
            dynamic_pred = str(self.dynamic_le.classes_[dyn_proba.argmax()])

        # Slightly discount confidence when reusing a held landmark frame
        conf_scale = 0.85 if held_frame else 1.0
        static_conf_use = static_conf * conf_scale
        dynamic_conf_use = dynamic_conf * conf_scale

        raw_pred: str | None = None
        raw_conf = 0.0
        raw_source = ""
        mode = self.mode

        if mode == "AUTO":
            if self.dynamic_model is None:
                if static_conf_use >= P.STATIC_THRESHOLD:
                    raw_pred, raw_conf, raw_source = static_pred, static_conf_use, "STATIC"
            else:
                raw_pred, raw_conf, raw_source = self._pick_auto(
                    static_pred, static_conf_use, dynamic_pred, dynamic_conf_use
                )
        elif mode == "STATIC" and static_conf_use >= P.STATIC_THRESHOLD:
            raw_pred, raw_conf, raw_source = static_pred, static_conf_use, "STATIC"
        elif (
            mode == "DYNAMIC"
            and dynamic_pred
            and dynamic_conf_use >= P.DYNAMIC_THRESHOLD
        ):
            raw_pred, raw_conf, raw_source = dynamic_pred, dynamic_conf_use, "MOTION"

        final_pred, ema_score = self._update_ema(raw_pred, raw_conf)
        final_conf = float(ema_score) if final_pred else 0.0
        final_source = ""
        if final_pred:
            if final_pred == dynamic_pred:
                final_source = "MOTION"
                final_conf = max(final_conf, dynamic_conf_use)
            elif final_pred == static_pred:
                final_source = "STATIC"
                final_conf = max(final_conf, static_conf_use)
            else:
                final_source = raw_source or "STATIC"

        hold_progress, confirmed = self._confirm_logic(final_pred, now)

        return {
            "static_pred": static_pred,
            "static_conf": static_conf,
            "dynamic_pred": dynamic_pred,
            "dynamic_conf": dynamic_conf,
            "final_pred": final_pred,
            "final_conf": final_conf,
            "final_source": final_source,
            "hold_progress": hold_progress,
            "confirmed": confirmed,
            "sentence": self.sentence.copy(),
            "sequence_len": len(self.sequence),
            "sequence_max": P.SEQUENCE_LENGTH,
            "mode": self.mode,
            "stable_count": self.stable_count,
            "stable_needed": P.STABLE_FRAMES,
            "ema_score": final_conf,
            "hand_held": held_frame,
        }

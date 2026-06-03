"""Shared sign recognition, grammar correction, and TTS for web + desktop."""
from __future__ import annotations

import io
import os
from collections import deque

import joblib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tensorflow.keras.models import load_model

load_dotenv()

SEQUENCE_LENGTH = 40
CONFIRM_SECONDS = 1.0
COOLDOWN_SECONDS = 2.0
DYNAMIC_THRESHOLD = 0.65
STATIC_THRESHOLD = 0.60

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


def _env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    return value.strip().strip('"').strip("'")


def correct_sentence(words: list[str]) -> dict:
    raw = " ".join(words)
    key = _env("DEEPSEEK_API_KEY")
    if not key:
        return {"raw": raw, "corrected": raw, "grammar_active": False}

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help mute people communicate. Convert sign language word "
                        "sequences into natural grammatically correct sentences. "
                        "Return ONLY the sentence, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Signs detected: {raw}\n\nNatural sentence:",
                },
            ],
        )
        corrected = (response.choices[0].message.content or "").strip()
        if not corrected:
            corrected = raw
        return {"raw": raw, "corrected": corrected, "grammar_active": True}
    except Exception as exc:
        return {"raw": raw, "corrected": raw, "grammar_active": False, "error": str(exc)}


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
        self.static_model = joblib.load("models/sign_model.pkl")
        self.static_le = joblib.load("models/label_encoder.pkl")
        self.dynamic_model = load_model("models/dynamic_model.keras")
        self.dynamic_le = joblib.load("models/dynamic_label_encoder.pkl")
        self.reset_session()

    def reset_session(self) -> None:
        self.sequence: deque = deque(maxlen=SEQUENCE_LENGTH)
        self.sentence: list[str] = []
        self.buffer_label: str | None = None
        self.buffer_start: float | None = None
        self.last_spoken_time = 0.0
        self.mode = "AUTO"

    def process_frame(self, features: list[float], now: float) -> dict:
        static_proba = self.static_model.predict_proba([features])[0]
        static_conf = float(static_proba.max())
        static_pred = str(self.static_le.classes_[static_proba.argmax()])

        dynamic_pred = None
        dynamic_conf = 0.0
        self.sequence.append(features)
        if len(self.sequence) == SEQUENCE_LENGTH:
            seq_array = np.array(self.sequence)[np.newaxis, ...]
            dyn_proba = self.dynamic_model.predict(seq_array, verbose=0)[0]
            dynamic_conf = float(dyn_proba.max())
            dynamic_pred = str(self.dynamic_le.classes_[dyn_proba.argmax()])

        final_pred, final_conf, final_source = None, 0.0, ""
        mode = self.mode

        if mode == "AUTO":
            if dynamic_pred and dynamic_conf >= DYNAMIC_THRESHOLD:
                if static_conf >= STATIC_THRESHOLD:
                    if dynamic_conf >= static_conf:
                        final_pred, final_conf, final_source = (
                            dynamic_pred,
                            dynamic_conf,
                            "MOTION",
                        )
                    else:
                        final_pred, final_conf, final_source = (
                            static_pred,
                            static_conf,
                            "STATIC",
                        )
                else:
                    final_pred, final_conf, final_source = (
                        dynamic_pred,
                        dynamic_conf,
                        "MOTION",
                    )
            elif static_conf >= STATIC_THRESHOLD:
                final_pred, final_conf, final_source = static_pred, static_conf, "STATIC"
        elif mode == "STATIC" and static_conf >= STATIC_THRESHOLD:
            final_pred, final_conf, final_source = static_pred, static_conf, "STATIC"
        elif mode == "DYNAMIC" and dynamic_pred and dynamic_conf >= DYNAMIC_THRESHOLD:
            final_pred, final_conf, final_source = dynamic_pred, dynamic_conf, "MOTION"

        hold_progress = 0.0
        confirmed = None

        if final_pred:
            if final_pred == self.buffer_label and self.buffer_start is not None:
                held = now - self.buffer_start
                hold_progress = min(held / CONFIRM_SECONDS, 1.0)
                if held >= CONFIRM_SECONDS and (now - self.last_spoken_time) > COOLDOWN_SECONDS:
                    self.sentence.append(final_pred)
                    confirmed = final_pred
                    self.last_spoken_time = now
                    self.buffer_label = None
                    self.buffer_start = None
            else:
                self.buffer_label = final_pred
                self.buffer_start = now
        else:
            self.buffer_label = None
            self.buffer_start = None

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
            "sequence_max": SEQUENCE_LENGTH,
            "mode": self.mode,
        }

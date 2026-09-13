"""Desktop OpenCV runner — uses shared SignEngine + params (same as web/mobile API)."""
import io
import os
import threading
import time

import cv2
import mediapipe as mp
import pygame
import pyttsx3
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

import params as P
from api.inference import SignEngine, correct_sentence
from utils.landmarks import extract_landmarks

load_dotenv()


def _env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    return value.strip().strip('"').strip("'")


print("Loading models...")
engine = SignEngine()
print("✅ Both models loaded\n")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=P.MP_MAX_HANDS,
    min_detection_confidence=P.MP_DET_CONF,
    min_tracking_confidence=P.MP_TRACK_CONF,
)

ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY")
USE_ELEVENLABS = bool(ELEVENLABS_API_KEY)
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if USE_ELEVENLABS else None
tts_offline = pyttsx3.init()
tts_offline.setProperty("rate", 150)
tts_offline.setProperty("volume", 1.0)
pygame.mixer.init()


def speak(text: str) -> None:
    def _speak():
        if USE_ELEVENLABS and el_client:
            try:
                audio = el_client.text_to_speech.convert(
                    text=text,
                    voice_id=_env("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_flash_v2_5",
                    output_format="mp3_44100_128",
                )
                audio_bytes = b"".join(audio)
                pygame.mixer.music.load(io.BytesIO(audio_bytes))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
                return
            except Exception as e:
                print(f"⚠️  ElevenLabs failed ({e}), using offline TTS")
        tts_offline.say(text)
        tts_offline.runAndWait()

    threading.Thread(target=_speak, daemon=True).start()


def draw_panel(img, x1, y1, x2, y2, color=(20, 20, 20), alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def conf_bar(frame, conf, x, y, w=180, h=10):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    color = (0, 255, 100) if conf > 0.85 else (0, 165, 255) if conf > 0.6 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + int(conf * w), y + h), color, -1)
    cv2.putText(
        frame,
        f"{conf * 100:.0f}%",
        (x + w + 6, y + 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
    )


last_spoken = None
last_spoken_flash = 0.0
last_corrected = ""

from utils.camera import open_camera

cap, cam_idx = open_camera(0)
if cap is None:
    raise SystemExit(
        "❌ Webcam failed. Try: python test_camera.py --probe\n"
        "   then: python test_camera.py --index 1"
    )

# Default to STATIC so freshly trained landmark models are easy to test
engine.mode = "STATIC"

print("🟢 Running!")
print(f"   camera index={cam_idx}  mode={engine.mode}")
print(
    f"   thresholds: static>={P.STATIC_THRESHOLD} dynamic>={P.DYNAMIC_THRESHOLD} "
    f"confirm={P.CONFIRM_SECONDS}s"
)
print("   SPACE = speak sentence | C = clear | M = toggle mode | Q = quit\n")
print("   Tip: press M for AUTO/DYNAMIC after you retrain the LSTM (train_dynamic.py)\n")

cv2.namedWindow("Sign to Speech", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Lost camera frames.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    now = time.time()

    final_pred = None
    final_conf = 0.0
    final_source = ""
    hold_progress = 0.0
    out = {
        "sentence": engine.sentence,
        "sequence_len": len(engine.sequence),
        "mode": engine.mode,
    }

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        features = extract_landmarks(hand_lm)
        out = engine.process_frame(features, now)
        final_pred = out.get("final_pred")
        final_conf = float(out.get("final_conf") or 0)
        final_source = out.get("final_source") or ""
        hold_progress = float(out.get("hold_progress") or 0)

        if out.get("confirmed"):
            last_spoken = out["confirmed"]
            last_spoken_flash = now
            speak(out["confirmed"])
            print(f"✅ [{final_source}] {out['confirmed']}")

        if hold_progress > 0:
            cx, cy = frame.shape[1] - 55, 55
            cv2.ellipse(
                frame,
                (cx, cy),
                (38, 38),
                -90,
                0,
                int(360 * hold_progress),
                (0, 255, 100),
                5,
            )
    else:
        cv2.putText(
            frame,
            "Show your hand...",
            (170, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 80),
            2,
        )

    draw_panel(frame, 0, 0, 640, 72)
    if final_pred:
        color = (0, 255, 180) if final_source == "MOTION" else (255, 220, 0)
        cv2.putText(frame, final_pred, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        conf_bar(frame, final_conf, 12, 56, w=200)
        tag_color = (0, 200, 140) if final_source == "MOTION" else (180, 160, 0)
        cv2.putText(
            frame,
            final_source,
            (225, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tag_color,
            2,
        )

    cv2.putText(
        frame,
        f"MODE: {engine.mode}",
        (490, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 200, 255),
        1,
    )

    if last_spoken and (now - last_spoken_flash) < 2.0:
        cv2.putText(
            frame,
            f"Spoke: {last_spoken}",
            (12, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )

    if last_corrected:
        draw_panel(frame, 0, 370, 640, 408, color=(0, 40, 0))
        display = last_corrected if len(last_corrected) < 50 else last_corrected[:47] + "..."
        cv2.putText(
            frame,
            f"AI: {display}",
            (12, 395),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 100),
            2,
        )

    seq_len = int(out.get("sequence_len") or 0)
    buf_fill = seq_len / P.SEQUENCE_LENGTH
    cv2.rectangle(frame, (0, 408), (int(640 * buf_fill), 412), (80, 80, 200), -1)
    cv2.putText(
        frame,
        f"Motion buffer: {seq_len}/{P.SEQUENCE_LENGTH}",
        (440, 405),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (120, 120, 200),
        1,
    )

    draw_panel(frame, 0, 412, 640, 480)
    sentence = out.get("sentence") or engine.sentence
    sentence_text = " ".join(sentence) if sentence else "—"
    if len(sentence_text) > 42:
        sentence_text = "..." + sentence_text[-39:]
    cv2.putText(
        frame,
        sentence_text,
        (12, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "SPACE=speak  C=clear  M=mode  Q=quit",
        (340, 472),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 100, 100),
        1,
    )

    cv2.imshow("Sign to Speech", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        keep_mode = engine.mode
        engine.reset_session()
        engine.mode = keep_mode
        print("🗑️  Cleared")
    elif key == ord(" ") and engine.sentence:
        words = engine.sentence.copy()
        print(f"📝 Raw signs: {' '.join(words)}")

        def correct_and_speak():
            global last_corrected
            result = correct_sentence(words)
            corrected = result.get("corrected") or " ".join(words)
            last_corrected = corrected
            if result.get("grammar_active"):
                print(f"✅ Corrected: {corrected}")
            else:
                print(f"⚠️  Grammar offline ({result.get('error')}) → {corrected}")
            speak(corrected)

        threading.Thread(target=correct_and_speak, daemon=True).start()
    elif key == ord("m"):
        modes = ["AUTO", "STATIC", "DYNAMIC"]
        engine.mode = modes[(modes.index(engine.mode) + 1) % 3]
        print(f"🔁 Mode: {engine.mode}")

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")

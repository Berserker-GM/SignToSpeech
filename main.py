import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import threading
from tensorflow.keras.models import load_model
from collections import deque
import pygame
import io
from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
from openai import OpenAI

# Load .env before reading any API keys
load_dotenv()

def _env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        return None
    return value.strip().strip('"').strip("'")


# ── AI Sentence Corrector ─────────────────────────────────────────
_deepseek_key = _env("DEEPSEEK_API_KEY")
deepseek_client = (
    OpenAI(api_key=_deepseek_key, base_url="https://api.deepseek.com")
    if _deepseek_key
    else None
)

def correct_sentence(words: list) -> str:
    raw = " ".join(words)
    if not deepseek_client:
        print("⚠️  DEEPSEEK_API_KEY not set — using raw signs")
        return raw
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": "You help mute people communicate. Convert sign language word sequences into natural grammatically correct sentences. Return ONLY the sentence, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Signs detected: {raw}\n\nNatural sentence:"
                }
            ]
        )
        corrected = (response.choices[0].message.content or "").strip()
        if not corrected:
            print("⚠️  DeepSeek returned empty text — using raw signs")
            return raw
        return corrected
    except Exception as e:
        print(f"⚠️  DeepSeek failed ({e}), using raw words")
        return raw

# ── Load models ───────────────────────────────────────────────────
print("Loading models...")
static_model  = joblib.load("models/sign_model.pkl")
static_le     = joblib.load("models/label_encoder.pkl")
dynamic_model = load_model("models/dynamic_model.keras")
dynamic_le    = joblib.load("models/dynamic_label_encoder.pkl")
print("✅ Both models loaded\n")

# ── MediaPipe ─────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── TTS ───────────────────────────────────────────────────────────


ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY")
USE_ELEVENLABS     = bool(ELEVENLABS_API_KEY)

el_client   = ElevenLabs(api_key=ELEVENLABS_API_KEY) if USE_ELEVENLABS else None
tts_offline = pyttsx3.init()
tts_offline.setProperty("rate", 150)
tts_offline.setProperty("volume", 1.0)
pygame.mixer.init()

def speak(text):
    def _speak():
        if USE_ELEVENLABS:
            try:
                audio = el_client.text_to_speech.convert(
                    text=text,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",  # George
                    model_id="eleven_flash_v2_5",
                    output_format="mp3_44100_128"
                )
                audio_bytes = b"".join(audio)
                pygame.mixer.music.load(io.BytesIO(audio_bytes))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
            except Exception as e:
                print(f"⚠️  ElevenLabs failed ({e}), using offline TTS")
                tts_offline.say(text)
                tts_offline.runAndWait()
        else:
            tts_offline.say(text)
            tts_offline.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── Landmark extraction ───────────────────────────────────────────
from utils.landmarks import extract_landmarks

# ── State ─────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 40      # must match training
CONFIRM_SECONDS  = 1.0     # hold time for static signs
COOLDOWN_SECONDS = 2.0     # gap between confirmations
DYNAMIC_THRESHOLD = 0.65   # min confidence for dynamic prediction
STATIC_THRESHOLD  = 0.60

sequence         = deque(maxlen=SEQUENCE_LENGTH)  # rolling frame buffer
sentence         = []
buffer_label     = None
buffer_start     = None
last_spoken_time = 0
last_spoken      = None
mode             = "AUTO"  # AUTO / STATIC / DYNAMIC
last_corrected   = ""
# ── UI helpers ────────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, color=(20,20,20), alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def conf_bar(frame, conf, x, y, w=180, h=10):
    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,50), -1)
    color = (0,255,100) if conf > 0.85 else (0,165,255) if conf > 0.6 else (0,0,255)
    cv2.rectangle(frame, (x,y), (x+int(conf*w), y+h), color, -1)
    cv2.putText(frame, f"{conf*100:.0f}%", (x+w+6, y+9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

# ── Main loop ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🟢 Running!")
print("   SPACE = speak sentence | C = clear | M = toggle mode | Q = quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    now    = time.time()

    static_pred   = None
    static_conf   = 0.0
    dynamic_pred  = None
    dynamic_conf  = 0.0
    final_pred    = None
    final_conf    = 0.0
    final_source  = ""

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        features = extract_landmarks(hand_lm)

        # ── Static prediction (per frame) ────────────────────────
        proba       = static_model.predict_proba([features])[0]
        static_conf = proba.max()
        static_pred = static_le.classes_[proba.argmax()]

        # ── Dynamic prediction (rolling window) ──────────────────
        sequence.append(features)
        if len(sequence) == SEQUENCE_LENGTH:
            seq_array    = np.array(sequence)[np.newaxis, ...]  # (1, 40, 63)
            dyn_proba    = dynamic_model.predict(seq_array, verbose=0)[0]
            dynamic_conf = dyn_proba.max()
            dynamic_pred = dynamic_le.classes_[dyn_proba.argmax()]

        # ── AUTO mode: pick whichever is more confident ───────────
        if mode == "AUTO":
            if dynamic_pred and dynamic_conf >= DYNAMIC_THRESHOLD:
                if static_conf >= STATIC_THRESHOLD:
                    # both confident — pick higher
                    if dynamic_conf >= static_conf:
                        final_pred, final_conf, final_source = dynamic_pred, dynamic_conf, "MOTION"
                    else:
                        final_pred, final_conf, final_source = static_pred, static_conf, "STATIC"
                else:
                    final_pred, final_conf, final_source = dynamic_pred, dynamic_conf, "MOTION"
            elif static_conf >= STATIC_THRESHOLD:
                final_pred, final_conf, final_source = static_pred, static_conf, "STATIC"
        elif mode == "STATIC" and static_conf >= STATIC_THRESHOLD:
            final_pred, final_conf, final_source = static_pred, static_conf, "STATIC"
        elif mode == "DYNAMIC" and dynamic_pred and dynamic_conf >= DYNAMIC_THRESHOLD:
            final_pred, final_conf, final_source = dynamic_pred, dynamic_conf, "MOTION"

        # ── Confirm logic: hold same sign for CONFIRM_SECONDS ─────
        if final_pred:
            if final_pred == buffer_label:
                held = now - buffer_start
                # Draw hold arc
                progress = min(held / CONFIRM_SECONDS, 1.0)
                cx, cy   = frame.shape[1] - 55, 55
                cv2.ellipse(frame, (cx,cy), (38,38), -90, 0,
                            int(360*progress), (0,255,100), 5)

                if held >= CONFIRM_SECONDS:
                    if (now - last_spoken_time) > COOLDOWN_SECONDS:
                        sentence.append(final_pred)
                        speak(final_pred)
                        last_spoken      = final_pred
                        last_spoken_time = now
                        print(f"✅ [{final_source}] {final_pred}")
                    buffer_label = None
                    buffer_start = None
            else:
                buffer_label = final_pred
                buffer_start = now
    else:
        buffer_label = None
        buffer_start = None
        pass  # keep buffer — hand may reappear mid-motion
        cv2.putText(frame, "Show your hand...", (170, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,80), 2)

    # ── UI ────────────────────────────────────────────────────────

    # Top bar
    draw_panel(frame, 0, 0, 640, 72)
    if final_pred:
        color = (0,255,180) if final_source == "MOTION" else (255,220,0)
        cv2.putText(frame, final_pred, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        conf_bar(frame, final_conf, 12, 56, w=200)
        # Source tag
        tag_color = (0,200,140) if final_source == "MOTION" else (180,160,0)
        cv2.putText(frame, final_source, (225, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_color, 2)

    # Mode indicator
    mode_color = (100,200,255)
    cv2.putText(frame, f"MODE: {mode}", (490, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)

    # Last spoken flash
    if last_spoken and (now - last_spoken_time) < 2.0:
        cv2.putText(frame, f"Spoke: {last_spoken}", (12, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
    
    if last_corrected:
        draw_panel(frame, 0, 370, 640, 408, color=(0, 40, 0))
        display = last_corrected if len(last_corrected) < 50 else last_corrected[:47] + "..."
        cv2.putText(frame, f"AI: {display}", (12, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)    

    # Dynamic buffer fill bar
    buf_fill = len(sequence) / SEQUENCE_LENGTH
    cv2.rectangle(frame, (0, 408), (int(640 * buf_fill), 412), (80, 80, 200), -1)
    cv2.putText(frame, f"Motion buffer: {len(sequence)}/{SEQUENCE_LENGTH}",
                (440, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 200), 1)
    
    # Bottom sentence bar
    draw_panel(frame, 0, 412, 640, 480)
    sentence_text = " ".join(sentence) if sentence else "—"
    if len(sentence_text) > 42:
        sentence_text = "..." + sentence_text[-39:]
    cv2.putText(frame, sentence_text, (12, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    cv2.putText(frame, "SPACE=speak  C=clear  M=mode  Q=quit",
                (340, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
    

    cv2.imshow("Sign to Speech", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence.clear()
        print("🗑️  Cleared")
    elif key == ord(' ') and sentence:
        raw = " ".join(sentence)
        print(f"📝 Raw signs: {raw}")
        def correct_and_speak():
            global last_corrected
            corrected = correct_sentence(sentence.copy())
            last_corrected = corrected
            print(f"✅ Corrected: {corrected}")
            speak(corrected)
        threading.Thread(target=correct_and_speak, daemon=True).start()
    elif key == ord('m'):
        modes = ["AUTO", "STATIC", "DYNAMIC"]
        mode  = modes[(modes.index(mode) + 1) % 3]
        print(f"🔁 Mode: {mode}")

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")
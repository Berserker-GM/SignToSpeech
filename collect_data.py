import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Signs to collect ──────────────────────────────────────────────
# Letters A-Z plus common words
SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "HELLO", "HELP", "THANK_YOU", "YES", "NO",
    "PLEASE", "SORRY", "WATER", "PAIN", "STOP"
]

SAMPLES_PER_SIGN = 100   # how many samples to collect per sign
DATA_FILE = "data/landmarks.csv"

# ── Setup CSV ─────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
file_exists = os.path.isfile(DATA_FILE)

csvfile = open(DATA_FILE, "a", newline="")
writer = csv.writer(csvfile)

# Header: 21 landmarks × (x, y, z) = 63 features + label
if not file_exists:
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

# ── Helper: extract + normalize landmarks ────────────────────────
# DELETE the old extract_landmarks function and replace with this import
import sys
sys.path.insert(0, '.')
from utils.landmarks import extract_landmarks

# ── Main collection loop ──────────────────────────────────────────
def collect_sign(sign_label):
    cap = cv2.VideoCapture(0)
    count = 0
    collecting = False

    print(f"\n📌 Get ready to show sign: {sign_label}")
    print("  Press SPACE to start collecting | Press Q to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Status overlay
        status_color = (0, 255, 0) if collecting else (0, 165, 255)
        status_text = f"Collecting: {count}/{SAMPLES_PER_SIGN}" if collecting else "Press SPACE to start"

        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {sign_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Progress bar
        if collecting and SAMPLES_PER_SIGN > 0:
            bar_width = int((count / SAMPLES_PER_SIGN) * 620)
            cv2.rectangle(frame, (10, 58), (10 + bar_width, 65), (0, 255, 0), -1)

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                if collecting:
                    row = extract_landmarks(hand_lm)
                    row.append(sign_label)
                    writer.writerow(row)
                    csvfile.flush()
                    count += 1

        else:
            cv2.putText(frame, "NO HAND DETECTED", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False   # user quit

        # Auto-stop when enough samples collected
        if count >= SAMPLES_PER_SIGN:
            print(f"  ✅ Done! {count} samples saved for '{sign_label}'")
            time.sleep(0.5)
            break

    cap.release()
    cv2.destroyAllWindows()
    return True   # finished this sign, continue to next


# ── Run collection for each sign ─────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SIGN LANGUAGE DATA COLLECTOR")
    print("=" * 50)
    print(f"Signs to collect: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"Saving to: {DATA_FILE}\n")

    # Let user pick a starting sign (so you can resume later)
    print("Which sign do you want to start from?")
    for i, s in enumerate(SIGNS):
        print(f"  {i:2d}: {s}")
    start_idx = int(input("\nEnter number (0 to start from beginning): "))

    for sign in SIGNS[start_idx:]:
        should_continue = collect_sign(sign)
        if not should_continue:
            print("\n⚠️  Collection stopped early. Run again to resume from any sign.")
            break

    csvfile.close()
    print("\n✅ All done! Data saved to", DATA_FILE)
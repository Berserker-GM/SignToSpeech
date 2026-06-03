import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Motion signs to collect ───────────────────────────────────────
MOTION_SIGNS = [
    "HELLO", #wave from forehead
    "HELP", #gentle tap of non-dominant hand
    "THANK_YOU", #hand moves from chin outward
    "YES", #fist bounces up and down
    "NO", #index and middle fingers tap together
    "PLEASE", #circular motion of flat hand on chest
    "SORRY" #circular motion of fist on chest
]

SEQUENCE_LENGTH = 40     # frames per sample
SAMPLES_PER_SIGN = 90    # sequences per sign
DATA_DIR = "data/dynamic"

os.makedirs(DATA_DIR, exist_ok=True)

# ── Extract landmarks (same normalization as static) ──────────────
# DELETE the old extract_landmarks function and replace with this import
import sys
sys.path.insert(0, '.')
from utils.landmarks import extract_landmarks  # 63 values

def empty_frame():
    return [0.0] * 63

# ── Collect one sign ──────────────────────────────────────────────
def collect_sign(sign_label):
    sign_dir = os.path.join(DATA_DIR, sign_label)
    os.makedirs(sign_dir, exist_ok=True)

    # Count existing samples so we can resume
    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    print(f"\n📌 Sign: {sign_label}  (have {existing}/{SAMPLES_PER_SIGN})")
    print("   Press SPACE to record each sample | Q to quit")

    cap = cv2.VideoCapture(0)
    sample_count = existing

    while sample_count < SAMPLES_PER_SIGN and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # UI
        cv2.rectangle(frame, (0, 0), (640, 65), (20, 20, 20), -1)
        cv2.putText(frame, f"Sign: {sign_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_PER_SIGN}  |  Press SPACE to record",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, "NO HAND", (260, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Dynamic Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

        if key == ord(' '):
            # ── Auto-record all samples in one go ────────────────
            print(f"\n🚀 Auto-recording {SAMPLES_PER_SIGN - sample_count} samples...")
            print("   Get ready — starting in 3 seconds\n")

            for countdown in range(3, 0, -1):
                ret2, frame2 = cap.read()
                frame2 = cv2.flip(frame2, 1)
                cv2.rectangle(frame2, (0,0), (640,480), (20,20,20), -1)
                cv2.putText(frame2, f"Starting in {countdown}...",
                            (180, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 100), 4)
                cv2.putText(frame2, f"Do the '{sign_label}' motion repeatedly",
                            (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,255), 2)
                cv2.imshow("Dynamic Data Collection", frame2)
                cv2.waitKey(1)
                time.sleep(1)

            while sample_count < SAMPLES_PER_SIGN:
                sequence = []
                frame_count = 0

                # Record one sequence of SEQUENCE_LENGTH frames
                while frame_count < SEQUENCE_LENGTH:
                    ret3, frame3 = cap.read()
                    if not ret3:
                        break
                    frame3 = cv2.flip(frame3, 1)
                    rgb3   = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                    res3   = hands.process(rgb3)

                    # Progress bar
                    progress = int((frame_count / SEQUENCE_LENGTH) * 620)
                    cv2.rectangle(frame3, (0, 460), (progress, 480), (0,255,0), -1)

                    # Status overlay
                    cv2.rectangle(frame3, (0,0), (640,65), (20,20,20), -1)
                    cv2.putText(frame3, f"Sample {sample_count+1}/{SAMPLES_PER_SIGN}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,180), 2)
                    cv2.putText(frame3, f"Frame {frame_count+1}/{SEQUENCE_LENGTH}  — keep doing the motion!",
                                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                    if res3.multi_hand_landmarks:
                        lm_data = extract_landmarks(res3.multi_hand_landmarks[0])
                        mp_draw.draw_landmarks(frame3,
                                               res3.multi_hand_landmarks[0],
                                               mp_hands.HAND_CONNECTIONS)
                    else:
                        lm_data = empty_frame()
                        cv2.putText(frame3, "NO HAND — keep hand visible!",
                                    (150, 260), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0,0,255), 2)

                    sequence.append(lm_data)
                    frame_count += 1
                    cv2.imshow("Dynamic Data Collection", frame3)
                    cv2.waitKey(1)

                # Save this sequence
                seq_array = np.array(sequence)
                save_path  = os.path.join(sign_dir, f"{sample_count}.npy")
                np.save(save_path, seq_array)
                sample_count += 1
                print(f"  ✅ Sample {sample_count}/{SAMPLES_PER_SIGN} saved")

                # Brief flash between samples so you know one ended
                flash = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(flash, f"✅ {sample_count}/{SAMPLES_PER_SIGN}  Keep going!",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 100), 3)
                cv2.imshow("Dynamic Data Collection", flash)
                cv2.waitKey(200)   # 200ms flash between samples

            print(f"\n✅ All {SAMPLES_PER_SIGN} samples collected for '{sign_label}'!")
            sequence = []
            countdown = 3

            # 3-second countdown
            while countdown > 0:
                ret2, frame2 = cap.read()
                frame2 = cv2.flip(frame2, 1)
                cv2.rectangle(frame2, (0,0), (640,480), (20,20,20), -1)
                cv2.putText(frame2, f"Starting in {countdown}...",
                            (180, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 100), 4)
                cv2.putText(frame2, f"Do the '{sign_label}' motion",
                            (160, 300), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,255), 2)
                cv2.imshow("Dynamic Data Collection", frame2)
                cv2.waitKey(1)
                time.sleep(1)
                countdown -= 1

            # Record SEQUENCE_LENGTH frames
            print(f"  🎬 Recording sample {sample_count + 1}...", end=" ")
            frame_count = 0

            while frame_count < SEQUENCE_LENGTH:
                ret3, frame3 = cap.read()
                if not ret3:
                    break
                frame3 = cv2.flip(frame3, 1)
                rgb3   = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                res3   = hands.process(rgb3)

                # Progress bar
                progress = int((frame_count / SEQUENCE_LENGTH) * 620)
                cv2.rectangle(frame3, (0, 460), (progress, 480), (0,255,0), -1)
                cv2.putText(frame3, f"Recording... {frame_count}/{SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if res3.multi_hand_landmarks:
                    lm_data = extract_landmarks(res3.multi_hand_landmarks[0])
                    mp_draw.draw_landmarks(frame3,
                                           res3.multi_hand_landmarks[0],
                                           mp_hands.HAND_CONNECTIONS)
                else:
                    lm_data = empty_frame()

                sequence.append(lm_data)
                frame_count += 1

                cv2.imshow("Dynamic Data Collection", frame3)
                cv2.waitKey(1)

            # Save as numpy array: shape (40, 63)
            seq_array = np.array(sequence)
            save_path = os.path.join(sign_dir, f"{sample_count}.npy")
            np.save(save_path, seq_array)
            sample_count += 1
            print(f"saved! ({sample_count}/{SAMPLES_PER_SIGN})")

    cap.release()
    cv2.destroyAllWindows()
    return True

# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  DYNAMIC SIGN DATA COLLECTOR")
    print("=" * 50)
    print(f"Motion signs: {MOTION_SIGNS}")
    print(f"Frames per sample: {SEQUENCE_LENGTH}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}\n")

    print("Which sign to start from?")
    for i, s in enumerate(MOTION_SIGNS):
        print(f"  {i}: {s}")
    start = int(input("\nEnter number (0 = beginning): "))

    for sign in MOTION_SIGNS[start:]:
        ok = collect_sign(sign)
        if not ok:
            print("\n⚠️  Stopped. Run again to resume.")
            break

    print("\n✅ Done! Data saved to data/dynamic/")
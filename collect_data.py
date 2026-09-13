import csv
import os
import sys
import time

import cv2
import mediapipe as mp

sys.path.insert(0, ".")
import params as P
from utils.camera import open_camera
from utils.landmarks import extract_landmarks

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=P.MP_MAX_HANDS,
    min_detection_confidence=P.MP_DET_CONF,
    min_tracking_confidence=P.MP_TRACK_CONF,
)

# ── Signs to collect ──────────────────────────────────────────────
SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "HELLO",
    "HELP",
    "THANK_YOU",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "WATER",
    "PAIN",
    "STOP",
    "MORE",
    "RESTROOM",
    "EAT",
    "DRINK",
    "FRIEND",
    "FAMILY",
    "GOOD",
    "BAD",
    "HAPPY",
    "SAD",
]

SAMPLES_PER_SIGN = 100
SAMPLE_INTERVAL_MS = 200  # only save a row every N ms (reduces near-duplicates)
DATA_FILE = "data/landmarks.csv"

# CSV schema: 63 landmark cols + session + label
FEATURE_HEADER = []
for i in range(21):
    FEATURE_HEADER += [f"x{i}", f"y{i}", f"z{i}"]
CSV_HEADER = FEATURE_HEADER + ["session", "label"]

os.makedirs("data", exist_ok=True)


def _ensure_csv_schema():
    """Create file or migrate legacy CSVs (no session column) → add session=0."""
    if not os.path.isfile(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        return

    with open(DATA_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

    if header == CSV_HEADER:
        return

    # Legacy: 63 features + label (no session)
    if header and header[-1] == "label" and "session" not in header:
        print("♻️  Migrating landmarks.csv → add session column (legacy rows get session=0)")
        import pandas as pd

        df = pd.read_csv(DATA_FILE)
        if "session" not in df.columns:
            # Insert session before label
            cols = [c for c in df.columns if c != "label"]
            df["session"] = 0
            df = df[cols + ["session", "label"]]
            df.to_csv(DATA_FILE, index=False)
        return

    if not header:
        with open(DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


_ensure_csv_schema()

# Next session id = max existing + 1
def _next_session_start() -> int:
    try:
        import pandas as pd

        df = pd.read_csv(DATA_FILE, usecols=["session"])
        if len(df) == 0:
            return 1
        return int(df["session"].max()) + 1
    except Exception:
        return 1


_session_counter = _next_session_start()

csvfile = open(DATA_FILE, "a", newline="")
writer = csv.writer(csvfile)


def collect_sign(cap, sign_label):
    """Collect one sign using an already-open camera. Returns False to stop the session."""
    global _session_counter

    count = 0
    collecting = False
    session_id = None
    last_sample_t = 0.0
    vary_hint_t = 0.0

    print(f"\n📌 Get ready to show sign: {sign_label}")
    print("  Press SPACE to start collecting | Press Q to quit")
    print(f"  Sampling every {SAMPLE_INTERVAL_MS}ms — vary angle/distance while holding\n")
    print("  → Click the 'Data Collection' window, then press SPACE.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera stopped returning frames.")
            return False

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        now = time.time()

        status_color = (0, 255, 0) if collecting else (0, 165, 255)
        status_text = (
            f"Collecting: {count}/{SAMPLES_PER_SIGN}  sess#{session_id}"
            if collecting
            else "Press SPACE to start"
        )

        cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Sign: {sign_label}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            status_text,
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        if collecting and SAMPLES_PER_SIGN > 0:
            bar_width = int((count / SAMPLES_PER_SIGN) * 620)
            cv2.rectangle(frame, (10, 62), (10 + bar_width, 68), (0, 255, 0), -1)

        # Periodic variety prompt
        if collecting and (now - vary_hint_t) > 2.5:
            cv2.putText(
                frame,
                "Slightly change angle / distance / position",
                (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                if collecting and session_id is not None:
                    elapsed_ms = (now - last_sample_t) * 1000.0
                    if elapsed_ms >= SAMPLE_INTERVAL_MS:
                        row = extract_landmarks(hand_lm)
                        row.append(session_id)
                        row.append(sign_label)
                        writer.writerow(row)
                        csvfile.flush()
                        count += 1
                        last_sample_t = now
                        if count % 8 == 0:
                            vary_hint_t = now  # flash hint periodically
        else:
            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (200, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not collecting:
                collecting = True
                session_id = _session_counter
                _session_counter += 1
                last_sample_t = 0.0  # allow immediate first sample
                vary_hint_t = time.time()
                print(f"  ▶ Session {session_id} started for '{sign_label}'")
        elif key == ord("q"):
            return False

        if count >= SAMPLES_PER_SIGN:
            print(f"  ✅ Done! {count} samples saved for '{sign_label}' (session {session_id})")
            time.sleep(0.5)
            return True

    return False


if __name__ == "__main__":
    print("=" * 50)
    print("  SIGN LANGUAGE DATA COLLECTOR")
    print("=" * 50)
    print(f"Signs to collect: {SIGNS}")
    print(f"Samples per sign: {SAMPLES_PER_SIGN}")
    print(f"Sample interval: {SAMPLE_INTERVAL_MS} ms")
    print(f"MediaPipe det/track: {P.MP_DET_CONF}/{P.MP_TRACK_CONF}")
    print(f"Saving to: {DATA_FILE}")
    print("CSV columns: 63 landmarks + session + label\n")

    print("Which sign do you want to start from?")
    for i, s in enumerate(SIGNS):
        print(f"  {i:2d}: {s}")
    start_idx = int(input("\nEnter number (0 to start from beginning): "))

    cap, cam_idx = open_camera(0)
    if cap is None:
        csvfile.close()
        raise SystemExit(
            "\n❌ Could not open any webcam.\n"
            "   • Close Zoom/Teams/browser tabs using the camera\n"
            "   • Windows Settings → Privacy → Camera → allow desktop apps\n"
            "   • Test with: python test_camera.py\n"
        )

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    print(f"\n📷 Webcam ready (index {cam_idx}). Look for the OpenCV window.")

    try:
        for sign in SIGNS[start_idx:]:
            should_continue = collect_sign(cap, sign)
            if not should_continue:
                print("\n⚠️  Collection stopped early. Run again to resume from any sign.")
                break
        else:
            print("\n✅ All done! Data saved to", DATA_FILE)
            print("   Train with: python train_model.py  (uses GroupShuffleSplit on session)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csvfile.close()


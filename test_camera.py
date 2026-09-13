"""
Quick webcam smoke test.

  python test_camera.py              # auto-pick a working camera
  python test_camera.py --index 1    # force camera index
  python test_camera.py --probe      # list which devices/backends work
  python test_camera.py --raw        # video only (no MediaPipe)
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from utils.camera import frame_looks_valid, open_camera, probe_cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Test webcam for STS data collection")
    parser.add_argument("--index", type=int, default=0, help="Preferred camera index")
    parser.add_argument("--probe", action="store_true", help="Probe all cameras and exit")
    parser.add_argument("--raw", action="store_true", help="Skip MediaPipe (video only)")
    args = parser.parse_args()

    if args.probe:
        probe_cameras()
        return

    cap, idx = open_camera(args.index)
    if cap is None:
        print("\nRun with --probe to see details:", file=sys.stderr)
        print("  python test_camera.py --probe", file=sys.stderr)
        raise SystemExit(1)

    hands = None
    mp_draw = None
    hand_connections = None
    if not args.raw:
        try:
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            hand_connections = mp_hands.HAND_CONNECTIONS
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"MediaPipe unavailable ({e}); showing raw video only.")

    win = "Hand Test - Press Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"Showing camera index {idx}. Press Q in the window to quit.")

    bad_streak = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lost camera frames.")
            break

        if not frame_looks_valid(frame):
            bad_streak += 1
            if bad_streak == 15:
                print(
                    "⚠️  Still getting garbage frames. Try:\n"
                    "   python test_camera.py --probe\n"
                    "   python test_camera.py --index 1"
                )
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "Bad camera frame — try --probe / --index 1",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            frame = canvas
        else:
            bad_streak = 0
            frame = cv2.flip(frame, 1)

            if hands is not None and mp_draw is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for hand_lm in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_lm, hand_connections)
                        cv2.putText(
                            frame,
                            f"Landmarks: {len(hand_lm.landmark)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

            cv2.putText(
                frame,
                f"cam {idx}  {frame.shape[1]}x{frame.shape[0]}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

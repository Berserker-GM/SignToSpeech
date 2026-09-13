"""Open a webcam reliably on Windows (DirectShow / MSMF) and other platforms."""

from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np


def frame_looks_valid(frame) -> bool:
    """Reject black / static-garbage buffers common with wrong Windows backends."""
    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return False
    if frame.ndim < 2:
        return False

    h, w = frame.shape[:2]
    if h < 48 or w < 48:
        return False

    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3
        else frame
    )
    mean = float(gray.mean())
    std = float(gray.std())

    # Nearly black / empty
    if mean < 6 and std < 12:
        return False

    # Classic bad OpenCV Windows frame: noisy strip on top, black below
    top = gray[: max(1, h // 5)]
    bot = gray[h // 5 :]
    if float(top.std()) > 35 and float(bot.mean()) < 8 and float(bot.std()) < 12:
        return False

    # Completely flat
    if std < 2.5:
        return False

    return True


def _try_open(
    index: int,
    backend: Optional[int],
    backend_name: str,
    *,
    width: Optional[int],
    height: Optional[int],
    fourcc: Optional[str],
) -> Optional[cv2.VideoCapture]:
    cap = (
        cv2.VideoCapture(index, backend)
        if backend is not None
        else cv2.VideoCapture(index)
    )
    if not cap.isOpened():
        cap.release()
        return None

    # Convert buffers to numpy (helps some MSMF builds)
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass

    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Drop a few frames, then require a *valid-looking* frame
    good = None
    for i in range(20):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if frame_looks_valid(frame):
            good = frame
            break
        # keep last raw frame for debugging if nothing validates
        if i == 19:
            good = None

    if good is None:
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"📷 Camera OK  index={index}  backend={backend_name}  "
        f"size={actual_w}x{actual_h}  fourcc={fourcc or 'default'}"
    )
    return cap


def open_camera(preferred_index: int = 0, width: int = 640, height: int = 480):
    """
    Try camera indices / backends / pixel formats until a sane frame appears.
    Returns (cap, index) or (None, None).
    """
    if sys.platform.startswith("win"):
        backends: list[tuple[str, Optional[int]]] = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", None),
        ]
    else:
        backends = [("ANY", None)]

    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]

    # Prefer MJPG at 640x480 on Windows laptops; also try native + YUY2
    configs: list[tuple[Optional[int], Optional[int], Optional[str]]] = [
        (width, height, "MJPG"),
        (None, None, "MJPG"),
        (width, height, None),
        (None, None, None),
        (1280, 720, "MJPG"),
        (width, height, "YUY2"),
        (None, None, "YUY2"),
    ]

    for index in indices:
        for bname, backend in backends:
            for w, h, fourcc in configs:
                label = f"{bname}/idx{index}/{fourcc or 'raw'}/{w or 'auto'}x{h or 'auto'}"
                try:
                    cap = _try_open(
                        index,
                        backend,
                        bname,
                        width=w,
                        height=h,
                        fourcc=fourcc,
                    )
                except Exception as e:
                    print(f"  skip {label}: {e}")
                    continue
                if cap is not None:
                    return cap, index

    print(
        "❌ No usable camera frame found.\n"
        "   Tips:\n"
        "   • Close Zoom / Teams / browser / Phone Link using the camera\n"
        "   • Windows Settings → Privacy & security → Camera → allow desktop apps\n"
        "   • Try: python test_camera.py --index 1\n"
        "   • Try: python test_camera.py --probe  (lists every device attempt)"
    )
    return None, None


def probe_cameras(max_index: int = 5) -> None:
    """Print which index/backend combinations return a valid frame."""
    if sys.platform.startswith("win"):
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    else:
        backends = [("ANY", None)]

    print("Probing cameras…\n")
    found = False
    for index in range(max_index):
        for bname, backend in backends:
            for fourcc in ("MJPG", None):
                cap = (
                    cv2.VideoCapture(index, backend)
                    if backend is not None
                    else cv2.VideoCapture(index)
                )
                opened = cap.isOpened()
                if not opened:
                    print(f"  [{index}] {bname} fourcc={fourcc or '-'}  NOT OPEN")
                    cap.release()
                    continue
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = False, None
                for _ in range(12):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                ok = ret and frame_looks_valid(frame)
                shape = getattr(frame, "shape", None)
                mean = float(np.mean(frame)) if ok or (ret and frame is not None) else 0
                status = "VALID" if ok else ("GARBAGE/BLACK" if ret else "NO FRAME")
                print(
                    f"  [{index}] {bname} fourcc={fourcc or '-'}  {status}  "
                    f"shape={shape} mean={mean:.1f}"
                )
                if ok:
                    found = True
                cap.release()
    if not found:
        print("\nNo valid camera stream found.")
    else:
        print("\nUse a VALID line’s index, e.g. python test_camera.py --index 0")

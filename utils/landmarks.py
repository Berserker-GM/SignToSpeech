# utils/landmarks.py
import numpy as np

def extract_landmarks(hand_landmarks):
    lm = hand_landmarks.landmark
    
    # Step 1: subtract wrist (position invariant)
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for point in lm:
        coords.append([
            point.x - wrist_x,
            point.y - wrist_y,
            point.z - wrist_z
        ])
    
    coords = np.array(coords)  # (21, 3)
    
    # Step 2: scale by hand size (distance invariant)
    # Use middle finger tip (landmark 12) to wrist as reference scale
    scale = np.linalg.norm(coords[12])
    if scale > 0:
        coords = coords / scale
    
    return coords.flatten().tolist()  # 63 values
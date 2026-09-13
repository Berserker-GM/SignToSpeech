"""
Sequence feature helpers for dynamic LSTM.

Production path (train_dynamic.py + api/inference.py):
  pose (T, 63) → add_velocity → (T, 126)

Rotation-invariance helpers below are for a *future* retrain only.
"""
from __future__ import annotations

import numpy as np

# MediaPipe landmark indices
WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17


def normalize_xyz(coords: np.ndarray) -> np.ndarray:
    """coords: (21, 3) absolute → wrist-centered, scale-normalized (current production)."""
    wrist = coords[WRIST].copy()
    rel = coords - wrist
    scale = np.linalg.norm(rel[12])  # middle tip
    if scale > 1e-6:
        rel = rel / scale
    return rel


def normalize_xyz_rot_invar(coords: np.ndarray) -> np.ndarray:
    """
    Future-train option: also rotate so palm axes align (wrist→middle_mcp = +Y).
    Do not enable until you recollect + retrain both models.
    """
    rel = normalize_xyz(coords)
    y_axis = rel[MIDDLE_MCP]
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return rel
    y_axis = y_axis / y_norm
    # Rough palm x from index-pinky
    x_axis = rel[INDEX_MCP] - rel[PINKY_MCP]
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return rel
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # world → palm
    return (R @ rel.T).T


def add_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    sequence: (T, 63) → (T, 126) with frame-to-frame deltas.
    First frame delta is zeros.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=-1)

"""
Tunable recognition / grammar / MediaPipe parameters for STS.

Latency budget (≈30 FPS webcam):
  EMA lock (~4–6 frames) + CONFIRM_SECONDS ≈ time-to-word
  Do NOT also stack a large STABLE_FRAMES on top of heavy majority voting.
"""

# ── Landmark sequence (must match train_dynamic.py / collect_dynamic.py) ──
SEQUENCE_LENGTH = 40          # frames in the LSTM motion window
# At 30 FPS → 1.33s of motion; at 15 FPS → 2.67s. Prefer ~30 FPS camera.

# ── Confirm / cooldown (sign acceptance) ──────────────────────────────────
# Old stack (SMOOTH_MIN_VOTES=6 + STABLE_FRAMES=5 + CONFIRM=2s) ≈ 2.4s+ lag.
# New: EMA decides the label; confirm is the only hold gate.
CONFIRM_SECONDS = 0.70        # was 2.0 — main hold after EMA locks a label
COOLDOWN_SECONDS = 1.20       # was 3.0 — still prevents double-fire
GAP_TOLERANCE_SECONDS = 0.40  # reuse last landmarks / keep hold across brief dropouts

# ── Model confidence gates ────────────────────────────────────────────────
# 0.80 / 0.75 is aggressive → many true signs never enter the smoother.
# Calibrate with:  python calibrate_thresholds.py
STATIC_THRESHOLD = 0.68
DYNAMIC_THRESHOLD = 0.65
AUTO_MOTION_MARGIN = 0.08     # larger margin → fewer letter flickers during motion

# ── Temporal smoothing (EMA of class confidence — replaces huge majority vote) ──
EMA_ALPHA = 0.35              # higher = snappier, lower = smoother
EMA_LOCK_THRESHOLD = 0.55     # smoothed score needed to expose a label
EMA_SWITCH_MARGIN = 0.12      # new label must beat current by this to steal the lock
# Kept for API compat / light consecutive check (1–2 only; do not re-introduce lag)
SMOOTH_WINDOW = 8             # legacy field (unused by EMA path)
SMOOTH_MIN_VOTES = 1          # legacy
STABLE_FRAMES = 2             # tiny debounce after EMA lock before hold timer starts

# ── MediaPipe Hands ───────────────────────────────────────────────────────
MP_MAX_HANDS = 3              # two-handed signs (MORE/FRIEND) need architecture change
MP_DET_CONF = 0.60            # keep collect_* and live inference identical
MP_TRACK_CONF = 0.60
MP_PRESENCE_CONF = 0.60
LANDMARK_HOLD_SECONDS = 0.40  # server reuses last 63-vector during brief MP dropouts
DYNAMIC_FEATURE_DIM = 126     # pose(63) + velocity(63) — must match train_dynamic.py

# ── DeepSeek grammar ──────────────────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MAX_TOKENS = 160
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_TIMEOUT_S = 20
DEEPSEEK_DISABLE_THINKING = True

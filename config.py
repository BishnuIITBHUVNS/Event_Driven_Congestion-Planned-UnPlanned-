"""
config.py — All constants in one place.
Edit paths here if you move files.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"   / "events.csv"
DATA_PROC   = ROOT / "data" / "processed" / "events_features.csv"
MODEL_DIR   = ROOT / "models"

# ── Timezone ──────────────────────────────────────────────────────────────────
import pandas as pd
IST_OFFSET  = pd.Timedelta(hours=5, minutes=30)   # UTC → IST

# ── Domain severity maps (calibrated from EDA) ────────────────────────────────

# How severe each event_cause is for congestion (0 = none, 5 = critical)
CAUSE_SEVERITY = {
    "congestion":          5,
    "accident":            5,
    "debris":              5,
    "Debris":              5,
    "vip_movement":        4,
    "public_event":        4,
    "protest":             4,
    "procession":          3,
    "construction":        3,
    "tree_fall":           3,
    "water_logging":       3,
    "road_conditions":     2,
    "vehicle_breakdown":   2,
    "pot_holes":           2,
    "others":              1,
    "Fog / Low Visibility":1,
    "test_demo":           0,
}

# Empirical road-closure probability per cause (from EDA)
CAUSE_CLOSURE_PROB = {
    "debris":              1.00,
    "Debris":              1.00,
    "vip_movement":        0.80,
    "public_event":        0.46,
    "protest":             0.40,
    "tree_fall":           0.39,
    "construction":        0.27,
    "procession":          0.26,
    "road_conditions":     0.12,
    "others":              0.09,
    "water_logging":       0.09,
    "congestion":          0.04,
    "vehicle_breakdown":   0.04,
    "accident":            0.03,
    "pot_holes":           0.02,
    "Fog / Low Visibility":0.00,
    "test_demo":           0.00,
}

# Congestion impact weight per vehicle type (heavier = bigger blockage)
VEH_SEVERITY = {
    "heavy_vehicle": 4,
    "truck":         4,
    "private_bus":   3,
    "bmtc_bus":      3,
    "ksrtc_bus":     3,
    "lcv":           2,
    "private_car":   1,
    "taxi":          1,
    "auto":          1,
    "others":        1,
}

# ── Feature column list (used in train.py and app.py) ─────────────────────────
FEATURE_COLS = [
    # Temporal
    "hour_ist", "day_of_week", "month", "is_weekend",
    "is_morning_rush", "is_evening_rush", "is_peak", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Event  (requires_road_closure_int REMOVED — it IS the classification target)
    "cause_severity", "cause_closure_prob", "veh_severity",
    "is_planned", "has_vehicle",
    # Spatial
    "is_named_corridor", "corridor_freq", "zone_priority_rate",
    "has_junction", "dist_from_center_km",
    # Historical density
    "corridor_density_2h", "zone_density_2h",
]

# ── Target ────────────────────────────────────────────────────────────────────
#
# NOTE: priority_class is 99.6% determined by is_named_corridor — not a useful
# ML target. The genuinely hard targets are:
#
#   TARGET_CLS = "road_closure_class"  (1 = closure required, 0 = not)
#                8.3% positive, varies meaningfully by cause & location.
#                This is the actionable barricading/deployment decision.
#
#   TARGET_REG = "log_duration_min"   (log1p of minutes to close)
#                Available on 38% of rows — more meaningful than a composite.
#
TARGET_CLS     = "road_closure_class"   # primary classification target
TARGET_REG     = "log_duration_min"     # primary regression target
TARGET_REG_ALT = "impact_score"        # secondary / dashboard display

# Bengaluru city center (for distance feature)
CITY_LAT, CITY_LON = 12.9716, 77.5946

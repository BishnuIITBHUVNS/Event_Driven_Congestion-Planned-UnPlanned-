"""
config.py — All constants in one place.
Paths are set for FLAT repo structure (all files at root).
"""
from pathlib import Path

# - Paths (flat repo — everything sits at the repo root)
ROOT      = Path(__file__).resolve().parent   # repo root
DATA_RAW  = ROOT / "events.csv"               # not needed at deploy time
DATA_PROC = ROOT / "events_features.csv"      # uploaded to root
MODEL_DIR = ROOT                              # pkl files are at root too

# ─ Timezone 
import pandas as pd
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

# ─ Domain severity maps
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

FEATURE_COLS = [
    "hour_ist", "day_of_week", "month", "is_weekend",
    "is_morning_rush", "is_evening_rush", "is_peak", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "cause_severity", "cause_closure_prob", "veh_severity",
    "is_planned", "has_vehicle",
    "is_named_corridor", "corridor_freq", "zone_priority_rate",
    "has_junction", "dist_from_center_km",
    "corridor_density_2h", "zone_density_2h",
]

TARGET_CLS     = "road_closure_class"
TARGET_REG     = "log_duration_min"
TARGET_REG_ALT = "impact_score"

CITY_LAT, CITY_LON = 12.9716, 77.5946

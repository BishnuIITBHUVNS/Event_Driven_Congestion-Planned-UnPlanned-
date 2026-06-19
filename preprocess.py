"""
preprocess.py  —  Step 1 of the pipeline.

Run:
    python src/preprocess.py

Reads  : data/raw/events.csv           (8,173 rows × 46 cols)
Writes : data/processed/events_features.csv

What happens here:
    1. Parse & timezone-correct datetimes (timestamps are UTC, we convert to IST)
    2. Compute event duration from closed_datetime − start_datetime
    3. Engineer temporal, event, spatial, and historical-density features
    4. Build two targets:
       - priority_class  (1 = High, 0 = Low)  → for classification
       - impact_score    (0–10 composite)      → for regression
    5. Drop columns that are all-NaN or leak the target
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    DATA_RAW, DATA_PROC, MODEL_DIR, IST_OFFSET,
    CAUSE_SEVERITY, CAUSE_CLOSURE_PROB, VEH_SEVERITY,
    CITY_LAT, CITY_LON, FEATURE_COLS, TARGET_CLS, TARGET_REG,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path=DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded  : {len(df):,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATETIMES + DURATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    All timestamps in the raw file are UTC (suffix +00).
    We keep the UTC column for duration maths, and add an IST column
    for all time-of-day features.
    
    NOTE: the raw hour distribution in UTC looks like 2 AM is the busiest,
    which converts to ~9:30 AM IST — that aligns with the morning rush.
    Always use hour_ist in features.
    """
    for col in ["start_datetime", "closed_datetime", "modified_datetime"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["start_ist"] = df["start_datetime"] + IST_OFFSET

    # Duration: prefer closed_datetime (7,095 valid) over resolved_datetime (only 72)
    df["duration_min"] = (
        (df["closed_datetime"] - df["start_datetime"])
        .dt.total_seconds() / 60
    )
    # Clip negative/extreme values at the 95th percentile
    p95 = df.loc[df["duration_min"] > 0, "duration_min"].quantile(0.95)
    df["duration_min"] = df["duration_min"].clip(lower=0, upper=p95).fillna(0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["start_ist"]
    df["hour_ist"]    = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek       # 0=Mon … 6=Sun
    df["month"]       = dt.dt.month

    df["is_weekend"]       = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_morning_rush"]  = df["hour_ist"].isin(range(7, 11)).astype(int)
    df["is_evening_rush"]  = df["hour_ist"].isin(range(17, 21)).astype(int)
    df["is_peak"]          = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)
    df["is_night"]         = df["hour_ist"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    # Cyclic encoding so the model sees hour 23 and hour 0 as neighbours
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_ist"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_ist"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df["cause_severity"]    = df["event_cause"].map(CAUSE_SEVERITY).fillna(1)
    df["cause_closure_prob"]= df["event_cause"].map(CAUSE_CLOSURE_PROB).fillna(0.05)
    df["veh_severity"]      = df["veh_type"].map(VEH_SEVERITY).fillna(0)
    df["is_planned"]        = (df["event_type"] == "planned").astype(int)
    df["has_vehicle"]       = df["veh_type"].notna().astype(int)
    df["requires_road_closure_int"] = df["requires_road_closure"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPATIAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_spatial(df: pd.DataFrame) -> pd.DataFrame:
    # Named corridor vs generic "Non-corridor"
    df["is_named_corridor"] = (df["corridor"] != "Non-corridor").astype(int)

    # Frequency-encode corridor (popular corridors = higher traffic stakes)
    corridor_counts = df["corridor"].value_counts()
    df["corridor_freq"] = df["corridor"].map(corridor_counts).fillna(0)

    # Zone-level average high-priority rate (target-encoded, leakage-safe: fill with global mean)
    global_rate = (df["priority"] == "High").mean()
    zone_rate   = df.groupby("zone")["priority"].apply(lambda x: (x == "High").mean())
    df["zone_priority_rate"] = df["zone"].map(zone_rate).fillna(global_rate)

    # Named junction present
    df["has_junction"] = df["junction"].notna().astype(int)

    # Distance from Bengaluru city centre (degrees → rough km)
    df["dist_from_center_km"] = np.sqrt(
        (df["latitude"]  - CITY_LAT) ** 2 +
        (df["longitude"] - CITY_LON) ** 2
    ) * 111.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. HISTORICAL DENSITY
#    How many events happened in the same corridor / zone in the 2 hours
#    before this event? This is the "congestion memory" signal.
# ─────────────────────────────────────────────────────────────────────────────

def add_historical_density(df: pd.DataFrame, window_hours: int = 2) -> pd.DataFrame:
    """
    O(n²) in the worst case but tractable at 8k rows.
    For production use a groupby + rolling approach.
    """
    df = df.sort_values("start_datetime").reset_index(drop=True)
    window = pd.Timedelta(hours=window_hours)

    corridor_density = np.zeros(len(df), dtype=int)
    zone_density     = np.zeros(len(df), dtype=int)

    print("  Computing historical density (corridor + zone, 2-hour window)…")
    for i in tqdm(range(len(df))):
        t         = df.at[i, "start_datetime"]
        corridor  = df.at[i, "corridor"]
        zone      = df.at[i, "zone"]
        mask_time = (df["start_datetime"] >= t - window) & (df["start_datetime"] < t)

        if pd.notna(corridor):
            corridor_density[i] = int((df.loc[mask_time, "corridor"] == corridor).sum())
        if pd.notna(zone):
            zone_density[i] = int((df.loc[mask_time, "zone"] == zone).sum())

    df["corridor_density_2h"] = corridor_density
    df["zone_density_2h"]     = zone_density
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD TARGETS
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    road_closure_class : binary — does this event require a road closure?
        Fully populated (8,173 rows).  8.3% positive.
        Much harder and more actionable than priority_class, which is
        trivially 99.6% determined by corridor type.

    log_duration_min : log1p(minutes from start to closed_datetime).
        Available on 3,126 rows (38%).  NaN elsewhere — models trained
        on this target will use only those rows.

    priority_class : kept as a derived column (useful for the dashboard and
        as a feature for other models) but not the primary ML target.

    impact_score   : composite 0–10 retained for the dashboard display.
    """
    # ── primary classification target ─────────────────────────────────────
    df["road_closure_class"] = df["requires_road_closure_int"]   # already 0/1

    # ── primary regression target ─────────────────────────────────────────
    # log1p of duration; NaN where duration is 0 (not yet closed)
    df["log_duration_min"] = np.where(
        df["duration_min"] > 0,
        np.log1p(df["duration_min"]),
        np.nan
    )

    # ── keep priority_class as derived column ─────────────────────────────
    df["priority_class"] = (df["priority"] == "High").astype(int)

    # ── keep impact_score for dashboard ──────────────────────────────────
    p95_dur = df.loc[df["duration_min"] > 0, "duration_min"].quantile(0.95)
    dur_norm = np.log1p(df["duration_min"]) / np.log1p(max(p95_dur, 1))
    dur_norm = dur_norm.clip(0, 1)
    df["impact_score"] = (
        0.40 * df["priority_class"].astype(float) +
        0.35 * df["road_closure_class"].astype(float) +
        0.25 * dur_norm
    ) * 10

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame):
    print("\n── Sanity check ──────────────────────────────────────────────")
    feats_present = [f for f in FEATURE_COLS if f in df.columns]
    feats_missing = [f for f in FEATURE_COLS if f not in df.columns]
    print(f"  Features ready   : {len(feats_present)}/{len(FEATURE_COLS)}")
    if feats_missing:
        print(f"  ⚠ Missing        : {feats_missing}")

    X = df[feats_present].fillna(0)
    nan_pct = X.isnull().mean().sort_values(ascending=False)
    if nan_pct.max() > 0:
        print(f"  NaN in features  :\n{nan_pct[nan_pct>0]}")
    else:
        print("  NaN in features  : none ✓")

    print(f"\n  road_closure_class  : {df['road_closure_class'].value_counts().to_dict()} "
          f"({df['road_closure_class'].mean():.1%} positive)")
    print(f"  log_duration_min    : {df['log_duration_min'].notna().sum():,} valid rows "
          f"({df['log_duration_min'].notna().mean():.1%})")
    print(f"  impact_score (dash) : mean={df['impact_score'].mean():.2f}  "
          f"std={df['impact_score'].std():.2f}")
    print("──────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(skip_density: bool = False):
    """
    skip_density=True lets you re-run faster during debugging.
    Set False for the final submission build.
    """
    DATA_PROC.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()

    print("Step 1: parsing datetimes…")
    df = parse_datetimes(df)

    print("Step 2: temporal features…")
    df = add_temporal(df)

    print("Step 3: event features…")
    df = add_event_features(df)

    print("Step 4: spatial features…")
    df = add_spatial(df)

    if skip_density:
        print("Step 5: SKIPPED historical density (skip_density=True)")
        df["corridor_density_2h"] = 0
        df["zone_density_2h"]     = 0
    else:
        print("Step 5: historical density…")
        df = add_historical_density(df)

    print("Step 6: building targets…")
    df = build_targets(df)

    sanity_check(df)

    df.to_csv(DATA_PROC, index=False)
    print(f"Saved  : {DATA_PROC}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-density", action="store_true",
                        help="Skip the slow historical-density step (for quick iteration)")
    args = parser.parse_args()
    run(skip_density=args.skip_density)

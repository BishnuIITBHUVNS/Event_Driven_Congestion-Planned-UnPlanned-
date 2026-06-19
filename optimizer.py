"""
optimizer.py  —  Resource allocation over predicted congestion zones.

Given a DataFrame of corridor/zone congestion predictions,
allocates police personnel and barricades using Google OR-Tools CP-SAT.

Usage (standalone):
    from src.optimizer import allocate
    result_df = allocate(predicted_zones_df, total_personnel=80, total_barricades=40)
"""

import pandas as pd
import numpy as np
from ortools.sat.python import cp_model


def allocate(
    zones: pd.DataFrame,
    total_personnel: int = 80,
    total_barricades: int = 40,
    max_per_zone_p: int = 20,
    max_per_zone_b: int = 12,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    zones : DataFrame with columns
        - zone_id        (str)
        - label          (str, e.g. corridor or zone name)
        - impact_score   (float 0–10)
        - event_count    (int, # predicted events in this zone)
    total_personnel : int
    total_barricades : int

    Returns
    -------
    zones DataFrame with added columns:
        allocated_personnel, allocated_barricades, risk_level, priority_rank
    """
    zones = zones.copy().reset_index(drop=True)
    n = len(zones)
    if n == 0:
        return zones

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Integer-ise scores for CP-SAT
    scores    = (zones["impact_score"].fillna(0).values * 100).astype(int).clip(0, 1000)
    ev_counts = (zones.get("event_count", pd.Series([1]*n)).fillna(1).values).astype(int).clip(1)

    # ── Decision variables ────────────────────────────────────────────────────
    p = [model.NewIntVar(0, max_per_zone_p, f"p_{i}") for i in range(n)]
    b = [model.NewIntVar(0, max_per_zone_b, f"b_{i}") for i in range(n)]

    # ── Global capacity constraints ───────────────────────────────────────────
    model.Add(sum(p) <= total_personnel)
    model.Add(sum(b) <= total_barricades)

    # ── Minimum resources for high-risk zones ────────────────────────────────
    for i in range(n):
        score = zones.at[i, "impact_score"]
        if score >= 7.0:             # critical
            model.Add(p[i] >= 5)
            model.Add(b[i] >= 3)
        elif score >= 5.0:           # high
            model.Add(p[i] >= 2)
            model.Add(b[i] >= 1)

    # ── Objective: maximise weighted coverage ────────────────────────────────
    # Each unit of resource deployed to zone i earns scores[i] points
    model.Maximize(sum(scores[i] * (p[i] + b[i]) for i in range(n)))

    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        zones["allocated_personnel"]  = [solver.Value(p[i]) for i in range(n)]
        zones["allocated_barricades"] = [solver.Value(b[i]) for i in range(n)]
    else:
        # Proportional fallback
        w = zones["impact_score"].fillna(0)
        w = w / w.sum() if w.sum() > 0 else pd.Series([1/n]*n)
        zones["allocated_personnel"]  = (w * total_personnel).round().astype(int).clip(0, max_per_zone_p)
        zones["allocated_barricades"] = (w * total_barricades).round().astype(int).clip(0, max_per_zone_b)

    # ── Labels ────────────────────────────────────────────────────────────────
    zones["risk_level"] = pd.cut(
        zones["impact_score"],
        bins=[-0.01, 3, 5, 7, 10.01],
        labels=["low", "medium", "high", "critical"],
    )
    zones["priority_rank"] = zones["impact_score"].rank(ascending=False).astype(int)

    return zones.sort_values("priority_rank").reset_index(drop=True)


def aggregate_predictions(df: pd.DataFrame, group_col: str = "corridor") -> pd.DataFrame:
    """
    Aggregate row-level impact_score predictions to corridor/zone level.
    Input df must have columns: [group_col, impact_score, priority_class]
    """
    agg = (
        df.groupby(group_col)
        .agg(
            impact_score=("impact_score", "mean"),
            event_count=("impact_score", "count"),
            high_priority_count=("priority_class", "sum"),
            road_closure_count=("requires_road_closure_int", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "label"})
    )
    agg["zone_id"] = agg["label"].str[:8].str.replace(" ", "_")
    return agg.sort_values("impact_score", ascending=False).reset_index(drop=True)

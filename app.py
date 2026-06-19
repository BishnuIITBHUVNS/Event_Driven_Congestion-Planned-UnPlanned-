"""
app.py  —  Streamlit prototype dashboard.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import joblib
import folium
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import DATA_PROC, MODEL_DIR, FEATURE_COLS, TARGET_CLS, TARGET_REG
from optimizer import aggregate_predictions, allocate

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridLock — Event Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & DATA (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        return {
            "xgb_clf":      joblib.load(MODEL_DIR / "xgb_clf.pkl"),
            "lgb_clf":      joblib.load(MODEL_DIR / "lgb_clf.pkl"),
            "xgb_reg":      joblib.load(MODEL_DIR / "xgb_reg.pkl"),
            "lgb_reg":      joblib.load(MODEL_DIR / "lgb_reg.pkl"),
            "feature_cols": joblib.load(MODEL_DIR / "feature_cols.pkl"),
        }
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    if DATA_PROC.exists():
        return pd.read_csv(DATA_PROC)
    return None

models = load_models()
df_proc = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🚦 GridLock — Event-Driven Congestion Intelligence")
st.caption("Bengaluru traffic event prediction · Resource optimisation · Diversion planning")

if models is None:
    st.warning("⚠ Models not found. Run `python src/train.py` first, then refresh.")
    st.stop()

if df_proc is None:
    st.warning("⚠ Processed data not found. Run `python src/preprocess.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — event input
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📋 New Event")

    event_cause = st.selectbox("Event cause", [
        "vehicle_breakdown", "congestion", "accident", "construction",
        "water_logging", "tree_fall", "public_event", "procession",
        "vip_movement", "protest", "road_conditions", "pot_holes", "others",
    ])
    event_type = st.selectbox("Event type", ["unplanned", "planned"])
    veh_type   = st.selectbox("Vehicle type", [
        "none", "heavy_vehicle", "truck", "bmtc_bus", "ksrtc_bus",
        "private_bus", "lcv", "private_car", "taxi", "auto", "others",
    ])
    corridor = st.selectbox("Corridor", [
        "Mysore Road", "Bellary Road 1", "Bellary Road 2", "Tumkur Road",
        "Hosur Road", "ORR North 1", "ORR North 2", "ORR East 1", "ORR East 2",
        "Old Madras Road", "Magadi Road", "Bannerghata Road", "Non-corridor",
    ])
    zone = st.selectbox("Zone", [
        "Central Zone 1", "Central Zone 2",
        "North Zone 1", "North Zone 2",
        "South Zone 1", "South Zone 2",
        "East Zone 1", "East Zone 2",
        "West Zone 1", "West Zone 2",
    ])
    road_closure = st.checkbox("Requires road closure?", value=False)
    event_date   = st.date_input("Date", datetime.today())
    event_time   = st.time_input("Start time (IST)", datetime.strptime("08:30", "%H:%M").time())

    st.divider()
    st.subheader("Resources available")
    total_p = st.number_input("Police personnel", 10, 300, 80)
    total_b = st.number_input("Barricades",        5, 150, 40)

    predict_btn = st.button("🔍 Predict & Optimise", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Congestion Map", "🚔 Resource Plan", "📊 Model Performance", "🔎 Dataset Explorer"
])

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    CAUSE_SEVERITY, CAUSE_CLOSURE_PROB, VEH_SEVERITY, CITY_LAT, CITY_LON
)

def build_input_row(
    event_cause, event_type, veh_type, corridor, zone,
    road_closure, event_date, event_time,
):
    """Build a single-row feature DataFrame matching training schema."""
    from config import IST_OFFSET
    dt = datetime.combine(event_date, event_time)
    hour = dt.hour
    dow  = dt.weekday()
    month = dt.month

    # Corridor frequency from training data
    corridor_counts = df_proc["corridor"].value_counts()
    zone_rate       = df_proc.groupby("zone")["priority_class"].mean()
    global_rate     = df_proc["priority_class"].mean()

    row = {
        "hour_ist":                hour,
        "day_of_week":             dow,
        "month":                   month,
        "is_weekend":              int(dow in [5, 6]),
        "is_morning_rush":         int(hour in range(7, 11)),
        "is_evening_rush":         int(hour in range(17, 21)),
        "is_peak":                 int(hour in list(range(7,11)) + list(range(17,21))),
        "is_night":                int(hour in [22, 23, 0, 1, 2, 3, 4]),
        "hour_sin":                np.sin(2 * np.pi * hour / 24),
        "hour_cos":                np.cos(2 * np.pi * hour / 24),
        "dow_sin":                 np.sin(2 * np.pi * dow / 7),
        "dow_cos":                 np.cos(2 * np.pi * dow / 7),
        "cause_severity":          CAUSE_SEVERITY.get(event_cause, 1),
        "cause_closure_prob":      CAUSE_CLOSURE_PROB.get(event_cause, 0.05),
        "veh_severity":            VEH_SEVERITY.get(veh_type, 0),
        "is_planned":              int(event_type == "planned"),
        "has_vehicle":             int(veh_type != "none"),
        "requires_road_closure_int": int(road_closure),
        "is_named_corridor":       int(corridor != "Non-corridor"),
        "corridor_freq":           corridor_counts.get(corridor, 1),
        "zone_priority_rate":      zone_rate.get(zone, global_rate),
        "has_junction":            0,
        "dist_from_center_km":     0.0,
        "corridor_density_2h":     0,
        "zone_density_2h":         0,
    }

    feat_cols = models["feature_cols"]
    X = pd.DataFrame([row])[feat_cols].fillna(0)
    return X

if predict_btn:
    X_input = build_input_row(
        event_cause, event_type, veh_type, corridor, zone,
        road_closure, event_date, event_time,
    )

    feat_cols = models["feature_cols"]
    p_proba = 0.5 * models["xgb_clf"].predict_proba(X_input)[:, 1] + \
              0.5 * models["lgb_clf"].predict_proba(X_input)[:, 1]
    p_class = int(p_proba[0] >= 0.5)

    r_pred = np.clip(
        0.5 * models["xgb_reg"].predict(X_input) +
        0.5 * models["lgb_reg"].predict(X_input), 0, 10,
    )[0]

    st.session_state["prediction"] = {
        "priority":     "High" if p_class else "Low",
        "proba":        float(p_proba[0]),
        "impact_score": float(r_pred),
        "corridor":     corridor,
        "zone":         zone,
        "cause":        event_cause,
        "time":         event_time.strftime("%H:%M"),
    }

    # ── Aggregate predictions over ALL corridors for resource map ────────────
    all_preds = df_proc.copy()
    feat_df = all_preds[[c for c in feat_cols if c in all_preds.columns]].fillna(0)
    all_preds["impact_score_pred"] = np.clip(
        0.5 * models["xgb_reg"].predict(feat_df) +
        0.5 * models["lgb_reg"].predict(feat_df), 0, 10,
    )
    all_preds["priority_class_pred"] = (
        0.5 * models["xgb_clf"].predict_proba(feat_df)[:, 1] +
        0.5 * models["lgb_clf"].predict_proba(feat_df)[:, 1] >= 0.5
    ).astype(int)
    all_preds["requires_road_closure_int"] = df_proc.get("requires_road_closure_int", 0)

    agg = aggregate_predictions(all_preds.assign(impact_score=all_preds["impact_score_pred"]), "corridor")
    result = allocate(agg, total_personnel=total_p, total_barricades=total_b)
    st.session_state["resource_plan"] = result
    st.session_state["all_preds"]     = all_preds


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Congestion map
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        col1, col2, col3, col4 = st.columns(4)
        risk_color = "🔴" if pred["priority"] == "High" else "🟡"
        col1.metric("Priority",     f"{risk_color} {pred['priority']}")
        col2.metric("Confidence",   f"{pred['proba']:.0%}")
        col3.metric("Impact score", f"{pred['impact_score']:.1f} / 10")
        col4.metric("Corridor",     pred["corridor"])

        # ── Folium map: show all event locations coloured by predicted impact ──
        ap = st.session_state["all_preds"]
        ap_geo = ap[(ap["latitude"].between(12.5, 13.5)) & (ap["longitude"].between(77.0, 78.0))]

        m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="CartoDB positron")

        def impact_color(score):
            if score >= 7: return "#dc3545"
            if score >= 5: return "#fd7e14"
            if score >= 3: return "#ffc107"
            return "#28a745"

        for _, row in ap_geo.sample(min(800, len(ap_geo)), random_state=42).iterrows():
            sc = row.get("impact_score_pred", 0)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=max(3, sc * 1.2),
                color=impact_color(sc),
                fill=True,
                fill_opacity=0.45,
                popup=folium.Popup(
                    f"<b>{row.get('event_cause','—')}</b><br>"
                    f"Corridor: {row.get('corridor','—')}<br>"
                    f"Impact: {sc:.1f}<br>"
                    f"Road closure: {row.get('requires_road_closure','—')}",
                    max_width=200,
                ),
            ).add_to(m)

        st_folium(m, width=None, height=500, returned_objects=[])

        # ── Timeline: hourly event count from processed data ──────────────────
        if "hour_ist" in df_proc.columns:
            hourly = df_proc.groupby("hour_ist").size().reset_index(name="count")
            fig = px.bar(hourly, x="hour_ist", y="count", title="Event frequency by hour (IST)",
                         labels={"hour_ist": "Hour (IST)", "count": "Event count"})
            fig.add_vrect(x0=7, x1=11, fillcolor="orange", opacity=0.1, annotation_text="Morning rush")
            fig.add_vrect(x0=17, x1=21, fillcolor="red",   opacity=0.1, annotation_text="Evening rush")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Configure an event in the sidebar and click **Predict & Optimise**.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Resource plan
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    if "resource_plan" in st.session_state:
        plan = st.session_state["resource_plan"]
        st.subheader("Optimal deployment — by corridor")
        display_cols = ["label", "impact_score", "event_count",
                        "high_priority_count", "allocated_personnel",
                        "allocated_barricades", "risk_level", "priority_rank"]
        show = plan[[c for c in display_cols if c in plan.columns]].copy()
        show["impact_score"] = show["impact_score"].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(plan.head(10), x="label", y="allocated_personnel",
                         color="impact_score", color_continuous_scale="Reds",
                         title="Personnel allocation (top 10 corridors)")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(plan.head(10), x="label", y="allocated_barricades",
                         color="impact_score", color_continuous_scale="Oranges",
                         title="Barricade allocation (top 10 corridors)")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = plan.to_csv(index=False)
        st.download_button("⬇ Download deployment plan (CSV)", data=csv,
                           file_name="resource_plan.csv", mime="text/csv")
    else:
        st.info("Run prediction first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model performance
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Evaluation on chronological test split (last 20% of events by date)")
    st.caption("Metrics are computed at inference time on held-out data.")

    if df_proc is not None and models is not None:
        feat_cols = models["feature_cols"]
        df_sorted = df_proc.sort_values("start_datetime").reset_index(drop=True)
        split = int(len(df_sorted) * 0.8)
        df_test = df_sorted.iloc[split:]

        X_test = df_test[[c for c in feat_cols if c in df_test.columns]].fillna(0)
        y_cls_true = df_test[TARGET_CLS]

        cls_proba = 0.5 * models["xgb_clf"].predict_proba(X_test)[:, 1] + \
                    0.5 * models["lgb_clf"].predict_proba(X_test)[:, 1]
        cls_pred  = (cls_proba >= 0.5).astype(int)

        # Regression: only evaluate on rows that have a valid duration (38% of data)
        reg_mask        = df_test[TARGET_REG].notna()
        X_test_reg      = X_test[reg_mask]
        y_reg_true      = df_test.loc[reg_mask, TARGET_REG]
        reg_pred_all    = np.clip(
            0.5 * models["xgb_reg"].predict(X_test) +
            0.5 * models["lgb_reg"].predict(X_test), 0, None,
        )
        reg_pred        = reg_pred_all[reg_mask]

        from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, r2_score
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 (road closure)",   f"{f1_score(y_cls_true, cls_pred):.3f}")
        col2.metric("ROC-AUC",             f"{roc_auc_score(y_cls_true, cls_proba):.3f}")
        col3.metric("MAE (log duration)",  f"{mean_absolute_error(y_reg_true, reg_pred):.3f}")
        col4.metric("R² (log duration)",   f"{r2_score(y_reg_true, reg_pred):.3f}")
        st.caption(f"Regression metrics computed on {reg_mask.sum()} / {len(df_test)} test rows that have a closed_datetime.")

        # Actual vs predicted scatter (regression subset only)
        fig = px.scatter(
            x=y_reg_true, y=reg_pred,
            labels={"x": "Actual log duration", "y": "Predicted log duration"},
            title="Actual vs Predicted log duration (test rows with closed_datetime)",
            opacity=0.4, color_discrete_sequence=["#3B82F6"],
        )
        fig.add_shape(type="line",
                      x0=float(y_reg_true.min()), y0=float(y_reg_true.min()),
                      x1=float(y_reg_true.max()), y1=float(y_reg_true.max()),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        # Error histogram
        errors = pd.Series(reg_pred - y_reg_true.values, name="error")
        fig2 = px.histogram(errors, title="Prediction error distribution (log duration)",
                             labels={"value": "error (pred − actual)"}, nbins=40)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Dataset explorer
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Raw dataset overview")
    if df_proc is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total events",     f"{len(df_proc):,}")
        col2.metric("Planned events",   f"{df_proc.get('is_planned', pd.Series(0)).sum():,}")
        col3.metric("Road closures",    f"{df_proc.get('requires_road_closure_int', pd.Series(0)).sum():,}")

        # Cause breakdown
        cause_agg = df_proc.groupby("event_cause").agg(
            count=("event_cause","count"),
            high_prio_rate=("priority_class","mean"),
        ).sort_values("count", ascending=False).reset_index()
        fig = px.bar(cause_agg, x="event_cause", y="count",
                     color="high_prio_rate", color_continuous_scale="RdYlGn_r",
                     title="Event count by cause (colour = High-priority rate)")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        # Corridor breakdown
        corr_agg = df_proc.groupby("corridor").agg(
            count=("corridor","count"),
        ).sort_values("count", ascending=False).head(15).reset_index()
        fig2 = px.bar(corr_agg, x="corridor", y="count",
                      title="Top 15 corridors by event volume")
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

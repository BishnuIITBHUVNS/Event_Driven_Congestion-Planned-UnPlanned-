"""
app.py — Streamlit dashboard (flat repo structure).
Run: streamlit run app.py
"""

import sys
from pathlib import Path
# Flat repo: config.py and optimizer.py are in the same folder as app.py
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import folium
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
from datetime import datetime

from config import (
    DATA_PROC, MODEL_DIR, FEATURE_COLS, TARGET_CLS, TARGET_REG,
    CAUSE_SEVERITY, CAUSE_CLOSURE_PROB, VEH_SEVERITY, CITY_LAT, CITY_LON,
)
from optimizer import aggregate_predictions, allocate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridLock — Event Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
)

# ── Load models & data (cached) ───────────────────────────────────────────────
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
    except FileNotFoundError as e:
        return None

@st.cache_data
def load_data():
    if DATA_PROC.exists():
        return pd.read_csv(DATA_PROC)
    return None

models  = load_models()
df_proc = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 GridLock — Event-Driven Congestion Intelligence")
st.caption("Bengaluru traffic event prediction · Resource optimisation · Diversion planning")

if models is None:
    st.error("❌ Model files not found. Make sure xgb_clf.pkl, lgb_clf.pkl, xgb_reg.pkl, lgb_reg.pkl, feature_cols.pkl are in the repo root.")
    st.stop()

if df_proc is None:
    st.error("❌ events_features.csv not found in repo root.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
        "North Zone 1",   "North Zone 2",
        "South Zone 1",   "South Zone 2",
        "East Zone 1",    "East Zone 2",
        "West Zone 1",    "West Zone 2",
    ])
    road_closure = st.checkbox("Requires road closure?", value=False)
    event_date   = st.date_input("Date", datetime.today())
    event_time   = st.time_input("Start time (IST)", datetime.strptime("08:30", "%H:%M").time())

    st.divider()
    st.subheader("Resources available")
    total_p = st.number_input("Police personnel", 10, 300, 80)
    total_b = st.number_input("Barricades",        5, 150, 40)

    predict_btn = st.button("🔍 Predict & Optimise", use_container_width=True, type="primary")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Congestion Map", "🚔 Resource Plan", "📊 Model Performance", "🔎 Dataset Explorer"
])

# ── Build feature row from sidebar input ──────────────────────────────────────
def build_input_row(event_cause, event_type, veh_type, corridor, zone,
                    road_closure, event_date, event_time):
    dt    = datetime.combine(event_date, event_time)
    hour  = dt.hour
    dow   = dt.weekday()
    month = dt.month

    corridor_counts = df_proc["corridor"].value_counts()
    zone_rate       = df_proc.groupby("zone")["road_closure_class"].mean()
    global_rate     = df_proc["road_closure_class"].mean()

    row = {
        "hour_ist":           hour,
        "day_of_week":        dow,
        "month":              month,
        "is_weekend":         int(dow in [5, 6]),
        "is_morning_rush":    int(hour in range(7, 11)),
        "is_evening_rush":    int(hour in range(17, 21)),
        "is_peak":            int(hour in list(range(7,11)) + list(range(17,21))),
        "is_night":           int(hour in [22, 23, 0, 1, 2, 3, 4]),
        "hour_sin":           np.sin(2 * np.pi * hour / 24),
        "hour_cos":           np.cos(2 * np.pi * hour / 24),
        "dow_sin":            np.sin(2 * np.pi * dow / 7),
        "dow_cos":            np.cos(2 * np.pi * dow / 7),
        "cause_severity":     CAUSE_SEVERITY.get(event_cause, 1),
        "cause_closure_prob": CAUSE_CLOSURE_PROB.get(event_cause, 0.05),
        "veh_severity":       VEH_SEVERITY.get(veh_type, 0),
        "is_planned":         int(event_type == "planned"),
        "has_vehicle":        int(veh_type != "none"),
        "is_named_corridor":  int(corridor != "Non-corridor"),
        "corridor_freq":      corridor_counts.get(corridor, 1),
        "zone_priority_rate": zone_rate.get(zone, global_rate),
        "has_junction":       0,
        "dist_from_center_km":0.0,
        "corridor_density_2h":0,
        "zone_density_2h":    0,
    }
    feat_cols = models["feature_cols"]
    return pd.DataFrame([row])[feat_cols].fillna(0)

# ── Predict on button click ───────────────────────────────────────────────────
if predict_btn:
    X_input = build_input_row(
        event_cause, event_type, veh_type, corridor, zone,
        road_closure, event_date, event_time,
    )
    feat_cols = models["feature_cols"]

    p_proba = (0.5 * models["xgb_clf"].predict_proba(X_input)[:, 1] +
               0.5 * models["lgb_clf"].predict_proba(X_input)[:, 1])
    r_pred  = float(np.clip(
        0.5 * models["xgb_reg"].predict(X_input) +
        0.5 * models["lgb_reg"].predict(X_input), 0, None,
    )[0])

    st.session_state["prediction"] = {
        "closure":      "Yes ⚠️" if p_proba[0] >= 0.5 else "No ✅",
        "proba":        float(p_proba[0]),
        "log_duration": r_pred,
        "est_minutes":  round(np.expm1(r_pred)),
        "corridor":     corridor,
        "zone":         zone,
    }

    # Run predictions over full dataset for map & optimizer
    all_preds = df_proc.copy()
    feat_df   = all_preds[[c for c in feat_cols if c in all_preds.columns]].fillna(0)
    all_preds["impact_score"] = np.clip(
        0.5 * models["xgb_reg"].predict(feat_df) +
        0.5 * models["lgb_reg"].predict(feat_df), 0, None,
    )
    all_preds["priority_class"] = (
        0.5 * models["xgb_clf"].predict_proba(feat_df)[:, 1] +
        0.5 * models["lgb_clf"].predict_proba(feat_df)[:, 1] >= 0.5
    ).astype(int)

    agg    = aggregate_predictions(all_preds, "corridor")
    result = allocate(agg, total_personnel=total_p, total_barricades=total_b)
    st.session_state["resource_plan"] = result
    st.session_state["all_preds"]     = all_preds

# ── Tab 1: Congestion Map ─────────────────────────────────────────────────────
with tab1:
    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Road closure predicted", pred["closure"])
        c2.metric("Closure probability",    f"{pred['proba']:.0%}")
        c3.metric("Est. resolution time",   f"{pred['est_minutes']} min")
        c4.metric("Corridor",               pred["corridor"])

        ap     = st.session_state["all_preds"]
        ap_geo = ap[ap["latitude"].between(12.5, 13.5) & ap["longitude"].between(77.0, 78.0)]

        m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="CartoDB positron")

        def impact_color(s):
            if s >= 7: return "#dc3545"
            if s >= 5: return "#fd7e14"
            if s >= 3: return "#ffc107"
            return "#28a745"

        for _, row in ap_geo.sample(min(800, len(ap_geo)), random_state=42).iterrows():
            sc = row.get("impact_score", 0)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=max(3, sc * 1.2),
                color=impact_color(sc),
                fill=True, fill_opacity=0.45,
                popup=folium.Popup(
                    f"<b>{row.get('event_cause','—')}</b><br>"
                    f"Corridor: {row.get('corridor','—')}<br>"
                    f"Score: {sc:.1f}",
                    max_width=200,
                ),
            ).add_to(m)

        st_folium(m, width=None, height=500, returned_objects=[])

        if "hour_ist" in df_proc.columns:
            hourly = df_proc.groupby("hour_ist").size().reset_index(name="count")
            fig = px.bar(hourly, x="hour_ist", y="count",
                         title="Event frequency by hour (IST)",
                         labels={"hour_ist": "Hour (IST)", "count": "Event count"})
            fig.add_vrect(x0=7,  x1=11, fillcolor="orange", opacity=0.1, annotation_text="Morning rush")
            fig.add_vrect(x0=17, x1=21, fillcolor="red",    opacity=0.1, annotation_text="Evening rush")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Configure an event in the sidebar and click **Predict & Optimise**.")

# ── Tab 2: Resource Plan ──────────────────────────────────────────────────────
with tab2:
    if "resource_plan" in st.session_state:
        plan = st.session_state["resource_plan"]
        st.subheader("Optimal deployment — by corridor")
        show_cols = ["label", "impact_score", "event_count", "high_priority_count",
                     "allocated_personnel", "allocated_barricades", "risk_level", "priority_rank"]
        show = plan[[c for c in show_cols if c in plan.columns]].copy()
        show["impact_score"] = show["impact_score"].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(plan.head(10), x="label", y="allocated_personnel",
                         color="impact_score", color_continuous_scale="Reds",
                         title="Personnel allocation (top 10 corridors)")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(plan.head(10), x="label", y="allocated_barricades",
                         color="impact_score", color_continuous_scale="Oranges",
                         title="Barricade allocation (top 10 corridors)")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        csv = plan.to_csv(index=False)
        st.download_button("⬇ Download deployment plan", data=csv,
                           file_name="resource_plan.csv", mime="text/csv")
    else:
        st.info("Run prediction first.")

# ── Tab 3: Model Performance ──────────────────────────────────────────────────
with tab3:
    st.subheader("Evaluation on chronological test split (last 20% of events by date)")
    if df_proc is not None and models is not None:
        feat_cols = models["feature_cols"]
        df_sorted = df_proc.sort_values("start_datetime").reset_index(drop=True)
        split     = int(len(df_sorted) * 0.8)
        df_test   = df_sorted.iloc[split:]

        X_test     = df_test[[c for c in feat_cols if c in df_test.columns]].fillna(0)
        y_cls_true = df_test[TARGET_CLS]

        cls_proba = (0.5 * models["xgb_clf"].predict_proba(X_test)[:, 1] +
                     0.5 * models["lgb_clf"].predict_proba(X_test)[:, 1])
        cls_pred  = (cls_proba >= 0.5).astype(int)

        reg_mask   = df_test[TARGET_REG].notna()
        y_reg_true = df_test.loc[reg_mask, TARGET_REG]
        reg_pred   = np.clip(
            0.5 * models["xgb_reg"].predict(X_test[reg_mask]) +
            0.5 * models["lgb_reg"].predict(X_test[reg_mask]), 0, None,
        )

        from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, r2_score
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 (road closure)",  f"{f1_score(y_cls_true, cls_pred):.3f}")
        c2.metric("ROC-AUC",            f"{roc_auc_score(y_cls_true, cls_proba):.3f}")
        c3.metric("MAE (log duration)", f"{mean_absolute_error(y_reg_true, reg_pred):.3f}")
        c4.metric("R² (log duration)",  f"{r2_score(y_reg_true, reg_pred):.3f}")
        st.caption(f"Regression on {reg_mask.sum()} / {len(df_test)} test rows with closed_datetime.")

        fig = px.scatter(x=y_reg_true, y=reg_pred, opacity=0.4,
                         labels={"x": "Actual log duration", "y": "Predicted"},
                         title="Actual vs Predicted log duration",
                         color_discrete_sequence=["#3B82F6"])
        fig.add_shape(type="line",
                      x0=float(y_reg_true.min()), y0=float(y_reg_true.min()),
                      x1=float(y_reg_true.max()), y1=float(y_reg_true.max()),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        errors = pd.Series(reg_pred - y_reg_true.values, name="error")
        fig2 = px.histogram(errors, nbins=40,
                            title="Prediction error distribution",
                            labels={"value": "error (pred − actual)"})
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: Dataset Explorer ───────────────────────────────────────────────────
with tab4:
    st.subheader("Raw dataset overview")
    if df_proc is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total events",   f"{len(df_proc):,}")
        c2.metric("Planned events", f"{int(df_proc.get('is_planned', pd.Series(0)).sum())}")
        c3.metric("Road closures",  f"{int(df_proc.get('road_closure_class', pd.Series(0)).sum())}")

        cause_agg = df_proc.groupby("event_cause").agg(
            count=("event_cause","count"),
            closure_rate=("road_closure_class","mean"),
        ).sort_values("count", ascending=False).reset_index()
        fig = px.bar(cause_agg, x="event_cause", y="count",
                     color="closure_rate", color_continuous_scale="RdYlGn_r",
                     title="Event count by cause (colour = road closure rate)")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        corr_agg = (df_proc.groupby("corridor").size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False).head(15))
        fig2 = px.bar(corr_agg, x="corridor", y="count",
                      title="Top 15 corridors by event volume")
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

# Event-Driven-Congestion_Planned-UnPlanned
# GridLock - Event-Driven Congestion Intelligence

A prototype built for the Flipkart Gridlock Hackathon, addressing the "Event-Driven Congestion (Planned & Unplanned)" theme. It predicts whether a traffic event will require a road closure, estimates how long it'll take to clear, and recommends how to split limited police personnel and barricades across active corridors  all from a single event description.

## The problem we're solving

Traffic control rooms in Bengaluru currently respond to events after congestion has already built up. A VIP movement, a protest, debris on the road, a fallen tree - by the time enforcement is deployed, the gridlock has already formed. There's no system today that looks at an event before it escalates and tells you: is this going to be serious, how long will it last, and where should resources go first. That's the gap this project tries to close.

## What's actually in here

This isn't a notebook with some plots in it. It's a full pipeline: raw data goes in one end, a working Streamlit dashboard comes out the other.

- **preprocess.py** takes the raw event log and turns it into 24 usable features - time-of-day encoding, empirical severity scores per event cause, corridor and zone statistics, rolling event density.
- **train.py** trains an XGBoost + LightGBM ensemble for two tasks: will this event require a road closure (classification), and how long will it take to resolve (regression). Both are evaluated on a chronological train/test split, not a random one, because random splits let the model peek at the future.
- **optimizer.py** wraps Google OR-Tools' CP-SAT solver to allocate a fixed personnel and barricade budget across corridors, maximizing coverage of the highest-predicted-impact zones while guaranteeing minimum coverage for anything flagged critical.
- **app.py** is the live dashboard - describe a new event in the sidebar, get a prediction, see it plotted against historical events on that corridor, see the optimal resource split, and check the model's actual test-set performance, all in one place.

## A finding worth being upfront about

The first version of this project trained directly on the dataset's `priority` label, and the results looked almost too good. Running a correlation check explained why: `priority` turned out to be 99.6% determined by a single column - whether the event happened on a named corridor or not. The model wasn't learning anything; it was just reading off a lookup table that was already baked into the data.

We re-targeted the classifier on `requires_road_closure` instead - a genuinely harder target, only 8.3% positive, and a far more actionable signal for a control room deciding whether to deploy barricades. The regression target moved from a synthetic composite score to actual incident duration. The numbers got less flattering, but they got real.

## Honest performance numbers

On a held-out chronological test set (most recent 20% of events by date):

- Road closure classification - F1 around 0.78, ROC-AUC around 0.72
- Duration regression - R² around 0.81, MAPE around 35%

These aren't competition-winning numbers, and we're not going to pretend otherwise. The dataset has about 8,000 rows, with an 8 per cent positive class rate for the hardest target, which is a tough place to start. What we'd point to instead is the pipeline itself - the chronological split, the leakage catch, the constraint-based optimiser, the fact that every number in the dashboard is computed live rather than hardcoded.

## Running it locally

Clone the repo, then from inside it:

```
pip install -r requirements.txt
streamlit run app.py
```

That's it. The trained models and processed dataset are already included, so the app runs immediately without needing a training step.

If you want to regenerate everything from the raw data:

```
python preprocess.py
python train.py
```

This rebuilds the feature set and retrains both models from scratch, overwriting the saved files.

## Repository layout

```
app.py                  the Streamlit dashboard
config.py                all constants, paths, and domain lookup tables
optimizer.py              OR-Tools resource allocation logic
preprocess.py             raw data to feature pipeline
train.py                  model training and evaluation
requirements.txt          dependencies
events_features.csv       processed dataset (output of preprocess.py)
xgb_clf.pkl / lgb_clf.pkl  classification models
xgb_reg.pkl / lgb_reg.pkl  regression models
feature_cols.pkl          exact feature ordering used at inference time
```

## What's not built yet

Diversion routing is scaffolded in the architecture but not wired into the app - the plan is to use OSMnx and a shortest-path algorithm over the road graph once a corridor is flagged for closure, so the system can suggest an actual alternate route rather than just flagging the problem. Real-time data ingestion (live camera feeds, GPS speed data) is also not connected; everything currently runs against the static historical dataset.

## Data

Built on an anonymized Bengaluru traffic event dataset covering November 2023 through April 2024, provided as part of the hackathon.

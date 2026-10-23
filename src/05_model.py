"""
Phase 4 — Can we predict which appointments will fail?

Run:  .venv\\Scripts\\python.exe src/05_model.py
"""
# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
pd.set_option("display.width", 200)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")

all_jobs = sa[
    ~sa["is_placeholder"] & sa["in_demand_window"]
].copy()

FINAL_STATUSES = ["Completed", "CannotComplete", "Canceled"]

jobs = all_jobs.loc[
    all_jobs["Status"].isin(FINAL_STATUSES)
    & all_jobs["region"].notna()
].copy()

print("MODEL POPULATION CHECK")
print(f"Operational appointments: {len(all_jobs):,}")
print(f"Appointments included in model: {len(jobs):,}")
print(f"Excluded from model: {len(all_jobs) - len(jobs):,}")

# %%
# --- Features the planner would actually have ------------------------
# Only things known WHEN THE JOB IS BOOKED. Anything recorded during or
# after the visit would leak the answer: Sub_Categories__c IS the failure
# reason, so a model using it would score beautifully and be worthless.
jobs["work_type_grp"] = np.where(
    jobs["Work Type: Work Type Name"].str.startswith("Inspect & Fit", na=False),
    "Inspect & Fit",
    jobs["Work Type: Work Type Name"],
)
jobs["is_retail_str"] = jobs["Is Retail"].astype(str)

jobs["booked_hour"] = (
    jobs["Booked Slot Start Time"].dt.hour
)

jobs["scheduled_dow"] = (
    jobs["Booked Slot Start Time"].dt.day_name()
)

# Conservative feature selection:
# Exclude FSL__Efficiency__c because its availability at booking
# time has not been established.
#
# Exclude own_territory because technician assignments may change
# after a booking is created.
#
# Keep only candidate planning-time features. Their actual
# historical availability should be confirmed before deployment.

NUMERIC = [
    "Lead Time",
    "Number of Tyres",
    "planned_minutes",
    "booked_hour",
]

CATEGORICAL = [
    "work_type_grp",
    "region",
    "scheduled_dow",
    "is_retail_str",
    "working_hours",
]
# %%
# --- Split by TIME, not at random ------------------------------------
# A random split lets the model learn from December to predict November.
# In real use you only ever have the past, so train on Oct-Nov, test on Dec.
train = jobs[jobs["effective_date"] < "2024-12-01"]
test = jobs[jobs["effective_date"] >= "2024-12-01"]

print(f"train {len(train):,} (Oct-Nov)   test {len(test):,} (Dec)")
print(f"failure rate   train {train['failed'].mean():.1%}   test {test['failed'].mean():.1%}")
print(
    "This is a retrospective time-based test. "
    "Differences in failure rates between periods may affect performance."
)
# %%
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=30))]), CATEGORICAL),
])

model = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=3000))])
model.fit(train[NUMERIC + CATEGORICAL], train["failed"])

risk = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
baseline = test["failed"].mean()

print("\n" + "=" * 78)
print("HOW GOOD IS IT?")
print("=" * 78)
print(f"AUC: {roc_auc_score(test['failed'], risk):.3f}")
print(
    "AUC evaluates discrimination between failed and non-failed "
    "appointments. Operational usefulness also depends on "
    "precision, calibration, intervention costs and capacity."
)
# %%
# --- The number a planner actually cares about ------------------------
# Not AUC. "If I check the riskiest N jobs, how many really fail?"
scored = test.assign(risk=risk)
days = scored["effective_date"].nunique()

print("\n" + "=" * 78)
print("RETROSPECTIVE DECEMBER RISK RANKING - FULL TEST PERIOD")
print("=" * 78)
print(f"(December: {len(scored):,} appointments over {days} days, "
      f"baseline failure rate {baseline:.1%})\n")

for pct in [5, 10, 20, 30]:
    n = int(len(scored) * pct / 100)
    top = scored.nlargest(n, "risk")
    print(f"  top {pct:>2}%  {n / days:>5.0f} jobs/day  "
          f"precision {top['failed'].mean():>5.1%}  "
          f"lift {top['failed'].mean() / baseline:.2f}x  "
          f"catches {top['failed'].sum() / scored['failed'].sum():>4.0%} of failures")

# %%
# --- Can it find the expensive failure mode? -------------------------
# Wrong parts ordered sends a technician out 95% of the time. If anything
# is worth predicting, it is this one.
print("\n" + "=" * 78)
print("THE SAME MODEL, AIMED AT SPECIFIC FAILURE TYPES")
print("=" * 78)

STOCK = ["Wrong parts ordered", "Tyres not in Stock",
         "Locking Wheel Nut Unavailable", "Stock too far for pickup",
         "Centre refused to release stock"]

targets = {
    "any failure": ("failed", None),
    "stock failure (any)": (None, STOCK),
    "wrong parts ordered": (None, ["Wrong parts ordered"]),
}

results = []
for label, (col, cats) in targets.items():
    if col:
        y_tr, y_te = train[col], test[col]
    else:
        y_tr = train["Sub_Categories__c"].isin(cats).astype(int)
        y_te = test["Sub_Categories__c"].isin(cats).astype(int)

    m = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=3000))])
    m.fit(train[NUMERIC + CATEGORICAL], y_tr)
    p = m.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]

    base = y_te.mean()
    t = test.assign(p=p, y=y_te)
    top = t.nlargest(int(len(t) * 0.10), "p")
    results.append({
        "target": label,
        "base_rate": f"{base:.1%}",
        "AUC": round(roc_auc_score(y_te, p), 3),
        "top10pct_precision": f"{top['y'].mean():.1%}",
        "lift": round(top["y"].mean() / base, 2),
    })

results = pd.DataFrame(results)
print(results.to_string(index=False))
results.to_csv(TABLES / "model_results.csv", index=False)

print(
    "\nMODEL INTERPRETATION: Performance should be assessed "
    "against the operational value of acting on predictions."
)

print(
    "Weak discrimination with the available features does not prove "
    "that failure is inherently unpredictable."
)

print(
    "A pre-dispatch readiness check is a practical alternative "
    "worth testing, especially for parts-dependent appointments."
)
# %%
# --- What does the model think drives failure? ------------------------
names = model.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()

print("\n" + "=" * 78)
print("WHAT PUSHES RISK UP AND DOWN")
print("=" * 78)
print("\nraises risk:")
print(coefs.tail(6).iloc[::-1].round(3).to_string())
print("\nlowers risk:")
print(coefs.head(6).round(3).to_string())
coefs.to_csv(TABLES / "model_coefficients.csv")
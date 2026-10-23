from pathlib import Path
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

sa = pd.read_parquet(PROCESSED / "appointments.parquet")
jobs = sa[~sa["is_placeholder"] & sa["in_demand_window"]].copy()

daily = jobs.groupby("effective_date").agg(jobs=("Id", "size")).reset_index()
daily["dow"] = daily["effective_date"].dt.day_name()

train = daily[daily["effective_date"] < "2024-12-01"]
test = daily[daily["effective_date"] >= "2024-12-01"].copy()

dow_avg = train.groupby("dow")["jobs"].mean()
test["forecast"] = test["dow"].map(dow_avg)
test["error"] = test["jobs"] - test["forecast"]
test["abs_error"] = test["error"].abs()
test["pct_error"] = test["abs_error"] / test["jobs"] * 100

mae = test["abs_error"].mean()
mape = test["pct_error"].mean()
rmse = np.sqrt((test["error"]**2).mean())
persistence_mae = test["jobs"].diff().abs().mean()

print(f"Train days: {len(train)}  Test days: {len(test)}")
print("\nDay-of-week average (train):")
print(dow_avg.round(0).to_string())
print(f"\nMAE  {mae:.1f}   MAPE {mape:.1f}%   RMSE {rmse:.1f}")
print(f"vs. same-as-yesterday baseline MAE: {persistence_mae:.1f}")

test.to_csv(TABLES / "forecast_dow_naive.csv", index=False)
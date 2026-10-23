"""
Phase 2, Step 1 — Where is the demand?

Run:  python src/02_demand.py
Reads the analysis base built by 01_clean.py.
"""
# %%
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# %%
sa = pd.read_parquet(PROCESSED / "appointments.parquet")

# The analysis base: real customer jobs, dated inside the window.
jobs = sa[~sa["is_placeholder"] & sa["in_demand_window"]].copy()

print(f"Analysis base: {len(jobs):,} jobs")
print(f"  no territory matched: {jobs['customer_territory'].isna().sum():,} "
      f"({jobs['customer_territory'].isna().mean():.1%})")
print(f"  dates: {jobs['effective_date'].min():%d %b} to {jobs['effective_date'].max():%d %b %Y}")
print(f"  calendar days covered: {jobs['effective_date'].nunique()}")

# %%
# --- Demand measured two ways -----------------------------------------
# A tyre fitting takes about 28 minutes. An inspection takes about 180.
# Counting jobs treats those as equal demand. Counting hours does not.

demand = (
    jobs.dropna(subset=["customer_territory"])
    .groupby(["region", "customer_territory"], as_index=False)
    .agg(jobs=("Id", "size"), planned_min=("planned_minutes", "sum"))
)
demand["hours"] = (demand["planned_min"] / 60).round(0)
demand["avg_job_min"] = (demand["planned_min"] / demand["jobs"]).round(1)

# Rank each territory on both measures, then see how far it moves.
demand["rank_jobs"] = demand["jobs"].rank(ascending=False, method="min").astype(int)
demand["rank_hours"] = demand["hours"].rank(ascending=False, method="min").astype(int)
demand["rank_move"] = demand["rank_jobs"] - demand["rank_hours"]

demand = demand.sort_values("rank_hours")
demand.to_csv(TABLES / "demand_by_territory.csv", index=False)

print("\n" + "=" * 78)
print("TOP 15 TERRITORIES BY HOURS OF DEMAND")
print("=" * 78)
cols = ["customer_territory", "region", "jobs", "hours", "avg_job_min",
        "rank_jobs", "rank_hours"]
print(demand[cols].head(15).to_string(index=False))

# %%
# --- Where the two measures disagree most -----------------------------
print("\n" + "=" * 78)
print("BIGGEST RANK MOVES  (positive = busier than job count suggests)")
print("=" * 78)
movers = demand.loc[demand["jobs"] >= 200].copy()
movers = movers.reindex(movers["rank_move"].abs().sort_values(ascending=False).index)
print(movers[cols + ["rank_move"]].head(10).to_string(index=False))

# %%
# --- Region summary ---------------------------------------------------
print("\n" + "=" * 78)
print("BY REGION")
print("=" * 78)
region = (
    jobs.dropna(subset=["region"])
    .groupby("region", as_index=False)
    .agg(jobs=("Id", "size"), planned_min=("planned_minutes", "sum"),
         territories=("customer_territory", "nunique"))
)
region["hours"] = (region["planned_min"] / 60).round(0)
region["share_jobs"] = (region["jobs"] / region["jobs"].sum() * 100).round(1)
region["share_hours"] = (region["hours"] / region["hours"].sum() * 100).round(1)
print(region[["region", "territories", "jobs", "hours", "share_jobs", "share_hours"]]
      .sort_values("hours", ascending=False).to_string(index=False))

# %%
# --- The weekly shape -------------------------------------------------
# Divide by the number of that weekday in the window, so Mondays are
# compared with Mondays rather than with a count of calendar days.
print("\n" + "=" * 78)
print("DEMAND BY DAY OF WEEK")
print("=" * 78)
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
         "Saturday", "Sunday"]

dow = jobs.groupby("dow", as_index=False).agg(
    jobs=("Id", "size"), planned_min=("planned_minutes", "sum")
)
weekdays_in_window = (
    jobs.drop_duplicates("effective_date").groupby("dow").size().rename("n_days")
)
dow = dow.merge(weekdays_in_window, on="dow")
dow["jobs_per_day"] = (dow["jobs"] / dow["n_days"]).round(0)
dow["hours_per_day"] = (dow["planned_min"] / 60 / dow["n_days"]).round(0)
dow["dow"] = pd.Categorical(dow["dow"], categories=order, ordered=True)
print(dow.sort_values("dow").to_string(index=False))

# %%
# --- Trend across the quarter ----------------------------------------
# Use the week's start DATE, not its ISO number: ISO puts 30-31 December
# into week 1, which sorts to the front and makes December look like January.
print("\n" + "=" * 78)
print("BY WEEK  (partial = window starts or ends mid-week)")
print("=" * 78)

jobs["week_start"] = jobs["effective_date"] - pd.to_timedelta(
    jobs["effective_date"].dt.dayofweek, unit="D"
)

weekly = jobs.groupby("week_start", as_index=False).agg(
    jobs=("Id", "size"),
    planned_min=("planned_minutes", "sum"),
    days=("effective_date", "nunique"),
)
weekly["hours"] = (weekly["planned_min"] / 60).round(0)
weekly["partial"] = np.where(weekly["days"] < 7, "partial", "")
weekly["jobs_per_day"] = (weekly["jobs"] / weekly["days"]).round(0)

print(weekly[["week_start", "days", "jobs", "hours", "jobs_per_day", "partial"]]
      .to_string(index=False))
"""
Phase 2, Step 2 — Capacity: who is where, and what did they deliver?

Run:  .venv\\Scripts\\python.exe src/03_capacity.py
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

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")
roster = pd.read_parquet(PROCESSED / "tech_raw.parquet")

jobs = sa[~sa["is_placeholder"] & sa["in_demand_window"]].copy()
WINDOW_DAYS = (
    pd.Timestamp("2024-12-22")
    - pd.Timestamp("2024-10-08")
).days + 1

# %%
# --- Who on the roster actually worked? -------------------------------
# 199 technicians are listed. Not all of them appear in the appointments.
active_ids = set(jobs["ServiceResourceId"].dropna())
roster["active"] = roster["ServiceResourceId"].isin(active_ids)

print(f"Roster: {len(roster)} technicians")
print(f"  worked at least one job: {roster['active'].sum()}")
print(f"  no jobs at all:          {(~roster['active']).sum()}")

print("\nInactive technicians by territory:")
idle = (
    roster.loc[~roster["active"]]
    .groupby("Territory")
    .size()
    .sort_values(ascending=False)
)
print(idle.head(12).to_string())
print(f"\nterritories affected: {idle.size} of {roster['Territory'].nunique()}")

# %%
# --- OBSERVED WORK BY TECHNICIAN HOME TERRITORY ------------------------
# This measures appointments performed by technicians based in each
# territory, regardless of where the customer appointment occurred.
#
# These are observed activity measures, NOT available capacity.
# We do not have reliable scheduled hours, leave or availability data.

work = jobs.dropna(subset=["tech_territory"]).copy()
tech_days = (
    work.drop_duplicates(["ServiceResourceId", "effective_date"])
    .groupby("tech_territory")
    .size()
    .rename("tech_days")
)

capacity = work.groupby("tech_territory").agg(
    techs=("ServiceResourceId", "nunique"),
    delivered_jobs=("Id", "size"),
    delivered_min=("planned_minutes", "sum"),
)
capacity = capacity.join(tech_days)
capacity["delivered_hours"] = (capacity["delivered_min"] / 60).round(0)
capacity["jobs_per_tech_day"] = (capacity["delivered_jobs"] / capacity["tech_days"]).round(1)
capacity["hours_per_tech_day"] = (capacity["delivered_min"] / 60 / capacity["tech_days"]).round(1)
capacity["days_per_tech"] = (capacity["tech_days"] / capacity["techs"]).round(0)

# %%
# --- Demand side, by the CUSTOMER's territory -------------------------
demand = (
    jobs.dropna(subset=["customer_territory"])
    .groupby("customer_territory")
    .agg(demand_jobs=("Id", "size"), demand_min=("planned_minutes", "sum"))
)
demand["demand_hours"] = (demand["demand_min"] / 60).round(0)

# %%

# --- Compare customer demand with home-based technician activity ------
# Demand is attributed to CUSTOMER territory.
# Technician activity is attributed to TECHNICIAN HOME territory.
#
# Comparing them can reveal geographic differences, but does NOT
# measure local utilisation, spare capacity or service coverage.

view = demand[["demand_jobs", "demand_hours"]].join(
    capacity[
        [
            "techs",
            "tech_days",
            "delivered_jobs",
            "delivered_hours",
            "jobs_per_tech_day",
            "hours_per_tech_day",
            "days_per_tech",
        ]
    ],
    how="outer",
)

# Use explicit names so the two geographies are never confused.
view = view.rename(
    columns={
        "techs": "active_home_technicians",
        "tech_days": "home_tech_days",
        "delivered_jobs": "jobs_by_home_technicians",
        "delivered_hours": "hours_by_home_technicians",
        "jobs_per_tech_day": "jobs_per_home_tech_day",
        "hours_per_tech_day": "hours_per_home_tech_day",
        "days_per_tech": "days_per_home_technician",
    }
)

# Diagnostic only: customer demand relative to the number of
# active technicians based in that territory.
# This does not measure utilisation or spare capacity.
view["demand_hours_per_active_home_technician"] = (
    view["demand_hours"]
    / view["active_home_technicians"].replace(0, np.nan)
).round(1)

view = view.sort_values(
    "demand_hours_per_active_home_technician",
    ascending=False
)

# The previous "coverage" ratio has deliberately been removed.
assert "coverage" not in view.columns

view.to_csv(TABLES / "territory_capacity.csv")


print("\n" + "=" * 110)
print("HIGHEST DEMAND PER ACTIVE HOME-BASED TECHNICIAN (DIAGNOSTIC)")
print("=" * 110)
print(view.head(15).to_string())

print("\n" + "=" * 110)
print("LOWEST DEMAND PER ACTIVE HOME-BASED TECHNICIAN (DIAGNOSTIC)")
print("=" * 110)
print(
    view.dropna(subset=["active_home_technicians"])
    .tail(10)
    .to_string()
)

# %%

# --- Territories with demand but no active home-based technician -------
# No active home-based technician does not necessarily mean that
# nobody is assigned to the territory.
# An assigned technician may have had no observed appointments.

no_active_home = view.loc[
    view["demand_jobs"].notna()
    & (view["active_home_technicians"].fillna(0) == 0)
]

print(
    "\nTerritories with demand but no active home-based technician:",
    len(no_active_home)
)

if not no_active_home.empty:
    print(
        no_active_home[
            ["demand_jobs", "demand_hours"]
        ].to_string()
    )


# %%
# --- National picture --------------------------------------------------
print("\n" + "=" * 110)
print("NATIONAL")
print("=" * 110)
print(f"active technicians      : {work['ServiceResourceId'].nunique()}")
print(f"technician-days worked  : {len(work.drop_duplicates(['ServiceResourceId', 'effective_date'])):,}")
print(
    f"median observed appointment-days per technician: "
    f"{work.groupby('ServiceResourceId')['effective_date'].nunique().median():.0f} "
    f"within {WINDOW_DAYS} calendar days"
)
print(f"median jobs/tech-day    : {work.groupby(['ServiceResourceId', 'effective_date']).size().median():.1f}")
print(f"median hours/tech-day   : {work.groupby(['ServiceResourceId', 'effective_date'])['planned_minutes'].sum().median() / 60:.1f}")
# %%
# --- OBSERVED TERRITORY SERVICE COVERAGE -------------------------------
# Count technicians actually serving customers in each territory.
# This is observed appointment coverage, not available capacity.
#
# Missing technician-home information must not be classified as
# cross-territory work.

served = jobs.dropna(subset=["customer_territory"]).copy()

# A technician's home territory is considered known only when
# both their technician ID and home-territory assignment exist.
served["known_home"] = (
    served["ServiceResourceId"].notna()
    & served["tech_territory"].notna()
)

# Unknown assignments should not automatically count as
# cross-territory appointments.
served["own_known"] = np.where(
    served["known_home"],
    (
        served["customer_territory"]
        == served["tech_territory"]
    ).astype(float),
    np.nan,
)

# Appointment demand and technicians serving the customer territory.
effective = served.groupby("customer_territory").agg(
    demand_jobs=("Id", "size"),
    demand_min=("planned_minutes", "sum"),
    techs_serving=("ServiceResourceId", "nunique"),
    known_home_jobs=("known_home", "sum"),
    own_territory_rate=("own_known", "mean"),
)

effective["demand_hours"] = (
    effective["demand_min"] / 60
).round(1)

# Track records excluded from the own-territory denominator.
effective["unknown_home_jobs"] = (
    effective["demand_jobs"] - effective["known_home_jobs"]
)

# Each technician counts once per date in each customer territory.
# A technician serving two territories on the same date contributes
# one territory-technician-day to each, not two national workdays.
days_in_terr = (
    served.dropna(subset=["ServiceResourceId", "effective_date"])
    .drop_duplicates(
        ["ServiceResourceId", "effective_date", "customer_territory"]
    )
    .groupby("customer_territory")
    .size()
    .rename("tech_days_in_territory")
)

effective = effective.join(days_in_terr)

# Number of distinct technicians assigned to each territory
# according to the technician roster.
assigned = (
    roster.groupby("Territory")["ServiceResourceId"]
    .nunique()
    .rename("techs_assigned")
)

effective = effective.join(assigned)

# Observed workload per technician serving the territory.
# NOT available-capacity utilisation.
effective["hours_per_serving_tech"] = (
    effective["demand_hours"]
    / effective["techs_serving"].replace(0, np.nan)
).round(1)

# Own-territory rate is calculated among appointments with
# known technician-home assignments only.
effective["own_territory_rate"] = (
    effective["own_territory_rate"].round(3)
)

# Validation checks.
assert (
    effective["known_home_jobs"]
    + effective["unknown_home_jobs"]
    == effective["demand_jobs"]
).all()

assert (
    effective["own_territory_rate"]
    .dropna()
    .between(0, 1)
    .all()
)

print("\nTERRITORY DATA QUALITY CHECK")
print(f"Customer-territory appointments: {len(served):,}")
print(f"Known technician-home assignments: {served['known_home'].sum():,}")
print(f"Unknown technician-home assignments: {(~served['known_home']).sum():,}")
print(
    "Territories with unknown assignments:",
    int((effective["unknown_home_jobs"] > 0).sum())
)

cols = [
    "demand_jobs",
    "demand_hours",
    "techs_assigned",
    "techs_serving",
    "tech_days_in_territory",
    "hours_per_serving_tech",
    "own_territory_rate",
    "known_home_jobs",
    "unknown_home_jobs",
]

print("\n" + "=" * 100)
print("TERRITORIES WHERE ASSIGNED AND SERVING TECHNICIANS DIFFER MOST")
print("=" * 100)
gap = effective.copy()
gap["gap"] = gap["techs_serving"] - gap["techs_assigned"]
print(gap.sort_values("gap", ascending=False)[cols + ["gap"]].head(12).to_string())

print("\n" + "=" * 100)
print("LOWEST OWN-TERRITORY RATE  (served mostly from outside)")
print("=" * 100)
print(effective[effective["demand_jobs"] >= 150]
      .sort_values("own_territory_rate")[cols].head(10).to_string())

effective.to_csv(TABLES / "territory_effective.csv")

# %%
# --- Q3: how evenly is work shared between technicians? ---------------
# Compare like with like: shift pattern first, then the spread within it.
print("\n" + "=" * 100)
print("Q3 — WORK PER TECHNICIAN")
print("=" * 100)

per_tech = jobs.groupby("ServiceResourceId").agg(
    jobs=("Id", "size"),
    days=("effective_date", "nunique"),
    planned_min=("planned_minutes", "sum"),
    territories=("customer_territory", "nunique"),
)
per_tech["hours"] = (per_tech["planned_min"] / 60).round(0)
per_tech["jobs_per_day"] = (per_tech["jobs"] / per_tech["days"]).round(2)
per_tech["hours_per_day"] = (per_tech["planned_min"] / 60 / per_tech["days"]).round(2)

per_tech = per_tech.join(
    roster.set_index("ServiceResourceId")[["Working Hours", "Territory"]]
)

print("\nBy shift pattern:")
print(per_tech.groupby("Working Hours").agg(
    n=("jobs", "size"),
    med_days=("days", "median"),
    med_jobs=("jobs", "median"),
    med_jobs_day=("jobs_per_day", "median"),
    med_hours_day=("hours_per_day", "median"),
).round(1).to_string())

print("\nSpread across all active technicians:")
print(per_tech[["jobs", "days", "jobs_per_day", "hours_per_day", "territories"]]
      .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_string())

for col in ["jobs", "jobs_per_day", "hours_per_day"]:
    lo, hi = per_tech[col].quantile([0.1, 0.9])
    print(f"  {col:<14} p10={lo:7.1f}  p90={hi:7.1f}  ratio={hi / lo:.2f}x")

per_tech.to_csv(TABLES / "technician_workload.csv")

# %%
# --- Territories where the entire assigned team is inactive -----------
by_terr = roster.groupby("Territory")["active"].agg(on_roster="size", active="sum")
by_terr["inactive"] = by_terr["on_roster"] - by_terr["active"]
dead = by_terr[by_terr["active"] == 0]

demand_simple = jobs.groupby("customer_territory").agg(
    jobs=("Id", "size"), hours=("planned_minutes", lambda s: s.sum() / 60)
).round(0)

print("\n" + "=" * 80)
print("TERRITORIES WITH A ROSTER BUT NOBODY WORKING")
print("=" * 80)
print(dead.join(demand_simple, how="left").to_string())

# %%
# --- Region level: where territory lines stop leaking -----------------
# 80.6% of cross-territory work stays inside the same region group,
# so this is the level at which a load comparison is trustworthy.
# A territory can span two regions: six of them do, because their postcode
# districts are split. drop_duplicates() would silently take whichever
# district appeared first in the file, which is arbitrary. Use the region
# that holds most of the territory's districts instead.
roster["region"] = "Group " + roster["Territory"].str.extract(r"^G(\d+)-")[0]
print(f"\ntechnicians not matching G<N>- prefix: {roster['region'].isna().sum()}")
supply = roster.groupby("region")["active"].agg(on_roster="size", active="sum")
dem_r = jobs.dropna(subset=["region"]).groupby("region").agg(
    jobs=("Id", "size"), planned_min=("planned_minutes", "sum")
)

# Count distinct technician-date combinations WITHIN each region.
# A technician working in two regions on the same date contributes
# one observed technician-region-day to each region.
#
# Regional technician-days are not additive to national
# technician-days because cross-region work can occur.

days_r = (
    jobs.dropna(
        subset=["region", "ServiceResourceId", "effective_date"]
    )
    .drop_duplicates(
        ["region", "ServiceResourceId", "effective_date"]
    )
    .groupby("region")
    .size()
    .rename("tech_days")
)


# --- Regional data coverage and technician-day validation -------------

national_days = (
    jobs.dropna(subset=["ServiceResourceId", "effective_date"])
    .drop_duplicates(["ServiceResourceId", "effective_date"])
    .shape[0]
)

regional_days = int(days_r.sum())

print("\n" + "=" * 80)
print("REGIONAL DATA COVERAGE CHECK")
print("=" * 80)

print(f"Total appointments: {len(jobs):,}")

print(
    "Appointments with customer territory:",
    f"{jobs['customer_territory'].notna().sum():,}"
)

print(
    "Appointments with region:",
    f"{jobs['region'].notna().sum():,}"
)

print(
    "Missing region despite known customer territory:",
    f"{(jobs['region'].isna() & jobs['customer_territory'].notna()).sum():,}"
)
# --- Investigate appointments excluded from regional reporting --------

missing_region = jobs.loc[
    jobs["region"].isna()
    & jobs["customer_territory"].notna()
]

print("\n" + "=" * 80)
print("CUSTOMER TERRITORIES WITHOUT REGION MAPPING")
print("=" * 80)

print(
    missing_region["customer_territory"]
    .value_counts(dropna=False)
    .to_string()
)

missing_territory = jobs.loc[
    jobs["customer_territory"].isna()
]

print("\n" + "=" * 80)
print("APPOINTMENTS WITHOUT CUSTOMER TERRITORY")
print("=" * 80)

print(f"Number of appointments: {len(missing_territory):,}")

print(
    "Number with a region:",
    missing_territory["region"].notna().sum()
)

print(
    "Distinct technicians involved:",
    missing_territory["ServiceResourceId"].nunique()
)
print(
    "National distinct technician-days:",
    f"{national_days:,}"
)

print(
    "Sum of observed technician-region-days:",
    f"{regional_days:,}"
)

print(
    "Technicians serving multiple regions on the same date:",
    jobs.dropna(
        subset=["ServiceResourceId", "effective_date", "region"]
    )
    .groupby(["ServiceResourceId", "effective_date"])["region"]
    .nunique()
    .gt(1)
    .sum()
)

# --- Regional demand and observed workload ----------------------------
reg = supply.join(dem_r).join(days_r)
reg["hours"] = (reg["planned_min"] / 60).round(0)

# Both units, side by side. Jobs is the demand unit, hours is the capacity
# unit, and they answer different questions - so report both and watch
# where they disagree.
reg["jobs_per_active_tech"] = (reg["jobs"] / reg["active"]).round(0)
reg["hours_per_active_tech"] = (reg["hours"] / reg["active"]).round(0)
reg["jobs_per_tech_day"] = (reg["jobs"] / reg["tech_days"]).round(2)
reg["hours_per_tech_day"] = (reg["hours"] / reg["tech_days"]).round(2)
reg["avg_job_min"] = (reg["planned_min"] / reg["jobs"]).round(1)

reg["rank_jobs"] = reg["jobs_per_active_tech"].rank(ascending=False).astype("Int64")
reg["rank_hours"] = reg["hours_per_active_tech"].rank(ascending=False).astype("Int64")
reg["inactive_pct"] = ((reg["on_roster"] - reg["active"]) / reg["on_roster"] * 100).round(0)

print("\n" + "=" * 100)
print("Q2 — REGION LOAD, MEASURED BOTH WAYS")
print("=" * 100)
print(reg[["on_roster", "active", "inactive_pct",
           "jobs_per_active_tech", "rank_jobs",
           "hours_per_active_tech", "rank_hours",
           "jobs_per_tech_day", "hours_per_tech_day", "avg_job_min"]]
      .sort_values("rank_hours").to_string())

print("\nWhere the two units disagree:")
for region, row in reg.iterrows():
    if row["rank_jobs"] != row["rank_hours"]:
        print(f"  {region}: {row['rank_jobs']} by jobs, {row['rank_hours']} by hours "
              f"(average job {row['avg_job_min']:.0f} min)")

reg.to_csv(TABLES / "region_supply_demand.csv")

# --- Final geographic population reconciliation -----------------------

mapped = jobs["region"].notna()

suspended_or_unmapped = (
    jobs["customer_territory"].notna()
    & jobs["region"].isna()
)

missing_territory = jobs["customer_territory"].isna()

# These categories must be mutually exclusive.
assert not (mapped & missing_territory).any()

# Every operational appointment must belong to exactly one category.
assert (
    int(mapped.sum())
    + int(suspended_or_unmapped.sum())
    + int(missing_territory.sum())
) == len(jobs)

# Regional demand must match the regional table.
assert int(reg["jobs"].sum()) == int(mapped.sum())

print("\n" + "=" * 80)
print("FINAL GEOGRAPHIC RECONCILIATION")
print("=" * 80)

print(f"National appointments: {len(jobs):,}")
print(f"Mapped to six regions: {mapped.sum():,}")
print(f"Suspended/unmapped territory: {suspended_or_unmapped.sum():,}")
print(f"Missing customer territory: {missing_territory.sum():,}")
print(f"Regional mapping completeness: {mapped.mean():.2%}")
print("All geographic reconciliation checks passed.")


# %%
# --- Q3 fairness, comparing like with like ----------------------------
# Drop technicians with under 20 working days: joiners and leavers would
# otherwise look like underperformers.
steady = per_tech[per_tech["days"] >= 20]
print("\n" + "=" * 80)
print(f"Q3 — SPREAD WITHIN SHIFT PATTERN  ({len(steady)} technicians, "
      f"{len(per_tech) - len(steady)} dropped for under 20 days)")
print("=" * 80)

for pattern, grp in steady.groupby("Working Hours"):
    if len(grp) < 5:
        continue
    lo, hi = grp["hours_per_day"].quantile([0.1, 0.9])
    print(f"  {pattern:<22} n={len(grp):>3}  p10={lo:.2f}  p90={hi:.2f}  "
          f"ratio={hi / lo:.2f}x  median={grp['hours_per_day'].median():.2f}")

lo, hi = steady["hours_per_day"].quantile([0.1, 0.9])
print(f"  {'ALL PATTERNS':<22} n={len(steady):>3}  p10={lo:.2f}  p90={hi:.2f}  "
      f"ratio={hi / lo:.2f}x  median={steady['hours_per_day'].median():.2f}")
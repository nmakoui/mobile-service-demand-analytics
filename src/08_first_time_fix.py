
"""
Identify subsequent same-vehicle appointments within 60 days.

This is a repeat-appointment proxy, NOT confirmed rework
or a validated first-time-fix measure.

Run: python src/08_first_time_fix.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")

LOOKAHEAD_DAYS = 60

# -----------------------------------------------------------
# 1. All available non-placeholder appointments
# -----------------------------------------------------------

all_real = sa.loc[
    ~sa["is_placeholder"]
    & sa["effective_date"].notna()
].copy()

all_real["vehicle_key"] = (
    all_real["VRN_formula__c"]
    .astype("string")
    .str.upper()
    .str.replace(" ", "", regex=False)
    .str.strip()
)

all_real = all_real.loc[
    all_real["vehicle_key"].notna()
    & all_real["vehicle_key"].ne("")
].copy()

max_date = all_real["effective_date"].max()

assert all_real["Id"].is_unique

# -----------------------------------------------------------
# 2. Failed appointments in the operational window
# -----------------------------------------------------------

jobs = sa.loc[
    ~sa["is_placeholder"] & sa["in_demand_window"]
].copy()

all_failed = jobs.loc[jobs["failed"]].copy()

assert all_failed["Id"].is_unique

all_failed["vehicle_key"] = (
    all_failed["VRN_formula__c"]
    .astype("string")
    .str.upper()
    .str.replace(" ", "", regex=False)
    .str.strip()
)

with_vehicle = all_failed.loc[
    all_failed["vehicle_key"].notna()
    & all_failed["vehicle_key"].ne("")
].copy()

# -----------------------------------------------------------
# 3. Date-window eligibility
# -----------------------------------------------------------

# A failed appointment must be at least 60 calendar days
# before the latest appointment date observed in the extract.
#
# IMPORTANT:
# This is a date-window eligibility proxy.
# It DOES NOT prove complete follow-up capture because the
# source extract is bounded by Created Date.

cutoff = max_date - pd.Timedelta(days=LOOKAHEAD_DAYS)

scoreable = with_vehicle.loc[
    with_vehicle["effective_date"] <= cutoff
].copy()

# -----------------------------------------------------------
# 4. Find subsequent appointments for the same vehicle
# -----------------------------------------------------------

merged = scoreable.merge(
    all_real[
        [
            "vehicle_key",
            "effective_date",
            "Id",
            "planned_minutes",
        ]
    ],
    on="vehicle_key",
    suffixes=("", "_other"),
)

merged = merged.loc[
    (merged["Id_other"] != merged["Id"])
    & (merged["effective_date_other"] > merged["effective_date"])
    & (
        merged["effective_date_other"]
        <= merged["effective_date"]
        + pd.Timedelta(days=LOOKAHEAD_DAYS)
    )
]

if not merged.empty:
    first_repeat = (
        merged.sort_values("effective_date_other")
        .groupby("Id")
        .first()[
            ["effective_date_other", "planned_minutes_other"]
        ]
        .rename(
            columns={
                "effective_date_other": "repeat_date",
                "planned_minutes_other": "repeat_minutes",
            }
        )
    )

    scoreable = scoreable.join(first_repeat, on="Id")

else:
    scoreable["repeat_date"] = pd.NaT
    scoreable["repeat_minutes"] = float("nan")

scoreable["repeat_visit"] = scoreable["repeat_date"].notna()

# -----------------------------------------------------------
# 5. Export ALL failed appointments with eligibility flags
# -----------------------------------------------------------

output = all_failed[
    ["Id", "failure_owner", "Sub_Categories__c"]
].copy()

output["repeat_eligible"] = output["Id"].isin(
    scoreable["Id"]
)

output = output.merge(
    scoreable[
        ["Id", "repeat_visit", "repeat_minutes"]
    ],
    on="Id",
    how="left",
    validate="one_to_one",
)

# Ineligible = unknown outcome, NOT no repeat appointment.
assert len(output) == len(all_failed)

assert (
    int(output["repeat_eligible"].sum()) == len(scoreable)
)

assert (
    output.loc[
        ~output["repeat_eligible"], "repeat_visit"
    ].isna().all()
)

output.to_csv(
    TABLES / "first_time_fix.csv",
    index=False,
)

# -----------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------

print("\n" + "=" * 80)
print("SUBSEQUENT SAME-VEHICLE APPOINTMENT ANALYSIS")
print("=" * 80)

print(f"Total failed appointments: {len(all_failed):,}")

print(
    f"Failed appointments with vehicle identifier: "
    f"{len(with_vehicle):,}"
)

print(f"Latest observed appointment date: {max_date}")

print(f"Date-window eligibility cutoff: {cutoff}")

print(
    f"Eligible failed appointments: {len(scoreable):,}"
)

print(
    f"Ineligible / unscoreable failed appointments: "
    f"{len(all_failed) - len(scoreable):,}"
)

if len(scoreable):
    print(
        f"Observed subsequent appointment within 60 days: "
        f"{scoreable['repeat_visit'].sum():,}"
    )

    print(
        f"Observed subsequent-appointment rate: "
        f"{scoreable['repeat_visit'].mean():.1%}"
    )

    print("\nBy failure owner:")
    print(
        scoreable.groupby("failure_owner")["repeat_visit"]
        .agg(["size", "mean"])
        .round(3)
        .to_string()
    )

print("\nLIMITATIONS:")
print(
    "A subsequent appointment may be unrelated to the original failure."
)
print(
    "Created-Date extraction limits may omit later appointments, "
    "even when the date-window criterion is satisfied."
)
print(
    "Do not interpret this result as a confirmed first-time-fix rate."
)

print("\nAll repeat-appointment checks passed.")

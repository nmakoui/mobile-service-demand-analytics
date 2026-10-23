
"""
Evidence-based recommendation for stock-related service failures.

Measures recorded appointment starts and associated booked hours.
Does not assume physical attendance, wasted labour or guaranteed savings.

Run: python src/06_recommendation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")

jobs = sa.loc[
    ~sa["is_placeholder"] & sa["in_demand_window"]
].copy()

jobs["recorded_start"] = jobs["Actual Start"].notna()
failed = jobs.loc[jobs["failed"]].copy()

# -----------------------------------------------------------
# 1. Compare the key stock failure reasons
# -----------------------------------------------------------

REASONS = [
    "Wrong parts ordered",
    "Tyres not in Stock",
    "Locking Wheel Nut Unavailable",
]

rows = []

for reason in REASONS:
    group = failed.loc[
        failed["Sub_Categories__c"] == reason
    ]

    started = group.loc[group["recorded_start"]]

    rows.append({
        "failure_reason": reason,
        "failed_appointments": len(group),
        "with_recorded_start": len(started),
        "without_recorded_start": (
            len(group) - len(started)
        ),
        "recorded_start_rate_pct": (
            len(started) / len(group) * 100
            if len(group) else np.nan
        ),
        "booked_hours_all": (
            group["planned_minutes"].sum() / 60
        ),
        "booked_hours_with_start": (
            started["planned_minutes"].sum() / 60
        ),
    })

report = pd.DataFrame(rows)

# -----------------------------------------------------------
# 2. Validation
# -----------------------------------------------------------

assert (
    report["with_recorded_start"]
    <= report["failed_appointments"]
).all()

assert (
    report["with_recorded_start"]
    + report["without_recorded_start"]
    == report["failed_appointments"]
).all()

assert (
    report["booked_hours_with_start"]
    <= report["booked_hours_all"]
).all()

report.round(2).to_csv(
    TABLES / "recommendation_evidence.csv",
    index=False
)

print("\n" + "=" * 80)
print("STOCK FAILURE EVIDENCE")
print("=" * 80)

print(report.round(2).to_string(index=False))

# -----------------------------------------------------------
# 3. Quantify exposure for the proposed pilot
# -----------------------------------------------------------

TARGET = [
    "Wrong parts ordered",
    "Locking Wheel Nut Unavailable",
]

target = report.loc[
    report["failure_reason"].isin(TARGET)
]

target_failures = int(target["failed_appointments"].sum())
target_starts = int(target["with_recorded_start"].sum())

target_booked_hours = (
    target["booked_hours_with_start"].sum()
)

print("\n" + "=" * 80)
print("PROPOSED PRE-DISPATCH READINESS PILOT")
print("=" * 80)

print(f"Target failure categories: {', '.join(TARGET)}")
print(f"Failed appointments: {target_failures:,}")
print(f"Failures with recorded start: {target_starts:,}")

print(
    f"Planned hours associated with recorded starts: "
    f"{target_booked_hours:,.1f}"
)

print("\nRECOMMENDATION:")
print(
    "Pilot a pre-dispatch parts-readiness check for "
    "parts-dependent appointments."
)

print(
    "Verify parts compatibility, stock readiness and "
    "locking-wheel-nut requirements before dispatch."
)

print("\nPILOT DESIGN:")
print(
    "1. Select comparable territories or booking teams "
    "for a controlled operational pilot."
)

print(
    "2. Record whether readiness checks find problems "
    "before the scheduled appointment."
)

print(
    "3. Measure overall failure rates, failure reasons, "
    "recorded-start rates and rescheduling outcomes."
)

print(
    "4. Record additional checking time and any "
    "verified technician time made available."
)

print(
    "5. Compare results with a suitable baseline or "
    "comparison group before scaling."
)

print("\nIMPORTANT LIMITATIONS:")
print(
    "- Actual Start is a recorded-start proxy, "
    "not proof of travel or physical attendance."
)

print(
    "- Planned hours are booked workload, "
    "not confirmed wasted or recoverable labour."
)

print(
    "- A lower recorded-start rate does not by itself "
    "mean the failure was prevented."
)

print(
    "- Different stock failure categories may have "
    "different operational causes."
)

print(
    "- No technician headcount or financial savings "
    "are claimed from this analysis."
)

print("\nAll recommendation checks passed.")

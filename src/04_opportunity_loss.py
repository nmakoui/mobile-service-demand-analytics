"""
Phase 3 — Q4: how much capacity is lost to causes inside the business?

Run:  .venv\\Scripts\\python.exe src/04_opportunity_loss.py
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

pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 40)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")
jobs = sa[~sa["is_placeholder"] & sa["in_demand_window"]].copy()

# An Actual Start timestamp indicates a recorded appointment start.
# It is a proxy for technician attendance, not proof of physical
# arrival, travel or time spent at the customer location.
jobs["attended"] = jobs["Actual Start"].notna()

failed = jobs[jobs["failed"]].copy()

print(f"Appointments: {len(jobs):,}")
print(
    f"Appointments flagged as failed: {len(failed):,} "
    f"({len(failed) / len(jobs):.1%})"
)

# %%
# --- Did anyone actually travel? --------------------------------------
print("\n" + "=" * 80)
print("DID A TECHNICIAN ATTEND?")
print("=" * 80)
print(pd.crosstab(failed["Status"], failed["attended"],
                  rownames=["Status"], colnames=["attended"]).to_string())

# %%
# --- The recorded duration is a default, not a measurement ------------
print("\n" + "=" * 80)
print("RECORDED DURATION ON ATTENDED FAILURES")
print("=" * 80)
print(failed.loc[failed["attended"], "ActualDuration"]
      .describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1).to_string())
print("\nMedian, 75th and 90th percentile are the same value, so the system")
print("writes a fixed number rather than timing the visit. The booked slot")
print("is used below instead of this column.")

# %%
# --- Anatomy of the loss ----------------------------------------------
anatomy = failed.groupby("failure_owner").agg(
    appointments=("Id", "size"),
    attended=("attended", "sum"),
    booked_min=("planned_minutes", "sum"),
)
anatomy["attended_pct"] = (anatomy["attended"] / anatomy["appointments"] * 100).round(0)
anatomy["booked_hours"] = (anatomy["booked_min"] / 60).round(0)
anatomy["attended_hours"] = (
    failed[failed["attended"]].groupby("failure_owner")["planned_minutes"].sum() / 60
).round(0)
anatomy = anatomy.sort_values("appointments", ascending=False)

print("\n" + "=" * 80)
print("WHERE THE LOSS COMES FROM")
print("=" * 80)
print(anatomy[["appointments", "attended", "attended_pct",
               "booked_hours", "attended_hours"]].to_string())
anatomy.to_csv(TABLES / "opportunity_loss_anatomy.csv")

# %%

# --- Cancellation notice by calendar day ------------------------------
# Compare calendar dates, not timestamps measured against midnight.
#
# effective_date is our appointment-date proxy.
# Time of Cancellation is the recorded cancellation timestamp.
#
# This measures calendar-day notice only.
# It does not establish whether a cancelled slot was refilled.

cancelled = failed[
    failed["Status"] == "Canceled"
].copy()

cancelled["appointment_day"] = pd.to_datetime(
    cancelled["effective_date"],
    errors="coerce"
).dt.normalize()

cancelled["cancellation_day"] = pd.to_datetime(
    cancelled["Time of Cancellation"],
    errors="coerce"
).dt.normalize()

cancelled["notice_calendar_days"] = (
    cancelled["appointment_day"]
    - cancelled["cancellation_day"]
).dt.days

# Negative days = cancellation recorded after appointment date.
# Zero = cancellation recorded on the appointment date.
# Positive days = cancellation recorded before appointment date.

bins = [-np.inf, -1, 0, 1, 3, 7, np.inf]

labels = [
    "Recorded after appointment date",
    "Same appointment date",
    "1 calendar day before",
    "2-3 calendar days before",
    "4-7 calendar days before",
    "8+ calendar days before",
]

notice_category = pd.cut(
    cancelled["notice_calendar_days"],
    bins=bins,
    labels=labels
)

notice_counts = (
    notice_category
    .value_counts()
    .reindex(labels, fill_value=0)
)

unknown_notice = int(notice_category.isna().sum())

# Validate that every cancellation has been accounted for.
assert (
    int(notice_counts.sum()) + unknown_notice
    == len(cancelled)
)

# Save the breakdown for reporting and validation.
notice_output = pd.DataFrame({
    "calendar_notice_category": (
        labels + ["Missing/invalid date"]
    ),
    "appointments": (
        [int(notice_counts.loc[label]) for label in labels]
        + [unknown_notice]
    )
})

notice_output.to_csv(
    TABLES / "cancellation_notice_by_calendar_day.csv",
    index=False
)

print("\n" + "=" * 80)
print("CANCELLATION NOTICE - CALENDAR-DAY ANALYSIS")
print("=" * 80)

print(f"Total cancelled appointments: {len(cancelled):,}")

print("\nCancellation notice distribution:")
print(notice_output.to_string(index=False))

same_or_after = int(
    notice_counts["Recorded after appointment date"]
    + notice_counts["Same appointment date"]
)

print(
    "\nCancelled on or recorded after appointment date: "
    f"{same_or_after:,} "
    f"({same_or_after / len(cancelled):.1%})"
)

print(
    "\nNote: Calendar-day notice does not establish "
    "whether a cancelled appointment slot was refilled."
)

print("Cancellation reconciliation passed.")


# %%

# --- Booked-hours exposure associated with internal failures ----------
# We examine failures attributed to internal business processes.
#
# A recorded Actual Start is used as a proxy for attendance.
# Planned minutes represent the booked appointment duration.
#
# These values DO NOT establish:
# - actual technician time wasted
# - capacity that could definitely have been recovered
# - additional technicians required
# - financial losses or savings

INTERNAL = ["Stock", "Resource", "Booking admin"]

internal = failed[
    failed["failure_owner"].isin(INTERNAL)
].copy()

# Retain the name "lost" temporarily for compatibility with
# the existing regional and weekday analysis below.
# This subset means INTERNAL FAILURES WITH A RECORDED START.
lost = internal[internal["attended"]].copy()

internal_booked_hours = (
    internal["planned_minutes"].sum() / 60
)

started_internal_booked_hours = (
    lost["planned_minutes"].sum() / 60
)

national_booked_hours = (
    jobs["planned_minutes"].sum() / 60
)

internal_start_rate = (
    len(lost) / len(internal)
    if len(internal) > 0
    else np.nan
)

booked_hours_share = (
    started_internal_booked_hours / national_booked_hours
    if national_booked_hours > 0
    else np.nan
)

missing_planned_minutes = (
    lost["planned_minutes"].isna().sum()
)

# Validation
assert len(lost) <= len(internal)
assert lost["attended"].all()
assert lost["failed"].all()
assert lost["failure_owner"].isin(INTERNAL).all()

print("\n" + "=" * 80)
print("Q4 - BOOKED-HOURS EXPOSURE FROM INTERNAL FAILURES")
print("=" * 80)

print(
    f"Appointments with internal failure owner: "
    f"{len(internal):,}"
)

print(
    f"Internal failures with recorded Actual Start: "
    f"{len(lost):,}"
)

print(
    f"Recorded-start rate among internal failures: "
    f"{internal_start_rate:.1%}"
)

print(
    f"Booked hours across all internal failures: "
    f"{internal_booked_hours:,.1f}"
)

print(
    f"Booked hours on internal failures with recorded start: "
    f"{started_internal_booked_hours:,.1f}"
)

print(
    f"Share of national booked appointment hours: "
    f"{booked_hours_share:.2%}"
)

print(
    f"Missing planned durations in this subset: "
    f"{missing_planned_minutes:,}"
)

print("\nINTERPRETATION:")
print(
    "These are planned appointment hours associated with "
    "internal failures that have a recorded Actual Start."
)

print(
    "They do not represent confirmed wasted technician "
    "hours or recoverable resource capacity."
)

print("\nAll internal-failure validation checks passed.")

# %%

# --- Regional booked-hours exposure from internal failures ------------
# These are booked appointment hours, not confirmed wasted labour.
# Regional comparisons use customer-region attribution.

regional_demand = (
    jobs.dropna(subset=["region"])
    .groupby("region")
    .agg(
        appointments=("Id", "size"),
        total_planned_hours=(
            "planned_minutes",
            lambda s: s.sum() / 60
        )
    )
)

regional_exposure = (
    lost.dropna(subset=["region"])
    .groupby("region")
    .agg(
        started_internal_failures=("Id", "size"),
        booked_hours_on_started_failures=(
            "planned_minutes",
            lambda s: s.sum() / 60
        )
    )
)

regional_report = regional_demand.join(
    regional_exposure,
    how="left"
)

regional_report[
    [
        "started_internal_failures",
        "booked_hours_on_started_failures"
    ]
] = regional_report[
    [
        "started_internal_failures",
        "booked_hours_on_started_failures"
    ]
].fillna(0)

regional_report["exposure_pct_of_booked_hours"] = (
    regional_report["booked_hours_on_started_failures"]
    / regional_report["total_planned_hours"]
    * 100
)

regional_report = regional_report.round(2)

# --- Reconcile national and regional exposure -------------------------

unmapped = lost[lost["region"].isna()]

unmapped_hours = (
    unmapped["planned_minutes"].sum() / 60
)

regional_started_count = int(
    regional_report["started_internal_failures"].sum()
)

regional_started_hours = (
    lost.loc[lost["region"].notna(), "planned_minutes"].sum()
    / 60
)

assert (
    regional_started_count + len(unmapped)
    == len(lost)
)

assert np.isclose(
    regional_started_hours + unmapped_hours,
    started_internal_booked_hours
)

print("\n" + "=" * 80)
print("REGIONAL BOOKED-HOURS EXPOSURE")
print("=" * 80)

print(regional_report.to_string())

print("\nREGIONAL RECONCILIATION")
print(
    f"National internal failures with recorded start: "
    f"{len(lost):,}"
)

print(
    f"Included in six regions: "
    f"{regional_started_count:,}"
)

print(
    f"Excluded due to missing/unmapped region: "
    f"{len(unmapped):,}"
)

print(
    f"Booked hours on excluded appointments: "
    f"{unmapped_hours:,.2f}"
)

print(
    "\nAll regional exposure reconciliation checks passed."
)

# Save a correctly labelled regional table.
regional_report.to_csv(
    TABLES / "regional_failure_exposure.csv"
)


# %%

# --- Failure patterns by appointment day of week -----------------------
# Compare total appointment volume, failure rates and internal
# failures with recorded appointment starts.
#
# A recorded Actual Start is a proxy, not proof of physical attendance.
# The figures do not represent confirmed lost technician visits.

order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_report = pd.DataFrame(index=order)
weekday_report.index.name = "day_of_week"

weekday_report["appointments"] = (
    jobs.groupby("dow")
    .size()
    .reindex(order, fill_value=0)
)

weekday_report["failed_appointments"] = (
    failed.groupby("dow")
    .size()
    .reindex(order, fill_value=0)
)

weekday_report["started_internal_failures"] = (
    lost.groupby("dow")
    .size()
    .reindex(order, fill_value=0)
)

weekday_report["failure_rate_pct"] = (
    weekday_report["failed_appointments"]
    / weekday_report["appointments"].replace(0, np.nan)
    * 100
).round(2)

weekday_report["started_internal_failure_rate_pct"] = (
    weekday_report["started_internal_failures"]
    / weekday_report["appointments"].replace(0, np.nan)
    * 100
).round(2)

# --- Validation --------------------------------------------------------

assert weekday_report["appointments"].sum() == len(jobs)

assert (
    weekday_report["failed_appointments"].sum()
    == len(failed)
)

assert (
    weekday_report["started_internal_failures"].sum()
    == len(lost)
)

assert (
    weekday_report["started_internal_failures"]
    <= weekday_report["failed_appointments"]
).all()

# --- Export ------------------------------------------------------------

weekday_report.to_csv(
    TABLES / "weekday_failure_activity.csv"
)

# --- Display -----------------------------------------------------------

print("\n" + "=" * 80)
print("FAILURE ACTIVITY BY APPOINTMENT DAY OF WEEK")
print("=" * 80)

print(weekday_report.to_string())

print("\nWEEKDAY RECONCILIATION")
print(f"Total appointments: {weekday_report['appointments'].sum():,}")
print(f"Failed appointments: {weekday_report['failed_appointments'].sum():,}")
print(
    "Started internal failures:",
    f"{weekday_report['started_internal_failures'].sum():,}"
)

print("All weekday reconciliation checks passed.")


# %%

# --- Stock failures: reason, recorded start and booked hours -----------
# Actual Start is used as a recorded-start proxy, not proof
# that a technician physically attended or travelled.
#
# Planned duration measures booked workload, not actual
# wasted labour or confirmed recoverable capacity.

stock = failed.loc[
    failed["failure_owner"] == "Stock"
].copy()

# Preserve missing or blank failure reasons as an explicit category.
stock["stock_failure_reason"] = (
    stock["Sub_Categories__c"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("Not recorded")
)

# Include planned minutes only when an Actual Start was recorded.
stock["planned_minutes_started"] = (
    stock["planned_minutes"].where(stock["attended"], 0)
)

# Aggregate by failure reason.
stock_report = (
    stock.groupby("stock_failure_reason")
    .agg(
        failed_appointments=("Id", "size"),
        with_recorded_start=("attended", "sum"),
        booked_minutes_all=("planned_minutes", "sum"),
        booked_minutes_started=("planned_minutes_started", "sum"),
    )
)

# Share of stock failures attributable to each recorded reason.
stock_report["share_stock_failures_pct"] = (
    stock_report["failed_appointments"]
    / len(stock)
    * 100
)

# Among failed appointments in each category, how many
# have a recorded Actual Start?
stock_report["recorded_start_pct"] = (
    stock_report["with_recorded_start"]
    / stock_report["failed_appointments"]
    * 100
)

stock_report["booked_hours_all"] = (
    stock_report["booked_minutes_all"] / 60
)

stock_report["booked_hours_with_start"] = (
    stock_report["booked_minutes_started"] / 60
)

# Select useful reporting columns.
stock_report = stock_report[
    [
        "failed_appointments",
        "share_stock_failures_pct",
        "with_recorded_start",
        "recorded_start_pct",
        "booked_hours_all",
        "booked_hours_with_start",
    ]
].sort_values(
    "failed_appointments",
    ascending=False
)

# --- Validation --------------------------------------------------------

assert stock_report["failed_appointments"].sum() == len(stock)

assert (
    stock_report["with_recorded_start"].sum()
    == int(stock["attended"].sum())
)

assert np.isclose(
    stock_report["booked_hours_all"].sum(),
    stock["planned_minutes"].sum() / 60
)

assert np.isclose(
    stock_report["booked_hours_with_start"].sum(),
    stock.loc[stock["attended"], "planned_minutes"].sum() / 60
)

assert (
    stock_report["recorded_start_pct"]
    .between(0, 100)
    .all()
)

# --- Export ------------------------------------------------------------

stock_report.round(2).to_csv(
    TABLES / "stock_failure_start_analysis.csv"
)

# --- Display -----------------------------------------------------------

print("\n" + "=" * 80)
print("STOCK FAILURE REASONS AND BOOKED-HOURS EXPOSURE")
print("=" * 80)

print(stock_report.round(2).to_string())

print("\nSTOCK FAILURE RECONCILIATION")

print(f"Total stock failures: {len(stock):,}")

print(
    "Stock failures with recorded start:",
    f"{stock['attended'].sum():,}"
)

print(
    "Booked hours across all stock failures:",
    f"{stock['planned_minutes'].sum() / 60:,.2f}"
)

print(
    "Booked hours with recorded start:",
    f"{stock.loc[stock['attended'], 'planned_minutes'].sum() / 60:,.2f}"
)

print(
    "Missing planned durations:",
    f"{stock['planned_minutes'].isna().sum():,}"
)

print("All stock failure reconciliation checks passed.")

# --- Compare the two principal failure reasons ------------------------

reason_a = "Wrong parts ordered"
reason_b = "Tyres not in Stock"

if (
    reason_a in stock_report.index
    and reason_b in stock_report.index
):
    a = stock_report.loc[reason_a]
    b = stock_report.loc[reason_b]

    print("\n" + "=" * 80)
    print("WRONG PARTS VS TYRE STOCK FAILURES")
    print("=" * 80)

    print(
        f"Wrong parts ordered - recorded-start rate: "
        f"{a['recorded_start_pct']:.1f}%"
    )

    print(
        f"Tyres not in stock - recorded-start rate: "
        f"{b['recorded_start_pct']:.1f}%"
    )

    print(
        "Difference in recorded-start rates:",
        f"{a['recorded_start_pct'] - b['recorded_start_pct']:.1f}"
        " percentage points"
    )

    print(
        "Booked hours on started wrong-parts failures:",
        f"{a['booked_hours_with_start']:,.1f}"
    )

    print(
        "Booked hours on started tyre-stock failures:",
        f"{b['booked_hours_with_start']:,.1f}"
    )

print("\nInterpretation:")
print(
    "Differences in recorded-start rates identify opportunities "
    "for process investigation. They do not establish why failures "
    "occurred or prove that the booked hours were wasted."
)


# %%

# --- Booking lead-time analysis with complete reconciliation ----------
# Lead time describes the interval between booking and appointment.
# Missing categories are retained rather than silently excluded.
#
# Observed failure-rate differences are descriptive, not causal.

lead_jobs = jobs.copy()

lead_jobs["lead_band_audit"] = (
    lead_jobs["lead_band"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("Missing lead band")
)

lead_report = (
    lead_jobs.groupby("lead_band_audit")
    .agg(
        appointments=("Id", "size"),
        failed_appointments=("failed", "sum")
    )
)

lead_report["failure_rate_pct"] = (
    lead_report["failed_appointments"]
    / lead_report["appointments"]
    * 100
).round(2)

lead_report["share_of_all_appointments_pct"] = (
    lead_report["appointments"]
    / len(jobs)
    * 100
).round(2)

# Show the expected categories in a meaningful order.
expected_bands = [
    "Same day",
    "Next day",
    "2-3 days",
    "4-7 days",
    "8+ days",
    "Missing lead band"
]

# Preserve unexpected category labels for investigation.
unexpected_bands = [
    band for band in lead_report.index
    if band not in expected_bands
]

lead_report = lead_report.reindex(
    expected_bands + unexpected_bands,
    fill_value=0
)

# --- Validation --------------------------------------------------------

assert int(lead_report["appointments"].sum()) == len(jobs)

assert (
    int(lead_report["failed_appointments"].sum())
    == len(failed)
)

assert (
    lead_report["failed_appointments"]
    <= lead_report["appointments"]
).all()

# --- Export ------------------------------------------------------------

lead_report.to_csv(
    TABLES / "booking_lead_time_audit.csv",
    index_label="lead_band"
)

# --- Display -----------------------------------------------------------

print("\n" + "=" * 80)
print("BOOKING LEAD-TIME ANALYSIS - VALIDATED")
print("=" * 80)

print(lead_report.to_string())

missing = int(
    lead_report.loc["Missing lead band", "appointments"]
)

missing_failed = int(
    lead_report.loc["Missing lead band", "failed_appointments"]
)

print("\nLEAD-TIME DATA QUALITY")

print(f"Total appointments: {len(jobs):,}")

print(
    f"Missing lead-time category: {missing:,} "
    f"({missing / len(jobs):.2%})"
)

print(
    f"Failed appointments with missing lead time: "
    f"{missing_failed:,}"
)

print(
    "Unexpected lead-time categories:",
    unexpected_bands
)

print("\nAll lead-time reconciliation checks passed.")

# %%
# --- Demand booked into suspended districts ---------------------------
susp = jobs[jobs["suspended_district"]]
print("\n" + "=" * 80)
print("APPOINTMENTS IN SUSPENDED DISTRICTS")
print("=" * 80)
print(f"appointments : {len(susp):,}")
print(f"failure rate : {susp['failed'].mean():.1%}  (national {jobs['failed'].mean():.1%})")
print(f"hours booked : {susp['planned_minutes'].sum() / 60:.0f}")
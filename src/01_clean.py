"""
Clean the Kwik Fit appointment data.

Run:  python src/01_clean.py
Reads data/processed/*_raw.parquet. The raw files are never modified.
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

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# %%
sa = pd.read_parquet(PROCESSED / "sa_raw.parquet")
tech = pd.read_parquet(PROCESSED / "tech_raw.parquet")
terr = pd.read_parquet(PROCESSED / "terr_raw.parquet")
print(f"loaded  sa={len(sa):,}  tech={len(tech):,}  terr={len(terr):,}")

# --- Audit lead-time values before numeric conversion -----------------
# Distinguish missing values already present in sa_raw.parquet
# from non-missing values that fail numeric conversion.

lead_input = sa["Lead Time"]

lead_blank = (
    lead_input.isna()
    | lead_input.astype("string").str.strip().eq("").fillna(False)
)

lead_numeric = pd.to_numeric(lead_input, errors="coerce")

sa["lead_time_missing_in_input"] = lead_blank.astype(bool)

sa["lead_time_parse_error"] = (
    ~sa["lead_time_missing_in_input"]
    & lead_numeric.isna()
)

# %%
# --- Excel serial numbers -> real dates -------------------------------
# Excel counts days from 1899-12-30. 45569.54 is not a quantity,
# it is 2024-10-01 13:05. Rounding to the second removes floating-point
# artefacts like 23:59:59.999999.

DATE_COLS = [
    "Created Date",
    "Appointment Date",
    "SchedStartTime",
    "SchedEndTime",
    "Actual Start",
    "ActualEndTime",
    "Booked Slot Start Time",
    "Booked Slot End Time",
    "Time of Cancellation",
    "Mobile Booking Form: Created Date",
]


def excel_to_datetime(s: pd.Series) -> pd.Series:
    """Convert an Excel serial-number column to real datetimes."""
    numbers = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(numbers, unit="D", origin="1899-12-30").dt.round("s")


for col in DATE_COLS:
    sa[col] = excel_to_datetime(sa[col])

# %%
# --- Columns that should be numbers -----------------------------------
# errors="coerce" turns anything unparseable into NaN rather than crashing,
# so one bad cell cannot take down the whole run.

NUM_COLS = [
    "ActualDuration",
    "DurationInMinutes",
    "Lead Time",
    "Number of Tyres",
    "FSL__Efficiency__c",
    "Latitude",
    "Longitude",
]

for col in NUM_COLS:
    sa[col] = pd.to_numeric(sa[col], errors="coerce")

# %%
# --- Did the conversion work? -----------------------------------------
print("Date ranges after conversion:\n")
for col in DATE_COLS:
    lo, hi = sa[col].min(), sa[col].max()
    filled = sa[col].notna().sum()
    print(f"  {col:<36} {str(lo)[:16]}  ->  {str(hi)[:16]}   ({filled:,} values)")
    # %%
# --- Evidence for the remaining data-quality issues -------------------

print("=" * 70)
print("1. Work types — are placeholder jobs real customer demand?")
print("=" * 70)
print(sa["Work Type: Work Type Name"].value_counts().head(8).to_string())

placeholder = sa["Work Type: Work Type Name"] == "Manual Placeholder Job"
print(f"\nPlaceholder rows: {placeholder.sum():,}")
print(f"  with a postcode:   {sa.loc[placeholder, 'PostalCode'].notna().mean():.1%}")
print(f"  with a vehicle reg:{sa.loc[placeholder, 'VRN_formula__c'].notna().mean():.1%}")
print("  status mix:")
print(sa.loc[placeholder, "Status"].value_counts().to_string())

print("\nFor contrast, real tyre-fitting jobs:")
real = sa["Work Type: Work Type Name"] == "Tyre Fitting - Request"
print(f"  with a postcode:   {sa.loc[real, 'PostalCode'].notna().mean():.1%}")
print(f"  with a vehicle reg:{sa.loc[real, 'VRN_formula__c'].notna().mean():.1%}")

# %%
print("=" * 70)
print("2. DurationType — does the unit match the value?")
print("=" * 70)
print(sa.groupby("DurationType")["DurationInMinutes"].describe().round(1).to_string())

print("\nWork types among the rows marked 'Hours':")
hours = sa["DurationType"] == "Hours"
print(
    sa.loc[hours]
    .groupby("Work Type: Work Type Name")["DurationInMinutes"]
    .agg(["size", "median"])
    .to_string()
)

# %%
print("=" * 70)
print("3. ActualDuration — are there jobs nobody closed?")
print("=" * 70)
print(sa["ActualDuration"].describe().round(1).to_string())
for threshold in [120, 240, 480, 1440]:
    n = (sa["ActualDuration"] > threshold).sum()
    print(f"  longer than {threshold:>5,} minutes: {n:>5,}")

# %%
# --- Apply fixes, recording every one ---------------------------------
OUTPUTS = ROOT / "outputs" / "tables"
OUTPUTS.mkdir(parents=True, exist_ok=True)

log = []


def record(issue, decision, rows, note=""):
    log.append(
        {"issue": issue, "decision": decision, "rows_affected": rows, "note": note}
    )
    print(f"  {rows:>6,}  {issue}")


print("Cleaning decisions:\n")

# 1. Header rows pasted into the data (fixed in 00_convert.py).
record(
    "Header rows inside the data",
    "Dropped",
    2,
    "Rows where Id held the literal text 'Id'. Removed in 00_convert.py.",
)

# 2. Duplicate appointment numbers.
dupes = sa["Appointment Number"].duplicated().sum()
record(
    "Duplicate appointment numbers",
    "None remain" if dupes == 0 else "Investigate",
    dupes,
    "The only duplicates were the header rows above.",
)

# 3. Placeholder jobs are not customer demand.
sa["is_placeholder"] = sa["Work Type: Work Type Name"] == "Manual Placeholder Job"
record(
    "Manual Placeholder Job records",
    "Flagged, excluded from demand",
    int(sa["is_placeholder"].sum()),
    "0% carry a vehicle registration vs 100% of real jobs. Admin holds, not visits.",
)
# 4. Appointment Date is missing for 95.7% of Inspect & Fit jobs but 0% of
#    tyre fittings. Those are the longest jobs, so filtering on that column
#    silently removes the heaviest capacity. Fall back to when work actually
#    started, then to the scheduled start.
sa["effective_date"] = (
    sa["Appointment Date"]
    .fillna(sa["Actual Start"])
    .fillna(sa["SchedStartTime"])
    .dt.normalize()
)

# Record which timestamp supplied each appointment date.
# This is important because fallback dates are not equivalent
# to an explicitly recorded Appointment Date.

sa["effective_date_source"] = np.select(
    [
        sa["Appointment Date"].notna(),
        sa["Actual Start"].notna(),
        sa["SchedStartTime"].notna(),
    ],
    [
        "Appointment Date",
        "Actual Start",
        "SchedStartTime",
    ],
    default="Unavailable",
)

recovered = int(sa["Appointment Date"].isna().sum() - sa["effective_date"].isna().sum())
record(
    "Appointment Date missing",
    "Filled from Actual Start / SchedStartTime",
    int(sa["Appointment Date"].isna().sum()),
    f"95.7% of Inspect & Fit jobs vs 0% of tyre fittings. {recovered} recovered.",
)

WINDOW_START = pd.Timestamp("2024-10-08")
WINDOW_END = pd.Timestamp("2024-12-22")
sa["in_demand_window"] = sa["effective_date"].between(WINDOW_START, WINDOW_END)
record(
    "Dated outside 8 Oct-22 Dec 2024",
    "Flagged, excluded from demand only",
    int((~sa["in_demand_window"]).sum()),
    "Window trimmed from both ends to avoid Created-Date censoring: the first/last "
    "few days of the raw window were undercounted because the extract filters on "
    "Created Date, not appointment date."
)

# 5. DurationType says 'Hours' but the values are already minutes.
#    Proven: every 'Inspect & Fit [N]' work type has a median of N x 60.
hours_rows = int((sa["DurationType"] == "Hours").sum())
record(
    "DurationType labelled 'Hours'",
    "No change - column is already minutes",
    hours_rows,
    "DurationType describes how the job was estimated, not the unit.",
)

# 6. Jobs left open on the device.
planned = sa["DurationInMinutes"].replace(0, np.nan)
ratio = sa["ActualDuration"] / planned
sa["duration_suspect"] = (sa["ActualDuration"] > 1440) | (
    (ratio > 4) & (sa["ActualDuration"] > 120)
)
sa["duration_clean"] = sa["ActualDuration"].where(~sa["duration_suspect"])
record(
    "Implausible ActualDuration",
    "Flagged, blanked in duration_clean",
    int(sa["duration_suspect"].sum()),
    "Over 24h, or over 4x planned. A flat cap would delete real 9.5h jobs.",
)

# %%
# --- Save the log -----------------------------------------------------
log_df = pd.DataFrame(log)
log_df.to_csv(OUTPUTS / "cleaning_log.csv", index=False)

print("\n" + "=" * 70)
print(log_df[["issue", "decision", "rows_affected"]].to_string(index=False))
print("=" * 70)

print(f"\nEffect on average job duration:")
print(f"  before flagging: {sa['ActualDuration'].mean():.1f} min")
print(f"  after flagging:  {sa['duration_clean'].mean():.1f} min")
print(f"  median (either): {sa['duration_clean'].median():.0f} min")

print(f"\nRows: {len(sa):,} total")
print(f"  real customer jobs:        {(~sa['is_placeholder']).sum():,}")
print(f"  in the demand window:      {(sa['is_placeholder'].eq(False) & sa['in_demand_window']).sum():,}")
# %%
# --- Join technician and territory information ------------------------
# Both join keys were checked for duplicates first. If either repeated,
# the merge would multiply rows instead of enriching them.

assert tech["ServiceResourceId"].is_unique, "technician key is not unique"
assert terr["District"].is_unique, "district key is not unique"

before = len(sa)

# Where the technician is based.
sa = sa.merge(
    tech[["ServiceResourceId", "Service_Resource_Name", "Territory",
          "Base Centre", "Base Centre Postal Code", "Working Hours"]]
    .rename(columns={
        "Service_Resource_Name": "tech_name",
        "Territory": "tech_territory",
        "Base Centre": "base_centre",
        "Base Centre Postal Code": "base_postcode",
        "Working Hours": "working_hours",
    }),
    on="ServiceResourceId",
    how="left",
)

# Where the customer is.
sa = sa.merge(
    terr.rename(columns={
        "Territory1": "customer_territory",
        "Restricted Territory name": "territory_restriction",
    }),
    on="District",
    how="left",
)
sa["region"] = "Group " + sa["customer_territory"].str.extract(r"^G(\d+)-")[0]
unmatched = sa["customer_territory"].notna() & sa["region"].isna()
print(f"customer_territory not matching G<N>- prefix: {unmatched.sum()}")
print(sa.loc[unmatched, "customer_territory"].value_counts().head(10))

assert len(sa) == before, f"join changed row count: {before} -> {len(sa)}"
print(f"joined, still {len(sa):,} rows")
print(f"  no technician matched: {sa['tech_territory'].isna().sum():,}")
print(f"  no territory matched:  {sa['region'].isna().sum():,}")

# %%
# --- Derived fields ---------------------------------------------------
sa["dow"] = sa["effective_date"].dt.day_name()
sa["dow_num"] = sa["effective_date"].dt.dayofweek
sa["iso_week"] = sa["effective_date"].dt.isocalendar().week.astype("Int64")
sa["is_weekend"] = sa["dow_num"] >= 5

# An appointment that did not deliver, whatever the reason.
sa["failed"] = sa["Status"].isin(["CannotComplete", "Canceled"])

# Who owns the failure. The team can only act on some of these.
FAILURE_OWNER = {
    "Wrong parts ordered": "Stock",
    "Tyres not in Stock": "Stock",
    "Stock too far for pickup": "Stock",
    "Centre refused to release stock": "Stock",
    "Locking Wheel Nut Unavailable": "Stock",
    "Cannot fit into Tech's Schedule": "Resource",
    "Tech sick": "Resource",
    "Tech running late": "Resource",
    "Van breakdown": "Resource",
    "Failed Attendance": "Resource",
    "Rearranged by Customer": "Customer",
    "Customer/Vehicle not present": "Customer",
    "Issue with Vehicle": "Customer",
    "Slot Time Not Suitable": "Customer",
    "Lead Time Not Suitable": "Customer",
    "Customer Emergency": "Customer",
    "Write off": "Customer",
    "Dups/Booked & Authed": "Booking admin",
    "Duplicate Booking (Cancelled ahead of time)": "Booking admin",
    "Job already done by KFM": "Booking admin",
    "Booking Error": "Booking admin",
    "Account Amendment": "Booking admin",
}
sa["failure_owner"] = sa["Sub_Categories__c"].map(FAILURE_OWNER)
sa.loc[sa["failed"] & sa["failure_owner"].isna(), "failure_owner"] = "Not recorded"


# --- Own-territory assignment -----------------------------------------
# Compare customer territory with technician home territory.
# Missing assignments should remain unknown, not False.

sa["own_territory_known"] = (
    sa["tech_territory"].notna()
    & sa["customer_territory"].notna()
)

sa["own_territory"] = (
    sa["tech_territory"]
    .eq(sa["customer_territory"])
    .astype("boolean")
    .where(sa["own_territory_known"], pd.NA)
)

# Demand booked into an area the business suspended.
sa["suspended_district"] = sa["territory_restriction"] == "Suspended"

# Booking behaviour.
sa["lead_band"] = pd.cut(
    sa["Lead Time"],
    bins=[-0.1, 0.5, 1.5, 3.5, 7.5, 1000],
    labels=["Same day", "Next day", "2-3 days", "4-7 days", "8+ days"],
)
sa["slot_hour"] = sa["Booked Slot Start Time"].dt.hour

sa["planned_minutes"] = sa["DurationInMinutes"]

# --- Final data-quality audit -----------------------------------------
# Audit only the operational appointment population.

audit = sa.loc[
    ~sa["is_placeholder"] & sa["in_demand_window"]
].copy()

missing_band = audit["lead_band"].isna()

outside_bins = (
    audit["Lead Time"].notna()
    & audit["lead_band"].isna()
)

missing_in_input = int(
    audit.loc[
        missing_band, "lead_time_missing_in_input"
    ].sum()
)

parse_errors = int(
    audit.loc[
        missing_band, "lead_time_parse_error"
    ].sum()
)

outside_count = int(outside_bins.sum())

print("\n" + "=" * 80)
print("FINAL CLEANING AUDIT")
print("=" * 80)

print(f"\nOperational appointments: {len(audit):,}")

print("\n1. LEAD-TIME DATA QUALITY")
print(f"Missing lead bands: {missing_band.sum():,}")
print(f"Already missing in input parquet: {missing_in_input:,}")
print(f"Numeric conversion failures: {parse_errors:,}")
print(f"Numeric values outside category bins: {outside_count:,}")

assert (
    missing_in_input + parse_errors + outside_count
    == int(missing_band.sum())
), "Unexplained missing lead-time categories"

print("\nWork types among missing lead times:")
print(
    audit.loc[missing_band, "Work Type: Work Type Name"]
    .value_counts(dropna=False)
    .head(10)
    .to_string()
)

print("\nStatuses among missing lead times:")
print(
    audit.loc[missing_band, "Status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n2. APPOINTMENT DATE SOURCES")
print(
    audit["effective_date_source"]
    .value_counts(dropna=False)
    .to_string()
)

assert audit["effective_date"].notna().all()

print("\n3. TERRITORY DATA QUALITY")
print(
    "Missing customer territory:",
    int(audit["customer_territory"].isna().sum())
)

print(
    "Missing technician home territory:",
    int(audit["tech_territory"].isna().sum())
)

print(
    "Unknown own-territory comparison:",
    int(audit["own_territory"].isna().sum())
)

print(
    "Missing/unmapped region:",
    int(audit["region"].isna().sum())
)

print("\n4. PLANNED DURATION DATA QUALITY")

invalid_duration = (
    audit["planned_minutes"].isna()
    | (audit["planned_minutes"] <= 0)
)

print(
    "Missing, zero or negative planned durations:",
    int(invalid_duration.sum())
)

print("\n5. APPOINTMENT STATUS DISTRIBUTION")
print(
    audit["Status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAll final cleaning audit checks passed.")


# %%
# --- Save -------------------------------------------------------------
sa.to_parquet(PROCESSED / "appointments.parquet", index=False)
print(f"\nSaved appointments.parquet  ({len(sa):,} rows, {sa.shape[1]} cols)")

analysis = sa[~sa["is_placeholder"] & sa["in_demand_window"]]
print(f"\nAnalysis base (real jobs, Oct-Dec): {len(analysis):,}")
print(f"  failure rate      : {analysis['failed'].mean():.1%}")
print(f"  regions covered   : {analysis['region'].nunique()}")
print(f"  territories        : {analysis['customer_territory'].nunique()}")
print(f"  technicians active : {analysis['ServiceResourceId'].nunique()}")
print("\nFailure ownership:")
print(analysis.loc[analysis["failed"], "failure_owner"].value_counts().to_string())
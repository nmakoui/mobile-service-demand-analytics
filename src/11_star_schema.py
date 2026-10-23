"""
Phase 5 — Export the Power BI star schema.

Run:  python src/11_star_schema.py
"""
from pathlib import Path
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PBI = ROOT / "outputs" / "powerbi"
PBI.mkdir(parents=True, exist_ok=True)

sa = pd.read_parquet(PROCESSED / "appointments.parquet")
jobs = sa[~sa["is_placeholder"] & sa["in_demand_window"]].copy()
jobs["Attended"] = jobs["Actual Start"].notna()
tech_raw = pd.read_parquet(PROCESSED / "tech_raw.parquet")


# --- Fact_Appointments (grain: one appointment) -----------------------

fact_appt = jobs[
    [
        "Id",
        "ServiceResourceId",
        "customer_territory",
        "effective_date",
        "effective_date_source",
        "Status",
        "failed",
        "failure_owner",
        "Sub_Categories__c",
        "planned_minutes",
        "Lead Time",
        "lead_band",
        "slot_hour",
        "Is Retail",
        "Work Type: Work Type Name",
        "own_territory",
        "own_territory_known",
        "suspended_district",
        "Attended",
    ]
].rename(
    columns={
        "Id": "AppointmentId",
        "ServiceResourceId": "TechnicianKey",
        "customer_territory": "TerritoryKey",
        "effective_date": "DateKey",
        "effective_date_source": "DateSource",
        "failed": "Failed",
        "failure_owner": "FailureOwner",
        "Sub_Categories__c": "FailureReason",
        "planned_minutes": "PlannedMinutes",
        "Lead Time": "LeadTimeDays",
        "lead_band": "LeadBand",
        "slot_hour": "BookedHour",
        "Is Retail": "IsRetail",
        "Work Type: Work Type Name": "WorkType",
        "own_territory": "OwnTerritory",
        "own_territory_known": "OwnTerritoryKnown",
        "suspended_district": "SuspendedDistrict",
    }
)

assert len(fact_appt) == len(jobs)
assert fact_appt["AppointmentId"].is_unique

fact_appt["DateKey"] = fact_appt["DateKey"].dt.date


# --- Subsequent appointment matching ----------------------------------

ftf_path = TABLES / "first_time_fix.csv"

if not ftf_path.exists():
    raise FileNotFoundError(
        "first_time_fix.csv is missing. "
        "Run 08_first_time_fix.py before this script."
    )

ftf = pd.read_csv(ftf_path)[
    [
        "Id",
        "repeat_eligible",
        "repeat_visit",
        "repeat_minutes",
    ]
].rename(
    columns={
        "Id": "AppointmentId",
        "repeat_eligible": "RepeatEligible",
        "repeat_visit": "RepeatVisit",
        "repeat_minutes": "RepeatMinutes",
    }
)

assert ftf["AppointmentId"].is_unique

fact_appt = fact_appt.merge(
    ftf,
    on="AppointmentId",
    how="left",
    validate="one_to_one",
)

# Non-failed appointments have no repeat eligibility in this
# failure-specific analysis.
eligibility = (
    fact_appt["RepeatEligible"]
    .astype("string")
    .str.strip()
    .str.lower()
)

fact_appt["RepeatEligible"] = (
    eligibility.eq("true")
    .fillna(False)
    .astype(bool)
)

# Preserve three states: True, False, or unknown.
repeat_value = (
    fact_appt["RepeatVisit"]
    .astype("string")
    .str.strip()
    .str.lower()
)

fact_appt["RepeatVisit"] = (
    repeat_value
    .map({"true": True, "false": False})
    .astype("boolean")
)

assert (
    fact_appt.loc[
        ~fact_appt["RepeatEligible"], "RepeatVisit"
    ].isna().all()
)

assert (
    fact_appt.loc[
        fact_appt["RepeatEligible"], "Failed"
    ].all()
)

print("\nREPEAT APPOINTMENT EXPORT CHECK")

print(
    "Eligible failed appointments:",
    int(fact_appt["RepeatEligible"].sum())
)

print(
    "Observed subsequent appointments:",
    int(fact_appt["RepeatVisit"].fillna(False).sum())
)

print(
    "Unscoreable / not applicable:",
    int(fact_appt["RepeatVisit"].isna().sum())
)

fact_appt.to_csv(PBI / "fact_appointments.csv", index=False)
print(f"Fact_Appointments: {len(fact_appt):,} rows")

# --- Dim_Technician -------------------------------------------------
dim_tech = tech_raw[[
    "ServiceResourceId", "Service_Resource_Name", "Territory",
    "Base Centre", "Base Centre Postal Code", "Working Hours",
]].rename(columns={
    "ServiceResourceId": "TechnicianKey", "Service_Resource_Name": "TechName",
    "Territory": "HomeTerritory", "Base Centre": "BaseCentre",
    "Base Centre Postal Code": "BaseCentrePostcode", "Working Hours": "WorkingHoursPattern",
})
dim_tech["HomeRegion"] = "Group " + dim_tech["HomeTerritory"].str.extract(r"^G(\d+)-")[0]
dim_tech.to_csv(PBI / "dim_technician.csv", index=False)
print(f"Dim_Technician: {len(dim_tech):,} rows")

# --- Dim_Territory ----------------------------------------------------
dim_terr = (
    jobs[["customer_territory", "region", "territory_restriction"]]
    .dropna(subset=["customer_territory"])
    .drop_duplicates(subset=["customer_territory"])
    .rename(columns={"customer_territory": "TerritoryKey", "region": "Region",
                      "territory_restriction": "RestrictionStatus"})
)
dupes = dim_terr["TerritoryKey"].duplicated().sum()
print(f"Dim_Territory key duplicates (should be 0): {dupes}")
dim_terr["TerritoryName"] = dim_terr["TerritoryKey"].str.replace(r"^G\d+-\d+-", "", regex=True)
dim_terr.to_csv(PBI / "dim_territory.csv", index=False)
print(f"Dim_Territory: {len(dim_terr):,} rows")

# --- Fact_TechnicianDay (grain: technician x date — a second, smaller fact) ---
tdp = TABLES / "travel_by_tech_day.csv"
if tdp.exists():
    tech_day = pd.read_csv(tdp).rename(
        columns={"ServiceResourceId": "TechnicianKey", "effective_date": "DateKey"}
    )
    tech_day.to_csv(PBI / "fact_technician_day.csv", index=False)
    print(f"Fact_TechnicianDay: {len(tech_day):,} rows")
else:
    print("travel_by_tech_day.csv not found - run 07_travel.py first for this table.")
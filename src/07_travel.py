
"""
Exploratory technician-day activity and geographic movement analysis.

Straight-line distances between recorded-start appointments are
illustrative geographic proxies, not actual driven routes.

Run: python src/07_travel.py
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

KEYS = ["ServiceResourceId", "effective_date"]

# -----------------------------------------------------------
# 1. All observed technician-appointment-days
# -----------------------------------------------------------

base = (
    jobs.dropna(subset=KEYS)
    .groupby(KEYS)
    .agg(
        jobs=("Id", "size"),
        job_minutes=("planned_minutes", "sum"),
    )
)

base["job_hours"] = base["job_minutes"] / 60

# -----------------------------------------------------------
# 2. Appointments with recorded starts and valid coordinates
# -----------------------------------------------------------

geo = jobs.loc[
    jobs["Actual Start"].notna()
].dropna(
    subset=KEYS + ["Latitude", "Longitude"]
).copy()

valid_coordinates = (
    geo["Latitude"].between(-90, 90)
    & geo["Longitude"].between(-180, 180)
    & ~(
        geo["Latitude"].eq(0)
        & geo["Longitude"].eq(0)
    )
)

geo = geo.loc[valid_coordinates].copy()

geo = geo.sort_values(
    KEYS + ["Actual Start", "Id"]
)

grouped = geo.groupby(KEYS)

geo["previous_lat"] = grouped["Latitude"].shift()
geo["previous_lon"] = grouped["Longitude"].shift()

# -----------------------------------------------------------
# 3. Haversine straight-line distance
# -----------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    return 2 * radius_km * np.arcsin(np.sqrt(a))


geo["straight_line_leg_km"] = haversine_km(
    geo["previous_lat"],
    geo["previous_lon"],
    geo["Latitude"],
    geo["Longitude"],
)

# Only legs between successive geocoded appointments with
# recorded starts are represented. Depot journeys and
# movements involving unobserved stops are not measured.

legs = geo.dropna(
    subset=["straight_line_leg_km"]
)

leg_summary = (
    legs.groupby(KEYS)
    .agg(
        observable_legs=("Id", "size"),
        straight_line_km=(
            "straight_line_leg_km", "sum"
        ),
    )
)

# -----------------------------------------------------------
# 4. Combine with the full technician-day population
# -----------------------------------------------------------

tech_day = base.join(
    leg_summary,
    how="left"
)

tech_day["observable_legs"] = (
    tech_day["observable_legs"]
    .fillna(0)
    .astype(int)
)

# No observed leg means distance is UNKNOWN, not zero.
tech_day.loc[
    tech_day["observable_legs"] == 0,
    "straight_line_km"
] = np.nan

# Optional illustration only. This is not actual travel time.
ASSUMED_SPEED_KMH = 30

tech_day["illustrative_minutes_at_30kmh"] = (
    tech_day["straight_line_km"]
    / ASSUMED_SPEED_KMH
    * 60
)

tech_day = tech_day.reset_index()

# -----------------------------------------------------------
# 5. Validation and export
# -----------------------------------------------------------

national_tech_days = (
    jobs.dropna(subset=KEYS)
    .drop_duplicates(KEYS)
    .shape[0]
)

assert len(tech_day) == national_tech_days

assert tech_day["jobs"].sum() == len(
    jobs.dropna(subset=KEYS)
)

tech_day.to_csv(
    TABLES / "travel_by_tech_day.csv",
    index=False
)

print("\n" + "=" * 80)
print("TECHNICIAN-DAY AND EXPLORATORY MOVEMENT CHECK")
print("=" * 80)

print(f"Observed technician-days: {len(tech_day):,}")

print(
    "Days with at least one observed geographic leg:",
    f"{(tech_day['observable_legs'] > 0).sum():,}"
)

print(
    "Days without an observable geographic leg:",
    f"{(tech_day['observable_legs'] == 0).sum():,}"
)

print(
    "Median straight-line distance on days with observed legs:",
    f"{tech_day['straight_line_km'].median():.1f} km"
)

print("\nLIMITATIONS:")
print(
    "Straight-line distances are not road distances."
)

print(
    "Journeys from or to a technician's base are not included."
)

print(
    "Appointments lacking valid coordinates or recorded starts "
    "cannot contribute to the observed geographic legs."
)

print(
    "The assumed 30 km/h speed is illustrative only."
)

print(
    "Booked job hours are planned workload, "
    "not actual technician working hours."
)

print("\nAll technician-day checks passed.")

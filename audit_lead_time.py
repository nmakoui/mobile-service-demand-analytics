
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

sa = pd.read_parquet(
    ROOT / "data" / "processed" / "appointments.parquet"
)

jobs = sa[
    ~sa["is_placeholder"] & sa["in_demand_window"]
].copy()

raw = jobs["Lead Time"]
lead = pd.to_numeric(raw, errors="coerce")
missing_band = jobs["lead_band"].isna()

print("\n=== LEAD-TIME DIAGNOSTIC ===")
print(f"Total appointments: {len(jobs):,}")
print(f"Missing lead bands: {missing_band.sum():,}")

checks = {
    "Lead Time is null": raw.isna(),
    "Lead Time is non-numeric": raw.notna() & lead.isna(),
    "Lead Time <= -0.1 days": lead <= -0.1,
    "Lead Time > 1000 days": lead > 1000,
    "Within valid range but uncategorised": (
        lead.notna() & (lead > -0.1) & (lead <= 1000)
    ),
}

print("\n=== WHY LEAD BANDS ARE MISSING ===")

total_explained = 0

for label, condition in checks.items():
    count = int((missing_band & condition).sum())
    total_explained += count
    print(f"{label}: {count:,}")

print(f"\nTotal explained: {total_explained:,}")
print(f"Total missing bands: {missing_band.sum():,}")

print("\n=== ORIGINAL LEAD TIME VALUES ===")
print(
    lead.loc[missing_band]
    .value_counts(dropna=False)
    .head(20)
    .to_string()
)

print("\n=== NEGATIVE LEAD TIMES ===")
print(f"All negative lead times: {(lead < 0).sum():,}")

print("\n=== FAILURE RATES ===")
print(
    jobs.assign(missing_lead_band=missing_band)
    .groupby("missing_lead_band")["failed"]
    .agg(["size", "sum", "mean"])
    .to_string()
)

print("\n=== DIAGNOSTIC COMPLETE ===")

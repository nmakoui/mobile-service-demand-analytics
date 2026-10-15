"""
Convert the raw Kwik Fit files into fast parquet copies.

Run once:  python src/00_convert.py
The raw files in data/raw are never modified.
"""
# %%
import time
from pathlib import Path

import pandas as pd

# Works both when run as a script and when run cell-by-cell in VS Code.
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# %%
# --- Load -------------------------------------------------------------
# SA.xlsb is binary Excel, so it needs the pyxlsb engine. This is the slow bit.
t0 = time.time()
sa = pd.read_excel(RAW / "SA.xlsb", engine="pyxlsb")
print(f"SA.xlsb loaded in {time.time() - t0:.0f}s")

tech = pd.read_excel(RAW / "Tech.xlsx")

# utf-8-sig strips the hidden byte-order mark at the start of the file.
# Without it the first column is named 'ï»¿District' and every join on it fails.
terr = pd.read_csv(RAW / "Terr.csv", encoding="utf-8-sig")

print(f"sa    {sa.shape[0]:>7,} rows x {sa.shape[1]:>2} cols")
print(f"tech  {tech.shape[0]:>7,} rows x {tech.shape[1]:>2} cols")
print(f"terr  {terr.shape[0]:>7,} rows x {terr.shape[1]:>2} cols")

# %%
# --- Repair: header rows pasted into the middle of the data -----------
# Someone has appended exports together and left the header line in twice.
# These rows have the literal text "Id" in the Id column.
# They are dropped here rather than in the cleaning step because the file
# cannot be saved properly while a numeric column also contains text.
header_rows = sa["Id"] == "Id"
print(f"\nEmbedded header rows found: {header_rows.sum()}")
sa = sa.loc[~header_rows].copy()
print(f"SA rows after repair: {len(sa):,}")

# %%
# --- Check for columns pandas could not type cleanly ------------------
# Any column holding more than one Python type is a sign of messy source data.
for col in sa.columns:
    kinds = sa[col].dropna().map(type).unique()
    if len(kinds) > 1:
        print(f"  mixed types in '{col}': {[k.__name__ for k in kinds]}")

# %%
# --- Save -------------------------------------------------------------
sa.to_parquet(PROCESSED / "sa_raw.parquet", index=False)
tech.to_parquet(PROCESSED / "tech_raw.parquet", index=False)
terr.to_parquet(PROCESSED / "terr_raw.parquet", index=False)
print("\nSaved three parquet files to data/processed/")
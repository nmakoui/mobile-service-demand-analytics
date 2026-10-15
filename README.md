# Kwik Fit Mobile — Resource & Demand Planning Assessment

Analysis of service appointment, technician and territory data for the
Junior Resource & Demand Planning Analyst assessment task.

## Running it

python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy pyxlsb openpyxl pyarrow matplotlib scikit-learn
python src/00_convert.py # once: raw files -> parquet
python src/01_clean.py # cleaning, joins, analysis base


## Structure

- `src/00_convert.py` — reads the raw files, repairs structural defects, saves parquet
- `src/01_clean.py` — data-quality fixes, joins, derived fields, analysis base
- `outputs/tables/cleaning_log.csv` — every cleaning decision and how many rows it touched

## Data

Source files are not in this repository. They contain company operational
data and are excluded by `.gitignore`.
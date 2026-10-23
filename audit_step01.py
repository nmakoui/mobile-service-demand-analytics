
from pathlib import Path
import csv
import os
import re

ROOT = Path(__file__).resolve().parent

SKIP_FOLDERS = {
    ".venv", "venv", ".git", "__pycache__",
    "node_modules", ".idea"
}

EXTENSIONS = {
    ".py", ".csv", ".pbix", ".xlsx",
    ".parquet", ".sql", ".md"
}

files = []

for folder, subfolders, filenames in os.walk(ROOT):
    subfolders[:] = [
        d for d in subfolders if d not in SKIP_FOLDERS
    ]

    for filename in filenames:
        path = Path(folder) / filename
        if path.suffix.lower() in EXTENSIONS:
            files.append(path)

files.sort()

def relative(path):
    return str(path.relative_to(ROOT))

print("\n=== 1. PROJECT FILES ===")

for path in files:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"{relative(path)}  ({size_mb:.2f} MB)")

print("\n=== 2. CSV DATASET COLUMNS ===")

for path in files:
    if path.suffix.lower() != ".csv":
        continue

    print(f"\nFILE: {relative(path)}")

    try:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as file:
            reader = csv.reader(file)
            columns = next(reader, [])
            print("COLUMNS:", ", ".join(columns))
    except Exception as error:
        print("READ ERROR:", error)

print("\n=== 3. POTENTIALLY MISLEADING METRICS ===")

pattern = re.compile(
    r"coverage|slack|utili[sz]ation|"
    r"lost.?hours|technician.?days?|"
    r"own.?territory|repeat.?visit",
    re.IGNORECASE
)

matches = 0

for path in files:
    if path.suffix.lower() not in {".py", ".sql", ".md"}:
        continue

    if path.stat().st_size > 2_000_000:
        continue

    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

        for line_number, line in enumerate(lines, 1):
            if pattern.search(line):
                print(
                    f"{relative(path)}:{line_number}: "
                    f"{line.strip()[:180]}"
                )
                matches += 1

                if matches >= 100:
                    print("Stopped after 100 matches.")
                    break

        if matches >= 100:
            break

    except Exception as error:
        print(f"ERROR in {relative(path)}: {error}")

print("\n=== AUDIT COMPLETE ===")

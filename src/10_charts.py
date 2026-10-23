
"""
Four supporting charts for the Resource and Demand Planning assessment.

Run: python src/10_charts.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
CHARTS = ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------
# CHART 1 — Demand rank comparison
# -----------------------------------------------------------

demand = pd.read_csv(
    TABLES / "demand_by_territory.csv"
)

top = demand.nsmallest(15, "rank_hours").copy()

leicester = demand.loc[
    demand["customer_territory"]
    .str.contains("Leicester", case=False, na=False)
].iloc[0]

fig, ax = plt.subplots(figsize=(8, 8))

for _, row in top.iterrows():
    is_leicester = (
        "Leicester" in row["customer_territory"]
    )

    color = "#D35445" if is_leicester else "#BBBBBB"

    ax.plot(
        [0, 1],
        [row["rank_jobs"], row["rank_hours"]],
        color=color,
        linewidth=3 if is_leicester else 1,
        marker="o",
    )

    ax.text(
        -0.05,
        row["rank_jobs"],
        row["customer_territory"],
        ha="right",
        va="center",
        fontsize=8,
        color=color,
    )

ax.set_xlim(-0.65, 1.1)
ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Rank by appointments", "Rank by planned hours"]
)

ax.invert_yaxis()
ax.set_yticks([])

ax.set_title(
    f"Leicester: #{int(leicester['rank_jobs'])} by appointments, "
    f"#{int(leicester['rank_hours'])} by planned hours",
    fontsize=12,
    fontweight="bold",
)

clean_axes(ax)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig(CHARTS / "01_slope_leicester.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------
# CHART 2 — Regional demand intensity
# -----------------------------------------------------------

reg = pd.read_csv(
    TABLES / "region_supply_demand.csv"
)

fig, ax = plt.subplots(figsize=(8, 6))

for _, row in reg.iterrows():
    ax.scatter(
        row["jobs_per_active_tech"],
        row["hours_per_active_tech"],
        s=200,
        edgecolor="white",
        linewidth=1.5,
    )

    ax.annotate(
        row["region"],
        (
            row["jobs_per_active_tech"],
            row["hours_per_active_tech"],
        ),
        xytext=(7, 4),
        textcoords="offset points",
    )

ax.set_xlabel(
    "Customer-region appointments per active home-region technician"
)

ax.set_ylabel(
    "Customer-region planned hours per active home-region technician"
)

ax.set_title(
    "Regional demand intensity differs by work mix",
    fontsize=12,
    fontweight="bold",
)

ax.text(
    0.5,
    -0.23,
    "Diagnostic comparison only: not technician utilisation or spare capacity.\n"
    "Technicians may serve customers outside their home region.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

clean_axes(ax)
plt.tight_layout()
plt.savefig(CHARTS / "02_region_scatter.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------
# CHART 3 — Recorded starts among failed stock appointments
# -----------------------------------------------------------

stock = pd.read_csv(
    TABLES / "stock_failure_start_analysis.csv"
)

reasons = [
    "Wrong parts ordered",
    "Tyres not in Stock",
    "Locking Wheel Nut Unavailable",
]

stock = stock.set_index("stock_failure_reason")
bench = stock.loc[reasons].copy()

bench = bench.sort_values("recorded_start_pct")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(
    bench.index,
    bench["recorded_start_pct"],
    color=[
        "#D35445" if p >= 50 else "#447FA3"
        for p in bench["recorded_start_pct"]
    ],
)

for bar, (_, row) in zip(bars, bench.iterrows()):
    ax.text(
        bar.get_width() + 1.5,
        bar.get_y() + bar.get_height() / 2,
        (
            f"{row['recorded_start_pct']:.1f}% "
            f"(n={int(row['failed_appointments']):,})"
        ),
        va="center",
        fontsize=10,
    )

ax.set_xlim(0, 120)

ax.set_xlabel(
    "Failed appointments with a recorded Actual Start (%)"
)

ax.set_title(
    "Recorded-start rates differ sharply by stock failure reason",
    fontsize=12,
    fontweight="bold",
)

ax.text(
    0.5,
    -0.22,
    "Recorded start is a proxy, not confirmation of technician arrival.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

clean_axes(ax)
plt.tight_layout()
plt.savefig(CHARTS / "03_benchmark_bar.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------
# CHART 4 — Lead time and failure rate
# -----------------------------------------------------------

lead = pd.read_csv(
    TABLES / "booking_lead_time_audit.csv"
)

order = [
    "Same day",
    "Next day",
    "2-3 days",
    "4-7 days",
    "8+ days",
    "Missing lead band",
]

lead = (
    lead.set_index("lead_band")
    .reindex(order)
)

assert lead["appointments"].notna().all()

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    range(len(lead)),
    lead["failure_rate_pct"],
    color=[
        "#D35445" if name == "8+ days" else "#8D9DAA"
        for name in order
    ],
)

for bar, (_, row) in zip(bars, lead.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        (
            f"{row['failure_rate_pct']:.1f}%\n"
            f"n={int(row['appointments']):,}"
        ),
        ha="center",
        fontsize=9,
    )

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=15, ha="right")
ax.set_ylabel("Observed failure rate (%)")
ax.set_ylim(0, max(lead["failure_rate_pct"]) * 1.3)

ax.set_title(
    "Failure rates vary with booking lead time",
    fontsize=12,
    fontweight="bold",
)

ax.text(
    0.5,
    -0.30,
    "Descriptive association, not causation.\n"
    "Missing lead-time records are concentrated in Inspect & Fit work.",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

clean_axes(ax)
plt.tight_layout()
plt.savefig(CHARTS / "04_leadtime_bar.png", dpi=200)
plt.close(fig)

print("Saved 4 supporting charts to outputs/charts/")

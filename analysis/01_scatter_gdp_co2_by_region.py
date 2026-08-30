import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_FILE = FIGURES_DIR / "scatter_gdp_co2_by_region.png"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_FILE)

# -----------------------------
# Check required columns
# -----------------------------
required_cols = ["region", "year", "gdp_pc_region", "co2_pc_region"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# -----------------------------
# Prepare output folder
# -----------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sort data for line connections
# -----------------------------
df = df.sort_values(["region", "year"]).copy()

# -----------------------------
# Make plot
# -----------------------------
plt.figure(figsize=(10, 7))

regions = df["region"].unique()

for region in regions:
    sub = df[df["region"] == region].copy()

    # Scatter points
    plt.scatter(
        sub["gdp_pc_region"],
        sub["co2_pc_region"],
        label=region,
        alpha=0.8
    )

    # Connect points through time
    plt.plot(
        sub["gdp_pc_region"],
        sub["co2_pc_region"],
        alpha=0.6
    )

    # Label first and last year
    first_row = sub.iloc[0]
    last_row = sub.iloc[-1]

    plt.annotate(
        str(int(first_row["year"])),
        (first_row["gdp_pc_region"], first_row["co2_pc_region"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

    plt.annotate(
        str(int(last_row["year"])),
        (last_row["gdp_pc_region"], last_row["co2_pc_region"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

# -----------------------------
# Labels and formatting
# -----------------------------
plt.xlabel("Regional GDP per capita (constant USD)")
plt.ylabel("Regional CO₂ emissions per capita")
plt.title("GDP per Capita vs CO₂ Emissions per Capita by Region-Year")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Save and show
# -----------------------------
plt.savefig(Output_FILE := OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {Output_FILE}")

plt.show()

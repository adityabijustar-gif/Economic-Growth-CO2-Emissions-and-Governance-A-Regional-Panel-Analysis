import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_FILE = FIGURES_DIR / "gdp_trends_by_region.png"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_FILE)

# -----------------------------
# Check required columns
# -----------------------------
required_cols = ["region", "year", "gdp_pc_region"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# -----------------------------
# Prepare output folder
# -----------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sort data
# -----------------------------
df = df.sort_values(["region", "year"]).copy()

# -----------------------------
# Make plot
# -----------------------------
plt.figure(figsize=(10, 7))

regions = df["region"].unique()

for region in regions:
    sub = df[df["region"] == region].copy()

    plt.plot(
        sub["year"],
        sub["gdp_pc_region"],
        marker="o",
        label=region
    )

# -----------------------------
# Labels and formatting
# -----------------------------
plt.xlabel("Year")
plt.ylabel("Regional GDP per capita (constant USD)")
plt.title("Regional GDP per Capita Trends Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Save and show
# -----------------------------
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_FILE}")

plt.show()

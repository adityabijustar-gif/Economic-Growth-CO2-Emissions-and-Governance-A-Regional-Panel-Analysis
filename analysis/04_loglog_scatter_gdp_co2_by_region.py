import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
 
# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_FILE = FIGURES_DIR / "loglog_scatter_gdp_co2_by_region.png"
 
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
# Keep only positive values
# -----------------------------
df = df.dropna(subset=required_cols).copy()
df = df[(df["gdp_pc_region"] > 0) & (df["co2_pc_region"] > 0)].copy()

# -----------------------------
# Log transform
# -----------------------------
df["log_gdp_pc_region"] = np.log(df["gdp_pc_region"])
df["log_co2_pc_region"] = np.log(df["co2_pc_region"])

# -----------------------------
# Prepare output folder
# -----------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sort for time paths
# -----------------------------
df = df.sort_values(["region", "year"]).copy()

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 7))

regions = df["region"].unique()

for region in regions:
    sub = df[df["region"] == region].copy()

    plt.scatter(
        sub["log_gdp_pc_region"],
        sub["log_co2_pc_region"],
        label=region,
        alpha=0.8
    )

    plt.plot(
        sub["log_gdp_pc_region"],
        sub["log_co2_pc_region"],
        alpha=0.6
    )

    first_row = sub.iloc[0]
    last_row = sub.iloc[-1]

    plt.annotate(
        str(int(first_row["year"])),
        (first_row["log_gdp_pc_region"], first_row["log_co2_pc_region"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

    plt.annotate(
        str(int(last_row["year"])),
        (last_row["log_gdp_pc_region"], last_row["log_co2_pc_region"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

# -----------------------------
# Labels and formatting
# -----------------------------
plt.xlabel("Log of regional GDP per capita (constant USD)")
plt.ylabel("Log of regional CO₂ emissions per capita")
plt.title("Log-Log GDP per Capita vs CO₂ Emissions per Capita by Region-Year")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Save and show
# -----------------------------
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_FILE}")

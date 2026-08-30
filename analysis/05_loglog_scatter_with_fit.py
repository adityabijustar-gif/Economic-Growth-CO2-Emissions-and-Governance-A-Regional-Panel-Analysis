
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_FILE = FIGURES_DIR / "loglog_scatter_with_fit.png"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_FILE)
 
# -----------------------------
# Check required columns
# -----------------------------
required_cols = ["region", "year", "gdp_pc_region", "co2_pc_region"]
missing_cols = [col for col in required_cols if col not in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
 
# -----------------------------
# Clean and log transform
# -----------------------------
df = df.dropna(subset=required_cols).copy()
df = df[(df["gdp_pc_region"] > 0) & (df["co2_pc_region"] > 0)].copy()
 
df["log_gdp_pc_region"] = np.log(df["gdp_pc_region"])
df["log_co2_pc_region"] = np.log(df["co2_pc_region"])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Fit simple linear line in logs
# -----------------------------
x = df["log_gdp_pc_region"].values
y = df["log_co2_pc_region"].values

slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(x.min(), x.max(), 200)
y_line = intercept + slope * x_line

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 7))

for region in df["region"].unique():
    sub = df[df["region"] == region]
    plt.scatter(
        sub["log_gdp_pc_region"],
        sub["log_co2_pc_region"],
        label=region,
        alpha=0.75
    )

plt.plot(x_line, y_line, linewidth=2, label=f"Overall fit: slope = {slope:.2f}")

plt.xlabel("Log of regional GDP per capita (constant USD)")
plt.ylabel("Log of regional CO₂ emissions per capita")
plt.title("Log-Log GDP per Capita vs CO₂ Emissions per Capita with Fitted Line")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_FILE}")

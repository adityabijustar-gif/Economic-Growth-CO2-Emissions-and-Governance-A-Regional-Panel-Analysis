import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_FILE = FIGURES_DIR / "ekc_quadratic_loglog_plot.png"
 
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
# Clean and log transform
# -----------------------------
df = df.dropna(subset=required_cols).copy()
df = df[(df["gdp_pc_region"] > 0) & (df["co2_pc_region"] > 0)].copy()
 
df["log_gdp_pc_region"] = np.log(df["gdp_pc_region"])
df["log_co2_pc_region"] = np.log(df["co2_pc_region"])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Quadratic fit
# y = a + b*x + c*x^2
# -----------------------------
x = df["log_gdp_pc_region"].values
y = df["log_co2_pc_region"].values

c2, c1, c0 = np.polyfit(x, y, 2)

x_curve = np.linspace(x.min(), x.max(), 300)
y_curve = c2 * x_curve**2 + c1 * x_curve + c0

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

plt.plot(x_curve, y_curve, linewidth=2, label="Quadratic EKC fit")

plt.xlabel("Log of regional GDP per capita (constant USD)")
plt.ylabel("Log of regional CO₂ emissions per capita")
plt.title("EKC-Style Quadratic Fit: Log GDP per Capita vs Log CO₂ per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_FILE}")

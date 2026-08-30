import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "region_year_panel.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures" / "loglog_by_region"

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

# -----------------------------
# Prepare output folder
# -----------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Create one plot per region
# -----------------------------
for region in df["region"].unique():
    sub = df[df["region"] == region].sort_values("year").copy()

    x = sub["log_gdp_pc_region"].values
    y = sub["log_co2_pc_region"].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.8)
    plt.plot(x, y, alpha=0.6)

    if len(sub) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = intercept + slope * x_line
        plt.plot(x_line, y_line, linewidth=2, label=f"Fitted slope = {slope:.2f}")

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

    plt.xlabel("Log of regional GDP per capita (constant USD)")
    plt.ylabel("Log of regional CO₂ emissions per capita")
    plt.title(f"Log-Log GDP vs CO₂: {region}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = FIGURES_DIR / f"loglog_{region}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_file}")

import pandas as pd
from pathlib import Path


def pop_weighted_mean(df, value_col, weight_col):
    return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()


# Make paths work regardless of where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parents[1]

panel_path = PROJECT_ROOT / "data" / "cleaned" / "wdi_gdp_co2_population_panel.csv"
regions_path = PROJECT_ROOT / "data" / "meta" / "country_regions.csv"
output_dir = PROJECT_ROOT / "data" / "analysis"
output_file = output_dir / "region_year_panel.csv"

# Load files
panel = pd.read_csv(panel_path)
regions = pd.read_csv(regions_path)

# Merge region mapping into the country-year panel
panel = panel.merge(regions, on="iso3c", how="left")

# Check for countries with no region assigned
missing = panel.loc[panel["region"].isna(), "iso3c"].dropna().unique()
if len(missing) > 0:
    raise ValueError(
        f"Missing region assignment for {len(missing)} countries: {missing}"
    )

# Build region-year panel using population-weighted means
region_panel = (
    panel
    .dropna(subset=[
        "population",
        "gdp_per_capita_const2015_usd",
        "co2_per_capita_tons"
    ])
    .groupby(["region", "year"])
    .apply(lambda x: pd.Series({
        "gdp_pc_region": pop_weighted_mean(
            x, "gdp_per_capita_const2015_usd", "population"
        ),
        "co2_pc_region": pop_weighted_mean(
            x, "co2_per_capita_tons", "population"
        ),
        "population_region": x["population"].sum()
    }))
    .reset_index()
)

# Save output
output_dir.mkdir(parents=True, exist_ok=True)
region_panel.to_csv(output_file, index=False)

# Print checks
print(region_panel.head())
print("Regions:", region_panel["region"].unique())
print(region_panel.groupby("region").size())
print("Years:", region_panel["year"].min(), "-", region_panel["year"].max())
print("Saved to:", output_file)

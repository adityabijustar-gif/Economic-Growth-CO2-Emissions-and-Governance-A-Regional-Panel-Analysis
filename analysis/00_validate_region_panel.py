from pathlib import Path
import numpy as np
import pandas as pd
# ============================================================
# 1. PROJECT PATHS
# ============================================================
# This allows the script to work regardless of the current terminal directory, as long as it remains inside /analysis.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNTRY_PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "wdi_gdp_co2_population_panel.csv"
)
REGION_MAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "meta"
    / "country_regions.csv"
)
REGION_PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "region_year_panel.csv"
)
QC_DIR = PROJECT_ROOT / "data" / "analysis" / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# 2. PROJECT EXPECTATIONS
# ============================================================
EXPECTED_REGIONS = {
    "Europe_NorthAmerica",
    "DevelopedAsia_Oceania",
    "China",
    "India",
    "Global_South",
}
EXPECTED_START_YEAR = 1990
# This is not an exclusion threshold. It simply generates a warning where the complete-case countries account for less than 95% of observed regional population in a particular year.
POPULATION_COVERAGE_WARNING_THRESHOLD = 0.95
# Numerical tolerance when reproducing saved regional averages.
RTOL = 1e-8
ATOL = 1e-10
# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
errors = []
warnings = []
def require_columns(df, required, dataset_name):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )
def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
# ============================================================
# 4. LOAD DATA
# ============================================================
print_section("LOADING DATA")
country_panel = pd.read_csv(COUNTRY_PANEL_FILE)
region_map = pd.read_csv(REGION_MAP_FILE)
region_panel = pd.read_csv(REGION_PANEL_FILE)
print(f"Country-year panel: {COUNTRY_PANEL_FILE}")
print(f"Rows: {len(country_panel):,}")
print(f"\nRegion mapping: {REGION_MAP_FILE}")
print(f"Rows: {len(region_map):,}")
print(f"\nSaved region-year panel: {REGION_PANEL_FILE}")
print(f"Rows: {len(region_panel):,}")
# ============================================================
# 5. REQUIRED COLUMN CHECKS
# ============================================================
require_columns(
    country_panel,
    [
        "country",
        "iso3c",
        "year",
        "gdp_per_capita_const2015_usd",
        "co2_per_capita_tons",
        "population",
    ],
    "Country-year panel",
)
require_columns(
    region_map,
    [
        "iso3c",
        "region",
    ],
    "Region mapping",
)
require_columns(
    region_panel,
    [
        "region",
        "year",
        "gdp_pc_region",
        "co2_pc_region",
        "population_region",
    ],
    "Region-year panel",
)
# ============================================================
# 6. BASIC TYPE STANDARDISATION
# ============================================================
# Check missing mapping identifiers BEFORE converting to strings.
if region_map[["iso3c", "region"]].isna().any().any():
    errors.append(
        "country_regions.csv contains missing iso3c or region values."
    )
country_panel["iso3c"] = (
    country_panel["iso3c"]
    .astype(str)
    .str.strip()
)
region_map["iso3c"] = (
    region_map["iso3c"]
    .astype(str)
    .str.strip()
)
region_map["region"] = (
    region_map["region"]
    .astype(str)
    .str.strip()
)
region_panel["region"] = (
    region_panel["region"]
    .astype(str)
    .str.strip()
)
country_panel["year"] = pd.to_numeric(
    country_panel["year"],
    errors="raise",
).astype(int)
region_panel["year"] = pd.to_numeric(
    region_panel["year"],
    errors="raise",
).astype(int)
# ============================================================
# 7. REGION MAPPING QC
# ============================================================
print_section("REGION MAPPING QC")
duplicate_mapping = region_map[
    region_map.duplicated("iso3c", keep=False)
].sort_values("iso3c")
if not duplicate_mapping.empty:
    duplicate_mapping.to_csv(
        QC_DIR / "duplicate_region_assignments.csv",
        index=False,
    )
    errors.append(
        f"{duplicate_mapping['iso3c'].nunique()} ISO codes have "
        "more than one region assignment."
    )
actual_regions = set(region_map["region"].unique())
missing_expected_regions = EXPECTED_REGIONS - actual_regions
unexpected_regions = actual_regions - EXPECTED_REGIONS
if missing_expected_regions:
    errors.append(
        "Expected region labels missing from country_regions.csv: "
        f"{sorted(missing_expected_regions)}"
    )
if unexpected_regions:
    errors.append(
        "Unexpected region labels found in country_regions.csv: "
        f"{sorted(unexpected_regions)}"
    )
print("\nCountries mapped by region:")
print(
    region_map.groupby("region")["iso3c"]
    .nunique()
    .sort_index()
)
mapped_codes = set(region_map["iso3c"])
panel_codes = set(country_panel["iso3c"].dropna())
mapping_codes_not_in_panel = sorted(
    mapped_codes - panel_codes
)
if mapping_codes_not_in_panel:
    pd.DataFrame(
        {"iso3c": mapping_codes_not_in_panel}
    ).to_csv(
        QC_DIR / "mapped_codes_not_found_in_country_panel.csv",
        index=False,
    )
    warnings.append(
        f"{len(mapping_codes_not_in_panel)} mapped ISO codes do not "
        "appear in the cleaned country-year panel."
    )
# These are not automatically errors because WDI contains regional/income aggregates as well as countries.
unmapped_panel_codes = sorted(
    panel_codes - mapped_codes
)
pd.DataFrame(
    {"iso3c": unmapped_panel_codes}
).to_csv(
    QC_DIR / "panel_codes_not_in_region_map.csv",
    index=False,
)
# ============================================================
# 8. COUNTRY-YEAR KEY CHECK
# ============================================================
print_section("COUNTRY-YEAR PANEL STRUCTURE")
country_year_duplicates = country_panel[
    country_panel.duplicated(
        ["iso3c", "year"],
        keep=False,
    )
].sort_values(["iso3c", "year"])
if not country_year_duplicates.empty:
    country_year_duplicates.to_csv(
        QC_DIR / "duplicate_country_year_rows.csv",
        index=False,
    )
    errors.append(
        f"{len(country_year_duplicates)} duplicated country-year rows "
        "exist in the cleaned panel."
    )
else:
    print("PASS: No duplicated country-year observations.")
# ============================================================
# 9. REGION-YEAR PANEL STRUCTURE
# ============================================================
print_section("REGION-YEAR PANEL STRUCTURE")
region_year_duplicates = region_panel[
    region_panel.duplicated(
        ["region", "year"],
        keep=False,
    )
].sort_values(["region", "year"])
if not region_year_duplicates.empty:
    region_year_duplicates.to_csv(
        QC_DIR / "duplicate_region_year_rows.csv",
        index=False,
    )
    errors.append(
        f"{len(region_year_duplicates)} duplicated region-year rows "
        "exist."
    )
else:
    print("PASS: No duplicated region-year observations.")
# Missing values
regional_core_columns = [
    "region",
    "year",
    "gdp_pc_region",
    "co2_pc_region",
    "population_region",
]
na_counts = (
    region_panel[regional_core_columns]
    .isna()
    .sum()
)
print("\nMissing values:")
print(na_counts)
if na_counts.sum() > 0:
    errors.append(
        "The saved region-year panel contains missing values."
    )
else:
    print("\nPASS: No missing values in the regional panel.")
# Positive values
invalid_gdp = region_panel[
    region_panel["gdp_pc_region"] <= 0
]
invalid_co2 = region_panel[
    region_panel["co2_pc_region"] <= 0
]
invalid_population = region_panel[
    region_panel["population_region"] <= 0
]
if not invalid_gdp.empty:
    errors.append(
        f"{len(invalid_gdp)} region-year observations have "
        "non-positive GDP per capita."
    )
if not invalid_co2.empty:
    errors.append(
        f"{len(invalid_co2)} region-year observations have "
        "non-positive CO2 per capita. These cannot be logged."
    )
if not invalid_population.empty:
    errors.append(
        f"{len(invalid_population)} region-year observations have "
        "non-positive population."
    )
# ============================================================
# 10. TIME COVERAGE
# ============================================================
print_section("TIME COVERAGE")
min_year = int(region_panel["year"].min())
max_year = int(region_panel["year"].max())
print(f"Overall year range: {min_year} - {max_year}")
if min_year != EXPECTED_START_YEAR:
    warnings.append(
        f"Regional panel starts in {min_year}, while the planned "
        f"sample begins in {EXPECTED_START_YEAR}."
    )
year_summary = (
    region_panel
    .groupby("region")
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_years=("year", "nunique"),
        n_observations=("year", "size"),
    )
    .reset_index()
)
year_summary.to_csv(
    QC_DIR / "region_time_coverage.csv",
    index=False,
)
print("\nCoverage by region:")
print(year_summary.to_string(index=False))
# Check whether all regions contain the same set of years
all_years = set(region_panel["year"].unique())
for region, group in region_panel.groupby("region"):
    region_years = set(group["year"].unique())
    missing_years = sorted(all_years - region_years)
    if missing_years:
        warnings.append(
            f"{region} is missing {len(missing_years)} years from the "
            "overall regional-panel time range."
        )
# ============================================================
# 11. RECONSTRUCT REGIONAL AGGREGATES FROM COUNTRY DATA
# ============================================================
print_section("RECONSTRUCTING REGIONAL AGGREGATES")
country_with_region = country_panel.merge(
    region_map,
    on="iso3c",
    how="inner",
    validate="many_to_one",
)
core_variables = [
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_tons",
    "population",
]
# This reproduces the approach documented in the existing project: a country-year contributes only where all three variables needed for aggregation are present.
complete = (
    country_with_region
    .dropna(subset=core_variables)
    .copy()
)
complete["gdp_weighted"] = (
    complete["gdp_per_capita_const2015_usd"]
    * complete["population"]
)
complete["co2_weighted"] = (
    complete["co2_per_capita_tons"]
    * complete["population"]
)
recalculated = (
    complete
    .groupby(
        ["region", "year"],
        as_index=False,
    )
    .agg(
        gdp_weighted_sum=("gdp_weighted", "sum"),
        co2_weighted_sum=("co2_weighted", "sum"),
        population_region_recalc=("population", "sum"),
        n_countries_used=("iso3c", "nunique"),
    )
)
recalculated["gdp_pc_region_recalc"] = (
    recalculated["gdp_weighted_sum"]
    / recalculated["population_region_recalc"]
)
recalculated["co2_pc_region_recalc"] = (
    recalculated["co2_weighted_sum"]
    / recalculated["population_region_recalc"]
)
# ============================================================
# 12. COUNTRY-COMPOSITION / COVERAGE CHECK
# ============================================================
print_section("COUNTRY COMPOSITION AND COVERAGE")
countries_in_map = (
    region_map
    .groupby("region", as_index=False)
    .agg(
        n_countries_in_map=("iso3c", "nunique")
    )
)
# Population available from all mapped countries in a year, even if GDP or CO2 is missing.
population_possible = (
    country_with_region
    .dropna(subset=["population"])
    .groupby(
        ["region", "year"],
        as_index=False,
    )
    .agg(
        population_with_population_data=(
            "population",
            "sum",
        ),
        n_countries_with_population=(
            "iso3c",
            "nunique",
        ),
    )
)
coverage = (
    recalculated[
        [
            "region",
            "year",
            "population_region_recalc",
            "n_countries_used",
        ]
    ]
    .merge(
        countries_in_map,
        on="region",
        how="left",
    )
    .merge(
        population_possible,
        on=["region", "year"],
        how="left",
    )
)
coverage["country_coverage_pct"] = (
    coverage["n_countries_used"]
    / coverage["n_countries_in_map"]
    * 100
)
coverage["population_coverage_pct"] = (
    coverage["population_region_recalc"]
    / coverage["population_with_population_data"]
    * 100
)
coverage.to_csv(
    QC_DIR / "region_year_country_coverage.csv",
    index=False,
)
coverage_summary = (
    coverage
    .groupby("region", as_index=False)
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        min_countries_used=("n_countries_used", "min"),
        max_countries_used=("n_countries_used", "max"),
        countries_in_map=("n_countries_in_map", "max"),
        min_country_coverage_pct=(
            "country_coverage_pct",
            "min",
        ),
        min_population_coverage_pct=(
            "population_coverage_pct",
            "min",
        ),
    )
)
coverage_summary.to_csv(
    QC_DIR / "region_coverage_summary.csv",
    index=False,
)
print(coverage_summary.to_string(index=False))
# Warn if the number of contributing countries changes
for _, row in coverage_summary.iterrows():
    if row["min_countries_used"] != row["max_countries_used"]:
        warnings.append(
            f"{row['region']}: contributing-country count changes "
            f"from {int(row['min_countries_used'])} to "
            f"{int(row['max_countries_used'])} across years."
        )
    if (
        row["min_population_coverage_pct"]
        <
        POPULATION_COVERAGE_WARNING_THRESHOLD * 100
    ):
        warnings.append(
            f"{row['region']}: population coverage falls below "
            f"{POPULATION_COVERAGE_WARNING_THRESHOLD:.0%} "
            "in at least one year."
        )
# ============================================================
# 13. VERIFY SAVED REGIONAL PANEL AGAINST RECALCULATION
# ============================================================
print_section("CHECKING SAVED AGGREGATES")
comparison = region_panel.merge(
    recalculated[
        [
            "region",
            "year",
            "gdp_pc_region_recalc",
            "co2_pc_region_recalc",
            "population_region_recalc",
            "n_countries_used",
        ]
    ],
    on=["region", "year"],
    how="outer",
    indicator=True,
)
missing_keys = comparison[
    comparison["_merge"] != "both"
]
if not missing_keys.empty:
    errors.append(
        "Saved and reconstructed regional panels do not contain "
        "the same region-year observations."
    )
both = comparison[
    comparison["_merge"] == "both"
].copy()
variable_pairs = {
    "gdp_pc_region": "gdp_pc_region_recalc",
    "co2_pc_region": "co2_pc_region_recalc",
    "population_region": "population_region_recalc",
}
for saved_col, recalculated_col in variable_pairs.items():
    both[f"{saved_col}_abs_diff"] = (
        both[saved_col]
        - both[recalculated_col]
    ).abs()
    both[f"{saved_col}_matches"] = np.isclose(
        both[saved_col],
        both[recalculated_col],
        rtol=RTOL,
        atol=ATOL,
        equal_nan=False,
    )
    if not both[f"{saved_col}_matches"].all():
        mismatch_count = (
            ~both[f"{saved_col}_matches"]
        ).sum()
        errors.append(
            f"{mismatch_count} observations fail the "
            f"recalculation check for {saved_col}."
        )
comparison_output = comparison.copy()
comparison_output.to_csv(
    QC_DIR / "aggregation_recheck.csv",
    index=False,
)
# ============================================================
# 14. FINAL QC REPORT
# ============================================================
print_section("FINAL QC RESULT")
summary_lines = []
summary_lines.append(
    "REGIONAL PANEL QUALITY CONTROL REPORT"
)
summary_lines.append(
    "=" * 50
)
summary_lines.append(
    f"Country panel rows: {len(country_panel):,}"
)
summary_lines.append(
    f"Region panel rows: {len(region_panel):,}"
)
summary_lines.append(
    f"Regional sample: {min_year}-{max_year}"
)
summary_lines.append(
    "Regions: "
    + ", ".join(
        sorted(region_panel["region"].unique())
    )
)
summary_lines.append("")
summary_lines.append("ERRORS:")
if errors:
    summary_lines.extend(
        [f"- {message}" for message in errors]
    )
else:
    summary_lines.append("- None")
summary_lines.append("")
summary_lines.append("WARNINGS:")
if warnings:
    summary_lines.extend(
        [f"- {message}" for message in warnings]
    )
else:
    summary_lines.append("- None")
summary_lines.append("")
summary_lines.append(
    "Generated QC files:"
)
summary_lines.append(
    "- region_time_coverage.csv"
)
summary_lines.append(
    "- region_year_country_coverage.csv"
)
summary_lines.append(
    "- region_coverage_summary.csv"
)
summary_lines.append(
    "- aggregation_recheck.csv"
)
summary_text = "\n".join(summary_lines)
(QC_DIR / "qc_summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)
print(summary_text)
print(
    f"\nQC output directory:\n{QC_DIR}"
)
if errors:
    print(
        "\nQC RESULT: FAIL\n"
        "Do not proceed to regression estimation until the "
        "errors above have been resolved."
    )
    raise SystemExit(1)
else:
    print(
        "\nQC RESULT: PASS\n"
        "The regional panel has passed the structural and "
        "aggregation checks."
    )

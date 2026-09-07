from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================
# 1. PROJECT PATHS
# ============================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "country_panel_results"
)

SUMMARY_DIR = (
    OUTPUT_DIR
    / "summaries"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. RESEARCH-DESIGN SETTINGS
# ============================================================

# Keep the country-panel analysis aligned with the regional
# analysis already completed in Script 08.
START_YEAR = 1990
END_YEAR = 2024

EXPECTED_YEARS = list(
    range(
        START_YEAR,
        END_YEAR + 1,
    )
)

N_EXPECTED_YEARS = len(
    EXPECTED_YEARS
)


EXPECTED_REGIONS = [
    "Europe_NorthAmerica",
    "DevelopedAsia_Oceania",
    "China",
    "India",
    "Global_South",
]


# A country needs at least two usable annual observations to
# contribute meaningful within-country information.
MIN_YEARS_PER_COUNTRY = 2


# Minimum number of completely balanced countries required
# before estimating the balanced-panel robustness model.
MIN_BALANCED_COUNTRIES = 10


# Confidence interval level.
ALPHA = 0.05


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """
    Print a clean section heading in Terminal.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


def require_columns(
    df,
    required_columns,
    dataset_name,
):
    """
    Ensure that all variables required by the analysis exist.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


def format_p_value(
    p_value,
):
    """
    Produce readable p-values for Terminal output.

    Very small p-values are displayed as <0.001 rather than
    scientific-notation values that provide little additional
    substantive information.
    """

    if pd.isna(
        p_value
    ):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.4f}"


def fit_clustered_model(
    formula,
    data,
    weights=None,
):
    """
    Estimate OLS or WLS with standard errors clustered by country.

    Main features:
    - clustering at the country level;
    - finite-sample covariance correction;
    - degrees-of-freedom correction;
    - Student-t inference.

    If weights are supplied, weighted least squares is used.
    Otherwise ordinary least squares is used.
    """

    cluster_groups = pd.Categorical(
        data["iso3c"]
    ).codes

    if weights is None:

        estimator = smf.ols(
            formula=formula,
            data=data,
        )

    else:

        estimator = smf.wls(
            formula=formula,
            data=data,
            weights=weights,
        )

    model = estimator.fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                cluster_groups,

            "use_correction":
                True,

            "df_correction":
                True,
        },
        use_t=True,
    )

    return model


def save_model_summary(
    model,
    filename,
):
    """
    Save full Statsmodels results locally.

    These files are useful for checking diagnostics but are
    generated automatically and therefore need not be committed
    to GitHub.
    """

    output_file = (
        SUMMARY_DIR
        / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            model.summary().as_text()
        )


def extract_model_result(
    model,
    data,
    term,
    model_name,
    sample_name,
    country_fe,
    year_fe,
    weighted,
    first_difference,
    aic_bic_group,
    notes="",
):
    """
    Extract the economically important coefficient and model
    diagnostics into a single machine-readable record.
    """

    if term not in model.params.index:

        raise ValueError(
            f"Coefficient '{term}' was not found in model "
            f"'{model_name}'."
        )


    confidence_interval = (
        model.conf_int(
            alpha=ALPHA
        )
        .loc[
            term
        ]
    )


    record = {
        "model":
            model_name,

        "sample":
            sample_name,

        "coefficient_name":
            term,

        "beta":
            float(
                model.params[
                    term
                ]
            ),

        "std_error":
            float(
                model.bse[
                    term
                ]
            ),

        "ci95_low":
            float(
                confidence_interval.iloc[0]
            ),

        "ci95_high":
            float(
                confidence_interval.iloc[1]
            ),

        "p_value":
            float(
                model.pvalues[
                    term
                ]
            ),

        "n_obs":
            int(
                model.nobs
            ),

        "n_countries":
            int(
                data[
                    "iso3c"
                ]
                .nunique()
            ),

        "n_years":
            int(
                data[
                    "year"
                ]
                .nunique()
            ),

        "first_year":
            int(
                data[
                    "year"
                ]
                .min()
            ),

        "last_year":
            int(
                data[
                    "year"
                ]
                .max()
            ),

        # These are overall R-squared measures.
        # In fixed-effects models they include explanatory power
        # coming from the fixed-effect dummy variables.
        "overall_r_squared":
            float(
                model.rsquared
            ),

        "overall_adjusted_r_squared":
            float(
                model.rsquared_adj
            ),

        "aic":
            float(
                model.aic
            ),

        "bic":
            float(
                model.bic
            ),

        "country_fixed_effects":
            bool(
                country_fe
            ),

        "year_fixed_effects":
            bool(
                year_fe
            ),

        "population_weighted":
            bool(
                weighted
            ),

        "first_difference":
            bool(
                first_difference
            ),

        # AIC/BIC should only be mechanically compared between
        # models belonging to the same group and using the same
        # dependent variable/sample structure.
        "aic_bic_comparison_group":
            aic_bic_group,

        "covariance":
            (
                "Country-clustered SE; "
                "finite-sample correction; "
                "t-based inference"
            ),

        "notes":
            notes,
    }


    return record


def print_key_result(
    result,
):
    """
    Print only the economically relevant result rather than
    hundreds of fixed-effect dummy coefficients.
    """

    print(
        f"\n{result['model']}"
    )

    print(
        "-" * len(
            result["model"]
        )
    )

    print(
        "Coefficient:",
        f"{result['beta']:.4f}",
    )

    print(
        "Std. error:",
        f"{result['std_error']:.4f}",
    )

    print(
        "95% CI:",
        (
            f"[{result['ci95_low']:.4f}, "
            f"{result['ci95_high']:.4f}]"
        ),
    )

    print(
        "p-value:",
        format_p_value(
            result[
                "p_value"
            ]
        ),
    )

    print(
        "N:",
        result[
            "n_obs"
        ],
    )

    print(
        "Countries:",
        result[
            "n_countries"
        ],
    )


def missing_year_string(
    observed_years,
):
    """
    Return a comma-separated list of missing analysis years for
    a country.
    """

    observed_set = set(
        int(year)
        for year in observed_years
    )

    missing = [
        year
        for year in EXPECTED_YEARS
        if year not in observed_set
    ]

    return ",".join(
        str(year)
        for year in missing
    )


# ============================================================
# 4. LOAD SOURCE DATA
# ============================================================

print_section(
    "LOADING COUNTRY PANEL AND REGION MAP"
)


country_panel = pd.read_csv(
    COUNTRY_PANEL_FILE
)

region_map = pd.read_csv(
    REGION_MAP_FILE
)


print(
    f"Country panel:\n{COUNTRY_PANEL_FILE}"
)

print(
    f"Rows loaded: {len(country_panel):,}"
)


print(
    f"\nRegion map:\n{REGION_MAP_FILE}"
)

print(
    f"Rows loaded: {len(region_map):,}"
)


# ============================================================
# 5. REQUIRED COLUMN VALIDATION
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
    "Country panel",
)


require_columns(
    region_map,
    [
        "iso3c",
        "region",
    ],
    "Region map",
)


# ============================================================
# 6. STANDARDISE IDENTIFIERS AND TYPES
# ============================================================

country_panel = (
    country_panel
    .copy()
)

region_map = (
    region_map
    .copy()
)


country_panel["iso3c"] = (
    country_panel[
        "iso3c"
    ]
    .astype(str)
    .str.strip()
)


region_map["iso3c"] = (
    region_map[
        "iso3c"
    ]
    .astype(str)
    .str.strip()
)


region_map["region"] = (
    region_map[
        "region"
    ]
    .astype(str)
    .str.strip()
)


country_panel["year"] = (
    pd.to_numeric(
        country_panel[
            "year"
        ],
        errors="raise",
    )
    .astype(int)
)


numeric_columns = [
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_tons",
    "population",
]


for column in numeric_columns:

    country_panel[column] = (
        pd.to_numeric(
            country_panel[
                column
            ],
            errors="coerce",
        )
    )


# ============================================================
# 7. REGION-MAPPING VALIDATION
# ============================================================

print_section(
    "VALIDATING REGION MAPPING"
)


if region_map[
    [
        "iso3c",
        "region",
    ]
].isna().any().any():

    raise ValueError(
        "country_regions.csv contains missing ISO or region values."
    )


duplicate_regions = (
    region_map[
        region_map.duplicated(
            "iso3c",
            keep=False,
        )
    ]
    .sort_values(
        "iso3c"
    )
)


if not duplicate_regions.empty:

    duplicate_regions.to_csv(
        OUTPUT_DIR
        / "duplicate_region_assignments.csv",
        index=False,
    )

    raise ValueError(
        "Some ISO codes have multiple region assignments. "
        "See duplicate_region_assignments.csv."
    )


actual_regions = set(
    region_map[
        "region"
    ]
    .unique()
)


expected_regions = set(
    EXPECTED_REGIONS
)


if actual_regions != expected_regions:

    raise ValueError(
        "\nRegion labels do not match the research design.\n"
        f"Expected: {sorted(expected_regions)}\n"
        f"Found: {sorted(actual_regions)}"
    )


print(
    "PASS: Region mapping contains exactly the expected five "
    "regional-development clusters."
)


# ============================================================
# 8. COUNTRY-YEAR KEY VALIDATION
# ============================================================

print_section(
    "VALIDATING COUNTRY-YEAR PANEL KEYS"
)


duplicate_country_year = (
    country_panel[
        country_panel.duplicated(
            [
                "iso3c",
                "year",
            ],
            keep=False,
        )
    ]
    .sort_values(
        [
            "iso3c",
            "year",
        ]
    )
)


if not duplicate_country_year.empty:

    duplicate_country_year.to_csv(
        OUTPUT_DIR
        / "duplicate_country_year_rows.csv",
        index=False,
    )

    raise ValueError(
        "Duplicate country-year observations found. "
        "See duplicate_country_year_rows.csv."
    )


print(
    "PASS: No duplicate country-year observations."
)


# ============================================================
# 9. RESTRICT TO THE PRE-SPECIFIED ANALYSIS WINDOW
# ============================================================

original_row_count = len(
    country_panel
)


panel_window = (
    country_panel[
        (
            country_panel[
                "year"
            ]
            >= START_YEAR
        )
        &
        (
            country_panel[
                "year"
            ]
            <= END_YEAR
        )
    ]
    .copy()
)


print_section(
    "ANALYSIS WINDOW"
)


print(
    f"Research window: {START_YEAR}-{END_YEAR}"
)

print(
    f"Rows in research window: {len(panel_window):,}"
)


# ============================================================
# 10. IDENTIFY WDI CODES NOT PRESENT IN THE REGION MAP
# ============================================================

mapped_codes = set(
    region_map[
        "iso3c"
    ]
)


unmapped_rows = (
    panel_window[
        ~panel_window[
            "iso3c"
        ]
        .isin(
            mapped_codes
        )
    ]
    .copy()
)


unmapped_summary = (
    unmapped_rows[
        [
            "iso3c",
            "country",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "country",
            "iso3c",
        ]
    )
)


unmapped_summary.to_csv(
    OUTPUT_DIR
    / "unmapped_iso_codes_in_analysis_window.csv",
    index=False,
)


# These rows commonly include World Bank aggregate groups.
# They are deliberately excluded by keeping only ISO codes
# explicitly included in country_regions.csv.


# ============================================================
# 11. MERGE THE EXPLICIT COUNTRY-REGION CLASSIFICATION
# ============================================================

mapped_panel = (
    panel_window
    .merge(
        region_map,
        on="iso3c",
        how="inner",
        validate="many_to_one",
    )
)


print_section(
    "COUNTRY MAPPING"
)


print(
    f"Rows after explicit country-region mapping: "
    f"{len(mapped_panel):,}"
)

print(
    f"Unique mapped ISO codes: "
    f"{mapped_panel['iso3c'].nunique():,}"
)


# ============================================================
# 12. CONSTRUCT THE CORE REGRESSION SAMPLE
# ============================================================

missing_gdp = (
    mapped_panel[
        "gdp_per_capita_const2015_usd"
    ]
    .isna()
)


missing_co2 = (
    mapped_panel[
        "co2_per_capita_tons"
    ]
    .isna()
)


nonpositive_gdp = (
    mapped_panel[
        "gdp_per_capita_const2015_usd"
    ]
    .notna()
    &
    (
        mapped_panel[
            "gdp_per_capita_const2015_usd"
        ]
        <= 0
    )
)


nonpositive_co2 = (
    mapped_panel[
        "co2_per_capita_tons"
    ]
    .notna()
    &
    (
        mapped_panel[
            "co2_per_capita_tons"
        ]
        <= 0
    )
)


valid_core = (
    ~missing_gdp
    &
    ~missing_co2
    &
    ~nonpositive_gdp
    &
    ~nonpositive_co2
)


valid_panel = (
    mapped_panel[
        valid_core
    ]
    .copy()
)


# ------------------------------------------------------------
# Require at least two usable annual observations per country.
# ------------------------------------------------------------

usable_year_counts = (
    valid_panel
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .nunique()
)


eligible_codes = set(
    usable_year_counts[
        usable_year_counts
        >= MIN_YEARS_PER_COUNTRY
    ]
    .index
)


estimation_sample = (
    valid_panel[
        valid_panel[
            "iso3c"
        ]
        .isin(
            eligible_codes
        )
    ]
    .copy()
)


estimation_sample = (
    estimation_sample
    .sort_values(
        [
            "iso3c",
            "year",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 13. CREATE LOG VARIABLES
# ============================================================

estimation_sample[
    "log_gdp"
] = np.log(
    estimation_sample[
        "gdp_per_capita_const2015_usd"
    ]
)


estimation_sample[
    "log_co2"
] = np.log(
    estimation_sample[
        "co2_per_capita_tons"
    ]
)


if not np.isfinite(
    estimation_sample[
        [
            "log_gdp",
            "log_co2",
        ]
    ]
    .to_numpy()
).all():

    raise ValueError(
        "Non-finite log values remain in the estimation sample."
    )


# ============================================================
# 14. SAMPLE-EXCLUSION AUDIT
# ============================================================

print_section(
    "SAMPLE EXCLUSION AUDIT"
)


rows_valid_before_min_year_filter = len(
    valid_panel
)


rows_final = len(
    estimation_sample
)


countries_before_min_year_filter = (
    valid_panel[
        "iso3c"
    ]
    .nunique()
)


countries_final = (
    estimation_sample[
        "iso3c"
    ]
    .nunique()
)


exclusion_audit = pd.DataFrame(
    [
        {
            "stage":
                "Rows in original cleaned country panel",

            "count":
                original_row_count,

            "notes":
                "Before restricting years or mapped countries",
        },

        {
            "stage":
                f"Rows within {START_YEAR}-{END_YEAR}",

            "count":
                len(
                    panel_window
                ),

            "notes":
                "Pre-specified research window",
        },

        {
            "stage":
                "Rows after explicit region mapping",

            "count":
                len(
                    mapped_panel
                ),

            "notes":
                (
                    "Retains economies explicitly present in "
                    "country_regions.csv. In the current cleaned "
                    "WDI extract, all 217 economy codes are mapped "
                    "and no unmatched aggregate codes are present."
                ),
        },

        {
            "stage":
                "Rows with missing GDP",

            "count":
                int(
                    missing_gdp.sum()
                ),

            "notes":
                "Diagnostic count; categories may overlap",
        },

        {
            "stage":
                "Rows with missing CO2",

            "count":
                int(
                    missing_co2.sum()
                ),

            "notes":
                "Diagnostic count; categories may overlap",
        },

        {
            "stage":
                "Rows with non-positive GDP",

            "count":
                int(
                    nonpositive_gdp.sum()
                ),

            "notes":
                "Cannot be log transformed",
        },

        {
            "stage":
                "Rows with non-positive CO2",

            "count":
                int(
                    nonpositive_co2.sum()
                ),

            "notes":
                "Cannot be log transformed",
        },

        {
            "stage":
                "Valid GDP-CO2 country-year rows",

            "count":
                rows_valid_before_min_year_filter,

            "notes":
                "Before minimum-years-per-country requirement",
        },

        {
            "stage":
                "Countries before minimum-years filter",

            "count":
                countries_before_min_year_filter,

            "notes":
                "",
        },

        {
            "stage":
                "Final regression rows",

            "count":
                rows_final,

            "notes":
                (
                    f"Countries require at least "
                    f"{MIN_YEARS_PER_COUNTRY} usable observations"
                ),
        },

        {
            "stage":
                "Final regression countries",

            "count":
                countries_final,

            "notes":
                "",
        },
    ]
)


exclusion_audit.to_csv(
    OUTPUT_DIR
    / "country_panel_exclusion_audit.csv",
    index=False,
)


print(
    exclusion_audit.to_string(
        index=False
    )
)
# ============================================================
# ECONOMY-UNIVERSE AUDIT
# ============================================================

mapped_panel[
    "valid_core"
] = valid_core


economy_universe = (
    mapped_panel
    .groupby(
        [
            "iso3c",
            "country",
            "region",
        ],
        as_index=False,
    )
    .agg(
        n_years_in_source=(
            "year",
            "nunique",
        ),

        n_valid_regression_observations=(
            "valid_core",
            "sum",
        ),
    )
)


economy_universe[
    "enters_main_regression"
] = (
    economy_universe[
        "iso3c"
    ]
    .isin(
        eligible_codes
    )
)


def exclusion_reason(row):

    if row[
        "enters_main_regression"
    ]:
        return "Included"

    if (
        row[
            "n_valid_regression_observations"
        ]
        == 0
    ):
        return (
            "No usable positive GDP-CO2 "
            "economy-year observations"
        )

    if (
        row[
            "n_valid_regression_observations"
        ]
        <
        MIN_YEARS_PER_COUNTRY
    ):
        return (
            f"Fewer than {MIN_YEARS_PER_COUNTRY} "
            "usable observations"
        )

    return "Other"


economy_universe[
    "regression_sample_status"
] = (
    economy_universe
    .apply(
        exclusion_reason,
        axis=1,
    )
)


economy_universe.to_csv(
    OUTPUT_DIR
    / "country_panel_economy_universe.csv",
    index=False,
)

# ============================================================
# 15. COUNTRY COVERAGE TABLE
# ============================================================

print_section(
    "COUNTRY COVERAGE"
)


country_coverage = (
    estimation_sample
    .groupby(
        [
            "iso3c",
            "country",
            "region",
        ],
        as_index=False,
    )
    .agg(
        first_year=(
            "year",
            "min",
        ),

        last_year=(
            "year",
            "max",
        ),

        n_years=(
            "year",
            "nunique",
        ),

        within_log_gdp_sd=(
            "log_gdp",
            "std",
        ),

        within_log_co2_sd=(
            "log_co2",
            "std",
        ),
    )
)


years_by_country = (
    estimation_sample
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .apply(
        lambda series:
            sorted(
                set(
                    int(year)
                    for year in series
                )
            )
    )
)


country_coverage[
    "missing_years"
] = (
    country_coverage[
        "iso3c"
    ]
    .map(
        years_by_country.apply(
            missing_year_string
        )
    )
)


country_coverage[
    "coverage_pct"
] = (
    country_coverage[
        "n_years"
    ]
    /
    N_EXPECTED_YEARS
    *
    100
)


country_coverage[
    "is_balanced_1990_2024"
] = (
    (
        country_coverage[
            "first_year"
        ]
        == START_YEAR
    )
    &
    (
        country_coverage[
            "last_year"
        ]
        == END_YEAR
    )
    &
    (
        country_coverage[
            "n_years"
        ]
        == N_EXPECTED_YEARS
    )
    &
    (
        country_coverage[
            "missing_years"
        ]
        == ""
    )
)


country_coverage.to_csv(
    OUTPUT_DIR
    / "country_panel_country_coverage.csv",
    index=False,
)


balanced_codes = set(
    country_coverage.loc[
        country_coverage[
            "is_balanced_1990_2024"
        ],
        "iso3c",
    ]
)


n_balanced_countries = len(
    balanced_codes
)


print(
    f"Countries in estimation sample: "
    f"{countries_final}"
)

print(
    f"Completely balanced countries "
    f"({START_YEAR}-{END_YEAR}): "
    f"{n_balanced_countries}"
)


# ============================================================
# 16. REGION COVERAGE TABLE
# ============================================================

region_coverage = (
    estimation_sample
    .groupby(
        "region",
        as_index=False,
    )
    .agg(
        n_countries=(
            "iso3c",
            "nunique",
        ),

        n_observations=(
            "year",
            "size",
        ),

        first_year=(
            "year",
            "min",
        ),

        last_year=(
            "year",
            "max",
        ),
    )
)


mean_years_by_region = (
    country_coverage
    .groupby(
        "region"
    )[
        "n_years"
    ]
    .mean()
)


median_years_by_region = (
    country_coverage
    .groupby(
        "region"
    )[
        "n_years"
    ]
    .median()
)


region_coverage[
    "mean_years_per_country"
] = (
    region_coverage[
        "region"
    ]
    .map(
        mean_years_by_region
    )
)


region_coverage[
    "median_years_per_country"
] = (
    region_coverage[
        "region"
    ]
    .map(
        median_years_by_region
    )
)


region_coverage.to_csv(
    OUTPUT_DIR
    / "country_panel_region_coverage.csv",
    index=False,
)


print(
    "\nCountry-panel coverage by region:"
)

print(
    region_coverage
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. YEAR COVERAGE TABLE
# ============================================================

year_coverage = (
    estimation_sample
    .groupby(
        "year",
        as_index=False,
    )
    .agg(
        n_countries=(
            "iso3c",
            "nunique",
        ),

        n_observations=(
            "iso3c",
            "size",
        ),
    )
)


year_coverage.to_csv(
    OUTPUT_DIR
    / "country_panel_year_coverage.csv",
    index=False,
)


# ============================================================
# 18. MAIN SAMPLE SUMMARY
# ============================================================

years_per_country = (
    country_coverage[
        "n_years"
    ]
)


zero_within_gdp_variation = int(
    (
        country_coverage[
            "within_log_gdp_sd"
        ]
        .fillna(0)
        == 0
    )
    .sum()
)


sample_summary = pd.DataFrame(
    [
        {
            "metric":
                "Analysis start year",

            "value":
                START_YEAR,
        },

        {
            "metric":
                "Analysis end year",

            "value":
                END_YEAR,
        },

        {
            "metric":
                "Potential years per country",

            "value":
                N_EXPECTED_YEARS,
        },

        {
            "metric":
                "Final country-year observations",

            "value":
                len(
                    estimation_sample
                ),
        },

        {
            "metric":
                "Countries",

            "value":
                countries_final,
        },

        {
            "metric":
                "Regions",

            "value":
                estimation_sample[
                    "region"
                ]
                .nunique(),
        },

        {
            "metric":
                "Minimum usable years per country",

            "value":
                int(
                    years_per_country.min()
                ),
        },

        {
            "metric":
                "Median usable years per country",

            "value":
                float(
                    years_per_country.median()
                ),
        },

        {
            "metric":
                "Mean usable years per country",

            "value":
                float(
                    years_per_country.mean()
                ),
        },

        {
            "metric":
                "Maximum usable years per country",

            "value":
                int(
                    years_per_country.max()
                ),
        },

        {
            "metric":
                "Completely balanced countries",

            "value":
                n_balanced_countries,
        },

        {
            "metric":
                "Countries with zero within-log-GDP variation",

            "value":
                zero_within_gdp_variation,
        },
    ]
)


sample_summary.to_csv(
    OUTPUT_DIR
    / "country_panel_sample_summary.csv",
    index=False,
)


print(
    "\nSample summary:"
)

print(
    sample_summary.to_string(
        index=False
    )
)


# ============================================================
# 19. WARN ABOUT SMALL NUMBER OF COUNTRY CLUSTERS
# ============================================================

if countries_final < 30:

    print(
        "\nWARNING: Fewer than 30 country clusters are available. "
        "Cluster-robust inference should be interpreted cautiously."
    )


# ============================================================
# 20. MODEL 1 — POOLED OLS BENCHMARK
# ============================================================

print_section(
    "MODEL 1: POOLED OLS BENCHMARK"
)


pooled_model = fit_clustered_model(
    formula=(
        "log_co2 ~ log_gdp"
    ),
    data=estimation_sample,
)


pooled_result = extract_model_result(
    model=pooled_model,
    data=estimation_sample,
    term="log_gdp",
    model_name="Pooled OLS",
    sample_name="Main country panel",
    country_fe=False,
    year_fe=False,
    weighted=False,
    first_difference=False,
    aic_bic_group="levels_main_sample",
    notes=(
        "Benchmark only. Mixes between-country and "
        "within-country variation."
    ),
)


print_key_result(
    pooled_result
)


save_model_summary(
    pooled_model,
    "01_pooled_ols_summary.txt",
)


# ============================================================
# 21. MODEL 2 — COUNTRY FIXED EFFECTS
# ============================================================

print_section(
    "MODEL 2: COUNTRY FIXED EFFECTS"
)


country_fe_model = fit_clustered_model(
    formula=(
        "log_co2 ~ "
        "log_gdp + "
        "C(iso3c)"
    ),
    data=estimation_sample,
)


country_fe_result = extract_model_result(
    model=country_fe_model,
    data=estimation_sample,
    term="log_gdp",
    model_name="Country fixed effects",
    sample_name="Main country panel",
    country_fe=True,
    year_fe=False,
    weighted=False,
    first_difference=False,
    aic_bic_group="levels_main_sample",
    notes=(
        "Controls for time-invariant country characteristics."
    ),
)


print_key_result(
    country_fe_result
)


save_model_summary(
    country_fe_model,
    "02_country_fe_summary.txt",
)


# ============================================================
# 22. MODEL 3 — COUNTRY + YEAR FIXED EFFECTS
# ============================================================

print_section(
    "MODEL 3: COUNTRY + YEAR FIXED EFFECTS — MAIN SPECIFICATION"
)


two_way_fe_model = fit_clustered_model(
    formula=(
        "log_co2 ~ "
        "log_gdp + "
        "C(iso3c) + "
        "C(year)"
    ),
    data=estimation_sample,
)


two_way_fe_result = extract_model_result(
    model=two_way_fe_model,
    data=estimation_sample,
    term="log_gdp",
    model_name="Country + year fixed effects",
    sample_name="Main country panel",
    country_fe=True,
    year_fe=True,
    weighted=False,
    first_difference=False,
    aic_bic_group="levels_main_sample",
    notes=(
        "MAIN SPECIFICATION. Controls for persistent country "
        "heterogeneity and common year-specific shocks."
    ),
)


print_key_result(
    two_way_fe_result
)


save_model_summary(
    two_way_fe_model,
    "03_country_year_fe_summary.txt",
)


# ============================================================
# 23. MODEL 4 — BALANCED-PANEL TWO-WAY FE ROBUSTNESS
# ============================================================

print_section(
    "MODEL 4: BALANCED-PANEL TWO-WAY FE ROBUSTNESS"
)


balanced_result = None


balanced_sample = (
    estimation_sample[
        estimation_sample[
            "iso3c"
        ]
        .isin(
            balanced_codes
        )
    ]
    .copy()
)


if (
    n_balanced_countries
    >= MIN_BALANCED_COUNTRIES
):

    balanced_model = fit_clustered_model(
        formula=(
            "log_co2 ~ "
            "log_gdp + "
            "C(iso3c) + "
            "C(year)"
        ),
        data=balanced_sample,
    )


    balanced_result = extract_model_result(
        model=balanced_model,
        data=balanced_sample,
        term="log_gdp",
        model_name=(
            "Balanced-panel country + year fixed effects"
        ),
        sample_name=(
            f"Countries observed in every year "
            f"{START_YEAR}-{END_YEAR}"
        ),
        country_fe=True,
        year_fe=True,
        weighted=False,
        first_difference=False,
        aic_bic_group="balanced_levels_sample",
        notes=(
            "Robustness check for changing country coverage."
        ),
    )


    print_key_result(
        balanced_result
    )


    save_model_summary(
        balanced_model,
        "04_balanced_country_year_fe_summary.txt",
    )


else:

    print(
        f"Balanced robustness model skipped because only "
        f"{n_balanced_countries} completely balanced countries "
        f"were available. Minimum required: "
        f"{MIN_BALANCED_COUNTRIES}."
    )


# ============================================================
# 24. MODEL 5 — POPULATION-WEIGHTED TWO-WAY FE ROBUSTNESS
# ============================================================

print_section(
    "MODEL 5: POPULATION-WEIGHTED TWO-WAY FE ROBUSTNESS"
)


weighted_sample = (
    estimation_sample[
        estimation_sample[
            "population"
        ]
        .notna()
        &
        (
            estimation_sample[
                "population"
            ]
            > 0
        )
    ]
    .copy()
)


if weighted_sample.empty:

    raise ValueError(
        "No valid population observations are available for "
        "the population-weighted robustness model."
    )


# Weight scaling does not change WLS coefficient estimates.
# Normalisation simply keeps numerical values manageable.
weighted_sample[
    "population_weight"
] = (
    weighted_sample[
        "population"
    ]
    /
    weighted_sample[
        "population"
    ]
    .mean()
)


weighted_model = fit_clustered_model(
    formula=(
        "log_co2 ~ "
        "log_gdp + "
        "C(iso3c) + "
        "C(year)"
    ),
    data=weighted_sample,
    weights=weighted_sample[
        "population_weight"
    ],
)


weighted_result = extract_model_result(
    model=weighted_model,
    data=weighted_sample,
    term="log_gdp",
    model_name=(
        "Population-weighted country + year fixed effects"
    ),
    sample_name=(
        "Main panel observations with positive population"
    ),
    country_fe=True,
    year_fe=True,
    weighted=True,
    first_difference=False,
    aic_bic_group="population_weighted_levels",
    notes=(
        "Robustness specification. Population weighting changes "
        "the estimand from approximately the average country "
        "towards the average person."
    ),
)


print_key_result(
    weighted_result
)


save_model_summary(
    weighted_model,
    "05_population_weighted_country_year_fe_summary.txt",
)


# ============================================================
# 25. BUILD FIRST-DIFFERENCE COUNTRY PANEL
# ============================================================

print_section(
    "CONSTRUCTING FIRST-DIFFERENCE PANEL"
)


difference_source = (
    estimation_sample
    .sort_values(
        [
            "iso3c",
            "year",
        ]
    )
    .copy()
)


difference_source[
    "previous_year"
] = (
    difference_source
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .shift(1)
)


difference_source[
    "d_log_gdp"
] = (
    difference_source
    .groupby(
        "iso3c"
    )[
        "log_gdp"
    ]
    .diff()
)


difference_source[
    "d_log_co2"
] = (
    difference_source
    .groupby(
        "iso3c"
    )[
        "log_co2"
    ]
    .diff()
)


difference_source[
    "year_gap"
] = (
    difference_source[
        "year"
    ]
    -
    difference_source[
        "previous_year"
    ]
)


# ------------------------------------------------------------
# CRITICAL:
#
# Only retain differences where the two observations are
# consecutive years.
#
# Without this check, a country missing 2004 could generate:
#
#     2005 minus 2003
#
# and incorrectly treat a two-year change as an annual change.
# ------------------------------------------------------------

difference_sample = (
    difference_source[
        (
            difference_source[
                "year_gap"
            ]
            == 1
        )
        &
        difference_source[
            "d_log_gdp"
        ]
        .notna()
        &
        difference_source[
            "d_log_co2"
        ]
        .notna()
    ]
    .copy()
)


difference_sample = (
    difference_sample
    .sort_values(
        [
            "iso3c",
            "year",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"First-difference observations: "
    f"{len(difference_sample):,}"
)

print(
    f"Countries contributing first differences: "
    f"{difference_sample['iso3c'].nunique():,}"
)


# ============================================================
# 26. FIRST-DIFFERENCE COVERAGE
# ============================================================

difference_coverage = (
    difference_sample
    .groupby(
        [
            "iso3c",
            "country",
            "region",
        ],
        as_index=False,
    )
    .agg(
        first_difference_year=(
            "year",
            "min",
        ),

        last_difference_year=(
            "year",
            "max",
        ),

        n_difference_observations=(
            "year",
            "size",
        ),
    )
)


difference_coverage.to_csv(
    OUTPUT_DIR
    / "country_panel_first_difference_coverage.csv",
    index=False,
)


difference_region_coverage = (
    difference_sample
    .groupby(
        "region",
        as_index=False,
    )
    .agg(
        n_countries=(
            "iso3c",
            "nunique",
        ),

        n_difference_observations=(
            "year",
            "size",
        ),
    )
)


difference_region_coverage.to_csv(
    OUTPUT_DIR
    / "country_panel_first_difference_region_coverage.csv",
    index=False,
)


# ============================================================
# 27. MODEL 6 — FIRST DIFFERENCES + YEAR FIXED EFFECTS
# ============================================================

print_section(
    "MODEL 6: FIRST DIFFERENCES + YEAR FIXED EFFECTS"
)


first_difference_model = fit_clustered_model(
    formula=(
        "d_log_co2 ~ "
        "d_log_gdp + "
        "C(year)"
    ),
    data=difference_sample,
)


first_difference_result = extract_model_result(
    model=first_difference_model,
    data=difference_sample,
    term="d_log_gdp",
    model_name=(
        "First differences + year fixed effects"
    ),
    sample_name=(
        "Consecutive country-year pairs"
    ),
    country_fe=False,
    year_fe=True,
    weighted=False,
    first_difference=True,
    aic_bic_group="first_difference_sample",
    notes=(
        "Robustness check using annual changes. "
        "Only consecutive-year differences are retained."
    ),
)


print_key_result(
    first_difference_result
)


save_model_summary(
    first_difference_model,
    "06_first_difference_year_fe_summary.txt",
)


# ============================================================
# 28. EXPORT MAIN FIXED-EFFECTS RESULTS
# ============================================================

levels_results = [
    pooled_result,
    country_fe_result,
    two_way_fe_result,
]


if balanced_result is not None:

    levels_results.append(
        balanced_result
    )


levels_results.append(
    weighted_result
)


levels_results_df = pd.DataFrame(
    levels_results
)


levels_results_df.to_csv(
    OUTPUT_DIR
    / "country_panel_fe_results.csv",
    index=False,
)


# ============================================================
# 29. EXPORT FIRST-DIFFERENCE RESULT
# ============================================================

first_difference_results_df = pd.DataFrame(
    [
        first_difference_result
    ]
)


first_difference_results_df.to_csv(
    OUTPUT_DIR
    / "country_panel_first_difference_results.csv",
    index=False,
)


# ============================================================
# 30. COMBINED MODEL-COMPARISON TABLE
# ============================================================

combined_results = (
    levels_results
    +
    [
        first_difference_result
    ]
)


combined_results_df = pd.DataFrame(
    combined_results
)


# ------------------------------------------------------------
# Compare each coefficient with the pooled benchmark where
# meaningful.
#
# This is descriptive only. Different specifications estimate
# different sources of variation and should not be interpreted
# as interchangeable estimands.
# ------------------------------------------------------------

pooled_beta = (
    pooled_result[
        "beta"
    ]
)


combined_results_df[
    "beta_minus_pooled"
] = (
    combined_results_df[
        "beta"
    ]
    -
    pooled_beta
)


combined_results_df.to_csv(
    OUTPUT_DIR
    / "country_panel_model_comparison.csv",
    index=False,
)


# ============================================================
# 31. AIC/BIC COMPARISON FOR THE SAME MAIN LEVELS SAMPLE
# ============================================================

# Only these three models share:
#
# - the same dependent variable;
# - the same observations;
# - an unweighted likelihood.
#
# Therefore their AIC/BIC values can at least be compared
# mechanically. These criteria should not be compared directly
# with the first-difference, balanced-sample or WLS models.

main_sample_information_criteria = (
    combined_results_df[
        combined_results_df[
            "aic_bic_comparison_group"
        ]
        == "levels_main_sample"
    ][
        [
            "model",
            "aic",
            "bic",
            "n_obs",
        ]
    ]
    .copy()
)


main_sample_information_criteria[
    "delta_aic_vs_min"
] = (
    main_sample_information_criteria[
        "aic"
    ]
    -
    main_sample_information_criteria[
        "aic"
    ]
    .min()
)


main_sample_information_criteria[
    "delta_bic_vs_min"
] = (
    main_sample_information_criteria[
        "bic"
    ]
    -
    main_sample_information_criteria[
        "bic"
    ]
    .min()
)


main_sample_information_criteria.to_csv(
    OUTPUT_DIR
    / "country_panel_main_sample_information_criteria.csv",
    index=False,
)


# ============================================================
# 32. CLEAN TERMINAL RESULTS TABLE
# ============================================================

print_section(
    "COUNTRY-PANEL RESULTS SUMMARY"
)


display_columns = [
    "model",
    "beta",
    "std_error",
    "ci95_low",
    "ci95_high",
    "p_value",
    "n_obs",
    "n_countries",
    "country_fixed_effects",
    "year_fixed_effects",
    "population_weighted",
    "first_difference",
]


terminal_results = (
    combined_results_df[
        display_columns
    ]
    .copy()
)


print(
    terminal_results
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 33. MAIN-SPECIFICATION EMPHASIS
# ============================================================

print_section(
    "MAIN SPECIFICATION"
)


print(
    "Preferred country-panel specification:"
)

print(
    "log(CO2_it) = country FE + year FE "
    "+ beta * log(GDP_it) + error_it"
)


print(
    "\nEstimated within-country GDP-emissions elasticity:"
)

print(
    f"{two_way_fe_result['beta']:.4f}"
)


print(
    "\n95% confidence interval:"
)

print(
    (
        f"[{two_way_fe_result['ci95_low']:.4f}, "
        f"{two_way_fe_result['ci95_high']:.4f}]"
    )
)


print(
    "\np-value:"
)

print(
    format_p_value(
        two_way_fe_result[
            "p_value"
        ]
    )
)


print(
    "\nThis is the coefficient that should receive primary "
    "attention before regional heterogeneity is introduced "
    "in Script 10."
)


# ============================================================
# 34. OUTPUT MANIFEST
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


print(
    f"Outputs saved to:\n{OUTPUT_DIR}"
)


print(
    "\nKey machine-readable files:"
)


key_files = [
    "country_panel_sample_summary.csv",
    "country_panel_exclusion_audit.csv",
    "country_panel_country_coverage.csv",
    "country_panel_region_coverage.csv",
    "country_panel_year_coverage.csv",
    "unmapped_iso_codes_in_analysis_window.csv",
    "country_panel_fe_results.csv",
    "country_panel_first_difference_results.csv",
    "country_panel_first_difference_coverage.csv",
    "country_panel_first_difference_region_coverage.csv",
    "country_panel_model_comparison.csv",
    "country_panel_main_sample_information_criteria.csv",
]


for filename in key_files:

    print(
        f"- {filename}"
    )


print(
    "\nFull Statsmodels summaries are stored locally under:"
)

print(
    SUMMARY_DIR
)


print(
    "\nSCRIPT 09 COMPLETE."
)

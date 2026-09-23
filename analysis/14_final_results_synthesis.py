from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

REGIONAL_REGRESSION_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "regression_results"
)

GOVERNANCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "governance_results"
)

FINAL_ROBUSTNESS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "final_robustness"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "final_synthesis"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FILES
# ============================================================

GLOBAL_ROBUSTNESS_FILE = (
    FINAL_ROBUSTNESS_DIR
    / "global_twfe_robustness.csv"
)

REGIONAL_ROBUSTNESS_FILE = (
    FINAL_ROBUSTNESS_DIR
    / "regional_heterogeneity_robustness.csv"
)

SAMPLE_AUDIT_FILE = (
    FINAL_ROBUSTNESS_DIR
    / "sample_robustness_audit.csv"
)

FINAL_CORE_RESULTS_FILE = (
    FINAL_ROBUSTNESS_DIR
    / "final_core_results_summary.csv"
)

GOVERNANCE_INTERACTION_FILE = (
    GOVERNANCE_DIR
    / "governance_interaction_summary.csv"
)

GOVERNANCE_SPECIFICATION_FILE = (
    GOVERNANCE_DIR
    / "governance_specification_robustness.csv"
)

GOVERNANCE_BETWEEN_WITHIN_FILE = (
    GOVERNANCE_DIR
    / "governance_between_within_decomposition.csv"
)

GOVERNANCE_COMMON_BALANCED_FILE = (
    GOVERNANCE_DIR
    / "governance_common_balanced_annual_interactions.csv"
)

REGIONAL_EKC_FILE = (
    REGIONAL_REGRESSION_DIR
    / "regional_ekc_results.csv"
)


# ============================================================
# 3. RESEARCH SETTINGS
# ============================================================

START_YEAR = 1990
END_YEAR = 2024

ALPHA = 0.05

TOLERANCE = 1e-8


EXPECTED_REGIONS = [
    "Europe_NorthAmerica",
    "DevelopedAsia_Oceania",
    "China",
    "India",
    "Global_South",
]


REGION_DISPLAY_NAMES = {
    "Europe_NorthAmerica":
        "Europe & North America",

    "DevelopedAsia_Oceania":
        "Developed Asia & Oceania",

    "China":
        "China",

    "India":
        "India",

    "Global_South":
        "Global South",
}


GOVERNANCE_ORDER = [
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
]


GOVERNANCE_DISPLAY_NAMES = {
    "government_effectiveness":
        "Government Effectiveness",

    "regulatory_quality":
        "Regulatory Quality",

    "rule_of_law":
        "Rule of Law",
}


DECOUPLING_CATEGORIES = [
    "Absolute decoupling",
    "Relative decoupling",
    "Coupled expansion",
]


# ============================================================
# 4. GENERAL HELPERS
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


def require_file(path):
    """
    Fail immediately if a required frozen-analysis source file
    is missing.
    """

    if not path.exists():

        raise FileNotFoundError(
            "Required input file not found:\n"
            f"{path}"
        )


def require_columns(
    df,
    columns,
    dataset_name,
):
    """
    Verify that all required columns exist.
    """

    missing = [
        column
        for column
        in columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


def sha256_file(path):
    """
    Compute a SHA-256 hash for source provenance.
    """

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )


    return digest.hexdigest()


def relative_path(path):
    """
    Convert an absolute project path into a portable relative
    repository path.
    """

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def parse_bool(value):
    """
    Safely convert a CSV-loaded Boolean-like value to Python
    bool.

    Handles:
    - True / False
    - numpy.bool_
    - "True" / "False"
    - 1 / 0

    Missing values return np.nan.

    Important:
        bool("False") evaluates to True in Python, so direct
        bool(...) conversion must not be used for CSV strings.
    """

    if pd.isna(
        value
    ):

        return np.nan


    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):

        return bool(
            value
        )


    if isinstance(
        value,
        (
            int,
            np.integer,
            float,
            np.floating,
        ),
    ):

        if value == 1:

            return True


        if value == 0:

            return False


    value_string = (
        str(
            value
        )
        .strip()
        .lower()
    )


    if value_string in [
        "true",
        "t",
        "yes",
        "y",
        "1",
    ]:

        return True


    if value_string in [
        "false",
        "f",
        "no",
        "n",
        "0",
    ]:

        return False


    raise ValueError(
        f"Could not interpret Boolean value: {value}"
    )


def safe_float(value):
    """
    Convert a numeric-like value to float while preserving
    missing values.
    """

    if pd.isna(
        value
    ):

        return np.nan


    return float(
        value
    )


def safe_int(value):
    """
    Convert a numeric-like value to integer while preserving
    missing values as np.nan.
    """

    if pd.isna(
        value
    ):

        return np.nan


    return int(
        value
    )


def json_native(value):
    """
    Convert NumPy/Pandas objects into JSON-safe Python objects.
    """

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )


    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(
            value
        ):

            return None

        return float(
            value
        )


    if isinstance(
        value,
        np.bool_,
    ):

        return bool(
            value
        )


    if pd.isna(
        value
    ):

        return None


    return value


def dataframe_records_for_json(df):
    """
    Convert a DataFrame into JSON-safe record dictionaries.
    """

    records = []


    for record in df.to_dict(
        orient="records"
    ):

        records.append(
            {
                key:
                    json_native(
                        value
                    )
                for (
                    key,
                    value,
                ) in record.items()
            }
        )


    return records


def classify_decoupling(
    d_log_gdp,
    d_log_co2,
):
    """
    Classify decoupling conditional on positive GDP growth.

    Absolute decoupling:
        GDP per capita grows while CO2 per capita falls.

    Relative decoupling:
        GDP and CO2 both grow, but CO2 grows more slowly.

    Coupled expansion:
        CO2 grows at least as fast as GDP.

    Log changes preserve the ordering of proportional changes.
    """

    if d_log_gdp <= 0:

        return None


    if d_log_co2 < 0:

        return (
            "Absolute decoupling"
        )


    if d_log_co2 < d_log_gdp:

        return (
            "Relative decoupling"
        )


    return (
        "Coupled expansion"
    )


# ============================================================
# 5. VERIFY REQUIRED FROZEN OUTPUTS
# ============================================================

print_section(
    "VERIFYING FROZEN ANALYSIS OUTPUTS"
)


required_files = [
    COUNTRY_PANEL_FILE,
    REGION_MAP_FILE,
    GLOBAL_ROBUSTNESS_FILE,
    REGIONAL_ROBUSTNESS_FILE,
    SAMPLE_AUDIT_FILE,
    FINAL_CORE_RESULTS_FILE,
    GOVERNANCE_INTERACTION_FILE,
    GOVERNANCE_SPECIFICATION_FILE,
    GOVERNANCE_BETWEEN_WITHIN_FILE,
    GOVERNANCE_COMMON_BALANCED_FILE,
    REGIONAL_EKC_FILE,
]


for path in required_files:

    require_file(
        path
    )


print(
    f"PASS: all {len(required_files)} required source files exist."
)


# ============================================================
# 6. LOAD SCRIPT 13 FINAL ROBUSTNESS OUTPUTS
# ============================================================

print_section(
    "LOADING SCRIPT 13 RESULTS"
)


global_robustness = pd.read_csv(
    GLOBAL_ROBUSTNESS_FILE
)


regional_robustness = pd.read_csv(
    REGIONAL_ROBUSTNESS_FILE
)


sample_audit = pd.read_csv(
    SAMPLE_AUDIT_FILE
)


final_core = pd.read_csv(
    FINAL_CORE_RESULTS_FILE
)


require_columns(
    global_robustness,
    [
        "specification",
        "sample_type",
        "coefficient",
        "std_error",
        "ci95_low",
        "ci95_high",
        "p_value",
        "n_obs",
        "n_economies",
        "n_years",
        "first_year",
        "last_year",
        "covariance",
    ],
    "Script 13 global robustness",
)


require_columns(
    regional_robustness,
    [
        "specification",
        "region",
        "region_display",
        "beta_log_gdp",
        "std_error",
        "ci95_low",
        "ci95_high",
        "p_value",
        "n_region_obs",
        "n_region_economies",
        "inferential_caution",
    ],
    "Script 13 regional robustness",
)


require_columns(
    sample_audit,
    [
        "specification",
        "sample_type",
        "n_obs",
        "n_economies",
        "n_years",
        "first_year",
        "last_year",
        "minimum_obs_per_economy",
        "median_obs_per_economy",
        "maximum_obs_per_economy",
        "is_strictly_balanced",
    ],
    "Script 13 sample robustness audit",
)


require_columns(
    final_core,
    [
        "result_type",
        "scope",
        "baseline_beta",
        "baseline_ci95_low",
        "baseline_ci95_high",
        "baseline_p_value",
        "robustness_min_beta",
        "robustness_max_beta",
    ],
    "Script 13 final core summary",
)


# ============================================================
# 7. BUILD AUTHORITATIVE CORE ELASTICITY TABLE
# ============================================================

print_section(
    "BUILDING CORE ELASTICITY TABLE"
)


baseline_global = (
    global_robustness[
        global_robustness[
            "specification"
        ]
        ==
        "Baseline 1990-2024"
    ]
)


if len(
    baseline_global
) != 1:

    raise ValueError(
        "Could not uniquely identify the baseline global result."
    )


baseline_global = (
    baseline_global.iloc[0]
)


baseline_regions = (
    regional_robustness[
        regional_robustness[
            "specification"
        ]
        ==
        "Baseline 1990-2024"
    ]
    .copy()
)


if set(
    baseline_regions[
        "region"
    ]
) != set(
    EXPECTED_REGIONS
):

    raise ValueError(
        "Baseline regional results do not contain exactly the "
        "five expected development clusters."
    )


core_elasticity_rows = [
    {
        "scope_type":
            "Global pooled economy-level",

        "scope":
            "All economies",

        "estimate":
            float(
                baseline_global[
                    "coefficient"
                ]
            ),

        "std_error":
            float(
                baseline_global[
                    "std_error"
                ]
            ),

        "ci95_low":
            float(
                baseline_global[
                    "ci95_low"
                ]
            ),

        "ci95_high":
            float(
                baseline_global[
                    "ci95_high"
                ]
            ),

        "p_value":
            float(
                baseline_global[
                    "p_value"
                ]
            ),

        "n_obs":
            int(
                baseline_global[
                    "n_obs"
                ]
            ),

        "n_economies":
            int(
                baseline_global[
                    "n_economies"
                ]
            ),

        "first_year":
            int(
                baseline_global[
                    "first_year"
                ]
            ),

        "last_year":
            int(
                baseline_global[
                    "last_year"
                ]
            ),

        "interpretation":
            (
                "Within-economy GDP-per-capita elasticity of "
                "CO2 emissions per capita, conditional on "
                "economy and year fixed effects."
            ),

        "inferential_caution":
            "",

        "source_script":
            "13",

        "source_file":
            relative_path(
                GLOBAL_ROBUSTNESS_FILE
            ),
    }
]


for region in EXPECTED_REGIONS:

    row = (
        baseline_regions[
            baseline_regions[
                "region"
            ]
            ==
            region
        ]
    )


    if len(
        row
    ) != 1:

        raise ValueError(
            f"Could not uniquely identify baseline result for "
            f"{region}."
        )


    row = (
        row.iloc[0]
    )


    core_elasticity_rows.append(
        {
            "scope_type":
                "Development-cluster elasticity",

            "scope":
                REGION_DISPLAY_NAMES[
                    region
                ],

            "estimate":
                float(
                    row[
                        "beta_log_gdp"
                    ]
                ),

            "std_error":
                float(
                    row[
                        "std_error"
                    ]
                ),

            "ci95_low":
                float(
                    row[
                        "ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    row[
                        "ci95_high"
                    ]
                ),

            "p_value":
                float(
                    row[
                        "p_value"
                    ]
                ),

            "n_obs":
                int(
                    row[
                        "n_region_obs"
                    ]
                ),

            "n_economies":
                int(
                    row[
                        "n_region_economies"
                    ]
                ),

            "first_year":
                START_YEAR,

            "last_year":
                END_YEAR,

            "interpretation":
                (
                    "Development-cluster-specific GDP-per-capita "
                    "elasticity of CO2 emissions per capita in "
                    "the economy and year fixed-effects model."
                ),

            "inferential_caution":
                (
                    str(
                        row[
                            "inferential_caution"
                        ]
                    )
                    if pd.notna(
                        row[
                            "inferential_caution"
                        ]
                    )
                    else ""
                ),

            "source_script":
                "13",

            "source_file":
                relative_path(
                    REGIONAL_ROBUSTNESS_FILE
                ),
        }
    )


table_core_elasticities = pd.DataFrame(
    core_elasticity_rows
)


table_core_elasticities.to_csv(
    OUTPUT_DIR
    / "table_01_core_elasticities.csv",
    index=False,
)


# ============================================================
# 8. CROSS-CHECK SCRIPT 13 CORE SUMMARY
# ============================================================

print_section(
    "CROSS-CHECKING CORE RESULTS"
)


global_summary_row = (
    final_core[
        final_core[
            "result_type"
        ]
        ==
        "Global GDP-CO2 elasticity"
    ]
)


if len(
    global_summary_row
) != 1:

    raise ValueError(
        "Final core summary does not contain exactly one global "
        "elasticity row."
    )


global_summary_row = (
    global_summary_row.iloc[0]
)


if not np.isclose(
    float(
        global_summary_row[
            "baseline_beta"
        ]
    ),
    float(
        baseline_global[
            "coefficient"
        ]
    ),
    atol=TOLERANCE,
    rtol=0,
):

    raise ValueError(
        "Global baseline coefficient differs between Script 13 "
        "source tables."
    )


for region in EXPECTED_REGIONS:

    display = (
        REGION_DISPLAY_NAMES[
            region
        ]
    )


    core_row = (
        final_core[
            (
                final_core[
                    "result_type"
                ]
                ==
                "Regional GDP-CO2 elasticity"
            )
            &
            (
                final_core[
                    "scope"
                ]
                ==
                display
            )
        ]
    )


    regression_row = (
        baseline_regions[
            baseline_regions[
                "region"
            ]
            ==
            region
        ]
    )


    if (
        len(
            core_row
        )
        != 1
        or
        len(
            regression_row
        )
        != 1
    ):

        raise ValueError(
            f"Regional cross-check failed for {display}."
        )


    if not np.isclose(
        float(
            core_row.iloc[0][
                "baseline_beta"
            ]
        ),
        float(
            regression_row.iloc[0][
                "beta_log_gdp"
            ]
        ),
        atol=TOLERANCE,
        rtol=0,
    ):

        raise ValueError(
            f"Regional coefficient mismatch for {display}."
        )


print(
    "PASS: Script 13 global and regional headline results agree "
    "across source tables."
)


# ============================================================
# 9. BUILD GLOBAL ROBUSTNESS PAPER TABLE
# ============================================================

print_section(
    "BUILDING GLOBAL ROBUSTNESS TABLE"
)


global_table_columns = [
    "specification",
    "sample_type",
    "coefficient",
    "std_error",
    "ci95_low",
    "ci95_high",
    "p_value",
    "n_obs",
    "n_economies",
    "n_years",
    "first_year",
    "last_year",
    "covariance",
    "excluded_region",
    "notes",
]


table_global_robustness = (
    global_robustness[
        global_table_columns
    ]
    .copy()
)


table_global_robustness.to_csv(
    OUTPUT_DIR
    / "table_02_global_robustness.csv",
    index=False,
)


# ============================================================
# 10. LOAD DATA FOR DECOUPLING RECONSTRUCTION
# ============================================================

print_section(
    "RECONSTRUCTING DECOUPLING SUMMARY"
)


panel = pd.read_csv(
    COUNTRY_PANEL_FILE
)


region_map = pd.read_csv(
    REGION_MAP_FILE
)


require_columns(
    panel,
    [
        "country",
        "iso3c",
        "year",
        "gdp_per_capita_const2015_usd",
        "co2_per_capita_tons",
    ],
    "Clean economy panel",
)


require_columns(
    region_map,
    [
        "iso3c",
        "region",
    ],
    "Economy-region mapping",
)


panel = (
    panel
    .copy()
)


region_map = (
    region_map
    .copy()
)


panel[
    "iso3c"
] = (
    panel[
        "iso3c"
    ]
    .astype(str)
    .str.strip()
)


region_map[
    "iso3c"
] = (
    region_map[
        "iso3c"
    ]
    .astype(str)
    .str.strip()
)


region_map[
    "region"
] = (
    region_map[
        "region"
    ]
    .astype(str)
    .str.strip()
)


panel[
    "year"
] = pd.to_numeric(
    panel[
        "year"
    ],
    errors="raise",
).astype(int)


for column in [
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_tons",
]:

    panel[
        column
    ] = pd.to_numeric(
        panel[
            column
        ],
        errors="coerce",
    )


if panel.duplicated(
    [
        "iso3c",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate economy-year observations found while "
        "reconstructing decoupling."
    )


if region_map.duplicated(
    "iso3c"
).any():

    raise ValueError(
        "Duplicate economy-region mappings found."
    )


decoupling_panel = (
    panel[
        (
            panel[
                "year"
            ]
            >= START_YEAR
        )
        &
        (
            panel[
                "year"
            ]
            <= END_YEAR
        )
        &
        panel[
            "gdp_per_capita_const2015_usd"
        ]
        .notna()
        &
        panel[
            "co2_per_capita_tons"
        ]
        .notna()
        &
        (
            panel[
                "gdp_per_capita_const2015_usd"
            ]
            > 0
        )
        &
        (
            panel[
                "co2_per_capita_tons"
            ]
            > 0
        )
    ]
    .merge(
        region_map,
        on="iso3c",
        how="inner",
        validate="many_to_one",
    )
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


if set(
    decoupling_panel[
        "region"
    ]
    .unique()
) != set(
    EXPECTED_REGIONS
):

    raise ValueError(
        "Decoupling reconstruction does not contain all five "
        "development clusters."
    )


decoupling_panel[
    "log_gdp"
] = np.log(
    decoupling_panel[
        "gdp_per_capita_const2015_usd"
    ]
)


decoupling_panel[
    "log_co2"
] = np.log(
    decoupling_panel[
        "co2_per_capita_tons"
    ]
)


# ============================================================
# 11. ANNUAL DECOUPLING
# ============================================================

annual = (
    decoupling_panel
    .copy()
)


annual[
    "previous_year"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .shift(1)
)


annual[
    "previous_log_gdp"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "log_gdp"
    ]
    .shift(1)
)


annual[
    "previous_log_co2"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "log_co2"
    ]
    .shift(1)
)


annual[
    "year_gap"
] = (
    annual[
        "year"
    ]
    -
    annual[
        "previous_year"
    ]
)


annual[
    "d_log_gdp"
] = (
    annual[
        "log_gdp"
    ]
    -
    annual[
        "previous_log_gdp"
    ]
)


annual[
    "d_log_co2"
] = (
    annual[
        "log_co2"
    ]
    -
    annual[
        "previous_log_co2"
    ]
)


annual_valid = (
    annual[
        (
            annual[
                "year_gap"
            ]
            == 1
        )
        &
        annual[
            "d_log_gdp"
        ]
        .notna()
        &
        annual[
            "d_log_co2"
        ]
        .notna()
    ]
    .copy()
)


annual_valid[
    "decoupling_category"
] = [
    classify_decoupling(
        d_log_gdp,
        d_log_co2,
    )
    for (
        d_log_gdp,
        d_log_co2,
    ) in zip(
        annual_valid[
            "d_log_gdp"
        ],
        annual_valid[
            "d_log_co2"
        ],
    )
]


annual_positive = (
    annual_valid[
        annual_valid[
            "d_log_gdp"
        ]
        > 0
    ]
    .copy()
)


annual_rows = []


for region in EXPECTED_REGIONS:

    region_data = (
        annual_positive[
            annual_positive[
                "region"
            ]
            ==
            region
        ]
    )


    denominator = (
        len(
            region_data
        )
    )


    if denominator == 0:

        raise ValueError(
            f"No positive-GDP annual observations for {region}."
        )


    for category in DECOUPLING_CATEGORIES:

        count = int(
            (
                region_data[
                    "decoupling_category"
                ]
                ==
                category
            )
            .sum()
        )


        annual_rows.append(
            {
                "analysis":
                    "Annual",

                "region":
                    region,

                "region_display":
                    REGION_DISPLAY_NAMES[
                        region
                    ],

                "category":
                    category,

                "count":
                    count,

                "denominator":
                    denominator,

                "share_percent":
                    (
                        count
                        /
                        denominator
                        *
                        100
                    ),

                "economies_with_both_endpoints":
                    np.nan,

                "denominator_definition":
                    (
                        "Economy-year observations with "
                        "consecutive annual observations and "
                        "positive GDP-per-capita growth"
                    ),

                "period":
                    "1990-2024",
            }
        )


# ============================================================
# 12. LONG-PERIOD ENDPOINT DECOUPLING
# ============================================================

endpoint_start = (
    decoupling_panel[
        decoupling_panel[
            "year"
        ]
        ==
        START_YEAR
    ][
        [
            "iso3c",
            "country",
            "region",
            "log_gdp",
            "log_co2",
        ]
    ]
    .rename(
        columns={
            "log_gdp":
                "log_gdp_start",

            "log_co2":
                "log_co2_start",
        }
    )
)


endpoint_end = (
    decoupling_panel[
        decoupling_panel[
            "year"
        ]
        ==
        END_YEAR
    ][
        [
            "iso3c",
            "log_gdp",
            "log_co2",
        ]
    ]
    .rename(
        columns={
            "log_gdp":
                "log_gdp_end",

            "log_co2":
                "log_co2_end",
        }
    )
)


endpoints = (
    endpoint_start
    .merge(
        endpoint_end,
        on="iso3c",
        how="inner",
        validate="one_to_one",
    )
)


endpoints[
    "d_log_gdp"
] = (
    endpoints[
        "log_gdp_end"
    ]
    -
    endpoints[
        "log_gdp_start"
    ]
)


endpoints[
    "d_log_co2"
] = (
    endpoints[
        "log_co2_end"
    ]
    -
    endpoints[
        "log_co2_start"
    ]
)


endpoints[
    "decoupling_category"
] = [
    classify_decoupling(
        d_log_gdp,
        d_log_co2,
    )
    for (
        d_log_gdp,
        d_log_co2,
    ) in zip(
        endpoints[
            "d_log_gdp"
        ],
        endpoints[
            "d_log_co2"
        ],
    )
]


endpoint_expanding = (
    endpoints[
        endpoints[
            "d_log_gdp"
        ]
        > 0
    ]
    .copy()
)


long_period_rows = []


for region in EXPECTED_REGIONS:

    both_endpoints_count = int(
        (
            endpoints[
                "region"
            ]
            ==
            region
        )
        .sum()
    )


    region_data = (
        endpoint_expanding[
            endpoint_expanding[
                "region"
            ]
            ==
            region
        ]
    )


    denominator = (
        len(
            region_data
        )
    )


    if denominator == 0:

        raise ValueError(
            f"No expanding endpoint economies for {region}."
        )


    for category in DECOUPLING_CATEGORIES:

        count = int(
            (
                region_data[
                    "decoupling_category"
                ]
                ==
                category
            )
            .sum()
        )


        long_period_rows.append(
            {
                "analysis":
                    "Long-period endpoints",

                "region":
                    region,

                "region_display":
                    REGION_DISPLAY_NAMES[
                        region
                    ],

                "category":
                    category,

                "count":
                    count,

                "denominator":
                    denominator,

                "share_percent":
                    (
                        count
                        /
                        denominator
                        *
                        100
                    ),

                "economies_with_both_endpoints":
                    both_endpoints_count,

                "denominator_definition":
                    (
                        "Economies with valid observations in "
                        "both 1990 and 2024 and positive "
                        "long-period GDP-per-capita growth"
                    ),

                "period":
                    "1990-2024",
            }
        )


table_decoupling = pd.DataFrame(
    annual_rows
    +
    long_period_rows
)


table_decoupling.to_csv(
    OUTPUT_DIR
    / "table_03_decoupling_summary.csv",
    index=False,
)


# ============================================================
# 13. VALIDATE DECOUPLING SHARES
# ============================================================

for (
    analysis_name,
    region,
), group in table_decoupling.groupby(
    [
        "analysis",
        "region",
    ]
):

    total_share = (
        group[
            "share_percent"
        ]
        .sum()
    )


    total_count = (
        group[
            "count"
        ]
        .sum()
    )


    denominator = int(
        group[
            "denominator"
        ]
        .iloc[0]
    )


    if not np.isclose(
        total_share,
        100.0,
        atol=1e-8,
        rtol=0,
    ):

        raise ValueError(
            "Decoupling shares do not sum to 100 for "
            f"{analysis_name}, {region}: {total_share}"
        )


    if total_count != denominator:

        raise ValueError(
            "Decoupling category counts do not equal the "
            f"denominator for {analysis_name}, {region}."
        )


print(
    "PASS: annual and long-period decoupling shares sum to 100% "
    "for every development cluster."
)


# ============================================================
# 14. LOAD GOVERNANCE RESULTS
# ============================================================

print_section(
    "BUILDING GOVERNANCE SYNTHESIS TABLE"
)


governance_interactions = pd.read_csv(
    GOVERNANCE_INTERACTION_FILE
)


governance_specifications = pd.read_csv(
    GOVERNANCE_SPECIFICATION_FILE
)


governance_between_within = pd.read_csv(
    GOVERNANCE_BETWEEN_WITHIN_FILE
)


governance_common_balanced = pd.read_csv(
    GOVERNANCE_COMMON_BALANCED_FILE
)


require_columns(
    governance_interactions,
    [
        "indicator",
        "indicator_display",
        "sample",
        "delta_elasticity_change_per_10pt_governance",
        "delta_std_error",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_p_value",
        "delta_p_value_holm",
    ],
    "Governance interaction summary",
)


require_columns(
    governance_specifications,
    [
        "indicator",
        "indicator_display",
        "sample",
        "specification",
        "delta_elasticity_change_per_10pt_governance",
        "delta_std_error",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_p_value",
        "delta_p_value_holm",
    ],
    "Governance specification robustness",
)


require_columns(
    governance_between_within,
    [
        "indicator",
        "indicator_display",
        "sample",
        "component",
        "coefficient",
        "std_error",
        "ci95_low",
        "ci95_high",
        "p_value",
        "p_value_holm",
    ],
    "Governance between/within decomposition",
)


require_columns(
    governance_common_balanced,
    [
        "indicator",
        "indicator_display",
        "sample",
        "delta_elasticity_change_per_10pt_governance",
        "delta_std_error",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_p_value",
        "delta_p_value_holm",
    ],
    "Common balanced governance robustness",
)


# ============================================================
# 15. BUILD GOVERNANCE SYNTHESIS TABLE
# ============================================================

governance_rows = []


# ------------------------------------------------------------
# A. Common complete-case baseline interaction
# ------------------------------------------------------------

baseline_governance = (
    governance_interactions[
        governance_interactions[
            "sample"
        ]
        ==
        "Common complete-case governance sample"
    ]
    .copy()
)


if set(
    baseline_governance[
        "indicator"
    ]
) != set(
    GOVERNANCE_ORDER
):

    raise ValueError(
        "Common governance sample does not contain the three "
        "expected governance indicators."
    )


for _, row in baseline_governance.iterrows():

    governance_rows.append(
        {
            "indicator":
                row[
                    "indicator"
                ],

            "indicator_display":
                row[
                    "indicator_display"
                ],

            "result_family":
                "Baseline linear interaction",

            "specification":
                "Baseline linear governance interaction",

            "component":
                "GDP x governance",

            "estimate":
                float(
                    row[
                        "delta_elasticity_change_per_10pt_governance"
                    ]
                ),

            "std_error":
                float(
                    row[
                        "delta_std_error"
                    ]
                ),

            "ci95_low":
                float(
                    row[
                        "delta_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    row[
                        "delta_ci95_high"
                    ]
                ),

            "p_value":
                float(
                    row[
                        "delta_p_value"
                    ]
                ),

            "p_value_holm":
                float(
                    row[
                        "delta_p_value_holm"
                    ]
                ),

            "sample":
                row[
                    "sample"
                ],

            "interpretation_unit":
                (
                    "Change in GDP-CO2 elasticity per 10-point "
                    "higher governance score"
                ),

            "source_script":
                "12",
        }
    )


# ------------------------------------------------------------
# B. Common balanced 2002-2024 robustness
# ------------------------------------------------------------

if set(
    governance_common_balanced[
        "indicator"
    ]
) != set(
    GOVERNANCE_ORDER
):

    raise ValueError(
        "Common balanced governance file does not contain the "
        "three expected governance indicators."
    )


for _, row in governance_common_balanced.iterrows():

    governance_rows.append(
        {
            "indicator":
                row[
                    "indicator"
                ],

            "indicator_display":
                row[
                    "indicator_display"
                ],

            "result_family":
                "Common balanced robustness",

            "specification":
                "Common balanced 2002-2024 interaction",

            "component":
                "GDP x governance",

            "estimate":
                float(
                    row[
                        "delta_elasticity_change_per_10pt_governance"
                    ]
                ),

            "std_error":
                float(
                    row[
                        "delta_std_error"
                    ]
                ),

            "ci95_low":
                float(
                    row[
                        "delta_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    row[
                        "delta_ci95_high"
                    ]
                ),

            "p_value":
                float(
                    row[
                        "delta_p_value"
                    ]
                ),

            "p_value_holm":
                float(
                    row[
                        "delta_p_value_holm"
                    ]
                ),

            "sample":
                row[
                    "sample"
                ],

            "interpretation_unit":
                (
                    "Change in GDP-CO2 elasticity per 10-point "
                    "higher governance score"
                ),

            "source_script":
                "12",
        }
    )


# ------------------------------------------------------------
# C. Specification sensitivity
# ------------------------------------------------------------

for _, row in governance_specifications.iterrows():

    governance_rows.append(
        {
            "indicator":
                row[
                    "indicator"
                ],

            "indicator_display":
                row[
                    "indicator_display"
                ],

            "result_family":
                "Specification sensitivity",

            "specification":
                row[
                    "specification"
                ],

            "component":
                "GDP x governance",

            "estimate":
                float(
                    row[
                        "delta_elasticity_change_per_10pt_governance"
                    ]
                ),

            "std_error":
                float(
                    row[
                        "delta_std_error"
                    ]
                ),

            "ci95_low":
                float(
                    row[
                        "delta_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    row[
                        "delta_ci95_high"
                    ]
                ),

            "p_value":
                float(
                    row[
                        "delta_p_value"
                    ]
                ),

            "p_value_holm":
                float(
                    row[
                        "delta_p_value_holm"
                    ]
                ),

            "sample":
                row[
                    "sample"
                ],

            "interpretation_unit":
                (
                    "Change in GDP-CO2 elasticity per 10-point "
                    "higher governance score"
                ),

            "source_script":
                "12",
        }
    )


# ------------------------------------------------------------
# D. Between / within decomposition
# ------------------------------------------------------------

for _, row in governance_between_within.iterrows():

    governance_rows.append(
        {
            "indicator":
                row[
                    "indicator"
                ],

            "indicator_display":
                row[
                    "indicator_display"
                ],

            "result_family":
                "Between/within decomposition",

            "specification":
                "Between/within governance decomposition",

            "component":
                row[
                    "component"
                ],

            "estimate":
                float(
                    row[
                        "coefficient"
                    ]
                ),

            "std_error":
                float(
                    row[
                        "std_error"
                    ]
                ),

            "ci95_low":
                float(
                    row[
                        "ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    row[
                        "ci95_high"
                    ]
                ),

            "p_value":
                float(
                    row[
                        "p_value"
                    ]
                ),

            "p_value_holm":
                float(
                    row[
                        "p_value_holm"
                    ]
                ),

            "sample":
                row[
                    "sample"
                ],

            "interpretation_unit":
                (
                    "Difference in GDP-CO2 elasticity per "
                    "10-point governance difference"
                ),

            "source_script":
                "12",
        }
    )


table_governance = pd.DataFrame(
    governance_rows
)


table_governance.to_csv(
    OUTPUT_DIR
    / "table_04_governance_summary.csv",
    index=False,
)


# ============================================================
# 16. GOVERNANCE STRUCTURAL VALIDATION
# ============================================================

baseline_spec_rows = (
    governance_specifications[
        governance_specifications[
            "specification"
        ]
        ==
        "Baseline linear governance interaction"
    ]
)


region_adjusted_rows = (
    governance_specifications[
        governance_specifications[
            "specification"
        ]
        ==
        "Region-adjusted governance interaction"
    ]
)


quadratic_rows = (
    governance_specifications[
        governance_specifications[
            "specification"
        ]
        ==
        "Quadratic-GDP governance interaction"
    ]
)


region_quadratic_rows = (
    governance_specifications[
        governance_specifications[
            "specification"
        ]
        ==
        "Region + quadratic governance interaction"
    ]
)


for (
    label,
    frame,
) in [
    (
        "baseline",
        baseline_spec_rows,
    ),
    (
        "region-adjusted",
        region_adjusted_rows,
    ),
    (
        "quadratic",
        quadratic_rows,
    ),
    (
        "region + quadratic",
        region_quadratic_rows,
    ),
]:

    if set(
        frame[
            "indicator"
        ]
    ) != set(
        GOVERNANCE_ORDER
    ):

        raise ValueError(
            f"Governance {label} specification is incomplete."
        )


if not (
    baseline_spec_rows[
        "delta_elasticity_change_per_10pt_governance"
    ]
    < 0
).all():

    raise ValueError(
        "Expected baseline governance interactions are not all "
        "negative."
    )


if not (
    quadratic_rows[
        "delta_elasticity_change_per_10pt_governance"
    ]
    > 0
).all():

    raise ValueError(
        "Expected quadratic governance interactions are not all "
        "positive."
    )


if not (
    region_quadratic_rows[
        "delta_elasticity_change_per_10pt_governance"
    ]
    > 0
).all():

    raise ValueError(
        "Expected region + quadratic governance interactions "
        "are not all positive."
    )


print(
    "PASS: governance specification-sensitivity pattern matches "
    "the frozen Script 12 results."
)


# ============================================================
# 17. LOAD FINAL REGIONAL EKC RESULTS
# ============================================================

print_section(
    "BUILDING REGIONAL EKC SUMMARY"
)


regional_ekc = pd.read_csv(
    REGIONAL_EKC_FILE
)


required_ekc_columns = [
    "region",
    "beta1_log_gdp",
    "beta1_std_error",
    "beta1_ci95_low",
    "beta1_ci95_high",
    "beta1_p_value",
    "beta2_log_gdp_sq",
    "beta2_std_error",
    "beta2_ci95_low",
    "beta2_ci95_high",
    "beta2_p_value",
    "r_squared",
    "adjusted_r_squared",
    "aic",
    "bic",
    "n_obs",
    "observed_min_gdp",
    "observed_max_gdp",
    "elasticity_at_min_gdp",
    "elasticity_at_max_gdp",
    "turning_point_log_gdp",
    "turning_point_gdp",
    "turning_point_inside_sample",
    "turning_point_position_pct_of_log_income_range",
    "n_obs_below_turning_point",
    "n_obs_above_turning_point",
    "share_obs_below_turning_point",
    "share_obs_above_turning_point",
    "ekc_sign_pattern",
    "within_sample_ekc_candidate",
    "covariance",
]


require_columns(
    regional_ekc,
    required_ekc_columns,
    "Script 08 regional EKC results",
)


regional_ekc[
    "region"
] = (
    regional_ekc[
        "region"
    ]
    .astype(str)
    .str.strip()
)


if regional_ekc.duplicated(
    "region"
).any():

    raise ValueError(
        "regional_ekc_results.csv contains duplicate region rows."
    )


observed_ekc_regions = set(
    regional_ekc[
        "region"
    ]
)


if observed_ekc_regions != set(
    EXPECTED_REGIONS
):

    missing_regions = (
        set(
            EXPECTED_REGIONS
        )
        -
        observed_ekc_regions
    )


    extra_regions = (
        observed_ekc_regions
        -
        set(
            EXPECTED_REGIONS
        )
    )


    raise ValueError(
        "Regional EKC file does not contain exactly the five "
        "expected development clusters. "
        f"Missing={sorted(missing_regions)}, "
        f"extra={sorted(extra_regions)}"
    )


ekc_rows = []


for region in EXPECTED_REGIONS:

    source_row = (
        regional_ekc[
            regional_ekc[
                "region"
            ]
            ==
            region
        ]
    )


    if len(
        source_row
    ) != 1:

        raise ValueError(
            f"Could not uniquely identify EKC result for "
            f"{region}."
        )


    source_row = (
        source_row.iloc[0]
    )


    turning_point_inside = parse_bool(
        source_row[
            "turning_point_inside_sample"
        ]
    )


    sign_pattern = parse_bool(
        source_row[
            "ekc_sign_pattern"
        ]
    )


    within_sample_candidate = parse_bool(
        source_row[
            "within_sample_ekc_candidate"
        ]
    )


    beta1 = safe_float(
        source_row[
            "beta1_log_gdp"
        ]
    )


    beta2 = safe_float(
        source_row[
            "beta2_log_gdp_sq"
        ]
    )


    beta2_p = safe_float(
        source_row[
            "beta2_p_value"
        ]
    )


    turning_point_gdp = safe_float(
        source_row[
            "turning_point_gdp"
        ]
    )


    observed_min_gdp = safe_float(
        source_row[
            "observed_min_gdp"
        ]
    )


    observed_max_gdp = safe_float(
        source_row[
            "observed_max_gdp"
        ]
    )


    share_below = safe_float(
        source_row[
            "share_obs_below_turning_point"
        ]
    )


    share_above = safe_float(
        source_row[
            "share_obs_above_turning_point"
        ]
    )


    n_below = safe_int(
        source_row[
            "n_obs_below_turning_point"
        ]
    )


    n_above = safe_int(
        source_row[
            "n_obs_above_turning_point"
        ]
    )


    # --------------------------------------------------------
    # Construct cautious EKC interpretation
    # --------------------------------------------------------

    if within_sample_candidate is True:

        if (
            pd.notna(
                share_below
            )
            and
            pd.notna(
                share_above
            )
        ):

            interpretation_note = (
                "The quadratic model satisfies the Script 08 "
                "within-sample EKC candidate screening rule. "
                f"The estimated turning point is approximately "
                f"{turning_point_gdp:,.0f} GDP per capita, "
                f"within the observed range of "
                f"{observed_min_gdp:,.0f} to "
                f"{observed_max_gdp:,.0f}. Approximately "
                f"{share_below * 100:.1f}% of observations lie "
                f"below the turning point and "
                f"{share_above * 100:.1f}% lie above it. "
                "This remains exploratory evidence from a "
                "regional time-series quadratic specification "
                "and is not causal or structural proof of an "
                "Environmental Kuznets Curve."
            )

        else:

            interpretation_note = (
                "The quadratic model satisfies the Script 08 "
                "within-sample EKC candidate screening rule and "
                "places its estimated turning point inside the "
                "observed GDP-per-capita range. This remains "
                "exploratory evidence rather than causal or "
                "structural proof of an Environmental Kuznets "
                "Curve."
            )


    elif (
        sign_pattern is True
        and
        turning_point_inside is not True
    ):

        interpretation_note = (
            "The fitted quadratic has the inverted-U sign "
            "pattern, but its estimated turning point lies "
            "outside the observed GDP-per-capita range. The "
            "sample therefore does not provide within-sample "
            "turning-point evidence for an Environmental "
            "Kuznets Curve."
        )


    elif sign_pattern is True:

        interpretation_note = (
            "The fitted quadratic has the inverted-U sign "
            "pattern, but it does not satisfy the full Script 08 "
            "within-sample candidate screening rule. It should "
            "not be presented as evidence of a confirmed "
            "Environmental Kuznets Curve."
        )


    else:

        interpretation_note = (
            "The quadratic model does not satisfy the Script 08 "
            "inverted-U screening conditions and therefore does "
            "not provide within-sample EKC candidate evidence."
        )


    ekc_rows.append(
        {
            "region":
                region,

            "region_display":
                REGION_DISPLAY_NAMES[
                    region
                ],

            "beta1_log_gdp":
                beta1,

            "beta1_std_error":
                safe_float(
                    source_row[
                        "beta1_std_error"
                    ]
                ),

            "beta1_ci95_low":
                safe_float(
                    source_row[
                        "beta1_ci95_low"
                    ]
                ),

            "beta1_ci95_high":
                safe_float(
                    source_row[
                        "beta1_ci95_high"
                    ]
                ),

            "beta1_p_value":
                safe_float(
                    source_row[
                        "beta1_p_value"
                    ]
                ),

            "beta2_log_gdp_sq":
                beta2,

            "beta2_std_error":
                safe_float(
                    source_row[
                        "beta2_std_error"
                    ]
                ),

            "beta2_ci95_low":
                safe_float(
                    source_row[
                        "beta2_ci95_low"
                    ]
                ),

            "beta2_ci95_high":
                safe_float(
                    source_row[
                        "beta2_ci95_high"
                    ]
                ),

            "beta2_p_value":
                beta2_p,

            "r_squared":
                safe_float(
                    source_row[
                        "r_squared"
                    ]
                ),

            "adjusted_r_squared":
                safe_float(
                    source_row[
                        "adjusted_r_squared"
                    ]
                ),

            "aic":
                safe_float(
                    source_row[
                        "aic"
                    ]
                ),

            "bic":
                safe_float(
                    source_row[
                        "bic"
                    ]
                ),

            "n_obs":
                safe_int(
                    source_row[
                        "n_obs"
                    ]
                ),

            "observed_min_gdp":
                observed_min_gdp,

            "observed_max_gdp":
                observed_max_gdp,

            "elasticity_at_min_gdp":
                safe_float(
                    source_row[
                        "elasticity_at_min_gdp"
                    ]
                ),

            "elasticity_at_max_gdp":
                safe_float(
                    source_row[
                        "elasticity_at_max_gdp"
                    ]
                ),

            "turning_point_log_gdp":
                safe_float(
                    source_row[
                        "turning_point_log_gdp"
                    ]
                ),

            "turning_point_gdp":
                turning_point_gdp,

            "turning_point_inside_sample":
                turning_point_inside,

            "turning_point_position_pct_of_log_income_range":
                safe_float(
                    source_row[
                        "turning_point_position_pct_of_log_income_range"
                    ]
                ),

            "n_obs_below_turning_point":
                n_below,

            "n_obs_above_turning_point":
                n_above,

            "share_obs_below_turning_point":
                share_below,

            "share_obs_above_turning_point":
                share_above,

            "ekc_sign_pattern":
                sign_pattern,

            "within_sample_ekc_candidate":
                within_sample_candidate,

            "covariance":
                source_row[
                    "covariance"
                ],

            "interpretation_note":
                interpretation_note,

            "source_script":
                "08",

            "source_file":
                relative_path(
                    REGIONAL_EKC_FILE
                ),
        }
    )


table_ekc = pd.DataFrame(
    ekc_rows
)


# ============================================================
# 18. VALIDATE REGIONAL EKC SYNTHESIS
# ============================================================

if len(
    table_ekc
) != len(
    EXPECTED_REGIONS
):

    raise ValueError(
        "Regional EKC synthesis does not contain exactly five "
        "development clusters."
    )


if table_ekc[
    "region"
].duplicated().any():

    raise ValueError(
        "Duplicate regions found in regional EKC synthesis."
    )


candidate_rows = (
    table_ekc[
        table_ekc[
            "within_sample_ekc_candidate"
        ]
        ==
        True
    ]
)


if not candidate_rows.empty:

    if not (
        candidate_rows[
            "turning_point_inside_sample"
        ]
        ==
        True
    ).all():

        raise ValueError(
            "At least one within-sample EKC candidate has a "
            "turning point outside the observed sample."
        )


    if not (
        (
            candidate_rows[
                "turning_point_gdp"
            ]
            >=
            candidate_rows[
                "observed_min_gdp"
            ]
        )
        &
        (
            candidate_rows[
                "turning_point_gdp"
            ]
            <=
            candidate_rows[
                "observed_max_gdp"
            ]
        )
    ).all():

        raise ValueError(
            "At least one EKC candidate turning point falls "
            "outside the reported GDP range."
        )


    for _, row in candidate_rows.iterrows():

        n_below = (
            row[
                "n_obs_below_turning_point"
            ]
        )


        n_above = (
            row[
                "n_obs_above_turning_point"
            ]
        )


        n_obs = (
            row[
                "n_obs"
            ]
        )


        if (
            pd.notna(
                n_below
            )
            and
            pd.notna(
                n_above
            )
            and
            pd.notna(
                n_obs
            )
        ):

            if (
                int(
                    n_below
                )
                +
                int(
                    n_above
                )
            ) != int(
                n_obs
            ):

                raise ValueError(
                    "EKC turning-point observation counts do "
                    "not sum to N for "
                    f"{row['region_display']}."
                )


        share_below = (
            row[
                "share_obs_below_turning_point"
            ]
        )


        share_above = (
            row[
                "share_obs_above_turning_point"
            ]
        )


        if (
            pd.notna(
                share_below
            )
            and
            pd.notna(
                share_above
            )
        ):

            if not np.isclose(
                float(
                    share_below
                )
                +
                float(
                    share_above
                ),
                1.0,
                atol=1e-8,
                rtol=0,
            ):

                raise ValueError(
                    "EKC turning-point shares do not sum to 1 "
                    f"for {row['region_display']}."
                )


table_ekc.to_csv(
    OUTPUT_DIR
    / "table_05_regional_ekc_summary.csv",
    index=False,
)


print(
    table_ekc[
        [
            "region_display",
            "beta1_log_gdp",
            "beta2_log_gdp_sq",
            "beta2_p_value",
            "turning_point_gdp",
            "turning_point_inside_sample",
            "ekc_sign_pattern",
            "within_sample_ekc_candidate",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nPASS: regional EKC results loaded using the final "
    "Script 08 column structure."
)


# ============================================================
# 19. BUILD FINAL HEADLINE RESULTS TABLE
# ============================================================

print_section(
    "BUILDING FINAL HEADLINE RESULTS TABLE"
)


headline_rows = []


# ------------------------------------------------------------
# A. Global and regional elasticity results
# ------------------------------------------------------------

for _, row in table_core_elasticities.iterrows():

    if (
        row[
            "scope"
        ]
        ==
        "All economies"
    ):

        result_id = (
            "global_elasticity"
        )

    else:

        result_id = (
            "regional_elasticity_"
            +
            row[
                "scope"
            ]
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "&",
                "and",
            )
        )


    headline_rows.append(
        {
            "result_id":
                result_id,

            "result_family":
                "GDP-CO2 elasticity",

            "scope":
                row[
                    "scope"
                ],

            "estimate":
                row[
                    "estimate"
                ],

            "ci95_low":
                row[
                    "ci95_low"
                ],

            "ci95_high":
                row[
                    "ci95_high"
                ],

            "p_value":
                row[
                    "p_value"
                ],

            "unit":
                (
                    "CO2-per-capita elasticity with respect to "
                    "GDP per capita"
                ),

            "preferred_interpretation":
                (
                    "Association, not causal effect."
                ),

            "source_script":
                row[
                    "source_script"
                ],

            "source_file":
                row[
                    "source_file"
                ],
        }
    )


# ------------------------------------------------------------
# B. Global robustness range
# ------------------------------------------------------------

sample_robustness = (
    global_robustness[
        global_robustness[
            "sample_type"
        ]
        !=
        "Inference robustness"
    ]
    .copy()
)


robustness_min = float(
    sample_robustness[
        "coefficient"
    ]
    .min()
)


robustness_max = float(
    sample_robustness[
        "coefficient"
    ]
    .max()
)


headline_rows.append(
    {
        "result_id":
            "global_elasticity_robustness_range",

        "result_family":
            "GDP-CO2 robustness",

        "scope":
            "All sample-based robustness specifications",

        "estimate":
            float(
                baseline_global[
                    "coefficient"
                ]
            ),

        "ci95_low":
            np.nan,

        "ci95_high":
            np.nan,

        "p_value":
            np.nan,

        "unit":
            (
                "Baseline elasticity; robustness range described "
                "in preferred_interpretation"
            ),

        "preferred_interpretation":
            (
                "Sample-based elasticity estimates range from "
                f"{robustness_min:.3f} to "
                f"{robustness_max:.3f}; all corresponding "
                "95% confidence intervals remain above zero."
            ),

        "source_script":
            "13",

        "source_file":
            relative_path(
                GLOBAL_ROBUSTNESS_FILE
            ),
    }
)


# ------------------------------------------------------------
# C. Governance headline results
# ------------------------------------------------------------

for indicator in GOVERNANCE_ORDER:

    indicator_baseline = (
        baseline_spec_rows[
            baseline_spec_rows[
                "indicator"
            ]
            ==
            indicator
        ]
    )


    indicator_region = (
        region_adjusted_rows[
            region_adjusted_rows[
                "indicator"
            ]
            ==
            indicator
        ]
    )


    indicator_quadratic = (
        quadratic_rows[
            quadratic_rows[
                "indicator"
            ]
            ==
            indicator
        ]
    )


    if (
        len(
            indicator_baseline
        )
        != 1
        or
        len(
            indicator_region
        )
        != 1
        or
        len(
            indicator_quadratic
        )
        != 1
    ):

        raise ValueError(
            f"Governance headline result incomplete for "
            f"{indicator}."
        )


    indicator_baseline = (
        indicator_baseline.iloc[0]
    )


    indicator_region = (
        indicator_region.iloc[0]
    )


    indicator_quadratic = (
        indicator_quadratic.iloc[0]
    )


    headline_rows.append(
        {
            "result_id":
                (
                    "governance_"
                    +
                    indicator
                ),

            "result_family":
                "Governance interaction",

            "scope":
                GOVERNANCE_DISPLAY_NAMES[
                    indicator
                ],

            "estimate":
                float(
                    indicator_baseline[
                        "delta_elasticity_change_per_10pt_governance"
                    ]
                ),

            "ci95_low":
                float(
                    indicator_baseline[
                        "delta_ci95_low"
                    ]
                ),

            "ci95_high":
                float(
                    indicator_baseline[
                        "delta_ci95_high"
                    ]
                ),

            "p_value":
                float(
                    indicator_baseline[
                        "delta_p_value_holm"
                    ]
                ),

            "unit":
                (
                    "Change in GDP-CO2 elasticity per 10-point "
                    "higher governance score"
                ),

            "preferred_interpretation":
                (
                    "The baseline linear interaction is negative, "
                    f"but the region-adjusted estimate is "
                    f"{indicator_region['delta_elasticity_change_per_10pt_governance']:.3f} "
                    "and the quadratic-GDP estimate is "
                    f"{indicator_quadratic['delta_elasticity_change_per_10pt_governance']:.3f}. "
                    "The governance moderation result is "
                    "therefore specification-sensitive."
                ),

            "source_script":
                "12",

            "source_file":
                relative_path(
                    GOVERNANCE_SPECIFICATION_FILE
                ),
        }
    )


final_headline_results = pd.DataFrame(
    headline_rows
)


if final_headline_results[
    "result_id"
].duplicated().any():

    raise ValueError(
        "Duplicate final headline result IDs detected."
    )


final_headline_results.to_csv(
    OUTPUT_DIR
    / "final_headline_results.csv",
    index=False,
)


# ============================================================
# 20. BUILD PAPER-NUMBERS JSON
# ============================================================

print_section(
    "BUILDING PAPER-NUMBERS JSON"
)


global_baseline_json = {
    "estimate":
        float(
            baseline_global[
                "coefficient"
            ]
        ),

    "std_error":
        float(
            baseline_global[
                "std_error"
            ]
        ),

    "ci95":
        [
            float(
                baseline_global[
                    "ci95_low"
                ]
            ),

            float(
                baseline_global[
                    "ci95_high"
                ]
            ),
        ],

    "p_value":
        float(
            baseline_global[
                "p_value"
            ]
        ),

    "n_obs":
        int(
            baseline_global[
                "n_obs"
            ]
        ),

    "n_economies":
        int(
            baseline_global[
                "n_economies"
            ]
        ),

    "period":
        "1990-2024",
}


regional_json = {}


for _, row in (
    table_core_elasticities[
        table_core_elasticities[
            "scope_type"
        ]
        ==
        "Development-cluster elasticity"
    ]
    .iterrows()
):

    regional_json[
        row[
            "scope"
        ]
    ] = {
        "estimate":
            float(
                row[
                    "estimate"
                ]
            ),

        "ci95":
            [
                float(
                    row[
                        "ci95_low"
                    ]
                ),

                float(
                    row[
                        "ci95_high"
                    ]
                ),
            ],

        "p_value":
            float(
                row[
                    "p_value"
                ]
            ),

        "n_economies":
            int(
                row[
                    "n_economies"
                ]
            ),

        "inferential_caution":
            row[
                "inferential_caution"
            ],
    }


annual_decoupling_json = {}


long_decoupling_json = {}


for region in EXPECTED_REGIONS:

    display = (
        REGION_DISPLAY_NAMES[
            region
        ]
    )


    annual_group = (
        table_decoupling[
            (
                table_decoupling[
                    "analysis"
                ]
                ==
                "Annual"
            )
            &
            (
                table_decoupling[
                    "region"
                ]
                ==
                region
            )
        ]
    )


    long_group = (
        table_decoupling[
            (
                table_decoupling[
                    "analysis"
                ]
                ==
                "Long-period endpoints"
            )
            &
            (
                table_decoupling[
                    "region"
                ]
                ==
                region
            )
        ]
    )


    annual_decoupling_json[
        display
    ] = {
        row[
            "category"
        ]:
            float(
                row[
                    "share_percent"
                ]
            )
        for _, row in annual_group.iterrows()
    }


    long_decoupling_json[
        display
    ] = {
        row[
            "category"
        ]:
            float(
                row[
                    "share_percent"
                ]
            )
        for _, row in long_group.iterrows()
    }


governance_json = {}


for indicator in GOVERNANCE_ORDER:

    subset = (
        governance_specifications[
            governance_specifications[
                "indicator"
            ]
            ==
            indicator
        ]
    )


    governance_json[
        GOVERNANCE_DISPLAY_NAMES[
            indicator
        ]
    ] = {
        row[
            "specification"
        ]:
            {
                "estimate":
                    float(
                        row[
                            "delta_elasticity_change_per_10pt_governance"
                        ]
                    ),

                "ci95":
                    [
                        float(
                            row[
                                "delta_ci95_low"
                            ]
                        ),

                        float(
                            row[
                                "delta_ci95_high"
                            ]
                        ),
                    ],

                "p_value_holm":
                    float(
                        row[
                            "delta_p_value_holm"
                        ]
                    ),
            }
        for _, row in subset.iterrows()
    }


common_balanced_json = {}


for _, row in governance_common_balanced.iterrows():

    common_balanced_json[
        row[
            "indicator_display"
        ]
    ] = {
        "estimate":
            float(
                row[
                    "delta_elasticity_change_per_10pt_governance"
                ]
            ),

        "ci95":
            [
                float(
                    row[
                        "delta_ci95_low"
                    ]
                ),

                float(
                    row[
                        "delta_ci95_high"
                    ]
                ),
            ],

        "p_value":
            float(
                row[
                    "delta_p_value"
                ]
            ),

        "p_value_holm":
            float(
                row[
                    "delta_p_value_holm"
                ]
            ),
    }


between_within_json = {}


for indicator in GOVERNANCE_ORDER:

    indicator_rows = (
        governance_between_within[
            governance_between_within[
                "indicator"
            ]
            ==
            indicator
        ]
    )


    between_within_json[
        GOVERNANCE_DISPLAY_NAMES[
            indicator
        ]
    ] = {
        row[
            "component"
        ]:
            {
                "estimate":
                    float(
                        row[
                            "coefficient"
                        ]
                    ),

                "ci95":
                    [
                        float(
                            row[
                                "ci95_low"
                            ]
                        ),

                        float(
                            row[
                                "ci95_high"
                            ]
                        ),
                    ],

                "p_value_holm":
                    float(
                        row[
                            "p_value_holm"
                        ]
                    ),
            }
        for _, row in indicator_rows.iterrows()
    }


paper_numbers = {
    "project":
        {
            "title":
                (
                    "Economic Growth, CO2 Emissions, and "
                    "Governance: A Regional Panel Analysis"
                ),

            "analysis_period":
                "1990-2024",

            "analysis_status":
                (
                    "Econometric analysis frozen at "
                    "v1.0-analysis"
                ),

            "interpretive_rule":
                (
                    "Regression estimates are observational "
                    "associations rather than causal effects."
                ),
        },

    "global_elasticity":
        global_baseline_json,

    "global_sample_robustness_range":
        {
            "minimum":
                robustness_min,

            "maximum":
                robustness_max,
        },

    "regional_elasticities":
        regional_json,

    "decoupling":
        {
            "annual_positive_gdp_growth_observations":
                annual_decoupling_json,

            "long_period_1990_2024":
                long_decoupling_json,
        },

    "governance":
        {
            "specification_sensitivity":
                governance_json,

            "common_balanced_2002_2024":
                common_balanced_json,

            "between_within_decomposition":
                between_within_json,
        },

    "regional_ekc":
        dataframe_records_for_json(
            table_ekc
        ),
}


with open(
    OUTPUT_DIR
    / "paper_numbers.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        paper_numbers,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 21. SOURCE-PROVENANCE MANIFEST
# ============================================================

print_section(
    "BUILDING SOURCE PROVENANCE MANIFEST"
)


manifest_rows = []


for path in required_files:

    manifest_rows.append(
        {
            "source_file":
                relative_path(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "size_bytes":
                int(
                    path.stat().st_size
                ),

            "purpose":
                (
                    "Input to Script 14 final synthesis"
                ),
        }
    )


source_manifest = pd.DataFrame(
    manifest_rows
)


source_manifest.to_csv(
    OUTPUT_DIR
    / "source_manifest.csv",
    index=False,
)


# ============================================================
# 22. FINAL CORE ELASTICITY FIGURE
# ============================================================

print_section(
    "CREATING FINAL CORE ELASTICITY FIGURE"
)


figure_data = (
    table_core_elasticities
    .copy()
)


y_positions = np.arange(
    len(
        figure_data
    )
)


estimates = (
    figure_data[
        "estimate"
    ]
    .to_numpy()
)


lower_errors = (
    estimates
    -
    figure_data[
        "ci95_low"
    ]
    .to_numpy()
)


upper_errors = (
    figure_data[
        "ci95_high"
    ]
    .to_numpy()
    -
    estimates
)


fig, ax = plt.subplots(
    figsize=(
        10,
        6.5,
    )
)


ax.errorbar(
    estimates,
    y_positions,
    xerr=np.vstack(
        [
            lower_errors,
            upper_errors,
        ]
    ),
    fmt="o",
    capsize=4,
)


ax.axvline(
    0,
    linewidth=1,
    linestyle="--",
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    figure_data[
        "scope"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    "GDP per capita elasticity of CO2 emissions per capita"
)


ax.set_title(
    "Core GDP–CO2 Elasticity Results\n"
    "Economy and Year Fixed Effects, 1990–2024"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "Bars show 95% confidence intervals. China and India are "
        "single-economy development groups, so their regional "
        "inference requires additional caution."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.06,
        1,
        1,
    ]
)


FINAL_CORE_FIGURE = (
    FIGURES_DIR
    / "14_core_elasticities_summary.png"
)


fig.savefig(
    FINAL_CORE_FIGURE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved:\n{FINAL_CORE_FIGURE}"
)


# ============================================================
# 23. GENERATE FINAL RESULTS MARKDOWN
# ============================================================

print_section(
    "GENERATING FINAL RESULTS SUMMARY"
)


global_beta = float(
    baseline_global[
        "coefficient"
    ]
)


global_ci_low = float(
    baseline_global[
        "ci95_low"
    ]
)


global_ci_high = float(
    baseline_global[
        "ci95_high"
    ]
)


global_south_beta = float(
    baseline_regions.loc[
        baseline_regions[
            "region"
        ]
        ==
        "Global_South",
        "beta_log_gdp",
    ]
    .iloc[0]
)


china_beta = float(
    baseline_regions.loc[
        baseline_regions[
            "region"
        ]
        ==
        "China",
        "beta_log_gdp",
    ]
    .iloc[0]
)


india_beta = float(
    baseline_regions.loc[
        baseline_regions[
            "region"
        ]
        ==
        "India",
        "beta_log_gdp",
    ]
    .iloc[0]
)


europe_beta = float(
    baseline_regions.loc[
        baseline_regions[
            "region"
        ]
        ==
        "Europe_NorthAmerica",
        "beta_log_gdp",
    ]
    .iloc[0]
)


developed_asia_beta = float(
    baseline_regions.loc[
        baseline_regions[
            "region"
        ]
        ==
        "DevelopedAsia_Oceania",
        "beta_log_gdp",
    ]
    .iloc[0]
)


markdown_text = f"""# Final Empirical Results Synthesis

## Analysis status

The empirical analysis covers 1990–2024 and is frozen at the
`v1.0-analysis` milestone.

Script 14 does not estimate new econometric models. It consolidates
and validates results from the completed empirical pipeline.

## 1. Global GDP–CO2 relationship

The preferred economy and year fixed-effects specification gives a
GDP-per-capita elasticity of CO2 emissions per capita of
**{global_beta:.3f}**, with a 95% confidence interval of
**[{global_ci_low:.3f}, {global_ci_high:.3f}]**.

Within economies and conditional on common year effects, a 1%
increase in GDP per capita is therefore associated with approximately
a **{global_beta:.2f}% increase in CO2 emissions per capita**.

This is an observational association and should not be interpreted as
a causal effect.

Across the final sample-based robustness checks, the global elasticity
ranges from **{robustness_min:.3f}** to **{robustness_max:.3f}**.

## 2. Development-cluster heterogeneity

The baseline development-cluster elasticities are approximately:

- Europe & North America: **{europe_beta:.3f}**
- Developed Asia & Oceania: **{developed_asia_beta:.3f}**
- China: **{china_beta:.3f}**
- India: **{india_beta:.3f}**
- Global South: **{global_south_beta:.3f}**

The pooled global relationship therefore masks substantial
heterogeneity.

Europe & North America and Developed Asia & Oceania have small and
statistically imprecise slopes around zero, whereas the Global South
retains a much stronger positive income-emissions relationship.

China and India are intentionally retained as separate development
groups, but each contains only one economy. Their point estimates are
useful descriptions of their trajectories, while formal cluster-based
regional inference requires additional caution.

## 3. Decoupling

Annual decoupling is classified only among economy-year observations
with positive GDP-per-capita growth and consecutive annual data.

Long-period decoupling compares exact 1990 and 2024 endpoints and is
classified only among economies whose GDP per capita increased over
the period.

The paper-ready category shares are stored in:

`table_03_decoupling_summary.csv`

The three categories are:

- absolute decoupling;
- relative decoupling; and
- coupled expansion.

## 4. Governance

The baseline linear governance specifications suggest that higher
governance scores are associated with weaker GDP–CO2 coupling.

However, the relationship is not structurally robust.

After allowing development clusters to have different GDP slopes, the
negative governance interaction becomes small and statistically
imprecise. When nonlinear GDP dynamics are introduced, the governance
interaction changes sign.

The common balanced 2002–2024 comparison also shows that Government
Effectiveness provides the strongest negative baseline signal, while
Regulatory Quality and Rule of Law are weaker under this demanding
common-sample restriction.

The between/within decomposition further shows that the negative
baseline relationship is concentrated in persistent differences
between economies rather than changes in governance within the same
economy over time.

The governance extension therefore does **not** establish a stable
independent governance mechanism. Governance is better interpreted as
being associated with broader structural development differences.

## 5. Environmental Kuznets Curve evidence

Regional quadratic models are retained as exploratory benchmarks.

The Script 08 field `within_sample_ekc_candidate` identifies cases in
which the estimated quadratic satisfies a first-pass within-sample
turning-point screening rule.

These candidate flags should not be interpreted as causal or
structural proof of an Environmental Kuznets Curve.

The economy-level fixed-effects, regional-heterogeneity, decoupling,
governance-sensitivity and final-robustness results are given greater
weight in the final interpretation.

## 6. Final empirical conclusion

The central empirical result is not that economic growth has one
universal emissions consequence.

Instead:

1. the pooled within-economy GDP–CO2 relationship remains positive
   and robust;
2. the strength of that relationship differs substantially across
   development groups;
3. advanced-economy clusters display much greater evidence of
   decoupling;
4. China and India retain positive but lower elasticities than the
   Global South;
5. the Global South remains the most strongly growth-coupled
   multi-economy development cluster; and
6. governance correlates with these development patterns, but the
   analysis does not identify a stable independent governance
   moderation mechanism.

All regression results are interpreted as observational associations
rather than causal estimates.
"""


with open(
    OUTPUT_DIR
    / "final_results_summary.md",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        markdown_text
    )


# ============================================================
# 24. FINAL SYNTHESIS VALIDATION
# ============================================================

print_section(
    "FINAL SYNTHESIS VALIDATION"
)


validation_messages = []


def record_pass(message):
    """
    Store and print one successful validation result.
    """

    validation_messages.append(
        "PASS: "
        +
        message
    )


    print(
        "PASS: "
        +
        message
    )


# ------------------------------------------------------------
# A. Core elasticity table
# ------------------------------------------------------------

if len(
    table_core_elasticities
) != 6:

    raise ValueError(
        "Core elasticity table should contain one global and "
        "five development-cluster results."
    )


record_pass(
    "core elasticity table contains 6 expected results."
)


# ------------------------------------------------------------
# B. Global robustness
# ------------------------------------------------------------

if len(
    global_robustness
) != 11:

    raise ValueError(
        "Global robustness input should contain 11 results."
    )


record_pass(
    "global robustness table contains 11 expected results."
)


sample_only = (
    global_robustness[
        global_robustness[
            "sample_type"
        ]
        !=
        "Inference robustness"
    ]
)


if not (
    sample_only[
        "coefficient"
    ]
    > 0
).all():

    raise ValueError(
        "At least one sample-based global robustness estimate "
        "is non-positive."
    )


if not (
    sample_only[
        "ci95_low"
    ]
    > 0
).all():

    raise ValueError(
        "At least one sample-based global 95% confidence "
        "interval crosses zero."
    )


record_pass(
    "all sample-based global robustness estimates and 95% "
    "confidence intervals remain positive."
)


# ------------------------------------------------------------
# C. Regional robustness
# ------------------------------------------------------------

if len(
    regional_robustness
) != 25:

    raise ValueError(
        "Regional robustness input should contain 25 results."
    )


record_pass(
    "regional robustness table contains 25 expected results."
)


# ------------------------------------------------------------
# D. Strict balanced panel
# ------------------------------------------------------------

balanced_audit = (
    sample_audit[
        sample_audit[
            "specification"
        ]
        ==
        "Strict balanced panel"
    ]
)


if len(
    balanced_audit
) != 1:

    raise ValueError(
        "Could not identify strict balanced-panel audit row."
    )


balanced_audit = (
    balanced_audit.iloc[0]
)


if not parse_bool(
    balanced_audit[
        "is_strictly_balanced"
    ]
):

    raise ValueError(
        "Strict balanced-panel audit is not marked balanced."
    )


record_pass(
    "strict balanced-panel result remains validated."
)


# ------------------------------------------------------------
# E. Decoupling
# ------------------------------------------------------------

expected_decoupling_rows = (
    2
    *
    len(
        EXPECTED_REGIONS
    )
    *
    len(
        DECOUPLING_CATEGORIES
    )
)


if len(
    table_decoupling
) != expected_decoupling_rows:

    raise ValueError(
        "Decoupling synthesis has an unexpected number of rows. "
        f"Expected {expected_decoupling_rows}, "
        f"found {len(table_decoupling)}."
    )


record_pass(
    "annual and long-period decoupling contain all expected "
    "region-category combinations."
)


# ------------------------------------------------------------
# F. Governance
# ------------------------------------------------------------

if len(
    governance_specifications
) != 12:

    raise ValueError(
        "Governance specification table should contain "
        "3 indicators x 4 specifications = 12 rows."
    )


if len(
    governance_between_within
) != 6:

    raise ValueError(
        "Governance between/within table should contain "
        "3 indicators x 2 components = 6 rows."
    )


if len(
    governance_common_balanced
) != 3:

    raise ValueError(
        "Common balanced governance table should contain one "
        "row for each of the three governance indicators."
    )


record_pass(
    "governance specification sensitivity contains 12 expected "
    "results."
)


record_pass(
    "governance between/within decomposition contains 6 "
    "expected results."
)


record_pass(
    "common balanced governance robustness contains 3 expected "
    "results."
)


# ------------------------------------------------------------
# G. EKC
# ------------------------------------------------------------

if len(
    table_ekc
) != 5:

    raise ValueError(
        "Regional EKC summary should contain five regions."
    )


record_pass(
    "regional EKC benchmark contains five expected regions."
)


# ------------------------------------------------------------
# H. Source provenance
# ------------------------------------------------------------

if len(
    source_manifest
) != len(
    required_files
):

    raise ValueError(
        "Source provenance manifest is incomplete."
    )


if source_manifest[
    "sha256"
].duplicated().any():

    print(
        "NOTE: at least two source files have identical SHA-256 "
        "content. This is not necessarily an error."
    )


record_pass(
    "source provenance manifest contains every Script 14 input."
)


# ------------------------------------------------------------
# I. Headline results
# ------------------------------------------------------------

if final_headline_results[
    "result_id"
].duplicated().any():

    raise ValueError(
        "Headline results contain duplicate IDs."
    )


EXPECTED_HEADLINE_ROWS = (
    6
    +
    1
    +
    3
)


if len(
    final_headline_results
) != EXPECTED_HEADLINE_ROWS:

    raise ValueError(
        "Unexpected number of final headline results. "
        f"Expected {EXPECTED_HEADLINE_ROWS}, "
        f"found {len(final_headline_results)}."
    )


record_pass(
    "headline result identifiers are unique."
)


record_pass(
    "final headline table contains 10 authoritative results."
)


# ------------------------------------------------------------
# J. Paper numbers JSON exists and can be loaded
# ------------------------------------------------------------

PAPER_NUMBERS_FILE = (
    OUTPUT_DIR
    / "paper_numbers.json"
)


require_file(
    PAPER_NUMBERS_FILE
)


with open(
    PAPER_NUMBERS_FILE,
    "r",
    encoding="utf-8",
) as file:

    json_check = json.load(
        file
    )


required_json_sections = {
    "project",
    "global_elasticity",
    "global_sample_robustness_range",
    "regional_elasticities",
    "decoupling",
    "governance",
    "regional_ekc",
}


if not required_json_sections.issubset(
    set(
        json_check.keys()
    )
):

    raise ValueError(
        "paper_numbers.json is missing one or more required "
        "top-level sections."
    )


record_pass(
    "paper_numbers.json loads and contains all required sections."
)


# ============================================================
# 25. WRITE VALIDATION REPORT
# ============================================================

validation_report = (
    "\n".join(
        validation_messages
    )
    +
    "\n\n"
    +
    "SCRIPT 14 FINAL SYNTHESIS: PASS\n"
)


with open(
    OUTPUT_DIR
    / "final_synthesis_validation.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        validation_report
    )


# ============================================================
# 26. OUTPUT MANIFEST
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


output_files = [
    "table_01_core_elasticities.csv",
    "table_02_global_robustness.csv",
    "table_03_decoupling_summary.csv",
    "table_04_governance_summary.csv",
    "table_05_regional_ekc_summary.csv",
    "final_headline_results.csv",
    "paper_numbers.json",
    "source_manifest.csv",
    "final_results_summary.md",
    "final_synthesis_validation.txt",
]


print(
    f"Final synthesis directory:\n{OUTPUT_DIR}"
)


print(
    "\nGenerated files:"
)


for filename in output_files:

    print(
        f"- {filename}"
    )


print(
    "\nFigure:"
)


print(
    f"- {FINAL_CORE_FIGURE}"
)


print(
    "\nSCRIPT 14 COMPLETE."
)

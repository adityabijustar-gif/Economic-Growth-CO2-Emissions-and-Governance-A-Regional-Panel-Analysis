from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


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

SCRIPT09_RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "country_panel_results"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "country_panel_heterogeneity_results"
)

SUMMARY_DIR = (
    OUTPUT_DIR
    / "summaries"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. RESEARCH-DESIGN SETTINGS
# ============================================================

START_YEAR = 1990
END_YEAR = 2024

EXPECTED_YEARS = list(
    range(
        START_YEAR,
        END_YEAR + 1,
    )
)

EXPECTED_YEAR_SET = set(
    EXPECTED_YEARS
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


# These are the three development clusters containing multiple
# independent economy clusters.
#
# They are used for the preferred formal heterogeneity test
# because China and India each contain only one economy.
MULTI_ECONOMY_REGIONS = [
    "Europe_NorthAmerica",
    "DevelopedAsia_Oceania",
    "Global_South",
]


SINGLE_ECONOMY_REGIONS = [
    "China",
    "India",
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


MIN_YEARS_PER_ECONOMY = 2

ALPHA = 0.05

SIGNIFICANCE_LEVEL = 0.05

# Script 10 should reconstruct the Script 09 common-slope
# coefficient essentially exactly.
SCRIPT09_BETA_TOLERANCE = 1e-8


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
    Confirm that all required variables exist.
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
    """

    if pd.isna(
        p_value
    ):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.4f}"


def scalar(
    value,
):
    """
    Safely convert a Statsmodels scalar or one-element array
    into a standard Python float.
    """

    array = np.asarray(
        value
    )

    return float(
        array.squeeze()
    )


def fit_clustered_model(
    formula,
    data,
    weights=None,
):
    """
    Estimate OLS or WLS with economy-clustered standard errors.

    Features:
    - clustering at the economy level;
    - finite-sample covariance correction;
    - degrees-of-freedom correction;
    - Student-t inference.

    IMPORTANT:
    Cluster-robust inference is strongest for coefficients
    identified across many economy clusters. China and India
    each form one-economy development groups, so their
    group-specific p-values and confidence intervals require
    additional caution.
    """

    cluster_groups = pd.Categorical(
        data[
            "iso3c"
        ]
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


def check_design_rank(
    model,
    model_name,
):
    """
    Check that the regression design matrix is full rank.

    This protects against accidental exact collinearity.
    """

    design_matrix = (
        model.model.exog
    )

    rank = np.linalg.matrix_rank(
        design_matrix
    )

    n_columns = (
        design_matrix.shape[1]
    )


    if rank < n_columns:

        raise ValueError(
            f"{model_name} is rank deficient: "
            f"rank={rank}, columns={n_columns}."
        )


def save_model_summary(
    model,
    filename,
):
    """
    Save the complete Statsmodels summary locally.

    Existing filenames are deliberately retained so rerunning
    Script 10 cleanly replaces the previous summaries.
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


def create_slope_variables(
    data,
    base_variable,
    prefix,
):
    """
    Create one region-specific slope variable for each
    development cluster.

    Example:

        slope_China =
            log_gdp for China observations
            0 otherwise

    The corresponding estimated coefficient is therefore the
    China-specific GDP-CO2 elasticity directly.
    """

    data = data.copy()

    slope_columns = {}


    for region in EXPECTED_REGIONS:

        column_name = (
            f"{prefix}_{region}"
        )

        data[
            column_name
        ] = np.where(
            data[
                "region"
            ].eq(
                region
            ),

            data[
                base_variable
            ],

            0.0,
        )

        slope_columns[
            region
        ] = column_name


    return (
        data,
        slope_columns,
    )


def heterogeneous_formula(
    dependent_variable,
    slope_columns,
    include_economy_fe,
    include_year_fe,
    include_region_intercepts=False,
):
    """
    Construct the heterogeneous-slope regression formula.

    LEVELS MODELS
    -------------
    Region main effects are NOT included because regional
    membership does not vary within an economy and is therefore
    absorbed by economy fixed effects.

    FIRST-DIFFERENCE MODEL
    ----------------------
    Economy fixed effects disappear under differencing.
    Region intercepts can therefore be included to allow average
    annual emissions growth to differ systematically between
    development clusters.
    """

    components = list(
        slope_columns.values()
    )


    if include_region_intercepts:

        components.append(
            "C(region)"
        )


    if include_economy_fe:

        components.append(
            "C(iso3c)"
        )


    if include_year_fe:

        components.append(
            "C(year)"
        )


    formula = (
        f"{dependent_variable} ~ "
        +
        " + ".join(
            components
        )
    )


    return formula


def extract_regional_slopes(
    model,
    data,
    slope_columns,
    specification,
    first_difference=False,
    weighted=False,
    balanced=False,
):
    """
    Extract the regional GDP-CO2 elasticities.

    China and India remain in the output, but are explicitly
    labelled as single-economy groups so conventional
    cluster-robust inferential claims are not overstated.
    """

    rows = []

    confidence_intervals = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    for region in EXPECTED_REGIONS:

        coefficient_name = (
            slope_columns[
                region
            ]
        )


        if (
            coefficient_name
            not in
            model.params.index
        ):

            raise ValueError(
                f"Missing regional slope coefficient: "
                f"{coefficient_name}"
            )


        region_data = (
            data[
                data[
                    "region"
                ]
                == region
            ]
        )


        n_economies = int(
            region_data[
                "iso3c"
            ]
            .nunique()
        )


        n_observations = int(
            len(
                region_data
            )
        )


        ci = (
            confidence_intervals
            .loc[
                coefficient_name
            ]
        )


        single_economy = bool(
            n_economies
            == 1
        )


        if single_economy:

            inference_note = (
                "Single-economy development group. "
                "Point estimate is informative, but the "
                "conventional economy-clustered standard error, "
                "confidence interval and p-value should not be "
                "used for strong inferential claims."
            )

            primary_inference = False

        else:

            inference_note = (
                "Multi-economy development group. "
                "Economy-clustered inference is used."
            )

            primary_inference = True


        rows.append(
            {
                "specification":
                    specification,

                "region":
                    region,

                "region_display":
                    REGION_DISPLAY_NAMES[
                        region
                    ],

                "coefficient_name":
                    coefficient_name,

                "beta":
                    float(
                        model.params[
                            coefficient_name
                        ]
                    ),

                "std_error":
                    float(
                        model.bse[
                            coefficient_name
                        ]
                    ),

                "ci95_low":
                    float(
                        ci.iloc[0]
                    ),

                "ci95_high":
                    float(
                        ci.iloc[1]
                    ),

                "p_value":
                    float(
                        model.pvalues[
                            coefficient_name
                        ]
                    ),

                "n_observations":
                    n_observations,

                "n_economies":
                    n_economies,

                "first_year":
                    int(
                        region_data[
                            "year"
                        ]
                        .min()
                    ),

                "last_year":
                    int(
                        region_data[
                            "year"
                        ]
                        .max()
                    ),

                "single_economy_group":
                    single_economy,

                "primary_cluster_inference":
                    primary_inference,

                "population_weighted":
                    bool(
                        weighted
                    ),

                "balanced_panel":
                    bool(
                        balanced
                    ),

                "first_difference":
                    bool(
                        first_difference
                    ),

                "inference_note":
                    inference_note,
            }
        )


    return pd.DataFrame(
        rows
    )


def joint_equality_test(
    model,
    slope_columns,
    specification,
    regions_to_test,
    test_scope,
    primary_inference,
):
    """
    Jointly test equality of a specified collection of regional
    GDP-CO2 elasticities.

    Examples
    --------

    ALL-FIVE TEST:

        H0:
        beta_Europe
        =
        beta_DevelopedAsia
        =
        beta_China
        =
        beta_India
        =
        beta_GlobalSouth


    MULTI-ECONOMY TEST:

        H0:
        beta_Europe
        =
        beta_DevelopedAsia
        =
        beta_GlobalSouth

    The latter is the preferred inferential test because all
    three coefficients are identified from multiple economy
    clusters.
    """

    if len(
        regions_to_test
    ) < 2:

        raise ValueError(
            "A joint equality test requires at least two regions."
        )


    parameter_names = list(
        model.params.index
    )


    parameter_lookup = {
        name: index
        for index, name
        in enumerate(
            parameter_names
        )
    }


    # Use the final region in the requested collection as the
    # comparison baseline.
    baseline_region = (
        regions_to_test[-1]
    )


    baseline_column = (
        slope_columns[
            baseline_region
        ]
    )


    restrictions = []


    for region in regions_to_test[:-1]:

        row = np.zeros(
            len(
                parameter_names
            )
        )


        region_column = (
            slope_columns[
                region
            ]
        )


        row[
            parameter_lookup[
                region_column
            ]
        ] = 1.0


        row[
            parameter_lookup[
                baseline_column
            ]
        ] = -1.0


        restrictions.append(
            row
        )


    restriction_matrix = np.vstack(
        restrictions
    )


    test = model.f_test(
        restriction_matrix
    )


    p_value = scalar(
        test.pvalue
    )


    includes_singleton = any(
        region in SINGLE_ECONOMY_REGIONS
        for region in regions_to_test
    )


    if includes_singleton:

        inference_note = (
            "Includes China and/or India, which are "
            "single-economy development groups. "
            "Treat this test as supplementary."
        )

    else:

        inference_note = (
            "All tested groups contain multiple economies. "
            "This is the preferred formal heterogeneity test."
        )


    return {
        "specification":
            specification,

        "test_scope":
            test_scope,

        "regions_tested":
            " | ".join(
                REGION_DISPLAY_NAMES[
                    region
                ]
                for region
                in regions_to_test
            ),

        "n_regions_tested":
            len(
                regions_to_test
            ),

        "includes_single_economy_group":
            includes_singleton,

        "primary_inference":
            bool(
                primary_inference
            ),

        "null_hypothesis":
            (
                "All tested regional GDP-emissions "
                "elasticities are equal"
            ),

        "test":
            "Economy-cluster-robust F test",

        "f_statistic":
            scalar(
                test.fvalue
            ),

        "p_value":
            p_value,

        "df_num":
            float(
                test.df_num
            ),

        "df_denom":
            float(
                test.df_denom
            ),

        "reject_equal_slopes_at_5pct":
            bool(
                p_value
                <
                SIGNIFICANCE_LEVEL
            ),

        "inference_note":
            inference_note,
    }


def run_joint_tests(
    model,
    slope_columns,
    specification,
):
    """
    Run both:

    1. the original five-group equality test; and
    2. the preferred three-multi-economy-group equality test.

    Both are returned to the EXISTING
    regional_heterogeneity_joint_tests.csv file.
    """

    all_five_test = joint_equality_test(
        model=model,
        slope_columns=slope_columns,
        specification=specification,
        regions_to_test=EXPECTED_REGIONS,
        test_scope="All five development groups",
        primary_inference=False,
    )


    multi_economy_test = joint_equality_test(
        model=model,
        slope_columns=slope_columns,
        specification=specification,
        regions_to_test=MULTI_ECONOMY_REGIONS,
        test_scope="Three multi-economy development groups",
        primary_inference=True,
    )


    return [
        all_five_test,
        multi_economy_test,
    ]


def pairwise_regional_tests(
    model,
    data,
    slope_columns,
    specification,
):
    """
    Test all 10 pairwise differences between the five regional
    elasticities.

    The original Holm adjustment across all 10 comparisons is
    retained exactly as the primary multiple-testing correction.

    Comparisons involving China or India are explicitly flagged
    because those development groups contain only one economy.
    """

    parameter_names = list(
        model.params.index
    )


    parameter_lookup = {
        name: index
        for index, name
        in enumerate(
            parameter_names
        )
    }


    economy_counts = (
        data
        .groupby(
            "region"
        )[
            "iso3c"
        ]
        .nunique()
        .to_dict()
    )


    rows = []


    for (
        region_a,
        region_b,
    ) in combinations(
        EXPECTED_REGIONS,
        2,
    ):

        contrast = np.zeros(
            len(
                parameter_names
            )
        )


        coefficient_a = (
            slope_columns[
                region_a
            ]
        )


        coefficient_b = (
            slope_columns[
                region_b
            ]
        )


        contrast[
            parameter_lookup[
                coefficient_a
            ]
        ] = 1.0


        contrast[
            parameter_lookup[
                coefficient_b
            ]
        ] = -1.0


        test = model.t_test(
            contrast
        )


        confidence_interval = (
            test.conf_int(
                alpha=ALPHA
            )[0]
        )


        n_economies_a = int(
            economy_counts[
                region_a
            ]
        )


        n_economies_b = int(
            economy_counts[
                region_b
            ]
        )


        involves_single_economy = bool(
            n_economies_a
            == 1
            or
            n_economies_b
            == 1
        )


        both_multi_economy = bool(
            n_economies_a
            > 1
            and
            n_economies_b
            > 1
        )


        if involves_single_economy:

            inference_note = (
                "Comparison involves China and/or India, "
                "a single-economy development group. "
                "Treat conventional cluster-robust inference "
                "as supplementary."
            )

        else:

            inference_note = (
                "Both development groups contain multiple "
                "economies; conventional economy-clustered "
                "inference is appropriate."
            )


        rows.append(
            {
                "specification":
                    specification,

                "region_a":
                    region_a,

                "region_a_display":
                    REGION_DISPLAY_NAMES[
                        region_a
                    ],

                "region_b":
                    region_b,

                "region_b_display":
                    REGION_DISPLAY_NAMES[
                        region_b
                    ],

                "region_a_n_economies":
                    n_economies_a,

                "region_b_n_economies":
                    n_economies_b,

                "both_multi_economy_groups":
                    both_multi_economy,

                "involves_single_economy_group":
                    involves_single_economy,

                "difference_beta_a_minus_b":
                    scalar(
                        test.effect
                    ),

                "std_error":
                    scalar(
                        test.sd
                    ),

                "ci95_low":
                    float(
                        confidence_interval[
                            0
                        ]
                    ),

                "ci95_high":
                    float(
                        confidence_interval[
                            1
                        ]
                    ),

                "t_statistic":
                    scalar(
                        test.tvalue
                    ),

                "p_value_raw":
                    scalar(
                        test.pvalue
                    ),

                "inference_note":
                    inference_note,
            }
        )


    result = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Retain the ORIGINAL Holm correction across all 10
    # pairwise tests.
    #
    # This preserves comparability with previous Script 10
    # outputs and remains a conservative procedure.
    # --------------------------------------------------------

    (
        reject,
        p_adjusted,
        _,
        _,
    ) = multipletests(
        result[
            "p_value_raw"
        ],
        alpha=SIGNIFICANCE_LEVEL,
        method="holm",
    )


    result[
        "p_value_holm"
    ] = (
        p_adjusted
    )


    result[
        "significant_raw_5pct"
    ] = (
        result[
            "p_value_raw"
        ]
        <
        SIGNIFICANCE_LEVEL
    )


    result[
        "significant_holm_5pct"
    ] = (
        reject
    )


    return result


def print_regional_slopes(
    slope_results,
    title,
):
    """
    Print a compact regional coefficient table.
    """

    print_section(
        title
    )


    display_columns = [
        "region_display",
        "beta",
        "std_error",
        "ci95_low",
        "ci95_high",
        "p_value",
        "n_economies",
        "n_observations",
        "primary_cluster_inference",
    ]


    print(
        slope_results[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print_section(
    "LOADING SOURCE DATA"
)


panel = pd.read_csv(
    COUNTRY_PANEL_FILE
)


region_map = pd.read_csv(
    REGION_MAP_FILE
)


print(
    f"Economy-year panel rows: "
    f"{len(panel):,}"
)

print(
    f"Region-map rows: "
    f"{len(region_map):,}"
)


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

require_columns(
    panel,
    [
        "country",
        "iso3c",
        "year",
        "gdp_per_capita_const2015_usd",
        "co2_per_capita_tons",
        "population",
    ],
    "WDI economy-year panel",
)


require_columns(
    region_map,
    [
        "iso3c",
        "region",
    ],
    "Economy-region mapping",
)


# ============================================================
# 6. STANDARDISE IDENTIFIERS AND TYPES
# ============================================================

panel = panel.copy()

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
] = (
    pd.to_numeric(
        panel[
            "year"
        ],
        errors="raise",
    )
    .astype(int)
)


for column in [
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_tons",
    "population",
]:

    panel[
        column
    ] = pd.to_numeric(
        panel[
            column
        ],
        errors="coerce",
    )


# ============================================================
# 7. VALIDATE REGION MAPPING
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
        "Region mapping contains missing economy codes "
        "or region labels."
    )


duplicate_region_codes = (
    region_map[
        region_map.duplicated(
            "iso3c",
            keep=False,
        )
    ]
)


if not duplicate_region_codes.empty:

    raise ValueError(
        "Some economy codes have multiple regional assignments."
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
    "PASS: five expected regional-development clusters found."
)


# ============================================================
# 8. VALIDATE ECONOMY-YEAR KEYS
# ============================================================

if panel.duplicated(
    [
        "iso3c",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate economy-year observations found."
    )


print(
    "PASS: no duplicate economy-year observations."
)


# ============================================================
# 9. RECONSTRUCT THE EXACT SCRIPT 09 MAIN SAMPLE
# ============================================================

print_section(
    "RECONSTRUCTING SCRIPT 09 MAIN SAMPLE"
)


panel = (
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
    ]
    .copy()
)


panel = (
    panel
    .merge(
        region_map,
        on="iso3c",
        how="inner",
        validate="many_to_one",
    )
)


valid_core = (
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
)


valid_panel = (
    panel[
        valid_core
    ]
    .copy()
)


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
        >= MIN_YEARS_PER_ECONOMY
    ]
    .index
)


main_sample = (
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


main_sample = (
    main_sample
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


main_sample[
    "log_gdp"
] = np.log(
    main_sample[
        "gdp_per_capita_const2015_usd"
    ]
)


main_sample[
    "log_co2"
] = np.log(
    main_sample[
        "co2_per_capita_tons"
    ]
)


if not np.isfinite(
    main_sample[
        [
            "log_gdp",
            "log_co2",
        ]
    ]
    .to_numpy()
).all():

    raise ValueError(
        "Non-finite log values remain in the main sample."
    )


print(
    f"Main-sample observations: "
    f"{len(main_sample):,}"
)

print(
    f"Main-sample economies: "
    f"{main_sample['iso3c'].nunique():,}"
)


# ============================================================
# 10. CROSS-CHECK AGAINST SCRIPT 09
# ============================================================

print_section(
    "SCRIPT 09 CONSISTENCY CHECK"
)


SCRIPT09_FE_FILE = (
    SCRIPT09_RESULTS_DIR
    / "country_panel_fe_results.csv"
)


if SCRIPT09_FE_FILE.exists():

    script09_results = pd.read_csv(
        SCRIPT09_FE_FILE
    )


    require_columns(
        script09_results,
        [
            "model",
            "beta",
            "n_obs",
            "n_countries",
        ],
        "Script 09 FE results",
    )


    script09_main = (
        script09_results[
            script09_results[
                "model"
            ]
            ==
            "Country + year fixed effects"
        ]
    )


    if len(
        script09_main
    ) != 1:

        raise ValueError(
            "Could not uniquely identify the Script 09 "
            "country + year FE result."
        )


    script09_main = (
        script09_main.iloc[0]
    )


    expected_n_obs = int(
        script09_main[
            "n_obs"
        ]
    )


    expected_n_economies = int(
        script09_main[
            "n_countries"
        ]
    )


    if (
        len(
            main_sample
        )
        != expected_n_obs
    ):

        raise ValueError(
            "Script 10 main-sample observation count does not "
            "match Script 09."
        )


    if (
        main_sample[
            "iso3c"
        ]
        .nunique()
        != expected_n_economies
    ):

        raise ValueError(
            "Script 10 economy count does not match Script 09."
        )


    print(
        "PASS: Script 10 reconstructs the same observations "
        "and economy count as Script 09."
    )


else:

    script09_main = None

    print(
        "WARNING: Script 09 result file was not found. "
        "Sample reconstruction will continue independently."
    )


# ============================================================
# 11. REGIONAL SAMPLE DIAGNOSTICS
# ============================================================

print_section(
    "REGIONAL SAMPLE DIAGNOSTICS"
)


main_sample[
    "within_log_gdp"
] = (
    main_sample[
        "log_gdp"
    ]
    -
    main_sample
    .groupby(
        "iso3c"
    )[
        "log_gdp"
    ]
    .transform(
        "mean"
    )
)


region_sample_summary = (
    main_sample
    .groupby(
        "region",
        as_index=False,
    )
    .agg(
        n_economies=(
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

        within_log_gdp_sd=(
            "within_log_gdp",
            "std",
        ),
    )
)


region_sample_summary[
    "region_display"
] = (
    region_sample_summary[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


region_sample_summary[
    "single_economy_group"
] = (
    region_sample_summary[
        "n_economies"
    ]
    == 1
)


region_sample_summary[
    "primary_cluster_inference"
] = (
    region_sample_summary[
        "n_economies"
    ]
    > 1
)


if (
    region_sample_summary[
        "within_log_gdp_sd"
    ]
    <= 0
).any():

    raise ValueError(
        "At least one development cluster contains no usable "
        "within-economy GDP variation."
    )


region_sample_summary.to_csv(
    OUTPUT_DIR
    / "regional_heterogeneity_region_sample_summary.csv",
    index=False,
)


print(
    region_sample_summary[
        [
            "region_display",
            "n_economies",
            "n_observations",
            "within_log_gdp_sd",
            "single_economy_group",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. COMMON-SLOPE TWO-WAY FE BENCHMARK
# ============================================================

print_section(
    "COMMON-SLOPE TWO-WAY FIXED EFFECTS BENCHMARK"
)


common_slope_model = fit_clustered_model(
    formula=(
        "log_co2 ~ "
        "log_gdp + "
        "C(iso3c) + "
        "C(year)"
    ),
    data=main_sample,
)


check_design_rank(
    common_slope_model,
    "Common-slope TWFE model",
)


common_slope_beta = float(
    common_slope_model.params[
        "log_gdp"
    ]
)


print(
    "Common GDP-emissions elasticity:",
    f"{common_slope_beta:.6f}",
)


if script09_main is not None:

    script09_beta = float(
        script09_main[
            "beta"
        ]
    )


    if not np.isclose(
        common_slope_beta,
        script09_beta,
        atol=SCRIPT09_BETA_TOLERANCE,
        rtol=0,
    ):

        raise ValueError(
            "Script 10 common-slope coefficient does not "
            "reproduce Script 09 within tolerance."
        )


    print(
        "PASS: common-slope coefficient reproduces Script 09."
    )


save_model_summary(
    common_slope_model,
    "01_common_slope_twfe_summary.txt",
)


# ============================================================
# 13. MAIN REGIONAL-HETEROGENEITY TWFE MODEL
# ============================================================

print_section(
    "MAIN REGIONAL-HETEROGENEITY TWO-WAY FE MODEL"
)


(
    main_heterogeneity_sample,
    main_slope_columns,
) = create_slope_variables(
    data=main_sample,
    base_variable="log_gdp",
    prefix="slope",
)


main_formula = heterogeneous_formula(
    dependent_variable="log_co2",
    slope_columns=main_slope_columns,
    include_economy_fe=True,
    include_year_fe=True,
    include_region_intercepts=False,
)


main_heterogeneity_model = fit_clustered_model(
    formula=main_formula,
    data=main_heterogeneity_sample,
)


check_design_rank(
    main_heterogeneity_model,
    "Main regional-heterogeneity TWFE model",
)


save_model_summary(
    main_heterogeneity_model,
    "02_main_regional_heterogeneity_twfe_summary.txt",
)


main_slopes = extract_regional_slopes(
    model=main_heterogeneity_model,
    data=main_heterogeneity_sample,
    slope_columns=main_slope_columns,
    specification="Main two-way FE regional slopes",
    first_difference=False,
    weighted=False,
    balanced=False,
)


print_regional_slopes(
    main_slopes,
    "MAIN WITHIN-ECONOMY REGIONAL ELASTICITIES",
)


# ============================================================
# 14. MAIN JOINT HETEROGENEITY TESTS
# ============================================================

print_section(
    "MAIN JOINT TESTS OF REGIONAL HETEROGENEITY"
)


main_joint_tests = run_joint_tests(
    model=main_heterogeneity_model,
    slope_columns=main_slope_columns,
    specification="Main two-way FE regional slopes",
)


for test_result in main_joint_tests:

    print(
        f"\n{test_result['test_scope']}"
    )

    print(
        "H0: all tested regional elasticities are equal"
    )

    print(
        "F-statistic:",
        f"{test_result['f_statistic']:.4f}",
    )

    print(
        "p-value:",
        format_p_value(
            test_result[
                "p_value"
            ]
        ),
    )

    print(
        "Primary inference:",
        test_result[
            "primary_inference"
        ],
    )

    print(
        "Reject equal slopes at 5%:",
        test_result[
            "reject_equal_slopes_at_5pct"
        ],
    )


# ============================================================
# 15. MAIN PAIRWISE REGIONAL TESTS
# ============================================================

main_pairwise = pairwise_regional_tests(
    model=main_heterogeneity_model,
    data=main_heterogeneity_sample,
    slope_columns=main_slope_columns,
    specification="Main two-way FE regional slopes",
)


print_section(
    "MAIN PAIRWISE REGIONAL SLOPE TESTS"
)


print(
    main_pairwise[
        [
            "region_a_display",
            "region_b_display",
            "difference_beta_a_minus_b",
            "p_value_raw",
            "p_value_holm",
            "significant_holm_5pct",
            "both_multi_economy_groups",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. COMMON VS HETEROGENEOUS MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame(
    [
        {
            "model":
                "Common-slope economy + year FE",

            "n_obs":
                int(
                    common_slope_model.nobs
                ),

            "n_parameters":
                int(
                    len(
                        common_slope_model.params
                    )
                ),

            "r_squared":
                float(
                    common_slope_model.rsquared
                ),

            "adjusted_r_squared":
                float(
                    common_slope_model.rsquared_adj
                ),

            "aic":
                float(
                    common_slope_model.aic
                ),

            "bic":
                float(
                    common_slope_model.bic
                ),
        },

        {
            "model":
                "Region-specific-slope economy + year FE",

            "n_obs":
                int(
                    main_heterogeneity_model.nobs
                ),

            "n_parameters":
                int(
                    len(
                        main_heterogeneity_model.params
                    )
                ),

            "r_squared":
                float(
                    main_heterogeneity_model.rsquared
                ),

            "adjusted_r_squared":
                float(
                    main_heterogeneity_model.rsquared_adj
                ),

            "aic":
                float(
                    main_heterogeneity_model.aic
                ),

            "bic":
                float(
                    main_heterogeneity_model.bic
                ),
        },
    ]
)


for criterion in [
    "adjusted_r_squared",
    "aic",
    "bic",
]:

    baseline_value = (
        model_comparison
        .loc[
            model_comparison[
                "model"
            ]
            ==
            "Common-slope economy + year FE",
            criterion,
        ]
        .iloc[0]
    )


    model_comparison[
        f"delta_{criterion}_vs_common"
    ] = (
        model_comparison[
            criterion
        ]
        -
        baseline_value
    )


model_comparison[
    "note"
] = [
    (
        "Script 09 common-slope benchmark."
    ),

    (
        "Allows five region-specific GDP-emissions slopes. "
        "The preferred formal heterogeneity test is the joint "
        "test across Europe/North America, Developed Asia & "
        "Oceania, and the Global South. "
        "AIC/BIC are supplementary."
    ),
]


model_comparison.to_csv(
    OUTPUT_DIR
    / "common_vs_heterogeneous_model_comparison.csv",
    index=False,
)


# ============================================================
# 17. BALANCED-PANEL ROBUSTNESS SAMPLE
# ============================================================

print_section(
    "BALANCED-PANEL REGIONAL-HETEROGENEITY ROBUSTNESS"
)


def is_complete_year_set(
    series,
):
    """
    Return True only when an economy has every year
    from 1990 through 2024.
    """

    return (
        set(
            int(year)
            for year
            in series
        )
        ==
        EXPECTED_YEAR_SET
    )


balanced_status = (
    main_sample
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .apply(
        is_complete_year_set
    )
)


balanced_codes = set(
    balanced_status[
        balanced_status
    ]
    .index
)


balanced_sample = (
    main_sample[
        main_sample[
            "iso3c"
        ]
        .isin(
            balanced_codes
        )
    ]
    .copy()
)


(
    balanced_sample,
    balanced_slope_columns,
) = create_slope_variables(
    data=balanced_sample,
    base_variable="log_gdp",
    prefix="slope",
)


balanced_formula = heterogeneous_formula(
    dependent_variable="log_co2",
    slope_columns=balanced_slope_columns,
    include_economy_fe=True,
    include_year_fe=True,
    include_region_intercepts=False,
)


balanced_model = fit_clustered_model(
    formula=balanced_formula,
    data=balanced_sample,
)


check_design_rank(
    balanced_model,
    "Balanced regional-heterogeneity TWFE model",
)


save_model_summary(
    balanced_model,
    "03_balanced_regional_heterogeneity_twfe_summary.txt",
)


balanced_slopes = extract_regional_slopes(
    model=balanced_model,
    data=balanced_sample,
    slope_columns=balanced_slope_columns,
    specification="Balanced-panel two-way FE regional slopes",
    first_difference=False,
    weighted=False,
    balanced=True,
)


balanced_joint_tests = run_joint_tests(
    model=balanced_model,
    slope_columns=balanced_slope_columns,
    specification="Balanced-panel two-way FE regional slopes",
)


balanced_pairwise = pairwise_regional_tests(
    model=balanced_model,
    data=balanced_sample,
    slope_columns=balanced_slope_columns,
    specification="Balanced-panel two-way FE regional slopes",
)


print_regional_slopes(
    balanced_slopes,
    "BALANCED-PANEL REGIONAL ELASTICITIES",
)


# ============================================================
# 18. POPULATION-WEIGHTED ROBUSTNESS
# ============================================================

print_section(
    "POPULATION-WEIGHTED REGIONAL-HETEROGENEITY ROBUSTNESS"
)


weighted_sample = (
    main_sample[
        main_sample[
            "population"
        ]
        .notna()
        &
        (
            main_sample[
                "population"
            ]
            > 0
        )
    ]
    .copy()
)


if weighted_sample.empty:

    raise ValueError(
        "No positive population observations are available "
        "for the weighted robustness specification."
    )


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


(
    weighted_sample,
    weighted_slope_columns,
) = create_slope_variables(
    data=weighted_sample,
    base_variable="log_gdp",
    prefix="slope",
)


weighted_formula = heterogeneous_formula(
    dependent_variable="log_co2",
    slope_columns=weighted_slope_columns,
    include_economy_fe=True,
    include_year_fe=True,
    include_region_intercepts=False,
)


weighted_model = fit_clustered_model(
    formula=weighted_formula,
    data=weighted_sample,
    weights=weighted_sample[
        "population_weight"
    ],
)


check_design_rank(
    weighted_model,
    "Population-weighted regional-heterogeneity TWFE model",
)


save_model_summary(
    weighted_model,
    "04_population_weighted_regional_heterogeneity_summary.txt",
)


weighted_slopes = extract_regional_slopes(
    model=weighted_model,
    data=weighted_sample,
    slope_columns=weighted_slope_columns,
    specification="Population-weighted two-way FE regional slopes",
    first_difference=False,
    weighted=True,
    balanced=False,
)


weighted_joint_tests = run_joint_tests(
    model=weighted_model,
    slope_columns=weighted_slope_columns,
    specification="Population-weighted two-way FE regional slopes",
)


weighted_pairwise = pairwise_regional_tests(
    model=weighted_model,
    data=weighted_sample,
    slope_columns=weighted_slope_columns,
    specification="Population-weighted two-way FE regional slopes",
)


print_regional_slopes(
    weighted_slopes,
    "POPULATION-WEIGHTED REGIONAL ELASTICITIES",
)


# ============================================================
# 19. CONSTRUCT FIRST-DIFFERENCE SAMPLE
# ============================================================

print_section(
    "CONSTRUCTING FIRST-DIFFERENCE HETEROGENEITY SAMPLE"
)


difference_source = (
    main_sample
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
    f"Economies contributing differences: "
    f"{difference_sample['iso3c'].nunique():,}"
)


# ============================================================
# 20. CROSS-CHECK FIRST-DIFFERENCE SAMPLE AGAINST SCRIPT 09
# ============================================================

SCRIPT09_FD_FILE = (
    SCRIPT09_RESULTS_DIR
    / "country_panel_first_difference_results.csv"
)


if SCRIPT09_FD_FILE.exists():

    script09_fd = pd.read_csv(
        SCRIPT09_FD_FILE
    )


    if len(
        script09_fd
    ) != 1:

        raise ValueError(
            "Script 09 first-difference results contain an "
            "unexpected number of rows."
        )


    script09_fd = (
        script09_fd.iloc[0]
    )


    if (
        len(
            difference_sample
        )
        !=
        int(
            script09_fd[
                "n_obs"
            ]
        )
    ):

        raise ValueError(
            "Script 10 first-difference sample does not match "
            "Script 09."
        )


    if (
        difference_sample[
            "iso3c"
        ]
        .nunique()
        !=
        int(
            script09_fd[
                "n_countries"
            ]
        )
    ):

        raise ValueError(
            "Script 10 first-difference economy count does not "
            "match Script 09."
        )


    print(
        "PASS: first-difference sample matches Script 09."
    )


# ============================================================
# 21. FIRST-DIFFERENCE REGIONAL-HETEROGENEITY MODEL
# ============================================================

(
    difference_sample,
    difference_slope_columns,
) = create_slope_variables(
    data=difference_sample,
    base_variable="d_log_gdp",
    prefix="d_slope",
)


# ------------------------------------------------------------
# Unlike the levels model, economy fixed effects have already
# been removed through first differencing.
#
# Region intercepts are therefore included so average annual
# emissions growth may differ across development clusters
# independently of the GDP-growth slope.
# ------------------------------------------------------------

difference_formula = heterogeneous_formula(
    dependent_variable="d_log_co2",
    slope_columns=difference_slope_columns,
    include_economy_fe=False,
    include_year_fe=True,
    include_region_intercepts=True,
)


difference_model = fit_clustered_model(
    formula=difference_formula,
    data=difference_sample,
)


check_design_rank(
    difference_model,
    "First-difference regional-heterogeneity model",
)


save_model_summary(
    difference_model,
    "05_first_difference_regional_heterogeneity_summary.txt",
)


difference_slopes = extract_regional_slopes(
    model=difference_model,
    data=difference_sample,
    slope_columns=difference_slope_columns,
    specification="First-difference regional slopes",
    first_difference=True,
    weighted=False,
    balanced=False,
)


difference_joint_tests = run_joint_tests(
    model=difference_model,
    slope_columns=difference_slope_columns,
    specification="First-difference regional slopes",
)


difference_pairwise = pairwise_regional_tests(
    model=difference_model,
    data=difference_sample,
    slope_columns=difference_slope_columns,
    specification="First-difference regional slopes",
)


print_regional_slopes(
    difference_slopes,
    "FIRST-DIFFERENCE REGIONAL ELASTICITIES",
)


# ============================================================
# 22. COMBINE AND EXPORT REGIONAL SLOPE RESULTS
# ============================================================

all_slope_results = pd.concat(
    [
        main_slopes,
        balanced_slopes,
        weighted_slopes,
        difference_slopes,
    ],
    ignore_index=True,
)


# EXISTING filename retained.
all_slope_results.to_csv(
    OUTPUT_DIR
    / "regional_heterogeneity_slopes.csv",
    index=False,
)


# EXISTING filename retained.
main_slopes.to_csv(
    OUTPUT_DIR
    / "main_regional_twfe_slopes.csv",
    index=False,
)


# ============================================================
# 23. COMBINE AND EXPORT JOINT HETEROGENEITY TESTS
# ============================================================

# Each specification now contributes TWO rows:
#
# 1. all five development groups;
# 2. the preferred three-multi-economy-group test.
#
# The EXISTING output filename is retained.

joint_tests = pd.DataFrame(
    (
        main_joint_tests
        +
        balanced_joint_tests
        +
        weighted_joint_tests
        +
        difference_joint_tests
    )
)


joint_tests.to_csv(
    OUTPUT_DIR
    / "regional_heterogeneity_joint_tests.csv",
    index=False,
)


# ============================================================
# 24. COMBINE AND EXPORT PAIRWISE TESTS
# ============================================================

all_pairwise_tests = pd.concat(
    [
        main_pairwise,
        balanced_pairwise,
        weighted_pairwise,
        difference_pairwise,
    ],
    ignore_index=True,
)


# EXISTING filename retained.
all_pairwise_tests.to_csv(
    OUTPUT_DIR
    / "regional_pairwise_slope_tests.csv",
    index=False,
)


# EXISTING filename retained.
main_pairwise.to_csv(
    OUTPUT_DIR
    / "main_regional_pairwise_slope_tests.csv",
    index=False,
)


# ============================================================
# 25. ROBUSTNESS COMPARISON TABLE
# ============================================================

main_for_merge = (
    main_slopes[
        [
            "region",
            "region_display",
            "beta",
        ]
    ]
    .rename(
        columns={
            "beta":
                "main_twfe_beta"
        }
    )
)


balanced_for_merge = (
    balanced_slopes[
        [
            "region",
            "beta",
        ]
    ]
    .rename(
        columns={
            "beta":
                "balanced_twfe_beta"
        }
    )
)


weighted_for_merge = (
    weighted_slopes[
        [
            "region",
            "beta",
        ]
    ]
    .rename(
        columns={
            "beta":
                "population_weighted_twfe_beta"
        }
    )
)


difference_for_merge = (
    difference_slopes[
        [
            "region",
            "beta",
        ]
    ]
    .rename(
        columns={
            "beta":
                "first_difference_beta"
        }
    )
)


robustness_comparison = (
    main_for_merge
    .merge(
        balanced_for_merge,
        on="region",
        how="left",
    )
    .merge(
        weighted_for_merge,
        on="region",
        how="left",
    )
    .merge(
        difference_for_merge,
        on="region",
        how="left",
    )
)


robustness_comparison[
    "balanced_minus_main"
] = (
    robustness_comparison[
        "balanced_twfe_beta"
    ]
    -
    robustness_comparison[
        "main_twfe_beta"
    ]
)


robustness_comparison[
    "population_weighted_minus_main"
] = (
    robustness_comparison[
        "population_weighted_twfe_beta"
    ]
    -
    robustness_comparison[
        "main_twfe_beta"
    ]
)


robustness_comparison[
    "first_difference_minus_main"
] = (
    robustness_comparison[
        "first_difference_beta"
    ]
    -
    robustness_comparison[
        "main_twfe_beta"
    ]
)


# EXISTING filename retained.
robustness_comparison.to_csv(
    OUTPUT_DIR
    / "regional_slope_robustness_comparison.csv",
    index=False,
)


# ============================================================
# 26. SAMPLE SUMMARY
# ============================================================

sample_summary = pd.DataFrame(
    [
        {
            "specification":
                "Main two-way FE",

            "n_observations":
                len(
                    main_heterogeneity_sample
                ),

            "n_economies":
                main_heterogeneity_sample[
                    "iso3c"
                ]
                .nunique(),

            "first_year":
                main_heterogeneity_sample[
                    "year"
                ]
                .min(),

            "last_year":
                main_heterogeneity_sample[
                    "year"
                ]
                .max(),
        },

        {
            "specification":
                "Balanced-panel two-way FE",

            "n_observations":
                len(
                    balanced_sample
                ),

            "n_economies":
                balanced_sample[
                    "iso3c"
                ]
                .nunique(),

            "first_year":
                balanced_sample[
                    "year"
                ]
                .min(),

            "last_year":
                balanced_sample[
                    "year"
                ]
                .max(),
        },

        {
            "specification":
                "Population-weighted two-way FE",

            "n_observations":
                len(
                    weighted_sample
                ),

            "n_economies":
                weighted_sample[
                    "iso3c"
                ]
                .nunique(),

            "first_year":
                weighted_sample[
                    "year"
                ]
                .min(),

            "last_year":
                weighted_sample[
                    "year"
                ]
                .max(),
        },

        {
            "specification":
                "First differences",

            "n_observations":
                len(
                    difference_sample
                ),

            "n_economies":
                difference_sample[
                    "iso3c"
                ]
                .nunique(),

            "first_year":
                difference_sample[
                    "year"
                ]
                .min(),

            "last_year":
                difference_sample[
                    "year"
                ]
                .max(),
        },
    ]
)


# EXISTING filename retained.
sample_summary.to_csv(
    OUTPUT_DIR
    / "regional_heterogeneity_sample_summary.csv",
    index=False,
)


# ============================================================
# 27. MAIN COEFFICIENT FIGURE
# ============================================================

print_section(
    "CREATING REGIONAL ELASTICITY FIGURE"
)


plot_data = (
    main_slopes
    .set_index(
        "region"
    )
    .loc[
        EXPECTED_REGIONS
    ]
    .reset_index()
)


plot_labels = []


for _, row in plot_data.iterrows():

    n_economies = int(
        row[
            "n_economies"
        ]
    )


    if n_economies == 1:

        label = (
            f"{row['region_display']} "
            "(1 economy)"
        )

    else:

        label = (
            f"{row['region_display']} "
            f"({n_economies} economies)"
        )


    plot_labels.append(
        label
    )


y_positions = np.arange(
    len(
        plot_data
    )
)


lower_errors = (
    plot_data[
        "beta"
    ]
    -
    plot_data[
        "ci95_low"
    ]
)


upper_errors = (
    plot_data[
        "ci95_high"
    ]
    -
    plot_data[
        "beta"
    ]
)


fig, ax = plt.subplots(
    figsize=(
        10,
        6.4,
    )
)


ax.errorbar(
    plot_data[
        "beta"
    ],
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
    plot_labels
)


ax.invert_yaxis()


ax.set_xlabel(
    "GDP per capita elasticity of CO₂ emissions per capita"
)


ax.set_title(
    "Regional GDP–CO₂ Elasticities\n"
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
        "Bars show economy-clustered 95% confidence intervals. "
        "China and India are single-economy development groups; "
        "their intervals are shown for completeness but require "
        "additional inferential caution."
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


# EXISTING filename retained.
FIGURE_FILE = (
    FIGURES_DIR
    / "10_regional_twfe_elasticities.png"
)


fig.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved figure:\n{FIGURE_FILE}"
)


# ============================================================
# 28. CLEAN TERMINAL OUTPUT — JOINT TESTS
# ============================================================

print_section(
    "JOINT HETEROGENEITY TESTS"
)


print(
    joint_tests[
        [
            "specification",
            "test_scope",
            "f_statistic",
            "p_value",
            "df_num",
            "df_denom",
            "primary_inference",
            "reject_equal_slopes_at_5pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. CLEAN TERMINAL OUTPUT — MAIN MULTI-ECONOMY TEST
# ============================================================

print_section(
    "PREFERRED MAIN HETEROGENEITY TEST"
)


preferred_main_test = (
    joint_tests[
        (
            joint_tests[
                "specification"
            ]
            ==
            "Main two-way FE regional slopes"
        )
        &
        (
            joint_tests[
                "primary_inference"
            ]
            == True
        )
    ]
)


if len(
    preferred_main_test
) != 1:

    raise ValueError(
        "Could not uniquely identify the preferred main "
        "multi-economy heterogeneity test."
    )


preferred_main_test = (
    preferred_main_test.iloc[0]
)


print(
    "Regions tested:"
)

print(
    preferred_main_test[
        "regions_tested"
    ]
)


print(
    "\nF-statistic:",
    f"{preferred_main_test['f_statistic']:.4f}",
)


print(
    "p-value:",
    format_p_value(
        preferred_main_test[
            "p_value"
        ]
    ),
)


print(
    "Reject equal slopes at 5%:",
    preferred_main_test[
        "reject_equal_slopes_at_5pct"
    ],
)


# ============================================================
# 30. CLEAN TERMINAL OUTPUT — ROBUSTNESS
# ============================================================

print_section(
    "REGIONAL SLOPE ROBUSTNESS COMPARISON"
)


print(
    robustness_comparison[
        [
            "region_display",
            "main_twfe_beta",
            "balanced_twfe_beta",
            "population_weighted_twfe_beta",
            "first_difference_beta",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 31. SINGLE-ECONOMY GROUP WARNING
# ============================================================

print_section(
    "SINGLE-ECONOMY INFERENCE WARNING"
)


single_economy_regions = (
    main_slopes[
        main_slopes[
            "single_economy_group"
        ]
    ][
        "region_display"
    ]
    .tolist()
)


print(
    "The following development groups contain only one economy:"
)


for region_name in single_economy_regions:

    print(
        f"- {region_name}"
    )


print(
    "\nTheir slope point estimates remain economically useful."
)

print(
    "However, conventional economy-clustered p-values and "
    "confidence intervals for those group-specific slopes, "
    "and pairwise tests involving them, should not be used "
    "for strong inferential claims."
)

print(
    "\nThe preferred formal regional heterogeneity test is "
    "therefore the joint equality test across:"
)

print(
    "- Europe & North America"
)

print(
    "- Developed Asia & Oceania"
)

print(
    "- Global South"
)


# ============================================================
# 32. OUTPUT MANIFEST
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


print(
    f"Results saved to:\n{OUTPUT_DIR}"
)


print(
    "\nKey machine-readable files:"
)


# IMPORTANT:
# These are the SAME filenames used by the previous Script 10.
# Running this updated script cleanly overwrites them.

key_files = [
    "regional_heterogeneity_region_sample_summary.csv",
    "regional_heterogeneity_sample_summary.csv",
    "main_regional_twfe_slopes.csv",
    "regional_heterogeneity_slopes.csv",
    "regional_heterogeneity_joint_tests.csv",
    "main_regional_pairwise_slope_tests.csv",
    "regional_pairwise_slope_tests.csv",
    "common_vs_heterogeneous_model_comparison.csv",
    "regional_slope_robustness_comparison.csv",
]


for filename in key_files:

    print(
        f"- {filename}"
    )


print(
    "\nFigure:"
)

print(
    FIGURE_FILE
)


print(
    "\nFull Statsmodels summaries are stored locally under:"
)

print(
    SUMMARY_DIR
)


print(
    "\nSCRIPT 10 COMPLETE."
)
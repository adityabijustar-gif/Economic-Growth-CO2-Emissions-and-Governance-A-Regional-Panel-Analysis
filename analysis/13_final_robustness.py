from pathlib import Path

import matplotlib.pyplot as plt
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

SCRIPT09_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "country_panel_results"
    / "country_panel_fe_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "final_robustness"
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

EXPECTED_YEARS = set(
    range(
        START_YEAR,
        END_YEAR + 1,
    )
)

MIN_OBS_PER_ECONOMY = 2

ALPHA = 0.05

SCRIPT09_BETA_TOLERANCE = 1e-8


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


# ============================================================
# 3. GENERAL HELPERS
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
    Verify that required variables exist.
    """

    missing = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


def save_model_summary(
    model,
    filename,
):
    """
    Save a Statsmodels regression summary locally.
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


def check_design_rank(
    model,
    model_name,
):
    """
    Confirm that a regression design matrix is full rank.
    """

    design = (
        model.model.exog
    )

    rank = np.linalg.matrix_rank(
        design
    )

    n_columns = (
        design.shape[1]
    )

    if rank < n_columns:

        raise ValueError(
            f"{model_name} is rank deficient: "
            f"rank={rank}, columns={n_columns}."
        )


def retain_minimum_observations(
    df,
    minimum=MIN_OBS_PER_ECONOMY,
):
    """
    Retain economies with at least the requested number of
    observations in the sample being estimated.

    This prevents countries with only one remaining observation
    after a robustness restriction from entering a fixed-effects
    regression without within-economy information.
    """

    counts = (
        df
        .groupby(
            "iso3c"
        )[
            "year"
        ]
        .nunique()
    )

    eligible = set(
        counts[
            counts
            >= minimum
        ]
        .index
    )

    return (
        df[
            df[
                "iso3c"
            ]
            .isin(
                eligible
            )
        ]
        .copy()
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


def fit_economy_clustered_model(
    formula,
    data,
):
    """
    Estimate OLS with economy-clustered standard errors.

    This matches the principal inference approach used in
    Scripts 09-12.
    """

    if data.empty:

        raise ValueError(
            "Cannot estimate model on an empty sample."
        )

    n_economies = (
        data[
            "iso3c"
        ]
        .nunique()
    )

    if n_economies < 2:

        raise ValueError(
            "At least two economy clusters are required."
        )

    economy_groups = pd.Categorical(
        data[
            "iso3c"
        ]
    ).codes

    model = smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="cluster",

        cov_kwds={
            "groups":
                economy_groups,

            "use_correction":
                True,

            "df_correction":
                True,
        },

        use_t=True,
    )

    return (
        model
    )


def fit_two_way_clustered_model(
    formula,
    data,
):
    """
    Estimate OLS with two-way clustered covariance by:

    1. economy; and
    2. calendar year.

    The coefficient estimates are identical to the one-way
    clustered model because clustering affects covariance
    estimation, not OLS point estimates.

    This tests sensitivity of inference to residual dependence
    both within economies over time and across economies within
    the same year.
    """

    if data.empty:

        raise ValueError(
            "Cannot estimate model on an empty sample."
        )

    economy_codes = pd.Categorical(
        data[
            "iso3c"
        ]
    ).codes

    year_codes = pd.Categorical(
        data[
            "year"
        ]
    ).codes

    groups = np.column_stack(
        [
            economy_codes,
            year_codes,
        ]
    )

    model = smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="cluster",

        cov_kwds={
            "groups":
                groups,

            "use_correction":
                True,

            "df_correction":
                True,
        },

        use_t=True,
    )

    return (
        model
    )


def extract_term_result(
    model,
    data,
    term,
    specification,
    sample_type,
    covariance_description,
    notes="",
    excluded_region="",
):
    """
    Extract a coefficient and its inference information.
    """

    if term not in model.params.index:

        raise ValueError(
            f"Term '{term}' not found in "
            f"'{specification}'."
        )

    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )

    ci = (
        confidence
        .loc[
            term
        ]
    )

    return {
        "specification":
            specification,

        "sample_type":
            sample_type,

        "term":
            term,

        "coefficient":
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
                ci.iloc[0]
            ),

        "ci95_high":
            float(
                ci.iloc[1]
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

        "n_economies":
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

        "r_squared":
            float(
                model.rsquared
            ),

        "adjusted_r_squared":
            float(
                model.rsquared_adj
            ),

        "covariance":
            covariance_description,

        "excluded_region":
            excluded_region,

        "notes":
            notes,
    }


def sample_audit_row(
    data,
    specification,
    sample_type,
    notes="",
    excluded_region="",
):
    """
    Record the size and structure of one robustness sample.
    """

    counts = (
        data
        .groupby(
            "iso3c"
        )[
            "year"
        ]
        .nunique()
    )

    return {
        "specification":
            specification,

        "sample_type":
            sample_type,

        "n_obs":
            int(
                len(
                    data
                )
            ),

        "n_economies":
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

        "minimum_obs_per_economy":
            int(
                counts.min()
            ),

        "median_obs_per_economy":
            float(
                counts.median()
            ),

        "maximum_obs_per_economy":
            int(
                counts.max()
            ),

        "is_strictly_balanced":
            bool(
                counts.nunique()
                == 1
                and
                counts.iloc[0]
                ==
                data[
                    "year"
                ]
                .nunique()
            ),

        "excluded_region":
            excluded_region,

        "notes":
            notes,
    }


# ============================================================
# 4. LOAD SOURCE DATA
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
    "Clean WDI economy panel",
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
# 5. STANDARDISE SOURCE DATA
# ============================================================

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
# 6. SOURCE VALIDATION
# ============================================================

print_section(
    "VALIDATING SOURCE DATA"
)


if panel.duplicated(
    [
        "iso3c",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate economy-year observations found."
    )


if region_map.duplicated(
    "iso3c"
).any():

    raise ValueError(
        "Duplicate economy-region assignments found."
    )


if set(
    region_map[
        "region"
    ]
    .unique()
) != set(
    EXPECTED_REGIONS
):

    raise ValueError(
        "Region mapping does not contain exactly the expected "
        "five development clusters."
    )


print(
    "PASS: source keys and regional labels are valid."
)


# ============================================================
# 7. RECONSTRUCT SCRIPT 09 MAIN SAMPLE
# ============================================================

print_section(
    "RECONSTRUCTING MAIN 1990-2024 SAMPLE"
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


main_sample = (
    retain_minimum_observations(
        valid_panel,
        minimum=MIN_OBS_PER_ECONOMY,
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


main_sample[
    "region_display"
] = (
    main_sample[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


print(
    f"Main observations: "
    f"{len(main_sample):,}"
)

print(
    f"Main economies: "
    f"{main_sample['iso3c'].nunique():,}"
)

print(
    f"Years: "
    f"{main_sample['year'].min()}-"
    f"{main_sample['year'].max()}"
)


# ============================================================
# 8. GLOBAL BASELINE MODEL
# ============================================================

GLOBAL_FORMULA = (
    "log_co2 ~ "
    "log_gdp + "
    "C(iso3c) + "
    "C(year)"
)


baseline_model = (
    fit_economy_clustered_model(
        formula=GLOBAL_FORMULA,
        data=main_sample,
    )
)


check_design_rank(
    baseline_model,
    "Baseline global TWFE model",
)


save_model_summary(
    baseline_model,
    "baseline_global_twfe.txt",
)


baseline_beta = float(
    baseline_model.params[
        "log_gdp"
    ]
)


print(
    "Baseline GDP elasticity: "
    f"{baseline_beta:.6f}"
)


# ============================================================
# 9. CROSS-CHECK AGAINST SCRIPT 09
# ============================================================

print_section(
    "SCRIPT 09 CONSISTENCY CHECK"
)


if SCRIPT09_RESULTS_FILE.exists():

    script09_results = pd.read_csv(
        SCRIPT09_RESULTS_FILE
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
            "Could not uniquely identify the Script 09 main "
            "country + year fixed-effects result."
        )


    script09_main = (
        script09_main
        .iloc[0]
    )


    beta_column = None


    for candidate in [
        "beta",
        "coefficient",
    ]:

        if candidate in script09_main.index:

            beta_column = (
                candidate
            )

            break


    if beta_column is None:

        raise ValueError(
            "Script 09 result file does not contain a beta or "
            "coefficient column."
        )


    script09_beta = float(
        script09_main[
            beta_column
        ]
    )


    if not np.isclose(
        baseline_beta,
        script09_beta,
        atol=SCRIPT09_BETA_TOLERANCE,
        rtol=0,
    ):

        raise ValueError(
            "Script 13 baseline coefficient does not reproduce "
            "Script 09. "
            f"Script 13={baseline_beta:.10f}, "
            f"Script 09={script09_beta:.10f}"
        )


    if (
        "n_obs"
        in script09_main.index
        and
        int(
            baseline_model.nobs
        )
        !=
        int(
            script09_main[
                "n_obs"
            ]
        )
    ):

        raise ValueError(
            "Script 13 baseline N does not reproduce Script 09."
        )


    print(
        "PASS: Script 13 baseline reproduces Script 09."
    )


else:

    print(
        "WARNING: Script 09 result file not found. "
        "Cross-check skipped."
    )


# ============================================================
# 10. CONSTRUCT FINAL ROBUSTNESS SAMPLES
# ============================================================

print_section(
    "CONSTRUCTING ROBUSTNESS SAMPLES"
)


robustness_samples = {}


# ------------------------------------------------------------
# A. Baseline 1990-2024
# ------------------------------------------------------------

robustness_samples[
    "Baseline 1990-2024"
] = {
    "data":
        main_sample.copy(),

    "sample_type":
        "Baseline",

    "notes":
        (
            "Main Script 09 economy-year sample."
        ),

    "excluded_region":
        "",
}


# ------------------------------------------------------------
# B. End-year sensitivity: 2022 and 2023
# ------------------------------------------------------------

for end_year in [
    2022,
    2023,
]:

    sample = (
        main_sample[
            main_sample[
                "year"
            ]
            <= end_year
        ]
        .copy()
    )


    sample = (
        retain_minimum_observations(
            sample
        )
    )


    robustness_samples[
        f"End year {end_year}"
    ] = {
        "data":
            sample,

        "sample_type":
            "End-year sensitivity",

        "notes":
            (
                f"Baseline sample truncated at {end_year}."
            ),

        "excluded_region":
            "",
    }


# ------------------------------------------------------------
# C. Exclude COVID disruption and immediate rebound
# ------------------------------------------------------------

covid_excluded_sample = (
    main_sample[
        ~main_sample[
            "year"
        ]
        .isin(
            [
                2020,
                2021,
            ]
        )
    ]
    .copy()
)


covid_excluded_sample = (
    retain_minimum_observations(
        covid_excluded_sample
    )
)


robustness_samples[
    "Exclude 2020-2021"
] = {
    "data":
        covid_excluded_sample,

    "sample_type":
        "COVID-period exclusion",

    "notes":
        (
            "Excludes 2020 and 2021 to reduce influence of the "
            "pandemic contraction and immediate rebound."
        ),

    "excluded_region":
        "",
}


# ------------------------------------------------------------
# D. Strict balanced panel, 1990-2024
# ------------------------------------------------------------

balance_status = (
    main_sample
    .groupby(
        "iso3c"
    )[
        "year"
    ]
    .apply(
        lambda series:
            (
                set(
                    int(
                        year
                    )
                    for year
                    in series
                )
                ==
                EXPECTED_YEARS
            )
    )
)


balanced_codes = set(
    balance_status[
        balance_status
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


if balanced_sample.empty:

    raise ValueError(
        "Strict 1990-2024 balanced panel is empty."
    )


expected_balanced_rows = (
    balanced_sample[
        "iso3c"
    ]
    .nunique()
    *
    len(
        EXPECTED_YEARS
    )
)


if len(
    balanced_sample
) != expected_balanced_rows:

    raise ValueError(
        "Balanced-panel construction failed."
    )


robustness_samples[
    "Strict balanced panel"
] = {
    "data":
        balanced_sample,

    "sample_type":
        "Balanced-panel robustness",

    "notes":
        (
            "Economies must have valid positive GDP and CO2 "
            "observations in every year from 1990 through 2024."
        ),

    "excluded_region":
        "",
}


print(
    "Strict balanced panel: "
    f"{balanced_sample['iso3c'].nunique():,} economies, "
    f"{len(balanced_sample):,} observations."
)


# ------------------------------------------------------------
# E. Leave one development cluster out
# ------------------------------------------------------------

for region in EXPECTED_REGIONS:

    sample = (
        main_sample[
            main_sample[
                "region"
            ]
            != region
        ]
        .copy()
    )


    sample = (
        retain_minimum_observations(
            sample
        )
    )


    display_name = (
        REGION_DISPLAY_NAMES[
            region
        ]
    )


    robustness_samples[
        f"Exclude {display_name}"
    ] = {
        "data":
            sample,

        "sample_type":
            "Leave-one-cluster-out",

        "notes":
            (
                "Tests whether the global elasticity is "
                "disproportionately driven by one development "
                "cluster."
            ),

        "excluded_region":
            region,
    }


# ============================================================
# 11. SAMPLE AUDIT
# ============================================================

sample_audit_rows = []


for (
    specification,
    details,
) in robustness_samples.items():

    sample_audit_rows.append(
        sample_audit_row(
            data=
                details[
                    "data"
                ],

            specification=
                specification,

            sample_type=
                details[
                    "sample_type"
                ],

            notes=
                details[
                    "notes"
                ],

            excluded_region=
                details[
                    "excluded_region"
                ],
        )
    )


sample_audit = (
    pd.DataFrame(
        sample_audit_rows
    )
)


sample_audit.to_csv(
    OUTPUT_DIR
    / "sample_robustness_audit.csv",
    index=False,
)


print(
    sample_audit[
        [
            "specification",
            "n_obs",
            "n_economies",
            "n_years",
            "is_strictly_balanced",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 12. GLOBAL TWFE ROBUSTNESS
# ============================================================

print_section(
    "ESTIMATING GLOBAL TWFE ROBUSTNESS MODELS"
)


global_result_rows = []


for (
    specification,
    details,
) in robustness_samples.items():

    sample = (
        details[
            "data"
        ]
    )


    model = (
        fit_economy_clustered_model(
            formula=GLOBAL_FORMULA,
            data=sample,
        )
    )


    check_design_rank(
        model,
        specification,
    )


    safe_name = (
        specification
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "&",
            "and",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "–",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


    save_model_summary(
        model,
        f"global_{safe_name}.txt",
    )


    global_result_rows.append(
        extract_term_result(
            model=model,
            data=sample,
            term="log_gdp",
            specification=specification,
            sample_type=
                details[
                    "sample_type"
                ],
            covariance_description=
                "Economy-clustered standard errors",
            notes=
                details[
                    "notes"
                ],
            excluded_region=
                details[
                    "excluded_region"
                ],
        )
    )


    print(
        f"{specification}: "
        f"beta={model.params['log_gdp']:.4f}, "
        f"N={int(model.nobs):,}, "
        f"economies={sample['iso3c'].nunique():,}"
    )


# ============================================================
# 13. TWO-WAY CLUSTERING ROBUSTNESS
# ============================================================

print_section(
    "TWO-WAY CLUSTERING ROBUSTNESS"
)


two_way_model = (
    fit_two_way_clustered_model(
        formula=GLOBAL_FORMULA,
        data=main_sample,
    )
)


check_design_rank(
    two_way_model,
    "Baseline two-way clustered model",
)


save_model_summary(
    two_way_model,
    "baseline_global_twfe_two_way_clustered.txt",
)


two_way_result = (
    extract_term_result(
        model=two_way_model,
        data=main_sample,
        term="log_gdp",
        specification=
            "Baseline two-way clustering",
        sample_type=
            "Inference robustness",
        covariance_description=
            "Two-way clustered by economy and year",
        notes=
            (
                "Identical OLS coefficient to baseline; "
                "covariance allows residual dependence within "
                "economies and within calendar years."
            ),
    )
)


global_result_rows.append(
    two_way_result
)


if not np.isclose(
    two_way_result[
        "coefficient"
    ],
    baseline_beta,
    atol=1e-10,
    rtol=0,
):

    raise ValueError(
        "Two-way clustered model changed the OLS coefficient. "
        "This should not occur on the same estimation sample."
    )


print(
    "Economy-clustered baseline:"
)

print(
    f"  beta = "
    f"{baseline_model.params['log_gdp']:.6f}"
)

print(
    f"  SE   = "
    f"{baseline_model.bse['log_gdp']:.6f}"
)

print(
    f"  p    = "
    f"{baseline_model.pvalues['log_gdp']:.6g}"
)


print(
    "\nTwo-way economy/year clustered:"
)

print(
    f"  beta = "
    f"{two_way_model.params['log_gdp']:.6f}"
)

print(
    f"  SE   = "
    f"{two_way_model.bse['log_gdp']:.6f}"
)

print(
    f"  p    = "
    f"{two_way_model.pvalues['log_gdp']:.6g}"
)


# ============================================================
# 14. EXPORT GLOBAL ROBUSTNESS TABLE
# ============================================================

global_robustness = (
    pd.DataFrame(
        global_result_rows
    )
)


global_robustness.to_csv(
    OUTPUT_DIR
    / "global_twfe_robustness.csv",
    index=False,
)


# ============================================================
# 15. REGIONAL HETEROGENEITY HELPERS
# ============================================================

def prepare_region_slope_sample(
    data,
):
    """
    Create region-specific GDP slope regressors.

    The model contains no common log-GDP slope. Instead, each
    development cluster receives its own log-GDP coefficient.

    Economy and year fixed effects remain included.
    """

    prepared = (
        data
        .copy()
    )


    slope_columns = {}


    for region in EXPECTED_REGIONS:

        if region not in set(
            prepared[
                "region"
            ]
            .unique()
        ):

            raise ValueError(
                f"Region missing from regional robustness "
                f"sample: {region}"
            )


        column = (
            f"region_slope_{region}"
        )


        prepared[
            column
        ] = np.where(
            prepared[
                "region"
            ]
            ==
            region,

            prepared[
                "log_gdp"
            ],

            0.0,
        )


        slope_columns[
            region
        ] = (
            column
        )


    return (
        prepared,
        slope_columns,
    )


def estimate_regional_slopes(
    data,
    specification,
):
    """
    Estimate Script-10-style regional GDP elasticities with
    economy and year fixed effects and economy-clustered
    standard errors.
    """

    (
        prepared,
        slope_columns,
    ) = prepare_region_slope_sample(
        data
    )


    slope_formula = (
        " + "
        .join(
            slope_columns.values()
        )
    )


    formula = (
        "log_co2 ~ "
        f"{slope_formula} + "
        "C(iso3c) + C(year)"
    )


    model = (
        fit_economy_clustered_model(
            formula=formula,
            data=prepared,
        )
    )


    check_design_rank(
        model,
        (
            "Regional heterogeneity: "
            f"{specification}"
        ),
    )


    safe_name = (
        specification
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


    save_model_summary(
        model,
        (
            f"regional_heterogeneity_"
            f"{safe_name}.txt"
        ),
    )


    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    rows = []


    for region in EXPECTED_REGIONS:

        term = (
            slope_columns[
                region
            ]
        )


        region_sample = (
            prepared[
                prepared[
                    "region"
                ]
                ==
                region
            ]
        )


        n_region_economies = (
            region_sample[
                "iso3c"
            ]
            .nunique()
        )


        ci = (
            confidence
            .loc[
                term
            ]
        )


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

                "term":
                    term,

                "beta_log_gdp":
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
                        ci.iloc[0]
                    ),

                "ci95_high":
                    float(
                        ci.iloc[1]
                    ),

                "p_value":
                    float(
                        model.pvalues[
                            term
                        ]
                    ),

                "n_total_obs_model":
                    int(
                        model.nobs
                    ),

                "n_total_economies_model":
                    int(
                        prepared[
                            "iso3c"
                        ]
                        .nunique()
                    ),

                "n_region_obs":
                    int(
                        len(
                            region_sample
                        )
                    ),

                "n_region_economies":
                    int(
                        n_region_economies
                    ),

                "inferential_caution":
                    (
                        "Single-economy development group; "
                        "clustered inference for this regional "
                        "slope requires additional caution."
                        if n_region_economies
                        == 1
                        else ""
                    ),
            }
        )


    return (
        rows
    )


# ============================================================
# 16. REGIONAL HETEROGENEITY ROBUSTNESS
# ============================================================

print_section(
    "ESTIMATING REGIONAL HETEROGENEITY ROBUSTNESS"
)


regional_sample_definitions = {
    "Baseline 1990-2024":
        main_sample,

    "End year 2022":
        robustness_samples[
            "End year 2022"
        ][
            "data"
        ],

    "End year 2023":
        robustness_samples[
            "End year 2023"
        ][
            "data"
        ],

    "Exclude 2020-2021":
        robustness_samples[
            "Exclude 2020-2021"
        ][
            "data"
        ],

    "Strict balanced panel":
        balanced_sample,
}


regional_result_rows = []


for (
    specification,
    sample,
) in regional_sample_definitions.items():

    regional_result_rows.extend(
        estimate_regional_slopes(
            data=sample,
            specification=specification,
        )
    )


regional_robustness = (
    pd.DataFrame(
        regional_result_rows
    )
)


regional_robustness.to_csv(
    OUTPUT_DIR
    / "regional_heterogeneity_robustness.csv",
    index=False,
)


print(
    regional_robustness[
        [
            "specification",
            "region_display",
            "beta_log_gdp",
            "std_error",
            "p_value",
            "n_region_economies",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. FINAL CORE RESULTS SUMMARY
# ============================================================

print_section(
    "BUILDING FINAL CORE RESULTS SUMMARY"
)


summary_rows = []


# ------------------------------------------------------------
# Global result stability
#
# Exclude two-way clustering from coefficient-range calculation
# because it uses the identical baseline sample and coefficient;
# it is an inference robustness test rather than a sample test.
# ------------------------------------------------------------

global_sample_results = (
    global_robustness[
        global_robustness[
            "sample_type"
        ]
        !=
        "Inference robustness"
    ]
    .copy()
)


baseline_global_row = (
    global_robustness[
        global_robustness[
            "specification"
        ]
        ==
        "Baseline 1990-2024"
    ]
)


if len(
    baseline_global_row
) != 1:

    raise ValueError(
        "Could not identify unique baseline global result."
    )


baseline_global_row = (
    baseline_global_row
    .iloc[0]
)


summary_rows.append(
    {
        "result_type":
            "Global GDP-CO2 elasticity",

        "scope":
            "All economies",

        "baseline_beta":
            float(
                baseline_global_row[
                    "coefficient"
                ]
            ),

        "baseline_ci95_low":
            float(
                baseline_global_row[
                    "ci95_low"
                ]
            ),

        "baseline_ci95_high":
            float(
                baseline_global_row[
                    "ci95_high"
                ]
            ),

        "baseline_p_value":
            float(
                baseline_global_row[
                    "p_value"
                ]
            ),

        "robustness_min_beta":
            float(
                global_sample_results[
                    "coefficient"
                ]
                .min()
            ),

        "robustness_max_beta":
            float(
                global_sample_results[
                    "coefficient"
                ]
                .max()
            ),

        "all_robustness_coefficients_positive":
            bool(
                (
                    global_sample_results[
                        "coefficient"
                    ]
                    > 0
                )
                .all()
            ),

        "all_robustness_p_below_0_05":
            bool(
                (
                    global_sample_results[
                        "p_value"
                    ]
                    < 0.05
                )
                .all()
            ),

        "n_robustness_specifications":
            int(
                len(
                    global_sample_results
                )
            ),

        "notes":
            (
                "Range covers end-year, balanced-panel, "
                "COVID-exclusion and leave-one-cluster-out "
                "sample checks."
            ),
    }
)


# ------------------------------------------------------------
# Regional result stability
# ------------------------------------------------------------

for region in EXPECTED_REGIONS:

    region_results = (
        regional_robustness[
            regional_robustness[
                "region"
            ]
            ==
            region
        ]
        .copy()
    )


    baseline_region = (
        region_results[
            region_results[
                "specification"
            ]
            ==
            "Baseline 1990-2024"
        ]
    )


    if len(
        baseline_region
    ) != 1:

        raise ValueError(
            f"Could not identify baseline regional result "
            f"for {region}."
        )


    baseline_region = (
        baseline_region
        .iloc[0]
    )


    summary_rows.append(
        {
            "result_type":
                "Regional GDP-CO2 elasticity",

            "scope":
                REGION_DISPLAY_NAMES[
                    region
                ],

            "baseline_beta":
                float(
                    baseline_region[
                        "beta_log_gdp"
                    ]
                ),

            "baseline_ci95_low":
                float(
                    baseline_region[
                        "ci95_low"
                    ]
                ),

            "baseline_ci95_high":
                float(
                    baseline_region[
                        "ci95_high"
                    ]
                ),

            "baseline_p_value":
                float(
                    baseline_region[
                        "p_value"
                    ]
                ),

            "robustness_min_beta":
                float(
                    region_results[
                        "beta_log_gdp"
                    ]
                    .min()
                ),

            "robustness_max_beta":
                float(
                    region_results[
                        "beta_log_gdp"
                    ]
                    .max()
                ),

            "all_robustness_coefficients_positive":
                bool(
                    (
                        region_results[
                            "beta_log_gdp"
                        ]
                        > 0
                    )
                    .all()
                ),

            "all_robustness_p_below_0_05":
                bool(
                    (
                        region_results[
                            "p_value"
                        ]
                        < 0.05
                    )
                    .all()
                ),

            "n_robustness_specifications":
                int(
                    len(
                        region_results
                    )
                ),

            "notes":
                (
                    "China and India are single-economy "
                    "development groups; regional clustered "
                    "inference requires additional caution."
                    if region
                    in [
                        "China",
                        "India",
                    ]
                    else
                    (
                        "Regional slope stability across "
                        "end-year, balanced-panel and "
                        "COVID-exclusion checks."
                    )
                ),
        }
    )


final_core_summary = (
    pd.DataFrame(
        summary_rows
    )
)


final_core_summary.to_csv(
    OUTPUT_DIR
    / "final_core_results_summary.csv",
    index=False,
)


print(
    final_core_summary[
        [
            "result_type",
            "scope",
            "baseline_beta",
            "robustness_min_beta",
            "robustness_max_beta",
            "all_robustness_coefficients_positive",
            "all_robustness_p_below_0_05",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. GLOBAL ROBUSTNESS FOREST PLOT
# ============================================================

print_section(
    "CREATING FINAL ROBUSTNESS FIGURE"
)


plot_order = [
    "Baseline 1990-2024",
    "Baseline two-way clustering",
    "End year 2022",
    "End year 2023",
    "Exclude 2020-2021",
    "Strict balanced panel",
    "Exclude Europe & North America",
    "Exclude Developed Asia & Oceania",
    "Exclude China",
    "Exclude India",
    "Exclude Global South",
]


plot_data = (
    global_robustness
    .set_index(
        "specification"
    )
    .loc[
        plot_order
    ]
    .reset_index()
)


y_positions = np.arange(
    len(
        plot_data
    )
)


coefficients = (
    plot_data[
        "coefficient"
    ]
    .to_numpy()
)


lower_errors = (
    coefficients
    -
    plot_data[
        "ci95_low"
    ]
    .to_numpy()
)


upper_errors = (
    plot_data[
        "ci95_high"
    ]
    .to_numpy()
    -
    coefficients
)


fig, ax = plt.subplots(
    figsize=(
        10,
        7.5,
    )
)


ax.errorbar(
    coefficients,
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
    plot_data[
        "specification"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    "GDP per capita elasticity of CO2 emissions per capita"
)


ax.set_title(
    "Global GDP–CO2 Elasticity: Final Robustness Checks\n"
    "Economy and Year Fixed Effects"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "Bars show 95% confidence intervals. The baseline and "
        "sample-robustness models use economy-clustered inference; "
        "the two-way specification clusters by both economy and year."
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


ROBUSTNESS_FIGURE_FILE = (
    FIGURES_DIR
    / "13_global_gdp_elasticity_robustness.png"
)


fig.savefig(
    ROBUSTNESS_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved:\n{ROBUSTNESS_FIGURE_FILE}"
)


# ============================================================
# 19. FINAL VALIDATION
# ============================================================

print_section(
    "FINAL VALIDATION"
)


EXPECTED_GLOBAL_RESULT_ROWS = 11


if len(
    global_robustness
) != EXPECTED_GLOBAL_RESULT_ROWS:

    raise ValueError(
        "Unexpected number of global robustness results. "
        f"Expected {EXPECTED_GLOBAL_RESULT_ROWS}, "
        f"found {len(global_robustness)}."
    )


EXPECTED_REGIONAL_RESULT_ROWS = (
    len(
        EXPECTED_REGIONS
    )
    *
    len(
        regional_sample_definitions
    )
)


if len(
    regional_robustness
) != EXPECTED_REGIONAL_RESULT_ROWS:

    raise ValueError(
        "Unexpected number of regional robustness rows. "
        f"Expected {EXPECTED_REGIONAL_RESULT_ROWS}, "
        f"found {len(regional_robustness)}."
    )


if len(
    final_core_summary
) != (
    1
    +
    len(
        EXPECTED_REGIONS
    )
):

    raise ValueError(
        "Final core-results summary has an unexpected number "
        "of rows."
    )


if not np.isclose(
    global_robustness.loc[
        global_robustness[
            "specification"
        ]
        ==
        "Baseline 1990-2024",
        "coefficient",
    ]
    .iloc[0],

    global_robustness.loc[
        global_robustness[
            "specification"
        ]
        ==
        "Baseline two-way clustering",
        "coefficient",
    ]
    .iloc[0],

    atol=1e-10,
    rtol=0,
):

    raise ValueError(
        "Baseline coefficient differs under two-way clustering."
    )


if (
    balanced_sample[
        "iso3c"
    ]
    .nunique()
    *
    len(
        EXPECTED_YEARS
    )
    !=
    len(
        balanced_sample
    )
):

    raise ValueError(
        "Strict balanced sample failed final validation."
    )


if (
    set(
        regional_robustness[
            "region"
        ]
        .unique()
    )
    !=
    set(
        EXPECTED_REGIONS
    )
):

    raise ValueError(
        "Regional robustness output does not contain all five "
        "development clusters."
    )


print(
    "PASS: Script 13 baseline reproduces the established "
    "economy-level specification."
)

print(
    "PASS: end-year sensitivity models completed."
)

print(
    "PASS: strict balanced-panel robustness completed."
)

print(
    "PASS: 2020-2021 exclusion robustness completed."
)

print(
    "PASS: all five leave-one-cluster-out tests completed."
)

print(
    "PASS: two-way economy/year clustering completed."
)

print(
    "PASS: regional heterogeneity robustness completed."
)

print(
    f"PASS: global robustness table contains "
    f"{len(global_robustness)} expected results."
)

print(
    f"PASS: regional robustness table contains "
    f"{len(regional_robustness)} expected results."
)


# ============================================================
# 20. OUTPUT MANIFEST
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


print(
    f"Results saved to:\n{OUTPUT_DIR}"
)


print(
    "\nMachine-readable outputs:"
)


output_files = [
    "global_twfe_robustness.csv",
    "regional_heterogeneity_robustness.csv",
    "sample_robustness_audit.csv",
    "final_core_results_summary.csv",
]


for filename in output_files:

    print(
        f"- {filename}"
    )


print(
    "\nFigure:"
)

print(
    f"- {ROBUSTNESS_FIGURE_FILE}"
)


print(
    "\nFull Statsmodels summaries are stored locally under:"
)

print(
    SUMMARY_DIR
)


print(
    "\nSCRIPT 13 COMPLETE."
)
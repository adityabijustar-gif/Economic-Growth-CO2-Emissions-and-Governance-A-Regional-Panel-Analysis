from pathlib import Path

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

GOVERNANCE_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "worldbank_wdi_download"
    / "P_Data_Extract_From_World_Development_Indicators 2"
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
    / "governance_results"
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

ANNUAL_GOV_START_YEAR = 2002
ANNUAL_GOV_END_YEAR = 2024

ANNUAL_GOV_YEARS = set(
    range(
        ANNUAL_GOV_START_YEAR,
        ANNUAL_GOV_END_YEAR + 1,
    )
)

MIN_GOV_OBS_PER_ECONOMY = 2

ALPHA = 0.05
SIGNIFICANCE_LEVEL = 0.05


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


# WGI-derived governance scores accessed through WDI.
GOVERNANCE_INDICATORS = {
    "government_effectiveness": {
        "series_code":
            "GOV_WGI_GE_SC",

        "display_name":
            "Government Effectiveness",

        "column":
            "gov_effectiveness_score",
    },

    "regulatory_quality": {
        "series_code":
            "GOV_WGI_RQ_SC",

        "display_name":
            "Regulatory Quality",

        "column":
            "regulatory_quality_score",
    },

    "rule_of_law": {
        "series_code":
            "GOV_WGI_RL_SC",

        "display_name":
            "Rule of Law",

        "column":
            "rule_of_law_score",
    },
}


EXPECTED_GOV_CODES = {
    details[
        "series_code"
    ]
    for details
    in GOVERNANCE_INDICATORS.values()
}


GOVERNANCE_SCORE_COLUMNS = [
    details[
        "column"
    ]
    for details
    in GOVERNANCE_INDICATORS.values()
]


# ============================================================
# 3. GENERAL HELPER FUNCTIONS
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
    Verify that all required columns are present.
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


def scalar(value):
    """
    Convert a Statsmodels scalar or one-element array to float.
    """

    return float(
        np.asarray(
            value
        )
        .squeeze()
    )


def format_p_value(
    p_value,
):
    """
    Human-readable p-value formatting.
    """

    if pd.isna(
        p_value
    ):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.4f}"


# ============================================================
# 4. GOVERNANCE FILE DISCOVERY
# ============================================================

def find_governance_data_file(
    raw_dir,
):
    """
    Identify the WDI CSV containing all three required
    governance indicators.

    Identification uses indicator codes rather than assuming a
    particular local filename.
    """

    if not raw_dir.exists():

        raise FileNotFoundError(
            "Governance raw-data directory does not exist:\n"
            f"{raw_dir}"
        )


    candidate_files = sorted(
        raw_dir.glob(
            "*.csv"
        )
    )


    if not candidate_files:

        raise FileNotFoundError(
            "No CSV files were found under:\n"
            f"{raw_dir}"
        )


    matches = []


    for candidate in candidate_files:

        try:

            codes = pd.read_csv(
                candidate,
                usecols=[
                    "Series Code",
                ],
            )

        except Exception:

            continue


        observed_codes = set(
            codes[
                "Series Code"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )


        if EXPECTED_GOV_CODES.issubset(
            observed_codes
        ):

            matches.append(
                candidate
            )


    if len(
        matches
    ) != 1:

        raise ValueError(
            "Could not uniquely identify the governance "
            "WDI data CSV. Matching files: "
            f"{[str(path) for path in matches]}"
        )


    return matches[0]


def parse_year_columns(
    columns,
):
    """
    Convert WDI year labels such as:

        2002 [YR2002]

    to integer years.
    """

    year_map = {}


    for column in columns:

        column_string = str(
            column
        )


        if "[YR" not in column_string:
            continue


        year_text = (
            column_string
            .split(
                " [YR",
                1,
            )[0]
        )


        try:

            year = int(
                year_text
            )

        except ValueError:

            continue


        year_map[
            column
        ] = (
            year
        )


    if not year_map:

        raise ValueError(
            "No WDI year columns were detected."
        )


    return year_map


# ============================================================
# 5. REGRESSION HELPERS
# ============================================================

def fit_clustered_model(
    formula,
    data,
):
    """
    Estimate OLS with standard errors clustered at economy
    level.

    Uses:
    - finite-sample covariance correction;
    - degrees-of-freedom correction;
    - Student-t inference.
    """

    if data.empty:

        raise ValueError(
            "Cannot estimate a model on an empty dataset."
        )


    if (
        data[
            "iso3c"
        ]
        .nunique()
        < 2
    ):

        raise ValueError(
            "At least two economy clusters are required."
        )


    cluster_groups = pd.Categorical(
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
    Confirm that a model design matrix is full rank.
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


def save_model_summary(
    model,
    filename,
):
    """
    Save full Statsmodels output locally.
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


# ============================================================
# 6. SAMPLE-CONSTRUCTION HELPERS
# ============================================================

def restrict_min_governance_observations(
    df,
    score_columns,
):
    """
    Retain economies with at least the required number of
    complete governance observations.
    """

    complete = (
        df
        .dropna(
            subset=score_columns
        )
        .copy()
    )


    counts = (
        complete
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
            >= MIN_GOV_OBS_PER_ECONOMY
        ]
        .index
    )


    return (
        complete[
            complete[
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


def prepare_centered_sample(
    df,
    governance_column,
):
    """
    Center GDP and governance within the estimation sample.

    Governance is expressed in 10-point units:

        gov_c10 =
            (governance - sample mean) / 10

    Thus the GDP × governance coefficient measures the change
    in GDP-emissions elasticity associated with a 10-point
    higher governance score.
    """

    if df.empty:

        raise ValueError(
            "Cannot prepare an empty governance sample."
        )


    prepared = (
        df
        .copy()
    )


    log_gdp_mean = float(
        prepared[
            "log_gdp"
        ]
        .mean()
    )


    governance_mean = float(
        prepared[
            governance_column
        ]
        .mean()
    )


    prepared[
        "log_gdp_c"
    ] = (
        prepared[
            "log_gdp"
        ]
        -
        log_gdp_mean
    )


    prepared[
        "log_gdp_c_sq"
    ] = (
        prepared[
            "log_gdp_c"
        ]
        ** 2
    )


    prepared[
        "gov_c10"
    ] = (
        prepared[
            governance_column
        ]
        -
        governance_mean
    ) / 10.0


    prepared[
        "gdp_x_gov"
    ] = (
        prepared[
            "log_gdp_c"
        ]
        *
        prepared[
            "gov_c10"
        ]
    )


    centering = {
        "log_gdp_mean":
            log_gdp_mean,

        "governance_mean":
            governance_mean,

        "governance_sd":
            float(
                prepared[
                    governance_column
                ]
                .std()
            ),

        "governance_p25":
            float(
                prepared[
                    governance_column
                ]
                .quantile(
                    0.25
                )
            ),

        "governance_p50":
            float(
                prepared[
                    governance_column
                ]
                .quantile(
                    0.50
                )
            ),

        "governance_p75":
            float(
                prepared[
                    governance_column
                ]
                .quantile(
                    0.75
                )
            ),
    }


    return (
        prepared,
        centering,
    )


def create_region_slope_variables(
    df,
    base_variable="log_gdp_c",
):
    """
    Create five development-cluster-specific GDP slopes.

    These replicate the core regional-slope idea from Script 10
    when testing whether governance adds explanatory power
    beyond regional development heterogeneity.
    """

    data = (
        df
        .copy()
    )


    slope_columns = {}


    for region in EXPECTED_REGIONS:

        column_name = (
            f"region_slope_{region}"
        )


        data[
            column_name
        ] = np.where(
            data[
                "region"
            ]
            .eq(
                region
            ),

            data[
                base_variable
            ],

            0.0,
        )


        slope_columns[
            region
        ] = (
            column_name
        )


    return (
        data,
        slope_columns,
    )


def prepare_between_within_governance(
    df,
    governance_column,
):
    """
    Decompose governance into persistent between-economy and
    within-economy components.

    The persistent economy-average governance level is absorbed
    by economy fixed effects as a standalone level, but its
    interaction with time-varying GDP remains identifiable.
    """

    (
        prepared,
        centering,
    ) = prepare_centered_sample(
        df,
        governance_column,
    )


    prepared[
        "gov_economy_mean"
    ] = (
        prepared
        .groupby(
            "iso3c"
        )[
            governance_column
        ]
        .transform(
            "mean"
        )
    )


    prepared[
        "gov_between_c10"
    ] = (
        prepared[
            "gov_economy_mean"
        ]
        -
        centering[
            "governance_mean"
        ]
    ) / 10.0


    prepared[
        "gov_within_c10"
    ] = (
        prepared[
            governance_column
        ]
        -
        prepared[
            "gov_economy_mean"
        ]
    ) / 10.0


    prepared[
        "gdp_x_gov_between"
    ] = (
        prepared[
            "log_gdp_c"
        ]
        *
        prepared[
            "gov_between_c10"
        ]
    )


    prepared[
        "gdp_x_gov_within"
    ] = (
        prepared[
            "log_gdp_c"
        ]
        *
        prepared[
            "gov_within_c10"
        ]
    )


    return (
        prepared,
        centering,
    )


def build_first_difference_sample(
    prepared,
):
    """
    First-difference the centered governance-interaction model.

    Only genuinely consecutive calendar years are retained.

    The differenced interaction is:

        Delta(log_gdp_c * gov_c10)

    rather than:

        Delta(log GDP) * Delta(governance).
    """

    source = (
        prepared
        .sort_values(
            [
                "iso3c",
                "year",
            ]
        )
        .copy()
    )


    source[
        "previous_year"
    ] = (
        source
        .groupby(
            "iso3c"
        )[
            "year"
        ]
        .shift(1)
    )


    source[
        "year_gap"
    ] = (
        source[
            "year"
        ]
        -
        source[
            "previous_year"
        ]
    )


    source[
        "d_log_co2"
    ] = (
        source
        .groupby(
            "iso3c"
        )[
            "log_co2"
        ]
        .diff()
    )


    source[
        "d_log_gdp"
    ] = (
        source
        .groupby(
            "iso3c"
        )[
            "log_gdp_c"
        ]
        .diff()
    )


    source[
        "d_gov_c10"
    ] = (
        source
        .groupby(
            "iso3c"
        )[
            "gov_c10"
        ]
        .diff()
    )


    source[
        "d_gdp_x_gov"
    ] = (
        source
        .groupby(
            "iso3c"
        )[
            "gdp_x_gov"
        ]
        .diff()
    )


    difference = (
        source[
            (
                source[
                    "year_gap"
                ]
                == 1
            )
            &
            source[
                "d_log_co2"
            ]
            .notna()
            &
            source[
                "d_log_gdp"
            ]
            .notna()
            &
            source[
                "d_gov_c10"
            ]
            .notna()
            &
            source[
                "d_gdp_x_gov"
            ]
            .notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    return (
        difference
    )


# ============================================================
# 7. RESULT-EXTRACTION HELPERS
# ============================================================

def sample_summary_row(
    data,
    indicator_key,
    indicator_name,
    sample_name,
):
    """
    Create one sample-audit record.
    """

    if data.empty:

        raise ValueError(
            f"Empty sample: {sample_name}"
        )


    return {
        "indicator":
            indicator_key,

        "indicator_display":
            indicator_name,

        "sample":
            sample_name,

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
    }


def extract_model_terms(
    model,
    data,
    indicator_key,
    indicator_name,
    specification,
    sample_name,
    terms,
    centering,
):
    """
    Export selected model coefficients to the master
    coefficient table.
    """

    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    rows = []


    for (
        term,
        interpretation,
    ) in terms.items():

        if (
            term
            not in
            model.params.index
        ):

            raise ValueError(
                f"Term '{term}' not found in model "
                f"'{specification}'."
            )


        ci = (
            confidence
            .loc[
                term
            ]
        )


        rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    indicator_name,

                "specification":
                    specification,

                "sample":
                    sample_name,

                "term":
                    term,

                "term_interpretation":
                    interpretation,

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

                "aic":
                    float(
                        model.aic
                    ),

                "bic":
                    float(
                        model.bic
                    ),

                "log_gdp_center":
                    centering.get(
                        "log_gdp_mean",
                        np.nan,
                    ),

                "governance_center":
                    centering.get(
                        "governance_mean",
                        np.nan,
                    ),

                "covariance":
                    (
                        "Economy-clustered SE; "
                        "finite-sample correction; "
                        "t-based inference"
                    ),
            }
        )


    return (
        rows
    )


def interaction_summary_row(
    model,
    data,
    indicator_key,
    indicator_name,
    sample_name,
    centering,
):
    """
    Extract the principal linear GDP × governance interaction.
    """

    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    beta_ci = (
        confidence
        .loc[
            "log_gdp_c"
        ]
    )


    gamma_ci = (
        confidence
        .loc[
            "gov_c10"
        ]
    )


    delta_ci = (
        confidence
        .loc[
            "gdp_x_gov"
        ]
    )


    beta = float(
        model.params[
            "log_gdp_c"
        ]
    )


    gamma = float(
        model.params[
            "gov_c10"
        ]
    )


    delta = float(
        model.params[
            "gdp_x_gov"
        ]
    )


    delta_p = float(
        model.pvalues[
            "gdp_x_gov"
        ]
    )


    return {
        "indicator":
            indicator_key,

        "indicator_display":
            indicator_name,

        "sample":
            sample_name,

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

        "governance_mean_score":
            centering[
                "governance_mean"
            ],

        "governance_sd_score":
            centering[
                "governance_sd"
            ],

        "governance_p25":
            centering[
                "governance_p25"
            ],

        "governance_p50":
            centering[
                "governance_p50"
            ],

        "governance_p75":
            centering[
                "governance_p75"
            ],

        "beta_gdp_at_mean_governance":
            beta,

        "beta_std_error":
            float(
                model.bse[
                    "log_gdp_c"
                ]
            ),

        "beta_ci95_low":
            float(
                beta_ci.iloc[0]
            ),

        "beta_ci95_high":
            float(
                beta_ci.iloc[1]
            ),

        "beta_p_value":
            float(
                model.pvalues[
                    "log_gdp_c"
                ]
            ),

        "gamma_governance_10pt_at_mean_gdp":
            gamma,

        "gamma_std_error":
            float(
                model.bse[
                    "gov_c10"
                ]
            ),

        "gamma_ci95_low":
            float(
                gamma_ci.iloc[0]
            ),

        "gamma_ci95_high":
            float(
                gamma_ci.iloc[1]
            ),

        "gamma_p_value":
            float(
                model.pvalues[
                    "gov_c10"
                ]
            ),

        "implied_co2_pct_change_for_10pt_governance_at_mean_gdp":
            (
                np.exp(
                    gamma
                )
                -
                1
            )
            *
            100,

        "delta_elasticity_change_per_10pt_governance":
            delta,

        "delta_std_error":
            float(
                model.bse[
                    "gdp_x_gov"
                ]
            ),

        "delta_ci95_low":
            float(
                delta_ci.iloc[0]
            ),

        "delta_ci95_high":
            float(
                delta_ci.iloc[1]
            ),

        "delta_p_value":
            delta_p,

        "interaction_significant_raw_5pct":
            bool(
                delta_p
                <
                SIGNIFICANCE_LEVEL
            ),
    }


def marginal_gdp_elasticities(
    model,
    indicator_key,
    indicator_name,
    sample_name,
    centering,
):
    """
    Calculate the GDP-emissions elasticity at governance P25,
    P50 and P75 for the baseline linear interaction model.
    """

    parameter_names = list(
        model.params.index
    )


    parameter_lookup = {
        name:
            index
        for (
            index,
            name,
        )
        in enumerate(
            parameter_names
        )
    }


    rows = []


    for (
        percentile_label,
        score_key,
    ) in [
        (
            "P25",
            "governance_p25",
        ),

        (
            "P50",
            "governance_p50",
        ),

        (
            "P75",
            "governance_p75",
        ),
    ]:

        score = float(
            centering[
                score_key
            ]
        )


        gov_c10_value = (
            score
            -
            centering[
                "governance_mean"
            ]
        ) / 10.0


        contrast = np.zeros(
            len(
                parameter_names
            )
        )


        contrast[
            parameter_lookup[
                "log_gdp_c"
            ]
        ] = (
            1.0
        )


        contrast[
            parameter_lookup[
                "gdp_x_gov"
            ]
        ] = (
            gov_c10_value
        )


        test = (
            model.t_test(
                contrast
            )
        )


        ci = (
            test.conf_int(
                alpha=ALPHA
            )[0]
        )


        rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    indicator_name,

                "sample":
                    sample_name,

                "governance_percentile":
                    percentile_label,

                "governance_score":
                    score,

                "governance_centered_10pt_units":
                    gov_c10_value,

                "marginal_gdp_elasticity":
                    scalar(
                        test.effect
                    ),

                "std_error":
                    scalar(
                        test.sd
                    ),

                "ci95_low":
                    float(
                        ci[0]
                    ),

                "ci95_high":
                    float(
                        ci[1]
                    ),

                "p_value":
                    scalar(
                        test.pvalue
                    ),
            }
        )


    return (
        rows
    )


# ============================================================
# 8. MULTIPLE-TESTING CORRECTION
# ============================================================

def add_holm_adjustment(
    df,
    group_columns,
    p_value_column,
    adjusted_column,
    reject_column,
):
    """
    Apply Holm family-wise-error correction across governance
    indicators within each specified model family.
    """

    output = (
        df
        .copy()
    )


    output[
        adjusted_column
    ] = (
        np.nan
    )


    output[
        reject_column
    ] = (
        False
    )


    grouped = (
        output
        .groupby(
            group_columns,
            dropna=False,
        )
    )


    for _, indices in grouped.groups.items():

        indices = list(
            indices
        )


        valid_indices = [
            index
            for index
            in indices
            if pd.notna(
                output.loc[
                    index,
                    p_value_column,
                ]
            )
        ]


        if not valid_indices:
            continue


        p_values = (
            output.loc[
                valid_indices,
                p_value_column,
            ]
            .astype(float)
            .to_numpy()
        )


        (
            reject,
            adjusted,
            _,
            _,
        ) = multipletests(
            p_values,
            alpha=SIGNIFICANCE_LEVEL,
            method="holm",
        )


        output.loc[
            valid_indices,
            adjusted_column,
        ] = (
            adjusted
        )


        output.loc[
            valid_indices,
            reject_column,
        ] = (
            reject
        )


    return (
        output
    )


# ============================================================
# 9. SPECIFICATION-ROBUSTNESS HELPERS
# ============================================================

def extract_delta_robustness_row(
    model,
    data,
    indicator_key,
    indicator_name,
    sample_name,
    specification,
    interaction_term="gdp_x_gov",
):
    """
    Extract the governance interaction from one robustness
    specification.
    """

    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    ci = (
        confidence
        .loc[
            interaction_term
        ]
    )


    return {
        "indicator":
            indicator_key,

        "indicator_display":
            indicator_name,

        "sample":
            sample_name,

        "specification":
            specification,

        "interaction_term":
            interaction_term,

        "delta_elasticity_change_per_10pt_governance":
            float(
                model.params[
                    interaction_term
                ]
            ),

        "delta_std_error":
            float(
                model.bse[
                    interaction_term
                ]
            ),

        "delta_ci95_low":
            float(
                ci.iloc[0]
            ),

        "delta_ci95_high":
            float(
                ci.iloc[1]
            ),

        "delta_p_value":
            float(
                model.pvalues[
                    interaction_term
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

        "n_parameters":
            int(
                len(
                    model.params
                )
            ),

        "r_squared":
            float(
                model.rsquared
            ),

        "adjusted_r_squared":
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
    }


def fit_specification_robustness(
    df,
    governance_column,
    indicator_key,
    indicator_name,
    sample_name,
):
    """
    Estimate governance moderation under:

    1. baseline linear interaction;
    2. region-specific GDP slopes;
    3. quadratic GDP;
    4. region-specific GDP slopes + quadratic GDP.

    Also estimates a quadratic common-slope model without
    governance for model-fit comparison.
    """

    (
        prepared,
        centering,
    ) = prepare_centered_sample(
        df,
        governance_column,
    )


    (
        region_prepared,
        slope_columns,
    ) = create_region_slope_variables(
        prepared,
        base_variable="log_gdp_c",
    )


    slope_terms = (
        " + "
        .join(
            slope_columns.values()
        )
    )


    model_definitions = {
        "Baseline linear governance interaction":
            (
                prepared,

                (
                    "log_co2 ~ "
                    "log_gdp_c + "
                    "gov_c10 + "
                    "gdp_x_gov + "
                    "C(iso3c) + C(year)"
                ),
            ),

        "Region-adjusted governance interaction":
            (
                region_prepared,

                (
                    "log_co2 ~ "
                    f"{slope_terms} + "
                    "gov_c10 + "
                    "gdp_x_gov + "
                    "C(iso3c) + C(year)"
                ),
            ),

        "Quadratic-GDP governance interaction":
            (
                prepared,

                (
                    "log_co2 ~ "
                    "log_gdp_c + "
                    "log_gdp_c_sq + "
                    "gov_c10 + "
                    "gdp_x_gov + "
                    "C(iso3c) + C(year)"
                ),
            ),

        "Region + quadratic governance interaction":
            (
                region_prepared,

                (
                    "log_co2 ~ "
                    f"{slope_terms} + "
                    "log_gdp_c_sq + "
                    "gov_c10 + "
                    "gdp_x_gov + "
                    "C(iso3c) + C(year)"
                ),
            ),
    }


    robustness_rows = []

    fit_rows = []


    for (
        specification,
        (
            model_data,
            formula,
        ),
    ) in model_definitions.items():

        model = fit_clustered_model(
            formula=formula,
            data=model_data,
        )


        check_design_rank(
            model,
            (
                f"{indicator_name}: "
                f"{sample_name}: "
                f"{specification}"
            ),
        )


        safe_spec = (
            specification
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "+",
                "plus",
            )
            .replace(
                "-",
                "_",
            )
        )


        save_model_summary(
            model,
            (
                f"{indicator_key}_robustness_"
                f"{safe_spec}.txt"
            ),
        )


        robustness_rows.append(
            extract_delta_robustness_row(
                model=model,
                data=model_data,
                indicator_key=indicator_key,
                indicator_name=indicator_name,
                sample_name=sample_name,
                specification=specification,
            )
        )


        fit_rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    indicator_name,

                "sample":
                    sample_name,

                "specification":
                    specification,

                "n_obs":
                    int(
                        model.nobs
                    ),

                "n_economies":
                    int(
                        model_data[
                            "iso3c"
                        ]
                        .nunique()
                    ),

                "n_parameters":
                    int(
                        len(
                            model.params
                        )
                    ),

                "r_squared":
                    float(
                        model.rsquared
                    ),

                "adjusted_r_squared":
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
            }
        )


    quadratic_common_model = fit_clustered_model(
        formula=(
            "log_co2 ~ "
            "log_gdp_c + "
            "log_gdp_c_sq + "
            "C(iso3c) + C(year)"
        ),
        data=prepared,
    )


    check_design_rank(
        quadratic_common_model,
        (
            f"{indicator_name}: "
            f"{sample_name}: "
            "Quadratic common-slope TWFE"
        ),
    )


    save_model_summary(
        quadratic_common_model,
        (
            f"{indicator_key}_robustness_"
            "quadratic_common_slope.txt"
        ),
    )


    fit_rows.append(
        {
            "indicator":
                indicator_key,

            "indicator_display":
                indicator_name,

            "sample":
                sample_name,

            "specification":
                "Quadratic common-slope TWFE",

            "n_obs":
                int(
                    quadratic_common_model.nobs
                ),

            "n_economies":
                int(
                    prepared[
                        "iso3c"
                    ]
                    .nunique()
                ),

            "n_parameters":
                int(
                    len(
                        quadratic_common_model.params
                    )
                ),

            "r_squared":
                float(
                    quadratic_common_model.rsquared
                ),

            "adjusted_r_squared":
                float(
                    quadratic_common_model.rsquared_adj
                ),

            "aic":
                float(
                    quadratic_common_model.aic
                ),

            "bic":
                float(
                    quadratic_common_model.bic
                ),
        }
    )


    return (
        robustness_rows,
        fit_rows,
        centering,
    )


def fit_between_within_decomposition(
    df,
    governance_column,
    indicator_key,
    indicator_name,
    sample_name,
):
    """
    Estimate separate persistent between-economy and
    within-economy governance moderation components.
    """

    (
        prepared,
        centering,
    ) = prepare_between_within_governance(
        df,
        governance_column,
    )


    model = fit_clustered_model(
        formula=(
            "log_co2 ~ "
            "log_gdp_c + "
            "gov_within_c10 + "
            "gdp_x_gov_between + "
            "gdp_x_gov_within + "
            "C(iso3c) + C(year)"
        ),
        data=prepared,
    )


    check_design_rank(
        model,
        (
            f"{indicator_name}: "
            f"{sample_name}: "
            "between-within governance decomposition"
        ),
    )


    save_model_summary(
        model,
        (
            f"{indicator_key}_"
            "between_within_decomposition.txt"
        ),
    )


    confidence = (
        model.conf_int(
            alpha=ALPHA
        )
    )


    rows = []


    for (
        component,
        term,
        interpretation,
    ) in [
        (
            "between",
            "gdp_x_gov_between",
            (
                "Difference in GDP-emissions elasticity "
                "associated with a 10-point higher persistent "
                "economy-average governance score"
            ),
        ),

        (
            "within",
            "gdp_x_gov_within",
            (
                "Difference in GDP-emissions elasticity "
                "associated with governance being 10 points "
                "above an economy's own long-run mean"
            ),
        ),
    ]:

        ci = (
            confidence
            .loc[
                term
            ]
        )


        rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    indicator_name,

                "sample":
                    sample_name,

                "component":
                    component,

                "term":
                    term,

                "interpretation":
                    interpretation,

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
                        prepared[
                            "iso3c"
                        ]
                        .nunique()
                    ),

                "r_squared":
                    float(
                        model.rsquared
                    ),

                "adjusted_r_squared":
                    float(
                        model.rsquared_adj
                    ),
            }
        )


    return (
        rows
    )


# ============================================================
# 10. LOCATE AND LOAD GOVERNANCE SOURCE
# ============================================================

print_section(
    "LOCATING GOVERNANCE WDI DATA"
)


GOVERNANCE_DATA_FILE = (
    find_governance_data_file(
        GOVERNANCE_RAW_DIR
    )
)


print(
    "Governance source file:\n"
    f"{GOVERNANCE_DATA_FILE}"
)


raw_governance = pd.read_csv(
    GOVERNANCE_DATA_FILE
)


require_columns(
    raw_governance,
    [
        "Country Name",
        "Country Code",
        "Series Name",
        "Series Code",
    ],
    "Governance WDI source",
)


source_row_count = (
    len(
        raw_governance
    )
)


governance_rows = (
    raw_governance[
        raw_governance[
            "Country Code"
        ]
        .notna()
        &
        raw_governance[
            "Series Code"
        ]
        .notna()
    ]
    .copy()
)


governance_rows[
    "Country Code"
] = (
    governance_rows[
        "Country Code"
    ]
    .astype(str)
    .str.strip()
)


governance_rows[
    "Series Code"
] = (
    governance_rows[
        "Series Code"
    ]
    .astype(str)
    .str.strip()
)


observed_codes = set(
    governance_rows[
        "Series Code"
    ]
    .unique()
)


missing_codes = (
    EXPECTED_GOV_CODES
    -
    observed_codes
)


if missing_codes:

    raise ValueError(
        "Governance source is missing expected series codes: "
        f"{missing_codes}"
    )


governance_rows = (
    governance_rows[
        governance_rows[
            "Series Code"
        ]
        .isin(
            EXPECTED_GOV_CODES
        )
    ]
    .copy()
)


if governance_rows.duplicated(
    [
        "Country Code",
        "Series Code",
    ]
).any():

    raise ValueError(
        "Duplicate economy-indicator rows found in governance "
        "source."
    )


print(
    f"Raw CSV rows: "
    f"{source_row_count:,}"
)

print(
    f"Valid governance series rows: "
    f"{len(governance_rows):,}"
)

print(
    f"Unique governance economies: "
    f"{governance_rows['Country Code'].nunique():,}"
)


# ============================================================
# 11. RESHAPE GOVERNANCE DATA
# ============================================================

print_section(
    "RESHAPING GOVERNANCE DATA"
)


year_map = (
    parse_year_columns(
        governance_rows.columns
    )
)


long_governance = (
    governance_rows
    .melt(
        id_vars=[
            "Country Name",
            "Country Code",
            "Series Name",
            "Series Code",
        ],

        value_vars=
            list(
                year_map.keys()
            ),

        var_name=
            "year_column",

        value_name=
            "governance_value_raw",
    )
)


long_governance[
    "year"
] = (
    long_governance[
        "year_column"
    ]
    .map(
        year_map
    )
    .astype(int)
)


long_governance[
    "governance_value"
] = pd.to_numeric(
    long_governance[
        "governance_value_raw"
    ],
    errors="coerce",
)


long_governance = (
    long_governance[
        (
            long_governance[
                "year"
            ]
            >= START_YEAR
        )
        &
        (
            long_governance[
                "year"
            ]
            <= END_YEAR
        )
    ]
    .copy()
)


observed_values = (
    long_governance[
        "governance_value"
    ]
    .dropna()
)


if (
    (
        observed_values
        < 0
    )
    .any()
    or
    (
        observed_values
        > 100
    )
    .any()
):

    raise ValueError(
        "Governance scores outside 0-100 were found."
    )


code_to_column = {
    details[
        "series_code"
    ]:
        details[
            "column"
        ]
    for details
    in GOVERNANCE_INDICATORS.values()
}


clean_governance = (
    long_governance
    .pivot_table(
        index=[
            "Country Code",
            "Country Name",
            "year",
        ],

        columns=
            "Series Code",

        values=
            "governance_value",

        aggfunc=
            "first",
    )
    .reset_index()
    .rename(
        columns={
            "Country Code":
                "iso3c",

            "Country Name":
                "governance_country_name",

            **code_to_column,
        }
    )
)


clean_governance = (
    clean_governance[
        clean_governance[
            GOVERNANCE_SCORE_COLUMNS
        ]
        .notna()
        .any(
            axis=1
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


if clean_governance.duplicated(
    [
        "iso3c",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate governance economy-year rows remain."
    )


clean_governance.to_csv(
    OUTPUT_DIR
    / "governance_wdi_clean_panel.csv",
    index=False,
)


print(
    f"Clean governance economy-year rows: "
    f"{len(clean_governance):,}"
)

print(
    f"Governance years: "
    f"{clean_governance['year'].min()}-"
    f"{clean_governance['year'].max()}"
)


# ============================================================
# 12. GOVERNANCE SOURCE AUDITS
# ============================================================

indicator_audit_rows = []


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    code = (
        details[
            "series_code"
        ]
    )


    source_subset = (
        long_governance[
            long_governance[
                "Series Code"
            ]
            ==
            code
        ]
    )


    observed = (
        source_subset[
            source_subset[
                "governance_value"
            ]
            .notna()
        ]
    )


    indicator_audit_rows.append(
        {
            "source_file":
                str(
                    GOVERNANCE_DATA_FILE
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),

            "indicator":
                indicator_key,

            "indicator_display":
                details[
                    "display_name"
                ],

            "series_code":
                code,

            "series_name":
                (
                    source_subset[
                        "Series Name"
                    ]
                    .dropna()
                    .iloc[0]
                ),

            "n_observations":
                int(
                    len(
                        observed
                    )
                ),

            "n_economies":
                int(
                    observed[
                        "Country Code"
                    ]
                    .nunique()
                ),

            "first_observed_year":
                int(
                    observed[
                        "year"
                    ]
                    .min()
                ),

            "last_observed_year":
                int(
                    observed[
                        "year"
                    ]
                    .max()
                ),

            "minimum_score":
                float(
                    observed[
                        "governance_value"
                    ]
                    .min()
                ),

            "maximum_score":
                float(
                    observed[
                        "governance_value"
                    ]
                    .max()
                ),

            "interpolation_used":
                False,
        }
    )


indicator_audit = (
    pd.DataFrame(
        indicator_audit_rows
    )
)


indicator_audit.to_csv(
    OUTPUT_DIR
    / "governance_source_indicator_audit.csv",
    index=False,
)


source_year_coverage = pd.DataFrame(
    {
        "year":
            list(
                range(
                    START_YEAR,
                    END_YEAR + 1,
                )
            )
    }
)


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    column = (
        details[
            "column"
        ]
    )


    counts = (
        clean_governance
        .groupby(
            "year"
        )[
            column
        ]
        .count()
    )


    source_year_coverage[
        f"n_{indicator_key}"
    ] = (
        source_year_coverage[
            "year"
        ]
        .map(
            counts
        )
        .fillna(0)
        .astype(int)
    )


complete_all_three_counts = (
    clean_governance[
        clean_governance[
            GOVERNANCE_SCORE_COLUMNS
        ]
        .notna()
        .all(
            axis=1
        )
    ]
    .groupby(
        "year"
    )[
        "iso3c"
    ]
    .nunique()
)


source_year_coverage[
    "n_complete_all_three"
] = (
    source_year_coverage[
        "year"
    ]
    .map(
        complete_all_three_counts
    )
    .fillna(0)
    .astype(int)
)


source_year_coverage.to_csv(
    OUTPUT_DIR
    / "governance_source_year_coverage.csv",
    index=False,
)


# ============================================================
# 13. RECONSTRUCT SCRIPT 09 ECONOMY SAMPLE
# ============================================================

print_section(
    "RECONSTRUCTING SCRIPT 09 ECONOMY SAMPLE"
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


if panel.duplicated(
    [
        "iso3c",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate WDI economy-year rows found."
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
        "Region mapping does not match the five-cluster design."
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
        >= 2
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


print(
    f"Main WDI observations: "
    f"{len(main_sample):,}"
)

print(
    f"Main WDI economies: "
    f"{main_sample['iso3c'].nunique():,}"
)


# ============================================================
# 14. CROSS-CHECK AGAINST SCRIPT 09
# ============================================================

SCRIPT09_FE_FILE = (
    SCRIPT09_RESULTS_DIR
    / "country_panel_fe_results.csv"
)


if SCRIPT09_FE_FILE.exists():

    script09 = pd.read_csv(
        SCRIPT09_FE_FILE
    )


    script09_main = (
        script09[
            script09[
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
            "Could not uniquely identify Script 09 main TWFE "
            "result."
        )


    script09_main = (
        script09_main.iloc[0]
    )


    if (
        len(
            main_sample
        )
        !=
        int(
            script09_main[
                "n_obs"
            ]
        )
    ):

        raise ValueError(
            "Script 12 sample does not reproduce Script 09 N."
        )


    if (
        main_sample[
            "iso3c"
        ]
        .nunique()
        !=
        int(
            script09_main[
                "n_countries"
            ]
        )
    ):

        raise ValueError(
            "Script 12 economy count does not reproduce "
            "Script 09."
        )


    print(
        "PASS: Script 12 reconstructs Script 09 levels sample."
    )


else:

    script09_main = None

    print(
        "WARNING: Script 09 result file was not found."
    )


# ============================================================
# 15. MERGE GOVERNANCE WITH VALIDATED WDI PANEL
# ============================================================

print_section(
    "MERGING GOVERNANCE WITH WDI PANEL"
)


merged = (
    main_sample
    .merge(
        clean_governance,
        on=[
            "iso3c",
            "year",
        ],
        how="left",
        validate="one_to_one",
    )
)


merged[
    "region_display"
] = (
    merged[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


if len(
    merged
) != len(
    main_sample
):

    raise ValueError(
        "Governance merge changed WDI row count."
    )


merged.to_csv(
    OUTPUT_DIR
    / "governance_merged_economy_panel.csv",
    index=False,
)


merge_audit_rows = [
    {
        "stage":
            "Script 09 WDI main sample",

        "n_obs":
            int(
                len(
                    main_sample
                )
            ),

        "n_economies":
            int(
                main_sample[
                    "iso3c"
                ]
                .nunique()
            ),

        "notes":
            "Validated 1990-2024 positive GDP/CO2 panel",
    },

    {
        "stage":
            "After left merge with governance",

        "n_obs":
            int(
                len(
                    merged
                )
            ),

        "n_economies":
            int(
                merged[
                    "iso3c"
                ]
                .nunique()
            ),

        "notes":
            "No governance interpolation or filling used",
    },
]


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    column = (
        details[
            "column"
        ]
    )


    available = (
        merged[
            merged[
                column
            ]
            .notna()
        ]
    )


    merge_audit_rows.append(
        {
            "stage":
                (
                    "Rows with "
                    f"{details['display_name']}"
                ),

            "n_obs":
                int(
                    len(
                        available
                    )
                ),

            "n_economies":
                int(
                    available[
                        "iso3c"
                    ]
                    .nunique()
                ),

            "notes":
                "Observed governance score only",
        }
    )


common_available = (
    merged
    .dropna(
        subset=
            GOVERNANCE_SCORE_COLUMNS
    )
)


merge_audit_rows.append(
    {
        "stage":
            "Rows complete on all three governance indicators",

        "n_obs":
            int(
                len(
                    common_available
                )
            ),

        "n_economies":
            int(
                common_available[
                    "iso3c"
                ]
                .nunique()
            ),

        "notes":
            "Before minimum-observation filter",
    }
)


merge_audit = (
    pd.DataFrame(
        merge_audit_rows
    )
)


merge_audit.to_csv(
    OUTPUT_DIR
    / "governance_merge_audit.csv",
    index=False,
)


# ============================================================
# 16. COVERAGE AUDITS
# ============================================================

main_year_coverage = pd.DataFrame(
    {
        "year":
            list(
                range(
                    START_YEAR,
                    END_YEAR + 1,
                )
            )
    }
)


main_wdi_counts = (
    merged
    .groupby(
        "year"
    )[
        "iso3c"
    ]
    .nunique()
)


main_year_coverage[
    "n_main_wdi_economies"
] = (
    main_year_coverage[
        "year"
    ]
    .map(
        main_wdi_counts
    )
    .fillna(0)
    .astype(int)
)


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    column = (
        details[
            "column"
        ]
    )


    counts = (
        merged[
            merged[
                column
            ]
            .notna()
        ]
        .groupby(
            "year"
        )[
            "iso3c"
        ]
        .nunique()
    )


    main_year_coverage[
        f"n_{indicator_key}"
    ] = (
        main_year_coverage[
            "year"
        ]
        .map(
            counts
        )
        .fillna(0)
        .astype(int)
    )


complete_counts = (
    merged[
        merged[
            GOVERNANCE_SCORE_COLUMNS
        ]
        .notna()
        .all(
            axis=1
        )
    ]
    .groupby(
        "year"
    )[
        "iso3c"
    ]
    .nunique()
)


main_year_coverage[
    "n_complete_all_three"
] = (
    main_year_coverage[
        "year"
    ]
    .map(
        complete_counts
    )
    .fillna(0)
    .astype(int)
)


main_year_coverage.to_csv(
    OUTPUT_DIR
    / "governance_main_sample_year_coverage.csv",
    index=False,
)


coverage_rows = []


for (
    iso3c,
    group,
) in merged.groupby(
    "iso3c"
):

    row = {
        "iso3c":
            iso3c,

        "country":
            group[
                "country"
            ]
            .iloc[0],

        "region":
            group[
                "region"
            ]
            .iloc[0],

        "region_display":
            group[
                "region_display"
            ]
            .iloc[0],
    }


    for (
        indicator_key,
        details,
    ) in GOVERNANCE_INDICATORS.items():

        column = (
            details[
                "column"
            ]
        )


        observed = (
            group[
                group[
                    column
                ]
                .notna()
            ]
        )


        row[
            f"n_{indicator_key}"
        ] = (
            int(
                len(
                    observed
                )
            )
        )


        row[
            f"first_year_{indicator_key}"
        ] = (
            int(
                observed[
                    "year"
                ]
                .min()
            )
            if not observed.empty
            else np.nan
        )


        row[
            f"last_year_{indicator_key}"
        ] = (
            int(
                observed[
                    "year"
                ]
                .max()
            )
            if not observed.empty
            else np.nan
        )


    row[
        "n_complete_all_three"
    ] = (
        int(
            len(
                group
                .dropna(
                    subset=
                        GOVERNANCE_SCORE_COLUMNS
                )
            )
        )
    )


    coverage_rows.append(
        row
    )


coverage_table = (
    pd.DataFrame(
        coverage_rows
    )
)


coverage_table.to_csv(
    OUTPUT_DIR
    / "governance_economy_coverage.csv",
    index=False,
)


# ============================================================
# 17. DESCRIPTIVE GOVERNANCE DIAGNOSTICS
# ============================================================

print_section(
    "GOVERNANCE DESCRIPTIVE DIAGNOSTICS"
)


descriptive_rows = []


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    column = (
        details[
            "column"
        ]
    )


    data = (
        merged[
            merged[
                column
            ]
            .notna()
        ]
        .copy()
    )


    economy_means = (
        data
        .groupby(
            "iso3c"
        )[
            column
        ]
        .mean()
    )


    within = (
        data[
            column
        ]
        -
        data
        .groupby(
            "iso3c"
        )[
            column
        ]
        .transform(
            "mean"
        )
    )


    descriptive_rows.append(
        {
            "indicator":
                indicator_key,

            "indicator_display":
                details[
                    "display_name"
                ],

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

            "mean":
                float(
                    data[
                        column
                    ]
                    .mean()
                ),

            "std":
                float(
                    data[
                        column
                    ]
                    .std()
                ),

            "median":
                float(
                    data[
                        column
                    ]
                    .median()
                ),

            "p25":
                float(
                    data[
                        column
                    ]
                    .quantile(
                        0.25
                    )
                ),

            "p75":
                float(
                    data[
                        column
                    ]
                    .quantile(
                        0.75
                    )
                ),

            "minimum":
                float(
                    data[
                        column
                    ]
                    .min()
                ),

            "maximum":
                float(
                    data[
                        column
                    ]
                    .max()
                ),

            "between_economy_sd_of_means":
                float(
                    economy_means
                    .std()
                ),

            "within_economy_sd":
                float(
                    within
                    .std()
                ),

            "economies_with_zero_within_variation":
                int(
                    (
                        data
                        .groupby(
                            "iso3c"
                        )[
                            column
                        ]
                        .std()
                        .fillna(0)
                        == 0
                    )
                    .sum()
                ),
        }
    )


descriptive_stats = (
    pd.DataFrame(
        descriptive_rows
    )
)


descriptive_stats.to_csv(
    OUTPUT_DIR
    / "governance_descriptive_statistics.csv",
    index=False,
)


correlation_sample = (
    merged
    .dropna(
        subset=
            GOVERNANCE_SCORE_COLUMNS
    )
    .copy()
)


pooled_correlations = (
    correlation_sample[
        GOVERNANCE_SCORE_COLUMNS
    ]
    .corr()
)


pooled_correlations.index.name = (
    "indicator"
)


pooled_correlations.to_csv(
    OUTPUT_DIR
    / "governance_correlations_pooled.csv"
)


within_correlation_sample = (
    correlation_sample
    .copy()
)


for column in GOVERNANCE_SCORE_COLUMNS:

    within_correlation_sample[
        column
    ] = (
        within_correlation_sample[
            column
        ]
        -
        within_correlation_sample
        .groupby(
            "iso3c"
        )[
            column
        ]
        .transform(
            "mean"
        )
    )


within_correlations = (
    within_correlation_sample[
        GOVERNANCE_SCORE_COLUMNS
    ]
    .corr()
)


within_correlations.index.name = (
    "indicator"
)


within_correlations.to_csv(
    OUTPUT_DIR
    / "governance_correlations_within_economy.csv"
)


region_summary_rows = []


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    column = (
        details[
            "column"
        ]
    )


    for (
        region,
        group,
    ) in (
        merged[
            merged[
                column
            ]
            .notna()
        ]
        .groupby(
            "region"
        )
    ):

        within = (
            group[
                column
            ]
            -
            group
            .groupby(
                "iso3c"
            )[
                column
            ]
            .transform(
                "mean"
            )
        )


        region_summary_rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    details[
                        "display_name"
                    ],

                "region":
                    region,

                "region_display":
                    REGION_DISPLAY_NAMES[
                        region
                    ],

                "n_obs":
                    int(
                        len(
                            group
                        )
                    ),

                "n_economies":
                    int(
                        group[
                            "iso3c"
                        ]
                        .nunique()
                    ),

                "mean_score":
                    float(
                        group[
                            column
                        ]
                        .mean()
                    ),

                "median_score":
                    float(
                        group[
                            column
                        ]
                        .median()
                    ),

                "within_economy_sd":
                    float(
                        within
                        .std()
                    ),
            }
        )


region_governance_summary = (
    pd.DataFrame(
        region_summary_rows
    )
)


region_governance_summary.to_csv(
    OUTPUT_DIR
    / "governance_region_summary.csv",
    index=False,
)


# ============================================================
# 18. PREPARE COMMON GOVERNANCE SAMPLES
# ============================================================

common_sample = (
    restrict_min_governance_observations(
        merged,
        GOVERNANCE_SCORE_COLUMNS,
    )
)


annual_common_complete = (
    merged[
        (
            merged[
                "year"
            ]
            >= ANNUAL_GOV_START_YEAR
        )
        &
        (
            merged[
                "year"
            ]
            <= ANNUAL_GOV_END_YEAR
        )
    ]
    .dropna(
        subset=
            GOVERNANCE_SCORE_COLUMNS
    )
    .copy()
)


common_balanced_status = (
    annual_common_complete
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
                ANNUAL_GOV_YEARS
            )
    )
)


common_balanced_codes = set(
    common_balanced_status[
        common_balanced_status
    ]
    .index
)


common_balanced_annual_sample = (
    annual_common_complete[
        annual_common_complete[
            "iso3c"
        ]
        .isin(
            common_balanced_codes
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


print(
    "Common balanced all-three governance sample: "
    f"{common_balanced_annual_sample['iso3c'].nunique():,} "
    "economies, "
    f"{len(common_balanced_annual_sample):,} observations."
)


# ============================================================
# 19. RESULT CONTAINERS
# ============================================================

model_result_rows = []

interaction_rows = []

marginal_rows = []

model_fit_rows = []

sample_rows = []

first_difference_rows = []

common_balanced_interaction_rows = []

specification_robustness_rows = []

specification_fit_rows = []

between_within_rows = []


# ============================================================
# 20. BASELINE GOVERNANCE MODELS
# ============================================================

print_section(
    "ESTIMATING BASELINE GOVERNANCE MODELS"
)


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    indicator_name = (
        details[
            "display_name"
        ]
    )


    governance_column = (
        details[
            "column"
        ]
    )


    print_section(
        indicator_name.upper()
    )


    # --------------------------------------------------------
    # A. MAIN AVAILABLE GOVERNANCE SAMPLE
    # --------------------------------------------------------

    main_indicator_sample = (
        restrict_min_governance_observations(
            merged,
            [
                governance_column
            ],
        )
    )


    (
        prepared_main,
        main_centering,
    ) = prepare_centered_sample(
        main_indicator_sample,
        governance_column,
    )


    sample_rows.append(
        sample_summary_row(
            prepared_main,
            indicator_key,
            indicator_name,
            "Main available governance sample",
        )
    )


    common_slope_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_main,
        )
    )


    additive_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_main,
        )
    )


    interaction_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "gdp_x_gov + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_main,
        )
    )


    for (
        model,
        model_name,
    ) in [
        (
            common_slope_model,
            "Common-slope TWFE",
        ),

        (
            additive_model,
            "Additive-governance TWFE",
        ),

        (
            interaction_model,
            "GDP-governance interaction TWFE",
        ),
    ]:

        check_design_rank(
            model,
            (
                f"{indicator_name}: "
                f"{model_name}"
            ),
        )


    save_model_summary(
        common_slope_model,
        (
            f"{indicator_key}_"
            "01_common_slope_twfe.txt"
        ),
    )


    save_model_summary(
        additive_model,
        (
            f"{indicator_key}_"
            "02_additive_governance_twfe.txt"
        ),
    )


    save_model_summary(
        interaction_model,
        (
            f"{indicator_key}_"
            "03_interaction_twfe.txt"
        ),
    )


    model_result_rows.extend(
        extract_model_terms(
            common_slope_model,
            prepared_main,
            indicator_key,
            indicator_name,
            "Common-slope TWFE",
            "Main available governance sample",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity on matched "
                        "governance sample"
                    ),
            },
            main_centering,
        )
    )


    model_result_rows.extend(
        extract_model_terms(
            additive_model,
            prepared_main,
            indicator_key,
            indicator_name,
            "Additive-governance TWFE",
            "Main available governance sample",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity conditional "
                        "on governance"
                    ),

                "gov_c10":
                    (
                        "Association of a 10-point higher "
                        "governance score with log CO2 at "
                        "average log GDP"
                    ),
            },
            main_centering,
        )
    )


    model_result_rows.extend(
        extract_model_terms(
            interaction_model,
            prepared_main,
            indicator_key,
            indicator_name,
            "GDP-governance interaction TWFE",
            "Main available governance sample",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity at mean "
                        "governance"
                    ),

                "gov_c10":
                    (
                        "Governance association at average "
                        "log GDP"
                    ),

                "gdp_x_gov":
                    (
                        "Change in GDP-emissions elasticity "
                        "for a 10-point higher governance score"
                    ),
            },
            main_centering,
        )
    )


    interaction_rows.append(
        interaction_summary_row(
            interaction_model,
            prepared_main,
            indicator_key,
            indicator_name,
            "Main available governance sample",
            main_centering,
        )
    )


    marginal_rows.extend(
        marginal_gdp_elasticities(
            interaction_model,
            indicator_key,
            indicator_name,
            "Main available governance sample",
            main_centering,
        )
    )


    baseline_aic = float(
        common_slope_model.aic
    )


    baseline_bic = float(
        common_slope_model.bic
    )


    for (
        model,
        model_name,
    ) in [
        (
            common_slope_model,
            "Common-slope TWFE",
        ),

        (
            additive_model,
            "Additive-governance TWFE",
        ),

        (
            interaction_model,
            "GDP-governance interaction TWFE",
        ),
    ]:

        model_fit_rows.append(
            {
                "indicator":
                    indicator_key,

                "indicator_display":
                    indicator_name,

                "sample":
                    "Main available governance sample",

                "model":
                    model_name,

                "n_obs":
                    int(
                        model.nobs
                    ),

                "n_parameters":
                    int(
                        len(
                            model.params
                        )
                    ),

                "r_squared":
                    float(
                        model.rsquared
                    ),

                "adjusted_r_squared":
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

                "delta_aic_vs_common_slope":
                    float(
                        model.aic
                        -
                        baseline_aic
                    ),

                "delta_bic_vs_common_slope":
                    float(
                        model.bic
                        -
                        baseline_bic
                    ),
            }
        )


    # --------------------------------------------------------
    # B. COMMON COMPLETE-CASE SAMPLE
    # --------------------------------------------------------

    (
        prepared_common,
        common_centering,
    ) = prepare_centered_sample(
        common_sample,
        governance_column,
    )


    sample_rows.append(
        sample_summary_row(
            prepared_common,
            indicator_key,
            indicator_name,
            "Common complete-case governance sample",
        )
    )


    common_interaction_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "gdp_x_gov + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_common,
        )
    )


    check_design_rank(
        common_interaction_model,
        (
            f"{indicator_name}: "
            "common complete-case interaction"
        ),
    )


    save_model_summary(
        common_interaction_model,
        (
            f"{indicator_key}_"
            "04_common_sample_interaction.txt"
        ),
    )


    # FIX:
    # Restore all three coefficient rows to the master results file.
    model_result_rows.extend(
        extract_model_terms(
            common_interaction_model,
            prepared_common,
            indicator_key,
            indicator_name,
            "GDP-governance interaction TWFE",
            "Common complete-case governance sample",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity at mean "
                        "governance"
                    ),

                "gov_c10":
                    (
                        "Governance association at average "
                        "log GDP"
                    ),

                "gdp_x_gov":
                    (
                        "Change in GDP-emissions elasticity "
                        "for a 10-point higher governance score"
                    ),
            },
            common_centering,
        )
    )


    interaction_rows.append(
        interaction_summary_row(
            common_interaction_model,
            prepared_common,
            indicator_key,
            indicator_name,
            "Common complete-case governance sample",
            common_centering,
        )
    )


    marginal_rows.extend(
        marginal_gdp_elasticities(
            common_interaction_model,
            indicator_key,
            indicator_name,
            "Common complete-case governance sample",
            common_centering,
        )
    )


    # --------------------------------------------------------
    # C. ANNUAL GOVERNANCE ERA: 2002-2024
    # --------------------------------------------------------

    annual_era = (
        restrict_min_governance_observations(
            merged[
                (
                    merged[
                        "year"
                    ]
                    >= ANNUAL_GOV_START_YEAR
                )
                &
                (
                    merged[
                        "year"
                    ]
                    <= ANNUAL_GOV_END_YEAR
                )
            ],
            [
                governance_column
            ],
        )
    )


    (
        prepared_annual,
        annual_centering,
    ) = prepare_centered_sample(
        annual_era,
        governance_column,
    )


    sample_rows.append(
        sample_summary_row(
            prepared_annual,
            indicator_key,
            indicator_name,
            "Annual governance era 2002-2024",
        )
    )


    annual_interaction_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "gdp_x_gov + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_annual,
        )
    )


    check_design_rank(
        annual_interaction_model,
        (
            f"{indicator_name}: "
            "annual-era interaction"
        ),
    )


    save_model_summary(
        annual_interaction_model,
        (
            f"{indicator_key}_"
            "05_annual_era_interaction.txt"
        ),
    )


    # FIX:
    # Restore detailed annual-era coefficients.
    model_result_rows.extend(
        extract_model_terms(
            annual_interaction_model,
            prepared_annual,
            indicator_key,
            indicator_name,
            "GDP-governance interaction TWFE",
            "Annual governance era 2002-2024",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity at mean "
                        "governance"
                    ),

                "gov_c10":
                    (
                        "Governance association at average "
                        "log GDP"
                    ),

                "gdp_x_gov":
                    (
                        "Change in GDP-emissions elasticity "
                        "for a 10-point higher governance score"
                    ),
            },
            annual_centering,
        )
    )


    interaction_rows.append(
        interaction_summary_row(
            annual_interaction_model,
            prepared_annual,
            indicator_key,
            indicator_name,
            "Annual governance era 2002-2024",
            annual_centering,
        )
    )


    marginal_rows.extend(
        marginal_gdp_elasticities(
            annual_interaction_model,
            indicator_key,
            indicator_name,
            "Annual governance era 2002-2024",
            annual_centering,
        )
    )


    # --------------------------------------------------------
    # D. INDICATOR-SPECIFIC BALANCED ANNUAL SAMPLE
    # --------------------------------------------------------

    annual_nonmissing = (
        merged[
            (
                merged[
                    "year"
                ]
                >= ANNUAL_GOV_START_YEAR
            )
            &
            (
                merged[
                    "year"
                ]
                <= ANNUAL_GOV_END_YEAR
            )
            &
            merged[
                governance_column
            ]
            .notna()
        ]
        .copy()
    )


    balanced_status = (
        annual_nonmissing
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
                    ANNUAL_GOV_YEARS
                )
        )
    )


    balanced_codes = set(
        balanced_status[
            balanced_status
        ]
        .index
    )


    balanced_annual = (
        annual_nonmissing[
            annual_nonmissing[
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


    (
        prepared_balanced,
        balanced_centering,
    ) = prepare_centered_sample(
        balanced_annual,
        governance_column,
    )


    sample_rows.append(
        sample_summary_row(
            prepared_balanced,
            indicator_key,
            indicator_name,
            "Balanced annual governance era 2002-2024",
        )
    )


    balanced_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "gdp_x_gov + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_balanced,
        )
    )


    check_design_rank(
        balanced_model,
        (
            f"{indicator_name}: "
            "balanced annual interaction"
        ),
    )


    save_model_summary(
        balanced_model,
        (
            f"{indicator_key}_"
            "06_balanced_annual_interaction.txt"
        ),
    )


    # FIX:
    # Restore detailed balanced-sample coefficients.
    model_result_rows.extend(
        extract_model_terms(
            balanced_model,
            prepared_balanced,
            indicator_key,
            indicator_name,
            "GDP-governance interaction TWFE",
            "Balanced annual governance era 2002-2024",
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity at mean "
                        "governance"
                    ),

                "gov_c10":
                    (
                        "Governance association at average "
                        "log GDP"
                    ),

                "gdp_x_gov":
                    (
                        "Change in GDP-emissions elasticity "
                        "for a 10-point higher governance score"
                    ),
            },
            balanced_centering,
        )
    )


    interaction_rows.append(
        interaction_summary_row(
            balanced_model,
            prepared_balanced,
            indicator_key,
            indicator_name,
            "Balanced annual governance era 2002-2024",
            balanced_centering,
        )
    )


    marginal_rows.extend(
        marginal_gdp_elasticities(
            balanced_model,
            indicator_key,
            indicator_name,
            "Balanced annual governance era 2002-2024",
            balanced_centering,
        )
    )


    # --------------------------------------------------------
    # E. COMMON BALANCED ALL-THREE SAMPLE
    # --------------------------------------------------------

    (
        prepared_common_balanced,
        common_balanced_centering,
    ) = prepare_centered_sample(
        common_balanced_annual_sample,
        governance_column,
    )


    sample_rows.append(
        sample_summary_row(
            prepared_common_balanced,
            indicator_key,
            indicator_name,
            (
                "Common balanced all-three governance "
                "sample 2002-2024"
            ),
        )
    )


    common_balanced_model = (
        fit_clustered_model(
            formula=(
                "log_co2 ~ "
                "log_gdp_c + "
                "gov_c10 + "
                "gdp_x_gov + "
                "C(iso3c) + C(year)"
            ),
            data=prepared_common_balanced,
        )
    )


    check_design_rank(
        common_balanced_model,
        (
            f"{indicator_name}: "
            "common balanced all-three interaction"
        ),
    )


    save_model_summary(
        common_balanced_model,
        (
            f"{indicator_key}_"
            "06b_common_balanced_interaction.txt"
        ),
    )


    # NEW MASTER-TABLE EXPORT:
    # Include the common balanced model in governance_model_results.csv.
    model_result_rows.extend(
        extract_model_terms(
            common_balanced_model,
            prepared_common_balanced,
            indicator_key,
            indicator_name,
            "GDP-governance interaction TWFE",
            (
                "Common balanced all-three governance "
                "sample 2002-2024"
            ),
            {
                "log_gdp_c":
                    (
                        "GDP-emissions elasticity at mean "
                        "governance"
                    ),

                "gov_c10":
                    (
                        "Governance association at average "
                        "log GDP"
                    ),

                "gdp_x_gov":
                    (
                        "Change in GDP-emissions elasticity "
                        "for a 10-point higher governance score"
                    ),
            },
            common_balanced_centering,
        )
    )


    common_balanced_interaction_rows.append(
        interaction_summary_row(
            common_balanced_model,
            prepared_common_balanced,
            indicator_key,
            indicator_name,
            (
                "Common balanced all-three governance "
                "sample 2002-2024"
            ),
            common_balanced_centering,
        )
    )


    interaction_rows.append(
        common_balanced_interaction_rows[
            -1
        ]
    )


    # --------------------------------------------------------
    # F. FIRST-DIFFERENCE ROBUSTNESS
    # --------------------------------------------------------

    difference_sample = (
        build_first_difference_sample(
            prepared_main
        )
    )


    if difference_sample.empty:

        raise ValueError(
            "No consecutive-year first differences for "
            f"{indicator_name}."
        )


    if difference_sample[
        "year_gap"
    ].ne(
        1
    ).any():

        raise ValueError(
            "Non-consecutive first differences remain for "
            f"{indicator_name}."
        )


    sample_rows.append(
        sample_summary_row(
            difference_sample,
            indicator_key,
            indicator_name,
            "First-difference consecutive-year sample",
        )
    )


    fd_model = (
        fit_clustered_model(
            formula=(
                "d_log_co2 ~ "
                "d_log_gdp + "
                "d_gov_c10 + "
                "d_gdp_x_gov + "
                "C(year)"
            ),
            data=difference_sample,
        )
    )


    check_design_rank(
        fd_model,
        (
            f"{indicator_name}: "
            "first-difference interaction"
        ),
    )


    save_model_summary(
        fd_model,
        (
            f"{indicator_key}_"
            "07_first_difference_interaction.txt"
        ),
    )


    fd_rows = (
        extract_model_terms(
            fd_model,
            difference_sample,
            indicator_key,
            indicator_name,
            "First-difference governance interaction",
            "First-difference consecutive-year sample",
            {
                "d_log_gdp":
                    (
                        "Short-run GDP-emissions elasticity "
                        "component"
                    ),

                "d_gov_c10":
                    (
                        "Association of a 10-point governance "
                        "change with emissions growth"
                    ),

                "d_gdp_x_gov":
                    (
                        "First-difference governance "
                        "moderation term"
                    ),
            },
            main_centering,
        )
    )


    model_result_rows.extend(
        fd_rows
    )


    first_difference_rows.extend(
        fd_rows
    )


    # --------------------------------------------------------
    # G. SPECIFICATION ROBUSTNESS
    #
    # All three governance indicators use the same common
    # complete-case sample here.
    # --------------------------------------------------------

    (
        robust_rows,
        robust_fit_rows,
        _,
    ) = fit_specification_robustness(
        df=common_sample,
        governance_column=governance_column,
        indicator_key=indicator_key,
        indicator_name=indicator_name,
        sample_name=
            "Common complete-case governance sample",
    )


    specification_robustness_rows.extend(
        robust_rows
    )


    specification_fit_rows.extend(
        robust_fit_rows
    )


    # --------------------------------------------------------
    # H. BETWEEN / WITHIN GOVERNANCE DECOMPOSITION
    # --------------------------------------------------------

    decomposition_rows = (
        fit_between_within_decomposition(
            df=common_sample,
            governance_column=governance_column,
            indicator_key=indicator_key,
            indicator_name=indicator_name,
            sample_name=
                "Common complete-case governance sample",
        )
    )


    between_within_rows.extend(
        decomposition_rows
    )


# ============================================================
# 21. BUILD BASELINE RESULT TABLES
# ============================================================

print_section(
    "EXPORTING GOVERNANCE RESULTS"
)


model_results = (
    pd.DataFrame(
        model_result_rows
    )
)


interaction_summary = (
    pd.DataFrame(
        interaction_rows
    )
)


marginal_elasticity_results = (
    pd.DataFrame(
        marginal_rows
    )
)


model_fit_comparison = (
    pd.DataFrame(
        model_fit_rows
    )
)


sample_summary = (
    pd.DataFrame(
        sample_rows
    )
)


first_difference_results = (
    pd.DataFrame(
        first_difference_rows
    )
)


# ============================================================
# 22. HOLM CORRECTION — BASELINE INTERACTION MODELS
# ============================================================

interaction_summary = (
    add_holm_adjustment(
        df=interaction_summary,
        group_columns=[
            "sample",
        ],
        p_value_column=
            "delta_p_value",
        adjusted_column=
            "delta_p_value_holm",
        reject_column=
            "interaction_significant_holm_5pct",
    )
)


first_difference_results[
    "p_value_holm"
] = (
    np.nan
)


first_difference_results[
    "significant_holm_5pct"
] = (
    False
)


fd_interaction_mask = (
    first_difference_results[
        "term"
    ]
    ==
    "d_gdp_x_gov"
)


fd_interactions_for_adjustment = (
    first_difference_results[
        fd_interaction_mask
    ]
    .copy()
)


if len(
    fd_interactions_for_adjustment
) != len(
    GOVERNANCE_INDICATORS
):

    raise ValueError(
        "First-difference Holm correction expected exactly "
        "one governance interaction per indicator."
    )


(
    fd_reject,
    fd_adjusted,
    _,
    _,
) = multipletests(
    fd_interactions_for_adjustment[
        "p_value"
    ]
    .to_numpy(),

    alpha=SIGNIFICANCE_LEVEL,
    method="holm",
)


first_difference_results.loc[
    fd_interaction_mask,
    "p_value_holm",
] = (
    fd_adjusted
)


first_difference_results.loc[
    fd_interaction_mask,
    "significant_holm_5pct",
] = (
    fd_reject
)


# ============================================================
# 23. EXPORT BASELINE FILES
# ============================================================

model_results.to_csv(
    OUTPUT_DIR
    / "governance_model_results.csv",
    index=False,
)


interaction_summary.to_csv(
    OUTPUT_DIR
    / "governance_interaction_summary.csv",
    index=False,
)


marginal_elasticity_results.to_csv(
    OUTPUT_DIR
    / "governance_marginal_gdp_elasticities.csv",
    index=False,
)


model_fit_comparison.to_csv(
    OUTPUT_DIR
    / "governance_model_fit_comparison.csv",
    index=False,
)


sample_summary.to_csv(
    OUTPUT_DIR
    / "governance_sample_summary.csv",
    index=False,
)


first_difference_results.to_csv(
    OUTPUT_DIR
    / "governance_first_difference_results.csv",
    index=False,
)


common_sample_interactions = (
    interaction_summary[
        interaction_summary[
            "sample"
        ]
        ==
        "Common complete-case governance sample"
    ]
    .copy()
)


common_sample_interactions.to_csv(
    OUTPUT_DIR
    / "governance_common_sample_interactions.csv",
    index=False,
)


balanced_annual_interactions = (
    interaction_summary[
        interaction_summary[
            "sample"
        ]
        ==
        "Balanced annual governance era 2002-2024"
    ]
    .copy()
)


balanced_annual_interactions.to_csv(
    OUTPUT_DIR
    / "governance_balanced_annual_interactions.csv",
    index=False,
)


common_balanced_interactions = (
    interaction_summary[
        interaction_summary[
            "sample"
        ]
        ==
        (
            "Common balanced all-three governance "
            "sample 2002-2024"
        )
    ]
    .copy()
)


common_balanced_interactions.to_csv(
    OUTPUT_DIR
    / "governance_common_balanced_annual_interactions.csv",
    index=False,
)


interaction_summary[
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
        "interaction_significant_raw_5pct",
        "interaction_significant_holm_5pct",
    ]
].to_csv(
    OUTPUT_DIR
    / "governance_interaction_holm_adjusted.csv",
    index=False,
)


# ============================================================
# 24. SPECIFICATION-ROBUSTNESS RESULTS
# ============================================================

specification_robustness = (
    pd.DataFrame(
        specification_robustness_rows
    )
)


specification_robustness = (
    add_holm_adjustment(
        df=specification_robustness,
        group_columns=[
            "sample",
            "specification",
        ],
        p_value_column=
            "delta_p_value",
        adjusted_column=
            "delta_p_value_holm",
        reject_column=
            "delta_significant_holm_5pct",
    )
)


specification_robustness[
    "delta_significant_raw_5pct"
] = (
    specification_robustness[
        "delta_p_value"
    ]
    <
    SIGNIFICANCE_LEVEL
)


specification_robustness.to_csv(
    OUTPUT_DIR
    / "governance_specification_robustness.csv",
    index=False,
)


specification_robustness[
    specification_robustness[
        "specification"
    ]
    ==
    "Region-adjusted governance interaction"
].to_csv(
    OUTPUT_DIR
    / "governance_region_adjusted_interactions.csv",
    index=False,
)


specification_robustness[
    specification_robustness[
        "specification"
    ]
    ==
    "Quadratic-GDP governance interaction"
].to_csv(
    OUTPUT_DIR
    / "governance_quadratic_gdp_interactions.csv",
    index=False,
)


specification_robustness[
    specification_robustness[
        "specification"
    ]
    ==
    "Region + quadratic governance interaction"
].to_csv(
    OUTPUT_DIR
    / "governance_region_quadratic_interactions.csv",
    index=False,
)


specification_fit = (
    pd.DataFrame(
        specification_fit_rows
    )
)


for (
    indicator,
    sample,
), group in specification_fit.groupby(
    [
        "indicator",
        "sample",
    ]
):

    baseline = (
        group[
            group[
                "specification"
            ]
            ==
            "Baseline linear governance interaction"
        ]
    )


    if len(
        baseline
    ) != 1:

        raise ValueError(
            "Could not identify unique baseline interaction "
            f"for {indicator}, {sample}."
        )


    baseline_aic = float(
        baseline[
            "aic"
        ]
        .iloc[0]
    )


    baseline_bic = float(
        baseline[
            "bic"
        ]
        .iloc[0]
    )


    mask = (
        (
            specification_fit[
                "indicator"
            ]
            ==
            indicator
        )
        &
        (
            specification_fit[
                "sample"
            ]
            ==
            sample
        )
    )


    specification_fit.loc[
        mask,
        "delta_aic_vs_linear_interaction",
    ] = (
        specification_fit.loc[
            mask,
            "aic",
        ]
        -
        baseline_aic
    )


    specification_fit.loc[
        mask,
        "delta_bic_vs_linear_interaction",
    ] = (
        specification_fit.loc[
            mask,
            "bic",
        ]
        -
        baseline_bic
    )


specification_fit.to_csv(
    OUTPUT_DIR
    / "governance_specification_fit_comparison.csv",
    index=False,
)


# ============================================================
# 25. BETWEEN / WITHIN GOVERNANCE DECOMPOSITION
# ============================================================

between_within_results = (
    pd.DataFrame(
        between_within_rows
    )
)


between_within_results = (
    add_holm_adjustment(
        df=between_within_results,
        group_columns=[
            "component",
        ],
        p_value_column=
            "p_value",
        adjusted_column=
            "p_value_holm",
        reject_column=
            "significant_holm_5pct",
    )
)


between_within_results[
    "significant_raw_5pct"
] = (
    between_within_results[
        "p_value"
    ]
    <
    SIGNIFICANCE_LEVEL
)


between_within_results.to_csv(
    OUTPUT_DIR
    / "governance_between_within_decomposition.csv",
    index=False,
)


# ============================================================
# 26. TERMINAL OUTPUT — BASELINE INTERACTIONS
# ============================================================

print_section(
    "BASELINE GOVERNANCE INTERACTION RESULTS"
)


main_interactions = (
    interaction_summary[
        interaction_summary[
            "sample"
        ]
        ==
        "Main available governance sample"
    ]
    .copy()
)


print(
    main_interactions[
        [
            "indicator_display",
            "n_obs",
            "n_economies",
            "beta_gdp_at_mean_governance",
            "delta_elasticity_change_per_10pt_governance",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_p_value",
            "delta_p_value_holm",
            "interaction_significant_holm_5pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 27. TERMINAL OUTPUT — SPECIFICATION SENSITIVITY
# ============================================================

print_section(
    "GOVERNANCE SPECIFICATION SENSITIVITY"
)


print(
    specification_robustness[
        [
            "indicator_display",
            "specification",
            "delta_elasticity_change_per_10pt_governance",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_p_value",
            "delta_p_value_holm",
            "delta_significant_holm_5pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 28. TERMINAL OUTPUT — BETWEEN / WITHIN DECOMPOSITION
# ============================================================

print_section(
    "BETWEEN / WITHIN GOVERNANCE DECOMPOSITION"
)


print(
    between_within_results[
        [
            "indicator_display",
            "component",
            "coefficient",
            "ci95_low",
            "ci95_high",
            "p_value",
            "p_value_holm",
            "significant_holm_5pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. FIGURE — BASELINE LINEAR INTERACTION
# ============================================================

print_section(
    "CREATING BASELINE GOVERNANCE INTERACTION FIGURE"
)


plot_data = (
    main_interactions
    .set_index(
        "indicator"
    )
    .loc[
        list(
            GOVERNANCE_INDICATORS.keys()
        )
    ]
    .reset_index()
)


y_positions = np.arange(
    len(
        plot_data
    )
)


coefficient = (
    plot_data[
        "delta_elasticity_change_per_10pt_governance"
    ]
    .to_numpy()
)


lower_error = (
    coefficient
    -
    plot_data[
        "delta_ci95_low"
    ]
    .to_numpy()
)


upper_error = (
    plot_data[
        "delta_ci95_high"
    ]
    .to_numpy()
    -
    coefficient
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5,
    )
)


ax.errorbar(
    coefficient,
    y_positions,
    xerr=np.vstack(
        [
            lower_error,
            upper_error,
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
        "indicator_display"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    "Change in GDP–CO2 elasticity per 10-point higher "
    "governance score"
)


ax.set_title(
    "Governance and the GDP–CO2 Relationship\n"
    "Baseline Linear Economy and Year Fixed-Effects Model"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "Negative coefficients indicate weaker GDP–CO2 coupling "
        "in the baseline linear specification. Bars show "
        "economy-clustered 95% confidence intervals. "
        "Holm-adjusted p-values are reported in the result tables."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.07,
        1,
        1,
    ]
)


INTERACTION_FIGURE_FILE = (
    FIGURES_DIR
    / "12_governance_interaction_coefficients.png"
)


fig.savefig(
    INTERACTION_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# 30. FIGURE — BASELINE MARGINAL GDP ELASTICITIES
# ============================================================

print_section(
    "CREATING BASELINE MARGINAL-ELASTICITY FIGURE"
)


marginal_plot = (
    marginal_elasticity_results[
        marginal_elasticity_results[
            "sample"
        ]
        ==
        "Main available governance sample"
    ]
    .copy()
)


percentile_order = [
    "P25",
    "P50",
    "P75",
]


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5,
    )
)


x_positions = np.arange(
    len(
        percentile_order
    )
)


for (
    indicator_key,
    details,
) in GOVERNANCE_INDICATORS.items():

    subset = (
        marginal_plot[
            marginal_plot[
                "indicator"
            ]
            ==
            indicator_key
        ]
        .set_index(
            "governance_percentile"
        )
        .loc[
            percentile_order
        ]
    )


    ax.errorbar(
        x_positions,
        subset[
            "marginal_gdp_elasticity"
        ]
        .to_numpy(),

        yerr=np.vstack(
            [
                (
                    subset[
                        "marginal_gdp_elasticity"
                    ]
                    -
                    subset[
                        "ci95_low"
                    ]
                )
                .to_numpy(),

                (
                    subset[
                        "ci95_high"
                    ]
                    -
                    subset[
                        "marginal_gdp_elasticity"
                    ]
                )
                .to_numpy(),
            ]
        ),

        marker="o",
        capsize=4,
        label=
            details[
                "display_name"
            ],
    )


ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
)


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(
    percentile_order
)


ax.set_xlabel(
    "Governance-score percentile"
)


ax.set_ylabel(
    "Estimated GDP per capita elasticity of CO2 emissions "
    "per capita"
)


ax.set_title(
    "Baseline Linear Marginal GDP–CO2 Elasticity\n"
    "Across Governance Levels"
)


ax.legend(
    loc="best"
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "These marginal effects belong to the baseline linear "
        "governance specification and should be interpreted "
        "alongside the specification-sensitivity tests."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.07,
        1,
        1,
    ]
)


MARGINAL_FIGURE_FILE = (
    FIGURES_DIR
    / "12_marginal_gdp_elasticity_by_governance.png"
)


fig.savefig(
    MARGINAL_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# 31. FIGURE — SPECIFICATION SENSITIVITY
# ============================================================

print_section(
    "CREATING GOVERNANCE SPECIFICATION-SENSITIVITY FIGURE"
)


spec_order = [
    "Baseline linear governance interaction",
    "Region-adjusted governance interaction",
    "Quadratic-GDP governance interaction",
    "Region + quadratic governance interaction",
]


sensitivity_plot = (
    specification_robustness
    .copy()
)


sensitivity_plot[
    "row_label"
] = (
    sensitivity_plot[
        "indicator_display"
    ]
    +
    " — "
    +
    sensitivity_plot[
        "specification"
    ]
)


ordered_rows = []


for indicator_key in GOVERNANCE_INDICATORS.keys():

    for specification in spec_order:

        match = (
            sensitivity_plot[
                (
                    sensitivity_plot[
                        "indicator"
                    ]
                    ==
                    indicator_key
                )
                &
                (
                    sensitivity_plot[
                        "specification"
                    ]
                    ==
                    specification
                )
            ]
        )


        if len(
            match
        ) != 1:

            raise ValueError(
                "Specification-sensitivity figure could not "
                "identify a unique result."
            )


        ordered_rows.append(
            match.iloc[0]
        )


ordered_plot = (
    pd.DataFrame(
        ordered_rows
    )
    .reset_index(
        drop=True
    )
)


y_positions = np.arange(
    len(
        ordered_plot
    )
)


coef = (
    ordered_plot[
        "delta_elasticity_change_per_10pt_governance"
    ]
    .to_numpy()
)


lower = (
    coef
    -
    ordered_plot[
        "delta_ci95_low"
    ]
    .to_numpy()
)


upper = (
    ordered_plot[
        "delta_ci95_high"
    ]
    .to_numpy()
    -
    coef
)


fig, ax = plt.subplots(
    figsize=(
        11,
        9,
    )
)


ax.errorbar(
    coef,
    y_positions,
    xerr=np.vstack(
        [
            lower,
            upper,
        ]
    ),
    fmt="o",
    capsize=3,
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
    ordered_plot[
        "row_label"
    ]
)


ax.invert_yaxis()


ax.set_xlabel(
    "GDP × governance interaction per 10-point governance score"
)


ax.set_title(
    "Governance Interaction Sensitivity to Model Specification"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.01,
    (
        "All specifications use the common complete-case "
        "governance sample. Bars show economy-clustered "
        "95% confidence intervals."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        1,
    ]
)


SENSITIVITY_FIGURE_FILE = (
    FIGURES_DIR
    / "12_governance_specification_sensitivity.png"
)


fig.savefig(
    SENSITIVITY_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# 32. FINAL VALIDATION
# ============================================================

print_section(
    "FINAL VALIDATION"
)


if len(
    main_interactions
) != len(
    GOVERNANCE_INDICATORS
):

    raise ValueError(
        "Main interaction output does not contain one row per "
        "governance indicator."
    )


if len(
    common_sample_interactions
) != len(
    GOVERNANCE_INDICATORS
):

    raise ValueError(
        "Common-sample interaction output is incomplete."
    )


if len(
    balanced_annual_interactions
) != len(
    GOVERNANCE_INDICATORS
):

    raise ValueError(
        "Indicator-specific balanced output is incomplete."
    )


if len(
    common_balanced_interactions
) != len(
    GOVERNANCE_INDICATORS
):

    raise ValueError(
        "Common balanced annual output is incomplete."
    )


expected_robustness_rows = (
    len(
        GOVERNANCE_INDICATORS
    )
    *
    4
)


if len(
    specification_robustness
) != expected_robustness_rows:

    raise ValueError(
        "Specification robustness output has an unexpected "
        "number of rows."
    )


expected_decomposition_rows = (
    len(
        GOVERNANCE_INDICATORS
    )
    *
    2
)


if len(
    between_within_results
) != expected_decomposition_rows:

    raise ValueError(
        "Between/within governance output is incomplete."
    )


if (
    common_balanced_annual_sample[
        "iso3c"
    ]
    .nunique()
    *
    len(
        ANNUAL_GOV_YEARS
    )
    !=
    len(
        common_balanced_annual_sample
    )
):

    raise ValueError(
        "Common balanced governance sample is not actually "
        "balanced."
    )


# ------------------------------------------------------------
# MASTER COEFFICIENT TABLE VALIDATION
#
# Per indicator:
#
# Main sample:
#   Common-slope model                    = 1 row
#   Additive-governance model             = 2 rows
#   Interaction model                     = 3 rows
#
# Other samples:
#   Common complete-case interaction      = 3 rows
#   Annual-era interaction                = 3 rows
#   Indicator-specific balanced model     = 3 rows
#   Common-balanced all-three model       = 3 rows
#   First-difference model                = 3 rows
#
# Total per indicator = 21
# 3 indicators × 21 = 63 rows.
# ------------------------------------------------------------

EXPECTED_MODEL_RESULT_ROWS = (
    len(
        GOVERNANCE_INDICATORS
    )
    *
    21
)


if len(
    model_results
) != EXPECTED_MODEL_RESULT_ROWS:

    raise ValueError(
        "governance_model_results.csv has an unexpected "
        "number of coefficient rows. "
        f"Expected {EXPECTED_MODEL_RESULT_ROWS}, "
        f"found {len(model_results)}."
    )


# Validate detailed interaction coefficient presence for every
# intended levels sample.
expected_levels_samples = [
    "Main available governance sample",
    "Common complete-case governance sample",
    "Annual governance era 2002-2024",
    "Balanced annual governance era 2002-2024",
    (
        "Common balanced all-three governance "
        "sample 2002-2024"
    ),
]


for sample_name in expected_levels_samples:

    interaction_terms = (
        model_results[
            (
                model_results[
                    "sample"
                ]
                ==
                sample_name
            )
            &
            (
                model_results[
                    "term"
                ]
                ==
                "gdp_x_gov"
            )
        ]
    )


    if len(
        interaction_terms
    ) != len(
        GOVERNANCE_INDICATORS
    ):

        raise ValueError(
            "Master coefficient table does not contain exactly "
            "one governance interaction per indicator for: "
            f"{sample_name}"
        )


print(
    "PASS: governance source and merge checks completed."
)

print(
    "PASS: three baseline governance interactions estimated."
)

print(
    "PASS: Holm multiple-testing correction applied."
)

print(
    "PASS: common all-three balanced sample estimated."
)

print(
    "PASS: region-adjusted governance interactions estimated."
)

print(
    "PASS: quadratic-GDP governance interactions estimated."
)

print(
    "PASS: region + quadratic interactions estimated."
)

print(
    "PASS: between/within governance decomposition estimated."
)

print(
    "PASS: first-difference governance robustness estimated."
)

print(
    "PASS: governance values were used only when actually observed."
)

print(
    "PASS: governance_model_results.csv contains "
    f"{len(model_results)} expected coefficient rows."
)


# ============================================================
# 33. OUTPUT MANIFEST
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


print(
    f"Results saved to:\n{OUTPUT_DIR}"
)


print(
    "\nCore Script 12 files:"
)


core_files = [
    "governance_wdi_clean_panel.csv",
    "governance_merged_economy_panel.csv",
    "governance_source_indicator_audit.csv",
    "governance_source_year_coverage.csv",
    "governance_main_sample_year_coverage.csv",
    "governance_economy_coverage.csv",
    "governance_merge_audit.csv",
    "governance_descriptive_statistics.csv",
    "governance_correlations_pooled.csv",
    "governance_correlations_within_economy.csv",
    "governance_region_summary.csv",
    "governance_sample_summary.csv",
    "governance_model_results.csv",
    "governance_interaction_summary.csv",
    "governance_common_sample_interactions.csv",
    "governance_balanced_annual_interactions.csv",
    "governance_common_balanced_annual_interactions.csv",
    "governance_first_difference_results.csv",
    "governance_marginal_gdp_elasticities.csv",
    "governance_model_fit_comparison.csv",
]


for filename in core_files:

    print(
        f"- {filename}"
    )


print(
    "\nGovernance robustness files:"
)


robustness_files = [
    "governance_interaction_holm_adjusted.csv",
    "governance_specification_robustness.csv",
    "governance_region_adjusted_interactions.csv",
    "governance_quadratic_gdp_interactions.csv",
    "governance_region_quadratic_interactions.csv",
    "governance_specification_fit_comparison.csv",
    "governance_between_within_decomposition.csv",
]


for filename in robustness_files:

    print(
        f"- {filename}"
    )


print(
    "\nFigures:"
)


print(
    f"- {INTERACTION_FIGURE_FILE}"
)

print(
    f"- {MARGINAL_FIGURE_FILE}"
)

print(
    f"- {SENSITIVITY_FIGURE_FILE}"
)


print(
    "\nFull Statsmodels summaries are stored locally under:"
)

print(
    SUMMARY_DIR
)


print(
    "\nSCRIPT 12 COMPLETE."
)
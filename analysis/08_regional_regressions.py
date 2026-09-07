from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "region_year_panel.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "regression_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

EXPECTED_REGIONS = [
    "Europe_NorthAmerica",
    "DevelopedAsia_Oceania",
    "China",
    "India",
    "Global_South",
]

# Levels regressions use two Newey-West lags.
HAC_LAGS_LEVELS = 2

# First-difference regressions use one lag.
HAC_LAGS_DIFF = 1

# Significance threshold used only for the preliminary EKC screen.
SIGNIFICANCE_LEVEL = 0.05

# Confidence intervals
ALPHA = 0.05


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(df, required_columns):
    """
    Check that all required variables are present.
    """
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def fit_hac_ols(
    y,
    X,
    maxlags,
):
    """
    Estimate OLS with Newey-West/HAC standard errors.

    Improvements over the original specification:
    - HAC correction for heteroskedasticity and serial correlation
    - small-sample covariance correction
    - Student-t inference rather than asymptotic normal inference
    """

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "use_correction": True,
        },
        use_t=True,
    )

    return model


def fit_hac_panel_ols(
    y,
    X,
    groups,
    maxlags,
):
    """
    Panel-HAC model for the pooled regional benchmark.

    This is preferable to treating the 175 stacked region-year
    observations as one continuous time series.

    The pooled model remains a benchmark rather than the main
    inferential specification.
    """

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="hac-panel",
        cov_kwds={
            "groups": groups,
            "maxlags": maxlags,
            "use_correction": "hac",
            "df_correction": False,
        },
        use_t=True,
    )

    return model


def confidence_interval(
    model,
    parameter,
    alpha=ALPHA,
):
    """
    Return lower and upper confidence limits for one parameter.
    """

    ci = model.conf_int(alpha=alpha)

    lower = float(
        ci.loc[parameter, 0]
    )

    upper = float(
        ci.loc[parameter, 1]
    )

    return lower, upper


def calculate_turning_point(
    beta1,
    beta2,
):
    """
    For:

        log(CO2) =
        a + beta1*log(GDP) + beta2*log(GDP)^2

    turning point in log GDP is:

        -beta1 / (2*beta2)

    GDP turning point is its exponential.
    """

    if np.isclose(beta2, 0.0):
        return np.nan, np.nan

    turning_point_log = (
        -beta1
        / (2 * beta2)
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        turning_point_gdp = float(
            np.exp(turning_point_log)
        )

    return (
        float(turning_point_log),
        turning_point_gdp,
    )


def save_model_summary(
    model,
    filename,
):
    """
    Save full statsmodels output as text.
    """

    output_file = (
        OUTPUT_DIR
        / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            model.summary().as_text()
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print_section("LOADING REGIONAL PANEL")

df = pd.read_csv(DATA_FILE)

print(
    f"Loaded: {DATA_FILE}"
)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# 5. VALIDATE DATA FOR REGRESSION
# ============================================================

required_columns = [
    "region",
    "year",
    "gdp_pc_region",
    "co2_pc_region",
]

require_columns(
    df,
    required_columns,
)


# Missing values
if df[
    required_columns
].isna().any().any():

    raise ValueError(
        "Regression variables contain missing values."
    )


# Positive values required for logarithms
if (
    df["gdp_pc_region"] <= 0
).any():

    raise ValueError(
        "GDP per capita contains zero or negative values."
    )


if (
    df["co2_pc_region"] <= 0
).any():

    raise ValueError(
        "CO2 per capita contains zero or negative values."
    )


# Duplicated region-years
if df.duplicated(
    [
        "region",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate region-year observations found."
    )


# Standardise types
df = df.copy()

df["region"] = (
    df["region"]
    .astype(str)
    .str.strip()
)

df["year"] = pd.to_numeric(
    df["year"],
    errors="raise",
).astype(int)


# Check exact region labels
actual_regions = set(
    df["region"].unique()
)

expected_regions = set(
    EXPECTED_REGIONS
)

if actual_regions != expected_regions:

    raise ValueError(
        "\nRegion labels do not match expectations.\n"
        f"Expected: {sorted(expected_regions)}\n"
        f"Found: {sorted(actual_regions)}"
    )


# Sort panel
df = (
    df
    .sort_values(
        [
            "region",
            "year",
        ]
    )
    .reset_index(drop=True)
)


# Check that yearly series are continuous
for region, group in df.groupby("region"):

    years = (
        group["year"]
        .sort_values()
        .to_numpy()
    )

    gaps = np.diff(years)

    if (
        len(gaps) > 0
        and not np.all(gaps == 1)
    ):

        raise ValueError(
            f"{region} has gaps in its yearly series."
        )


print(
    "\nRegression panel validated successfully."
)

print(
    "\nObservations by region:"
)

print(
    df.groupby("region")["year"]
    .agg(
        [
            "min",
            "max",
            "count",
        ]
    )
)


# ============================================================
# 6. CREATE LOG VARIABLES
# ============================================================

df["log_gdp"] = np.log(
    df["gdp_pc_region"]
)

df["log_co2"] = np.log(
    df["co2_pc_region"]
)

df["log_gdp_sq"] = (
    df["log_gdp"] ** 2
)


# ============================================================
# 7. POOLED LOG-LOG BENCHMARK
# ============================================================

print_section(
    "POOLED REGIONAL LOG-LOG BENCHMARK"
)

# Encode each region as a panel group.
pooled_groups = pd.Categorical(
    df["region"],
    categories=EXPECTED_REGIONS,
    ordered=True,
).codes


pooled_linear = fit_hac_panel_ols(
    y=df["log_co2"],
    X=df[["log_gdp"]],
    groups=pooled_groups,
    maxlags=HAC_LAGS_LEVELS,
)


print(
    pooled_linear.summary()
)


pooled_ci_low, pooled_ci_high = (
    confidence_interval(
        pooled_linear,
        "log_gdp",
    )
)


pooled_results = pd.DataFrame(
    [
        {
            "model":
                "Pooled regional log-log benchmark",

            "beta_log_gdp":
                pooled_linear.params[
                    "log_gdp"
                ],

            "std_error":
                pooled_linear.bse[
                    "log_gdp"
                ],

            "ci95_low":
                pooled_ci_low,

            "ci95_high":
                pooled_ci_high,

            "p_value":
                pooled_linear.pvalues[
                    "log_gdp"
                ],

            "r_squared":
                pooled_linear.rsquared,

            "adjusted_r_squared":
                pooled_linear.rsquared_adj,

            "aic":
                pooled_linear.aic,

            "bic":
                pooled_linear.bic,

            "n_obs":
                int(
                    pooled_linear.nobs
                ),

            "covariance":
                (
                    "HAC-panel, "
                    f"{HAC_LAGS_LEVELS} lags"
                ),
        }
    ]
)


pooled_results.to_csv(
    OUTPUT_DIR
    / "pooled_loglog_results.csv",
    index=False,
)


save_model_summary(
    pooled_linear,
    "pooled_loglog_summary.txt",
)


# ============================================================
# 8. REGION-SPECIFIC LINEAR LOG-LOG MODELS
# ============================================================

print_section(
    "REGION-SPECIFIC LOG-LOG MODELS"
)


linear_results = []


for region in EXPECTED_REGIONS:

    subset = (
        df[
            df["region"] == region
        ]
        .sort_values("year")
        .copy()
    )


    model = fit_hac_ols(
        y=subset["log_co2"],
        X=subset[
            [
                "log_gdp",
            ]
        ],
        maxlags=HAC_LAGS_LEVELS,
    )


    beta = float(
        model.params[
            "log_gdp"
        ]
    )

    std_error = float(
        model.bse[
            "log_gdp"
        ]
    )

    p_value = float(
        model.pvalues[
            "log_gdp"
        ]
    )


    ci_low, ci_high = (
        confidence_interval(
            model,
            "log_gdp",
        )
    )


    linear_results.append(
        {
            "region":
                region,

            "beta_log_gdp":
                beta,

            "std_error":
                std_error,

            "ci95_low":
                ci_low,

            "ci95_high":
                ci_high,

            "p_value":
                p_value,

            "r_squared":
                model.rsquared,

            "adjusted_r_squared":
                model.rsquared_adj,

            "aic":
                model.aic,

            "bic":
                model.bic,

            "n_obs":
                int(
                    model.nobs
                ),

            "first_year":
                int(
                    subset[
                        "year"
                    ].min()
                ),

            "last_year":
                int(
                    subset[
                        "year"
                    ].max()
                ),

            "covariance":
                (
                    f"HAC({HAC_LAGS_LEVELS}) "
                    "+ small-sample correction "
                    "+ t inference"
                ),
        }
    )


    print(
        f"\n--- {region} ---"
    )

    print(
        model.summary()
    )


    save_model_summary(
        model,
        f"loglog_{region}_summary.txt",
    )


linear_results_df = pd.DataFrame(
    linear_results
)


linear_results_df.to_csv(
    OUTPUT_DIR
    / "regional_loglog_results.csv",
    index=False,
)


# ============================================================
# 9. REGION-SPECIFIC QUADRATIC EKC MODELS
# ============================================================

print_section(
    "REGION-SPECIFIC QUADRATIC EKC MODELS"
)


ekc_results = []


for region in EXPECTED_REGIONS:

    subset = (
        df[
            df["region"] == region
        ]
        .sort_values("year")
        .copy()
    )


    model = fit_hac_ols(
        y=subset["log_co2"],
        X=subset[
            [
                "log_gdp",
                "log_gdp_sq",
            ]
        ],
        maxlags=HAC_LAGS_LEVELS,
    )


    beta1 = float(
        model.params[
            "log_gdp"
        ]
    )

    beta2 = float(
        model.params[
            "log_gdp_sq"
        ]
    )


    beta1_se = float(
        model.bse[
            "log_gdp"
        ]
    )

    beta2_se = float(
        model.bse[
            "log_gdp_sq"
        ]
    )


    beta1_p = float(
        model.pvalues[
            "log_gdp"
        ]
    )

    beta2_p = float(
        model.pvalues[
            "log_gdp_sq"
        ]
    )


    beta1_ci_low, beta1_ci_high = (
        confidence_interval(
            model,
            "log_gdp",
        )
    )


    beta2_ci_low, beta2_ci_high = (
        confidence_interval(
            model,
            "log_gdp_sq",
        )
    )


    # --------------------------------------------------------
    # Observed GDP range
    # --------------------------------------------------------

    observed_min_gdp = float(
        subset[
            "gdp_pc_region"
        ].min()
    )

    observed_max_gdp = float(
        subset[
            "gdp_pc_region"
        ].max()
    )


    observed_min_log_gdp = float(
        subset[
            "log_gdp"
        ].min()
    )

    observed_max_log_gdp = float(
        subset[
            "log_gdp"
        ].max()
    )


    # --------------------------------------------------------
    # EKC turning point
    # --------------------------------------------------------

    (
        turning_point_log_gdp,
        turning_point_gdp,
    ) = calculate_turning_point(
        beta1,
        beta2,
    )


    turning_point_inside_sample = bool(
        np.isfinite(
            turning_point_gdp
        )
        and
        observed_min_gdp
        <= turning_point_gdp
        <= observed_max_gdp
    )


    # --------------------------------------------------------
    # Position of turning point within observed log-income
    # range.
    #
    # 0% = lower boundary
    # 100% = upper boundary
    # --------------------------------------------------------

    if (
        np.isfinite(
            turning_point_log_gdp
        )
        and
        observed_max_log_gdp
        >
        observed_min_log_gdp
    ):

        turning_point_position_pct = (
            (
                turning_point_log_gdp
                -
                observed_min_log_gdp
            )
            /
            (
                observed_max_log_gdp
                -
                observed_min_log_gdp
            )
            *
            100
        )

    else:

        turning_point_position_pct = np.nan


    # --------------------------------------------------------
    # Number of actual observations on each side
    # --------------------------------------------------------

    if np.isfinite(
        turning_point_gdp
    ):

        n_below = int(
            (
                subset[
                    "gdp_pc_region"
                ]
                <
                turning_point_gdp
            ).sum()
        )

        n_above = int(
            (
                subset[
                    "gdp_pc_region"
                ]
                >
                turning_point_gdp
            ).sum()
        )

    else:

        n_below = 0
        n_above = 0


    n_obs = len(
        subset
    )


    share_below = (
        n_below
        /
        n_obs
    )

    share_above = (
        n_above
        /
        n_obs
    )


    # --------------------------------------------------------
    # Marginal GDP elasticity at observed income endpoints
    #
    # In the quadratic log-log model:
    #
    # elasticity =
    # beta1 + 2*beta2*log(GDP)
    # --------------------------------------------------------

    elasticity_at_min_gdp = (
        beta1
        +
        2
        *
        beta2
        *
        np.log(
            observed_min_gdp
        )
    )


    elasticity_at_max_gdp = (
        beta1
        +
        2
        *
        beta2
        *
        np.log(
            observed_max_gdp
        )
    )


    # --------------------------------------------------------
    # Preliminary EKC screen
    # --------------------------------------------------------

    ekc_sign_pattern = bool(
        beta1 > 0
        and
        beta2 < 0
    )


    within_sample_ekc_candidate = bool(
        ekc_sign_pattern
        and
        beta2_p
        <
        SIGNIFICANCE_LEVEL
        and
        turning_point_inside_sample
    )


    ekc_results.append(
        {
            "region":
                region,

            "beta1_log_gdp":
                beta1,

            "beta1_std_error":
                beta1_se,

            "beta1_ci95_low":
                beta1_ci_low,

            "beta1_ci95_high":
                beta1_ci_high,

            "beta1_p_value":
                beta1_p,

            "beta2_log_gdp_sq":
                beta2,

            "beta2_std_error":
                beta2_se,

            "beta2_ci95_low":
                beta2_ci_low,

            "beta2_ci95_high":
                beta2_ci_high,

            "beta2_p_value":
                beta2_p,

            "r_squared":
                model.rsquared,

            "adjusted_r_squared":
                model.rsquared_adj,

            "aic":
                model.aic,

            "bic":
                model.bic,

            "n_obs":
                int(
                    model.nobs
                ),

            "observed_min_gdp":
                observed_min_gdp,

            "observed_max_gdp":
                observed_max_gdp,

            "elasticity_at_min_gdp":
                elasticity_at_min_gdp,

            "elasticity_at_max_gdp":
                elasticity_at_max_gdp,

            "turning_point_log_gdp":
                turning_point_log_gdp,

            "turning_point_gdp":
                turning_point_gdp,

            "turning_point_inside_sample":
                turning_point_inside_sample,

            "turning_point_position_pct_of_log_income_range":
                turning_point_position_pct,

            "n_obs_below_turning_point":
                n_below,

            "n_obs_above_turning_point":
                n_above,

            "share_obs_below_turning_point":
                share_below,

            "share_obs_above_turning_point":
                share_above,

            "ekc_sign_pattern":
                ekc_sign_pattern,

            "within_sample_ekc_candidate":
                within_sample_ekc_candidate,

            "covariance":
                (
                    f"HAC({HAC_LAGS_LEVELS}) "
                    "+ small-sample correction "
                    "+ t inference"
                ),
        }
    )


    print(
        f"\n--- {region} ---"
    )

    print(
        model.summary()
    )


    print(
        "\nTurning point GDP:"
    )

    print(
        turning_point_gdp
    )


    print(
        "Turning point inside sample:"
    )

    print(
        turning_point_inside_sample
    )


    print(
        "Turning-point position "
        "within log-income range (%):"
    )

    print(
        turning_point_position_pct
    )


    print(
        "Observations below / above "
        "turning point:"
    )

    print(
        n_below,
        "/",
        n_above,
    )


    print(
        "Elasticity at minimum GDP:"
    )

    print(
        elasticity_at_min_gdp
    )


    print(
        "Elasticity at maximum GDP:"
    )

    print(
        elasticity_at_max_gdp
    )


    print(
        "Within-sample EKC candidate:"
    )

    print(
        within_sample_ekc_candidate
    )


    save_model_summary(
        model,
        f"ekc_{region}_summary.txt",
    )


ekc_results_df = pd.DataFrame(
    ekc_results
)


ekc_results_df.to_csv(
    OUTPUT_DIR
    / "regional_ekc_results.csv",
    index=False,
)


# ============================================================
# 10. LINEAR VS QUADRATIC MODEL COMPARISON
# ============================================================

print_section(
    "LINEAR VS EKC MODEL COMPARISON"
)


linear_comparison = (
    linear_results_df[
        [
            "region",
            "r_squared",
            "adjusted_r_squared",
            "aic",
            "bic",
            "n_obs",
        ]
    ]
    .rename(
        columns={
            "r_squared":
                "linear_r_squared",

            "adjusted_r_squared":
                "linear_adjusted_r_squared",

            "aic":
                "linear_aic",

            "bic":
                "linear_bic",
        }
    )
)


ekc_comparison = (
    ekc_results_df[
        [
            "region",
            "r_squared",
            "adjusted_r_squared",
            "aic",
            "bic",
            "within_sample_ekc_candidate",
        ]
    ]
    .rename(
        columns={
            "r_squared":
                "ekc_r_squared",

            "adjusted_r_squared":
                "ekc_adjusted_r_squared",

            "aic":
                "ekc_aic",

            "bic":
                "ekc_bic",
        }
    )
)


comparison = (
    linear_comparison
    .merge(
        ekc_comparison,
        on="region",
        how="left",
    )
)


comparison[
    "delta_adjusted_r_squared"
] = (
    comparison[
        "ekc_adjusted_r_squared"
    ]
    -
    comparison[
        "linear_adjusted_r_squared"
    ]
)


# Negative delta AIC/BIC favours the EKC model.
comparison[
    "delta_aic_ekc_minus_linear"
] = (
    comparison[
        "ekc_aic"
    ]
    -
    comparison[
        "linear_aic"
    ]
)


comparison[
    "delta_bic_ekc_minus_linear"
] = (
    comparison[
        "ekc_bic"
    ]
    -
    comparison[
        "linear_bic"
    ]
)


comparison.to_csv(
    OUTPUT_DIR
    / "linear_vs_ekc_comparison.csv",
    index=False,
)


# ============================================================
# 11. FIRST-DIFFERENCE ROBUSTNESS REGRESSIONS
# ============================================================

print_section(
    "FIRST-DIFFERENCE ROBUSTNESS MODELS"
)


first_difference_results = []


for region in EXPECTED_REGIONS:

    subset = (
        df[
            df["region"] == region
        ]
        .sort_values("year")
        .copy()
    )


    # --------------------------------------------------------
    # Approximate annual percentage changes:
    #
    # Δlog(GDP)
    # Δlog(CO2)
    # --------------------------------------------------------

    subset[
        "d_log_gdp"
    ] = (
        subset[
            "log_gdp"
        ]
        .diff()
    )


    subset[
        "d_log_co2"
    ] = (
        subset[
            "log_co2"
        ]
        .diff()
    )


    diff_data = (
        subset
        .dropna(
            subset=[
                "d_log_gdp",
                "d_log_co2",
            ]
        )
        .copy()
    )


    model = fit_hac_ols(
        y=diff_data[
            "d_log_co2"
        ],
        X=diff_data[
            [
                "d_log_gdp",
            ]
        ],
        maxlags=HAC_LAGS_DIFF,
    )


    beta = float(
        model.params[
            "d_log_gdp"
        ]
    )

    std_error = float(
        model.bse[
            "d_log_gdp"
        ]
    )

    p_value = float(
        model.pvalues[
            "d_log_gdp"
        ]
    )


    ci_low, ci_high = (
        confidence_interval(
            model,
            "d_log_gdp",
        )
    )


    first_difference_results.append(
        {
            "region":
                region,

            "beta_dlog_gdp":
                beta,

            "std_error":
                std_error,

            "ci95_low":
                ci_low,

            "ci95_high":
                ci_high,

            "p_value":
                p_value,

            "r_squared":
                model.rsquared,

            "adjusted_r_squared":
                model.rsquared_adj,

            "aic":
                model.aic,

            "bic":
                model.bic,

            "n_obs":
                int(
                    model.nobs
                ),

            "first_difference_start_year":
                int(
                    diff_data[
                        "year"
                    ].min()
                ),

            "last_year":
                int(
                    diff_data[
                        "year"
                    ].max()
                ),

            "covariance":
                (
                    f"HAC({HAC_LAGS_DIFF}) "
                    "+ small-sample correction "
                    "+ t inference"
                ),
        }
    )


    print(
        f"\n--- {region} ---"
    )

    print(
        model.summary()
    )


    save_model_summary(
        model,
        f"first_difference_{region}_summary.txt",
    )


first_difference_results_df = pd.DataFrame(
    first_difference_results
)


first_difference_results_df.to_csv(
    OUTPUT_DIR
    / "regional_first_difference_results.csv",
    index=False,
)


# ============================================================
# 12. CLEAN TERMINAL SUMMARY
# ============================================================

print_section(
    "LINEAR LOG-LOG RESULTS"
)


linear_display = (
    linear_results_df[
        [
            "region",
            "beta_log_gdp",
            "std_error",
            "ci95_low",
            "ci95_high",
            "p_value",
            "adjusted_r_squared",
            "n_obs",
        ]
    ]
    .copy()
)


print(
    linear_display
    .round(4)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------

print_section(
    "EKC RESULTS"
)


ekc_display = (
    ekc_results_df[
        [
            "region",
            "beta1_log_gdp",
            "beta1_p_value",
            "beta2_log_gdp_sq",
            "beta2_p_value",
            "adjusted_r_squared",
            "turning_point_gdp",
            "turning_point_inside_sample",
            "turning_point_position_pct_of_log_income_range",
            "elasticity_at_min_gdp",
            "elasticity_at_max_gdp",
            "within_sample_ekc_candidate",
        ]
    ]
    .copy()
)


print(
    ekc_display
    .round(4)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------

print_section(
    "LINEAR VS EKC COMPARISON"
)


print(
    comparison
    .round(4)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------

print_section(
    "FIRST-DIFFERENCE ROBUSTNESS RESULTS"
)


first_difference_display = (
    first_difference_results_df[
        [
            "region",
            "beta_dlog_gdp",
            "std_error",
            "ci95_low",
            "ci95_high",
            "p_value",
            "adjusted_r_squared",
            "n_obs",
        ]
    ]
    .copy()
)


print(
    first_difference_display
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. OUTPUT COMPLETE
# ============================================================

print_section(
    "OUTPUT COMPLETE"
)


print(
    "Regression outputs saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nKey files:"
)

print(
    "- pooled_loglog_results.csv"
)

print(
    "- regional_loglog_results.csv"
)

print(
    "- regional_ekc_results.csv"
)

print(
    "- linear_vs_ekc_comparison.csv"
)

print(
    "- regional_first_difference_results.csv"
)

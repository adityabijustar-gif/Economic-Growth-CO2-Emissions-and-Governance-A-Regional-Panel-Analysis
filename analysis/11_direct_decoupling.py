from pathlib import Path

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

REGION_PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "region_year_panel.csv"
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
    / "decoupling_results"
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


MIN_YEARS_PER_ECONOMY = 2


# Numerical tolerance only. This is NOT an economically meaningful
# threshold; it only prevents floating-point noise around zero from
# changing a classification.
LOG_CHANGE_TOL = 1e-12


# Number of largest absolute annual CO2-per-capita percentage changes
# to export for manual source-data inspection.
OUTLIER_TOP_N = 50


# Pre-specified endpoint comparisons. These periods are descriptive and
# are not chosen according to whichever dates produce the strongest
# result.
LONG_PERIODS = [
    {
        "period":
            "1990-2004",

        "start_year":
            1990,

        "end_year":
            2004,
    },

    {
        "period":
            "2005-2014",

        "start_year":
            2005,

        "end_year":
            2014,
    },

    {
        "period":
            "2015-2024",

        "start_year":
            2015,

        "end_year":
            2024,
    },

    {
        "period":
            "1990-2024",

        "start_year":
            1990,

        "end_year":
            2024,
    },
]


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
    Verify that required variables are present.
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


def classify_change(
    d_log_gdp,
    d_log_co2,
):
    """
    Classify one observed per-capita GDP-emissions change.

    Positive-GDP-growth observations are divided into:

    1. Absolute decoupling
       GDP per capita rises and CO2 per capita falls.

    2. Relative decoupling
       GDP per capita rises and CO2 per capita is non-negative
       but rises more slowly than GDP per capita.

    3. Coupled expansion
       GDP per capita rises and CO2 per capita rises at least as
       quickly as GDP per capita.

    GDP contractions and approximately zero-GDP-growth periods
    are classified separately so declining emissions during a
    recession are not incorrectly described as decoupling.
    """

    tol = LOG_CHANGE_TOL

    if d_log_gdp > tol:

        if d_log_co2 < -tol:

            return (
                "Absolute decoupling"
            )

        if (
            d_log_co2
            <
            d_log_gdp - tol
        ):

            return (
                "Relative decoupling"
            )

        return (
            "Coupled expansion"
        )

    if abs(
        d_log_gdp
    ) <= tol:

        if d_log_co2 < -tol:

            return (
                "Stable GDP, emissions falling"
            )

        if d_log_co2 > tol:

            return (
                "Stable GDP, emissions rising"
            )

        return (
            "Stable GDP and emissions"
        )

    if d_log_co2 < -tol:

        return (
            "Economic contraction, emissions falling"
        )

    if d_log_co2 > tol:

        return (
            "Economic contraction, emissions rising"
        )

    return (
        "Economic contraction, emissions stable"
    )


def add_change_metrics(
    df,
    gdp_change_column,
    co2_change_column,
    weight_column,
):
    """
    Add common decoupling indicators and diagnostic variables.

    The decoupling ratio is reported only for GDP-expansion
    observations and is NOT used as the primary classification
    rule.
    """

    df = df.copy()

    df[
        "decoupling_classification"
    ] = [
        classify_change(
            gdp_change,
            co2_change,
        )
        for (
            gdp_change,
            co2_change,
        )
        in zip(
            df[
                gdp_change_column
            ],
            df[
                co2_change_column
            ],
        )
    ]

    df[
        "gdp_expansion"
    ] = (
        df[
            gdp_change_column
        ]
        >
        LOG_CHANGE_TOL
    )

    df[
        "absolute_decoupling"
    ] = (
        df[
            "decoupling_classification"
        ]
        ==
        "Absolute decoupling"
    )

    df[
        "relative_decoupling"
    ] = (
        df[
            "decoupling_classification"
        ]
        ==
        "Relative decoupling"
    )

    df[
        "coupled_expansion"
    ] = (
        df[
            "decoupling_classification"
        ]
        ==
        "Coupled expansion"
    )

    df[
        "any_decoupling"
    ] = (
        df[
            "absolute_decoupling"
        ]
        |
        df[
            "relative_decoupling"
        ]
    )

    df[
        "gdp_change_pct"
    ] = (
        np.exp(
            df[
                gdp_change_column
            ]
        )
        -
        1
    ) * 100

    df[
        "co2_change_pct"
    ] = (
        np.exp(
            df[
                co2_change_column
            ]
        )
        -
        1
    ) * 100

    df[
        "decoupling_ratio"
    ] = np.where(
        df[
            "gdp_expansion"
        ],

        (
            df[
                co2_change_column
            ]
            /
            df[
                gdp_change_column
            ]
        ),

        np.nan,
    )

    df[
        "valid_population_weight"
    ] = (
        df[
            weight_column
        ]
        .notna()
        &
        (
            df[
                weight_column
            ]
            > 0
        )
    )

    return df


def summarise_decoupling(
    df,
    group_columns,
    weight_column,
):
    """
    Summarise decoupling outcomes within specified groups.

    Primary shares use GDP-expansion observations as the
    denominator.

    Both mean and MEDIAN GDP/CO2 changes are reported. Medians
    are preferred for descriptive interpretation because annual
    percentage changes can contain extreme low-base outliers.

    Population-weighted shares are also reported as robustness
    measures.
    """

    rows = []

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )
    )

    for keys, group in grouped:

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            column:
                value
            for (
                column,
                value,
            )
            in zip(
                group_columns,
                keys,
            )
        }

        expansion = (
            group[
                group[
                    "gdp_expansion"
                ]
            ]
        )

        n_total = len(
            group
        )

        n_expansion = len(
            expansion
        )

        n_absolute = int(
            expansion[
                "absolute_decoupling"
            ]
            .sum()
        )

        n_relative = int(
            expansion[
                "relative_decoupling"
            ]
            .sum()
        )

        n_coupled = int(
            expansion[
                "coupled_expansion"
            ]
            .sum()
        )

        if (
            n_absolute
            +
            n_relative
            +
            n_coupled
            !=
            n_expansion
        ):

            raise ValueError(
                "GDP-expansion classifications do not sum "
                "to the number of GDP-expansion observations."
            )

        row[
            "n_total_changes"
        ] = (
            n_total
        )

        row[
            "n_gdp_expansions"
        ] = (
            n_expansion
        )

        row[
            "n_absolute_decoupling"
        ] = (
            n_absolute
        )

        row[
            "n_relative_decoupling"
        ] = (
            n_relative
        )

        row[
            "n_coupled_expansion"
        ] = (
            n_coupled
        )

        row[
            "n_any_decoupling"
        ] = (
            n_absolute
            +
            n_relative
        )

        row[
            "share_gdp_expansion_of_all_changes"
        ] = (
            n_expansion
            /
            n_total
            if n_total > 0
            else np.nan
        )

        if n_expansion > 0:

            row[
                "share_absolute_of_expansions"
            ] = (
                n_absolute
                /
                n_expansion
            )

            row[
                "share_relative_of_expansions"
            ] = (
                n_relative
                /
                n_expansion
            )

            row[
                "share_coupled_of_expansions"
            ] = (
                n_coupled
                /
                n_expansion
            )

            row[
                "share_any_decoupling_of_expansions"
            ] = (
                (
                    n_absolute
                    +
                    n_relative
                )
                /
                n_expansion
            )

            row[
                "mean_gdp_change_pct_during_expansions"
            ] = (
                expansion[
                    "gdp_change_pct"
                ]
                .mean()
            )

            row[
                "median_gdp_change_pct_during_expansions"
            ] = (
                expansion[
                    "gdp_change_pct"
                ]
                .median()
            )

            row[
                "mean_co2_change_pct_during_expansions"
            ] = (
                expansion[
                    "co2_change_pct"
                ]
                .mean()
            )

            row[
                "median_co2_change_pct_during_expansions"
            ] = (
                expansion[
                    "co2_change_pct"
                ]
                .median()
            )

            # If the input is a longer-period endpoint table,
            # also report annualized growth statistics so periods
            # of different lengths can be compared more sensibly.
            if (
                "annualized_gdp_growth_pct"
                in expansion.columns
            ):

                row[
                    "mean_annualized_gdp_growth_pct_during_expansions"
                ] = (
                    expansion[
                        "annualized_gdp_growth_pct"
                    ]
                    .mean()
                )

                row[
                    "median_annualized_gdp_growth_pct_during_expansions"
                ] = (
                    expansion[
                        "annualized_gdp_growth_pct"
                    ]
                    .median()
                )

                row[
                    "mean_annualized_co2_growth_pct_during_expansions"
                ] = (
                    expansion[
                        "annualized_co2_growth_pct"
                    ]
                    .mean()
                )

                row[
                    "median_annualized_co2_growth_pct_during_expansions"
                ] = (
                    expansion[
                        "annualized_co2_growth_pct"
                    ]
                    .median()
                )

            weighted_expansion = (
                expansion[
                    expansion[
                        "valid_population_weight"
                    ]
                ]
            )

            weight_total = (
                weighted_expansion[
                    weight_column
                ]
                .sum()
            )

            if weight_total > 0:

                row[
                    "population_weighted_share_absolute"
                ] = (
                    weighted_expansion.loc[
                        weighted_expansion[
                            "absolute_decoupling"
                        ],
                        weight_column,
                    ]
                    .sum()
                    /
                    weight_total
                )

                row[
                    "population_weighted_share_relative"
                ] = (
                    weighted_expansion.loc[
                        weighted_expansion[
                            "relative_decoupling"
                        ],
                        weight_column,
                    ]
                    .sum()
                    /
                    weight_total
                )

                row[
                    "population_weighted_share_coupled"
                ] = (
                    weighted_expansion.loc[
                        weighted_expansion[
                            "coupled_expansion"
                        ],
                        weight_column,
                    ]
                    .sum()
                    /
                    weight_total
                )

                row[
                    "population_weighted_share_any_decoupling"
                ] = (
                    weighted_expansion.loc[
                        weighted_expansion[
                            "any_decoupling"
                        ],
                        weight_column,
                    ]
                    .sum()
                    /
                    weight_total
                )

            else:

                row[
                    "population_weighted_share_absolute"
                ] = np.nan

                row[
                    "population_weighted_share_relative"
                ] = np.nan

                row[
                    "population_weighted_share_coupled"
                ] = np.nan

                row[
                    "population_weighted_share_any_decoupling"
                ] = np.nan

        else:

            for column in [
                "share_absolute_of_expansions",
                "share_relative_of_expansions",
                "share_coupled_of_expansions",
                "share_any_decoupling_of_expansions",
                "mean_gdp_change_pct_during_expansions",
                "median_gdp_change_pct_during_expansions",
                "mean_co2_change_pct_during_expansions",
                "median_co2_change_pct_during_expansions",
                "population_weighted_share_absolute",
                "population_weighted_share_relative",
                "population_weighted_share_coupled",
                "population_weighted_share_any_decoupling",
            ]:

                row[
                    column
                ] = np.nan

            if (
                "annualized_gdp_growth_pct"
                in group.columns
            ):

                for column in [
                    "mean_annualized_gdp_growth_pct_during_expansions",
                    "median_annualized_gdp_growth_pct_during_expansions",
                    "mean_annualized_co2_growth_pct_during_expansions",
                    "median_annualized_co2_growth_pct_during_expansions",
                ]:

                    row[
                        column
                    ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_long_period_changes(
    sample,
    periods,
):
    """
    Construct pre-specified endpoint comparisons.

    An economy enters a period only if it has valid observations
    in BOTH the specified start year and end year.

    Intermediate observations are not required for an endpoint
    comparison.
    """

    output_rows = []

    for period in periods:

        period_name = (
            period[
                "period"
            ]
        )

        start_year = int(
            period[
                "start_year"
            ]
        )

        end_year = int(
            period[
                "end_year"
            ]
        )

        years_elapsed = (
            end_year
            -
            start_year
        )

        if years_elapsed <= 0:

            raise ValueError(
                f"Invalid period: {period_name}"
            )

        start = (
            sample[
                sample[
                    "year"
                ]
                == start_year
            ][
                [
                    "iso3c",
                    "country",
                    "region",
                    "log_gdp",
                    "log_co2",
                    "population",
                    "gdp_per_capita_const2015_usd",
                    "co2_per_capita_tons",
                ]
            ]
            .copy()
        )

        start = start.rename(
            columns={
                "log_gdp":
                    "log_gdp_start",

                "log_co2":
                    "log_co2_start",

                "population":
                    "population_start",

                "gdp_per_capita_const2015_usd":
                    "gdp_per_capita_start",

                "co2_per_capita_tons":
                    "co2_per_capita_start",
            }
        )

        end = (
            sample[
                sample[
                    "year"
                ]
                == end_year
            ][
                [
                    "iso3c",
                    "country",
                    "region",
                    "log_gdp",
                    "log_co2",
                    "population",
                    "gdp_per_capita_const2015_usd",
                    "co2_per_capita_tons",
                ]
            ]
            .copy()
        )

        end = end.rename(
            columns={
                "log_gdp":
                    "log_gdp_end",

                "log_co2":
                    "log_co2_end",

                "population":
                    "population_end",

                "gdp_per_capita_const2015_usd":
                    "gdp_per_capita_end",

                "co2_per_capita_tons":
                    "co2_per_capita_end",
            }
        )

        period_data = (
            start
            .merge(
                end,
                on=[
                    "iso3c",
                    "country",
                    "region",
                ],
                how="inner",
                validate="one_to_one",
            )
        )

        period_data[
            "period"
        ] = (
            period_name
        )

        period_data[
            "start_year"
        ] = (
            start_year
        )

        period_data[
            "end_year"
        ] = (
            end_year
        )

        period_data[
            "years_elapsed"
        ] = (
            years_elapsed
        )

        period_data[
            "d_log_gdp"
        ] = (
            period_data[
                "log_gdp_end"
            ]
            -
            period_data[
                "log_gdp_start"
            ]
        )

        period_data[
            "d_log_co2"
        ] = (
            period_data[
                "log_co2_end"
            ]
            -
            period_data[
                "log_co2_start"
            ]
        )

        period_data[
            "average_population"
        ] = (
            (
                period_data[
                    "population_start"
                ]
                +
                period_data[
                    "population_end"
                ]
            )
            /
            2
        )

        period_data = add_change_metrics(
            df=period_data,
            gdp_change_column="d_log_gdp",
            co2_change_column="d_log_co2",
            weight_column="average_population",
        )

        period_data[
            "annualized_gdp_growth_pct"
        ] = (
            np.exp(
                period_data[
                    "d_log_gdp"
                ]
                /
                years_elapsed
            )
            -
            1
        ) * 100

        period_data[
            "annualized_co2_growth_pct"
        ] = (
            np.exp(
                period_data[
                    "d_log_co2"
                ]
                /
                years_elapsed
            )
            -
            1
        ) * 100

        output_rows.append(
            period_data
        )

    return pd.concat(
        output_rows,
        ignore_index=True,
    )


def build_regional_long_period_changes(
    regional_panel,
    periods,
):
    """
    Construct the same endpoint analysis using the
    population-weighted regional aggregate panel.
    """

    output_rows = []

    for period in periods:

        period_name = (
            period[
                "period"
            ]
        )

        start_year = int(
            period[
                "start_year"
            ]
        )

        end_year = int(
            period[
                "end_year"
            ]
        )

        years_elapsed = (
            end_year
            -
            start_year
        )

        start = (
            regional_panel[
                regional_panel[
                    "year"
                ]
                == start_year
            ][
                [
                    "region",
                    "log_gdp_region",
                    "log_co2_region",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_gdp_region":
                        "log_gdp_start",

                    "log_co2_region":
                        "log_co2_start",
                }
            )
        )

        end = (
            regional_panel[
                regional_panel[
                    "year"
                ]
                == end_year
            ][
                [
                    "region",
                    "log_gdp_region",
                    "log_co2_region",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "log_gdp_region":
                        "log_gdp_end",

                    "log_co2_region":
                        "log_co2_end",
                }
            )
        )

        merged = (
            start
            .merge(
                end,
                on="region",
                how="inner",
                validate="one_to_one",
            )
        )

        merged[
            "period"
        ] = (
            period_name
        )

        merged[
            "start_year"
        ] = (
            start_year
        )

        merged[
            "end_year"
        ] = (
            end_year
        )

        merged[
            "years_elapsed"
        ] = (
            years_elapsed
        )

        merged[
            "d_log_gdp"
        ] = (
            merged[
                "log_gdp_end"
            ]
            -
            merged[
                "log_gdp_start"
            ]
        )

        merged[
            "d_log_co2"
        ] = (
            merged[
                "log_co2_end"
            ]
            -
            merged[
                "log_co2_start"
            ]
        )

        merged[
            "gdp_change_pct"
        ] = (
            np.exp(
                merged[
                    "d_log_gdp"
                ]
            )
            -
            1
        ) * 100

        merged[
            "co2_change_pct"
        ] = (
            np.exp(
                merged[
                    "d_log_co2"
                ]
            )
            -
            1
        ) * 100

        merged[
            "annualized_gdp_growth_pct"
        ] = (
            np.exp(
                merged[
                    "d_log_gdp"
                ]
                /
                years_elapsed
            )
            -
            1
        ) * 100

        merged[
            "annualized_co2_growth_pct"
        ] = (
            np.exp(
                merged[
                    "d_log_co2"
                ]
                /
                years_elapsed
            )
            -
            1
        ) * 100

        merged[
            "decoupling_classification"
        ] = [
            classify_change(
                d_gdp,
                d_co2,
            )
            for (
                d_gdp,
                d_co2,
            )
            in zip(
                merged[
                    "d_log_gdp"
                ],
                merged[
                    "d_log_co2"
                ],
            )
        ]

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

        output_rows.append(
            merged
        )

    return pd.concat(
        output_rows,
        ignore_index=True,
    )


# ============================================================
# 4. DECOUPLING METHODOLOGY TABLE
# ============================================================

print_section(
    "WRITING DECOUPLING METHODOLOGY"
)


methodology = pd.DataFrame(
    [
        {
            "classification":
                "Absolute decoupling",

            "gdp_condition":
                "d_log_gdp > 0",

            "co2_condition":
                "d_log_co2 < 0",

            "interpretation":
                (
                    "GDP per capita rises while CO2 emissions "
                    "per capita fall."
                ),
        },

        {
            "classification":
                "Relative decoupling",

            "gdp_condition":
                "d_log_gdp > 0",

            "co2_condition":
                "0 <= d_log_co2 < d_log_gdp",

            "interpretation":
                (
                    "GDP per capita and emissions per capita "
                    "both rise, but emissions rise more slowly "
                    "than GDP."
                ),
        },

        {
            "classification":
                "Coupled expansion",

            "gdp_condition":
                "d_log_gdp > 0",

            "co2_condition":
                "d_log_co2 >= d_log_gdp",

            "interpretation":
                (
                    "CO2 emissions per capita rise at least as "
                    "quickly as GDP per capita during expansion."
                ),
        },

        {
            "classification":
                "Economic contraction, emissions falling",

            "gdp_condition":
                "d_log_gdp < 0",

            "co2_condition":
                "d_log_co2 < 0",

            "interpretation":
                (
                    "GDP per capita and emissions per capita both "
                    "fall. This is not classified as successful "
                    "decoupling."
                ),
        },

        {
            "classification":
                "Economic contraction, emissions rising",

            "gdp_condition":
                "d_log_gdp < 0",

            "co2_condition":
                "d_log_co2 > 0",

            "interpretation":
                (
                    "GDP per capita falls while emissions per "
                    "capita rise."
                ),
        },

        {
            "classification":
                "Stable-GDP cases",

            "gdp_condition":
                "approximately zero GDP change",

            "co2_condition":
                "classified according to emissions sign",

            "interpretation":
                (
                    "Separated from expansion-based decoupling "
                    "categories."
                ),
        },
    ]
)


methodology.to_csv(
    OUTPUT_DIR
    / "decoupling_methodology.csv",
    index=False,
)


# ============================================================
# 5. LOAD SOURCE DATA
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


regional_panel = pd.read_csv(
    REGION_PANEL_FILE
)


print(
    f"Economy-year source rows: "
    f"{len(panel):,}"
)

print(
    f"Economy-region mappings: "
    f"{len(region_map):,}"
)

print(
    f"Region-year rows: "
    f"{len(regional_panel):,}"
)


# ============================================================
# 6. VALIDATE REQUIRED COLUMNS
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
    "Economy-year panel",
)


require_columns(
    region_map,
    [
        "iso3c",
        "region",
    ],
    "Economy-region mapping",
)


require_columns(
    regional_panel,
    [
        "region",
        "year",
        "gdp_pc_region",
        "co2_pc_region",
    ],
    "Region-year panel",
)


# ============================================================
# 7. STANDARDISE SOURCE DATA
# ============================================================

panel = panel.copy()

region_map = region_map.copy()

regional_panel = regional_panel.copy()


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


regional_panel[
    "region"
] = (
    regional_panel[
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


regional_panel[
    "year"
] = (
    pd.to_numeric(
        regional_panel[
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


for column in [
    "gdp_pc_region",
    "co2_pc_region",
]:

    regional_panel[
        column
    ] = pd.to_numeric(
        regional_panel[
            column
        ],
        errors="coerce",
    )


# ============================================================
# 8. VALIDATE KEYS AND REGIONAL LABELS
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


if regional_panel.duplicated(
    [
        "region",
        "year",
    ]
).any():

    raise ValueError(
        "Duplicate region-year observations found."
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
        "Economy-region mapping does not contain exactly "
        "the expected development clusters."
    )


if set(
    regional_panel[
        "region"
    ]
    .unique()
) != set(
    EXPECTED_REGIONS
):

    raise ValueError(
        "Regional panel does not contain exactly the expected "
        "development clusters."
    )


print(
    "PASS: keys and regional labels are valid."
)


# ============================================================
# 9. RECONSTRUCT SCRIPT 09 MAIN ECONOMY SAMPLE
# ============================================================

print_section(
    "RECONSTRUCTING MAIN ECONOMY SAMPLE"
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


print(
    f"Main economy-year observations: "
    f"{len(main_sample):,}"
)

print(
    f"Main economies: "
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


SCRIPT09_FD_FILE = (
    SCRIPT09_RESULTS_DIR
    / "country_panel_first_difference_results.csv"
)


if SCRIPT09_FE_FILE.exists():

    script09_fe = pd.read_csv(
        SCRIPT09_FE_FILE
    )


    script09_main = (
        script09_fe[
            script09_fe[
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
            "Could not uniquely identify Script 09 main model."
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
            "Script 11 observation count does not match "
            "Script 09."
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
            "Script 11 economy count does not match Script 09."
        )


    print(
        "PASS: levels sample matches Script 09."
    )


else:

    print(
        "WARNING: Script 09 levels-result file not found."
    )


# ============================================================
# 11. CONSTRUCT ANNUAL ECONOMY-LEVEL CHANGES
# ============================================================

print_section(
    "CONSTRUCTING ANNUAL ECONOMY-LEVEL CHANGES"
)


annual = (
    main_sample
    .sort_values(
        [
            "iso3c",
            "year",
        ]
    )
    .copy()
)


annual[
    "year_from"
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
    "log_gdp_previous"
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
    "log_co2_previous"
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
    "gdp_per_capita_previous"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "gdp_per_capita_const2015_usd"
    ]
    .shift(1)
)


annual[
    "co2_per_capita_previous"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "co2_per_capita_tons"
    ]
    .shift(1)
)


annual[
    "population_previous"
] = (
    annual
    .groupby(
        "iso3c"
    )[
        "population"
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
        "year_from"
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
        "log_gdp_previous"
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
        "log_co2_previous"
    ]
)


annual = (
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


annual[
    "year_from"
] = (
    annual[
        "year_from"
    ]
    .astype(int)
)


annual = annual.rename(
    columns={
        "year":
            "year_to"
    }
)


annual[
    "average_population"
] = (
    (
        annual[
            "population_previous"
        ]
        +
        annual[
            "population"
        ]
    )
    /
    2
)


annual = add_change_metrics(
    df=annual,
    gdp_change_column="d_log_gdp",
    co2_change_column="d_log_co2",
    weight_column="average_population",
)


annual[
    "region_display"
] = (
    annual[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


# ============================================================
# 12. CROSS-CHECK FIRST-DIFFERENCE SAMPLE AGAINST SCRIPT 09
# ============================================================

if SCRIPT09_FD_FILE.exists():

    script09_fd = pd.read_csv(
        SCRIPT09_FD_FILE
    )


    if len(
        script09_fd
    ) != 1:

        raise ValueError(
            "Unexpected number of Script 09 first-difference "
            "result rows."
        )


    script09_fd = (
        script09_fd.iloc[0]
    )


    if (
        len(
            annual
        )
        !=
        int(
            script09_fd[
                "n_obs"
            ]
        )
    ):

        raise ValueError(
            "Annual decoupling pair count does not match "
            "Script 09 first-difference sample."
        )


    if (
        annual[
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
            "Annual decoupling economy count does not match "
            "Script 09 first-difference sample."
        )


    print(
        "PASS: annual decoupling sample exactly matches "
        "Script 09 consecutive-year sample."
    )


print(
    f"Annual consecutive changes: "
    f"{len(annual):,}"
)

print(
    f"Economies represented: "
    f"{annual['iso3c'].nunique():,}"
)


# ============================================================
# 13. VALIDATE ANNUAL CLASSIFICATION EXHAUSTIVENESS
# ============================================================

if annual[
    "decoupling_classification"
].isna().any():

    raise ValueError(
        "Some annual observations were not classified."
    )


annual_expansion = (
    annual[
        annual[
            "gdp_expansion"
        ]
    ]
)


if (
    annual_expansion[
        "absolute_decoupling"
    ]
    .sum()
    +
    annual_expansion[
        "relative_decoupling"
    ]
    .sum()
    +
    annual_expansion[
        "coupled_expansion"
    ]
    .sum()
    !=
    len(
        annual_expansion
    )
):

    raise ValueError(
        "Annual GDP-expansion classifications are not "
        "mutually exhaustive."
    )


print(
    "PASS: annual decoupling classifications are exhaustive."
)


# ============================================================
# 14. EXPORT ANNUAL ECONOMY-LEVEL EVENTS
# ============================================================

annual_output_columns = [
    "iso3c",
    "country",
    "region",
    "region_display",
    "year_from",
    "year_to",
    "gdp_per_capita_previous",
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_previous",
    "co2_per_capita_tons",
    "d_log_gdp",
    "d_log_co2",
    "gdp_change_pct",
    "co2_change_pct",
    "decoupling_ratio",
    "decoupling_classification",
    "gdp_expansion",
    "absolute_decoupling",
    "relative_decoupling",
    "coupled_expansion",
    "any_decoupling",
    "average_population",
]


annual[
    annual_output_columns
].to_csv(
    OUTPUT_DIR
    / "annual_economy_decoupling.csv",
    index=False,
)


# ============================================================
# 15. ANNUAL CO2 OUTLIER AUDIT
# ============================================================

print_section(
    "ANNUAL CO2 CHANGE OUTLIER AUDIT"
)


outlier_audit = (
    annual[
        [
            "iso3c",
            "country",
            "region",
            "region_display",
            "year_from",
            "year_to",
            "gdp_per_capita_previous",
            "gdp_per_capita_const2015_usd",
            "co2_per_capita_previous",
            "co2_per_capita_tons",
            "gdp_change_pct",
            "co2_change_pct",
            "decoupling_classification",
        ]
    ]
    .copy()
)


outlier_audit[
    "abs_co2_change_pct"
] = (
    outlier_audit[
        "co2_change_pct"
    ]
    .abs()
)


outlier_audit[
    "co2_previous_below_0_1_tons"
] = (
    outlier_audit[
        "co2_per_capita_previous"
    ]
    < 0.1
)


outlier_audit = (
    outlier_audit
    .sort_values(
        "abs_co2_change_pct",
        ascending=False,
    )
    .head(
        OUTLIER_TOP_N
    )
    .reset_index(
        drop=True
    )
)


outlier_audit.insert(
    0,
    "outlier_rank",
    np.arange(
        1,
        len(
            outlier_audit
        )
        + 1,
    ),
)


outlier_audit.to_csv(
    OUTPUT_DIR
    / "annual_co2_change_outlier_audit.csv",
    index=False,
)


print(
    f"Exported top {len(outlier_audit)} annual absolute CO2 "
    "percentage changes for manual inspection."
)


print(
    outlier_audit[
        [
            "outlier_rank",
            "country",
            "year_from",
            "year_to",
            "co2_per_capita_previous",
            "co2_per_capita_tons",
            "co2_change_pct",
        ]
    ]
    .head(10)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. ANNUAL ECONOMY SUMMARY
# ============================================================

print_section(
    "ANNUAL DECOUPLING SUMMARY BY ECONOMY"
)


economy_summary = summarise_decoupling(
    df=annual,
    group_columns=[
        "iso3c",
        "country",
        "region",
        "region_display",
    ],
    weight_column="average_population",
)


# Determine each economy's most common expansion category.
dominant_rows = []


for (
    iso3c,
    country,
    region,
    region_display,
), group in (
    annual[
        annual[
            "gdp_expansion"
        ]
    ]
    .groupby(
        [
            "iso3c",
            "country",
            "region",
            "region_display",
        ]
    )
):

    counts = (
        group[
            "decoupling_classification"
        ]
        .value_counts()
    )


    if counts.empty:

        dominant = np.nan

    else:

        maximum = (
            counts.max()
        )

        tied = (
            counts[
                counts
                == maximum
            ]
            .index
            .tolist()
        )

        dominant = (
            tied[0]
            if len(
                tied
            )
            == 1
            else "Tie"
        )


    dominant_rows.append(
        {
            "iso3c":
                iso3c,

            "country":
                country,

            "region":
                region,

            "region_display":
                region_display,

            "dominant_expansion_classification":
                dominant,
        }
    )


dominant_df = pd.DataFrame(
    dominant_rows
)


economy_summary = (
    economy_summary
    .merge(
        dominant_df,
        on=[
            "iso3c",
            "country",
            "region",
            "region_display",
        ],
        how="left",
        validate="one_to_one",
    )
)


economy_summary.to_csv(
    OUTPUT_DIR
    / "annual_decoupling_economy_summary.csv",
    index=False,
)


# ============================================================
# 17. ANNUAL REGIONAL SUMMARY
# ============================================================

region_summary = summarise_decoupling(
    df=annual,
    group_columns=[
        "region",
        "region_display",
    ],
    weight_column="average_population",
)


region_summary.to_csv(
    OUTPUT_DIR
    / "annual_decoupling_region_summary.csv",
    index=False,
)


print(
    region_summary[
        [
            "region_display",
            "n_gdp_expansions",
            "share_absolute_of_expansions",
            "share_relative_of_expansions",
            "share_coupled_of_expansions",
            "share_any_decoupling_of_expansions",
            "median_gdp_change_pct_during_expansions",
            "median_co2_change_pct_during_expansions",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. ANNUAL REGION-YEAR SUMMARY
# ============================================================

region_year_summary = summarise_decoupling(
    df=annual,
    group_columns=[
        "region",
        "region_display",
        "year_to",
    ],
    weight_column="average_population",
)


region_year_summary.to_csv(
    OUTPUT_DIR
    / "annual_decoupling_region_year.csv",
    index=False,
)


# ============================================================
# 19. ANNUAL GLOBAL YEAR SUMMARY
# ============================================================

annual_global = annual.copy()


annual_global[
    "scope"
] = (
    "All economies"
)


global_year_summary = summarise_decoupling(
    df=annual_global,
    group_columns=[
        "year_to",
    ],
    weight_column="average_population",
)


global_year_summary.to_csv(
    OUTPUT_DIR
    / "annual_decoupling_global_year.csv",
    index=False,
)


global_summary = summarise_decoupling(
    df=annual_global,
    group_columns=[
        "scope",
    ],
    weight_column="average_population",
)


global_summary.to_csv(
    OUTPUT_DIR
    / "annual_decoupling_global_summary.csv",
    index=False,
)


# ============================================================
# 20. LONG-PERIOD ECONOMY COMPARISONS
# ============================================================

print_section(
    "LONG-PERIOD ECONOMY DECOUPLING"
)


long_period = build_long_period_changes(
    sample=main_sample,
    periods=LONG_PERIODS,
)


long_period[
    "region_display"
] = (
    long_period[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


long_period_output_columns = [
    "period",
    "start_year",
    "end_year",
    "years_elapsed",
    "iso3c",
    "country",
    "region",
    "region_display",
    "gdp_per_capita_start",
    "gdp_per_capita_end",
    "co2_per_capita_start",
    "co2_per_capita_end",
    "d_log_gdp",
    "d_log_co2",
    "gdp_change_pct",
    "co2_change_pct",
    "annualized_gdp_growth_pct",
    "annualized_co2_growth_pct",
    "decoupling_ratio",
    "decoupling_classification",
    "gdp_expansion",
    "absolute_decoupling",
    "relative_decoupling",
    "coupled_expansion",
    "any_decoupling",
    "average_population",
]


long_period[
    long_period_output_columns
].to_csv(
    OUTPUT_DIR
    / "long_period_economy_decoupling.csv",
    index=False,
)


# ============================================================
# 21. LONG-PERIOD REGIONAL SUMMARY
# ============================================================

long_region_summary = summarise_decoupling(
    df=long_period,
    group_columns=[
        "period",
        "start_year",
        "end_year",
        "region",
        "region_display",
    ],
    weight_column="average_population",
)


main_region_counts = (
    main_sample[
        [
            "iso3c",
            "region",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "region"
    )[
        "iso3c"
    ]
    .nunique()
    .to_dict()
)


long_region_summary[
    "total_main_sample_economies_in_region"
] = (
    long_region_summary[
        "region"
    ]
    .map(
        main_region_counts
    )
)


long_region_summary[
    "endpoint_coverage_pct"
] = (
    long_region_summary[
        "n_total_changes"
    ]
    /
    long_region_summary[
        "total_main_sample_economies_in_region"
    ]
    *
    100
)


long_region_summary.to_csv(
    OUTPUT_DIR
    / "long_period_decoupling_region_summary.csv",
    index=False,
)


# ============================================================
# 22. LONG-PERIOD GLOBAL SUMMARY
# ============================================================

long_global = long_period.copy()


long_global[
    "scope"
] = (
    "All economies"
)


long_global_summary = summarise_decoupling(
    df=long_global,
    group_columns=[
        "period",
        "start_year",
        "end_year",
        "scope",
    ],
    weight_column="average_population",
)


long_global_summary[
    "total_main_sample_economies"
] = (
    main_sample[
        "iso3c"
    ]
    .nunique()
)


long_global_summary[
    "endpoint_coverage_pct"
] = (
    long_global_summary[
        "n_total_changes"
    ]
    /
    long_global_summary[
        "total_main_sample_economies"
    ]
    *
    100
)


long_global_summary.to_csv(
    OUTPUT_DIR
    / "long_period_decoupling_global_summary.csv",
    index=False,
)


# ============================================================
# 23. PREPARE REGIONAL AGGREGATE PANEL
# ============================================================

print_section(
    "REGIONAL AGGREGATE DECOUPLING"
)


regional_panel = (
    regional_panel[
        (
            regional_panel[
                "year"
            ]
            >= START_YEAR
        )
        &
        (
            regional_panel[
                "year"
            ]
            <= END_YEAR
        )
    ]
    .copy()
)


if (
    regional_panel[
        "gdp_pc_region"
    ]
    <= 0
).any():

    raise ValueError(
        "Regional GDP contains non-positive values."
    )


if (
    regional_panel[
        "co2_pc_region"
    ]
    <= 0
).any():

    raise ValueError(
        "Regional CO2 contains non-positive values."
    )


regional_panel[
    "log_gdp_region"
] = np.log(
    regional_panel[
        "gdp_pc_region"
    ]
)


regional_panel[
    "log_co2_region"
] = np.log(
    regional_panel[
        "co2_pc_region"
    ]
)


regional_panel = (
    regional_panel
    .sort_values(
        [
            "region",
            "year",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 24. ANNUAL REGIONAL AGGREGATE CHANGES
# ============================================================

regional_annual = regional_panel.copy()


regional_annual[
    "year_from"
] = (
    regional_annual
    .groupby(
        "region"
    )[
        "year"
    ]
    .shift(1)
)


regional_annual[
    "d_log_gdp"
] = (
    regional_annual
    .groupby(
        "region"
    )[
        "log_gdp_region"
    ]
    .diff()
)


regional_annual[
    "d_log_co2"
] = (
    regional_annual
    .groupby(
        "region"
    )[
        "log_co2_region"
    ]
    .diff()
)


regional_annual[
    "year_gap"
] = (
    regional_annual[
        "year"
    ]
    -
    regional_annual[
        "year_from"
    ]
)


regional_annual = (
    regional_annual[
        regional_annual[
            "year_gap"
        ]
        == 1
    ]
    .copy()
)


regional_annual[
    "year_from"
] = (
    regional_annual[
        "year_from"
    ]
    .astype(int)
)


regional_annual = regional_annual.rename(
    columns={
        "year":
            "year_to"
    }
)


regional_annual[
    "gdp_change_pct"
] = (
    np.exp(
        regional_annual[
            "d_log_gdp"
        ]
    )
    -
    1
) * 100


regional_annual[
    "co2_change_pct"
] = (
    np.exp(
        regional_annual[
            "d_log_co2"
        ]
    )
    -
    1
) * 100


regional_annual[
    "decoupling_classification"
] = [
    classify_change(
        d_gdp,
        d_co2,
    )
    for (
        d_gdp,
        d_co2,
    )
    in zip(
        regional_annual[
            "d_log_gdp"
        ],
        regional_annual[
            "d_log_co2"
        ],
    )
]


regional_annual[
    "gdp_expansion"
] = (
    regional_annual[
        "d_log_gdp"
    ]
    >
    LOG_CHANGE_TOL
)


regional_annual[
    "region_display"
] = (
    regional_annual[
        "region"
    ]
    .map(
        REGION_DISPLAY_NAMES
    )
)


expected_regional_changes = (
    len(
        EXPECTED_REGIONS
    )
    *
    (
        len(
            EXPECTED_YEARS
        )
        -
        1
    )
)


if len(
    regional_annual
) != expected_regional_changes:

    raise ValueError(
        "Unexpected number of annual regional aggregate changes."
    )


regional_annual.to_csv(
    OUTPUT_DIR
    / "regional_aggregate_annual_decoupling.csv",
    index=False,
)


# ============================================================
# 25. REGIONAL AGGREGATE ANNUAL SUMMARY
# ============================================================

regional_annual_summary_rows = []


for region, group in regional_annual.groupby(
    "region"
):

    expansion = (
        group[
            group[
                "gdp_expansion"
            ]
        ]
    )


    n_expansion = len(
        expansion
    )


    absolute = int(
        (
            expansion[
                "decoupling_classification"
            ]
            ==
            "Absolute decoupling"
        )
        .sum()
    )


    relative = int(
        (
            expansion[
                "decoupling_classification"
            ]
            ==
            "Relative decoupling"
        )
        .sum()
    )


    coupled = int(
        (
            expansion[
                "decoupling_classification"
            ]
            ==
            "Coupled expansion"
        )
        .sum()
    )


    regional_annual_summary_rows.append(
        {
            "region":
                region,

            "region_display":
                REGION_DISPLAY_NAMES[
                    region
                ],

            "n_annual_changes":
                len(
                    group
                ),

            "n_gdp_expansions":
                n_expansion,

            "n_absolute_decoupling":
                absolute,

            "n_relative_decoupling":
                relative,

            "n_coupled_expansion":
                coupled,

            "share_absolute_of_expansions":
                (
                    absolute
                    /
                    n_expansion
                    if n_expansion > 0
                    else np.nan
                ),

            "share_relative_of_expansions":
                (
                    relative
                    /
                    n_expansion
                    if n_expansion > 0
                    else np.nan
                ),

            "share_coupled_of_expansions":
                (
                    coupled
                    /
                    n_expansion
                    if n_expansion > 0
                    else np.nan
                ),

            "share_any_decoupling_of_expansions":
                (
                    (
                        absolute
                        +
                        relative
                    )
                    /
                    n_expansion
                    if n_expansion > 0
                    else np.nan
                ),
        }
    )


regional_aggregate_summary = pd.DataFrame(
    regional_annual_summary_rows
)


regional_aggregate_summary.to_csv(
    OUTPUT_DIR
    / "regional_aggregate_annual_summary.csv",
    index=False,
)


# ============================================================
# 26. REGIONAL AGGREGATE LONG-PERIOD COMPARISONS
# ============================================================

regional_long_period = (
    build_regional_long_period_changes(
        regional_panel=regional_panel,
        periods=LONG_PERIODS,
    )
)


regional_long_period.to_csv(
    OUTPUT_DIR
    / "regional_aggregate_long_period_decoupling.csv",
    index=False,
)


# ============================================================
# 27. FIGURE — ANNUAL EXPANSION CLASSIFICATIONS BY REGION
# ============================================================

print_section(
    "CREATING ANNUAL DECOUPLING FIGURE"
)


figure_data = (
    region_summary
    .set_index(
        "region"
    )
    .loc[
        EXPECTED_REGIONS
    ]
    .reset_index()
)


labels = (
    figure_data[
        "region_display"
    ]
    .tolist()
)


absolute_share = (
    figure_data[
        "share_absolute_of_expansions"
    ]
    .to_numpy()
    *
    100
)


relative_share = (
    figure_data[
        "share_relative_of_expansions"
    ]
    .to_numpy()
    *
    100
)


coupled_share = (
    figure_data[
        "share_coupled_of_expansions"
    ]
    .to_numpy()
    *
    100
)


y_positions = np.arange(
    len(
        labels
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)


ax.barh(
    y_positions,
    absolute_share,
    label="Absolute decoupling",
)


ax.barh(
    y_positions,
    relative_share,
    left=absolute_share,
    label="Relative decoupling",
)


ax.barh(
    y_positions,
    coupled_share,
    left=(
        absolute_share
        +
        relative_share
    ),
    label="Coupled expansion",
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    labels
)


ax.invert_yaxis()


ax.set_xlim(
    0,
    100,
)


ax.set_xlabel(
    "Share of positive-GDP-per-capita-growth economy-year observations (%)"
)


ax.set_title(
    "Annual GDP–CO2 Per-Capita Decoupling Outcomes by Development Cluster\n"
    "Economy-Level Observations, 1990–2024"
)


ax.legend(
    loc="best"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "Shares are calculated only among economy-year observations "
        "with positive GDP-per-capita growth. Economic contractions "
        "are classified separately."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        1,
    ]
)


ANNUAL_FIGURE_FILE = (
    FIGURES_DIR
    / "11_annual_decoupling_shares_by_region.png"
)


fig.savefig(
    ANNUAL_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved:\n{ANNUAL_FIGURE_FILE}"
)


# ============================================================
# 28. FIGURE — FULL-PERIOD ENDPOINT COMPARISON
# ============================================================

print_section(
    "CREATING LONG-PERIOD DECOUPLING FIGURE"
)


full_period_summary = (
    long_region_summary[
        long_region_summary[
            "period"
        ]
        ==
        "1990-2024"
    ]
    .set_index(
        "region"
    )
    .reindex(
        EXPECTED_REGIONS
    )
    .reset_index()
)


full_absolute = (
    full_period_summary[
        "share_absolute_of_expansions"
    ]
    .to_numpy()
    *
    100
)


full_relative = (
    full_period_summary[
        "share_relative_of_expansions"
    ]
    .to_numpy()
    *
    100
)


full_coupled = (
    full_period_summary[
        "share_coupled_of_expansions"
    ]
    .to_numpy()
    *
    100
)


# Refinement: include the number of expanding economies in the
# plotted labels so a one-economy China or India bar is not
# visually mistaken for the same evidential weight as a large
# multi-economy group.
full_labels = [
    (
        f"{row['region_display']} "
        f"(n={int(row['n_gdp_expansions'])} expanding economies)"
    )
    for _, row
    in full_period_summary.iterrows()
]


y_positions = np.arange(
    len(
        full_labels
    )
)


fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)


ax.barh(
    y_positions,
    full_absolute,
    label="Absolute decoupling",
)


ax.barh(
    y_positions,
    full_relative,
    left=full_absolute,
    label="Relative decoupling",
)


ax.barh(
    y_positions,
    full_coupled,
    left=(
        full_absolute
        +
        full_relative
    ),
    label="Coupled expansion",
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    full_labels
)


ax.invert_yaxis()


ax.set_xlim(
    0,
    100,
)


ax.set_xlabel(
    "Share of expanding economies with both endpoints available (%)"
)


ax.set_title(
    "Long-Period GDP–CO2 Per-Capita Decoupling by Development Cluster\n"
    "1990–2024 Endpoint Comparison"
)


ax.legend(
    loc="best"
)


ax.grid(
    axis="x",
    alpha=0.25,
)


fig.text(
    0.5,
    0.015,
    (
        "Only economies with usable observations at both 1990 and 2024 "
        "endpoints enter this comparison. Labels report the number of "
        "expanding economies; endpoint coverage is reported separately."
    ),
    ha="center",
    va="bottom",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        1,
    ]
)


LONG_FIGURE_FILE = (
    FIGURES_DIR
    / "11_long_period_decoupling_shares_by_region.png"
)


fig.savefig(
    LONG_FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    f"Saved:\n{LONG_FIGURE_FILE}"
)


# ============================================================
# 29. CLEAN TERMINAL SUMMARY — ANNUAL ECONOMY EVENTS
# ============================================================

print_section(
    "ANNUAL ECONOMY-LEVEL DECOUPLING SUMMARY"
)


print(
    region_summary[
        [
            "region_display",
            "n_gdp_expansions",
            "share_absolute_of_expansions",
            "share_relative_of_expansions",
            "share_coupled_of_expansions",
            "share_any_decoupling_of_expansions",
            "median_gdp_change_pct_during_expansions",
            "median_co2_change_pct_during_expansions",
            "population_weighted_share_any_decoupling",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 30. CLEAN TERMINAL SUMMARY — LONG PERIODS
# ============================================================

print_section(
    "LONG-PERIOD DECOUPLING SUMMARY"
)


print(
    long_region_summary[
        [
            "period",
            "region_display",
            "n_total_changes",
            "endpoint_coverage_pct",
            "n_gdp_expansions",
            "share_absolute_of_expansions",
            "share_relative_of_expansions",
            "share_coupled_of_expansions",
            "median_gdp_change_pct_during_expansions",
            "median_co2_change_pct_during_expansions",
            "median_annualized_gdp_growth_pct_during_expansions",
            "median_annualized_co2_growth_pct_during_expansions",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 31. CLEAN TERMINAL SUMMARY — REGIONAL AGGREGATES
# ============================================================

print_section(
    "REGIONAL AGGREGATE LONG-PERIOD OUTCOMES"
)


print(
    regional_long_period[
        [
            "period",
            "region_display",
            "gdp_change_pct",
            "co2_change_pct",
            "annualized_gdp_growth_pct",
            "annualized_co2_growth_pct",
            "decoupling_classification",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 32. FINAL VALIDATION
# ============================================================

print_section(
    "FINAL VALIDATION"
)


if len(
    regional_annual
) != expected_regional_changes:

    raise ValueError(
        "Regional annual change count failed final validation."
    )


if annual[
    "year_gap"
].ne(
    1
).any():

    raise ValueError(
        "Non-consecutive annual changes remain in the "
        "economy-level annual sample."
    )


if annual[
    "decoupling_classification"
].isna().any():

    raise ValueError(
        "Missing annual decoupling classifications remain."
    )


if long_period[
    "decoupling_classification"
].isna().any():

    raise ValueError(
        "Missing long-period classifications remain."
    )


# Validate that expansion shares sum to one wherever an
# expansion denominator exists.
for summary_name, summary_df in [
    (
        "annual region summary",
        region_summary,
    ),
    (
        "long-period region summary",
        long_region_summary,
    ),
]:

    valid = (
        summary_df[
            "n_gdp_expansions"
        ]
        > 0
    )

    share_sum = (
        summary_df.loc[
            valid,
            "share_absolute_of_expansions"
        ]
        +
        summary_df.loc[
            valid,
            "share_relative_of_expansions"
        ]
        +
        summary_df.loc[
            valid,
            "share_coupled_of_expansions"
        ]
    )


    if not np.allclose(
        share_sum,
        1.0,
        atol=1e-12,
        rtol=0,
    ):

        raise ValueError(
            f"Expansion shares do not sum to one in {summary_name}."
        )


print(
    "PASS: all economy-level annual changes are consecutive."
)

print(
    "PASS: annual classifications are complete."
)

print(
    "PASS: long-period classifications are complete."
)

print(
    "PASS: expansion shares sum to one where defined."
)

print(
    "PASS: regional aggregate annual panel contains "
    f"{expected_regional_changes} expected changes."
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
    "\nMachine-readable outputs:"
)


# Existing filenames are retained. The refined script adds one
# new audit file: annual_co2_change_outlier_audit.csv.
key_files = [
    "decoupling_methodology.csv",
    "annual_economy_decoupling.csv",
    "annual_co2_change_outlier_audit.csv",
    "annual_decoupling_economy_summary.csv",
    "annual_decoupling_region_summary.csv",
    "annual_decoupling_region_year.csv",
    "annual_decoupling_global_year.csv",
    "annual_decoupling_global_summary.csv",
    "long_period_economy_decoupling.csv",
    "long_period_decoupling_region_summary.csv",
    "long_period_decoupling_global_summary.csv",
    "regional_aggregate_annual_decoupling.csv",
    "regional_aggregate_annual_summary.csv",
    "regional_aggregate_long_period_decoupling.csv",
]


for filename in key_files:

    print(
        f"- {filename}"
    )


print(
    "\nFigures:"
)

print(
    f"- {ANNUAL_FIGURE_FILE}"
)

print(
    f"- {LONG_FIGURE_FILE}"
)


print(
    "\nSCRIPT 11 COMPLETE."
)
# Economic Growth, CO₂ Emissions, and Governance

## A Regional Panel Analysis

**Independent Climate Economics Research Project**

## Project Overview

This project investigates the relationship between economic development and
carbon emissions, with particular attention to regional heterogeneity,
decoupling, and the potential role of institutional quality.

The central question is whether higher GDP per capita is systematically
associated with higher CO₂ emissions per capita, and whether this relationship
changes across different stages and patterns of development.

The project uses a country-year panel constructed from World Bank data and a
population-weighted regional panel covering five regional-development
clusters.

## Research Questions

### Primary question

What is the relationship between GDP per capita and CO₂ emissions per capita
across regions and over time?

### Secondary questions

- Does the income-emissions relationship differ systematically across
  regional-development clusters?
- Is there evidence of absolute or relative decoupling at higher income levels?
- Is there empirical support for an Environmental Kuznets Curve relationship?
- Do stronger institutions weaken the relationship between economic growth
  and emissions?

## Conceptual Framework

### Growth-emissions channel

Economic development can affect emissions through several stages:

- Early development: industrialisation, infrastructure expansion and greater
  fossil-fuel use can increase emissions.
- Later development: structural change towards services, technological
  upgrading, energy efficiency and environmental regulation can weaken the
  relationship between income and emissions.

This motivates testing for non-linear Environmental Kuznets Curve patterns.

### Governance channel

Institutional quality may affect emissions through:

- environmental regulation and enforcement,
- renewable-energy investment,
- carbon-pricing mechanisms,
- green innovation,
- infrastructure and energy policy.

Governance is therefore treated as a potential mechanism that may alter the
income-emissions relationship. The governance analysis is a planned extension
and is not yet interpreted as an empirical result.

## Data

### World Development Indicators

Core indicators are obtained from the World Bank World Development Indicators.

| Concept | Indicator | Code |
|---|---|---|
| GDP per capita | GDP per capita, constant 2015 US$ | `NY.GDP.PCAP.KD` |
| CO₂ emissions | CO₂ emissions excluding LULUCF per capita, AR5-consistent | `EN.GHG.CO2.PC.CE.AR5` |
| Population | Population, total | `SP.POP.TOTL` |

The main sample begins in 1990.

## Dataset Construction

The WDI download is transformed from wide format into a country-year panel.

The cleaned country-level dataset is:

`data/cleaned/wdi_gdp_co2_population_panel.csv`

Each row represents one country-year observation.

Core variables are:

- `country`
- `iso3c`
- `year`
- `gdp_per_capita_const2015_usd`
- `co2_per_capita_tons`
- `population`

## Regional Classification

Countries are assigned to five regional-development clusters:

1. `Europe_NorthAmerica`
2. `DevelopedAsia_Oceania`
3. `China`
4. `India`
5. `Global_South`

China and India are retained as standalone groups because of their population,
economic scale and distinct development-emissions trajectories.

Developed Asia & Oceania contains mature Asia-Pacific economies including
Japan, South Korea, Australia, New Zealand and Singapore, together with Israel
under the project's development-based classification.

These groups should therefore be interpreted as regional-development clusters
rather than strictly geographical regions.

The explicit ISO3-to-region mapping is stored in:

`data/meta/country_regions.csv`

## Regional Aggregation

Country-year observations are aggregated to region-year observations using
population weighting.

For GDP per capita:

GDPpc_region =
sum(GDPpc_country × population_country) /
sum(population_country)

The same method is used for CO₂ emissions per capita.

Regional population is calculated as the sum of the populations of
contributing countries.

The resulting dataset is:

`data/analysis/region_year_panel.csv`

Each row represents one region-year observation.

## Current Analysis

The project has completed the core descriptive and exploratory stages.

Current outputs include:

- GDP per capita versus CO₂ emissions scatter plots,
- GDP trends by region,
- CO₂ trends by region,
- log-log GDP-emissions plots,
- region-specific development paths,
- fitted log-log relationships,
- pooled quadratic Environmental Kuznets Curve visualisation,
- separate regional log-log figures.

These figures are stored under:

`figures/`

The fitted curves are currently treated as exploratory evidence rather than
formal econometric results.

## Empirical Strategy

The next stage is formal econometric estimation.

### Regional benchmark models

The first formal specifications will estimate:

1. pooled log-log income-emissions regression,
2. region-specific log-log regressions,
3. quadratic EKC specifications.

These models will report coefficient estimates, standard errors, p-values,
sample sizes and goodness-of-fit statistics.

### Country-panel extension

The main econometric extension will return to the underlying country-year
panel and estimate models with country and year fixed effects.

This allows regional heterogeneity to be examined without discarding the
within-region country-level variation.

### Governance extension

World Governance Indicators are planned for a later stage.

Candidate indicators include:

- Government Effectiveness,
- Regulatory Quality,
- Rule of Law.

Governance will initially be merged at country-year level rather than directly
aggregated to five regional observations.

## Repository Structure

```text
worldbank_project/
│
├── analysis/
│   ├── 00_validate_region_panel.py
│   ├── descriptive and plotting scripts
│   └── aggregation scripts
│
├── data/
│   ├── raw/
│   │   └── World Bank WDI download
│   │
│   ├── cleaned/
│   │   └── wdi_gdp_co2_population_panel.csv
│   │
│   ├── meta/
│   │   ├── country_regions.csv
│   │   └── data_checksums.sha256
│   │
│   └── analysis/
│       ├── region_year_panel.csv
│       └── qc/
│
├── figures/
│   ├── regional descriptive figures
│   └── loglog_by_region/
│
├── README.md
├── requirements.txt
└── .gitignore
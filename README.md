# Economic Growth, CO₂ Emissions, and Governance

## A Regional Panel Analysis

**Independent Climate Economics Research Project**

---

## 1. Project Overview

This project investigates the relationship between economic development and
carbon emissions, with particular attention to regional heterogeneity,
long-run decoupling, non-linear development paths, and the potential role of
institutional quality.

The central question is whether higher GDP per capita is systematically
associated with higher CO₂ emissions per capita, and whether this relationship
changes across different development trajectories.

The project combines:

- a country-year panel constructed from World Bank data;
- a population-weighted regional panel covering five regional-development
  clusters;
- descriptive regional analysis;
- log-log income-emissions regressions;
- Environmental Kuznets Curve (EKC) specifications;
- first-difference robustness regressions.

The next econometric stage will return to the underlying country-year panel
and estimate models with country and year fixed effects.

---

## 2. Research Questions

### Primary Question

What is the relationship between GDP per capita and CO₂ emissions per capita
across regions and over time?

### Secondary Questions

- Does the income-emissions relationship differ systematically across
  regional-development clusters?
- Is there evidence of long-run decoupling at higher income levels?
- Is there empirical support for an Environmental Kuznets Curve relationship?
- Does the responsiveness of emissions to income weaken as economies develop?
- Do stronger institutions help explain differences in long-run
  income-emissions trajectories?

---

## 3. Conceptual Framework

### 3.1 Growth-Emissions Channel

Economic development may affect emissions differently across stages of
development.

A simplified development pathway is:

- Early development → industrialisation, infrastructure expansion and greater
  fossil-fuel use → rising emissions.
- Later development → structural change, technological upgrading, energy
  efficiency and environmental regulation → weaker emissions growth or
  declining emissions.

This motivates testing both linear income-emissions relationships and
non-linear Environmental Kuznets Curve specifications.

### 3.2 Governance Channel

Institutional quality may influence long-run emissions trajectories through:

- environmental regulation and enforcement;
- public investment in low-carbon infrastructure;
- renewable-energy policy;
- carbon-pricing mechanisms;
- green innovation;
- energy and industrial policy.

Governance is treated as a potential explanatory mechanism rather than an
established result at the current stage of the project.

---

## 4. Data

### 4.1 World Development Indicators

Core variables are obtained from the World Bank World Development Indicators
(WDI).

| Concept | Indicator | Code |
|---|---|---|
| GDP per capita | GDP per capita, constant 2015 US$ | `NY.GDP.PCAP.KD` |
| CO₂ emissions per capita | CO₂ emissions excluding LULUCF per capita, AR5-consistent | `EN.GHG.CO2.PC.CE.AR5` |
| Population | Population, total | `SP.POP.TOTL` |

The main regional sample covers:

**1990–2024**

---

## 5. Country-Year Dataset Construction

The original WDI download is provided in wide format, with years stored as
columns.

The data-processing pipeline:

1. loads the World Bank WDI extract;
2. programmatically identifies year columns;
3. reshapes the data from wide to long format;
4. converts values to numeric format;
5. retains the required indicators;
6. pivots the data to one row per country-year;
7. exports a cleaned country-year panel.

The cleaned dataset is stored at:

`data/cleaned/wdi_gdp_co2_population_panel.csv`

Core variables are:

- `country`
- `iso3c`
- `year`
- `gdp_per_capita_const2015_usd`
- `co2_per_capita_tons`
- `population`

---

## 6. Regional Classification

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

The groupings should therefore be interpreted as
**regional-development clusters**, rather than purely geographical regions.

The explicit ISO3-to-region mapping is stored at:

`data/meta/country_regions.csv`

---

## 7. Regional Aggregation

Country-year observations are converted into region-year observations using
population-weighted aggregation.

For GDP per capita:

\[
GDPpc_{rt}
=
\frac{
\sum_i GDPpc_{irt} \times Population_{irt}
}{
\sum_i Population_{irt}
}
\]

The same population-weighting procedure is applied to CO₂ emissions per
capita.

Regional population is calculated as the sum of the populations of
contributing countries.

The resulting dataset is stored at:

`data/analysis/region_year_panel.csv`

Each observation represents one region-year.

The current panel is balanced over time:

- 5 regional-development clusters;
- 35 annual observations per region;
- 1990–2024;
- 175 region-year observations in total.

---

## 8. Data Validation

The regional panel is independently validated using:

`analysis/00_validate_region_panel.py`

The validation procedure checks:

- expected region labels;
- duplicate country-year observations;
- duplicate region-year observations;
- missing values;
- positive GDP, emissions and population values;
- continuous time coverage;
- country coverage by region-year;
- population coverage;
- consistency between the saved regional panel and independently
  reconstructed population-weighted aggregates.

Quality-control outputs are stored under:

`data/analysis/qc/`

Dataset checksums are stored at:

`data/meta/data_checksums.sha256`

---

## 9. Descriptive Analysis

The descriptive stage examines:

- GDP per capita versus CO₂ emissions per capita;
- GDP trends by region;
- CO₂ trends by region;
- log-log GDP-emissions relationships;
- region-specific development paths;
- exploratory quadratic EKC relationships.

Figures are stored under:

`figures/`

The descriptive analysis indicates substantial heterogeneity across regional
development trajectories, motivating region-specific econometric estimation.

---

## 10. Econometric Strategy

### 10.1 Pooled Regional Benchmark

The baseline pooled regional specification is:

\[
\ln(CO_{2,rt})
=
\alpha
+
\beta \ln(GDP_{rt})
+
\varepsilon_{rt}
\]

The pooled model provides a benchmark relationship across all region-year
observations.

Because the regions exhibit substantially different trajectories, the pooled
coefficient is not interpreted as a universal income-emissions elasticity.

### 10.2 Region-Specific Log-Log Models

For each regional-development cluster:

\[
\ln(CO_{2,rt})
=
\alpha_r
+
\beta_r \ln(GDP_{rt})
+
\varepsilon_{rt}
\]

The coefficient \(\beta_r\) represents the estimated income elasticity of CO₂
emissions within the regional time series.

Region-specific regressions use Newey-West/HAC standard errors with
small-sample correction and t-based inference.

### 10.3 Environmental Kuznets Curve Models

Regional non-linearity is tested using:

\[
\ln(CO_{2,rt})
=
\alpha_r
+
\beta_{1r}\ln(GDP_{rt})
+
\beta_{2r}[\ln(GDP_{rt})]^2
+
\varepsilon_{rt}
\]

A region is treated as a preliminary within-sample EKC candidate only where:

- the estimated relationship has an inverted-U sign pattern;
- the quadratic term is statistically significant;
- the estimated turning point lies within the observed GDP range.

Model choice is also evaluated using:

- adjusted \(R^2\);
- Akaike Information Criterion (AIC);
- Bayesian Information Criterion (BIC).

### 10.4 First-Difference Robustness Models

Because GDP per capita and CO₂ emissions per capita are strongly trending
macroeconomic variables, the regional analysis also estimates:

\[
\Delta\ln(CO_{2,rt})
=
\alpha_r
+
\beta_r\Delta\ln(GDP_{rt})
+
\varepsilon_{rt}
\]

This specification examines whether annual changes in income and emissions
co-move after removing the long-run level trend.

First differencing reduces the regional sample from 35 to 34 observations per
region.

---

## 11. Main Regional Results

The regional regressions indicate that the pooled positive relationship between
GDP and emissions masks substantial heterogeneity.

| Region | Levels GDP-CO₂ Elasticity | First-Difference Elasticity | Preferred Shape | EKC Assessment |
|---|---:|---:|---|---|
| Europe & North America | **−0.539** | **+1.256** | Non-linear | Possible high-income EKC / long-run decoupling pattern |
| Developed Asia & Oceania | +0.112, statistically imprecise | **+1.064** | Strongly non-linear | **Strongest preliminary within-sample EKC candidate** |
| China | **+0.621** | **+1.018** | Linear | No credible EKC evidence |
| India | **+0.747** | **+0.725** | Linear | No credible EKC evidence |
| Global South | **+0.687** | **+0.800** | Non-linear | Significant concavity, but turning point remains outside observed range |

### 11.1 Europe & North America

The levels regression produces a negative GDP-emissions elasticity of
approximately **−0.54**, consistent with a long-run pattern of rising income
and declining per-capita emissions.

The quadratic model is preferred to the linear specification and produces an
estimated turning point of approximately **$25,400 GDP per capita**, which lies
within the observed income range.

However, relatively few observations lie below the estimated turning point.
The result is therefore interpreted as evidence consistent with high-income
non-linearity and long-run decoupling rather than definitive evidence of a
complete textbook EKC.

The first-difference coefficient is strongly positive, indicating that
short-run economic expansions remain associated with emissions growth even
within a long-run declining emissions trajectory.

### 11.2 Developed Asia & Oceania

The linear levels relationship is weak and statistically imprecise.

Allowing for non-linearity substantially improves model fit.

The estimated quadratic relationship:

- has an inverted-U form;
- produces a turning point of approximately **$32,900 GDP per capita**;
- places the turning point near the middle of the observed income range;
- has approximately equal numbers of observations on either side of the peak.

Developed Asia & Oceania therefore provides the strongest preliminary
region-level evidence consistent with an EKC-type transition.

However, the positive first-difference relationship indicates that short-run
GDP growth remains associated with emissions growth.

### 11.3 China

China exhibits a strong positive levels relationship:

\[
\beta \approx 0.62
\]

The positive relationship also survives first differencing.

The quadratic specification provides no meaningful improvement over the linear
model, and the estimated turning point lies far outside the observed income
range.

The regional evidence therefore does not support an EKC for China during the
sample period.

### 11.4 India

India has the largest positive levels elasticity among the five regional
groups:

\[
\beta \approx 0.75
\]

The relationship remains positive in first differences.

The quadratic specification does not materially improve model fit and produces
an economically irrelevant turning point far outside the observed data.

The linear specification is therefore preferred.

### 11.5 Global South

The Global South exhibits a strong positive levels relationship:

\[
\beta \approx 0.69
\]

and a strong positive first-difference relationship.

However, the quadratic model is strongly preferred to the linear model.

The estimated marginal responsiveness of emissions to income declines
substantially as GDP rises, indicating meaningful concavity in the development
path.

The implied turning point of approximately **$5,900 GDP per capita** remains
above the observed maximum income in the regional sample.

The evidence therefore supports a weakening income-emissions relationship, but
not an observed EKC transition.

---

## 12. Interpretation of the Regional Evidence

The current evidence does **not** support a single universal Environmental
Kuznets Curve.

Instead, income-emissions relationships appear to differ substantially across
development trajectories.

China, India and the Global South exhibit positive long-run and short-run
income-emissions relationships.

The Global South nevertheless shows evidence that emissions become less
responsive to additional income growth at higher development levels.

High-income regions exhibit a different long-run pattern.

Europe & North America displays a negative levels relationship consistent with
long-run decoupling, while Developed Asia & Oceania provides the strongest
preliminary evidence of an inverted-U regional development path.

Importantly, annual GDP growth remains positively associated with annual
emissions growth in all five regions.

This suggests that high-income long-run decoupling should not be interpreted
as economic growth automatically reducing emissions.

Instead, long-run structural transformation, energy transition, technological
change and policy may help shift the emissions intensity of economic
activity.

These mechanisms will be investigated more directly in subsequent stages of
the project.

---

## 13. Current Regression Outputs

Formal regional results are generated by:

`analysis/08_regional_regressions.py`

Key outputs are stored under:

`data/analysis/regression_results/`

The principal machine-readable tables are:

- `pooled_loglog_results.csv`
- `regional_loglog_results.csv`
- `regional_ekc_results.csv`
- `linear_vs_ekc_comparison.csv`
- `regional_first_difference_results.csv`

Full Statsmodels summaries can be reproduced directly by running the regression
script.

---

## 14. Next Econometric Stage

The next stage returns to the underlying country-year panel.

The main specification will estimate:

\[
\ln(CO_{2,it})
=
\alpha_i
+
\lambda_t
+
\beta\ln(GDP_{it})
+
\varepsilon_{it}
\]

where:

- \(\alpha_i\) represents country fixed effects;
- \(\lambda_t\) represents year fixed effects.

Standard errors will be clustered at the country level.

This specification exploits substantially more variation than the
five-region aggregate panel and controls for:

- time-invariant country characteristics;
- global year-specific shocks.

Subsequent models will test regional heterogeneity using interactions between
regional classification and GDP per capita.

---

## 15. Planned Extensions

### Country Panel Fixed Effects

Estimate within-country income-emissions relationships while controlling for
country and year fixed effects.

### Regional Heterogeneity

Formally test whether income-emissions elasticities differ between regional
development clusters.

### Direct Decoupling Analysis

Distinguish:

- absolute decoupling: GDP rises while emissions fall;
- relative decoupling: emissions continue to rise but more slowly than GDP.

### Governance Extension

World Governance Indicators are planned for a later stage.

Candidate indicators include:

- Government Effectiveness;
- Regulatory Quality;
- Rule of Law.

Governance will initially be merged at country-year level so that the analysis
retains cross-country institutional variation.

A later specification may examine whether governance changes the
income-emissions relationship through an interaction term.

---

## 16. Limitations

The current regional analysis has several important limitations.

- The analysis is observational and does not establish causality.
- Regional aggregation masks substantial within-region country heterogeneity.
- Each regional regression contains only 35 annual observations.
- GDP and emissions exhibit strong time trends.
- First differencing reduces, but does not eliminate, all time-series concerns.
- The regional aggregates may reflect changing country coverage where source
  data are unavailable.
- Energy mix, industrial structure and trade-related emissions are not yet
  explicitly controlled for.
- Governance indicators contain measurement uncertainty.
- Regional EKC turning points should be interpreted cautiously until supported
  by the country-panel analysis.

For these reasons, the regional regressions are treated as benchmark and
descriptive econometric evidence rather than the final inferential model.

---

## 17. Repository Structure

```text
worldbank_project/
│
├── analysis/
│   ├── 00_validate_region_panel.py
│   ├── 01_scatter_...
│   ├── 02_gdp_trends_by_region.py
│   ├── 03_co2_trends_by_region.py
│   ├── 04_loglog_...
│   ├── 05_loglog_scatter_with_fit.py
│   ├── 06_loglog_by_region_separate.py
│   ├── 07_ekc_quadratic_plot.py
│   ├── 08_regional_regressions.py
│   └── regional aggregation / cleaning scripts
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
│       ├── qc/
│       └── regression_results/
│
├── figures/
│   ├── regional descriptive figures
│   └── loglog_by_region/
│
├── README.md
├── requirements.txt
└── .gitignore
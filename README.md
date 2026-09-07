# Economic Growth, CO₂ Emissions, and Governance

## A Regional and Economy-Level Panel Analysis

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

The empirical design combines two levels of analysis:

1. a population-weighted regional panel covering five regional-development
   clusters; and
2. an economy-year panel covering World Bank reporting economies.

The regional analysis is used to describe heterogeneous development-emissions
trajectories and investigate possible Environmental Kuznets Curve (EKC)
patterns.

The economy-level panel is then used for the main econometric analysis,
including economy fixed effects, year fixed effects, population-weighted
robustness, balanced-panel robustness, and first-difference specifications.

The next stage will formally test whether within-economy GDP-emissions
elasticities differ across the five regional-development clusters.

---

## 2. Research Questions

### Primary Question

What is the relationship between GDP per capita and CO₂ emissions per capita
across economies, regions and over time?

### Secondary Questions

- Does the income-emissions relationship differ systematically across
  regional-development clusters?
- Is there evidence of long-run decoupling at higher income levels?
- Is there empirical support for an Environmental Kuznets Curve relationship?
- Does the responsiveness of emissions to income weaken as economies develop?
- Does the positive GDP-emissions relationship persist after controlling for
  persistent economy-specific characteristics and common global shocks?
- Do stronger institutions help explain differences in long-run
  income-emissions trajectories?

---

## 3. Conceptual Framework

### 3.1 Growth-Emissions Channel

Economic development may affect carbon emissions differently across stages of
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

Governance is therefore treated as a potential explanatory mechanism rather
than an established result at the current stage of the project.

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

The main analysis period is:

**1990–2024**

### 4.2 Economy Coverage

The cleaned WDI extract contains **217 World Bank reporting economies and
jurisdictions** rather than sovereign states only.

The project deliberately retains the full World Bank economy universe,
including territories and special administrative regions where the required
data are available.

All 217 economy codes have an explicit regional assignment in:

`data/meta/country_regions.csv`

There are no unmatched economy codes in the cleaned WDI analysis window.

The main log GDP-log CO₂ regression sample contains **197 economies**.

Of the remaining 20 mapped economies:

- 19 have no usable positive GDP-CO₂ economy-year observations during
  1990–2024;
- New Caledonia has only one usable joint observation and therefore does not
  meet the minimum two-observation requirement for panel estimation.

The 70 non-positive CO₂ observations in the original source panel are entirely
accounted for by Nauru and Tuvalu, whose CO₂ series are recorded as zero
throughout the analysis period. Since the logarithm of zero is undefined,
these observations cannot enter the log specifications.

The complete economy-universe audit is stored at:

`data/analysis/country_panel_results/country_panel_economy_universe.csv`

---

## 5. Economy-Year Dataset Construction

The original WDI download is provided in wide format, with years stored as
columns.

The data-processing pipeline:

1. loads the World Bank WDI extract;
2. programmatically identifies year columns;
3. reshapes the data from wide to long format;
4. converts values to numeric format;
5. retains the required indicators;
6. pivots the data to one row per economy-year;
7. exports a cleaned panel dataset.

The cleaned dataset is stored at:

`data/cleaned/wdi_gdp_co2_population_panel.csv`

Core variables are:

- `country`
- `iso3c`
- `year`
- `gdp_per_capita_const2015_usd`
- `co2_per_capita_tons`
- `population`

The terms `country` and `country_panel` are retained in some technical
filenames and variable names for continuity with the original data pipeline.
In the research interpretation, the statistical units are World Bank
reporting economies.

---

## 6. Regional Classification

The 217 World Bank economies are assigned to five regional-development
clusters:

1. `Europe_NorthAmerica`
2. `DevelopedAsia_Oceania`
3. `China`
4. `India`
5. `Global_South`

The classification is intended to capture differences in development paths,
institutional maturity, industrialisation history and emissions structure.
It should therefore be interpreted as a set of
**regional-development clusters**, rather than purely geographical regions.

### China, Hong Kong and Macao

Mainland China (`CHN`) is treated as the standalone China cluster.

Hong Kong SAR (`HKG`) and Macao SAR (`MAC`) are classified within
Developed Asia & Oceania because their development trajectories and economic
structures differ substantially from mainland China.

The classification therefore follows development characteristics rather than
political geography alone.

China and India are retained as standalone groups because of their population,
economic scale and distinctive development-emissions trajectories.

The explicit economy mapping is stored at:

`data/meta/country_regions.csv`

### Full mapped economy counts

| Regional-development cluster | Mapped economies |
|---|---:|
| Europe & North America | 53 |
| Developed Asia & Oceania | 8 |
| China | 1 |
| India | 1 |
| Global South | 154 |
| **Total** | **217** |

---

## 7. Regional Aggregation

Economy-year observations are converted into region-year observations using
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
contributing economies.

The resulting dataset is stored at:

`data/analysis/region_year_panel.csv`

Each observation represents one region-year.

The regional panel contains:

- 5 regional-development clusters;
- 35 annual observations per cluster;
- 1990–2024;
- 175 region-year observations in total.

---

## 8. Data Validation and Reproducibility

The regional panel is independently validated using:

`analysis/00_validate_region_panel.py`

The validation procedure checks:

- expected regional labels;
- duplicate economy-year observations;
- duplicate region-year observations;
- missing values;
- positive GDP, emissions and population values;
- continuous time coverage;
- economy coverage by region-year;
- population coverage;
- consistency between the saved regional panel and independently
  reconstructed population-weighted aggregates.

Quality-control outputs are stored under:

`data/analysis/qc/`

Dataset checksums are stored at:

`data/meta/data_checksums.sha256`

The economy-level regression pipeline performs additional checks for:

- duplicate economy-year keys;
- complete regional mapping;
- regression-sample eligibility;
- positive values before logarithmic transformation;
- panel coverage;
- balanced-panel status;
- economy coverage by region and year;
- consecutive-year validity before first differencing.

---

## 9. Descriptive Regional Analysis

The descriptive stage examines:

- GDP per capita versus CO₂ emissions per capita;
- GDP trends by region;
- CO₂ trends by region;
- log-log GDP-emissions relationships;
- region-specific development paths;
- exploratory quadratic EKC relationships.

Figures are stored under:

`figures/`

The descriptive analysis reveals substantial differences in development and
emissions trajectories across the five clusters, motivating formal
region-specific estimation.

---

## 10. Regional Econometric Strategy

Regional regressions are estimated using the 175-observation region-year
panel.

### 10.1 Pooled Regional Benchmark

\[
\ln(CO_{2,rt})
=
\alpha
+
\beta \ln(GDP_{rt})
+
\varepsilon_{rt}
\]

The pooled model provides a benchmark but is not interpreted as a universal
income-emissions elasticity because the regional paths are highly
heterogeneous.

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

Region-specific models use Newey-West/HAC standard errors with small-sample
correction and t-based inference.

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
- the estimated turning point lies within the observed income range.

Adjusted \(R^2\), AIC and BIC are also used to assess whether the quadratic
specification meaningfully improves fit relative to the linear model.

### 10.4 First-Difference Robustness

Because GDP and emissions are strongly trending macroeconomic variables, the
regional analysis also estimates:

\[
\Delta\ln(CO_{2,rt})
=
\alpha_r
+
\beta_r\Delta\ln(GDP_{rt})
+
\varepsilon_{rt}
\]

This tests whether annual GDP and emissions changes co-move after removing the
long-run levels trend.

---

## 11. Main Regional Results

The regional regressions indicate that the pooled positive GDP-emissions
relationship masks substantial heterogeneity.

| Region | Levels GDP-CO₂ Elasticity | First-Difference Elasticity | Preferred Shape | EKC Assessment |
|---|---:|---:|---|---|
| Europe & North America | **−0.539** | **+1.256** | Non-linear | Possible high-income EKC / long-run decoupling pattern |
| Developed Asia & Oceania | +0.112, statistically imprecise | **+1.064** | Strongly non-linear | **Strongest preliminary within-sample EKC candidate** |
| China | **+0.621** | **+1.018** | Linear | No credible EKC evidence |
| India | **+0.747** | **+0.725** | Linear | No credible EKC evidence |
| Global South | **+0.687** | **+0.800** | Non-linear | Significant concavity; turning point remains outside the observed range |

### Europe & North America

The levels relationship is negative, consistent with a long-run pattern of
rising income and declining per-capita emissions.

The quadratic model is preferred to the linear model and produces an estimated
turning point of approximately **$25,400 GDP per capita**, inside the observed
income range.

However, relatively few observations lie below the turning point. The result
is therefore interpreted as evidence consistent with high-income non-linearity
and long-run decoupling rather than definitive evidence of a complete
textbook EKC.

The first-difference relationship remains strongly positive, indicating that
short-run economic expansions can remain associated with emissions growth
within a longer-run declining emissions trajectory.

### Developed Asia & Oceania

The simple linear relationship is weak and statistically imprecise, while the
quadratic specification substantially improves model fit.

The estimated turning point is approximately **$32,900 GDP per capita** and
lies near the centre of the observed income range, with substantial
observations on both sides.

This cluster therefore provides the strongest preliminary regional evidence
consistent with an EKC-type transition.

The first-difference relationship nevertheless remains positive.

### China

China exhibits a strong positive income-emissions relationship in both levels
and first differences.

The quadratic specification provides no meaningful improvement over the
linear model and implies a turning point far outside the observed income
range.

The regional evidence therefore does not support an EKC for China over
1990–2024.

### India

India also exhibits a strong positive levels relationship, which survives
first differencing.

The quadratic specification adds little explanatory value and produces an
economically irrelevant turning point outside the observed data.

The linear specification is therefore preferred.

### Global South

The Global South exhibits strong positive levels and first-difference
relationships.

The quadratic specification is strongly preferred to a straight-line model,
however, indicating that the responsiveness of emissions to income falls
substantially as income rises.

The implied turning point of approximately **$5,900 GDP per capita** remains
above the maximum income observed in the regional sample.

The evidence therefore supports increasing concavity and a weakening
income-emissions relationship, but not an observed EKC transition.

---

## 12. Interpretation of the Regional Evidence

The regional evidence does not support a single universal Environmental
Kuznets Curve.

Instead, income-emissions relationships vary substantially across development
trajectories.

China, India and the Global South continue to exhibit positive long-run and
short-run GDP-emissions relationships.

The Global South nevertheless shows evidence that emissions become less
responsive to additional income growth at higher development levels.

High-income regional trajectories differ. Europe & North America exhibits a
negative levels relationship consistent with long-run decoupling, while
Developed Asia & Oceania provides the strongest preliminary evidence of an
inverted-U regional development path.

Importantly, annual GDP and emissions changes remain positively associated in
all five regional clusters.

Long-run decoupling should therefore not be interpreted as economic growth
automatically reducing emissions. Structural transformation, technology,
energy systems and policy may instead alter the emissions intensity of
economic activity over time.

---

## 13. Economy-Level Panel Sample

Regional aggregation is useful for describing development trajectories, but it
discards substantial cross-economy variation.

The main econometric stage therefore returns to the underlying economy-year
panel.

### Main estimation sample

The final levels panel contains:

- **197 World Bank economies**;
- **6,725 economy-year observations**;
- **35 potential years**, 1990–2024;
- median coverage of **35 years per economy**;
- mean coverage of approximately **34.14 years per economy**;
- **180 completely balanced economies**.

The theoretical maximum for 197 economies over 35 years is:

\[
197 \times 35 = 6,895
\]

so the main estimation panel contains approximately:

\[
\frac{6,725}{6,895}
\approx
97.5\%
\]

of all possible economy-year observations.

The incomplete series contain no internal missing-year gaps in the final
estimation sample; missing observations occur at the beginning or end of
individual economy series.

### Estimation-sample distribution

| Regional-development cluster | Economies | Economy-year observations |
|---|---:|---:|
| Europe & North America | 43 | 1,486 |
| Developed Asia & Oceania | 8 | 280 |
| China | 1 | 35 |
| India | 1 | 35 |
| Global South | 144 | 4,889 |
| **Total** | **197** | **6,725** |

This unequal distribution is one reason why the global fixed-effects
coefficient should not be interpreted as equally representative of all five
clusters.

---

## 14. Economy-Level Econometric Strategy

Formal economy-level results are generated by:

`analysis/09_country_panel_fixed_effects.py`

Despite the historical technical filename, the units are World Bank reporting
economies.

### 14.1 Pooled OLS Benchmark

\[
\ln(CO_{2,it})
=
\alpha
+
\beta\ln(GDP_{it})
+
\varepsilon_{it}
\]

This combines between-economy and within-economy variation and is used only as
a benchmark.

### 14.2 Economy Fixed Effects

\[
\ln(CO_{2,it})
=
\alpha_i
+
\beta\ln(GDP_{it})
+
\varepsilon_{it}
\]

Economy fixed effects absorb persistent characteristics such as geography,
resource endowments, historical industrial structure and other
time-invariant economy-specific factors.

### 14.3 Economy + Year Fixed Effects

The preferred specification is:

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

- \(\alpha_i\) = economy fixed effects;
- \(\lambda_t\) = year fixed effects.

Year effects control for shocks shared across economies, including global
recessions, energy-market shocks, common technological changes and other
year-specific global developments.

Standard errors are clustered at the economy level using finite-sample
correction and t-based inference.

### 14.4 Balanced-Panel Robustness

The preferred two-way fixed-effects specification is re-estimated using only
the **180 economies observed in every year from 1990 through 2024**.

This tests whether changing panel composition materially affects the estimated
GDP-emissions elasticity.

### 14.5 Population-Weighted Robustness

The two-way fixed-effects model is also estimated using population weights.

The unweighted specification remains the main model because it approximates
the relationship for the average reporting economy.

Population weighting instead places greater influence on developments in
high-population economies and shifts the estimand towards the experience of
the average person.

### 14.6 First-Difference Robustness

The economy-level analysis also estimates:

\[
\Delta\ln(CO_{2,it})
=
\lambda_t
+
\beta\Delta\ln(GDP_{it})
+
\varepsilon_{it}
\]

using only consecutive economy-year pairs.

The differencing procedure explicitly verifies that the two observations are
one year apart, preventing multi-year gaps from being incorrectly treated as
annual changes.

The resulting sample contains:

- **197 economies**;
- **6,528 consecutive economy-year changes**;
- 1991–2024.

---

## 15. Main Economy-Level Results

| Specification | GDP-CO₂ Elasticity | 95% CI | Role |
|---|---:|---:|---|
| Pooled OLS | **0.829** | [0.716, 0.942] | Benchmark |
| Economy fixed effects | **0.674** | [0.527, 0.821] | Controls for persistent economy heterogeneity |
| **Economy + year fixed effects** | **0.653** | **[0.505, 0.801]** | **Main specification** |
| Balanced-panel economy + year FE | **0.613** | [0.457, 0.769] | Coverage robustness |
| Population-weighted economy + year FE | **0.724** | [0.583, 0.865] | Population-weighting robustness |
| First differences + year FE | **0.459** | [0.350, 0.568] | Trend robustness |

All reported GDP coefficients are statistically significant at conventional
levels (\(p<0.001\)).

### 15.1 Pooled versus Fixed Effects

The pooled estimate is:

\[
\beta = 0.829
\]

while the preferred two-way fixed-effects estimate is:

\[
\boxed{\beta = 0.653}
\]

The coefficient therefore falls by approximately **21%** once persistent
economy characteristics and common year shocks are accounted for.

This indicates that part of the raw cross-economy relationship reflects
persistent structural differences between economies.

However, the positive relationship remains economically substantial and
statistically precise after these controls are introduced.

### 15.2 Main Two-Way Fixed-Effects Result

The preferred estimate is:

\[
\boxed{
\beta = 0.653
}
\]

with:

\[
95\%\,CI =
[0.505,\ 0.801]
\]

and:

\[
p<0.001
\]

The result implies that, within an economy over time, a 1% increase in GDP per
capita is associated with approximately a **0.65% increase in CO₂ emissions
per capita**, conditional on economy fixed effects and common year effects.

This is an association rather than a causal estimate.

Because:

\[
0 < \beta < 1
\]

emissions rise less than proportionally with income on average.

The global economy-level results are therefore consistent with a positive but
sub-unitary income-emissions elasticity rather than average absolute
decoupling.

### 15.3 Balanced-Panel Robustness

Restricting the sample to 180 completely balanced economies produces:

\[
\beta = 0.613
\]

compared with:

\[
0.653
\]

in the main model.

The balanced estimate is only about **6% lower**, providing evidence that
changes in economy coverage are not driving the principal result.

### 15.4 Population-Weighted Robustness

The population-weighted two-way fixed-effects model gives:

\[
\beta = 0.724
\]

approximately **11% higher** than the unweighted main estimate.

The positive relationship therefore remains strong when more populous
economies receive greater influence.

### 15.5 First-Difference Robustness

The first-difference estimate is:

\[
\boxed{
\beta_{\Delta}=0.459
}
\]

with:

\[
95\%\,CI=
[0.350,\ 0.568]
\]

and:

\[
p<0.001
\]

This indicates that annual GDP-per-capita growth remains positively associated
with annual emissions growth after removing the long-run levels trend.

The first-difference elasticity is smaller than the levels two-way
fixed-effects estimate, suggesting that short-run annual co-movement is weaker
than the longer-run within-economy association.

The positive result nevertheless substantially reduces the concern that the
main relationship is simply produced by GDP and emissions sharing common
long-run trends.

---

## 16. Interpretation of the Combined Evidence

The regional and economy-level analyses point to a consistent but
heterogeneous relationship between economic development and carbon emissions.

At the global economy level, GDP per capita remains positively associated with
CO₂ emissions per capita after controlling for persistent economy-specific
characteristics and common global shocks.

The estimated elasticity falls from approximately **0.83** in pooled OLS to
approximately **0.65** under economy and year fixed effects, indicating that
persistent structural differences explain part, but not all, of the raw
income-emissions relationship.

The positive association survives:

- balanced-panel estimation;
- population weighting;
- first differencing.

At the same time, the regional analysis shows that this global average masks
substantial heterogeneity.

China, India and the Global South display positive development-emissions
trajectories, while high-income clusters show stronger evidence of long-run
flattening or declining emissions at higher income levels.

The emerging evidence therefore does not support either of two extreme
claims:

1. that economic growth universally produces the same increase in emissions;
   or
2. that economic growth automatically causes emissions to decline once income
   becomes sufficiently high.

Instead, the relationship appears to depend on development trajectory and
structural change.

The next empirical stage formally tests this regional heterogeneity within the
economy-level fixed-effects framework.

---

## 17. Current Machine-Readable Outputs

### Regional regressions

Generated by:

`analysis/08_regional_regressions.py`

Stored under:

`data/analysis/regression_results/`

Key outputs include:

- `pooled_loglog_results.csv`
- `regional_loglog_results.csv`
- `regional_ekc_results.csv`
- `linear_vs_ekc_comparison.csv`
- `regional_first_difference_results.csv`

### Economy-level panel regressions

Generated by:

`analysis/09_country_panel_fixed_effects.py`

Stored under:

`data/analysis/country_panel_results/`

Key outputs include:

- `country_panel_sample_summary.csv`
- `country_panel_exclusion_audit.csv`
- `country_panel_economy_universe.csv`
- `country_panel_country_coverage.csv`
- `country_panel_region_coverage.csv`
- `country_panel_year_coverage.csv`
- `unmapped_iso_codes_in_analysis_window.csv`
- `country_panel_fe_results.csv`
- `country_panel_first_difference_results.csv`
- `country_panel_first_difference_coverage.csv`
- `country_panel_first_difference_region_coverage.csv`
- `country_panel_model_comparison.csv`
- `country_panel_main_sample_information_criteria.csv`

Full Statsmodels summaries are reproducible from the analysis scripts and are
kept as generated local outputs rather than core repository results.

---

## 18. Next Econometric Stage: Formal Regional Heterogeneity

The next stage will test whether the within-economy GDP-emissions elasticity
differs statistically across the five regional-development clusters.

The planned specification is:

\[
\ln(CO_{2,it})
=
\alpha_i
+
\lambda_t
+
\sum_r
\beta_r
\left[
Region_{ir}\times\ln(GDP_{it})
\right]
+
\varepsilon_{it}
\]

Because an economy's regional classification does not change over time, the
standalone region indicators are absorbed by economy fixed effects.

The interaction terms remain identifiable and allow the income-emissions
elasticity to vary across development clusters.

This stage will estimate regional slopes and formally test hypotheses such as:

\[
H_0:
\beta_{\text{China}}
=
\beta_{\text{Europe/NorthAmerica}}
\]

and analogous comparisons across the other clusters.

China and India each contain one economy and will therefore be interpreted as
individual economy trajectories rather than multi-economy regional averages.

---

## 19. Planned Direct Decoupling Analysis

A later stage will measure decoupling directly rather than inferring it solely
from regression slopes.

### Absolute Decoupling

GDP per capita rises while CO₂ emissions per capita fall.

### Relative Decoupling

GDP and emissions both rise, but emissions increase more slowly than GDP.

This will allow the descriptive language of "decoupling" to be linked to
explicit observed changes over defined time periods.

---

## 20. Planned Governance Extension

World Governance Indicators will be incorporated after the core
growth-emissions relationship and regional heterogeneity results are
established.

Candidate indicators include:

- Government Effectiveness;
- Regulatory Quality;
- Rule of Law.

Governance will be merged at the economy-year level so that cross-economy and
within-economy institutional variation is retained.

An initial specification will estimate the direct conditional association
between governance and emissions.

A later interaction model may test whether governance changes the
income-emissions relationship:

\[
\ln(CO_{2,it})
=
\alpha_i
+
\lambda_t
+
\beta\ln(GDP_{it})
+
\gamma Governance_{it}
+
\delta
[
Governance_{it}\times\ln(GDP_{it})
]
+
\varepsilon_{it}
\]

The interaction coefficient \(\delta\) will be central to evaluating whether
institutional quality is associated with a weaker growth-emissions link.

---

## 21. Limitations

The current analysis has several important limitations.

- The regressions are observational and do not establish causality.
- Regional aggregation masks substantial within-region heterogeneity.
- The regional regressions contain only 35 annual observations per cluster.
- GDP and emissions exhibit strong time trends.
- First differencing reduces, but does not eliminate, all time-series concerns.
- The World Bank economy universe includes territories and special
  administrative regions as well as sovereign states.
- The main fixed-effects coefficient represents an average across economies
  and may mask substantial regional heterogeneity.
- The Global South contributes the majority of economy-year observations to
  the current global panel.
- China and India are standalone economy trajectories rather than
  multi-economy regional clusters.
- The latest years, particularly 2024, have somewhat lower economy coverage
  than much of the earlier panel.
- Energy mix, industrial structure, trade-embedded emissions and other
  structural factors are not yet explicitly controlled for.
- Governance indicators contain measurement uncertainty.
- Regional EKC turning points should be interpreted cautiously until supported
  by the economy-level heterogeneity analysis.

For these reasons, the project focuses on robust association, heterogeneity
and empirical pattern rather than causal claims.

---

## 22. Repository Structure

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
│   ├── 09_country_panel_fixed_effects.py
│   └── regional aggregation / data-processing scripts
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
│       ├── regression_results/
│       └── country_panel_results/
│
├── figures/
│   ├── regional descriptive figures
│   └── loglog_by_region/
│
├── README.md
├── requirements.txt
└── .gitignore
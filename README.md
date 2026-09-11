# Economic Growth, CO₂ Emissions, and Governance

## Regional Heterogeneity in a Global Economy-Level Panel

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

The empirical design combines two complementary levels of analysis:

1. a population-weighted regional panel covering five regional-development
   clusters; and
2. an economy-year panel covering World Bank reporting economies.

The regional analysis describes long-run development-emissions trajectories
and investigates possible Environmental Kuznets Curve (EKC) patterns.

The economy-level panel provides the main inferential framework through:

- economy fixed effects;
- year fixed effects;
- economy-clustered inference;
- balanced-panel robustness;
- population-weighted robustness;
- first-difference specifications;
- formal regional slope interactions.

The core econometric results show that the positive global GDP-emissions
relationship masks substantial heterogeneity across development trajectories.

---

## 2. Research Questions

### Primary Question

How does the relationship between GDP per capita and CO₂ emissions per capita
differ across economies, development clusters, and over time?

### Secondary Questions

- Does the income-emissions relationship differ systematically across
  regional-development clusters?
- Does the positive global GDP-emissions association survive economy and year
  fixed effects?
- Is there evidence of long-run decoupling at higher income levels?
- Is there empirical support for Environmental Kuznets Curve behaviour?
- Does the responsiveness of emissions to income weaken as economies develop?
- Do high-income and developing economies exhibit different short-run
  growth-emissions relationships?
- Do stronger institutions help explain differences in long-run
  development-emissions trajectories?

---

## 3. Conceptual Framework

### 3.1 Growth-Emissions Channel

Economic development can affect emissions differently across stages of
development.

A simplified development pathway is:

- early development → industrialisation, infrastructure expansion and greater
  fossil-fuel use → rising emissions;
- later development → structural transformation, technological upgrading,
  energy efficiency, cleaner energy systems and environmental regulation →
  weaker emissions growth or declining emissions.

This motivates examining both linear and non-linear income-emissions
relationships.

### 3.2 Structural Decoupling

Economic growth and emissions need not move proportionately.

An economy may experience:

- **absolute decoupling**, where GDP per capita rises while CO₂ emissions per
  capita fall;
- **relative decoupling**, where both GDP and emissions rise, but emissions
  increase more slowly than GDP;
- **continued coupling**, where emissions rise at least as quickly as income.

Regression elasticities provide evidence about the strength of the
income-emissions relationship, but direct decoupling is treated as a separate
empirical concept and is measured explicitly using observed changes.

### 3.3 Governance Channel

Institutional quality may influence long-run emissions trajectories through:

- environmental regulation and enforcement;
- public investment in low-carbon infrastructure;
- renewable-energy policy;
- carbon-pricing mechanisms;
- green innovation;
- energy and industrial policy.

Governance is therefore treated as a potential mechanism capable of changing
the strength of the growth-emissions relationship rather than as an
established causal result.

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
jurisdictions**, rather than sovereign states only.

The project deliberately retains the full World Bank economy universe,
including territories and special administrative regions where the required
data are available.

All 217 economy codes have an explicit regional assignment in:

`data/meta/country_regions.csv`

There are no unmatched economy codes in the cleaned WDI analysis window.

The principal log GDP-log CO₂ regression sample contains **197 economies**.

Of the remaining 20 mapped economies:

- 19 have no usable positive GDP-CO₂ economy-year observations during
  1990–2024;
- New Caledonia has only one usable joint observation and therefore does not
  satisfy the minimum two-observation requirement for panel estimation.

The 70 non-positive CO₂ observations in the source panel are entirely
accounted for by Nauru and Tuvalu, whose CO₂ series are recorded as zero
throughout the analysis period. Because the logarithm of zero is undefined,
these observations cannot enter logarithmic specifications.

The complete economy-universe audit is stored at:

`data/analysis/country_panel_results/country_panel_economy_universe.csv`

---

## 5. Economy-Year Dataset Construction

The original WDI download is supplied in wide format with years stored as
columns.

The data pipeline:

1. loads the World Bank WDI extract;
2. programmatically identifies year columns;
3. reshapes the data from wide to long format;
4. converts observations to numeric form;
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

The terms `country` and `country_panel` remain in some technical filenames and
variable names for continuity with the original pipeline. The empirical units
should be interpreted as World Bank reporting economies.

---

## 6. Regional-Development Classification

The 217 World Bank economies are assigned to five regional-development
clusters:

1. `Europe_NorthAmerica`
2. `DevelopedAsia_Oceania`
3. `China`
4. `India`
5. `Global_South`

These groups are intended to capture differences in development paths,
institutional maturity, industrialisation history and emissions structure.
They should therefore be interpreted as **regional-development clusters**
rather than purely geographical regions.

### 6.1 China, Hong Kong and Macao

Mainland China (`CHN`) forms the standalone China cluster.

Hong Kong SAR (`HKG`) and Macao SAR (`MAC`) are classified within
Developed Asia & Oceania because their development trajectories and economic
structures differ substantially from mainland China.

The classification therefore follows development characteristics rather than
political geography alone.

China and India are retained as standalone groups because of their population,
economic scale and distinctive development-emissions trajectories.

The explicit mapping is stored at:

`data/meta/country_regions.csv`

### 6.2 Full Economy Mapping

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
\sum_i GDPpc_{irt}\times Population_{irt}
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

The regional panel contains:

- 5 regional-development clusters;
- 35 annual observations per cluster;
- 1990–2024;
- 175 region-year observations.

---

## 8. Data Validation and Reproducibility

The regional panel is independently validated using:

`analysis/00_validate_region_panel.py`

Checks include:

- expected regional labels;
- duplicate economy-year observations;
- duplicate region-year observations;
- missing values;
- positive GDP, emissions and population values;
- continuous time coverage;
- economy coverage by region-year;
- population coverage;
- independent reconstruction of population-weighted aggregates.

Quality-control outputs are stored under:

`data/analysis/qc/`

Dataset checksums are stored at:

`data/meta/data_checksums.sha256`

The economy-level regression pipeline performs additional checks for:

- duplicate economy-year keys;
- complete regional mapping;
- regression-sample eligibility;
- positive values prior to logarithmic transformation;
- balanced-panel status;
- regional and annual sample coverage;
- consecutive-year validity before first differencing;
- consistency between successive regression scripts.

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

The descriptive evidence reveals substantial differences in development and
emissions trajectories across the five clusters.

---

## 10. Regional Benchmark Econometrics

Formal regional benchmark results are generated by:

`analysis/08_regional_regressions.py`

### 10.1 Region-Specific Log-Log Models

For each cluster:

\[
\ln(CO_{2,rt})
=
\alpha_r+
\beta_r\ln(GDP_{rt})
+\varepsilon_{rt}
\]

Region-specific regressions use Newey-West/HAC inference with small-sample
correction and t-based inference.

### 10.2 Environmental Kuznets Curve Models

Regional non-linearity is examined using:

\[
\ln(CO_{2,rt})
=
\alpha_r+
\beta_{1r}\ln(GDP_{rt})
+
\beta_{2r}[\ln(GDP_{rt})]^2
+
\varepsilon_{rt}
\]

A region is treated as a preliminary within-sample EKC candidate only if:

- the estimated curve has an inverted-U sign pattern;
- the quadratic term is statistically meaningful;
- the implied turning point lies within the observed income range.

Model comparison also considers adjusted \(R^2\), AIC and BIC.

### 10.3 Regional First Differences

The regional analysis additionally estimates:

\[
\Delta\ln(CO_{2,rt})
=
\alpha_r+
\beta_r\Delta\ln(GDP_{rt})
+
\varepsilon_{rt}
\]

to investigate whether annual GDP and emissions changes co-move after removing
their long-run levels trends.

---

## 11. Regional Benchmark Results

| Region | Levels GDP-CO₂ elasticity | First-difference elasticity | Preferred shape | EKC assessment |
|---|---:|---:|---|---|
| Europe & North America | **−0.539** | **+1.256** | Non-linear | Possible high-income EKC / long-run decoupling pattern |
| Developed Asia & Oceania | +0.112, statistically imprecise | **+1.064** | Strongly non-linear | Strongest preliminary within-sample EKC candidate |
| China | **+0.621** | **+1.018** | Linear | No credible observed EKC |
| India | **+0.747** | **+0.725** | Linear | No credible observed EKC |
| Global South | **+0.687** | **+0.800** | Non-linear | Significant concavity; turning point outside observed range |

These models describe aggregate regional trajectories and should not be
interpreted as equivalent to the economy-level fixed-effects estimates below.

### 11.1 High-Income Regional Trajectories

Europe & North America exhibits a negative long-run regional levels
relationship and a preferred non-linear specification, consistent with
high-income decoupling.

Developed Asia & Oceania provides the strongest preliminary regional evidence
of an inverted-U trajectory, with an estimated turning point around
**$32,900 GDP per capita** located near the middle of its observed income
range.

However, aggregate regional relationships do not imply that every constituent
economy follows the same trajectory.

### 11.2 Developing Regional Trajectories

China and India exhibit strong positive relationships between GDP per capita
and CO₂ emissions per capita, with little support for an observed quadratic
turning point.

The Global South also exhibits a strong positive relationship, although the
quadratic specification indicates that the responsiveness of emissions to
income weakens as income rises.

Its implied turning point remains above the observed regional income range.

---

## 12. Economy-Level Panel Sample

Regional aggregation provides a useful long-run narrative but removes
substantial cross-economy variation.

The main econometric analysis therefore returns to the underlying economy-year
panel.

The principal levels estimation sample contains:

- **197 World Bank economies**;
- **6,725 economy-year observations**;
- **35 potential years**, 1990–2024;
- median coverage of **35 years per economy**;
- mean coverage of approximately **34.14 years per economy**;
- **180 completely balanced economies**.

The theoretical maximum is:

\[
197\times35=6,895
\]

so the estimation panel contains approximately:

\[
\frac{6,725}{6,895}\approx97.5\%
\]

of all possible observations.

The incomplete economy series contain no internal missing-year gaps in the
final estimation sample; missing observations occur at the beginning or end of
individual series.

### 12.1 Estimation-Sample Distribution

| Development cluster | Economies | Economy-year observations |
|---|---:|---:|
| Europe & North America | 43 | 1,486 |
| Developed Asia & Oceania | 8 | 280 |
| China | 1 | 35 |
| India | 1 | 35 |
| Global South | 144 | 4,889 |
| **Total** | **197** | **6,725** |

The unequal distribution means that a single global coefficient should not be
interpreted as equally representative of every development cluster.

---

## 13. Global Economy-Level Fixed Effects

Economy-level fixed-effects results are generated by:

`analysis/09_country_panel_fixed_effects.py`

### 13.1 Main Specification

The preferred common-slope specification is:

\[
\ln(CO_{2,it})
=
\alpha_i+
\lambda_t+
\beta\ln(GDP_{it})
+
\varepsilon_{it}
\]

where:

- \(\alpha_i\) = economy fixed effects;
- \(\lambda_t\) = year fixed effects.

Standard errors are clustered at the economy level using finite-sample
correction and t-based inference.

### 13.2 Main Results

| Specification | GDP-CO₂ elasticity | 95% CI | Role |
|---|---:|---:|---|
| Pooled OLS | **0.829** | [0.716, 0.942] | Benchmark |
| Economy fixed effects | **0.674** | [0.527, 0.821] | Persistent economy heterogeneity |
| **Economy + year fixed effects** | **0.653** | **[0.505, 0.801]** | **Main common-slope specification** |
| Balanced-panel TWFE | **0.613** | [0.457, 0.769] | Coverage robustness |
| Population-weighted TWFE | **0.724** | [0.583, 0.865] | Population-weighting robustness |
| First differences + year FE | **0.459** | [0.350, 0.568] | Trend robustness |

The preferred common-slope estimate is:

\[
\boxed{\beta=0.653}
\]

with:

\[
95\%\,CI=[0.505,\;0.801].
\]

A 1% within-economy increase in GDP per capita is therefore associated with
approximately a **0.65% increase in CO₂ emissions per capita**, conditional on
economy fixed effects and common year shocks.

The estimate is an association rather than a causal effect.

The coefficient is below one, implying that emissions rise less than
proportionately with income on average, but remain positively associated with
economic development.

---

## 14. Formal Regional Heterogeneity

The common global coefficient imposes the assumption:

\[
\beta_{\text{Europe}}
=
\beta_{\text{Developed Asia}}
=
\beta_{\text{China}}
=
\beta_{\text{India}}
=
\beta_{\text{Global South}}.
\]

This restriction is formally tested by:

`analysis/10_regional_heterogeneity.py`

The heterogeneous specification is:

\[
\ln(CO_{2,it})
=
\alpha_i+
\lambda_t+
\sum_r
\beta_r
\left[
Region_{ir}\times\ln(GDP_{it})
\right]
+
\varepsilon_{it}.
\]

Because regional membership is time invariant, standalone region indicators
are absorbed by economy fixed effects. The interaction slopes remain
identified through within-economy changes in GDP.

---

## 15. Main Regional Fixed-Effects Results

| Development cluster | GDP-CO₂ elasticity | 95% CI | Economies | Interpretation |
|---|---:|---:|---:|---|
| Europe & North America | **0.086** | [−0.322, 0.495] | 43 | No statistically detectable average levels relationship |
| Developed Asia & Oceania | **−0.056** | [−0.357, 0.246] | 8 | Approximately flat unweighted average relationship |
| China | **0.592** | [0.545, 0.639]* | 1 | Large positive point estimate |
| India | **0.699** | [0.618, 0.781]* | 1 | Large positive point estimate |
| Global South | **0.801** | [0.634, 0.969] | 144 | Strong positive relationship |

\* China and India are single-economy development groups. Their point estimates
are economically informative, but conventional economy-clustered confidence
intervals and p-values are not used as primary evidence of regional
heterogeneity.

The central pattern is therefore:

\[
\boxed{
\text{comparatively flat high-income relationships}
\quad\text{versus}\quad
\text{a strongly positive Global South relationship}.
}
\]

---

## 16. Formal Tests of Regional Heterogeneity

### 16.1 Preferred Multi-Economy Joint Test

The preferred formal test is restricted to the three development groups
containing multiple economies:

- Europe & North America;
- Developed Asia & Oceania;
- Global South.

The null hypothesis is:

\[
H_0:
\beta_{\text{Europe}}
=
\beta_{\text{Developed Asia}}
=
\beta_{\text{Global South}}.
\]

The result is:

\[
\boxed{
F(2,196)=17.20,\qquad p<0.001
}
\]

with an exact p-value of approximately \(1.3\times10^{-7}\).

The equality of the three multi-economy slopes is therefore strongly rejected.

### 16.2 Five-Group Joint Test

The corresponding five-group test gives:

\[
F(4,196)=21.20,\qquad p<0.001.
\]

Because this test includes the single-economy China and India groups, it is
treated as supplementary rather than the principal inferential result.

---

## 17. Pairwise Regional Comparisons

Among the three multi-economy groups, Holm-adjusted pairwise tests show:

### Europe & North America vs Developed Asia & Oceania

\[
\hat\beta_E-\hat\beta_D\approx0.142
\]

The difference is not statistically significant after multiple-testing
adjustment.

### Europe & North America vs Global South

\[
\hat\beta_E-\hat\beta_G\approx-0.715
\]

with a Holm-adjusted p-value of approximately:

\[
0.013.
\]

The Global South therefore exhibits a significantly steeper positive
GDP-emissions relationship.

### Developed Asia & Oceania vs Global South

\[
\hat\beta_D-\hat\beta_G\approx-0.857
\]

with:

\[
p_{\text{Holm}}<0.001.
\]

This provides particularly strong evidence that the Global South's
income-emissions relationship is steeper than the unweighted Developed Asia &
Oceania relationship.

Comparisons involving China and India are retained for completeness but are
treated cautiously because each is a single-economy development group.

---

## 18. Common-Slope vs Heterogeneous-Slope Model

Allowing the GDP-emissions elasticity to vary across development groups
improves model fit relative to the Script 09 common-slope specification.

### Common-Slope TWFE

\[
R^2\approx0.9667
\]

\[
\bar R^2\approx0.9655
\]

### Region-Specific-Slope TWFE

\[
R^2\approx0.9689
\]

\[
\bar R^2\approx0.9677.
\]

The heterogeneous model also produces approximately:

\[
\Delta AIC=-447.5
\]

and:

\[
\Delta BIC=-420.3
\]

relative to the common-slope model.

Because both models use the same dependent variable and estimation sample,
these information criteria provide supplementary evidence in favour of
regional heterogeneity.

The cluster-robust joint tests remain the primary inferential evidence.

---

## 19. Regional-Heterogeneity Robustness

### 19.1 Balanced Panel

Restricting the analysis to 180 economies observed throughout 1990–2024 gives
approximately:

| Development cluster | Main TWFE | Balanced-panel TWFE |
|---|---:|---:|
| Europe & North America | +0.086 | −0.100 |
| Developed Asia & Oceania | −0.056 | −0.043 |
| China | +0.592 | +0.596 |
| India | +0.699 | +0.706 |
| Global South | +0.801 | +0.795 |

The preferred multi-economy equality test remains strongly significant:

\[
F(2,179)\approx25.13,\qquad p<0.001.
\]

Changing economy coverage therefore does not drive the principal
heterogeneity result.

### 19.2 Population Weighting

Population-weighted estimates are approximately:

| Development cluster | Unweighted TWFE | Population-weighted TWFE |
|---|---:|---:|
| Europe & North America | +0.086 | −0.080 |
| Developed Asia & Oceania | −0.056 | +0.408 |
| China | +0.592 | +0.670 |
| India | +0.699 | +0.835 |
| Global South | +0.801 | +0.985 |

The most substantial change occurs in Developed Asia & Oceania.

The unweighted specification estimates the relationship for something closer
to the average economy, while population weighting gives more influence to
larger economies and moves the estimand towards the experience of the average
person.

This difference indicates meaningful within-group heterogeneity rather than a
failure of the main specification.

The population-weighted multi-economy joint test remains strongly significant:

\[
F(2,196)\approx24.15,\qquad p<0.001.
\]

### 19.3 First Differences

The regional first-difference specification is:

\[
\Delta\ln(CO_{2,it})
=
\gamma_r+
\lambda_t+
\sum_r
\beta_r
[
Region_{ir}\times\Delta\ln(GDP_{it})
]
+
\varepsilon_{it}.
\]

Regional intercepts \(\gamma_r\) allow average annual emissions growth to
differ across development groups independently of GDP growth.

The estimated short-run elasticities are approximately:

| Development cluster | First-difference elasticity |
|---|---:|
| Europe & North America | **0.745** |
| Developed Asia & Oceania | **0.031** |
| China | **0.784*** |
| India | **0.520*** |
| Global South | **0.451** |

\* Single-economy inference caveat applies.

The preferred three-group first-difference equality test gives:

\[
F(2,196)\approx17.61,\qquad p<0.001.
\]

Short-run GDP-emissions relationships therefore also differ materially across
development clusters.

---

## 20. Interpreting Levels and First Differences Together

The levels and first-difference models answer different economic questions.

For Europe & North America, the levels elasticity is approximately:

\[
0.086
\]

while the annual first-difference elasticity is approximately:

\[
0.745.
\]

This is consistent with a setting in which long-run structural transformation
weakens the income-emissions relationship, while short-run economic
expansions remain associated with higher emissions growth.

Developed Asia & Oceania exhibits an approximately zero unweighted elasticity
in both the levels and first-difference specifications, although
population-weighted results are more positive.

The Global South exhibits:

\[
\beta_{\text{levels}}\approx0.801
\]

and:

\[
\beta_{\Delta}\approx0.451.
\]

Its growth-emissions relationship therefore remains positive in both longer-run
within-economy variation and annual changes.

---

## 21. Direct Decoupling Analysis

Regression elasticities do not by themselves establish whether an economy is
experiencing absolute or relative decoupling.

The direct-decoupling module is reserved as:

`analysis/11_direct_decoupling.py`

Machine-readable outputs from this stage are reserved under:

`data/analysis/decoupling_results/`

The analysis uses observed log changes in GDP per capita and CO₂ emissions per
capita.

For consecutive observations where GDP per capita increases:

### Absolute Decoupling

\[
\Delta\ln GDP>0
\]

and:

\[
\Delta\ln CO_2<0.
\]

### Relative Decoupling

\[
\Delta\ln GDP>0,
\]

\[
\Delta\ln CO_2\ge0,
\]

and:

\[
\Delta\ln CO_2<\Delta\ln GDP.
\]

### Coupled Expansion

\[
\Delta\ln GDP>0
\]

and:

\[
\Delta\ln CO_2\ge\Delta\ln GDP.
\]

Periods of economic contraction are classified separately rather than being
labelled as successful decoupling simply because emissions also decline.

The module is designed to summarise decoupling patterns:

- by economy;
- by regional-development cluster;
- across time;
- using both annual changes and longer-period comparisons where appropriate.

This direct classification complements, rather than replaces, the regression
analysis.

---

## 22. Governance Extension

World Governance Indicators will be incorporated after the core
growth-emissions relationship and regional heterogeneity are established.

Candidate indicators include:

- Government Effectiveness;
- Regulatory Quality;
- Rule of Law.

Governance will be merged at economy-year level.

A baseline governance specification can be written as:

\[
\ln(CO_{2,it})
=
\alpha_i+
\lambda_t+
\beta\ln(GDP_{it})
+
\gamma Governance_{it}
+
\varepsilon_{it}.
\]

The more substantively important specification allows governance to modify the
income-emissions relationship:

\[
\ln(CO_{2,it})
=
\alpha_i+
\lambda_t+
\beta\ln(GDP_{it})
+
\gamma Governance_{it}
+
\delta
[
Governance_{it}\times\ln(GDP_{it})
]
+
\varepsilon_{it}.
\]

If:

\[
\delta<0,
\]

stronger governance would be associated with a weaker GDP-emissions
relationship, conditional on the model specification.

No causal interpretation is assumed from this observational design.

---

## 23. Interpretation of the Combined Evidence

The project produces three complementary empirical findings.

First, aggregate regional trajectories differ substantially. High-income
regional aggregates show stronger evidence of flattening, non-linearity and
long-run decoupling, while developing regional aggregates remain more strongly
associated with rising emissions.

Second, across the 197-economy panel, GDP per capita remains positively
associated with CO₂ emissions per capita after controlling for persistent
economy characteristics and common year shocks:

\[
\beta_{\text{common}}\approx0.653.
\]

Third, the common global elasticity masks substantial development-stage
heterogeneity.

In the preferred heterogeneous two-way fixed-effects model:

\[
\beta_{\text{Europe/NA}}\approx0.086,
\]

\[
\beta_{\text{Developed Asia/Oceania}}\approx-0.056,
\]

and:

\[
\beta_{\text{Global South}}\approx0.801.
\]

The equality of these three multi-economy slopes is strongly rejected.

The evidence therefore does not support a single universal
income-emissions relationship.

Instead, the strength of coupling between economic development and emissions
appears to depend materially on development trajectory and structural
conditions.

---

## 24. Statistical Interpretation and Caveats

### 24.1 Association, Not Causality

All regressions are observational.

Statements should therefore use language such as:

- "is associated with";
- "is consistent with";
- "the estimated relationship";

rather than causal wording.

### 24.2 China and India

China and India deliberately form standalone development groups.

Their slope point estimates are economically useful, but each is identified
from a single economy trajectory.

Conventional economy-clustered confidence intervals, p-values and pairwise
tests involving these groups are therefore treated as supplementary rather
than primary evidence.

### 24.3 Developed Asia & Oceania

Developed Asia & Oceania contains only eight economies.

Economy-clustered inference is available, but its small number of constituent
economies warrants more caution than inference for Europe & North America or
the Global South.

The central regional-heterogeneity result does not depend solely on this
eight-economy cluster: the Europe/North America versus Global South comparison
also shows a statistically meaningful difference.

### 24.4 Fixed-Effects \(R^2\)

The high \(R^2\) values of the levels regressions should not be interpreted as
GDP alone explaining almost all emissions variation.

Economy and year fixed effects account for substantial levels variation.

The economically relevant evidence is therefore concentrated in:

- coefficient estimates;
- confidence intervals;
- joint slope tests;
- pairwise comparisons;
- robustness specifications.

---

## 25. Main Machine-Readable Outputs

### Regional Benchmark Regressions

Generated by:

`analysis/08_regional_regressions.py`

Stored under:

`data/analysis/regression_results/`

Principal outputs include:

- `pooled_loglog_results.csv`
- `regional_loglog_results.csv`
- `regional_ekc_results.csv`
- `linear_vs_ekc_comparison.csv`
- `regional_first_difference_results.csv`

### Economy-Level Fixed Effects

Generated by:

`analysis/09_country_panel_fixed_effects.py`

Stored under:

`data/analysis/country_panel_results/`

Principal outputs include:

- `country_panel_sample_summary.csv`
- `country_panel_exclusion_audit.csv`
- `country_panel_economy_universe.csv`
- `country_panel_country_coverage.csv`
- `country_panel_region_coverage.csv`
- `country_panel_year_coverage.csv`
- `country_panel_fe_results.csv`
- `country_panel_first_difference_results.csv`
- `country_panel_model_comparison.csv`

### Formal Regional Heterogeneity

Generated by:

`analysis/10_regional_heterogeneity.py`

Stored under:

`data/analysis/country_panel_heterogeneity_results/`

Principal outputs include:

- `regional_heterogeneity_region_sample_summary.csv`
- `regional_heterogeneity_sample_summary.csv`
- `main_regional_twfe_slopes.csv`
- `regional_heterogeneity_slopes.csv`
- `regional_heterogeneity_joint_tests.csv`
- `main_regional_pairwise_slope_tests.csv`
- `regional_pairwise_slope_tests.csv`
- `common_vs_heterogeneous_model_comparison.csv`
- `regional_slope_robustness_comparison.csv`

The principal coefficient figure is:

`figures/10_regional_twfe_elasticities.png`

Generated full Statsmodels summaries are stored locally and can be recreated
from the scripts.

---

## 26. Repository Structure

```text
worldbank_project/
│
├── analysis/
│   ├── 00_validate_region_panel.py
│   ├── 01_...
│   ├── 02_...
│   ├── 03_...
│   ├── 04_...
│   ├── 05_...
│   ├── 06_...
│   ├── 07_...
│   ├── 08_regional_regressions.py
│   ├── 09_country_panel_fixed_effects.py
│   ├── 10_regional_heterogeneity.py
│   └── numbered extension modules
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
│       ├── country_panel_results/
│       ├── country_panel_heterogeneity_results/
│       └── decoupling_results/
│
├── figures/
│   ├── regional descriptive figures
│   ├── loglog_by_region/
│   └── 10_regional_twfe_elasticities.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

The path:

`analysis/11_direct_decoupling.py`

is reserved for the direct-decoupling module, with outputs under:

`data/analysis/decoupling_results/`

when that module is run.

---

## 27. Reproducibility

Install dependencies using:

```bash
python3 -m pip install -r requirements.txt
```

Run regional-panel validation:

```bash
python3 analysis/00_validate_region_panel.py
```

Run regional benchmark regressions:

```bash
python3 analysis/08_regional_regressions.py
```

Run economy-level fixed effects:

```bash
python3 analysis/09_country_panel_fixed_effects.py
```

Run formal regional heterogeneity analysis:

```bash
python3 analysis/10_regional_heterogeneity.py
```

The scripts use paths relative to the repository root and therefore do not
depend on a particular local Desktop path.

The direct-decoupling module follows the reserved path:

```bash
python3 analysis/11_direct_decoupling.py
```

once that module is present.

---

## 28. Analytical Pipeline

The research architecture is:

```text
World Bank WDI data
        ↓
Clean economy-year panel
        ↓
Explicit economy-region mapping
        ↓
Population-weighted regional panel
        ↓
Regional descriptive analysis
        ↓
Regional log-log / EKC benchmarks
        ↓
Economy + year fixed-effects model
        ↓
Formal regional slope heterogeneity
        ↓
Direct decoupling measurement
        ↓
Governance extension
        ↓
Final robustness, interpretation and research paper
```

The first three econometric layers are complementary rather than substitutes:

- regional aggregates describe long-run development trajectories;
- common-slope fixed effects estimate the average within-economy relationship;
- regional interactions test whether that average relationship differs across
  development trajectories.

Direct decoupling then translates the statistical relationships into observed
growth-emissions outcomes.

---

## 29. Limitations

The analysis has several important limitations.

- The research design is observational and does not establish causality.
- Regional aggregation masks substantial within-region heterogeneity.
- Regional benchmark regressions contain only 35 annual observations per
  cluster.
- GDP and emissions contain strong long-run trends.
- First differencing reduces, but does not eliminate, every possible
  time-series concern.
- The World Bank economy universe includes territories and special
  administrative regions as well as sovereign states.
- The Global South contributes the majority of economy-year observations.
- China and India are standalone economy trajectories rather than
  multi-economy regions.
- Developed Asia & Oceania contains only eight economies, so cluster-based
  inference should be interpreted with additional caution.
- Population weighting changes the estimand and reveals important
  within-cluster heterogeneity.
- 2024 has somewhat lower economy coverage than much of the earlier panel.
- Energy mix, industrial structure, trade-embedded emissions and other
  structural mechanisms are not yet explicitly controlled for.
- Governance indicators contain measurement uncertainty.
- EKC turning points from aggregate regional models should not be interpreted
  as universal thresholds applicable to individual economies.

The project therefore focuses on robust association, heterogeneity and
development patterns rather than causal claims.

---

## 30. Policy Relevance

The evidence indicates that economic growth does not have a uniform
relationship with carbon emissions.

Across the global economy-level panel, economic development remains positively
associated with emissions on average.

However, formal interaction models show that this global average masks major
differences across development trajectories.

The Global South remains substantially more tightly coupled to emissions
growth than the high-income multi-economy groups in the preferred levels
specification.

This distinction is relevant for:

- climate finance;
- international burden sharing;
- energy-transition policy;
- green industrial strategy;
- technological diffusion;
- institutional development;
- sustainable-growth policy.

A policy implication is not that developing economies should avoid economic
growth, but that the technologies, institutions and energy systems through
which growth occurs are likely to determine whether development remains
carbon-intensive or becomes progressively decoupled from emissions.
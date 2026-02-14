# Economic Growth, CO₂ Emissions, and Governance  
## A Regional Panel Analysis

**Work in Progress – Climate Economics Research Project**

---

## 1. Project Overview

This project investigates the relationship between **economic growth** and **carbon emissions**, with a focus on **regional heterogeneity** and the role of **institutional quality**.

The central question is whether economic growth necessarily leads to higher emissions — and whether strong governance can weaken this link.

The analysis combines cross-country panel data with regional aggregation to examine long-run development–emissions dynamics.

---

## 2. Research Questions

### Primary Question

> What is the relationship between GDP per capita and CO₂ emissions per capita across regions and over time?

### Secondary Questions

- Does the income–emissions relationship differ systematically across regions?
- Is there evidence of **decoupling** at higher income levels?
- Do regions with stronger institutions emit less CO₂ at comparable income levels?
- Is there empirical support for the **Environmental Kuznets Curve (EKC)** hypothesis?

---

## 3. Conceptual Framework

### 3.1 Growth–Emissions Channel

Economic development often follows a structural transition:

- **Early development** → industrialisation → fossil fuel dependence → rising emissions  
- **Later development** → services, technological upgrading, regulation → slower emissions growth  

This motivates testing for an **Environmental Kuznets Curve (EKC)** relationship.

### 3.2 Governance Channel

Institutions may influence emissions through:

- Enforcement of environmental regulations  
- Public investment in renewable energy  
- Carbon pricing mechanisms  
- Support for green innovation  

**Key intuition:**  
Income enables emissions reductions, but governance determines whether they occur.

---

## 4. Data Sources

### Core Data

Data are drawn from the **World Bank – World Development Indicators (WDI)**:

- GDP per capita (constant 2015 USD)  
- CO₂ emissions per capita (AR5-consistent series)  
- Total population  

### Governance Data (Planned Extension)

From the **World Bank – World Governance Indicators (WGI)**:

- Government effectiveness  
- Regulatory quality  
- Rule of law  

Time coverage: 1990–latest available year.

---

## 5. Dataset Construction

The current pipeline produces a clean **country–year panel dataset**.

### Variables

| Variable | Description |
|-----------|-------------|
| `country` | Country name |
| `iso3c` | ISO-3 country code |
| `year` | Calendar year |
| `gdp_per_capita_const2015_usd` | GDP per capita (constant 2015 USD) |
| `co2_per_capita_tons` | CO₂ emissions per capita |
| `population` | Total population |

Each row corresponds to a **country–year observation**.

### Data Processing Steps

1. Load World Bank WDI data (wide format).
2. Detect year columns programmatically.
3. Reshape from wide to long format.
4. Convert missing values to `NaN`.
5. Filter to core indicators.
6. Pivot to one row per country–year.
7. Export cleaned panel dataset.

No aggregation or modeling is performed at this stage.

---

## 6. Regional Classification (Planned Aggregation)

Countries will be grouped into five development clusters:

1. **Europe_NorthAmerica**  
2. **Advanced_Asia_Oceania**  
3. **China** (standalone)  
4. **India** (standalone)  
5. **Global_South**

### Rationale

Regions reflect differences in:

- Stage of development  
- Institutional maturity  
- Industrialisation history  
- Emissions structure  

Japan, South Korea, Australia, New Zealand, Israel, and Singapore are classified as **Advanced_Asia_Oceania** due to their high-income status and early industrial transition.

### Aggregation Rules

- GDP per capita → population-weighted average  
- CO₂ per capita → population-weighted average  
- Governance indicators → simple average  

This produces a **region–year panel dataset** for analysis.

---

## 7. Planned Empirical Analysis

### Descriptive Analysis

- GDP vs CO₂ scatter plots (region–year observations)
- Region-specific development paths
- Time-series trends to identify decoupling

### Econometric Specifications

- Log–log income–emissions regression
- EKC specification (quadratic income term)
- Governance-augmented models
- Optional regional fixed effects

---

## 8. Output Files

Current cleaned dataset: data/cleaned/wdi_gdp_co2_population_panel.csv

Planned regional dataset: data/meta/country_regions.csv

---

## 9. Limitations

- Correlation does not imply causation.
- Governance indicators are measured with error.
- Regional aggregation masks within-region heterogeneity.
- Energy mix and structural factors are not yet controlled for.

---

## 10. Policy Relevance

Understanding whether growth automatically reduces emissions — or whether institutional strength is required — is central to:

- Climate finance allocation  
- Green industrial policy  
- Sustainable development strategy  

This project aims to contribute to the empirical discussion on growth–environment trade-offs.

---

## 11. Repository Structure

data/
raw/
cleaned/
analysis/
meta/

scripts/
notebooks/
figures/
paper/

---

## 12. Project Status

- ✔ Country–year panel constructed  
- ⏳ Regional aggregation in progress  
- ⏳ Visual analysis  
- ⏳ Econometric estimation  
- ⏳ Governance extension  

---

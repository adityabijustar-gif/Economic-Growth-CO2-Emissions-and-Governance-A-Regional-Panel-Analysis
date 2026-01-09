# Economic Growth, CO₂ Emissions, and Governance

## A Regional Panel Analysis

**Data Construction and Research Design (Work in Progress)**

---

## 1. Project Motivation

This project studies the relationship between **economic growth** and **carbon emissions** over time, with a particular focus on **regional heterogeneity** and the role of **institutions and governance**.

Understanding whether economic growth necessarily leads to higher emissions—and whether strong institutions can weaken this link—is central to both climate economics and development policy.

---

## 2. Research Questions

### Primary Question

**What is the relationship between GDP per capita and CO₂ emissions per capita across regions and over time?**

### Secondary Questions

Does this relationship differ between:

* **Europe & North America**
* **Global South**
* **China** (standalone)
* **India** (standalone)

Additional questions:

* Is there evidence of **decoupling** at higher income levels?
* Do regions with **stronger governance and institutions** emit less CO₂ at comparable income levels?

---

## 3. Hypotheses

* Economic growth is generally associated with higher CO₂ emissions.
* The income–emissions relationship varies systematically by region.
* High-income regions with strong institutions (e.g. Scandinavia) emit less CO₂ than predicted by income alone.
* Governance quality is negatively correlated with emissions intensity.

---

## 4. Conceptual Framework

### 4.1 Growth–Emissions Channel

Economic development often follows a predictable pattern:

* **Early development** → industrialisation → fossil-fuel dependence → ↑ emissions
* **Later development** → services, technology, regulation → slower emissions growth

This motivates the **Environmental Kuznets Curve (EKC)** hypothesis.

### 4.2 Governance Channel

Institutions may shape emissions outcomes through:

* Enforcement of environmental regulation
* Public investment in renewable energy
* Carbon pricing and innovation incentives

**Key intuition:**

> Income enables emissions reductions, but governance determines whether they occur.

---

## 5. Data Sources

### Core Data (Current Stage)

* **World Bank – World Development Indicators (WDI)**

  * GDP per capita (constant 2015 USD)
  * CO₂ emissions per capita (AR5-consistent series)
  * Total population

### Governance Data (Planned)

* **World Governance Indicators (WGI)**

  * Government effectiveness
  * Regulatory quality
  * Rule of law

---

## 6. Constructed Variables (Country–Year Panel)

The current script produces a clean **country–year panel dataset** with the following variables:

| Variable                       | Description                        |
| ------------------------------ | ---------------------------------- |
| `country`                      | Country name                       |
| `iso3c`                        | ISO-3 country code                 |
| `year`                         | Calendar year                      |
| `gdp_per_capita_const2015_usd` | GDP per capita (constant 2015 USD) |
| `co2_per_capita_tons`          | CO₂ emissions per capita           |
| `population`                   | Total population                   |

Each row corresponds to a **country–year observation**.

---

## 7. Data Construction Pipeline

1. Load World Bank WDI CSV (wide format).
2. Detect year columns programmatically (1960–2025).
3. Reshape data from wide to long format.
4. Convert missing values (`..`) to `NaN`.
5. Filter to core indicators (GDP, CO₂, population).
6. Pivot to one row per country-year.
7. Save a clean CSV for analysis.

No aggregation or modeling is performed at this stage.

---

## 8. Output Files

The script generates:

```
data/cleaned/wdi_gdp_co2_population_panel.csv
```

This file serves as the **foundation for all subsequent analysis**, including regional aggregation, visualization, and econometric estimation.

---

## 9. Planned Regional Aggregation (Next Step)

Countries will later be grouped into:

* **Europe & North America**
* **Global South**
* **China** (standalone)
* **India** (standalone)

Aggregation rules:

* GDP per capita and CO₂ per capita: **population-weighted averages**
* Governance indices: **simple averages**

---

## 10. Planned Empirical Analysis

### Visual Analysis

* **GDP vs CO₂ scatter plots** (region-year observations)
* **Region-specific development paths**
* **Time trends** to identify decoupling

### Econometric Specifications

* Log–log income–emissions regressions
* EKC specifications with squared income terms
* Governance-augmented models
* Optional regional fixed effects

---

## 11. Limitations

* Correlation does not imply causation.
* Governance indicators are measured with error.
* Regional aggregation masks within-region heterogeneity.

---

## 12. Why This Project Matters

This project demonstrates:

* Clear economic intuition
* Correct panel-data design
* Awareness of heterogeneity and institutions
* Policy relevance for climate and development economics

---


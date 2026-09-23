# Final Empirical Results Synthesis

## Analysis status

The empirical analysis covers 1990–2024 and is frozen at the
`v1.0-analysis` milestone.

Script 14 does not estimate new econometric models. It consolidates
and validates results from the completed empirical pipeline.

## 1. Global GDP–CO2 relationship

The preferred economy and year fixed-effects specification gives a
GDP-per-capita elasticity of CO2 emissions per capita of
**0.653**, with a 95% confidence interval of
**[0.505, 0.801]**.

Within economies and conditional on common year effects, a 1%
increase in GDP per capita is therefore associated with approximately
a **0.65% increase in CO2 emissions per capita**.

This is an observational association and should not be interpreted as
a causal effect.

Across the final sample-based robustness checks, the global elasticity
ranges from **0.613** to **0.688**.

## 2. Development-cluster heterogeneity

The baseline development-cluster elasticities are approximately:

- Europe & North America: **0.086**
- Developed Asia & Oceania: **-0.056**
- China: **0.592**
- India: **0.699**
- Global South: **0.801**

The pooled global relationship therefore masks substantial
heterogeneity.

Europe & North America and Developed Asia & Oceania have small and
statistically imprecise slopes around zero, whereas the Global South
retains a much stronger positive income-emissions relationship.

China and India are intentionally retained as separate development
groups, but each contains only one economy. Their point estimates are
useful descriptions of their trajectories, while formal cluster-based
regional inference requires additional caution.

## 3. Decoupling

Annual decoupling is classified only among economy-year observations
with positive GDP-per-capita growth and consecutive annual data.

Long-period decoupling compares exact 1990 and 2024 endpoints and is
classified only among economies whose GDP per capita increased over
the period.

The paper-ready category shares are stored in:

`table_03_decoupling_summary.csv`

The three categories are:

- absolute decoupling;
- relative decoupling; and
- coupled expansion.

## 4. Governance

The baseline linear governance specifications suggest that higher
governance scores are associated with weaker GDP–CO2 coupling.

However, the relationship is not structurally robust.

After allowing development clusters to have different GDP slopes, the
negative governance interaction becomes small and statistically
imprecise. When nonlinear GDP dynamics are introduced, the governance
interaction changes sign.

The common balanced 2002–2024 comparison also shows that Government
Effectiveness provides the strongest negative baseline signal, while
Regulatory Quality and Rule of Law are weaker under this demanding
common-sample restriction.

The between/within decomposition further shows that the negative
baseline relationship is concentrated in persistent differences
between economies rather than changes in governance within the same
economy over time.

The governance extension therefore does **not** establish a stable
independent governance mechanism. Governance is better interpreted as
being associated with broader structural development differences.

## 5. Environmental Kuznets Curve evidence

Regional quadratic models are retained as exploratory benchmarks.

The Script 08 field `within_sample_ekc_candidate` identifies cases in
which the estimated quadratic satisfies a first-pass within-sample
turning-point screening rule.

These candidate flags should not be interpreted as causal or
structural proof of an Environmental Kuznets Curve.

The economy-level fixed-effects, regional-heterogeneity, decoupling,
governance-sensitivity and final-robustness results are given greater
weight in the final interpretation.

## 6. Final empirical conclusion

The central empirical result is not that economic growth has one
universal emissions consequence.

Instead:

1. the pooled within-economy GDP–CO2 relationship remains positive
   and robust;
2. the strength of that relationship differs substantially across
   development groups;
3. advanced-economy clusters display much greater evidence of
   decoupling;
4. China and India retain positive but lower elasticities than the
   Global South;
5. the Global South remains the most strongly growth-coupled
   multi-economy development cluster; and
6. governance correlates with these development patterns, but the
   analysis does not identify a stable independent governance
   moderation mechanism.

All regression results are interpreted as observational associations
rather than causal estimates.

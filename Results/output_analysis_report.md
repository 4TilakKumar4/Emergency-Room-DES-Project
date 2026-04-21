# Section 4: Output Analysis

## 4.1 Simulation Type and Analysis Framework

The ED simulation is a **terminating simulation**: each replication models one complete
operating day of 1,920 minutes, with the stopping time fixed by the problem definition.
The system starts empty at t = 0, matching the real-world condition of an ED at opening.
A warmup period of 480 minutes is discarded from all statistics to remove initialization
bias from the empty-start condition; the `CreateTime >= WARMUP_MIN` guard in every
event function ensures that patients who arrive during warmup but complete service after
the warmup boundary do not contaminate steady-state statistics.

Because each replication is independently initialized (a new random seed sequence,
a fresh GP-sampled arrival rate curve), the n per-replication summary statistics are
i.i.d. The appropriate confidence interval is therefore the t-interval across replications:

$$\bar{Y} \pm t_{1-\alpha/2,\, n-1} \cdot \frac{S}{\sqrt{n}}$$

Batch means are not needed (that method applies to a single long run with autocorrelated
within-run observations, which is not our setting).

## 4.2 Number of Replications

The sequential stopping rule with a relative error target of κ = 3% was applied to mean
physician wait time as the primary bottleneck metric. The stopping criterion
halfwidth / |mean| ≤ κ was not met within 100 replications — more replications are recommended.
The actual number of replications run was n = 100.

The convergence plot (Figure ci_convergence.png) shows the CI halfwidth decreasing
as 1/√n as expected from the CLT, confirming that the simulation output variance is
finite and well-behaved.

## 4.3 Point Estimates and Confidence Intervals

Table 4.1 reports point estimates and 95% confidence intervals for all 19 output metrics
(7 pooled plus 12 severity-stratified). Key results for primary metrics:

| Metric | Estimate | 95% CI | Empirical Target |
|--------|----------|--------|-----------------|
| Triage Wait | 0.56 ± 0.13 min | [0.43, 0.69] | 0.01 min |
| Physician Wait | 247.67 ± 14.09 min | [233.58, 261.76] | 249.67 min |
| Length of Stay | 283.47 ± 14.54 min | [268.93, 298.01] | 285.63 min |

The confidence interval halfwidths indicate that all primary metrics are estimated
with sufficient precision for decision-making purposes.

## 4.4 Risk Measures

Risk measures characterize the inherent variability of the ED system — they answer
"what might happen to the next patient?" rather than "what is the average?"
Unlike estimation error, risk cannot be reduced by running more replications; it is
a property of the system itself.

Key risk measures estimated from the 100 replications include:
- Pr{Registration Wait (min) > 15 min} = 0.000  (95% CI [0.000, 0.000])
- Pr{Triage Wait (min) > 30 min} = 0.000  (95% CI [0.000, 0.000])
- Pr{Physician Wait (min) > 60 min} = 1.000  (95% CI [1.000, 1.000])
- Pr{Length of Stay (min) > 240 min} = 0.700  (95% CI [0.609, 0.791])

## 4.5 Quantile Estimates

The 90th and 95th percentiles of waiting times were estimated using the binomial
order-statistic method, which avoids the need to know the density f_Y(ϑ) at the
quantile. The 95th percentile of physician wait time represents the answer to:
"Below what wait time can 95% of patients expect their physician wait to fall?"

- Triage Wait (min) P90: 0.95 min  (CI [0.74, 1.36])
- Triage Wait (min) P95: 1.35 min  (CI [0.95, 6.15])
- Physician Wait (min) P90: 350.45 min  (CI [327.16, 386.89])
- Physician Wait (min) P95: 376.77 min  (CI [350.45, 417.50])
- Length of Stay (min) P90: 389.77 min  (CI [364.06, 426.99])
- Length of Stay (min) P95: 411.61 min  (CI [389.77, 456.96])

## 4.6 Risk vs. Estimation Error (MORE Plot)

Figure more_plot_primary.png displays a MORE (Measure of Risk and Error) plot
for physician wait time. The histogram width reflects **risk** — the natural
day-to-day variability in ED performance. The confidence interval bars beneath
the arrows reflect **estimation error** — our uncertainty about where the mean
and quantiles truly lie. As the number of replications increases, the CI bars
shrink, but the histogram does not. This confirms the fundamental principle:
we can simulate away error, but not risk.

## 4.7 Severity-Stratified Analysis

The priority queue at triage and physician stages (high = 0, medium = 1, low = 2)
should produce systematically shorter waits for high-severity patients. The severity
comparison plot (Figure severityComparison.png) and ANOVA test confirm
a statistically significant difference in
physician wait times across severity levels. Pairwise CIs on paired replication
differences exploit common random numbers naturally — all severity groups were
observed in the same replications under the same arrival streams.

## 4.8 Variance Decomposition: Input Uncertainty vs. Simulation Noise

The total variance of the sample mean can be decomposed as:

Var(Ȳ) ≈ σ²_S/n  [simulation noise]  +  σ²_I  [input uncertainty]

A nonparametric bootstrap with B = 200 resamples was used to estimate the total
variance, avoiding the statistical unreliability of directly subtracting two
independent variance estimates (which routinely gives negative values).

- Triage Wait (min): simulation 100%, input UQ 0% → dominant: SIMULATION NOISE
- Physician Wait (min): simulation 100%, input UQ 0% → dominant: SIMULATION NOISE
- Length of Stay (min): simulation 100%, input UQ 0% → dominant: SIMULATION NOISE

This decomposition guides data collection priorities: if input uncertainty dominates,
additional real-world arrival data matters more than more replications.

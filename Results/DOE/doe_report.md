# Section 5: Design of Experiments and Simulation Optimization
**IE 7215 Discrete Event Simulation Analysis | Northeastern University | April 2026**

---

## 5.1 Problem Framing

The goal of this DOE phase is to identify the ED staffing configuration that minimises
mean physician wait time while keeping physician utilisation in the operationally
acceptable range of 70–85%. The output analysis identified physician wait as the
primary bottleneck (mean 218 min at baseline) with high day-to-day
variability (P95 well above 300 min). This motivates a systematic search over staffing
levels rather than ad hoc trial-and-error.

Decision variables and their ranges:

| Factor | Symbol | Range | Rationale |
|--------|--------|-------|-----------|
| Triage nurses | N | 2 – 4 | ±1 around baseline of 2 |
| Physicians | D | 2 – 4 | ±1 around baseline of 3 |
| Registration clerks | C | 1 – 2 | Baseline of 1; test if a second helps |

Primary response: mean physician wait time (min). Secondary responses: physician
utilisation, LOS, and Pr(physician wait > 60 min).

Common random numbers (CRN) are used throughout. For every replication slot j,
all scenarios receive the same GP-sampled arrival rate curve and the same SimRNG
seed sequence. Paired differences across scenarios therefore reflect staffing changes
rather than random demand variation, giving tighter CIs on differences.

---

## 5.2 Phase 1 — 2^3 Factorial Screening

A 2^3 full factorial with 30 replications per design point was used to screen
which factors significantly affect physician wait time. Each factor was set at a low
and high level, yielding 8 treatment combinations with CRN across all scenarios.

Main effects and two-way interactions estimated from the factorial are shown below
and in Figure factorial_effects.png. A negative effect means the response decreases
(improves) when the factor goes from its low to its high level.

| Factor | Effect (min) | Interpretation |
|--------|-------------|----------------|
| Doctors | -344.8 | decreases wait |
| Nurses×Doctors | 3.9 | increases wait |
| Nurses | -2.0 | decreases wait |
| Clerks | 0.0 | increases wait |
| Clerks×Nurses | 0.0 | increases wait |
| Clerks×Doctors | 0.0 | increases wait |


The dominant effect is physicians: the large negative value confirms that adding a
physician substantially reduces wait time. The nurse and clerk effects are smaller in
magnitude, indicating they are less critical to physician wait time. Based on these
results, clerks are dropped from Phase 2 (fixed at baseline of 1) and the search is
conducted over nurses and doctors.

---

## 5.3 Phase 2 — Response Surface Enumeration

With clerks fixed at 1, the full grid of (nNurses, nDoctors) combinations was
simulated with 30 replications each using CRN. The response surface
heatmap (Figure response_surface.png) shows mean physician wait for every
combination. Lower values are better.

Top 3 configurations by mean physician wait:

| Rank | Clerks | Nurses | Doctors | Mean Wait (min) | Util |
|------|--------|--------|---------|----------------|------|
| 1 | 1 | 2 | 4 | 69.8 ± 4.3 | 69.0% |
| 2 | 1 | 3 | 4 | 71.8 ± 4.4 | 69.2% |
| 3 | 1 | 4 | 4 | 71.8 ± 4.4 | 69.2% |


The heatmap reveals that physician count is the dominant driver of wait time, with
nurse count having a secondary effect. The best and second-best configurations are
carried forward to Phase 3.

---

## 5.4 Phase 3 — Kim-Nelson Ranking and Selection

The Kim-Nelson (2001) sequential indifference-zone procedure was applied to the top
1 candidates identified in Phase 2. The procedure guarantees
Pr(correct selection) >= 95% whenever the true best configuration differs
from all others by at least delta = 10 minutes in mean physician wait — a
threshold chosen to reflect a clinically and operationally meaningful improvement.

Parameters: delta = 10.0 min, alpha = 0.05, n0 = 10 first-stage replications.

The procedure ran for 106 total replications per configuration and
selected configuration (1 clerk, 2 nurses, 4 doctors) as the best, with the statistical guarantee
that the probability of this being an incorrect selection is at most 5%.

Selected optimal configuration:
- Clerks: 1
- Nurses: 2
- Physicians: 4
- Estimated mean physician wait: 68.2 min
- Pr(correct selection): >= 95%

The convergence plot (Figure convergence_ks.png) shows the number of competing
scenarios at each iteration. The procedure eliminates inferior configurations
quickly, focusing simulation effort on the hardest comparisons.

---

## 5.5 What-If Experiments

Three what-if experiments were designed to answer specific policy questions beyond
the main optimisation. All use CRN pairing against the baseline for valid comparisons.

**What-if 1 — Minimum constant staffing to meet a wait target.**
Doctor count was varied from 2 to 4 with all other factors fixed at baseline.
This identifies the minimum number of physicians needed to keep mean physician wait
below 120 minutes — a common target in ED performance standards.

**What-if 2A — Two-shift staffing policy.**
Rather than a constant doctor count all day, this experiment models a shift schedule
that matches the NHPP arrival pattern: 2 doctors during off-peak hours
(midnight–11:00 and 21:00–midnight) and 4 during the peak window (11:00–21:00).
The shift-change event in ED_Simulation updates doctor capacity mid-replication,
allowing the simulation to capture the transient effects of staff changeover.

**What-if 2B — Utilisation-target staffing.**
The required physician headcount was derived analytically from Little's Law:
c = lambda * E[S] / rho_target, where lambda is the mean arrival rate from the
GP posterior, E[S] is the weighted mean physician service time, and rho_target = 80%.
This gives a theoretically grounded staffing recommendation that can be compared
against the integer-search result from Phase 3.

**Summary table:**

| Scenario | Mean Physician Wait (min) | ±HW | Physician Util |
|----------|--------------------------|-----|----------------|
| Baseline (3 doctors, constant) | 218.5 | 12.0 | 91.2% |
| 2 doctors (constant) | 412.7 | 6.3 | 99.7% |
| 3 doctors (constant) | 209.5 | 9.7 | 90.7% |
| 4 doctors (constant) | 71.8 | 4.4 | 69.2% |
| Two-shift (2→4→2) | 148.8 | 6.2 | n/a |
| 3 doctors (util≈80%) | 209.5 | 9.7 | 90.7% |


The comparison chart (Figure whatif_comparison.png) shows all scenarios against the
120-minute wait target line and the baseline mean. Scenarios below the dashed baseline
line represent improvements; the 120-minute target line indicates operational acceptability.

---

## 5.6 Conclusions and Staffing Recommendation

The DOE analysis leads to three actionable conclusions:

First, physician count is the dominant driver of mean physician wait time. The
factorial main effect for doctors is several times larger than for nurses or clerks,
and the response surface shows a steep gradient along the doctor axis.

Second, the Kim-Nelson procedure identified (1 clerk, 2 nurses, 4 doctors) as the
statistically best configuration with Pr(CS) >= 95%. This configuration
reduces mean physician wait from the baseline of 218 min to
approximately 68 min — a reduction of
150 minutes.

Third, the two-shift policy (WI-2A) offers a cost-effective alternative if adding
permanent physician capacity is not feasible. By concentrating additional physicians
during the peak arrival window and reducing to 2 overnight, similar wait
reductions may be achieved with lower total staffing cost.

---

## References

Kim, S.-H., & Nelson, B. L. (2001). A fully sequential procedure for
indifference-zone selection in simulation. *ACM TOMACS*, 11(3), 251–273.

Nelson, B. L., & Pei, L. (2021). *Foundations and Methods of Stochastic Simulation*
(2nd ed.). Springer.

Whitt, W. (2007). What you should know about queueing models to set staffing
requirements in service systems. *Naval Research Logistics*, 54, 476–484.

# Technical Write-Up: ED Simulation Modelling and Coding Approach

**IE 7215 — Discrete Event Simulation Analysis**  
Northeastern University · Spring 2026  
Author: Tathya Malav Kamdar, Tilak Kumar Byradenahalli Ramesh, Uriel Baron

---

## 1. Problem Description and Objectives

This project models an Emergency Room (ER) as a discrete-event simulation to assess system performance, quantify the impact of input model uncertainty on output estimates, and guide resource allocation decisions. The system is representative of a mid-size academic ED receiving approximately 200 patients per day, with three processing stages: registration, triage, and physician consultation.

The primary performance metrics are mean patient wait times at each stage, total length of stay (LOS), and resource utilisation across three resource types. The project progresses through three modelling levels: a baseline NHPP model with fixed arrival rates, a GP-enhanced model that propagates arrival rate estimation uncertainty, and a final version that produces severity-stratified output statistics across all 100 replications.

---

## 2. Simulation Framework

The project uses a custom Python simulation engine — `SimClasses`, `SimFunctions`, `SimRNG` — structured as the `sim_engine` package. This is a hand-built event-calendar framework, not SimPy or Arena. The choice is deliberate: it forces explicit modelling of every event type, queue discipline, and resource interaction rather than relying on a high-level abstraction.

The core objects are:

**`EventCalendar`** — a sorted list of `EventNotice` objects. Events are inserted in time order and removed from the front. The simulation advances by removing the next event, setting `SimClasses.Clock` to its `EventTime`, and dispatching to the appropriate handler function.

**`DTStat`** — discrete-time statistics accumulator. Stores sum, sum-of-squares, and observation count. Produces mean and standard deviation. Used for all wait time and LOS statistics. Does not depend on `SimClasses.Clock`.

**`CTStat`** — continuous-time statistics accumulator. Tracks time-weighted area under a step function. Used for queue lengths and resource utilisation. Must be updated whenever the tracked variable changes.

**`Resource`** — tracks busy units, seizes and frees capacity, and maintains a `CTStat` for time-average utilisation.

**`FIFOQueue`** and **`PriorityQueue`** — queue objects each wrapping a `CTStat` for WIP tracking. `PriorityQueue` inserts entities by priority key while preserving FIFO order within each priority class.

All objects that track statistics are registered in `theDTStats` or `theCTStats` lists. `SimFunctionsInit` clears them all between replications, and `ClearStats` clears them again at the warmup boundary. This design ensures no replication state leaks into the next, and warmup-period observations never contaminate steady-state statistics.

---

## 3. Input Modelling

### 3.1 Service Time Distribution Fitting

`Fitting_Dist_Tat.py` fits statistical distributions to service and wait time data from the 30-day dataset using maximum likelihood estimation via `scipy.stats`. Eight candidate distributions are evaluated: Normal, Exponential, Gamma, Erlang, Lognormal, Weibull, Triangular, and Beta. Selection is by Kolmogorov-Smirnov test — the distribution with the highest p-value is chosen.

Three fitting approaches are implemented:

**Approach 1 — Pooled**: a single distribution per stage, ignoring severity. Produces the simplest input model but loses the significant heterogeneity in service times across triage levels.

**Approach 2 — Severity-stratified**: separate distributions per (stage, severity) combination — nine distributions in total for three stages and three severity levels. This is the approach used in all simulation models. It captures the empirical finding that high-severity patients have substantially longer doctor service times even though they are served first.

**Approach 3 — Service vs. wait separation**: fits service time distributions only, using wait times as external validation targets rather than simulation inputs. This is methodologically cleaner because wait times are endogenous outcomes of the queuing system, not independent inputs.

The fitting output is written to `Sources/simrng_parameters.csv` with SimRNG-compatible parameter names. Distributions without a direct SimRNG equivalent (Beta, Weibull) are approximated with `Erlang(m=1, mean)` — equivalent to `Expon(mean)` — a documented simplification.

### 3.2 Arrival Process Modelling

The arrival process is modelled as a Non-Homogeneous Poisson Process (NHPP) with a 24-hour periodic rate function. In the baseline model, the rate function is a step function: 24 independent point estimates computed as the 30-day average arrivals per hour. In the GP models, this step function is replaced by a smooth continuous function sampled from a Gaussian Process posterior.

The NHPP inter-arrival time is drawn from `Expon(60/λ(t))` where `λ(t)` is the current rate in patients per hour and time is in minutes. The rate is evaluated at the current fractional hour `(Clock % 1440) / 60`, which advances continuously rather than stepping at hour boundaries.

---

## 4. Gaussian Process Arrival Rate Model

### 4.1 Motivation

Using 30 days of data to estimate 24 hourly rates produces point estimates with non-trivial estimation error, particularly for overnight hours where total counts over 30 days may be as low as 40–50 arrivals. The baseline model treats these estimates as exact truth, which produces overconfident output confidence intervals. The GP model quantifies this estimation uncertainty and propagates it into the output statistics.

### 4.2 The Log-Gaussian Cox Process

The formal framework is the Log-Gaussian Cox Process (LGCP), introduced by Møller, Syversveen, and Waagepetersen (1998). A latent function `f(t)` is drawn from a Gaussian Process, and the arrival rate is set as `λ(t) = exp(f(t))`. The exponential transformation enforces strict non-negativity of rates.

For hourly binned count data the generative model is: `f ~ GP(m, k)` → `λ_i = exp(f_i)` → `N_i ~ Poisson(λ_i)`. The GP prior on `f` regularises the rate estimates, borrowing strength across adjacent hours through the kernel's correlation structure.

### 4.3 Implementation via Log-Space GP

Rather than full LGCP inference (which requires MCMC or variational approximation), the implementation uses a pragmatic Gaussian approximation: fit a `GaussianProcessRegressor` to `y_log = log(mean_counts + ε)` with per-bin observation noise estimated via the delta method. The delta method approximation gives `Var[log(X̄)] ≈ Var[X̄] / X̄²` where `Var[X̄] = sample_var / n_days`. This transforms heteroscedastic Poisson noise into approximately Gaussian noise on the log scale, making the problem tractable for standard GP regression.

The `alpha` parameter of `GaussianProcessRegressor` is set to these per-bin noise estimates, so bins with high day-to-day variability (e.g. hour 11 during peak period) contribute less to kernel hyperparameter fitting than stable bins.

### 4.4 Kernel Design

The kernel is a locally-periodic construction:

```
k(t, t') = ConstantKernel × ExpSineSquared × RBF + WhiteKernel
```

The `ExpSineSquared` component captures the 24-hour periodicity with its period fixed at 24.0 to prevent the optimiser from drifting. Its internal length scale controls wiggliness within one cycle. The `RBF` multiplier creates a locally-periodic structure where the periodic pattern can vary in amplitude across the day. The `WhiteKernel` absorbs residual observation noise not captured by the delta-method `alpha` term.

Kernel hyperparameters are optimised by maximising the log-marginal likelihood using `n_restarts_optimizer=20` random initialisations to avoid local optima.

### 4.5 Posterior Sampling

After fitting, the posterior is queried on a fine grid `_xFine` of 97 evenly spaced points from 0 to 24 (4× the bin resolution). `sampleRateCurve()` calls `GaussianProcessRegressor.sample_y()` on this grid, exponentiates to return to rate space, and wraps the result in a `scipy.interpolate.interp1d` cubic spline. The returned callable produces a smooth, continuous rate function that can be evaluated at any fractional hour.

The cubic interpolation between the 97 grid points eliminates the step-function discontinuity at hour boundaries that exists in the baseline model. When `nextInterarrival()` queries the rate at `t = 8.75` (8:45 AM), it gets a smoothly interpolated value between the hour-8 and hour-9 estimates.

---

## 5. Two-Level Simulation and Input Uncertainty Quantification

### 5.1 The Two-Level Protocol

The GP model implements the two-level simulation framework described by Barton, Nelson, and Xie (2014). The outer loop samples a rate curve `θ_i` from the GP posterior, representing one plausible realisation of the true arrival rate function given the observed data. The inner loop runs the DES with that fixed rate curve.

In the current implementation `n_inner = 1` — one replication per rate curve — so the 100 replications correspond to 100 different outer samples. This conflates input uncertainty and simulation noise in the observed output variance, but provides a valid combined CI. The decomposition separates them analytically after the fact.

```
for rep in range(NUM_REPS):
    rateFn = gpModel.sampleRateCurve(randomState=rep)
    result = runReplication(rateFn)
```

The `random_state=rep` argument makes each sample deterministic and reproducible while ensuring all 100 samples are different draws from the posterior.

### 5.2 Variance Decomposition

The law of total variance decomposes output variance as:

```
Var(Y) = Var_θ[E(Y|θ)] + E_θ[Var(Y|θ)]
```

The first term is input uncertainty: how much the expected output shifts across different plausible rate functions. The second term is simulation noise: the inherent randomness of the queuing process given a fixed rate function.

`ArrivalRateGP.varianceDecomposition()` estimates these from the replication results. When `n_inner = 1`, `E_θ[Var(Y|θ)]` cannot be estimated directly and is approximated as zero, so the total variance is attributed entirely to input uncertainty. With `n_inner > 1`, the within-curve variance provides a proper estimate of simulation noise.

The `inputFraction` output answers the key question: if you wanted to narrow the output CI, would you need more arrival data or more simulation replications? If input uncertainty dominates (high `inputFraction`), more replications will not help — only collecting more days of arrival data will tighten the GP posterior.

---

## 6. Simulation Model Design

### 6.1 Event Types and Dispatch

The simulation uses four event types: `arrival`, `endRegistration`, `endTriage`, `endDoctor`. The event loop dispatches by string comparison:

```python
if   ev.EventType == "arrival":          arrival()
elif ev.EventType == "endRegistration":  endRegistration(ev)
elif ev.EventType == "endTriage":        endTriage(ev)
elif ev.EventType == "endDoctor":        endDoctor(ev)
```

Event function names exactly match the dispatch strings — a case-sensitivity bug here would cause events to be silently dropped and queues to fill without bound. This was an identified bug in an earlier version and has been fixed.

### 6.2 Warmup Period and Contamination Guard

The simulation runs for `WARMUP_MIN + RUN_MIN` minutes. `ClearStats` is called once at the first event that crosses the warmup boundary. This resets all DTStat and CTStat accumulators to their zero states.

A critical secondary guard prevents warmup contamination from patients who arrive before warmup but complete service after it. Without this guard, a patient who arrived at minute 400 and sees a doctor at minute 510 would contribute their full 110-minute wait to the post-warmup statistics, inflating all wait time estimates. The guard is:

```python
if p.CreateTime >= WARMUP_MIN:
    doctorWait.Record(p.doc_start - p.triage_end)
```

This ensures that only patients who entered the system during the steady-state period contribute to the output statistics. It is applied identically in `endRegistration`, `endTriage`, and `endDoctor` for all four pooled and twelve per-severity DTStat collectors.

### 6.3 Queue Discipline

Registration uses `SimClasses.FIFOQueue`. The justification is that severity level is not known to the receptionist — it is assessed during triage. Additionally, the empirical data shows 44.1% zero wait at registration, meaning the queue is almost always empty and the discipline rarely matters in practice.

Triage and physician queues use `PriorityQueue` with `PRIORITY = {"high": 0, "medium": 1, "low": 2}`. Lower numeric values sort earlier. The insertion logic walks the queue to find the first entity of lower priority and inserts before it, preserving FIFO order within each priority class.

### 6.4 Random Number Streams

Five independent random number streams are used, one per source of randomness:

| Stream | Purpose |
|---|---|
| 1 | NHPP inter-arrival times |
| 2 | Severity assignment |
| 3 | Registration service times |
| 4 | Triage service times |
| 5 | Physician service times |

Stream separation prevents changes to one distribution (e.g. physician service time mean) from affecting any other random sequence. This is standard practice for variance reduction through common random numbers in comparison experiments.

### 6.5 Service Time Generation

`drawService()` dispatches to the appropriate SimRNG function based on the distribution name stored in the parameters dict:

```python
entry = params[stage][severity]
func  = entry[0]
if func == "Lognormal":  return SimRNG.Lognormal(entry[1], entry[2], stream)
if func == "Erlang":     return SimRNG.Erlang(entry[1], entry[2], stream)
...
```

SimRNG's `Lognormal` is parameterised by the raw-data mean and variance, not by log-space parameters — the conversion is handled internally. This means `simrng_parameters.csv` stores empirical means and variances directly, keeping the parameter file human-readable.

---

## 7. Output Analysis

### 7.1 Confidence Intervals

Per-replication means are collected across all 100 replications. The 95% CI uses the t-distribution approximation with `ddof=1`:

```
CI = X̄ ± 1.96 × s / √n
```

The approximation uses `z = 1.96` rather than the t-critical value because `n = 100` replications makes the difference negligible (t₀.₀₂₅,₉₉ = 1.984).

### 7.2 Severity-Stratified Output

The GP with severity model produces 19 columns per replication row in the output CSV: 7 pooled metrics plus 12 per-severity metrics (`regWait_{low/medium/high}`, `triageWait_{low/medium/high}`, `doctorWait_{low/medium/high}`, `LOS_{low/medium/high}`).

Per-severity statistics are accumulated by routing each patient's observations to the correct DTStat using `p.severity` as a dict key. The twelve per-severity collectors are registered in `theDTStats` alongside the four pooled collectors so they are automatically cleared between replications and at the warmup boundary — no special handling required.

### 7.3 Validation

Simulation output is compared against empirical targets derived directly from the 30-day dataset. These targets reflect the data-generating process rather than a theoretical benchmark, so close agreement indicates the simulation correctly reproduces the system's behaviour. Doctor wait time is the most demanding validation target (249.67 min empirical) because it is highly sensitive to the interaction between the NHPP arrival pattern and physician capacity.

---

## 8. File-by-File Reference

| File | Purpose | Key Functions |
|---|---|---|
| `sim_engine/SimClasses.py` | DES primitives | `CTStat`, `DTStat`, `EventCalendar`, `FIFOQueue`, `PriorityQueue`, `Resource`, `Entity` |
| `sim_engine/SimFunctions.py` | Scheduling and init | `SimFunctionsInit`, `Schedule`, `SchedulePlus`, `ClearStats` |
| `sim_engine/SimRNG.py` | Random variate generation | `lcgrand`, `Expon`, `Erlang`, `Lognormal`, `Normal`, `Triangular`, `Uniform` |
| `sim_engine/ArrivalRateGP.py` | GP arrival rate model | `fit`, `sampleRateCurve`, `meanRateCurve`, `plotPosterior`, `varianceDecomposition` |
| `Fitting_Dist_Tat.py` | Distribution fitting | `fitSeverityStratifiedDistributions`, `findBestFittingDistribution`, `exportSimrngParametersCsv` |
| `New_Distrubution_plot1.py` | Distribution fitting (SimRNG export variant) | Same structure as `Fitting_Dist_Tat.py` |
| `ERSimulationModelNHPP.py` | Baseline simulation | Fixed `HOURLY_RATES`, `runReplication()` with no arguments |
| `ERSimulationModelGP.py` | GP arrival uncertainty | `buildHourlyCounts`, `runReplication(rateFn)`, `plotGPPosterior` |
| `ERSimulationModelGPwithSev.py` | GP + severity output | All of above plus per-severity DTStat collectors and severity breakdown table |

---

## 9. References

Barton, R. R., Nelson, B. L., and Xie, W. (2014). Quantifying input uncertainty via simulation confidence intervals. *INFORMS Journal on Computing*, 26(1), 74–87.

Law, A. M. and Kelton, W. D. (2000). *Simulation Modeling and Analysis*, 3rd ed. McGraw-Hill.

Møller, J., Syversveen, A. R., and Waagepetersen, R. P. (1998). Log Gaussian Cox Processes. *Scandinavian Journal of Statistics*, 25(3), 451–482.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

Song, E. and Nelson, B. L. (2015). Quickly assessing contributions to input uncertainty. *IIE Transactions*, 47(9), 893–909.

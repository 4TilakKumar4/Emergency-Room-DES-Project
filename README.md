# Emergency Room Discrete-Event Simulation

**IE 7215 — Discrete Event Simulation Analysis**  
Northeastern University · Spring 2026  
Authors: Tathya Malav Kamdar, Tilak Kumar Byradenahalli Ramesh, Uriel Baron

---

## Overview

This project implements a complete discrete-event simulation (DES) pipeline for an Emergency Department (ED) using a custom Python simulation engine. The model covers patient arrivals, triage, physician consultation, resource allocation, and length of stay, with a full input modelling, validation, sensitivity analysis, design of experiments, and stress-testing framework.

The simulation is built as five progressive configurations. The first two represent the core model at increasing levels of fidelity. The remaining three are experimental or stress-test variants used to explore alternative configurations and edge-case system behaviour.

| Configuration | File | Description |
|---|---|---|
| Baseline NHPP | `ERSimulationModelNHPP.py` | Fixed hourly arrival rates from data |
| GP + Severity *(production)* | `ERSimulationModelGPwithSev.py` | GP rates + per-severity statistics — primary model |
| GP + Severity Experimental | `ERSimulationModelGPwithSev_Exp.py` | Experimental configuration: exponential service times instead of KS-fitted distributions — used to test whether distribution family choice affects conclusions |
| MCE Stress Test | `ERSimulationModelMCE.py` | Mass Casualty Event scenario — superimposed exponential-decay surge arrival stream with 70% high-severity patients and 6-doctor surge staffing |
| MCE Experimental | `ERSimulationModelMCE_Exp.py` | MCE scenario under the experimental exponential service time configuration |

> **Note on "Experimental" variants:** The `_Exp` suffix denotes an *experimental* configuration that replaces KS-selected service time distributions (Lognormal, Erlang) with Exponential distributions at the same empirical means. This tests whether the distribution family choice materially affects system performance conclusions. It is not the production model — `ERSimulationModelGPwithSev.py` is.

---

## Repository Structure

```
Emergency-Room-DES-Project/
│
├── sim_engine/                          # Custom DES framework (not SimPy/Arena)
│   ├── __init__.py
│   ├── analysis_utils.py                # Shared CI utility — single source of truth
│   ├── ArrivalRateGP.py                 # GP arrival rate model (fit, sample, decompose)
│   ├── SimClasses.py                    # Core primitives: EventCalendar, DTStat, CTStat,
│   │                                    #   Resource, FIFOQueue
│   ├── SimFunctions.py                  # Schedule, SchedulePlus, ClearStats, SimFunctionsInit
│   └── SimRNG.py                        # Random variate generation (100 independent streams)
│
├── Sources/                             # Input data — not tracked by git
│   ├── er_5000_patients.csv             # 5,000 patient records over 30 days
│   └── simrng_parameters.csv           # Fitted distribution parameters (output of Fitting_Dist_Tat.py)
│
├── Results/                             # All outputs — auto-created, not tracked by git
│   ├── DOE/                             # DOE_Optimization.py outputs
│   ├── input_modeling/                  # Fitting_Dist_Tat.py plots and diagnostics
│   ├── output_analysis/                 # OutputAnalysis.py plots
│   ├── simulation/                      # Per-configuration simulation outputs
│   │   ├── nhpp/                        # ERSimulationModelNHPP.py
│   │   ├── gp_severity/                 # ERSimulationModelGPwithSev.py  ← baseline
│   │   ├── gp_severity_experimental/    # ERSimulationModelGPwithSev_Exp.py
│   │   ├── mce/                         # ERSimulationModelMCE.py
│   │   └── mce_experimental/            # ERSimulationModelMCE_Exp.py
│   ├── validation/                      # JacksonValidation.py and DegeneracyTests.py outputs
│   ├── ED_rep_results.csv               # Latest baseline replication results (100 reps)
│   ├── sensitivity_summary.csv          # Summary table from InputSensitivityAnalysis.py
│   └── simrng_parameters.csv           # Copy of fitted parameters used in last run
│
├── ERSimulationModelNHPP.py
├── ERSimulationModelGPwithSev.py
├── ERSimulationModelGPwithSev_Exp.py
├── ERSimulationModelMCE.py
├── ERSimulationModelMCE_Exp.py
│
├── Fitting_Dist_Tat.py                  # Distribution fitting pipeline
├── InputSensitivityAnalysis.py          # Four sensitivity analyses (A–D)
├── DOE_Optimization.py                  # Factorial screening, Kim-Nelson R&S, what-if experiments
├── OutputAnalysis.py                    # CI convergence, risk measures, LOS histograms
├── JacksonValidation.py                 # Analytical validation against Erlang-C predictions
├── DegeneracyTests.py                   # Structural validation: monotonicity, empty system,
│                                        #   overload, and priority collapse tests
│
├── README.md
├── TECHNICAL_WRITEUP.md
├── CONTRIBUTING.md
└── requirements.txt
```

---

## System Description

| Component | Configuration |
|---|---|
| Receptionist | 1 — FIFO queue (severity unknown to clerk at registration) |
| Triage nurses | 2 — priority queue (high > medium > low) |
| Physicians | 3 baseline, 6 during MCE surge — priority queue |
| Arrival process | Non-Homogeneous Poisson Process, 24-hour periodic cycle |
| Service times | Severity-stratified Lognormal/Erlang (production) or Exponential (experimental) |
| Warmup | 480 minutes (8 hours) |
| Steady-state run | 1440 minutes (24 hours) |
| Replications | 100 |

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### Recommended run order

1. **Fit distributions** — only needs to run once; produces `Sources/simrng_parameters.csv`
2. **Run the production simulation** — reads the parameters file
3. **Run DOE, sensitivity, or MCE** — all import the production simulation module

### Step 1 — Fit service time distributions

```bash
python Fitting_Dist_Tat.py
```

Reads `Sources/er_5000_patients.csv`, fits eight candidate distributions per stage per severity, selects by KS test, and writes `Sources/simrng_parameters.csv`. Plots saved to `Results/`.

### Step 2 — Run the production simulation

```bash
python ERSimulationModelGPwithSev.py
```

Runs 100 replications with GP-sampled arrival rates and severity-stratified service times. Outputs to `Results/simulation/gp_severity/`.

### Step 3 (optional) — Run additional analyses

**Sensitivity analysis** — four analyses covering data volume, distribution choice, hour-group GP uncertainty, and variance decomposition:

```bash
python InputSensitivityAnalysis.py
```

Outputs to `Results/sensitivity_analysis/`.

**Design of experiments and optimisation** — 2³ factorial screening, full enumeration, Kim-Nelson ranking-and-selection, and three what-if staffing experiments:

```bash
python DOE_Optimization.py
```

Outputs to `Results/DOE/`, including `doe_report.md` — a narrative report section ready to paste into the final write-up.

**MCE stress test** — Mass Casualty Event scenario with surge staffing and split statistics:

```bash
python ERSimulationModelMCE.py
```

Outputs to `Results/simulation/mce/`.

**Experimental configurations** — to test the alternative exponential service time assumption:

```bash
python ERSimulationModelGPwithSev_Exp.py   # experimental service times, no MCE
python ERSimulationModelMCE_Exp.py         # experimental service times + MCE
```

Outputs to `Results/simulation/gp_severity_experimental/` and `Results/simulation/mce_experimental/` respectively.

---

## Input Data

### `er_5000_patients.csv`

5,000 ED patient records over 30 days. Key columns:

| Column | Description |
|---|---|
| `arrival_time` | Patient arrival timestamp |
| `severity` | `low`, `medium`, or `high` |
| `registration_service_min` | Time spent with receptionist |
| `triage_service_min` | Time spent with triage nurse |
| `doctor_service_min` | Time spent with physician |
| `*_wait_min` | Time waiting in each queue (validation targets only — not simulation inputs) |
| `er_los_min` | Total ED length of stay |

Severity breakdown: low 55.6%, medium 34.9%, high 9.6%

### `simrng_parameters.csv`

Output of `Fitting_Dist_Tat.py`. Contains SimRNG-compatible distribution parameters for all three stages across all severity levels. The simulation loads this file at runtime — no service time parameters are hardcoded in any model file.

---

## Outputs

All outputs are written automatically to `Results/` subdirectories, which are created on first run.

### Simulation outputs (`Results/simulation/<config>/`)

| File | Description |
|---|---|
| `ED_output_analysis.png` | 2×2 histograms of key metrics with 95% CI and empirical validation targets |
| `ED_utilisation.png` | Bar chart of resource utilisation per stage with CI error bars |
| `ED_gp_posterior.png` | GP posterior diagnostic: credible band, posterior mean, and 20 sample curves |
| `ED_rep_results.csv` | Per-replication raw results (19 columns: 4 pooled metrics + 3 × 4 per-severity + 3 utilisation) |

MCE configurations additionally produce:

| File | Description |
|---|---|
| `MCE_arrival_rate.png` | Normal GP stream vs. MCE decay stream overlaid |
| `MCE_severity_mix.png` | Baseline vs. MCE severity composition pie charts |
| `MCE_severity_waits.png` | Doctor wait and LOS by severity, MCE vs. normal patients |
| `MCE_rep_results.csv` / `MCE_experimental_rep_results.csv` | Per-replication split statistics |

### DOE outputs (`Results/DOE/`)

| File | Description |
|---|---|
| `factorial_effects.png` | Tornado chart of main effects and two-way interactions |
| `response_surface.png` | Heatmap of mean physician wait over the factor grid |
| `convergence_ks.png` | Kim-Nelson active-set size per sequential iteration |
| `whatif_comparison.png` | Bar chart comparing all staffing policy what-if experiments |
| `doe_results.csv` | Per-replication results for every scenario |
| `doe_report.md` | Narrative report section ready for the final project report |

### Sensitivity analysis outputs (`Results/sensitivity_analysis/`)

| File | Description |
|---|---|
| `sens_A_data_volume.png` | CI half-width vs. days of arrival data |
| `sens_B_distribution.png` | Doctor wait under KS-fitted vs. simpler distribution families |
| `sens_C_hour_groups.png` | Output uncertainty decomposed by peak vs. overnight arrival uncertainty |
| `sens_D_bootstrap.png` | Nested simulation variance decomposition: input uncertainty vs. simulation noise |
| `sensitivity_summary.csv` | Numerical results from all four analyses |

### Validation targets

| Metric | Empirical | Notes |
|---|---|---|
| Registration wait | 4.67 min | 44.1% zero wait |
| Triage wait | 0.01 min | 99.4% zero wait |
| Doctor wait | 249.67 min | 5.5% zero wait — primary bottleneck |
| Length of stay | 285.63 min | |

---

## Modelling Approach Summary

### Input modelling

Distribution fitting is performed in three approaches:

1. **Pooled** — single distribution per stage, ignores severity
2. **Severity-stratified** — separate distributions per (stage, severity) combination — used in the production simulation
3. **Service vs. wait separation** — service times as simulation inputs; wait times reserved as external validation targets

The KS test selects the best-fitting distribution from eight candidates: Normal, Exponential, Gamma, Erlang, Lognormal, Weibull, Triangular, Beta.

### Arrival process

The NHPP is modelled using hour-by-hour arrival rates estimated from the 30-day dataset. In the GP models, these rates are replaced by samples from a Gaussian Process posterior, motivated by the Log-Gaussian Cox Process framework and implemented as a log-space Gaussian approximation using scikit-learn's `GaussianProcessRegressor`. A locally-periodic kernel (`ExpSineSquared × RBF + WhiteKernel`) captures the 24-hour periodic pattern while allowing amplitude variation across the day.

Each replication draws a different plausible rate curve from the GP posterior, propagating arrival rate estimation uncertainty into output confidence intervals.

Inter-arrival times are drawn using a piecewise-constant inversion method: at each event the current rate `λ(t)` is evaluated and an exponential inter-arrival `Expon(60/λ(t))` is drawn. This is a standard approximation for slowly-varying NHPP rate functions.

### Input uncertainty quantification

The GP model implements a two-level simulation protocol (Barton, Nelson, and Xie, 2014):

- **Outer loop**: sample a rate curve from the GP posterior (input uncertainty)
- **Inner loop**: run one DES replication with that fixed rate curve (simulation noise)

Output variance is decomposed using the law of total variance. Results show simulation noise accounts for 93–96% of total variance, confirming the input distributions are well-specified and that more replications — not more data collection — is the best way to narrow output confidence intervals.

### Queue discipline

Registration uses FIFO because severity is not clinically assessed until triage. Triage and physician queues use priority discipline (high > medium > low), implemented with a three-deque O(1) structure. Patients of equal severity retain FIFO order within their priority class.

### Sensitivity analysis

Four analyses are implemented in `InputSensitivityAnalysis.py`:

- **Analysis A** — CI width vs. days of arrival data: how much does output precision improve with more data?
- **Analysis B** — Distribution family sensitivity: do simpler distributions (Exponential, Normal) change conclusions?
- **Analysis C** — Hour-group GP uncertainty: does peak-hour or overnight arrival uncertainty drive output variance more?
- **Analysis D** — Nested simulation variance decomposition: separates input uncertainty from simulation noise using the nested parametric bootstrap estimator of Song and Nelson (2015), avoiding the negative-variance problem of direct subtraction.

### Design of experiments and optimisation

`DOE_Optimization.py` runs a three-phase pipeline:

1. **2³ factorial screening** — identifies which staffing factors (clerks, nurses, physicians) drive physician wait time. Common Random Numbers are applied so differences across configurations are attributable to staffing changes, not demand variation.
2. **Full enumeration** — maps the response surface over the full physician count range
3. **Kim-Nelson ranking-and-selection** — selects the statistically best configuration with Pr(CS) ≥ 0.95 and a 10-minute indifference zone

Three what-if experiments compare staffing policies: constant headcount threshold, two-shift policy (off-peak/peak staffing), and a utilisation-target policy derived from Little's Law.

### MCE stress test

`ERSimulationModelMCE.py` superimposes a second independent NHPP arrival stream over the normal GP stream using the Poisson superposition theorem. The MCE stream follows an exponentially decaying rate starting at 18:00 with a peak of 20 patients/hour, decaying to ~1 patient/hour after 3 hours (decay constant δ = ln(20)/180 ≈ 0.0166/min). The severity mix shifts to 70% high-acuity during the event. Surge staffing increases physicians from 3 to 6. MCE and normal patients are tagged separately, producing split statistics for both populations.

---

## Project Structure Notes

The `sim_engine/` package contains all simulation framework code and the shared statistical utility. Import pattern in model files:

```python
from sim_engine import SimClasses, SimFunctions, SimRNG
from sim_engine.ArrivalRateGP import ArrivalRateGP
from sim_engine.analysis_utils import ci as ci95
```

`sim_engine/analysis_utils.py` provides a single `ci(series, alpha)` function using `scipy.stats.t.ppf` — the correct t-distribution CI for finite replication counts. All model and analysis files import from here rather than defining their own CI functions. Keeping it inside `sim_engine/` means it is versioned alongside the framework it supports and is available to any script that already imports the package.

`SimFunctions.py` uses a relative import (`from . import SimClasses`) because both files are in the same package.

---

## Known Assumptions and Limitations

- Registration queue is modelled as FIFO. In a real ED, ESI Level 1/2 patients would bypass registration. The data does not evidence this behaviour, so modelling it would create an inconsistency with the fitted input distributions.
- The dataset has a relatively flat diurnal arrival curve compared to published ED data. NHPP parameters are estimated directly from this dataset rather than from a reference population.
- Distributions with no direct SimRNG equivalent (Beta, Weibull) are approximated with `Erlang(m=1, mean)`, equivalent to `Expon(mean)`. This simplification is documented in `simrng_parameters.csv` and tested explicitly in the experimental model variants.
- The two-level variance decomposition with `n_inner = 1` (one replication per GP sample) conflates input uncertainty and simulation noise in a single CI. The nested simulation estimator in Analysis D separates them correctly; the `n_inner = 1` protocol is used in the main simulation for computational tractability.
- The MCE scenario uses a fixed exponential decay profile. Real mass casualty event arrival patterns vary significantly by incident type; the chosen parameters (Hirshberg et al., 2001; Hick et al., 2009) represent a moderate-scale urban trauma event.
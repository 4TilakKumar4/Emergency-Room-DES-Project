# Emergency Room Discrete-Event Simulation

**IE 7215 — Discrete Event Simulation Analysis**  
Northeastern University · Spring 2026  
Author: Tathya Malav Kamdar, Tilak Kumar Byradenahalli Ramesh, Uriel Baron

---

## Overview

This project implements a complete discrete-event simulation (DES) pipeline for an Emergency Room (ER) using a custom Python simulation engine. The model covers patient arrivals, triage, physician consultation, resource allocation, and length of stay, with a full input modelling and output analysis framework.

The simulation is built in three progressive versions:

| Model | File | Arrival Process |
|---|---|---|
| Baseline NHPP | `ERSimulationModelNHPP.py` | Fixed hourly rates from data |
| GP Arrival Model | `ERSimulationModelGP.py` | GP posterior samples per replication |
| GP + Severity Output | `ERSimulationModelGPwithSev.py` | GP rates + per-severity statistics |

---

## Repository Structure

```
Emergency-Room-DES-Project/
│
├── sim_engine/                       # Simulation framework
│   ├── __init__.py
│   ├── SimClasses.py                 # Core DES primitives
│   ├── SimFunctions.py               # Scheduling and initialisation
│   ├── SimRNG.py                     # Random number generation (100 streams)
│   └── ArrivalRateGP.py              # GP arrival rate model
│
├── Sources/                          # Input data (not tracked by git)
│   ├── er_5000_patients.csv
│   └── simrng_parameters.csv
│
├── Results/                          # Output plots and CSVs (auto-created)
│
├── ERSimulationModelNHPP.py          # Baseline simulation model
├── ERSimulationModelGP.py            # GP arrival uncertainty model
├── ERSimulationModelGPwithSev.py     # GP model with severity-stratified output
├── Fitting_Dist_Tat.py               # Distribution fitting (primary)
├── New_Distrubution_plot1.py         # Distribution fitting (SimRNG export)
│
├── README.md
├── CONTRIBUTING.md
└── requirements.txt
```

---

## System Description

| Component | Configuration |
|---|---|
| Receptionist | 1 (FIFO queue — severity assigned on arrival but unknown to clerk) |
| Triage nurses | 2 (priority queue — high before medium before low) |
| Physicians | 3 (priority queue — high before medium before low) |
| Arrival process | Non-Homogeneous Poisson Process (NHPP), 24-hour cycle |
| Service times | Severity-specific Lognormal/Erlang distributions |
| Warmup | 480 minutes (8 hours) |
| Steady-state run | 1440 minutes (24 hours) |
| Replications | 100 |

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### Running the simulation

Place the data files in `Sources/`:

```
Sources/er_5000_patients.csv
Sources/simrng_parameters.csv
```

Run the GP model with severity-stratified output (recommended):

```bash
python ERSimulationModelGPwithSev.py
```

Or run the baseline NHPP model:

```bash
python ERSimulationModelNHPP.py
```

### Running distribution fitting

```bash
python Fitting_Dist_Tat.py
```

This reads `Sources/er_5000_patients.csv`, fits distributions across three approaches, and writes `Results/simrng_parameters.csv`.

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
| `*_wait_min` | Time waiting in each queue |
| `er_los_min` | Total ED length of stay |

Severity breakdown: low 55.6%, medium 34.9%, high 9.6%

### `simrng_parameters.csv`

Output of `Fitting_Dist_Tat.py`. Contains SimRNG-compatible distribution parameters for all three service stages across all severity levels. The simulation loads this file at runtime — no hardcoded service time parameters exist in the model.

---

## Outputs

All outputs are written to the `Results/` directory, which is created automatically.

| File | Description |
|---|---|
| `ED_output_analysis.png` | 2×2 histograms of key metrics with 95% CI and empirical targets |
| `ED_utilisation.png` | Bar chart of resource utilisation with CI error bars |
| `ED_gp_posterior.png` | GP posterior diagnostic with credible band and sample curves |
| `ED_rep_results.csv` | Per-replication raw results (19 columns including per-severity) |
| `simrng_parameters.csv` | Fitted distribution parameters for all stages (from fitting script) |

### Validation Targets

| Metric | Empirical | Notes |
|---|---|---|
| Registration wait | 4.67 min | 44.1% zero wait |
| Triage wait | 0.01 min | 99.4% zero wait |
| Doctor wait | 249.67 min | 5.5% zero wait — primary bottleneck |
| Length of stay | 285.63 min | |

---

## Modelling Approach Summary

### Input Modelling

Distribution fitting is performed in three approaches:

1. **Pooled** — single distribution per stage, ignores severity
2. **Severity-stratified** — separate distributions per (stage, severity) combination — used in simulation
3. **Service vs. wait separation** — service times as simulation inputs, wait times as validation targets only

The Kolmogorov-Smirnov test selects the best-fitting distribution from eight candidates: Normal, Exponential, Gamma, Erlang, Lognormal, Weibull, Triangular, Beta.

### Arrival Process

The NHPP is modelled using hour-by-hour arrival rates estimated from the 30-day dataset. In the GP models, these rates are replaced by samples from a Gaussian Process posterior — a Log-Gaussian Cox Process fitted in log-space using a locally-periodic kernel (`ExpSineSquared × RBF + WhiteKernel`). Each replication draws a different plausible rate curve, propagating arrival rate estimation uncertainty into the output confidence intervals.

### Input Uncertainty Quantification

The GP model implements a two-level simulation protocol:

- **Outer loop**: sample a rate curve from the GP posterior (input uncertainty)
- **Inner loop**: run the DES replication with that rate curve (simulation noise)

Output variance is decomposed using the law of total variance into input uncertainty and simulation noise components. This identifies whether reducing output CI width requires more arrival data or more simulation replications.

### Queue Discipline

Registration uses FIFO because severity is not clinically assessed until triage. Triage and physician queues use priority discipline (high > medium > low). Patients of equal severity retain FIFO order within their priority class.

---

## Project Structure Notes

The `sim_engine/` package contains all simulation framework code. Import pattern in model files:

```python
from sim_engine import SimClasses, SimFunctions, SimRNG
from sim_engine.ArrivalRateGP import ArrivalRateGP
```

`SimFunctions.py` uses a relative import (`from . import SimClasses`) because both files are in the same package.

---

## Known Assumptions and Limitations

- Registration queue is modelled as FIFO. In a real ED, ESI Level 1/2 patients would bypass registration. The data does not evidence this behaviour, so modelling it would create an inconsistency with the fitted input distributions.
- The dataset has a relatively flat diurnal arrival curve compared to real ED data. NHPP parameters are estimated directly from this dataset.
- Distributions with no direct SimRNG equivalent (Beta, Weibull) are approximated with `Erlang(m=1, mean)`, equivalent to `Expon(mean)`. This is documented in `simrng_parameters.csv`.
- The two-level variance decomposition with `n_inner = 1` (one replication per GP sample) slightly underestimates the simulation noise component. With 100 outer samples this is acceptable.

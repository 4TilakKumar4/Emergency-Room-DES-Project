"""
SensitivityAnalysis.py — Input sensitivity analyses for the ED simulation.

This module implements four sensitivity analyses to understand how input uncertainty propagates to output performance metrics: 
  A. Data volume sensitivity  — CI width vs days of arrival data
  B. Distribution sensitivity — KS winner vs simpler alternatives
  C. Hour-group sensitivity   — peak-hour vs overnight GP uncertainty
  D. Service time bootstrap   — variance decomposition across all input sources

Each analysis is self-contained and writes its output to Results/sensitivity_analysis/.

Inputs  : Sources/er_5000_patients.csv, Sources/simrng_parameters.csv
Outputs : Results/sensitivity_analysis/sens_A_data_volume.png
          Results/sensitivity_analysis/sens_B_distribution.png
          Results/sensitivity_analysis/sens_C_hour_groups.png
          Results/sensitivity_analysis/sens_D_bootstrap.png
          Results/sensitivity_analysis/sensitivity_summary.csv
"""

import os
import math
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import interp1d

# Import the simulation engine and shared simulation state.
# All event functions and module-level objects (queues, resources, stats)
# live in ERSimulationModelGPwithSev — imported here and used directly.
import ERSimulationModelGPwithSev as sim
from sim_engine.ArrivalRateGP import ArrivalRateGP


# File paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "Sources", "er_5000_patients.csv")
PARAMS_FILE = os.path.join(BASE_DIR, "Sources", "simrng_parameters.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "sensitivity_analysis")

# Analysis parameters — reduce NUM_REPS for faster iteration if needed
NUM_REPS   = 100
DATA_STEPS = [10, 15, 20, 25, 30]      # Analysis A: days of data
# Analysis D nested simulation budget is set inside analysisD() as R_INNER and B_OUTER

PEAK_HOURS      = list(range(8, 18))    # Analysis C: hours 8–17 (peak)
OVERNIGHT_HOURS = list(range(0, 8))     # Analysis C: hours 0–7  (overnight)


def ensureResultsDir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def runBatch(rateFn, nReps: int, params: dict | None = None) -> pd.DataFrame:
    """
    Run nReps replications, all using the same rateFn and optional params override.
    If params is provided it temporarily replaces sim.theParams for the batch.
    """
    originalParams = sim.theParams
    if params is not None:
        sim.theParams = params
    try:
        rows = [sim.runReplication(rateFn) for _ in range(nReps)]
    finally:
        sim.theParams = originalParams
    return pd.DataFrame(rows)


def ci95(series: pd.Series) -> tuple[float, float]:
    m  = series.mean()
    hw = 1.96 * series.std(ddof=1) / math.sqrt(len(series))
    return m, hw


# Analysis A — Data Volume Sensitivity

def analysisA(hourlyCounts: np.ndarray) -> pd.DataFrame:
    """
    Refit GP with progressively fewer days of arrival data.
    For each data level, run NUM_REPS replications and record CI width.

    The CI width measures how uncertain our performance estimate is given
    only N days of arrival data. A steep decline toward 30 days means
    collecting more data would substantially narrow output uncertainty.
    A flat curve means 30 days is adequate.
    """
    print("\nAnalysis A: Data Volume Sensitivity")
    rows = []

    for nDays in DATA_STEPS:
        subsample = hourlyCounts[:nDays, :]
        gp        = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
        gp.fit(subsample)

        repResults = []
        for r in range(NUM_REPS):
            rateFn = gp.sampleRateCurve(randomState=r)
            repResults.append(sim.runReplication(rateFn))

        df         = pd.DataFrame(repResults)
        dw_m, dw_hw = ci95(df["doctorWait"])
        ls_m, ls_hw = ci95(df["LOS"])

        rows.append({
            "nDays":         nDays,
            "doctorWait_m":  dw_m,
            "doctorWait_hw": dw_hw,
            "LOS_m":         ls_m,
            "LOS_hw":        ls_hw,
        })
        print(f"  {nDays:2d} days │ doctorWait {dw_m:7.2f} ± {dw_hw:5.2f}  │  "
              f"LOS {ls_m:7.2f} ± {ls_hw:5.2f}")

    return pd.DataFrame(rows)


def plotA(resultsA: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analysis A — CI Half-Width vs Days of Arrival Data",
                 fontsize=13, fontweight="bold")

    for ax, col, label, color in [
        (ax1, "doctorWait_hw", "Doctor Wait CI Half-Width (min)", "#e74c3c"),
        (ax2, "LOS_hw",        "LOS CI Half-Width (min)",         "#3498db"),
    ]:
        ax.plot(resultsA["nDays"], resultsA[col], "o-", color=color,
                linewidth=2, markersize=8)
        ax.set_xlabel("Days of arrival data")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(DATA_STEPS)
        ax.grid(True, alpha=0.3)

        # Annotate the slope between last two points as diminishing returns indicator
        x1, x2 = resultsA["nDays"].iloc[-2], resultsA["nDays"].iloc[-1]
        y1, y2 = resultsA[col].iloc[-2], resultsA[col].iloc[-1]
        slope  = (y2 - y1) / (x2 - x1)
        ax.annotate(f"Δ = {slope:.2f} min/day\nat 25–30 days",
                    xy=(x2, y2), xytext=(x2 - 6, y2 + (y2 - y1) * 2),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color))

    plt.tight_layout()
    _save(fig, "sens_A_data_volume.png")


# Analysis B — Distribution Choice Sensitivity

def buildAlternativeParams(rawDf: pd.DataFrame, baseParams: dict) -> dict:
    """
    Build two alternative parameter dicts for the doctor stage:
      - expon_mean: Expon(mean) — simplest possible approximation
      - erlang2:    Erlang(m=2, mean) — moderate complexity alternative

    All other stages retain baseParams to isolate the doctor service effect.
    """
    alternativeParams = {}

    for label, builderFn in [
        ("expon_mean", lambda mean, _: ("Expon", mean)),
        ("erlang2",    lambda mean, _: ("Erlang", 2, mean)),
    ]:
        p = copy.deepcopy(baseParams)
        for sev in sim.SEVERITIES:
            col  = f"doctor_service_min"
            data = rawDf[rawDf["severity"] == sev]["doctor_service_min"].dropna().values
            data = data[data > 0]
            if len(data) < 10:
                continue
            mean = float(np.mean(data))
            var  = float(np.var(data, ddof=1))
            p["doctor"][sev] = builderFn(mean, var)
        alternativeParams[label] = p

    return alternativeParams


def analysisB(hourlyCounts: np.ndarray, baseParams: dict,
              rawDf: pd.DataFrame) -> pd.DataFrame:
    """
    Compare output CIs under three doctor service time model assumptions.
    All three use the same GP rate curves for fair comparison — only the
    service time distribution changes.
    """
    print("\nAnalysis B: Distribution Choice Sensitivity")

    # Generate one set of rate functions to reuse across all variants
    gp = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gp.fit(hourlyCounts)
    rateFns = [gp.sampleRateCurve(randomState=r) for r in range(NUM_REPS)]

    altParams = buildAlternativeParams(rawDf, baseParams)
    scenarios = {
        "KS winner (fitted)": baseParams,
        **altParams,
    }

    rows = []
    for label, params in scenarios.items():
        repResults = []
        for r, rateFn in enumerate(rateFns):
            originalParams = sim.theParams
            sim.theParams  = params
            try:
                repResults.append(sim.runReplication(rateFn))
            finally:
                sim.theParams = originalParams

        df         = pd.DataFrame(repResults)
        dw_m, dw_hw = ci95(df["doctorWait"])
        ls_m, ls_hw = ci95(df["LOS"])

        rows.append({
            "scenario":      label,
            "doctorWait_m":  dw_m,
            "doctorWait_hw": dw_hw,
            "LOS_m":         ls_m,
            "LOS_hw":        ls_hw,
        })
        print(f"  {label:<28} │ doctorWait {dw_m:7.2f} ± {dw_hw:5.2f}  │  "
              f"LOS {ls_m:7.2f} ± {ls_hw:5.2f}")

    return pd.DataFrame(rows)


def plotB(resultsB: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analysis B — Sensitivity to Doctor Service Time Distribution",
                 fontsize=13, fontweight="bold")

    colors = ["#27ae60", "#e74c3c", "#f39c12"]
    for ax, col, label in [
        (axes[0], "doctorWait", "Doctor Wait (min)"),
        (axes[1], "LOS",        "Length of Stay (min)"),
    ]:
        x     = np.arange(len(resultsB))
        means  = resultsB[f"{col}_m"].values
        errors = resultsB[f"{col}_hw"].values

        bars = ax.bar(x, means, yerr=errors, color=colors[:len(resultsB)],
                      alpha=0.8, capsize=6, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(resultsB["scenario"], rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val, err in zip(bars, means, errors):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + err + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "sens_B_distribution.png")


# Analysis C — Hour-Group Sensitivity

def pinnedRateFn(gp: ArrivalRateGP, pinnedHours: list[int],
                 randomState: int) -> callable:
    """
    Draw a GP posterior sample but replace the rate values for pinnedHours
    with their posterior mean. This isolates the uncertainty contribution
    from the remaining (unpinned) hours.

    pinnedHours: list of integer hours (0–23) to freeze at posterior mean.
    """
    xFine   = gp._xFine.ravel()
    meanFn  = gp.meanRateCurve()
    sampleFn = gp.sampleRateCurve(randomState=randomState)

    # Evaluate both at the fine grid
    meanRates   = np.array([float(meanFn(x))   for x in xFine])
    sampleRates = np.array([float(sampleFn(x)) for x in xFine])

    # For each grid point, check if it falls inside a pinned hour and replace
    pinnedRates = sampleRates.copy()
    for i, x in enumerate(xFine):
        hour = int(x) % 24
        if hour in pinnedHours:
            pinnedRates[i] = meanRates[i]

    return interp1d(xFine, np.maximum(pinnedRates, 1e-6),
                    kind="cubic", bounds_error=False,
                    fill_value=(pinnedRates[0], pinnedRates[-1]))


def analysisC(hourlyCounts: np.ndarray) -> pd.DataFrame:
    """
    Run three scenarios:
      1. Baseline   — full GP posterior free (all hours uncertain)
      2. Pin peak   — hours 8–17 fixed at posterior mean; overnight free
      3. Pin night  — hours 0–7 fixed at posterior mean; peak free

    Comparing output variance across scenarios isolates which hour group
    drives the arrival-rate component of output uncertainty.
    """
    print("\nAnalysis C: Hour-Group Sensitivity")

    gp = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gp.fit(hourlyCounts)

    scenarios = {
        "Baseline (all free)":     [],
        "Peak pinned (8–17)":      PEAK_HOURS,
        "Overnight pinned (0–7)":  OVERNIGHT_HOURS,
    }

    rows = []
    for label, pinned in scenarios.items():
        repResults = []
        for r in range(NUM_REPS):
            if pinned:
                rateFn = pinnedRateFn(gp, pinned, randomState=r)
            else:
                rateFn = gp.sampleRateCurve(randomState=r)
            repResults.append(sim.runReplication(rateFn))

        df           = pd.DataFrame(repResults)
        dw_m, dw_hw  = ci95(df["doctorWait"])
        dw_var       = float(df["doctorWait"].var(ddof=1))

        rows.append({
            "scenario":      label,
            "doctorWait_m":  dw_m,
            "doctorWait_hw": dw_hw,
            "doctorWait_var": dw_var,
        })
        print(f"  {label:<30} │ doctorWait {dw_m:7.2f} ± {dw_hw:5.2f}  "
              f"│ var = {dw_var:8.2f}")

    results = pd.DataFrame(rows)

    # Variance reduction from pinning tells you the contribution of that group
    baseVar   = results.loc[results["scenario"] == "Baseline (all free)",
                            "doctorWait_var"].values[0]
    for _, row in results.iterrows():
        if row["scenario"] != "Baseline (all free)":
            reduction = (baseVar - row["doctorWait_var"]) / baseVar * 100
            print(f"  Pinning {row['scenario'][:25]:<25} reduces variance by "
                  f"{reduction:.1f}%")

    return results


def plotC(resultsC: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analysis C — Doctor Wait Sensitivity to Hour-Group Uncertainty",
                 fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e74c3c", "#27ae60"]
    x      = np.arange(len(resultsC))

    ax1.bar(x, resultsC["doctorWait_m"], yerr=resultsC["doctorWait_hw"],
            color=colors[:len(resultsC)], alpha=0.8, capsize=6, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(resultsC["scenario"], rotation=10, ha="right", fontsize=9)
    ax1.set_ylabel("Doctor Wait (min)")
    ax1.set_title("Mean ± 95% CI", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, resultsC["doctorWait_var"],
            color=colors[:len(resultsC)], alpha=0.8, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(resultsC["scenario"], rotation=10, ha="right", fontsize=9)
    ax2.set_ylabel("Doctor Wait Variance (min²)")
    ax2.set_title("Output Variance by Scenario", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(ax2.patches, resultsC["doctorWait_var"]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.5, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "sens_C_hour_groups.png")


# Analysis D — Service Time Bootstrap + Variance Decomposition

def detectServiceCols(rawDf: pd.DataFrame) -> dict:
    """
    Auto-detect service time column names from the CSV by searching for
    columns whose names contain each stage keyword and 'service' or 'min'.
    Falls back to the most likely patterns if detection is ambiguous.
    """
    cols = rawDf.columns.str.lower().tolist()
    mapping = {}
    patterns = {
        "reg":    ["registr"],
        "triage": ["triage"],
        "doctor": ["doctor", "physician"],
    }
    for stage, keywords in patterns.items():
        candidates = [c for c in cols
                      if any(kw in c for kw in keywords)
                      and ("service" in c or "min" in c)
                      and "wait" not in c]
        mapping[stage] = rawDf.columns[cols.index(candidates[0])] if candidates else None
    return mapping


def bootstrapServiceParams(rawDf: pd.DataFrame, rng: np.random.Generator,
                           colMap: dict) -> dict:
    """
    Resample patients with replacement and refit service time parameters.
    Uses Lognormal(mean, var) — only the MLE estimates change per resample,
    reflecting finite-sample parameter uncertainty.

    colMap must be {stage: column_name} as returned by detectServiceCols().
    Returns a params dict in the same format as loadParams().
    """
    bootDf = rawDf.sample(n=len(rawDf), replace=True,
                          random_state=int(rng.integers(1e9)))
    params = {"reg": {}, "triage": {}, "doctor": {}}

    for stage, col in colMap.items():
        if col is None:
            params[stage] = copy.deepcopy(sim.theParams[stage])
            continue
        for sev in sim.SEVERITIES:
            data = bootDf[bootDf["severity"] == sev][col].dropna().values
            data = data[data > 0]
            if len(data) < 5:
                params[stage][sev] = sim.theParams[stage][sev]
                continue
            mean = float(np.mean(data))
            var  = float(np.var(data, ddof=1))
            params[stage][sev] = ("Lognormal", mean, var)

    return params


def nestedVariance(outerSamples: list[np.ndarray]) -> tuple[float, float]:
    """
    Proper law of total variance estimator for nested simulation.

    outerSamples: list of arrays, one per outer sample.
                  Each array contains R_inner output values.

    Returns (inputVar, simVar) where:
      inputVar = Var[E(Y|θ)]  — variance of inner means across outer samples
      simVar   = E[Var(Y|θ)]  — mean inner variance (simulation noise)

    This avoids the subtraction-of-independent-estimates flaw.
    The two components are estimated from the same simulation budget,
    so their sum equals the total output variance asymptotically.
    """
    innerMeans = np.array([arr.mean() for arr in outerSamples])
    innerVars  = np.array([arr.var(ddof=1) if len(arr) > 1
                           else 0.0 for arr in outerSamples])
    inputVar = float(np.var(innerMeans, ddof=1))
    simVar   = float(np.mean(innerVars))
    return inputVar, simVar


def analysisD(hourlyCounts: np.ndarray, baseParams: dict,
              rawDf: pd.DataFrame) -> dict:
    """
    Variance decomposition using nested simulation (law of total variance).

    Two separate nested loops isolate each input source:

      Arrival loop:  outer = GP posterior samples, inner = R_INNER replications
                     each with that rate curve and fixed service params.
                     inputVar_arrival = Var[E(Y|θ_arrival)]

      Service loop:  outer = bootstrap service params, inner = R_INNER replications
                     each with those params and fixed GP mean rate.
                     inputVar_service = Var[E(Y|θ_service)]

    Using nested simulation instead of subtracting independent scenario variances
    avoids the statistical flaw where sampling noise causes negative estimates
    that collapse to zero under the max(..., 0) clamp.
    """
    print("\nAnalysis D: Service Time Bootstrap + Variance Decomposition")

    # R_INNER > 1 is essential — it gives us within-outer-sample variance,
    # which is the simulation noise estimate. Without this we cannot separate
    # input uncertainty from simulation noise.
    R_INNER = 10
    B_OUTER = 50   # outer samples per source — 50 × 10 = 500 total runs per source

    gp     = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gp.fit(hourlyCounts)
    meanFn = gp.meanRateCurve()
    rng    = np.random.default_rng(42)
    colMap = detectServiceCols(rawDf)

    print(f"  Detected service time columns: {colMap}")
    print(f"  Design: {B_OUTER} outer × {R_INNER} inner = "
          f"{B_OUTER * R_INNER} runs per source")

    # Arrival nested loop
    print("  Running arrival uncertainty loop...")
    arrivalOuter = []
    for b in range(B_OUTER):
        rateFn  = gp.sampleRateCurve(randomState=b)
        inner   = np.array([
            sim.runReplication(rateFn)["doctorWait"]
            for _ in range(R_INNER)
        ])
        arrivalOuter.append(inner)

    inputVarArrival, simVarArrival = nestedVariance(arrivalOuter)

    # Service bootstrap nested loop
    print("  Running service time uncertainty loop...")
    serviceOuter = []
    for _ in range(B_OUTER):
        bootParams     = bootstrapServiceParams(rawDf, rng, colMap)
        originalParams = sim.theParams
        sim.theParams  = bootParams
        try:
            inner = np.array([
                sim.runReplication(meanFn)["doctorWait"]
                for _ in range(R_INNER)
            ])
        finally:
            sim.theParams = originalParams
        serviceOuter.append(inner)

    inputVarService, simVarService = nestedVariance(serviceOuter)

    # Use the average simulation noise estimate from both loops
    simVar   = (simVarArrival + simVarService) / 2.0
    total    = inputVarArrival + inputVarService + simVar

    decomp = {
        "simNoise":        simVar,
        "arrivalInput":    inputVarArrival,
        "serviceInput":    inputVarService,
        "total":           total,
        "simFraction":     simVar           / total if total > 0 else 0,
        "arrivalFraction": inputVarArrival  / total if total > 0 else 0,
        "serviceFraction": inputVarService  / total if total > 0 else 0,
    }

    print(f"\n  VARIANCE DECOMPOSITION (doctor wait, nested simulation)")
    print(f"  {'Source':<30}  {'Variance':>10}  {'Share':>8}")
    print(f"  {'Simulation noise':<30}  {simVar:10.3f}  "
          f"{decomp['simFraction'] * 100:7.1f}%")
    print(f"  {'Arrival rate uncertainty':<30}  {inputVarArrival:10.3f}  "
          f"{decomp['arrivalFraction'] * 100:7.1f}%")
    print(f"  {'Service time uncertainty':<30}  {inputVarService:10.3f}  "
          f"{decomp['serviceFraction'] * 100:7.1f}%")
    print(f"  {'Total':<30}  {total:10.3f}")
    print(f"\n  Interpretation:")
    if decomp["arrivalFraction"] > decomp["serviceFraction"]:
        print(f"  Arrival rate uncertainty dominates ({decomp['arrivalFraction']*100:.0f}%).")
        print(f"  Collecting more days of arrival data would narrow output CI most.")
    else:
        print(f"  Service time uncertainty dominates ({decomp['serviceFraction']*100:.0f}%).")
        print(f"  Collecting more patient-level service time records would help most.")

    return decomp


def plotD(decomp: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Analysis D — Doctor Wait Variance Decomposition",
                 fontsize=13, fontweight="bold")

    labels = ["Simulation\nnoise", "Arrival rate\nuncertainty",
              "Service time\nuncertainty"]
    values = [decomp["simNoise"], decomp["arrivalInput"], decomp["serviceInput"]]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    total  = decomp["total"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white",
                  width=0.5)
    ax.set_ylabel("Variance contribution (min²)")
    ax.set_title(
        "Law of total variance: Var(Y) = Var_arrival + Var_service + Var_sim",
        fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + total * 0.01,
                f"{val:.1f}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, "sens_D_bootstrap.png")


# Main

def main() -> None:
    ensureResultsDir()

    print("Loading data...")
    hourlyCounts = sim.buildHourlyCounts(SOURCE_FILE)
    rawDf        = pd.read_csv(SOURCE_FILE)
    print(f"  Count matrix: {hourlyCounts.shape}")

    print("Loading base service time parameters...")
    sim.theParams = sim.loadParams(PARAMS_FILE)
    baseParams    = copy.deepcopy(sim.theParams)

    summaryRows = []

    # Analysis A
    resultsA = analysisA(hourlyCounts)
    plotA(resultsA)
    summaryRows.append({"analysis": "A",
                        "description": "CI half-width at 30 days (doctorWait)",
                        "value": round(resultsA["doctorWait_hw"].iloc[-1], 3)})

    # Analysis B
    resultsB = analysisB(hourlyCounts, baseParams, rawDf)
    plotB(resultsB)
    ksRow  = resultsB[resultsB["scenario"] == "KS winner (fitted)"]
    expRow = resultsB[resultsB["scenario"] == "expon_mean"]
    if len(ksRow) and len(expRow):
        diff = abs(float(ksRow["doctorWait_m"].values[0]) -
                   float(expRow["doctorWait_m"].values[0]))
        summaryRows.append({"analysis": "B",
                            "description": "KS vs Expon mean difference (doctorWait)",
                            "value": round(diff, 3)})

    # Analysis C
    resultsC = analysisC(hourlyCounts)
    plotC(resultsC)
    baseVar = resultsC.loc[
        resultsC["scenario"] == "Baseline (all free)", "doctorWait_var"
    ].values[0]
    peakVar = resultsC.loc[
        resultsC["scenario"] == "Peak pinned (8–17)", "doctorWait_var"
    ].values[0]
    summaryRows.append({"analysis": "C",
                        "description": "Variance reduction from pinning peak hours (%)",
                        "value": round((baseVar - peakVar) / baseVar * 100, 1)})

    # Analysis D
    decomp = analysisD(hourlyCounts, baseParams, rawDf)
    plotD(decomp)
    summaryRows.append({"analysis": "D",
                        "description": "Arrival rate uncertainty fraction (%)",
                        "value": round(decomp["arrivalFraction"] * 100, 1)})
    summaryRows.append({"analysis": "D",
                        "description": "Service time uncertainty fraction (%)",
                        "value": round(decomp["serviceFraction"] * 100, 1)})
    summaryRows.append({"analysis": "D",
                        "description": "Simulation noise fraction (%)",
                        "value": round(decomp["simFraction"] * 100, 1)})

    # Save summary table
    summary = pd.DataFrame(summaryRows)
    outPath = os.path.join(RESULTS_DIR, "sensitivity_summary.csv")
    summary.to_csv(outPath, index=False)
    print(f"\n  Summary → {outPath}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
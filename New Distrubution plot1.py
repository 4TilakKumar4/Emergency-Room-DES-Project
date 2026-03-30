"""
ER Patient Flow — Distribution Fitting Analysis

Fits statistical distributions to ER service times, wait times, and
inter-arrival times using three approaches:
  1. Pooled (urgency-independent)
  2. Severity-stratified
  3. Service times vs. wait times (simulation input vs. validation)

Input   : Sources/er_synthetic_5000_patients.csv
Outputs : Results/*.png  (10 diagnostic plots)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# File paths — edit here only if the project structure changes
SOURCE_FILE = os.path.join("Sources", "er_synthetic_5000_patients.csv")
RESULTS_DIR = "Results"

# Candidate distribution families tested against every time variable
DISTRIBUTIONS: dict[str, stats.rv_continuous] = {
    "Normal":      stats.norm,
    "Exponential": stats.expon,
    "Gamma":       stats.gamma,
    "Erlang":      stats.erlang,
    "Lognormal":   stats.lognorm,
    "Weibull":     stats.weibull_min,
    "Triangular":  stats.triang,
    "Beta":        stats.beta,
}

# Severity levels present in the dataset, listed low to high
SEVERITIES = ["low", "medium", "high"]

# Plot colours: one per severity level and one per ER stage
SEV_COLORS   = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
STAGE_COLORS = ["#3498db", "#f39c12", "#e74c3c", "#27ae60"]

# KS p-value threshold — fits above this are labelled "GOOD" in console output
SIGNIFICANCE = 0.05


def createOutputDirectory(path: str) -> None:
    """Create the results directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def findBestFittingDistribution(data: np.ndarray) -> tuple[str, tuple, float, float, list[dict]]:
    """
    Fit every distribution in DISTRIBUTIONS to *data* and return the best one.

    The Kolmogorov-Smirnov (KS) test is used for goodness-of-fit.
    The distribution with the highest p-value is declared the winner.

    Parameters
    ----------
    data : 1-D array of strictly filterPositiveValues floats.

    Returns
    -------
    best_name   : name of the winning distribution
    best_params : MLE parameter tuple for that distribution
    best_ks     : KS statistic of the winner
    best_pval   : KS p-value of the winner
    all_fits    : list of dicts {name, params, ks, pval} for every candidate
    """
    best_name:   str | None   = None
    best_params: tuple | None = None
    best_ks:     float        = np.inf
    best_pval:   float        = -1.0
    all_fits:    list[dict]   = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            # Estimate distribution parameters via Maximum Likelihood Estimation
            params = dist.fit(data)

            # Measure how well the fitted CDF matches the empirical CDF
            ks, pval = stats.kstest(data, dist.cdf, args=params)
            all_fits.append({"name": name, "params": params, "ks": ks, "pval": pval})

            # Keep whichever distribution has the highest p-value so far
            if pval > best_pval:
                best_name, best_params, best_ks, best_pval = name, params, ks, pval

        except Exception as exc:
            print(f"    [WARN] {name} failed on this data: {exc}")

    return best_name, best_params, best_ks, best_pval, all_fits


def filterPositiveValues(series: pd.Series | np.ndarray) -> np.ndarray:
    """Strip zeros and NaNs — fitting requires strictly filterPositiveValues values."""
    arr = series.values if isinstance(series, pd.Series) else np.asarray(series)
    return arr[arr > 0]


def printKsTestResults(all_fits: list[dict], best_name: str) -> None:
    """Print KS results for every candidate, highlighting the winner."""
    for f in all_fits:
        tag  = "GOOD " if f["pval"] > SIGNIFICANCE else "     "
        star = " ★ WINNER" if f["name"] == best_name else ""
        print(f"    {tag} {f['name']:15s}: KS={f['ks']:.4f},  p={f['pval']:.4f}{star}")


def saveFigure(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the Results folder and immediately close it to free memory."""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def evaluatePdfOverRange(
    dist_name: str, params: tuple, data: np.ndarray, x_vals: np.ndarray
) -> np.ndarray | None:
    """Evaluate the PDF of a fitted distribution over x_vals; return None on error."""
    try:
        return DISTRIBUTIONS[dist_name].pdf(x_vals, *params)
    except Exception:
        return None


def loadAndPreparePatientData(filepath: str) -> pd.DataFrame:
    """
    Read the patient CSV, parse arrival timestamps, and derive the
    inter-arrival time column by differencing consecutive arrival times.
    """
    print(f"\nLoading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Parse arrival_time as proper datetime and sort chronologically
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df = df.sort_values("arrival_time").reset_index(drop=True)

    # Time between consecutive arrivals in minutes (first row will be NaN)
    df["inter_arrival_min"] = df["arrival_time"].diff().dt.total_seconds() / 60

    print(f"  {len(df):,} patients loaded.")
    return df


def printSummaryStatistics(df: pd.DataFrame) -> None:
    """Print patient counts, date range, resource counts, and mean service times."""
    print(f"\n{'=' * 80}")
    print("DATASET OVERVIEW")
    print(f"{'=' * 80}")
    print(f"  Total patients : {len(df):,}")
    print(f"  Date range     : {df['arrival_time'].min()}  →  {df['arrival_time'].max()}")
    print(
        f"  Resources      : {df['registration_resource_id'].nunique()} Clerks, "
        f"{df['triage_resource_id'].nunique()} Nurses, "
        f"{df['doctor_resource_id'].nunique()} Doctors"
    )

    print("\n  Severity distribution:")
    for sev in SEVERITIES:
        n = (df["severity"] == sev).sum()
        print(f"    {sev:<8}: {n:>5,}  ({n / len(df) * 100:.1f}%)")

    # Map human-readable labels to column names for the averages block
    metrics = {
        "Registration Service": "registration_service_min",
        "Registration Wait":    "registration_wait_min",
        "Triage Service":       "triage_service_min",
        "Triage Wait":          "triage_wait_min",
        "Doctor Service":       "doctor_service_min",
        "Doctor Wait":          "doctor_wait_min",
        "Inter-arrival":        "inter_arrival_min",
        "Total ER LOS":         "er_los_min",
    }
    print("\n  Key averages (minutes):")
    for label, col in metrics.items():
        print(f"    {label:<25}: {df[col].dropna().mean():6.2f} min")


def fitPooledDistributions(df: pd.DataFrame) -> list[dict]:
    """
    Fit a single distribution to ALL patients for each time variable.
    Assumption: severity level does not affect service or inter-arrival times.
    This is the simplest possible simulation input model.
    """
    print(f"\n\n{'#' * 80}")
    print("APPROACH 1: POOLED DISTRIBUTIONS (Urgency-Independent)")
    print("Assumption: service times are the same regardless of severity.")
    print(f"{'#' * 80}")

    # Inter-arrival drives patient generation; service times drive each station's speed
    variables = {
        "Registration Service": df["registration_service_min"],
        "Triage Service":       df["triage_service_min"],
        "Doctor Service":       df["doctor_service_min"],
        "Inter-arrival":        df["inter_arrival_min"].dropna(),
    }

    results = []
    for name, series in variables.items():
        data = filterPositiveValues(series)
        best_name, best_params, best_ks, best_pval, all_fits = findBestFittingDistribution(data)
        print(f"\n--- {name}  (n={len(data):,},  mean={data.mean():.2f},  sd={data.std():.2f}) ---")
        printKsTestResults(all_fits, best_name)
        results.append({
            "Variable": name, "n": len(data),
            "Mean": data.mean(), "Std": data.std(),
            "Best Dist": best_name, "KS": best_ks, "p-value": best_pval,
            "Params": best_params, "all_fits": all_fits, "data": data,
        })

    print(f"\n\nAPPROACH 1 SUMMARY")
    print(f"{'Variable':<25} {'n':>6}  {'Mean':>7}  {'Best Dist':<15}  {'p-value':>10}")
    print("-" * 70)
    for r in results:
        print(f"  {r['Variable']:<25} {r['n']:>6}  {r['Mean']:>7.2f}  "
              f"{r['Best Dist']:<15}  {r['p-value']:>10.4f}")
    return results


def fitSeverityStratifiedDistributions(df: pd.DataFrame) -> list[dict]:
    """
    Fit a separate distribution for every (stage, severity) combination.
    Quantifies whether severity materially changes service time and determines
    whether the simulation needs severity-specific inputs.
    """
    print(f"\n\n{'#' * 80}")
    print("APPROACH 2: SEVERITY-SPECIFIC DISTRIBUTIONS")
    print("Tests whether service times differ meaningfully by severity level.")
    print(f"{'#' * 80}")

    # Only service stages are stratified; inter-arrival does not depend on severity
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }

    results = []
    for stage_name, col in stage_cols.items():
        print(f"\n{'=' * 60}\nSTAGE: {stage_name}\n{'=' * 60}")
        for sev in SEVERITIES:
            data = filterPositiveValues(df[df["severity"] == sev][col])

            # Skip severity groups that are too small to fit reliably
            if len(data) < 10:
                print(f"  [{sev.upper()}]  Skipped — fewer than 10 observations.")
                continue

            best_name, best_params, best_ks, best_pval, all_fits = findBestFittingDistribution(data)
            print(f"\n  {sev.upper()}  (n={len(data):,},  mean={data.mean():.2f},  sd={data.std():.2f}):")
            printKsTestResults(all_fits, best_name)
            results.append({
                "Stage": stage_name, "Severity": sev, "n": len(data),
                "Mean": data.mean(), "Std": data.std(),
                "Best Dist": best_name, "KS": best_ks, "p-value": best_pval,
                "Params": best_params, "all_fits": all_fits, "data": data,
            })

    print(f"\n\nAPPROACH 2 SUMMARY")
    print(f"{'Stage':<25} {'Severity':<10} {'n':>6}  {'Mean':>7}  {'Best Dist':<15}  {'p-value':>10}")
    print("-" * 78)
    for r in results:
        print(f"  {r['Stage']:<25} {r['Severity']:<10} {r['n']:>6}  {r['Mean']:>7.2f}  "
              f"{r['Best Dist']:<15}  {r['p-value']:>10.4f}")
    return results


def evaluateSeverityImpact(
    approach1Results: list[dict],
    approach2Results: list[dict],
    df: pd.DataFrame,
) -> None:
    """
    Compare pooled means against severity-specific means for each service stage.
    A spread greater than 20% of the pooled mean is flagged as practically
    significant, meaning severity-specific distributions should be used.
    """
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }
    print(f"\n\nPOOLED vs. SEVERITY-SPECIFIC — DOES SEVERITY MATTER?")
    print("-" * 70)
    for stage_name in stage_cols:
        pooled   = next(r for r in approach1Results if r["Variable"] == stage_name)
        sev_rows = [r for r in approach2Results if r["Stage"] == stage_name]
        means    = [r["Mean"] for r in sev_rows]

        # Guard: if all severity groups were skipped (< 10 obs each), skip this stage
        if not means:
            print(f"\n  {stage_name}: SKIPPED — no severity groups had enough data.")
            continue

        # Spread is the absolute range across the three severity groups
        spread     = max(means) - min(means)
        pct_spread = spread / pooled["Mean"] * 100

        print(f"\n  {stage_name}:")
        print(f"    Pooled mean: {pooled['Mean']:.2f} min  (p={pooled['p-value']:.4f})")
        for r in sev_rows:
            print(f"    {r['Severity']:<8} mean: {r['Mean']:.2f} min  (p={r['p-value']:.4f})")
        print(f"    Spread : {spread:.2f} min  ({pct_spread:.1f}% of pooled mean)")
        verdict = "SEVERITY MATTERS" if pct_spread > 20 else "SEVERITY EFFECT IS SMALL"
        print(f"    >>> {verdict}  ({pct_spread:.0f}% spread)")


def separateServiceAndWaitFits(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Separate service-time distributions (fed into the simulation as inputs)
    from wait-time distributions (used only to validate simulation output).
    Wait times are emergent — they arise from queue dynamics — so fitting them
    as inputs would be circular.
    """
    print(f"\n\n{'#' * 80}")
    print("APPROACH 3: SERVICE TIMES (Input) vs. WAIT TIMES (Validation)")
    print("Service times drive the simulation; wait times validate its output.")
    print(f"{'#' * 80}")

    service_vars = {
        "Registration Service": df["registration_service_min"],
        "Triage Service":       df["triage_service_min"],
        "Doctor Service":       df["doctor_service_min"],
    }
    wait_vars = {
        "Registration Wait": df["registration_wait_min"],
        "Doctor Wait":       df["doctor_wait_min"],
    }

    print(f"\n--- SERVICE TIMES (simulation inputs) ---")
    service_results = []
    for name, series in service_vars.items():
        data = filterPositiveValues(series)
        best_name, best_params, best_ks, best_pval, all_fits = findBestFittingDistribution(data)
        print(f"  {name:<25}: {best_name}  (mean={data.mean():.2f},  p={best_pval:.4f})")
        service_results.append({
            "Variable": name, "Type": "SERVICE (Input)", "n": len(data),
            "Mean": data.mean(), "Best Dist": best_name, "p-value": best_pval,
            "Params": best_params, "data": data, "all_fits": all_fits,
        })

    print(f"\n--- WAIT TIMES (validation targets) ---")
    wait_results = []
    for name, series in wait_vars.items():
        data   = filterPositiveValues(series)
        total  = len(series)
        waited = len(data)
        pct_w  = waited / total * 100

        # If almost no one waits, fitting is unreliable and unnecessary
        if waited < 10:
            print(f"  {name:<25}: SKIPPED — only {waited} patients waited.")
            continue

        best_name, best_params, best_ks, best_pval, all_fits = findBestFittingDistribution(data)
        print(f"  {name:<25}: {best_name}  (mean={data.mean():.2f},  p={best_pval:.4f})")
        print(f"    {waited:,}/{total:,} patients waited ({pct_w:.1f}%);  "
              f"{total - waited:,} had zero wait.")
        wait_results.append({
            "Variable": name, "Type": "WAIT (Validation)", "n": waited,
            "Mean": data.mean(), "Best Dist": best_name, "p-value": best_pval,
            "Params": best_params, "data": data, "all_fits": all_fits,
            "pct_waited": pct_w,
        })

    # Triage wait is so rare that a fitted distribution is not warranted
    triage_waited = (df["triage_wait_min"] > 0).sum()
    triage_total  = len(df)
    print(
        f"  {'Triage Wait':<25}: NEGLIGIBLE — "
        f"{triage_waited}/{triage_total} ({triage_waited / triage_total * 100:.1f}%) "
        f"patients waited.  (3 nurses absorb all demand.)"
    )

    print(f"\n\nAPPROACH 3 SUMMARY")
    hdr = f"{'Variable':<25} {'Type':<20} {'n':>6}  {'Mean':>7}  {'Best Dist':<15}  {'p-value':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in service_results:
        print(f"  {r['Variable']:<25} {'SERVICE (Input)':<20} {r['n']:>6}  "
              f"{r['Mean']:>7.2f}  {r['Best Dist']:<15}  {r['p-value']:>10.4f}")
    for r in wait_results:
        print(f"  {r['Variable']:<25} {'WAIT (Validation)':<20} {r['n']:>6}  "
              f"{r['Mean']:>7.2f}  {r['Best Dist']:<15}  {r['p-value']:>10.4f}")

    return service_results, wait_results


def plotPooledHistogramsWithFits(results: list[dict]) -> None:
    """
    2×2 grid of histograms with all candidate PDF curves overlaid.
    The winning distribution is drawn in solid black; others are grey dashed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Approach 1: Pooled Distribution Fits (Urgency-Independent)",
        fontsize=14, fontweight="bold",
    )
    for idx, r in enumerate(results):
        ax   = axes[idx // 2, idx % 2]
        data = r["data"]

        ax.hist(data, bins=40, density=True, alpha=0.5,
                color=STAGE_COLORS[idx], edgecolor="white", label="Data")

        # Evaluate every candidate PDF and overlay, highlighting the winner
        x_vals = np.linspace(max(1e-6, data.min()), np.percentile(data, 99), 300)
        for f in r["all_fits"]:
            pdf = evaluatePdfOverRange(f["name"], f["params"], data, x_vals)
            if pdf is None:
                continue
            is_best = f["name"] == r["Best Dist"]
            ax.plot(
                x_vals, pdf,
                linewidth=3 if is_best else 1,
                color="black" if is_best else "gray",
                linestyle="-" if is_best else "--",
                alpha=1.0 if is_best else 0.35,
                label=f'{f["name"]} (p={f["pval"]:.3f}){"  ★" if is_best else ""}',
            )
        ax.set_title(f'{r["Variable"]}: {r["Best Dist"]} (p={r["p-value"]:.4f})',
                     fontweight="bold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    saveFigure(fig, "approach1_pooled_fits.png")


def plotPooledQqPlots(results: list[dict]) -> None:
    """
    2×2 Q-Q plots for the pooled fits.
    Points close to the red 45° line indicate a good fit; deviations in the
    tails reveal where the theoretical distribution diverges from reality.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Approach 1: Q-Q Plots (Pooled)", fontsize=14, fontweight="bold")

    for idx, r in enumerate(results):
        ax   = axes[idx // 2, idx % 2]
        data = np.sort(r["data"])
        dist = DISTRIBUTIONS[r["Best Dist"]]
        n    = len(data)

        # Use uniform order statistics i/(n+1) to avoid 0 and 1 at the boundaries
        probs       = np.arange(1, n + 1) / (n + 1)
        theoretical = dist.ppf(probs, *r["Params"])

        ax.scatter(theoretical, data, alpha=0.2, s=8, color=STAGE_COLORS[idx])
        lo = min(theoretical.min(), data.min())
        hi = max(theoretical.max(), data.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect fit")
        ax.set_title(f'{r["Variable"]}: {r["Best Dist"]} (p={r["p-value"]:.4f})',
                     fontweight="bold")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Observed quantiles")
        ax.legend(fontsize=8)

    plt.tight_layout()
    saveFigure(fig, "approach1_qq_plots.png")


def plotSeverityHistogramsPerStage(approach2Results: list[dict]) -> None:
    """
    One 1×3 histogram figure per service stage, with one subplot per severity.
    Lets you visually compare whether the distribution shape shifts across
    low / medium / high patients.
    """
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }
    for stage_name in stage_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Approach 2: {stage_name} by Severity",
                     fontsize=14, fontweight="bold")

        for idx, sev in enumerate(SEVERITIES):
            ax   = axes[idx]
            rows = [r for r in approach2Results
                    if r["Stage"] == stage_name and r["Severity"] == sev]

            # Hide the subplot if this severity group has no fit results
            if not rows:
                ax.set_visible(False)
                continue

            r    = rows[0]
            data = r["data"]

            ax.hist(data, bins=30, density=True, alpha=0.6,
                    color=SEV_COLORS[sev], edgecolor="white", label="Data")
            x_vals = np.linspace(max(1e-6, data.min()), np.percentile(data, 99), 300)
            pdf    = evaluatePdfOverRange(r["Best Dist"], r["Params"], data, x_vals)
            if pdf is not None:
                ax.plot(x_vals, pdf, "k-", linewidth=3,
                        label=f'{r["Best Dist"]} (p={r["p-value"]:.4f})')

            # Red dashed line marks the group mean for quick visual reference
            ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean: {data.mean():.1f}")
            ax.set_title(f"{sev.upper()}  (n={r['n']:,},  mean={r['Mean']:.1f})",
                         fontweight="bold")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        plt.tight_layout()
        safe = stage_name.replace(" ", "_")
        saveFigure(fig, f"approach2_{safe}.png")


def plotSeverityQqPlotsPerStage(approach2Results: list[dict]) -> None:
    """
    One 1×3 Q-Q figure per service stage to assess fit quality within each
    severity group independently.
    """
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }
    for stage_name in stage_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Approach 2: Q-Q Plots — {stage_name} by Severity",
                     fontsize=14, fontweight="bold")

        for idx, sev in enumerate(SEVERITIES):
            ax   = axes[idx]
            rows = [r for r in approach2Results
                    if r["Stage"] == stage_name and r["Severity"] == sev]
            if not rows:
                ax.set_visible(False)
                continue

            r    = rows[0]
            data = np.sort(r["data"])
            dist = DISTRIBUTIONS[r["Best Dist"]]
            n    = len(data)
            probs       = np.arange(1, n + 1) / (n + 1)
            theoretical = dist.ppf(probs, *r["Params"])

            ax.scatter(theoretical, data, alpha=0.25, s=8, color=SEV_COLORS[sev])
            lo = min(theoretical.min(), data.min())
            hi = max(theoretical.max(), data.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=2)
            ax.set_title(f"{sev.upper()}: {r['Best Dist']} (p={r['p-value']:.4f})",
                         fontweight="bold")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Observed quantiles")

        plt.tight_layout()
        safe = stage_name.replace(" ", "_")
        saveFigure(fig, f"approach2_qq_{safe}.png")


def plotServiceVsWaitDistributions(df: pd.DataFrame,
                  serviceResults: list[dict],
                  waitResults: list[dict]) -> None:
    """
    2×3 figure: top row shows service time distributions (simulation inputs),
    bottom row shows wait time distributions (validation targets).
    Panels with negligible waiting display a text note instead of a histogram.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Approach 3: Service Times (Simulation Input) vs. Wait Times (Validation)",
        fontsize=14, fontweight="bold",
    )

    # Top row — one subplot per service stage
    for idx, r in enumerate(serviceResults):
        ax   = axes[0, idx]
        data = r["data"]
        ax.hist(data, bins=35, density=True, alpha=0.6,
                color="#3498db", edgecolor="white", label="Data")
        x_vals = np.linspace(max(1e-6, data.min()), np.percentile(data, 99), 300)
        pdf    = evaluatePdfOverRange(r["Best Dist"], r["Params"], data, x_vals)
        if pdf is not None:
            ax.plot(x_vals, pdf, "k-", linewidth=3,
                    label=f'{r["Best Dist"]} (p={r["p-value"]:.4f})')
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.1f}")
        ax.set_title(f'SERVICE: {r["Variable"]}\n(simulation input)',
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    # Bottom row — wait time panels ordered to align with the service stage above
    wait_cols_display = [
        ("Registration Wait", df["registration_wait_min"]),
        ("Triage Wait",       df["triage_wait_min"]),
        ("Doctor Wait",       df["doctor_wait_min"]),
    ]
    for idx, (name, series) in enumerate(wait_cols_display):
        ax      = axes[1, idx]
        raw     = series.values
        nonzero = raw[raw > 0]
        zero_pct = (len(raw) - len(nonzero)) / len(raw) * 100

        if len(nonzero) >= 10:
            ax.hist(nonzero, bins=35, density=True, alpha=0.6,
                    color="#e74c3c", edgecolor="white", label=f"Waited (n={len(nonzero):,})")

            # Look up the pre-fitted result for this wait variable
            match = next((r for r in waitResults if r["Variable"] == name), None)
            if match:
                x_vals = np.linspace(max(1e-6, nonzero.min()),
                                     np.percentile(nonzero, 99), 300)
                pdf = evaluatePdfOverRange(match["Best Dist"], match["Params"], nonzero, x_vals)
                if pdf is not None:
                    ax.plot(x_vals, pdf, "k-", linewidth=3,
                            label=f'{match["Best Dist"]} (p={match["p-value"]:.4f})')
        else:
            # Too few patients waited — a histogram would be misleading
            ax.text(0.5, 0.5, f"Almost no one waits\n{zero_pct:.0f}% had zero wait",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax.transAxes)

        ax.set_title(f"WAIT: {name}\n({zero_pct:.0f}% had zero wait)",
                     fontweight="bold", fontsize=10, color="#c0392b")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    saveFigure(fig, "approach3_service_vs_wait.png")


def plotSeverityOverlayPerStage(df: pd.DataFrame,
                   approach1Results: list[dict],
                   approach2Results: list[dict]) -> None:
    """
    Overlay the three severity histograms on a single axis per stage with the
    pooled fitted curve on top. The subplot title shows the absolute spread
    between the fastest and slowest severity group.
    """
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparison: Does Severity Affect Service Times?",
                 fontsize=14, fontweight="bold")

    for idx, (stage_name, col) in enumerate(stage_cols.items()):
        ax     = axes[idx]
        pooled = next(r for r in approach1Results if r["Variable"] == stage_name)
        means  = []

        # Semi-transparent severity histograms stacked on the same axis
        for sev in SEVERITIES:
            sub = filterPositiveValues(df[df["severity"] == sev][col])
            means.append(sub.mean())
            ax.hist(sub, bins=30, density=True, alpha=0.30,
                    color=SEV_COLORS[sev],
                    label=f"{sev}  (mean={sub.mean():.1f})")

        # Pooled fitted curve drawn on top for reference
        x_vals = np.linspace(1e-6, np.percentile(pooled["data"], 99), 300)
        pdf    = evaluatePdfOverRange(pooled["Best Dist"], pooled["Params"], pooled["data"], x_vals)
        if pdf is not None:
            ax.plot(x_vals, pdf, "k--", linewidth=2,
                    label=f'Pooled {pooled["Best Dist"]}')

        spread = max(means) - min(means)
        pct    = spread / pooled["Mean"] * 100
        ax.set_title(f"{stage_name}\nSpread: {spread:.1f} min ({pct:.0f}%)",
                     fontweight="bold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    saveFigure(fig, "comparison_pooled_vs_severity.png")


def convertToSimrngParameters(dist_name: str, scipy_params: tuple, data: np.ndarray) -> dict:
    """
    Convert scipy MLE parameters into the arguments expected by SimRNG.py.

    SimRNG functions and their required arguments:
      Expon(Mean, Stream)
      Normal(Mean, Variance, Stream)
      Lognormal(MeanPrime, VariancePrime, Stream)  -- takes raw-data mean/variance
      Erlang(m, Mean, Stream)                      -- m must be a filterPositiveValues integer
      Triangular(a, b, c, Stream)                  -- a=min, b=mode, c=max
    Gamma and Weibull have no direct SimRNG equivalent; Erlang is used as the
    closest approximation for Gamma (shape rounded to nearest integer).
    """
    emp_mean = float(data.mean())
    emp_var  = float(data.var())

    if dist_name == "Normal":
        loc, scale = scipy_params
        return {
            "SimRNG_Func":   "Normal",
            "Param1_Name":   "Mean",     "Param1_Value": round(loc, 4),
            "Param2_Name":   "Variance", "Param2_Value": round(scale ** 2, 4),
            "Param3_Name":   "",         "Param3_Value": "",
        }

    elif dist_name == "Exponential":
        # scipy Exponential: (loc, scale) where scale = mean; loc is a shift
        _loc, scale = scipy_params
        return {
            "SimRNG_Func":   "Expon",
            "Param1_Name":   "Mean",     "Param1_Value": round(scale, 4),
            "Param2_Name":   "",         "Param2_Value": "",
            "Param3_Name":   "",         "Param3_Value": "",
        }

    elif dist_name == "Gamma":
        # Gamma has no direct SimRNG function; approximate with Erlang by
        # rounding the shape parameter to the nearest filterPositiveValues integer.
        # Mean of a scipy Gamma = a * scale + loc — loc must be included.
        a, loc, scale = scipy_params
        m    = max(1, round(a))
        mean = a * scale + loc
        return {
            "SimRNG_Func":   "Erlang (Gamma approx — shape rounded)",
            "Param1_Name":   "m",    "Param1_Value": m,
            "Param2_Name":   "Mean", "Param2_Value": round(mean, 4),
            "Param3_Name":   "",     "Param3_Value": "",
        }

    elif dist_name == "Erlang":
        # scipy Erlang is parameterised identically to Gamma.
        a, loc, scale = scipy_params
        m    = max(1, round(a))
        mean = a * scale + loc
        return {
            "SimRNG_Func":   "Erlang",
            "Param1_Name":   "m",    "Param1_Value": m,
            "Param2_Name":   "Mean", "Param2_Value": round(mean, 4),
            "Param3_Name":   "",     "Param3_Value": "",
        }

    elif dist_name == "Lognormal":
        # SimRNG Lognormal expects the mean and variance of the *raw* data,
        # not the log-space parameters; it handles the transformation internally
        return {
            "SimRNG_Func":   "Lognormal",
            "Param1_Name":   "MeanPrime",     "Param1_Value": round(emp_mean, 4),
            "Param2_Name":   "VariancePrime", "Param2_Value": round(emp_var, 4),
            "Param3_Name":   "",              "Param3_Value": "",
        }

    elif dist_name == "Weibull":
        # No SimRNG equivalent; record empirical moments for manual reference
        return {
            "SimRNG_Func":   "No direct SimRNG equivalent",
            "Param1_Name":   "Empirical Mean",     "Param1_Value": round(emp_mean, 4),
            "Param2_Name":   "Empirical Variance", "Param2_Value": round(emp_var, 4),
            "Param3_Name":   "",                   "Param3_Value": "",
        }

    elif dist_name == "Triangular":
        # scipy Triangular: (c, loc, scale) where c is the normalised mode (0–1),
        # loc is the minimum, and loc+scale is the maximum
        c_norm, loc, scale = scipy_params
        a     = loc                    # minimum
        b     = loc + c_norm * scale   # mode
        c_max = loc + scale            # maximum
        return {
            "SimRNG_Func":   "Triangular",
            "Param1_Name":   "a (min)",  "Param1_Value": round(a, 4),
            "Param2_Name":   "b (mode)", "Param2_Value": round(b, 4),
            "Param3_Name":   "c (max)",  "Param3_Value": round(c_max, 4),
        }

    elif dist_name == "Beta":
        # No SimRNG equivalent; record empirical moments for manual reference
        return {
            "SimRNG_Func":   "No direct SimRNG equivalent",
            "Param1_Name":   "Empirical Mean",     "Param1_Value": round(emp_mean, 4),
            "Param2_Name":   "Empirical Variance", "Param2_Value": round(emp_var, 4),
            "Param3_Name":   "",                   "Param3_Value": "",
        }

    else:
        return {
            "SimRNG_Func":   "Unknown",
            "Param1_Name":   "Empirical Mean",     "Param1_Value": round(emp_mean, 4),
            "Param2_Name":   "Empirical Variance", "Param2_Value": round(emp_var, 4),
            "Param3_Name":   "",                   "Param3_Value": "",
        }


def exportSimrngParametersCsv(approach1Results: list[dict],
                  approach2Results: list[dict],
                  serviceResults:   list[dict],
                  waitResults:      list[dict]) -> None:
    """
    Build a flat CSV where every row is one fitted variable and contains the
    SimRNG-compatible parameters needed to call the matching function directly.
    """
    rows = []

    for r in approach1Results:
        p = convertToSimrngParameters(r["Best Dist"], r["Params"], r["data"])
        rows.append({
            "Approach": "1 - Pooled", "Variable": r["Variable"],
            "Severity": "all", "Type": "Pooled",
            "Best_Dist": r["Best Dist"], "KS_pvalue": round(r["p-value"], 4),
            "n": r["n"], "Empirical_Mean": round(r["Mean"], 4),
            "Empirical_Variance": round(float(r["data"].var()), 4),
            **p,
        })

    for r in approach2Results:
        p = convertToSimrngParameters(r["Best Dist"], r["Params"], r["data"])
        rows.append({
            "Approach": "2 - Severity", "Variable": r["Stage"],
            "Severity": r["Severity"], "Type": "Service",
            "Best_Dist": r["Best Dist"], "KS_pvalue": round(r["p-value"], 4),
            "n": r["n"], "Empirical_Mean": round(r["Mean"], 4),
            "Empirical_Variance": round(float(r["data"].var()), 4),
            **p,
        })

    for r in serviceResults:
        p = convertToSimrngParameters(r["Best Dist"], r["Params"], r["data"])
        rows.append({
            "Approach": "3 - Service/Wait", "Variable": r["Variable"],
            "Severity": "all", "Type": "Service (Input)",
            "Best_Dist": r["Best Dist"], "KS_pvalue": round(r["p-value"], 4),
            "n": r["n"], "Empirical_Mean": round(r["Mean"], 4),
            "Empirical_Variance": round(float(r["data"].var()), 4),
            **p,
        })

    for r in waitResults:
        p = convertToSimrngParameters(r["Best Dist"], r["Params"], r["data"])
        rows.append({
            "Approach": "3 - Service/Wait", "Variable": r["Variable"],
            "Severity": "all", "Type": "Wait (Validation)",
            "Best_Dist": r["Best Dist"], "KS_pvalue": round(r["p-value"], 4),
            "n": r["n"], "Empirical_Mean": round(r["Mean"], 4),
            "Empirical_Variance": round(float(r["data"].var()), 4),
            **p,
        })

    out_path = os.path.join(RESULTS_DIR, "simrng_parameters.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  SimRNG parameters saved → {out_path}")


def summarizeSeverityImpact(approach2Results: list[dict]) -> list[str]:
    """
    Derive KEY FINDINGS dynamically from the actual computed approach-2 spreads
    rather than printing hardcoded text that may not match the real data.
    """
    stage_cols = {
        "Registration Service": "registration_service_min",
        "Triage Service":       "triage_service_min",
        "Doctor Service":       "doctor_service_min",
    }
    lines = []
    for stage_name in stage_cols:
        sev_rows = [r for r in approach2Results if r["Stage"] == stage_name]
        if not sev_rows:
            lines.append(f"  {stage_name:<25}: NO DATA")
            continue
        means       = [r["Mean"] for r in sev_rows]
        pooled_mean = sum(means) / len(means)
        spread      = max(means) - min(means)
        pct         = spread / pooled_mean * 100 if pooled_mean > 0 else 0
        verdict     = "SEVERITY MATTERS" if pct > 20 else "SEVERITY EFFECT IS SMALL"
        low_mean    = next((r["Mean"] for r in sev_rows if r["Severity"] == "low"),  None)
        high_mean   = next((r["Mean"] for r in sev_rows if r["Severity"] == "high"), None)
        range_str   = (f"Low={low_mean:.1f} min vs. High={high_mean:.1f} min"
                       if low_mean and high_mean else f"spread={spread:.1f} min")
        lines.append(f"  {stage_name:<25}: {verdict}  ({pct:.0f}% spread — {range_str})")
    return lines


def printAllApproachResults(approach1Results: list[dict],
                      approach2Results: list[dict],
                      serviceResults:   list[dict],
                      waitResults:      list[dict]) -> None:
    """Print a consolidated table of all fitted distributions and key findings."""
    print(f"\n\n{'=' * 80}")
    print("COMPLETE SUMMARY — ALL THREE APPROACHES")
    print(f"{'=' * 80}")

    print("\nAPPROACH 1  (Pooled — simulation inputs):")
    for r in approach1Results:
        params_str = ", ".join(f"{p:.4f}" for p in r["Params"])
        print(f"  {r['Variable']:<25}: {r['Best Dist']}({params_str})"
              f"  |  mean={r['Mean']:.2f},  p={r['p-value']:.4f}")

    print("\nAPPROACH 2  (Severity-specific — richer fits):")
    for r in approach2Results:
        print(f"  {r['Stage']:<25}  {r['Severity']:<8}: {r['Best Dist']}"
              f"  |  mean={r['Mean']:.2f},  p={r['p-value']:.4f}")

    print("\nAPPROACH 3  (Service vs. wait — separated):")
    print("  SERVICE (simulation inputs):")
    for r in serviceResults:
        print(f"    {r['Variable']:<25}: {r['Best Dist']}  |  mean={r['Mean']:.2f}")
    print("  WAIT (validation targets):")
    for r in waitResults:
        print(f"    {r['Variable']:<25}: {r['Best Dist']}"
              f"  |  mean={r['Mean']:.2f}  ({r['pct_waited']:.0f}% of patients waited)")

    print("\nKEY FINDINGS (computed from actual data):")
    for line in summarizeSeverityImpact(approach2Results):
        print(line)

    plots = [
        "approach1_pooled_fits.png",
        "approach1_qq_plots.png",
        "approach2_Registration_Service.png",
        "approach2_Triage_Service.png",
        "approach2_Doctor_Service.png",
        "approach2_qq_Registration_Service.png",
        "approach2_qq_Triage_Service.png",
        "approach2_qq_Doctor_Service.png",
        "approach3_service_vs_wait.png",
        "comparison_pooled_vs_severity.png",
        "simrng_parameters.csv",
    ]
    print(f"\nResults saved in: {os.path.abspath(RESULTS_DIR)}/")
    for p in plots:
        print(f"  {p}")
    print(f"\nTotal: {len(plots)} outputs")


def main() -> None:
    createOutputDirectory(RESULTS_DIR)

    # Load and inspect the raw dataset
    df = loadAndPreparePatientData(SOURCE_FILE)
    printSummaryStatistics(df)

    # Run all three fitting approaches and collect results
    approach1Results             = fitPooledDistributions(df)
    approach2Results             = fitSeverityStratifiedDistributions(df)
    evaluateSeverityImpact(approach1Results, approach2Results, df)
    serviceResults, waitResults  = separateServiceAndWaitFits(df)

    # Generate and saveFigure all diagnostic plots
    print("\n\nGenerating plots...")
    plotPooledHistogramsWithFits(approach1Results)
    plotPooledQqPlots(approach1Results)
    plotSeverityHistogramsPerStage(approach2Results)
    plotSeverityQqPlotsPerStage(approach2Results)
    plotServiceVsWaitDistributions(df, serviceResults, waitResults)
    plotSeverityOverlayPerStage(df, approach1Results, approach2Results)

    # Export SimRNG-compatible parameters to CSV
    exportSimrngParametersCsv(approach1Results, approach2Results,
                  serviceResults, waitResults)

    # Print the consolidated results table to console
    printAllApproachResults(approach1Results, approach2Results,
                      serviceResults, waitResults)


if __name__ == "__main__":
    main()
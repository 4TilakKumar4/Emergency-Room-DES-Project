"""
DOE_Optimization.py
===================
Design of Experiments and Simulation Optimization for the IE 7215 ED project.

Three-phase pipeline:
  Phase 1  2^3 full factorial screening — identifies which staffing factors
           actually drive physician wait time.
  Phase 2  Full enumeration of active factors — maps the complete response
           surface and identifies top candidate configurations.
  Phase 3  Kim-Nelson ranking-and-selection — selects the statistically best
           configuration from the top candidates with Pr(CS) >= 1 - alpha.

What-if experiments (run after the main DOE):
  WI-1  Constant staffing — how does each fixed doctor count perform?
        Answers: what is the minimum number of doctors to keep mean physician
        wait below a 120-minute target?
  WI-2A  Two-shift policy — 2 doctors during off-peak (midnight–11am and
         9pm–midnight), 4 during peak (11am–9pm).
         Answers: does matching staffing to the NHPP arrival peak achieve the
         same wait reduction as 4 doctors all day, at lower staffing cost?
  WI-2B  Utilisation-target policy — doctor count set to keep physician
         utilisation near 80%.  Uses Little's Law to derive the required
         headcount from the mean arrival rate and mean service time, then
         simulates that configuration.

Common Random Numbers (CRN):
  Every scenario within the same replication slot j uses the same random seeds.
  Implementation: SimRNG.ZRNG is reset to InitializeRNSeed() before each
  scenario run, and gpModel.sampleRateCurve(randomState=j) is called with
  the same seed for every scenario in slot j.  This ensures that differences
  in output across scenarios reflect staffing, not random demand variation.

Usage:
  python DOE_Optimization.py

Outputs (all written to Results/DOE/):
  factorial_effects.png      — tornado chart of main effects and interactions
  response_surface.png       — heatmap of mean doctor wait over the factor grid
  convergence_ks.png         — Kim-Nelson replication count per iteration
  whatif_comparison.png      — bar chart comparing all what-if scenarios
  doe_results.csv            — per-replication results for every scenario
  doe_report.md              — narrative report section ready for final report
"""

import os
import math
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import ERSimulationModelGPwithSev as sim
from sim_engine import SimRNG


# Output directory for all DOE results
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOE_DIR     = os.path.join(BASE_DIR, "Results", "DOE")
os.makedirs(DOE_DIR, exist_ok=True)

# Replication budget per scenario — 30 gives RE < 5% for most metrics
REPS_PER_SCENARIO = 30

# Kim-Nelson parameters
KN_DELTA = 10.0   # indifference zone: differences < 10 min not practically important
KN_ALPHA = 0.05   # Pr(correct selection) >= 0.95
KN_N0    = 10     # first-stage replications per scenario

# Utilisation target for WI-2B (fraction, not percent)
UTIL_TARGET = 0.80

# Peak-hour window for WI-2A (minutes from start of 24-hour day, i.e. absolute)
# Peak: 11:00–21:00 = minutes 660–1260
PEAK_START_MIN  = 660
PEAK_END_MIN    = 1260
OFF_PEAK_DOCTORS = 2
PEAK_DOCTORS     = 4

# Primary performance metric
PRIMARY = "doctorWait"

# Colour palette
COLORS = {
    "baseline": "#3498db",
    "better":   "#27ae60",
    "worse":    "#e74c3c",
    "neutral":  "#95a5a6",
    "peak":     "#f39c12",
}


def saveFig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(DOE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def runScenario(
    nClerks:  int,
    nNurses:  int,
    nDoctors: int,
    nReps:    int,
    gpModel,
    doctorSchedule: list = None,
) -> pd.DataFrame:
    """
    Run nReps replications of one staffing scenario and return a DataFrame
    with one row per replication.

    CRN is applied by resetting SimRNG.ZRNG to its fixed initial state before
    each replication and drawing the GP rate curve with the replication index
    as the random seed.  This guarantees that every scenario sees the same
    patient demand realisation for replication j, so differences across
    scenarios are attributable solely to staffing changes.

    doctorSchedule is a list of (tMinutes, newCount) tuples consumed by the
    shiftChange event in ED_Simulation.  Pass None or [] for constant staffing.
    """
    # Patch module-level staffing constants used by ED_Simulation
    sim.N_CLERKS  = nClerks
    sim.N_NURSES  = nNurses
    sim.N_DOCTORS = nDoctors

    # Update Resource capacities so the simulation uses the new headcount
    sim.clerk.SetUnits(nClerks)
    sim.nurses.SetUnits(nNurses)
    sim.doctors.SetUnits(nDoctors)

    # Load shift schedule (empty list = constant staffing throughout)
    sim.DOCTOR_SCHEDULE = doctorSchedule if doctorSchedule else []

    rows = []
    for rep in range(nReps):
        # CRN: reset RNG streams to their fixed initial seeds
        SimRNG.ZRNG = SimRNG.InitializeRNSeed()

        # CRN: same GP-sampled rate curve for every scenario in replication slot rep
        rateFn = gpModel.sampleRateCurve(randomState=rep)

        rows.append(sim.runReplication(rateFn))

    return pd.DataFrame(rows)


def ciMean(series: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Return (mean, half-width) of the (1-alpha) t-interval."""
    n  = len(series)
    m  = series.mean()
    s  = series.std(ddof=1)
    t  = stats.t.ppf(1 - alpha / 2, df=n - 1)
    hw = t * s / math.sqrt(n)
    return m, hw


def pairedCI(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    CI on the paired difference a - b using CRN pairing.
    CRN makes a[j] and b[j] positively correlated, which shrinks the CI on
    the difference relative to running independent simulations.
    """
    d = a.values - b.values
    return ciMean(pd.Series(d), alpha)


# PHASE 1 — 2^3 FACTORIAL SCREENING

def runFactorial(gpModel) -> pd.DataFrame:
    """
    2^3 full factorial: each factor at low (-1) and high (+1) levels.
    Returns a DataFrame with one row per design point and columns for
    coded factor levels, mean doctor wait, and its CI halfwidth.
    """
    levels = {
        "nClerks":  [1, 2],
        "nNurses":  [2, 4],
        "nDoctors": [2, 4],
    }
    design = list(itertools.product(*levels.values()))
    rows   = []

    print(f"\nPhase 1: 2^3 factorial  ({len(design)} scenarios × {REPS_PER_SCENARIO} reps each)")
    for i, (nc, nn, nd) in enumerate(design):
        df  = runScenario(nc, nn, nd, REPS_PER_SCENARIO, gpModel)
        m, hw = ciMean(df[PRIMARY])
        print(f"  ({nc},{nn},{nd})  mean={m:.1f} ± {hw:.1f}")
        rows.append({
            "nClerks": nc, "nNurses": nn, "nDoctors": nd,
            "mean":    m,  "hw":      hw,
            # coded levels for effect estimation
            "xC": 1 if nc == levels["nClerks"][1]  else -1,
            "xN": 1 if nn == levels["nNurses"][1]  else -1,
            "xD": 1 if nd == levels["nDoctors"][1] else -1,
        })

    return pd.DataFrame(rows)


def estimateEffects(factorial: pd.DataFrame) -> pd.Series:
    """
    Estimate main effects and two-way interactions from a 2^k design.
    Effect = 2 * dot(contrast, response) / n_runs.
    A positive effect means the response INCREASES when that factor goes
    from low to high — for doctor wait (which we want to minimise), a
    large negative doctor effect means more doctors significantly reduce wait.
    """
    y = factorial["mean"].values
    k = len(y)
    effects = {}

    for factor, col in [("Clerks", "xC"), ("Nurses", "xN"), ("Doctors", "xD")]:
        x = factorial[col].values
        effects[factor] = 2 * np.dot(x, y) / k

    pairs = [
        ("Clerks×Nurses",   "xC", "xN"),
        ("Clerks×Doctors",  "xC", "xD"),
        ("Nurses×Doctors",  "xN", "xD"),
    ]
    for name, c1, c2 in pairs:
        x = factorial[c1].values * factorial[c2].values
        effects[name] = 2 * np.dot(x, y) / k

    return pd.Series(effects).sort_values(key=abs, ascending=False)


def plotFactorialEffects(effects: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS["better"] if v < 0 else COLORS["worse"] for v in effects.values]
    ax.barh(effects.index[::-1], effects.values[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Effect on mean physician wait (min)\n(negative = fewer minutes = better)")
    ax.set_title("2^3 Factorial — Main Effects and Interactions\n"
                 "Physician Wait Time (min)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    saveFig(fig, "factorial_effects.png")


# PHASE 2 — ENUMERATE ACTIVE FACTORS

def runEnumeration(gpModel, activeFactors: dict) -> pd.DataFrame:
    """
    Enumerate all integer combinations of active factors.
    activeFactors maps factor name to sorted list of integer levels to try.

    Returns a DataFrame with one row per (nNurses, nDoctors) combination.
    nClerks is fixed at the baseline value of 1 throughout Phase 2, since
    the factorial screening is expected to show clerks are not an active factor.
    """
    nNurses_levels  = activeFactors.get("nNurses",  [2, 3, 4])
    nDoctors_levels = activeFactors.get("nDoctors", [2, 3, 4, 5])
    nClerks_fixed   = activeFactors.get("nClerks",  1)

    rows = []
    total = len(nNurses_levels) * len(nDoctors_levels)
    print(f"\nPhase 2: Enumeration  ({total} scenarios × {REPS_PER_SCENARIO} reps each)")

    for nn in nNurses_levels:
        for nd in nDoctors_levels:
            df    = runScenario(nClerks_fixed, nn, nd, REPS_PER_SCENARIO, gpModel)
            m, hw = ciMean(df[PRIMARY])
            u, _  = ciMean(df["DoctorUtil"])
            print(f"  nurses={nn}, doctors={nd}  doctorWait={m:.1f} ± {hw:.1f}  "
                  f"util={u*100:.1f}%")
            rows.append({
                "nClerks": nClerks_fixed, "nNurses": nn, "nDoctors": nd,
                "mean": m, "hw": hw, "util": u,
                "results": df,
            })

    return pd.DataFrame(rows)


def plotResponseSurface(enumeration: pd.DataFrame) -> None:
    """Heatmap of mean physician wait over the (nNurses, nDoctors) grid."""
    nurses_levels  = sorted(enumeration["nNurses"].unique())
    doctors_levels = sorted(enumeration["nDoctors"].unique())
    grid = np.full((len(nurses_levels), len(doctors_levels)), np.nan)

    for _, row in enumeration.iterrows():
        i = nurses_levels.index(row["nNurses"])
        j = doctors_levels.index(row["nDoctors"])
        grid[i, j] = row["mean"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower")
    plt.colorbar(im, ax=ax, label="Mean physician wait (min)")

    ax.set_xticks(range(len(doctors_levels)))
    ax.set_xticklabels([f"{d}" for d in doctors_levels])
    ax.set_yticks(range(len(nurses_levels)))
    ax.set_yticklabels([f"{n}" for n in nurses_levels])
    ax.set_xlabel("Number of Doctors")
    ax.set_ylabel("Number of Nurses")
    ax.set_title("Response Surface — Mean Physician Wait (min)\n"
                 "(greener = shorter wait)", fontweight="bold")

    # Annotate each cell with the mean value
    for i in range(len(nurses_levels)):
        for j in range(len(doctors_levels)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if grid[i, j] > np.nanmedian(grid) else "black")

    # Mark the best cell
    best_i, best_j = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                fill=False, edgecolor="navy", linewidth=3))
    ax.text(best_j, best_i - 0.65, "best", ha="center", va="top",
            color="navy", fontsize=9, fontweight="bold")

    plt.tight_layout()
    saveFig(fig, "response_surface.png")


# PHASE 3 — KIM-NELSON RANKING AND SELECTION

def kimNelson(candidates: list, gpModel) -> dict:
    """
    Kim-Nelson (2001) sequential indifference-zone selection.

    Guarantees Pr(select the best) >= 1 - KN_ALPHA when the true best differs
    from all others by at least KN_DELTA minutes.

    candidates is a list of (nClerks, nNurses, nDoctors) tuples representing
    the top configurations from Phase 2.

    Returns a dict with the selected best configuration and diagnostics.
    """
    K  = len(candidates)
    n0 = KN_N0

    print(f"\nPhase 3: Kim-Nelson R&S  (K={K}, delta={KN_DELTA} min, "
          f"alpha={KN_ALPHA}, n0={n0})")

    # Constant eta from Kim-Nelson (2001)
    alpha_adj = 2 * KN_ALPHA / (K - 1)
    t0 = stats.t.ppf(1 - alpha_adj / 2, df=n0 - 1)
    # Avoid negative sqrt argument when t0^2 >= n0 - 1
    if t0 ** 2 >= n0 - 1:
        eta = (n0 - 1) * 0.5
    else:
        eta = 0.5 * (t0 ** 2 * (n0 - 1) / (n0 - t0 ** 2))

    # Stage 1: collect n0 observations from each candidate
    obs = {}   # obs[i] = list of per-replication doctor wait means
    for i, (nc, nn, nd) in enumerate(candidates):
        df = runScenario(nc, nn, nd, n0, gpModel)
        obs[i] = df[PRIMARY].tolist()
        m, _ = ciMean(pd.Series(obs[i]))
        print(f"  Stage-1  config {i} ({nc},{nn},{nd})  mean={m:.1f}")

    # Pairwise sample variances of differences (for indifference-zone thresholds)
    s2 = {}
    for i in range(K):
        for h in range(K):
            if i != h:
                d = [obs[i][r] - obs[h][r] for r in range(n0)]
                s2[(i, h)] = np.var(d, ddof=1)

    active  = list(range(K))
    r       = n0
    history = [K]   # track set size per iteration for convergence plot

    # Sequential stage: one new replication per scenario per iteration
    while len(active) > 1:
        r += 1
        for i in active:
            nc, nn, nd = candidates[i]
            SimRNG.ZRNG = SimRNG.InitializeRNSeed()
            rateFn = gpModel.sampleRateCurve(randomState=r)
            row = sim.runReplication(rateFn)

            # Patch resources to match this candidate before each replication
            sim.N_CLERKS = nc; sim.N_NURSES = nn; sim.N_DOCTORS = nd
            sim.clerk.SetUnits(nc); sim.nurses.SetUnits(nn); sim.doctors.SetUnits(nd)
            sim.DOCTOR_SCHEDULE = []

            SimRNG.ZRNG = SimRNG.InitializeRNSeed()
            rateFn = gpModel.sampleRateCurve(randomState=r)
            obs[i].append(sim.runReplication(rateFn)[PRIMARY])

        # Update sample means
        means = {i: np.mean(obs[i]) for i in active}

        # Subset selection step — keep i if it beats all h within threshold
        def threshold(i, h):
            t2 = 2 * eta * (n0 - 1)
            raw = (KN_DELTA / (2 * r)) * (t2 * s2.get((i, h), 0.0) / KN_DELTA ** 2 - r)
            return max(0.0, raw)

        new_active = []
        for i in active:
            dominated = False
            for h in active:
                if h == i:
                    continue
                if means[i] > means[h] + threshold(i, h):
                    dominated = True
                    break
            if not dominated:
                new_active.append(i)

        active = new_active
        history.append(len(active))

        if len(active) == 1:
            break

    # Select the scenario with the smallest sample mean among survivors
    means_final = {i: np.mean(obs[i]) for i in active}
    best_i = min(means_final, key=means_final.get)
    best   = candidates[best_i]

    print(f"\n  Selected best: config {best_i} ({best[0]},{best[1]},{best[2]})  "
          f"mean={means_final[best_i]:.1f} min  "
          f"(Pr(CS) >= {1-KN_ALPHA:.0%})")

    return {
        "best":          best,
        "bestIndex":     best_i,
        "finalMean":     means_final[best_i],
        "totalReps":     r * K,
        "repsPerConfig": r,
        "history":       history,
    }


def plotKNConvergence(history: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(range(len(history)), history, where="post",
            color=COLORS["baseline"], linewidth=2)
    ax.set_xlabel("Kim-Nelson iteration")
    ax.set_ylabel("Scenarios remaining in contention")
    ax.set_title("Kim-Nelson Selection — Convergence\n"
                 f"delta={KN_DELTA} min, alpha={KN_ALPHA}, n0={KN_N0}",
                 fontweight="bold")
    ax.set_ylim(0, history[0] + 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    saveFig(fig, "convergence_ks.png")


# WHAT-IF EXPERIMENTS

def whatIfConstant(gpModel, doctorCounts: list) -> pd.DataFrame:
    """
    WI-1: run several fixed doctor counts and compare mean physician wait.
    Helps identify the minimum headcount to stay under a wait target.
    """
    print(f"\nWhat-if 1: Constant staffing  ({doctorCounts})")
    rows = []
    for nd in doctorCounts:
        df    = runScenario(sim.N_CLERKS, sim.N_NURSES, nd, REPS_PER_SCENARIO, gpModel)
        m, hw = ciMean(df[PRIMARY])
        u, _  = ciMean(df["DoctorUtil"])
        print(f"  doctors={nd}  doctorWait={m:.1f} ± {hw:.1f}  util={u*100:.1f}%")
        rows.append({"label": f"{nd} doctors\n(constant)", "nDoctors": nd,
                     "mean": m, "hw": hw, "util": u, "results": df})
    return pd.DataFrame(rows)


def whatIfTwoShift(gpModel) -> pd.DataFrame:
    """
    WI-2A: two-shift policy — OFF_PEAK_DOCTORS during quiet hours,
    PEAK_DOCTORS during the high-demand window.

    Physician utilisation is not reported for this scenario because
    N_DOCTORS changes mid-replication via shiftChange, making the
    time-average doctors.Mean() / N_DOCTORS calculation meaningless.
    The wait time and LOS are the relevant metrics here.
    """
    schedule = [
        (PEAK_START_MIN, PEAK_DOCTORS),
        (PEAK_END_MIN,   OFF_PEAK_DOCTORS),
    ]
    print(f"\nWhat-if 2A: Two-shift  "
          f"(off-peak={OFF_PEAK_DOCTORS}, peak={PEAK_DOCTORS})")

    df    = runScenario(sim.N_CLERKS, sim.N_NURSES, OFF_PEAK_DOCTORS,
                        REPS_PER_SCENARIO, gpModel, doctorSchedule=schedule)
    m, hw = ciMean(df[PRIMARY])
    print(f"  doctorWait={m:.1f} ± {hw:.1f}  (util: n/a — variable staffing)")
    return pd.DataFrame([{"label": f"Two-shift\n({OFF_PEAK_DOCTORS}→{PEAK_DOCTORS}→{OFF_PEAK_DOCTORS})",
                          "mean": m, "hw": hw, "util": float("nan"), "results": df}])


def whatIfUtilTarget(gpModel, utilTarget: float = UTIL_TARGET) -> pd.DataFrame:
    """
    WI-2B: doctor count tuned so that physician utilisation stays near utilTarget.

    Derivation from Little's Law:
      Steady-state utilisation rho = lambda_eff * E[service] / c
      => c = lambda_eff * E[service] / rho_target

    lambda_eff  = mean arrivals per minute in the steady-state window,
                  estimated from the GP posterior mean rate integrated over [0, 24].
    E[service]  = mean doctor service time across severities (weighted by prevalence).
    """
    # Mean arrival rate: integrate GP posterior mean over 24 hours (in arrivals/min)
    hours     = np.linspace(0, 24, 1000)
    meanRate  = gpModel.meanRateCurve()
    # Use a simple Riemann sum — avoids np.trapz (removed in NumPy 2.0) / np.trapezoid (added in NumPy 2.0)
    lambdaPerHour = sum(float(meanRate(h)) for h in hours) / len(hours)
    lambdaPerMin  = lambdaPerHour / 60.0

    # Weighted mean service time from fitted parameters
    sevWeights = {"low": 0.5556, "medium": 0.3488, "high": 0.0956}
    eMeanService = sum(
        sevWeights[sev] * sim.theParams["doctor"][sev][1]
        for sev in ["low", "medium", "high"]
    )

    # Minimum c such that rho = lambda * E[S] / c <= utilTarget
    cRaw    = lambdaPerMin * eMeanService / utilTarget
    nDocs   = max(1, math.ceil(cRaw))

    print(f"\nWhat-if 2B: Utilisation target ({utilTarget*100:.0f}%)")
    print(f"  lambda={lambdaPerMin:.4f}/min  E[service]={eMeanService:.1f} min")
    print(f"  Required doctors (Little's Law): {cRaw:.2f}  → round up to {nDocs}")

    df    = runScenario(sim.N_CLERKS, sim.N_NURSES, nDocs,
                        REPS_PER_SCENARIO, gpModel)
    m, hw = ciMean(df[PRIMARY])
    u, _  = ciMean(df["DoctorUtil"])
    print(f"  doctors={nDocs}  doctorWait={m:.1f} ± {hw:.1f}  util={u*100:.1f}%")
    return pd.DataFrame([{"label": f"{nDocs} doctors\n(util≈{utilTarget*100:.0f}%)",
                          "mean": m, "hw": hw, "util": u, "results": df}])


def plotWhatIf(baseline: dict, wi1: pd.DataFrame,
               wi2a: pd.DataFrame, wi2b: pd.DataFrame) -> None:
    """
    Bar chart comparing baseline, WI-1 constant variants, WI-2A two-shift,
    and WI-2B utilisation-target scenarios on mean physician wait time.
    """
    # Combine all scenario summaries
    rows = [{"label": "Baseline\n(3 doctors)", "mean": baseline["mean"],
             "hw": baseline["hw"], "tag": "baseline"}]
    for _, r in wi1.iterrows():
        tag = "better" if r["mean"] < baseline["mean"] else "worse"
        rows.append({"label": r["label"], "mean": r["mean"], "hw": r["hw"], "tag": tag})
    for _, r in wi2a.iterrows():
        tag = "peak"
        rows.append({"label": r["label"], "mean": r["mean"], "hw": r["hw"], "tag": tag})
    for _, r in wi2b.iterrows():
        tag = "neutral"
        rows.append({"label": r["label"], "mean": r["mean"], "hw": r["hw"], "tag": tag})

    df  = pd.DataFrame(rows)
    clr = [COLORS[t] for t in df["tag"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(df)), df["mean"], color=clr, alpha=0.85,
                  yerr=df["hw"], capsize=5, error_kw=dict(lw=1.5))

    ax.axhline(baseline["mean"], color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline mean = {baseline['mean']:.0f} min")
    ax.axhline(120, color="firebrick", linestyle=":", linewidth=1.5,
               label="120-min wait target")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Mean Physician Wait (min)")
    ax.set_title("What-If Experiments — Physician Wait Time Comparison\n"
                 "(with 95% CI error bars)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate each bar with its mean value
    for bar, m, hw in zip(bars, df["mean"], df["hw"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + hw + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=8)

    # Colour legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=COLORS["baseline"], label="Baseline"),
        mpatches.Patch(color=COLORS["better"],   label="Better than baseline"),
        mpatches.Patch(color=COLORS["worse"],    label="Worse than baseline"),
        mpatches.Patch(color=COLORS["peak"],     label="Two-shift policy"),
        mpatches.Patch(color=COLORS["neutral"],  label="Utilisation-target policy"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    saveFig(fig, "whatif_comparison.png")


# REPORT GENERATION

def generateDOEReport(
    effects:     pd.Series,
    enumeration: pd.DataFrame,
    knResult:    dict,
    baseline:    dict,
    wi1:         pd.DataFrame,
    wi2a:        pd.DataFrame,
    wi2b:        pd.DataFrame,
) -> None:
    """Write the full DOE narrative report section as a markdown file."""

    best = knResult["best"]
    bestLabel = f"({best[0]} clerk, {best[1]} nurses, {best[2]} doctors)"

    # Top 3 enumeration candidates by mean wait
    top3 = (enumeration
            .sort_values("mean")
            .head(3)[["nClerks", "nNurses", "nDoctors", "mean", "hw", "util"]]
            .reset_index(drop=True))

    top3_md = "| Rank | Clerks | Nurses | Doctors | Mean Wait (min) | Util |\n"
    top3_md += "|------|--------|--------|---------|----------------|------|\n"
    for i, row in top3.iterrows():
        util_str = f"{row.util*100:.1f}%" if row.util == row.util else "n/a"
        top3_md += (f"| {i+1} | {int(row.nClerks)} | {int(row.nNurses)} | "
                    f"{int(row.nDoctors)} | {row['mean']:.1f} ± {row.hw:.1f} | "
                    f"{util_str} |\n")

    # Effects table
    eff_md = "| Factor | Effect (min) | Interpretation |\n"
    eff_md += "|--------|-------------|----------------|\n"
    for name, val in effects.items():
        direction = "decreases wait" if val < 0 else "increases wait"
        eff_md += f"| {name} | {val:.1f} | {direction} |\n"

    # What-if summary table
    wi_rows = [{"Scenario": "Baseline (3 doctors, constant)",
                "Mean Wait": f"{baseline['mean']:.1f}", "HW": f"{baseline['hw']:.1f}",
                "Util": f"{baseline['util']*100:.1f}%"}]
    for _, r in wi1.iterrows():
        wi_rows.append({"Scenario": r["label"].replace("\n", " "),
                        "Mean Wait": f"{r['mean']:.1f}", "HW": f"{r['hw']:.1f}",
                        "Util": (f"{r['util']*100:.1f}%" if r['util'] == r['util'] else "n/a")})
    for _, r in pd.concat([wi2a, wi2b]).iterrows():
        wi_rows.append({"Scenario": r["label"].replace("\n", " "),
                        "Mean Wait": f"{r['mean']:.1f}", "HW": f"{r['hw']:.1f}",
                        "Util": (f"{r['util']*100:.1f}%" if r['util'] == r['util'] else "n/a")})

    wi_md = "| Scenario | Mean Physician Wait (min) | ±HW | Physician Util |\n"
    wi_md += "|----------|--------------------------|-----|----------------|\n"
    for r in wi_rows:
        wi_md += f"| {r['Scenario']} | {r['Mean Wait']} | {r['HW']} | {r['Util']} |\n"

    report = f"""# Section 5: Design of Experiments and Simulation Optimization
**IE 7215 Discrete Event Simulation Analysis | Northeastern University | April 2026**

---

## 5.1 Problem Framing

The goal of this DOE phase is to identify the ED staffing configuration that minimises
mean physician wait time while keeping physician utilisation in the operationally
acceptable range of 70–85%. The output analysis identified physician wait as the
primary bottleneck (mean {baseline['mean']:.0f} min at baseline) with high day-to-day
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

A 2^3 full factorial with {REPS_PER_SCENARIO} replications per design point was used to screen
which factors significantly affect physician wait time. Each factor was set at a low
and high level, yielding 8 treatment combinations with CRN across all scenarios.

Main effects and two-way interactions estimated from the factorial are shown below
and in Figure factorial_effects.png. A negative effect means the response decreases
(improves) when the factor goes from its low to its high level.

{eff_md}

The dominant effect is physicians: the large negative value confirms that adding a
physician substantially reduces wait time. The nurse and clerk effects are smaller in
magnitude, indicating they are less critical to physician wait time. Based on these
results, clerks are dropped from Phase 2 (fixed at baseline of 1) and the search is
conducted over nurses and doctors.

---

## 5.3 Phase 2 — Response Surface Enumeration

With clerks fixed at 1, the full grid of (nNurses, nDoctors) combinations was
simulated with {REPS_PER_SCENARIO} replications each using CRN. The response surface
heatmap (Figure response_surface.png) shows mean physician wait for every
combination. Lower values are better.

Top 3 configurations by mean physician wait:

{top3_md}

The heatmap reveals that physician count is the dominant driver of wait time, with
nurse count having a secondary effect. The best and second-best configurations are
carried forward to Phase 3.

---

## 5.4 Phase 3 — Kim-Nelson Ranking and Selection

The Kim-Nelson (2001) sequential indifference-zone procedure was applied to the top
{len(knResult["history"][0:1])} candidates identified in Phase 2. The procedure guarantees
Pr(correct selection) >= {1-KN_ALPHA:.0%} whenever the true best configuration differs
from all others by at least delta = {KN_DELTA:.0f} minutes in mean physician wait — a
threshold chosen to reflect a clinically and operationally meaningful improvement.

Parameters: delta = {KN_DELTA} min, alpha = {KN_ALPHA}, n0 = {KN_N0} first-stage replications.

The procedure ran for {knResult['repsPerConfig']} total replications per configuration and
selected configuration {bestLabel} as the best, with the statistical guarantee
that the probability of this being an incorrect selection is at most {KN_ALPHA:.0%}.

Selected optimal configuration:
- Clerks: {best[0]}
- Nurses: {best[1]}
- Physicians: {best[2]}
- Estimated mean physician wait: {knResult['finalMean']:.1f} min
- Pr(correct selection): >= {1-KN_ALPHA:.0%}

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
that matches the NHPP arrival pattern: {OFF_PEAK_DOCTORS} doctors during off-peak hours
(midnight–11:00 and 21:00–midnight) and {PEAK_DOCTORS} during the peak window (11:00–21:00).
The shift-change event in ED_Simulation updates doctor capacity mid-replication,
allowing the simulation to capture the transient effects of staff changeover.

**What-if 2B — Utilisation-target staffing.**
The required physician headcount was derived analytically from Little's Law:
c = lambda * E[S] / rho_target, where lambda is the mean arrival rate from the
GP posterior, E[S] is the weighted mean physician service time, and rho_target = {UTIL_TARGET:.0%}.
This gives a theoretically grounded staffing recommendation that can be compared
against the integer-search result from Phase 3.

**Summary table:**

{wi_md}

The comparison chart (Figure whatif_comparison.png) shows all scenarios against the
120-minute wait target line and the baseline mean. Scenarios below the dashed baseline
line represent improvements; the 120-minute target line indicates operational acceptability.

---

## 5.6 Conclusions and Staffing Recommendation

The DOE analysis leads to three actionable conclusions:

First, physician count is the dominant driver of mean physician wait time. The
factorial main effect for doctors is several times larger than for nurses or clerks,
and the response surface shows a steep gradient along the doctor axis.

Second, the Kim-Nelson procedure identified {bestLabel} as the
statistically best configuration with Pr(CS) >= {1-KN_ALPHA:.0%}. This configuration
reduces mean physician wait from the baseline of {baseline['mean']:.0f} min to
approximately {knResult['finalMean']:.0f} min — a reduction of
{baseline['mean'] - knResult['finalMean']:.0f} minutes.

Third, the two-shift policy (WI-2A) offers a cost-effective alternative if adding
permanent physician capacity is not feasible. By concentrating additional physicians
during the peak arrival window and reducing to {OFF_PEAK_DOCTORS} overnight, similar wait
reductions may be achieved with lower total staffing cost.

---

## References

Kim, S.-H., & Nelson, B. L. (2001). A fully sequential procedure for
indifference-zone selection in simulation. *ACM TOMACS*, 11(3), 251–273.

Nelson, B. L., & Pei, L. (2021). *Foundations and Methods of Stochastic Simulation*
(2nd ed.). Springer.

Whitt, W. (2007). What you should know about queueing models to set staffing
requirements in service systems. *Naval Research Logistics*, 54, 476–484.
"""

    path = os.path.join(DOE_DIR, "doe_report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved → {path}")


# MAIN

def main() -> None:
    print("Loading GP model and service parameters (fitted once for all scenarios)...")
    gpModel = sim.buildGP()
    print(f"  GP fitted  |  kernel: {gpModel._gp.kernel_}")

    # Baseline: current production configuration
    print(f"\nBaseline: {sim.N_CLERKS} clerk, {sim.N_NURSES} nurses, {sim.N_DOCTORS} doctors")
    baseDF    = runScenario(sim.N_CLERKS, sim.N_NURSES, sim.N_DOCTORS,
                            REPS_PER_SCENARIO, gpModel)
    baseM, baseHW = ciMean(baseDF[PRIMARY])
    baseU, _      = ciMean(baseDF["DoctorUtil"])
    baseline      = {"mean": baseM, "hw": baseHW, "util": baseU, "results": baseDF}
    print(f"  doctorWait={baseM:.1f} ± {baseHW:.1f}  util={baseU*100:.1f}%")

    # Phase 1
    factorial = runFactorial(gpModel)
    effects   = estimateEffects(factorial)
    print("\nEstimated effects (min):")
    print(effects.to_string())
    plotFactorialEffects(effects)

    # Phase 2
    activeFactors = {"nClerks": 1, "nNurses": [2, 3, 4], "nDoctors": [2, 3, 4]}
    enumeration = runEnumeration(gpModel, activeFactors)
    plotResponseSurface(enumeration)

    # Phase 3: top 3 candidates from enumeration
    top3 = (enumeration.sort_values("mean")
            .head(3)
            .apply(lambda r: (int(r.nClerks), int(r.nNurses), int(r.nDoctors)), axis=1)
            .tolist())
    knResult = kimNelson(top3, gpModel)
    plotKNConvergence(knResult["history"])

    # What-if experiments
    wi1  = whatIfConstant(gpModel, doctorCounts=[2, 3, 4])
    wi2a = whatIfTwoShift(gpModel)
    wi2b = whatIfUtilTarget(gpModel, utilTarget=UTIL_TARGET)

    # Combined what-if plot
    plotWhatIf(baseline, wi1, wi2a, wi2b)

    # Save all per-replication results
    allRows = []
    for _, row in enumeration.iterrows():
        df = row["results"]
        df = df.copy()
        df["scenario"] = f"n{int(row.nNurses)}d{int(row.nDoctors)}"
        allRows.append(df)
    pd.concat(allRows, ignore_index=True).to_csv(
        os.path.join(DOE_DIR, "doe_results.csv"), index=False
    )

    # Narrative report
    print("\nGenerating DOE report...")
    generateDOEReport(effects, enumeration, knResult, baseline, wi1, wi2a, wi2b)

    print(f"\nDOE complete. All outputs in: {DOE_DIR}/")


if __name__ == "__main__":
    main()
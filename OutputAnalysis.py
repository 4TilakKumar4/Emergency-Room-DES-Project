"""
OutputAnalysis.py — Post-hoc output analysis for the IE 7215 ED simulation.

Reads the per-replication results CSV produced by ERSimulationModelGPwithSev.py
and computes two categories of output measures, following the distinction drawn
by Nelson and Pei (2021, §7.1.2):

  MEASURES OF ERROR — tell us whether we have run enough replications:
    1. CI convergence     — how 95% CI half-width shrinks as n increases
    2. Relative half-width (RE = hw / mean) — precision relative to magnitude
    3. Validation relative error — |sim_mean - empirical| / empirical × 100%

  MEASURES OF RISK — support operational/clinical decision-making:
    4. Tail probabilities  — P(mean wait > threshold) for a range of thresholds
    5. Quantile estimates  — 0.50, 0.75, 0.90, 0.95 quantiles with
                             order-statistic confidence intervals
    6. MORE plots          — Measure of Risk and Error plots (Nelson & Pei, Fig 7.2):
                             histogram + LIKELY/UNLIKELY regions + CI markers on
                             mean and 0.05/0.95 quantile estimates

Inputs  : Results/simulation/gp_severity/ED_rep_results.csv
Outputs : Results/output_analysis/
            ci_convergence.png
            more_plots.png
            risk_measures.png
            output_analysis_summary.csv

References:
  Nelson, B. L. and Pei, L. (2021). Foundations and Methods of Stochastic
    Simulation, 2nd ed. Springer. Chapter 7.
"""

import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sim_engine.analysis_utils import ci


# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "Results", "simulation", "gp_severity",
                          "ED_rep_results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results", "output_analysis")

# Empirical validation targets (from er_5000_patients.csv)
VALIDATION = {
    "regWait":    4.67,
    "triageWait": 0.01,
    "doctorWait": 249.67,
    "LOS":        285.63,
}

# Risk thresholds — clinically motivated (minutes)
DOCTOR_THRESHOLDS = [30, 60, 120, 180, 240]
LOS_THRESHOLDS    = [120, 180, 240, 300, 360]

# Colours — consistent with the rest of the codebase
C_BLUE   = "#3498db"
C_RED    = "#e74c3c"
C_GREEN  = "#27ae60"
C_ORANGE = "#f39c12"
C_DARK   = "#2c3e50"


# Utilities

def ensureDir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def loadResults() -> pd.DataFrame:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Baseline results not found at:\n  {INPUT_FILE}\n"
            "Run ERSimulationModelGPwithSev.py first to generate the baseline CSV."
        )
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} replications  ({INPUT_FILE})")
    return df


def _quantileCI(series: pd.Series, q: float,
                alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Point estimate and 95% order-statistic CI for quantile q.
    Normal approximation to the binomial (Nelson & Pei 2021, eq. 7.7).
    Returns (point, lower_bound, upper_bound).
    """
    n         = len(series)
    sorted_v  = np.sort(series.values)
    point     = float(np.quantile(sorted_v, q))
    z         = stats.norm.ppf(1.0 - alpha / 2.0)
    lo_i      = max(0,   int(math.floor(n * q - z * math.sqrt(n * q * (1 - q)))))
    hi_i      = min(n-1, int(math.ceil( n * q + z * math.sqrt(n * q * (1 - q)))))
    return point, float(sorted_v[lo_i]), float(sorted_v[hi_i])


# 1. CI CONVERGENCE  (measure of error)

def ciConvergence(results: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute 95% CI half-width for k = 10, 20, ..., n replications.
    Also records the relative half-width RE = hw / mean (dimensionless precision).
    A flat curve after some k* indicates the replication budget is adequate.
    """
    n     = len(results)
    steps = list(range(10, n + 1, 10))
    rows  = []
    for k in steps:
        dw_m, dw_hw = ci(results["doctorWait"].iloc[:k])
        ls_m, ls_hw = ci(results["LOS"].iloc[:k])
        rows.append({
            "n":      k,
            "dw_m":   dw_m,
            "dw_hw":  dw_hw,
            "dw_re":  dw_hw / dw_m if dw_m > 0 else 0.0,
            "los_m":  ls_m,
            "los_hw": ls_hw,
            "los_re": ls_hw / ls_m if ls_m > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def plotCIConvergence(conv: pd.DataFrame) -> None:
    """
    Two-panel chart: absolute CI half-width (left axis) and relative
    half-width RE % (right axis, dashed) vs replication count.
    A vertical marker shows where RE drops below 5%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Measure of Error — CI Half-Width Convergence vs. Replications",
        fontsize=13, fontweight="bold",
    )

    cfg = [
        (axes[0], "dw_hw",  "dw_re",  "Doctor Wait",    C_RED),
        (axes[1], "los_hw", "los_re", "Length of Stay", C_BLUE),
    ]
    for ax, hw_col, re_col, label, color in cfg:
        ax2 = ax.twinx()
        ax.plot(conv["n"], conv[hw_col], color=color, linewidth=2.5,
                label="Half-width (min)")
        ax2.plot(conv["n"], conv[re_col] * 100, color=color, linewidth=1.5,
                 linestyle="--", alpha=0.65, label="RE (%)")

        # Mark where RE first drops below 5 %
        below5 = conv[conv[re_col] < 0.05]
        if not below5.empty:
            n5 = int(below5.iloc[0]["n"])
            ax.axvline(n5, color="gray", linestyle=":", linewidth=1.2,
                       label=f"RE < 5 % at n = {n5}")

        ax.set_xlabel("Number of replications (n)")
        ax.set_ylabel("CI Half-Width (min)", color=color)
        ax2.set_ylabel("Relative Half-Width  RE = hw/mean (%)", color="gray",
                       fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor="gray")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "ci_convergence.png")


# 2. VALIDATION RELATIVE ERROR  (measure of error)

def validationError(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each primary KPI: absolute error and relative error against empirical
    target, plus RE = hw / mean as a measure of estimation precision.
    """
    metrics = [
        ("regWait",    "Registration Wait"),
        ("triageWait", "Triage Wait"),
        ("doctorWait", "Doctor Wait"),
        ("LOS",        "Length of Stay"),
    ]
    rows = []
    for col, label in metrics:
        sim_m, sim_hw = ci(results[col])
        empirical     = VALIDATION[col]
        abs_err = abs(sim_m - empirical)
        rel_err = abs_err / empirical * 100 if empirical > 0 else float("nan")
        re_pct  = sim_hw / sim_m * 100 if sim_m > 0 else 0.0
        within  = abs_err <= sim_hw
        rows.append({
            "Metric":      label,
            "Simulated":   round(sim_m,    2),
            "HalfWidth":   round(sim_hw,   2),
            "RE_pct":      round(re_pct,   2),
            "Empirical":   empirical,
            "AbsError":    round(abs_err,  2),
            "RelError_pct": round(rel_err, 2),
            "WithinCI":    within,
        })
    return pd.DataFrame(rows)


def printValidationTable(vdf: pd.DataFrame) -> None:
    print()
    print(f"  {'Metric':<22} {'Sim mean':>9} {'±hw':>7}  {'RE%':>5}  "
          f"{'Empirical':>9}  {'|err|':>7}  {'RelErr%':>8}  {'InCI?':>5}")
    print("  " + "─" * 80)
    for _, r in vdf.iterrows():
        flag = "✓" if r["WithinCI"] else "✗"
        print(
            f"  {r['Metric']:<22} {r['Simulated']:>9.2f} {r['HalfWidth']:>7.2f}"
            f"  {r['RE_pct']:>4.1f}%  {r['Empirical']:>9.2f}"
            f"  {r['AbsError']:>7.2f}  {r['RelError_pct']:>7.2f}%  {flag:>5}"
        )


# 3. RISK MEASURES — tail probabilities and quantile estimates

def riskMeasures(results: pd.DataFrame) -> dict:
    """
    Compute tail probabilities and quantile estimates.

    These are measured from the distribution of per-replication *means*, so
    P(doctorWait > 120) means: the probability that the *average* patient
    wait in a given day exceeds 120 minutes — not the individual patient
    probability.  This is the relevant measure for operational planning.

    Quantile CIs use the order-statistic method (Nelson & Pei, 2021, eq. 7.7).
    """
    tail_dw = {t: float((results["doctorWait"] > t).mean())
               for t in DOCTOR_THRESHOLDS}
    tail_ls = {t: float((results["LOS"]        > t).mean())
               for t in LOS_THRESHOLDS}

    quantiles = {}
    for col, tag in [("doctorWait", "dw"), ("LOS", "los")]:
        for q in [0.50, 0.75, 0.90, 0.95]:
            pt, lo, hi = _quantileCI(results[col], q)
            quantiles[f"{tag}_q{int(q * 100)}"] = {
                "point": pt, "lo": lo, "hi": hi
            }

    return {"tail_dw": tail_dw, "tail_ls": tail_ls, "quantiles": quantiles}


def printRiskMeasures(risk: dict) -> None:
    print()
    print("  Doctor Wait — tail probabilities:")
    for t, p in risk["tail_dw"].items():
        bar = "█" * int(round(p * 20))
        print(f"    P(doctorWait > {t:>3} min) = {p:.3f}  {bar}")

    print()
    print("  LOS — tail probabilities:")
    for t, p in risk["tail_ls"].items():
        bar = "█" * int(round(p * 20))
        print(f"    P(LOS > {t:>3} min)        = {p:.3f}  {bar}")

    print()
    print(f"  {'Quantile':<28} {'Point':>8}   95 % order-statistic CI")
    print("  " + "─" * 55)
    labels = {
        "dw_q50":  "Doctor Wait  p50 (median)",
        "dw_q75":  "Doctor Wait  p75",
        "dw_q90":  "Doctor Wait  p90",
        "dw_q95":  "Doctor Wait  p95",
        "los_q50": "LOS          p50 (median)",
        "los_q75": "LOS          p75",
        "los_q90": "LOS          p90",
        "los_q95": "LOS          p95",
    }
    for key, label in labels.items():
        d = risk["quantiles"][key]
        print(f"  {label:<28} {d['point']:>8.1f}   [{d['lo']:>7.1f}, {d['hi']:>7.1f}]")


def plotRiskMeasures(results: pd.DataFrame, risk: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Measures of Risk — Tail Probabilities from Distribution of Replication Means",
        fontsize=13, fontweight="bold",
    )

    panel_cfg = [
        (axes[0], risk["tail_dw"], "Threshold (min)",
         "P(mean doctor wait > threshold)", "Doctor Wait"),
        (axes[1], risk["tail_ls"], "Threshold (min)",
         "P(mean LOS > threshold)", "Length of Stay"),
    ]
    for ax, tail_dict, xlabel, ylabel, title in panel_cfg:
        ths   = list(tail_dict.keys())
        probs = list(tail_dict.values())
        clrs  = [C_RED    if p > 0.5
                 else C_ORANGE if p > 0.2
                 else C_GREEN
                 for p in probs]
        bars = ax.bar([str(t) for t in ths], probs, color=clrs,
                      alpha=0.85, edgecolor="white")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
                   alpha=0.4, label="P = 0.50")
        ax.set_ylim(0, 1.12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{p:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "risk_measures.png")


# 4. MORE PLOTS  (Nelson & Pei, 2021, §7.1.2)

def plotMORE(results: pd.DataFrame) -> None:
    """
    Measure of Risk and Error (MORE) plot for doctor wait and LOS.

    Layout (following Nelson & Pei, 2021, Figure 7.2):
      - Histogram of the 100 per-replication means
      - Central 90 % box (p05–p95 of rep means) labelled LIKELY
      - Tails labelled UNLIKELY
      - Downward arrow at the sample mean with its 95 % CI shown below
      - Downward arrows at the p05 and p95 quantile estimates with
        order-statistic CIs shown below
      - Vertical dotted line for the empirical validation target

    The plot makes both error (CI widths on the arrows) and risk (position of
    the LIKELY region relative to thresholds) visible simultaneously.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "MORE Plot — Measure of Risk and Error  (Nelson & Pei, 2021, §7.1.2)",
        fontsize=13, fontweight="bold",
    )

    panel_cfg = [
        (axes[0], "doctorWait", "Doctor Wait (min)",    C_RED,  249.67),
        (axes[1], "LOS",        "Length of Stay (min)", C_BLUE, 285.63),
    ]

    for ax, col, xlabel, color, empirical in panel_cfg:
        data     = results[col].values
        n        = len(data)
        mean_val = float(np.mean(data))
        q05      = float(np.percentile(data, 5))
        q95      = float(np.percentile(data, 95))

        _, hw_mean          = ci(results[col])
        _, q05_lo, q05_hi   = _quantileCI(results[col], 0.05)
        _, q95_lo, q95_hi   = _quantileCI(results[col], 0.95)

        # ── histogram ──
        counts, bin_edges, _ = ax.hist(
            data, bins=15, color=color, alpha=0.55,
            edgecolor="white", zorder=2,
        )
        max_count = counts.max()

        # ── LIKELY / UNLIKELY shading ──
        ax.axvspan(data.min() * 0.97, q05,        alpha=0.13, color="gray",  zorder=1)
        ax.axvspan(q95,  data.max() * 1.03,        alpha=0.13, color="gray",  zorder=1)
        ax.axvspan(q05,  q95,                       alpha=0.09, color=color,  zorder=1)

        y_label = max_count * 0.91
        mid_lo  = (data.min() + q05) / 2
        mid_hi  = (q95 + data.max()) / 2
        ax.text(mid_lo,                  y_label, "UNLIKELY",
                ha="center", fontsize=8, color="gray", style="italic")
        ax.text((q05 + q95) / 2,         y_label, "LIKELY",
                ha="center", fontsize=9, color=color, fontweight="bold")
        ax.text(mid_hi,                  y_label, "UNLIKELY",
                ha="center", fontsize=8, color="gray", style="italic")

        # ── annotations below the histogram ──
        # Reserve space below x = 0
        gap      = max_count * 0.13   # vertical gap per annotation row
        arrow_y  = -gap               # tip of downward arrows
        ci_y     = -gap * 1.7         # where CI bars are drawn
        text_y   = -gap * 2.6         # where text labels sit

        def annotate_point(x, hw_or_bounds, label_str, ann_color):
            # Downward arrow from above 0 to 0
            ax.annotate(
                "", xy=(x, 0), xytext=(x, arrow_y * 0.5),
                arrowprops=dict(arrowstyle="->", color=ann_color, lw=1.8),
            )
            # CI bar
            if isinstance(hw_or_bounds, tuple):
                lo, hi = hw_or_bounds
            else:
                lo, hi = x - hw_or_bounds, x + hw_or_bounds
            ax.plot([lo, hi], [ci_y, ci_y], color=ann_color,
                    linewidth=2.5, solid_capstyle="round")
            ax.text(x, text_y, label_str,
                    ha="center", va="top", fontsize=7.2, color=ann_color)

        annotate_point(
            mean_val, hw_mean,
            f"mean={mean_val:.1f}\nCI ±{hw_mean:.1f}",
            C_DARK,
        )
        annotate_point(
            q05, (q05_lo, q05_hi),
            f"p05={q05:.1f}\n[{q05_lo:.0f}, {q05_hi:.0f}]",
            "gray",
        )
        annotate_point(
            q95, (q95_lo, q95_hi),
            f"p95={q95:.1f}\n[{q95_lo:.0f}, {q95_hi:.0f}]",
            "gray",
        )

        # ── empirical target ──
        ax.axvline(empirical, color="firebrick", linestyle=":",
                   linewidth=1.8, label=f"Empirical = {empirical}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(xlabel, fontweight="bold")
        ax.legend(fontsize=8)
        # Extend y-axis downward to fit annotation rows
        ax.set_ylim(text_y * 1.4, max_count * 1.12)

    plt.tight_layout()
    _save(fig, "more_plots.png")


# 5. SUMMARY CSV

def saveSummaryCSV(vdf: pd.DataFrame, risk: dict, conv: pd.DataFrame) -> None:
    rows = []

    # Validation / estimation error
    for _, r in vdf.iterrows():
        rows.append({
            "Section":    "Validation",
            "Metric":     r["Metric"],
            "Value":      r["Simulated"],
            "HalfWidth":  r["HalfWidth"],
            "RE_pct":     r["RE_pct"],
            "Note":       (f"Empirical={r['Empirical']}, "
                           f"RelErr={r['RelError_pct']}%, "
                           f"WithinCI={r['WithinCI']}"),
        })

    # CI convergence (final row — RE at n = 100)
    final = conv.iloc[-1]
    for tag, val in [("doctorWait_RE_at_n100", final["dw_re"] * 100),
                     ("LOS_RE_at_n100",         final["los_re"] * 100)]:
        rows.append({
            "Section": "Convergence", "Metric": tag,
            "Value": round(val, 2), "HalfWidth": "",
            "RE_pct": "", "Note": "Relative half-width (%) at n=100",
        })

    # Tail probabilities
    for t, p in risk["tail_dw"].items():
        rows.append({
            "Section": "Risk", "Metric": f"P(doctorWait>{t})",
            "Value": round(p, 4), "HalfWidth": "", "RE_pct": "", "Note": "",
        })
    for t, p in risk["tail_ls"].items():
        rows.append({
            "Section": "Risk", "Metric": f"P(LOS>{t})",
            "Value": round(p, 4), "HalfWidth": "", "RE_pct": "", "Note": "",
        })

    # Quantile estimates
    for key, d in risk["quantiles"].items():
        rows.append({
            "Section": "Quantile", "Metric": key,
            "Value": round(d["point"], 2), "HalfWidth": "", "RE_pct": "",
            "Note": f"CI=[{d['lo']:.1f}, {d['hi']:.1f}]",
        })

    path = os.path.join(OUTPUT_DIR, "output_analysis_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved → {path}")


# MAIN

def main() -> None:
    ensureDir()
    print("\nED Simulation — Output Analysis")
    print("=" * 65)

    results = loadResults()
    n       = len(results)

    # 1. CI convergence
    print(f"\n[1/4]  CI convergence and relative half-width  (n = {n})")
    conv   = ciConvergence(results)
    plotCIConvergence(conv)
    final  = conv.iloc[-1]
    print(f"  Doctor wait : mean = {final['dw_m']:.2f} min | "
          f"hw = {final['dw_hw']:.2f} | RE = {final['dw_re']*100:.1f}%")
    print(f"  LOS         : mean = {final['los_m']:.2f} min | "
          f"hw = {final['los_hw']:.2f} | RE = {final['los_re']*100:.1f}%")

    # 2. Validation error
    print("\n[2/4]  Validation relative error")
    vdf = validationError(results)
    printValidationTable(vdf)

    # 3. Risk measures
    print("\n[3/4]  Risk measures — tail probabilities and quantile estimates")
    risk = riskMeasures(results)
    printRiskMeasures(risk)
    plotRiskMeasures(results, risk)

    # 4. MORE plots
    print("\n[4/4]  MORE plots")
    plotMORE(results)

    # 5. Summary CSV
    print("\nWriting summary CSV...")
    saveSummaryCSV(vdf, risk, conv)

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
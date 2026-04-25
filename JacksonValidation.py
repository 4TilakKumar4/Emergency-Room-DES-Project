"""
Jackson Network validation for the IE 7215 ED Simulation.

Structural validation strategy
-------------------------------
The full simulation uses NHPP arrivals, non-exponential service times, and
priority queues — none of which admit closed-form solutions.  To validate
that the simulation mechanics are correct (event scheduling, seize/free logic,
queue discipline, warmup clearing) we temporarily simplify the model into a
tractable Jackson Network and compare simulation output against the analytical
M/M/c result for each stage.

Simplifications applied for this validation run only
-----------------------------------------------------
1. Arrivals : NHPP replaced by HPP with rate lambda = mean GP posterior rate.
2. Service  : All stage/severity distributions replaced by Exponential(mean)
              where mean is the severity-weighted empirical mean from theParams.
3. Queues   : PriorityQueues replaced by FIFO at triage and doctor stages.
4. Warmup   : Same 480-min warmup as production model.

Under these conditions each stage is an independent M/M/c queue and the
product-form Jackson theorem gives a closed-form expected wait in queue:

    Erlang-C formula:
        C(c, rho) = (c*rho)^c / (c! * (1-rho)) * P0
        P0 = [ sum_{k=0}^{c-1} (c*rho)^k/k!  +  (c*rho)^c/(c!*(1-rho)) ]^{-1}
        E[Wq] = C(c, rho) / (c * mu - lambda)

where rho = lambda / (c * mu), c = number of servers, mu = 1 / mean_service.

Validation criterion
--------------------
For each stage the simulation 95% CI must contain the analytical E[Wq].
An error below 5% is considered acceptable for structural validation.

Outputs
-------
  Results/Validation/validation_jackson.png   — comparison bar chart
  Results/Validation/validation_jackson.csv   — table of analytical vs simulated values
"""

import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import ERSimulationModelGPwithSev as sim
from sim_engine import SimClasses, SimFunctions, SimRNG

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "Results", "Validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Validation run parameters — same warmup as production model
VAL_REPS   = 100
VAL_WARMUP = sim.WARMUP_MIN
VAL_RUN    = sim.RUN_MIN
ALPHA      = 0.05

# Stages to validate, mapped to their resource and server count
STAGES = [
    ("registration", "reg",    sim.N_CLERKS,  1),
    ("triage",       "triage", sim.N_NURSES,  2),
    ("doctor",       "doctor", sim.N_DOCTORS, 3),
]


# M/M/c analytical formulas

def erlangC(lam: float, mu: float, c: int) -> float:
    """
    Erlang-C probability: Pr(arriving customer waits) for an M/M/c queue.
    Returns C(c, rho) where rho = lambda / (c * mu).
    """
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0

    # Compute P0 (probability all servers idle)
    sum_terms = sum((c * rho) ** k / math.factorial(k) for k in range(c))
    last_term  = (c * rho) ** c / (math.factorial(c) * (1 - rho))
    p0         = 1.0 / (sum_terms + last_term)

    # Erlang-C numerator
    C = (c * rho) ** c / (math.factorial(c) * (1 - rho)) * p0
    return C


def mmcWaitInQueue(lam: float, mu: float, c: int) -> float:
    """
    Expected wait in queue for M/M/c: E[Wq] = C(c,rho) / (c*mu - lambda).
    Returns infinity when rho >= 1 (unstable system).
    """
    rho = lam / (c * mu)
    if rho >= 1.0:
        return float("inf")
    C = erlangC(lam, mu, c)
    return C / (c * mu - lam)


# Derive validation parameters from the fitted model

def getMeanArrivalRate(gpModel) -> float:
    """
    Compute the mean arrival rate (patients per minute) by integrating the GP
    posterior mean curve over a full 24-hour period and dividing by 1440 min.
    The result is used as the constant HPP rate lambda for the validation run.
    """
    hours = np.linspace(0, 24, 2000)
    meanCurve = gpModel.meanRateCurve()
    # GP curve returns patients per hour; convert to per minute
    ratePerHour = sum(float(meanCurve(h)) for h in hours) / len(hours)
    return ratePerHour / 60.0


def _meanFromEntry(entry: tuple) -> float:
    """
    Extract the mean service time from a SimRNG parameter tuple.

    Tuple formats by distribution:
      Lognormal  : ("Lognormal",  mean,    variance)   → entry[1]
      Erlang     : ("Erlang",     m,       mean)        → entry[2]
      Expon      : ("Expon",      mean)                 → entry[1]
      Normal     : ("Normal",     mean,    variance)    → entry[1]
      Triangular : ("Triangular", a,       b,       c)  → (a+b+c)/3
    """
    func = entry[0]
    if func == "Erlang":
        return float(entry[2])
    if func == "Triangular":
        return (float(entry[1]) + float(entry[2]) + float(entry[3])) / 3.0
    return float(entry[1])


def getWeightedMeanService(params: dict, stage: str) -> float:
    """
    Compute the severity-weighted mean service time for a stage.
    Weights match the empirical severity distribution in the simulation.
    """
    weights = {"low": 0.5556, "medium": 0.3488, "high": 0.0956}
    return sum(w * _meanFromEntry(params[stage][sev])
               for sev, w in weights.items())


# Patching helpers — temporarily replace simulation behaviour

def _fifoRemove(queue):
    """FIFO remove from the front of the queue list (same as default pop(0))."""
    if not queue.ThisQueue:
        return None
    entity = queue.ThisQueue.pop(0)
    queue.WIP.Record(float(len(queue.ThisQueue)))
    return entity


def _constantInterarrival(mean_iat: float) -> float:
    """Return HPP inter-arrival time: Expon(mean_iat) from stream 1."""
    return SimRNG.Expon(mean_iat, sim.STREAM_ARRIVAL)


def runValidationReplication(lam_per_min: float,
                              mean_services: dict,
                              original_remove_triage,
                              original_remove_doctor) -> dict:
    """
    Run one validation replication under simplified Jackson Network conditions.

    Patches applied inside this function:
    - nextInterarrival uses constant HPP rate (lam_per_min).
    - drawService always returns Expon(weighted mean) regardless of severity.
    - triageQueue.Remove and doctorQueue.Remove use FIFO order.

    All patches are reverted before the function returns.
    """
    # Patch 1: constant HPP arrival rate
    mean_iat = 1.0 / lam_per_min
    original_nextInterarrival = sim.nextInterarrival

    def hppInterarrival():
        return _constantInterarrival(mean_iat)

    sim.nextInterarrival = hppInterarrival

    # Patch 2: exponential service times (weighted mean per stage)
    original_drawService = sim.drawService

    def exponService(params, stage, severity, stream):
        return SimRNG.Expon(mean_services[stage], stream)

    sim.drawService = exponService

    # Patch 3: FIFO discipline at triage and doctor queues
    sim.triageQueue.Remove = lambda: _fifoRemove(sim.triageQueue)
    sim.doctorQueue.Remove = lambda: _fifoRemove(sim.doctorQueue)

    # Patch 4: disable shift-change schedule for this clean validation run
    original_schedule = sim.DOCTOR_SCHEDULE
    sim.DOCTOR_SCHEDULE = []

    # Run one replication — use a fixed dummy rate function (not used since
    # nextInterarrival is patched, but runReplication expects a callable)
    dummy_rate = lambda t: lam_per_min * 60.0
    result = sim.runReplication(dummy_rate)

    # Revert all patches
    sim.nextInterarrival          = original_nextInterarrival
    sim.drawService               = original_drawService
    sim.triageQueue.Remove        = original_remove_triage
    sim.doctorQueue.Remove        = original_remove_doctor
    sim.DOCTOR_SCHEDULE           = original_schedule

    return result


def runJacksonValidation(gpModel) -> pd.DataFrame:
    """
    Run VAL_REPS replications under Jackson Network conditions and compare
    simulated wait times against the M/M/c analytical formula for each stage.

    Returns a DataFrame with one row per stage containing analytical and
    simulated estimates and the validation verdict.
    """
    lam     = getMeanArrivalRate(gpModel)
    params  = sim.theParams

    mean_services = {
        "reg":    getWeightedMeanService(params, "reg"),
        "triage": getWeightedMeanService(params, "triage"),
        "doctor": getWeightedMeanService(params, "doctor"),
    }

    print(f"\nJackson Network Validation")
    print(f"  HPP rate lambda = {lam:.5f} patients/min  "
          f"({lam*60:.2f} patients/hr)")
    for stage, ms in mean_services.items():
        mu  = 1.0 / ms
        print(f"  {stage:<12}: mean service = {ms:.2f} min  "
              f"mu = {mu:.4f}/min")

    # Analytical expected wait in queue for each stage
    analytical = {
        "registration": mmcWaitInQueue(lam, 1.0 / mean_services["reg"],    1),
        "triage":       mmcWaitInQueue(lam, 1.0 / mean_services["triage"],  2),
        "doctor":       mmcWaitInQueue(lam, 1.0 / mean_services["doctor"],  3),
    }
    for stage, wq in analytical.items():
        rho = lam / ({"registration":1,"triage":2,"doctor":3}[stage]
                     * (1.0 / mean_services[{"registration":"reg","triage":"triage","doctor":"doctor"}[stage]]))
        print(f"  Analytical E[Wq] {stage:<14}: {wq:.2f} min  (rho={rho:.3f})")

    # Store original Remove methods before patching
    original_remove_triage = sim.triageQueue.Remove
    original_remove_doctor = sim.doctorQueue.Remove

    # Run validation replications
    print(f"\n  Running {VAL_REPS} validation replications...")
    sim.N_CLERKS  = 1
    sim.N_NURSES  = 2
    sim.N_DOCTORS = 3
    sim.clerk.SetUnits(1)
    sim.nurses.SetUnits(2)
    sim.doctors.SetUnits(3)

    rows = []
    SimRNG.ZRNG = SimRNG.InitializeRNSeed()
    for rep in range(VAL_REPS):
        row = runValidationReplication(
            lam, mean_services,
            original_remove_triage, original_remove_doctor
        )
        rows.append(row)
        if (rep + 1) % 20 == 0:
            print(f"    Completed {rep+1}/{VAL_REPS}")

    results = pd.DataFrame(rows)

    # Build comparison table
    stage_map = {
        "registration": ("regWait",    1),
        "triage":       ("triageWait", 2),
        "doctor":       ("doctorWait", 3),
    }
    table_rows = []
    for stage_label, (col, c) in stage_map.items():
        y   = results[col].dropna()
        n   = len(y)
        m   = y.mean()
        s   = y.std(ddof=1)
        t   = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
        hw  = t * s / math.sqrt(n)
        ci_lo, ci_hi = m - hw, m + hw

        analytic = analytical[stage_label]
        error_pct = abs(m - analytic) / analytic * 100 if analytic > 0 else float("nan")
        contained = (ci_lo <= analytic <= ci_hi) if analytic < float("inf") else False
        verdict   = "PASS" if contained or error_pct < 5.0 else "FAIL"

        print(f"\n  {stage_label.capitalize()}:")
        print(f"    Analytical E[Wq] = {analytic:.2f} min")
        print(f"    Simulated  E[Wq] = {m:.2f} ± {hw:.2f} min  "
              f"[{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"    Error = {error_pct:.1f}%   CI contains analytic: {contained}   {verdict}")

        table_rows.append({
            "stage":        stage_label,
            "servers_c":    c,
            "analytical":   round(analytic, 3),
            "simulated":    round(m, 3),
            "hw":           round(hw, 3),
            "ci_lower":     round(ci_lo, 3),
            "ci_upper":     round(ci_hi, 3),
            "error_pct":    round(error_pct, 1),
            "ci_contains":  contained,
            "verdict":      verdict,
        })

    table = pd.DataFrame(table_rows)
    return table, results


def plotValidation(table: pd.DataFrame) -> None:
    """
    Bar chart comparing analytical M/M/c wait times against simulation CIs.
    Each stage gets a bar for the simulated mean with a CI error bar, and a
    horizontal line showing the analytical target value.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x      = np.arange(len(table))
    width  = 0.5
    colors = ["#27ae60" if v == "PASS" else "#e74c3c"
              for v in table["verdict"]]

    bars = ax.bar(
        x, table["simulated"],
        width=width,
        color=colors, alpha=0.8,
        yerr=table["hw"],
        capsize=6,
        error_kw=dict(lw=2, color="black"),
        label="Simulated mean ± 95% CI",
    )

    # Analytical target markers
    for i, row in table.iterrows():
        ax.hlines(
            row["analytical"],
            i - width / 2, i + width / 2,
            colors="navy", linewidths=2.5,
            label="M/M/c analytical" if i == 0 else "_nolegend_",
        )
        ax.text(
            i, row["analytical"] + table["hw"].max() * 0.05,
            f"Analytic\n{row['analytical']:.1f}",
            ha="center", va="bottom", fontsize=8, color="navy",
        )

    # Verdict annotations
    for bar, row in zip(bars, table.itertuples()):
        color  = "#27ae60" if row.verdict == "PASS" else "#e74c3c"
        label  = f"{row.verdict}\n({row.error_pct:.1f}% err)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row.hw + table["hw"].max() * 0.15,
            label,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{r.stage.capitalize()}\n(M/M/{r.servers_c})"
        for r in table.itertuples()
    ])
    ax.set_ylabel("Expected Wait in Queue (min)")
    ax.set_title(
        "Jackson Network Validation — Simulation vs. M/M/c Analytical\n"
        "Simplified conditions: HPP arrivals, Exponential service, FIFO queues",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "validation_jackson.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {path}")


def main() -> None:
    print("Loading GP model and service parameters...")
    gpModel = sim.buildGP()
    print(f"  GP fitted  |  kernel: {gpModel._gp.kernel_}")

    table, results = runJacksonValidation(gpModel)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "validation_jackson.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # Save plot
    plotValidation(table)

    # Summary
    passed = (table["verdict"] == "PASS").sum()
    total  = len(table)
    print(f"\nValidation complete: {passed}/{total} stages passed")
    print(table[["stage", "analytical", "simulated", "error_pct", "verdict"]].to_string(index=False))

    if passed == total:
        print("\nAll stages PASS — simulation mechanics are structurally validated.")
    else:
        failed = table[table["verdict"] == "FAIL"]["stage"].tolist()
        print(f"\nWARNING: {failed} failed. Check service time parameters and queue logic.")


if __name__ == "__main__":
    main()
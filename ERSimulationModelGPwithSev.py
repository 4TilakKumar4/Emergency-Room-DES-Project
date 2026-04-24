"""
Emergency Room Discrete-Event Simulation

System:   1 receptionist, 2 triage nurses, 3 physicians
Arrivals: Non-Homogeneous Poisson Process (NHPP) with GP-modeled rates
          Rate function sampled from GP posterior each replication to
          propagate arrival rate input uncertainty into output CIs.
Service:  Severity-specific distributions from simrng_parameters.csv
Queues:   Registration = FIFO; Triage and Doctor = priority (high > med > low)

Outputs (100 replications, 95% CI):
  - Mean wait time per stage
  - Mean length of stay
  - Resource utilisation per resource type
  - Variance decomposition: input uncertainty vs simulation noise
  - Results/simulation/gp_severity/ED_output_analysis.png
  - Results/simulation/gp_severity/ED_utilisation.png
  - Results/simulation/gp_severity/ED_gp_posterior.png
  - Results/simulation/gp_severity/ED_rep_results.csv

Validation targets (empirical from er_5000_patients.csv):
  E[reg wait]    ≈   4.67 min   44.1% zero wait
  E[triage wait] ≈   0.01 min   99.4% zero wait
  E[doctor wait] ≈ 249.67 min    5.5% zero wait
  E[LOS]         ≈ 285.63 min

Input   : Sources/er_5000_patients.csv,  Sources/simrng_parameters.csv
Outputs : Results/simulation/gp_severity/*.png,  Results/simulation/gp_severity/ED_rep_results.csv
"""

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sim_engine import SimFunctions
from sim_engine import SimRNG
from sim_engine import SimClasses
from sim_engine.ArrivalRateGP import ArrivalRateGP
from analysis_utils import ci as ci95   # single source of truth for CI calculation


# File paths — edit here only if the project structure changes
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "Sources", "er_5000_patients.csv")
PARAMS_FILE = os.path.join(BASE_DIR, "Sources", "simrng_parameters.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "simulation", "gp_severity")

# Simulation run parameters
NUM_REPS   = 100
WARMUP_MIN = 480.0    # 8-hour warmup
RUN_MIN    = 1440.0   # 24-hour steady-state run

# System configuration — can be patched by DOE_Optimization before each scenario
N_CLERKS  = 1
N_NURSES  = 2
N_DOCTORS = 3

# Shift-change schedule for variable-staffing what-if experiments.
# Each entry is (time_in_minutes, new_doctor_count).
# Empty list means constant staffing throughout (default baseline behaviour).
DOCTOR_SCHEDULE = []

# Random number streams — one per independent source of randomness
STREAM_ARRIVAL  = 1
STREAM_SEVERITY = 2
STREAM_REG      = 3
STREAM_TRIAGE   = 4
STREAM_DOCTOR   = 5

# Severity levels and priority map (lower number = higher priority)
SEVERITIES = ["low", "medium", "high"]
PRIORITY   = {"high": 0, "medium": 1, "low": 2}

# Plot colours
SEV_COLORS    = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
METRIC_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

# Empirical validation targets
VALIDATION = {
    "regWait":    4.67,
    "triageWait": 0.01,
    "doctorWait": 249.67,
    "LOS":        285.63,
}

# Severity assignment — cumulative CDF for SimRNG.Random_integer
# low = 0.5556, medium = 0.3488, high = 0.0956
SEV_PROBS  = [0.5556, 0.9044, 1.0000]
SEV_LABELS = {1: "low", 2: "medium", 3: "high"}

# Module-level rate function — set at the start of each replication
_currentRateFn = None


def ensureResultsDir(path: str) -> None:
    """Create the results directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def buildHourlyCounts(filepath: str) -> np.ndarray:
    """
    Read the patient CSV and build a (n_days, 24) integer count matrix.
    Each cell is the number of patients arriving in that hour on that day.
    This is the raw input the GP needs — not the fitted distributions.
    """
    df = pd.read_csv(filepath)
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df["arrival_date"] = df["arrival_time"].dt.date
    df["arrival_hour"] = df["arrival_time"].dt.hour

    counts = (
        df.groupby(["arrival_date", "arrival_hour"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all 24 hour columns are present even if some had zero arrivals
    counts = counts.reindex(columns=range(24), fill_value=0)
    return counts.values.astype(float)


def loadParams(filepath: str) -> dict:
    """
    Read simrng_parameters.csv and build a lookup:
      params[stage][severity] = (func, p1, p2[, p3])

    Distributions with no direct SimRNG equivalent (Beta, Weibull) are
    approximated with Erlang(m=1, mean) — equivalent to Expon(mean).
    """
    df       = pd.read_csv(filepath)
    a2       = df[df["Approach"] == "2 - Severity"]
    stageMap = {
        "Registration Service": "reg",
        "Triage Service":       "triage",
        "Doctor Service":       "doctor",
    }
    params = {s: {} for s in stageMap.values()}

    for _, row in a2.iterrows():
        stage = stageMap.get(row["Variable"])
        if stage is None:
            continue
        sev  = row["Severity"]
        func = row["SimRNG_Func"]
        p1   = float(row["Param1_Value"])
        p2   = float(row["Param2_Value"]) if str(row["Param2_Value"]).strip() else 0.0

        if func == "No direct SimRNG equivalent":
            params[stage][sev] = ("Erlang", 1, p1)
        elif func == "Lognormal":
            params[stage][sev] = ("Lognormal", p1, p2)
        elif func == "Erlang":
            params[stage][sev] = ("Erlang", int(p1), p2)
        elif func == "Expon":
            params[stage][sev] = ("Expon", p1)
        elif func == "Normal":
            params[stage][sev] = ("Normal", p1, p2)
        elif func == "Triangular":
            p3 = float(row["Param3_Value"])
            params[stage][sev] = ("Triangular", p1, p2, p3)
        else:
            params[stage][sev] = ("Erlang", 1, p1)

    return params


def drawService(params: dict, stage: str, severity: str, stream: int) -> float:
    """Sample a service time using the fitted distribution for this stage/severity."""
    entry = params[stage][severity]
    func  = entry[0]
    if func == "Lognormal":   return SimRNG.Lognormal(entry[1], entry[2], stream)
    if func == "Erlang":      return SimRNG.Erlang(entry[1], entry[2], stream)
    if func == "Expon":       return SimRNG.Expon(entry[1], stream)
    if func == "Normal":      return max(0.0, SimRNG.Normal(entry[1], entry[2], stream))
    if func == "Triangular":  return SimRNG.Triangular(entry[1], entry[2], entry[3], stream)
    return SimRNG.Expon(entry[1], stream)


def nextInterarrival() -> float:
    """
    NHPP inter-arrival using the current replication's GP-sampled rate curve.
    Converts clock minutes to a fractional hour, queries the callable rate
    function, then draws Expon(60/rate).
    """
    tHours = (SimClasses.Clock % 1440) / 60   # fractional hour in [0, 24)
    rate   = max(float(_currentRateFn(tHours)), 0.01)
    return SimRNG.Expon(60.0 / rate, STREAM_ARRIVAL)


def assignSeverity() -> str:
    return SEV_LABELS[SimRNG.Random_integer(SEV_PROBS, STREAM_SEVERITY)]


def buildGP() -> "ArrivalRateGP":
    """
    Load arrival data, fit the GP arrival rate model, and populate theParams.

    Called once by DOE_Optimization before running any scenarios so the GP
    is not re-fitted for every staffing configuration.

    Returns the fitted ArrivalRateGP instance ready for sampleRateCurve().
    """
    global theParams
    hourlyCounts = buildHourlyCounts(SOURCE_FILE)
    gp = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gp.fit(hourlyCounts)
    theParams = loadParams(PARAMS_FILE)
    return gp


def printParams(params: dict) -> None:
    print("\nSERVICE TIME PARAMETERS  (Approach 2 — severity-stratified)")
    labels = {"reg": "Registration", "triage": "Triage", "doctor": "Doctor"}
    for stage, label in labels.items():
        print(f"\n  {label}:")
        for sev in SEVERITIES:
            print(f"    {sev:<8}: {params[stage][sev]}")


class PriorityQueue:
    """
    Priority queue for ED patients that serves by acuity (high > medium > low).
    Patients of equal severity retain FIFO order within their class.

    Implementation: three deques keyed by PRIORITY value (0, 1, 2).
    Add and Remove are both O(1) — previously the list-based implementation
    was O(n) for both the scan and the insert, which mattered at high
    physician-queue depths (up to 100+ patients at the 3-doctor baseline).
    """

    def __init__(self):
        self.WIP    = SimClasses.CTStat()
        self._lanes: dict[int, deque] = {0: deque(), 1: deque(), 2: deque()}

    def NumQueue(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def Add(self, patient) -> None:
        self._lanes[PRIORITY[patient.severity]].append(patient)   # O(1)
        self.WIP.Record(float(self.NumQueue()))

    def Remove(self):
        for key in (0, 1, 2):                    # high → medium → low
            if self._lanes[key]:
                entity = self._lanes[key].popleft()   # O(1)
                self.WIP.Record(float(self.NumQueue()))
                return entity
        return None

    def Mean(self) -> float:
        return self.WIP.Mean()


class Patient(SimClasses.Entity):
    """
    ED patient entity. Inherits CreateTime from Entity (set to Clock on init).
    Stage timestamps are stamped as the patient moves through the system and
    used to compute per-stage wait times at service completion.
    """

    def __init__(self, severity: str):
        super().__init__()
        self.severity     = severity
        self.reg_start    = 0.0
        self.reg_end      = 0.0
        self.triage_start = 0.0
        self.triage_end   = 0.0
        self.doc_start    = 0.0


zSimRNG = SimRNG.InitializeRNSeed()

clerk   = SimClasses.Resource()
nurses  = SimClasses.Resource()
doctors = SimClasses.Resource()

clerk.SetUnits(N_CLERKS)
nurses.SetUnits(N_NURSES)
doctors.SetUnits(N_DOCTORS)

regQueue    = SimClasses.FIFOQueue()
triageQueue = PriorityQueue()
doctorQueue = PriorityQueue()

regWait    = SimClasses.DTStat()
triageWait = SimClasses.DTStat()
doctorWait = SimClasses.DTStat()
LOS        = SimClasses.DTStat()

# Per-severity collectors — one DTStat per (stage, severity) combination
regWaitBySev    = {sev: SimClasses.DTStat() for sev in SEVERITIES}
triageWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
doctorWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
losBySev        = {sev: SimClasses.DTStat() for sev in SEVERITIES}

calendar = SimClasses.EventCalendar()

theCTStats   = []
theDTStats   = [
    regWait, triageWait, doctorWait, LOS,
    *regWaitBySev.values(),
    *triageWaitBySev.values(),
    *doctorWaitBySev.values(),
    *losBySev.values(),
]
theQueues    = [regQueue, triageQueue, doctorQueue]
theResources = [clerk, nurses, doctors]

theParams = {}


def arrival() -> None:
    # Schedule next arrival using the GP-sampled rate for the current fractional hour
    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())
    p = Patient(assignSeverity())
    regQueue.Add(p)
    if clerk.Busy == 0:
        startRegistration()


def startRegistration() -> None:
    p           = regQueue.Remove()
    p.reg_start = SimClasses.Clock
    clerk.Seize(1)
    SimFunctions.SchedulePlus(calendar, "endRegistration",
                              drawService(theParams, "reg", p.severity, STREAM_REG), p)


def endRegistration(ev) -> None:
    p         = ev.WhichObject
    p.reg_end = SimClasses.Clock
    # Wait = time patient spent in the registration queue before the clerk was free
    if p.CreateTime >= WARMUP_MIN:
        regWait.Record(p.reg_start - p.CreateTime)
        regWaitBySev[p.severity].Record(p.reg_start - p.CreateTime)
    clerk.Free(1)
    triageQueue.Add(p)
    if nurses.Busy < nurses.NumberOfUnits:
        startTriage()
    if regQueue.NumQueue() > 0:
        startRegistration()


def startTriage() -> None:
    p              = triageQueue.Remove()
    p.triage_start = SimClasses.Clock
    nurses.Seize(1)
    SimFunctions.SchedulePlus(calendar, "endTriage",
                              drawService(theParams, "triage", p.severity, STREAM_TRIAGE), p)


def endTriage(ev) -> None:
    p            = ev.WhichObject
    p.triage_end = SimClasses.Clock
    # Wait = time between leaving registration and a nurse becoming available
    if p.CreateTime >= WARMUP_MIN:
        triageWait.Record(p.triage_start - p.reg_end)
        triageWaitBySev[p.severity].Record(p.triage_start - p.reg_end)
    nurses.Free(1)
    doctorQueue.Add(p)
    if doctors.Busy < doctors.NumberOfUnits:
        startDoctor()
    if triageQueue.NumQueue() > 0:
        startTriage()


def startDoctor() -> None:
    p           = doctorQueue.Remove()
    p.doc_start = SimClasses.Clock
    doctors.Seize(1)
    SimFunctions.SchedulePlus(calendar, "endDoctor",
                              drawService(theParams, "doctor", p.severity, STREAM_DOCTOR), p)


def endDoctor(ev) -> None:
    p = ev.WhichObject
    # Wait = time between leaving triage and a doctor becoming available
    if p.CreateTime >= WARMUP_MIN:
        doctorWait.Record(p.doc_start - p.triage_end)
        doctorWaitBySev[p.severity].Record(p.doc_start - p.triage_end)
    # Only record LOS for patients who arrived after the warmup period
    if p.CreateTime >= WARMUP_MIN:
        LOS.Record(SimClasses.Clock - p.CreateTime)
        losBySev[p.severity].Record(SimClasses.Clock - p.CreateTime)
    doctors.Free(1)
    if doctorQueue.NumQueue() > 0:
        startDoctor()


def shiftChange(ev) -> None:
    """
    Adjust the number of doctors at a scheduled shift boundary.

    The new doctor count is stored in ev.WhichObject (an integer).
    Both the module-level N_DOCTORS constant and the Resource capacity
    are updated so utilisation calculations remain correct.

    If the new count is lower and some doctors are busy, SetUnits records
    the new capacity — busy doctors finish their current patients before
    the reduction takes effect, mirroring real shift hand-off behaviour.
    """
    global N_DOCTORS
    newCount  = ev.WhichObject
    N_DOCTORS = newCount
    doctors.SetUnits(newCount)

    # If capacity increased, start serving any waiting patients immediately
    while doctors.Busy < doctors.NumberOfUnits and doctorQueue.NumQueue() > 0:
        startDoctor()


def runReplication(rateFn: callable) -> dict:
    """
    Run one replication with the given GP-sampled rate function.

    The rate function is stored in _currentRateFn so nextInterarrival() can
    access it without changing its signature — keeping all event function
    interfaces unchanged from the baseline model.

    Shift-change events are scheduled from DOCTOR_SCHEDULE if non-empty.
    Each entry (tMinutes, newCount) fires a shiftChange event at that absolute
    simulation clock time, allowing mid-replication doctor count adjustment.
    """
    global _currentRateFn
    _currentRateFn = rateFn

    SimFunctions.SimFunctionsInit(
        calendar, theQueues, theCTStats, theDTStats, theResources
    )
    for pq in [triageQueue, doctorQueue]:
        pq.WIP.Clear()
        pq.WIP.Xlast = 0.0

    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())

    # Schedule all doctor shift changes for this replication
    for tMinutes, newCount in DOCTOR_SCHEDULE:
        SimFunctions.SchedulePlus(calendar, "shiftChange", tMinutes, newCount)

    warmupCleared = False

    while SimClasses.Clock < WARMUP_MIN + RUN_MIN:
        ev = calendar.Remove()
        SimClasses.Clock = ev.EventTime

        if not warmupCleared and SimClasses.Clock >= WARMUP_MIN:
            SimFunctions.ClearStats(theCTStats, theDTStats)
            for pq in [triageQueue, doctorQueue]:
                pq.WIP.Clear()
                pq.WIP.Xlast = 0.0
            warmupCleared = True

        if   ev.EventType == "arrival":          arrival()
        elif ev.EventType == "endRegistration":  endRegistration(ev)
        elif ev.EventType == "endTriage":        endTriage(ev)
        elif ev.EventType == "endDoctor":        endDoctor(ev)
        elif ev.EventType == "shiftChange":      shiftChange(ev)
        else:
            raise RuntimeError(f"Unknown event type in calendar: {ev.EventType!r}")

    row = {
        "regWait":    regWait.Mean(),
        "triageWait": triageWait.Mean(),
        "doctorWait": doctorWait.Mean(),
        "LOS":        LOS.Mean(),
        "ClerkUtil":  clerk.Mean()   / N_CLERKS,
        "NurseUtil":  nurses.Mean()  / N_NURSES,
        "DoctorUtil": doctors.Mean() / N_DOCTORS,
    }
    for sev in SEVERITIES:
        row[f"regWait_{sev}"]    = regWaitBySev[sev].Mean()
        row[f"triageWait_{sev}"] = triageWaitBySev[sev].Mean()
        row[f"doctorWait_{sev}"] = doctorWaitBySev[sev].Mean()
        row[f"LOS_{sev}"]        = losBySev[sev].Mean()
    return row


def printResults(results: pd.DataFrame, decomp: dict) -> None:
    print(f"\nRESULTS SUMMARY  ({NUM_REPS} replications, 95% CI)")

    metrics = [
        ("regWait",    "Reg wait        (min)"),
        ("triageWait", "Triage wait     (min)"),
        ("doctorWait", "Doctor wait     (min)"),
        ("LOS",        "Length of stay  (min)"),
        ("ClerkUtil",  "Clerk util      (%)  "),
        ("NurseUtil",  "Nurse util      (%)  "),
        ("DoctorUtil", "Doctor util     (%)  "),
    ]
    for col, label in metrics:
        m, hw = ci95(results[col])
        scale = 100 if "util" in col.lower() else 1
        print(f"  {label}: {m * scale:8.3f}  ±  {hw * scale:.3f}")

    print(f"\n  VALIDATION TARGETS (empirical)")
    targets = [
        ("Reg wait",    VALIDATION["regWait"],    "44.1% zero wait"),
        ("Triage wait", VALIDATION["triageWait"], "99.4% zero wait"),
        ("Doctor wait", VALIDATION["doctorWait"], " 5.5% zero wait"),
        ("LOS",         VALIDATION["LOS"],        ""),
    ]
    for label, val, note in targets:
        print(f"  {label:<16}: {val:8.2f}  {note}")

    print(f"\n  VARIANCE DECOMPOSITION (doctor wait)")
    print(f"  Input uncertainty (GP arrival rates) : "
          f"{decomp['inputVar']:8.4f}  ({decomp['inputFraction'] * 100:.1f}%)")
    print(f"  Simulation noise                     : "
          f"{decomp['simVar']:8.4f}  ({decomp['simFraction'] * 100:.1f}%)")
    print(f"  Total variance                       : {decomp['totalVar']:8.4f}")
    print(f"\n  Interpretation: {decomp['inputFraction'] * 100:.0f}% of output variance "
          f"comes from arrival rate uncertainty — "
          + ("collecting more data would reduce CI width."
             if decomp["inputFraction"] > 0.3
             else "simulation noise dominates; more replications help more than more data."))

    print(f"\n  WAIT TIMES BY SEVERITY (doctor wait and LOS, 95% CI)")
    print(f"  {'Severity':<10} {'DoctorWait (min)':>20}  {'LOS (min)':>18}")
    for sev in SEVERITIES:
        dw_m, dw_hw = ci95(results[f"doctorWait_{sev}"])
        ls_m, ls_hw = ci95(results[f"LOS_{sev}"])
        print(f"  {sev:<10} {dw_m:8.2f} ± {dw_hw:5.2f}       {ls_m:8.2f} ± {ls_hw:5.2f}")

    print(f"\nResults saved in: {os.path.abspath(RESULTS_DIR)}/")
    for f in ["ED_output_analysis.png", "ED_utilisation.png",
              "ED_gp_posterior.png", "ED_rep_results.csv"]:
        print(f"  {f}")


def _save(fig, filename: str) -> None:
    """Save a figure to the Results folder and immediately close it to free memory."""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plotOutputAnalysis(results: pd.DataFrame) -> None:
    """
    2×2 histogram grid for the four primary output metrics.
    Overlays simulation mean, 95% CI bounds, and empirical validation target.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ED Simulation — Output Analysis ({NUM_REPS} Replications, GP Arrival Rates)",
        fontsize=14, fontweight="bold",
    )
    plotCfg = [
        ("regWait",    "Registration Wait (min)",  VALIDATION["regWait"],    METRIC_COLORS[0]),
        ("doctorWait", "Doctor Wait (min)",         VALIDATION["doctorWait"], METRIC_COLORS[1]),
        ("LOS",        "Length of Stay (min)",      VALIDATION["LOS"],        METRIC_COLORS[2]),
        ("DoctorUtil", "Doctor Utilisation (%)",    None,                     METRIC_COLORS[3]),
    ]
    for ax, (col, title, target, color) in zip(axes.flat, plotCfg):
        scale = 100 if "util" in col.lower() else 1
        data  = results[col] * scale
        m, hw = ci95(results[col])
        mScaled, hwScaled = m * scale, hw * scale

        ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(mScaled, color="black", linestyle="-",  linewidth=2,
                   label=f"Mean = {mScaled:.2f}")
        ax.axvline(mScaled - hwScaled, color="black", linestyle="--", linewidth=1.2,
                   label=f"95% CI ± {hwScaled:.2f}")
        ax.axvline(mScaled + hwScaled, color="black", linestyle="--", linewidth=1.2)
        if target is not None:
            ax.axvline(target, color="red", linestyle=":", linewidth=2,
                       label=f"Empirical = {target}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Minutes" if scale == 1 else "Utilisation (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "ED_output_analysis.png")


def plotUtilisation(results: pd.DataFrame) -> None:
    """Bar chart of mean resource utilisation with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Resource Utilisation with 95% CI", fontsize=14, fontweight="bold")

    cols   = ["ClerkUtil", "NurseUtil", "DoctorUtil"]
    labels = ["clerk\n(1 unit)", "nurses\n(2 units)", "doctors\n(3 units)"]
    colors = [METRIC_COLORS[0], METRIC_COLORS[3], METRIC_COLORS[1]]
    means  = [ci95(results[c])[0] * 100 for c in cols]
    errors = [ci95(results[c])[1] * 100 for c in cols]

    bars = ax.bar(labels, means, color=colors, alpha=0.8,
                  edgecolor="white", yerr=errors, capsize=5)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.4,
               label="100% capacity")
    ax.set_ylabel("Mean Utilisation (%)")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)

    for bar, val, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    _save(fig, "ED_utilisation.png")


def plotGPPosterior(gpModel: ArrivalRateGP, hourlyCounts: np.ndarray) -> None:
    """
    GP posterior diagnostic: posterior mean, 95% credible band, sample curves,
    and the empirical hourly means overlaid as reference points.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    gpModel.plotPosterior(
        nSamples=30,
        title="GP Posterior — ED Arrival Rate (pts/hr)",
        ax=ax,
    )
    empiricalMeans = hourlyCounts.mean(axis=0)
    ax.scatter(
        np.arange(24) + 0.5, empiricalMeans,
        color="#e74c3c", zorder=5, s=60,
        label="Empirical mean (30-day avg)", marker="D",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "ED_gp_posterior.png")


def main() -> None:
    ensureResultsDir(RESULTS_DIR)

    # Build the (n_days, 24) count matrix from the raw CSV
    print("\nLoading arrival data...")
    hourlyCounts = buildHourlyCounts(SOURCE_FILE)
    print(f"  Built ({hourlyCounts.shape[0]} days × {hourlyCounts.shape[1]} hours) count matrix")

    # Fit the GP once before any replications run
    print("Fitting GP arrival rate model...")
    gpModel = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gpModel.fit(hourlyCounts)
    print(f"  Fitted  |  kernel: {gpModel._gp.kernel_}")

    # Load fitted service time parameters
    global theParams
    theParams = loadParams(PARAMS_FILE)
    printParams(theParams)

    # Run all replications
    print("\nRUNNING SIMULATION  (GP arrival rates — two-level input uncertainty)")
    print(f"  {NUM_REPS} replications  |  {int(WARMUP_MIN)} min warmup  |  "
          f"{int(RUN_MIN)} min steady-state run")

    repResults = []
    for rep in range(NUM_REPS):
        # Each replication samples a different plausible rate curve from the GP posterior
        rateFn = gpModel.sampleRateCurve(randomState=rep)
        repResults.append(runReplication(rateFn))
        if (rep + 1) % 10 == 0:
            print(f"  Completed replication {rep + 1:>3} / {NUM_REPS}")

    results = pd.DataFrame(repResults)

    # Variance decomposition on doctor wait — primary bottleneck metric
    decomp = gpModel.varianceDecomposition(results["doctorWait"].values)

    printResults(results, decomp)

    print("\n\nGenerating plots...")
    plotOutputAnalysis(results)
    plotUtilisation(results)
    plotGPPosterior(gpModel, hourlyCounts)

    outPath = os.path.join(RESULTS_DIR, "ED_rep_results.csv")
    results.to_csv(outPath, index=False)
    print(f"  Saved → {outPath}")


if __name__ == "__main__":
    main()
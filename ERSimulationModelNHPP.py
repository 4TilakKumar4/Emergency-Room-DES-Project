"""
Emergency Room Discrete-Event Simulation


System:   1 receptionist, 2 triage nurses, 3 physicians
Arrivals: Non-Homogeneous Poisson Process (NHPP), hour-by-hour rates
Service:  Severity-specific distributions from simrng_parameters.csv
Queues:   Registration = FIFO; Triage and Doctor = priority (high > med > low)

Outputs (50 replications, 95% CI):
  - Mean wait time per stage
  - Mean length of stay
  - Resource utilisation per resource type
  - Results/simulation/nhpp/ED_output_analysis.png
  - Results/simulation/nhpp/ED_utilisation.png
  - Results/simulation/nhpp/ED_rep_results.csv

Validation targets (empirical from er_5000_patients.csv):
  E[reg wait]    ≈   4.67 min   44.1% zero wait
  E[triage wait] ≈   0.01 min   99.4% zero wait
  E[doctor wait] ≈ 249.67 min    5.5% zero wait
  E[LOS]         ≈ 285.63 min

Input   : Sources/simrng_parameters.csv
Outputs : Results/simulation/nhpp/*.png,  Results/simulation/nhpp/ED_rep_results.csv
"""

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sim_engine import SimFunctions
from sim_engine   import SimRNG
from sim_engine import SimClasses
from sim_engine.analysis_utils import ci as ci95   # single source of truth for CI calculation

# File paths — edit here only if the project structure changes
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(BASE_DIR, "Sources", "simrng_parameters.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "simulation", "nhpp")


# Simulation run parameters
NUM_REPS   = 100
WARMUP_MIN = 480.0    # 8-hour warmup
RUN_MIN    = 1440.0   # 24-hour steady-state run

# System configuration
N_CLERKS  = 1
N_NURSES  = 2
N_DOCTORS = 3

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

# NHPP hourly arrival rates (patients/hr) — estimated from 30-day dataset
HOURLY_RATES = {
     0:  1.7000,  1:  1.6667,  2:  1.4667,  3:  1.5333,
     4:  1.5000,  5:  2.4000,  6:  4.0333,  7:  6.0000,
     8:  9.5667,  9: 11.6000, 10: 10.3000, 11: 14.0000,
    12: 13.8667, 13: 13.6667, 14: 12.2333, 15: 12.4000,
    16: 10.7000, 17: 11.1000, 18:  7.4667, 19:  5.9000,
    20:  4.9000, 21:  3.3000, 22:  2.8667, 23:  2.5000,
}

# Severity assignment — cumulative CDF for SimRNG.Random_integer
# low = 0.5556, medium = 0.3488, high = 0.0956
SEV_PROBS  = [0.5556, 0.9044, 1.0000]
SEV_LABELS = {1: "low", 2: "medium", 3: "high"}

# Empirical validation targets
VALIDATION = {
    "regWait":    4.67,
    "triageWait": 0.01,
    "doctorWait": 249.67,
    "LOS":        285.63,
}


def ensureResultsDir(path: str) -> None:
    """Create the results directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def loadParams(filepath: str) -> dict:
    """
    Read simrng_parameters.csv and build a lookup:
      params[stage][severity] = (func, p1, p2[, p3])

    Distributions with no direct SimRNG equivalent (Beta, Weibull) are
    approximated with Erlang(m=1, mean) — equivalent to Expon(mean).
    """
    df    = pd.read_csv(filepath)
    a2    = df[df["Approach"] == "2 - Severity"]
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
    """NHPP inter-arrival: Expon(60/rate) where rate is the current hour's rate."""
    hour = int(SimClasses.Clock % 1440 / 60) % 24
    return SimRNG.Expon(60.0 / HOURLY_RATES[hour], STREAM_ARRIVAL)


def assignSeverity() -> str:
    return SEV_LABELS[SimRNG.Random_integer(SEV_PROBS, STREAM_SEVERITY)]

def printConfig() -> None:
    print(f"\n\n{'-' * 80}")
    print(f"SIMULATION CONFIGURATION")
    print(f"{'-' * 80}")
    print(f"Number of replications : {NUM_REPS}")
    print(f"Warmup period         : {int(WARMUP_MIN)} minutes")
    print(f"Steady-state run      : {int(RUN_MIN)} minutes")
    print(f"Resources             : {N_CLERKS} clerk(s), "
          f"{N_NURSES} nurse(s), {N_DOCTORS} doctor(s)")
    print(f"Random streams        : arrival={STREAM_ARRIVAL}, "
          f"severity={STREAM_SEVERITY}, reg={STREAM_REG}, "
          f"triage={STREAM_TRIAGE}, doctor={STREAM_DOCTOR}")


def printParams(params: dict) -> None:
    print(f"\n{'-' * 80}")
    print("SERVICE TIME PARAMETERS  (Approach 2 — severity-stratified)")
    print(f"{'-' * 80}")
    labels = {"reg": "Registration", "triage": "Triage", "doctor": "Doctor"}
    for stage, label in labels.items():
        print(f"\n  {label}:")
        for sev in SEVERITIES:
            print(f"    {sev:<8}: {params[stage][sev]}")


class PriorityQueue:
    """
    Priority queue serving by acuity (high > medium > low), FIFO within class.
    Three deques — O(1) Add and Remove.
    Exposes the same interface as FIFOQueue so event functions are unchanged.
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
        for key in (0, 1, 2):                         # high → medium → low
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

calendar = SimClasses.EventCalendar()

theCTStats   = []
theDTStats   = [regWait, triageWait, doctorWait, LOS]
theQueues    = [regQueue, triageQueue, doctorQueue]
theResources = [clerk, nurses, doctors]

theParams = {}


def arrival() -> None:
    # Schedule the next arrival before processing the current one (NHPP rate for current hour)
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
    # Only record LOS for patients who arrived after the warmup period
    if p.CreateTime >= WARMUP_MIN:
        LOS.Record(SimClasses.Clock - p.CreateTime)
    doctors.Free(1)
    if doctorQueue.NumQueue() > 0:
        startDoctor()


def runReplication() -> dict:
    """Run one replication and return a dict of steady-state performance metrics."""
    SimFunctions.SimFunctionsInit(
        calendar, theQueues, theCTStats, theDTStats, theResources
    )

    for pq in [triageQueue, doctorQueue]:
        pq.WIP.Clear()
        pq.WIP.Xlast = 0.0

    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())

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
        else:
            raise RuntimeError(f"Unknown event type in calendar: {ev.EventType!r}")

    return {
        "regWait":    regWait.Mean(),
        "triageWait": triageWait.Mean(),
        "doctorWait": doctorWait.Mean(),
        "LOS":        LOS.Mean(),
        "ClerkUtil":  clerk.Mean()   / N_CLERKS,
        "NurseUtil":  nurses.Mean()  / N_NURSES,
        "DoctorUtil": doctors.Mean() / N_DOCTORS,
    }


def printResults(results: pd.DataFrame) -> None:
    print(f"\n\n")
    print("RESULTS SUMMARY  (50 replications, 95% CI)")
    print(f"{'-' * 80}")

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
    print("  " + "-" * 45)
    targets = [
        ("Reg wait",    VALIDATION["regWait"],    "44.1% zero wait"),
        ("Triage wait", VALIDATION["triageWait"], "99.4% zero wait"),
        ("Doctor wait", VALIDATION["doctorWait"], " 5.5% zero wait"),
        ("LOS",         VALIDATION["LOS"],        ""),
    ]
    for label, val, note in targets:
        print(f"  {label:<16}: {val:8.2f}  {note}")

    print(f"\nResults saved in: {os.path.abspath(RESULTS_DIR)}/")
    for f in ["ED_output_analysis.png", "ED_utilisation.png", "ED_rep_results.csv"]:
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
    fig.suptitle("ED Simulation — Output Analysis (50 Replications)",
                 fontsize=14, fontweight="bold")

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
    fig.suptitle("Resource Utilisation with 95% CI",
                 fontsize=14, fontweight="bold")

    cols    = ["ClerkUtil", "NurseUtil", "DoctorUtil"]
    labels  = ["clerk\n(1 unit)", "nurses\n(2 units)", "doctors\n(3 units)"]
    colors  = [METRIC_COLORS[0], METRIC_COLORS[3], METRIC_COLORS[1]]
    means   = [ci95(results[c])[0] * 100 for c in cols]
    errors  = [ci95(results[c])[1] * 100 for c in cols]

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


def main() -> None:
    ensureResultsDir(RESULTS_DIR)
    

    # Load fitted service time parameters from simrng_parameters.csv
    global theParams
    theParams = loadParams(PARAMS_FILE)
    printConfig()
    printParams(theParams)


    # Run all replications
    print(f"\n\n")
    print("RUNNING SIMULATION")
    print(f"\n  {NUM_REPS} replications  |  {int(WARMUP_MIN)} min warmup  |  "
          f"{int(RUN_MIN)} min steady-state run")
    print(f"{'-' * 80}")

    print(f"\n Progress:")

    repResults = []
    for rep in range(NUM_REPS):
        repResults.append(runReplication())
        if (rep + 1) % 10 == 0:
            print(f"  Completed replication {rep + 1:>3} / {NUM_REPS}")

    results = pd.DataFrame(repResults)

    printResults(results)

    # Generate and save all diagnostic plots
    print("\n\nGenerating plots...")
    plotOutputAnalysis(results)
    plotUtilisation(results)

    outPath = os.path.join(RESULTS_DIR, "ED_rep_results.csv")
    results.to_csv(outPath, index=False)
    print(f"  Saved → {outPath}")


if __name__ == "__main__":
    main()
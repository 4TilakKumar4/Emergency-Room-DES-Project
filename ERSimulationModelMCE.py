"""
ERSimulationModelMCE.py — ED Simulation with Mass Casualty Event (MCE) Scenario

Extends the baseline GP-arrival model with a superimposed MCE arrival stream
that begins at 18:00 (6 PM) and decays exponentially over 3 hours.

The two arrival processes are kept strictly independent (Poisson superposition):
  Stream 1 — Normal ED visitors: NHPP with GP-sampled rate curve (unchanged)
  Stream 2 — MCE casualties:     NHPP with rate(t) = 20 · exp(−λ · elapsed_min)

Both streams feed the same queues and resources, so the ED responds to the
combined load. MCE patients are tagged (isMce=True) enabling split statistics
for MCE vs. normal patients within each replication.

Scenario parameters vs. baseline (ERSimulationModelGPwithSev.py):
  N_DOCTORS        : 3  →  6   (surge staffing)
  RESULTS_DIR      : Results/  →  Results/MCE/
  MCE_START_MIN    : 1080.0    (18:00 = 6 PM)
  MCE_DURATION     : 180.0     (3 hours)
  MCE_PEAK_RATE    : 20.0      (patients/hr at event onset)
  MCE_DECAY        : ln(20)/180 ≈ 0.01664 /min  (rate ≈ 1 pt/hr at hour 3)

Decay constant derivation:
  Target: rate(180 min) ≈ 1 pt/hr
  20 · exp(−λ · 180) = 1  →  λ = ln(20) / 180

Outputs (100 replications, 95% CI):
  - Mean wait time per stage  (all patients, MCE-only, normal-only)
  - Mean length of stay       (all patients, MCE-only, normal-only)
  - Resource utilisation per resource type
  - Variance decomposition: input uncertainty vs simulation noise
  - Results/MCE/MCE_output_analysis.png
  - Results/MCE/MCE_utilisation.png
  - Results/MCE/MCE_gp_posterior.png
  - Results/MCE/MCE_arrival_rate.png
  - Results/MCE/MCE_rep_results.csv

Reference:
  Hirshberg A, Holcomb JB, Mattox KL (2001). Hospital trauma care in
    multiple-casualty incidents: a critical view. Ann Emerg Med 37(6):647-52.
  Hick JL, Barbera JA, Kelen GD (2009). Refining surge capacity: conventional,
    contingency, and crisis capacity. Disaster Med Public Health Prep 3(S1).

Input   : Sources/er_5000_patients.csv,  Sources/simrng_parameters.csv
Outputs : Results/MCE/*.png,  Results/MCE/MCE_rep_results.csv
"""

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sim_engine import SimFunctions
from sim_engine import SimRNG
from sim_engine import SimClasses
from sim_engine.ArrivalRateGP import ArrivalRateGP


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "Sources", "er_5000_patients.csv")
PARAMS_FILE = os.path.join(BASE_DIR, "Sources", "simrng_parameters.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "MCE")

# ---------------------------------------------------------------------------
# Simulation run parameters  (identical to baseline)
# ---------------------------------------------------------------------------
NUM_REPS   = 100
WARMUP_MIN = 480.0    # 8-hour warmup  (midnight → 8 AM)
RUN_MIN    = 1440.0   # 24-hour steady-state run

# ---------------------------------------------------------------------------
# System configuration  — 6 doctors for MCE surge scenario
# ---------------------------------------------------------------------------
N_CLERKS  = 1
N_NURSES  = 2
N_DOCTORS = 3    # increased from baseline 3

DOCTOR_SCHEDULE = []    # no mid-run shift changes in this scenario

# ---------------------------------------------------------------------------
# Random number streams
# Streams 1-5 are identical to the baseline so GP-sampled arrival sequences
# and service times are comparable across the two models.
# Streams 6-7 are exclusive to the MCE process — full independence guaranteed.
# ---------------------------------------------------------------------------
STREAM_ARRIVAL      = 1
STREAM_SEVERITY     = 2
STREAM_REG          = 3
STREAM_TRIAGE       = 4
STREAM_DOCTOR       = 5
STREAM_MCE_ARRIVAL  = 6   # MCE inter-arrival draws
STREAM_MCE_SEVERITY = 7   # MCE severity assignment draws

# ---------------------------------------------------------------------------
# Severity levels and priority map
# ---------------------------------------------------------------------------
SEVERITIES = ["low", "medium", "high"]
PRIORITY   = {"high": 0, "medium": 1, "low": 2}

# Baseline severity probabilities (cumulative CDF)
# low ≈ 55.56%, medium ≈ 34.88%, high ≈ 9.56%
SEV_PROBS  = [0.5556, 0.9044, 1.0000]
SEV_LABELS = {1: "low", 2: "medium", 3: "high"}

# MCE severity probabilities — high-severity dominant
# Reflects trauma/rescue scenario: ~70% high, ~20% medium, ~10% low
MCE_SEV_PROBS = [0.10, 0.30, 1.00]

# ---------------------------------------------------------------------------
# Mass Casualty Event parameters
# ---------------------------------------------------------------------------
MCE_START_MIN  = 1080.0               # 18:00 (6 PM) — minutes from midnight
MCE_DURATION   = 180.0                # 3-hour rescue window
MCE_PEAK_RATE  = 20.0                 # patients/hr at event onset
MCE_DECAY      = math.log(MCE_PEAK_RATE) / MCE_DURATION
# Derivation: rate(180) = 1 pt/hr  →  20·exp(−λ·180) = 1  →  λ = ln(20)/180
# At t=0 min:   rate = 20.0 pts/hr
# At t=60 min:  rate ≈  8.0 pts/hr
# At t=120 min: rate ≈  3.2 pts/hr
# At t=180 min: rate ≈  1.0 pts/hr  (event effectively over)

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
SEV_COLORS    = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
METRIC_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_currentRateFn = None   # GP-sampled rate function, set per replication


# ===========================================================================
# Helper utilities
# ===========================================================================

def ensureResultsDir(path: str) -> None:
    """Create the results directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def buildHourlyCounts(filepath: str) -> np.ndarray:
    """
    Read the patient CSV and build a (n_days, 24) integer count matrix.
    Each cell is the number of patients arriving in that hour on that day.
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
    counts = counts.reindex(columns=range(24), fill_value=0)
    return counts.values.astype(float)


def loadParams(filepath: str) -> dict:
    """
    Read simrng_parameters.csv and build a lookup:
      params[stage][severity] = (func, p1, p2[, p3])
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


# ===========================================================================
# Arrival rate functions
# ===========================================================================

def nextInterarrival() -> float:
    """
    NHPP inter-arrival for the normal patient stream.
    Uses the GP-sampled rate function for the current replication.
    Identical to the baseline model — this function is untouched.
    """
    tHours = (SimClasses.Clock % 1440) / 60
    rate   = max(float(_currentRateFn(tHours)), 0.01)
    return SimRNG.Expon(60.0 / rate, STREAM_ARRIVAL)


def nextMceInterarrival() -> float:
    """
    NHPP inter-arrival for the MCE casualty stream.

    Rate decays exponentially from MCE_PEAK_RATE as rescues slow down:
      rate(t) = MCE_PEAK_RATE · exp(−MCE_DECAY · elapsed_min)

    elapsed_min is the time since MCE_START_MIN, computed from the live clock.
    This implements a time-varying Poisson process by computing the local rate
    at each self-scheduling step — valid for monotone decreasing rates.
    Uses STREAM_MCE_ARRIVAL (stream 6) — fully independent of all other streams.
    """
    elapsed = max(SimClasses.Clock - MCE_START_MIN, 0.0)
    rate    = max(MCE_PEAK_RATE * math.exp(-MCE_DECAY * elapsed), 0.01)
    return SimRNG.Expon(60.0 / rate, STREAM_MCE_ARRIVAL)


def assignSeverity() -> str:
    """Sample severity for a normal patient using the baseline probability vector."""
    return SEV_LABELS[SimRNG.Random_integer(SEV_PROBS, STREAM_SEVERITY)]


def assignMceSeverity() -> str:
    """
    Sample severity for an MCE patient.
    Uses the high-dominant MCE_SEV_PROBS vector and STREAM_MCE_SEVERITY (stream 7).
    """
    return SEV_LABELS[SimRNG.Random_integer(MCE_SEV_PROBS, STREAM_MCE_SEVERITY)]


def buildGP() -> "ArrivalRateGP":
    """
    Load arrival data, fit the GP arrival rate model, and populate theParams.
    Called once before running replications so the GP is not re-fitted per scenario.
    """
    global theParams
    hourlyCounts = buildHourlyCounts(SOURCE_FILE)
    gp = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gp.fit(hourlyCounts)
    theParams = loadParams(PARAMS_FILE)
    return gp


# ===========================================================================
# Priority queue
# ===========================================================================

class PriorityQueue:
    """
    Serves patients by severity. High-severity patients are inserted ahead of
    medium and low. Patients of equal severity retain FIFO order.
    """

    def __init__(self):
        self.WIP       = SimClasses.CTStat()
        self.ThisQueue = []

    def NumQueue(self) -> int:
        return len(self.ThisQueue)

    def Add(self, patient) -> None:
        pos = len(self.ThisQueue)
        for i, p in enumerate(self.ThisQueue):
            if PRIORITY[patient.severity] < PRIORITY[p.severity]:
                pos = i
                break
        self.ThisQueue.insert(pos, patient)
        self.WIP.Record(float(len(self.ThisQueue)))

    def Remove(self):
        if not self.ThisQueue:
            return None
        entity = self.ThisQueue.pop(0)
        self.WIP.Record(float(len(self.ThisQueue)))
        return entity

    def Mean(self) -> float:
        return self.WIP.Mean()


# ===========================================================================
# Patient entity
# ===========================================================================

class Patient(SimClasses.Entity):
    """
    ED patient entity.
    isMce = True flags patients originating from the MCE arrival stream,
    enabling split statistics without changing the service logic.
    """

    def __init__(self, severity: str, isMce: bool = False):
        super().__init__()
        self.severity     = severity
        self.isMce        = isMce
        self.reg_start    = 0.0
        self.reg_end      = 0.0
        self.triage_start = 0.0
        self.triage_end   = 0.0
        self.doc_start    = 0.0


# ===========================================================================
# Simulation objects  (module-level, reset each replication)
# ===========================================================================

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

# Aggregate stat collectors (all patients)
regWait    = SimClasses.DTStat()
triageWait = SimClasses.DTStat()
doctorWait = SimClasses.DTStat()
LOS        = SimClasses.DTStat()

# Per-severity collectors
regWaitBySev    = {sev: SimClasses.DTStat() for sev in SEVERITIES}
triageWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
doctorWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
losBySev        = {sev: SimClasses.DTStat() for sev in SEVERITIES}

# MCE-specific collectors — isolate MCE patient outcomes for reporting
doctorWaitMce = SimClasses.DTStat()
losMce        = SimClasses.DTStat()

# Normal-patient-only collectors — for direct comparison within same replication
doctorWaitNormal = SimClasses.DTStat()
losNormal        = SimClasses.DTStat()

calendar = SimClasses.EventCalendar()

theCTStats   = []
theDTStats   = [
    regWait, triageWait, doctorWait, LOS,
    *regWaitBySev.values(),
    *triageWaitBySev.values(),
    *doctorWaitBySev.values(),
    *losBySev.values(),
    doctorWaitMce, losMce,
    doctorWaitNormal, losNormal,
]
theQueues    = [regQueue, triageQueue, doctorQueue]
theResources = [clerk, nurses, doctors]

theParams = {}


# ===========================================================================
# Event handlers — normal patient stream (identical logic to baseline)
# ===========================================================================

def arrival() -> None:
    """Schedule next normal arrival; create patient and enter registration queue."""
    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())
    p = Patient(assignSeverity(), isMce=False)
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
    if p.CreateTime >= WARMUP_MIN:
        dw = p.doc_start - p.triage_end
        ls = SimClasses.Clock - p.CreateTime

        doctorWait.Record(dw)
        LOS.Record(ls)
        doctorWaitBySev[p.severity].Record(dw)
        losBySev[p.severity].Record(ls)

        # Split into MCE vs. normal sub-populations
        if p.isMce:
            doctorWaitMce.Record(dw)
            losMce.Record(ls)
        else:
            doctorWaitNormal.Record(dw)
            losNormal.Record(ls)

    doctors.Free(1)
    if doctorQueue.NumQueue() > 0:
        startDoctor()


# ===========================================================================
# Event handlers — MCE casualty stream
# ===========================================================================

def mceArrival() -> None:
    """
    MCE casualty arrival handler.

    Creates a high-severity-dominant patient tagged with isMce=True and adds
    them to the shared registration queue. Reschedules itself only while still
    within the MCE window — the stream dies naturally at MCE_START_MIN + MCE_DURATION.

    Both streams (normal and MCE) feed the same queues and resources, so the
    ED responds to the combined load as Poisson superposition.
    """
    p = Patient(assignMceSeverity(), isMce=True)
    regQueue.Add(p)
    if clerk.Busy == 0:
        startRegistration()

    # Continue self-scheduling only while the rescue window is still open
    if SimClasses.Clock < MCE_START_MIN + MCE_DURATION:
        SimFunctions.Schedule(calendar, "mceArrival", nextMceInterarrival())


def mceStart(ev) -> None:
    """
    Kick off the MCE casualty stream at MCE_START_MIN.
    Schedules the first MCE arrival; subsequent arrivals self-schedule via mceArrival().
    """
    print(f"  [MCE] Event started at t={SimClasses.Clock:.1f} min  "
          f"(clock = {SimClasses.Clock/60:.1f} hr,  "
          f"initial rate = {MCE_PEAK_RATE:.1f} pts/hr)")
    SimFunctions.Schedule(calendar, "mceArrival", nextMceInterarrival())


def shiftChange(ev) -> None:
    """Adjust the number of doctors at a scheduled shift boundary."""
    global N_DOCTORS
    newCount  = ev.WhichObject
    N_DOCTORS = newCount
    doctors.SetUnits(newCount)
    while doctors.Busy < doctors.NumberOfUnits and doctorQueue.NumQueue() > 0:
        startDoctor()


# ===========================================================================
# Replication runner
# ===========================================================================

def runReplication(rateFn: callable) -> dict:
    """
    Run one replication with the given GP-sampled rate function.

    Both the normal NHPP stream and the MCE stream are active.
    The MCE stream is seeded by a single mceStart event scheduled at MCE_START_MIN.
    """
    global _currentRateFn
    _currentRateFn = rateFn

    SimFunctions.SimFunctionsInit(
        calendar, theQueues, theCTStats, theDTStats, theResources
    )
    for pq in [triageQueue, doctorQueue]:
        pq.WIP.Clear()
        pq.WIP.Xlast = 0.0

    # Seed the normal arrival stream
    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())

    # Seed the MCE event — fires once at 6 PM to kick off the casualty stream
    SimFunctions.Schedule(calendar, "mceStart", MCE_START_MIN)

    # Schedule any mid-run doctor shift changes
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
        elif ev.EventType == "mceStart":         mceStart(ev)
        elif ev.EventType == "mceArrival":       mceArrival()

    row = {
        "regWait":          regWait.Mean(),
        "triageWait":       triageWait.Mean(),
        "doctorWait":       doctorWait.Mean(),
        "LOS":              LOS.Mean(),
        "ClerkUtil":        clerk.Mean()   / N_CLERKS,
        "NurseUtil":        nurses.Mean()  / N_NURSES,
        "DoctorUtil":       doctors.Mean() / N_DOCTORS,
        "doctorWait_mce":   doctorWaitMce.Mean(),
        "LOS_mce":          losMce.Mean(),
        "doctorWait_normal":doctorWaitNormal.Mean(),
        "LOS_normal":       losNormal.Mean(),
    }
    for sev in SEVERITIES:
        row[f"regWait_{sev}"]    = regWaitBySev[sev].Mean()
        row[f"triageWait_{sev}"] = triageWaitBySev[sev].Mean()
        row[f"doctorWait_{sev}"] = doctorWaitBySev[sev].Mean()
        row[f"LOS_{sev}"]        = losBySev[sev].Mean()

    return row


# ===========================================================================
# Output utilities
# ===========================================================================

def ci95(series: pd.Series) -> tuple[float, float]:
    """Return (mean, half-width) of the 95% CI for a column of replication results."""
    m  = series.mean()
    hw = 1.96 * series.std(ddof=1) / math.sqrt(len(series))
    return m, hw


def printResults(results: pd.DataFrame, decomp: dict) -> None:
    print(f"\nMCE SCENARIO RESULTS  ({NUM_REPS} replications, 95% CI)")
    print(f"  Configuration: {N_DOCTORS} doctors | MCE at {MCE_START_MIN/60:.0f}:00 "
          f"for {MCE_DURATION/60:.0f} hr | peak rate {MCE_PEAK_RATE:.0f} pts/hr\n")

    metrics = [
        ("regWait",           "Reg wait — all patients  (min)"),
        ("triageWait",        "Triage wait — all patients (min)"),
        ("doctorWait",        "Doctor wait — all patients (min)"),
        ("LOS",               "LOS — all patients       (min)"),
        ("doctorWait_mce",    "Doctor wait — MCE only   (min)"),
        ("LOS_mce",           "LOS — MCE only           (min)"),
        ("doctorWait_normal", "Doctor wait — normal only(min)"),
        ("LOS_normal",        "LOS — normal only        (min)"),
        ("ClerkUtil",         "Clerk util               (%)  "),
        ("NurseUtil",         "Nurse util               (%)  "),
        ("DoctorUtil",        "Doctor util              (%)  "),
    ]
    for col, label in metrics:
        m, hw = ci95(results[col])
        scale = 100 if "util" in col.lower() else 1
        print(f"  {label}: {m * scale:8.3f}  ±  {hw * scale:.3f}")

    print(f"\n  VARIANCE DECOMPOSITION (doctor wait — all patients)")
    print(f"  Input uncertainty : {decomp['inputVar']:8.4f}  "
          f"({decomp['inputFraction'] * 100:.1f}%)")
    print(f"  Simulation noise  : {decomp['simVar']:8.4f}  "
          f"({decomp['simFraction'] * 100:.1f}%)")
    print(f"  Total variance    : {decomp['totalVar']:8.4f}")

    print(f"\n  WAIT TIMES BY SEVERITY — doctor wait and LOS  (95% CI)")
    print(f"  {'Severity':<10} {'DoctorWait (min)':>22}  {'LOS (min)':>20}")
    for sev in SEVERITIES:
        dw_m, dw_hw = ci95(results[f"doctorWait_{sev}"])
        ls_m, ls_hw = ci95(results[f"LOS_{sev}"])
        print(f"  {sev:<10} {dw_m:8.2f} ± {dw_hw:5.2f}       {ls_m:8.2f} ± {ls_hw:5.2f}")

    print(f"\nResults saved in: {os.path.abspath(RESULTS_DIR)}/")


def _save(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plotOutputAnalysis(results: pd.DataFrame) -> None:
    """
    2×3 grid comparing all-patient, MCE-only, and normal-only doctor wait and LOS,
    plus resource utilisation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"ED MCE Scenario — Output Analysis ({NUM_REPS} Replications, 6 Doctors)",
        fontsize=14, fontweight="bold",
    )
    plotCfg = [
        ("doctorWait",        "Doctor Wait — All Patients (min)",    METRIC_COLORS[1]),
        ("doctorWait_mce",    "Doctor Wait — MCE Patients (min)",    "#e74c3c"),
        ("doctorWait_normal", "Doctor Wait — Normal Patients (min)", METRIC_COLORS[0]),
        ("LOS",               "LOS — All Patients (min)",            METRIC_COLORS[2]),
        ("LOS_mce",           "LOS — MCE Patients (min)",            "#c0392b"),
        ("LOS_normal",        "LOS — Normal Patients (min)",         "#1a7a4a"),
    ]
    for ax, (col, title, color) in zip(axes.flat, plotCfg):
        data  = results[col]
        m, hw = ci95(results[col])
        ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(m,      color="black", linestyle="-",  linewidth=2,
                   label=f"Mean = {m:.2f}")
        ax.axvline(m - hw, color="black", linestyle="--", linewidth=1.2,
                   label=f"95% CI ± {hw:.2f}")
        ax.axvline(m + hw, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "MCE_output_analysis.png")


def plotUtilisation(results: pd.DataFrame) -> None:
    """Bar chart of mean resource utilisation with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "MCE Scenario — Resource Utilisation with 95% CI",
        fontsize=14, fontweight="bold",
    )
    cols   = ["ClerkUtil", "NurseUtil", "DoctorUtil"]
    labels = ["clerk\n(1 unit)", "nurses\n(2 units)", "doctors\n(6 units)"]
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
    _save(fig, "MCE_utilisation.png")


def plotMceArrivalRate() -> None:
    """
    Diagnostic plot of the MCE arrival rate curve over the event window.
    Annotates the rate at 0, 60, 120, and 180 minutes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, MCE_DURATION, 300)
    rates = MCE_PEAK_RATE * np.exp(-MCE_DECAY * t)

    ax.plot(t, rates, color="#e74c3c", linewidth=2.5, label="MCE arrival rate")
    ax.fill_between(t, 0, rates, alpha=0.15, color="#e74c3c")

    for mark in [0, 60, 120, 180]:
        r = MCE_PEAK_RATE * math.exp(-MCE_DECAY * mark)
        ax.annotate(
            f"{r:.1f} pts/hr",
            xy=(mark, r), xytext=(mark + 5, r + 1.5),
            fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

    ax.set_xlabel("Minutes elapsed since MCE onset (18:00)")
    ax.set_ylabel("Arrival rate (patients / hr)")
    ax.set_title(
        f"MCE Casualty Stream — Exponential Decay  "
        f"(λ = {MCE_DECAY:.4f}/min,  peak = {MCE_PEAK_RATE:.0f} pts/hr)",
        fontweight="bold",
    )
    ax.set_xlim(0, MCE_DURATION)
    ax.set_ylim(0, MCE_PEAK_RATE * 1.15)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "MCE_arrival_rate.png")


def plotGPPosterior(gpModel: ArrivalRateGP, hourlyCounts: np.ndarray) -> None:
    """GP posterior diagnostic with empirical hourly means overlaid."""
    fig, ax = plt.subplots(figsize=(13, 6))
    gpModel.plotPosterior(
        nSamples=30,
        title="GP Posterior — ED Arrival Rate (pts/hr)  [MCE scenario baseline]",
        ax=ax,
    )
    empiricalMeans = hourlyCounts.mean(axis=0)
    ax.scatter(
        np.arange(24) + 0.5, empiricalMeans,
        color="#e74c3c", zorder=5, s=60,
        label="Empirical mean (30-day avg)", marker="D",
    )
    ax.axvline(MCE_START_MIN / 60, color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"MCE onset ({MCE_START_MIN/60:.0f}:00)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "MCE_gp_posterior.png")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    ensureResultsDir(RESULTS_DIR)

    print("\nMASS CASUALTY EVENT SCENARIO")
    print(f"  Doctors          : {N_DOCTORS}")
    print(f"  MCE onset        : {MCE_START_MIN/60:.0f}:00  "
          f"(t = {MCE_START_MIN:.0f} min)")
    print(f"  MCE duration     : {MCE_DURATION/60:.0f} hr  "
          f"({MCE_DURATION:.0f} min)")
    print(f"  Peak arrival rate: {MCE_PEAK_RATE:.0f} pts/hr  "
          f"→  decay const λ = {MCE_DECAY:.4f}/min")
    print(f"  Rate at t+1 hr   : {MCE_PEAK_RATE * math.exp(-MCE_DECAY*60):.1f} pts/hr")
    print(f"  Rate at t+2 hr   : {MCE_PEAK_RATE * math.exp(-MCE_DECAY*120):.1f} pts/hr")
    print(f"  Rate at t+3 hr   : {MCE_PEAK_RATE * math.exp(-MCE_DECAY*180):.1f} pts/hr")

    print("\nLoading arrival data...")
    hourlyCounts = buildHourlyCounts(SOURCE_FILE)
    print(f"  Built ({hourlyCounts.shape[0]} days × {hourlyCounts.shape[1]} hours) "
          f"count matrix")

    print("Fitting GP arrival rate model...")
    gpModel = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gpModel.fit(hourlyCounts)
    print(f"  Fitted  |  kernel: {gpModel._gp.kernel_}")

    global theParams
    theParams = loadParams(PARAMS_FILE)

    print("\nRUNNING MCE SIMULATION  (GP arrival rates + MCE superposition stream)")
    print(f"  {NUM_REPS} replications  |  {int(WARMUP_MIN)} min warmup  |  "
          f"{int(RUN_MIN)} min steady-state run")

    repResults = []
    for rep in range(NUM_REPS):
        rateFn = gpModel.sampleRateCurve(randomState=rep)
        repResults.append(runReplication(rateFn))
        if (rep + 1) % 10 == 0:
            print(f"  Completed replication {rep + 1:>3} / {NUM_REPS}")

    results = pd.DataFrame(repResults)
    decomp  = gpModel.varianceDecomposition(results["doctorWait"].values)

    printResults(results, decomp)

    print("\nGenerating plots...")
    plotOutputAnalysis(results)
    plotUtilisation(results)
    plotMceArrivalRate()
    plotGPPosterior(gpModel, hourlyCounts)

    outPath = os.path.join(RESULTS_DIR, "MCE_rep_results.csv")
    results.to_csv(outPath, index=False)
    print(f"  Saved → {outPath}")


if __name__ == "__main__":
    main()

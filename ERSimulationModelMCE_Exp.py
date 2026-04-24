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
from sim_engine.analysis_utils import ci as ci95   # single source of truth for CI calculation


# File paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "Sources", "er_5000_patients.csv")
PARAMS_FILE = os.path.join(BASE_DIR, "Sources", "simrng_parameters.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "simulation", "mce_experimental")


# Simulation run parameters
NUM_REPS   = 100
WARMUP_MIN = 480.0
RUN_MIN    = 1440.0


# System configuration
# Registration and triage shared
# Doctors dedicated by severity
N_CLERKS  = 1
N_NURSES  = 2

N_DOCTORS_HIGH   = 3
N_DOCTORS_MEDIUM = 2
N_DOCTORS_LOW    = 1

N_DOCTORS = N_DOCTORS_HIGH + N_DOCTORS_MEDIUM + N_DOCTORS_LOW

DOCTOR_SCHEDULE = []


# Random number streams
STREAM_ARRIVAL      = 1
STREAM_SEVERITY     = 2
STREAM_REG          = 3
STREAM_TRIAGE       = 4
STREAM_DOCTOR       = 5
STREAM_MCE_ARRIVAL  = 6
STREAM_MCE_SEVERITY = 7


# Severity setup
SEVERITIES = ["low", "medium", "high"]
PRIORITY   = {"high": 0, "medium": 1, "low": 2}

SEV_PROBS  = [0.5556, 0.9044, 1.0000]
SEV_LABELS = {1: "low", 2: "medium", 3: "high"}

# MCE severity mix: low 10%, medium 20%, high 70%
MCE_SEV_PROBS = [0.10, 0.30, 1.00]


# MCE parameters
MCE_START_MIN = 1080.0
MCE_DURATION  = 180.0
MCE_PEAK_RATE = 20.0
MCE_DECAY     = math.log(MCE_PEAK_RATE) / MCE_DURATION


# Plot colors
SEV_COLORS = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
METRIC_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]


# Validation targets
VALIDATION = {
    "regWait":    4.67,
    "triageWait": 0.01,
    "doctorWait": 249.67,
    "LOS":        285.63,
}


# Module-level state
_currentRateFn = None
theParams = {}


# Helper functions

def ensureResultsDir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def buildHourlyCounts(filepath: str) -> np.ndarray:
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
    entry = params[stage][severity]
    func  = entry[0]
    if func == "Lognormal":
        return SimRNG.Lognormal(entry[1], entry[2], stream)
    if func == "Erlang":
        return SimRNG.Erlang(entry[1], entry[2], stream)
    if func == "Expon":
        return SimRNG.Expon(entry[1], stream)
    if func == "Normal":
        return max(0.0, SimRNG.Normal(entry[1], entry[2], stream))
    if func == "Triangular":
        return SimRNG.Triangular(entry[1], entry[2], entry[3], stream)
    return SimRNG.Expon(entry[1], stream)


# Arrival logic

def nextInterarrival() -> float:
    tHours = (SimClasses.Clock % 1440) / 60
    rate   = max(float(_currentRateFn(tHours)), 0.01)
    return SimRNG.Expon(60.0 / rate, STREAM_ARRIVAL)


def nextMceInterarrival() -> float:
    elapsed = max(SimClasses.Clock - MCE_START_MIN, 0.0)
    rate = max(MCE_PEAK_RATE * math.exp(-MCE_DECAY * elapsed), 0.01)
    return SimRNG.Expon(60.0 / rate, STREAM_MCE_ARRIVAL)


def assignSeverity() -> str:
    return SEV_LABELS[SimRNG.Random_integer(SEV_PROBS, STREAM_SEVERITY)]


def assignMceSeverity() -> str:
    return SEV_LABELS[SimRNG.Random_integer(MCE_SEV_PROBS, STREAM_MCE_SEVERITY)]


# Priority queue

class PriorityQueue:
    """
    Priority queue serving by acuity (high > medium > low), FIFO within class.
    Three deques — O(1) Add and Remove.
    """
    def __init__(self):
        self.WIP    = SimClasses.CTStat()
        self._lanes: dict[int, deque] = {0: deque(), 1: deque(), 2: deque()}

    def NumQueue(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def Add(self, patient) -> None:
        self._lanes[PRIORITY[patient.severity]].append(patient)
        self.WIP.Record(float(self.NumQueue()))

    def Remove(self):
        for key in (0, 1, 2):
            if self._lanes[key]:
                entity = self._lanes[key].popleft()
                self.WIP.Record(float(self.NumQueue()))
                return entity
        return None

    def Mean(self) -> float:
        return self.WIP.Mean()


# Patient entity

class Patient(SimClasses.Entity):
    def __init__(self, severity: str, isMce: bool = False):
        super().__init__()
        self.severity     = severity
        self.isMce        = isMce
        self.reg_start    = 0.0
        self.reg_end      = 0.0
        self.triage_start = 0.0
        self.triage_end   = 0.0
        self.doc_start    = 0.0


# Simulation objects

zSimRNG = SimRNG.InitializeRNSeed()

clerk = SimClasses.Resource()
nurses = SimClasses.Resource()

doctorHigh = SimClasses.Resource()
doctorMed  = SimClasses.Resource()
doctorLow  = SimClasses.Resource()

clerk.SetUnits(N_CLERKS)
nurses.SetUnits(N_NURSES)
doctorHigh.SetUnits(N_DOCTORS_HIGH)
doctorMed.SetUnits(N_DOCTORS_MEDIUM)
doctorLow.SetUnits(N_DOCTORS_LOW)

regQueue = SimClasses.FIFOQueue()
triageQueue = PriorityQueue()

doctorQueueHigh = SimClasses.FIFOQueue()
doctorQueueMed  = SimClasses.FIFOQueue()
doctorQueueLow  = SimClasses.FIFOQueue()

regWait    = SimClasses.DTStat()
triageWait = SimClasses.DTStat()
doctorWait = SimClasses.DTStat()
LOS        = SimClasses.DTStat()

regWaitBySev    = {sev: SimClasses.DTStat() for sev in SEVERITIES}
triageWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
doctorWaitBySev = {sev: SimClasses.DTStat() for sev in SEVERITIES}
losBySev        = {sev: SimClasses.DTStat() for sev in SEVERITIES}

doctorWaitMce = SimClasses.DTStat()
losMce        = SimClasses.DTStat()
doctorWaitNormal = SimClasses.DTStat()
losNormal        = SimClasses.DTStat()

calendar = SimClasses.EventCalendar()

theCTStats = []
theDTStats = [
    regWait, triageWait, doctorWait, LOS,
    *regWaitBySev.values(),
    *triageWaitBySev.values(),
    *doctorWaitBySev.values(),
    *losBySev.values(),
    doctorWaitMce, losMce,
    doctorWaitNormal, losNormal,
]

theQueues = [
    regQueue,
    triageQueue,
    doctorQueueHigh,
    doctorQueueMed,
    doctorQueueLow,
]

theResources = [
    clerk,
    nurses,
    doctorHigh,
    doctorMed,
    doctorLow,
]


# Doctor routing helpers

def getDoctorQueue(severity: str):
    if severity == "high":
        return doctorQueueHigh
    if severity == "medium":
        return doctorQueueMed
    return doctorQueueLow


def getDoctorResource(severity: str):
    if severity == "high":
        return doctorHigh
    if severity == "medium":
        return doctorMed
    return doctorLow


# Event handlers: normal arrivals

def arrival() -> None:
    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())
    p = Patient(assignSeverity(), isMce=False)
    regQueue.Add(p)
    if clerk.Busy == 0:
        startRegistration()


def startRegistration() -> None:
    p = regQueue.Remove()
    if p is None:
        return
    p.reg_start = SimClasses.Clock
    clerk.Seize(1)
    SimFunctions.SchedulePlus(
        calendar,
        "endRegistration",
        drawService(theParams, "reg", p.severity, STREAM_REG),
        p,
    )


def endRegistration(ev) -> None:
    p = ev.WhichObject
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
    p = triageQueue.Remove()
    if p is None:
        return
    p.triage_start = SimClasses.Clock
    nurses.Seize(1)
    SimFunctions.SchedulePlus(
        calendar,
        "endTriage",
        drawService(theParams, "triage", p.severity, STREAM_TRIAGE),
        p,
    )


def endTriage(ev) -> None:
    p = ev.WhichObject
    p.triage_end = SimClasses.Clock

    if p.CreateTime >= WARMUP_MIN:
        triageWait.Record(p.triage_start - p.reg_end)
        triageWaitBySev[p.severity].Record(p.triage_start - p.reg_end)

    nurses.Free(1)

    q = getDoctorQueue(p.severity)
    r = getDoctorResource(p.severity)

    q.Add(p)
    if r.Busy < r.NumberOfUnits:
        startDoctor(p.severity)

    if triageQueue.NumQueue() > 0:
        startTriage()


def startDoctor(severity: str) -> None:
    q = getDoctorQueue(severity)
    r = getDoctorResource(severity)

    p = q.Remove()
    if p is None:
        return

    p.doc_start = SimClasses.Clock
    r.Seize(1)

    SimFunctions.SchedulePlus(
        calendar,
        "endDoctor",
        drawService(theParams, "doctor", p.severity, STREAM_DOCTOR),
        p,
    )


def endDoctor(ev) -> None:
    p = ev.WhichObject
    q = getDoctorQueue(p.severity)
    r = getDoctorResource(p.severity)

    if p.CreateTime >= WARMUP_MIN:
        dw = p.doc_start - p.triage_end
        ls = SimClasses.Clock - p.CreateTime

        doctorWait.Record(dw)
        doctorWaitBySev[p.severity].Record(dw)

        LOS.Record(ls)
        losBySev[p.severity].Record(ls)

        if p.isMce:
            doctorWaitMce.Record(dw)
            losMce.Record(ls)
        else:
            doctorWaitNormal.Record(dw)
            losNormal.Record(ls)

    r.Free(1)

    if q.NumQueue() > 0:
        startDoctor(p.severity)


# Event handlers: MCE arrivals

def mceArrival() -> None:
    p = Patient(assignMceSeverity(), isMce=True)
    regQueue.Add(p)

    if clerk.Busy == 0:
        startRegistration()

    if SimClasses.Clock < MCE_START_MIN + MCE_DURATION:
        SimFunctions.Schedule(calendar, "mceArrival", nextMceInterarrival())


def mceStart(ev) -> None:
    print(
        f"  [MCE] Event started at t={SimClasses.Clock:.1f} min "
        f"(clock = {SimClasses.Clock / 60:.1f} hr, "
        f"initial rate = {MCE_PEAK_RATE:.1f} pts/hr)"
    )
    SimFunctions.Schedule(calendar, "mceArrival", nextMceInterarrival())


def shiftChange(ev) -> None:
    return


# Utility helpers

def clearQueueStats() -> None:
    for q in [triageQueue, doctorQueueHigh, doctorQueueMed, doctorQueueLow]:
        if hasattr(q, "WIP"):
            q.WIP.Clear()
            q.WIP.Xlast = 0.0




# Replication runner

def runReplication(rateFn: callable) -> dict:
    global _currentRateFn
    _currentRateFn = rateFn

    clerk.SetUnits(N_CLERKS)
    nurses.SetUnits(N_NURSES)
    doctorHigh.SetUnits(N_DOCTORS_HIGH)
    doctorMed.SetUnits(N_DOCTORS_MEDIUM)
    doctorLow.SetUnits(N_DOCTORS_LOW)

    SimFunctions.SimFunctionsInit(
        calendar, theQueues, theCTStats, theDTStats, theResources
    )
    clearQueueStats()

    SimFunctions.Schedule(calendar, "arrival", nextInterarrival())
    SimFunctions.Schedule(calendar, "mceStart", MCE_START_MIN)

    warmupCleared = False

    while SimClasses.Clock < WARMUP_MIN + RUN_MIN:
        ev = calendar.Remove()
        SimClasses.Clock = ev.EventTime

        if not warmupCleared and SimClasses.Clock >= WARMUP_MIN:
            SimFunctions.ClearStats(theCTStats, theDTStats)
            clearQueueStats()
            warmupCleared = True

        if ev.EventType == "arrival":
            arrival()
        elif ev.EventType == "endRegistration":
            endRegistration(ev)
        elif ev.EventType == "endTriage":
            endTriage(ev)
        elif ev.EventType == "endDoctor":
            endDoctor(ev)
        elif ev.EventType == "shiftChange":
            shiftChange(ev)
        elif ev.EventType == "mceStart":
            mceStart(ev)
        elif ev.EventType == "mceArrival":
            mceArrival()
        else:
            raise RuntimeError(f"Unknown event type in calendar: {ev.EventType!r}")

    total_doc_busy = doctorHigh.Mean() + doctorMed.Mean() + doctorLow.Mean()

    row = {
        "regWait":    regWait.Mean(),
        "triageWait": triageWait.Mean(),
        "doctorWait": doctorWait.Mean(),
        "LOS":        LOS.Mean(),
        "ClerkUtil":  clerk.Mean() / N_CLERKS,
        "NurseUtil":  nurses.Mean() / N_NURSES,
        "DoctorUtil": total_doc_busy / N_DOCTORS,
        "DoctorUtil_high":   doctorHigh.Mean() / N_DOCTORS_HIGH,
        "DoctorUtil_medium": doctorMed.Mean() / N_DOCTORS_MEDIUM,
        "DoctorUtil_low":    doctorLow.Mean() / N_DOCTORS_LOW,
        "doctorWait_mce":    doctorWaitMce.Mean(),
        "LOS_mce":           losMce.Mean(),
        "doctorWait_normal": doctorWaitNormal.Mean(),
        "LOS_normal":        losNormal.Mean(),
    }

    for sev in SEVERITIES:
        row[f"regWait_{sev}"]    = regWaitBySev[sev].Mean()
        row[f"triageWait_{sev}"] = triageWaitBySev[sev].Mean()
        row[f"doctorWait_{sev}"] = doctorWaitBySev[sev].Mean()
        row[f"LOS_{sev}"]        = losBySev[sev].Mean()

    return row


# Output and plotting

def printResults(results: pd.DataFrame, decomp: dict) -> None:
    print(f"\nMCE + DEDICATED DOCTOR RESULTS  ({NUM_REPS} replications, 95% CI)")
    print(
        f"  Doctor config: high={N_DOCTORS_HIGH}, medium={N_DOCTORS_MEDIUM}, low={N_DOCTORS_LOW}"
    )
    print(
        f"  MCE at {MCE_START_MIN/60:.0f}:00 for {MCE_DURATION/60:.0f} hr | peak {MCE_PEAK_RATE:.0f} pts/hr"
    )

    metrics = [
        ("regWait",    "Reg wait                 (min)"),
        ("triageWait", "Triage wait              (min)"),
        ("doctorWait", "Doctor wait              (min)"),
        ("LOS",        "Length of stay           (min)"),
        ("doctorWait_mce",    "Doctor wait — MCE only    (min)"),
        ("LOS_mce",           "LOS — MCE only            (min)"),
        ("doctorWait_normal", "Doctor wait — normal only (min)"),
        ("LOS_normal",        "LOS — normal only         (min)"),
        ("ClerkUtil",  "Clerk util               (%)  "),
        ("NurseUtil",  "Nurse util               (%)  "),
        ("DoctorUtil", "Doctor util overall      (%)  "),
    ]

    for col, label in metrics:
        m, hw = ci95(results[col])
        scale = 100 if "util" in col.lower() else 1
        print(f"  {label}: {m * scale:8.3f}  ±  {hw * scale:.3f}")

    print("\n  DEDICATED DOCTOR UTILISATION (95% CI)")
    for col, label in [
        ("DoctorUtil_high", "High severity doctor util   (%)"),
        ("DoctorUtil_medium", "Medium severity doctor util (%)"),
        ("DoctorUtil_low", "Low severity doctor util    (%)"),
    ]:
        m, hw = ci95(results[col])
        print(f"  {label}: {m * 100:8.3f}  ±  {hw * 100:.3f}")

    print("\n  WAIT TIMES BY SEVERITY (95% CI)")
    print(f"  {'Severity':<10} {'RegWait':>14}  {'TriageWait':>14}  {'DoctorWait':>14}  {'LOS':>14}")
    for sev in SEVERITIES:
        rw_m, rw_hw = ci95(results[f"regWait_{sev}"])
        tw_m, tw_hw = ci95(results[f"triageWait_{sev}"])
        dw_m, dw_hw = ci95(results[f"doctorWait_{sev}"])
        ls_m, ls_hw = ci95(results[f"LOS_{sev}"])
        print(
            f"  {sev:<10} "
            f"{rw_m:7.2f} ± {rw_hw:5.2f}   "
            f"{tw_m:7.2f} ± {tw_hw:5.2f}   "
            f"{dw_m:7.2f} ± {dw_hw:5.2f}   "
            f"{ls_m:7.2f} ± {ls_hw:5.2f}"
        )

    print("\n  VARIANCE DECOMPOSITION (doctor wait)")
    print(f"  Input uncertainty : {decomp['inputVar']:8.4f}  ({decomp['inputFraction'] * 100:.1f}%)")
    print(f"  Simulation noise  : {decomp['simVar']:8.4f}  ({decomp['simFraction'] * 100:.1f}%)")
    print(f"  Total variance    : {decomp['totalVar']:8.4f}")

    print(f"\nResults saved in: {os.path.abspath(RESULTS_DIR)}/")


def _save(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plotUtilisation(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("MCE + Dedicated Resource Utilisation", fontsize=14, fontweight="bold")

    cols = [
        "ClerkUtil",
        "NurseUtil",
        "DoctorUtil_high",
        "DoctorUtil_medium",
        "DoctorUtil_low",
        "DoctorUtil",
    ]
    labels = [
        "clerk\n(1 unit)",
        "nurses\n(2 units)",
        "high docs\n(3 units)",
        "med docs\n(2 units)",
        "low docs\n(1 unit)",
        "all docs\n(6 units)",
    ]
    means  = [ci95(results[c])[0] * 100 for c in cols]
    errors = [ci95(results[c])[1] * 100 for c in cols]

    bars = ax.bar(labels, means, yerr=errors, capsize=5)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_ylabel("Mean Utilisation (%)")
    ax.set_ylim(0, 120)

    for bar, val, err in zip(bars, means, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    _save(fig, "MCE_EXP_utilisation.png")


def plotMceArrivalRate() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, MCE_DURATION, 300)
    rates = MCE_PEAK_RATE * np.exp(-MCE_DECAY * t)

    ax.plot(t, rates, linewidth=2.5)
    ax.fill_between(t, 0, rates, alpha=0.15)

    ax.set_xlabel("Minutes since MCE onset")
    ax.set_ylabel("Arrival rate (patients / hr)")
    ax.set_title("MCE Arrival Rate Decay", fontweight="bold")
    ax.set_xlim(0, MCE_DURATION)

    plt.tight_layout()
    _save(fig, "MCE_EXP_arrival_rate.png")


def plotOutputAnalysis(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MCE + Dedicated Doctors Output Analysis", fontsize=14, fontweight="bold")

    plotCfg = [
        ("regWait",    "Registration Wait (min)", METRIC_COLORS[0]),
        ("triageWait", "Triage Wait (min)", METRIC_COLORS[3]),
        ("doctorWait", "Doctor Wait (min)", METRIC_COLORS[1]),
        ("LOS",        "Length of Stay (min)", METRIC_COLORS[2]),
    ]

    for ax, (col, title, color) in zip(axes.flat, plotCfg):
        data = results[col]
        m, hw = ci95(results[col])

        ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(m, color="black", linestyle="-", linewidth=2, label=f"Mean = {m:.2f}")
        ax.axvline(m - hw, color="black", linestyle="--", linewidth=1.2, label=f"95% CI ± {hw:.2f}")
        ax.axvline(m + hw, color="black", linestyle="--", linewidth=1.2)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "MCE_EXP_output_analysis.png")


def plotGPPosterior(gpModel: ArrivalRateGP, hourlyCounts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    gpModel.plotPosterior(
        nSamples=30,
        title="GP Posterior — ED Arrival Rate",
        ax=ax,
    )
    empiricalMeans = hourlyCounts.mean(axis=0)
    ax.scatter(
        np.arange(24) + 0.5,
        empiricalMeans,
        color="#e74c3c",
        zorder=5,
        s=60,
        label="Empirical mean",
        marker="D",
    )
    ax.axvline(MCE_START_MIN / 60, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"MCE onset ({MCE_START_MIN/60:.0f}:00)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "MCE_EXP_gp_posterior.png")


# Main

def main() -> None:
    ensureResultsDir(RESULTS_DIR)

    print("\nLoading arrival data...")
    hourlyCounts = buildHourlyCounts(SOURCE_FILE)
    print(f"  Built ({hourlyCounts.shape[0]} days × {hourlyCounts.shape[1]} hours) count matrix")

    print("Fitting GP arrival rate model...")
    gpModel = ArrivalRateGP(period=24.0, nRestarts=20, randomState=42)
    gpModel.fit(hourlyCounts)
    print(f"  Fitted  |  kernel: {gpModel._gp.kernel_}")

    global theParams
    theParams = loadParams(PARAMS_FILE)

    print("\nRUNNING MCE + DEDICATED DOCTOR SIMULATION")
    print(f"  {NUM_REPS} replications  |  warmup {int(WARMUP_MIN)} min  |  run {int(RUN_MIN)} min")
    print(f"  Doctor capacities: high={N_DOCTORS_HIGH}, medium={N_DOCTORS_MEDIUM}, low={N_DOCTORS_LOW}")
    print(f"  MCE starts at {MCE_START_MIN/60:.0f}:00 with peak {MCE_PEAK_RATE:.0f} pts/hr")

    repResults = []
    for rep in range(NUM_REPS):
        rateFn = gpModel.sampleRateCurve(randomState=rep)
        repResults.append(runReplication(rateFn))
        if (rep + 1) % 10 == 0:
            print(f"  Completed replication {rep + 1:>3} / {NUM_REPS}")

    results = pd.DataFrame(repResults)
    decomp = gpModel.varianceDecomposition(results["doctorWait"].values)

    printResults(results, decomp)

    print("\nGenerating plots...")
    plotOutputAnalysis(results)
    plotUtilisation(results)
    plotMceArrivalRate()
    plotGPPosterior(gpModel, hourlyCounts)

    outPath = os.path.join(RESULTS_DIR, "MCE_experimental_rep_results.csv")
    results.to_csv(outPath, index=False)
    print(f"  Saved → {outPath}")


if __name__ == "__main__":
    main()
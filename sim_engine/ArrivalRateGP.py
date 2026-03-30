"""
ArrivalRateGP.py — Gaussian Process model for NHPP arrival rate estimation.

Fits a GP in log-space to binned arrival count data and produces posterior
rate curve samples for two-level simulation input uncertainty analysis.

Domain-agnostic: accepts any (n_days, n_bins) integer count matrix and
returns callable rate functions. Knows nothing about patients or EDs.

Kernel: locally periodic (ExpSineSquared × RBF) + WhiteKernel noise.
The period is fixed at n_bins. The RBF component allows the amplitude
envelope to vary rather than enforcing a perfectly rigid periodic pattern.

Reference:
  Møller, Syversveen, Waagepetersen (1998). Log Gaussian Cox Processes.
  Barton, Nelson, Xie (2014). Input uncertainty in stochastic simulation.

Author  : Tilak
Input   : (n_days, n_bins) numpy array of arrival counts
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, ExpSineSquared, RBF, WhiteKernel,
)


class ArrivalRateGP:
    """
    GP-based arrival rate model for NHPP simulation.

    Fits a Gaussian Process to log-transformed hourly mean counts. The
    posterior represents uncertainty about the true rate function given
    finite observation data. Call fit() once, then sampleRateCurve() per
    replication to implement the two-level input uncertainty protocol.
    """

    def __init__(self, period: float = 24.0, nRestarts: int = 20,
                 randomState: int = 42):
        self.period      = period
        self.nRestarts   = nRestarts
        self.randomState = randomState

        self._gp      = None
        self._xFine   = None
        self._nBins   = 0
        self._fitted  = False

    def fit(self, counts: np.ndarray) -> "ArrivalRateGP":
        """
        Fit the GP to observed arrival counts.

        counts must be a 2-D array of shape (n_days, n_bins) where each
        cell is the integer number of arrivals in that bin on that day.
        Returns self to allow method chaining.
        """
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 2:
            raise ValueError("counts must be shape (n_days, n_bins)")

        nDays, nBins = counts.shape
        self._nBins  = nBins

        # Bin centroids as GP training inputs: 0.5, 1.5, ..., nBins - 0.5
        xTrain = np.arange(nBins).reshape(-1, 1) + 0.5

        meanCounts = counts.mean(axis=0)
        varCounts  = counts.var(axis=0, ddof=1)

        # Log-transform the per-bin means; epsilon guards against zero bins
        yLog = np.log(meanCounts + 1e-6)

        # Observation noise in log-space via the delta method:
        # Var[log(X̄)] ≈ Var[X̄] / X̄²  where Var[X̄] = sample_var / nDays
        alpha = (varCounts / nDays) / (meanCounts ** 2 + 1e-6)
        alpha = np.clip(alpha, 1e-4, 10.0)

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.01, 100.0))
            * ExpSineSquared(
                length_scale=2.0,
                periodicity=float(self.period),
                length_scale_bounds=(0.3, 12.0),
                periodicity_bounds="fixed",
            )
            * RBF(
                length_scale=float(nBins) * 2.0,
                length_scale_bounds=(float(nBins) * 0.5, float(nBins) * 10.0),
            )
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 2.0))
        )

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=self.nRestarts,
            normalize_y=True,
            random_state=self.randomState,
        )
        self._gp.fit(xTrain, yLog)

        # Fine-resolution query grid — 4 points per bin for smooth cubic interpolation
        self._xFine  = np.linspace(0, nBins, nBins * 4 + 1).reshape(-1, 1)
        self._fitted = True
        return self

    def _checkFitted(self) -> None:
        """Raise if fit() has not been called."""
        if not self._fitted:
            raise RuntimeError("Call fit() before sampling or querying the GP.")

    def meanRateCurve(self) -> callable:
        """
        Return the GP posterior mean as a callable rate function.
        Maps a fractional bin index to a rate in the same units as the input counts.
        """
        self._checkFitted()
        logMean, _ = self._gp.predict(self._xFine, return_std=True)
        rates      = np.exp(logMean)
        return interp1d(
            self._xFine.ravel(), rates,
            kind="cubic", bounds_error=False,
            fill_value=(rates[0], rates[-1]),
        )

    def sampleRateCurve(self, randomState: int | None = None) -> callable:
        """
        Draw one rate curve from the GP posterior.

        Each call returns a different plausible rate function consistent with
        the observed data. Passing one into each simulation replication
        implements the outer loop of the two-level input uncertainty protocol.
        Returns a callable mapping fractional bin index to a non-negative rate.
        """
        self._checkFitted()
        logSample = self._gp.sample_y(
            self._xFine, n_samples=1, random_state=randomState
        ).ravel()
        rates = np.maximum(np.exp(logSample), 1e-6)
        return interp1d(
            self._xFine.ravel(), rates,
            kind="cubic", bounds_error=False,
            fill_value=(rates[0], rates[-1]),
        )

    def posteriorSamples(self, nSamples: int = 100,
                         randomState: int | None = None) -> np.ndarray:
        """
        Draw nSamples rate curves from the GP posterior at once.
        Returns an array of shape (n_query_points, nSamples) in rate units.
        """
        self._checkFitted()
        logSamples = self._gp.sample_y(
            self._xFine, n_samples=nSamples, random_state=randomState
        )
        return np.maximum(np.exp(logSamples), 1e-6)

    def plotPosterior(self, nSamples: int = 20,
                      title: str = "GP Posterior — Arrival Rate",
                      ax: plt.Axes | None = None) -> plt.Axes:
        """
        Diagnostic plot: posterior mean, 95% credible band, and sample curves.
        Draws into ax if provided, otherwise creates a new figure.
        """
        self._checkFitted()

        logMean, logStd = self._gp.predict(self._xFine, return_std=True)
        x          = self._xFine.ravel()
        meanRates  = np.exp(logMean)
        upper      = np.exp(logMean + 1.96 * logStd)
        lower      = np.maximum(np.exp(logMean - 1.96 * logStd), 0)
        samples    = self.posteriorSamples(nSamples=nSamples,
                                           randomState=self.randomState)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        for i in range(nSamples):
            ax.plot(x, samples[:, i], color="#3498db", alpha=0.15, linewidth=0.8)

        ax.fill_between(x, lower, upper, alpha=0.25, color="#3498db",
                        label="95% credible band")
        ax.plot(x, meanRates, color="#2c3e50", linewidth=2.5, label="Posterior mean")
        ax.set_xlabel(f"Hour of day (Range: 0-{self._nBins-1}, bin width: {self.period / self._nBins:.2f} hours)")
        ax.set_ylabel("Arrivals per hour")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, self._nBins)
        return ax

    def varianceDecomposition(self, results: np.ndarray) -> dict:
        """
        Decompose total output variance into input uncertainty and simulation
        noise using the law of total variance: Var(Y) = Var[E(Y|θ)] + E[Var(Y|θ)].

        results must be shape (nOuter,) or (nOuter, nInner). When nInner == 1
        (one replication per rate curve) the simulation noise term is estimated
        from the between-curve spread, which underestimates it slightly but is
        acceptable when nOuter is large.

        Returns a dict with keys: inputVar, simVar, totalVar,
        inputFraction, simFraction.
        """
        if results.ndim == 1:
            results = results.reshape(-1, 1)

        innerMeans = results.mean(axis=1)
        innerVars  = (results.var(axis=1, ddof=1)
                      if results.shape[1] > 1
                      else np.zeros(len(results)))

        inputVar = float(np.var(innerMeans, ddof=1))
        simVar   = float(np.mean(innerVars) / results.shape[1])
        totalVar = inputVar + simVar

        return {
            "inputVar":      inputVar,
            "simVar":        simVar,
            "totalVar":      totalVar,
            "inputFraction": inputVar / totalVar if totalVar > 0 else 0.0,
            "simFraction":   simVar   / totalVar if totalVar > 0 else 0.0,
        }
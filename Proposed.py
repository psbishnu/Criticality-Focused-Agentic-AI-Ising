#!/usr/bin/env python3
"""
PRE_agentic_ising_pipeline.py

Current lattice sizes
---------------------
L = 16, 32, 64

Expected CSV format
-------------------
Temperature,Phase,spin_0,spin_1,...,spin_(L*L-1)

IMPORTANT SCIENTIFIC NOTE
-------------------------
This script operates on PRECOMPUTED datasets and treats each temperature as
an "oracle query". Therefore:
    - temperature-query reduction and configuration-usage reduction are valid;
    - raw Monte Carlo CPU-time savings are NOT directly measured here because
      all datasets already exist.
For a strong computational-efficiency claim in the final paper, pair these
results with an on-demand Monte Carlo experiment where only agent-selected
temperatures are simulated.

Exact 2D Ising values are used ONLY for final evaluation/plot reference.
They are never used for temperature acquisition, stopping, Binder crossing,
finite-size scaling, or exponent estimation.
"""

# ======================================================================
# 0. ALL USER / EXPERIMENT PARAMETERS -- EDIT ONLY THIS BLOCK
# ======================================================================

from pathlib import Path

DATASETS = {
    32: Path("../JOB5_Noise/J5Data/MCD32.csv"),
    64: Path("../JOB5_Noise/J5Data/MCD64.csv"),
    128: Path("../JOB5_Noise/J5Data/MCD128.csv"),
}

LATTICE_SIZES = [32, 64, 128]

# Ising physics
J = 1.0
K_B = 1.0
EXTERNAL_FIELD = 0.0

TEMPERATURE_COLUMN = "Temperature"
PHASE_COLUMN = "Phase"
SPIN_PREFIX = "spin_"

# ------------------ Agent parameters ------------------
N_INITIAL_TEMPERATURES = 6
MAX_TEMPERATURE_QUERIES = 20
MIN_QUERIES_BEFORE_STOP = 12
STABILITY_WINDOW = 5
TC_STABILITY_TOL = 0.015

GP_LENGTH_SCALE = 0.20
GP_NOISE_LEVEL = 1e-6

EXPLORATION_WEIGHT = 0.40
CRITICALITY_WEIGHT = 0.45
BRACKETING_WEIGHT = 0.15

CRITICALITY_WIDTH = 0.20
BRACKET_WIDTH = 0.18

# ------------------ Binder / FSS ------------------
BINDER_INTERPOLATION_POINTS = 4000
BINDER_APPROX_TOL = 0.025
BINDER_SLOPE_WINDOW = 0.30
BINDER_SLOPE_MIN_POINTS = 3

# ------------------ Bootstrap uncertainty ------------------
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 2026
CI_LOW = 2.5
CI_HIGH = 97.5

# ------------------ Optional time-series diagnostics ------------------
# Set True ONLY if, within each temperature, rows preserve chronological
# Monte Carlo sampling order. Otherwise autocorrelation/ESS are invalid.
ROWS_ARE_TIME_ORDERED = False
MAX_AUTOCORR_LAG = 500

# ------------------ Reproducibility ------------------
RANDOM_SEED = 42

# ------------------ Outputs ------------------
RESULTS_ROOT = Path("Results")
COMBINED_DIRNAME = "PRE_Combined"

# Figures: PDF ONLY
FIG_WIDTH = 11
FIG_HEIGHT = 8
FONT_SIZE = 17
AXIS_FONT_SIZE = 19
TITLE_FONT_SIZE = 21
LEGEND_FONT_SIZE = 13
LINE_WIDTH = 3.0
MARKER_SIZE = 65

# ------------------ Exact values: evaluation only ------------------
USE_EXACT_VALUES_FOR_EVALUATION = True

EXACT_TC = 2.0 / __import__("math").log(1.0 + __import__("math").sqrt(2.0))
EXACT_BETA = 1.0 / 8.0
EXACT_GAMMA = 7.0 / 4.0
EXACT_NU = 1.0

# ======================================================================

import argparse
import gc
import math
import time
import warnings
import resource
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.weight": "bold",
    "axes.labelsize": AXIS_FONT_SIZE,
    "axes.labelweight": "bold",
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": TITLE_FONT_SIZE,
})


# ======================================================================
# 1. Utility functions
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-queries",
        type=int,
        default=MAX_TEMPERATURE_QUERIES,
        help="Maximum number of temperature queries per lattice size.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap replicates.",
    )
    return p.parse_args()


def save_pdf(fig, path_without_suffix):
    fig.tight_layout()
    fig.savefig(str(path_without_suffix) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def spin_sort_key(name):
    return int(name.split("_")[-1])


def normalize(v):
    v = np.asarray(v, dtype=float)
    finite = np.isfinite(v)
    if not np.any(finite):
        return np.zeros_like(v)
    vv = v.copy()
    vv[~finite] = np.nanmin(vv[finite])
    lo, hi = np.min(vv), np.max(vv)
    if hi - lo < 1e-14:
        return np.zeros_like(vv)
    return (vv - lo) / (hi - lo)


def parabolic_peak_tc(T, y):
    """Estimate peak location using a local quadratic fit."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(T) & np.isfinite(y)
    T, y = T[ok], y[ok]
    if len(T) == 0:
        return np.nan

    order = np.argsort(T)
    T, y = T[order], y[order]
    i = int(np.argmax(y))

    if 0 < i < len(T) - 1:
        x3 = T[i - 1:i + 2]
        y3 = y[i - 1:i + 2]
        try:
            a, b, _ = np.polyfit(x3, y3, 2)
            if a < 0:
                xv = -b / (2.0 * a)
                if x3.min() <= xv <= x3.max():
                    return float(xv)
        except Exception:
            pass

    return float(T[i])


def ci(values, low=CI_LOW, high=CI_HIGH):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.median(a)),
        float(np.percentile(a, low)),
        float(np.percentile(a, high)),
    )


def interpolate_at(df, column, x):
    d = df[["Temperature", column]].dropna().sort_values("Temperature")
    if len(d) < 2:
        return np.nan
    T = d["Temperature"].to_numpy(float)
    y = d[column].to_numpy(float)
    if x < T.min() or x > T.max():
        return np.nan
    return float(np.interp(x, T, y))


# ======================================================================
# 2. Dataset loading and observables
# ======================================================================

def load_raw_observables(path: Path, lattice: int):
    """
    Load dataset with compact spin dtype and immediately reduce each
    configuration to scalar observables.

    Returns a much smaller dataframe:
        Temperature, m, abs_m, e, c1
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    if TEMPERATURE_COLUMN not in header:
        raise ValueError(f"{path}: missing {TEMPERATURE_COLUMN}")

    spin_cols = sorted(
        [c for c in header if c.startswith(SPIN_PREFIX)],
        key=spin_sort_key,
    )

    expected = lattice * lattice
    if len(spin_cols) != expected:
        raise ValueError(
            f"{path}: L={lattice} requires {expected} spin columns; "
            f"found {len(spin_cols)}."
        )

    dtype_map = {c: np.int8 for c in spin_cols}
    dtype_map[TEMPERATURE_COLUMN] = np.float64

    usecols = [TEMPERATURE_COLUMN] + spin_cols
    df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)

    spins = df[spin_cols].to_numpy(dtype=np.int8, copy=False)
    vals = np.unique(spins)
    if not np.all(np.isin(vals, [-1, 1])):
        raise ValueError(f"{path}: spin values must be +/-1; found {vals[:10]}")

    T = df[TEMPERATURE_COLUMN].to_numpy(float)
    if np.any(T <= 0):
        raise ValueError(f"{path}: all temperatures must be > 0.")

    n = lattice * lattice
    s2 = spins.reshape(-1, lattice, lattice)

    m = spins.mean(axis=1)
    abs_m = np.abs(m)

    # Unique nearest-neighbour bonds: right + down => 2N bonds.
    bond_sum = (
        s2 * np.roll(s2, -1, axis=1)
        + s2 * np.roll(s2, -1, axis=2)
    ).sum(axis=(1, 2))

    e = -J * bond_sum / n - EXTERNAL_FIELD * m

    # C1 = mean nearest-neighbour correlation over unique bonds.
    # For h=0 on square lattice: E/N = -2 J C1.
    c1 = bond_sum / (2.0 * n)

    raw = pd.DataFrame({
        "Temperature": T,
        "m": m.astype(np.float64),
        "abs_m": abs_m.astype(np.float64),
        "e": e.astype(np.float64),
        "c1": c1.astype(np.float64),
    })

    del df, spins, s2, bond_sum
    gc.collect()

    return raw


def aggregate_temperature_group(T, g, lattice):
    n = lattice * lattice

    m = g["m"].to_numpy(float)
    am = g["abs_m"].to_numpy(float)
    e = g["e"].to_numpy(float)
    c1 = g["c1"].to_numpy(float)

    m2 = np.mean(m ** 2)
    m4 = np.mean(m ** 4)
    e2 = np.mean(e ** 2)

    chi = (n / (K_B * T)) * max(m2 - np.mean(am) ** 2, 0.0)
    cv = (n / (K_B * T ** 2)) * max(e2 - np.mean(e) ** 2, 0.0)

    binder = np.nan
    if m2 > 1e-14:
        binder = 1.0 - m4 / (3.0 * m2 ** 2)

    return {
        "Temperature": float(T),
        "N_configs": int(len(g)),
        "Abs_Magnetization": float(np.mean(am)),
        "Energy_per_spin": float(np.mean(e)),
        "C1": float(np.mean(c1)),
        "Susceptibility": float(chi),
        "Heat_Capacity": float(cv),
        "Binder_U4": float(binder),
    }


def aggregate_by_temperature(raw, lattice):
    rows = [
        aggregate_temperature_group(T, g, lattice)
        for T, g in raw.groupby("Temperature", sort=True)
    ]
    agg = pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)

    if len(agg) < 7:
        raise ValueError(
            f"L={lattice}: only {len(agg)} unique temperatures; "
            "at least 7 are recommended."
        )

    return agg


# ======================================================================
# 3. Optional autocorrelation / effective sample size
# ======================================================================

def integrated_autocorrelation_time(x, max_lag=MAX_AUTOCORR_LAG):
    """
    FFT autocorrelation with initial-positive truncation.
    Use ONLY if rows are chronological MC samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 8:
        return np.nan

    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 1e-15:
        return 0.5

    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
    acf /= np.arange(n, 0, -1)
    acf /= acf[0]

    m = min(max_lag, n - 1)
    tau = 0.5
    for lag in range(1, m + 1):
        if acf[lag] <= 0:
            break
        tau += acf[lag]

    return float(max(tau, 0.5))


def time_series_diagnostics(raw):
    rows = []
    for T, g in raw.groupby("Temperature", sort=True):
        tau = integrated_autocorrelation_time(g["m"].to_numpy(float))
        ess = np.nan
        if np.isfinite(tau) and tau > 0:
            ess = len(g) / (2.0 * tau)

        rows.append({
            "Temperature": float(T),
            "N_configs": int(len(g)),
            "tau_int_magnetization": float(tau) if np.isfinite(tau) else np.nan,
            "effective_sample_size": float(ess) if np.isfinite(ess) else np.nan,
        })

    return pd.DataFrame(rows)


# ======================================================================
# 4. Physics-informed agent
# ======================================================================

@dataclass
class AgentDecision:
    index: int
    temperature: float
    acquisition: float
    predicted_tc: float
    reason: str


class PhysicsInformedAgent:
    """
    Autonomous temperature selection.

    The agent sees only queried temperatures.
    Exact Tc and exact exponents are never used.
    """

    def __init__(self, full_aggregate, max_queries):
        self.data = full_aggregate.copy()
        self.temps = self.data["Temperature"].to_numpy(float)
        self.max_queries = min(max_queries, len(self.data))
        self.selected = []
        self.memory = []
        self.tc_history = []

    def initialize(self):
        n0 = min(N_INITIAL_TEMPERATURES, self.max_queries)
        idx = np.linspace(0, len(self.temps) - 1, n0, dtype=int)
        self.selected = list(dict.fromkeys(idx.tolist()))

        while len(self.selected) < n0:
            candidates = [i for i in range(len(self.temps)) if i not in self.selected]
            self.selected.append(candidates[0])

        for idx in self.selected.copy():
            self._record(idx, np.nan, "initial coarse exploration")

    def _fit_gp(self):
        ids = np.array(sorted(set(self.selected)), dtype=int)
        X = self.temps[ids].reshape(-1, 1)
        y = self.data.loc[ids, "Susceptibility"].to_numpy(float)

        scaler = StandardScaler()
        ys = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(
                length_scale=GP_LENGTH_SCALE,
                length_scale_bounds=(1e-3, 10.0),
            )
            + WhiteKernel(
                noise_level=GP_NOISE_LEVEL,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=RANDOM_SEED,
        )
        gp.fit(X, ys)
        return gp

    def current_tc(self):
        ids = np.array(sorted(set(self.selected)), dtype=int)
        return parabolic_peak_tc(
            self.temps[ids],
            self.data.loc[ids, "Susceptibility"].to_numpy(float),
        )

    def decide_next(self):
        gp = self._fit_gp()
        mu, std = gp.predict(self.temps.reshape(-1, 1), return_std=True)

        predicted_tc = float(self.temps[int(np.argmax(mu))])

        criticality = np.exp(
            -0.5
            * ((self.temps - predicted_tc) / max(CRITICALITY_WIDTH, 1e-8)) ** 2
        )

        selected_t = self.temps[np.array(self.selected, dtype=int)]

        left_near = np.any(
            (selected_t < predicted_tc)
            & (selected_t > predicted_tc - BRACKET_WIDTH)
        )
        right_near = np.any(
            (selected_t > predicted_tc)
            & (selected_t < predicted_tc + BRACKET_WIDTH)
        )

        bracket = np.zeros_like(self.temps, dtype=float)

        if not left_near:
            bracket += np.exp(
                -0.5
                * (
                    (self.temps - (predicted_tc - BRACKET_WIDTH / 2.0))
                    / max(BRACKET_WIDTH / 2.0, 1e-8)
                )
                ** 2
            )

        if not right_near:
            bracket += np.exp(
                -0.5
                * (
                    (self.temps - (predicted_tc + BRACKET_WIDTH / 2.0))
                    / max(BRACKET_WIDTH / 2.0, 1e-8)
                )
                ** 2
            )

        acquisition = (
            EXPLORATION_WEIGHT * normalize(std)
            + CRITICALITY_WEIGHT * normalize(criticality)
            + BRACKETING_WEIGHT * normalize(bracket)
        )

        acquisition[np.array(self.selected, dtype=int)] = -np.inf
        idx = int(np.argmax(acquisition))

        reasons = []
        if normalize(std)[idx] > 0.60:
            reasons.append("high GP uncertainty")
        if normalize(criticality)[idx] > 0.60:
            reasons.append("predicted critical region")
        if normalize(bracket)[idx] > 0.60:
            reasons.append("physics-critic bracketing")
        if not reasons:
            reasons.append("maximum combined acquisition")

        return AgentDecision(
            index=idx,
            temperature=float(self.temps[idx]),
            acquisition=float(acquisition[idx]),
            predicted_tc=predicted_tc,
            reason=", ".join(reasons),
        )

    def _record(self, idx, acquisition, reason):
        if idx not in self.selected:
            self.selected.append(idx)

        tc = self.current_tc()
        self.tc_history.append(tc)

        r = self.data.iloc[idx]

        self.memory.append({
            "Step": len(self.memory) + 1,
            "Temperature": float(r["Temperature"]),
            "Abs_Magnetization": float(r["Abs_Magnetization"]),
            "Energy_per_spin": float(r["Energy_per_spin"]),
            "C1": float(r["C1"]),
            "Susceptibility": float(r["Susceptibility"]),
            "Heat_Capacity": float(r["Heat_Capacity"]),
            "Binder_U4": float(r["Binder_U4"]),
            "Tc_estimate": float(tc),
            "Acquisition": float(acquisition) if np.isfinite(acquisition) else np.nan,
            "Reason": reason,
        })

    def critic_allows_stop(self):
        if len(set(self.selected)) < MIN_QUERIES_BEFORE_STOP:
            return False

        if len(self.tc_history) < STABILITY_WINDOW:
            return False

        recent = np.asarray(self.tc_history[-STABILITY_WINDOW:], dtype=float)
        if not np.all(np.isfinite(recent)):
            return False

        if np.max(recent) - np.min(recent) > TC_STABILITY_TOL:
            return False

        tc = recent[-1]
        selected_t = self.temps[np.array(self.selected, dtype=int)]

        left = np.any(
            (selected_t < tc)
            & (selected_t > tc - BRACKET_WIDTH)
        )
        right = np.any(
            (selected_t > tc)
            & (selected_t < tc + BRACKET_WIDTH)
        )

        return bool(left and right)

    def run(self):
        self.initialize()

        while len(set(self.selected)) < self.max_queries:
            if self.critic_allows_stop():
                break

            d = self.decide_next()
            self._record(d.index, d.acquisition, d.reason)

        ids = sorted(set(self.selected))
        selected_df = self.data.iloc[ids].sort_values("Temperature").reset_index(drop=True)

        return {
            "selected_indices": ids,
            "selected_observables": selected_df,
            "memory": pd.DataFrame(self.memory),
            "tc": self.current_tc(),
        }


# ======================================================================
# 5. Binder crossing and critical scaling
# ======================================================================

def binder_crossing(df1, df2, tc_hint):
    """
    Find crossing U4_L1(T)=U4_L2(T) by linear interpolation.
    Uses only selected/observed agentic temperatures.
    """
    a = df1[["Temperature", "Binder_U4"]].dropna().sort_values("Temperature")
    b = df2[["Temperature", "Binder_U4"]].dropna().sort_values("Temperature")

    if len(a) < 2 or len(b) < 2:
        return np.nan

    lo = max(a["Temperature"].min(), b["Temperature"].min())
    hi = min(a["Temperature"].max(), b["Temperature"].max())

    if lo >= hi:
        return np.nan

    grid = np.linspace(lo, hi, BINDER_INTERPOLATION_POINTS)

    ua = np.interp(
        grid,
        a["Temperature"].to_numpy(float),
        a["Binder_U4"].to_numpy(float),
    )
    ub = np.interp(
        grid,
        b["Temperature"].to_numpy(float),
        b["Binder_U4"].to_numpy(float),
    )

    diff = ua - ub
    candidates = []

    for i in range(len(grid) - 1):
        if diff[i] == 0:
            candidates.append(float(grid[i]))
        elif diff[i] * diff[i + 1] < 0:
            x1, x2 = grid[i], grid[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            x = x1 - y1 * (x2 - x1) / (y2 - y1)
            candidates.append(float(x))

    if candidates:
        return min(candidates, key=lambda x: abs(x - tc_hint))

    # Optional approximate crossing only when curves are already very close.
    i = int(np.argmin(np.abs(diff)))
    if abs(diff[i]) <= BINDER_APPROX_TOL:
        return float(grid[i])

    return np.nan


def local_binder_slope(df, tc):
    d = df[["Temperature", "Binder_U4"]].dropna().sort_values("Temperature")
    if len(d) < BINDER_SLOPE_MIN_POINTS:
        return np.nan

    local = d[
        np.abs(d["Temperature"] - tc) <= BINDER_SLOPE_WINDOW
    ].copy()

    if len(local) < BINDER_SLOPE_MIN_POINTS:
        # Use nearest points instead.
        local["distance"] = np.abs(local["Temperature"] - tc)
        local = local.sort_values("distance").head(BINDER_SLOPE_MIN_POINTS)

    if len(local) < 2:
        return np.nan

    x = local["Temperature"].to_numpy(float)
    y = local["Binder_U4"].to_numpy(float)

    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(abs(slope))
    except Exception:
        return np.nan


def scaling_from_selected(selected_by_L, pseudo_tc_by_L):
    """
    Estimate:
        binder Tc,
        nu from dU4/dT ~ L^(1/nu),
        beta/nu from M(Tc,L) ~ L^(-beta/nu),
        gamma/nu from chi(Tc,L) ~ L^(gamma/nu),
        beta, gamma,
        Tc(infinity) from Tc(L)=Tc(inf)+a L^(-1/nu).

    With only L=16,32,64 these exponent estimates are exploratory and should
    be reported with bootstrap confidence intervals.
    """
    sizes = sorted(selected_by_L)
    pseudo_vals = np.array([pseudo_tc_by_L[L] for L in sizes], dtype=float)
    tc_hint = float(np.nanmedian(pseudo_vals))

    crossing_rows = []
    crossings = []

    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        tcx = binder_crossing(
            selected_by_L[L1],
            selected_by_L[L2],
            tc_hint,
        )
        crossing_rows.append({
            "L1": L1,
            "L2": L2,
            "Binder_crossing_Tc": tcx,
        })
        if np.isfinite(tcx):
            crossings.append(tcx)

    binder_tc = float(np.nanmedian(crossings)) if crossings else tc_hint

    # Binder derivative scaling -> nu
    L_arr = np.asarray(sizes, dtype=float)
    slopes = np.asarray(
        [local_binder_slope(selected_by_L[L], binder_tc) for L in sizes],
        dtype=float,
    )

    nu = np.nan
    inv_nu = np.nan
    nu_r2 = np.nan

    ok = np.isfinite(slopes) & (slopes > 0)
    if np.sum(ok) >= 3:
        reg = linregress(np.log(L_arr[ok]), np.log(slopes[ok]))
        inv_nu = float(reg.slope)
        nu_r2 = float(reg.rvalue ** 2)
        if inv_nu > 0:
            nu = float(1.0 / inv_nu)

    # Evaluate M and chi at binder Tc
    M_tc = np.asarray(
        [interpolate_at(selected_by_L[L], "Abs_Magnetization", binder_tc) for L in sizes],
        dtype=float,
    )
    chi_tc = np.asarray(
        [interpolate_at(selected_by_L[L], "Susceptibility", binder_tc) for L in sizes],
        dtype=float,
    )

    beta_over_nu = np.nan
    gamma_over_nu = np.nan
    beta_ratio_r2 = np.nan
    gamma_ratio_r2 = np.nan

    okm = np.isfinite(M_tc) & (M_tc > 0)
    if np.sum(okm) >= 3:
        reg_m = linregress(np.log(L_arr[okm]), np.log(M_tc[okm]))
        beta_over_nu = float(-reg_m.slope)
        beta_ratio_r2 = float(reg_m.rvalue ** 2)

    okc = np.isfinite(chi_tc) & (chi_tc > 0)
    if np.sum(okc) >= 3:
        reg_c = linregress(np.log(L_arr[okc]), np.log(chi_tc[okc]))
        gamma_over_nu = float(reg_c.slope)
        gamma_ratio_r2 = float(reg_c.rvalue ** 2)

    beta = (
        float(beta_over_nu * nu)
        if np.isfinite(beta_over_nu) and np.isfinite(nu)
        else np.nan
    )
    gamma = (
        float(gamma_over_nu * nu)
        if np.isfinite(gamma_over_nu) and np.isfinite(nu)
        else np.nan
    )

    # Finite-size extrapolation Tc(L)=Tc_inf + a L^(-1/nu)
    tc_inf = np.nan
    tc_fss_r2 = np.nan
    fss_slope = np.nan

    if np.isfinite(nu) and nu > 0 and np.all(np.isfinite(pseudo_vals)):
        x = L_arr ** (-1.0 / nu)
        reg_t = linregress(x, pseudo_vals)
        tc_inf = float(reg_t.intercept)
        fss_slope = float(reg_t.slope)
        tc_fss_r2 = float(reg_t.rvalue ** 2)

    metrics = {
        "Binder_Tc": binder_tc,
        "nu": nu,
        "inv_nu": inv_nu,
        "nu_scaling_R2": nu_r2,
        "beta_over_nu": beta_over_nu,
        "gamma_over_nu": gamma_over_nu,
        "beta_over_nu_R2": beta_ratio_r2,
        "gamma_over_nu_R2": gamma_ratio_r2,
        "beta": beta,
        "gamma": gamma,
        "Tc_infinity_FSS": tc_inf,
        "Tc_FSS_R2": tc_fss_r2,
        "Tc_FSS_slope": fss_slope,
    }

    scaling_points = pd.DataFrame({
        "L": sizes,
        "Tc_pseudocritical": pseudo_vals,
        "M_at_Binder_Tc": M_tc,
        "Chi_at_Binder_Tc": chi_tc,
        "Abs_dU4_dT_at_Binder_Tc": slopes,
    })

    return metrics, pd.DataFrame(crossing_rows), scaling_points


# ======================================================================
# 6. Bootstrap uncertainty
# ======================================================================

def bootstrap_selected_aggregate(raw, selected_temperatures, lattice, rng):
    rows = []

    selected_set = set(float(x) for x in selected_temperatures)

    for T, g in raw.groupby("Temperature", sort=True):
        if float(T) not in selected_set:
            continue

        n = len(g)
        if n == 0:
            continue

        idx = rng.integers(0, n, size=n)
        gb = g.iloc[idx]
        rows.append(aggregate_temperature_group(T, gb, lattice))

    return pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)


def bootstrap_fss(
    raw_by_L,
    selected_by_L,
    n_bootstrap,
):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    selected_temps_by_L = {
        L: selected_by_L[L]["Temperature"].to_numpy(float)
        for L in selected_by_L
    }

    for b in range(n_bootstrap):
        boot_selected = {}
        boot_pseudo = {}

        for L in sorted(raw_by_L):
            ba = bootstrap_selected_aggregate(
                raw_by_L[L],
                selected_temps_by_L[L],
                L,
                rng,
            )
            boot_selected[L] = ba
            boot_pseudo[L] = parabolic_peak_tc(
                ba["Temperature"],
                ba["Susceptibility"],
            )

        metrics, _, _ = scaling_from_selected(
            boot_selected,
            boot_pseudo,
        )

        row = {
            "Bootstrap": b + 1,
            **{f"Tc_L{L}": boot_pseudo[L] for L in sorted(boot_pseudo)},
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# 7. Plotting
# ======================================================================

def add_tc_lines(ax, agent_tc, exact=True):
    if np.isfinite(agent_tc):
        ax.axvline(
            agent_tc,
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=f"Agentic Tc(L) = {agent_tc:.6f}",
        )

    if exact and USE_EXACT_VALUES_FOR_EVALUATION:
        ax.axvline(
            EXACT_TC,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=f"Exact thermodynamic Tc = {EXACT_TC:.6f}",
        )


def plot_single_curve(
    agg,
    ycol,
    ylabel,
    title,
    outbase,
    agent_tc,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(
        agg["Temperature"],
        agg[ycol],
        linewidth=LINE_WIDTH,
        label=title,
    )

    add_tc_lines(ax, agent_tc)

    ax.set_xlabel("Temperature T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_combined_thermodynamic_curves(
    agg,
    outbase,
    agent_tc,
    lattice,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    curve_specs = [
        ("Abs_Magnetization", r"$|M|$", "-"),
        ("Energy_per_spin", "E / spin", "--"),
        ("C1", r"$C_1$", "-."),
        ("Heat_Capacity", r"$C_V$", ":"),
        ("Binder_U4", r"$U_4$", (0, (6, 2, 1.5, 2))),
    ]

    for ycol, label, linestyle in curve_specs:
        ax.plot(
            agg["Temperature"],
            agg[ycol],
            linewidth=LINE_WIDTH,
            linestyle=linestyle,
            label=label,
        )

    add_tc_lines(ax, agent_tc)

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Observable value")
    ax.set_title(f"Thermodynamic Observables vs Temperature (L={lattice})")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE}, ncol=2)

    save_pdf(fig, outbase)


def plot_agent_sampling(agg, selected, agent_tc, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(
        agg["Temperature"],
        agg["Susceptibility"],
        linewidth=LINE_WIDTH,
        label="Available temperature grid",
    )

    ax.scatter(
        selected["Temperature"],
        selected["Susceptibility"],
        s=MARKER_SIZE,
        zorder=5,
        label="Agent-selected temperatures",
    )

    add_tc_lines(ax, agent_tc)

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility χ")
    ax.set_title("Agentic Critical-Region Sampling")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_tc_convergence(memory, final_tc, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(
        memory["Step"],
        memory["Tc_estimate"],
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=7,
        label="Agent Tc estimate",
    )

    if np.isfinite(final_tc):
        ax.axhline(
            final_tc,
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=f"Final Agentic Tc(L) = {final_tc:.6f}",
        )

    if USE_EXACT_VALUES_FOR_EVALUATION:
        ax.axhline(
            EXACT_TC,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=f"Exact thermodynamic Tc = {EXACT_TC:.6f}",
        )

    ax.set_xlabel("Temperature-query step")
    ax.set_ylabel("Estimated Tc")
    ax.set_title("Autonomous Tc Convergence")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_binder_combined(selected_by_L, binder_tc, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for L in sorted(selected_by_L):
        d = selected_by_L[L].sort_values("Temperature")
        ax.plot(
            d["Temperature"],
            d["Binder_U4"],
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=6,
            label=f"L = {L}",
        )

    if np.isfinite(binder_tc):
        ax.axvline(
            binder_tc,
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=f"Agentic Binder Tc = {binder_tc:.6f}",
        )

    if USE_EXACT_VALUES_FOR_EVALUATION:
        ax.axvline(
            EXACT_TC,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=f"Exact Tc = {EXACT_TC:.6f}",
        )

    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Binder cumulant U4")
    ax.set_title("Agent-Observed Binder Cumulant Crossing")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_pseudocritical_fss(
    scaling_points,
    nu,
    tc_inf,
    outbase,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    Ls = scaling_points["L"].to_numpy(float)
    tcs = scaling_points["Tc_pseudocritical"].to_numpy(float)

    if np.isfinite(nu) and nu > 0:
        x = Ls ** (-1.0 / nu)
        ax.scatter(x, tcs, s=100, label="Agentic pseudocritical Tc(L)")

        if np.isfinite(tc_inf):
            reg = linregress(x, tcs)
            xx = np.linspace(0, x.max() * 1.08, 250)
            yy = reg.intercept + reg.slope * xx
            ax.plot(
                xx,
                yy,
                linewidth=LINE_WIDTH,
                label=f"FSS extrapolation Tc(∞) = {tc_inf:.6f}",
            )

        ax.set_xlabel(r"$L^{-1/\nu}$")
    else:
        x = 1.0 / Ls
        ax.scatter(x, tcs, s=100, label="Agentic pseudocritical Tc(L)")
        ax.set_xlabel(r"$1/L$")

    if USE_EXACT_VALUES_FOR_EVALUATION:
        ax.axhline(
            EXACT_TC,
            linestyle=":",
            linewidth=LINE_WIDTH,
            label=f"Exact Tc = {EXACT_TC:.6f}",
        )

    ax.set_ylabel("Critical temperature")
    ax.set_title("Finite-Size Scaling of Agentic Tc")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_magnetization_collapse(
    selected_by_L,
    tc,
    nu,
    beta,
    outbase,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    if not (np.isfinite(tc) and np.isfinite(nu) and np.isfinite(beta) and nu > 0):
        ax.text(
            0.5,
            0.5,
            "Insufficient stable estimates for magnetization data collapse",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
        )
    else:
        for L in sorted(selected_by_L):
            d = selected_by_L[L].sort_values("Temperature")
            x = (d["Temperature"].to_numpy(float) - tc) * (L ** (1.0 / nu))
            y = d["Abs_Magnetization"].to_numpy(float) * (L ** (beta / nu))

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=6,
                label=f"L = {L}",
            )

    ax.set_xlabel(r"$(T-T_c)L^{1/\nu}$")
    ax.set_ylabel(r"$|M|L^{\beta/\nu}$")
    ax.set_title("Agentic Magnetization Finite-Size Data Collapse")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_susceptibility_collapse(
    selected_by_L,
    tc,
    nu,
    gamma,
    outbase,
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    if not (np.isfinite(tc) and np.isfinite(nu) and np.isfinite(gamma) and nu > 0):
        ax.text(
            0.5,
            0.5,
            "Insufficient stable estimates for susceptibility data collapse",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
        )
    else:
        for L in sorted(selected_by_L):
            d = selected_by_L[L].sort_values("Temperature")
            x = (d["Temperature"].to_numpy(float) - tc) * (L ** (1.0 / nu))
            y = d["Susceptibility"].to_numpy(float) * (L ** (-gamma / nu))

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=6,
                label=f"L = {L}",
            )

    ax.set_xlabel(r"$(T-T_c)L^{1/\nu}$")
    ax.set_ylabel(r"$\chi L^{-\gamma/\nu}$")
    ax.set_title("Agentic Susceptibility Finite-Size Data Collapse")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_efficiency(efficiency_df, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    x = np.arange(len(efficiency_df))
    vals = efficiency_df["Temperature_Reduction_Percent"].to_numpy(float)

    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"L={int(v)}" for v in efficiency_df["L"]],
        fontweight="bold",
    )

    ax.set_ylabel("Temperature evaluations avoided (%)")
    ax.set_xlabel("Lattice size")
    ax.set_title("Agentic Sampling Efficiency")
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(vals):
        ax.text(
            i,
            v,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    save_pdf(fig, outbase)


def plot_beta_ratio_scaling(scaling_points, metrics, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    d = scaling_points.dropna(subset=["M_at_Binder_Tc"])
    d = d[d["M_at_Binder_Tc"] > 0]

    x = np.log(d["L"].to_numpy(float))
    y = np.log(d["M_at_Binder_Tc"].to_numpy(float))

    ax.scatter(x, y, s=100, label="Agentic measurements")

    if len(x) >= 2:
        reg = linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xx,
            reg.intercept + reg.slope * xx,
            linewidth=LINE_WIDTH,
            label=rf"$\beta/\nu$ = {metrics['beta_over_nu']:.4f}",
        )

    ax.set_xlabel(r"$\log L$")
    ax.set_ylabel(r"$\log |M(T_c,L)|$")
    ax.set_title(r"Critical Scaling for $\beta/\nu$")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_gamma_ratio_scaling(scaling_points, metrics, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    d = scaling_points.dropna(subset=["Chi_at_Binder_Tc"])
    d = d[d["Chi_at_Binder_Tc"] > 0]

    x = np.log(d["L"].to_numpy(float))
    y = np.log(d["Chi_at_Binder_Tc"].to_numpy(float))

    ax.scatter(x, y, s=100, label="Agentic measurements")

    if len(x) >= 2:
        reg = linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xx,
            reg.intercept + reg.slope * xx,
            linewidth=LINE_WIDTH,
            label=rf"$\gamma/\nu$ = {metrics['gamma_over_nu']:.4f}",
        )

    ax.set_xlabel(r"$\log L$")
    ax.set_ylabel(r"$\log \chi(T_c,L)$")
    ax.set_title(r"Critical Scaling for $\gamma/\nu$")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


def plot_nu_scaling(scaling_points, metrics, outbase):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    d = scaling_points.dropna(subset=["Abs_dU4_dT_at_Binder_Tc"])
    d = d[d["Abs_dU4_dT_at_Binder_Tc"] > 0]

    x = np.log(d["L"].to_numpy(float))
    y = np.log(d["Abs_dU4_dT_at_Binder_Tc"].to_numpy(float))

    ax.scatter(x, y, s=100, label="Agentic measurements")

    if len(x) >= 2:
        reg = linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xx,
            reg.intercept + reg.slope * xx,
            linewidth=LINE_WIDTH,
            label=rf"$1/\nu$ = {metrics['inv_nu']:.4f}",
        )

    ax.set_xlabel(r"$\log L$")
    ax.set_ylabel(r"$\log |dU_4/dT|_{T_c}$")
    ax.set_title(r"Binder-Slope Scaling for $\nu$")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight": "bold", "size": LEGEND_FONT_SIZE})

    save_pdf(fig, outbase)


# ======================================================================
# 8. Main
# ======================================================================

def main():
    args = parse_args()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    combined_dir = RESULTS_ROOT / COMBINED_DIRNAME
    combined_dir.mkdir(parents=True, exist_ok=True)

    raw_by_L = {}
    full_agg_by_L = {}
    selected_by_L = {}
    pseudo_tc_by_L = {}

    lattice_summaries = []
    efficiency_rows = []
    runtime_rows = []

    # --------------------------------------------------------------
    # A. Run the proposed agent independently for L=16,32,64
    # --------------------------------------------------------------
    for L in LATTICE_SIZES:
        lattice_wall_start = time.perf_counter()
        lattice_cpu_start = time.process_time()

        path = DATASETS[L]
        outdir = RESULTS_ROOT / f"Results_{L}"
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== L={L}: loading {path} ===")

        raw = load_raw_observables(path, L)
        agg = aggregate_by_temperature(raw, L)

        full_grid_tc = parabolic_peak_tc(
            agg["Temperature"],
            agg["Susceptibility"],
        )

        agent = PhysicsInformedAgent(
            agg,
            max_queries=args.max_queries,
        )
        result = agent.run()

        selected = result["selected_observables"]
        memory = result["memory"]
        tc_agent = float(result["tc"])

        raw_by_L[L] = raw
        full_agg_by_L[L] = agg
        selected_by_L[L] = selected
        pseudo_tc_by_L[L] = tc_agent

        total_temps = len(agg)
        queried_temps = len(selected)

        total_configs = int(agg["N_configs"].sum())
        selected_configs = int(selected["N_configs"].sum())

        temp_fraction = queried_temps / total_temps
        cfg_fraction = selected_configs / total_configs if total_configs > 0 else np.nan

        temp_reduction = (1.0 - temp_fraction) * 100.0
        cfg_reduction = (1.0 - cfg_fraction) * 100.0 if np.isfinite(cfg_fraction) else np.nan

        # Optional autocorrelation / ESS
        if ROWS_ARE_TIME_ORDERED:
            ts_diag = time_series_diagnostics(raw)
            ts_diag.to_csv(
                outdir / "agentic_autocorrelation_ess.csv",
                index=False,
            )
            selected_tau = ts_diag[
                ts_diag["Temperature"].isin(selected["Temperature"])
            ]
            median_tau = float(selected_tau["tau_int_magnetization"].median())
            median_ess = float(selected_tau["effective_sample_size"].median())
        else:
            median_tau = np.nan
            median_ess = np.nan

        summary = {
            "L": L,
            "Dataset": str(path),
            "Total_Configurations": total_configs,
            "Total_Unique_Temperatures": total_temps,
            "Agent_Queried_Temperatures": queried_temps,
            "Temperature_Query_Fraction": temp_fraction,
            "Temperature_Reduction_Percent": temp_reduction,
            "Counterfactual_Selected_Configurations": selected_configs,
            "Counterfactual_Configuration_Fraction": cfg_fraction,
            "Counterfactual_Configuration_Reduction_Percent": cfg_reduction,
            "Agentic_Pseudocritical_Tc": tc_agent,
            "Full_Grid_Pseudocritical_Tc": full_grid_tc,
            "Agent_vs_FullGrid_Abs_Difference": abs(tc_agent - full_grid_tc),
            "Median_tau_int_if_ordered": median_tau,
            "Median_ESS_if_ordered": median_ess,
        }

        if USE_EXACT_VALUES_FOR_EVALUATION:
            summary.update({
                "Exact_Thermodynamic_Tc_eval_only": EXACT_TC,
                "Agent_vs_Exact_Abs_Error_eval_only": abs(tc_agent - EXACT_TC),
                "Agent_vs_Exact_Relative_Error_Percent_eval_only":
                    abs(tc_agent - EXACT_TC) / EXACT_TC * 100.0,
            })

        lattice_summaries.append(summary)

        efficiency_rows.append({
            "L": L,
            "Total_Temperatures": total_temps,
            "Queried_Temperatures": queried_temps,
            "Temperature_Reduction_Percent": temp_reduction,
            "Total_Configurations": total_configs,
            "Counterfactual_Selected_Configurations": selected_configs,
            "Counterfactual_Configuration_Reduction_Percent": cfg_reduction,
        })

        # Numeric outputs
        pd.DataFrame([summary]).to_csv(
            outdir / "proposed_agentic_summary.csv",
            index=False,
        )
        selected.to_csv(
            outdir / "proposed_agentic_selected_observables.csv",
            index=False,
        )
        memory.to_csv(
            outdir / "proposed_agentic_memory.csv",
            index=False,
        )

        # Figures: PDF ONLY; thermodynamic observables merged, susceptibility separate
        plot_agent_sampling(
            agg,
            selected,
            tc_agent,
            outdir / "01_agentic_sampling",
        )

        plot_tc_convergence(
            memory,
            tc_agent,
            outdir / "02_tc_convergence",
        )

        plot_combined_thermodynamic_curves(
            agg,
            outdir / "03_thermodynamic_summary",
            tc_agent,
            L,
        )

        plot_single_curve(
            agg,
            "Susceptibility",
            "χ",
            f"Susceptibility vs Temperature (L={L})",
            outdir / "04_susceptibility_curve",
            tc_agent,
        )

        lattice_wall = time.perf_counter() - lattice_wall_start
        lattice_cpu = time.process_time() - lattice_cpu_start

        runtime_rows.append({
            "Scope": f"L={L}",
            "Wall_Clock_Seconds": lattice_wall,
            "CPU_Process_Seconds": lattice_cpu,
            "Theoretical_Preprocessing_Time":
                f"O(N_configs * {L}^2)",
            "Theoretical_Agent_Time":
                "O(Q^4 + N_T * Q^2) worst-case sequential GP fitting/prediction",
            "Theoretical_Space":
                f"O(N_configs * {L}^2) while loading spins; reduced to O(N_configs) after scalarization",
        })

        print(
            f"L={L}: agent Tc={tc_agent:.6f}, "
            f"queries={queried_temps}/{total_temps}, "
            f"temperature reduction={temp_reduction:.2f}%"
        )

    # --------------------------------------------------------------
    # B. Agentic Binder crossing + FSS + critical exponents
    # --------------------------------------------------------------
    metrics, crossings, scaling_points = scaling_from_selected(
        selected_by_L,
        pseudo_tc_by_L,
    )

    crossings.to_csv(
        combined_dir / "binder_crossings.csv",
        index=False,
    )
    scaling_points.to_csv(
        combined_dir / "critical_scaling_points.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # C. Bootstrap confidence intervals
    # --------------------------------------------------------------
    print(f"\nRunning {args.bootstrap} bootstrap replicates...")

    boot = bootstrap_fss(
        raw_by_L,
        selected_by_L,
        args.bootstrap,
    )
    boot.to_csv(
        combined_dir / "bootstrap_distributions.csv",
        index=False,
    )

    ci_rows = []
    for metric_name in [
        "Binder_Tc",
        "Tc_infinity_FSS",
        "nu",
        "beta",
        "gamma",
        "beta_over_nu",
        "gamma_over_nu",
    ]:
        med, low, high = ci(boot[metric_name].to_numpy(float))
        ci_rows.append({
            "Metric": metric_name,
            "Point_Estimate": metrics.get(metric_name, np.nan),
            "Bootstrap_Median": med,
            f"CI_{CI_LOW}_Percent": low,
            f"CI_{CI_HIGH}_Percent": high,
        })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(
        combined_dir / "critical_parameters_with_bootstrap_CI.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # D. Evaluation-only exact-value comparison
    # --------------------------------------------------------------
    final_eval = {
        **metrics,
        "Exact_Tc_eval_only": EXACT_TC,
        "Exact_beta_eval_only": EXACT_BETA,
        "Exact_gamma_eval_only": EXACT_GAMMA,
        "Exact_nu_eval_only": EXACT_NU,
    }

    if np.isfinite(metrics["Tc_infinity_FSS"]):
        final_eval["Tc_infinity_abs_error_eval_only"] = abs(
            metrics["Tc_infinity_FSS"] - EXACT_TC
        )
    else:
        final_eval["Tc_infinity_abs_error_eval_only"] = np.nan

    if np.isfinite(metrics["beta"]):
        final_eval["beta_abs_error_eval_only"] = abs(
            metrics["beta"] - EXACT_BETA
        )
    else:
        final_eval["beta_abs_error_eval_only"] = np.nan

    if np.isfinite(metrics["gamma"]):
        final_eval["gamma_abs_error_eval_only"] = abs(
            metrics["gamma"] - EXACT_GAMMA
        )
    else:
        final_eval["gamma_abs_error_eval_only"] = np.nan

    if np.isfinite(metrics["nu"]):
        final_eval["nu_abs_error_eval_only"] = abs(
            metrics["nu"] - EXACT_NU
        )
    else:
        final_eval["nu_abs_error_eval_only"] = np.nan

    pd.DataFrame([final_eval]).to_csv(
        combined_dir / "PRE_agentic_final_physics_summary.csv",
        index=False,
    )

    pd.DataFrame(lattice_summaries).to_csv(
        combined_dir / "all_lattice_agentic_summary.csv",
        index=False,
    )

    efficiency_df = pd.DataFrame(efficiency_rows)
    efficiency_df.to_csv(
        combined_dir / "computational_sampling_efficiency.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # E. Combined publication figures
    # --------------------------------------------------------------
    plot_binder_combined(
        selected_by_L,
        metrics["Binder_Tc"],
        combined_dir / "01_binder_crossing",
    )

    plot_pseudocritical_fss(
        scaling_points,
        metrics["nu"],
        metrics["Tc_infinity_FSS"],
        combined_dir / "02_pseudocritical_Tc_finite_size_scaling",
    )

    plot_magnetization_collapse(
        selected_by_L,
        metrics["Binder_Tc"],
        metrics["nu"],
        metrics["beta"],
        combined_dir / "03_magnetization_data_collapse",
    )

    plot_susceptibility_collapse(
        selected_by_L,
        metrics["Binder_Tc"],
        metrics["nu"],
        metrics["gamma"],
        combined_dir / "04_susceptibility_data_collapse",
    )

    plot_efficiency(
        efficiency_df,
        combined_dir / "05_computational_sampling_efficiency",
    )

    plot_beta_ratio_scaling(
        scaling_points,
        metrics,
        combined_dir / "06_beta_over_nu_scaling",
    )

    plot_gamma_ratio_scaling(
        scaling_points,
        metrics,
        combined_dir / "07_gamma_over_nu_scaling",
    )

    plot_nu_scaling(
        scaling_points,
        metrics,
        combined_dir / "08_nu_binder_slope_scaling",
    )

    # --------------------------------------------------------------
    # F. Time + space complexity CSV
    # --------------------------------------------------------------
    total_wall = time.perf_counter() - wall_start
    total_cpu = time.process_time() - cpu_start

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = float(usage.ru_maxrss) / 1024.0  # Linux HPC convention

    runtime_rows.append({
        "Scope": "TOTAL_PIPELINE",
        "Wall_Clock_Seconds": total_wall,
        "CPU_Process_Seconds": total_cpu,
        "Theoretical_Preprocessing_Time":
            "O(sum_L N_configs(L) * L^2)",
        "Theoretical_Agent_Time":
            "O(sum_L [Q^4 + N_T(L)*Q^2]) worst-case sequential GP",
        "Theoretical_Space":
            "Peak dominated by current spin matrix O(N_configs(L)*L^2); "
            "after reduction O(sum_L N_configs(L))",
    })

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df["Peak_RSS_MB_TOTAL_PROCESS"] = peak_rss_mb

    runtime_df.to_csv(
        combined_dir / "time_and_space_complexity.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # G. Console report
    # --------------------------------------------------------------
    print("\n============================================================")
    print("PRE-ORIENTED AGENTIC ISING RESULTS")
    print("============================================================")

    print("\nPseudocritical Tc(L):")
    for L in sorted(pseudo_tc_by_L):
        print(f"  L={L:>2}: {pseudo_tc_by_L[L]:.6f}")

    print(f"\nBinder-based agentic Tc: {metrics['Binder_Tc']:.6f}")
    print(f"Estimated nu:            {metrics['nu']}")
    print(f"Estimated beta:          {metrics['beta']}")
    print(f"Estimated gamma:         {metrics['gamma']}")
    print(f"FSS Tc(infinity):        {metrics['Tc_infinity_FSS']}")

    if USE_EXACT_VALUES_FOR_EVALUATION:
        print("\nEvaluation-only exact values:")
        print(f"  Tc    = {EXACT_TC:.8f}")
        print(f"  beta  = {EXACT_BETA:.8f}")
        print(f"  gamma = {EXACT_GAMMA:.8f}")
        print(f"  nu    = {EXACT_NU:.8f}")

    print("\nSampling efficiency:")
    print(
        efficiency_df[
            [
                "L",
                "Queried_Temperatures",
                "Total_Temperatures",
                "Temperature_Reduction_Percent",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT:")
    print(
        "The current datasets are precomputed. Therefore the code directly "
        "supports a temperature-query / data-usage efficiency claim. "
        "For a strict Monte Carlo CPU-savings claim, run the final agent "
        "in an on-demand simulation loop."
    )

    print(f"\nAll combined outputs saved in: {combined_dir.resolve()}")


if __name__ == "__main__":
    main()

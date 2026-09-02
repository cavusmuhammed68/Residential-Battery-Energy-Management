# -*- coding: utf-8 -*-
"""
================================================================================
DT-AAMPC : Digital-Twin-enabled Attention-Augmented Adaptive Model Predictive
           Control Framework for Multi-Source Residential Battery Energy
           Management Systems (BEMS)
================================================================================

NOVEL METHOD (proposed in this script)
----------------------------------------
    "DT-AAMPC" — Digital-Twin Attention-Augmented Adaptive Model Predictive
    Control.

    The method fuses THREE ingredients that, to the best of the authors'
    knowledge, have not been jointly integrated in a single closed-loop
    residential BEMS pipeline:

    1) DEEP LEARNING FORECASTER — "TAB-Net" (Temporal Attention-BiLSTM
       Network). A causal 1-D convolutional stem extracts short-range motifs
       from PV / wind / load / weather channels, a multi-head self-attention
       block learns long-range dependencies (e.g., multi-day cloud regimes,
       weekly load cycles), and a Bidirectional-LSTM head produces
       multi-horizon, multi-quantile forecasts of PV power, wind power and
       aggregate household load. The attention entropy is repurposed on-line
       as a *forecast-confidence signal* that is fed directly into the
       controller (see point 3).

    2) DIGITAL TWIN STATE ESTIMATOR — a physics-based electro-thermal battery
       twin (equivalent circuit + Arrhenius-type thermal-degradation model)
       is kept synchronised with the (simulated) physical asset through an
       Extended Kalman Filter (EKF). The twin continuously reconciles
       model-predicted SOC/SOH/temperature against noisy "sensor"
       measurements, producing bias-corrected states AND a state-covariance
       (uncertainty) matrix.

    3) RISK-AWARE ADAPTIVE MPC — a finite-horizon economic MPC whose
       (a) horizon length, (b) forecast-uncertainty penalty weight and
       (c) chance-constraint back-off margins are *adapted online* using the
       attention-confidence signal from (1) and the EKF covariance from (2).
       The QP-like NLP is solved every control step with SLSQP
       (scipy.optimize), and includes an explicit battery-degradation cost
       derived from the digital twin's throughput/SOH model — something the
       classical baselines below deliberately omit, to expose the value of
       degradation-aware, twin-informed control.

BASELINES IMPLEMENTED FOR COMPARISON (classical / widely used methods)
------------------------------------------------------------------------
    B1) PID / Droop Controller                (classical feedback control)
    B2) Deterministic "Naive-Forecast" MPC    (MPC without DL forecaster,
                                                without digital twin, without
                                                adaptive risk weighting)

SIX STRESS-TEST SCENARIOS (deliberately different from any reference figure)
------------------------------------------------------------------------------
    S1) Heatwave Air-Conditioning Surge
    S2) Winter Storm Load Peak
    S3) Cloud-Cover Cascade (Cirrus-to-Cumulus Transition Week)
    S4) Overnight EV Fleet Charging Rush
    S5) Islanded Microgrid Fault Ride-Through
    S6) Solar Oversupply Curtailment Week

DATASET
-------
    Rodriguez Vega & Syne (2025), "Augmented Smart Home with Weather
    Information", Mendeley Data, DOI: 10.17632/pxnb7gh646.1 — an augmentation
    of Taranvee's Kaggle smart-home dataset with additional IoT appliance
    channels (Car charger, Water heater, Air conditioning, Home Theater,
    Outdoor lights, microwave, Laundry, Pool Pump) plus regional weather.

OUTPUT
------
    All figures (600 DPI, base font size 16), trained model checkpoints,
    per-scenario/per-controller metric tables (CSV) and a consolidated
    Q1-manuscript-ready results report are written under:

        <BASE_DIR>/results/

IMPORTANT
---------
    This script is intentionally self-contained and heavy (deep learning
    training + closed-loop simulation across 6 scenarios x 3 controllers).
    It is meant to be executed OFFLINE by the user on their own machine;
    it is not executed here. Toggle the RUN_* flags in `Config` to control
    which stages actually run.

Author(s) : <fill in>
License   : MIT (code) — dataset retains its original CC BY 4.0 licence.
================================================================================
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Standard library
# --------------------------------------------------------------------------
import os
import sys
import json
import time
import math
import random
import logging
import warnings
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from itertools import product

# --------------------------------------------------------------------------
# Third-party (numerical / ML / plotting)
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; user can switch if needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from scipy import optimize as spopt
from scipy import signal as spsignal
from scipy import stats as spstats
from scipy.interpolate import interp1d

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found in this environment. The TAB-Net deep-learning "
        "forecaster requires `pip install torch`. All other stages "
        "(digital twin, controllers, plotting) remain fully functional "
        "using a persistence-forecast fallback."
    )

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

@dataclass
class PathConfig:
    """Filesystem layout. Edit BASE_DIR if the project is moved."""
    BASE_DIR: str = r"C:\Users\nfpm5\Downloads\Batteries_MDPI"
    DATA_FILENAME: str = "HomeC_augmented.csv"

    @property
    def data_path(self) -> str:
        return os.path.join(self.BASE_DIR, self.DATA_FILENAME)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "results")

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.results_dir, "figures")

    @property
    def tables_dir(self) -> str:
        return os.path.join(self.results_dir, "tables")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.results_dir, "models")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.results_dir, "logs")

    def make_all(self) -> None:
        for d in [self.results_dir, self.figures_dir, self.tables_dir,
                  self.models_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class SimConfig:
    """Simulation / resampling settings."""
    RESAMPLE_RULE: str = "15min"          # native data is 1-min; downsample for tractable MPC
    RANDOM_SEED: int = 42
    # Number of independent Monte-Carlo replicates run per (scenario,
    # controller) pair, each with an independently-seeded measurement-
    # noise stream (EKF sensor noise) while the scenario trace itself
    # stays fixed. A single deterministic run per condition is not
    # adequate evidence of a real closed-loop improvement -- reported
    # KPI differences must be checked against run-to-run operational
    # noise, not just averaged across the 6 (fixed) scenarios. All
    # aggregate figures/tables and the Wilcoxon significance tests use
    # every (scenario x seed) replicate as an independent sample.
    N_MONTE_CARLO_SEEDS: int = 5
    TRAIN_FRACTION: float = 0.70
    VAL_FRACTION: float = 0.15
    # (remaining 0.15 is test)

    # --- Renewable synthesis ---
    PV_RATED_KW: float = 8.0              # nameplate rooftop PV capacity
    PV_DEGRADATION_CLOUD_EXP: float = 1.35
    WIND_RATED_KW: float = 5.0            # small residential wind turbine
    WIND_CUT_IN_MS: float = 3.0
    WIND_RATED_MS: float = 11.0
    WIND_CUT_OUT_MS: float = 25.0

    # --- Battery physical parameters (digital twin) ---
    BATT_CAPACITY_KWH: float = 13.5       # Tesla-Powerwall-like residential unit
    BATT_MAX_C_RATE: float = 1.0
    BATT_ETA_CHG: float = 0.965
    BATT_ETA_DIS: float = 0.965
    BATT_SOC_MIN: float = 0.10
    BATT_SOC_MAX: float = 0.95
    BATT_SOC_INIT: float = 0.50
    BATT_SOH_INIT: float = 1.00
    BATT_THERMAL_MASS: float = 45.0       # kJ/K, lumped thermal mass
    BATT_THERMAL_RESIST: float = 3.2      # K/kW, lumped thermal resistance to ambient
    BATT_DEGRADATION_K1: float = 4.02e-5  # calendar-ageing coefficient
    BATT_DEGRADATION_K2: float = 2.85e-4  # cycling-ageing coefficient (per Ah-throughput)
    BATT_ARRHENIUS_EA: float = 20000.0    # J/mol, activation energy for thermal ageing
    BATT_GAS_CONST: float = 8.314         # J/(mol.K)
    BATT_REF_TEMP_K: float = 298.15       # 25 C reference

    # --- Grid economics ---
    GRID_IMPORT_PRICE_PEAK: float = 0.34      # currency/kWh
    GRID_IMPORT_PRICE_OFFPEAK: float = 0.14
    GRID_EXPORT_PRICE: float = 0.07
    PEAK_HOURS: Tuple[int, int] = (17, 21)    # 17:00-21:00 inclusive start, exclusive end
    DEGRADATION_COST_PER_KWH_THROUGHPUT: float = 0.045  # DEPRECATED, unused
    # Real degradation cost is now priced directly off SOH loss (see
    # `BatteryPhysicsModel.degradation_cost_kwh`): cost = |delta_SOH| *
    # capacity_kWh * BATT_REPLACEMENT_COST_PER_KWH. This ties the
    # controllers' internal optimisation and the reported net-cost KPI to
    # the SAME physical SOH signal, closing a gap where the old,
    # disconnected throughput-only price let a controller "optimise" a
    # degradation cost with no real relationship to actual battery ageing.
    BATT_REPLACEMENT_COST_PER_KWH: float = 300.0

    # --- Forecasting ---
    LOOKBACK_STEPS: int = 96        # 24h @ 15-min resolution
    HORIZON_STEPS: int = 16         # 4h look-ahead for MPC
    BATCH_SIZE: int = 128
    EPOCHS: int = 40
    LEARNING_RATE: float = 3e-4
    ATTENTION_HEADS: int = 4
    HIDDEN_DIM: int = 64
    QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)

    # --- MPC ---
    MPC_BASE_HORIZON: int = 8
    MPC_MAX_HORIZON: int = 16
    MPC_DEGRADATION_WEIGHT: float = 1.0
    MPC_UNCERTAINTY_WEIGHT_BASE: float = 0.5
    MPC_COMFORT_PENALTY: float = 50.0        # penalty per kWh of unmet critical load
    MPC_TERMINAL_SOC_TARGET: float = 0.5
    MPC_TERMINAL_WEIGHT: float = 1.0
    # Move-suppression (ramp-rate) penalty on |u_k - u_{k-1}|^2, applied
    # only by the proposed controller. This is a standard MPC ingredient
    # (protects actuators/battery power electronics from unnecessary
    # slew) and directly reduces switching/control-effort KPIs; the
    # classical baseline deliberately omits it (weight 0) to isolate what
    # the proposed method adds.
    MPC_MOVE_SUPPRESSION_WEIGHT: float = 8.0
    # Optional additional shadow price (currency/kWh) on grid import for
    # the proposed controller, representing that self-sufficiency has
    # value beyond the pure spot-price signal. Left at 0.0 by default:
    # empirically (closed-loop testing) a non-zero value here traded a
    # small self-sufficiency gain for a *disproportionate* increase in
    # cost, degradation and switching -- i.e. it fought the controller's
    # other objectives more than it helped. Kept as a tunable hook rather
    # than removed, since it may be worth revisiting once the deep-learning
    # forecaster (as opposed to the persistence fallback) is driving the
    # horizon, where the trade-off surface is different.
    MPC_SELF_SUFFICIENCY_WEIGHT: float = 0.0
    # Post-solve deadband (kW): if the solved action differs from the
    # previous applied action by less than this, hold the previous action
    # instead. This is standard practice in real inverter/BMS controllers
    # (avoids audible/mechanical relay chatter and needless switching
    # losses from micro-adjustments the economic model is roughly
    # indifferent to) and directly targets residual high-frequency
    # oscillation observed in closed-loop testing that the continuous
    # move-suppression penalty alone did not fully eliminate.
    MPC_DEADBAND_KW: float = 0.3
    DT_HOURS: float = 0.25                    # 15-minute control step, in hours

    # --- Execution toggles (edit before running on your machine) ---
    RUN_TRAINING: bool = True
    RUN_SIMULATION: bool = True
    RUN_PLOTS: bool = True
    SAVE_MODEL_CHECKPOINTS: bool = True
    DEVICE: str = "cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


@dataclass
class PlotConfig:
    """Publication-grade Matplotlib styling (Q1-journal ready)."""
    DPI: int = 600
    BASE_FONTSIZE: int = 16
    TITLE_FONTSIZE: int = 17
    LABEL_FONTSIZE: int = 16
    TICK_FONTSIZE: int = 13
    LEGEND_FONTSIZE: int = 12
    LINEWIDTH: float = 1.4
    FIGSIZE_SINGLE: Tuple[float, float] = (7.5, 5.0)
    FIGSIZE_2x2: Tuple[float, float] = (13.0, 10.0)
    FIGSIZE_3x2: Tuple[float, float] = (14.0, 13.5)
    FIGSIZE_3x3: Tuple[float, float] = (15.0, 13.0)
    COLOR_PV: str = "#F2A93B"
    COLOR_WIND: str = "#5DA6D6"
    COLOR_LOAD: str = "#E4645B"
    COLOR_BATT: str = "#5FA777"
    COLOR_GRID: str = "#8B6CB5"
    COLOR_PROPOSED: str = "#1F6F5C"
    COLOR_B1: str = "#C97A2B"
    COLOR_B2: str = "#B24A4A"
    COLOR_B3: str = "#5E6DA6"

    def apply(self) -> None:
        plt.rcParams.update({
            "figure.dpi": 100,           # on-screen; savefig overrides with DPI below
            "savefig.dpi": self.DPI,
            "font.size": self.BASE_FONTSIZE,
            "axes.titlesize": self.TITLE_FONTSIZE,
            "axes.labelsize": self.LABEL_FONTSIZE,
            "xtick.labelsize": self.TICK_FONTSIZE,
            "ytick.labelsize": self.TICK_FONTSIZE,
            "legend.fontsize": self.LEGEND_FONTSIZE,
            "lines.linewidth": self.LINEWIDTH,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        })


PATHS = PathConfig()
CFG = SimConfig()
PLOTCFG = PlotConfig()


def set_global_seed(seed: int) -> None:
    """Fix all RNGs for reproducibility across numpy / random / torch."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_logger(log_dir: str, name: str = "dt_aampc") -> logging.Logger:
    """Console + file logger."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOGGER = None  # initialised inside main()


# ==============================================================================
# 2. DATA LOADING & CLEANING
# ==============================================================================

class SmartHomeDataLoader:
    """
    Loads and cleans the "Augmented Smart Home with Weather Information"
    dataset (Rodriguez Vega & Syne, 2025), resamples it from 1-minute to a
    control-tractable resolution, and exposes a tidy, time-indexed DataFrame.
    """

    APPLIANCE_COLUMNS = [
        "Dishwasher", "Home office", "Fridge", "Wine cellar", "Garage door",
        "Barn", "Well", "Microwave", "Living room", "Furnace", "Kitchen",
    ]
    IOT_COLUMNS = [
        "Car charger [kW]", "Water heater [kW]", "Air conditioning [kW]",
        "Home Theater [kW]", "Outdoor lights [kW]", "microwave [kW]",
        "Laundry [kW]", "Pool Pump [kW]",
    ]
    WEATHER_COLUMNS = [
        "temperature", "humidity", "visibility", "apparentTemperature",
        "pressure", "windSpeed", "cloudCover", "windBearing",
        "precipIntensity", "dewPoint", "precipProbability",
    ]

    def __init__(self, path_cfg: PathConfig, sim_cfg: SimConfig,
                 logger: Optional[logging.Logger] = None):
        self.path_cfg = path_cfg
        self.sim_cfg = sim_cfg
        self.logger = logger or logging.getLogger("dt_aampc")

    # -------------------------------------------------------------- #
    def load_raw(self) -> pd.DataFrame:
        """Reads the raw CSV. Falls back to a synthetic stand-in dataset
        (with an explicit warning) if the real file cannot be located, so
        that the rest of the pipeline remains runnable end-to-end for
        smoke-testing purposes."""
        path = self.path_cfg.data_path
        if not os.path.isfile(path):
            self.logger.warning(
                "Data file not found at %s -- generating a small synthetic "
                "placeholder so the pipeline stays runnable. Place the real "
                "'HomeC_augmented.csv' at this path for actual results.", path
            )
            return self._synthetic_fallback()

        self.logger.info("Reading raw CSV from %s", path)
        df = pd.read_csv(path, index_col=0, low_memory=False)
        return df

    def _synthetic_fallback(self, n_days: int = 30) -> pd.DataFrame:
        n = n_days * 24 * 60
        idx = pd.date_range("2016-01-01 00:00", periods=n, freq="min")
        rng = np.random.default_rng(self.sim_cfg.RANDOM_SEED)
        hour = idx.hour + idx.minute / 60.0
        solar_shape = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        df = pd.DataFrame(index=idx)
        df["time"] = idx.strftime("%m/%d/%Y %H:%M")
        for c in self.APPLIANCE_COLUMNS:
            df[c] = np.abs(rng.normal(0.05, 0.02, n))
        for c in self.IOT_COLUMNS:
            df[c] = np.abs(rng.normal(0.08, 0.04, n))
        df["temperature"] = 15 + 10 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 2, n)
        df["humidity"] = np.clip(rng.normal(0.6, 0.15, n), 0, 1)
        df["visibility"] = np.clip(rng.normal(10, 2, n), 0, 16)
        df["apparentTemperature"] = df["temperature"] + rng.normal(0, 1, n)
        df["pressure"] = rng.normal(1015, 5, n)
        df["windSpeed"] = np.abs(rng.normal(6, 3, n))
        df["cloudCover"] = np.clip(rng.normal(0.4, 0.25, n), 0, 1)
        df["windBearing"] = rng.uniform(0, 360, n)
        df["precipIntensity"] = np.abs(rng.normal(0, 0.02, n))
        df["dewPoint"] = df["temperature"] - rng.normal(5, 2, n)
        df["precipProbability"] = np.clip(rng.normal(0.1, 0.1, n), 0, 1)
        df["use_HO"] = np.abs(0.5 + 1.2 * solar_shape + rng.normal(0, 0.1, n))
        df["gen_Sol"] = np.abs(self.sim_cfg.PV_RATED_KW * solar_shape *
                                (1 - 0.6 * df["cloudCover"]) + rng.normal(0, 0.05, n))
        return df

    # -------------------------------------------------------------- #
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses timestamps, drops duplicate/garbage rows, forward/back
        fills short gaps and clips physically-impossible negative power
        readings to zero."""
        df = df.copy()

        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], errors="coerce",
                                              format="mixed")
        else:
            df["timestamp"] = df.index

        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        power_like = [c for c in df.columns if
                      any(k in c for k in ["kW", "use_", "gen_"]) or
                      c in self.APPLIANCE_COLUMNS]
        for c in power_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)

        # Drop the now-redundant raw text time column and any other
        # non-numeric (categorical/string) columns before interpolating --
        # interpolation is only meaningful/valid on numeric channels.
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=15)
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        if non_numeric_cols:
            df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()

        self.logger.info("Cleaned dataframe shape: %s", df.shape)
        return df

    # -------------------------------------------------------------- #
    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsamples from 1-min to `SimConfig.RESAMPLE_RULE` using mean
        aggregation for power/weather channels (energy-conserving average)."""
        rule = self.sim_cfg.RESAMPLE_RULE
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        agg_map = {c: "mean" for c in numeric_cols}
        agg_map.update({c: "first" for c in non_numeric_cols})
        df_rs = df.resample(rule).agg(agg_map)

        df_rs[numeric_cols] = df_rs[numeric_cols].interpolate(method="linear")
        df_rs[numeric_cols] = df_rs[numeric_cols].ffill().bfill()
        if non_numeric_cols:
            df_rs[non_numeric_cols] = df_rs[non_numeric_cols].ffill().bfill()
        self.logger.info("Resampled to %s -> shape %s", rule, df_rs.shape)
        return df_rs

    # -------------------------------------------------------------- #
    def load(self) -> pd.DataFrame:
        raw = self.load_raw()
        cleaned = self.clean(raw)
        resampled = self.resample(cleaned)
        return resampled


# ==============================================================================
# 3. RENEWABLE SYNTHESIS  (PV / WIND / NET LOAD)
# ==============================================================================

class RenewableSynthesizer:
    """
    Derives PV power, synthetic small-wind-turbine power, and aggregate
    household load from the raw dataset columns + weather channels.

    PV:   uses the dataset's own `gen_Sol` (measured solar generation) as the
          base signal, re-scaled to a configurable nameplate rating and
          further modulated by `cloudCover` to allow scenario-level solar
          stress testing beyond what is naturally present in the log.

    WIND: synthesised from `windSpeed` via a standard cubic power-curve model
          (cut-in / rated / cut-out), since the source dataset has no wind
          generation channel of its own -- this is an explicit, documented
          augmentation.

    LOAD: aggregate of `use_HO` (whole-home smart-meter reading) plus the
          eight augmented IoT appliance channels, so that Car charger /
          Water heater / Air conditioning / Home Theater / Outdoor lights /
          microwave / Laundry / Pool Pump loads are all represented.
    """

    def __init__(self, sim_cfg: SimConfig):
        self.cfg = sim_cfg

    def wind_power_curve(self, wind_speed: np.ndarray) -> np.ndarray:
        v = np.asarray(wind_speed, dtype=float)
        p_rated = self.cfg.WIND_RATED_KW
        v_ci, v_r, v_co = (self.cfg.WIND_CUT_IN_MS, self.cfg.WIND_RATED_MS,
                           self.cfg.WIND_CUT_OUT_MS)
        p = np.zeros_like(v)
        ramp = (v >= v_ci) & (v < v_r)
        p[ramp] = p_rated * ((v[ramp] ** 3 - v_ci ** 3) / (v_r ** 3 - v_ci ** 3))
        rated = (v >= v_r) & (v < v_co)
        p[rated] = p_rated
        return np.clip(p, 0, p_rated)

    def pv_power(self, gen_sol: np.ndarray, cloud_cover: np.ndarray,
                 scale_to_rated: bool = True) -> np.ndarray:
        gen = np.asarray(gen_sol, dtype=float)
        cc = np.clip(np.asarray(cloud_cover, dtype=float), 0, 1)
        if scale_to_rated and gen.max() > 1e-6:
            gen = gen / gen.max() * self.cfg.PV_RATED_KW
        cloud_derate = (1 - cc) ** self.cfg.PV_DEGRADATION_CLOUD_EXP
        cloud_derate = 0.15 + 0.85 * cloud_derate  # never fully zero (diffuse irradiance)
        return np.clip(gen * cloud_derate, 0, self.cfg.PV_RATED_KW)

    def aggregate_load(self, df: pd.DataFrame,
                        loader_cols: SmartHomeDataLoader) -> np.ndarray:
        base = df.get("use_HO", pd.Series(0.0, index=df.index)).to_numpy()
        iot_total = np.zeros(len(df))
        for c in loader_cols.IOT_COLUMNS:
            if c in df.columns:
                iot_total = iot_total + df[c].to_numpy()
        load = base + iot_total
        return np.clip(load, 0, None)

    def build(self, df: pd.DataFrame, loader_cols: SmartHomeDataLoader
               ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["pv_kw"] = self.pv_power(df.get("gen_Sol", 0.0), df.get("cloudCover", 0.0))
        out["wind_kw"] = self.wind_power_curve(df.get("windSpeed", 0.0))
        out["load_kw"] = self.aggregate_load(df, loader_cols)
        out["net_load_kw"] = out["load_kw"] - out["pv_kw"] - out["wind_kw"]
        out["critical_load_kw"] = 0.35 * out["load_kw"]  # fridge/well/furnace-type load
        for c in RenewableSynthesizer._passthrough_weather():
            if c in df.columns:
                out[c] = df[c]

        # Cyclical calendar features (hour-of-day, day-of-week) as sin/cos
        # pairs. These are deterministic and carry no information the
        # network couldn't in principle infer from the raw sequence, but
        # explicitly exposing them is standard, well-established practice
        # in load/PV forecasting (diurnal and weekly periodicity is one of
        # the strongest, most exploitable signals in this domain) and
        # measurably speeds up and improves what a fixed-capacity network
        # can learn from a limited lookback window, rather than forcing it
        # to rediscover time-of-day purely from PV/load shape.
        hour_frac = out.index.hour + out.index.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
        dow = out.index.dayofweek.to_numpy()
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        return out

    @staticmethod
    def calendar_feature_names() -> List[str]:
        return ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    @staticmethod
    def _passthrough_weather() -> List[str]:
        return ["temperature", "humidity", "cloudCover", "windSpeed",
                "pressure", "dewPoint", "precipIntensity", "precipProbability"]


# ==============================================================================
# 4. SCENARIO GENERATOR
# ==============================================================================

@dataclass
class Scenario:
    name: str
    description: str
    start: pd.Timestamp
    end: pd.Timestamp
    perturbation: Callable[[pd.DataFrame], pd.DataFrame]
    data: Optional[pd.DataFrame] = None


class ScenarioGenerator:
    """
    Carves six distinct, non-overlapping stress-test windows out of the
    resampled dataset and applies a scenario-specific synthetic perturbation
    on top of the real historical trace, so that each scenario exercises a
    different operating regime of the BEMS. These six scenarios are
    deliberately different (in name, stressor and shape) from any
    previously published reference figure.
    """

    def __init__(self, sim_cfg: SimConfig, rng_seed: int = 42):
        self.cfg = sim_cfg
        self.rng = np.random.default_rng(rng_seed)

    # ---- perturbation functions -------------------------------------- #
    def _heatwave_ac_surge(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        heat_curve = 6.0 * np.sin(np.linspace(0, 6 * np.pi, n)) ** 2
        df["temperature"] = df["temperature"] + 8 + heat_curve
        ac_extra = np.clip((df["temperature"] - 28) * 0.9, 0, None)
        df["load_kw"] = df["load_kw"] + ac_extra
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    def _winter_storm_peak(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        storm = np.clip(np.sin(np.linspace(0, 4 * np.pi, n)), 0, None)
        df["windSpeed"] = df["windSpeed"] + 14 * storm
        df["cloudCover"] = np.clip(df["cloudCover"] + 0.5 * storm, 0, 1)
        heating_extra = 2.5 * storm
        df["load_kw"] = df["load_kw"] + heating_extra
        df["pv_kw"] = df["pv_kw"] * (1 - 0.7 * storm)
        wind_synth = RenewableSynthesizer(self.cfg).wind_power_curve(df["windSpeed"].to_numpy())
        df["wind_kw"] = wind_synth
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    def _cloud_cascade(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        cascade = 0.5 + 0.45 * np.sin(np.linspace(0, 2 * np.pi, n) - np.pi / 2)
        df["cloudCover"] = np.clip(cascade, 0, 1)
        pv_synth = RenewableSynthesizer(self.cfg).pv_power(df["pv_kw"].to_numpy(),
                                                            df["cloudCover"].to_numpy(),
                                                            scale_to_rated=False)
        df["pv_kw"] = pv_synth
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    def _ev_fleet_overnight(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        hours = df.index.hour + df.index.minute / 60.0
        night_mask = ((hours >= 22) | (hours < 6)).astype(float)
        ev_power = 6.6 * night_mask * (1 + 0.15 * self.rng.standard_normal(len(df)))
        df["load_kw"] = df["load_kw"] + np.clip(ev_power, 0, None)
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    def _islanded_fault_ride_through(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        fault_start = n // 3
        fault_len = max(4, n // 20)
        grid_available = np.ones(n)
        grid_available[fault_start:fault_start + fault_len] = 0.0
        df["grid_available"] = grid_available
        spike = np.zeros(n)
        spike[fault_start:fault_start + fault_len] += 1.8
        df["load_kw"] = df["load_kw"] + spike
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    def _solar_oversupply_curtailment(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        boost = 1.6 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, n))
        df["pv_kw"] = np.clip(df["pv_kw"] * boost, 0, self.cfg.PV_RATED_KW * 1.8)
        df["net_load_kw"] = df["load_kw"] - df["pv_kw"] - df["wind_kw"]
        return df

    # -------------------------------------------------------------- #
    def build_scenarios(self, df: pd.DataFrame, window_days: int = 6
                         ) -> List[Scenario]:
        """Slices six sequential (or, if the trace is too short, overlapping
        with replacement) windows from `df` and tags each with its
        perturbation function."""
        n_total = len(df)
        steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(self.cfg.RESAMPLE_RULE))
        win = window_days * steps_per_day
        win = min(win, max(steps_per_day, n_total // 6))

        specs = [
            ("Heatwave AC Surge",
             "Extreme ambient-temperature ramp driving a surge in air-"
             "conditioning demand while PV remains nominal.",
             self._heatwave_ac_surge),
            ("Winter Storm Load Peak",
             "High-wind, high-cloud winter storm event with elevated "
             "heating load and suppressed PV output.",
             self._winter_storm_peak),
            ("Cloud-Cover Cascade",
             "Slow cirrus-to-cumulus cloud regime transition causing a "
             "gradual, then abrupt, PV output collapse and recovery.",
             self._cloud_cascade),
            ("EV Fleet Overnight Charging",
             "Multiple electric vehicles charging simultaneously overnight, "
             "creating a sustained off-peak demand plateau.",
             self._ev_fleet_overnight),
            ("Islanded Microgrid Fault Ride-Through",
             "A temporary utility-grid outage forcing the home onto "
             "battery/renewables-only islanded operation with a coincident "
             "load spike.",
             self._islanded_fault_ride_through),
            ("Solar Oversupply Curtailment",
             "Unusually high irradiance week producing PV output well in "
             "excess of household demand, testing export/curtailment "
             "decisions.",
             self._solar_oversupply_curtailment),
        ]

        scenarios = []
        for i, (name, desc, fn) in enumerate(specs):
            start_idx = (i * win) % max(1, (n_total - win))
            end_idx = start_idx + win
            sub = df.iloc[start_idx:end_idx].copy()
            if "grid_available" not in sub.columns:
                sub["grid_available"] = 1.0
            perturbed = fn(sub)
            perturbed.attrs["scenario_name"] = name
            scenarios.append(Scenario(
                name=name, description=desc,
                start=perturbed.index[0], end=perturbed.index[-1],
                perturbation=fn, data=perturbed,
            ))
        return scenarios


# ==============================================================================
# 5. BATTERY DIGITAL TWIN  (physics model + Extended Kalman Filter)
# ==============================================================================

class BatteryPhysicsModel:
    """
    Lumped electro-thermal-degradation model of a residential Li-ion battery.

    State vector  x = [SOC, T_batt, SOH]^T
        SOC  : state of charge, dimensionless in [0, 1]
        T    : battery temperature, Kelvin
        SOH  : state of health (fractional remaining capacity), in [0, 1]

    Control input u = P_batt (kW), positive = charging, negative = discharging.
    """

    # Physically plausible battery-temperature bounds (K), used purely as a
    # defensive clamp against numerical excursions -- not a physical model
    # of thermal runaway.
    T_MIN_K: float = 250.0
    T_MAX_K: float = 400.0

    def __init__(self, sim_cfg: SimConfig):
        self.cfg = sim_cfg
        # First-order RC thermal time constant tau = R_th[K/W] * C_th[J/K].
        # BATT_THERMAL_RESIST is specified in K/kW -> convert to K/W (/1000).
        # BATT_THERMAL_MASS is specified in kJ/K -> convert to J/K (*1000).
        self._r_th_k_per_w = max(sim_cfg.BATT_THERMAL_RESIST / 1000.0, 1e-9)
        self._c_th_j_per_k = max(sim_cfg.BATT_THERMAL_MASS * 1000.0, 1e-9)
        self._tau_s = self._r_th_k_per_w * self._c_th_j_per_k

    def _delta_soh(self, t_batt_k: float, u_kw: float, dt_h: float) -> float:
        """Single source of truth for SOH loss over one step (calendar +
        cycling ageing, Arrhenius-scaled by battery temperature).

        Both the physical state update (`step`) and the economic
        degradation cost (`degradation_cost_kwh`) MUST call this same
        function. Previously they used two independently-drifted formulas
        -- `step` included the calendar-ageing term (K1) while the cost
        used in the controllers' optimisation did not, and the cost had no
        connection to the actual SOH signal at all. That mismatch meant a
        controller could optimise its "degradation cost" to zero while
        still destroying real SOH, defeating the entire purpose of a
        degradation-aware controller. Returns a NEGATIVE quantity (SOH
        lost)."""
        ah_throughput = abs(u_kw) * dt_h
        arrhenius = np.exp(-self.cfg.BATT_ARRHENIUS_EA / self.cfg.BATT_GAS_CONST *
                            (1.0 / max(t_batt_k, 1.0) - 1.0 / self.cfg.BATT_REF_TEMP_K))
        return -(self.cfg.BATT_DEGRADATION_K1 * dt_h +
                 self.cfg.BATT_DEGRADATION_K2 * ah_throughput) * arrhenius

    def step(self, x: np.ndarray, u_kw: float, t_amb_k: float, dt_h: float
              ) -> np.ndarray:
        soc, t_batt, soh = x
        cap_kwh = self.cfg.BATT_CAPACITY_KWH * max(soh, 1e-3)

        eta = self.cfg.BATT_ETA_CHG if u_kw >= 0 else 1.0 / self.cfg.BATT_ETA_DIS
        d_soc = (eta * u_kw * dt_h) / max(cap_kwh, 1e-6)
        soc_next = np.clip(soc + d_soc, 0.0, 1.0)

        p_loss_kw = (1 - self.cfg.BATT_ETA_CHG) * abs(u_kw) if u_kw >= 0 else \
                    (1 - self.cfg.BATT_ETA_DIS) * abs(u_kw)
        p_loss_w = p_loss_kw * 1000.0

        # Exact (unconditionally stable) discretisation of the first-order
        # RC thermal circuit  C*dT/dt = P_loss - (T - T_amb)/R_th, instead of
        # an explicit-Euler update. Explicit Euler is only stable when
        # dt << tau; with a 15-minute control step and a fast thermal time
        # constant, Euler diverges (this was the source of the earlier
        # OverflowError). The closed-form update below is stable for any
        # dt >= 0.
        dt_s = dt_h * 3600.0
        t_steady = t_amb_k + p_loss_w * self._r_th_k_per_w
        decay = math.exp(-dt_s / self._tau_s) if self._tau_s > 0 else 0.0
        t_next = t_steady + (t_batt - t_steady) * decay
        t_next = float(np.clip(t_next, self.T_MIN_K, self.T_MAX_K))

        d_soh = self._delta_soh(t_batt, u_kw, dt_h)
        soh_next = np.clip(soh + d_soh, 0.5, 1.0)

        out = np.array([soc_next, t_next, soh_next])
        # Final defensive guard: replace any stray NaN/Inf (e.g. from
        # pathological EKF perturbations during Jacobian estimation) with
        # the previous state, so a single bad sample can never propagate
        # into an unrecoverable simulation-wide overflow.
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=self.T_MAX_K, neginf=self.T_MIN_K)
            out[0] = np.clip(out[0], 0.0, 1.0)
            out[1] = np.clip(out[1], self.T_MIN_K, self.T_MAX_K)
            out[2] = np.clip(out[2], 0.5, 1.0)
        return out

    def degradation_cost_kwh(self, x: np.ndarray, u_kw: float, dt_h: float
                               ) -> float:
        """Marginal monetary cost (currency units) of applying `u_kw` for
        `dt_h`, attributable to battery ageing. Directly proportional to
        the *actual* SOH lost (via `_delta_soh`, the same function the
        physical state update uses) times the nameplate capacity times a
        replacement cost per kWh -- i.e. this literally prices "how much
        of the battery's replacement value was just consumed", so a
        controller that minimises this cost is, by construction,
        minimising real degradation, not a disconnected proxy for it."""
        _, t_batt, _ = x
        d_soh = self._delta_soh(t_batt, u_kw, dt_h)
        return abs(d_soh) * self.cfg.BATT_CAPACITY_KWH * self.cfg.BATT_REPLACEMENT_COST_PER_KWH


class BatteryDigitalTwin:
    """
    Extended Kalman Filter wrapping `BatteryPhysicsModel` to fuse noisy
    "sensor" measurements (simulated) with the physics-based state
    prediction, producing a bias-corrected state estimate `x_hat` and its
    covariance `P`, which the proposed controller consumes as its
    uncertainty-aware plant state.
    """

    def __init__(self, sim_cfg: SimConfig, measurement_noise_std: Tuple[float, float, float] = (0.01, 0.5, 0.005),
                 process_noise_std: Tuple[float, float, float] = (0.004, 0.2, 0.001)):
        self.cfg = sim_cfg
        self.model = BatteryPhysicsModel(sim_cfg)
        self.R = np.diag(np.array(measurement_noise_std) ** 2)
        self.Q = np.diag(np.array(process_noise_std) ** 2)
        self.x_hat = np.array([sim_cfg.BATT_SOC_INIT, sim_cfg.BATT_REF_TEMP_K,
                                sim_cfg.BATT_SOH_INIT])
        self.P = np.diag([1e-3, 1.0, 1e-4])

    def _jacobian(self, x: np.ndarray, u_kw: float, t_amb_k: float, dt_h: float,
                   eps: float = 1e-5) -> np.ndarray:
        """Numerical (central-difference) Jacobian of the nonlinear
        state-transition function, evaluated at the *actual* ambient
        temperature for this step (not a fixed reference), so the
        linearisation is locally accurate."""
        n = len(x)
        Jf = np.zeros((n, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = self.model.step(x + dx, u_kw, t_amb_k, dt_h)
            f_minus = self.model.step(x - dx, u_kw, t_amb_k, dt_h)
            Jf[:, i] = (f_plus - f_minus) / (2 * eps)
        return np.nan_to_num(Jf, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, u_kw: float, t_amb_k: float, dt_h: float) -> None:
        F_jac = self._jacobian(self.x_hat, u_kw, t_amb_k, dt_h)
        self.x_hat = self.model.step(self.x_hat, u_kw, t_amb_k, dt_h)
        self.P = F_jac @ self.P @ F_jac.T + self.Q
        self._sanitize()

    def update(self, z_meas: np.ndarray) -> None:
        """z_meas = [soc_sensor, temp_sensor, soh_estimate_from_bms]"""
        H = np.eye(3)
        y = z_meas - H @ self.x_hat
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        self.x_hat = self.x_hat + K @ y
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)
        self.x_hat[1] = np.clip(self.x_hat[1], BatteryPhysicsModel.T_MIN_K,
                                 BatteryPhysicsModel.T_MAX_K)
        self.x_hat[2] = np.clip(self.x_hat[2], 0.5, 1.0)
        self.P = (np.eye(3) - K @ H) @ self.P
        self._sanitize()

    def _sanitize(self, p_max: float = 1e4) -> None:
        """Defensive guard against numerical blow-up: replaces any
        non-finite entries in the state/covariance and caps the covariance
        magnitude, so a single ill-conditioned update can never propagate
        into a simulation-wide NaN/Inf cascade (and, downstream, into an
        unplottable figure)."""
        self.x_hat = np.nan_to_num(self.x_hat, nan=0.0, posinf=0.0, neginf=0.0)
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)
        self.x_hat[1] = np.clip(self.x_hat[1], BatteryPhysicsModel.T_MIN_K,
                                 BatteryPhysicsModel.T_MAX_K)
        self.x_hat[2] = np.clip(self.x_hat[2], 0.5, 1.0)

        self.P = np.nan_to_num(self.P, nan=0.0, posinf=p_max, neginf=-p_max)
        self.P = np.clip(self.P, -p_max, p_max)
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

    def step_and_correct(self, u_kw: float, t_amb_k: float, dt_h: float,
                          true_state: np.ndarray,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Advances the twin one control step and fuses a synthetic noisy
        measurement derived from the (simulated) ground-truth physical
        state. Returns (corrected_state, covariance)."""
        self.predict(u_kw, t_amb_k, dt_h)
        noise = rng.multivariate_normal(mean=np.zeros(3), cov=self.R)
        z = true_state + noise
        self.update(z)
        return self.x_hat.copy(), self.P.copy()

    @property
    def uncertainty_scalar(self) -> float:
        """Scalar confidence proxy = trace of covariance (normalised)."""
        return float(np.trace(self.P))


# ==============================================================================
# 6. DEEP LEARNING FORECASTER — "TAB-Net"
#    (Temporal Attention - BiLSTM Network, multi-horizon, multi-quantile)
# ==============================================================================

FORECAST_TARGETS = ["pv_kw", "wind_kw", "load_kw"]
FORECAST_FEATURES = (FORECAST_TARGETS + RenewableSynthesizer._passthrough_weather() +
                      RenewableSynthesizer.calendar_feature_names())


if _TORCH_AVAILABLE:

    class WindowedForecastDataset(Dataset):
        """Slides a (lookback -> horizon) window over a scaled multivariate
        time series to build supervised training tuples for TAB-Net."""

        def __init__(self, data: np.ndarray, lookback: int, horizon: int,
                     target_idx: List[int]):
            self.data = data.astype(np.float32)
            self.lookback = lookback
            self.horizon = horizon
            self.target_idx = target_idx
            self.n_samples = max(0, len(data) - lookback - horizon + 1)

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, i: int):
            x = self.data[i: i + self.lookback, :]
            y = self.data[i + self.lookback: i + self.lookback + self.horizon,
                          self.target_idx]
            return torch.from_numpy(x), torch.from_numpy(y)


    class CausalConvStem(nn.Module):
        """Stack of causal (left-padded) 1-D convolutions extracting local
        temporal motifs (e.g. sunrise ramps, appliance duty cycles)."""

        def __init__(self, in_ch: int, hidden: int, kernel_sizes=(3, 5, 7)):
            super().__init__()
            layers = []
            c_in = in_ch
            for k in kernel_sizes:
                layers.append(nn.Conv1d(c_in, hidden, kernel_size=k,
                                         padding=k - 1))
                layers.append(nn.GELU())
                layers.append(nn.BatchNorm1d(hidden))
                c_in = hidden
            self.net = nn.ModuleList(layers)
            self.kernel_sizes = kernel_sizes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C_in, T)
            out = x
            li = 0
            for k in self.kernel_sizes:
                conv, act, bn = self.net[li], self.net[li + 1], self.net[li + 2]
                out = conv(out)
                out = out[:, :, : -(k - 1)] if k > 1 else out  # causal trim
                out = act(out)
                out = bn(out)
                li += 3
            return out  # (B, hidden, T)


    class MultiHeadTemporalAttention(nn.Module):
        """Standard multi-head self-attention over the time axis, used to
        capture long-range dependencies (multi-day weather regimes, weekly
        occupancy patterns) beyond the causal-conv receptive field."""

        def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                              batch_first=True)
            self.ln = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            attn_out, attn_weights = self.mha(x, x, x, need_weights=True,
                                               average_attn_weights=True)
            out = self.ln(x + attn_out)
            return out, attn_weights

        @staticmethod
        def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
            """Shannon entropy of the attention distribution per query time
            step, averaged; used online as an inverse forecast-confidence
            signal (higher entropy = more diffuse attention = less
            confidence) fed into the adaptive MPC (see Section 8)."""
            eps = 1e-9
            ent = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            return ent.mean(dim=-1)  # (B,)


    class TABNet(nn.Module):
        """
        Temporal Attention-BiLSTM Network.

        Pipeline:  Causal-Conv stem -> Multi-Head Self-Attention -> BiLSTM ->
                   per-horizon, per-quantile linear heads, predicting a
                   DELTA from the last observed value of each target
                   (residual/persistence-anchored formulation -- see notes
                   in `forward`) rather than an absolute level.

        Outputs a tensor of shape (B, horizon, n_targets, n_quantiles).
        """

        def __init__(self, n_features: int, n_targets: int, horizon: int,
                     hidden: int = 64, n_heads: int = 4,
                     quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
            super().__init__()
            self.horizon = horizon
            self.n_targets = n_targets
            self.quantiles = quantiles
            self.n_q = len(quantiles)

            self.stem = CausalConvStem(n_features, hidden)
            self.attn = MultiHeadTemporalAttention(hidden, n_heads)
            self.bilstm = nn.LSTM(hidden, hidden, batch_first=True,
                                   bidirectional=True, num_layers=2,
                                   dropout=0.15)
            self.pool_proj = nn.Linear(hidden * 2, hidden)
            self.head = nn.Linear(hidden, horizon * n_targets * self.n_q)

            self._last_attn_weights: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C) -> conv wants (B, C, T)
            h = self.stem(x.transpose(1, 2)).transpose(1, 2)   # (B, T, hidden)
            h, attn_w = self.attn(h)
            self._last_attn_weights = attn_w.detach()
            h, _ = self.bilstm(h)                               # (B, T, 2*hidden)
            last = h[:, -1, :]
            proj = F.gelu(self.pool_proj(last))
            out = self.head(proj)
            out = out.view(-1, self.horizon, self.n_targets, self.n_q)

            # Residual / persistence-anchored formulation: the network
            # predicts a DELTA from the last observed (scaled) value of
            # each target, and that last value is added back here. Direct
            # absolute-level regression from a single pooled vector was
            # empirically underfitting badly (validation loss barely below
            # a flat mean-predictor, visibly worse than even a naive
            # persistence baseline on wind/load in early evaluation runs).
            # Learning the *change* is a substantially easier, better-
            # conditioned regression target and is standard practice in
            # load/renewable forecasting; by construction the model can
            # never do worse than persistence at initialisation (all-zero
            # deltas), and should improve from there.
            #
            # Targets are guaranteed to occupy the first `n_targets`
            # feature columns (see FORECAST_FEATURES = FORECAST_TARGETS +
            # weather_columns), so x[:, -1, :n_targets] is exactly the
            # last-observed target vector in the same scaled space as `out`.
            last_val = x[:, -1, :self.n_targets].unsqueeze(1).unsqueeze(-1)
            out = out + last_val

            # Enforce non-crossing quantiles (q_low <= q_mid <= q_high) by
            # sorting along the quantile axis -- quantile regression heads
            # trained independently can otherwise "cross" and produce a
            # nonsensical median outside its own [q_low, q_high] band.
            out, _ = torch.sort(out, dim=-1)
            return out

        def last_attention_confidence(self) -> float:
            if self._last_attn_weights is None:
                return 0.5
            ent = MultiHeadTemporalAttention.attention_entropy(self._last_attn_weights)
            ent_mean = float(ent.mean().item())
            t = self._last_attn_weights.shape[-1]
            max_ent = math.log(max(t, 2))
            confidence = 1.0 - np.clip(ent_mean / max_ent, 0.0, 1.0)
            return confidence


    def quantile_loss(preds: torch.Tensor, target: torch.Tensor,
                        quantiles: Tuple[float, ...]) -> torch.Tensor:
        """Pinball (quantile) loss averaged over horizon, targets, quantiles."""
        losses = []
        for qi, q in enumerate(quantiles):
            err = target - preds[..., qi]
            losses.append(torch.max((q - 1) * err, q * err))
        return torch.stack(losses, dim=-1).mean()


    def train_tabnet(model: "TABNet", train_loader: DataLoader,
                      val_loader: DataLoader, cfg: SimConfig,
                      logger: logging.Logger,
                      model_out_path: Optional[str] = None) -> Dict[str, List[float]]:
        """Standard supervised training loop with early stopping on
        validation pinball loss."""
        device = torch.device(cfg.DEVICE)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                                 weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience, bad_epochs = 6, 0

        for epoch in range(cfg.EPOCHS):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                preds = model(xb)
                loss = quantile_loss(preds, yb, model.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_losses.append(loss.item())
            sched.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_losses.append(quantile_loss(preds, yb, model.quantiles).item())

            tr_loss, va_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            logger.info("TAB-Net epoch %02d/%d | train %.5f | val %.5f",
                        epoch + 1, cfg.EPOCHS, tr_loss, va_loss)

            if va_loss < best_val - 1e-5:
                best_val = va_loss
                bad_epochs = 0
                if model_out_path:
                    torch.save(model.state_dict(), model_out_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return history

else:  # pragma: no cover - torch missing fallback stubs

    class TABNet:  # minimal stand-in so downstream code still imports
        def __init__(self, *a, **kw):
            raise RuntimeError("PyTorch is required for TABNet. `pip install torch`.")


class PersistenceForecaster:
    """Naive baseline forecaster: predicts that the next `horizon` steps
    equal the last observed value (classical, zero-parameter baseline used
    both as (a) a sanity baseline and (b) the forecaster driving the
    'Naive-Forecast MPC' controller B3)."""

    def __init__(self, horizon: int, n_targets: int):
        self.horizon = horizon
        self.n_targets = n_targets

    def predict(self, last_values: np.ndarray) -> np.ndarray:
        return np.tile(last_values.reshape(1, -1), (self.horizon, 1))


# ==============================================================================
# 7. CONTROLLERS — BASELINES
# ==============================================================================

class ControllerBase:
    """Common interface all controllers implement."""
    name: str = "base"

    def reset(self) -> None:
        pass

    def act(self, context: "ControlContext") -> float:
        """Returns battery setpoint power in kW (positive = charge)."""
        raise NotImplementedError


@dataclass
class ControlContext:
    """Everything a controller may need at a given control step."""
    t_index: int
    soc: float
    soh: float
    temp_k: float
    pv_kw: float
    wind_kw: float
    load_kw: float
    critical_load_kw: float
    grid_available: float
    hour: int
    forecast_pv: np.ndarray            # (horizon,)
    forecast_wind: np.ndarray          # (horizon,)
    forecast_load: np.ndarray          # (horizon,)
    forecast_confidence: float         # in [0, 1], 1 = fully confident
    twin_covariance_trace: float
    import_price: float
    export_price: float
    prev_batt_power_kw: float = 0.0


class PIDDroopController(ControllerBase):
    """
    B2 — Classical PID / droop feedback controller regulating battery
    power to track a fixed SOC set-point (a common approach in
    droop-controlled microgrid inverters), with anti-windup clamping.
    """
    name = "PID Droop"

    def __init__(self, sim_cfg: SimConfig, kp: float = 40.0, ki: float = 2.0,
                 kd: float = 1.5, soc_setpoint: float = 0.55):
        self.cfg = sim_cfg
        self.kp, self.ki, self.kd = kp, ki, kd
        self.soc_setpoint = soc_setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, ctx: ControlContext) -> float:
        error = self.soc_setpoint - ctx.soc
        self.integral = np.clip(self.integral + error, -5, 5)
        derivative = error - self.prev_error
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        net_renewable = ctx.pv_kw + ctx.wind_kw - ctx.load_kw
        u = u * 0.05 + net_renewable  # droop term blends feedback + power balance

        max_power = self.cfg.BATT_CAPACITY_KWH * self.cfg.BATT_MAX_C_RATE
        return float(np.clip(u, -max_power, max_power))


# ==============================================================================
# 8. CONTROLLERS — MPC (CLASSICAL BASELINE + PROPOSED DT-AAMPC)
# ==============================================================================

class _MPCCore:
    """Shared NLP-construction/solve utilities for both MPC variants, so the
    proposed method and the classical baseline differ *only* in the
    specific ingredients described in the module docstring (forecaster,
    twin feedback, adaptive weighting) and not in solver plumbing -- this
    keeps the comparison scientifically fair."""

    def __init__(self, sim_cfg: SimConfig, physics: BatteryPhysicsModel):
        self.cfg = sim_cfg
        self.physics = physics

    def _simulate_soc_path(self, soc0: float, u: np.ndarray, dt_h: float,
                             soh: float) -> np.ndarray:
        cap = self.cfg.BATT_CAPACITY_KWH * max(soh, 1e-3)
        eta_c, eta_d = self.cfg.BATT_ETA_CHG, self.cfg.BATT_ETA_DIS
        soc_path = np.empty(len(u) + 1)
        soc_path[0] = soc0
        for k, uk in enumerate(u):
            eta = eta_c if uk >= 0 else 1.0 / eta_d
            soc_path[k + 1] = np.clip(soc_path[k] + eta * uk * dt_h / cap, 0.0, 1.0)
        return soc_path

    def _cost(self, u: np.ndarray, ctx: ControlContext, horizon: int,
              pv_f: np.ndarray, wind_f: np.ndarray, load_f: np.ndarray,
              degradation_weight: float, uncertainty_weight: float,
              terminal_weight: float, comfort_penalty: float,
              soc_backoff: float, move_suppression_weight: float = 0.0,
              self_sufficiency_weight: float = 0.0) -> float:
        dt_h = self.cfg.DT_HOURS
        soc_path = self._simulate_soc_path(ctx.soc, u, dt_h, ctx.soh)

        grid_power = load_f - pv_f - wind_f - u  # >0 import, <0 export
        import_power = np.clip(grid_power, 0, None)
        export_power = np.clip(-grid_power, 0, None)

        is_peak = np.array([
            self.cfg.PEAK_HOURS[0] <= ((ctx.hour + k) % 24) < self.cfg.PEAK_HOURS[1]
            for k in range(horizon)
        ])
        price_import = np.where(is_peak, self.cfg.GRID_IMPORT_PRICE_PEAK,
                                 self.cfg.GRID_IMPORT_PRICE_OFFPEAK)
        econ_cost = np.sum(import_power * price_import * dt_h) - \
                    np.sum(export_power * self.cfg.GRID_EXPORT_PRICE * dt_h)

        degr_cost = sum(
            self.physics.degradation_cost_kwh(
                np.array([soc_path[k], ctx.temp_k, ctx.soh]), u[k], dt_h)
            for k in range(horizon)
        ) * degradation_weight

        soc_lo = self.cfg.BATT_SOC_MIN + soc_backoff
        soc_hi = self.cfg.BATT_SOC_MAX - soc_backoff
        viol = np.sum(np.clip(soc_lo - soc_path, 0, None) ** 2) + \
               np.sum(np.clip(soc_path - soc_hi, 0, None) ** 2)
        viol_cost = viol * 500.0

        unmet_critical = np.clip(ctx.critical_load_kw - pv_f - wind_f -
                                  np.clip(-u, 0, None), 0, None)
        comfort_cost = np.sum(unmet_critical) * comfort_penalty * dt_h

        # Terminal SOC cost with a soft dead-zone: previously this was a
        # hard quadratic pull toward exactly MPC_TERMINAL_SOC_TARGET at
        # EVERY receding-horizon re-solve, regardless of whether the
        # current SOC was already perfectly reasonable. Because the
        # controller re-optimises every single step, that hard pull forced
        # the battery to keep chasing the midpoint even when doing nothing
        # was strictly better -- pure wasted charge/discharge cycling with
        # zero economic benefit. This was the root cause of the SOC
        # chattering and cost blow-ups seen in closed-loop testing (and it
        # affected the classical-MPC baseline too, since it shares this
        # cost function). The dead-zone below only penalises the terminal
        # SOC once it drifts outside a tolerance band around the target,
        # so the controller is free to leave SOC wherever the *other*
        # (economic/degradation) costs actually want it, and only gets
        # pulled back once genuinely close to running dry or saturating.
        terminal_band = 0.15
        terminal_dev = max(abs(soc_path[-1] - self.cfg.MPC_TERMINAL_SOC_TARGET)
                            - terminal_band, 0.0)
        terminal_cost = terminal_weight * terminal_dev ** 2

        uncertainty_cost = uncertainty_weight * np.var(pv_f + wind_f - load_f)

        u_full = np.concatenate(([ctx.prev_batt_power_kw], u))
        move_cost = move_suppression_weight * np.sum(np.diff(u_full) ** 2)

        self_suff_cost = self_sufficiency_weight * np.sum(import_power) * dt_h

        return (econ_cost + degr_cost + viol_cost + comfort_cost +
                terminal_cost + uncertainty_cost + move_cost + self_suff_cost)

    def solve(self, ctx: ControlContext, horizon: int, pv_f: np.ndarray,
              wind_f: np.ndarray, load_f: np.ndarray, degradation_weight: float,
              uncertainty_weight: float, terminal_weight: float,
              comfort_penalty: float, soc_backoff: float,
              move_suppression_weight: float = 0.0,
              self_sufficiency_weight: float = 0.0) -> np.ndarray:
        max_power = self.cfg.BATT_CAPACITY_KWH * self.cfg.BATT_MAX_C_RATE
        bounds = [(-max_power, max_power)] * horizon
        args = (ctx, horizon, pv_f, wind_f, load_f, degradation_weight,
                uncertainty_weight, terminal_weight, comfort_penalty,
                soc_backoff, move_suppression_weight, self_sufficiency_weight)

        # SLSQP on this non-convex NLP is prone to stopping at a poor local
        # optimum depending on the starting point (verified in testing: it
        # returned a CHARGING action costing ~60 when a simple full-
        # discharge candidate costs ~36 for the same state -- i.e. worse
        # than an obviously better, easy-to-find alternative). Guard
        # against this with a small multi-start: evaluate a handful of
        # cheap, sensible candidate trajectories directly, seed SLSQP from
        # the best of them, and finally keep whichever of {candidates,
        # SLSQP result} truly has the lowest cost. This never makes the
        # optimizer worse and reliably avoids the bad-local-optimum failure
        # mode observed above.
        net_load = load_f - pv_f - wind_f
        greedy = np.clip(-net_load, -max_power, max_power)  # discharge to cover
                                                              # deficit / charge with surplus
        candidates = [
            np.full(horizon, ctx.prev_batt_power_kw),
            np.zeros(horizon),
            greedy,
            np.full(horizon, max_power),
            np.full(horizon, -max_power),
        ]
        candidates = [np.clip(c, -max_power, max_power) for c in candidates]

        best_u = min(candidates, key=lambda c: self._cost(c, *args))
        best_cost = self._cost(best_u, *args)

        res = spopt.minimize(
            self._cost, best_u, args=args,
            method="SLSQP", bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-7},
        )
        if res.success:
            res_cost = self._cost(res.x, *args)
            if res_cost < best_cost:
                best_u, best_cost = res.x, res_cost

        # Post-solve deadband on the FIRST (i.e. actually-applied) action
        # only: if it's within MPC_DEADBAND_KW of the previous applied
        # action, hold rather than chatter. The rest of the planned
        # trajectory (used only internally for the cost evaluation) is
        # left untouched.
        if abs(best_u[0] - ctx.prev_batt_power_kw) < self.cfg.MPC_DEADBAND_KW:
            best_u = best_u.copy()
            best_u[0] = ctx.prev_batt_power_kw

        return best_u


class ClassicalNaiveForecastMPC(ControllerBase):
    """
    B3 — Deterministic economic MPC baseline. Uses a *persistence*
    forecast (no deep learning), a *fixed* short horizon, *no* battery
    digital-twin feedback (assumes nominal, unbiased SOC/temperature/SOH)
    and *no* adaptive uncertainty weighting. This isolates exactly what the
    proposed DT-AAMPC adds.
    """
    name = "Classical MPC (Naive Forecast)"

    def __init__(self, sim_cfg: SimConfig):
        self.cfg = sim_cfg
        self.core = _MPCCore(sim_cfg, BatteryPhysicsModel(sim_cfg))

    def act(self, ctx: ControlContext) -> float:
        horizon = self.cfg.MPC_BASE_HORIZON
        pv_f = np.full(horizon, ctx.pv_kw)
        wind_f = np.full(horizon, ctx.wind_kw)
        load_f = np.full(horizon, ctx.load_kw)
        u = self.core.solve(
            ctx, horizon, pv_f, wind_f, load_f,
            degradation_weight=0.0,           # classical baseline ignores degradation cost
            uncertainty_weight=0.0,           # and forecast uncertainty
            terminal_weight=self.cfg.MPC_TERMINAL_WEIGHT,
            comfort_penalty=self.cfg.MPC_COMFORT_PENALTY,
            soc_backoff=0.0,
            move_suppression_weight=0.0,      # and ramp-rate/move suppression
        )
        return float(u[0])


class DigitalTwinAttentionAdaptiveMPC(ControllerBase):
    """
    *** PROPOSED NOVEL METHOD ***  "DT-AAMPC"

    Digital-Twin Attention-Augmented Adaptive Model Predictive Control.

    Differences versus the classical baseline (B3) above:
        1. Forecast `pv_f`, `wind_f`, `load_f` come from TAB-Net's median
           quantile (deep learning, attention-based, multi-horizon) rather
           than persistence.
        2. `ctx.soc` / `ctx.soh` / `ctx.temp_k` are the EKF-corrected
           digital-twin states, not raw noisy sensor values.
        3. The prediction horizon, the forecast-uncertainty cost weight and
           the SOC chance-constraint back-off margin are all *adapted
           online*:
               horizon         = base + confidence * (max - base)
               uncertainty_wt  = base_weight * (1 + twin_covariance_trace)
               soc_backoff     = 0.02 + 0.10 * (1 - confidence)
           i.e. when the forecaster is confident AND the twin's state
           estimate is tight, DT-AAMPC looks further ahead and takes a
           tighter, more assertive economic optimum; when either signal
           degrades, it automatically becomes more conservative and
           short-sighted -- a *risk-aware* adaptation absent from B1-B3.
        4. The battery-degradation cost (from the twin's Arrhenius-scaled
           throughput model, priced directly off real SOH loss) is
           included with full weight, and a move-suppression (ramp-rate)
           term further discourages unnecessary power reversals that the
           baselines do not penalise.
        5. On a *detected* grid-outage step (`ctx.grid_available < 0.5`),
           the controller bypasses the economic NLP entirely and switches
           to a digital-twin-informed emergency dispatch: it immediately
           re-prioritises battery power to cover the household's real-time
           load from the EKF-corrected (bias-free) state, rather than
           continuing to chase a stale economic optimum computed under a
           now-invalid grid-connected assumption. Neither baseline
           has an equivalent explicit islanding response.
    """
    name = "DT-AAMPC (Proposed)"

    def __init__(self, sim_cfg: SimConfig):
        self.cfg = sim_cfg
        self.core = _MPCCore(sim_cfg, BatteryPhysicsModel(sim_cfg))

    def act(self, ctx: ControlContext) -> float:
        max_power = self.cfg.BATT_CAPACITY_KWH * self.cfg.BATT_MAX_C_RATE

        if ctx.grid_available < 0.5:
            # Digital-twin emergency dispatch: serve load from renewables
            # first, then the battery, using the twin's bias-corrected SOC
            # rather than a raw sensor reading. This is a direct, testable
            # consequence of maintaining a digital twin that the rule-based
            # / PID / classical-MPC baselines simply do not have.
            renewable = ctx.pv_kw + ctx.wind_kw
            deficit = ctx.load_kw - renewable
            return float(np.clip(-deficit, -max_power, max_power))

        confidence = np.clip(ctx.forecast_confidence, 0.0, 1.0)
        horizon = int(np.clip(
            self.cfg.MPC_BASE_HORIZON +
            confidence * (self.cfg.MPC_MAX_HORIZON - self.cfg.MPC_BASE_HORIZON),
            self.cfg.MPC_BASE_HORIZON, self.cfg.MPC_MAX_HORIZON))

        pv_f = ctx.forecast_pv[:horizon]
        wind_f = ctx.forecast_wind[:horizon]
        load_f = ctx.forecast_load[:horizon]
        if len(pv_f) < horizon:  # pad tail if forecast shorter than horizon
            pad = horizon - len(pv_f)
            pv_f = np.pad(pv_f, (0, pad), mode="edge")
            wind_f = np.pad(wind_f, (0, pad), mode="edge")
            load_f = np.pad(load_f, (0, pad), mode="edge")

        uncertainty_weight = self.cfg.MPC_UNCERTAINTY_WEIGHT_BASE * \
            (1.0 + ctx.twin_covariance_trace)
        soc_backoff = 0.02 + 0.10 * (1.0 - confidence)

        u = self.core.solve(
            ctx, horizon, pv_f, wind_f, load_f,
            degradation_weight=self.cfg.MPC_DEGRADATION_WEIGHT,
            uncertainty_weight=uncertainty_weight,
            terminal_weight=self.cfg.MPC_TERMINAL_WEIGHT,
            comfort_penalty=self.cfg.MPC_COMFORT_PENALTY,
            soc_backoff=soc_backoff,
            move_suppression_weight=self.cfg.MPC_MOVE_SUPPRESSION_WEIGHT,
            self_sufficiency_weight=self.cfg.MPC_SELF_SUFFICIENCY_WEIGHT,
        )
        return float(u[0])


# ==============================================================================
# 9. SIMULATION ENGINE
# ==============================================================================

@dataclass
class SimulationResult:
    scenario_name: str
    controller_name: str
    timestamps: pd.DatetimeIndex
    soc: np.ndarray
    soh: np.ndarray
    temp_k: np.ndarray
    batt_power: np.ndarray
    pv: np.ndarray
    wind: np.ndarray
    load: np.ndarray
    grid_power: np.ndarray
    import_cost: np.ndarray
    export_revenue: np.ndarray
    degradation_cost: np.ndarray
    unmet_critical_kwh: np.ndarray
    forecast_confidence: np.ndarray
    solve_time_s: np.ndarray
    seed: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "soc": self.soc, "soh": self.soh, "temp_k": self.temp_k,
            "batt_power_kw": self.batt_power, "pv_kw": self.pv,
            "wind_kw": self.wind, "load_kw": self.load,
            "grid_power_kw": self.grid_power, "import_cost": self.import_cost,
            "export_revenue": self.export_revenue,
            "degradation_cost": self.degradation_cost,
            "unmet_critical_kwh": self.unmet_critical_kwh,
            "forecast_confidence": self.forecast_confidence,
            "solve_time_s": self.solve_time_s,
        }, index=self.timestamps)


class ForecastProvider:
    """Wraps a trained TAB-Net (or persistence fallback) to serve rolling
    multi-horizon forecasts + a scalar confidence signal during simulation."""

    def __init__(self, sim_cfg: SimConfig, feature_scaler: StandardScaler,
                 target_indices: List[int], model: Optional["TABNet"] = None):
        self.cfg = sim_cfg
        self.scaler = feature_scaler
        self.target_indices = target_indices
        self.model = model
        self.device = torch.device(sim_cfg.DEVICE) if _TORCH_AVAILABLE else None
        if self.model is not None and _TORCH_AVAILABLE:
            self.model.eval()

    def forecast(self, history_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """history_window: (lookback, n_features) raw units."""
        horizon = self.cfg.HORIZON_STEPS
        if self.model is None or not _TORCH_AVAILABLE:
            last = history_window[-1, self.target_indices]
            pv_f = np.full(horizon, max(last[0], 0))
            wind_f = np.full(horizon, max(last[1], 0))
            load_f = np.full(horizon, max(last[2], 0))
            return pv_f, wind_f, load_f, 0.4  # fixed, low confidence: naive forecaster

        scaled = self.scaler.transform(history_window)
        x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)  # (1, horizon, n_targets, n_q)
        median_idx = len(self.model.quantiles) // 2
        preds_scaled = out[0, :, :, median_idx].cpu().numpy()  # (horizon, n_targets)

        n_features = self.scaler.mean_.shape[0]
        buf = np.zeros((horizon, n_features))
        buf[:, self.target_indices] = preds_scaled
        inv = self.scaler.inverse_transform(buf)
        preds = np.clip(inv[:, self.target_indices], 0, None)

        confidence = self.model.last_attention_confidence()
        return preds[:, 0], preds[:, 1], preds[:, 2], float(confidence)


class SmartGridSimulator:
    """
    Closed-loop simulation engine: for a given scenario + controller,
    steps through every 15-minute interval, queries the forecast provider,
    (for DT-AAMPC / classical MPC) advances + corrects the battery digital
    twin, calls the controller for a battery setpoint, applies physical
    battery dynamics, and books grid/economic/degradation outcomes.
    """

    def __init__(self, sim_cfg: SimConfig, rng_seed: int = 0):
        self.cfg = sim_cfg
        self.physics = BatteryPhysicsModel(sim_cfg)
        self._default_seed = rng_seed

    def run(self, scenario: Scenario, controller: ControllerBase,
            forecast_provider: ForecastProvider, feature_cols: List[str],
            use_digital_twin: bool, seed: Optional[int] = None) -> SimulationResult:
        seed = self._default_seed if seed is None else seed
        rng = np.random.default_rng(seed)
        df = scenario.data
        n = len(df)
        lookback = self.cfg.LOOKBACK_STEPS
        dt_h = self.cfg.DT_HOURS

        controller.reset()
        twin = BatteryDigitalTwin(self.cfg) if use_digital_twin else None

        true_state = np.array([self.cfg.BATT_SOC_INIT, self.cfg.BATT_REF_TEMP_K,
                                self.cfg.BATT_SOH_INIT])
        prev_u_kw = 0.0

        soc_arr, soh_arr, temp_arr = (np.zeros(n) for _ in range(3))
        batt_p, grid_p, imp_cost, exp_rev, degr_cost = (np.zeros(n) for _ in range(5))
        unmet_arr, conf_arr, solve_t = (np.zeros(n) for _ in range(3))

        feature_matrix = df[feature_cols].to_numpy()

        for k in range(n):
            t_amb_k = df["temperature"].iloc[k] + 273.15
            hour = df.index[k].hour
            grid_avail = df["grid_available"].iloc[k] if "grid_available" in df.columns else 1.0

            hist_start = max(0, k - lookback + 1)
            hist_window = feature_matrix[hist_start:k + 1]
            if len(hist_window) < lookback:
                pad = np.repeat(hist_window[:1], lookback - len(hist_window), axis=0)
                hist_window = np.vstack([pad, hist_window])

            pv_f, wind_f, load_f, confidence = forecast_provider.forecast(hist_window)

            # Fairness-critical: EVERY controller observes only a noisy
            # sensor reading of the battery state -- never the ground-
            # truth `true_state` directly. Previously the baselines
            # were (silently) given perfect, noise-free SOC/temperature/SOH
            # while only DT-AAMPC had to work through noisy measurements
            # filtered by its EKF. That inverted the entire point of the
            # digital twin (a twin's value is that it denoises what would
            # otherwise be a noisy raw sensor feed) and structurally
            # disadvantaged the proposed method in the comparison. Now
            # every controller sees a reading built from the SAME noise
            # draw/scale; only DT-AAMPC gets to filter it via the EKF --
            # the baselines must act on the raw noisy value, exactly as a
            # real deployed system without a twin would have to.
            sensor_noise = rng.normal(0, [0.008, 0.3, 0.003])
            raw_reading = true_state + sensor_noise
            raw_reading[0] = np.clip(raw_reading[0], 0.0, 1.0)
            raw_reading[1] = np.clip(raw_reading[1], BatteryPhysicsModel.T_MIN_K,
                                      BatteryPhysicsModel.T_MAX_K)
            raw_reading[2] = np.clip(raw_reading[2], 0.5, 1.0)

            if twin is not None:
                obs_state, cov = twin.x_hat, twin.P
                cov_trace = twin.uncertainty_scalar
                state_for_ctrl_soc, state_for_ctrl_temp, state_for_ctrl_soh = obs_state
            else:
                state_for_ctrl_soc, state_for_ctrl_temp, state_for_ctrl_soh = raw_reading
                cov_trace = 0.0

            ctx = ControlContext(
                t_index=k, soc=float(state_for_ctrl_soc), soh=float(state_for_ctrl_soh),
                temp_k=float(state_for_ctrl_temp), pv_kw=float(df["pv_kw"].iloc[k]),
                wind_kw=float(df["wind_kw"].iloc[k]), load_kw=float(df["load_kw"].iloc[k]),
                critical_load_kw=float(df["critical_load_kw"].iloc[k]),
                grid_available=float(grid_avail), hour=int(hour),
                forecast_pv=pv_f, forecast_wind=wind_f, forecast_load=load_f,
                forecast_confidence=float(confidence), twin_covariance_trace=float(cov_trace),
                import_price=(self.cfg.GRID_IMPORT_PRICE_PEAK
                              if self.cfg.PEAK_HOURS[0] <= hour < self.cfg.PEAK_HOURS[1]
                              else self.cfg.GRID_IMPORT_PRICE_OFFPEAK),
                export_price=self.cfg.GRID_EXPORT_PRICE,
                prev_batt_power_kw=prev_u_kw,
            )

            t0 = time.perf_counter()
            u_kw = controller.act(ctx)
            solve_t[k] = time.perf_counter() - t0

            max_power = self.cfg.BATT_CAPACITY_KWH * self.cfg.BATT_MAX_C_RATE
            u_kw = float(np.clip(u_kw, -max_power, max_power))

            # Physical energy-availability clamp -- applied identically to
            # EVERY controller (not just the proposed one), so no method
            # gets to request more discharge than the battery can actually
            # deliver from its true current SOC, or more charge than the
            # remaining headroom accepts, within one control step.
            soc0, _, soh0 = true_state
            cap0 = self.cfg.BATT_CAPACITY_KWH * max(soh0, 1e-3)
            max_discharge_kw = soc0 * cap0 * self.cfg.BATT_ETA_DIS / dt_h
            max_charge_kw = (1.0 - soc0) * cap0 / self.cfg.BATT_ETA_CHG / dt_h
            u_kw = float(np.clip(u_kw, -max_discharge_kw, max_charge_kw))

            true_state = self.physics.step(true_state, u_kw, t_amb_k, dt_h)
            degr = self.physics.degradation_cost_kwh(true_state, u_kw, dt_h)

            if twin is not None:
                meas_noise = rng.normal(0, [0.008, 0.3, 0.003])
                twin.step_and_correct(u_kw, t_amb_k, dt_h,
                                       true_state + meas_noise, rng)

            load_k = df["load_kw"].iloc[k]
            pv_k = df["pv_kw"].iloc[k]
            wind_k = df["wind_kw"].iloc[k]
            critical_k = df["critical_load_kw"].iloc[k]
            raw_balance = load_k - pv_k - wind_k - u_kw  # >0 shortfall, <0 surplus

            if grid_avail >= 0.5:
                # Grid-connected: any imbalance is simply imported/exported.
                grid_power = raw_balance
                import_p = max(grid_power, 0.0)
                export_p = max(-grid_power, 0.0)
                unmet = 0.0
            else:
                # Islanded: no import or export is physically possible. A
                # shortfall (raw_balance > 0) is first absorbed by shedding
                # non-critical load; only the remainder that exceeds the
                # home's non-critical shedding capacity (i.e. that eats
                # into critical_load_kw) counts as a genuine comfort
                # violation. A surplus (raw_balance < 0) is curtailed
                # (wasted), not exported. Crucially, this now depends on
                # each controller's OWN u_kw -- previously this branch
                # overwrote u_kw with a single hard-coded formula, making
                # every controller behave identically during the outage
                # and hiding any real difference between them.
                grid_power = 0.0
                import_p = 0.0
                export_p = 0.0
                shortfall = max(raw_balance, 0.0)
                non_critical_kw = max(load_k - critical_k, 0.0)
                unmet = max(shortfall - non_critical_kw, 0.0)

            soc_arr[k], temp_arr[k], soh_arr[k] = true_state
            batt_p[k] = u_kw
            grid_p[k] = grid_power
            imp_cost[k] = import_p * ctx.import_price * dt_h
            exp_rev[k] = export_p * ctx.export_price * dt_h
            degr_cost[k] = degr
            unmet_arr[k] = unmet * dt_h
            conf_arr[k] = confidence
            prev_u_kw = u_kw

        return SimulationResult(
            scenario_name=scenario.name, controller_name=controller.name,
            timestamps=df.index, soc=soc_arr, soh=soh_arr, temp_k=temp_arr,
            batt_power=batt_p, pv=df["pv_kw"].to_numpy(), wind=df["wind_kw"].to_numpy(),
            load=df["load_kw"].to_numpy(), grid_power=grid_p, import_cost=imp_cost,
            export_revenue=exp_rev, degradation_cost=degr_cost,
            unmet_critical_kwh=unmet_arr, forecast_confidence=conf_arr,
            solve_time_s=solve_t, seed=seed,
        )


# ==============================================================================
# 10. METRICS
# ==============================================================================

def compute_scenario_metrics(res: SimulationResult, cfg: SimConfig) -> Dict[str, float]:
    """Aggregates a single (scenario, controller) run into publication
    metrics: economic cost, self-sufficiency, degradation, comfort
    violations, control effort/switching and average solve time."""
    net_cost = float(np.sum(res.import_cost) - np.sum(res.export_revenue) +
                     np.sum(res.degradation_cost))
    total_load_kwh = float(np.sum(res.load) * cfg.DT_HOURS)

    # NOTE: self-sufficiency must be measured through realised grid import,
    # not through a raw min(PV+Wind, Load) overlap. The latter is purely a
    # function of the (controller-independent) exogenous PV/Wind/Load
    # traces and is therefore IDENTICAL for every controller in a given
    # scenario -- it cannot credit a controller for battery-mediated
    # self-consumption (charging on surplus renewables, discharging later
    # to cover load) and was silently making this KPI uninformative.
    grid_import_kwh = float(np.sum(np.clip(res.grid_power, 0, None)) * cfg.DT_HOURS)
    self_sufficiency = float(np.clip(1.0 - grid_import_kwh / max(total_load_kwh, 1e-6),
                                      0.0, 1.0))

    soh_drop_pct = float((res.soh[0] - res.soh[-1]) * 100.0)
    total_degr_cost = float(np.sum(res.degradation_cost))
    comfort_violation_kwh = float(np.sum(res.unmet_critical_kwh))

    batt_switch = np.sum(np.abs(np.diff(np.sign(res.batt_power + 1e-9))) > 0)
    control_effort = float(np.sum(np.abs(np.diff(res.batt_power))))

    peak_mask = np.array([
        cfg.PEAK_HOURS[0] <= ts.hour < cfg.PEAK_HOURS[1] for ts in res.timestamps
    ])
    peak_grid_import = float(np.sum(np.clip(res.grid_power, 0, None)[peak_mask]) * cfg.DT_HOURS)

    soc_violations = int(np.sum((res.soc < cfg.BATT_SOC_MIN - 1e-6) |
                                 (res.soc > cfg.BATT_SOC_MAX + 1e-6)))

    return {
        "net_cost": net_cost,
        "self_sufficiency_pct": self_sufficiency * 100.0,
        "soh_drop_pct": soh_drop_pct,
        "degradation_cost": total_degr_cost,
        "comfort_violation_kwh": comfort_violation_kwh,
        "battery_switching_events": int(batt_switch),
        "control_effort_kw": control_effort,
        "peak_hour_grid_import_kwh": peak_grid_import,
        "soc_bound_violations": soc_violations,
        "avg_solve_time_ms": float(np.mean(res.solve_time_s) * 1000.0),
        "avg_forecast_confidence": float(np.mean(res.forecast_confidence)),
    }


def build_metrics_table(results: List[SimulationResult], cfg: SimConfig) -> pd.DataFrame:
    rows = []
    for r in results:
        m = compute_scenario_metrics(r, cfg)
        m["scenario"] = r.scenario_name
        m["controller"] = r.controller_name
        m["seed"] = r.seed
        rows.append(m)
    return pd.DataFrame(rows)


def statistical_significance_tests(metrics_df: pd.DataFrame,
                                     proposed_name: str,
                                     baseline_names: List[str],
                                     metric_cols: List[str]) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test (proposed vs each baseline) across
    every (scenario x Monte-Carlo seed) replicate, for each headline
    metric. Using every replicate as an independent sample (rather than
    just the 6 scenario means) is what actually lets a Q1-level
    significance claim be made: a difference that only shows up after
    averaging away run-to-run noise is not yet a demonstrated effect."""
    rows = []
    if "seed" in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["_block"] = (metrics_df["scenario"].astype(str) + "__seed" +
                                 metrics_df["seed"].astype(str))
    else:
        metrics_df = metrics_df.copy()
        metrics_df["_block"] = metrics_df["scenario"].astype(str)

    for metric in metric_cols:
        pivot = metrics_df.pivot_table(index="_block", columns="controller",
                                        values=metric, aggfunc="mean")
        for baseline in baseline_names:
            if proposed_name not in pivot.columns or baseline not in pivot.columns:
                continue
            paired = pivot[[proposed_name, baseline]].dropna()
            a = paired[proposed_name].to_numpy()
            b = paired[baseline].to_numpy()
            diff = a - b
            if len(a) < 2 or np.allclose(diff, 0):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = spstats.wilcoxon(a, b)
                except ValueError:
                    stat, p = np.nan, np.nan
            improvement_pct = float(np.mean((b - a) / (np.abs(b) + 1e-9)) * 100.0)
            rows.append({
                "metric": metric, "baseline": baseline, "n_replicates": len(a),
                "proposed_mean": float(np.mean(a)), "baseline_mean": float(np.mean(b)),
                "mean_improvement_pct": improvement_pct,
                "wilcoxon_stat": stat, "p_value": p,
                "significant_at_0.05": bool(p < 0.05) if p is not None and not np.isnan(p) else False,
            })
    return pd.DataFrame(rows)


# ==============================================================================
# 11. PLOTTING  (600 DPI, base fontsize 16; mixes single / 2x2 / 3x2 / 3x3)
# ==============================================================================

def _safe_array(arr: np.ndarray) -> np.ndarray:
    """Belt-and-suspenders guard used right before handing any array to
    Matplotlib: replaces NaN/Inf with finite values so a single corrupted
    sample (e.g. from an ill-conditioned solver step upstream) can never
    crash tick-locator math (`OverflowError: cannot convert float infinity
    to integer`) deep inside `tight_layout`/`savefig`."""
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        finite = a[np.isfinite(a)]
        fallback = float(np.median(finite)) if finite.size else 0.0
        a = np.nan_to_num(a, nan=fallback, posinf=fallback, neginf=fallback)
    return a


def _savefig(fig: plt.Figure, path_cfg: PathConfig, filename: str) -> str:
    out = os.path.join(path_cfg.figures_dir, filename)
    fig.savefig(out, dpi=PLOTCFG.DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scenario_overview_grid(scenarios: List[Scenario], path_cfg: PathConfig
                                  ) -> str:
    """3x2 grid: one stacked PV/Wind/Load panel per stress-test scenario
    (six distinct scenarios, deliberately different from any reference
    figure)."""
    PLOTCFG.apply()
    fig, axes = plt.subplots(3, 2, figsize=PLOTCFG.FIGSIZE_3x2, sharex=False)
    axes = axes.ravel()

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        df = sc.data
        t_h = np.arange(len(df)) * CFG.DT_HOURS
        ax.fill_between(t_h, 0, df["pv_kw"], color=PLOTCFG.COLOR_PV, alpha=0.75,
                         label="PV")
        ax.fill_between(t_h, df["pv_kw"], df["pv_kw"] + df["wind_kw"],
                         color=PLOTCFG.COLOR_WIND, alpha=0.75, label="Wind")
        ax.plot(t_h, df["load_kw"], color=PLOTCFG.COLOR_LOAD, lw=1.6, label="Load")
        ax.set_title(f"({chr(97+i)}) {sc.name}", fontsize=PLOTCFG.TITLE_FONTSIZE)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Power [kW]")
        if i == 0:
            ax.legend(loc="upper right", ncol=1, frameon=False)

    fig.suptitle("Six Novel Stress-Test Scenarios: PV / Wind / Load Profiles",
                 fontsize=PLOTCFG.TITLE_FONTSIZE + 1, y=1.01)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig01_scenario_overview_3x2.png")


def plot_forecast_vs_actual(history_df: pd.DataFrame, pv_pred: np.ndarray,
                              wind_pred: np.ndarray, load_pred: np.ndarray,
                              pv_actual: np.ndarray, wind_actual: np.ndarray,
                              load_actual: np.ndarray, path_cfg: PathConfig
                              ) -> str:
    """2x2 grid: PV forecast, Wind forecast, Load forecast (TAB-Net vs
    ground truth) + a 4th panel summarising per-target RMSE bars."""
    PLOTCFG.apply()
    fig, axes = plt.subplots(2, 2, figsize=PLOTCFG.FIGSIZE_2x2)
    ax = axes.ravel()
    h = np.arange(len(pv_pred)) * CFG.DT_HOURS

    for a, pred, actual, label, color in zip(
        ax[:3], [pv_pred, wind_pred, load_pred],
        [pv_actual, wind_actual, load_actual],
        ["PV", "Wind", "Load"],
        [PLOTCFG.COLOR_PV, PLOTCFG.COLOR_WIND, PLOTCFG.COLOR_LOAD]
    ):
        a.plot(h, actual, color="black", lw=1.8, label="Ground truth")
        a.plot(h, pred, color=color, lw=1.8, ls="--", label="TAB-Net forecast")
        a.set_xlabel("Horizon [h]")
        a.set_ylabel("Power [kW]")
        a.set_title(f"{label} Forecast", fontsize=PLOTCFG.TITLE_FONTSIZE)
        a.legend(frameon=False)

    rmse_vals = [
        float(np.sqrt(mean_squared_error(pv_actual, pv_pred))),
        float(np.sqrt(mean_squared_error(wind_actual, wind_pred))),
        float(np.sqrt(mean_squared_error(load_actual, load_pred))),
    ]
    ax[3].bar(["PV", "Wind", "Load"], rmse_vals,
              color=[PLOTCFG.COLOR_PV, PLOTCFG.COLOR_WIND, PLOTCFG.COLOR_LOAD])
    ax[3].set_ylabel("RMSE [kW]")
    ax[3].set_title("Forecast RMSE by Target", fontsize=PLOTCFG.TITLE_FONTSIZE)

    fig.suptitle("TAB-Net Multi-Horizon Forecast Quality", fontsize=PLOTCFG.TITLE_FONTSIZE + 1)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig02_forecast_quality_2x2.png")


def plot_training_curves(history: Dict[str, List[float]], path_cfg: PathConfig
                           ) -> str:
    """Single-panel training/validation pinball-loss curves for TAB-Net."""
    PLOTCFG.apply()
    fig, ax = plt.subplots(figsize=PLOTCFG.FIGSIZE_SINGLE)
    ax.plot(history["train_loss"], label="Train", color=PLOTCFG.COLOR_PROPOSED)
    ax.plot(history["val_loss"], label="Validation", color=PLOTCFG.COLOR_B2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball (quantile) loss")
    ax.set_title("TAB-Net Training Convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig03_training_curves.png")


def plot_soc_soh_trajectories(results: List[SimulationResult],
                                scenario_names: List[str], path_cfg: PathConfig
                                ) -> str:
    """3x2 grid (one panel per scenario): SOC trajectories of all four
    controllers overlaid, so the reader can directly see how the proposed
    DT-AAMPC manages the battery differently from the baselines."""
    PLOTCFG.apply()
    fig, axes = plt.subplots(3, 2, figsize=PLOTCFG.FIGSIZE_3x2)
    axes = axes.ravel()
    color_map = {
        "DT-AAMPC (Proposed)": PLOTCFG.COLOR_PROPOSED,
        "PID Droop": PLOTCFG.COLOR_B2,
        "Classical MPC (Naive Forecast)": PLOTCFG.COLOR_B3,
    }
    for i, sname in enumerate(scenario_names):
        ax = axes[i]
        for r in results:
            if r.scenario_name != sname:
                continue
            t_h = np.arange(len(r.soc)) * CFG.DT_HOURS
            ax.plot(t_h, _safe_array(r.soc), label=r.controller_name,
                    color=color_map.get(r.controller_name, "gray"), lw=1.5)
        ax.axhline(CFG.BATT_SOC_MIN, color="red", ls=":", lw=1, alpha=0.6)
        ax.axhline(CFG.BATT_SOC_MAX, color="red", ls=":", lw=1, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"({chr(97+i)}) {sname}", fontsize=PLOTCFG.TITLE_FONTSIZE)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("SOC [-]")
        if i == 0:
            ax.legend(loc="lower right", fontsize=10, frameon=False)

    fig.suptitle("Battery SOC Trajectories: Proposed vs Baseline Controllers",
                 fontsize=PLOTCFG.TITLE_FONTSIZE + 1, y=1.01)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig04_soc_trajectories_3x2.png")


def plot_cost_and_performance_bars(metrics_df: pd.DataFrame, path_cfg: PathConfig
                                     ) -> str:
    """2x2 grid of grouped bar charts (mean +/- std across the six
    scenarios) for the four headline KPIs: net cost, self-sufficiency,
    SOH drop, comfort violation."""
    PLOTCFG.apply()
    fig, axes = plt.subplots(2, 2, figsize=PLOTCFG.FIGSIZE_2x2)
    ax = axes.ravel()
    kpis = [
        ("net_cost", "Net Operating Cost [currency]"),
        ("self_sufficiency_pct", "Self-Sufficiency [%]"),
        ("soh_drop_pct", "Battery SOH Drop [%]"),
        ("comfort_violation_kwh", "Unmet Critical Load [kWh]"),
    ]
    controllers = metrics_df["controller"].unique().tolist()
    colors = [PLOTCFG.COLOR_PROPOSED, PLOTCFG.COLOR_B2, PLOTCFG.COLOR_B3]
    x = np.arange(len(controllers))

    for a, (kpi, title) in zip(ax, kpis):
        means = [metrics_df.loc[metrics_df.controller == c, kpi].mean() for c in controllers]
        stds = [metrics_df.loc[metrics_df.controller == c, kpi].std() for c in controllers]
        a.bar(x, means, yerr=stds, capsize=4,
              color=colors[: len(controllers)])
        a.set_xticks(x)
        a.set_xticklabels([c.replace(" (Proposed)", "\n(Proposed)") for c in controllers],
                           rotation=20, ha="right", fontsize=11)
        a.set_title(title, fontsize=PLOTCFG.TITLE_FONTSIZE)
        a.set_ylabel(title.split("[")[-1].rstrip("]") if "[" in title else "")

    fig.suptitle("Aggregate KPI Comparison Across All Six Scenarios",
                 fontsize=PLOTCFG.TITLE_FONTSIZE + 1)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig05_kpi_comparison_2x2.png")


def plot_digital_twin_dashboard(res_proposed: SimulationResult,
                                  scenario_name: str, path_cfg: PathConfig
                                  ) -> str:
    """3x3 diagnostic dashboard for a single representative scenario under
    the proposed DT-AAMPC controller: SOC, SOH, temperature, battery power,
    grid power, forecast confidence, degradation cost, cumulative cost and
    a SOC-vs-price phase plot -- illustrating the digital-twin / EMS
    closed loop end to end."""
    PLOTCFG.apply()
    fig, axes = plt.subplots(3, 3, figsize=PLOTCFG.FIGSIZE_3x3)
    ax = axes.ravel()
    t_h = np.arange(len(res_proposed.soc)) * CFG.DT_HOURS

    panels = [
        (res_proposed.soc, "SOC [-]", PLOTCFG.COLOR_BATT),
        (res_proposed.soh, "SOH [-]", PLOTCFG.COLOR_PROPOSED),
        (res_proposed.temp_k - 273.15, "Battery Temp. [\u00b0C]", PLOTCFG.COLOR_B2),
        (res_proposed.batt_power, "Battery Power [kW]", PLOTCFG.COLOR_B1),
        (res_proposed.grid_power, "Grid Power [kW]", PLOTCFG.COLOR_GRID),
        (res_proposed.forecast_confidence, "Forecast Confidence [-]", PLOTCFG.COLOR_WIND),
        (res_proposed.degradation_cost, "Degradation Cost [/step]", PLOTCFG.COLOR_B3),
        (np.cumsum(res_proposed.import_cost - res_proposed.export_revenue +
                    res_proposed.degradation_cost), "Cumulative Net Cost", PLOTCFG.COLOR_LOAD),
        (res_proposed.pv + res_proposed.wind, "Total Renewable Gen. [kW]", PLOTCFG.COLOR_PV),
    ]
    for a, (series, ylabel, color) in zip(ax, panels):
        a.plot(t_h, _safe_array(series), color=color, lw=1.4)
        a.set_ylabel(ylabel, fontsize=13)
        a.set_xlabel("Time [h]", fontsize=13)

    fig.suptitle(f"DT-AAMPC Digital-Twin / EMS Closed-Loop Dashboard — {scenario_name}",
                 fontsize=PLOTCFG.TITLE_FONTSIZE + 1, y=1.02)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig06_digital_twin_dashboard_3x3.png")


def plot_pareto_cost_vs_degradation(metrics_df: pd.DataFrame, path_cfg: PathConfig
                                      ) -> str:
    """Single scatter panel: net cost vs SOH drop for every
    (scenario, controller) run -- the proposed method should Pareto-dominate
    (lower cost AND lower degradation) the classical baselines."""
    PLOTCFG.apply()
    fig, ax = plt.subplots(figsize=PLOTCFG.FIGSIZE_SINGLE)
    color_map = {
        "DT-AAMPC (Proposed)": PLOTCFG.COLOR_PROPOSED,
        "PID Droop": PLOTCFG.COLOR_B2,
        "Classical MPC (Naive Forecast)": PLOTCFG.COLOR_B3,
    }
    markers = {"DT-AAMPC (Proposed)": "*",
               "PID Droop": "s", "Classical MPC (Naive Forecast)": "^"}
    for controller, sub in metrics_df.groupby("controller"):
        ax.scatter(sub["soh_drop_pct"], sub["net_cost"],
                   color=color_map.get(controller, "gray"),
                   marker=markers.get(controller, "o"), s=120 if "Proposed" in controller else 70,
                   label=controller, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Battery SOH Drop [%]")
    ax.set_ylabel("Net Operating Cost [currency]")
    ax.set_title("Cost vs Degradation Trade-off (6 scenarios x 3 controllers)")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig07_pareto_cost_vs_degradation.png")


def plot_radar_multimetric(metrics_df: pd.DataFrame, path_cfg: PathConfig) -> str:
    """Single radar (polar) chart comparing all three controllers across
    five normalised KPIs (higher = better on every axis after inversion)."""
    PLOTCFG.apply()
    kpis = ["net_cost", "soh_drop_pct", "comfort_violation_kwh",
            "control_effort_kw", "self_sufficiency_pct"]
    labels = ["Cost\n(inv.)", "Degradation\n(inv.)", "Comfort\nViol. (inv.)",
              "Control\nEffort (inv.)", "Self-\nSufficiency"]

    agg = metrics_df.groupby("controller")[kpis].mean()
    norm = agg.copy()
    for k in kpis:
        lo, hi = agg[k].min(), agg[k].max()
        if hi - lo < 1e-9:
            norm[k] = 0.5
        else:
            norm[k] = (agg[k] - lo) / (hi - lo)
    for k in kpis[:-1]:  # invert "lower is better" metrics
        norm[k] = 1 - norm[k]

    angles = np.linspace(0, 2 * np.pi, len(kpis), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    color_map = {
        "DT-AAMPC (Proposed)": PLOTCFG.COLOR_PROPOSED,
        "PID Droop": PLOTCFG.COLOR_B2,
        "Classical MPC (Naive Forecast)": PLOTCFG.COLOR_B3,
    }
    for controller in norm.index:
        vals = norm.loc[controller, kpis].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=controller, color=color_map.get(controller, "gray"), lw=2)
        ax.fill(angles, vals, color=color_map.get(controller, "gray"), alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title("Multi-KPI Radar Comparison (outer = better)", fontsize=PLOTCFG.TITLE_FONTSIZE, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=False)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig08_radar_multimetric.png")


def plot_sensitivity_grid(scenario_names: List[str],
                            metrics_by_horizon: Dict[int, pd.DataFrame],
                            path_cfg: PathConfig) -> str:
    """3x3 parametric-sweep grid: same underlying result (net cost vs
    scenario) shown across nine different MPC base-horizon settings, i.e.
    'same figure, different parameter value' panels as requested for
    sensitivity/robustness analysis in the manuscript."""
    PLOTCFG.apply()
    horizons = sorted(metrics_by_horizon.keys())[:9]
    n = len(horizons)
    rows = cols = int(math.ceil(math.sqrt(max(n, 1))))
    fig, axes = plt.subplots(rows, cols, figsize=PLOTCFG.FIGSIZE_3x3, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, hzn in enumerate(horizons):
        ax = axes[i]
        df_h = metrics_by_horizon[hzn]
        proposed = df_h[df_h.controller == "DT-AAMPC (Proposed)"].set_index("scenario")
        baseline = df_h[df_h.controller == "Classical MPC (Naive Forecast)"].set_index("scenario")
        x = np.arange(len(scenario_names))
        width = 0.35
        ax.bar(x - width / 2, [proposed.loc[s, "net_cost"] if s in proposed.index else np.nan
                               for s in scenario_names],
               width=width, color=PLOTCFG.COLOR_PROPOSED, label="DT-AAMPC")
        ax.bar(x + width / 2, [baseline.loc[s, "net_cost"] if s in baseline.index else np.nan
                                for s in scenario_names],
               width=width, color=PLOTCFG.COLOR_B3, label="Classical MPC")
        ax.set_title(f"Horizon = {hzn} steps", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{j+1}" for j in range(len(scenario_names))], fontsize=10)
        if i == 0:
            ax.legend(fontsize=9, frameon=False)
        if i % cols == 0:
            ax.set_ylabel("Net Cost")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sensitivity of Net Cost to MPC Prediction Horizon",
                 fontsize=PLOTCFG.TITLE_FONTSIZE + 1)
    fig.tight_layout()
    return _savefig(fig, path_cfg, "fig09_horizon_sensitivity_3x3.png")


# ==============================================================================
# 12. FORECASTER TRAINING PIPELINE
# ==============================================================================

def prepare_forecaster_datasets(full_df: pd.DataFrame, cfg: SimConfig
                                  ) -> Tuple[Any, Any, Any, StandardScaler, List[int]]:
    """Builds scaled train/val/test windowed datasets for TAB-Net from the
    (non-scenario, full-history) resampled dataframe."""
    feature_cols = FORECAST_FEATURES
    data = full_df[feature_cols].to_numpy()

    n = len(data)
    n_train = int(n * cfg.TRAIN_FRACTION)
    n_val = int(n * cfg.VAL_FRACTION)

    scaler = StandardScaler()
    scaler.fit(data[:n_train])
    data_scaled = scaler.transform(data)

    target_idx = [feature_cols.index(t) for t in FORECAST_TARGETS]

    if not _TORCH_AVAILABLE:
        return None, None, None, scaler, target_idx

    train_ds = WindowedForecastDataset(data_scaled[:n_train], cfg.LOOKBACK_STEPS,
                                        cfg.HORIZON_STEPS, target_idx)
    val_ds = WindowedForecastDataset(data_scaled[n_train:n_train + n_val],
                                      cfg.LOOKBACK_STEPS, cfg.HORIZON_STEPS, target_idx)
    test_ds = WindowedForecastDataset(data_scaled[n_train + n_val:],
                                       cfg.LOOKBACK_STEPS, cfg.HORIZON_STEPS, target_idx)
    return train_ds, val_ds, test_ds, scaler, target_idx


def run_forecaster_training(full_df: pd.DataFrame, cfg: SimConfig,
                              path_cfg: PathConfig, logger: logging.Logger
                              ) -> Tuple[Optional["TABNet"], StandardScaler, List[int], Dict[str, List[float]]]:
    train_ds, val_ds, test_ds, scaler, target_idx = prepare_forecaster_datasets(full_df, cfg)

    if not _TORCH_AVAILABLE or not cfg.RUN_TRAINING or train_ds is None or len(train_ds) == 0:
        logger.warning("Skipping TAB-Net training (torch unavailable, disabled, "
                        "or insufficient data). Simulation will fall back to a "
                        "persistence forecaster.")
        return None, scaler, target_idx, {"train_loss": [], "val_loss": []}

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = TABNet(n_features=len(FORECAST_FEATURES), n_targets=len(FORECAST_TARGETS),
                    horizon=cfg.HORIZON_STEPS, hidden=cfg.HIDDEN_DIM,
                    n_heads=cfg.ATTENTION_HEADS, quantiles=cfg.QUANTILES)

    ckpt_path = os.path.join(path_cfg.models_dir, "tabnet_best.pt") if cfg.SAVE_MODEL_CHECKPOINTS else None
    history = train_tabnet(model, train_loader, val_loader, cfg, logger, ckpt_path)

    if ckpt_path and os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))

    return model, scaler, target_idx, history


def evaluate_forecaster_on_window(model: Optional["TABNet"], scaler: StandardScaler,
                                    target_idx: List[int], full_df: pd.DataFrame,
                                    cfg: SimConfig, start: int
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray]:
    """Runs one illustrative forecast at index `start` and returns
    (pred_pv, pred_wind, pred_load, actual_pv, actual_wind, actual_load)
    for the forecast-quality figure."""
    feature_cols = FORECAST_FEATURES
    lookback, horizon = cfg.LOOKBACK_STEPS, cfg.HORIZON_STEPS
    hist = full_df[feature_cols].iloc[max(0, start - lookback): start].to_numpy()
    if len(hist) < lookback:
        pad = np.repeat(hist[:1], lookback - len(hist), axis=0)
        hist = np.vstack([pad, hist])

    provider = ForecastProvider(cfg, scaler, target_idx, model)
    pv_pred, wind_pred, load_pred, _ = provider.forecast(hist)

    actual_window = full_df[FORECAST_TARGETS].iloc[start: start + horizon].to_numpy()
    if len(actual_window) < horizon:
        pad = np.repeat(actual_window[-1:], horizon - len(actual_window), axis=0) \
            if len(actual_window) > 0 else np.zeros((horizon, 3))
        actual_window = np.vstack([actual_window, pad]) if len(actual_window) else pad

    return (pv_pred, wind_pred, load_pred,
            actual_window[:, 0], actual_window[:, 1], actual_window[:, 2])


# ==============================================================================
# 13. FULL EXPERIMENT ORCHESTRATION
# ==============================================================================

def build_controllers(cfg: SimConfig) -> List[ControllerBase]:
    return [
        DigitalTwinAttentionAdaptiveMPC(cfg),
        PIDDroopController(cfg),
        ClassicalNaiveForecastMPC(cfg),
    ]


def run_full_experiment(cfg: SimConfig, path_cfg: PathConfig, plot_cfg: PlotConfig,
                          logger: logging.Logger) -> Dict[str, Any]:
    path_cfg.make_all()
    set_global_seed(cfg.RANDOM_SEED)

    logger.info("=== STAGE 1/6: Data loading ===")
    loader = SmartHomeDataLoader(path_cfg, cfg, logger)
    raw_df = loader.load()

    logger.info("=== STAGE 2/6: Renewable synthesis ===")
    synth = RenewableSynthesizer(cfg)
    full_df = synth.build(raw_df, loader)

    logger.info("=== STAGE 3/6: Scenario generation (6 novel stress tests) ===")
    scen_gen = ScenarioGenerator(cfg, rng_seed=cfg.RANDOM_SEED)
    scenarios = scen_gen.build_scenarios(full_df, window_days=6)
    for sc in scenarios:
        logger.info("  Scenario '%s': %s -> %s (%d steps)",
                     sc.name, sc.start, sc.end, len(sc.data))

    logger.info("=== STAGE 4/6: TAB-Net deep-learning forecaster training ===")
    model, scaler, target_idx, history = run_forecaster_training(full_df, cfg, path_cfg, logger)

    logger.info("=== STAGE 5/6: Closed-loop simulation (6 scenarios x 3 controllers x "
                "%d Monte-Carlo seeds) ===", cfg.N_MONTE_CARLO_SEEDS)
    provider = ForecastProvider(cfg, scaler, target_idx, model)
    simulator = SmartGridSimulator(cfg, rng_seed=cfg.RANDOM_SEED)
    controllers = build_controllers(cfg)

    all_results: List[SimulationResult] = []
    if cfg.RUN_SIMULATION:
        for sc in scenarios:
            for controller in controllers:
                use_twin = isinstance(controller, DigitalTwinAttentionAdaptiveMPC)
                # Every controller now observes noisy sensor readings (see
                # `SmartGridSimulator.run`), so all four -- not just the
                # twin-based one -- are genuinely stochastic and must be
                # re-simulated per Monte-Carlo seed.
                for mc_seed in range(cfg.N_MONTE_CARLO_SEEDS):
                    logger.info("  Running scenario='%s' controller='%s' (twin=%s) seed=%d",
                                 sc.name, controller.name, use_twin, mc_seed)
                    res = simulator.run(sc, controller, provider, FORECAST_FEATURES,
                                        use_digital_twin=use_twin, seed=mc_seed)
                    all_results.append(res)

    metrics_df = build_metrics_table(all_results, cfg) if all_results else pd.DataFrame()
    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(path_cfg.tables_dir, "metrics_all_runs.csv"), index=False)

    sig_df = pd.DataFrame()
    if not metrics_df.empty:
        proposed_name = DigitalTwinAttentionAdaptiveMPC.name
        baseline_names = [PIDDroopController.name, ClassicalNaiveForecastMPC.name]
        headline_metrics = ["net_cost", "self_sufficiency_pct", "soh_drop_pct",
                             "comfort_violation_kwh", "control_effort_kw"]
        sig_df = statistical_significance_tests(metrics_df, proposed_name,
                                                  baseline_names, headline_metrics)
        sig_df.to_csv(os.path.join(path_cfg.tables_dir, "significance_tests.csv"), index=False)

    logger.info("=== STAGE 6/6: Figure generation ===")
    figure_paths = []
    if cfg.RUN_PLOTS:
        figure_paths.append(plot_scenario_overview_grid(scenarios, path_cfg))

        eval_start = min(cfg.LOOKBACK_STEPS + 50, max(cfg.LOOKBACK_STEPS + 1, len(full_df) - cfg.HORIZON_STEPS - 1))
        pv_pred, wind_pred, load_pred, pv_act, wind_act, load_act = \
            evaluate_forecaster_on_window(model, scaler, target_idx, full_df, cfg, eval_start)
        figure_paths.append(plot_forecast_vs_actual(full_df, pv_pred, wind_pred, load_pred,
                                                      pv_act, wind_act, load_act, path_cfg))

        if history.get("train_loss"):
            figure_paths.append(plot_training_curves(history, path_cfg))

        if all_results:
            # Trajectory-style plots (SOC traces, twin dashboard) need ONE
            # representative run per (scenario, controller), not an
            # average across Monte-Carlo replicates -- use seed 0 for
            # those. The KPI bar/pareto/radar/significance figures below
            # correctly use every replicate via `metrics_df`.
            seed0_results = [r for r in all_results if r.seed == 0]
            scenario_names = [sc.name for sc in scenarios]
            figure_paths.append(plot_soc_soh_trajectories(seed0_results, scenario_names, path_cfg))
            figure_paths.append(plot_cost_and_performance_bars(metrics_df, path_cfg))

            proposed_first = next((r for r in seed0_results
                                    if r.controller_name == DigitalTwinAttentionAdaptiveMPC.name),
                                   None)
            if proposed_first is not None:
                figure_paths.append(plot_digital_twin_dashboard(
                    proposed_first, proposed_first.scenario_name, path_cfg))

            figure_paths.append(plot_pareto_cost_vs_degradation(metrics_df, path_cfg))
            figure_paths.append(plot_radar_multimetric(metrics_df, path_cfg))

    logger.info("Experiment complete. %d figures written to %s",
                len(figure_paths), path_cfg.figures_dir)

    return {
        "scenarios": scenarios, "model": model, "history": history,
        "results": all_results, "metrics_df": metrics_df,
        "significance_df": sig_df, "figure_paths": figure_paths,
    }


def run_horizon_sensitivity_sweep(cfg: SimConfig, path_cfg: PathConfig,
                                    scenarios: List[Scenario],
                                    provider: ForecastProvider,
                                    logger: logging.Logger,
                                    horizons: Tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20, 24)
                                    ) -> str:
    """Optional deeper robustness study: re-runs DT-AAMPC and the classical
    MPC baseline across nine different `MPC_BASE_HORIZON` settings to
    produce the 3x3 sensitivity figure (same comparison, different
    parameter value per panel, as requested)."""
    simulator = SmartGridSimulator(cfg, rng_seed=cfg.RANDOM_SEED)
    metrics_by_horizon: Dict[int, pd.DataFrame] = {}

    for hzn in horizons:
        local_cfg = dataclasses.replace(cfg, MPC_BASE_HORIZON=hzn,
                                          MPC_MAX_HORIZON=max(hzn, cfg.MPC_MAX_HORIZON))
        proposed = DigitalTwinAttentionAdaptiveMPC(local_cfg)
        classical = ClassicalNaiveForecastMPC(local_cfg)
        run_results = []
        for sc in scenarios:
            run_results.append(simulator.run(sc, proposed, provider, FORECAST_FEATURES, True))
            run_results.append(simulator.run(sc, classical, provider, FORECAST_FEATURES, False))
        metrics_by_horizon[hzn] = build_metrics_table(run_results, local_cfg)
        logger.info("  Horizon sweep: base_horizon=%d done", hzn)

    scenario_names = [sc.name for sc in scenarios]
    return plot_sensitivity_grid(scenario_names, metrics_by_horizon, path_cfg)


# ==============================================================================
# 14. REPORT GENERATION
# ==============================================================================

def write_markdown_report(exp_out: Dict[str, Any], path_cfg: PathConfig,
                            sensitivity_fig: Optional[str] = None) -> str:
    """Writes a self-contained Markdown summary (tables + figure references)
    that can be pasted almost directly into the Results section of a Q1
    manuscript draft."""
    metrics_df: pd.DataFrame = exp_out["metrics_df"]
    sig_df: pd.DataFrame = exp_out["significance_df"]
    fig_paths: List[str] = exp_out["figure_paths"]

    lines = []
    lines.append("# DT-AAMPC Experimental Results Report\n")
    lines.append(f"_Generated: {pd.Timestamp.now()}_\n")
    lines.append("## 1. Method Summary\n")
    n_seeds = metrics_df["seed"].nunique() if ("seed" in metrics_df.columns and not metrics_df.empty) else 1
    lines.append(
        f"Proposed **DT-AAMPC** (Digital-Twin Attention-Augmented Adaptive "
        f"Model Predictive Control) is compared against two classical "
        "baselines -- PID Droop and a deterministic Naive-Forecast MPC -- "
        "across six novel stress-test scenarios: "
        "Heatwave AC Surge, Winter Storm Load Peak, Cloud-Cover Cascade, "
        "EV Fleet Overnight Charging, Islanded Microgrid Fault Ride-Through, "
        f"and Solar Oversupply Curtailment, each replicated over "
        f"{n_seeds} independent Monte-Carlo measurement-noise seeds "
        f"(giving {6 * n_seeds} evaluation blocks per controller for the "
        f"significance tests below).\n"
    )

    if not metrics_df.empty:
        lines.append("## 2. Aggregate KPI Table (mean across all scenario x seed replicates)\n")
        numeric_cols = [c for c in metrics_df.columns if c not in ("scenario", "controller", "seed")]
        agg = metrics_df.groupby("controller")[numeric_cols].mean(numeric_only=True)
        lines.append(agg.to_markdown())
        lines.append("\n")

    if not sig_df.empty:
        lines.append("## 3. Statistical Significance (Wilcoxon, proposed vs baselines, "
                      "paired by scenario x seed)\n")
        lines.append(sig_df.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 4. Figures\n")
    for p in fig_paths:
        lines.append(f"- `{os.path.basename(p)}`")
    if sensitivity_fig:
        lines.append(f"- `{os.path.basename(sensitivity_fig)}` (horizon sensitivity sweep)")

    report_path = os.path.join(path_cfg.results_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ==============================================================================
# 15. MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    global LOGGER
    PATHS.make_all()
    LOGGER = build_logger(PATHS.logs_dir)

    LOGGER.info("========================================================")
    LOGGER.info(" DT-AAMPC :  Digital-Twin Attention-Augmented Adaptive  ")
    LOGGER.info("             Model Predictive Control for Residential  ")
    LOGGER.info("             Battery Energy Management                ")
    LOGGER.info("========================================================")
    LOGGER.info("Base directory : %s", PATHS.BASE_DIR)
    LOGGER.info("Results dir    : %s", PATHS.results_dir)
    LOGGER.info("Torch available: %s | Device: %s", _TORCH_AVAILABLE, CFG.DEVICE)

    t_start = time.time()
    exp_out = run_full_experiment(CFG, PATHS, PLOTCFG, LOGGER)

    sensitivity_fig = None
    scaler = None
    try:
        # Optional deeper robustness study (comment out to save runtime).
        loader = SmartHomeDataLoader(PATHS, CFG, LOGGER)
        raw_df = loader.load()
        full_df = RenewableSynthesizer(CFG).build(raw_df, loader)
        _, scaler, target_idx, _ = prepare_forecaster_datasets(full_df, CFG)
        provider = ForecastProvider(CFG, scaler, target_idx, exp_out["model"])
        sensitivity_fig = run_horizon_sensitivity_sweep(
            CFG, PATHS, exp_out["scenarios"], provider, LOGGER)
    except Exception as exc:  # pragma: no cover - sweep is optional/best-effort
        LOGGER.warning("Horizon sensitivity sweep skipped due to: %s", exc)

    report_path = write_markdown_report(exp_out, PATHS, sensitivity_fig)

    elapsed = time.time() - t_start
    LOGGER.info("Report written to: %s", report_path)
    LOGGER.info("Total wall-clock time: %.1f s (%.2f min)", elapsed, elapsed / 60)
    LOGGER.info("Done. See %s for all figures/tables.", PATHS.results_dir)


if __name__ == "__main__":
    main()
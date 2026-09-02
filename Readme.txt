# DT-AAMPC: Digital-Twin Attention-Augmented Adaptive Model Predictive Control

Digital-Twin Attention-Augmented Adaptive Model Predictive Control (DT-AAMPC) for degradation-aware residential battery energy management. This repository accompanies the manuscript *"Digital-Twin Attention-Augmented Adaptive Model Predictive Control for Degradation-Aware Residential Battery Energy Management: A Stress-Tested, Statistically Validated Framework"*, submitted to a Q1 energy-systems journal.

DT-AAMPC unites three ingredients inside a single closed loop:

- **TAB-Net** (Temporal Attention-BiLSTM Network): a causal-convolution and multi-head-self-attention forecaster that produces multi-horizon, multi-quantile forecasts of photovoltaic, wind and load power, together with an online confidence signal derived from its attention entropy.
- **Battery digital twin**: a lumped electro-thermal-degradation physics model synchronised with noisy sensor readings through an extended Kalman filter, producing a bias-corrected state estimate and its covariance.
- **Risk-aware adaptive economic MPC**: a receding-horizon controller whose prediction horizon, forecast-uncertainty penalty and state-of-charge back-off margin are all adapted online from the forecaster confidence and the twin covariance.

The framework is evaluated against a classical PID/droop controller and a deterministic naive-forecast MPC across six deliberately novel stress-test scenarios, each replicated over five independent Monte-Carlo measurement-noise seeds, with paired Wilcoxon signed-rank significance testing (Holm-Bonferroni corrected) across all thirty evaluation blocks per controller.

## Headline Results

Averaged over all thirty (scenario, seed) evaluation blocks:

| Metric | DT-AAMPC (proposed) | PID / Droop | Classical MPC (naive forecast) |
|---|---|---|---|
| Net operating cost [currency] | **251.76** | 360.29 | 469.48 |
| Self-sufficiency [%] | **46.63** | 39.54 | 44.04 |
| Battery SOH drop [%] | **3.04** | 5.55 | 8.41 |
| Control effort [kW] | **28.46** | 216.54 | 452.43 |

All four differences above are statistically significant against both baselines (paired Wilcoxon, Holm-Bonferroni corrected, p < 0.001). The comfort-violation gain did not reach significance. A single scenario, an extreme heatwave-driven air-conditioning surge, was identified in which the simpler PID/droop baseline outperformed DT-AAMPC on both cost and degradation; this exception is discussed candidly in the manuscript rather than omitted.

## Repository Structure

```
.
├── README.md
├── src/
│   └── DT_asac_ampc_battery_ems.py     # Full simulation framework (see below)
├── data/
│   ├── metrics_all_runs.csv            # Per-(scenario, controller, seed) KPI table
│   └── significance_tests.csv          # Paired Wilcoxon test results
├── paper/
│   ├── main.tex                        # Manuscript source (MDPI journal template)
│   └── figures/
│       ├── DT-AAMPC_diagrams.pdf       # Methodology flowchart (Figure 1)
│       ├── fig01_scenario_overview_3x2.png
│       ├── fig02_forecast_skill_vs_horizon.png
│       ├── fig04_soc_trajectories_3x2.png
│       ├── fig05_kpi_boxplot.png
│       ├── fig06_digital_twin_dashboard_2x2.png
│       └── fig07_pareto_confidence_ellipse.png
└── results/                             # Created at runtime (see Output layout below)
```

### `src/DT_asac_ampc_battery_ems.py`

A single, self-contained script organised into fifteen numbered sections:

1. Global configuration (`PathConfig`, `SimConfig`, `PlotConfig`)
2. Data loading and cleaning
3. Renewable synthesis (PV, wind, net load)
4. Stress-test scenario generator (six scenarios)
5. Battery digital twin (physics model + extended Kalman filter)
6. TAB-Net deep-learning forecaster
7. Baseline controllers (PID/droop)
8. Proposed and baseline MPC controllers
9. Closed-loop simulation engine
10. Metrics computation and statistical significance testing
11. Publication-grade plotting (all figures)
12. Forecaster training pipeline
13. Full experiment orchestration
14. Markdown report generation
15. Main entry point

## Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
torch        # optional: TAB-Net training falls back to a persistence
             # forecaster automatically if PyTorch is not installed
```

Install with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch
```

## Dataset

The simulation is built on the *Augmented Smart Home with Weather Information* dataset (Rodríguez Vega & Syne, 2025), itself an augmentation of the one-minute-resolution smart-meter dataset released on Kaggle by the user Taranvee, extended with eight additional Internet-of-Things appliance channels (car charger, water heater, air conditioning, home theatre, outdoor lighting, microwave, laundry, pool pump).

- Augmented dataset (Mendeley Data): https://doi.org/10.17632/pxnb7gh646.1
- Original dataset (Kaggle): https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information

Place the downloaded CSV at the path configured in `PathConfig.BASE_DIR` / `PathConfig.DATA_FILENAME` inside `src/DT_asac_ampc_battery_ems.py`. If the file is not found, the script automatically falls back to a small synthetic placeholder dataset so that the pipeline remains runnable end to end for smoke-testing purposes; real results require the actual dataset.

## Quick Start

```bash
git clone https://github.com/<your-org>/DT-AAMPC.git
cd DT-AAMPC
pip install numpy pandas matplotlib scipy scikit-learn torch
python src/DT_asac_ampc_battery_ems.py
```

Before running, edit the `PathConfig.BASE_DIR` field near the top of the script to point at your local working directory, and place the dataset CSV there. The `SimConfig` toggles (`RUN_TRAINING`, `RUN_SIMULATION`, `RUN_PLOTS`) control which stages execute; all three default to `True`.

### Output layout

Running the script populates:

```
<BASE_DIR>/results/
├── figures/     # All PNG figures at 600 DPI
├── tables/      # metrics_all_runs.csv, significance_tests.csv
├── models/      # Trained TAB-Net checkpoint (tabnet_best.pt)
├── logs/        # Console + file log of the run
└── REPORT.md    # Consolidated, manuscript-ready results summary
```

This run is heavy (deep-learning training plus a closed-loop simulation across six scenarios, three controllers and five Monte-Carlo seeds each); expect a non-trivial wall-clock time on a CPU-only machine, considerably faster with a CUDA-capable GPU.

## Reproducing the Manuscript

The `data/` and `paper/figures/` files in this repository are the exact tables and figures cited in `paper/main.tex`. To recompile the manuscript locally:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # run twice to resolve cross-references
```


## Citation


```bibtex
@article{dtaampc2026,
  title   = {Digital-Twin Attention-Augmented Adaptive Model Predictive Control
             for Degradation-Aware Residential Battery Energy Management:
             A Stress-Tested, Statistically Validated Framework},
  author  = {Lastname, Firstname and Lastname, Firstname and Lastname, Firstname},
  journal = {Journal Not Specified},
  year    = {2026},
  doi     = {10.3390/1010000}
}
```

Please also cite the underlying dataset:

```bibtex
@misc{rodriguezvega2025dataset,
  title  = {Augmented Smart Home with Weather Information},
  author = {Rodr{\'i}guez Vega, Marcos and Syne, Lamine},
  year   = {2025},
  note   = {Mendeley Data, V1},
  doi    = {10.17632/pxnb7gh646.1}
}
```

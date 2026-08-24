# Time-Adjusted Cluster Load Allocation with Error Correction in Sparsely Metered Distribution Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhizterpaul/dsse/blob/main/report.ipynb)

## Research Summary & Scope

The primary research focus is distribution network state estimation using Time-Adjusted Cluster Load Allocation (CLA) and transformer dynamic signal processing in sparsely metered distribution networks:

*   **Known Upstream Plant & LV Networks**: OpenDSS model incorporating utility swing bus, substation step-down transformer, and 3 feeders with known radial LV topologies and transformer edge interfaces.

*   **Time-Adjusted Cluster Load Allocation**: Unmetered customer energy allocation ($E_U = E_F - E_M - E_L$) using class-based profile weights, time-adjustment integrals $\alpha_i(t)$, and technical loss accounting ($E_L = E_{\mathrm{transformer\_loss}} + E_{\mathrm{line\_loss}}$).

*   **ATP-EMTP Transient Coupling**: Coupling of sub-cycle transients (such as equipment switching and explicit line faults) at distribution transformer secondaries to extract high-frequency spectral and waveform residual signatures for transient error correction.

*   **Boundary & Consumer Measurements**: 36% consumer smart meter coverage combined with feeder boundary and transformer secondary monitoring.

*   **Feature Tabulation & Rendering**: Export of all steady-state and dynamic parameters directly to CSV datasets, rendering error metrics and error reduction factor calculations in `report.ipynb`.

## Repository Structure

- `src/power_plant/`: OpenDSS fixed plant model, measurement extraction, and ATP-EMTP dynamic transient extraction.
- `src/lv_networks/`: Known radial LV network topologies, consumer equipment models, and meter selection routines.
- `src/estimator/`: Cluster Load Allocation (`cla_estimator.py`) and Time-Adjusted CLA (`time_adjusted_cla_estimator.py`) state estimation engines.
- `src/simulation/`: Scenario definitions, co-simulation runner, and dataset generation routines.
- `src/report.ipynb`: Orchestration notebook for the complete research pipeline.

## Getting Started

### Installation
```bash
# Install core dependencies
pip install -r requirements.txt

```

### Execution
Run the complete research pipeline via the Jupyter notebook:
```bash
jupyter notebook src/report.ipynb
```

## References

- **Keywords**: Distribution System State Estimation (DSSE), Cluster Load Allocation, Time-Adjusted CLA, Transient Analysis, Statistical Analysis.
- **Modeling Framework**: OpenDSS (Distribution Power Flow), ATP-EMTP (transformer transients).

This repository contains the data and functions to reproduce all figures in Richter et al. (2024). Complete data and plots are stored in this repository (unsing `git lfs`) but can be recomputed (overwriting the repository files).

## Repository structure
- `png` plot outputs
- `data` simulation data
    - `3d_pfaller22` 3d simulation results published in pfaller22
    - `calibrated_richter24_from_{0,3}d` 0d simulations with elements calibrated from {0,3}d data
        - `input` calibration input: 0d `.json` files with all elements set to zero and solution vecotrs $y$, $\dot{y}$
        - `output` calibration output: 0d `.json` files with calibrated elements
    - `centerlines_pfaller22` centerlines for all 72 geometries in pfaller22
    - `geometric_pfaller22` geometric 0d models for all 72 geometries in pfaller22

## Running simulations
- `calibration_run_from_0d.py` calibrate 0d elements to simulation outputs
- `error_run_0d.py` run 0d simulations and save output

## Generating plots
All plots are generated in the `png` folder. The following scripts can be executed to generate plots:
- `calibration_plot.py` correlation of 0d elements: geometric vs. calibrated from 0d or 3d data
- `error_change_plot.py` change in error from geometric to calibrated 0d models (vs. 3d)
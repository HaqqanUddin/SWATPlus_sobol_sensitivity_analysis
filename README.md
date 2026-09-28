# SWAT+ Sobol Sensitivity Analysis

A parallel global sensitivity analysis framework for **SWAT+** models using the **Sobol method** (via SALib).

This script systematically evaluates the influence of key hydrological parameters on simulated streamflow by running thousands of SWAT+ simulations in parallel, extracting results, computing sensitivity indices, and generating a comprehensive visual dashboard.

---

## Overview

Sensitivity analysis is a critical step before model calibration. It helps identify which parameters most strongly control the model output, allowing modelers to focus calibration efforts on the most influential parameters and reduce computational cost.

This tool implements a **variance-based global sensitivity analysis** using the Sobol method, which can separate:

- **First-order effects (S1)** → individual contribution of each parameter  
- **Total-order effects (ST)** → total contribution including interactions with other parameters  

---

## Features

- Parallel execution using `ProcessPoolExecutor` (configurable number of workers)
- Automatic generation of Sobol sample sets with SALib
- Dynamic creation and cleanup of temporary SWAT+ run directories
- Extraction of average monthly flow (`flo_out`) from a specified channel
- Intermediate result saving (every N runs) for crash recovery
- Computation of Sobol sensitivity indices (S1 and ST)
- Automatic generation of a multi-panel dashboard:
  - Run summary statistics
  - Flow distribution histogram
  - First-order (S1) bar chart
  - Total-order (ST) bar chart
  - Scatter plot of all parameters vs. simulated flow

---

## Parameters Included

| Parameter | Full Name                        | Units / Range     | Change Type | Description |
|-----------|----------------------------------|-------------------|-------------|-------------|
| `cn2`     | SCS Curve Number                 | 35 – 95           | absval      | Controls surface runoff generation |
| `esco`    | Soil Evaporation Compensation    | 0.01 – 1.0        | absval      | Affects soil evaporation depth |
| `epco`    | Plant Uptake Compensation        | 0.01 – 1.0        | absval      | Controls plant water uptake from lower layers |
| `canmx`   | Maximum Canopy Storage           | 0 – 100 mm        | absval      | Interception capacity of vegetation |
| `awc`     | Available Water Capacity         | 0.05 – 0.40       | pctchg      | Soil water holding capacity |
| `k`       | Saturated Hydraulic Conductivity | 1 – 150 mm/hr     | pctchg      | Controls percolation rate |
| `perco`   | Percolation Coefficient          | 0.0 – 1.0         | absval      | Fraction of water that percolates |
| `surlag`  | Surface Runoff Lag               | 0.05 – 24 days    | absval      | Controls timing of surface runoff |

---

## Requirements

- Python 3.8 or higher
- SWAT+ executable (tested with rev61.0.1)
- Required Python packages:

```bash
pip install numpy pandas matplotlib SALib"# SWATPlus_sobol_sensitivity_analysis" 

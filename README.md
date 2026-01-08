# Cellular Decision Making

> **Tripathi, Dunkel & Skinner (2026)**  
> Code for "Collective is different: Information exchange and speed-accuracy trade-offs in self-organized patterning."

---

## Overview

This repository contains code for analyzing collective cellular patterning through a decentralized minimal model lateral inhibition, as used in [Tripathi, Dunkel, Skinner (2026)](https://doi.org/10.1103/lfnt-8qbt). The codebase includes:

- **Interior point optimization** for finding optimal strategies in multi-cell systems as posited in the paper
- **Sampling-based optimization** for finding approximate optimal strategies using score function and Monte-Carlo approximations
- **Information-theoretic calculations** for quantifying information theorey metrics for a continuous time markov chain
- **Experimental data analysis** for processing and calculating mutual information for microscopy data in ([Phan et al (2024)](https://doi.org/10.1242/dev.203165))

The relevant data for the project can be found ([10.5281/zenodo.18189942](10.5281/zenodo.18189942)). This contains the processed experimental data for mutual information calculation. It also contains results from the minimal model exploration, particularly the optimied strategies as almost all of the calculations were performed the MIT Engaging cluster -- the saved data means you don't need to repeat this computation.

---

## Prerequisites

- **Julia** (tested on v1.11.2)
- **Python** (tested on )
- **NPEET** for mutual information calculation (installled locally, see [https://github.com/gregversteeg/NPEET](https://github.com/gregversteeg/NPEET))
- **JuMP** and **Ipopt** for finding optimal strategies for patterning using interior point method
- **DifferentialEquations.jl** for solving ODE/SDE
- **CelularDecisions.jl** (installed locally; provides core data structures for the three cell system computations; can be found at [https://github.com/Dom-Skinner/CellularDecisions](https://github.com/Dom-Skinner/CellularDecisions))
- **Main Python dependencies** (see `requirements.txt`):  
  - numpy  
  - scipy  
  - pandas  
  - sympy  
  - matplotlib

---

## Repository Structure

```
├── example_calculation_scripts/
│   ├── three_cell_scripts/           # Three-cell system example calculation scripts
│   │   ├── info_calc_scripts/             # Scripts to calculate dynamic information metrics for three cell system, and create relevant plots 
│   │   └── opti_scripts/                  # Scripts to calculate optimal strategies (using interior point method and sampling-based method), and create relevant plots
│   │   └── SI_fig_final_state_mi/         # Plotting scripts for instanataneous information mesures for the three cells system
│   └── seven_cell_scripts/           # Seven-cell system example calculation scripts, primarily containing computational scripts to search for the optimal strategies using sampling based approximation
├── experimental_data_analysis/ # Mutual information calculation for experimental data
├── information_metrics/        # Functions for dynamic information theory calculations
├── mult_cell/                  # Function to define the multi-cell CTMC framework
├── sampling_based_optimization/# Function for sampling-based policy optimization method
└── utils/                      # Utility functions
```
---

## Computational scripts description
The following table describes the purpose of the different example calculation scripts.

| Script | Description |
| ------ | ----------- |
| `three_cell_scripts/opti_scripts/three_cell_optimisation.jl` | To find optimal solution (with a single initialiation) for the three cell system, using interior point method. |
| `three_cell_scripts/opti_scripts/three_cell_optimisation_multiple_warm_starts.jl` | To find optimal solution with warm starts for the three cell syste,, using interior point method. |
| `three_cell_scripts/opti_scripts/three_cell_optimisation_local.jl` | To search for the ``greedy'' solution for the three cell system, using interior point method. |
| `three_cell_scripts/opti_scripts/three_cell_run_sampling_sgd.jl` | To find optimal solution for the three cell system using the sampling-based approximate method. |
| `three_cell_scripts/opti_scripts/three_cell_sampling_phase_space.jl` | To sample the relevant manifold, finding and saving the sub-optimal solutions for the three-cell system. |
| `three_cell_scripts/opti_scripts/three_cell_getting_conditioned_rate_matrix.jl` | To create and save a new three cell system after Doob transformation so that the markov process always sucessfuly patterns and cell three is expressed. |
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple.jl` | To compute dynamic information metrics for information transfer between cell 1 and cell 3 in a three cell system for a fixed error rate. |
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple_error_sweep.jl` | To compute dynamic information metrics for information transfer between cell 1 and cell 3 in a three cell system for different error rates. |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one.jl` | To compute dynamic information metrics for information transfer between cell 3 and the tuple (cell 1, cell 2) in a three cell system for a fixed error rate. |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one_error_sweep.jl` | To compute dynamic information metrics for information transfer between cell 3 and the tuple (cell 1, cell 2) in a three cell system for different error rates. |
| `seven_cell_scripts/seven_cell_run_sampling_sgd.jl` | To find optimal solution for the seven cell system using the sampling-based approximate method. |

---
## Plotting Scripts
The following table describes which script is used to create the different plots for the various relevant figures in the paper. A lot of these scripts are also doing computations as well, and this is described in the relevant script.

| Script | Figures |
| ------ | ---------------- |
| `three_cell_scripts/opti_scripts/three_cell_arrhenius_landscape_plots.jl` | Fig. 2 |
| `three_cell_scripts/opti_scripts/three_cell_pareto_front_with_samples_fig_2.jl` | Fig. 2, S1 |
| `three_cell_scripts/opti_scripts/threecell_plotting_global.jl` | Fig. 2, 6|
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple.jl` | Fig. 3 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple_error_sweep.jl` | Fig. 3 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one.jl` | Fig. 3 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one_error_sweep.jl` | Fig. 3 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple_conditioned.jl` | Fig. 4 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one_conditioned.jl` | Fig. 4 |
| `three_cell_scripts/opti_scripts/threecell_all_instantaneous_mi_plot_maker.jl` | Fig. 5 |
| `three_cell_scripts/opti_scripts/threecell_plotting_global.jl` | Fig. 2, 6|
| `three_cell_scripts/opti_scripts/threecell_plotting_local_strategy.jl` | Fig. 6, S5 |
| `three_cell_scripts/opti_scripts/threecell_plotting_local_strategy_thrice.jl` | Fig. 6|
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple_high_error.jl` | Fig. 7 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_simple_low_error.jl` | Fig. 7 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one_high_error.jl` | Fig. 7 |
| `three_cell_scripts/info_calc_scripts/three_cell_info_two_v_one_low_error.jl` | Fig. 7 |
| `experimental_data_analysis/mutual_information_calculation.ipynb` | Fig. 8, S9, S10 |
| `seven_cell_scripts/seven_cell_pareto_front_si_fig.jl` | Fig. S1 |
| `seven_cell_scripts/seven_cell_arrhenius_landscape_plots.jl` | Fig. S2 |
| `seven_cell_scripts/seven_cell_simulations_fig_si.jl` | Fig. S2 |
| `three_cell_scripts/SI_fig_final_state_mi/final_state_mi_single.py` | Fig. S3 |
| `three_cell_scripts/SI_fig_final_state_mi/final_state_mi_two_v_one.py` | Fig. S4 |

---

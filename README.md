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

---

## Prerequisites

### Julia (tested on 1.11.2)

- **Julia** (tested on v1.11.2)
- **Python** (tested on )
- **NPEET** for mutual information calculation (installled locally, see [https://github.com/gregversteeg/NPEET](https://github.com/gregversteeg/NPEET))
- **JuMP** and **Ipopt** for finding optimal strategies for patterning using interior point method
- **DifferentialEquations** for solving ODE/SDE
- **CelularDecisions.jl** (installed locally; provides core data structures for the three cell system computations)
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
│   ├── three_cell_scripts/     # Three-cell system example calculation scripts
│   └── seven_cell_scripts/     # Seven-cell system example calculation scripts
├── experimental_data_analysis/ # Mutual information calculation for experimental data
├── information_metrics/        # Functions for dynamic information theory calculations
├── mult_cell/                  # Function to define the multi-cell CTMC framework
├── sampling_based_optimization/# Function for sampling-based policy optimization method
└── utils/                      # Utility functions
```
---

## Plotting Scripts

| Script | Figures / Tables |
| ------ | ---------------- |


---

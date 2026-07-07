# Mixed Finite Element for Plastic Localization

This repository contains the Firedrake implementation for the benchmark and mesh convergence analysis of the mixed finite element formulation (Argyris $P_5$ element) for plastic localization problems.

## 🔬 Experimental Design & Tasks

### Task 1: Benchmark Computations (Validation)
**Objective:** To rigorously validate this new Firedrake implementation against the original FreeFEM results published in our paper, establishing a solid baseline.

**Methodology:** - We strictly replicate the published macroscopic loading conditions: $g=1.4$ and $g=2.0$.
- The simulation runs for exactly 20 non-linear increments to match the reference data.
- **Outputs:** The script automatically generates isolated output directories containing Force-Displacement curves, ParaView dynamics (`.pvd`), and projects the macroscopic stress tensor for visual comparison.

---

### Task 2: Mesh Convergence Analysis
**Objective:** To mathematically demonstrate the superior convergence properties of the high-order Argyris ($P_5$) element and prove its computational efficiency over brute-force h-refinement.

**Methodology:**
- The macroscopic load is held constant (e.g., $g=1.4$).
- The characteristic mesh size (`clscale`) is sequentially refined (e.g., $0.1 \rightarrow 0.08 \rightarrow 0.05$).
- **Outputs:** At the end of each run, the script automatically integrates and extracts the **Total Global DOFs** and the **Final Elastoplastic Free Energy ($W$)**. 
- **Significance:** Tracking the total free energy against global DOFs proves that the $C^\infty$ internal continuity of the Argyris element can smoothly resolve the plastic shear band on coarse meshes, completely avoiding the numerical locking and severe artifacts seen in lower-order elements (like HCT).

## ⚙️ Prerequisites
To run this code, you need to have a working installation of [Firedrake](https://www.firedrakeproject.org/) and Gmsh.

## 🚀 How to Run the Code

The code is designed to be a "plug-and-play" script. All critical parameters are located at the very top of the script in the **CONTROL PANEL**.

Simply run the script via terminal:
```bash
python argyris_benchmark.py

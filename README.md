# Numerical Implementation of Incompatible Elements in Elasto-plasticity

This repository contains the numerical implementation of a novel thermodynamic formulation for small-strain elasto-plasticity. By treating the total strain tensor as an independent primary variable and utilizing the incompatibility operator ($\text{inc } E$), this model bypasses the classical multiplicative decomposition of the deformation gradient and captures plastic localization purely through thermodynamic energy minimization.

This project was implemented using the [Firedrake](https://www.firedrakeproject.org/) finite element framework, employing $C^1$-continuous elements to handle the second-order spatial derivatives required by the double-curl incompatibility operator.

## ✨ Key Features

* **Advanced $C^1$ Finite Elements:** Benchmarks and utilizes the 5th-order **Argyris** element (high-fidelity ground truth) and the 3rd-order **HCT** macro-element (computationally efficient).
* **Novel Thermodynamic Coupling:** Implements a scalar internal compatibility modulus ($\theta$) governed by the principle of maximal dissipation to model material yielding and softening naturally.
* **Custom Non-Linear Solver:** Bypasses standard black-box solvers with a custom Newton-Raphson loop featuring an analytical Jacobian for fast, quadratic local convergence.
* **Robust Saddle-Point Resolution:** Utilizes the **MUMPS** direct solver with a `NONZERO` shift factor ($10^{-10}$) to successfully invert highly indefinite mixed-space matrices without zero-pivot divergence.
* **Automated Parameter Sweeps:** Includes a fully automated batch loop for dynamic Gmsh generation, sequential element execution, and macroscopic F-U curve plotting.

## 🛠️ Dependencies

To run this simulation, you need to have the Firedrake environment installed, along with a few Python scientific libraries:
* [Firedrake](https://www.firedrakeproject.org/download.html) (with MUMPS and PETSc enabled)
* [Gmsh](https://gmsh.info/) (Python API)
* `numpy`
* `matplotlib`

## 🚀 Usage

### 1. Configuration
The main executable script contains a **Core Control Panel** at the top of the file. You can easily perform sensitivity analysis by modifying the thermodynamic variables:

```python
# --- Physical Parameter Sensitivity Settings ---
PARAM_B     = 0.014  # Hardening parameter (eta)
PARAM_KMU   = 1e4    # Shear modulus softening parameter (k)
PARAM_KALBE = 1.0    # Incompatibility penalty parameter (chi)

# --- Mesh Size ---
mesh_sizes_to_test = [0.04] # Define gmsh clscale here

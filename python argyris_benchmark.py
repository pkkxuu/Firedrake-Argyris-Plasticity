import numpy as np
import time
import gmsh
import sys
import os
from firedrake import *
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONTROL PANEL (BENCHMARK & CONVERGENCE SETTINGS)
# =====================================================================
element_type = "Argyris"

# [Benchmark Parameter] Current macroscopic load. 
# Change to 2.0 for the second benchmark run to match the paper.
g_load = 1.4       

# [Convergence Parameter] Characteristic mesh size. 
# For mesh convergence analysis, sequentially change to 0.1, 0.08, 0.05
clscale = 0.05     

# [Strict alignment with the publication]
nincrements = 20   

# Dynamically create isolated output directories to prevent overwriting results
output_dir = f"Results_{element_type}_g{g_load}_mesh{clscale}"
os.makedirs(output_dir, exist_ok=True)
print(f"🚀 Starting Benchmark: {element_type} Element | Load g={g_load} | Mesh {clscale}")

# =====================================================================
# 2. GEOMETRY & MESH GENERATION (Dynamically bound to clscale)
# =====================================================================
if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()
gmsh.model.add("SquareWithHole")
factory = gmsh.model.occ

square = factory.addRectangle(0, 0, 0, 1, 1)
hole = factory.addDisk(0.5, 0.5, 0, 0.1, 0.1)
factory.cut([(2, square)], [(2, hole)], removeObject=True, removeTool=True)
factory.synchronize()

curves = gmsh.model.getEntities(1)
left, bottom, right, top, hole_bnd = [], [], [], [], []

for dim, tag in curves:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    if abs(com[0] - 0.0) < 1e-6: left.append(tag)
    elif abs(com[1] - 0.0) < 1e-6: bottom.append(tag)
    elif abs(com[0] - 1.0) < 1e-6: right.append(tag)
    elif abs(com[1] - 1.0) < 1e-6: top.append(tag)
    else: hole_bnd.append(tag)

gmsh.model.addPhysicalGroup(1, left, 1)
gmsh.model.addPhysicalGroup(1, bottom, 2)
gmsh.model.addPhysicalGroup(1, right, 3)
gmsh.model.addPhysicalGroup(1, top, 4)
gmsh.model.addPhysicalGroup(1, hole_bnd, 5)
surfaces = gmsh.model.getEntities(2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)

gmsh.option.setNumber("Mesh.MeshSizeMin", clscale)
gmsh.option.setNumber("Mesh.MeshSizeMax", clscale)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
gmsh.model.mesh.generate(2)
gmsh.write("plate_with_hole.msh")
gmsh.finalize()

mesh = Mesh("plate_with_hole.msh")
print(f"Mesh generated successfully with {mesh.num_cells()} cells.")

# =====================================================================
# 3. MIXED FINITE ELEMENT SPACES
# =====================================================================
V_u = VectorFunctionSpace(mesh, "CG", 2)
V_rho = FunctionSpace(mesh, "DG", 0)
V_E = VectorFunctionSpace(mesh, "Argyris", 5, dim=3)
V_p = VectorFunctionSpace(mesh, "CG", 2)
W = MixedFunctionSpace([V_u, V_E, V_p])

print(f"\nDegrees of Freedom (DOFs):")
print(f" - Displacement (u): {V_u.dim()}")
print(f" - Internal variable (rho): {V_rho.dim()}")
print(f" - Strain component (E_ij): {V_E.dim()}")
print(f" => Total Global DOFs: {W.dim()}\n")

u, E, p = TrialFunctions(W)
v, F_test, q = TestFunctions(W)
rho = Function(V_rho, name="rho") 
rho.assign(2e-4)

# =====================================================================
# 4. MATHEMATICAL OPERATORS & VARIATIONAL FORMULATION
# =====================================================================
def eps(u): return sym(grad(u))
def inc(E): return E[0].dx(1).dx(1) + E[1].dx(0).dx(0) - 2 * E[2].dx(0).dx(1)
def as_sym_tensor(vec): return as_tensor([[vec[0], vec[2]], [vec[2], vec[1]]])
def tr_E(vec): return vec[0] + vec[1]

kmu = Constant(1e4)
kalbe = Constant(1.0)
mu0 = Constant(38.46)
kappa0 = Constant(83.0)
epsu = Constant(1e-4)
epsp = Constant(1e-4)

# Boundary load initializer (updated during the incremental loop)
Force = Constant(0.0) 

rho_safe = max_value(rho, 1e-8)
mut = kmu / rho_safe
mu_val = (mu0 * mut) / (mu0 + mut)
kappa_val = kappa0
albe_val = kalbe / rho_safe

a = (
    (kappa_val - (2.0/3.0)*mu_val) * (div(u) + tr_E(E)) * (div(v) + tr_E(F_test)) * dx
    + 2.0 * mu_val * inner(eps(u) + as_sym_tensor(E), eps(v) + as_sym_tensor(F_test)) * dx
    + albe_val * inc(E) * inc(F_test) * dx
    + inner(as_sym_tensor(E), eps(q)) * dx
    + inner(eps(p), as_sym_tensor(F_test)) * dx
    + epsu * inner(u, v) * dx
    - epsp * inner(p, q) * dx
)

L = (-Force * v[0]) * ds(1) + (Force * v[0]) * ds(3)

w_solution = Function(W)
problem = LinearVariationalProblem(a, L, w_solution)
solver = LinearVariationalSolver(problem, solver_parameters={
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 500,    
    "mat_mumps_icntl_24": 1,      
    "pc_factor_shift_type": "NONZERO", 
    "pc_factor_shift_amount": 1e-10
})

# =====================================================================
# 5. ITERATION PARAMETERS & PRE-COMPILED EXPRESSIONS
# =====================================================================
F_max = g_load  
b_val = 0.006
niter = 20
nnewt = 10
epsnewt = 1e-8
yieldsq = 1.0
mu0_f, kmu_f, kalbe_f = float(mu0), float(kmu), float(kalbe)
rho_tol = 1e-4

u_sol, E_sol, p_sol = w_solution.subfunctions
tre_expr = div(u_sol)
trEE_expr = tr_E(E_sol)
e_u_E = eps(u_sol) + as_sym_tensor(E_sol)

cokappa_expr = 0.5 * (tre_expr + trEE_expr)**2
comu_expr = - (1.0/3.0) * (tre_expr + trEE_expr)**2 + inner(e_u_E, e_u_E)
coalbe_expr = 0.5 * inc(E_sol)**2

cokappa_func = Function(V_rho)
comu_func = Function(V_rho)
coalbe_func = Function(V_rho)

U_history, F_history = [], []

# Prepare VTK outputs for ParaView visualization
vtk_file = VTKFile(f"{output_dir}/dynamics.pvd")
V_E_vis = VectorFunctionSpace(mesh, "CG", 1, dim=3)
E_vis = Function(V_E_vis, name="Strain_E_ij")
V_stress = TensorFunctionSpace(mesh, "DG", 1)
stress_vis = Function(V_stress, name="Stress")
I_tensor = Identity(2)

start_time = time.time()

# =====================================================================
# 6. NON-LINEAR INCREMENTAL LOOP
# =====================================================================
for istep in range(nincrements):
    current_force = F_max * (istep + 1) / nincrements
    Force.assign(current_force) 
    
    print(f"\n=== Load Step: {istep + 1}/{nincrements} | Force = {current_force:.4f} ===")
    rhol_array = rho.dat.data.copy()
    
    for iteration in range(1, niter):
        rho_old_check = rho.dat.data.copy()
        t_start = time.time()
        
        # A. MUMPS Direct LU Factorization
        solver.solve()
        t_solve = time.time()
        
        # B. Fast Data Extraction (Interpolation)
        cokappa = cokappa_func.interpolate(cokappa_expr).dat.data
        comu = comu_func.interpolate(comu_expr).dat.data
        coalbe = coalbe_func.interpolate(coalbe_expr).dat.data
        rho_array = rho.dat.data
        t_interp = time.time()
        
        # C. Vectorized NumPy Operations (In-memory computation)
        gammal = (yieldsq / (4.0 * kmu_f)) * np.sqrt(1.0 + b_val * rho_array)
        gammaEff = 0.5 * (1.0 / (istep + 0.5)) * gammal
        anewt = (2 * istep + 1) * comu / gammaEff
        bnewt = (2 * istep + 1) * coalbe * kalbe_f / gammaEff
        
        qstar = 1.0 - (anewt * kmu_f * mu0_f**2) / ((mu0_f * np.sqrt(bnewt) + kmu_f)**2)
        q = (np.maximum(0, qstar) + 1.0) / 2.0
        
        # Inner Newton method for q
        for inewt in range(1, nnewt):
            h = 2 * np.sqrt(bnewt * q) - (np.sqrt(anewt * mu0_f) - np.sqrt(kmu_f * (1 - q) / mu0_f))**2
            dh = np.sqrt(bnewt / q) + kmu_f / mu0_f - np.sqrt(anewt * kmu_f / (1 - q))
            ddh = -0.5 * (np.sqrt(bnewt) / (q**1.5) + np.sqrt(anewt * kmu_f) / ((1 - q)**1.5))
            q = np.maximum(np.maximum(qstar, epsnewt), np.minimum(1 - epsnewt, q - dh / ddh))
                
        rho_new = np.sqrt((anewt * kmu_f) / (1 - q)) - (kmu_f / mu0_f)
        t_numpy = time.time()

        print(f"   -> [Performance] Solve: {t_solve - t_start:.2f}s | Interp: {t_interp - t_solve:.2f}s | NumPy: {t_numpy - t_interp:.2f}s")

        # D & E. Field Update & Convergence Check
        rho.dat.data[:] = np.maximum(rhol_array, np.maximum(rho_new, 2e-4))
        rho_change = np.max(np.abs(rho.dat.data - rho_old_check))
        
        if rho_change < rho_tol and iteration > 1:
            print(f"   -> Converged early at iteration {iteration}. Max diff: {rho_change:.2e}")
            break

    # Export quantities for visualization (Displacement, Strain, Internal Variable, Stress)
    u_sol.rename("Displacement_u")
    rho.rename("Internal_Variable_rho")
    E_vis.project(E_sol)
    
    # Assembly of the macroscopic stress tensor
    stress_expr = (kappa_val - (2.0/3.0)*mu_val) * (div(u_sol) + tr_E(E_sol)) * I_tensor + 2.0 * mu_val * (eps(u_sol) + as_sym_tensor(E_sol))
    stress_vis.project(stress_expr)
    
    vtk_file.write(u_sol, E_vis, rho, stress_vis, time=istep+1)

    u_left = assemble(-u_sol[0] * ds(1))
    u_right = assemble(u_sol[0] * ds(3))
    Utot = u_left + u_right
    F_history.append(current_force)
    U_history.append(Utot)

end_time = time.time()

# =====================================================================
# 7. FINAL DATA EXTRACTION (FOR MESH CONVERGENCE ANALYSIS)
# =====================================================================
print("\n" + "="*55)
print("CONVERGENCE DATA EXTRACTION")
print("="*55)
print(f"Element Type        : {element_type}")
print(f"Load (g)            : {g_load}")
print(f"Mesh Size (clscale) : {clscale}")
print(f"Total Global DOFs   : {W.dim()}")
print(f"Total Compute Time  : {(end_time - start_time)/60:.2f} minutes")

# Compute the total elastoplastic free energy W at the final step
Total_Energy_expr = 0.5 * (kappa_val - (2.0/3.0)*mu_val) * (div(u_sol) + tr_E(E_sol))**2 \
                    + mu_val * inner(e_u_E, e_u_E) \
                    + 0.5 * albe_val * inc(E_sol)**2
Total_Energy = assemble(Total_Energy_expr * dx)

print(f"FINAL TOTAL ENERGY W: {Total_Energy:.8e}")
print("="*55 + "\n")

# Save Force-Displacement Curve
plt.figure(figsize=(8, 6))
plt.plot(U_history, F_history, 'b-o', linewidth=2, markersize=6)
plt.title(f"Force-Displacement ({element_type}, g={g_load}, clscale={clscale})", fontsize=14)
plt.xlabel("Total Displacement $U$", fontsize=12)
plt.ylabel("Applied Force $F$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/FD_curve.png")
print(f"Results saved successfully in folder: {output_dir}/")

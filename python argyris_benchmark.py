import numpy as np
import time
import gmsh
import sys
import os
from firedrake import *
import matplotlib.pyplot as plt

# =====================================================================
# 1. Core Control Panel: Physical Parameters & Mesh Settings
# =====================================================================
g_load = 2.0       
nincrements = 20   

# --- Mesh Size for Parameter Testing ---
mesh_sizes_to_test = [0.03] 

# --- Physical Parameter Sensitivity Settings ---
PARAM_B     = 0.014  # Hardening parameter (eta / b)
PARAM_KMU   = 1e4    # Shear modulus softening parameter (k)
PARAM_KALBE = 1.0    # Incompatibility penalty parameter (chi)

# =====================================================================
# 2. Core Solver Encapsulation 
# =====================================================================
def run_simulation(element_type, clscale, output_dir, mesh):
    print(f"\n   -> Running {element_type} element...")
    print(f"      Parameters: b={PARAM_B}, kmu={PARAM_KMU}, kalbe={PARAM_KALBE}")
    
    # Define mixed function spaces
    V_u = VectorFunctionSpace(mesh, "CG", 2)
    V_rho = FunctionSpace(mesh, "DG", 0)
    
    if element_type == "Argyris":
        V_E = VectorFunctionSpace(mesh, "Argyris", 5, dim=3)
    elif element_type == "HCT":
        V_E = VectorFunctionSpace(mesh, "HCT", 3, dim=3)
        
    V_p = VectorFunctionSpace(mesh, "CG", 2)
    W = MixedFunctionSpace([V_u, V_E, V_p])

    # Trial and Test functions
    u, E, p = TrialFunctions(W)
    v, F_test, q = TestFunctions(W)

    # Global solution variables
    u_tot = Function(V_u, name="Displacement_Total")
    E_tot = Function(V_E, name="Strain_Total")
    rho = Function(V_rho, name="rho") 
    rho.assign(2e-4) # Initial compatibility modulus

    # UFL Helper functions
    def eps(u): return sym(grad(u))
    def inc(E): return E[0].dx(1).dx(1) + E[1].dx(0).dx(0) - 2 * E[2].dx(0).dx(1)
    def as_sym_tensor(vec): return as_tensor([[vec[0], vec[2]], [vec[2], vec[1]]])
    def tr_E(vec): return vec[0] + vec[1]

    # Material constants
    kmu = Constant(PARAM_KMU)
    kalbe = Constant(PARAM_KALBE)
    mu0, kappa0 = Constant(38.46), Constant(83.0)
    epsu, epsp = Constant(1e-4), Constant(1e-4)
    Force_inc = Constant(g_load / nincrements) 

    # Dynamic tangent moduli dependent on rho (theta)
    rho_safe = max_value(rho, 1e-8)
    mut = kmu / rho_safe
    mu_val = (mu0 * mut) / (mu0 + mut)
    kappa_val = kappa0
    albe_val = kalbe / rho_safe

    # Variational Weak Form (LHS)
    a = (
        (kappa_val - (2.0/3.0)*mu_val) * (div(u) + tr_E(E)) * (div(v) + tr_E(F_test)) * dx
        + 2.0 * mu_val * inner(eps(u) + as_sym_tensor(E), eps(v) + as_sym_tensor(F_test)) * dx
        + albe_val * inc(E) * inc(F_test) * dx
        + inner(as_sym_tensor(E), eps(q)) * dx
        + inner(eps(p), as_sym_tensor(F_test)) * dx
        + epsu * inner(u, v) * dx
        - epsp * inner(p, q) * dx
    )

    # Variational Weak Form (RHS) - Uniaxial tension
    L = (-Force_inc * v[0]) * ds(1) + (Force_inc * v[0]) * ds(3)

    # Linear solver configuration (Direct solver for saddle-point matrix)
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

    # Custom Newton Loop Parameters
    b_val = PARAM_B
    niter, nnewt, epsnewt, yieldsq = 20, 10, 1e-8, 1.0
    mu0_f, kmu_f, kalbe_f = float(mu0), float(kmu), float(kalbe)
    rho_tol = 1e-4

    # Extract subfunctions for local stress/energy evaluation
    u_sol, E_sol, p_sol = w_solution.subfunctions
    tre_expr, trEE_expr = div(u_sol), tr_E(E_sol)
    e_u_E = eps(u_sol) + as_sym_tensor(E_sol)

    cokappa_expr = 0.5 * (tre_expr + trEE_expr)**2
    comu_expr = - (1.0/3.0) * (tre_expr + trEE_expr)**2 + inner(e_u_E, e_u_E)
    coalbe_expr = 0.5 * inc(E_sol)**2

    cokappa_func, comu_func, coalbe_func = Function(V_rho), Function(V_rho), Function(V_rho)

    # ==== ParaView Export Setup ====
    vtk_file = VTKFile(f"{output_dir}/dynamics_{element_type}.pvd")
    V_E_vis = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    E_vis = Function(V_E_vis, name="Strain_Total_vis")
    V_stress = TensorFunctionSpace(mesh, "DG", 1)
    stress_tot = Function(V_stress, name="Stress_Total")
    I_tensor = Identity(2)
    
    # Extract incompatible strain field
    V_incE = FunctionSpace(mesh, "DG", 1)
    incE_vis = Function(V_incE, name="SincEzz")

    U_history, F_history = [0.0], [0.0]

    # --- LOADING PHASE ---
    for istep in range(nincrements):
        current_force = g_load * (istep + 1) / nincrements
        rhol_array = rho.dat.data.copy()
        
        for iteration in range(1, niter):
            rho_old_check = rho.dat.data.copy()
            solver.solve()
            
            # Interpolate local energy terms
            cokappa = cokappa_func.interpolate(cokappa_expr).dat.data
            comu = comu_func.interpolate(comu_expr).dat.data
            coalbe = coalbe_func.interpolate(coalbe_expr).dat.data
            rho_array = rho.dat.data
            
            # Thermodynamic optimization variables
            gammal = (yieldsq / (4.0 * kmu_f)) * np.sqrt(1.0 + b_val * rho_array)
            gammaEff = 0.5 * (1.0 / (istep + 0.5)) * gammal
            anewt = (2 * istep + 1) * comu / gammaEff
            bnewt = (2 * istep + 1) * coalbe * kalbe_f / gammaEff
            
            qstar = 1.0 - (anewt * kmu_f * mu0_f**2) / ((mu0_f * np.sqrt(bnewt) + kmu_f)**2)
            q = (np.maximum(0, qstar) + 1.0) / 2.0
            
            # Custom Newton-Raphson iteration for local rho update
            for inewt in range(1, nnewt):
                h = 2 * np.sqrt(bnewt * q) - (np.sqrt(anewt * mu0_f) - np.sqrt(kmu_f * (1 - q) / mu0_f))**2
                dh = np.sqrt(bnewt / q) + kmu_f / mu0_f - np.sqrt(anewt * kmu_f / (1 - q))
                ddh = -0.5 * (np.sqrt(bnewt) / (q**1.5) + np.sqrt(anewt * kmu_f) / ((1 - q)**1.5))
                q = np.maximum(np.maximum(qstar, epsnewt), np.minimum(1 - epsnewt, q - dh / ddh))
                    
            rho_new = np.sqrt((anewt * kmu_f) / (1 - q)) - (kmu_f / mu0_f)

            # Update rho and check global vectorized convergence
            rho.dat.data[:] = np.maximum(rhol_array, np.maximum(rho_new, 2e-4))
            if np.max(np.abs(rho.dat.data - rho_old_check)) < rho_tol and iteration > 1:
                break

        # Accumulate state variables
        u_tot.assign(u_tot + u_sol)
        E_tot.assign(E_tot + E_sol)
        
        # Compute macroscopic stress
        stress_inc_expr = (kappa_val - (2.0/3.0)*mu_val) * (div(u_sol) + tr_E(E_sol)) * I_tensor + 2.0 * mu_val * (eps(u_sol) + as_sym_tensor(E_sol))
        stress_inc = project(stress_inc_expr, V_stress)
        stress_tot.assign(stress_tot + stress_inc)
        
        E_vis.project(E_tot)
        incE_vis.project(inc(E_tot)) 
        
        vtk_file.write(u_tot, E_vis, rho, stress_tot, incE_vis, time=istep+1)
        
        # Track macroscopic Force-Displacement
        u_left = assemble(-u_tot[0] * ds(1))
        u_right = assemble(u_tot[0] * ds(3))
        F_history.append(current_force)
        U_history.append(u_left + u_right)

    # --- UNLOADING PHASE ---
    Force_inc.assign(-g_load) 
    rho.assign(2e-4)          
    solver.solve()            
    
    u_tot.assign(u_tot + u_sol)
    E_tot.assign(E_tot + E_sol)
    
    stress_inc_expr = (kappa_val - (2.0/3.0)*mu_val) * (div(u_sol) + tr_E(E_sol)) * I_tensor + 2.0 * mu_val * (eps(u_sol) + as_sym_tensor(E_sol))
    stress_inc = project(stress_inc_expr, V_stress)
    stress_tot.assign(stress_tot + stress_inc)
    
    E_vis.project(E_tot)
    incE_vis.project(inc(E_tot))
    
    vtk_file.write(u_tot, E_vis, rho, stress_tot, incE_vis, time=nincrements+1) 
    
    u_left = assemble(-u_tot[0] * ds(1))
    u_right = assemble(u_tot[0] * ds(3))
    F_history.append(0.0)     
    U_history.append(u_left + u_right) 
    
    return U_history, F_history

# =====================================================================
# 3. Batch Automation Loop: Parameter Testing
# =====================================================================
for clscale in mesh_sizes_to_test:
    print(f"\n" + "="*60)
    print(f" STARTING PARAMETER TEST: clscale = {clscale}")
    print(f"   b={PARAM_B}, kmu={PARAM_KMU}, kalbe={PARAM_KALBE}")
    print("="*60)
    
    # Create distinct output directories based on active parameters
    output_dir = f"Results_ParamTest_b{PARAM_B}_k{PARAM_KMU}_chi{PARAM_KALBE}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dynamic Gmsh generation
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add("SquareWithHole")
    factory = gmsh.model.occ

    square = factory.addRectangle(0, 0, 0, 1, 1)
    hole = factory.addDisk(0.5, 0.5, 0, 0.1, 0.1)
    factory.cut([(2, square)], [(2, hole)], removeObject=True, removeTool=True)
    factory.synchronize()

    # Boundary tagging
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
    
    # Execute and capture results for both elements
    U_arg, F_arg = run_simulation("Argyris", clscale, output_dir, mesh)
    U_hct, F_hct = run_simulation("HCT", clscale, output_dir, mesh)
    
    # Automatically plot and save comparisons
    plt.figure(figsize=(10, 8))
    plt.plot(U_arg, F_arg, 'b-o', linewidth=2.5, markersize=7, zorder=2, label=f"Argyris")
    plt.plot(U_hct, F_hct, 'r--s', linewidth=2.5, markersize=7, zorder=3, label=f"HCT")

    plt.title(f"Parameter Sensitivity Analysis\n$b={PARAM_B},\ k={PARAM_KMU:.1e},\ \chi={PARAM_KALBE}$", fontsize=15, fontweight='bold')
    plt.xlabel("Total Displacement $U$", fontsize=13)
    plt.ylabel("Total Applied Force $F$", fontsize=13)
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)
    
    plt.savefig(f"{output_dir}/Param_Comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\n Parameter test complete. Please check the corresponding output directory.")

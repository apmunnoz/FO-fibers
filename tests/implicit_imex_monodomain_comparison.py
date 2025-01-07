from firedrake import *
from sys import argv
from tqdm import tqdm
import numpy as np 
import Mesh
import Fibers
import Ionics
import Electrics
from Units import Units
from Printing import parprint
from Parameters import IonicsParameters, ElectricsParameters
from RK4 import RK4
from time import perf_counter as time

parameters['form_compiler']['quadrature_degree'] = 3
#units = Units(Metres=1e3, Kilograms=1e3, Seconds=1e3, Kelvin=1, Ampere=1e3, mol=1e3)
units = Units(Metres=1, Kilograms=1, Seconds=1, Kelvin=1, Ampere=1, mol=1)

# Set by user ---------------------------------------------
dt = 2e-2 * units.ms(0)    # 5e-2 ms --> 5e-5 [s]
t0 = 0.0     # [s]
tf = 50 * dt
times_ODE = 1
times_exact = 100
dt_ode = dt/times_ODE
dt_exact = dt/times_exact # N_ode is this in exact solution

w_degree = 1
u_degree = 1

pde_model = Electrics.MONODOMAIN
ode_model = Ionics.FHN 
mesh_type = Mesh.PROLATE
if mesh_type == Mesh.UNITSQUARE: 
    N = int(argv[1])
    nx = ny = N
    mesh, dx, _ = Mesh.getMesh(mesh_type=mesh_type, nx=nx, ny=ny)  # [cm]
    fact = 10 * 1e-3 * units.L(0)
    mesh.coordinates.dat.data[:] *= fact  # [cm] to [m]
else:
    mesh, dx, _, _, _ = Mesh.getMesh(mesh_type=mesh_type)  # [cm]
    fact = 1e-3 * units.L(0)
    mesh.coordinates.dat.data[:] *= fact  # [cm] to [m]

parsol_pde = {'ksp_type': 'gmres',
              'pc_type': 'gamg',
              'ksp_norm_type': 'unpreconditioned',
              'snes_atol': 0.0,
              'snes_rtol': 1e-6,
              'snes_stol': 0.0,
              #'ksp_monitor': None,
              'ksp_gmres_restart': 200,
              'ksp_max_it': 200,
              'ksp_rtol': '1e-6',
              'ksp_atol': '0.0'}


time_expression = Constant(t0)

W = Ionics.getFunctionalSpace(mesh, w_degree, ode_model)
V = Electrics.getFunctionalSpace(mesh, u_degree, pde_model)
w, wn, w_test = Ionics.getFunctions(W, ode_model, units)
u, un, u_test = Electrics.getFunctions(V)
u0 = un.copy(True) # Used for restart
w0 = wn.copy(True)

f, s, n = Fibers.computeFibers(mesh, mesh_type=mesh_type)

sigma = Electrics.getDiffusion(f, s, n, mesh_type, pde_model, units)
v = Electrics.getTransmembranePotential(u, pde_model)
vn = Electrics.getTransmembranePotential(un, pde_model)
Iion_IMEX = Ionics.getIonicCurrent(vn, w, ode_model, units)
Iion = Ionics.getIonicCurrent(v, w, ode_model, units)
Iapp = Ionics.getIapp(mesh, time_expression, mesh_type, units)

def solveExact():

    Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
    Fu_imp_exact = Electrics.getResidual(u, un, u_test, w, sigma, Iion, Iapp, time_expression, dt_exact, pde_model, False, units)
    FRKw = lambda ww: Fw(v, ww)
    rk4 = RK4(w, wn, time_expression, dt_exact, FRKw)
    PDE_problem_exact = NonlinearVariationalProblem(Fu_imp_exact, u)

    Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
    A, b = Electrics.getIMEXAb(u, un, u_test, sigma, Iion_IMEX, Iapp, dt_exact, pde_model, units)
    PDE_problem = LinearVariationalProblem(A, b, u, constant_jacobian=True)
    PDE_solver_exact = LinearVariationalSolver(PDE_problem, solver_parameters=parsol_pde)

    # Init to solve
    un.assign(u0)
    wn.assign(w0)
    u.assign(un)
    w.assign(wn)
    us_exact = []
    ts_exact = []
    i = 1
    for t in tqdm(np.arange(t0+dt, tf, dt_exact)):

        t_init = time()
        rk4.solve(t)
        wn.assign(w)

        time_expression.assign(t)
        PDE_solver_exact.solve()
        t_done = time() - t_init
        un.assign(u)
        
        if i%times_exact==0: #export snaps of the inexact models
            us_exact.append(u.copy(True))
            ts_exact.append(t_done)
        i+=1
    return us_exact, ts_exact

def solveImplicit():

    Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
    FRKw = lambda ww: Fw(v, ww)
    rk4 = RK4(w, wn, time_expression, dt_ode, FRKw)

    params = parsol_pde.copy()
    params["snes_type"] = "newtonls"
    #params["snes_monitor"] = None
    #params["ksp_converged_reason"] = None
    #params["ksp_rtol"] = 1e-1
    Fu= Electrics.getResidual(u, un, u_test, w, sigma, Iion, Iapp, time_expression, dt, pde_model, False, units)
    PDE_problem_imp = NonlinearVariationalProblem(Fu, u)
    PDE_solver = NonlinearVariationalSolver(PDE_problem_imp, solver_parameters=params)

    # Init to solve
    un.assign(u0)
    wn.assign(w0)
    u.assign(un)
    w.assign(wn)
    us = []
    ts = []
    i = 1
    for t in tqdm(np.arange(t0+dt, tf, dt)):

        t_init = time()
        for j in range(times_ODE):
            tj = t - dt + j * dt_ode
            rk4.solve(tj)
            wn.assign(w)

        time_expression.assign(t)
        PDE_solver.solve()
        t_done = time() - t_init

        un.assign(u)
        us.append(u.copy(True))
        ts.append(t_done)
        i+=1
    return us, ts

def solveIMEX(NN):

    # Rescaled timesteps
    dt_ode_imex = dt_ode / NN
    dt_imex = dt / NN

    Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
    FRKw = lambda ww: Fw(v, ww)
    rk4 = RK4(w, wn, time_expression, dt_ode_imex, FRKw)

    Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
    A, b = Electrics.getIMEXAb(u, un, u_test, sigma, Iion_IMEX, Iapp, dt_imex, pde_model, units)
    PDE_problem = LinearVariationalProblem(A, b, u, constant_jacobian=True)
    PDE_solver = LinearVariationalSolver(PDE_problem, solver_parameters=parsol_pde)

    # Init to solve
    un.assign(u0)
    wn.assign(w0)
    u.assign(un)
    w.assign(wn)
    us = []
    ts = []
    i = 1
    for t in tqdm(np.arange(t0+dt, tf, dt_imex)):

        t_init = time()
        for j in range(times_ODE):
            tj = t - dt + j * dt_ode_imex
            rk4.solve(tj)
            wn.assign(w)

        time_expression.assign(t)
        PDE_solver.solve()
        t_done = time() - t_init

        un.assign(u)
        if i%NN == 0:
            us.append(u.copy(True))
            ts.append(t_done)
        i+=1
    return us, ts

def computeErrors(us_ex, us_imp, us_imex):
    err_imp = []
    err_imex = []
    for u_ex, u_imp, u_imex in zip(us_ex, us_imp, us_imex):
        err_imp.append(errornorm(u_ex, u_imp))
        err_imex.append(errornorm(u_ex, u_imex))
    return np.average(err_imp), np.average(err_imex)


#### INIT SCRIPT ####
us_ex, ts_ex = solveExact()
us_imex, ts_imex = solveIMEX(1)
us_imp, ts_imp = solveImplicit()

err_imp, err_imex = computeErrors(us_ex, us_imp, us_imex)
print("Errors:", err_imp, err_imex)
NN = 1
while True:
    err_imp, err_imex = computeErrors(us_ex, us_imp, us_imex)
    if err_imex > err_imp:
        NN = NN+1
        us_imex, ts_imex = solveIMEX(NN)
        print("IMEX worse:", err_imp, err_imex)
    else:
        break

print("Factor computed:", NN)
print(np.average(ts_imp), np.average(ts_imex))

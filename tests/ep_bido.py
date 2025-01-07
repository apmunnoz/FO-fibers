from firedrake import *
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

# DEBUG
from IonicsAux import TTPModel

parameters['form_compiler']['quadrature_degree'] = 3
#units = Units(Metres=1e3, Kilograms=1e3, Seconds=1e3, Kelvin=1, Ampere=1e3, mol=1e3)
units = Units(Metres=1, Kilograms=1, Seconds=1, Kelvin=1, Ampere=1, mol=1)

# Set by user ---------------------------------------------
dt = 2e-2 * units.ms(0)    # 5e-2 ms --> 5e-5 [s]
t0 = 0.0     # [s]
#tf = 1e3 * units.ms(0)       # [s]
tf = 10 * dt
saveEvery = 1e8 # big number to never save
times_ODE = 10
dt_ode = dt / times_ODE

w_degree = 1
u_degree = 1

pde_model = Electrics.BIDOMAIN
ode_model = Ionics.FHN  # FHN|TTP
mesh_type = Mesh.PROLATE  # PROLATE|BIVENTRICLE|UNITSQUARE
N = 100 # Elements per side for unit square

IMEX = True     # electric - True:IMEX|False:fully implicit

if IMEX:
    parsol_pde = {'ksp_type': 'gmres',
                  'pc_type': 'asm',
                  'ksp_norm_type': 'unpreconditioned', 
                  'snes_atol': 1e-14,
                  'snes_rtol': 1e-6,
                  'snes_stol': 0.0,
                  #'ksp_monitor': None,
                  'ksp_rtol': '1e-6',
                  'ksp_atol': '1e-14'}
else:
    parsol_pde = {'ksp_type': 'preonly',
                  'pc_type': 'asm',
                  'ksp_norm_type': 'none', 
                  'snes_atol': 1e-14,
                  'snes_rtol': 1e-6,
                  'snes_stol': 0.0,
                  'ksp_error_if_not_converged': False, 
                  'snes_qn_restart_type': 'none',
                  'snes_type': 'qn', 
                  'snes_qn_scale_type': 'jacobian',
                  'snes_lag_jacobian': -2,
                  'snes_lag_preconditioner': -2,
                  'snes_qn_m': 20,
                  'snes_converged_reason': None,
                  #'ksp_converged_reason': None,
                  'ksp_max_it': 5,
                  'ksp_rtol': '1e-6',
                  'ksp_atol': '1e-14'}


if mesh_type == Mesh.PROLATE:
    mesh, dx, _, _, _ = Mesh.getMesh(mesh_type=mesh_type)  # [mm]
    mesh.coordinates.dat.data[:] *= 1e-3 * units.L(0)  # [mm] to [m]
elif mesh_type == Mesh.BIVENTRICLE:
    mesh, dx, _, _, _, _ = Mesh.getMesh(mesh_type=mesh_type)  # [mm]
    mesh.coordinates.dat.data[:] *= 1e-3 * units.L(0)  # [mm] to [m]
elif mesh_type == Mesh.UNITSQUARE:
    nx = ny = N
    mesh, dx, _ = Mesh.getMesh(mesh_type=mesh_type, nx=nx, ny=ny)  # [cm]
    fact = 10 * 1e-3 * units.L(0)
    mesh.coordinates.dat.data[:] *= fact  # [cm] to [m]

time_expression = Constant(t0)

W = Ionics.getFunctionalSpace(mesh, w_degree, ode_model)
V = Electrics.getFunctionalSpace(mesh, u_degree, pde_model)
w, wn, w_test = Ionics.getFunctions(W, ode_model, units)
u, un, u_test = Electrics.getFunctions(V)

nullspace = MixedVectorSpaceBasis(
    V, [V.sub(0), VectorSpaceBasis(constant=True)])

f, s, n = Fibers.computeFibers(mesh, mesh_type=mesh_type)

sigma = Electrics.getDiffusion(
    f, s, n, mesh_type, pde_model, units)
v = Electrics.getTransmembranePotential(u, pde_model)
vn = Electrics.getTransmembranePotential(un, pde_model)
if IMEX: 
    Iion = Ionics.getIonicCurrent(vn, w, ode_model, units)
else:
    Iion = Ionics.getIonicCurrent(v, w, ode_model, units)
Iapp = Ionics.getIapp(mesh, time_expression, mesh_type, units)

Fw = Ionics.getResidualFunction(Iapp, ode_model, units)
Fu = Electrics.getResidual(u, un, u_test, w, sigma, Iion,
                           Iapp, time_expression, dt, pde_model, IMEX, units)


FRKw = lambda ww: Fw(v, ww)
rk4 = RK4(w, wn, time_expression, dt_ode, FRKw)

if IMEX:
    A, b = Electrics.getIMEXAb(u, un, u_test, sigma, Iion, Iapp, dt, pde_model, units)
    PDE_problem = LinearVariationalProblem(A, b, u)
    PDE_solver = LinearVariationalSolver(PDE_problem, solver_parameters=parsol_pde, nullspace=nullspace)
else:
    PDE_problem = NonlinearVariationalProblem(Fu, u)
    PDE_solver = NonlinearVariationalSolver(PDE_problem, solver_parameters=parsol_pde, nullspace=nullspace)

i = 1
if ode_model == Ionics.TTP:
    u.interpolate(Constant(-83.23 * units.mV))
un.assign(u)
wn.assign(w)

outfile = File("output/ep-mono.pvd")
outfile.write(*u.subfunctions, t=t0)
ts_tot = []
ts_pde = []
for t in np.arange(t0+dt, tf, dt):

    t_init = time()
    for j in range(times_ODE):
        tj = t - dt + j * dt_ode
        rk4.solve(tj)
        wn.assign(w)

    time_expression.assign(t)
    t_init_pde = time()
    PDE_solver.solve()
    t_done_pde = time() - t_init_pde
    t_done = time() - t_init

    un.assign(u)
    ts_tot.append(t_done)
    ts_pde.append(t_done_pde)
    if i % saveEvery == 0:
        outfile.write(*u.subfunctions, t=t)
        parprint(f"Solved time {t:4.5f} in {t_done:4.3f}s")
    i += 1

t_pde_avg = np.average(ts_pde)
t_avg = np.average(ts_tot)
parprint("Done")
parprint(f"Avg PDE solution time: \t{t_pde_avg:4.5f}s")
parprint(f"Avg Total solution time: \t{t_avg:4.5f}s")

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


parameters['form_compiler']['quadrature_degree'] = 3
units = Units(Metres=1e3, Kilograms=1e3, Seconds=1e3, Kelvin=1, Ampere=1e3, mol=1e3)
#units = Units(Metres=1, Kilograms=1, Seconds=1, Kelvin=1, Ampere=1e3, mol=1e3)

# Set by user ---------------------------------------------
dt = 1e-3 * units.ms(0)    # 5e-2 ms --> 5e-5 [s]
t0 = 0.0     # [s]
tf = 1e3 * units.ms(0)       # [s]
saveEvery = 1
times_ODE = 1
dt_ode = dt / times_ODE
N = 1

w_degree = 1
u_degree = 1

pde_model = Electrics.MONODOMAIN
ode_model = Ionics.TTP  # FHN|TTP
mesh_type = Mesh.UNITSQUARE  # PROLATE|BIVENTRICLE|UNITSQUARE

IMEX = True     # electric - True:IMEX|False:fully implicit

parsol_pde = {'ksp_type': 'cg',
              'pc_type': 'gamg',
              'ksp_norm_type': 'unpreconditioned', 
              'snes_atol': 1e-14,
              'snes_rtol': 1e-6,
              'snes_stol': 0.0,
              #'ksp_monitor': None,
              'ksp_rtol': '1e-6',
              'ksp_atol': '1e-14'}

mesh = UnitTriangleMesh()
fact = 10 * 1e-3 * units.L(0)
mesh.coordinates.dat.data[:] *= fact  # [cm] to [m]

time_expression = Constant(t0)

W = Ionics.getFunctionalSpace(mesh, w_degree, ode_model, 'DG')
V = FunctionSpace(mesh,  'DG', 0)
w, wn, w_test = Ionics.getFunctions(W, ode_model, units)
u, un, u_test = Electrics.getFunctions(V)


sigma = Constant(0.0, domain=mesh)
Iion = Ionics.getIonicCurrent(un, w, ode_model, units)
Iapp = IonicsParameters(units).Ia * conditional(le(time_expression, 2 * units.ms), 1, 0)

Fw = Ionics.getResidualFunction(Iapp, ode_model, units)

FRKw = lambda ww: Fw(u, ww)
rk4 = RK4(w, wn, time_expression, dt_ode, FRKw)

A, b = Electrics.getIMEXAb(u, un, u_test, sigma, Iion, Iapp, dt, pde_model, units)
PDE_problem = LinearVariationalProblem(A, b, u)
PDE_solver = LinearVariationalSolver(PDE_problem, solver_parameters=parsol_pde)

i = 1
if ode_model == Ionics.TTP:
    u.interpolate(Constant(-83.23 * units.mV))
un.assign(u)
wn.assign(w)

outfile = File("output/ode.pvd")
outfile.write(u, *w.subfunctions, t=t0)
#for i in range(18): print("DEBUG", i, w((0,0))[i])
#print("DONE")
for t in np.arange(t0+dt, tf, dt):

    t_init = time()
    for j in range(times_ODE):
        tj = t - dt + j * dt_ode
        rk4.solve(tj)
        wn.assign(w)
        #print("DEBUG", norm(wn))
        #for i in range(18): print("DEBUG", i, w((0,0))[i])
        #print("DONE")

    time_expression.assign(t)
    PDE_solver.solve()
    t_done = time() - t_init

    un.assign(u)
    if i % saveEvery == 0:
        outfile.write(u, *w.subfunctions, t=t)
        parprint(f"Solved time {t:4.5f} in {t_done:4.3f}s")
    i += 1

print("Done")

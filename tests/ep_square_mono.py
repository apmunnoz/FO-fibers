from firedrake import *
import numpy as np
import Mesh
import Fibers
import Ionics
import Electrics
from Units import SI
from Printing import parprint
from Parameters import IonicsParameters, ElectricsParameters

# Set by user ---------------------------------------------
dt = 5e-5    # 5e-2 ms --> 5e-5 [s]
t0 = 0.0     # [s]
tf = 3       # [s]

R = 0.1      # [mm]

theta = 1    # ionic - 0:explicit|1:implicit|0.5:midpoint
IMEX = 0     # electric - 0:IMEX|1:fully implicit
w_degree = 1
u_degree = 1

pde_model = Electrics.MONODOMAIN  # MONODOMAIN|BIDOMAIN
ode_model = Ionics.FHN  # FHN
mesh_type = Mesh.UNITSQUARE  # PROLATE|BIVENTRICLE|UNITSQUARE
# ----------------------------------------------------------

mesh, dx, _ = Mesh.getMesh(mesh_type=mesh_type)  # [cm]
mesh.coordinates.dat.data[:] *= 1e-2  # [m] to [cm]

time_expression = Constant(t0)

w, wn, w_test = Ionics.getIonicsFunctions(mesh, w_degree, ode_model=ode_model)
u, un, u_test = Electrics.getElectricsFunctions(
    mesh, u_degree, pde_model=pde_model)

f, s, n = Fibers.computeFibers(mesh, mesh_type=mesh_type)

sigma = Electrics.getDiffusion(
    f, s, n, mesh_type=mesh_type, pde_model=pde_model)
Iion = Ionics.getIonicCurrent(ode_model=ode_model)
Iapp = Ionics.getIapp(mesh, mesh_type=mesh_type, R=R)

Fw = Ionics.getODEForm(w, wn, w_test, u, dt, ode_model=ode_model, theta=theta)
Fu = Electrics.getPDEForm(u, un, u_test, w, sigma, Iion,
                          Iapp, time_expression, dt, pde_model=pde_model, IMEX=IMEX)

outfile = File("output/electrophysiology.pvd")

# --- Only for printing, not really necessary for simulation
outfile.write(u, t=t0)
# -----------------------------------------------------------

i = 0
for t in np.arange(t0, tf, dt):
    parprint("Solving time", t)
    time_expression.assign(t)
    # Define and solve ODE
    parprint("Solving ODE")
    solve(Fw == 0, w)
#    solve(Fw == 0, w, solver_parameters={'snes_type': 'ksponly',
#                                            'ksp_type': 'cg',
#                                            'pc_type': 'hypre'})
    wn.assign(w)
    # Define and solve PDE
    parprint("Solving PDE")
    solve(Fu == 0, u, solver_parameters={'snes_type': 'ksponly',
                                         'ksp_type': 'cg',
                                         'pc_type': 'hypre',
                                         'snes_converged_reason': None,
                                         'ksp_converged_reason': None})
    un.assign(u)
    if i % 2 == 0:
        outfile.write(u, t=t)
    i += 1

print("Done")

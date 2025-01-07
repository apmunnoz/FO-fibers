from firedrake import *
import numpy as np
import Fibers
import Mechanics
from EP import *
from ActiveStress import *
parameters['form_compiler']['quadrature_degree'] = 4


# EP setup
mesh_mm = Mesh('../prolate_4mm.msh')  # mm for EP. TODO: FIX
x, y, z = SpatialCoordinate(mesh_mm)
f, s, n = Fibers.generateFibersLV(mesh, 20, 10, 50)

dt_ep = 0.05       # timestep
Nsolves = 20  # hard coded, dt_mech / dt_ms
t_ep = 0.0
ta = 1.0        # time for applying current
Ia = 20         # applied current intensity

u_deg = 1    # Potential FEM degree
w_deg = 2    # Gating variables FEM degree
theta_ep = 0.5     # Theta method for time integration

chi = Constant(1.0)
cm = Constant(1.0)
time_expression_ep = Constant(0.0)


r = 2        # radius for applying current
app_curr = (x+26.8562)*(x+26.8562) + (y+7.91546) * \
    (y+7.91546) + (z-11.0003)*(z-11.0003)

#sigma = 2e-3 * outer(f, f) + 1.3514e-3 * outer(s, s) + 6.757e-4 * outer(n, n)


def Iapp(tt): return Ia * \
    conditional(And(le(sqrt(app_curr), r), le(tt, ta)), 1, 0)


Fu, Fw, u, un, w, wn = getElectrophysiologyForm(
    mesh, f, s, n, Iapp, time_expression_ep, dt, w_degree, u_degree, MONODOMAIN, FHN, theta=theta_ep)

ic = Constant(0.0)
u.assign(ic)
w.assign(ic)
un.assign(ic)
wn.assign(ic)

# Mechanics setup
mesh_lv = Mesh("../prolate_4mm.msh")
mesh_lv.coordinates.dat.data[:] *= 1e-3  # mm to m
f0, s0, n0 = Fibers.generateFibersLV(mesh_lv, 20, 10, 50)  # endo, epi, base

dx = dx(domain=mesh_lv)
ds = Measure('ds', domain=mesh_lv)
dt = 1e-3
t0 = 0.0
tf = 1.0
theta_mech = 1.0
d_degree = 1

dsRob = ds(10)
time_expression = Constant(t0)
AS = ActiveStress(mesh_lv, SS, v)
F, y, yn, ynn = Mechanics.getMechanicsForm(
    mesh_lv, f, s, n, dt, dx, dsRob, time_expression, AS, fe_degree=d_degree, theta=theta_mech)
outfile_mech = File("output/em_mechanics.pvd")


# Initial conditions
outfile_ep = File("output/mono_prolate.pvd")
outfile_ep.write(u, w)
outfile_mech.write(y)


# Solve
for t in np.arange(t0+dt, tf, dt):
    print("Solving time", t, flush=True)

    for i in range(Nsolves):
        print("Solving EP", flush=True)
        t_ep += dt_ep
        time_expression_ep.assign(t_ep)
        solve(Fw == 0, w, solver_parameters={'snes_type': 'newtonls', 'snes_atol': 1e-14, 'snes_rtol': 1e-6, 'snes_max_it': 1000,
              'ksp_type': 'cg', 'pc_type': 'jacobi', 'snes_converged_reason': None})  # Diagonal system, inverse = Jacobi
        wn.assign(w)
        solve(Fu == 0, u, solver_parameters={'snes_rtol': 0.99, 'snes_max_it': 1, 'ksp_type': 'cg',
              'pc_type': 'hypre', 'snes_converged_reason': None, 'ksp_converged_reason': None})
        un.assign(u)

    time_expression.assign(t)
    solve(F == 0, y, solver_parameters={"snes_type": "newtonls",
                                        "snes_monitor": None,
                                        "ksp_type": "gmres",
                                        "ksp_converged_reason": None,
                                        "pc_type": "gamg",
                                        "ksp_gmres_restart": 100,
                                        "snes_atol": 1e-10,
                                        "snes_rtol": 1e-6,
                                        "snes_stol": 0.0,
                                        "ksp_atol": 1e-14,
                                        "ksp_rtol": 1e-6})
    ynn.assign(yn)
    yn.assign(y)
    outfile_mech.write(y)
    outfile_ep.write(u, w)

print("Done")

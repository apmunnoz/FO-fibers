from firedrake import *
import numpy as np
import firdem.physics.Fibers as Fibers
import firdem.utils.Mesh as Mesh
import firdem.physics.Mechanics as Mechanics
import firdem.physics.ActiveStress as AS
from firdem.utils.Printing import parprint
from firdem.utils.Units import SI
from firdem.utils.Parameters import MechanicsParameters
from time import perf_counter as time
parameters['form_compiler']['quadrature_degree'] = 4

mesh_type = Mesh.PROLATE  # PROLATE|BIVENTRICLE|UNITSQUARE
mesh_lv, dx, ds_endo, ds_epi, ds_base = Mesh.getMesh(mesh_type=mesh_type)
mesh_lv.coordinates.dat.data[:] *= 1e-3  # mm to m
dsRob = ds_epi  # Friction at epicardium

f, s, n = Fibers.generateFibersLV(mesh_lv, 20, 10, 50)  # endo, epi, base

# Time discretization
dt = 5e-3
t0 = 0.0
tf = 1.0
theta = 0.5
fe_degree = 1
saveEvery = 2

time_expression = Constant(t0)
V = VectorFunctionSpace(mesh_lv, 'CG', fe_degree)
y = Function(V)
yn = Function(V)
ynn = Function(V)

AS = AS.ActiveStress(mesh_lv, AS.ANALYTICAL, SI)
params = MechanicsParameters(SI)
F = Mechanics.getMechanicsForm(y, yn, ynn, f, s, n, dt, dx,
                               dsRob, time_expression, AS, fe_degree=fe_degree, theta=theta)

problem = NonlinearVariationalProblem(F, y)
solver = NonlinearVariationalSolver(problem, solver_parameters={"snes_type": "newtonls",
                                                                # "snes_monitor": None,
                                                                # "ksp_converged_reason": None,
                                                                "ksp_type": "gmres",
                                                                "ksp_norm_type": "unpreconditioned",
                                                                "pc_type": "gamg",
                                                                # "sub_ksp_type": "preonly",
                                                                # "sub_pc_type": "ilu",
                                                                # "pc_mg_adapt_interp_coarse_space": "gdsw",
                                                                # "pc_mg_galerkin": None,
                                                                # "pc_mg_levels": 2,
                                                                # "mg_levels_pc_type": "asm",
                                                                "snes_lag_preconditioner": -2,
                                                                "snes_lag_preconditioner_persists": False,
                                                                "ksp_gmres_restart": 100,
                                                                "snes_atol": 1e-10,
                                                                "snes_rtol": 1e-6,
                                                                "snes_stol": 0.0,
                                                                "ksp_atol": 1e-14,
                                                                "ksp_rtol": 1e-2})
outfile = File("output/mechanics.pvd")
outfile.write(y)
i = 0
for t in np.arange(t0+dt, tf, dt):
    time_expression.assign(t)
    t0 = time()
    solver.solve()
    tf = time() - t0
    parprint(f"Solved time {t:4.3f} in {tf:4.3e}s")
    ynn.assign(yn)
    yn.assign(y)
    if i % saveEvery == 0:
        outfile.write(y)
    i = i+1

parprint("Done")

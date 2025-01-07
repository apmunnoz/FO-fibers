# Solver for Augmented Lagrangian formulation 
# by Minxia et al.
from firedrake import *
from sys import argv

N = int(argv[1])
theta0 = float(argv[2])  # Angle of initial vector
mesh = UnitSquareMesh(N, N)
gamma = 1e6  # Used for augmented Lagrangian
STAB = 0.0  # Used to add a small regularization

a_left = (0.0, -1.0)
def norm(a): return sqrt(a[0]*a[0]+a[1]*a[1])


a_left = Constant([i/norm(a_left) for i in a_left])

#a_right = (0,1)
a_right = (0.0, 1.0)
a_right = Constant([i/norm(a_right) for i in a_right])


class Mass(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a = -(1 + gamma)**(-1) * inner(test, trial)*dx
        bcs = None
        return (a, bcs)


params_minxia = {
    "snes_max_it":  1000,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-14,
    "ksp_max_it": 1000,
    "ksp_rtol": 1.0e-4,
    "ksp_atol": 1.0e-14,
    "snes_stol":    0.0,
    "snes_monitor": None,
    "snes_linesearch_type": "basic",
    "snes_linesearch_damping": 1.0,
    "snes_linesearch_maxstep": 1.0,
    "snes_converged_reason": None,
}

pbjacobi = {
    "mat_type":    "nest",
    "ksp_type":    "fgmres",
    "pc_type": "fieldsplit",
    "snes_divergence_tolerance": 1.0e10,
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0_ksp_type": "richardson",
    "fieldsplit_0_ksp_max_it": 1,
    "fieldsplit_0_ksp_convergence_test": "skip",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_0_mg_levels_ksp_type": "gmres",
    "fieldsplit_0_mg_levels_ksp_richardson_scale": 1/2,
    "fieldsplit_0_mg_levels_ksp_max_it": 3,
    "fieldsplit_0_mg_levels_ksp_convergence_test": "skip",
    "fieldsplit_0_mg_levels_pc_type": "pbjacobi",
    "fieldsplit_0_mg_coarse_pc_type": "python",
    "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_mg_coarse_assembled_pc_type": "cholesky",
    "fieldsplit_0_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mg_coarse_assembled_mat_mumps_icntl_14": "200",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.Mass",
    "fieldsplit_1_aux_pc_type": "cholesky",
    "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_aux_mat_mumps_icntl_14": "200",
}

params_minxia.update(pbjacobi)

Vvec = VectorFunctionSpace(mesh, 'CG', 2)
Vscl = FunctionSpace(mesh, 'CG', 1)
V = Vvec * Vscl

sol = Function(V)
u, l = split(sol)
v, e = TestFunctions(V)
du, dl = TrialFunctions(V)
n = FacetNormal(mesh)

POT = Constant(gamma/2) * (dot(u, u)-1) * (dot(u, u) - 1) * dx
F = inner(grad(u), grad(v)) * dx + 2 * l * dot(u, v) * dx + \
    e * (dot(u, u) - 1) * dx + derivative(POT, u, v)
J = derivative(F, sol, TrialFunction(V)) - Constant(gamma*2) * \
    inner(dot(u, u)-1, dot(du, v)) * dx
Jp = J
bcs = [DirichletBC(V.sub(0), a_left, 1),
       DirichletBC(V.sub(0), a_right, 2)]

# Uncomment for BCs on all the boundary
# bcs = [DirichletBC(V.sub(0), a_left, 1),
#        DirichletBC(V.sub(0), a_right, 2),
#        DirichletBC(V.sub(0), Constant((1,0)), 3),
#        DirichletBC(V.sub(0), Constant((-1,0)), 4)]
outfile = File("output/oseen-frank-saddle.pvd")

vec_rhs = Constant((cos(theta0), sin(theta0)))
sol.sub(0).interpolate(vec_rhs)
sol.sub(1).interpolate(-acos(dot(a_left, a_right))**2)
print("Done init")
solve(F == 0, sol, bcs=bcs, J=J, Jp=Jp, solver_parameters=params_minxia)
u, l = sol.split()
outfile.write(u, l)

from firedrake import *
from sys import argv

N = int(argv[1])
theta0 = float(argv[2])  # Direction of initial vector
mesh = UnitSquareMesh(N, N)
eta = 1e0  # "Learning rate" ;)
verbose = False

a_left = (0, -1)
def norm(a): return sqrt(a[0]*a[0]+a[1]*a[1])

a_left = Constant([i/norm(a_left) for i in a_left])

a_right = (0, 1)
a_right = Constant([i/norm(a_right) for i in a_right])

params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "hypre",
}

V = VectorFunctionSpace(mesh, 'CG', 1)
if verbose:
    print("Dofs:", V.dim())
u = Function(V, name="sol")
v = TestFunction(V)
bcs = [DirichletBC(V, a_left, 1),
       DirichletBC(V, a_right, 2)]

# Uncomment for BCs everywhere
# bcs = [DirichletBC(V, a_left, 1),
#        DirichletBC(V, a_right, 2),
#        DirichletBC(V, Constant((1,0)), 3),
#        DirichletBC(V, Constant((-1,0)), 4)]

# Set initial condition
u.interpolate(Constant((cos(theta0), sin(theta0))))
u.interpolate(u/sqrt(dot(u, u)))

outfile = File("output/oseen-frank-proj.pvd")

Du = grad(u)
Dv = grad(v)
F = inner(Du, Dv)
F = F * dx
DF = derivative(F, u, TrialFunction(V))
F_err = F - inner(Du, Du) * dot(u, v) * dx


# Homogenize BCs for have 0 BC for increment
for b in bcs:
    b.apply(u)
for b in bcs:
    b.homogenize()

dduu = Function(V)
dF = assemble(F_err, bcs=bcs)

# Set up solver
prob = LinearVariationalProblem(DF, -F_err, dduu, bcs=bcs)
solver = LinearVariationalSolver(prob, solver_parameters=params)

err = sqrt(assemble(dot(dF, dF) * dx))
err0 = err
it = 0
if verbose:
    print("It: {:4.0f}".format(0), err/err0)
while err > 1e-14 and err/err0 > 1e-8 and it < 1000:
    solver.solve()
    u.interpolate(u + eta * dduu)
    u.interpolate(u/sqrt(dot(u, u)))

    err = solver.snes.ksp.getRhs().norm()
    if verbose:
        print("It: {:4.0f}\tG err={:.4e}\tG err_rel={:.4e}".format(
            it, err, err/err0))
    err = min(err, err/err0)  # Coarse error control
    it += 1
outfile.write(u)
print("Done in {:4.0f} iterations.".format(it))

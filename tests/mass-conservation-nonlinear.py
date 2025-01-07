from firedrake import *
from firdem.physics.Constitutive import ConstitutiveModel
from firdem.physics.Perfusion import getNonlinearPerfusionMixedP

# Physical parameters
phi0 = 0.1
K = 1
ks = 1
beta = 10 # exchange coeff
p_ao = 2 # aortic bed pressure

# Discretization parameters
nx = ny = 100
Lx = Ly = 1
dt = 1e-2
tf = 1e-0

# Discrete setting
mesh = RectangleMesh(nx, ny, Lx, Ly)
Vphi = FunctionSpace(mesh, 'CG', 1)
Vmu = FunctionSpace(mesh, 'CG', 1)
V = Vphi * Vmu

# Functions
sol = Function(V)
sol.subfunctions[0].rename("phi")
sol.subfunctions[1].rename("mu")
sol_n = Function(V)
phi, mu = split(sol)
phi_n, _ = split(sol_n)
q, eta = TestFunctions(V)

# Initialize
sol.subfunctions[0].assign(Constant(phi0))
sol_n.assign(sol)

# Set up problem
model = ConstitutiveModel()
vphi = variable(phi)
vphi0 = variable(phi0)
model.addHelmholtz(0.5 * Constant(ks) * (vphi - vphi0) ** 2)
idt = Constant(1/dt)

model.normalizePressure(vphi, vphi0, Constant(1), Constant(1), 1) # The Constants are dummy values
p = model.getPressure(vphi)
sol.subfunctions[1].interpolate(p) # Initialize for visualization
dphidt = idt * (phi - phi_n)
source = -beta * (mu - p_ao)
FF = getNonlinearPerfusionMixedP(phi, idt * q, mu, eta, dphidt, K, p, source)
bcs = DirichletBC(V.sub(1), Constant(1), [1,3])

file = File("output/mass-nonlinear.pvd")
out = sol.subfunctions
file.write(*out)
for t in np.arange(dt, tf, dt):
    print(f"Solving instant t={t:1.4f}")
    solve(FF == 0, sol, bcs=bcs, solver_parameters={"ksp_type": "preonly", 
                                        "pc_type": "lu", 
                                        "pc_factor_mat_solver_type": "mumps"})
    sol_n.assign(sol)
    file.write(*out)
print("Done")

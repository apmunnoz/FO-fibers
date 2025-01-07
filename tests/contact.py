from firedrake import *
from firdem import Contact
from firdem import Mechanics

from firedrake.ufl_expr import FacetNormal,derivative
import numpy as np
from mpi4py import MPI
parameters['form_compiler']['quadrature_degree'] = 4

# Sim params
STAB = 0.0
step0 = 0.0
factr = 5
#max_force = 500
delta = 1.0e-3
epsilon = 1e6 # penalty parameter (adaptive to distance from gap function)


msh = BoxMesh(20, 20, 5, 0.08, 0.08, 0.02)
CONTACT_TAG = 6
BC_TAG = 5
msh.coordinates.dat.data[:] += [-0.04, -0.04, 0]

comm = MPI.COMM_WORLD
dim = msh.geometric_dimension()
x = SpatialCoordinate(msh)
dx = Measure("dx", msh)
ds = Measure("ds", msh)

# Define parameters
iter_coef = Constant(0)
tol = 5e-5
c1 = 75/(2*(1+0.499))*1e3; c2 = 500e3


F = Mechanics.getMechanicsForm(y, yn, ynn, f, s, n, dt, dx,
                               dsRob, time_expression, AS, fe_degree=fe_degree, theta=theta)


# Define material
def NeoHookean(u,c1,c2):
    dim = len(u)                                # Dimensions of the problem
    I = Identity(dim)                           # Identity tensor
    F = I + grad(u)                             # Deformation gradient
    C = F.T*F                                   # Right Cauchy-Green tensor
    E = 1/2*(C-I)                               # Green-Lagrange strain tensor

    J  = det(F)                                 # Volume ratio
    Fiso = J**(-1/3)*F                          # Isochoric deformation gradient
    Ciso = J**(-2/3)*C                          # Isochoric right Cauchy-Green tensor
    Eiso = 1/2*(Ciso - I)                       # Isochoric Green-Lagrange strain tensor
    Iiso = sum(Ciso[i,i] for i in range(dim))   # First invariant
    
    psi = c1/2*(Iiso-3) + c2/2*(ln(J))**2       # Strain energy function
    
    return psi,J,E

# TODO FROM HERE

##################################################
# Function spaces and definitions
V_scalar = FunctionSpace(msh, "CG", 1)
V_vector = VectorFunctionSpace(msh, "CG",2)

# Define functions
u = Function(V_vector,name='Displacement')
u.interpolate( Constant(tuple([0]*dim)))
gVal = Function(V_scalar)
gVal.interpolate(Constant(delta))
v = TestFunction(V_vector)

# Set Dirichlet BCs
bcs = [DirichletBC(V_vector, Constant(tuple([0]*dim)), BC_TAG)]

# Define material model
psi,J,E = NeoHookean(u,c1,c2)

# Define energy function
delta_const = Constant(delta)
Pi = psi*dx + iter_coef*epsilon*g(x+u,delta_const)**2*ds(CONTACT_TAG) + 1/2*epsilon*g(x+u,delta_const)**2*ds(CONTACT_TAG)

FNewton = derivative(Pi, u, v)
du = TrialFunction(V_vector)
J = derivative(FNewton, u, du) + Constant(STAB) * inner(grad(du), grad(v)) * dx

x = msh.coordinates
xfun = Function(V_vector)
xfun.interpolate(cpFun(x, delta)[0])

jj0 = step0/factr # So that step0 is the first one
jj = jj0
ii = 1
outfile = File("output/contact.pvd")
outfile.write(u, xfun, t=0)
def update_iter(current_solution):
    global ii, jj
    iter_coef.assign(jj)
    G = interpolate(g(x+u, delta_const), V_scalar)
    err = G.vector().get_local().max()

    if err > 5e-5:
        #jj = min(jj*factr , max_force)
        jj = jj + factr
        #print("UP", err, jj)
    outfile.write(u, xfun, t=ii)
    ii += 1
    #print("Ramp:", jj)

problem = NonlinearVariationalProblem(FNewton, u, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem, solver_parameters=params, pre_jacobian_callback=update_iter)

deltas = np.linspace(1e-3, 2e-3, 41)
# Define derivative of energy function
for delta in deltas:
    print("delta:", delta, "jj:", jj)
    delta_const.assign(delta)
    solver.solve()
    #iter_coef.assign(jj0)
    #jj = jj0

outfile.write(u, xfun, t=ii)


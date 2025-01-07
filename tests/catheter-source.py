from firedrake import RectangleMesh, as_tensor, Identity, FunctionSpace, SpatialCoordinate, DirichletBC, conditional, And, lt, gt, Constant, File, TestFunction, Function, ds
from firdem.physics.EMSource import getSource

nx = 50
ny = 50
lx = 0.04
ly = 0.02
mesh = RectangleMesh(nx,ny,lx,ly)
Pobj = 30
g0 = 1
regularization = 1e-10
gamma = 0.54
Romega = Constant(1.3) # Robin at sides

V = FunctionSpace(mesh, 'CG', 1)
x, y = SpatialCoordinate(mesh)
index = conditional(And(lt(x, lx/2 + 0.00233/2), gt(x, lx/2 - 0.00233/2)), 1.0, 0.0)
g_idx = 4
bc = DirichletBC(V, Constant(0.0), 3)
bcs = None

phi = Function(V, name="potential")
phi_test = TestFunction(V)

dsRobin = ds(1) + ds(2)
g = getSource(phi, phi_test, bcs, Pobj, gamma, g0, index, g_idx, Rrobin=Romega, dsRobin=dsRobin,
                       regularization=regularization, verbose=True)
outfile = File("output/EM.pvd")
outfile.write(phi)

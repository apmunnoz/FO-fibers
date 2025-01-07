from firedrake import *
from firedrake.output import VTKFile
from firdem.physics.Heat import getBioheatForm
import numpy as np

# Discretization parameters
nx = 100
ny = 10
lx = 0.03
ly = 0.01
mesh = RectangleMesh(nx,ny,lx,ly)
dt = 0.1
tf = 10
time = Constant(0)

# Functional setting
V = FunctionSpace(mesh, 'CG', 1)
T = Function(V, name="T")
Tn = Function(V, name="T")
q = TestFunction(V)
bcs = DirichletBC(V, Constant(0), 2)

idt = Constant(1/dt)
dTdt = idt * (T - Tn)

u_blood = Constant((1,0))
K = lambda T: 1e-5
rhof = 1e3
c = lambda T: 1

x,y = mesh.coordinates
def Min(a,b):
    return conditional(le(a,b), a, b)
source = Min(time, 1) * conditional(le(x,lx/2), -1, 0)

F = getBioheatForm(T, dTdt, q, c, u_blood, K, rhof, source=source)

file = VTKFile("output/bioheat.pvd")
file.write(T, t=0)
for t in np.arange(dt, tf, dt):
    time.assign(t)
    print("Solving time t={:2.3f}".format(t))
    solve(F==0, T, bcs=bcs)
    Tn.assign(T)
    file.write(T, t=t)

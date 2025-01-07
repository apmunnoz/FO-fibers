from firedrake import *
from firdem.physics.Heat import getNonlinearHeatForm
from firdem.physics.Constitutive import ConstitutiveModel
import numpy as np

# Discretization
nx = ny = 40
Lx = Ly = 1e-2
dt = 5e-1
tf = 5e1
mesh = RectangleMesh(nx, ny, Lx, Ly)
V = FunctionSpace(mesh, 'CG', 2)

T = Function(V, name="T")
T_n = T.copy(True)
qT = TestFunction(V)

# Parameters
T0 = Constant(273 + 37)
T.assign(T0)
T_n.assign(T)
KQ = Constant(0.518)
rhof = Constant(1020)
phi = Constant(0.1)
ecb = Constant(0.5 * 0.8 * 3617)
cT = Constant(3454)

# Modeling
model = ConstitutiveModel()
vT = variable(T)
model.addHelmholtz(rhof * cT * (vT - T0 - vT * ln(vT/T0)) )
model.addGibbs(-ecb * (T0 - vT)**2)

x,y = mesh.coordinates
p = -5e-2 * x
Vf = - grad(p)
bcs = DirichletBC(V, T0, [1,3,4])

source = conditional(le(x, Lx/2), Constant(1e8), 0.0)

Ss = model.getSolidEnthropy(vT)
Ss_n = replace(Ss, {T: T_n})
sf = model.getFluidEnthropy(vT)
sf_n = replace(sf, {vT: T_n})

# Forms
idt = Constant(1/dt)
dSsdt = idt * (Ss - Ss_n)
drhophisfdt = rhof * phi * idt * (sf - sf_n)
F = getNonlinearHeatForm(T, qT, dSsdt, drhophisfdt, p, sf, Vf, KQ, rhof, source=source)

file = File("output/heat-nonlinear.pvd")
file.write(T, t=0)
for t in np.arange(dt, tf, dt):
    print(f"Solving t={t}")
    solve(F==0, T, bcs=bcs, solver_parameters={"snes_monitor": None, "ksp_type": "preonly"})
    T_n.assign(T)
    file.write(T, t=t)
print("Done")

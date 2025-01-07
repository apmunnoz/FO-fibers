from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from firdem.physics.EMSource import getSource
from firdem.physics.Heat import getBioheatForm
parameters['form_compiler']['quadrature_degree'] = 6
print = PETSc.Sys.Print


# Discretization parameters
dim = 2
u_blood = None
if dim == 2:
    nx = 100
    ny = int(nx/4)
    lx = 0.08
    ly = 0.02
    mesh = RectangleMesh(nx,ny,lx,ly)
    x, y = SpatialCoordinate(mesh)
    T_robin = 4 # tag for Robin in T
    u_blood = Constant((0,0))
else:
    nx = ny = 40
    nz = int(nx)
    lx = ly = 0.08
    lz = 0.02
    mesh = BoxMesh(nx,ny,nz,lx,ly,lz)
    x, y, z = SpatialCoordinate(mesh)
    T_robin = 6 # tag for Robin in T
    u_blood = Constant((0,0,0))

dt = 0.1
tf = 30
time = Constant(0)

# Functional setting for T, Phi

## T
VT = FunctionSpace(mesh, 'CG', 1)
T = Function(VT, name="T")
T_n = Function(VT, name="T")
qT = TestFunction(VT)

## Phi
VPhi = FunctionSpace(mesh, 'CG', 1)
Phi = Function(VPhi, name="Phi")
qPhi = TestFunction(VPhi)

# Physical parameters

## T is in K
T0 = Constant(37+273)
Tb = Constant(37+273)
Tc = T - 273 # Variable in Celsius
Tsaline = Constant(29+273)
KQ = lambda _T: Tc * (0.5655 + 5.034e-12 * exp(0.263*Tc))
rhof = Constant(1020) # Blood
rhot = Constant(1000) # Tissue

# THIS TERM is the culprit (apparently) for divergence at 90 C
# To avoid this, we linearized the exponential
#rhocT = 1e6 * (3.643 - 0.003533 * exp(0.06263 * Tc))
rhocT = lambda _T: 1e6 / rhot * (3.643 - 0.003533 * (1 + 0.06263 * (_T-273)))

he = Constant(2550) # Robin at top
hb = Constant(4438) # Robin at top inter catheter

## Phi
Pobj = 5 if dim == 3 else 1e4 # More power for 2D
g0 = 1
regularization = 0.0
gamma = 0.54 * (1 + 0.015*(Tc - 37))
Romega = Constant(1.3) # Robin at sides

# Initial values
T.assign(T0)
T_n.assign(T0)

# BCs
T_idx = [1,2,3] if dim==2 else [1,2,3,4,5]
bcT = DirichletBC(VT, Constant(T0), T_idx)
bcPhi = None

# Catheter surface
index = None
if dim==2:
    index = conditional(And(lt(x, lx/2+0.00233/2), gt(x, lx/2-0.00233/2)), 1.0, 0.0)
    g_idx = 4
else: # dim==3  
    cond_x = And(lt(x, lx/2+0.00233/2), gt(x, lx/2-0.00233/2))
    cond_y = And(lt(y, ly/2+0.00233/2), gt(y, ly/2-0.00233/2))
    index = conditional(And(cond_x,cond_y), 1.0, 0.0)
    g_idx = 6


# Weak formulation

## T
source = gamma * grad(Phi)**2 # Catheter heating
idt = Constant(1/dt)
dTdt = idt * (T - T_n)

FT = getBioheatForm(T, dTdt, qT, rhocT, u_blood, KQ, rhot, source=source)
# Add Robin coefficients
FT += hb * (T - T0) * (1-index) * qT * ds(T_robin) + he * (T - Tsaline) * index * qT * ds(T_robin)

file = VTKFile(f"output/bioheat-catheter-{dim}D.pvd")
TC = Function(T, name="Tc")
TC.interpolate(Tc)
file.write(TC, Phi, time=0)
i = 1
dsRobin = None # For EM source
if dim==2:
    dsRobin = ds(1) + ds(2)
else:
    dsRobin = ds(1) + ds(2) + ds(3) + ds(4)
g0 = getSource(Phi, qPhi, bcPhi, Pobj, gamma, g0, index,
                   g_idx, Rrobin=Romega, dsRobin=dsRobin, regularization=regularization, verbose=True)
for t in np.arange(dt, tf, dt):
    time.assign(t)
    print("Solving time t={:2.3f}".format(t))
    # Recompute field every second. Keep still for now
    #if i % 10 == 0: 
        #g0 = getSource(Phi, qPhi, bcPhi, Pobj, gamma, g0, index,
                       #g_idx, Rrobin=Romega, dsRobin=dsRobin, regularization=regularization, verbose=True)
    
    solve(FT==0, T, bcs=bcT, solver_parameters={"ksp_type": "gmres", 
                                                "pc_type": "hypre",
                                                "snes_type": "newtonls", 
                                                "snes_monitor": None,
                                                "snes_stol": 0.0, 
                                                "snes_atol": 1e-12, 
                                                "snes_rtol": 1e-6, 
                                                #"snes_max_it": 10000, 
                                                "snes_converged_reason": None})
    T_n.assign(T)
    TC.interpolate(Tc)
    file.write(TC, Phi, time=t)
    i += 1
print("Done")

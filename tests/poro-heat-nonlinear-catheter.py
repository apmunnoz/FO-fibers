from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from firdem.physics.EMSource import getSource
from firdem.physics.Heat import getNonlinearHeatForm
from firdem.physics.Constitutive import ConstitutiveModel
from firdem.physics.Perfusion import getNonlinearPerfusionMixedP
#from firdem.utils.PressureFunction import getPressureTemperatureFunction
parameters['form_compiler']['quadrature_degree'] = 6
print = PETSc.Sys.Print


# Discretization parameters
dim = 3
if dim == 2:
    nx = 100
    ny = int(nx/4)
    lx = 0.08
    ly = 0.02
    mesh = RectangleMesh(nx,ny,lx,ly)
    x, y = SpatialCoordinate(mesh)
    T_robin = 4 # tag for Robin in T
else:
    nx = ny = 40
    nz = int(nx)
    lx = ly = 0.08
    lz = 0.02
    mesh = BoxMesh(nx,ny,nz,lx,ly,lz)
    x, y, z = SpatialCoordinate(mesh)
    T_robin = 6 # tag for Robin in T

dt = 0.1
tf = 30
time = Constant(0)

# Cauchy distribution parameters
c_gamma = 0.7
c_x0 = 36.54

# Functional setting for T, Phi, mu, phi

## T
VT = FunctionSpace(mesh, 'CG', 1)
T = Function(VT, name="T")
T_n = Function(VT, name="T")
qT = TestFunction(VT)

## Phi
VPhi = FunctionSpace(mesh, 'CG', 1)
Phi = Function(VPhi, name="Phi")
qPhi = TestFunction(VPhi)

## mu, phi
Vmu = FunctionSpace(mesh, 'CG', 1)
Vphi = FunctionSpace(mesh, 'CG', 1)
Vmuphi = Vmu*Vphi
muphi = Function(Vmuphi)
muphi.subfunctions[0].rename("mu")
muphi.subfunctions[1].rename("phi")
mu,phi = split(muphi)
muphi_n = muphi.copy(True)
_, phi_n = split(muphi_n)
qmu, qphi = TestFunctions(Vmuphi)


# Physical parameters

## T is in K
T0 = Constant(37+273)
Tb = Constant(37+273)
Tc = T - 273 # Variable in Celsius
Tsaline = Constant(29+273)
KQ = Tc * (0.5655 + 5.034e-12 * exp(0.263*Tc))
rhof = Constant(1020)
cf = Constant(3617)

# THIS TERM is the culprit (apparently) for divergence at 90 C
# To avoid this, we linearized the exponential
#rhocT = 1e6 * (3.643 - 0.003533 * exp(0.06263 * Tc))
rhocT = 1e6 * (3.643 - 0.003533 * (1 + 0.06263 * Tc))

he = Constant(2550) # Robin at top
hb = Constant(4438) # Robin at top inter catheter

## Phi
Pobj = 5 if dim == 3 else 1e4 # More power for 2D
g0 = 1
regularization = 0.0
gamma = 0.54 * (1 + 0.015*(Tc - 37))
Romega = Constant(1.3) # Robin at sides

## Porous media
phi0 = 0.1
p_ao = 2.7e3
p_ref = 2.7e3 
p_vn = 1.3e3
beta_aovn = 3.5e-5
q1 = 1.33 
q2 = 550
q3 = 45 
Kphi = 1e-7

# Initial values
T.assign(T0)
T_n.assign(T0)
muphi.subfunctions[0].assign(p_ref)
muphi.subfunctions[1].assign(phi0)
muphi_n.subfunctions[0].assign(p_ref)
muphi_n.subfunctions[1].assign(phi0)

# BCs
T_idx = [1,2,3] if dim==2 else [1,2,3,4,5]
bcT = DirichletBC(VT, Constant(T0), T_idx)
bcPhi = None
muphi_idx = [1,2] if dim==2 else [1,2,3,4]
bcMuphi = DirichletBC(Vmuphi.sub(0), Constant(p_ref), muphi_idx)

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


# Constitutive modeling
model = ConstitutiveModel()
vT = variable(T)
vphi = variable(phi)

## Enthropy terms
model.addHelmholtz(rhocT * (vT - T0 - (vT-273) * ln((vT-273)/(T0-273)) ))

## Vessel pressure law
model.addHelmholtz(q1/q3 * exp(q3*vphi) + q2 * vphi * (ln(q3 * vphi) - 1))

Ss = model.getSolidEnthropy(vT)
Ss_n = replace(Ss, {T: T_n, phi: phi_n})
p = model.getPressure(vphi)

## Water vapour pressure
## see wiki
p_T_fun = 0.61094 * exp(17.625 * Tc / (Tc + 243.04)) * 1e3 * phi0
Tc0 = T0-273 
p_T_fun0 = 0.61094 * exp(17.625 * Tc0 / (Tc0 + 243.04)) * 1e3 * phi0
p = p + p_T_fun - p_T_fun0

vphi0 = variable(phi0)
model.normalizePressure(vphi, vphi0, Constant(1), Constant(1), p_ref)

## Compatible Gibbs
vp = variable(p)
#model.addGibbs(cf * (Tb - vT - (vT-273) * ln((Tb-273)/(vT-273))) + (vp - p_ref) / rhof)
# Modified Gibbs potential from Scientific Reports enthropy function
model.addGibbs(pi * (c_gamma * Tc + 1/(3*c_gamma) * (Tc - c_x0)**3) + (vp - p_ref) / rhof)
sf = model.getFluidEnthropy(vT)
sf_n = replace(sf, {T: T_n, phi: phi_n})



# Weak formulation
idt = Constant(1/dt)

## T
Vf = -Kphi * grad(mu)
source = gamma * grad(Phi)**2 # Catheter heating
dSsdt = idt * (Ss - Ss_n)
#dSsdt = idt * rhocT * (T - T_n) / T
drhophisfdt = rhof * idt * (sf*phi - sf_n*phi_n)
FT = getNonlinearHeatForm(T, qT, dSsdt, drhophisfdt, mu, sf, Vf, KQ, rhof, source=source)
# Add Robin coefficients
FT += hb * (T - T0) * (1-index) * qT * ds(T_robin) + he * (T - Tsaline) * index * qT * ds(T_robin)

## Perfusion
dvarphidt = idt * (phi - phi_n)
sourcePerf = - beta_aovn * (mu - p_ao) - beta_aovn * (mu - p_vn)
sourcePerf = None
Fmuphi = getNonlinearPerfusionMixedP(phi, idt * qphi, mu, qmu, dvarphidt, Kphi, p, source=sourcePerf)

file = VTKFile(f"output/poro-heat-catheter-{dim}D.pvd")
out = muphi.subfunctions
TC = Function(T, name="Tc")
TC.interpolate(Tc)
file.write(TC, Phi, *out, time=0)
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
    solve(Fmuphi==0, muphi, bcs=bcMuphi, solver_parameters={"ksp_type": "gmres", 
                                                            "pc_type": "hypre",
                                                            #"snes_monitor": None,
                                                            "snes_stol": 0.0, 
                                                            "snes_atol": 1e-14, 
                                                            "snes_rtol": 1e-6, 
                                                            "snes_type": "newtonls",
                                                            "snes_linesearch_type": "none", 
                                                            "snes_converged_reason": None})
    T_n.assign(T)
    muphi_n.assign(muphi)
    TC.interpolate(Tc)
    file.write(TC, Phi, *out, time=t)
    i += 1
print("Done")

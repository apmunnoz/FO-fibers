from firedrake import *
from firdem.utils.Units import Units
from firdem.utils.Parameters import MechanicsParameters
from firdem.utils.Parameters import NEOHOOKEAN, MOONEYRIVLIN, USYK, GUCCIONE, HOLZAPFELOGDEN


# Default unit system
SI = Units()

def computeEnergy(F, f0, s0, n0, parameters):
    model = parameters.mechanicsModel
    C = F.T * F
    E = 0.5*(C - Identity(3))

    if model == NEOHOOKEAN: 
        C1 = parameters.C1
        I1 = tr(C)
        WP = Cg * (I1 - 3)
    elif model == MOONEYRIVLIN:
        C1 = parameters.C1
        C2 = parameters.C2
        I1 = tr(C)
        I2 = 0.5 * (tr(C)**2 - tr(C*C))
        WP =  C1 * (I1 - 3) + C2 * (I2 - 3)
    elif model == USYK:
        # Usyk,. mc Culloch 2002
        Cg = parameters.Cg       # [Pa]
        bf = parameters.bf       # [-]
        bs = parameters.bs       # [-]
        bn = parameters.bn       # [-]
        bfs = parameters.bfs      # [-]
        bfn = parameters.bfn       # [-]
        bsn = parameters.bsn       # [-]
        Eff, Efs, Efn = inner(E*f0, f0), inner(E*f0, s0), inner(E*f0, n0)
        Esf, Ess, Esn = inner(E*s0, f0), inner(E*s0, s0), inner(E*s0, n0)
        Enf, Ens, Enn = inner(E*n0, f0), inner(E*n0, s0), inner(E*n0, n0)

        Q = Constant(bf) * Eff**2 \
            + Constant(bs) * Ess**2 \
            + Constant(bn) * Enn**2 \
            + Constant(bfs) * 2.0 * Efs**2 \
            + Constant(bfn) * 2.0 * Efn**2 \
            + Constant(bsn) * 2.0 * Esn**2
        WP = 0.5*Constant(Cg)*(exp(Q)-1)
    if model == GUCCIONE:
        # Guccione
        Cg = parameters.Cg       # [Pa]
        bff = parameters.bff     # [-]
        bft = parameters.bft     # [-]
        btt = parameters.btt     # [-]
        Eff, Efs, Efn = inner(E*f0, f0), inner(E*f0, s0), inner(E*f0, n0)
        Esf, Ess, Esn = inner(E*s0, f0), inner(E*s0, s0), inner(E*s0, n0)
        Enf, Ens, Enn = inner(E*n0, f0), inner(E*n0, s0), inner(E*n0, n0)

        Q = Constant(bff) * Eff**2 \
            + 2 * Constant(bft) * (Efs**2 + Efn**2) \
            + Constant(btt) * (Ess**2 + Enn**2 + 2*Esn**2)
        WP = 0.5*Constant(Cg)*(exp(Q)-1)
    elif model == HOLZAPFELOGDEN:
        assert False, "HO model not implemented yet"
    return WP
      

def computePiola(_F, f0, s0, n0, parameters):
    F = variable(_F)
    J = det(F)
    Fbar=J**(-1./3.) * F

    WP = computeEnergy(Fbar, f0, s0, n0, parameters)
    k = parameters.k 
    WV = Constant(k)/2*(J-1)*ln(J)

    W = WP + WV

    return diff(W, F)


def getActive(F, f0, t, AS):
    C = F.T*F
    I4f = dot(f0, C*f0)
    Ta = AS.getTa(t)
    Pa = Ta*outer(F*f0, f0)/sqrt(I4f)
    return Pa

def getMechanicsForm(y, yn, ynn, f0, s0, n0, dt, dx, dSRob, time_expression, AS, fe_degree=1, theta=1.0, parameters=MechanicsParameters(SI), units=SI, model=GUCCIONE, static=False):

    ws = TestFunction(y.function_space())
    mesh = y.function_space().mesh()

    # Auxiliary variables
    idt = Constant(1/dt)
    dt = Constant(dt)
    us = idt * (y - yn)
    usn = idt * (yn - ynn)
    rhos = parameters.rhos

    F = Identity(3) + grad(y)
    Fn = Identity(3) + grad(yn)
    
    P = computePiola(F, f0, s0, n0, parameters) + getActive(F, f0, time_expression, AS)
    Pn = computePiola(Fn, f0, s0, n0, parameters) + getActive(Fn, f0, time_expression - dt, AS)

    # Pfaller et al.
    k_perp = Constant(2e5 * units.Pa / units.L )  # [Pa/m]
    c_perp = Constant(5e3 * units.Pa * units.S / units.L)  # [Pa*s/m]

    n = FacetNormal(mesh)
    rhos = Constant(rhos)
    ts_robin = -outer(n, n)*(k_perp*y + c_perp*us) - (Identity(3) - outer(n, n))*k_perp/10*y
    ts_robin_n = -outer(n, n)*(k_perp*yn + c_perp*usn) - (Identity(3) - outer(n, n))*k_perp/10*yn
    F_time = rhos * idt * dot(us - usn, ws) * dx
    Fs = inner(P, grad(ws))*dx - dot(ts_robin, ws)*dSRob
    Fsn = inner(Pn, grad(ws))*dx - dot(ts_robin_n, ws)*dSRob 

    if static: 
        return Fs
    else:
        Fout = F_time + Constant(theta) * Fs + Constant(1-theta) * Fsn  # Simple midpoint time integration 
        return Fout

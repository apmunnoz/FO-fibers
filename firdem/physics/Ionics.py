from firedrake import *
from Units import Units, SI
from Parameters import IonicsParameters
from Mesh import PROLATE, BIVENTRICLE, UNITSQUARE
from IonicsAux import TTPModel

# Names for model types
FHN = 8
TTP = 9

def getFunctionalSpace(mesh, w_degree, ode_model=FHN, space='CG'):
    """
    Generate the appropriate functional space for 
    the variational problem associated with the
    ionic model.
    """
    if ode_model == FHN:
        return FunctionSpace(mesh, space, w_degree) # We could also use DG
    elif ode_model == TTP:
        V = FunctionSpace(mesh, space, w_degree)
        return V * V * V * V * V * V * V * V * V * V * V * V * V * V * V * V * V * V
        #return VectorFunctionSpace(mesh, 'CG', w_degree, dim=18)
    else:
        assert False, "Ionic model {} not implemented yet".format(ode_model)

    
def getFunctions(V, ode_model, units):

    w = Function(V, name = "Gating")
    wn = Function(V, name = "Gating")
    w_test = TestFunction(V)

    if ode_model==TTP:
        ttp = TTPModel(units)
        Ca_iT, Ca_ssT, Ca_srT, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar = ttp.unpackGatingSubfunctions(w)
        
        Ca_iT.interpolate(Constant(ttp.Ca_ifun(0.000126 * units.mM)))
        Ca_srT.interpolate(Constant(ttp.Ca_srfun(3.64 * units.mM)))
        Ca_ssT.interpolate(Constant(ttp.Ca_ssfun(0.00036 * units.mM)))
        Na_i.interpolate(Constant(8.604 * units.mM))
        K_i.interpolate(Constant(136.89 * units.mM))
        m.interpolate(Constant(0.00172))
        h.interpolate(Constant(0.744))
        j.interpolate(Constant(0.7045))
        xr1.interpolate(Constant(0.00621))
        xr2.interpolate(Constant(0.4712))
        xs.interpolate(Constant(0.0095))
        r.interpolate(Constant(2.42e-8))
        s.interpolate(Constant(0.999998))
        d.interpolate(Constant(3.373e-5))
        f.interpolate(Constant(0.7888))
        f2.interpolate(Constant(0.9755))
        fcass.interpolate(Constant(0.9953))
        Rbar.interpolate(Constant(0.9073))
    wn.assign(w)
    return w, wn, w_test
                

def getIapp(mesh, tt, mesh_type, units):              
    """
    Applied current function 
        Iapp(x,t<ta)
    """  
    parameters = IonicsParameters(units)
    X = SpatialCoordinate(mesh)
    r = None
    if mesh_type == PROLATE:
        R=2
        p0 = 1e-3 * as_vector([-26.8562, -7.91546, 11.0003]) # [mm] -> [m]
        p0 = units.L * p0    # center 
        r  = R * (1e-3 * units.L)    # radius [mm]
    elif mesh_type == BIVENTRICLE:
        R=2
        p0 = 1e-3 * as_vector([103.724, -29.8749, -36.5508]) # [mm] -> [m]
        p0 = units.L * p0    # center
        r  = R * (1e-3 * units.L)    # radius [mm]
    elif mesh_type == UNITSQUARE:
        R=0.1
        p0 = 1e-3 * as_vector([0, 0])
        p0 = units.L * p0
        r  = 1e-3 * units.L  
    Xapp = X - p0    # center 
    app_curr = dot(Xapp, Xapp)
    Iapp = parameters.Ia * conditional(And(le( sqrt(app_curr), r ), le(tt, parameters.ta)), 1, 0)
    return Iapp


def getIonicCurrent(v, w, ode_model, units):
    """
    Return ionic current function.
    """
    if ode_model == FHN:
        b = 5 * (1e-3 * units.A) / (1e-2 * units.L)**2 / (1e-3 * units.Volt)**3  # [mA / cm^2 / mV^3]
        c = 0.1 * units.mV    # milliVolt
        delta = 1 * units.mV   # [mV]
        beta = 1 * (1e-3 * units.A) / (1e-2 * units.L)**2   # [mA / cm^2]
        Iion = -b * v * (v - c) * (delta - v) + beta * w
    else: # TTP
        ttp = TTPModel(units)
        Iion = ttp.I_ion(v, w)

    return Iion

def getResidualFunction(Iapp, ode_model, units):
    if ode_model == FHN:
        eta = 0.1 / (1e-3 * units.Volt) / (1e-3 * units.S)  # [1/(mV ms)]
        gamma = 0.025 / (1e-3 * units.S)  # [1/ms]
        G = lambda vv, ww: eta * vv - gamma * ww
    elif ode_model == TTP: # TTP
        ttp = TTPModel(units)
        G = lambda vv, ww: ttp.getGatingRHS(vv, ww, Iapp)

    return G
   
    
def getResidual(w, wn, w_test, u, dt, Iapp, ode_model, units):
    """
    Set the variational form of the ode model.
    Allow the user to choose the time splitting for
    its solution in time (theta-method).
        theta = 1:  implicit Euler
        theta = 0:  explicit Euler 
    """
    if u.ufl_shape == ():
        v = u
    else:
        v = u[0]-u[1] # ui - ue

    G = getResidualFunction(Iapp, ode_model, units)
    idt = Constant(1/dt)
    FF = idt * inner(w - wn,  w_test) * dx - dot(G(v, w), w_test) * dx    
    return FF   
    
        
        

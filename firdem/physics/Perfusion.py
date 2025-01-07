from firedrake import *
from firdem.utils.Units import Units
from firdem.utils.Parameters import PerfusionParameters


def getLinearPerfusionForm(p, u, Vmixed, parameters):

    V = Vmixed # shorthand
    v, q = TestFunctions(V)

    # Parameters
    beta_in = parameters.gamma
    beta_out = parameters.gamma
    beta = beta_in + beta_out
    K = parameters.perm
    p_art = parameters.p_art
    p_ven = parameters.p_ven
    rhof = parameters.rhof
    muf = parameters.muf
    invK = inv(K)

    # Gravity
    g = 9.8 * parameters.units.L / parameters.units.S**2
    Tb = parameters.T_blood

    eps_u = sym(grad(u))
    eps_v = sym(grad(v))

    # Mass conservation
    F1 = div(u) * q * dx + beta * p * q * dx- beta_in * p_art * q * dx - beta_out * p_ven * q * dx
    F2 = dot(invK * u, v) * dx + muf * inner(eps_u, eps_v) * dx - p * div(v) * dx
    F3 = rhof * g * (T - Tb) * v[2] * dx # Buoyancy
    return F1 + F2 + F3


def getNonlinearPerfusionMixedP(varphi, q, mu, eta, dvarphidt, K, p, source=None):
    """
    Get the nonlinear porous media equations:
        d varphi / dt - div K grad mu = source
                               mu - p = 0

    Problem variables: varphi, mu

    Parameters:
        - varphi: Lagrangian porosity
        - q: porosity test function
        - mu: Auxiliary variable for the pressure
        - eta: Auxiliary test function
        - dvarphidt: Time derivative of the Lagrangian porosity
        - K: Permeability
        - p: pressure function
        - source: External source term

    Output: Residual (UFL)

    Note: To obtain a symmetric problem, rescale q with q -> 1/dt * q
    """
    
    F = dvarphidt * eta * dx + dot(K * grad(mu), grad(eta)) * dx
    F += mu * q * dx - p * q * dx
    if source: 
        F -= source * eta * dx

    return F

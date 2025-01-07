from firedrake import *
from firdem.utils.Units import Units
from firdem.utils.Parameters import HeatParameters

def getBioheatForm(T, dTdt, q, c, u_blood, K, rhof, source=None):
    """
    Get Bioheat residual.

    rho c(T)(dTdt + u_blood.grad T) - div K grad T = source

    Parameters:
        T: Temperature Function
        dTdt: Time derivative of T
        c: specific heat capacity function
        u_blood: Blood velocity
        K: Thermal conductivity function
        rhof: Blood density
        source: Right hand side [optional]

    Output: Residual
    """

    F = rhof * c(T) * dTdt * q * dx \
        + rhof * c(T) * dot(u_blood, grad(T)) * q * dx \
        + K(T) * dot(grad(T), grad(q)) * dx
    if source: 
        F -= source * q * dx
    return F 

def getPMBioheatForm(T, dTdt, qT, c, u_blood, kt, kb, rhot, rhob, cb, phi, Q, source=None):
    """
    Get porous media Bioheat residuals.

    Lets write phis = 1-phi

    (phis rhot c(T) + phi * rhob * cb) * (dTdt + u_blood.grad T) - div((phis*kt(T)+phi*kb)grad(T)) + source

    Parameters:
        T: Tissue Temperature Function
        dTdt: Time derivative of T
        c: specific heat capacity function
        u_blood: Blood velocity
        kt: Thermal conductivity of tissue
        kb: Thermal conductivity of blood
        rhob: Blood density
        rhot: Tissue density
        phi: Porosity
        source: Right hand side [optional]

    Output: Residual

    """
    
    # Tissue temperature
    phis = 1-phi
    K = phis * kt(T) + phi * kb
    FF = (phis * rhot * c(T) + phi*rhob*cb) * (dTdt + dot(u_blood, grad(T))) * qT * dx \
         + dot(K * grad(T), grad(qT)) * dx \

    if source: 
        FF -= source * qT * dx
    return FF


def getNonlinearHeatForm(T, qT, dSsdt, drhophisfdt, p, sf, V, KQ, rhof, source=None):
    """
    Get residual associated with nonlinear porous heat equation:
    
    d(Ss + rho varphi sf)/dt + div(sf Vf) - div(1/T KQ grad(T)) = 1/T(source + Thetaf + Thetath)

    Parameters:
        - T: Temperature Function in Kelvin
        - dSsdt: Time derivative of (nonlinear) solid enthropy
        - drhophisfdt: Time derivative of fluid mass enthropy (mf sf = rhof varphi sf)
        - sf: fluid enthropy
        - V: fluid velocity 
        - KQ: Heat conductivity
        - source

    Output:  Residual (UFL)
    """

    mesh = T.function_space().mesh()
    dxmesh = dx(mesh)

    jf = rhof * V # fluid mass flux
    #F = T * dSsdt * qT * dxmesh 
    F = T * dSsdt * qT * dxmesh 
    F += T * drhophisfdt * qT * dxmesh
    # For this term, we use adjoint to avoid 2nd order derivatives
    F += -sf * dot(jf, grad(T * qT)) * dxmesh # T * div(sf * jf) * qT * dxmesh
    F +=  inner(KQ * grad(T), grad(qT)) * dxmesh

    # Thermal spontaneous dissipation
    Thetaf = -dot(V, grad(p))
    F += - Thetaf * qT * dxmesh
    if source: 
        F += -source * qT * dxmesh
    return F
    

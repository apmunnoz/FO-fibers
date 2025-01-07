from firedrake import *
from firdem.utils.Units import Units, SI
from firdem.utils.Parameters import ElectricsParameters
from firdem.utils.Mesh import PROLATE, BIVENTRICLE, UNITSQUARE

# Names for model types
MONODOMAIN = 21 
BIDOMAIN = 22

def getFunctionalSpace(mesh, u_degree, pde_model=MONODOMAIN):
    """
    Generate the appropriate functional space for 
    the variational problem associated with the
    electrical model.
    In case of BIDOMAIN, remember to set the associate
    nullspace to the extracellular space,
       nullspace = MixedVectorSpaceBasis(V, [V.sub(0), VectorSpaceBasis(constant=True)])
    in order to have uniqueness in the solution.
    """
    U = FunctionSpace(mesh, 'CG', u_degree)
    if pde_model == MONODOMAIN:
        V = U
    else: # BIDOMAIN
        V = U*U
    return V


def getFunctions(V):
    u = Function(V, name = "Potential")
    un = Function(V, name = "Potential")
    u_test = TestFunction(V)
    return u, un, u_test

def getBidomainNullspace(V):
    V = u.function_space()
    nullspace = MixedVectorSpaceBasis(
        V, [V.sub(0), VectorSpaceBasis(constant=True)])
    return nullspace
 
    

def getDiffusion(f, s, n, mesh_type, pde_model, units):
    """
    Create the diffusion tensors for the MONODOMAIN
    and BIDOMAIN models, using fibers computed in Fibers.py.
    Set to constant tensors in case of UNITSQUARE geometry.
    """
    uu = 1 / (units.Ohm * (1e-2 * units.L) ) # [1/(Ohm cm)]
    if pde_model == MONODOMAIN:
        sigma_f = 2e-3 * uu
        sigma_s = 1.3514e-3 * uu
        sigma_n = 6.757e-4 * uu
        sigma = sigma_f * outer(f, f) + sigma_s * outer(s, s) + sigma_n * outer(n, n)
        if mesh_type == UNITSQUARE:
            sigma = Constant( ( (sigma_f, 0), (0, sigma_s) ) )
    else: # BIDOMAIN
        sigma = [0, 0]
        sigma_f_e = 2e-3 * uu
        sigma_s_e = 1.3514e-3 * uu
        sigma_n_e = 6.757e-4 * uu
        sigma_f_i = 3e-3 * uu
        sigma_s_i = 3.1525e-4 * uu
        sigma_n_i = 3.1525e-5 * uu
        sigma[0] = sigma_f_i * outer(f, f) + sigma_s_i * outer(s, s) + sigma_n_i * outer(n, n)
        sigma[1] = sigma_f_e * outer(f, f) + sigma_s_e * outer(s, s) + sigma_n_e * outer(n, n)
        if mesh_type == UNITSQUARE:
            sigma[0] = Constant( ( (sigma_f_i, 0), (0, sigma_s_i) ) ) 
            sigma[1] = Constant( ( (sigma_f_e, 0), (0, sigma_s_e) ) ) 
    return sigma
    
    
def getResidual(u, un, u_test, w, sigma, Iion, Iapp, time_expression, dt, pde_model, IMEX, units):
    """
    Set the variational form of the pde model.
    Allow the user to choose the time splitting for
    its solution in time.
        IMEX = 1:  fully implicit Euler
        IMEX = 0:  Euler IMEX 
    Actual implementation: parabolic-parabolic formulation
    """
    parameters=ElectricsParameters(units)
    Chi = parameters.Chi
    cm = parameters.cm 
    
    FF = None
    idt = Constant(1/dt)
    
    if pde_model == MONODOMAIN:
        F1 = Chi * cm * idt * inner( u-un, u_test) * dx
        F2 = inner(sigma * grad(u), grad(u_test)) * dx 
        F3 = Chi * inner( Iion, u_test) * dx 
        F4 = inner(Iapp, u_test) * dx
        FF = F1 + F2 + F3 - F4
    else: # BIDOMAIN
        v = u[0]-u[1]
        vn = un[0]-un[1]
        v_test = u_test[0]-u_test[1]
        F1 = Chi * cm * idt * inner( v-vn, v_test) * dx
        F2 = inner(sigma[0] * grad(u[0]), grad(u_test[0])) * dx + inner(sigma[1] * grad(u[1]), grad(u_test[1])) * dx 
        F3 = Chi * inner( Iion, v_test) * dx
        F4 = inner(Iapp, v_test) * dx
        FF = F1 + F2 + F3 - F4
    return FF

def getIMEXAb(u, un, u_test, sigma, Iion, Iapp, dt, pde_model, units):
    parameters=ElectricsParameters(units)
    Chi = parameters.Chi
    cm = parameters.cm 
    V = u.function_space()
    u_trial = TrialFunction(V)
    
    FF = None
    idt = Constant(1/dt)
    
    if pde_model == MONODOMAIN:
        F1 = Chi * cm * idt * inner( u_trial-un, u_test) * dx
        F2 = inner(sigma * grad(u_trial), grad(u_test)) * dx 
        F3 = Chi * inner( Iion, u_test) * dx 
        F4 = inner(Iapp, u_test) * dx
        FF = F1 + F2 + F3 - F4
    else: # BIDOMAIN
        v = u_trial[0]-u_trial[1]
        vn = un[0]-un[1]
        v_test = u_test[0]-u_test[1]
        F1 = Chi * cm * idt * inner( v-vn, v_test) * dx
        F2 = inner(sigma[0] * grad(u_trial[0]), grad(u_test[0])) * dx + inner(sigma[1] * grad(u_trial[1]), grad(u_test[1])) * dx 
        F3 = Chi * inner( Iion, v_test) * dx
        F4 = inner(Iapp, v_test) * dx
        FF = F1 + F2 + F3 - F4
    return lhs(FF), rhs(FF)

def getTransmembranePotential(u, pde_model):
    if pde_model == MONODOMAIN:
        return u
    else:
        return u[0] - u[1]

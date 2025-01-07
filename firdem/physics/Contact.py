from firedrake import *


def getSphereContactLoad(deformation, penalization, center, R)
    """
    This function computes a Neumann loading term that enforces contact conditions through a penalized approach.

    Input: 
        - deformation: The new position vector, typically = x + u
        - penalization: epsilon parameter for enforcing contact
        - center: list with center coordinates
        - R: Radius of the sphere

    Output: 
        - Neumann load 'g' to be used in a given boundary. 
        - iter_coefficient: j function to be increased at each iteration

    The usage of this function should look as follows: 

    g, iter_coeff = getSphereContactLoad(x+u, 1e7, [0,0], 0.1)
    F_surf = dot(g, v) * ds(CONTACT_SURFACE)
    
    ii = 0
    def contactCallback(solution):
        global iter_coefficient, ii
        iter_coefficient.assign(ii)

    and then this function is given as a pre Jacobian callback function. 
    """
    
    iter_coefficient = Constant(0)
    epsilon = Constant(penalization)
    
    g_val = g(deformation, center, R)
    return iter_coefficient*epsilon*g_val**2 + 0.5*epsilon*g_val**2, iter_coefficient


class Contact:
    def __init__(self, u, iter_coefficient, increment):
        """
        Build a contact update class, where
        args:
            u : Displacement function. Must be a Function().
            iter_coefficient: Constant() to be updated in callbacks.
            increment: Additive increment to be used in the callbacks.
        """
        self.u = u
        self.ii = 0 # Iteration coefficient. 
        self.incr = increment
        self.mesh = u.function_space().mesh()
        self.V = FunctionSpace(self.mesh, 'CG', 1)
        self.G = Function(self.V)

    # Define closest point mapping from a circle
    def cpFun(x,center,radius):
        R = radius
        cen = as_vector(center) # Center of the sphere
        xMoved = x - cen
        r = sqrt(xMoved**2) # Conversion to polar coordinates
            
        cp = R/r * xMoved + cen
        n = 1/r * xMoved

        # TODO: generalize
        dim = cen.ufl_shape[0]
        orient = as_vector([0,-1]) if dim == 2 else as_vector([0,0,-1])

        prod = dot(n, orient)
        t = dot(cen - cp, orient)
        cp2 = cp + 2 * t * orient
        n2 = -n 
        cp = conditional(le(prod,0), cp2, cp)
        n = conditional(le(prod,0), n2, n)

        return cp,n

    # Define gap function
    def g(x,center,radius):
    
        cp, n = cpFun(x,center,radius)
            
        # Gap function g(x) = (x-cp(x))\cdot n
        gap = dot(x-cp, n)
        fun = conditional(le(gap, 0), np.abs(gap), 0)
    
        return fun


    def update_iter(current_solution):
        """
        This function should be given as the Jacobian callback function of the
        Newton solver where the mechanics are solved.
        """
        iter_coef.assign(jj)
        self.ii
        self.G.interpolate(g(x+u, delta_const), V_scalar)
        err = G.vector().get_local().max()

        if err > 5e-5:
            #jj = min(jj*factr , max_force)
            jj = jj + factr
            #print("UP", err, jj)
        ii += 1


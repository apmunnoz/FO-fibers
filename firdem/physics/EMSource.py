from firedrake import *

params = {"ksp_type": "cg",
          "pc_type": "gamg",
          "ksp_atol": 0.0,
          "ksp_rtol": 1e-4}


def getSource(phi, phi_test, bcs, P, gamma, g0, index, g_idx, Rrobin=None, dsRobin=None, regularization=1, verbose=False):
    v = phi_test
    V = phi.function_space()
    dphi = TrialFunction(V)
    dl = TrialFunction(V)
    mesh = phi.function_space().mesh()
    dxm = dx(mesh)
    dsm = ds(mesh)

    def a(u, v, _gamma):
        out = inner(_gamma * grad(u), grad(v)) * dxm
        if dsRobin: 
            out += 1/Rrobin * u * v * dsRobin
        return out

    def b(g, v):
        return - g * v * index * dsm(g_idx)

    def H1_norm(xx, _gamma): return assemble(inner(_gamma * grad(xx), grad(xx)) * dxm)

    # Auxiliary solvers
    l = phi.copy(True)
    g_fun = Constant(g0)
    integral = H1_norm(phi, gamma)
    c_int = Constant(2 * (integral - P))  # Placeholder for integral
    #Fstate = a(phi, v, gamma) + b(g_fun,  v)
    #Fadjoint = c_int * a(phi, v, gamma) + a(v, l, gamma)

    dFstate = a(dphi, v, gamma) + b(g_fun,v)
    dFadjoint = a(v, dl, gamma) + c_int * a(phi, v, gamma)
    problem_state = LinearVariationalProblem(lhs(dFstate), rhs(dFstate), phi, bcs=bcs)
    problem_adjoint = LinearVariationalProblem(lhs(dFadjoint), rhs(dFadjoint), l, bcs=bcs)
    solver_state = LinearVariationalSolver(problem_state, solver_parameters=params)
    solver_adjoint = LinearVariationalSolver(problem_adjoint, solver_parameters=params)

    # This solves phi for given control g
    def solveState(phi, g):
        g_fun.assign(g)
        #solve(Fstate == 0, phi, bcs=bcs, solver_parameters=params)
        solver_state.solve()

    # This solves l for given state phi
    def solveAdjoint(l, phi):
        integral = H1_norm(phi, gamma)
        c_int.assign(2 * (integral - P))  # Placeholder for integral
        #solve(Fadjoint == 0, l, bcs=bcs, solver_parameters=params)
        solver_adjoint.solve()

    def computeFunction(x, *args):
        phi = args[0]
        solveState(phi, x[0])
        return 0.5 * (H1_norm(phi, gamma) - P) ** 2 + 0.5 * regularization * x ** 2

    def computeJacobian(x, *args):
        phi = args[0]
        l = args[1]
        solveState(phi, x[0])
        solveAdjoint(l, phi)
        g_fun = Constant(x[0])
        return assemble(b(l, g_fun)) + regularization * x[0]

    g = Constant(g0)

    from scipy.optimize import minimize, fmin_l_bfgs_b, fmin_bfgs
    from numpy import inf
    ff = computeFunction
    df = computeJacobian
    #x, _, d = fmin_l_bfgs_b(ff, g0, fprime=df, args=(phi, l), approx_grad=False, m=20, factr=0.0, pgtol=0.0, disp=verbose, iprint=100, maxls=100)

    # For whatever reason, tolerances in l_bfgs_b are not being read. It actually converges for both tols = 0.
    sol = fmin_bfgs(ff, g0, fprime=df, args=(phi, l), gtol=0.0, norm=inf, xrtol=0)
    x = sol[0]

    #if verbose: print(d)
    g.assign(Constant(x))

    print("  EM g={}, P={}".format(g(0), H1_norm(phi, gamma)))
    return g(0)

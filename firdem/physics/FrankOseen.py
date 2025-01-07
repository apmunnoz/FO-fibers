from firedrake import grad, inner, dx, derivative, TrialFunction, dot, Function, assemble, sqrt, LinearVariationalProblem, LinearVariationalSolver, VectorFunctionSpace, TestFunction, FacetNormal, solve, ds, Constant, cos, sin, COMM_WORLD, DirichletBC, outer, CellDiameter
import numpy
#from AndersonAcceleration import AndersonAcceleration


def parprint(*args):
    if COMM_WORLD.rank == 0:
        print("[=]", *args, flush=True)


def solveFibers(u, bcs=None, eta=1.0, theta0=0.0, verbose=True, rtol=1e-6, stab=1e-12, ds_N=None, true_jacobian=False, init=True):

    params = {"snes_type": "ksponly", "ksp_type": "preonly",
              "pc_type": "hypre", "ksp_rtol": 0.1}

    V = u.function_space()
    v = TestFunction(V)
    dim = u.function_space().mesh().topological_dimension()
    if dim == 2 and init:
        u.interpolate(Constant((cos(theta0), sin(theta0))))
    elif dim == 3 and init:
        u.interpolate(Constant((cos(theta0), sin(theta0), 0)))
    elif init:
        u.interpolate(Constant((cos(theta0), sin(theta0))))
    un = Function(V)
    un.assign(u)

    Du = grad(u)
    Dv = grad(v)
    n = FacetNormal(V.mesh())
    nn = outer(n, n)
    F = inner(Du, Dv) * dx
    Cstab = Constant(1e1)/CellDiameter(V.mesh())
    if ds_N:
        F += - dot(nn*Du*n, v) * ds_N + Constant(-1) * dot(u, nn *
                                                           Dv*n) * ds_N + Cstab * dot(u, n) * dot(v, n) * ds_N

    DF = derivative(F, u, TrialFunction(V))
    F_err = F - inner(Du, Du) * dot(u, v) * dx
    if true_jacobian:
        DF = derivative(F_err, u, TrialFunction(V))

    # Homogenize BCs
    for b in bcs:
        b.apply(u)
    for b in bcs:
        b.homogenize()

    dduu = Function(V)

    dF = assemble(F_err, bcs=bcs)
    with dF.dat.vec_ro as v:
        err = v.norm()
    #err = sqrt(assemble(dot(dF, dF) * dx))
    err0 = err
    it = 0

    # Set up solver
    prob = LinearVariationalProblem(
        DF, -F_err, dduu, bcs=bcs, constant_jacobian=True)
    solver = LinearVariationalSolver(prob, solver_parameters=params)
    # Effective only with m=1... sometimes
    #anderson = AndersonAcceleration(0, 0)

    if verbose:
        parprint("It: {:4.0f}\tG err={:.4e}\tG err_rel={:.4e}".format(
            it, err, err/err0))
    while err > 1e-14 and err/err0 > rtol and it < 1000:
        solver.solve()
        dduu.interpolate(eta * dduu)
        newvec = u + dduu
        u.interpolate(newvec/(Constant(stab) + sqrt(dot(newvec, newvec))))
        #if anderson.order > 0:
        #    anderson.get_next_vector(u, un, dduu)
        #    u.interpolate(u/sqrt(Constant(stab) + dot(u, u)))
        un.assign(u)

        err = solver.snes.ksp.getRhs().norm()
        it += 1
        if verbose:
            parprint("It: {:4.0f}\tG err={:.4e}\tG err_rel={:.4e}".format(
                it, err, err/err0))
    if verbose:
        parprint("Done in {:4.0f} iterations.".format(it))


def generateNormalFunction(mesh, fiber_family, fiber_deg):
    V = VectorFunctionSpace(mesh, fiber_family, fiber_deg)
    u = Function(V)
    v = TestFunction(V)
    n = FacetNormal(mesh)
    F = dot(u-n, v) * ds
    solve(F == 0, u, solver_parameters={
          "snes_type": "ksponly", "ksp_type": "gmres", "pc_type": "none"})
    return u


def solveHeat(u, bcs=None, eta=1.0, theta0=0.0, verbose=True):
    params = {"snes_type": "ksponly", "ksp_type": "preonly",
              "pc_type": "hypre", "ksp_rtol": 1e-1}

    V = u.function_space()
    v = TestFunction(V)
    u.interpolate(Constant((cos(theta0), sin(theta0), 0)))
    #un = Function(V)
    # un.assign(u)

    Du = grad(u)
    Dv = grad(v)
    F = inner(Du, Dv) * dx
    DF = derivative(F, u, TrialFunction(V))

    # Homogenize BCs
    for b in bcs:
        b.apply(u)
    for b in bcs:
        b.homogenize()

    dduu = Function(V)

    dF = assemble(F, bcs=bcs)
    err = sqrt(assemble(dot(dF, dF) * dx))
    err0 = err
    it = 0

    # Set up solver
    prob = LinearVariationalProblem(DF, -F, dduu, bcs=bcs)
    solver = LinearVariationalSolver(prob, solver_parameters=params)
    # Effective only with m=1... sometimes
    #anderson = AndersonAcceleration(0, 0)

    if verbose:
        parprint("It: {:4.0f}\tG err={:.4e}\tG err_rel={:.4e}".format(
            it, err, err/err0))
    while err > 1e-14 and err/err0 > 1e-6 and it < 1000:
        solver.solve()
        dduu.interpolate(eta * dduu)
        newvec = u + dduu
        #anderson.get_next_vector(u, un, dduu)
        # un.assign(u)

        err = solver.snes.ksp.getRhs().norm()
        it += 1
        if verbose:
            parprint("It: {:4.0f}\tG err={:.4e}\tG err_rel={:.4e}".format(
                it, err, err/err0))
    if verbose:
        parprint("Done in {:4.0f} iterations.".format(it))
    u.interpolate(u/sqrt(Constant(1e-12) + dot(u, u)))


class PointBC(DirichletBC):
    def __init__(self, V, val, subdomain, point):
        super().__init__(V, val, subdomain)
        sec = V.dm.getDefaultSection()
        dm = V.mesh().topology_dm
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = dm.getCoordinateDim()
        coordsVec = dm.getCoordinatesLocal()
        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        for pt in range(vStart, vEnd):
            x = dm.getVecClosure(coordsSection, coordsVec,
                                 pt).reshape(-1, dim).mean(axis=0)
            # fix [0, 0] in original mesh coordinates (bottom left corner)
            if numpy.linalg.norm(x-point)/x.dot(x) <= 1e-4:
                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break
        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))
        self.nodes = numpy.asarray(nodes, dtype=int)
        #if len(self.nodes) > 0:
        #    print("Fixing nodes %s" % self.nodes, flush=True)
        #else:
        #    print("Not fixing any nodes", flush=True)

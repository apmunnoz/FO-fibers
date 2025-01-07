# This script computes the fiber field and all of the other related quantities using the standard potential method and also the Frank-Oseen formulation.
from firedrake import *
from FrankOseen import solveFibers, generateNormalFunction, PointBC, solveHeat, parprint


def generateLVFibers(mesh, output=False, verbose=False):

    # Boundary tags
    LV, EPI, BASE = 20, 10, 50

    # FEM spaces
    fiber_family = "CG"
    fiber_deg = 1

    # Boundary rotations
    alpha_endo = -40.0
    alpha_endo = pi/180 * alpha_endo
    alpha_epi = 50.0
    alpha_epi = pi/180 * alpha_epi
    beta_endo = 0.0  # -65
    beta_endo = pi/180 * beta_endo
    beta_epi = 0.0  # 25
    beta_epi = pi/180 * beta_epi

    # Projection smoothing
    EPS = 1e-8

    # Actual FEM spaces
    # Vector
    V = VectorFunctionSpace(mesh, fiber_family, fiber_deg)  # Potentials
    # Scalar
    Vs = FunctionSpace(mesh, fiber_family, fiber_deg+1)  # Potentials

    # Solution vectors
    phi_trans = Function(Vs, name="phi_transmural")
    phi_ab = Function(Vs, name="phi_apicobasal")
    d_trans_vec = Function(V, name="transmural_vec")
    d_ab_vec = Function(V, name="apicobasal_vec")
    d_trans = Function(V, name="transmural")
    d_ab = Function(V, name="apicobasal")
    d_vec = Function(V, name="transversal_vec")
    d = Function(V, name="transversal")
    f_vec = Function(V, name="fiber_vec")
    f = Function(V, name="fiber")
    cf_vec = Function(V, name="cross-fiber_vec")
    cf = Function(V, name="cross-fiber")

    # Normal vector as a vertexwise function
    N_fun = generateNormalFunction(mesh, fiber_family, fiber_deg)
    # FEM normal
    N = FacetNormal(mesh)

    
    ds_mesh = ds(mesh)
    vs = TestFunction(Vs)
    dus = TrialFunction(Vs)

    ## Transmural vector

    # Potential
    a = inner(grad(dus), grad(vs)) * dx
    L = Constant(0.0) * vs * dx
    bcs = [DirichletBC(Vs, Constant(1.0), EPI), DirichletBC(Vs, Constant(0.0), LV)]
    solve(a == L, phi_trans, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    d_trans.interpolate(grad(phi_trans)/(Constant(EPS) + sqrt(grad(phi_trans)**2)))

    # Vectorial
    bcs = [DirichletBC(V, N_fun, EPI), DirichletBC(V, -N_fun, LV)]
    ds_N = ds_mesh(BASE)
    solveFibers(d_trans_vec, bcs=bcs, eta=1, verbose=verbose, ds_N=ds_N)

    ## Apicobasal
    apex = Constant((-1.66032,  1.74604, 60.0145))

    # Potential
    Vs = FunctionSpace(mesh, fiber_family, fiber_deg+1)
    bcs = [DirichletBC(Vs, Constant(1), BASE), PointBC(
        Vs, Constant(0), "on_boundary", apex(0))]
    solve(a == L, phi_ab, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    d_ab.interpolate(grad(phi_ab)/(Constant(EPS) + sqrt(grad(phi_ab)**2)))

    # Vectorial
    bcs = [DirichletBC(V, N_fun, BASE), PointBC(
        V, Constant((0, 0, 0)), "on_boundary", apex(0))]
    ds_N = ds_mesh(EPI) + ds_mesh(LV)
    solveFibers(d_ab_vec, eta=1, stab=1e-8, bcs=bcs, verbose=verbose, ds_N=ds_N)
    d_ab_vec.interpolate(d_ab_vec - dot(d_ab_vec, d_trans_vec) * d_trans_vec)

    ## Transversal and fibers
    d.interpolate(cross(d_trans, d_ab))
    d_vec.interpolate(cross(d_trans_vec, d_ab_vec))


    # Rotation matrices
    def R1(alpha): return as_matrix(((cos(alpha), -sin(alpha), 0.0),
                                     (sin(alpha), cos(alpha), 0.0), (0.0, 0.0, 1.0)))


    def R2(beta): return as_matrix(((1.0, 0.0, 0.0), (0.0, cos(beta), sin(beta)),
                                    (0.0, -sin(beta), cos(beta))))

    # Orthonormal bases
    B = as_vector(((d[0], d_ab[0], d_trans[0]), (d[1], d_ab[1],
                  d_trans[1]), (d[2], d_ab[2], d_trans[2])))
    B_vec = as_vector(((d_vec[0], d_ab_vec[0], d_trans_vec[0]), (d_vec[1],
                      d_ab_vec[1], d_trans_vec[1]), (d_vec[2], d_ab_vec[2], d_trans_vec[2])))

    # Potential
    def alpha(phi): return alpha_endo * (1-phi) + alpha_epi * phi

    Q = B * R1(alpha(phi_trans)) * B.T
    f.interpolate(Q * d)
    cf.interpolate(d_ab - dot(d_ab, f) * f)

    # Vectorial
    Q_epi = B_vec * R1(alpha_epi) * B_vec.T
    Q_endo = B_vec * R1(alpha_endo) * B_vec.T
    bcs = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, LV)]
    solveFibers(f_vec, bcs=bcs, eta=1, verbose=verbose, ds_N=None)
    cf_vec.interpolate(d_ab_vec - dot(d_ab_vec, f_vec) * f_vec)

    # Output
    if output:
        F = Function(Vs, name="F")

        # Hypothesis for fibers as FO solutions
        lap = lambda f: div(grad(f))
        Q_vec_interp = B_vec * R1(alpha(phi_trans)) * B_vec.T
        LQ = lap(Q)
        F_exp = -0.5 * inner(LQ.T*Q - Q.T * LQ, outer(d, d))
        F.interpolate(F_exp)

        # Compute azimuth and elevation for both solutions
        # elevation = arccos(z/r), azimuth = sgn(y) * arccos(x/sqrt(x*x+y*y))
        def getElevation(vec):
            z = vec[2]
            r = sqrt(vec**2) # 1 for FO solutions, here for correctness
            return acos(z/r) - Constant(pi/2)

        def getAzimuth(vec):
            x = vec[0]
            y = vec[1]
            return sign(y) * acos(x / sqrt(x*x+y*y))


        elevation = Function(Vs, name="elevation")
        elevation.interpolate(getElevation(f))
        elevation_vec = Function(Vs, name="elevation_vec")
        elevation_vec.interpolate(getElevation(f_vec))
        azimuth = Function(Vs, name="azimuth")
        azimuth.interpolate(getAzimuth(f))
        azimuth_vec = Function(Vs, name="azimuth_vec")
        azimuth_vec.interpolate(getAzimuth(f_vec))

        # Compute centerline for Clairaut
        X = SpatialCoordinate(mesh)
        surf_base = assemble(1 * ds_mesh(BASE))
        dx_mesh = dx(mesh)
        vol = assemble(1 * dx_mesh)
        xs, ys, zs = assemble(X[0]*ds_mesh(BASE))/surf_base, assemble(X[1]*ds_mesh(BASE))/surf_base, assemble(X[2]*ds_mesh(BASE))/surf_base
        xv, yv, zv = assemble(X[0] * dx_mesh)/vol, assemble(X[1]*dx_mesh)/vol, assemble(X[2]*dx_mesh)/vol
        Xs = Constant((xs, ys, zs))
        Xv = Constant((xv, yv, xv))
        centerline = Xv - Xs
        centerline = centerline / sqrt(centerline**2)

        def getClairaut(vec):
            v1 = X - Xv
            Rvec = v1 - dot(v1, centerline) * centerline
            R = sqrt(Rvec**2)
            return R * cos(getElevation(vec))

        clairaut = Function(Vs, name="clairaut")
        clairaut.interpolate(getClairaut(f))
        clairaut_vec = Function(Vs, name="clairaut_vec")
        clairaut_vec.interpolate(getClairaut(f_vec))


        # Export Paraview file
        File("output/fibers-vec.pvd").write(phi_trans, phi_ab, d_trans,
                                            d_trans_vec, d_ab, d_ab_vec, d, d_vec, f, f_vec, cf, cf_vec, F,
                                            elevation, elevation_vec, azimuth, azimuth_vec, clairaut, clairaut_vec)
    return f_vec, d_ab_vec, cf_vec

if __name__ == "__main__": 

    mesh = Mesh("prolate_4mm.msh")
    f, n, s = generateLVFibers(mesh, output=True, verbose=True)
    parprint("Fibers OK")
    

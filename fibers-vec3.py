# This script computes the fiber field and all of the other related quantities using the standard potential method and also the Frank-Oseen formulation.
from firedrake import *
from FrankOseen import solveFibers, generateNormalFunction, PointBC, solveHeat, parprint


def generateLVFibers(mesh, output=False, verbose=False):

    # Boundary tags
    ENDO, EPI, BASE = 20, 10, 50

    # FEM spaces
    fiber_family = "CG"
    fiber_deg = 1

    # Boundary rotations
    alpha_endo = 80.0
    alpha_endo = pi/180 * alpha_endo
    alpha_epi = -70.0
    alpha_epi = pi/180 * alpha_epi

    # Projection smoothing
    EPS = 1e-8

    # Actual FEM spaces
    # Vector
    V = VectorFunctionSpace(mesh, fiber_family, fiber_deg)  
    # Scalar
    Vs = FunctionSpace(mesh, fiber_family, fiber_deg+1)  

    # Solution vectors
    phi_trans = Function(Vs, name="phi_transmural")
    phi_ab = Function(Vs, name="phi_apicobasal")
    d_trans_vec = Function(V, name="transmural_FO")
    d_ab_vec = Function(V, name="apicobasal_FO")
    d_trans = Function(V, name="transmural_RBM")
    d_ab = Function(V, name="apicobasal_RBM")
    d_vec = Function(V, name="transversal_FO")
    d = Function(V, name="transversal_RBM")
    f_vec = Function(V, name="fiber_FO")
    f = Function(V, name="fiber_RBM")

    # Normal vector as a vertexwise function
    N_fun = generateNormalFunction(mesh, fiber_family, fiber_deg)
    # FEM normal
    N = FacetNormal(mesh)

    
    ds_mesh = ds(mesh)
    vs = TestFunction(Vs)
    dus = TrialFunction(Vs)

    ## Transmural vector

    # Potential (RBM)
    a = inner(grad(dus), grad(vs)) * dx
    L = Constant(0.0) * vs * dx
    bcs = [DirichletBC(Vs, Constant(1.0), EPI), DirichletBC(Vs, Constant(0.0), ENDO)]
    solve(a == L, phi_trans, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    d_trans.interpolate(grad(phi_trans)/(Constant(EPS) + sqrt(grad(phi_trans)**2)))

    # Vectorial (FO)
    bcs = [DirichletBC(V, N_fun, EPI), DirichletBC(V, -N_fun, ENDO)]
    ds_N = ds_mesh(BASE)
    solveFibers(d_trans_vec, bcs=bcs, eta=1, verbose=verbose, ds_N=ds_N)

    ## Apicobasal
    apex = Constant((-1.66032,  1.74604, 60.0145))

    # Potential (RBM)
    Vs = FunctionSpace(mesh, fiber_family, fiber_deg+1)
    bcs = [DirichletBC(Vs, Constant(1), BASE), PointBC(
        Vs, Constant(0), "on_boundary", apex(0))]
    solve(a == L, phi_ab, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    d_ab.interpolate(grad(phi_ab)/(Constant(EPS) + sqrt(grad(phi_ab)**2)))

    # Vectorial (FO)
    bcs = [DirichletBC(V, N_fun, BASE), PointBC(
        V, Constant((0, 0, 0)), "on_boundary", apex(0))]
    ds_N = ds_mesh(EPI) + ds_mesh(ENDO)
    solveFibers(d_ab_vec, eta=1, stab=1e-8, bcs=bcs, verbose=verbose, ds_N=ds_N)
    d_ab_vec.interpolate(d_ab_vec - dot(d_ab_vec, d_trans_vec) * d_trans_vec)

    ## Transversal and fibers
    # Transversal
    d.interpolate(cross(d_trans, d_ab))
    d_vec.interpolate(cross(d_trans_vec, d_ab_vec))

    # Fibers

    # Rotation matrices
    def R1(alpha): return as_matrix(((cos(alpha), -sin(alpha), 0.0),
                                     (sin(alpha), cos(alpha), 0.0), (0.0, 0.0, 1.0)))

    # Orthonormal bases
    B = as_vector(((d[0], d_ab[0], d_trans[0]), (d[1], d_ab[1],
                  d_trans[1]), (d[2], d_ab[2], d_trans[2])))
    B_vec = as_vector(((d_vec[0], d_ab_vec[0], d_trans_vec[0]), (d_vec[1],
                      d_ab_vec[1], d_trans_vec[1]), (d_vec[2], d_ab_vec[2], d_trans_vec[2])))

    # Potential (RBM)
    def alpha(phi): return alpha_endo * (1-phi) + alpha_epi * phi

    Q = B * R1(alpha(phi_trans)) * B.T
    f.interpolate(Q * d)

    # Vectorial (FO)
    Q_epi = B_vec * R1(alpha_epi) * B_vec.T
    Q_endo = B_vec * R1(alpha_endo) * B_vec.T
    bcs = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, ENDO)]
    solveFibers(f_vec, bcs=bcs, eta=1, verbose=verbose, ds_N=None)

    # Vectorial (ROT)
    alpha_int = Function(Vs, name="alpha")
    a = inner(grad(dus), grad(vs)) * dx
    L = Constant(0.0) * vs * dx
    bcs = [DirichletBC(Vs, alpha_epi, EPI), DirichletBC(Vs, alpha_endo, ENDO)]
    solve(a == L, alpha_int, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    f2 = Function(V, name="fiber_ROT")
    f2.interpolate(cos(alpha_int) * d_vec + sin(alpha_int) * cross(-d_trans_vec, d_vec))

    # Output
    if output:
        ##################################################################################
        ### Hypothesis FO and Clairaut curves
        # Hypothesis for fibers as FO solutions
        F = Function(Vs, name="F")
        lap = lambda f: div(grad(f))
        Q_vec_interp = B_vec * R1(alpha(phi_trans)) * B_vec.T
        LQ = lap(Q)
        F_exp = -0.5 * inner(LQ.T*Q - Q.T * LQ, outer(d, d))
        F.interpolate(F_exp)

        ## Azimuth and Elevation angles
        # elevation = arcsin(z/r), azimuth = sgn(y) * arccos(x/sqrt(x*x+y*y))
        def getElevation(vec):
            z = vec[2]
            r = sqrt(vec**2) # 1 for FO solutions, here for correctness
            return asin(z/r)

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

        clairaut = Function(Vs, name="clairaut_RBM")
        clairaut.interpolate(getClairaut(f))
        clairaut_vec = Function(Vs, name="clairaut_FO")
        clairaut_vec.interpolate(getClairaut(f_vec))
        clairaut_f2 = Function(Vs, name="clairaut_ROT")
        clairaut_f2.interpolate(getClairaut(f2))
        
        # Export Paraview file
        File("output/fibers-vec_test.pvd").write(clairaut, clairaut_vec, clairaut_f2, f2, alpha_int, phi_trans, phi_ab, d_trans,
                                              d_trans_vec, d_ab, d_ab_vec, d, d_vec, f, f_vec, F)
    return

if __name__ == "__main__": 

    mesh = Mesh("prolate_4mm.msh")
    generateLVFibers(mesh, output=True, verbose=True)
    parprint("Fibers OK")
    

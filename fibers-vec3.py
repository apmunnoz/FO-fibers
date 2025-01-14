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
    d_trans_vec = Function(V, name="transmural_vec")
    d_ab_vec = Function(V, name="apicobasal_vec")
    d_trans = Function(V, name="transmural")
    d_ab = Function(V, name="apicobasal")
    d_vec = Function(V, name="transversal_vec")
    d = Function(V, name="transversal")
    f_vec = Function(V, name="fiber_vec")
    f = Function(V, name="fiber")

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

    # Vectorial (FO1)
    Q_epi = B_vec * R1(alpha_epi) * B_vec.T
    Q_endo = B_vec * R1(alpha_endo) * B_vec.T
    bcs = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, ENDO)]
    solveFibers(f_vec, bcs=bcs, eta=1, verbose=verbose, ds_N=None)

    # Vectorial (FO2)
    alpha_int = Function(Vs, name="alpha")
    a = inner(grad(dus), grad(vs)) * dx
    L = Constant(0.0) * vs * dx
    bcs = [DirichletBC(Vs, alpha_epi, EPI), DirichletBC(Vs, alpha_endo, ENDO)]
    solve(a == L, alpha_int, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    f2 = Function(V, name="fiber_vec2")
    f2.interpolate(cos(alpha_int) * d_vec + sin(alpha_int) * cross(-d_trans, d_vec))

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
            return asin(z/r)# + Constant(pi/2)

        def getAzimuth(vec):
            x = vec[0]
            y = vec[1]
            return sign(y) * acos(x / sqrt(x*x+y*y))


        #elevation = Function(Vs, name="elevation")
        #elevation.interpolate(getElevation(f))
        #elevation_vec = Function(Vs, name="elevation_vec")
        #elevation_vec.interpolate(getElevation(f_vec))
        #azimuth = Function(Vs, name="azimuth")
        #azimuth.interpolate(getAzimuth(f))
        #azimuth_vec = Function(Vs, name="azimuth_vec")
        #azimuth_vec.interpolate(getAzimuth(f_vec))

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

        def getClairaut2(vec):
            v1 = X - Xv
            Rvec = v1 - dot(v1, centerline) * centerline
            R = sqrt(Rvec**2)
            return R * cos(vec)

        
        #clair2 = Function(Vs, name="clairaut_cos(alpha)")
        #clair2.interpolate(getClairaut2(alpha_int))

        #clair3 = Function(Vs, name="clairaut_sin(alpha)")
        #clair3.interpolate(getClairaut_alpha(alpha_int))
        #####################################################################################
        ## Evaluating solutions model
        # Apicobasal
        n_hedg = Function(V, name="n_axis")
        denom = sqrt(dot(X-apex, X-apex))
        n_hedg.interpolate((X-apex)/(Constant(EPS)+denom))

        d_ab3 = Function(V, name="d_ab m3")
        temp = n_hedg - dot(n_hedg, d_trans_vec)*d_trans_vec
        norma2= sqrt(inner(temp, temp))
        d_ab3.interpolate(temp/norma2)

        d3 = Function(V, name= "transversal 3")
        d3.interpolate(cross(d_trans_vec, d_ab3))

        d3dotd = Function(Vs, name= "d3 dot d_vec")
        d3dotd.interpolate(dot(d3, d_vec))
        
        frot3 = Function(V, name="fiber_ev")
        frot3.interpolate(cos(alpha_int) * d3 + sin(alpha_int) * cross(-d_trans, d3))
        

        # Hypothesis for rotation as FO solutions
        R = Function(Vs, name="R")
        D = as_vector(((d[0], d_ab[0]), (d[1], d_ab[1]), (d[2], d_ab[2])))
        LD = lap(D)
        r = sqrt(X[0]*X[0] + X[1]*X[1])
        unit_r = as_vector([X[0] / r, X[1] / r])
        R_exp = -0.5 * inner(LD.T*D - D.T * LD, outer(unit_r, unit_r))
        R.interpolate(R_exp)

        
        
        # Export Paraview file
        File("output/fibers-vec_3.pvd").write(frot3, R, d_ab3, f2, alpha_int, phi_trans, phi_ab, d_trans,
                                              d_trans_vec, d_ab, d_ab_vec, d, d_vec, f, f_vec, F)
    return

if __name__ == "__main__": 

    mesh = Mesh("prolate_4mm.msh")
    generateLVFibers(mesh, output=True, verbose=True)
    parprint("Fibers OK")
    

# This script computes the fiber field and all of the other related quantities using the standard potential method and also the Frank-Oseen formulation.
from firedrake import *
from FrankOseen import solveFibers, generateNormalFunction, PointBC, solveHeat, parprint


def generateLVFibers(mesh, output=False, verbose=False):

    # Boundary tags
    LV, EPI, BASE = 20, 10, 50

    # FEM spaces
    fiber_family = "CG"
    fiber_deg = 2

    # Boundary rotations
    alpha_endo = 80.0#-60.0#-40.0
    alpha_endo = pi/180 * alpha_endo
    alpha_epi = -70.0#90.0#50.0
    alpha_epi = pi/180 * alpha_epi
    beta_endo = 0.0  # -65
    beta_endo = pi/180 * beta_endo
    beta_epi = 0.0  # 25
    beta_epi = pi/180 * beta_epi

    signo = -1#sign(alpha_endo - alpha_epi)
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
    d_ab_vec0 = Function(V, name="apicobasal_vec0")
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
    parprint("-----------Transmural vector-----------")

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
    parprint("-----------Apicobasal vector-----------")
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
    solveFibers(d_ab_vec0, eta=1, stab=1e-8, bcs=bcs, verbose=verbose, ds_N=ds_N)
    d_ab_vec = Function(V, name="apicobasal_vec")
    d_ab_vec.interpolate(d_ab_vec0 - dot(d_ab_vec0, d_trans_vec) * d_trans_vec)

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
    f.interpolate(-Q * d)
    cf.interpolate(d_ab - dot(d_ab, f) * f)

    # Vectorial
    parprint("-----------fiber vector FO-----------")
    Q_epi = B_vec * R1(alpha_epi) * B_vec.T
    Q_endo = B_vec * R1(alpha_endo) * B_vec.T
    bcs0 = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, LV)]
    solveFibers(f_vec, bcs=bcs0, eta=1, verbose=verbose, ds_N=None)
    cf_vec.interpolate(d_ab_vec - dot(d_ab_vec, f_vec) * f_vec)


    ## (P_alpha)
    alpha_int = Function(Vs, name="alpha")
    a = inner(grad(dus), grad(vs)) * dx
    L = Constant(0.0) * vs * dx
    bcs = [DirichletBC(Vs, alpha_epi, EPI), DirichletBC(Vs, alpha_endo, LV)]
    solve(a == L, alpha_int, bcs=bcs, solver_parameters={
          "ksp_type": "cg", "pc_type": "gamg"})
    grad_a = Function(V, name="grad_alph")
    grad_a.interpolate(project(grad(alpha_int), V))


    ## fibers from Cylindrical solution

    parprint("-----------u0 solution-----------")

    fsc = Function(V, name="fiber_from_cs1")
    # Cylindrical solution (u0)
    sc = Function(V, name="Cyl solution")
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]*X[0] + X[1]*X[1])
    hat_r = as_vector([X[0] / r, X[1] / r, 0])
    hat_th = as_vector([-X[1] / r, X[0] / r, 0])
    hat_z = as_vector([0, 0, 1])
    sc.interpolate(cos(alpha_int) * hat_th + sin(alpha_int) * cross(-hat_r, hat_th))

    # fiber FO
    bcs0 = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, LV)]
    solveFibers(fsc, bcs=bcs0, u0 = sc, eta=1, verbose=verbose, ds_N=None)

    ## fibers from Cylindrical solution axis \nabla \alpha / |\nabla \alpha|
    # CS axis (u1)
    parprint("-----------u1 solution-----------")
    f1_FO22 = Function(V, name="Cyl sol axis'")
    normgrada = sqrt(dot(grad_a, grad_a))
    nn = grad_a/normgrada
    f1_FO22.interpolate(cos(alpha_int) * hat_th + sin(alpha_int) * cross(nn, hat_th))

    #fiber FO
    f2_FO22 = Function(V, name="fiber_from_cs2")
    bcs0 = [DirichletBC(V, Q_epi * d_vec, EPI), DirichletBC(V, Q_endo * d_vec, LV)]
    solveFibers(f2_FO22, bcs=bcs0, u0 = f1_FO22, eta=1, verbose=verbose, ds_N=None)

    ### fibers from study of (P_{\alpha})
    ## FO fibers computed as rotation instead of solve FO

    f2 = Function(V, name="fiber_FO_rot")
    f2.interpolate(cos(alpha_int) * d_vec + sin(alpha_int) * cross(signo*d_trans, d_vec))

    ## FO and hedgehog function as d_ab

    n_hedg = Function(V, name="n_axis")
    n_hedg.interpolate((X-apex)/(Constant(EPS)+sqrt(dot(X-apex, X-apex))))

    d_ab3 = Function(V, name="d_ab_hedg")
    temp = n_hedg - dot(n_hedg, d_trans_vec)*d_trans_vec
    d_ab3.interpolate(temp/sqrt(inner(temp, temp)))        

    d3 = Function(V, name= "transversal_3")
    d3.interpolate(cross(d_trans_vec, d_ab3))

    fH = Function(V, name="fiber_hedge")
    fH.interpolate(cos(alpha_int) * d3 + sin(alpha_int) * cross(signo*d_trans, d3))


    B_vec2 = as_vector(((d3[0], d_ab3[0], d_trans_vec[0]), (d3[1],
                      d_ab3[1], d_trans_vec[1]), (d3[2], d_ab3[2], d_trans_vec[2])))


    # Vectorial hedgehog
    parprint("-----------fiber vector FO + hedgehog-----------")
    #fH2 = Function(V, name="fiber_FO_hedge")
    #Q_epi = B_vec2 * R1(alpha_epi) * B_vec2.T
    #Q_endo = B_vec2 * R1(alpha_endo) * B_vec2.T
    #bcs0 = [DirichletBC(V, Q_epi * d3, EPI), DirichletBC(V, Q_endo * d3, LV)]
    #solveFibers(fH2, bcs=bcs0, eta=1, verbose=verbose, ds_N=None)
    
    

    # Output
    if output:
        ## Hypothesis 
        F = Function(Vs, name="F")

        # Hypothesis for fibers as FO solutions
        lap = lambda f: div(grad(f))
        Q_vec_interp = B_vec * R1(alpha(phi_trans)) * B_vec.T
        LQ = lap(Q)
        F_exp = -0.5 * inner(LQ.T*Q - Q.T * LQ, outer(d, d))
        F.interpolate(F_exp)

        R = Function(Vs, name="R")

        # Hypothesis for rotation as FO solutions
        D = as_vector(((d[0], d_ab[0]), (d[1], d_ab[1]), (d[2], d_ab[2])))
        LD = lap(D)
        r = sqrt(X[0]*X[0] + X[1]*X[1])
        unit_r = as_vector([X[0] / r, X[1] / r])
        R_exp = -0.5 * inner(LD.T*D - D.T * LD, outer(unit_r, unit_r))
        R.interpolate(R_exp)

        
        ## Compute azimuth and elevation for both solutions
        # elevation = arcsin(z/r), azimuth = sgn(y) * arccos(x/sqrt(x*x+y*y))
        def getElevation(vec):
            z = vec[2]
            r = sqrt(vec**2) # 1 for FO solutions, here for correctness
            return asin(z/r)# + Constant(pi/2)

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
        surf_base = assemble(1 * ds_mesh(BASE))
        dx_mesh = dx(mesh)
        vol = assemble(1 * dx_mesh)
        xs, ys, zs = assemble(X[0]*ds_mesh(BASE))/surf_base, assemble(X[1]*ds_mesh(BASE))/surf_base, assemble(X[2]*ds_mesh(BASE))/surf_base
        xv, yv, zv = assemble(X[0] * dx_mesh)/vol, assemble(X[1]*dx_mesh)/vol, assemble(X[2]*dx_mesh)/vol
        Xs = Constant((xs, ys, zs))
        Xv = Constant((xv, yv, xv))
        centerline = Xv - Xs
        centerline = centerline / sqrt(centerline**2)


        ## Clairaut
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

        
        clairaut = Function(Vs, name="clairaut")
        clairaut.interpolate(getClairaut(f))
        clairaut_vec = Function(Vs, name="clairaut_vec")
        clairaut_vec.interpolate(getClairaut(f_vec))
        clair2 = Function(Vs, name="clairaut_cos(alpha)")
        clair2.interpolate(getClairaut2(alpha_int))
        clairaut_f3 = Function(Vs, name="clairaut_rotFO")
        clairaut_f3.interpolate(getClairaut(f2))

        CLA = Function(Vs, name="Clair_cos")
        CLA.interpolate(cos(getElevation(f_vec)))

        ## \alpha as elevation angle approximation
      
        err_alph = Function(Vs, name='error alpha')
        err_alph.interpolate(abs(alpha_int - asin(-sin(alpha_int)*d_ab3[2])))

        ## Comparation of angles

        def angle_between(v1,v2):
            nv1 = sqrt(dot(v1,v1)+ EPS)
            nv2 = sqrt(dot(v2,v2)+ EPS)
            return acos(dot(v1,v2)/(nv1*nv2))*180/3.1415
            #return dot(v1,v2)/nv2

        ang_FOPO = Function(Vs, name = "angle_FO_PO")
        ang_FOPO.interpolate(angle_between(f,f_vec))

        ang_HPO = Function(Vs, name = "angle_f(hedge)_PO")
        ang_HPO.interpolate(angle_between(f,fH))

        #ang_HPO2 = Function(Vs, name = "angle_FO(hedge)_PO")
        #ang_HPO2.interpolate(angle_between(f,fH2))

        ang_FOSC = Function(Vs, name = "angle_FO_CS(u0)")
        ang_FOSC.interpolate(angle_between(sc,f_vec))

        ang_FOfSC = Function(Vs, name = "angle_FO_f_CS")
        ang_FOfSC.interpolate(angle_between(fsc,f_vec))

        ang_FOu1 = Function(Vs, name = "angle_FO_u1")
        ang_FOu1.interpolate(angle_between(f1_FO22,f_vec))

        ang_FOfu1 = Function(Vs, name = "angle_FO_f(u1)")
        ang_FOfu1.interpolate(angle_between(f2_FO22,f_vec))

        ang_FOFO2 = Function(Vs, name = "angle_RBM_rotFO")
        ang_FOFO2.interpolate(angle_between(f2,f_vec))

        ang_RBMFO2 = Function(Vs, name = "angle_FO_rotFO")
        ang_RBMFO2.interpolate(angle_between(f2,f))

        ang_HFO2 = Function(Vs, name = "angle_f(hedge)_rotFO")
        ang_HFO2.interpolate(angle_between(f2,fH))

        #ang_H2FO2 = Function(Vs, name = "angle_FO(hedge)_rotFO")
        #ang_H2FO2.interpolate(angle_between(f2,fH2))

        ang_FOhedg = Function(Vs, name = "angle_FO_f(hedgehog)")
        ang_FOhedg.interpolate(angle_between(fH,f_vec))

        #ang_FOhedg2 = Function(Vs, name = "angle_FO_FO(hedgehog)")
        #ang_FOhedg2.interpolate(angle_between(fH2,f_vec))

        ang_ab_hedg = Function(Vs, name = "angle_dab_FO_vs_hedg")
        ang_ab_hedg.interpolate(angle_between(d_ab3,d_ab_vec))
        
        ## Calcul of S of rotation (via FO)

        def FO_ev(fib):
            Lap_f = lap(fib)
            norm_grad = inner(grad(fib), grad(fib))
            return Lap_f + norm_grad*fib
        
        S_FO = Function(V, name = "S_FO")
        S_FO.interpolate(FO_ev(f_vec))
        S_u0 = Function(V, name = "S_u0")
        S_u0.interpolate(FO_ev(sc))
        S_u1 = Function(V, name = "S_u1")
        S_u1.interpolate(FO_ev(f1_FO22))
        S_rotFO = Function(V, name = "S_rotFO")
        S_rotFO.interpolate(FO_ev(f2))
        S_hedge = Function(V, name = "S_f(hedgehog)")
        S_hedge.interpolate(FO_ev(fH))
        S_hedge2 = Function(V, name = "S_hedg_normal_dtrans")
        S_hedge2.interpolate(FO_ev(d_ab3))



        #-------------Alignment angle--------------------

        def align_ang(fib):
            #Lap_f = lap(fib)
            #norm_lap = sqrt(inner(Lap_f, Lap_f))
            #norm_f = sqrt(inner(fib, fib)) #|f|=1
            #norm_grad = inner(grad(fib), grad(fib))
            #cos_angle = 1 - norm_grad/(norm_lap*norm_f)
            #return abs(acos(cos_angle)*180/3.1415 -90)
            Lap_f = lap(fib)
            return angle_between(Lap_f, fib)

            
        align_angf2 = Function(Vs, name = "Alignment angle rotFO")
        align_angf2.interpolate(align_ang(f2))
        

        align_angFO = Function(Vs, name = "Alignment angle FO")
        align_angFO.interpolate(align_ang(f_vec))

        diff_align = Function(Vs, name = "Difference alignment")
        diff_align.interpolate(abs(align_angf2 - align_angFO))

        align_hedge = Function(Vs, name = "Alignment_hedg_normal_dtrans")
        align_hedge.interpolate(align_ang(d_ab3))

        align_hedgeFO = Function(Vs, name = "Alignment_dabFO")
        align_hedgeFO.interpolate(align_ang(d_ab_vec))
        

        diff_aligndab = Function(Vs, name = "Difference alignment dab")
        diff_aligndab.interpolate(abs(align_hedge - align_hedgeFO))

        
        #fH2, ang_H2FO2, ang_FOhedg2, ang_HPO2
        
        # Export Paraview file
        File("output/fibers-vec_grad2test.pvd").write(diff_aligndab, align_hedgeFO, align_hedge, diff_align, align_angFO, align_angf2, ang_HFO2, ang_HPO, ang_RBMFO2, ang_ab_hedg, S_hedge2, CLA, ang_FOPO, ang_FOSC, ang_FOfSC, ang_FOu1, ang_FOfu1, ang_FOFO2, ang_FOhedg, S_FO, S_u0,
                                                S_u1, S_rotFO, S_hedge, f2_FO22, f1_FO22, fsc, n_hedg, d_ab_vec0, sc, R, err_alph,
                                                clairaut_f3, fH, d_ab3, f2, clair2, grad_a, alpha_int, phi_trans,
                                                phi_ab, d_trans, d_trans_vec, d_ab, d_ab_vec, d, d_vec, f, f_vec, cf, cf_vec, F,
                                                elevation, elevation_vec, azimuth, azimuth_vec, clairaut, clairaut_vec)
    return f_vec, d_ab_vec, cf_vec

if __name__ == "__main__": 

    mesh = Mesh("prolate_4mm.msh")
    f, n, s = generateLVFibers(mesh, output=True, verbose=True)
    parprint("Fibers OK")
    

import numpy as np
from firedrake import *
from firdem.physics.Quaternions import *
from firdem.utils.Printing import parprint

# Names for mesh types
PROLATE = 14
BIVENTRICLE = 15

def ufl_norm(vec):
    return sqrt(dot(vec, vec))


def ufl_normalize(vec):
    vec.interpolate(vec / ufl_norm(vec))


def ufl_project(vec, orth):
    return vec - dot(vec, orth) * orth


def projectH1(sol, func, Vvec, C=0.0):
    tri = TrialFunction(Vvec)
    tes = TestFunction(Vvec)
    h = CellDiameter(Vvec.mesh())
    a = dot(tri, tes) + Constant(C) * h * h * inner(grad(tri), grad(tes))
    a = a * dx
    L = dot(func, tes) * dx
    pc_type = 'jacobi' if C == 0.0 else 'hypre'
    solve(a == L, sol, solver_parameters={ 'ksp_type': 'cg', 'pc_type': pc_type})


def axis(dk, dphi, Vv):
    # This function returns 3 NEW functions.
    en = dk.copy(True)
    et = Function(Vv)
    es = Function(Vv)
    ufl_normalize(en)
    et_ufl = ufl_project(dphi, en)
    et.interpolate(et_ufl / ufl_norm(et_ufl))
    ufl_normalize(et)
    es.interpolate(cross(en, et))
    return es, en, et


def orient(e0, e1, e2, alpha, beta):
    R1 = as_matrix(((cos(alpha), -sin(alpha), 0.0),
                   (sin(alpha), cos(alpha), 0.0), (0.0, 0.0, 1.0)))
    R2 = as_matrix(((1.0, 0.0, 0.0), (0.0, cos(beta), sin(beta)),
                   (0.0, -sin(beta), cos(beta))))
    Q = as_matrix(
        ((e0[0], e1[0], e2[0]), (e0[1], e1[1], e2[1]), (e0[2], e1[2], e2[2])))
    result = Q * R1 * R2
    return result


def interpolateSolutions(Q, f, s, n, name):
    if type(Q) == type(()):
        f.interpolate(Q[0])
        s.interpolate(Q[1])
        n.interpolate(Q[2])
    else:
        f.interpolate(Q[:, 0])
        s.interpolate(Q[:, 1])
        n.interpolate(Q[:, 2])
    outfile = File("{}.pvd".format(name))
    outfile.write(f, s, n)


def generateFibersBiventricle(mesh_biv, LV, RV, EPI, BASE,
                              potential_family='CG',
                              potential_deg=2,
                              fiber_family='CG',
                              alpha_endo=-40,
                              alpha_epi=-50,
                              beta_endo=0.0,
                              beta_epi=0.0,
                              fiber_deg=1,
                              TOL=1e-6):

    parprint("Computing fibers")
    V = FunctionSpace(mesh_biv, potential_family, potential_deg)  # Potentials
    Vfscal = FunctionSpace(mesh_biv, fiber_family, fiber_deg)  # Fibers
    Vfvec = VectorFunctionSpace(mesh_biv, fiber_family, fiber_deg)
    Vften = TensorFunctionSpace(mesh_biv, fiber_family, fiber_deg)
    solver_params = {'ksp_type': 'gmres', 'pc_type': 'hypre'}

    alpha_endo = pi/180 * alpha_endo
    alpha_epi = pi/180 * alpha_epi
    beta_endo = pi/180 * beta_endo
    beta_epi = pi/180 * beta_epi

    def a_s(d): return alpha_endo * (1 - d) - alpha_endo * d
    def a_w(d): return alpha_endo * (1 - d) + alpha_epi * d
    def b_s(d): return beta_endo * (1 - d) - beta_endo * d
    def b_w(d): return beta_endo * (1 - d) + beta_epi * d

    # Main definitions
    phi_trial = TrialFunction(V)
    phi_test = TestFunction(V)
    a = dot(grad(phi_trial), grad(phi_test)) * dx

    # Transmural distances
    phi_l = Function(V, name="phi_l")
    phi_r = Function(V, name="phi_r")
    phi_epi = Function(V, name="phi_epi")
    def bc_l(_x): return DirichletBC(V, _x, LV)
    def bc_r(_x): return DirichletBC(V, _x, RV)
    def bc_epi(_x): return DirichletBC(V, _x, EPI)

    solve(a == Constant(0.0)*phi_test*dx, phi_l,
          bcs=[bc_l(1.0), bc_r(0.0), bc_epi(0.0)], solver_parameters=solver_params)
    solve(a == Constant(0.0)*phi_test*dx, phi_r,
          bcs=[bc_l(0.0), bc_r(1.0), bc_epi(0.0)], solver_parameters=solver_params)
    solve(a == Constant(0.0)*phi_test*dx, phi_epi,
          bcs=[bc_l(0.0), bc_r(0.0), bc_epi(1.0)], solver_parameters=solver_params)

    # apicobasal function
    apex = Constant((100.265, -62.2247, -70.2126))
    k = Function(V, name="k")
    bcs = [DirichletBC(V, 1.0, BASE), PointBC(V, 0.0, "on_boundary", apex(0))]
    solve(a == -Constant(0.0) * phi_test * dx, k,
          bcs=bcs, solver_parameters=solver_params)

    # Compute gradients
    dphi_l = Function(Vfvec)
    dphi_r = Function(Vfvec)
    dphi_epi = Function(Vfvec)
    dk = Function(Vfvec)
    projectH1(dphi_l, grad(phi_l), Vfvec)
    projectH1(dphi_r, grad(phi_r), Vfvec)
    projectH1(dphi_epi, grad(phi_epi), Vfvec)
    projectH1(dk, grad(k), Vfvec)

    # Now we create all 3 bases
    basis_l = axis(dk, -dphi_l, Vfvec)
    basis_r = axis(dk, dphi_r, Vfvec)
    basis_epi = axis(dk, dphi_epi, Vfvec)

    d_l = phi_r / (phi_l + phi_r + TOL)
    d_r = phi_r / (phi_l + phi_r + TOL)
    Q_l_ufl = orient(*basis_l, a_s(d_l), b_s(d_l))
    Q_r_ufl = orient(*basis_r, a_s(d_r), b_s(d_r))
    Q_l = interpolate(Q_l_ufl, Vften)
    Q_r = interpolate(Q_r_ufl, Vften)
    f = Function(Vfvec, name="f")
    s = Function(Vfvec, name="s")
    n = Function(Vfvec, name="n")
    #interpolateSolutions(Q_l, f,s,n, "Q_l")
    #interpolateSolutions(Q_r, f,s,n, "Q_r")

    d_endo = phi_r / (phi_l + phi_r + TOL)
    Q_endo_ufl = bislerp(Q_l, Q_r, phi_r)
    Q_endo = interpolate(Q_endo_ufl, Vften)
    #interpolateSolutions(Q_endo, f,s,n, "Q_endo")
    Q_epi_ufl = orient(*basis_epi, a_w(phi_epi), b_w(phi_epi))
    Q_epi = interpolate(Q_epi_ufl, Vften)
    #interpolateSolutions(Q_epi, f,s,n, "Q_epi")
    Q = bislerp(Q_endo, Q_epi, phi_epi)
    #interpolateSolutions(Q, f,s,n, "Q")
    f.interpolate(Q[:, 0])
    s.interpolate(Q[:, 1])
    n.interpolate(Q[:, 2])
    ufl_normalize(f)
    ufl_normalize(s)
    ufl_normalize(n)
    File("output/fibers-BiV.pvd").write(phi_l, phi_r, phi_epi, k, f, s, n)
    parprint("Fibers OK")
    return f, s, n


def generateFibersLV(mesh_lv, LV, EPI, BASE,
                     potential_family='CG',
                     potential_deg=2,
                     fiber_family='CG',
                     fiber_def=1,
                     alpha_endo=40,
                     alpha_epi=-50,
                     beta_endo=-65,
                     beta_epi=25,
                     fiber_deg=1,
                     TOL=1e-6):

    parprint("Computing fibers")
    V = FunctionSpace(mesh_lv, potential_family, potential_deg)  # Potentials
    Vfscal = FunctionSpace(mesh_lv, fiber_family, fiber_deg)  # Fibers
    Vfvec = VectorFunctionSpace(mesh_lv, fiber_family, fiber_deg)
    Vften = TensorFunctionSpace(mesh_lv, fiber_family, fiber_deg)
    solver_params = {'ksp_type': 'gmres',
                     'pc_type': 'gamg', 'ksp_gmres_restart': 200}

    alpha_endo = pi/180 * alpha_endo
    alpha_epi = pi/180 * alpha_epi
    beta_endo = pi/180 * beta_endo
    beta_epi = pi/180 * beta_epi

    def a_s(d): return alpha_endo * (1 - d) - alpha_endo * d
    def a_w(d): return alpha_endo * (1 - d) + alpha_epi * d
    def b_s(d): return beta_endo * (1 - d) - beta_endo * d
    def b_w(d): return beta_endo * (1 - d) + beta_epi * d

    solver_params = {"ksp_type": "cg", "pc_type": "gamg"}

    # Main definitions
    phi_trial = TrialFunction(V)
    phi_test = TestFunction(V)
    a = dot(grad(phi_trial), grad(phi_test)) * dx

    # Transmural distances
    phi = Function(V, name="phi_l")
    def bc_l(_x): return DirichletBC(V, _x, LV)
    def bc_epi(_x): return DirichletBC(V, _x, EPI)

    solve(a == Constant(0.0)*phi_test*dx, phi,
          bcs=[bc_l(0.0), bc_epi(1.0)], solver_parameters=solver_params)

    # apicobasal function
    apex = Constant((-1.66032,  1.74604, 60.0145))
    k = Function(V, name="k")
    bcs = [DirichletBC(V, 1.0, BASE), PointBC(V, 0.0, "on_boundary", apex(0))]
    solve(a == -Constant(0.0) * phi_test * dx, k,
          bcs=bcs, solver_parameters=solver_params)

    # Compute gradients
    dphi = Function(Vfvec, name="transmural")
    dk = Function(Vfvec, name="apicobasal")
    projectH1(dphi, grad(phi), Vfvec)
    projectH1(dk, grad(k), Vfvec)
    ufl_normalize(dphi)
    ufl_normalize(dk)
    dk.interpolate(ufl_project(dk, dphi))

    basis = axis(dk, dphi, Vfvec)

    d = phi
    Q = orient(*basis, a_s(d), b_s(d))

    f = Function(Vfvec, name="f")
    s = Function(Vfvec, name="s")
    n = Function(Vfvec, name="n")
    f.interpolate(Q[:, 0])
    s.interpolate(Q[:, 1])
    n.interpolate(Q[:, 2])
    ufl_normalize(f)
    ufl_normalize(s)
    ufl_normalize(n)
    File("output/fibers-LV.pvd").write(phi, k, dphi, dk, f, s, n)
    parprint("Fibers OK")
    return f, s, n


def computeFibers(mesh, mesh_type=PROLATE):

    f = Constant(0)
    s = Constant(0)
    n = Constant(0)

    if mesh_type == PROLATE:
        LV = 20
        EPI = 10
        BASE = 50
        f, s, n = generateFibersLV(mesh, LV, EPI, BASE,
                                   potential_family='CG',
                                   potential_deg=2,
                                   fiber_family='CG',
                                   fiber_def=1,
                                   alpha_endo=-40,
                                   alpha_epi=50,
                                   beta_endo=0.0,
                                   beta_epi=0.0,
                                   fiber_deg=1,
                                   TOL=1e-6)
    elif mesh_type == BIVENTRICLE:
        LV = 10
        RV = 20
        EPI = 30
        BASE = 50
        f, s, n = generateFibersBiventricle(mesh, LV, RV, EPI, BASE,
                                            potential_family='CG',
                                            potential_deg=2,
                                            fiber_family='CG',
                                            alpha_endo=-40,
                                            alpha_epi=-50,
                                            beta_endo=0.0,
                                            beta_epi=0.0,
                                            fiber_deg=1,
                                            TOL=1e-6)
    else:
        pass
    return f, s, n

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
            if np.linalg.norm(x-point)/x.dot(x) <= 1e-4:
                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break
        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))
        self.nodes = np.asarray(nodes, dtype=int)
        #if len(self.nodes) > 0:
        #    print("Fixing nodes %s" % self.nodes, flush=True)
        #else:
        #    print("Not fixing any nodes", flush=True)


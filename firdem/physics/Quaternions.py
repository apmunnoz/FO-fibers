from firedrake import *

class Quaternion:
    def __init__(self, w, v):
        self.w = w
        self.v = v

    def dot(self, q):
        return self.w * q.w + dot(self.v, q.v)

    def norm(self):
        return sqrt(self.dot(self))

    def sum(self, q, in_place=False):
        scal = self.w + q.w
        vec = self.v + q.v
        if in_place:
            self.w = scal
            self.v = vec
        else:
            return Quaternion(scal, vec)

    def scale(self, a, in_place=False):
        scal = a * self.w
        vec = a * self.v
        if in_place:
            self.w = scal
            self.v = vec
        else:
            return Quaternion(scal, vec)
        
    def product(self, q, in_place=False):
        scal = self.w * q.w - dot(self.v, q.v)
        vec = self.w * q.v + q.w * self.v + cross(self.v, q.v)
        if in_place:
            self.w = scal
            self.v = vec
        else:
            return Quaternion(scal, vec)

    def interpolate(f_scal, f_vec):
        f_scal.interpolate(self.w)
        f_vec.interpolate(self.v)

    def normalize(self):
        norm = self.norm()
        self.w = self.w / norm
        self.v = self.v / norm


def rot2quat( R):
    # We comment the previous computations as they depend on computing angles.
    # The bible: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trR = tr(R)
    S1 = 2 * sqrt(trR + 1.0)
    S2 = 2 * sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1.0)
    S3 = 2 * sqrt(R[1, 1] - R[0, 0] - R[2, 2] + 1.0)
    S4 = 2 * sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1.0)

    def branch4(v1, v2, v3, v4):
        branch3 = conditional(gt(R[1, 1], R[2, 2]), v3, v4)
        branch2 = conditional(
            And(gt(R[0, 0], R[1, 1]), gt(R[0, 0], R[2, 2])), v2, branch3)
        branch1 = conditional(gt(trR, 0.0), v1, branch2)
        return branch1
    qw = branch4(0.25 * S1, (R[2, 1] - R[1, 2]) / S2,
                 (R[0, 2] - R[2, 0]) / S3, (R[1, 0] - R[0, 1]) / S4)
    qx = branch4((R[2, 1] - R[1, 2]) / S1, 0.25 * S2,
                 (R[0, 1] + R[1, 0]) / S3, (R[0, 2] + R[2, 0]) / S4)
    qy = branch4((R[0, 2] - R[2, 0]) / S1, (R[0, 1] + R[1, 0]
                                            ) / S2, 0.25 * S3, (R[1, 2] + R[2, 1]) / S4)
    qz = branch4((R[1, 0] - R[0, 1]) / S1, (R[0, 2] + R[2, 0]
                                            ) / S2, (R[1, 2] + R[2, 1]) / S3, 0.25 * S4)
    result = Quaternion(qw, as_vector((qx, qy, qz)))
    result.normalize()
    return result


def quat2rot(q):
    # We avoid the axis singularity and do
    # quat -> rot directly (as in Wiki)
    qr, qi, qj, qk = q.w, q.v[0], q.v[1], q.v[2]
    IR = as_matrix(((-qj*qj - qk*qk, qi*qj-qk*qr, qi*qk+qj*qr),
                    (qi*qj+qk*qr, -qi*qi - qk*qk, qj*qk - qi*qr),
                    (qi*qk - qj*qr, qj*qk + qi*qr, -qi*qi - qj*qj)))
    I = Identity(3) # dim = 3
    s = 1 / q.dot(q)  # norm squared
    return I + 2*s*IR


def slerp(qA, qB, t):
    TOL = 1e-6
    qM = qA
    _cosOmega = qA.dot(qB) # use _ for real (useless) one
    # First correct angle and compute out quaternion
    qM_scal = conditional(ge(_cosOmega, 0.0), qA.w, -qA.w)
    qM_vec = conditional(ge(_cosOmega, 0.0), qA.v, -qA.v)
    cosOmega = max_value(_cosOmega, -_cosOmega) 
    qM = Quaternion(qM_scal, qM_vec)
    # Then truncate to avoid errors (only positive side, we already guarantee it is not negative)
    cosOmega = min_value(1.0, cosOmega) 
    sinOmega = sqrt(1 - cosOmega * cosOmega)
    # Compute resulting vector
    omega = acos(cosOmega)
    c1 = conditional(ge(omega, TOL), sin(omega * (1-t)) / sin(omega), 1-t)
    c2 = conditional(ge(omega, TOL), sin(omega * t) / sin(omega), t)
    q1 = qM.scale(c1)
    q2 = qB.scale(c2)
    result = q1.sum(q2)
    result.normalize()
    # Branches:
    # 1) if abs(cosOmega) > 1-TOL, return qA,
    # 2) if abs(sinOmega) < TOL,   return 0.5(qA+qB)
    # 3) else, result
    cond1 = gt(abs(cosOmega), 1-TOL)
    cond2 = lt(abs(sinOmega), TOL)
    qAm = qA.scale(0.5)
    qBm = qB.scale(0.5)
    avg = qAm.sum(qBm)
    branch2_scal = conditional(cond2, avg.w, result.w)
    branch2_vec = conditional(cond2, avg.v, result.v)
    branch1_scal = conditional(cond1, qA.w, branch2_scal)
    branch1_vec = conditional(cond1, qA.v, branch2_vec)
    return Quaternion(branch1_scal, branch1_vec)


def getmaxprod(qA, qB):
    SCALAR = 0  # Arbitrary
    VECTOR = 1

    def computeCase(cases, out):  # Assume discarded ones dont exist
        q1 = cases[0]
        q2 = cases[1]
        fn1 = q1.product(qB).norm()
        fn2 = q2.product(qB).norm()
        cond = ge(fn1, fn2)
        if len(cases) == 2:
            if out == SCALAR:
                return conditional(cond, q1.w, q2.w)

            if out == VECTOR:
                return conditional(cond, q1.v, q2.v)

        else:
            cases_next1 = cases.copy()
            cases_next2 = cases.copy()
            cases_next1.pop(1)  # We remove the other one!
            cases_next2.pop(0)  # Bis
            return conditional(cond,
                               computeCase(cases_next1, out=out),
                               computeCase(cases_next2, out=out))

    quat_i = Quaternion(0.0, Constant((1., 0., 0.)))
    quat_j = Quaternion(0.0, Constant((0., 1., 0.)))
    quat_k = Quaternion(0.0, Constant((0., 0., 1.)))
    iqA = quat_i.product(qA)
    jqA = quat_j.product(qA)
    kqA = quat_k.product(qA)
    cases = [qA, qA.scale(-1.0)]#, iqA, iqA.scale(-1.0), jqA, jqA.scale(-1.0), kqA, kqA.scale(-1.0)]

    scalar = computeCase(cases, out=SCALAR)
    vector = computeCase(cases, out=VECTOR)
    qout = Quaternion(scalar, vector) 
    qout.normalize()
    return qout


def bislerp(Q_A, Q_B, t):
    q_A = rot2quat(Q_A)
    q_B = rot2quat(Q_B)
    q_M = getmaxprod(q_A, q_B)
    #q_M = q_A
    return quat2rot(slerp(q_M, q_B, t))


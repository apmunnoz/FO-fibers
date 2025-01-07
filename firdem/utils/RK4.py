from firedrake import *

class RK4:
    # 4th order Runge-Kutta for autonomous systems in Firedrake 

    def __init__(self, x, xn, tt, dt, F):
        self.x = x
        self.xn = xn
        self.dt = dt
        self.tt = tt
        self.F = F # Residual function x -> F(x)
        self.k1 = x.copy(True)
        self.k2 = x.copy(True)
        self.k3 = x.copy(True)
        self.k4 = x.copy(True)
        self.x1 = x.copy(True)
        self.x2 = x.copy(True)
        self.x3 = x.copy(True)
        self.x4 = x.copy(True)

    def assign(self, vec, fun):
        N = len(vec.subfunctions)
        if N == 1:
            vec.interpolate(fun)
        else:
            for i in range(N):
                vec.subfunctions[i].interpolate(fun[i])

    def solve(self, tn):
        F = self.F
        x = self.x
        xn = self.xn
        dt = self.dt
        tt = self.tt


        # First vector
        tt.assign(tn)
        self.assign(self.x1,xn)
        self.assign(self.k1,F(self.x1))

        # Second vector
        tt.assign(tn+dt/2)
        self.assign(self.x2,xn + Constant(0.5 * dt) * self.k1)
        self.assign(self.k2,F(self.x2))
    
        # Third vector
        tt.assign(tn+dt/2)
        self.assign(self.x3,xn + Constant(0.5 * dt) * self.k2)
        self.assign(self.k3,F(self.x3))
    
        # Fourth vector
        tt.assign(tn+dt)
        self.assign(self.x4,xn + Constant(dt) * self.k3)
        self.assign(self.k4,F(self.x4))

        # Finish and reset time
        self.assign(x, xn + Constant(dt/6) * (self.k1+2*(self.k2+self.k3)+self.k4))
        tt.assign(tn) 
        
        

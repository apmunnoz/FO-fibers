from firedrake import *
from ufl.variable import Variable

def assertIsVariable(vu):
    assert type(vu) == Variable, "Derivative argument must be of ufl.variable.Variable type, otherwise it will return 0"
    # Note that vu might be a variable and still the Form might not depend on it, so this is only a mild test. 

class ConstitutiveModel:
    
    def __init__(self):
        self.helmholtz = None
        self.gibbs = None

    def addHelmholtz(self,term):
        if self.helmholtz:
            self.helmholtz += term
        else:
            self.helmholtz = term

    def addGibbs(self, term):
        if self.gibbs: 
            self.gibbs += term
        else:
            self.gibbs = term

    def getPiola(self, F): 
        assertIsVariable(F)
        return diff(self.helmholtz, F)

    def getPressure(self, varphi):
        assertIsVariable(varphi)
        # Note that this works for multi-phase as well
        return diff(self.helmholtz, varphi)

    def getSolidEnthropy(self, T): 
        assertIsVariable(T)
        return -diff(self.helmholtz, T)

    def normalizePressure(self, varphi, varphi0, F, F0, p_ref):
        p = self.getPressure(varphi)
        p0 = replace(p, {varphi: varphi0, F:F0})
        self.addHelmholtz( - p0 * varphi + p_ref * varphi)

    def getFluidDensity(self, p): 
        assertIsVariable(p)
        return 1/diff(self.gibbs, p)

    def getFluidEnthropy(self, T):
        assertIsVariable(T)
        return -diff(self.gibbs, T)

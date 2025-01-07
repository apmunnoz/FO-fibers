from firedrake import *
from firdem.utils.Units import SI

NONE=0
ANALYTICAL=1
SS=2

class ActiveStress:
    def __init__(self, mesh, type, units):
        assert type in (ANALYTICAL, SS), "Wrong Active stress type"
        self.type = type
        self.mesh = mesh
        #self.v = v
        V = FunctionSpace(mesh, 'CG', 1)
        self.V = V
        self.Ta = Function(V)
        self.Tan = Function(V)
        self.units=units

    def getTa(self, t):
        if self.type == ANALYTICAL:
            T_wave = Constant(0.8*self.units.S) # seconds
            AS = Constant(5e4 * self.units.Pa) # [Pa]
            Ta = Constant(AS)* max_value(sin(2*pi*t/T_wave), 0)
            return Ta
        elif self.type == NONE: 
            return Constant(0.0)
        else: # SS
            return self.Ta

    def getForm(self, dt, v):
        v_peak = 1
        v_rest = 0
        vv = (v - v_rest) / (v_peak - v_rest)
        eps_Ta = conditional(lt(vv, 0.05), Constant(1.0), Constant(10.0))
        k_Ta = Cosntant(47.9e3)
        idt = Constant(1/dt)
        return idt * (Ta - Tan) - eps_Ta * (k_Ta * vv - Ta)



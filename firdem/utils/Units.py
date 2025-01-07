from firedrake import Constant

class Units:
    def __init__(self, Metres=1, Kilograms=1, Seconds=1, Kelvin=1, Ampere=1, mol=1):
        # SI by default
        self.S = Constant(Seconds) # 1 second
        self.L = Constant(Metres) # 1 metre 
        self.KG = Constant(Kilograms) # 1 kg
        self.T = Constant(Kelvin) # 1 Kelvin
        self.A = Constant(Ampere) # 1 Ampere
        self.mol = Constant(mol) # 1 mol

        # Derived units
        self.Vel = Constant(self.L/self.S) # Velocity
        self.N = Constant(self.KG * self.L / self.S**2) # Newton
        self.Pa = Constant(self.N / self.L**2) # Pascal
        self.J = Constant(self.N * self.L) # Joule
        self.W = Constant(self.J / self.S) # Watt
        self.Volt = Constant(self.W / self.A) # Volt
        self.C = Constant(self.A * self.S) # Coulomb
        self.Ohm = Constant(self.Volt / self.A) # Ohm
        self.ohm = Constant(self.Ohm)
        self.mho = Constant(1/self.ohm)# Siemens [S]
        self.Farad = Constant((self.S**4) * (self.A**2) / ( (self.L**2) * self.KG )) # Farad
        self.liter = Constant(1e-3 * self.L**3)
        self.M = Constant(self.mol / self.liter) # mol/L

        # milli units and similar
        self.mV = Constant(1e-3 * self.Volt) # mV
        self.mmol = Constant(1e-3 * self.mol) # mmol
        self.cm = Constant(1e-2 * self.L) # centimeter
        self.mm = Constant(1e-3 * self.L) # centimeter
        self.um = Constant(1e-6 * self.L) # micrometer
        self.mL = Constant(1e-3 * self.liter) # milliliter
        self.mM = Constant(1e-3 * self.M) # micromol/liter
        self.pF = Constant(1e-12 * self.Farad) # picoFarad
        self.uF = Constant(1e-6 * self.Farad) # microFarad
        self.ms = Constant(1e-3 * self.S) # milliseconds

    def rescale(self, unit, value):
        unit.assign(value) # Shorthand for rescaling basic units

SI=Units() # m, kg, s
CGS=Units(Metres=1e2, Kilograms=1e3) # cm, g, s

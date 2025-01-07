from firdem.utils.Units import SI

# Names for mechanics model types
GUCCIONE = 1
HOLZAPFELOGDEN = 2
USYK = 3
NEOHOOKEAN = 4
MOONEYRIVLIN = 5


class MechanicsParameters:

    def __init__(self, units):
        # Parameters are defined by default in SI
        # Guccione constants
        self.Cg  = 0.88e3 * units.Pa # [Pa]
        self.bf  = 8       # [-]
        self.bff = 8       # [-]
        self.bs  = 6       # [-]
        self.bn  = 3       # [-]
        self.bfs = 12      # [-]
        self.bfn = 3       # [-]
        self.bsn = 3       # [-]
        self.k   = 5e4 * units.Pa # [Pa]
        self.rhos = 1e3 * units.KG / units.L**3 # kg / m^3
        self.mechanicsModel = USYK



class IonicsParameters:

	def __init__(self, units):
		#self.Ia = 20 * (1e-3 * units.A) / (1e-2 * units.L)**3 # [mA/cm^3]
		self.Ia = 1 * (1e-3 * units.A) / (1e-2 * units.L)**3 # [mA/cm^3]
		self.ta = 1 * (1e-3 * units.S)	# [ms]
		


class ElectricsParameters:

	def __init__(self, units=SI):
		self.cm = 1 * (1e-3 * units.Farad) / (1e-2 * units.L)**2 # [mF/cm^2]
		self.Chi = 1 * (1 / (1e-2 * units.L) ) # [1/cm]
		

class PerfusionParameters:

    def __init__(self, units=SI):
        self.rhof = 1e3 * units.KG / units.L**3 # kg / m^3
        self.muf = 0.035 * units.Pa * units.S
        self.p_art = 1e4 * units.Pa
        self.p_ven = 1e3 * units.Pa
        self.perm = 1e-8 * units.L**2 / units.Pa / units.S
        self.gamma = 1e-4 / units.Pa / units.S

class HeatParameters:
    
    def __init__(self, units=SI):
        self.rhof = 1e3 * units.KG / units.L**3 # kg / m^3
        self.k = 1e-6 # TODO UNITS

mechanics_parameters = MechanicsParameters(SI)
ionics_parameters = IonicsParameters(SI)
electrics_parameters = ElectricsParameters(SI)
perfusion_parameters = PerfusionParameters(SI)


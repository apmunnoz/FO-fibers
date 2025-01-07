from firedrake import exp, ln, sqrt, conditional, gt, ge, lt, le, And, as_vector
from Units import Units, SI


# Hodkin-Huxley type right hand side
def HH(x_inf, tau_x, x):
    a = x_inf / tau_x
    b = (1-x_inf) / tau_x
    #return a * (1-x) + b*x
    return (x_inf - x) / tau_x

class TTPModel:

    def norm_exp(self, a, b, V, power=1):
        # Normalized exponential used in all HH-type dynamics
        return exp((V + a * self.units.mV)**power/(b * self.units.mV))

    def Ca_ifun(self, Ca_iT):
        b_Cai = self.K_bufc + self.Buf_c - Ca_iT
        c_Cai = self.K_bufc * Ca_iT
        Ca_i = 0.5 * (sqrt(b_Cai**2 + 4 * c_Cai) - b_Cai)
        return Ca_i

    def Ca_ssfun(self, Ca_ssT):
        b_Cass = self.K_bufss + self.Buf_ss - Ca_ssT
        c_Cass = self.K_bufss * Ca_ssT
        Ca_ss = 0.5 * (sqrt(b_Cass**2 + 4 * c_Cass) - b_Cass)
        return Ca_ss

    def Ca_srfun(self, Ca_srT):
        b_Casr = self.K_bufsr + self.Buf_sr - Ca_srT
        c_Casr   = self.K_bufsr  * Ca_srT
        Ca_sr = 0.5 * (sqrt(b_Casr**2 + 4 * c_Casr) - b_Casr)
        return Ca_sr

    def getHHIndexes(self):
        # m,h,j,xr1,xr2,xs,r,s,d,f,f2,fcass
        return 5,6,7,8,9,10,11,12,13,14,15,16

    def getNonHHIndexes(self):
        return (i for i in range(18) if i not in self.getHHIndexes())

    def unpackGatingSubfunctions(self, w):
        Ca_iT = w.subfunctions[0]
        Ca_ssT = w.subfunctions[1]
        Ca_srT = w.subfunctions[2]
        Na_i = w.subfunctions[3]
        K_i = w.subfunctions[4] 
        m = w.subfunctions[5]
        h = w.subfunctions[6]
        j = w.subfunctions[7]
        xr1 = w.subfunctions[8]
        xr2 = w.subfunctions[9]
        xs = w.subfunctions[10]
        r = w.subfunctions[11]
        s = w.subfunctions[12]
        d = w.subfunctions[13]
        f = w.subfunctions[14]
        f2 = w.subfunctions[15]
        fcass = w.subfunctions[16]
        Rbar = w.subfunctions[17]
        return Ca_iT, Ca_ssT, Ca_srT, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar

    def unpackGating(self, w, total=False):
        Ca_iT = w[0]
        Ca_ssT = w[1]
        Ca_srT = w[2]

        # Obtain free concentrations
        if total:
            Ca_i = Ca_iT
            Ca_ss = Ca_ssT
            Ca_sr = Ca_srT
        else: 
            Ca_i = self.Ca_ifun(Ca_iT)
            Ca_ss = self.Ca_ssfun(Ca_ssT)
            Ca_sr = self.Ca_srfun(Ca_srT)
        Na_i = w[3]
        K_i = w[4] 
        m = w[5]
        h = w[6]
        j = w[7]
        xr1 = w[8]
        xr2 = w[9]
        xs = w[10]
        r = w[11]
        s = w[12]
        d = w[13]
        f = w[14]
        f2 = w[15]
        fcass = w[16]
        Rbar = w[17]
        return Ca_i, Ca_ss, Ca_sr, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar


    def __init__(self, v, w, units):
        self.units = units
        # gas constant
        self.R = 8.3143 * units.J / units.T / units.mol # [J/K/mol]
        # temperature   
        self.T = 310 * units.T # [Kelvin]
        # Faraday constant
        self.F = 94.4867 * units.C / units.mmol
        # cell capacitance per unit surface area
        self.Cm = 2 * units.uF / (units.cm)**2 # uF /cm**2
        self.CAP = 0.185 * units.uF / (units.cm)**2
        # surface to volume ratio
        self.S = 0.2 / (units.um) # 1/um
        # cellular resistivity
        self.rho = 162 * units.Ohm * units.cm

        # cytoplasmic volume
        self.V_C = 16404 * units.um**3 # See code errata
        # sarcoplasmic reticulum volume
        self.V_SR = 1094 * units.um**3 # See code errata
        # Subspace volume
        self.V_ss = 54.68 * units.um**3 # See code errata
        # extracellular K+ concentration
        self.K_O = 5.4 * units.mM
        # extracellular Na+ concentration
        self.Na_O = 140 * units.mM
        # extracellular Ca2+ concentration
        self.Ca_O = 2 * units.mM
        # maximal I_Na conductance
        units.nS = 1e-9
        self.G_Na = 14.838 * units.nS / units.pF
        # maximal I_K1 conductance
        self.G_K1 = 5.405 * units.nS / units.pF
        # maximal epicardial I_to conductance
        self.G_to_epi = 0.294 * units.nS / units.pF
        # maximal endocardial I_to conductance
        self.G_to_endo = 0.073 * units.nS / units.pF
        # maximal I_Kr conductance
        self.G_Kr = 0.153 * units.nS / units.pF
        # maximal epi and endo I_Ks conductance
        self.G_Ks_epi = 0.392 * units.nS / units.pF
        # maximal M cell I_Ks conductance
        self.G_Ks_M = 0.098 * units.nS / units.pF
        #relative I_Ks permeability to Na+
        self.p_KNal = 0.03
        # maximal I_CaL conductance
        self.G_CaL = 3.980e-5 * units.cm / units.ms / units.uF
        # maximal I_NaCa
        self.k_NaCa = 1000 * 1e-12 * units.A / units.pF
        # voltage dependence of I_NaCa
        self.gamma = 0.35
        # Ca_i half saturation for I_NaCa
        self.K_mCa = 1.38 * units.mM
        # Na_i half saturation for I_NaCa
        self.K_mNai = 87.5 * units.mM
        # saturation factor for I_NaCa
        self.k_sat = 0.1
        # factor enhancing outward nature of I_NaCa
        self.alpha = 2.5
        # maximal I_NaK
        self.P_NaK = 2.724 * 1e-12 * units.A / units.pF
        # K_O half saturation of I_NaK
        self.K_mK = units.mM
        # Na_i half saturation constant of I_NaK
        self.K_mNa = 40 * units.mM
        # maximal I_pK conductance
        self.G_pK = 0.0146 * units.nS / units.pF
        # maximal I_pCa conductance
        self.G_pCa = 0.1238 * units.nS / units.pF
        # Ca_i half saturation of I_pCa 
        self.K_pCa = 0.0005 * units.mM
        # maximal I_bNa conductance
        self.G_bNa = 0.00029 * units.nS / units.pF
        # maximal I_bCa conductance
        self.G_bCa = 0.000592 * units.nS / units.pF
        # maximal I_up
        self.V_maxup = 0.006375 * units.mM / units.ms
        # half saturation of I_up
        self.K_up = 0.00025 * units.mM
        # Maximal I_rel conductance
        self.V_rel = 0.102 * units.mM / units.ms # see code errata
        # R to O and RI to I I_rel transition rate
        self.k_1prime = 0.15 * units.mM**(-2) / units.ms
        # O to I and R to RI I_rel transition rate
        self.k_2prime = 0.045 * units.mM**(-1) / units.ms
        # O to R and I to RI I_rel transition rate
        self.k_3 = 0.060 / units.ms
        # I to O and RI to I I_rel transition rate
        self.k_4 = 0.005 / units.ms # see code errata
        # Ca_SR half saturation
        self.EC = 1.5 * units.mM
        # Maximum value of k_casr
        self.max_sr = 2.5
        # Minumum value of k_case
        self.min_sr = 1
        # maximal I_leak 
        self.V_leak = 0.00036 * units.mM / units.ms
        # Maximal I_xfer conductance
        self.V_xfer = 0.0038 * units.mM / units.ms
        # total cytoplasmic buffer concentration
        self.Buf_c = 0.2 * units.mM
        # Ca_i half saturation constant for cytoplasmic buffer
        self.K_bufc = 0.001 * units.mM
        # total sarcoplasmic buffer concentration
        self.Buf_sr = 10 * units.mM
        # Ca_SR half saturation constant for sarcoplasmic buffer
        self.K_bufsr = 0.3 * units.mM
        # total subspace buffer concentration
        self.Buf_ss = 0.4 * units.mM
        # Ca_ss half-saturation constant
        self.K_bufss = 0.00025 * units.mM

        # Then, set all quantities to be used in the residuals
        Ca_i, Ca_ss, Ca_sr, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar = self.unpackGating(w)

        # I_Na
        self.E_Na = self.R * self.T / self.F * ln(self.Na_O / Na_i)
        I_Na = self.G_Na * m**3 * h * j * (v - self.E_Na)
        self.I_Na = I_Na
        
        # I_CaL
        fact = self.G_CaL * d * f * f2 * f_cass * 4 * (v-15 * self.units.mV) * self.F**2 / self.R / self.T
        VFRT = (v-15*self.units.mV) * self.F / self.R / self.T
        I_CaL = fact * (0.25 * Ca_ss*exp(2*VFRT) - self.Ca_O) / (exp(2 * VFRT) - 1)
        self.I_CaL = I_CaL 
    
        # I_Ks
        self.E_Ks = self.R * self.T / self.F * ln((self.K_O+self.P_NaK*self.Na_O) / (K_i+self.P_NaK*Na_i))
        I_Ks = self.G_Ks_M * xs**2 * (v - E_Ks) # FIX KS_EPI
        self.I_Ks = I_Ks 

        # I_Kr
        self.E_K = self.R * self.T / self.F * ln(self.K_O / K_i)
        denom = 5.4 * self.units.mM
        I_Kr =  self.G_Kr * sqrt(self.K_O/denom) * xr1 * xr2 * (v - self.E_K)
        self.I_Kr = I_Kr

        # I_K1
        self.E_K = self.R * self.T / self.F * ln(self.K_O / K_i)
        a_K1 = 0.1 / (1 + self.norm_exp(-200, 1/0.06, v-self.E_K))
        b_K1 = (3 * self.norm_exp(100, 1/0.0002, v-self.E_K) + self.norm_exp(-10, 10, v-self.E_K)) / (1 + self.norm_exp(0, -1/2, v-self.E_K) )
        x_K1_inf = a_K1 / (a_K1 + b_K1)
        denom = 5.4 * self.units.mM
        I_K1 =  self.G_K1 * sqrt(self.K_O / denom) * x_K1_inf * (v - self.E_K)
        self.I_K1 = I_K1
        
        # I_NaCa
        I_NaCa = self.k_NaCa 
        VFRT = v * self.F / self.R / self.T
        I_NaCa *= exp(self.gamma * VFRT) * Na_i**2 * self.Ca_O - exp((self.gamma-1) * VFRT) * self.Na_O**3*Ca_i*self.alpha
        I_NaCa /= (self.K_mNai**3 + self.Na_O**3) * (self.K_mCa + self.Ca_O) * (1 + self.k_sat * exp((self.gamma-1)*VFRT))
        self.I_NaCa = I_NaCa

        # I_NaK
        I_NaK = self.P_NaK * self.K_O * Na_i
        I_NaK /= (self.K_O+self.K_mK) * (Na_i + self.K_mNa) * (1 + 0.1245 * exp(-0.1 * VFRT) + 0.0353 * exp(-VFRT))
        self.I_NaK = I_NaK

        # I_pCa
        I_pCa = self.G_pCa * Ca_i / (self.K_pCa + Ca_i)
        self.I_pCa = I_pCa

        # I_pK
        I_pK = self.G_pK * (v-self.E_K) / (1 + self.norm_exp(25, 5.98, -v))
        self.I_pK = I_pK
    
        # I_bNa
        I_bNa = self.G_bNa * (v - self.E_Na)
        self.I_bNa = I_bNa
    
        # I_bCa
        self.E_Ca = 0.5 * self.R * self.T / self.F * ln(self.Ca_O / Ca_i)
        self.I_bCa = I_bCa

        # I_leak
        I_leak = self.V_leak * (Ca_sr - Ca_i)
        self.I_leak = I_leak

        # I_up
        I_up = self.V_maxup / (1 + self.K_up**2 / Ca_i**2)
        self.I_up = I_up

        # I_rel
        k_casr = self.max_sr - (self.max_sr - self.min_sr) / (1 + (self.EC/Ca_sr)**2)
        k_1 = self.k_1prime / k_casr
        O = k_1 * Ca_ss**2 * Rbar / (self.k_3 + k_1 * Ca_ss**2)
        I_rel = self.V_rel * O * (Ca_sr - Ca_ss)
        self.I_rel = I_rel
        
        # I_to
        I_to = self.G_to_endo * r * s * (v - self.E_K)
        self.I_to = I_to 

        # I_xfer
        I_xfer = self.V_xfer * (Ca_ss - Ca_i)
        self.I_xfer = I_xfer

        # I_ion
        I_ion = self.I_Na  + self.I_K1 + self.I_to + self.I_Kr + self.I_Ks + self.I_CaL  + self.I_NaCa + self.I_NaK + self.I_pCa + self.I_pK + self.I_bCa + self.I_bNa + self.I_xfer
        self.I_ion = I_ion

        #### Gating variables

        # m variable
        m_inf = 1.0/(1 + self.norm_exp(-56.86, 9.03, -V))**2.0
        a_m = 1.0/(1.0 + self.norm_exp(-60, 5, -V)) 
        b_m = 0.1  / (1 + self.norm_exp(35, 5, V)) + 0.1 / (1 + self.norm_exp(-50, 200, V))
        tau_m = a_m * b_m * self.units.ms
        self.m_inf = m_inf
        self.tau_m = tau_m

        # h variable
        h_inf = 1 / ( 1 + self.norm_exp(71.55, 7.43, V) )**2
        a_h = conditional(ge(V, -40 * self.units.mV), 
                    0, 
                    0.057 * self.norm_exp(80, -6.8, V))
        b_h = conditional(ge(V, -40 * self.units.mV), 
                    0.77 / (0.13 * (1 + self.norm_exp(10.66, -11.1, V))),
                    2.7 * exp(0.079 * V / self.units.mV) + 3.1e5 * exp(0.3485 * V / self.units.mV))
        tau_h = 1/(a_h + b_h) * self.units.ms
        self.h_inf = h_inf
        self.tau_h = tau_h
        
        # j variable
        j_inf = 1 / ( 1 + self.norm_exp(71.55, 7.43, V) )
        a_j = conditional(ge(V, -40 * self.units.mV),
                    0,
                    -2.5428e4 * self.norm_exp(0, 1/0.2444, V) - 6.948e-6 * self.norm_exp(0, -1/0.04391, V))
        b_j = conditional(ge(V, -40 * self.units.mV),
                    0.6 * self.norm_exp(0, 1/0.057, V) / (1 + self.norm_exp(32, -10, V)),
                    0.02424 * self.norm_exp(0, -1/0.01052, V) / (1 + self.norm_exp(40.14, -1/0.1378, V)))
        tau_j = 1 / (a_j + b_j) * self.units.ms
        self.j_inf = j_inf
        self.tau_j = tau_j
        
        # d variable
        d_inf = 1 / ( 1 + self.norm_exp(-8, 7.5, -V) )
        a_d = 1.4 / (1 + self.norm_exp(-35, 13, -V)) + 0.25
        b_d = 1.4 / (1 + self.norm_exp(5, 5, V))
        gamma_d = 1 / (1 + self.norm_exp(50, 20, -V))
        tau_d = (a_d * b_d + gamma_d) * self.units.ms
        self.d_inf = d_inf
        self.tau_d = tau_d

        # f variable
        f_inf = 1 / (1 + self.norm_exp(20, 7, V))
        a_f = 1102.5 * exp(-((V + 27 * self.units.mV) / (15*self.units.mV))**2)
        b_f = 200 / (1 + self.norm_exp(13, 10, -V))
        g_f = 20 + 180 / (1 + self.norm_exp(30, 10, V))
        tau_f = a_f + b_f + g_f
        tau_f = tau_f * self.units.ms
        self.f_inf = f_inf
        self.tau_f = tau_f

        # f2 variable
        f2_inf = 0.33 + 0.67 / (1 + self.norm_exp(35, 7, V))
        a_f = 600 * exp(-( (V+25*self.units.mV)/(sqrt(170)*self.units.mV))**2)
        b_f = 31 / (1 + self.norm_exp(25, 10, -V))
        g_f = 16 / (1 + self.norm_exp(30, 10, V))
        tau_f2 = a_f + b_f + g_f
        tau_f2 = tau_f2 * self.units.ms
        self.f2_inf = f2_inf
        self.tau_f2 = tau_f2


        # fCass variable
        fact =  1 + (Ca_ss / 0.05 / self.units.mM)**2
        fCass_inf = 0.6 / fact + 0.4
        tau_fCass = 2 + 80 / fact
        tau_fCass = tau_fCass * self.units.ms # milliSeconds
        self.fcass_inf = fcass_inf
        self.tau_fcass = tau_fcass
        
        # r variable
        r_inf = 1 / (1 + self.norm_exp(20, 6, -V))
        tau_r = 9.5 * self.norm_exp(40, -1800 * self.units.mV, V, power=2) + 0.8
        tau_r = tau_r * self.units.ms
        self.r_inf = r_inf
        self.tau_r = tau_r

        # s variable
        s_inf = 1 / (1 + self.norm_exp(20, 5, V))
        tau_s = 85 * self.norm_exp(45, -320 * self.units.mV , V, power=2) + 5 / (1 + self.norm_exp(-20, 5, V)) + 3
        tau_s = tau_s * self.units.ms
        self.s_inf = s_inf
        self.tau_s = tau_s

   # # Endocardial cells [TODO]
   # def s_HH_endo(self, V, s):
   #     s_inf = 1 / (1 + self.norm_exp(28, 5, V))
   #     tau_s = 1000 * self.norm_exp(67, -1000 * self.units.mV, V, power=2) + 8
   #     tau_s = tau_s * self.units.ms
   #     return HH(s_inf, tau_s, s)


        # xs variable
        xs_inf = 1 / (1 + self.norm_exp(5, -14, V))
        a_xs = 1400 / sqrt(1 + self.norm_exp(5, 6, -V))
        b_xs = 1 / (1 + self.norm_exp(-35, 15, V))
        tau_xs = a_xs * b_xs + 80
        tau_xs = tau_xs * self.units.ms
        self.xs_inf = xs_inf
        self.tau_xs = tau_xs

        # xr1 variable
        xr1_inf = 1 / (1 + self.norm_exp(26, -7, V))
        a_xr1 = 450 / (1 + self.norm_exp(45, -10, V))
        b_xr1 = 6 / (1 + self.norm_exp(30, 11.5, V))
        tau_xr1 = a_xr1 * b_xr1 * self.units.ms
        self.xr1_inf = xr1_inf
        self.tau_xr1 = tau_xr1

        # xr2 variable
        xr2_inf = 1 / (1 + self.norm_exp(88, 24, V))
        a_xr2 = 3 / (1 + self.norm_exp(60, -20, V))
        b_xr2 = 1.12 / (1 + self.norm_exp(-60, 20, V))
        tau_xr2 = a_xr2 * b_xr2 * self.units.ms
        self.xr2_inf = xr2_inf
        self.tau_xr2 = tau_xr2


        ### RHS for ODEs

        # Rbar variable
        k_casr = self.max_sr - (self.max_sr - self.min_sr) / (1 + (self.EC/Ca_sr)**2)
        k2 = self.k_2prime * k_casr
        self.dRbardt = -k2 * Ca_ss * Rbar + self.k_4 * (1 - Rbar)

    #def Ca_ibufc(self, Ca_i):
    #    return Ca_i * self.Buf_c / (Ca_i + self.K_bufc)
    #def Ca_srbufsr(self, Ca_sr):
    #    return Ca_sr * self.Buf_sr / (Ca_sr + self.K_bufsr)

        self.dCa_itotaldt = -(I_bCa + I_pCa -2*I_NaCa) * self.CAP / (2*self.V_C * self.F)+self.V_SR / self.V_C * (I_leak - I_up) + I_xfer
        self.dCa_srtotaldt = -I_leak + I_up - I_rel
        self.dCa_sstotaldt = -I_CaL * self.CAP / (2*self.V_ss*self.F) + self.V_SR * I_rel / self.V_ss - self.V_C / self.V_ss * I_xfer
        self.dNaidt = -self.CAP/(self.V_C * self.F) * (I_Na + I_bNa + 3 * I_NaK + 3 * I_NaCa)
        self.dKidt = -self.CAP/(self.V_C * self.F) * (I_K1 + I_to + I_Kr + I_Ks - 2*I_NaK + I_pK + I_stim)


    def getGatingRHS(self, v, w, I_stim):
        Ca_i, Ca_ss, Ca_sr, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar = self.unpackGating(w)
        Rm = self.m_HH(v, m)  
        Rh = self.h_HH(v, h) 
        Rj = self.j_HH(v, j)
        Rxr1 = self.xr1_HH(v, xr1) 
        Rxr2 = self.xr2_HH(v, xr2)
        Rxs = self.xs_HH(v, xs)
        Rr = self.r_HH(v, r)
        Rs = self.s_HH_endo(v, s)
        Rd = self.d_HH(v, d)
        Rf = self.f_HH(v, f)
        Rf2 = self.f2_HH(v, f2)
        Rfcass = self.fcass_HH(fcass, Ca_ss)
        RRbar = self.dRbardt(Rbar, Ca_ss, Ca_sr)
        RCa_i = self.dCa_itotaldt(v,w)
        RCa_sr = self.dCa_srtotaldt(v, w)
        RCa_ss = self.dCa_sstotaldt(v, w)
        RNa_i = self.dNaidt(v, w)
        RK_i =  self.dKidt(v, w, I_stim)
        vec = [RCa_i, RCa_ss, RCa_sr, RNa_i, RK_i, Rm, Rh, Rj, Rxr1, Rxr2, Rxs, Rr, Rs, Rd, Rf, Rf2, Rfcass, RRbar]
        return as_vector(vec)

        

    def getGatingResidual(self, v, w, w_t, I_stim): # Add test functions
        Ca_i, Ca_ss, Ca_sr, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar = self.unpackGating(w)
        #Ca_it, Ca_sst, Ca_srt, Na_it, K_it, mt, ht, jt, xr1t, xr2t, xst, rt, st, dt, ft, f2t, fcasst, Rbart = self.unpackGating(w_t, total=True)
        return dot(getGatingRHS(v, w, I_stim), w_t)
        #res = self.m_HH(v, m) * mt + self.h_HH(v, h) * ht + self.j_HH(v, j) * jt
        #res += self.xr1_HH(v, xr1) * xr1t + self.xr2_HH(v, xr2) * xr2t + self.xs_HH(v, xs) * xst
        #res += self.r_HH(v, r) * rt + self.s_HH_endo(v, s) * st +  self.d_HH(v, d) * dt
        #res += self.f_HH(v, f) * ft + self.f2_HH(v, f2) * f2t + self.fcass_HH(fcass, Ca_ss) * fcasst + self.dRbardt(Rbar, Ca_ss, Ca_sr) * Rbart
        #res += self.dCa_itotaldt(v,w) * Ca_it + self.dCa_srtotaldt(v, w) * Ca_srt + self.dCa_sstotaldt(v, w) * Ca_sst 
        #res += self.dNaidt(v, w) * Na_it + self.dKidt(v, w, I_stim) * K_it
        #return res

    def RushLarsenUpdate(x, xn, x_inf, tau_x, dt):
        x.interpolate(x_inf + exp(-Constant(dt)/tau_x) * (xn - x_inf))
        
    def solveHHVariables(self, v, w, wn, dt):
        # This solves ONLY the HH variables
        Ca_i, Ca_ss, Ca_sr, Na_i, K_i, m, h, j, xr1, xr2, xs, r, s, d, f, f2, fcass, Rbar = self.unpackGating(w)
        Ca_in, Ca_ssn, Ca_srn, Na_in, K_in, mn, hn, jn, xr1n, xr2n, xsn, rn, sn, dn, fn, f2n, fcassn, Rbarn = self.unpackGating(w)
        
        self.RushLarsenUpdate(m, mn, self.m_inf, self.tau_m, dt)
        self.RushLarsenUpdate(j, jn, self.j_inf, self.tau_j, dt)
        self.RushLarsenUpdate(h, hn, self.h_inf, self.tau_h, dt)
        self.RushLarsenUpdate(xr1, xr1n, self.xr1_inf, self.tau_xr1, dt)
        self.RushLarsenUpdate(xr2, xr2n, self.xr2_inf, self.tau_xr2, dt)
        self.RushLarsenUpdate(xs, xsn, self.xs_inf, self.tau_xs, dt)
        self.RushLarsenUpdate(r, rn, self.r_inf, self.tau_r, dt)
        self.RushLarsenUpdate(s, sn, self.s_inf, self.tau_s, dt)
        self.RushLarsenUpdate(d, dn, self.d_inf, self.tau_d, dt)
        self.RushLarsenUpdate(f, fn, self.f_inf, self.tau_f, dt)
        self.RushLarsenUpdate(f2, f2n, self.f2_inf, self.tau_f2, dt)
        self.RushLarsenUpdate(fcass, fcassn, self.fcass_inf, self.tau_fcass, dt)
        self.RushLarsenUpdate(m, mn, self.m_inf, self.tau_m, dt)
        self.RushLarsenUpdate(m, mn, self.m_inf, self.tau_m, dt)
        
              
  

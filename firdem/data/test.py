from firdem.utils.PressureFunction import getPressureTemperatureFunction
import numpy as np
import matplotlib.pyplot as plt

Ts = np.linspace(273 + 28, 273 + 60)
rho = 1020 
Ps = [ getPressureTemperatureFunction(rho, T, NCASE=1) for T in Ts]
plt.plot(Ts, Ps, label='NCASE=1')
Ps = [ getPressureTemperatureFunction(rho, T, NCASE=2) for T in Ts]
plt.plot(Ts, Ps, label='NCASE=2')
Ps = [ getPressureTemperatureFunction(rho, T, NCASE=3) for T in Ts]
plt.plot(Ts, Ps, label='NCASE=3')
plt.legend()
plt.show()

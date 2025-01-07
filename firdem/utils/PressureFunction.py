import firdem
import numpy as np
import csv
import matplotlib.pyplot as plt
from sys import argv

PATH = firdem.__path__[0]

def getPressureTemperatureFunction(density, T, SIZE=1, NCASE=3):
    """
    Get Pressure function in terms of the temperature for a given density. The density must be given in [kg/m^3] and the temperature in [K]. 

    Parameters:
        density: kg/m^3 of the fluid
        SIZE: 0 for 6x6 fit, 1 for 7x7 (see notes in data folder)
        NCASE: 1|2|3 (see notes in data folder)

    Output: 
        P = sum_ij A_ij V^i T^j
    """
    assert SIZE in (0,1), "SIZE arg must be 0|1"
    assert NCASE in (1,2,3), "NCASE arg must be 1|2|3"

    mat_size = 7+SIZE
    A = np.zeros((mat_size,mat_size)) # Case 6x6
    filename = "pressure-mishima-sumita" if SIZE==0 else "pressure-mishima-sumita-large"
    with open(f"{PATH}/data/{filename}.csv") as file: 
        reader = csv.reader(file, delimiter=';', quotechar='"')
        first = True
        for row in reader:
            if first: # skip header 
                first = False
                continue
            ii = int(row[0])
            jj = int(row[1])
            A[ii,jj] = row[NCASE+1]

    def P(V,T):
        """
        V in cm^3/gr, T in Celsius
        """
        t = T/300
        p = 0 
        for i in range(mat_size):
            for j in range(mat_size):
                p += A[i,j] * V**i * t**j
        # p is in MPa, convert to Pascal
        return p * 1e3

    V = 1/(density * 1e-3) # kg/m^3 = 1e3 g / 1e6 cm^3 = 1e-3 g/cm^3
    return P(V, T)

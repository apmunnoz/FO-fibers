from firedrake import *
import numpy as np
import Fibers
import Mesh
from Units import SI

mesh_type = Mesh.PROLATE  # PROLATE|BIVENTRICLE|UNITSQUARE
out = Mesh.getMesh(mesh_type=mesh_type)
mesh = out[0]

# Rescale for SI
f, s, n = Fibers.computeFibers(mesh, mesh_type)

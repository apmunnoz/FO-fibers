from firedrake import *
from firdem.utils.Units import Units

PROLATE = 14
BIVENTRICLE = 15
UNITSQUARE = 16

def getMesh(mesh_type=PROLATE, nx=100, ny=100): 

    dxx = None

    if mesh_type == PROLATE:
        mesh = Mesh("../meshes/prolate_4mm.msh")
        dxx = dx(domain=mesh)
        ds = Measure('ds', domain=mesh)
        ds_endo = ds(20)
        ds_epi = ds(10)
        ds_base = ds(50)
        return mesh, dxx, ds_endo, ds_epi, ds_base
    elif mesh_type == BIVENTRICLE:
        mesh = Mesh("../meshes/biv_2.0mm.msh")
        dxx = dx(domain=mesh)
        ds = Measure('ds', domain=mesh)
        ds_endo_lv = ds(10)
        ds_endo_rv = ds(20)
        ds_epi = ds(30)
        ds_base = ds(50)
        return mesh, dxx, ds_endo_lv, ds_endo_rv, ds_epi, ds_base
    elif mesh_type == UNITSQUARE:
        mesh = UnitSquareMesh(nx, ny)
        dxx = dx(domain=mesh)        
        ds = Measure('ds', domain=mesh)
        # Note: tags are 1,2,3,4 (x=0, x=L, y=0, y=L)
        return mesh, dxx, ds
        

import sys
import os
import time
import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.linalg as la
import scipy.io as sio

# import matplotlib.pyplot as plt

from ngsolve import *
from netgen.geom2d import SplineGeometry

# mute messages
ngsglobals.msg_level = 1


# static functions
def func_generator(params):
    # permittivity = 1.0
    # return permittivity
    N = len(params["x"])
    
    permittivity = 1
    
    for i in range(N):
        permittivity += (IfPos((x-params["x"][i])**2 + (y-params["y"][i])**2 - params["R"] **2, 0, 1) )
    
    return permittivity

def profile(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Function {func.__name__} took {end - start} seconds")
        return result
    return wrapper

class HelmholtzSolver:
    def __init__(self, maxh = (0.01, 0.03), pml_alpha=10j, order=3):

        t1 = time.perf_counter()

        self.geo = SplineGeometry()

        self.geo.AddRectangle((-2, -2), (2, 2), leftdomain=1, rightdomain=2)
        self.geo.AddRectangle((-3, -3), (3, 3), leftdomain=2, rightdomain=0, bc="outerbnd")

        self.geo.SetMaterial(1, "inner")
        self.geo.SetMaterial(2, "pml")

        self.geo.SetDomainMaxH(1, maxh[0])
        self.geo.SetDomainMaxH(2, maxh[1]) # mesh size should match the alpha value in PML.

        self.mesh = Mesh(self.geo.GenerateMesh())
        self.mesh.SetPML(pml.Cartesian(mins=(-2, -2), maxs=(2, 2), alpha=pml_alpha) , definedon=2)

        self.fes = H1(self.mesh, order=order, complex=True)
        self.u = self.fes.TrialFunction()
        self.v = self.fes.TestFunction()

        t2 = time.perf_counter()
        print(f" Preprocessing time: {t2 - t1:.2f} seconds")
   
    def solve(self, inc_kx, inc_ky, params):

        t1 = time.perf_counter()

        k_sq = inc_kx**2 + inc_ky**2
        u_inc = CF((exp(1j * inc_kx * x) * exp(1j * inc_ky * y)))

        permittivity = func_generator(params)
        
        f = LinearForm(self.fes)
        f += k_sq * (permittivity - 1) * u_inc * self.v * dx 
        
        a = BilinearForm(self.fes)
        a += grad(self.u) * grad(self.v) * dx - k_sq * permittivity * self.u * self.v * dx

        a.Assemble()
        f.Assemble()

        u_scat = GridFunction(self.fes)
        u_scat.vec.data = a.mat.Inverse(freedofs=self.fes.FreeDofs(), inverse="sparsecholesky") * f.vec

        t2 = time.perf_counter()
        print(f" Solving time: {t2 - t1:.2f} seconds")

        return u_scat
    
    def save(self, u, string_name, filename):
        sio.savemat(filename, {string_name: u.vec.FV().NumPy()})

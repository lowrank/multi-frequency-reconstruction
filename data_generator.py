"""
data_generator.py
"""

import time
from ngsolve import *
from netgen.geom2d import SplineGeometry
import scipy.io as sio
import numpy as np

# mute messages
ngsglobals.msg_level = 0

class DataGenerator:
    """
    solve Helmholtz equation with Absorbing Boundary Conditions (ABC) using NGSolve.

    # The computational domain is [-2, 2]. 

    @param maxh: the max diameter of triangle mesh in each subdomain.
    @param pml_alpha: the pml parameter for absorbing strength, smaller value needs a larger pml size.  
    @param order: order of finite element method
    @param radius: the data receivers (Cauchy data) are placed on a circle of this radius (should be inside the domain). 
    @param num_points: number of receivers on the circle.
    """
    def __init__(self, maxh = (0.01, 0.03), pml_alpha=10j, order=3, radius=1.5, num_points=360):

        t1 = time.perf_counter()

        self.cauchy_radius = radius

        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

        self.cauchy_points = [(self.cauchy_radius * cos(phi),
                               self.cauchy_radius * sin(phi)) for phi in theta]

        self.geo = SplineGeometry()

        # [-2, 2]^2 is the interior domain (label is 1)
        # [-3, 3]^2 is the full domain including the PML (label is 2)
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
        print(f"Mesh generation took {t2 - t1} seconds")

    def solve(self, inc_kx, inc_ky, permittivity, source = None):
        """
        solve the Helmholtz equation with incident wave exp(i kx x + i ky y).

        @param inc_kx: wave number in x direction
        @param inc_ky: wave number in y direction
        @param permittivity: permittivity function (q function).
        @param source: source function (f function)

        @return: the scattered wave u_scat = u_tot - u_inc

        The Helmholtz equation is

        - Delta u_{tot} - k^2 q u_{tot} = f(x)

        or 

        - Delta u_{scat} - k^2 q u_{scat} = k^2 (q - 1) u_{inc} + f(x)

        The weak formulation writes into (v as test function)

        int_{D} ( \nabla u_{scat} cdot \nabla v - k^2 q u_{scat} v ) dx  ---> Bilinear form

            = int_{D} k^2 (q - 1) u_{inc} v dx  ---> Linear form
        """

        t1 = time.perf_counter()

        k_sq = inc_kx**2 + inc_ky**2 # frequency squared
        u_inc = CF((exp(1j * inc_kx * x) * exp(1j * inc_ky * y))) # incident wave function

        linear_form = LinearForm(self.fes)
        linear_form += k_sq * (permittivity - 1) * u_inc * self.v * dx
        if source:
            linear_form += source * self.v * dx
        linear_form.Assemble()

        a = BilinearForm(self.fes)
        a += grad(self.u) * grad(self.v) * dx - k_sq * permittivity * self.u * self.v * dx
        a.Assemble()

        u_scat = GridFunction(self.fes)
        u_scat.vec.data = a.mat.Inverse(freedofs=self.fes.FreeDofs(),
                                        inverse="sparsecholesky") * linear_form.vec

        t2 = time.perf_counter()
        print(f"Solving took {t2 - t1} seconds")

        return u_scat

    def born_solve(self, inc_kx, inc_ky, permittivity, background_permittivity):
        """
        solve the Helmholtz equation with Born approximation.

        q \approx q_0 ---> background permittivity (not necessarily constant)

        Then Born approximation solves: 

        - Delta u_{born, scat} - k^2 q_0 u_{born, scat} = k^2 (q - 1) u_{inc}

        The solution u_{born, scat} is linear in q.         
        """

        k_sq = inc_kx**2 + inc_ky**2
        u_inc = CF((exp(1j * inc_kx * x) * exp(1j * inc_ky * y)))

        linear_form = LinearForm(self.fes)
        linear_form += k_sq * (permittivity - 1) * u_inc * self.v * dx
        linear_form.Assemble()

        a = BilinearForm(self.fes)
        a += grad(self.u) * grad(self.v) * dx - \
            k_sq * background_permittivity * self.u * self.v * dx
        a.Assemble()

        u_scat = GridFunction(self.fes)
        u_scat.vec.data = a.mat.Inverse(freedofs=self.fes.FreeDofs(),
                                        inverse="sparsecholesky") * linear_form.vec

        return u_scat

    def generate_cauchy_data(self, frequency, direction_angles, permittivity):
        """
        generate Cauchy data of u on a circle of radius r (no source function)

        @param frequency: k (here it refers to wave number). 
        @param direction_angles: the directions of incident waves.
        @param permittivity: the medium permittivity q. 

        Knowing the medium q, one can generate Cauchy data (Dirichlet-Neumann pairs) on the receivers. 
        """

        cauchy_u_vals = [] # Dirchlet data
        cauchy_gu_vals = [] # Neumann data

        for angle in direction_angles:

            kx = frequency * cos(angle)
            ky = frequency * sin(angle)

            u_scat = self.solve(kx, ky, permittivity)

            u_vals = [u_scat(self.mesh(*pt)) for pt in self.cauchy_points]
            gu = grad(u_scat)

            gu_vals = [(gu(self.mesh(*pt))[0] * pt[0] \
                            + gu(self.mesh(*pt))[1] * pt[1]) / self.cauchy_radius \
                                for pt in self.cauchy_points]

            cauchy_u_vals.append(u_vals)
            cauchy_gu_vals.append(gu_vals)

        return np.array(cauchy_u_vals), np.array(cauchy_gu_vals)


    def save(self, u, string_name, filename):
        """
        save the solution to a .mat file.
        """
        sio.savemat(filename, {string_name: u.vec.FV().NumPy()})

        
        

        

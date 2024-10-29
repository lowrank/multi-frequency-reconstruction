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
    """
    def __init__(self, maxh = (0.01, 0.03), pml_alpha=10j, order=3, radius=1.5, num_points=360):

        t1 = time.perf_counter()

        self.cauchy_radius = radius

        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

        self.cauchy_points = [(self.cauchy_radius * cos(phi),
                               self.cauchy_radius * sin(phi)) for phi in theta]

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
        print(f"Mesh generation took {t2 - t1} seconds")

    def solve(self, inc_kx, inc_ky, permittivity):
        """
        solve the Helmholtz equation with incident wave exp(i kx x + i ky y).

        @param inc_kx: wave number in x direction
        @param inc_ky: wave number in y direction
        @param permittivity: permittivity function

        @return: the scattered wave u_scat = u_tot - u_inc
        """

        t1 = time.perf_counter()

        k_sq = inc_kx**2 + inc_ky**2
        u_inc = CF((exp(1j * inc_kx * x) * exp(1j * inc_ky * y)))

        linear_form = LinearForm(self.fes)
        linear_form += k_sq * (permittivity - 1) * u_inc * self.v * dx
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
        generate Cauchy data of u on a circle of radius r.
        """

        cauchy_u_vals = []
        cauchy_gu_vals = []

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

    def linearization(self, frequency, direction_angles, cauchy_data, background_u_tot, background_permittivity):
        """
        linearize the Helmholtz equation with respect to the permittivity, then solve the reconstruction by least square.
        """

        u_vals, grad_u_vals = cauchy_data
        bg_u_vals, bg_grad_u_vals = self.generate_cauchy_data(frequency, direction_angles, background_permittivity)

        diff_u_vals = u_vals - bg_u_vals
        diff_grad_u_vals = grad_u_vals - bg_grad_u_vals

        for i, dir_v in enumerate(direction_angles):
            for j, dir_u in enumerate(direction_angles):
                diff = sum( [diff_grad_u_vals[i][k] * bg_u_vals[j][k] - bg_grad_u_vals[j][k] * diff_u_vals[i][k] for k in range(len(self.cauchy_points))] ) * (2 * pi * self.cauchy_radius / len(self.cauchy_points) ) / frequency**2
                print(i,j, diff)

        
        

        

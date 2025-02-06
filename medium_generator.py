"""
medium_generator.py
"""
from ngsolve import *
import numpy as np

class MediumGenerator:
    """
    Generate a medium with parameters given in params.
    """
    def __init__(self, medium_func):
        self.func = medium_func

    def generate(self, parameters):
        """
        Generate the medium with parameters given in parameters.
        """
        medium = 1 # background value
        for param in parameters:
            medium += self.func(param)
        return medium

    def plot(self, medium):
        """
        Plot the medium.
        """
        Draw(medium)

def disk_func(disk_params):
    """
    Generate a disk with parameters given in disk_params.
    """
    return IfPos((x - disk_params['x'])**2 + (y - disk_params['y'])**2 - (disk_params['r']) **2,
                 0,
                 disk_params['v'])


def cosine_func(cos_params):
    """
    Generate a cosine function with parameters given in cos_params.
    """
    return IfPos((x - cos_params['x'])**2 + (y - cos_params['y'])**2 - (cos_params['r']) **2,
                  0,
                  0.5 * cos_params['v'] * \
                    ( 1 + cos(pi * sqrt((x - cos_params['x'])**2 + \
                                        (y - cos_params['y'])**2)/(cos_params['r'])) ))

def export_mat(func, mesh, size=(128, 128)):
    """
    Generate the mat file for the function's image. The target domain is fixed in [-1.5, 1.5] x [-1.5, 1.5].
    """
    X = np.linspace(-1.5, 1.5, num=size[0])
    Y = np.linspace(-1.5, 1.5, num=size[1])
    pts = np.meshgrid(X, Y)
    return func(mesh(pts[0].flatten(), pts[1].flatten())).reshape(size).real
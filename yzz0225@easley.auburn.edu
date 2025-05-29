#!/usr/bin/env python
# coding: utf-8

# In[59]:


# Data generation

from IPython.display import clear_output
from data_generator import DataGenerator
from medium_generator import MediumGenerator, disk_func, cosine_func, export_mat
from ngsolve import * 
from ngsolve.webgui import Draw
from numpy.linalg import lstsq
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import scipy.io as sio
from tqdm import tqdm
import scipy
import time
import os

dg = DataGenerator(maxh = (0.05, 0.2))

n_process = 42 # less than the max CPU number

sample_size = 500

if not os.path.exists('data'):
    os.makedirs('data')
    
if not os.path.exists('models'):
    os.makedirs('models')
    
if not os.path.exists('train'):
    os.makedirs('train')


# In[60]:


medium = MediumGenerator(cosine_func) # MediumGenerator(disk_func)

background_params = [{"type": 0, "x": 0.0, "y": 0.0, "r": 0.5, "v": 0.0}]
background_permittivity = medium.generate(background_params)

# create a sample with several bumps (intrinsic dimension is n_bumps * 4).

def generate_media(n_bumps=8, sample_size=500):
    """
    The medium consists of bumps with radius uniformly in [0.2, 0.4], locations uniformly in unit disk, value uniformly in [0.5, 1.5]
    """

    start_time = time.time()
    
    for sample_id in tqdm(range(sample_size)):
        values = np.random.random((n_bumps, )) * 2 + 0.5

        r, theta, radius = np.sqrt(np.random.random((n_bumps, ))),                                              np.random.random((n_bumps, 1)) * 2 * np.pi,                                              np.random.random((n_bumps, 1)) * 0.1 + 0.3

        params = []

        for i in range(n_bumps):
            params.append({"type": 0, 
                           "x": r[i] * np.cos(theta[i]),
                           "y": r[i] * np.sin(theta[i]),
                           "r": radius[i], 
                           "v": values[i]})

        permittivity  = medium.generate(params)

        permittivity_mat = export_mat(permittivity, dg.mesh)

        scipy.io.savemat('data/data_' + str(sample_id) +'.mat', {'value':permittivity_mat, 'params': params})
        
        if (sample_id + 1)% 50 == 0:
            end_time = time.time()
            print('elapsed time: {:6.2f}, progress: {:4d} / {:4d}'.format(end_time - start_time, (sample_id+1), sample_size))
            


# In[61]:


generate_media() # uncomment to generate the data.


# In[62]:


def single_loop(i, pt, bpt, freq, inc_dir, out_dir):        
    mat = []
    vec = []
    
    kx = 2 * pi * freq * cos(inc_dir[i])
    ky = 2 * pi * freq * sin(inc_dir[i])

    psi = CF((exp(1j * kx * x) * exp(1j * ky * y)))

    u_scat = dg.solve(kx, ky, pt)

    for j_angle in out_dir:
        p_kx = 2 * pi * freq * cos(j_angle)
        p_ky = 2 * pi * freq * sin(j_angle)

        phi = CF((exp(1j * p_kx * x) * exp(1j * p_ky * y)))

        true_val = Integrate( (pt - bpt) * (phi) *  (u_scat + psi) * (IfPos((x)**2 + (y)**2 - (1.5) **2, 0, 1)), dg.mesh)

        test_func = dg.fes.TestFunction()

        linear_form = LinearForm(dg.fes)

        linear_form += test_func * (phi) *  (psi) * (IfPos((x)**2 + (y)**2 - (1.5) **2, 0, 1)) * dx

        linear_form.Assemble()
            
        mat.append(linear_form.vec.FV().NumPy())
        vec.append(true_val)

    return mat, vec

def assemble_linear_sys(pt, bpt, freq, n_in_dir=32, n_out_dir=8):

    A = Matrix(2 * n_in_dir * n_out_dir, dg.fes.ndof, complex=True)
    b = Vector(2 * n_in_dir * n_out_dir, complex=True)

    # incident and test directions are aligned to maximize the captured frequency domain
    inc_dir = np.linspace(0, 2 * np.pi, n_in_dir, endpoint=False)
    out_dir = np.linspace(0, 2 * np.pi, n_out_dir, endpoint=False)
    
    with multiprocessing.Pool(n_process) as p:
        tasks = [p.apply_async(single_loop, [i, pt, bpt, freq, inc_dir, out_dir]) for i in range(n_in_dir)]

        finished = {}
        
        # postprocessing step
        while len(finished) != len(tasks):
            for i, task in enumerate(tasks):
                if task.ready():
                    finished[i] = task.get()
                    for j in range(n_out_dir):
                        index = i * n_out_dir + j
                        A.NumPy()[2 * index, :]     = finished[i][0][j].real
                        A.NumPy()[2 * index + 1, :] = finished[i][0][j].imag
                        b[2 * index]                = finished[i][1][j].real
                        b[2 * index + 1]            = finished[i][1][j].imag
                    
    return A.NumPy(),  b.NumPy()


# In[75]:


data_dir = pjoin('.', 'data')

for freq_id in range(30):

    folder_path = 'train/' + str(freq_id)

    # if directory does not exist, create it. 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cur_freq = freq_id * 0.2 # step size 0.1 in frequency domain


    for sample_id in tqdm(range(sample_size)):
        data_filename = pjoin(data_dir, 'data_'+str(sample_id)+'.mat')
        mat_contents = sio.loadmat(data_filename)
        params_config = mat_contents['params'][0]
        params_size = len(params_config)

        params = []

        for i in range(params_size):
            config = np.concatenate(params_config[i].tolist()).flatten()

            params.append({"type": config[0], 
                           "x":    config[1],
                           "y":    config[2],
                           "r":    config[3], 
                           "v":    config[4]})

        permittivity  = medium.generate(params)

        A, b = assemble_linear_sys(permittivity, background_permittivity, freq=cur_freq) # assemble

        v = lstsq(A,  b, rcond=1e-2)[0] # least square through svd.    

        permittivity_update =  GridFunction(dg.fes)
        permittivity_update.vec.data = v

        permittivity_update_mat = export_mat(permittivity_update, dg.mesh)
        scipy.io.savemat('train/' + str(freq_id) + '/train_' + str(sample_id) +'.mat', {'value':permittivity_update_mat})


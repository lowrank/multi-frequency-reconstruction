from data_generator import * 
import numpy as np

hs = HelmholtzSolver(maxh = (0.01, 0.03))


params0 = {"type": 0, "x": [0], "y": [0], "R": 0.5, "val": 0.0}
params = {"type": 0, "x": [0], "y": [0], "R": 0.5, "val": 1.0}


# exp(i pi y) 
kx = 0
ky = 0.005 * pi
u_scat = hs.solve(kx, ky, params)
uB_scat = hs.Born_solve(kx, ky, params0, params)


# hs.save(u_scat, "scatter", "data/scatter.mat")
grad_uB_scat = grad(uB_scat)

# define domain B as a disk of radius 1.5
# generate Cauchy data

r_B = 1.5
theta_B = np.linspace(0, 2* pi, 1025)[0:-1]

pts = [(r_B * cos(theta_B[i]), r_B * sin(theta_B[i])) for i in range(len(theta_B))]

uB_scat_vals = [uB_scat(hs.mesh(*pt)) for pt in pts]
grad_uB_scat_vals = [ (grad_uB_scat(hs.mesh(*pt))[0] * pt[0] + grad_uB_scat(hs.mesh(*pt))[1] * pt[1]) / r_B for pt in pts ]


u_scat_vals = [u_scat(hs.mesh(*pt)) for pt in pts]
grad_u_scat_vals = [ (grad(u_scat)(hs.mesh(*pt))[0] * pt[0] + grad(u_scat)(hs.mesh(*pt))[1] * pt[1]) / r_B for pt in pts ]

# define the test function
p_kx = 0 
p_ky = -0.005 * pi 
psi_test = CF((exp(1j * p_kx * x) * exp(1j * p_ky * y)))
grad_psi_test_X = psi_test.Diff(x)
grad_psi_test_Y = psi_test.Diff(y)

psi_vals = [psi_test(hs.mesh(*pt)) for pt in pts]
grad_psi_vals = [ (grad_psi_test_X(hs.mesh(*pt)) * pt[0] + grad_psi_test_Y(hs.mesh(*pt)) * pt[1]) / r_B  for pt in pts ]

s = sum( [grad_psi_vals[i] * uB_scat_vals[i] - grad_uB_scat_vals[i] * psi_vals[i] for i in range(len(pts))] ) * (2 * pi * r_B / len(pts) ) / (kx**2 + ky**2)

print(s)

s = sum( [grad_psi_vals[i] * u_scat_vals[i] - grad_u_scat_vals[i] * psi_vals[i] for i in range(len(pts))] ) * (2 * pi * r_B / len(pts) ) / (kx**2 + ky**2)
print(s-pi/4)
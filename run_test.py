from solver.helmholtz_solver import * 

hs = HelmholtzSolver(maxh = (0.1, 0.3))

params = {"type": 0, "x": [0], "y": [0], "R": 0.5}

u_scat = hs.solve(0, pi, params)

hs.save(u_scat, "scatter", "data/scatter.mat")

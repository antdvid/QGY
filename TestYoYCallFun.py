import numpy as np
from QgyCapFloorPricer import *


qgy = IICapFloorQgy()
k = 1
T = 1.0
Sigma = 1.01
v_y = 0.6
rho_y = -0.1
rho_ny1 = -0.1
R_y = 0.5
P0T = np.exp(-0.01 * T)
qgy.set_spherical_parameters_at(k, Sigma, v_y, rho_y, rho_ny1, R_y)
price = qgy.price_caplet_floorlet_by_qgy(k, T, 0.05, P0T, True)
print(price/P0T)

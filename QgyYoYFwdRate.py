import numpy as np
from QgySwapPricer import *
import matplotlib.pyplot as plt


qgy = IISwapQGY()
# overwrite default parameters
Sigma_Tk_y = 0.045
v_Tk_y = 0.8
rho_Tk_y = -0.5
rho_t_ny1 = -0.1
R_Tk = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) * 0.01
for R in R_Tk:
    qgy.fill_spherical_parameters(Sigma_Tk_y, v_Tk_y, rho_Tk_y, rho_t_ny1, R)
    y_tT = qgy.price_yoy_infln_fwd()
    plt.plot(qgy.Tk[1:], y_tT[1:] * 100, 'o-', label="R = {}%".format(R * 100))

plt.plot(qgy.Tk[1:], 100 * (qgy.I0_Tk[1:]/qgy.I0_Tk[:-1] - 1), 'r-o', label='Naive forward')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()

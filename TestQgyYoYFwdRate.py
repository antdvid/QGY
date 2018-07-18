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

ax = plt.subplot(111)
plt.plot(qgy.Tk[1:], 100 * (qgy.I0_Tk[1:]/qgy.I0_Tk[:-1] - 1), 'r-o', label='Naive forward')
plt.xlabel('Maturity')
plt.ylabel('YoY Inflation Forward Rate')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

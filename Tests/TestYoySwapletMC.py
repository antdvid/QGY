import Model.QgyModel as qgy
import matplotlib.pyplot as plt
import numpy as np


qgy = qgy.QgyModel()
qgy.n_per_year = 50
num_path = 10000
P_Tk = qgy.P_0T(qgy.Tk)

Sigma_Tk_y = 0.045
v_Tk_y = 0.8
rho_Tk_y = -0.5
rho_t_ny1 = -0.1
#R_Tk = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) * 0.01
R_Tk = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256]) * 0.01
count = 0
for R in R_Tk:
    mc_res = np.zeros(len(qgy.Tk))
    qgy.fill_spherical_parameters(Sigma_Tk_y, v_Tk_y, rho_Tk_y, rho_t_ny1, R)
    for k in range(num_path):
        qgy.generate_terms_structure()
        disc = qgy.D_t
        yoy = qgy.Y_Tk
        disc_price = yoy * disc
        mc_res += disc_price
        #plt.plot(qgy.Tk, disc_price/P_Tk - 1, 'g-')
    avg_price = mc_res/num_path
    count += 1
    plt.plot(qgy.Tk, avg_price/P_Tk - 1, color=[0, count/len(R_Tk), 0])

plt.show()
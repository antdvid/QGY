import numpy as np
from QgyModel import *


qgy = QgyModel()

for k in range(1,30):
    ATk_sum = np.sum(qgy.A_Tk[1:k])
    phi_n_Tk = qgy.phi_n(k)
    psi_n_Tk = qgy.psi_n()
    E_n_Tk = qgy.E_tT_simple(0,k,phi_n_Tk,psi_n_Tk)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *

win_len = 9
poly_order = 3
color=['r', 'g', 'b', 'k']
params = np.loadtxt('LpiCalibrationParams.txt')
#params = np.loadtxt('CapletCalibrationParams.txt')
Tk = np.arange(1, 31)
plt.plot(Tk, params[:, 0], 'or',label='$\Sigma_{T_k}$')
plt.plot(Tk, params[:, 1], 'og', label='$\sin v_{T_k}$')
plt.plot(Tk, params[:, 2], 'ob', label=r'$\sin \rho_{T_k}$')
plt.plot(Tk, params[:, 3], 'ok', label=r'$R_{T_k}$')

for i in range(params.shape[1]):
    plt.plot(Tk, savgol_filter(params[:, i], win_len, poly_order), color[i])

plt.legend(loc='center right')
plt.show()
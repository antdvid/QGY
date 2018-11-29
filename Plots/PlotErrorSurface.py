import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from QgyCapFloorPricer import *
from QgyVolSurface import *


def generate_volsurfaces():
    rhoNY1 = -0.1
    vol_finder = QgyVolSurface(qgy.Tk, qgy.I0_Tk)
    surfs = []
    calibrate_params = ['CapletCalibrationParams.txt', 'LpiCalibrationParams2.txt']
    for i in range(len(calibrate_params)):
        param = np.loadtxt(calibrate_params[i])
        for i in range(len(param)):
            qgy.set_sin_parameters_at(i+1, param[i][0], param[i][1], param[i][2], rhoNY1, param[i][3])
        surf = vol_finder.find_volatility_surface(strikes, qgy)
        surfs.append(surf)
    return surfs


output = 'err_surf.txt'
overwrite = False
strikes = 0.01 * np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
qgy = QgyCapFloor()

if not Path(output).is_file() or overwrite:
    surfs = generate_volsurfaces()
    err = surfs[1] - surfs[0]
    np.savetxt(output, err)
else:
    err = np.loadtxt(output)

err = np.transpose(err)
plt.imshow(err, interpolation='none')
ax = plt.gca()
plt.xlabel('Maturity [Y]')
plt.ylabel('Strikes [%]')
plt.xticks(range(err.shape[1]), qgy.Tk[1:])
plt.yticks(range(err.shape[0]), strikes)

ax.set_xticks(np.arange(.5, 30, 1), minor=True)
ax.set_yticks(np.arange(.5, 10, 1), minor=True)

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
plt.colorbar()
plt.show()
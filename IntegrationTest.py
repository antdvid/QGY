import numpy as np
from QgIntegration import *


A = [ -0.000796723437167, -0.00205162559902, 0.0, -0.006496000348338571, -5.439195953378722e-06, -2.3003896658307634]
print('test 1: plot polynomial on r and theta')
qg_int = QgIntegration(A[0], A[1], A[2], A[3], A[4], A[5])
theta = np.linspace(0, 2 * np.pi, 200)
r = np.linspace(0, 8, 200)
X, Y = np.meshgrid(theta, r)
Z = qg_int.P(X, Y)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()

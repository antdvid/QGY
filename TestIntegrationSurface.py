import numpy as np
from QgIntegration import *


A = [ 0.001, 0.01, -7e-5, -0.01, 5e-6, -0.03]
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

from Model.QgyModel import *


qgy = QgyModel()
qgy.n_per_year = 10
N = qgy.n_per_year * (qgy.n - 1)
sigma1 = np.repeat(1, N)
sigma2 = np.repeat(2, N)
rho = -0.1
dt = 1/qgy.n_per_year

num_sim = 10000
res1 = []
res2 = []

for i in range(num_sim):
    [x_n, x_y1] = qgy.generate_two_correlated_gauss(sigma1, sigma2, rho, N, dt)
    res1.append(x_n[-1])
    res2.append(x_y1[-1])

corr = np.corrcoef(res1, res2)
print("corr = ", corr)
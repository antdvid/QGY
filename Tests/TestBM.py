from Model.QgyModel import *
from scipy import stats
import seaborn as sns

qgy = QgyModel()
qgy.n_per_year = 500
N = qgy.n_per_year * (qgy.n-1)
sigma2 = []
sigma2_prime = []
dt = 1 / qgy.n_per_year
for i in range(1, len(qgy.R_Tk_y)):
    for j in range(qgy.n_per_year):
        t = qgy.Tk[i-1] + dt * (j+1)
        sigma2.append(qgy.inf_vol(t))
        sigma2_prime.append(qgy.inf_vol_prime(t))

sigma_n = np.repeat(1, N)
sigma_n_prime = np.repeat(0, N)
t = np.linspace(1, qgy.Tk[-1], qgy.n_per_year * (qgy.n-1))

dist = []
N = 100
for i in range(0, N):
    [x_n, x_y1] = qgy.generate_two_correlated_gauss(sigma_n, sigma2, qgy.rho_n_y1, (qgy.n-1) * qgy.n_per_year,
                                                 1 / qgy.n_per_year, sigma_n_prime, sigma2_prime)
    x_y2 = qgy.generate_one_gauss(sigma2, (qgy.n-1) * qgy.n_per_year, 1 / qgy.n_per_year, sigma2_prime)
    x_Tk_y1 = x_y1[::qgy.n_per_year]
    x_Tk_y2 = x_y2[::qgy.n_per_year]
    plt.subplot(1,3,1)
    plt.plot(t, x_n, 'r')
    plt.subplot(1,3,2)
    plt.plot(t, x_y1, 'b')
    plt.subplot(1,3,3)
    plt.plot(t, x_y2, 'g')

plt.show()
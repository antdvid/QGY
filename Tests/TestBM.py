from Model.QgyModel import *
from scipy import stats
import seaborn as sns

qgy = QgyModel()
qgy.n_per_year = 5000
sigma2 = np.repeat(1, qgy.n_per_year * qgy.n)
sigma_n = np.repeat(1, qgy.n_per_year * qgy.n)
t = np.linspace(0, qgy.Tk[-1], qgy.n_per_year * qgy.n)

dist = []
N = 100
for i in range(0, N):
    [x_n, x_y1] = qgy.generate_two_correlated_gauss(sigma_n, sigma2, qgy.rho_n_y1, qgy.n * qgy.n_per_year,
                                                 1 / qgy.n_per_year)
    x_y2 = qgy.generate_one_gauss(sigma2, qgy.n * qgy.n_per_year, 1 / qgy.n_per_year)
    x_Tk_y1 = x_y1[::qgy.n_per_year]
    x_Tk_y2 = x_y2[::qgy.n_per_year]
    plt.subplot(1,3,1)
    plt.plot(t, x_n, 'r')
    plt.subplot(1,3,2)
    plt.plot(t, x_y1, 'b')
    plt.subplot(1,3,3)
    plt.plot(t, x_y2, 'g')

plt.show()
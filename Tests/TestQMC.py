from Model.QgyModel import *


num_path = 1000
num_steps = 100
permut_matrix = QgyModel.generate_permutation_matrix(num_path, num_steps)
sob_seq = QgyModel.generate_sobol_squence(num_path, 2)
sigma1 = 1
sigma2 = 1
dt = 0.1
Tk = [(i+1)*dt for i in range(num_steps)]

for i in range(num_path):
    #one_path = QgyModel.generate_one_quasi_gauss(sigma, dt, i, permut_matrix, sob_seq)
    [path1, path2] = QgyModel.generate_two_correlated_quasi_gauss(sigma1, sigma2, -0.1, dt, i, permut_matrix, sob_seq)
    plt.plot(Tk, path1, 'g-')
    plt.plot(Tk, path2, 'r-')

plt.show()
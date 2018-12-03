from Model.QgyModel import *


test = [10, 20, 40, 80, 160, 320]
num_sim = 10000

for num_per_year in test:
    dt = 1/num_per_year
    qgy = QgyModel()
    qgy.n_per_year = num_per_year
    N = num_per_year * (qgy.n-1)
    sigma = np.repeat(1, N)

    res = []
    for i in range(num_sim):
        one_path = qgy.generate_one_gauss(sigma, N, dt)
        res.append(one_path[-1])
        #plt.plot(one_path, 'g-')
    #plt.show()

    var = np.var(res)
    mean = np.mean(res)

    print("num_per_year = ", num_per_year, "mean = ", mean, ", var = ", var)



import Model.QgyModel as qgy
import numpy as np
import matplotlib.pyplot as plt


def compute_expectation_analytic(qgy):
    Tk = qgy.Tk
    ans = [1]
    for k in range(1, len(Tk)):
        EtT = qgy.E_tT_simple(0, k, qgy.phi_y(k), qgy.psi_y(k))
        XtT = 1
        ans.append(EtT * XtT)
    return ans


def compute_expectation_mc(qgy, num_sim, n_per_year):
    n = qgy.n
    sigma2 = []
    dt = 1 / n_per_year
    for i in range(1, n):
        for j in range(n_per_year):
            sigma2.append(np.exp(qgy.R_Tk_y[i] * (qgy.Tk[i - 1] + dt * (j + 1))))
    sigma_n = np.repeat(1, n_per_year * (n - 1))

    phi_Tk = gen_phi_vec_list(qgy)
    psi_Tk = gen_psi_matx_list(qgy)

#try quasi monte carlo
    permut_matrix = qgy.generate_permutation_matrix(num_sim, (n-1) * n_per_year)
    sob_seq = qgy.generate_sobol_squence(num_sim, 3)
######################

    np.random.seed(seed=12345)
    ans = np.zeros(n)
    for i in range(num_sim):
        #[x_n, x_y1] = qgy.generate_two_correlated_gauss(sigma_n, sigma2, qgy.rho_n_y1, (n-1) * n_per_year, 1/n_per_year)
        #x_y2 = qgy.generate_one_gauss(sigma2, (n-1) * n_per_year, 1/n_per_year)
        [x_n, x_y1] = qgy.generate_two_correlated_quasi_gauss(sigma_n, sigma2, qgy.rho_n_y1, dt, i, permut_matrix, sob_seq, [1, 2])
        x_y2 = qgy.generate_one_quasi_gauss(sigma2, dt, i, permut_matrix, sob_seq, 0)

        x_Tk_y1 = np.concatenate([[0], x_y1[n_per_year-1::n_per_year]])
        x_Tk_y2 = np.concatenate([[0], x_y2[n_per_year-1::n_per_year]])
        x_n_Tk = np.concatenate([[0], x_n[n_per_year - 1::n_per_year]])

        one_path = np.zeros(n)
        for j in range(len(x_Tk_y1)):
            x_Tk = np.matrix([x_n_Tk[j], x_Tk_y1[j], x_Tk_y2[j]]).T
            X_Tk = qgy.Xt(x_Tk, phi_Tk[j], psi_Tk[j])
            one_path[j] = X_Tk
        #plt.plot(qgy.Tk, one_path, 'g.')
        ans += one_path
    ans /= num_sim
    ans[0] = 1
    return ans

def gen_phi_vec_list(qgy):
    ans = []
    for i in range(qgy.n):
        ans.append(qgy.phi_y(i))
    return ans

def gen_psi_matx_list(qgy):
    ans = []
    for i in range(qgy.n):
        ans.append(qgy.psi_y(i))
    return ans

def test_mc_analy_comparison():
    qgy_md = qgy.QgyModel()
    num_sim = 1000
    n_per_year = 50
    EtXT_analy = compute_expectation_analytic(qgy_md)
    EtXT_mc = compute_expectation_mc(qgy_md, num_sim, n_per_year)
    plt.plot(qgy_md.Tk, EtXT_analy, 'r--')
    plt.plot(qgy_md.Tk, EtXT_mc, 'b-o')
    plt.show()

def test_convergence():
    qgy_md = qgy.QgyModel()
    num_sim = 100
    n_per_year = 100
    EtXT_analy = compute_expectation_analytic(qgy_md)
    for i in range(1, 10):
        current_num = num_sim * i * 2
        EtXT_mc = compute_expectation_mc(qgy_md, current_num, n_per_year)
        RMSE = np.sqrt(np.sum((EtXT_analy - EtXT_mc)**2))
        print("sim_num = ", current_num, "err = ", RMSE, "N * err = ", current_num * RMSE, "sqrt(N) * err = ", np.sqrt(current_num) * RMSE)


if __name__ == "__main__":
    test_mc_analy_comparison()


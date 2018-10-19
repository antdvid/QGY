from QgIntegration import *
from QgySwapPricer import *
import numpy as np
import scipy as sp


class QgyCapFloor(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)
        self.tol = 1e-10

    def price_caplet_floorlet_by_qgy(self, k, T, K_in, P_0T, is_caplet):
        dim = 3
        if abs(K_in) > 0.5:
            print("Warning: strike = {} is too large, make sure your definition of cap is for I_k/I_k-1 - 1".format(K_in))
            raise NotImplementedError
        K = K_in+1
        # intermediate variables
        Phi_T_n = self.phi_n_at(T)
        Phi_Tk_y = self.phi_y(k)
        Psi_T_n = self.psi_n()
        Psi_Tk_y = self.psi_y(k)
        G_Tk = self.G_tT(0,k)
        G_sqrt = np.linalg.cholesky(G_Tk)
        x_t = np.zeros([3, 1])
        L_Tk = self.I0_Tk[k]/self.I0_Tk[k-1] * np.exp(self.A_Tk[k])

        # compute ND1
        M = self.M_tT(Psi_T_n, self.G_tT(k,k,T))
        M_Psi = M.dot(Psi_T_n)
        M_Phi = M.dot(Phi_T_n)
        G_Tk_1 = self.transform_G(G_Tk, M_Psi)
        G_sqrt_1 = self.transform_G_sqrt(G_sqrt, M_Psi)
        x_t_1 = self.transform_x_t(x_t, G_Tk, G_Tk_1, M_Phi)
        ND1 = self.compute_Nd(G_sqrt_1, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t_1)

        # compute ND2
        G_Tk_2 = self.transform_G(G_Tk_1, Psi_Tk_y)
        G_sqrt_2 = self.transform_G_sqrt(G_sqrt_1, Psi_Tk_y)
        x_t_2 = self.transform_x_t(x_t_1, G_Tk_1, G_Tk_2, Phi_Tk_y)
        ND2 = self.compute_Nd(G_sqrt_2, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t_2)

        swaplet_pricer = IISwapQGY()
        E0_DY = swaplet_pricer.price_swaplet_by_qgy(k-1, k, T, P_0T)
        #E0_DY = swaplet_pricer.price_yoy_swaplet_by_qgy_2(k, T, P_0T)
        #E0_DY = self.compute_discount_YTk(k, Phi_Tk_y, Psi_Tk_y, G_Tk_1, P_0T)
        #E0_DY = np.asscalar(E0_DY)

        if is_caplet:
            ans = E0_DY * ND2 - K * P_0T * ND1
        else:
            ans = -E0_DY * (1 - ND2) + K * P_0T * (1 - ND1)
        if ans < -self.tol:
            print("ans =", ans)
            print("ND1 = ", ND1, "ND2 = ", ND2)
            print("E0_DY = ", E0_DY)
            # raise NotImplementedError
        return ans

    def verify_cap_floor_swap(self, cap_price, flr_price, swaplet_price):
        error = np.array(cap_price) - np.array(swaplet_price) - np.array(flr_price)
        return error

    def compute_discount_YTk(self, k, phi, psi, G, P_0T):
        M = self.M_tT(psi, G)
        E0Tk = np.power(np.linalg.det(M), -0.5) * np.exp(0.5 * phi.T.dot(G.dot(M).dot(phi)))
        ans = P_0T * self.I0_Tk[k]/self.I0_Tk[k-1] * np.exp(self.A_Tk[k]) * E0Tk
        return ans

    def compute_Nd(self, G_Tk_sqrt, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t):
        M = G_Tk_sqrt.T.dot(Phi_Tk_y + Psi_Tk_y.dot(x_t))
        N = 0.5 * G_Tk_sqrt.T.dot(Psi_Tk_y).dot(G_Tk_sqrt)
        F = -(np.log(L_Tk/K) - Phi_Tk_y.T.dot(x_t) - 0.5 * x_t.T.dot(Psi_Tk_y.dot(x_t)))
        # if np.fabs(M[0, 0]) > self.tol or np.fabs(N[0, 0] * N[0, 1] * N[0, 2] * N[1, 0] * N[2, 0]) > self.tol:
        #     print("M = ", M)
        #     print("N = ", N)
        #     raise NotImplementedError
        A = N[1,1]
        B = N[1,2] + N[2,1]
        C = N[2,2]
        D = M[1, 0]
        E = M[2, 0]
        F = np.asscalar(F)
        int_solver = QgIntegration(A, B, C, D, E, F)
        res = int_solver.compute_gaussian_integration()
        return res


if __name__ == "__main__":
    K_cap = 0.05
    K_floor = 0.0
    pricer = QgyCapFloor()
    cap_price = []
    floor_price = []
    for k in range(1, pricer.Tk.size):
        T = pricer.Tk[k]
        P_0T = np.exp(-0.01 * T)
        price = pricer.price_caplet_floorlet_by_qgy(k, T, K_cap, P_0T, True)
        cap_price.append(price)

        price = pricer.price_caplet_floorlet_by_qgy(k, T, K_floor, P_0T, False)
        floor_price.append(price)


    print(len(cap_price))
    plt.plot(pricer.Tk[1:], cap_price, 'o-', label='caplet')
    plt.plot(pricer.Tk[1:], floor_price, 'o-', label='floorlet')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()

from QgIntegration import *
from QgySwapPricer import *
import numpy as np
import scipy as sp


class IICapFloorQgy(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)
        self.tol = 1e-10

    def price_caplet_floorlet_by_qgy(self, k, T, K, P_0T, is_caplet):
        dim = 3

        # intermediate variables
        Phi_T_n = self.phi_n_at(T)
        Phi_Tk_y = self.phi_y(k)
        Psi_T_n = self.psi_n()
        Psi_Tk_y = self.psi_y(k)
        G_Tk = self.G_tT(0,k)
        G_sqrt = sp.linalg.cholesky(G_Tk)
        x_t = np.zeros([1, 3])
        L_Tk = self.I0_Tk[k]/self.I0_Tk[k-1] * np.exp(self.A_Tk[k])

        # compute ND1
        M = self.M_tT(Psi_T_n, self.G_tT(k,k,T))
        M_Psi = M.dot(Psi_T_n)
        G_Tk_1 = self.transform_G(G_Tk, M_Psi)
        G_sqrt_1 = self.transform_G_sqrt(G_sqrt, M_Psi)
        x_t_1 = self.transofrm_x_t(x_t, G_Tk, G_Tk_1, M, Phi_T_n).T
        ND1 = self.compute_Nd(G_sqrt_1, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t_1)

        # compute ND2
        G_Tk_2 = self.transform_G(G_Tk_1, Psi_Tk_y)
        G_sqrt_2 = self.transform_G_sqrt(G_sqrt_1, Psi_Tk_y)
        x_t_2 = self.transofrm_x_t(x_t_1, G_Tk_1, G_Tk_2, np.eye(dim), Phi_Tk_y).T
        ND2 = self.compute_Nd(G_sqrt_2, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t_2)

        swaplet_pricer = IISwapQGY()
        E0_DY = swaplet_pricer.price_swaplet_by_qgy(k-1, k, T, P_0T)

        if is_caplet:
            ans = E0_DY * ND2 - K * P_0T * ND1
        else:
            ans = -E0_DY * (1 - ND2) + K * P_0T * (1 - ND1)
        if ans < 0:
            print("ans =", ans)
            print("ND1 = ", ND1, "ND2 = ", ND2)
            raise NotImplementedError
        return ans

    def compute_Nd(self, G_Tk_sqrt, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t):
        M = G_Tk_sqrt.T.dot(Phi_Tk_y.T + Psi_Tk_y.dot(x_t.T))
        N = 0.5 * G_Tk_sqrt.T.dot(Psi_Tk_y).dot(G_Tk_sqrt)
        F = np.log(L_Tk/K) - Phi_Tk_y.dot(x_t.T) - 0.5 * x_t.dot(Psi_Tk_y.dot(x_t.T))
        if np.fabs(M[0, 0]) > self.tol or np.fabs(N[0, 0] * N[0, 1] * N[0, 2] * N[1, 0] * N[2, 0]) > self.tol:
            print("M = ", M)
            print("N = ", N)
            raise NotImplementedError
        A = np.asscalar(np.real(N[1,1]))
        B = np.asscalar(np.real(N[1,2] + N[2,1]))
        C = np.asscalar(np.real(N[2,2]))
        D = np.real(np.asscalar(M[1, 0]))
        E = np.real(np.asscalar(M[2, 0]))
        F = np.real(np.asscalar(F))
        if not np.isscalar(D):
            D = np.asscalar(D)
        if not np.isscalar(E):
            E = np.asscalar(E)
        int_solver = QgIntegration(A, B, C, D, E, -F)
        res = int_solver.compute_gaussian_integration()
        return res

    def transform_G(self, G_Tk, Psi, dim = 3):
        return G_Tk.dot(np.linalg.inv(np.eye(dim) + Psi.dot(G_Tk)))

    def transform_G_sqrt(self, G_Tk_sqrt, Psi, dim = 3):
        one_plus_Gsqrt_Psi_Gsqrt = np.eye(dim) + G_Tk_sqrt.T.dot(Psi).dot(G_Tk_sqrt)
        return G_Tk_sqrt.dot(np.linalg.inv(sp.linalg.cholesky(one_plus_Gsqrt_Psi_Gsqrt)))

    def transofrm_x_t(self, x_t, G_orig, G_new, M, Phi):
        return G_new.dot(np.linalg.inv(G_orig).dot(x_t.T) - M.dot(Phi.T))


if __name__ == "__main__":
    K_cap = 1.05
    K_floor = 1.00
    pricer = IICapFloorQgy()
    cap_price = []
    floor_price = []
    Tk = []
    for k in range(1, 30):
        T = pricer.Tk[k]
        P_0T = np.exp(-0.01 * T)
        price = pricer.price_caplet_floorlet_by_qgy(k, T, K_cap, P_0T, True)
        cap_price.append(price)
        price = pricer.price_caplet_floorlet_by_qgy(k, T, K_floor, P_0T, False)
        floor_price.append(price)
        Tk.append(T)

    plt.plot(Tk, cap_price, 'o-', label='caplet')
    plt.plot(Tk, floor_price, 'o-', label='floorlet')
    plt.legend()
    plt.show()

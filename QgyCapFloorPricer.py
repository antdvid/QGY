from QgIntegration import *
from QgySwapPricer import *
import numpy as np
import scipy as sp


class IICapFloorQgy(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)
        self.tol = 1e-4

    def price_caplet_by_qgy(self, k, T, K, P_0T):
        dim = 3

        # intermediate variables
        Phi_T_n = self.phi_n_at(T)
        Phi_Tk_y = self.phi_y(k)
        Psi_T_n = np.zeros([dim,dim])
        Psi_Tk_y = self.psi_y(k)
        G_Tk = self.G_tT(0,k)
        G_sqrt = sp.linalg.sqrtm(G_Tk)
        x_t = np.zeros([1, 3])
        L_Tk = self.I0_Tk[k]/self.I0_Tk[k-1] * np.exp(self.A_Tk[k])
        print("G_Tk = ", G_Tk)
        print("G_sqrt = ", G_sqrt)
        print("x_t = ", x_t)
        print("Phi_T_n = ", Phi_T_n)
        print("Phi_Tk_y = ", Phi_Tk_y)
        print("Psi_T_n = ", Psi_T_n)
        print("Psi_Tk_y = ", Psi_Tk_y)

        # compute ND1
        M = np.linalg.inv(np.eye(dim) + Psi_T_n.dot(self.G_tT(k, k, T)))
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
        E0_DY = swaplet_pricer.price_swaplet_by_qgy(k-1, k, T)

        return E0_DY * ND2 - K * P_0T * ND1

    def compute_Nd(self, G_Tk_sqrt, Phi_Tk_y, Psi_Tk_y, L_Tk, K, x_t):
        M = G_Tk_sqrt.dot(Phi_Tk_y.T + Psi_Tk_y.dot(x_t.T))
        N = 0.5 * G_Tk_sqrt.dot(Psi_Tk_y).dot(G_Tk_sqrt)
        F = np.log(L_Tk/K) - Phi_Tk_y.dot(x_t.T) - 0.5 * x_t.dot(Psi_Tk_y.dot(x_t.T))
        print("M = ", M)
        print("N = ", N)
        print("F = ", F)
        if np.fabs(M[0]) > self.tol or np.fabs(N[0, 0] * N[0, 1] * N[0, 2] * N[1, 0] * N[2, 0]) > self.tol:
            raise NotImplementedError
        A = N[1,1]
        B = N[1,2] + N[2,1]
        C = N[2,2]
        D = np.asscalar(M[1])
        E = np.asscalar(M[2])
        F = np.asscalar(F)
        print("coeff = ", A, B, C, D, E, -F)
        int_solver = QgIntegration(A, B, C, D, E, -F)
        res = int_solver.compute_gaussian_integration()
        return res

    def transform_G(self, G_Tk, Psi, dim = 3):
        return G_Tk.dot(np.linalg.inv(np.eye(dim) + Psi.dot(G_Tk)))

    def transform_G_sqrt(self, G_Tk_sqrt, Psi, dim = 3):
        return G_Tk_sqrt.dot(np.linalg.inv(sp.linalg.sqrtm(np.eye(dim) + G_Tk_sqrt.dot(Psi).dot(G_Tk_sqrt))))

    def transofrm_x_t(self, x_t, G_orig, G_new, M, Phi):
        return G_new.dot(np.linalg.inv(G_orig).dot(x_t.T) - M.dot(Phi.T))

if __name__ == "__main__":
    T = 15
    K = 0.1
    k = 5
    P_0T = np.exp(0.02 * T)
    pricer = IICapFloorQgy()
    price = pricer.price_caplet_by_qgy(k, T, K, P_0T)
    print("cap price = ", price)
from QgyCapFloorPricer import *
import numpy as np
import scipy as sp


class NominalCapFloor(IICapFloorQgy):
    def __init__(self):
        QgyModel.__init__(self)
        self.tol = 1e-10
        self.dim = 3

    def price_caplet_floorlet_by_qg(self, i_T, i_S, P_0T, P_0S, N, K, is_caplet):
        dim = 3
        Phi_iT = self.phi_n(i_T)
        Phi_iS = self.phi_n(i_S)
        Psi_iT = self.psi_n()
        Psi_iS = self.psi_n()
        tau = self.Tk[i_S] - self.Tk[i_T]
        res = 0

        # compute E0Y
        G_TS = self.G_tT(i_T, i_S)
        M_TS = self.M_tT(Psi_iS, G_TS)
        theta_Phi_iS = M_TS * Phi_iS
        theta_Psi_iS = M_TS * Psi_iS
        coeff = P_0T/P_0S * self.E_tT_simple(0, i_T, theta_Phi_iS, theta_Psi_iS)/self.E_tT_simple(0, i_S, Phi_iS, Psi_iS)

        x_t = np.zeros([3, 1])
        G_TS_1 = self.transform_G(G_TS, theta_Psi_iS)
        Gsqrt_TS_1 = np.linalg.cholesky(G_TS_1)
        x_t_1 = self.transform_x_t(x_t, G_TS, G_TS_1, theta_Phi_iS)
        Phi_X_T1 = (np.eye(3) - M_TS).dot(Phi_iS)
        Psi_X_T1 = (np.eye(3) - M_TS).dot(Psi_iS)
        X_T_1 = self.Xt(x_t_1, Phi_X_T1, Psi_X_T1)
        E0Y = coeff * X_T_1

        G_TS_2 = self.transform_G(G_TS_1, Psi_X_T1)
        Gsqrt_TS_2 = self.transform_G_sqrt(Gsqrt_TS_1, Psi_X_T1)
        x_t_2 = self.transform_x_t(x_t_1, G_TS_1, G_TS_2, Phi_X_T1)

        # compute Nd1
        Ln = np.log(1/(1 + K * tau))
        Nd1 = self.compute_Nd(Gsqrt_TS_1, Phi_X_T1, Psi_X_T1, Ln, K, x_t_1)
        Nd2 = self.compute_Nd(Gsqrt_TS_2, Phi_X_T1, Psi_X_T1, Ln, K, x_t_2)

        res = N*P_0S*E0Y*Nd2 - N * P_0S * (1 + K * tau)*Nd1
        return res

    def compute_Nd(self, G_Tk_sqrt, Phi_Tk_y, Psi_Tk_y, Ln, K, x_t):
        M = G_Tk_sqrt.T.dot(Phi_Tk_y + Psi_Tk_y.dot(x_t))
        N = 0.5 * G_Tk_sqrt.T.dot(Psi_Tk_y).dot(G_Tk_sqrt)
        F = -(Ln - Phi_Tk_y.T.dot(x_t) - 0.5 * x_t.T.dot(Psi_Tk_y.dot(x_t)))
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
        pricer = NominalCapFloor()
        T = 30
        r = 0.01
        N = 1
        K = 0.05
        isCaplet = True
        res = []
        for i in range(0, T):
            P_0T = np.exp(-r*pricer.Tk[i])
            P_0S = np.exp(-r*pricer.Tk[i+1])
            cap_price = pricer.price_caplet_floorlet_by_qg(i, i+1, P_0T, P_0S, N, K, isCaplet)
            res.append(cap_price)

        print(res)
        plt.plot(res)
        plt.show()
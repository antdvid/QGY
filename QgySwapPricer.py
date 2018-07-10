from QgyModel import *
import numpy as np


class IISwapQGY(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)


    def price_swaplet_by_qgy(self, h, k, T):
        r = 0.02
        P_0T = np.exp(-r * T)
        sum_A = 0
        for i in range(h+1, k+1):
            sum_A += self.A_Tk[i]
        res = P_0T * self.I0_Tk[k]/self.I0_Tk[h] * np.exp(sum_A)
        G = self.G_tT(0,k,T)
        M = self.M_tT(self.psi_y_at(T), G)
        E0T = self.E_tT(self.phi_y_at(T), M, G)
        res /= E0T

        G = self.G_tT(k,k,T)
        M = self.M_tT(self.psi_y_at(T), G)
        H0 = M.dot(self.phi_n_at(T).T).T + self.phi_y(k)
        H1 = M.dot(self.psi_n_at(T).T).T + self.psi_y(k)
        E_prod = 1
        for i in range(k, h+1, -1):
            G = self.G_tT(i-1,i)
            M = self.M_tT(H1, G)
            E_prod *= self.E_tT(H0, M, G)

            # update H0, H1
            G = self.G_tT(i - 2, i - 1)
            M = self.M_tT(H1, G)
            H0 = M.dot(H0.T).T + self.phi_y(i-1)
            H1 = M.dot(H1.T).T + self.psi_y(i-1)

        res *= E_prod
        G = self.G_tT(0, h+1)
        M = self.M_tT(H1, G)
        res *= self.E_tT(H0, M, G)

        return np.asscalar(res)


if __name__ == "__main__":
    pricer = IISwapQGY()
    res = pricer.price_swaplet_by_qgy(5, 10, 15)
    print("res = ", res)

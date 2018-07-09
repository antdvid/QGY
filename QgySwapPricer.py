import QgyModel
import numpy as np


class IISwapQGY(QgyModel):
    def price_swaplet_by_qgy(self, h, k, T):
        r = 0.02
        P_0T = np.exp(-r * T)
        sum_A = 0
        for i in range(h, k):
            sum_A += self.A_Tk[i]
        res = P_0T * self.I0_Tk[self.k]/self.I0_Tk[self.h] * np.exp(self.sum_A)
        G = self.G_tT(k,k,T)
        M = self.M_tT(self.psi_y_at(T), G)
        E0T = self.E_tT(self.phi_y_at(T), M, G)
        res /= E0T

        G = self.G_tT(k,k,T)
        M = self.M_tT(self.psi_y_at(T), G)
        H0 = M.dot(self.phi_n_at(T)) + self.phi_y(k-1)
        H1 = M.dot(self.psi_n_at(T)) + self.psi_y(k-1)
        E_prod = 1
        for i in range(k, h+1, -1):
            G = self.G_tT(i-1,i)
            M = self.M_tT(H1, G)
            E_prod *= self.E_tT(H0, M, G)

            # update H0, H1
            G = self.G_tT(i - 2, i - 1)
            M = self.M_tT(H1, G)
            H0 = M.dot(H0) + self.phi_y(i-2)
            H1 = M.dot(H1) + self.psi_y(i-2)

        res *= E_prod
        G = self.G_tT(0, h+1)
        M = self.MtT(H1, G)
        res *= self.E_tT(H0, M, G)

        return res


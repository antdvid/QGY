from QgyModel import *
import numpy as np
import scipy


class IISwapQGY(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)

    def price_swaplet_by_qgy(self, h, k, T, P_0T):
        sum_A = np.sum(self.A_Tk[h+1:k+1])
        res = P_0T * self.I0_Tk[k]/self.I0_Tk[h] * np.exp(sum_A)
        G = self.G_tT(0,k,T)
        M = self.M_tT(self.psi_n_at(T), G)
        E0T = self.E_tT(self.phi_n_at(T), M, G)
        res /= E0T

        G = self.G_tT(k,k,T)
        M = self.M_tT(self.psi_n_at(T), G)
        H0 = M.dot(self.phi_n_at(T).T).T + self.phi_y(k)
        H1 = M.dot(self.psi_n_at(T).T).T + self.psi_y(k)

        for i in range(k, h+1, -1):
            res *= self.E_tT_simple(i-1, i, H0, H1)

            # update H0, H1
            G = self.G_tT(i - 1, i)
            M = self.M_tT(H1, G)
            H0 = M.dot(H0.T).T + self.phi_y(i-1)
            H1 = M.dot(H1.T).T + self.psi_y(i-1)

        res *= self.E_tT_simple(0, h+1, H0, H1)

        return np.asscalar(res)

    def price_yoy_infln_fwd(self):
        swaplet_price = [0]
        Tk = [0]

        for k in range(1, self.Tk.size):
            T = self.Tk[k]
            P_0T = self.P_0T(T)
            price = self.price_swaplet_by_qgy(k - 1, k, T, P_0T) / P_0T - 1
            swaplet_price.append(price)
            Tk.append(T)
        return np.array(swaplet_price)

def test_swaplet():
    pricer = IISwapQGY()
    swaplet_price = []
    Tk = []
    for k in range(1, 30):
        T = pricer.Tk[k]
        P_0T = pricer.P_0T(T)
        price = pricer.price_swaplet_by_qgy(k-1, k, T, P_0T)
        swaplet_price.append(price)
        Tk.append(T)

    plt.plot(Tk, np.array(swaplet_price) - 1, 'o-')
    plt.show()


if __name__ == "__main__":
    pricer = IISwapQGY()
    T = 30
    res = pricer.price_swaplet_by_qgy(2, T, T, np.exp(-0.02 * T))
    print("res = ", res)


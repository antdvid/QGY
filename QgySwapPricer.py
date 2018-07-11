from QgyModel import *
import numpy as np
import scipy


class IISwapQGY(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)

    def price_yoy_inflation_forward_by_qgy(self):
        E_tT_xTk = np.zeros(self.Tk.size)
        for k in range(self.Tk.size):
            G = self.G_tT(0, k)
            M = self.M_tT(self.psi_y(k), G)
            E_tT_xTk[k] = self.E_tT(self.phi_y(k), M, G)
        y_tT = self.I0_Tk[1:]/self.I0_Tk[:-1] * np.exp(self.A_Tk[1:]) * E_tT_xTk[1:] - 1
        y_tT = np.insert(y_tT, 0, 0)
        return y_tT

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

def test_swaplet():
    pricer = IISwapQGY()
    swaplet_price = []
    Tk = []
    for k in range(1, 30):
        T = pricer.Tk[k]
        P_0T = np.exp(-0.02 * T)
        price = pricer.price_swaplet_by_qgy(k-1, k, T)
        swaplet_price.append(price)
        Tk.append(T)

    plt.plot(Tk, np.array(swaplet_price) - 1, 'o-')
    plt.show()

def test_yoy_infln_fwd():
    pricer = IISwapQGY()
    res = pricer.price_yoy_inflation_forward_by_qgy()
    plt.plot(pricer.Tk, res)
    plt.show()

if __name__ == "__main__":
    test_yoy_infln_fwd()


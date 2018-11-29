from Model.QgyModel import *
import numpy as np

class IISwapQGY(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)

    def price_yoy_swaplet_by_qgy_2(self, k, T, P_0T):
        Phi_Tk_y = self.phi_y(k)
        Psi_T_n = self.psi_n()
        Psi_Tk_y = self.psi_y(k)
        G_Tk = self.G_tT(0,k)

        M = self.M_tT(Psi_T_n, self.G_tT(k,k,T))
        M_Psi = M.dot(Psi_T_n)
        G_Tk_1 = self.transform_G(G_Tk, M_Psi)

        ans = self.compute_discount_YTk(k, Phi_Tk_y, Psi_Tk_y, G_Tk_1, P_0T)
        return ans

    def compute_discount_YTk(self, k, phi, psi, G, P_0T):
        M = self.M_tT(psi, G)
        E0Tk = np.power(np.linalg.det(M), -0.5) * np.exp(0.5 * phi.T.dot(G.dot(M).dot(phi)))
        ans = P_0T * self.I0_Tk[k] / self.I0_Tk[k - 1] * np.exp(self.A_Tk[k]) * E0Tk
        return np.asscalar(ans)


def test_swaplet():
    pricer = IISwapQGY()
    swaplet_price = []
    swaplet_price2 = []
    Tk = []
    for k in range(1, 30):
        T = pricer.Tk[k]
        P_0T = pricer.P_0T(T)
        price = pricer.price_swaplet_by_qgy(k-1, k, T, P_0T)
        swaplet_price.append(price)
        swaplet_price2.append(pricer.price_yoy_swaplet_by_qgy_2(k, T, P_0T))
        Tk.append(T)

    plt.plot(Tk, np.array(swaplet_price) - 1, 'o-', label='original')
    plt.plot(Tk, np.array(swaplet_price2) - 1, 's-', label='new')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # pricer = IISwapQGY()
    # T = 30
    # res_list = []
    # for i in range(1, T+1):
    #     res = pricer.price_swaplet_by_qgy(0, i-1, i, np.exp(-0.01 * i))
    #     res_list.append(res)
    # print(res_list)
    test_swaplet()
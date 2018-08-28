from QgyModel import *
import numpy as np

class IISwapQGY(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)

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
    res_list = []
    for i in range(1, T+1):
        res = pricer.price_swaplet_by_qgy(0, i-1, i, np.exp(-0.01 * i))
        res_list.append(res)
    print(res_list)

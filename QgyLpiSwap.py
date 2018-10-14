from QgyModel import *


class LpiSwapQgy(QgyModel):
    def __init__(self):
        QgyModel.__init__(self)
        self.NumIters = 100

    def price_lpi_by_qgy(self, indx, floor, cap, lpi0):
        price = 0
        np.random.seed(seed=12345)
        for i in range(self.NumIters):
            self.generate_terms_structure()
            Dt = self.D_t
            Y_Tk = self.Y_Tk
            P_Tk = self.generate_nodiscount_lpi_price(Y_Tk, floor, cap, lpi0)
            price += P_Tk[indx] * Dt[indx]/Dt[0]
        price /= self.NumIters
        return price

    def generate_discount_lpi_price(self, floor, cap, lpi0):
        price = np.zeros(len(self.Tk))
        np.random.seed(seed=12345)
        for i in range(self.NumIters):
            self.generate_terms_structure()
            Dt = self.D_t
            Y_Tk = self.Y_Tk
            P_Tk = self.generate_nodiscount_lpi_price(Y_Tk, floor, cap, lpi0)
            price += P_Tk * Dt/Dt[0]
        price /= self.NumIters
        return price

    def generate_nodiscount_lpi_price(self, Y_Tk, floor, cap, lpi0):
        n = self.Y_Tk.size
        P = np.empty(n)
        P[0] = lpi0
        for i in range(1, n):
            P[i] = P[i-1] * (1 + max(floor, min(cap, Y_Tk[i] - 1)))
        return P

    @staticmethod
    def generate_lpi_swap_rate(Tk, Lpi, I0t, Dt):
        P_tT = Dt/Dt[0]
        s = (Lpi * P_tT/Lpi[0])**(1/(Tk - Tk[0])) - (I0t/I0t[0])**(1/(Tk - Tk[0]))
        return s


if __name__ == '__main__':
    pricer = LpiSwapQgy()
    price = pricer.price_lpi_by_qgy(30, 0.00, 0.05, 231.0)
    print("price = ", price)
    #pricer.price_yoy_lpi_swap_by_qgy(0.00, 100)


from QgyModel import *


class LpiSwapQgy(QgyModel):
    def price_yoy_lpi_swap_by_qgy(self, floor, cap, lpi0):
        N = 200
        price = 0
        for i in range(N):
            self.generate_terms_structure()
            t = self.t
            Dt = self.D_t
            Tk = self.Tk
            Y_Tk = self.Y_Tk
            P_Tk = self.generate_yoy_lpi_price(Y_Tk, floor, cap, lpi0)
            S_Tk = self.generate_lpi_swap_rate(self.Tk, P_Tk, self.I0_Tk)
            plt.plot(Tk, P_Tk * 100)
            plt.xlabel('Maturity')
            plt.ylabel('LPI [%]')
        plt.show()

    def generate_yoy_lpi_price(self, Y_Tk, floor, cap, lpi0):
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
    pricer.price_yoy_lpi_swap_by_qgy(0.00, 0.05, 231.0)
    #pricer.price_yoy_lpi_swap_by_qgy(0.00, 100)


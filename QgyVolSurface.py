from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt


class QgyVolSurface():
    def __init__(self, Tk, I0_Tk):
        self.I0_Tk = np.array(I0_Tk)
        self.Tk = np.array(Tk)

    def compute_fwd_call_from_abs_lognorm_vol(self, sigma, h, k, strike):
        Y0_ST = self.I0_Tk[k]/self.I0_Tk[h]
        d = np.log(Y0_ST/strike)/sigma
        res = Y0_ST * stats.norm.cdf(d + 0.5 * sigma) - strike * stats.norm.cdf(d - 0.5*sigma)
        return res

    def compute_fwd_call_from_zc_vol(self, sigma, k, strike):
        tau = self.Tk[k] - 0.0
        return self.compute_fwd_call_from_abs_lognorm_vol(sigma * tau, 0, k, np.power(1.0 + strike, tau))

    def compute_fwd_call_from_yoy_vol(self, sigma, k, strike):
        return self.compute_fwd_call_from_abs_lognorm_vol(sigma, k-1, k, 1 + strike)

    def find_yoy_vol_from_fwd_caplet_price(self, price, k, strike):
        def target(sigma):
            lgnm_price = self.compute_fwd_call_from_yoy_vol(sigma, k, strike)
            ans = lgnm_price - price
            return ans
        opt_res = opt.brentq(target, -1, 1)
        return opt_res

    def find_zc_vol_from_fwd_caplet_price(self, price, k, strike):
        def target(sigma):
            ans = self.compute_fwd_call_from_zc_vol(sigma, k, strike) - price
            return ans
        opt_res = opt.brentq(target, -1, 1)
        return opt_res


if __name__ == "__main__":
    Tk = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
         29, 30])
    I0_Tk = [1.,          1.029,       1.0562685,   1.08690029,  1.1195073,   1.15645104,
             1.19461392,  1.23403618,  1.27475937,  1.31810119,  1.36159853,  1.40857368,
             1.45787376,  1.50889934,  1.56186171,  1.61683924,  1.67423703,  1.73450957,
             1.79695191,  1.86164218,  1.9286613,   1.99462151,  2.06263811,  2.1327678,
             2.20549519,  2.28092312,  2.36303635,  2.44928718,  2.53942095,  2.63210981,
             2.72818182]
    # quotations for the Caps Zero-Coupon on the European price index (HICPx) the 16th March 2010
    cap_strikes = 0.01 * np.array([1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])
    cap_maturity = np.array([1, 3, 5, 7, 10, 12, 15, 20, 30])
    cap_price = 1e-4 * np.asmatrix([[75.18, 49.31, 30.72, 18.45, 10.88, 6.39, 3.78, 2.27, 1.38],
                                    [289.14, 196.94, 127.53, 80.76, 51.53, 33.72, 22.77, 15.85, 11.35],
                                    [516.03, 360.05, 235.23, 147.44, 91.34, 57.30, 36.86, 24.41, 16.62],
                                    [773.98, 551.12, 368.15, 234.24, 146.05, 91.61, 58.70, 38.64, 26.15],
                                    [1127.44, 823.55, 563.73, 365.59, 230.66, 145.56, 93.51, 61.60, 41.69],
                                    [1311.75, 961.99, 656.82, 421.11, 260.90, 161.38, 101.76, 65.99, 44.06],
                                    [1513.46, 1119.27, 770.33, 496.04, 306.01, 186.36, 114.56, 71.92, 46.30],
                                    [1825.27, 1367.22, 951.66, 621.59, 393.09, 248.78, 160.67, 106.64, 72.80],
                                    [2500.60, 1926.13, 1373.61, 917.37, 596.82, 392.65, 265.78, 185.70, 133.64]])
    risk_free = 0.01

    vol_solver = QgyVolSurface(Tk, I0_Tk)
    vol = np.zeros([cap_maturity.size, cap_strikes.size])
    for i in range(cap_strikes.size):
        for j in range(cap_maturity.size):
            price = cap_price[j,i]
            T = cap_maturity[j]
            strike = cap_strikes[i]
            opt_res = vol_solver.find_zc_vol_from_fwd_caplet_price(price, int(T), strike/np.exp(-risk_free * T))
            vol[j][i] = opt_res
            # print("vol", j, i, "=", T, strike, vol[j][i], 'error = ', opt_res.fun, "niter = ", opt_res.nfev)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    XX, YY = np.meshgrid(cap_strikes, cap_maturity)

    surf = ax.plot_surface(XX, YY, vol, cmap=cm.coolwarm, linewidth=0.1, rstride=1, cstride=1, antialiased=False)
    plt.show()

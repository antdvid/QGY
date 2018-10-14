from QgyLpiSwap import *
import scipy as scp
import matplotlib.pyplot as plt

tenors_intrp = np.linspace(0, 30, 31)
qgy_pricer = LpiSwapQgy()
P0t = qgy_pricer.P_0T(qgy_pricer.Tk)
cap = 0.03
floor = 0.0
N = 400
price = 0
lpi0 = qgy_pricer.I0_Tk[0]
Tk = qgy_pricer.Tk
annual_return = np.zeros(len(Tk))
swap_rate_res = np.zeros(len(Tk))
yoy_res = np.zeros(len(Tk))
lpiTk_mean = np.zeros(len(Tk))
for i in range(N):
    qgy_pricer.generate_terms_structure()
    Y_Tk = qgy_pricer.Y_Tk
    D_Tk = qgy_pricer.D_t
    lpiTk = qgy_pricer.generate_nodiscount_lpi_price(Y_Tk, floor, cap, qgy_pricer.I0_Tk[0])
    swapRate = qgy_pricer.generate_lpi_swap_rate(Tk, lpiTk, qgy_pricer.I0_Tk, qgy_pricer.D_t)
    lpiReturn = np.power(lpiTk/lpiTk[0] * D_Tk/D_Tk[0], 1/(Tk - Tk[0])) - 1.0

    plt.subplot(1,3,1)
    plt.plot(Tk, swapRate * 100, 'g-')
    plt.xlabel('Maturity')
    plt.ylabel('Swap Rate [%]')
    plt.subplot(1,3,2)
    plt.plot(Tk, Y_Tk, 'g-')
    plt.xlabel('Maturity')
    plt.ylabel('YoY')
    plt.subplot(1,3,3)
    plt.plot(Tk, lpiReturn * 100, 'g-')
    plt.xlabel('Maturity')
    plt.ylabel('Return rate [%]')

    annual_return += lpiReturn
    swap_rate_res += swapRate
    yoy_res += Y_Tk
    lpiTk_mean += lpiTk

annual_return /= N
swap_rate_res /= N
yoy_res /= N
lpiTk_mean /= N
plt.subplot(1,3,1)
plt.plot(Tk[1:], swap_rate_res[1:] * 100, 'r')
plt.plot([Tk[0], Tk[-1]], [0, 0], '--b')
plt.subplot(1,3,2)
plt.plot(Tk, yoy_res, 'r')
# plot I_0Tk/I_0Tk-1
plt.plot(Tk[1:], qgy_pricer.I0_Tk[1:]/qgy_pricer.I0_Tk[0:-1],'b')
plt.subplot(1,3,3)
plt.plot(Tk, annual_return * 100, 'r')
plt.plot(Tk, (np.power(lpiTk_mean/lpi0, 1/Tk) - 1) * 100, 'k-')
yoyFwd = np.power(qgy_pricer.I0_Tk/qgy_pricer.I0_Tk[0], 1/(Tk - Tk[0])) - 1.0
plt.plot(Tk[1:], (yoyFwd[1:]) * 100, 'b')
plt.show()

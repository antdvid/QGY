from Model.QgyLpiSwap import *


#import data
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# the following data are annual return
rpi = [3.1775, 3.2641, 3.3139, 3.3406, 3.3498, 3.3800, 3.4386, 3.4625, 3.4595, 3.4131, 3.3316, 3.2923, 3.2533, 3.2699]
lpi03 = [2.6307, 2.6344, 2.6305, 2.6234, 2.5975, 2.5858, 2.6069, 2.6260, 2.6384, 2.6471, 2.6025, 2.6438, 2.5888, 2.6368]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
lpi0i = [3.1989, 3.3109, 3.3831, 3.4330, 3.4580, 3.5622, 3.6865, 3.7760, 3.8283, 3.8344, 3.8008, 3.8058, 3.7891, 3.8329]
lpi35 = [3.5337, 3.6080, 3.6553, 3.6890, 3.7111, 3.7757, 3.8453, 3.8904, 3.9093, 3.8932, 3.8808, 3.8383, 3.8602, 3.8656]
qgy_pricer = LpiSwapQgy()
case = 3
I_Tk = np.power(np.array(rpi) * 0.01 + 1, tenor)
I_Tk_intrp = np.interp(qgy_pricer.Tk, tenor, I_Tk)
print(I_Tk_intrp)
print(qgy_pricer.I0_Tk)
lpi_quotes = [[0.0, 0.03, np.array(lpi03) * 0.01], [0.0, 0.05, np.array(lpi05) * 0.01], [0.0, 100, np.array(lpi0i) * 0.01], [0.03, 0.05, np.array(lpi35) * 0.01]]
cap = lpi_quotes[case][1]
floor = lpi_quotes[case][0]
N = 200
price = 0
lpi0 = 1
Tk = qgy_pricer.Tk
annual_return = np.zeros(len(Tk))
lpi_price = np.zeros(len(Tk))
for i in range(N):
    lpiTk = qgy_pricer.generate_discount_lpi_price(floor, cap, qgy_pricer.I0_Tk[0])
    lpiReturn = np.power(lpiTk/lpi0, 1/Tk) - 1
    plt.plot(Tk[1:], lpiReturn[1:] * 100, 'g-')
    #plt.plot(Tk[1:], Y_Tk[1:], 'g-')

    annual_return += lpiReturn
    lpi_price += lpiTk

lpi_price /= N
annual_return = np.power(lpi_price/lpi0, 1/Tk) - 1

plt.plot(Tk[1:], annual_return[1:] * 100, 'b', label='simulation return')
plt.plot(tenor, lpi_quotes[case][2] * 100, 'r', label='market return')
plt.title("lpi with floor {} and cap {}".format(lpi_quotes[case][0], lpi_quotes[case][1]))
plt.legend(loc='lower right')
plt.xlabel('Maturity [Y]')
plt.ylabel('Annual return [%]')
plt.show()
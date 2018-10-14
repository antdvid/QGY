from QgyLpiSwap import *


#params = np.loadtxt('./LpiCalibrationParams.txt')
params = np.loadtxt('./CapletCalibrationParams.txt')
print(params.shape)
qgy = LpiSwapQgy()
Ry = 1.2
rhoNY1 = -0.1
# fill in parameters
for i in range(1, len(params)):
    if len(params[i]) == 4:
        Ry = 1.2
    else:
        Ry = params[i][3]

    qgy.set_sin_parameters_at(i+1, params[i][0], params[i][1], params[i][2], rhoNY1, Ry)


# set lpi market data
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# the following data are annual return
rpi = [3.1775, 3.2641, 3.3139, 3.3406, 3.3498, 3.3800, 3.4386, 3.4625, 3.4595, 3.4131, 3.3316, 3.2923, 3.2533, 3.2699]
lpi03 = [2.6307, 2.6344, 2.6305, 2.6234, 2.5975, 2.5858, 2.6069, 2.6260, 2.6384, 2.6471, 2.6025, 2.6438, 2.5888, 2.6368]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
lpi0i = [3.1989, 3.3109, 3.3831, 3.4330, 3.4580, 3.5622, 3.6865, 3.7760, 3.8283, 3.8344, 3.8008, 3.8058, 3.7891, 3.8329]
lpi35 = [3.5337, 3.6080, 3.6553, 3.6890, 3.7111, 3.7757, 3.8453, 3.8904, 3.9093, 3.8932, 3.8808, 3.8383, 3.8602, 3.8656]
# interpolate missing data
Tk = qgy.Tk[1:]
lpi03 = np.interp(Tk, tenor, lpi03)
lpi05 = np.interp(Tk, tenor, lpi05)
lpi0i = np.interp(Tk, tenor, lpi0i)
lpi35 = np.interp(Tk, tenor, lpi35)
tenor = Tk
#
lpi03 = dict(zip(tenor, lpi03))
lpi05 = dict(zip(tenor, lpi05))
lpi0i = dict(zip(tenor, lpi0i))
lpi35 = dict(zip(tenor, lpi35))

lpi0 = 1
lpi_quotes = [[0.0, 0.03, lpi03, '(0%, 3%)'], [0.0, 0.05, lpi05, '(0%, 5%)'], [0.0, 100, lpi0i, '(0%, inf)'], [0.03, 0.05, lpi35, '(3%, 5%)']]
color = ['r', 'g', 'b', 'k']
# pricing
count = 0
for quote in lpi_quotes:
    floor = quote[0]
    cap = quote[1]
    lpi = qgy.generate_discount_lpi_price(floor, cap, lpi0)
    print(lpi)
    lpi_ret = np.power(lpi/lpi0, 1/qgy.Tk) - 1
    plt.plot(qgy.Tk[1:], lpi_ret[1:] * 100, color[count], label='simulation ({})'.format(quote[3]))
    count += 1

count = 0
for quote in lpi_quotes:
    ret = quote[2]
    plt.plot(list(ret.keys()), list(ret.values()), "o" + color[count], label='market ({})'.format(quote[3]))
    count += 1

plt.xlabel('Maturity [year]')
plt.ylabel('Annual return [%]')
plt.title('Calibration to Lpi annual return')
plt.legend(loc='center right')
plt.show()
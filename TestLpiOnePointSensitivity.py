from QgyLpiSwap import *
import scipy.optimize as opt


def target_func(input):
    # parameters to move are Sigma_Tk_y, v_Tk_y, rho_Tk_y, R_Tk_y
    params = [qgy.Sigma_Tk_y[current_year], qgy.sinV_Tk_y[current_year], qgy.sinRho_Tk_y[current_year], qgy.R_Tk_y[current_year]]
    params[test_case] = input
    Sigma_Tk_y = params[0]
    sinv_Tk_y = params[1]
    sinrho_Tk_y = params[2]
    R_Tk_y = params[3]

    floor = lpi_quote[0]
    cap = lpi_quote[1]
    lpi_ret = lpi_quote[2][current_year]
    qgy.set_sin_parameters_at(current_year, Sigma_Tk_y, sinv_Tk_y, sinrho_Tk_y, rho_n_y1, R_Tk_y)
    model_price = qgy.price_lpi_by_qgy(current_year, floor, cap, lpi0)
    annual_ret = (model_price/lpi0) ** (1/current_year) - 1
    err = annual_ret - np.array(lpi_ret) * 0.01
    print("params = ", input, "err = ", err)
    return err

#import data
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# the following data are annual return
rpi = [3.1775, 3.2641, 3.3139, 3.3406, 3.3498, 3.3800, 3.4386, 3.4625, 3.4595, 3.4131, 3.3316, 3.2923, 3.2533, 3.2699]
lpi03 = [2.6307, 2.6344, 2.6305, 2.6234, 2.5975, 2.5858, 2.6069, 2.6260, 2.6384, 2.6471, 2.6025, 2.6438, 2.5888, 2.6368]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
lpi0i = [3.1989, 3.3109, 3.3831, 3.4330, 3.4580, 3.5622, 3.6865, 3.7760, 3.8283, 3.8344, 3.8008, 3.8058, 3.7891, 3.8329]
lpi35 = [3.5337, 3.6080, 3.6553, 3.6890, 3.7111, 3.7757, 3.8453, 3.8904, 3.9093, 3.8932, 3.8808, 3.8383, 3.8602, 3.8656]
qgy = LpiSwapQgy()
lpi03 = dict(zip(tenor, lpi03))
lpi05 = dict(zip(tenor, lpi05))
lpi0i = dict(zip(tenor, lpi0i))
lpi35 = dict(zip(tenor, lpi35))

test_case = 1     #change this to test Sigma, sinv, sinrho, R
range = [[0, 1], [0, 1], [-1, 1], [1, 1.5]]
#lpi_quote = [0.0, 1000, lpi0i] #change this to test different lpi
lpi_quote = [0.0, 1000, lpi0i]
current_year = 1
rho_n_y1 = -0.1
bnds = range[test_case]
lpi0 = 1


N = 50

qgy.NumIters = 500
x = []
y = []
for i in np.arange(0, N+1):
    lower = range[test_case][0]
    upper = range[test_case][1]
    param = lower + (upper - lower)/N * i
    err = target_func(param)
    #plt.plot(rho, err, 'o-')
    x.append(param)
    y.append(err)
plt.plot(x, y, 'o-')
plt.xlabel('param {}'.format(test_case))
plt.ylabel('error')
plt.show()
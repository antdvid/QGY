from QgyLpiSwap import *
import scipy.optimize as opt
import time as time
from scipy.signal import *


def target_func(input):
    # parameters to move are Sigma_Tk_y, v_Tk_y, rho_Tk_y, R_Tk_y
    Sigma_Tk_y = input[0]
    sinv_Tk_y = input[1]
    sinrho_Tk_y = input[2]
    if len(input) == 3:
        R_Tk_y = R_Y
    else:
        R_Tk_y = input[3]

    err = 0
    for quote in lpi_quotes:
        floor = quote[0]
        cap = quote[1]
        lpi_ret = np.array(quote[2][current_year]) * 0.01
        qgy.set_sin_parameters_at(current_year, Sigma_Tk_y, sinv_Tk_y, sinrho_Tk_y, rho_n_y1, R_Tk_y)
        mdl_price = qgy.price_lpi_by_qgy(current_year, floor, cap, lpi0)
        mdl_ret = (mdl_price/lpi0) ** (1/current_year) - 1
        err += (mdl_ret - lpi_ret) ** 2
    err = np.sqrt(err/len(lpi_quotes))
    print("  params = ", input, "err = ", err)
    return err


def plot_comparison():
    count = 0
    for quote in lpi_quotes:
        plt.plot(list(quote[2].keys()), np.array(list(quote[2].values())), 'o' + color[count], label='market ' + quote[3])
        count += 1

    count = 0
    for quote in lpi_quotes:
        floor = quote[0]
        cap = quote[1]
        model_price = qgy.generate_discount_lpi_price(floor, cap, lpi0)
        annual_ret = (model_price[1:] / lpi0) ** (1 / qgy.Tk[1:]) - 1
        plt.plot(qgy.Tk[1:], annual_ret * 100, '-' + color[count], label='calibration ' + quote[3])
        count += 1

    plt.xlabel('Maturity[Year]')
    plt.ylabel('Annual return [%]')
    plt.legend(loc='center right')
    plt.show()


# import data
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# the following data are annual return
rpi = [3.1775, 3.2641, 3.3139, 3.3406, 3.3498, 3.3800, 3.4386, 3.4625, 3.4595, 3.4131, 3.3316, 3.2923, 3.2533, 3.2699]
lpi03 = [2.6307, 2.6344, 2.6305, 2.6234, 2.5975, 2.5858, 2.6069, 2.6260, 2.6384, 2.6471, 2.6025, 2.6438, 2.5888, 2.6368]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
lpi0i = [3.1989, 3.3109, 3.3831, 3.4330, 3.4580, 3.5622, 3.6865, 3.7760, 3.8283, 3.8344, 3.8008, 3.8058, 3.7891, 3.8329]
lpi35 = [3.5337, 3.6080, 3.6553, 3.6890, 3.7111, 3.7757, 3.8453, 3.8904, 3.9093, 3.8932, 3.8808, 3.8383, 3.8602, 3.8656]
qgy = LpiSwapQgy()
lpis = [lpi03, lpi05, lpi0i, lpi35]

# interpolate missing data, should be avoid
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

lpi_quotes = [[0.0, 0.03, lpi03, '(0%, 3%)'], [0.0, 0.05, lpi05, '(0%, 5%)'], [0.0, 100, lpi0i, '(0%, inf)'], [0.03, 0.05, lpi35, '(3%, 5%)']]
#lpi_quotes = [[0.03, 0.05, lpi35, '[3%, 5%]']]
color = ['r', 'g', 'b', 'k']
rho_n_y1 = -0.1
max_tenor = 30
tol = 1e-6
x0 = [0.01, 0.5, -0.5, 0.8]
#x0 = [0.01, 0.5, -0.5]
R_Y = 1.2
bnds = [[0.0, 0.1], [0.4, 0.8], [-0.8, 0.1], [0, 10.0]]
#bnds = [[0.0, 0.1], [0, 0.8], [-1, 1]]
opt_bnds = opt.Bounds([a[0] for a in bnds], [a[1] for a in bnds])
lpi0 = 1
print("year  Sigma  v  rho  R   err     niters")
for current_year in tenor:
    if current_year > max_tenor:
        break
    start_time = time.time()
    opt_res = opt.minimize(target_func, x0=x0, bounds=opt_bnds, method='L-BFGS-B', jac=None, options={'maxiter': 50, 'maxfev': 50, 'xtol':tol, 'ftol':tol, 'gtol':tol, 'tol':tol, 'adaptive':True})

    # set lower bounds for Sigma(first parameter) to make it monotonic
    bnds[0][0] = opt_res.x[0] - 5e-4
    opt_bnds = opt.Bounds([a[0] for a in bnds], [a[1] for a in bnds])

    print(opt_res)
    elapsed_time = time.time() - start_time
    x0 = opt_res.x

    # constrain the initial guess
    for i in np.arange(len(x0)):
        x0[i] = max(x0[i], bnds[i][0])
        x0[i] = min(x0[i], bnds[i][1])
    if len(x0) == 3:
        print('#{:d}\t{:7.6f}\t\t{:5.4f}\t\t{:5.4f}\t\t{:5.4f}\t\t{:2.3e}\t\t{:d}\t\t{:5.4f}s'
              .format(current_year, x0[0], x0[1], x0[2], R_Y, opt_res.fun, opt_res.nfev, elapsed_time))
    else:
        print('#{:d}\t{:7.6f}\t\t{:7.6f}\t\t{:7.6f}\t\t{:7.6f}\t\t{:2.3e}\t\t{:d}\t\t{:5.4f}s'
              .format(current_year, x0[0], x0[1], x0[2], x0[3], opt_res.fun, opt_res.nfev, elapsed_time))
    # print("current = ", current_year, 'time cost = ', elapsed_time, 's')

plot_comparison()
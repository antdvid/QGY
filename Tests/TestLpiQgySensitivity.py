from Model.QgyLpiSwap import *


#import data
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# the following data are annual return
rpi = [3.1775, 3.2641, 3.3139, 3.3406, 3.3498, 3.3800, 3.4386, 3.4625, 3.4595, 3.4131, 3.3316, 3.2923, 3.2533, 3.2699]
lpi03 = [2.6307, 2.6344, 2.6305, 2.6234, 2.5975, 2.5858, 2.6069, 2.6260, 2.6384, 2.6471, 2.6025, 2.6438, 2.5888, 2.6368]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
lpi0i = [3.1989, 3.3109, 3.3831, 3.4330, 3.4580, 3.5622, 3.6865, 3.7760, 3.8283, 3.8344, 3.8008, 3.8058, 3.7891, 3.8329]
lpi35 = [3.5337, 3.6080, 3.6553, 3.6890, 3.7111, 3.7757, 3.8453, 3.8904, 3.9093, 3.8932, 3.8808, 3.8383, 3.8602, 3.8656]

lpi03 = dict(zip(tenor, lpi03))
lpi05 = dict(zip(tenor, lpi05))
lpi0i = dict(zip(tenor, lpi0i))
lpi35 = dict(zip(tenor, lpi35))
lpi_quote = [0.0, 1000, lpi0i]
lpi0 = 1

qgy = LpiSwapQgy()

test_space = [np.linspace(0, 0.05, 10),
              np.linspace(-0.05, 0.05, 10),
              np.linspace(-0.05, 0.05, 10),
              np.linspace(0, 3, 10)]
label = ['phi_Tk_y1', 'psi_Tk_y1', 'psi_Tk_y1y2', 'R_Tk']
test_tenor = 5
params = [0, 0, 0, 0]
for i in range(4):
    params = test_space[i]
    res = []
    for param in params:
        test_param = [qgy.phi_Tk_y1[test_tenor], qgy.psi_Tk_y1[test_tenor], qgy.psi_Tk_y1y2[test_tenor], qgy.R_Tk_y[test_tenor]]
        test_param[i] = param
        qgy.set_normalized_qgy_parameters_at(test_tenor, test_param[0], test_param[1], test_param[2], test_param[3])
        lpi = qgy.price_lpi_by_qgy(test_tenor, lpi_quote[0], lpi_quote[1], lpi0)
        lpi_ret = (np.power(lpi/lpi0, 1/test_tenor) - 1) * 100
        err = lpi_ret - lpi_quote[2][test_tenor]
        res.append(err)
    print(res)
    plt.plot(params, res, label='{}'.format(label[i]))

plt.legend()
plt.show()

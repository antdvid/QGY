from QgyLpiSwap import *


def generate_sigma_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    global R_Tk_y
    Sigma_test = 0.01 * np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    sin_v_test = np.ones(Sigma_test.size) * 0.8
    sin_rho_test = np.ones(Sigma_test.size) * 0
    R_Tk_y = np.ones(Sigma_test.size) * 0
    # Sigma_test = np.array([0.005])
    # sin_v_test = np.array([0.8])
    # sin_rho_test = np.array([0])

def generate_sinv_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    global R_Tk_y
    sin_v_test = 0.01 * np.arange(0, 110, 10)
    Sigma_test = np.ones(sin_v_test.size) * 0.045
    sin_rho_test = np.ones(sin_v_test.size) * 0
    R_Tk_y = np.ones(sin_v_test.size) * 0

def generate_sinrho_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    global R_Tk_y
    sin_rho_test = 0.01 * np.array([-100, -90, -70, -50, -30, 0, 30, 50, 70, 90, 100])
    Sigma_test = np.ones(sin_rho_test.size) * 0.045
    sin_v_test = np.ones(sin_rho_test.size) * 0.8
    R_Tk_y = np.ones(Sigma_test.size) * 0

def generate_RTkY_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    global R_Tk_y
    R_Tk_y = 0.01 * np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    Sigma_test = np.ones(R_Tk_y.size) * 0.045
    sin_v_test = np.ones(R_Tk_y.size) * np.sin(0.8)
    sin_rho_test = np.ones(R_Tk_y.size) * np.sin(-0.5)

# prepare data
qgy = LpiSwapQgy()
case = 3
options = {0: generate_sigma_test_data,
           1: generate_sinv_test_data,
           2: generate_sinrho_test_data,
           3: generate_RTkY_test_data}
options[case]()

case_var_map = {0: Sigma_test,
                1: sin_v_test,
                2: sin_rho_test,
                3: R_Tk_y}

case_name_map = {0: '$\Sigma_{T_k}$',
                 1: '$\sin v_{T_k}$',
                 2: r'$\sin \rho_{T_k}$',
                 3: '$R_{T_k}^y$'}

#strikes = np.array([-0.02])
rho_n_y1 = -0.1

ax = plt.subplot(111)
num_test = Sigma_test.size
year_index = 1
lpi0 = 1
floor = 0.00
cap = 0.05
tenor = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
lpi05 = [3.1644, 3.2438, 3.2887, 3.3169, 3.3228, 3.3830, 3.4771, 3.5447, 3.5812, 3.5774, 3.5376, 3.5268, 3.5104, 3.5262]
for i in range(0, num_test):
    smile = []
    qgy.fill_sin_parameters(Sigma_test[i], sin_v_test[i], sin_rho_test[i], rho_n_y1, R_Tk_y[i])
    price = qgy.generate_discount_lpi_price(floor, cap, lpi0)
    print(price)
    annual_ret = np.power(price/lpi0, 1/qgy.Tk) - 1
    legend_string = '{} = {:2.1f}%'.format(case_name_map[case], case_var_map[case][i] * 100)
    print(legend_string)
    plt.plot(qgy.Tk[1:], annual_ret[1:] * 100, 'o-', label=legend_string)

plt.plot(tenor, lpi05, '--')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Maturity')
plt.ylabel('Annual return[%]')
plt.show()


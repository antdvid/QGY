from QgyVolSurface import *
from QgyCapFloorPricer import *
from scipy.optimize import *
from QgyVolSurface import *

def generate_sigma_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    Sigma_test = 0.01 * np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    sin_v_test = np.ones(Sigma_test.size) * 0.8
    sin_rho_test = np.ones(Sigma_test.size) * 0

def generate_sinv_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    sin_v_test = 0.01 * np.arange(0, 110, 10)
    Sigma_test = np.ones(sin_v_test.size) * 0.045
    sin_rho_test = np.ones(sin_v_test.size) * 0

def generate_sinrho_test_data():
    global Sigma_test
    global sin_v_test
    global sin_rho_test
    sin_rho_test = 0.01 * np.array([-100, -90, -70, -50, -30, 0, 30, 50, 70, 90, 100])
    Sigma_test = np.ones(sin_rho_test.size) * 0.045
    sin_v_test = np.ones(sin_rho_test.size) * 0.8

# prepare data
case = 2
options = {0: generate_sigma_test_data,
           1: generate_sinv_test_data,
           2: generate_sinrho_test_data}
options[case]()

case_var_map = {0: Sigma_test,
                1: sin_v_test,
                2: sin_rho_test}

case_name_map = {0: '$\Sigma_{T_k}$',
                 1: '$\sin v_{T_k}$',
                 2: r'$\sin \rho_{T_k}$'}

strikes = 0.01 * np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
R_Tk_y = 0.0
rho_n_y1 = -0.1

ax = plt.subplot(111)
caplet_pricer = IICapFloorQgy()
vol_surface = QgyVolSurface(caplet_pricer.Tk, caplet_pricer.I0_Tk)
num_test = Sigma_test.size
year_index = 1
P_0T = np.exp(-0.01 * caplet_pricer.Tk[year_index])
for i in range(0, num_test):
    smile = []
    caplet_pricer.fill_sin_parameters(Sigma_test[i], sin_v_test[i], sin_rho_test[i], rho_n_y1, R_Tk_y)
    for stk in strikes:
        price = caplet_pricer.price_caplet_floorlet_by_qgy(year_index, caplet_pricer.Tk[year_index], stk, P_0T, True)
        opt_res = vol_surface.find_yoy_vol_from_fwd_caplet_price(price/P_0T, year_index, stk)
        smile.append(opt_res)
        #print("     calibration_err = ", opt_res.fun)
    legend_string = '{} = {:2.1f}%'.format(case_name_map[case], case_var_map[case][i] * 100)
    print(legend_string)
    plt.plot(strikes, smile, 'o-', label=legend_string)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Strike')
plt.ylabel('Year-on-year inflation volatility')
plt.show()


from QgyVolSurface import *
from QgyCapFloorPricer import *
from scipy.optimize import *

def target_func(input):
    # parameters to move are Sigma_Tk_y, v_Tk_y, rho_Tk_y, R_Tk_y
    Sigma_Tk_y = input[0]
    v_Tk_y = input[1]
    rho_Tk_y = input[2]
    R_Tk_y = input[3]
    rho_n_y1 = -0.1
    qgy.set_spherical_parameters_at(year_index, Sigma_Tk_y, v_Tk_y, rho_Tk_y, rho_n_y1, R_Tk_y)
    model_price = qgy.price_caplet_floorlet_by_qgy(year_index, Tk[year_index], cap_strike, P_0T, True)
    fwd_model_price = model_price/P_0T
    im_vol, opt_res = volsurf.find_yoy_vol_from_fwd_caplet_price(fwd_model_price, Tk[year_index], cap_strike)
    global nit
    nit = opt_res.iterations
    return (im_vol - vol_mkt[year_index])**2


# prepare data
price = 0.01 * np.array([0., 0.1, 0.2, 0.66, 1.33, 2, 2.6, 3.2, 3.8, 4, 4.6, 5.2, 5.8, 6, 6.6, 7, 7.6, 7.8, 8, 8.3, 8.6, 9, 9.2,
                         9.4, 9.8, 9.9, 10, 10.1, 10.6, 10.8, 11])
Tk = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
         29, 30])
I0_Tk = [1.,          1.029,       1.0562685,   1.08690029,  1.1195073,   1.15645104,
         1.19461392,  1.23403618,  1.27475937,  1.31810119,  1.36159853,  1.40857368,
         1.45787376,  1.50889934,  1.56186171,  1.61683924,  1.67423703,  1.73450957,
         1.79695191,  1.86164218,  1.9286613,   1.99462151,  2.06263811,  2.1327678,
         2.20549519,  2.28092312,  2.36303635,  2.44928718,  2.53942095,  2.63210981,
         2.72818182]

volsurf = QgyVolSurface(Tk, I0_Tk)
cap_strike = 0.05
risk_free = 0.01
vol_mkt = []
for k in range(price.size):
    P_0T = np.exp(-risk_free * Tk[k])
    vol, opt_res = volsurf.find_yoy_vol_from_fwd_caplet_price(price[k]/P_0T, k, cap_strike)
    vol_mkt.append(vol)
    print("market vol = ", vol, "niter = ", opt_res.iterations)

# do calibration
qgy = IICapFloorQgy()
year_index = 1
x0 = np.array([0.01, 0.6, -0.1, 0.5])
bnds = ((None, None), (0, np.pi/2), (-np.pi/2, np.pi/2), (0, None))
md_price_series = []
md_time_series = []
err = 0
nit = 0
print("year  Sigma  v  rho  R   err     niters")
for year_index in range(1, Tk.size):
    P_0T = np.exp(-risk_free * Tk[year_index])
    #try:
    opt_res = minimize(target_func, x0, method='L-BFGS-B', bounds=bnds)
    x0 = opt_res.x
    print('{:d}\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3e}\t\t{:d}'.format(year_index, x0[0], x0[1], x0[2], x0[3], opt_res.fun, opt_res.nfev))
    model_price = qgy.price_caplet_floorlet_by_qgy(year_index, Tk[year_index], cap_strike, P_0T, True)
    md_price_series.append(model_price)
    md_time_series.append(Tk[year_index])
    # except:
    #     print("warning: cannot find suitable parameters")
    #     print("parameters = ", qgy.Sigma_Tk_y[year_index], qgy.sinRho_Tk_y[year_index], qgy.sinV_Tk_y[year_index], qgy.R_Tk_y[year_index])

plt.plot(md_time_series, md_price_series, 'o', label='Model Price')
plt.plot(Tk, price, 'r--', label='Market price')
plt.xlabel('Maturity')
plt.ylabel('5% Cap')
plt.legend(loc='upper left')
plt.show()
from QgyCapFloorPricer import *
from QgyVolSurface import *


# import data
params = np.loadtxt('./LpiCalibrationParams.txt')
#params = np.loadtxt('./CapletCalibrationParams.txt')
print(params.shape)
qgy = IICapFloorQgy()
Ry = 1.2
rhoNY1 = -0.1
# fill in parameters
for i in range(1, len(params)):
     qgy.set_sin_parameters_at(i+1, params[i][0], params[i][1], params[i][2], rhoNY1, Ry)

# plot vol surface
strikes = 0.01 * np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
maturity = qgy.Tk
vol_finder = QgyVolSurface(qgy.Tk, qgy.I0_Tk)
v_surf = np.zeros([maturity.size-1, strikes.size])

for k in range(1, maturity.size):
    P_0T = qgy.P_0T(maturity[k])
    print(k)
    for j in range(strikes.size):
        price = qgy.price_caplet_floorlet_by_qgy(k, maturity[k], strikes[j], P_0T, True)
        vol_res = vol_finder.find_yoy_vol_from_fwd_caplet_price(price/P_0T, k, strikes[j])
        v_surf[k-1][j] = vol_res[0]

print(v_surf.shape)
for i in range(strikes.size):
    print(",".join(format(x, "10.3f") for x in v_surf[:, i]))

XX, YY = np.meshgrid(strikes, maturity[1:])
fig = plt.figure()
ax = fig.gca(projection='3d')

plt.figure(1)
surf = ax.plot_surface(XX, YY, v_surf, cmap=cm.coolwarm, linewidth=0.1, rstride=1, cstride=1, antialiased=False)
plt.xlabel('strikes')
plt.ylabel('maturity')

plt.figure(2)
N = v_surf.shape[0]
for i in range(1, N+1):
    plt.plot(strikes, v_surf[i-1, :], color=[0, (i-1)/N, 0])
plt.xlabel('strikes')
plt.ylabel('volatility')
plt.show()
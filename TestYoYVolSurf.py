from QgyCapFloorPricer import *
from QgyVolSurface import *


qgy = IICapFloorQgy()
strikes = 0.01 * np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
maturity = qgy.Tk
vol_surface = QgyVolSurface(qgy.Tk, qgy.I0_Tk)
v_surf = np.zeros([maturity.size-1, strikes.size])

for k in range(1, maturity.size):
    P_0T = np.exp(-0.01 * maturity[k])
    print(k)
    for j in range(strikes.size):
        price = qgy.price_caplet_floorlet_by_qgy(k, maturity[k], strikes[j], P_0T, True)
        vol_res = vol_surface.find_yoy_vol_from_fwd_caplet_price(price/P_0T, k, strikes[j])
        v_surf[k-1][j] = vol_res


XX, YY = np.meshgrid(strikes, maturity[1:])
fig = plt.figure()
ax = fig.gca(projection='3d')

print(XX.shape, YY.shape, v_surf.shape)
surf = ax.plot_surface(XX, YY, v_surf, cmap=cm.coolwarm, linewidth=0.1, rstride=1, cstride=1, antialiased=False)
plt.xlabel('strikes')
plt.ylabel('maturity')
plt.show()
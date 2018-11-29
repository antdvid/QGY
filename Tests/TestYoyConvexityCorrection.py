from Model.QgyModel import *

qgy = QgyModel()
ytT = qgy.price_yoy_infln_fwd()
I0_Tk_corr = qgy.fit_yoy_convexity_correction(qgy.I0_Tk)

plt.subplot(1,2,1)
plt.plot(qgy.Tk[1:], -0.00012662 * np.log(qgy.Tk[1:]), '-')
plt.plot(qgy.Tk[1:], ytT[1:] - (qgy.I0_Tk[1:]/qgy.I0_Tk[:-1] - 1), 'or')

plt.subplot(1,2,2)
plt.plot(qgy.Tk[1:], ytT[1:] - (I0_Tk_corr[1:]/I0_Tk_corr[:-1] - 1), 'or')
print(qgy.I0_Tk)
print(I0_Tk_corr)
plt.show()
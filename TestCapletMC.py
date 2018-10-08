import numpy as np
from QgyModel import *


qgy = QgyModel()
qgy.generate_terms_structure()
Tk = qgy.Tk
num_iters = 1000
strike = 0.05

avg = np.zeros(len(Tk))
for i in range(num_iters):
    qgy.generate_terms_structure()
    Y_Tk = qgy.Y_Tk
    D_Tk = qgy.D_t
    caplet = np.maximum(0.0, Y_Tk - (strike+1))* D_Tk/D_Tk[0]
    plt.plot(Tk, caplet, 'g')
    avg += caplet

avg /= num_iters
plt.plot(Tk, avg)
plt.show()
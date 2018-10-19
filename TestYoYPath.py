from QgyModel import *

qgy = QgyModel()
x1 = [-272797080.58203894] * 30
x2 = [-104016145.57136315] * 30
y = qgy.generate_yoy_structure_from_drivers(x1, x2)
print("y = ", y)
I_Tk = 1.61683924
I_Tk_1 = 1.56186171
A = 0.018957673336303969
x1 = -272797080.58203894
x2 = -104016145.57136315
phi = -5.053815176252262E-10
psi1 = 1.7120046393149053E-17
psi12 = 1.0861009779790263E-17
exponent = A - (phi + 0.5 * psi1 * x1 + psi12 * x2) * x1
print(exponent)
print(A)
print("phi = ", phi)
print(psi1 * 0.5 * x1)
print(phi + 0.5 * psi1 * x1)
print(phi + 0.5 * psi1 * x1 + psi12 * x2)

print(I_Tk/I_Tk_1 * np.exp(exponent))
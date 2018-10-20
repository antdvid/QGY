from QgyModel import *

qgy = QgyModel()
x1 = [-272797080.58203894] * 30
x2 = [-104016145.57136315] * 30
y = qgy.generate_yoy_structure_from_drivers(x1, x2)
print("y = ", y)
I_Tk = 1.6605024038790817
I_Tk_1 = 1.6072677347677349
A = 0.018957673336304
x1 = -272797080.582039
x2 =  -104016145.571363
phi = -5.05381517625226E-10
psi1 = 1.71200463931491E-17
psi12 = 1.08610097797903E-17
exponent = A - (phi + 0.5 * psi1 * x1 + psi12 * x2) * x1

print(I_Tk/I_Tk_1 * np.exp(exponent))
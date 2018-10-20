import numpy as np
import matplotlib.pyplot as plt


def compute_lpi_from_yoy(yoys):
    res = 1
    lpi = []
    for y in yoys:
        rate = np.minimum(cap, np.maximum(floor, y-1)) + 1
        res *= rate
        lpi.append(res)
    return lpi

outDir = 'D:/Results/'
fnames = [outDir + 'discProcess.txt', outDir + 'yoyProcess.txt']
disc = np.loadtxt(outDir + 'discProcess.txt')
yoy = np.loadtxt(outDir + 'yoyProcess.txt')
x_n1 = np.loadtxt(outDir + 'x_n1.txt')
x_y1 = np.loadtxt(outDir + 'x_y1.txt')
x_y2 = np.loadtxt(outDir + 'x_y2.txt')
floor = 0.0
cap = 0.03
i = 0
Tk = range(1, 31)
num_sim = yoy.shape[0]
num_ten = yoy.shape[1]
disc_avg = np.zeros(num_ten)
yoy_avg = np.zeros(num_ten)
lpi_avg = np.zeros(num_ten)

for i in range(num_sim):
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.plot(Tk, disc[i], 'g-')
    disc_avg += disc[i]
    plt.subplot(1,3,2)
    plt.plot(Tk, yoy[i], 'g-')
    yoy_avg += yoy[i]
    plt.subplot(1,3,3)
    lpi = compute_lpi_from_yoy(yoy[i])
    lpi_avg += lpi
    plt.plot(Tk, lpi, 'g-')

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.plot(Tk, x_n1[i,:], 'g-')
    plt.subplot(1,3,2)
    plt.plot(Tk, x_y1[i,:], 'g-')
    plt.subplot(1,3,3)
    plt.plot(Tk, x_y2[i,:], 'g-')

plt.figure(3)
plt.hist(x_y1[:,0], 50)

disc_avg /= num_sim
yoy_avg /= num_sim
lpi_avg /= num_sim

plt.figure(1)
plt.subplot(1,3,1)
plt.plot(Tk, disc_avg, 'r-')
plt.subplot(1,3,2)
plt.plot(Tk, yoy_avg, 'r-')
plt.subplot(1,3,3)
plt.plot(Tk, lpi_avg, 'r-')
plt.show()
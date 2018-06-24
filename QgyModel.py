import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class QgyModel:
    def __init__(self):
        #preset parameters
        self.R_Tk_y = -0.01 * np.array([0.85, 1.12, 0.15, 0.13, 0.19, 0.31, 0.39, 0.40, 0.55, 0.44, 0.39, 0.40, 0.54, 0.52, 0.47, 0.47, 0.28, 0.65])
        self.Phi_G =   0.01 * np.array([0.66, 1.58, 2.35, 2.81, 3.05, 3.33, 3.48, 3.59, 3.62, 3.66, 3.68, 3.72, 3.72, 3.78, 3.80, 3.83, 4.05, 3.84])
        self.v_Tk_y = -0.01 * np.array([13.68,22.53,6.77, 5.40, 6.46, 7.76, 8.25, 8.71, 11.97, 9.04, 9.24, 9.77,12.19,11.20,9.58,12.13, 6.45, 13.88])
        self.rho_Tk_y = 0.01 *np.array([0.0,  9.9, 13.4, 53.7, 43.0, 57.8, 52.9, 58.8, 56.8, 58.3, 61.1, 62.8, 60.2, 61.4, 61.5, 60.9, 62.2, 59.7])
        self.n = self.R_Tk_y.size

        self.n_per_year = 100
        self.rho_n_y1 = 0.6

        # n+1 points to deal with recurrence from k to k-1,
        # add a ghost point for T0 = 0
        self.I0_Tk = np.array([1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18])
        self.Tk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30])
        self.G_Tk_y1 = None
        self.G_Tk_y2 = None
        self.G_Tk_ny1 = None
        self.psi_Tk_y1y2 = None
        self.psi_Tk_y1 = None
        self.phi_Tk_y1 = None
        self.phi_Tk_n1 = np.ones(self.n+1) * 0.02
        self.A_tk = None

    def computeG_Tk_y(self):
        #integrate (sigma)^2 from 0 to t
        sigma = np.exp(self.R_Tk_y * self.Tk[1:])
        self.G_Tk_y1 = self.G_Tk_y2 = np.concatenate([[0], np.cumsum(sigma * sigma * (self.Tk[1:] - self.Tk[0:-1]))])

    def computeG_Tk_ny(self):
        #integrate (sigma)^2 from 0 to t
        sigma = np.exp(self.R_Tk_y * self.Tk[1:])
        self.G_Tk_ny1 = np.concatenate([[0], np.cumsum(self.rho_n_y1 * sigma * (self.Tk[1:] - self.Tk[:-1]))])

    def computePsi_Tk_y1y2(self):
        self.psi_Tk_y1y2 = np.concatenate([[0], self.v_Tk_y * np.sqrt(1 - self.rho_Tk_y * self.rho_Tk_y) / np.sqrt(self.G_Tk_y1[1:] * self.G_Tk_y2[1:])])

    def computePsi_Tk_y1(self):
        self.psi_Tk_y1 = np.concatenate([[0], -2 * np.abs(self.v_Tk_y) * self.rho_Tk_y/self.G_Tk_y1[1:]])


    def computePhi_tk_y(self):
        self.phi_Tk_y1 = np.concatenate([[0], self
                                        .Phi_G * np.sqrt(self.G_Tk_y1[1:])])

    def computeATk(self):
        #return is a vector on Tk
        n = self.Tk.size
        if self.A_tk == None:
            self.A_tk = np.empty(n)

        A_sum = 0
        for k in range(1, n):
            phi_n = np.array([self.phi_Tk_n1[k], 0, 0])
            psi_n = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            G = self.G_tT(0, k)
            M = self.M_tT(psi_n, G)
            E_0Tk = self.E_tT(phi_n, M, G)

            H0 = np.array([self.phi_Tk_n1[k], self.phi_Tk_y1[k], 0])
            H1 = np.array([[0, 0, 0],[0, self.psi_Tk_y1[k], self.psi_Tk_y1y2[k]],[0, self.psi_Tk_y1y2[k], 0]])
            E_prod = 1
            for i in range(k, 0, -1):
                G = self.G_tT(i-1, i)
                M = self.M_tT(H1, G)
                E_prod *= self.E_tT(H0, M, G)
                #update H
                phi_y = np.array([self.phi_Tk_n1[i-1], 0, 0])
                psi_y = np.array([[0, 0, 0], [0, self.psi_Tk_y1[i-1], self.psi_Tk_y1y2[i-1]], [0, self.psi_Tk_y1y2[i-1], 0]])
                H0 += M.dot(H0) + phi_y
                H1 += M.dot(H1) + psi_y
                print("k = ", k, "i = ", i)
                print(E_prod)
                print(H1)
            self.A_tk[k] = np.log(E_0Tk / E_prod) - A_sum
            A_sum += self.A_tk[k]

    def E_tT(self, phi, M, G):
        ans = np.power(np.linalg.det(M), 0.5) * np.exp(0.5 * np.dot(G.dot(M).dot(phi), phi))
        return ans

    def M_tT(self, psi, G):
        return np.linalg.inv(np.eye(3,3) + psi.dot(G))

    def G_tT(self, i, k):
        G_tT_ny1 = self.G_Tk_ny1[k]-self.G_Tk_ny1[i]
        G_t_y1 = self.G_Tk_y1[k] - self.G_Tk_y1[i]
        return np.array([[self.Tk[k] - self.Tk[i], G_tT_ny1, 0],
                        [G_tT_ny1, G_t_y1, 0],
                        [0, 0, G_t_y1]])

    def computeYtkDtk(self):
        sigma1 = 0.2
        sigma2 = 0.2
        [x_n, x_y1] = self.generate_two_correlated_gauss(sigma1, sigma2, self.rho_n_y1, self.n * self.n_per_year, 1/self.n_per_year)
        x_y2 = self.generate_one_gauss(sigma2, self.n * self.n_per_year, 1/self.n_per_year)

        x_Tk_y1 = x_y1[::self.n_per_year]
        x_Tk_y2 = x_y2[::self.n_per_year]

        print(self.A_tk)
        self.Y_Tk = self.I0_Tk[1:]/self.I0_Tk[0:-1] * np.exp(self.A_tk[1:] - (self.phi_Tk_y1[1:]
                                                                    + 0.5 * self.psi_Tk_y1[1:] * x_Tk_y1
                                                                    + self.psi_Tk_y1y2[1:] * x_Tk_y2) * x_Tk_y1)

    def doSimulation(self):
        #initialize, setup parameters
        self.computeG_Tk_ny()
        self.computeG_Tk_y()
        self.computePhi_tk_y()
        self.computePsi_Tk_y1()
        self.computePsi_Tk_y1y2()
        self.computeATk()

        #start simulation
        num_iters = 1
        for i in range(num_iters):
            self.computeYtkDtk()
            plt.plot(qgy.Tk[1:], qgy.Y_Tk)
        plt.show()

    @staticmethod
    def generate_two_correlated_gauss(sigma1, sigma2, rho, n, dt):
        dw1 = norm.rvs(size=n, scale=dt)
        dw2 = norm.rvs(size=n, scale=dt)

        dx1 = dw1 * sigma1
        dx2 = dw1 * sigma2 * rho + dw2 * np.sqrt(1 - rho * rho) * sigma2

        x1 = np.cumsum(dx1, axis=-1)
        x2 = np.cumsum(dx2, axis=-1)

        return [x1, x2]

    @staticmethod
    def generate_one_gauss(sigma, n, dt):
        dw = norm.rvs(size=n, scale=dt)
        dx = dw * sigma
        x = np.cumsum(dx, axis=-1)
        return x


if __name__ == "__main__":
    qgy = QgyModel()
    qgy.doSimulation()
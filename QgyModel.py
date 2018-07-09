import numpy as np
from scipy.stats import norm
import scipy.interpolate.interp1d as intrp
import matplotlib.pyplot as plt

class QgyModel:
    def __init__(self):
        #preset parameters
        self.Sigma_Tk_y = 0.01 * np.array([1.80, 2.72, 3.39, 3.84, 4.16, 4.43, 4.57, 4.48, 4.64, 4.66, 4.63, 4.61, 4.61, 4.60, 4.58, 4.58, 4.54, 4.51, 4.43, 4.38, 4.46, 4.47, 4.47, 4.46, 4.45, 4.60, 4.56, 4.56, 4.57, 4.60])
        self.sinV_Tk_y = -0.01 * np.array([64.9, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
        self.sinRho_Tk_y = -0.01 * np.array([11.7, 15.4, 21.3, 25.5, 32.9, 36.2, 39.5, 41.1, 47.0, 49.7, 53.1, 55.2, 57.7, 59.6, 61.9, 64.7, 65.4, 65.7, 65.0, 64.9, 62.6, 62.2, 62.6, 63.2, 63.6, 68.7, 69.0, 69.6, 70.5, 74.0])
        self.K_Tk_y = np.sqrt(np.square(self.sinV_Tk_y * self.sinRho_Tk_y) + 1)
        self.R_Tk_y = 0.01 * np.array([0.00, 59.2, 76.7, 89.5, 97.8, 104.4, 109.2, 112.2, 114.7, 116.7, 117.9, 119.1, 119.7, 120.5, 121.0, 121.0, 121.4, 121.5, 121.7, 121.8, 121.6, 121.8, 121.9, 122.1, 122.3, 122.6, 123.0, 123.4, 123.8, 124.2])
        self.n = self.R_Tk_y.size

        self.n_per_year = 100
        self.rho_n_y1 = -0.1
        self.psi_Tk_y1y2 = None
        self.psi_Tk_y1 = None
        self.G_Tk_y1 = None
        self.G_Tk_y2 = None
        self.phi_Tk_y1 = None
        self.phi_Tk_n1 = np.ones(self.n) * 0.02
        self.A_tk = None

        #n+1 points to deal with n and n-1
        self.I0_Tk = np.array([1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.30])
        self.Tk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        self.G_Tk_y1 = None  # integration, additional zero for the first point
        self.G_Tk_y2 = None  # integration, additional zero for the first point
        self.G_Tk_ny1 = None  # integration, additional zero for the first point

    def computeG_Tk_y(self):
        #integrate (sigma)^2 from 0 to t
        sigma = np.exp(self.R_Tk_y * self.Tk[1:])
        self.G_Tk_y1 = self.G_Tk_y2 = np.concatenate([[0], np.cumsum(sigma * sigma * np.diff(self.Tk))])

    def computeG_Tk_ny(self):
        #integrate (sigma)^2 from 0 to t
        sigma = np.exp(self.R_Tk_y * self.Tk[1:])
        self.G_Tk_ny1 = np.concatenate([[0], np.cumsum(self.rho_n_y1 * sigma * np.diff(self.Tk))])

    def computePsi_Tk_y1y2(self):
        cosRho_Tk_y = np.sqrt(1 - np.square(self.sinRho_Tk_y))
        psi_tilde =  - self.Sigma_Tk_y/ self.K_Tk_y * self.sinV_Tk_y * cosRho_Tk_y
        self.psi_Tk_y1y2 = - psi_tilde / np.sqrt(self.G_Tk_y1[1:] * self.G_Tk_y2[1:])

    def computePsi_Tk_y1(self):
        psi_tilde = self.Sigma_Tk_y/self.K_Tk_y * self.sinV_Tk_y * self.sinRho_Tk_y
        self.psi_Tk_y1 = -2 * psi_tilde / self.G_Tk_y1[1:]

    def computePhi_tk_y(self):
        cosV_Tk_y = np.sqrt(1 - np.square(self.sinV_Tk_y))
        phi_tilde = self.Sigma_Tk_y/self.K_Tk_y * cosV_Tk_y
        self.phi_Tk_y1 = -phi_tilde/np.sqrt(self.G_Tk_y1[1:])

    def phi_y(self, i):
        return np.array([[0, self.phi_Tk_y1[i], 0]])

    def phi_y_at(self, t):
        return np.array([[0, self.phi_Tk_y1_intrp(t), 0]])

    def phi_n_at(self, t):
        return np.array([[self.phi_Tk_n1_intrp(t), 0, 0]])

    def psi_y(self, i):
        return np.array([[0, 0, 0],[0, self.psi_Tk_y1[i], self.psi_Tk_y1y2[i]], [0, self.psi_Tk_y1y2[i], 0]])

    def psi_y_at(self, t):
        return np.array([[0, 0, 0], [0, self.psi_Tk_y1(t), self.psi_Tk_y1y2(t)], [0, self.psi_Tk_y1y2(t), 0]])

    def computeATk(self):
        n = self.Tk.size
        if self.A_tk is None:
            self.A_tk = np.empty(n-1)

        A_sum = 0
        for k in range(1, n):
            phi_n = np.array([self.phi_Tk_n1[k-1], 0, 0])
            psi_n = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            G = self.G_tT(0, k)
            M = self.M_tT(psi_n, G)
            E_0Tk = self.E_tT(phi_n, M, G)

            H0 = np.array([self.phi_Tk_n1[k-1], self.phi_Tk_y1[k-1], 0])
            H1 = np.array([[0, 0, 0],[0, self.psi_Tk_y1[k-1], self.psi_Tk_y1y2[k-1]], [0, self.psi_Tk_y1y2[k-1], 0]])
            E_prod = 1
            for i in range(k, 0, -1):
                G = self.G_tT(i - 1, i)
                M = np.linalg.inv(np.eye(3) + H1.dot(G))
                Theta0 = M.dot(H0)
                Theta1 = M.dot(H1)
                H0 = Theta0 + np.array([self.phi_Tk_n1[i - 1], self.phi_Tk_y1[i - 1], 0])
                H1 = Theta1 + np.array([[0, 0, 0], [0, self.psi_Tk_y1[i - 1], self.psi_Tk_y1y2[i - 1]],
                                        [0, self.psi_Tk_y1y2[i - 1], 0]])
                E_prod *= np.sqrt(np.linalg.det(M)) * np.exp(0.5 * H0.dot(G.dot(M.dot(H0))))

            self.A_tk[k-1] = np.log(E_0Tk / E_prod) - A_sum
            A_sum += self.A_tk[k-1]

    def computeYtkDtk(self):
        sigma1 = 0.1
        sigma1 = np.repeat(sigma1, self.n_per_year * self.n)
        sigma2 = sigma1

        [x_n, x_y1] = self.generate_two_correlated_gauss(sigma1, sigma2, self.rho_n_y1, self.n * self.n_per_year, 1/self.n_per_year)
        x_y2 = self.generate_one_gauss(sigma2, self.n * self.n_per_year, 1/self.n_per_year)

        x_Tk_y1 = x_y1[::self.n_per_year]
        x_Tk_y2 = x_y2[::self.n_per_year]

        self.Y_Tk = self.I0_Tk[1:]/self.I0_Tk[0:-1] * np.exp(self.A_tk - (self.phi_Tk_y1
                                                                    + 0.5 * self.psi_Tk_y1 * x_Tk_y1
                                                                    + self.psi_Tk_y1y2 * x_Tk_y2) * x_Tk_y1)
        self.t = np.linspace(0, self.Tk[-1], self.n * self.n_per_year)
        r = 0.02
        P0t = np.exp(-r * self.t)
        phi_t_n1 = np.repeat(self.phi_Tk_n1, self.n_per_year)
        G_t_n = self.t
        self.D_t = P0t * np.exp(-phi_t_n1 * x_n - 0.5 * np.square(phi_t_n1) * G_t_n)

    def initialize(self):
        #initialize, setup parameters
        self.computeG_Tk_ny()
        self.computeG_Tk_y()
        self.computePhi_tk_y()
        self.computePsi_Tk_y1()
        self.computePsi_Tk_y1y2()
        self.computeATk()
        self.generate_interpolation()
        self.print_debug()

    def generate_interpolation(self):
        self.psi_Tk_y1y2_intrp = intrp(self.Tk[1:], self.psi_Tk_y1y2)
        self.psi_Tk_y1_intrp = intrp(self.Tk[1:], self.psi_Tk_y1)
        self.phi_Tk_y1_intrp = intrp(self.Tk[1:], self.phi_Tk_y1)
        self.phi_Tk_n1_intrp = intrp(self.Tk[1:], self.phi_Tk_n1)

    def doSimulation(self):
        self.initialize()
        # start simulation
        for i in range(50):
            self.computeYtkDtk()
            plt.subplot(1,2,1)
            plt.plot(self.Tk[1:], self.Y_Tk)
            plt.subplot(1,2,2)
            plt.plot(self.t, self.D_t)
        plt.show()

    def E_tT(self, phi, M, G):
        ans = np.power(np.linalg.det(M), 0.5) * np.exp(0.5 * np.dot(G.dot(M).dot(phi), phi))
        return ans

    def M_tT(self, psi, G):
        return np.linalg.inv(np.eye(3,3) + psi * G)

    def G_tT(self, i, k, T=0):
        G_tT_ny1 = self.G_Tk_ny1[k]-self.G_Tk_ny1[i]
        G_tT_y1 = self.G_Tk_y1[k] - self.G_Tk_y1[i]

        T = max(T, self.Tk[k])
        sigma = np.exp(self.R_Tk_y[k-1] * self.Tk[k-1])
        G_tT_ny1 += (T - self.Tk[k]) * sigma * self.rho_n_y1
        G_tT_y1 += (T - self.Tk[k]) * sigma * sigma

        return np.array([[T - self.Tk[i], G_tT_ny1, 0],
                        [G_tT_ny1, G_tT_y1, 0],
                        [0, 0, G_tT_y1]])

    def print_debug(self):
        print("phi_tk_y1 = ", self.phi_Tk_y1)
        print("psi_tk_y1_y2 = ", self.psi_Tk_y1y2)
        print("psi_tk_y1 = ", self.psi_Tk_y1)
        print("A_Tk = ", self.A_tk)
        print("G_t_ny1 = ", self.G_Tk_ny1)
        print("G_t_y1 = ", self.G_Tk_y1)

    def price_by_qgy(self):
        raise NotImplementedError()

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
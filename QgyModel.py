import numpy as np
from scipy.stats import norm
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class QgyModel:
    def __init__(self):
        # input parameters, pad one ghost point at beginning to better alignment
        # the ghost number should not be used!!!
        self.dim = 3
        self.Sigma_Tk_y = 0.01 * np.array(
            [np.nan, 1.80, 2.72, 3.39, 3.84, 4.16, 4.43, 4.57, 4.48, 4.64, 4.66, 4.63, 4.61, 4.61, 4.60, 4.58, 4.58, 4.54, 4.51, 4.43, 4.38, 4.46, 4.47, 4.47, 4.46, 4.45, 4.60, 4.56, 4.56, 4.57, 4.60])
        self.sinV_Tk_y = 0.01 * np.array(
            [np.nan, 64.9, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
        self.sinRho_Tk_y = -0.01 * np.array(
            [np.nan, 11.7, 15.4, 21.3, 25.5, 32.9, 36.2, 39.5, 41.1, 47.0, 49.7, 53.1, 55.2, 57.7, 59.6, 61.9, 64.7, 65.4, 65.7, 65.0, 64.9, 62.6, 62.2, 62.6, 63.2, 63.6, 68.7, 69.0, 69.6, 70.5, 74.0])
        self.R_Tk_y = 0.01 * np.array(
            [np.nan, 0.0, 59.2, 76.7, 89.5, 97.8, 104.4, 109.2, 112.2, 114.7, 116.7, 117.9, 119.1, 119.7, 120.5, 121.0, 121.0, 121.4, 121.5, 121.7, 121.8, 121.6, 121.8, 121.9, 122.1, 122.3, 122.6, 123.0, 123.4, 123.8, 124.2])
        self.n = self.R_Tk_y.size

        self.n_per_year = 12
        self.rho_n_y1 = -0.1

        # n+1 points to deal with n and n-1 and align with Tk
        self.Tk = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30])
        # self.I0_Tk = np.array(
        #     [100.0, 103.3, 111.0, 119.5, 130.2, 135.6, 137.9, 141.3, 146.0, 150.2, 154.4, 159.5,
        #      163.4, 166.6, 171.1, 173.3, 178.4, 183.1, 188.9, 193.4, 201.6, 209.8, 210.1, 217.9,
        #      229.0, 238.0, 245.8, 252.6, 255.4, 258.8, 265.5])
        self.I0_Tk = self.gernate_fake_forward_inflation_index()

        self.G_Tk_y1 = None
        self.G_Tk_y2 = None
        self.G_Tk_ny1 = None
        self.psi_Tk_y1y2 = None
        self.psi_Tk_y1 = None
        self.phi_Tk_y1 = None
        self.phi_Tk_n1 = None
        self.A_Tk = None

        self.initialize()

        #for debugging
        self.x1min = 0
        self.x1max = 0
        self.x2min = 0
        self.x2max = 0

    def gernate_fake_forward_inflation_index(self):
        Y_Tk = 0.01 * np.array([0.0, 2.9, 2.65, 2.9, 3.0, 3.3,
                                     3.3, 3.3, 3.3, 3.4, 3.3,
                                     3.45, 3.5, 3.5, 3.51, 3.52,
                                     3.55, 3.6, 3.6, 3.6, 3.6,
                                     3.42, 3.41, 3.4, 3.41, 3.42,
                                     3.6, 3.65, 3.68, 3.65, 3.65]) + 1
        return self.Yt_to_It(Y_Tk)


    def integrate_piecewise_expfunc(self, x, R, t, T):
        ''' integrate function exp(R*s) from t to T '''
        if len(x) != len(R):
            print("Warning: len(x) is not equal to len(R)")
            raise SyntaxError()

        sum = 0
        for i in range(1, len(x)):
            lower = np.maximum(t, x[i-1])
            upper = np.minimum(T, x[i])
            if (lower >= upper):
                continue
            if R[i] == 0:
                sum += upper - lower
            else:
                sum += (np.exp(R[i] * upper) - np.exp(R[i] * lower))/R[i]
        return sum

    def computeG_Tk_y(self):
        # integrate (sigma)^2 from 0 to t
        self.G_Tk_y1 = self.G_Tk_y2 = []
        for i in range(0, len(self.R_Tk_y)):
            t = 0
            T = self.Tk[i]
            vol = self.integrate_piecewise_expfunc(self.Tk, self.R_Tk_y * 2, t, T)
            self.G_Tk_y1.append(vol)
        self.G_Tk_y1 = np.array(self.G_Tk_y1)
        self.G_Tk_y2 = np.array(self.G_Tk_y2)

        # check positive definite
        for i in range(1, len(self.Tk)):
            G = self.G_tT(0, i)
            try:
                G_sqrt = np.linalg.cholesky(G)
            except:
                print(G)

    def computeG_Tk_ny(self):
        # integrate (sigma)^2 from 0 to t
        self.G_Tk_ny1 = []
        for i in range(0, len(self.R_Tk_y)):
            t = 0
            T =self.Tk[i]
            intgVol = self.integrate_piecewise_expfunc(self.Tk, self.R_Tk_y, t, T)
            self.G_Tk_ny1.append(self.rho_n_y1 * intgVol)
        self.G_Tk_ny1 = np.array(self.G_Tk_ny1)

    def computePsi_Tk_y1y2(self):
        cosRho_Tk_y = np.sqrt(1 - np.square(self.sinRho_Tk_y))
        psi_tilde = - self.Sigma_Tk_y/ self.K_Tk_y * self.sinV_Tk_y * cosRho_Tk_y
        self.psi_Tk_y1y2 = np.empty(self.G_Tk_y1.size)
        self.psi_Tk_y1y2[0] = np.NaN
        self.psi_Tk_y1y2[1:] = - psi_tilde[1:] / np.sqrt(self.G_Tk_y1[1:] * self.G_Tk_y2[1:])

    def computePsi_Tk_y1(self):
        psi_tilde = self.Sigma_Tk_y/self.K_Tk_y * self.sinV_Tk_y * self.sinRho_Tk_y
        self.psi_Tk_y1 = np.empty(self.G_Tk_y1.size)
        self.psi_Tk_y1[0] = np.NaN
        self.psi_Tk_y1[1:] = -2 * psi_tilde[1:] / self.G_Tk_y1[1:]

    def computePhi_tk_y(self):
        cosV_Tk_y = np.sqrt(1 - np.square(self.sinV_Tk_y))
        phi_tilde = self.Sigma_Tk_y/self.K_Tk_y * cosV_Tk_y
        self.phi_Tk_y1 = np.empty(self.Tk.size)
        self.phi_Tk_y1[0] = np.NaN
        self.phi_Tk_y1[1:] = -phi_tilde[1:]/np.sqrt(self.G_Tk_y1[1:])

    def computePhi_tk_n1(self):
        #self.phi_Tk_n1 = np.zeros(self.n)
        self.phi_Tk_n1 = np.ones(self.n) * 0.01

    def computeK_Tk_y(self):
        self.K_Tk_y = np.sqrt(np.square(self.sinV_Tk_y * self.sinRho_Tk_y) + 1)

    def phi_y(self, i):
        return np.asmatrix([0, self.phi_Tk_y1[i], 0]).T

    def phi_y_at(self, t):
        return np.asmatrix([0, self.phi_Tk_y1_intrp(t), 0]).T

    def phi_n(self, i):
        return np.asmatrix([self.phi_Tk_n1[i], 0, 0]).T

    def phi_n_at(self, t):
        return np.asmatrix([self.phi_Tk_n1_intrp(t), 0, 0]).T

    def psi_n(self):
        return np.zeros([self.dim, self.dim])

    def psi_n_at(self, t):
        return np.zeros([self.dim, self.dim])

    def psi_y(self, i):
        return np.asmatrix([[0, 0, 0], [0, self.psi_Tk_y1[i], self.psi_Tk_y1y2[i]], [0, self.psi_Tk_y1y2[i], 0]])

    def psi_y_at(self, t):
        return np.asmatrix([[0, 0, 0], [0, self.psi_Tk_y1_intrp(t), self.psi_Tk_y1y2_intrp(t)], [0, self.psi_Tk_y1y2_intrp(t), 0]])

    def computeATk(self):
        n = self.Tk.size
        if self.A_Tk is None:
            self.A_Tk = np.empty(n)

        A_exp_prod = 1
        for k in range(1, n):
            phi_n = self.phi_n(k)
            psi_n = self.psi_n()
            G = self.G_tT(0, k)
            M = self.M_tT(psi_n, G)
            E_0Tk = self.E_tT(phi_n, M, G)

            H0 = self.phi_n(k) + self.phi_y(k)  # phi_n + phi_y
            H1 = self.psi_n() + self.psi_y(k)
            E_prod = 1
            for i in range(k, 0, -1):
                G = self.G_tT(i-1, i)
                M = self.M_tT(H1, G)
                E_prod *= self.E_tT(H0, M, G)

                Theta0 = M.dot(H0)
                Theta1 = M.dot(H1)
                H0 = Theta0 + self.phi_y(i-1)
                H1 = Theta1 + self.psi_y(i-1)

            self.A_Tk[k] = np.log(E_0Tk / E_prod / A_exp_prod)
            A_exp_prod = E_0Tk / E_prod

    def generate_yoy_structure_from_drivers(self, x_Tk_y1, x_Tk_y2):
        exponent = self.A_Tk[1:] - (self.phi_Tk_y1[1:] + 0.5 * self.psi_Tk_y1[1:] * x_Tk_y1 + self.psi_Tk_y1y2[1:] * x_Tk_y2) * x_Tk_y1
        return self.I0_Tk[1:]/self.I0_Tk[0:-1] * np.exp(exponent)

    def generate_terms_structure(self):
        # volatlity is backward filling, so the total number is self.n_per_year * (self.n - 1)
        sigma2 = []
        dt = 1/self.n_per_year
        for i in range(1, len(self.R_Tk_y)):
            for j in range(self.n_per_year):
                sigma2.append(np.exp(self.R_Tk_y[i] * (self.Tk[i-1] + dt * (j+1))))

        sigma_n = np.repeat(1, self.n_per_year * (self.n - 1))

        [x_n, x_y1] = self.generate_two_correlated_gauss(sigma_n, sigma2, self.rho_n_y1, (self.n-1) * self.n_per_year, 1/self.n_per_year)
        x_y2 = self.generate_one_gauss(sigma2, (self.n-1) * self.n_per_year, 1/self.n_per_year)

        x_Tk_y1 = x_y1[self.n_per_year-1::self.n_per_year]
        x_Tk_y2 = x_y2[self.n_per_year-1::self.n_per_year]

        self.Y_Tk = self.generate_yoy_structure_from_drivers(x_Tk_y1, x_Tk_y2)

        Y_0 = np.nan
        self.Y_Tk = np.insert(self.Y_Tk, 0, Y_0)

        x_n_Tk = x_n[self.n_per_year-1::self.n_per_year]
        x_n_Tk = np.insert(x_n_Tk, 0, 0)
        P0t = self.P_0T(self.Tk)
        self.D_t = P0t * np.exp(-self.phi_Tk_n1 * x_n_Tk - 0.5 * np.square(self.phi_Tk_n1) * self.Tk)
        self.D_t[0] = 1

    def P_0T(self, t):
        r = 0.002
        return np.exp(-r * t)

    def initialize(self):
        # initialize, setup parameters
        self.computeK_Tk_y()
        self.computeG_Tk_ny()
        self.computeG_Tk_y()
        self.computePhi_tk_y()
        self.computePhi_tk_n1()
        self.computePsi_Tk_y1()
        self.computePsi_Tk_y1y2()
        self.computeATk()
        self.generate_interpolation()
        #self.I0_Tk = self.fit_yoy_convexity_correction(self.I0_Tk)
        #self.print_debug()

    def reset_parameters(self):
        self.G_Tk_y1 = None
        self.G_Tk_y2 = None
        self.G_Tk_ny1 = None
        self.psi_Tk_y1y2 = None
        self.psi_Tk_y1 = None
        self.phi_Tk_y1 = None
        self.phi_Tk_n1 = None
        self.A_Tk = None

    def generate_interpolation(self):
        self.psi_Tk_y1y2_intrp = sp.interpolate.interp1d(self.Tk[1:], self.psi_Tk_y1y2[1:])
        self.psi_Tk_y1_intrp = sp.interpolate.interp1d(self.Tk[1:], self.psi_Tk_y1[1:])
        self.phi_Tk_y1_intrp = sp.interpolate.interp1d(self.Tk[1:], self.phi_Tk_y1[1:])
        self.phi_Tk_n1_intrp = sp.interpolate.interp1d(self.Tk[1:], self.phi_Tk_n1[1:])

    def E_tT_simple(self, h, k, phi, psi):
        G = self.G_tT(h, k)
        M = self.M_tT(psi, G)
        return self.E_tT(phi, M, G)

    def E_tT(self, phi, M, G):
        ans = np.power(np.linalg.det(M), 0.5) * np.exp(0.5 * phi.T.dot(G.dot(M).dot(phi)))
        return ans

    def M_tT(self, psi, G):
        return np.linalg.inv(np.eye(3,3) + psi.dot(G))

    def G_tT(self, i, k, T=0):
        G_tT_ny1 = self.G_Tk_ny1[k] - self.G_Tk_ny1[i]
        G_tT_y1 = self.G_Tk_y1[k] - self.G_Tk_y1[i]

        if T > self.Tk[k]:
            if self.R_Tk_y[k+1] == 0:
                G_tT_ny1 += self.rho_n_y1 * (T - self.Tk[k])
                G_tT_y1 += T - self.Tk[k]
            else:
                G_tT_ny1 += self.rho_n_y1 * (np.exp(T * self.R_Tk_y[k+1]) - np.exp(self.Tk[k] * self.R_Tk_y[k+1]))/self.R_Tk_y[k+1]
                G_tT_y1 += (np.exp(2.0 * T * self.R_Tk_y[k+1]) - np.exp(2.0 * self.Tk[k] * self.R_Tk_y[k+1]))/(2.0 * self.R_Tk_y[k+1])

        if T > self.Tk[k]:
            G_tn = T - self.Tk[i]
        else:
            G_tn = self.Tk[k] - self.Tk[i]

        return np.asmatrix([[G_tn, G_tT_ny1, 0],
                        [G_tT_ny1, G_tT_y1, 0],
                        [0, 0, G_tT_y1]])

    def fill_spherical_parameters(self, Sigma, v_y, rho_y, rho_ny1, R_y):
        self.reset_parameters()
        self.Sigma_Tk_y.fill(Sigma)
        self.sinV_Tk_y.fill(np.sin(v_y))
        self.sinRho_Tk_y.fill(np.sin(rho_y))
        self.rho_n_y1 = rho_ny1
        self.R_Tk_y.fill(R_y)
        self.initialize()

    def fill_sin_parameters(self, Sigma, sin_v_y, sin_rho_y, rho_ny1, R_y):
        self.reset_parameters()
        self.Sigma_Tk_y.fill(Sigma)
        self.sinV_Tk_y.fill(sin_v_y)
        self.sinRho_Tk_y.fill(sin_rho_y)
        self.rho_n_y1 = rho_ny1
        self.R_Tk_y.fill(R_y)
        self.initialize()

    def set_spherical_parameters_at(self, k, Sigma, v_y, rho_y, rho_ny1, R_y):
        self.Sigma_Tk_y[k] = Sigma
        self.sinV_Tk_y[k] = np.sin(v_y)
        self.sinRho_Tk_y[k] = np.sin(rho_y)
        self.rho_n_y1 = rho_ny1
        self.R_Tk_y[k] = R_y
        self.initialize()

    def set_sin_parameters_at(self, k, Sigma, sin_v_y, sin_rho_y, rho_ny1, R_y):
        self.Sigma_Tk_y[k] = Sigma
        self.sinV_Tk_y[k] = sin_v_y
        self.sinRho_Tk_y[k] = sin_rho_y
        self.rho_n_y1 = rho_ny1
        self.R_Tk_y[k] = R_y
        self.initialize()

    def set_qgy_parameters_at(self, k, phi_Tk_y1, psi_Tk_y1, psi_Tk_y1y2, R_Tk_y):
        self.phi_Tk_y1[k] = phi_Tk_y1
        self.psi_Tk_y1[k] = psi_Tk_y1
        self.psi_Tk_y1y2[k] = psi_Tk_y1y2
        self.R_Tk_y[k] = R_Tk_y
        self.computeATk()
        self.generate_interpolation()

    def set_normalized_qgy_parameters_at(self, k, phi_Tk_y1, psi_Tk_y1, psi_Tk_y1y2, R_Tk_y):
        self.R_Tk_y[k] = R_Tk_y
        self.computeG_Tk_ny()
        self.computeG_Tk_y()

        self.phi_Tk_y1[k] = -phi_Tk_y1/np.sqrt(self.G_Tk_y1[k])
        self.psi_Tk_y1[k] = -2.0 * psi_Tk_y1/self.G_Tk_y1[k]
        self.psi_Tk_y1y2[k] = -psi_Tk_y1y2/np.sqrt(self.G_Tk_y1[k] * self.G_Tk_y2[k])

        self.computeATk()
        self.generate_interpolation()

    def print_debug(self):
        print("phi_tk_y1 = ", self.phi_Tk_y1)
        print("psi_tk_y1_y2 = ", self.psi_Tk_y1y2)
        print("psi_tk_y1 = ", self.psi_Tk_y1)
        print("A_Tk = ", self.A_Tk)
        print("G_t_ny1 = ", self.G_Tk_ny1)
        print("G_t_y1 = ", self.G_Tk_y1)

    def price_by_qgy(self):
        raise NotImplementedError()

    def price_swaplet_by_qgy(self, h, k, T, P_0T):
        sum_A = np.sum(self.A_Tk[h+1:k+1])
        res = P_0T * self.I0_Tk[k]/self.I0_Tk[h] * np.exp(sum_A)
        G = self.G_tT(0,k,T)
        M = self.M_tT(self.psi_n_at(T), G)
        E0T = self.E_tT(self.phi_n_at(T), M, G)
        res /= E0T

        G = self.G_tT(k,k,T)
        M = self.M_tT(self.psi_n_at(T), G)
        H0 = M.dot(self.phi_n_at(T)) + self.phi_y(k)
        H1 = M.dot(self.psi_n_at(T)) + self.psi_y(k)

        for i in range(k, h+1, -1):
            res *= self.E_tT_simple(i-1, i, H0, H1)

            # update H0, H1
            G = self.G_tT(i - 1, i)
            M = self.M_tT(H1, G)
            H0 = M.dot(H0) + self.phi_y(i-1)
            H1 = M.dot(H1) + self.psi_y(i-1)

        res *= self.E_tT_simple(0, h+1, H0, H1)

        return np.asscalar(res)

    def price_yoy_infln_fwd(self):
        swaplet_price = [0]
        Tk = [0]

        for k in range(1, self.Tk.size):
            T = self.Tk[k]
            P_0T = self.P_0T(T)
            price = self.price_swaplet_by_qgy(k - 1, k, T, P_0T) / P_0T - 1
            swaplet_price.append(price)
            Tk.append(T)
        return np.array(swaplet_price)

    def fit_yoy_convexity_correction(self, I0_Tk):
        yoy_infln_fwd = self.price_yoy_infln_fwd()
        Tk = self.Tk

        def target(a):
            Y_0T_fit = yoy_infln_fwd[1:] + 1 + a * np.log(Tk[1:])
            I_0T_fit = self.Yt_to_It(Y_0T_fit)
            ans = np.sqrt(np.sum(np.square(I_0T_fit - I0_Tk[1:])))
            return ans

        a_fit = minimize(target, x0=np.array([0]), method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x
        Y_0T_fit = 1 + yoy_infln_fwd[1:] + a_fit * np.log(Tk[1:])
        print("a fit = ", a_fit)
        I_0T_fit = np.concatenate([[1], self.Yt_to_It(Y_0T_fit)])
        return I_0T_fit

    @staticmethod
    def transform_G(G_Tk, Psi, dim = 3):
        return G_Tk.dot(np.linalg.inv(np.eye(dim) + Psi.dot(G_Tk)))

    @staticmethod
    def transform_G_sqrt(G_Tk_sqrt, Psi, dim = 3):
        one_plus_Gsqrt_Psi_Gsqrt = np.eye(dim) + G_Tk_sqrt.T.dot(Psi).dot(G_Tk_sqrt)
        return G_Tk_sqrt.dot(np.linalg.cholesky(np.linalg.inv(one_plus_Gsqrt_Psi_Gsqrt)))

    @staticmethod
    def transform_x_t(x_t, G_orig, G_new, Phi):
        return G_new.dot(np.linalg.inv(G_orig).dot(x_t) - Phi)

    @staticmethod
    def pad_before_array(num, v):
        return np.concatenate([[num], v])

    @staticmethod
    def generate_two_correlated_gauss(sigma1, sigma2, rho, n, dt):
        sqrt_dt = np.sqrt(dt)
        dw1 = norm.rvs(size=n, scale=sqrt_dt)
        dw2 = norm.rvs(size=n, scale=sqrt_dt)

        dx1 = dw1 * sigma1
        dx2 = dw1 * sigma2 * rho + dw2 * np.sqrt(1 - rho * rho) * sigma2

        x1 = np.cumsum(dx1, axis=-1)
        x2 = np.cumsum(dx2, axis=-1)

        return [x1, x2]

    @staticmethod
    def generate_one_gauss(sigma, n, dt):
        dw = norm.rvs(size=n, scale=np.sqrt(dt))
        dx = dw * sigma
        x = np.cumsum(dx, axis=-1)
        return x

    @staticmethod
    def Yt_to_It(Yt):
        res = []
        I = 1
        for y in Yt:
            I *= y
            res.append(I)
        return np.array(res)

    @staticmethod
    def is_symmetric(a, tol=1e-8):
        return np.allclose(a, a.T, atol=tol)

    @staticmethod
    def Xt(xt, Phi, Psi):
        return np.exp(-Phi.T.dot(xt) - 0.5 * xt.T.dot(Psi.dot(xt)))


if __name__ == "__main__":
    qgy = QgyModel()
    qgy.initialize()
    # start simulation
    for i in range(50):
        qgy.generate_terms_structure()
        plt.subplot(1, 2, 1)
        plt.plot(qgy.Tk, qgy.Y_Tk, 'g-')
        plt.xlabel('Maturity')
        plt.ylabel('YoY inflation ratio')
        plt.subplot(1, 2, 2)
        plt.plot(qgy.Tk, qgy.D_t, 'g-')
        plt.xlabel('Maturity')
        plt.ylabel('Inverse numeraire')
    plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.integrate as integrate

class QgIntegration:
    def __init__(self, A, B, C, D, E, F):
        # P(x,y) = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
        # transform to polar coordinates theta, r
        # P(x,y) = Q(theta) r**2 + L(theta) r + F
        if not np.isscalar(A) \
                or not np.isscalar(B) \
                or not np.isscalar(C) \
                or not np.isscalar(D) \
                or not np.isscalar(E) \
                or not np.isscalar(F):
            print(A, B, C, D, E, F)
            print(np.isscalar(A), np.isscalar(B), np.isscalar(C), np.isscalar(D), np.isscalar(E), np.isscalar(F))
            raise NotImplementedError
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.tol = 1e-10

    def Q(self, theta):
        return self.A * np.cos(theta)**2 + self.B * np.sin(theta) * np.cos(theta) + self.C * np.sin(theta) ** 2

    def L(self, theta):
        return self.D * np.cos(theta) + self.E * np.sin(theta)

    def M(self, theta):
        return (self.D**2 - 4 * self.A*self.F) * np.cos(theta)**2 \
               + (2 * self.D * self.E - 4 * self.B * self.F) * np.sin(theta) * np.cos(theta) \
               + (self.E**2 - 4 * self.C * self.F) * np.sin(theta)**2

    def root_plus(self, theta):
        L = self.L(theta)
        Q = self.Q(theta)
        return (-L + np.sqrt(L**2 - 4 * self.F * Q))/(2 * Q)

    def root_minus(self, theta):
        L = self.L(theta)
        Q = self.Q(theta)
        return (-L - np.sqrt(L**2 - 4 * self.F * Q))/(2 * Q)

    def root_zero(self, theta):
        return - self.F/self.L(theta)

    def integrand(self, theta):
        F = self.F
        Q = self.Q(theta)
        L = self.L(theta)

        # not roots
        if L**2 - 4 * F * Q < 0 and F > 0:
            return 0
        elif L**2 - 4 * F * Q < 0 and F < 0:
            return 1

        # one or two roots
        r_p = self.root_plus(theta)
        r_m = self.root_minus(theta)
        r_0 = self.root_zero(theta)

        if F > 0 and Q < 0:
            return np.exp(-0.5 * r_m ** 2)
        elif F > 0 and Q > 0 and L > 0:
            return 0
        elif F > 0 and Q > 0 and L <= 0:
            return np.exp(-0.5 * r_m ** 2  - np.exp(-0.5 * r_p ** 2))
        elif F <= 0 and Q > 0:
            return 1.0 - np.exp(-0.5 * r_p ** 2)
        elif F <= 0 and Q < 0 and L <= 0:
            return 1.0
        elif F <= 0 and Q < 0 and L > 0:
            return 1 + np.exp(-0.5 * r_m ** 2) - np.exp(-0.5 * r_p ** 2)
        elif F > 0 and np.abs(Q) < self.tol and np.abs(L) < self.tol:
            return 0
        elif F > 0 and np.abs(Q)  < self.tol and L > 0:
            return 0
        elif F > 0 and np.abs(Q) < self.tol and L < 0:
            return np.exp(-0.5 * r_0 ** 2)
        elif F <= 0 and np.abs(Q) < self.tol and np.abs(L) < self.tol:
            return 1
        elif F <= 0 and np.abs(Q) < self.tol and L > 0:
            return 1 - np.exp(-0.5 * r_0 ** 2)
        elif F <= 0 and np.abs(Q) < self.tol and L < 0:
            return 1
        else:
            raise NotImplementedError

    def P(self, theta, r):
        Q = self.Q(theta)
        L = self.L(theta)
        return Q * r**2 + L *r + self.F

    def roots_of_Q(self, A, B, C):
        # roots of Q between [0, 2 * PI]
        ans = set()
        if  np.abs(A) < self.tol and np.abs(B) < self.tol and np.abs(C) < self.tol:
            return ans

        if np.abs(A) > self.tol and np.abs(B) < self.tol and np.abs(C) < self.tol:
            # A not = 0, B = 0, C = 0
            ans.add(0.5 * np.pi)
            ans.add(1.5 * np.pi)

        if np.abs(B) > self.tol and np.abs(C) < self.tol:
            # B not = 0, C = 0
            ans.add(0.5 * np.pi)
            ans = ans.union(self.find_periodic_root_between(np.arctan(-A/B), 0, 2*np.pi, np.pi))

        if  np.abs(C) > self.tol:
            # C not = 0
            one_root = np.arctan((-B + np.sqrt(B**2 - 4 * A * C))/(2 * C))
            ans = self.find_periodic_root_between(one_root, 0, 2*np.pi, np.pi)
            one_root = np.arctan((-B - np.sqrt(B**2 - 4 * A * C))/(2 * C))
            ans = ans.union(self.find_periodic_root_between(one_root, 0, 2*np.pi, np.pi))

        return ans

    def roots_of_L(self):
        ans = set()
        if np.abs(self.D) < self.tol and np.abs(self.E) < self.tol:
            return ans

        if np.abs(self.D) > self.tol and np.abs(self.E) < self.tol:
            ans.add(0.5 * np.pi)
            ans.add(1.5 * np.pi)

        if np.abs(self.D) < self.tol and np.abs(self.E) > self.tol:
            ans.add(0)
            ans.add(np.pi)

        if np.abs(self.D) > self.tol and np.abs(self.E) > self.tol:
            ans = self.find_periodic_root_between(np.arctan(-self.D/self.E), 0, 2*np.pi, np.pi)

        return ans

    def compute_gauss_integration_on_intervals(self):
        roots = self.find_coeff_roots()
        res = []
        for i in range(len(roots)-1):
            lb = roots[i]
            ub = roots[i+1]
            integ = integrate.quad(self.integrand, lb, ub)
            res.append(integ[0]/(2 * np.pi))
        return res

    def compute_gaussian_integration(self):
        res = self.compute_gauss_integration_on_intervals()
        return sum(res)

    def roots_of_M(self):
        return self.roots_of_Q(self.D**2 - 4 * self.A * self.F,
                               2 * self.D * self.E - 4 * self.B * self.F,
                               self.E**2 - 4 * self.C * self.F)

    def find_coeff_roots(self):
        ans = set()
        roots_of_Q = self.roots_of_Q(self.A, self.B, self.C)
        roots_of_L = self.roots_of_L()
        roots_of_M = self.roots_of_M()
        ans = ans.union(roots_of_Q)
        ans = ans.union(roots_of_L)
        ans = ans.union(roots_of_M)
        ans.add(min(ans) + 2 * np.pi)
        return sorted(ans)

    def find_periodic_root_between(self, init, lb, ub, period):
        ans = set()
        # cannot get an answer
        if (init >= ub or init < lb) and period > ub - lb:
            return ans

        # move init to [lb, ub)
        if init >= ub and period < ub - lb:
            init -= np.ceil((init - ub)/period) * period
        elif init < lb and period < ub - lb:
            init += np.ceil((lb - init)/period) * period

        # find root with period
        root = init
        while root < ub:
            ans.add(root)
            root += period
        root = init
        while lb <= root:
            ans.add(root)
            root -= period
        return ans

if __name__ == "__main__":
    A = 0.001
    B = 0.01
    C = -0.00007
    D = -0.01
    E = 0.000005
    F = -0.03

    # test 1
    print('test 1: plot polynomial on r and theta')
    qg_int = QgIntegration(A, B, C, D, E, F)
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(0, 8, 200)
    X, Y = np.meshgrid(theta, r)
    Z = qg_int.P(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    #plt.show()

    # test 2
    print('test 2: compute roots of coefficients')
    computed = qg_int.find_coeff_roots()
    computed = np.array(list(computed))/np.pi
    exact = {0.49777342465146, 0.96829651297154, 1.49777342465146, 1.96829651297154, 0.49984084507017,
           1.49984084507017, 0.49777454085691, 0.94235170892355, 1.49777454085691, 1.94235170892355, 2.49777342465146}
    exact = np.array(list(sorted(exact)))
    print(computed)
    err = np.linalg.norm(computed - exact, 1)
    print("error = ", err)


    # test 3
    print('test 3: compute integration on theta')
    res = np.array(qg_int.compute_gauss_integration_on_intervals())/(2*np.pi)
    exact = np.array([5.58102721790474e-7, 0.0010331521066328, 0.221255431926692, 0.0129326423061761, 0.232158529261464,
    5.58102721790474e-7, 0.00103315210663273, 0.221255431926692, 0.0129724020239904, 0.263981873139802])/(2 * np.pi)
    err = np.linalg.norm(res - exact, 1)
    print("error = ", err)
    print("integration = ", qg_int.compute_gaussian_integration())


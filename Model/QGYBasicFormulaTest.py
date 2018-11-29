from Model.QgyModel import *
import numpy as np


class QGYBasicTest(QgyModel):
    def testQXY(self):
        res_1 = []
        res_2 = []
        N = self.Tk.size
        for k in range(1, N):
            res_1.append(self.sol_1(k))
            res_2.append(self.sol_2(k))
        print(res_1)
        plt.plot(res_1, '-s')
        plt.plot(res_2, '-o')
        plt.show()

    def sol_1(self, k):
        Psi_Tk_y = self.psi_y(k)
        Psi_Tk_n = self.psi_n()
        Phi_Tk_y = self.phi_y(k)
        Phi_Tk_n = self.phi_n(k)

        # part 1
        GtT = self.G_tT(k, k)
        MtT = self.M_tT(Psi_Tk_n, GtT)
        MPhi = MtT.dot(Phi_Tk_n)
        MPsi = MtT.dot(Psi_Tk_n)
        res_1 = self.E_tT_simple(0, k, MPhi, MPsi)

        # part 2
        x_t = np.zeros([3, 1])
        GtT = self.G_tT(0, k)
        MtT = self.M_tT(Psi_Tk_y, GtT)
        MPhi = MtT.dot(Phi_Tk_y)
        MPsi = MtT.dot(Psi_Tk_y)
        GtT_1 = self.transform_G(GtT, MPsi)
        xt_1 = self.transofrm_x_t(x_t, GtT, GtT_1, MPhi)
        Xt_1 = self.Xt(xt_1, MPhi, MPsi)
        res_2 = self.E_tT_simple(0, k, Phi_Tk_y, Psi_Tk_y) * Xt_1

        return np.asscalar(res_1 * res_2)

    def sol_2(self, k):
        Psi_Tk_y = self.psi_y(k)
        Psi_Tk_n = self.psi_n()
        Phi_Tk_y = self.phi_y(k)
        Phi_Tk_n = self.phi_n(k)
        GtT = self.G_tT(k,k)
        M = self.M_tT(Psi_Tk_n, GtT)

        res = self.E_tT_simple(0, k, M.dot(Phi_Tk_n) + Phi_Tk_y, M.dot(Psi_Tk_n) + Psi_Tk_y)
        return np.asscalar(res)


test = QGYBasicTest()
test.testQXY()
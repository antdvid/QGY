import numpy as np
import matplotlib.pyplot as plt


def G_tT(t, T):
    if R == 0:
        return np.eye(3) * (T - t)
    else:
        return np.asmatrix([[T-t, 0, 0],[0, (np.exp(R*T) - np.exp(R*t))/R, 0],[0, 0, 0.5 * (np.exp(2*R*T) - np.exp(2*R*t))/R]])

def M_tT(t, T, psi):
    print("psi = ", psi, "G = ", G_tT(t, T))
    return np.linalg.inv(np.eye(3) + psi.dot(G_tT(t, T)))

def E_tT(t, T, phi, psi):
    G = G_tT(t, T)
    M = M_tT(t, T, psi)
    ans = np.sqrt(np.linalg.det(M_tT(t, T, psi))) * np.exp(0.5 * phi.dot(G.dot(M.dot(phi.T))))
    return ans

def Theta(t, T, phi, psi):
    M = M_tT(t, T, psi)
    print("M = ", M)
    return [M.dot(phi.T).T, M.dot(psi)]

def G_Tk_y1(G):
    return G[1,1]

def G_Tk_y2(G):
    return G[2,2]

def Exp_A_T2():
    theta = Theta(T[1], T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2))
    theta_phi = theta[0]
    theta_psi = theta[1]
    N = E_tT(0, T[2], phi_n(2), psi_n(2)) * E_tT(0, T[1], phi_n(1) + phi_y(1), psi_n(1) + psi_y(1))
    D = E_tT(T[1], T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2)) \
        * E_tT(0, T[1], phi_y(1) + theta_phi, psi_y(1) + theta_psi) \
        * E_tT(0, T[1], phi_n(1), psi_n(1))
    e_A_T1 =  E_tT(0, T[1], phi_n(1), psi_n(1)) / E_tT(0, T[1], phi_n(1) + phi_y(1), psi_n(1) + psi_y(1))
    e_A_T1_A_T2 = E_tT(0, T[2], phi_n(2), psi_n(2))/ (E_tT(T[1], T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2))
                                                      * E_tT(0, T[1], phi_y(1) + theta_phi, psi_y(1) + theta_psi))
    print("e_A_T1 = ", e_A_T1, "e_A_T1_T2 = ", e_A_T1_A_T2, "e_A_T2 = ", e_A_T1_A_T2/e_A_T1, "N/D = ", N/D)
    print("e_0_T2 = ", E_tT(0, T[2], phi_n(2), psi_n(2)), "e_prod = ", E_tT(T[1], T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2)) * E_tT(0, T[1], phi_y(1) + theta_phi, psi_y(1) + theta_psi))
    print("E_12 = ", E_tT(T[1], T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2)))
    print("H0 = ", phi_n(2) + phi_y(2), "H1 = ", psi_n(2) + psi_y(2))
    print("E_01 = ", E_tT(0, T[1], phi_y(1) + theta_phi, psi_y(1) + theta_psi))
    print("H0 = ", phi_y(1) + theta_phi, "H1 = ", psi_y(1) + theta_psi)
    return N/D

def phi_n(i):
    return np.matrix([0, 0, 0])

def psi_n(i):
    return np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def phi_y(i):
    return np.matrix([0, phi_y1[i-1], 0])

def psi_y(i):
    return np.matrix([[0, 0, 0], [0, psi_y1[i-1], psi_y1y2[i-1]], [0, psi_y1y2[i-1], 0]])


R_test = 0.01 * np.array([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
# R_test = 0.01 * np.array([0])
for R in R_test:
    T = np.array([0, 1, 2])
    G = [G_tT(0, T[1]), G_tT(0, T[2])]
    G_y1 = [G_Tk_y1(G[0]), G_Tk_y1(G[1])]
    G_y2 = [G_Tk_y2(G[0]), G_Tk_y2(G[1])]
    Sigma_Tk_y = 0.045
    v_Tk_y = 0.8
    rho_Tk_y = -np.pi/2
    rho_t_ny1 = 0.0
    K_Tk_y = np.sqrt(np.sin(v_Tk_y)**2 * np.sin(rho_Tk_y)**2 + 1)

    phi_y1_tild = Sigma_Tk_y * np.cos(v_Tk_y)/K_Tk_y
    phi_y1 = [-phi_y1_tild/np.sqrt(G_y1[0]), -phi_y1_tild/np.sqrt(G_y1[1])]

    psi_y1_tild = Sigma_Tk_y * np.sin(v_Tk_y) * np.sin(rho_Tk_y) / K_Tk_y
    psi_y1 = [-2 * psi_y1_tild/G_y1[0], -2 * psi_y1_tild/G_y1[1]]

    psi_y1y2_tild = -Sigma_Tk_y * np.sin(v_Tk_y) * np.cos(rho_Tk_y) / K_Tk_y
    psi_y1y2 = [-psi_y1y2_tild/np.sqrt(G_y1[0] * G_y2[0]), -psi_y1y2_tild/np.sqrt(G_y1[1] * G_y2[1])]

    e_A_T2 = Exp_A_T2()
    res = e_A_T2 * E_tT(0, T[2], phi_n(2) + phi_y(2), psi_n(2) + psi_y(2))/E_tT(0, T[2], phi_n(2), psi_n(2))
    plt.plot(R, res, 'o')
    print("Exp_A_T2 = ", e_A_T2, "res = ", res)

plt.show()
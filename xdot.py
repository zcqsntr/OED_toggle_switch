import numpy as np
from casadi import *


def xdot(sym_y, sym_theta, sym_u):

    IPTGi, aTci, RFP_LacI, GFP_TetR = [sym_y[i] for i in range(4)]

    IPTG, aTc = [sym_u[i] for i in range(2)]

    k_IPTG, k_aTc, k_L_pm0, k_L_pm, theta_T, theta_aTc, n_aTc, n_T, k_T_pm0, k_T_pm, theta_L, theta_IPTG, n_IPTG, n_L = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    dIPTGi = k_IPTG*(IPTG-IPTGi)-0.0165*IPTGi
    daTci  = k_aTc*(aTc-aTci)-0.0165*aTci
    dRFP_LacI  = ((1/0.1386)*(k_L_pm0+(k_L_pm/(1+(GFP_TetR/theta_T*1/(1+(aTci/theta_aTc)**n_aTc))**n_T))))-0.0165*RFP_LacI
    dGFP_TetR  = ((1/0.1386)*(k_T_pm0+(k_T_pm/(1+(RFP_LacI/theta_L*1/(1+(IPTGi/theta_IPTG)**n_IPTG))**n_L))))-0.0165*GFP_TetR

    xdot = SX.sym('xdot', 4)

    xdot[0] = dIPTGi
    xdot[1] = daTci
    xdot[2] = dRFP_LacI
    xdot[3] = dGFP_TetR

    return xdot

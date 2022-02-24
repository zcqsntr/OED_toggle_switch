import numpy as np
from casadi import *


def xdot(sym_y, sym_theta, sym_u):
    k_IPTG, k_aTc, k_L_pm0, k_L_pm, theta_T, theta_aTc, n_aTc, n_T, k_T_pm0, k_T_pm, theta_L, theta_IPTG, n_IPTG, n_L = \
        [0.4437322, 0.01334992, 0.05980078, 6.32831827, 91.80251520, 0.91779024, 0.44236521, 3.26829704, 0.27499435, 4.48180358, 2.68995461, 0.01720242, 0.80772911, 0.71316179]

    IPTGi, aTci, RFP_LacI, GFP_TetR = [sym_y[i] for i in range(4)]

    #actions are given straight forward
    #IPTG, aTc = [sym_u[i] for i in range(2)]

    # actions are linear combination
    IPTG = 1e-7 + sym_u[0]*(1- 1e-7)
    aTc = 1e-7 + (1-sym_u[0])*(100- 1e-7)

    # all params
    #k_IPTG, k_aTc, k_L_pm0, k_L_pm, theta_T, theta_aTc, n_aTc, n_T, k_T_pm0, k_T_pm, theta_L, theta_IPTG, n_IPTG, n_L = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    #highly identifiable
    k_aTc, k_L_pm, n_T, k_T_pm0, theta_L, n_L = [sym_theta[i] for i in range(len(sym_theta.elements()))]

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:50:52 2021

@author: Hans
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from RB_numerical_toy_model_v1 import depolarizing_noise

lm = 0.05  # must be the same as data lm

# def RB_Fm(m, p):

#     rho = np.array([[1,0],[0,0]])  # must be the same as data rho
#     ket_0 = np.array([1,0]).reshape(2,1)
#     proj_O = np.kron(ket_0, ket_0.T)
#     D = 2
#     # 4 looks better
#     a = rho - np.eye(2)/D
#     b = np.eye(2)/D
#     pre_A = depolarizing_noise(a, lm)
#     pre_B = depolarizing_noise(b, lm)
#     A = np.trace(pre_A @ proj_O).real
#     B = np.trace(pre_B @ proj_O).real
    
#     Fm_fit = A*p**m + B
#     # Fm_fit_ln = np.log(A*p**m + B)
#     return Fm_fit


# Fm = np.loadtxt("../data/Fm_60_avg_1000_depolar_01_seed_1_2021_07_01_17_14.txt")
# Fm = np.loadtxt("../data/Fm_80_avg_1000_depolar_005_seed_1_2021_07_01_21_45.txt")
# Fm = np.loadtxt("../data/Fm_100_avg_10_Amp_damp_001_K1_seed_1_2021_07_06_18_46.txt")

def compare_Fm(Fm, proj_O, rho, noise_mode, noise_para, pic_path):
    m = np.array(range(1, 1 + len(Fm)))
    # ket_0 = np.array([1,0]).reshape(2,1)
    # ket_1 = np.array([0,1]).reshape(2,1)
    # rho = np.array([[1,0],[0,0]])
    # proj_O = np.kron(ket_0, ket_0.T) 
    A = np.trace(proj_O @ (rho-np.eye(2)/2)).real
    B = np.trace(proj_O @ (np.eye(2)/2)).real
    
    # Not exactly, should write a function to calculate tr(Lambda)
    # The trace of a map (represented by Kraus operators) should be take care of, not a matrix.
    if noise_mode == 'depolar':
        theory_p = (4 - 3*noise_para -1)/3
    elif noise_mode == 'amp_damp':
        theory_p = (2 - noise_para + 2*np.sqrt(1-noise_para)-1)/3
    elif noise_mode == 'p_flip':
        theory_p = 1 - 4*noise_para/3
    # theory_amp_damp_p = (2-0.01 -1)/(4-1)  # sqrt(1-p) for amplitude damping, p=0.01
    theory_Fm = A*theory_p**m + B
    # m = np.array(range(1, 31))
    # Fm_ln = np.log(Fm)
    # fit_p, covariance = curve_fit(RB_Fm, m, Fm)
    # p = ((Fm[20]-B)/A)**(1/20)
    # fit_Fm = RB_Fm(m, fit_p)
    # fit_Fm = RB_Fm(m, lm)
    
    plt.plot(m, Fm, 'o', label='data')
    # plt.plot(m, theory_Fm, '-', label='estimate')
    # plt.plot(m, fit_Fm, '-', label='fit')
    plt.title(noise_mode)
    plt.xlabel('Sequence length')
    plt.ylabel('Average sequence fidelity')
    plt.legend()
    # plt.savefig(pic_path)
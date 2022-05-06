#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:14:26 2021

@author: sxyang
"""
#%%
import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group
from scipy import linalg
import random
import matplotlib.pyplot as plt
import time
# from repro_nonM_210705403 import non_Markovian_theory_Fm
# from fit_RB import compare_Fm

#----------------------------------------------------
# Reproducing Fig.4 unitary non-Markovian noise, 2 qubit, environment is the first qubit.
#----------------------------------------------------

class non_Markovian_theory_Fm():
    def __init__(self, noise_u, proj_O, rho, I, ds):
        self.u = noise_u
        self.M = proj_O
        self.rho = rho
        self.e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
        # self.e is rho_ES trace out the system (2nd qubit).
        self.I = I
        self.ds = ds
        ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
        ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
        self.ket = np.concatenate((ket_0, ket_1)).reshape(2,2)
        # Set self.ket as above so that I can have sum_b (I tensor |b><b|) with an index to pick one of ket_0 or ket_1.
        
    def theory_Fm(self, m):
        # Eq.(7)
        A_part = self._partially_depolar_A(self.rho, m)
        # print('A_part', A_part)
        B_part = self._completely_depolar_B(m)
        # print('B_part', B_part)
        AB = self._non_Markovian_unitary_map(A_part + B_part)
        # Tracing out E.
        AB = np.trace(AB.reshape(2,2,2,2), axis1=0, axis2=2)
        # print('m =', m)
        # print(AB)
        # print('AB', AB)
        theory_Fm = np.trace(self.M @ AB ).real
        return theory_Fm
    
    def _test_difference_A_B(self, m):
        # I found out that it is not correct to tear Am Bm apart, I try Am+Bm, it is not the same with what we want.
        # Try to debug, for ploting the Am, Bm.
        A_part = self._partially_depolar_A(self.rho, m)
        B_part = self._completely_depolar_B(m)
        Am = self._non_Markovian_unitary_map(A_part)
        Bm = self._non_Markovian_unitary_map(B_part)
        Am = np.trace(Am.reshape(2,2,2,2), axis1=1, axis2=3)
        Bm = np.trace(Bm.reshape(2,2,2,2), axis1=1, axis2=3)
        # AB = Am + Bm
        # print('m = ', m)
        # print(AB)
        Am = np.trace(self.M @ Am ).real
        Bm = np.trace(self.M @ Bm ).real
        return Am, Bm, Am+Bm
        
    def _partially_depolar_A(self, rho_ab, m):
        # Eq.8

        rho = rho_ab - np.kron(self.e, self.I/self.ds)
        M = 1
        Am = self._map_E_IS(rho)
        # while loop start from m = 2, so i start from 1 rather than 0.
        while M <= m:
            # A2 = [(dollar1 - theta1) tensor I] (A1)...
            Am = self._map_E_IS(Am)
            M += 1
            # print('Am', Am)
        
        # m + 1 for the power is the number of operations, not just a counter.
        Am = Am / (self.ds**2 - 1)**(m+1)
        return Am
        
    def _completely_depolar_B(self, m):
        # Eq.9. The input should be rho and then tr_S(rho), but I am lazy.
        e = self.e
        tmp = self._Theta_for_nonM_B(e)
        for i in range(1, m):
            tmp = self._Theta_for_nonM_B(tmp)
        Bm = np.kron(tmp, self.I/self.ds)
        return Bm
        
    def _map_E_IS(self, e):
        map_E_IS = np.zeros((4,4), dtype=complex)
        tmpe = np.zeros((2,2), dtype=complex)
        i = 0
        while i < 4:
            b = np.binary_repr(i, width=2)
            # print('Before first call')
            # print(np.kron(self.I, np.conj(self.ket[int(b[1])].reshape(1,2))).shape)
            # print(e.shape)
            # print(np.kron(self.I, self.ket[int(b[0])].reshape(2,1)).shape)
            tmpe = np.kron(self.I, np.conj(self.ket[int(b[1])].reshape(1,2))) @ e @ np.kron(self.I, self.ket[int(b[0])].reshape(2,1))
            # print('After')
            tmpe = self._D_minus_T(tmpe)
            tmp_s = np.kron(tmpe, np.kron(self.ket[int(b[0])].reshape(2,1), np.conj(self.ket[int(b[1])].reshape(1,2))))
            map_E_IS += tmp_s
            i += 1
        return map_E_IS
    
    def _D_minus_T(self, e):
        tmp_d = self._Dollar_for_nonM_A(e)
        tmp_t = self._Theta_for_nonM_B(e)
        d_minus_t = tmp_d - tmp_t
        return d_minus_t
        
    def _Dollar_for_nonM_A(self, e):
        # This part differ from Pedro's.
        # Eq.10
        dollar = np.zeros((2,2), dtype=complex)
        tmp = np.zeros((4,4), dtype=complex)
        i = 0
        while i < 4:
            # Using s to combine two loop into one loop.
            s = np.binary_repr(i, width=2)
            # Prepare the input state, (<b|rho_ab|b'> =) e tensor |s><s'|.
            # print('tmp0', tmp)
            # print('e', e)
            tmp = np.kron(e, np.kron(self.ket[int(s[1])].reshape(2,1), np.conj(self.ket[int(s[0])].reshape(1,2))))
            # print('tmp_state', tmp)
            # if i == 3:
            #     print('ooooooooooooooo')
            # Put the state into unitary nonM map.
            tmp = self._non_Markovian_unitary_map(tmp)

            # print('unitary_nonM', tmp)
            # Apply (Ie tensor <s|) Lambda() (Ie tensor |s'>)
            tmp = np.kron(self.I, np.conj(self.ket[int(s[1])].reshape(1,2))) @ tmp @ np.kron(self.I, self.ket[int(s[0])].reshape(2,1))
            # Sum over s, s' = 1,...,ds (ds =2)
            # print('dollar = ', tmp)
            dollar += tmp
            i += 1
        return dollar

    def _Theta_for_nonM_B(self, e):
        # Eq.11
        Theta = np.trace(self._non_Markovian_unitary_map(np.kron(e, self.I/self.ds)).reshape(2,2,2,2), axis1=1, axis2=3)
        return Theta
        
    def _non_Markovian_unitary_map(self, rho):
        return self.u @ rho @ np.conj(self.u).T
        
def sequence_with_unitary_noise(m, rho, noise_u, I):
    # apply unitary noise in the sequence
    tmp_rho = rho
    inver_op = np.eye(2)
    for i in range(m):
        u_map = unitary_group.rvs(2)
        tmp_rho = noise_u @ np.kron(I, u_map) @ tmp_rho @ np.conj(np.kron(I, u_map)).T @ np.conj(noise_u).T
        
        inver_op = inver_op @ inv(u_map)
    return tmp_rho, inver_op

def Markovianised_map(rho_e, rho_s, noise_u):
    # Input the state of environment, system and non-MArkovian noise, outputs the corresponding Markovinaised noise map in Eq.16.
    # Calculate Eq.16 to obtain A, B (Eq.14)
    # M_noise_u = noise_u * np.kron(rho_e, rho_s) * np.conj(noise_u).T
    M_noise_u = non_Markovian_unitary_map(np.kron(rho_e, rho_s), noise_u)
    M_noise_u = np.trace(M_noise_u.reshape(2,2,2,2), axis1=0, axis2=2)
    return M_noise_u

def non_Markovian_unitary_map(rho, noise_u):
    return noise_u @ rho @ np.conj(noise_u).T
    
def trace_Markovianised(noise_u):
    I = np.eye(2, dtype=complex)
    ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
    ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
    k0 = np.kron(I, np.conj(ket_0.T)) @ noise_u @ np.kron(I, ket_0)
    k1 = np.kron(I, np.conj(ket_1.T)) @ noise_u @ np.kron(I, ket_1)
    tr = np.trace(k0 @ np.conj(k0.T)) + np.trace(k1 @ np.conj(k1.T))
    return tr.real


if __name__ == "__main__":

    #====================================================
    # Seting unitary non-Markovian noise, states and measurement.
    # Set parameters for unitary
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    Z = np.array([[1, 0],[0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    ds = 2   # dim(rho_s)
    de = 2
    # J = 1.7
    # hx = 1.47
    # hy = -1.05
    # delta = 0.03
    J = 1.2
    hx = 1.17
    hy = -1.15
    delta = 0.05

    H = J * np.kron(X,X) + hx * (np.kron(X, I) + np.kron(I, X)) + hy * (np.kron(Y, I) + np.kron(I, Y))
    
    # Non-Markovian noise, assume unitary.
    noise_u = linalg.expm(-1j*delta*H)

    
    seed = 2
    np.random.seed(seed)
    time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    
    ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
    ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
    
    rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
    proj_O = np.kron(ket_0, np.conj(ket_0.T))        # np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T)
    
    M = 20
    #====================================================
    #====================================================
    # # This part generate data and caculate ASF. To debug, don't need to run this part.
    sample_size = int(50)
    fm = np.zeros(sample_size)
    Fm = np.zeros(M)
    # print('Setup state, projector, and some parameters.')
    
    for m in range(1, M+1):
        for i in range(sample_size):
            tmp_rho, inver_op = sequence_with_unitary_noise(m, rho, noise_u, I)
            tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
            # final_state = noise_u @ tmp_rho @ np.conj(noise_u).T
            final_state = non_Markovian_unitary_map(tmp_rho, noise_u)
            f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
            fm[i] = np.trace(proj_O @ f_sys_state).real
            
        Fm[m-1] = np.average(fm)
        print("m = ", str(m), " finished.")
    #====================================================
    
    #====================================================
    # To save some data if needed.
        # if m % 20 == 0:
        #     np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + '_' + noise_mode +'_'+ str(noise_para )+ "_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
            # np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + "_Amp_damp_001_K1_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
    
    # pic_path = "../data/Fm_" + str(M) + "_avg_" + str(saheory_mple_size) + '_' + noise_mode +'_'+ str(noise_para) + "_seed_" + str(seed) + "_" + time_mark + ".png"
    #====================================================
    
    # Set Markovianised Fm
    rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
    rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
    p = ((trace_Markovianised(noise_u) - 1)/(ds**2 - 1))
    A = np.trace(proj_O * Markovianised_map(rho_e, rho_s - I/ds, noise_u)).real
    B = np.trace(proj_O * Markovianised_map(rho_e, I/ds, noise_u)).real
    
    nonM_theory_Fm  = non_Markovian_theory_Fm(noise_u, proj_O, rho, I, ds)
    # nonM_theory_Fm  = non_Markovian_theory_Fm(np.kron(I,I), proj_O, rho, I, ds)
    # theory_Fm = nonM_theory_Fm.theory_Fm(3)
    theory_Fm = np.zeros(M)
    # Am = np.zeros(M)
    # Bm = np.zeros(M)
    for i in range(M):
        theory_Fm[i] = nonM_theory_Fm.theory_Fm(i)
    
    m = np.array(range(1, 1 + M))
    plt.plot(range(1, M+1), Fm, 'o', label='data')
    # plt.plot(range(1, M+1), A*p**m + B, '-', label=r'$F_m^{(M)}$')
    plt.plot(range(1, M+1), theory_Fm, '-', label=r'$F_m$')
    # plt.plot(range(1, M+1), Am, '-', label=r'$A_m$')
    # plt.plot(range(1, M+1), Bm, '-', label=r'$B_m$')
    plt.legend()
    # # plt.title()
    # plt.xlabel('Sequence length')
    # plt.ylabel('Average sequence fidelity')
        
# %%

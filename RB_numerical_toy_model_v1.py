# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:20:36 2021

@author: Hans
"""
#%%
import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group
from scipy import linalg
import matplotlib.pyplot as plt
import time
from fit_RB import compare_Fm

class noise_model():
    def __init__(self, noise_mode, noise_para):
        self.mode = noise_mode
        self.para = noise_para
        
    def apply_noise(self, state):
        if self.mode == 'depolar':
            # model = self._depolarizing_noise(state)
            model = self._depolarizing_noise_v2(state)
        elif self.mode == 'p_flip':
            model = self._phase_flip(state)
        elif self.mode == 'amp_damp':
            model = self._amplitude_damping(state)
        # elif noise_mode == 'b_flip':
            # tmp_rho = b_flip()
        return model
        
    def _depolarizing_noise(self, rho):
        # Something wrong with this function.
        # noise channel should be the same as sum over Kraus.
        # noise_para range from 0~1.5, depolarizing channel.
        noisy_rho = (1-self.para)*rho + (self.para/2)*np.eye(2)
        return noisy_rho
    
    def _depolarizing_noise_v2(self, rho):
        # noise channel should be the same as sum over Kraus.
        lm = 1 - self.para  # range from 0~1.5, depolarizing channel.
        noisy_rho = 1j*np.zeros((2,2))
        lmbda = lm
        K = 1j*np.zeros((4,2,2)) + 1j*np.zeros((4,2,2))
        K[0] = np.sqrt(1-3*lmbda/4)*np.eye(2)
        K[1] = np.sqrt(lmbda/4)*(np.kron(ket_0, ket_1.T) + np.kron(ket_1, ket_0.T))
        K[2] = np.sqrt(lmbda/4)*(-1j*np.kron(ket_0, ket_1.T) + 1j*np.kron(ket_1, ket_0.T))
        K[3] = np.sqrt(lmbda/4)*(np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T))
        for i in range(4):
            noisy_rho += K[i] @ rho @ np.conj(K[i]).T
            # print(noisy_rho)
        return noisy_rho
    
    def _phase_flip(self, rho):
        # noise channel should be the same as sum over Kraus.
        noisy_rho = 1j*np.zeros((2,2))
        ket_0 = np.array([1,0]).reshape(2,1)
        ket_1 = np.array([0,1]).reshape(2,1)
        K = np.zeros((2,2,2)) + 1j*np.zeros((2,2,2))
        K[0] = np.eye(2)*np.sqrt(1-self.para)
        K[1] = np.sqrt(self.para)*(np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T))
        # print(K[0])
        # print(K[1])
        for i in range(1,2):
            noisy_rho += K[i] @ rho @ np.conj(K[i]).T
        return noisy_rho
    
    def _amplitude_damping(self, rho):
        # noise channel should be the same as sum over Kraus.
        noisy_rho = 1j*np.zeros((2,2))
        K = np.zeros((2,2,2)) + 1j*np.zeros((2,2,2))
        K[0] = np.array([[1,0],[0, np.sqrt(1-self.para)]])
        K[1] = np.array([[0, np.sqrt(self.para)],[0,0]])
        # print(K[0])
        # print(K[1])
        for i in range(1,2):
            noisy_rho += K[i] @ rho @ np.conj(K[i]).T
        return noisy_rho

def sequence_with_noise(m, rho, noise_mode, noise_para, seed):
    np.random.seed(seed)
    tmp_rho = rho
    inver_op = np.eye(2)
    u = np.zeros((m, 2, 2), dtype=complex)
    for i in range(m):
        u_map = unitary_group.rvs(2)
        u[i-1] = u_map
        tmp_rho = u_map @ tmp_rho @ np.conj(u_map).T
        
        noise_type = noise_model(noise_mode, noise_para)
        tmp_rho = noise_type.apply_noise(tmp_rho)
        
        inver_op = inver_op @ inv(u_map)
    return tmp_rho, inver_op, noise_type, u


if __name__ == "__main__":
    
    noise_mode = 'depolar'
    noise_para = 0.7
    
    seed = 1
    # np.random.seed(seed)
    time_mark = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    
    ket_0 = np.array([1,0]).reshape(2,1) + 1j*np.zeros((2,1))
    ket_1 = np.array([0,1]).reshape(2,1) + 1j*np.zeros((2,1))
    
    rho = np.array([[1,0],[0,0]]) + 1j*np.zeros((2,2))
    # rho = np.array([[1,1],[1,1]])/2 + 1j*np.zeros((2,2))
    proj_O = np.kron(ket_0, ket_0.T)        # np.kron(ket_0, ket_0.T) - np.kron(ket_1, ket_1.T)

    M = 15
    sample_size = int(2)
    fm = np.zeros((M, sample_size))
    # u = []
    # Fm = np.zeros(M)
    print('Setup state, projector, and some parameters.')
    
    # tmp_rho = phase_flip(rho)
    u = []
    test_var = []
    # Kraus operators have 4 terms to sum, CP map (unitary, CPTP) may not be hermitian,
    # can not multiply rho after sum over.
    # Calculate A rho A+ each iteration.
    for i in range(sample_size):
        usamp = []
        test_var_samp = []
        for m in range(1, M+1):
            tmp_rho, inver_op, noise_type, tmp_u = sequence_with_noise(m, rho, noise_mode, noise_para, seed+i)
            usamp.append(tmp_u)
            tmp_rho = inver_op @ tmp_rho @ np.conj(inver_op).T
            final_state = noise_type.apply_noise(tmp_rho)
            test_var_samp.append(final_state)
            fm[m-1, i] = np.trace(proj_O @ final_state).real
        u.append(usamp)
        test_var.append(test_var_samp)
        # Fm[m-1] = np.average(fm)

    std_exp = np.std(fm, axis=1)
    Fm = np.mean(fm, axis=1)
        
        # if m % 20 == 0:
        #     np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + '_' + noise_mode +'_'+ str(noise_para )+ "_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
            # np.savetxt("../data/Fm_" + str(M) + "_avg_" + str(sample_size) + "_Amp_damp_001_K1_seed_" + str(seed) + "_" + time_mark + ".txt", Fm)
    
    pic_path = "../data/Fm_" + str(M) + "_avg_" + str(sample_size) + '_' + noise_mode +'_'+ str(noise_para) + "_seed_" + str(seed) + "_" + time_mark + ".png"
    compare_Fm(Fm, proj_O, rho, noise_mode, noise_para, pic_path)
    # plt.plot(range(1, M+1), Fm, 'o')
    # plt.title(noise_mode)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Average sequence fidelity')

# %%

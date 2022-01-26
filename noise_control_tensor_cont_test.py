#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:43:57 2022

@author: sxyang
"""

import numpy as np
from scipy import linalg
import random
from random import randint
import tensornetwork as tn
import matplotlib.pyplot as plt
import time

from utils_tn import initialized_lamdas_tn, gen_control_ten, edges_btw_ctr_nois, edges_in_lamdas, contract_edge_list, not_contract_edgs, pop_no_contract_edg, order4_to_2, rX, rY
from nonM_analytical_expression import non_Markovian_theory_Fm, non_Markovian_unitary_map, sequence_with_unitary_noise_list
from reproduce_210705403 import sequence_with_unitary_noise

def rand_clifford_sequence_unitary_noise_list(m, rho, noise_u, rand_clifford):
    # apply unitary noise in the sequence
    # Each step has different noise.
    I = np.eye(2, dtype=complex)
    tmp_rho = rho
    inver_op = np.eye(2)
    # lam_0 is not consider in the previous situation, adding it awkwardly here when all lams are the same.
    tmp_rho = noise_u[0] @ tmp_rho @ np.conj(noise_u[0].T)
    for i in range(m):
        gate = rand_clifford[i]
        tmp_rho = noise_u[i] @ np.kron(I, gate) @ tmp_rho @ np.conj(np.kron(I, gate)).T @ np.conj(noise_u[i]).T
        
        inver_op = inver_op @ np.conj(gate).T
    return tmp_rho, inver_op

clifford_list = [np.identity(2, dtype=complex),
                rX(1/2) @ rY(1/2),
                rY(-1/2) @ rX(-1/2),
                rX(1),
                rX(-1/2) @ rY(-1/2),
                rY(-1/2) @ rX(1/2),
                rY(1),
                rX(1/2) @ rY(-1/2),
                rY(1/2) @ rX(1/2),
                rY(1) @ rX(1),
                rX(-1/2) @ rY(1/2),
                rY(1/2) @ rX(-1/2),
                rX(1) @ rY(1/2),
                rX(-1/2),
                rX(-1/2) @ rY(-1/2) @ rX(1/2),
                rY(-1/2),
                rX(1/2),
                rX(1/2) @ rY(1/2) @ rX(1/2),
                rX(1) @ rY(-1/2),
                rY(1) @ rX(1/2),
                rX(1/2) @ rY(-1/2) @ rX(1/2),
                rY(1/2),
                rY(1) @ rX(-1/2),
                rX(-1/2) @ rY(1/2) @ rX(1/2)]

M = 3

X = np.array([[0, 1],[1, 0]], dtype=complex)
Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
Z = np.array([[1, 0],[0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

ds = 2   # dim(rho_s)
de = 2
J = 1.7
hx = 1.47
hy = -1.05
delta = 0.03

H = J * np.kron(X,X) + hx * (np.kron(X, I) + np.kron(I, X)) + hy * (np.kron(Y, I) + np.kron(I, Y))

# Non-Markovian noise, assume unitary.
# noise_u = linalg.expm(-1j*delta*H)
noise_u = []
# Generate the initial noise of lamda_ms + lamda_m+1, lamda_0
for i in range(M+2):
    noise_u.append(linalg.expm(-1j*delta*H))
    # noise_u.append(np.identity(4, dtype=complex))

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)

rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
proj_O = np.kron(ket_0, np.conj(ket_0.T))
rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)

#====================================================
#====================================================
# # This part generate data and caculate ASF. To debug, don't need to run this part.
sample_size = int(30)
fm = np.zeros(sample_size)
Fm_exp = np.zeros(M)
Fm_cont = np.zeros(M)
# print('Setup state, projector, and some parameters.')

# random.seed(4)
rand_clifford = []
clifford_int = np.zeros(M)
for m in range(M):
    clifford_int[m] = randint(0,23)
    rand_clifford.append(clifford_list[int(clifford_int[m])])

edg_time = np.zeros(M)
cont_time = np.zeros(M)
for m in range(1, M+1):
    # for i in range(sample_size):
    tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(m, rho, noise_u, rand_clifford)
    # 
    tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
    # final_state = noise_u @ tmp_rho @ np.conj(noise_u).T
    final_state = non_Markovian_unitary_map(tmp_rho, noise_u[m])
    f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
        # fm[i] = np.trace(proj_O @ f_sys_state).real
        
    Fm_exp[m-1] = np.trace(proj_O @ f_sys_state).real
    print("m = ", str(m), " finished.")
    
    lamdas = initialized_lamdas_tn(m, noise_u[m], rho_e)
    control_ten = gen_control_ten(rho_s, m, proj_O, rand_clifford)
    
    edg_time_s = time.perf_counter()
    e_edgs = edges_in_lamdas(lamdas, m)
    s_edgs = edges_btw_ctr_nois(control_ten, lamdas, m)
    edg_time_e = time.perf_counter()
    
    con_time_s = time.perf_counter()
    tmp_F = contract_edge_list(e_edgs)
    tmp_F = contract_edge_list(s_edgs, 'ASF')
    con_time_e = time.perf_counter()
    
    edg_time[m-1] = edg_time_e - edg_time_s
    cont_time[m-1] = con_time_e - con_time_s

    Fm_cont[m-1] = np.real(tmp_F.tensor)

# nonM_theory_Fm = non_Markovian_theory_Fm(proj_O, rho, I, ds)
# theory_Fm = np.zeros(M)
# for i in range(M):
#     theory_Fm[i] = nonM_theory_Fm.theory_Fm(i, noise_u[i])

# plt.plot(range(1, M+1), Fm_exp, 'o', label='data')

print('tr', Fm_exp)
# print('Theory', theory_Fm)
print('contract', Fm_cont)
print('edg time', edg_time)
print('cont_time', cont_time)

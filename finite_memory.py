#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:01:28 2022

@author: sxyang
"""

import numpy as np
from scipy import linalg
from random import randint
import tensornetwork as tn

from utils_tn import initialized_lamdas_tn, gen_control_ten, edges_btw_ctr_nois, edges_in_lamdas, contract_edge_list, not_contract_edgs, pop_no_contract_edg, order4_to_2, rX, rY
from nonM_analytical_expression import non_Markovian_theory_Fm

        
clifford_list = [ np.identity(2, dtype=complex),
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

m = 2
moves = 2

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
I = np.eye(2, dtype=complex)
rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
proj_O = np.kron(ket_0, np.conj(ket_0.T))

sys_dim = 2
bond_dim = 2
noise_u = np.identity(sys_dim * bond_dim, dtype=complex)

lamdas = initialized_lamdas_tn(m, noise_u, rho_e)

rand_clifford = []
for i in range(m):
    rand_clifford.append(clifford_list[randint(0,23)])
control_ten = gen_control_ten(rho_s, m, proj_O, rand_clifford)

# -------------------- Generate the F_exp here -------------
F_exp = 0.7*np.ones(m)

# ----------------------------------------------------------
F = np.zeros((moves, m))

for k in range(moves):
    # When updating the lams, also updating the lam_dgs.
    # i did not go to lam_0.
    i = k % (m+1)
    e_edgs = edges_in_lamdas(lamdas, m)
    s_edgs = edges_btw_ctr_nois(control_ten, lamdas, m)
    
    tmp_F = contract_edge_list(e_edgs)
    tmp_F = contract_edge_list(s_edgs, 'ASF')

    F[k,m-1] = np.real(tmp_F.tensor)
    
    
    for j in range(1,m+1):
        # This part has some redundent calculations, some F_pn did not change every time also they might not contribute to the cost func. But it will make things more complex.
        # Take the lamdas of sequence depth j and transform to order 2 tensor formate to calculate the analytical F.
        noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-j,m+2)))
        analytical_F = non_Markovian_theory_Fm(proj_O, rho, I, sys_dim)
        F[k,j-1] = analytical_F.theory_Fm(j, noise_u[j-1])
        
            
    # math: beta_{n} tilde_Theta_{n}^{i, i-1}
    # python: beta_{pn} tilde_Theta_{pn}^{i, i+1}
    tilde_theta_2 = np.zeros([2]*6, dtype=complex)
    for l in range(i+1):
        pn = m-l
        beta = F_exp[pn-1] - F[k, pn-1]
        exclude_2 = [i, i+1]
        exclude_1 = [i+1]
        if pn == m:
            tilde_ctr = control_ten
            lam_ex_2 = edges_in_lamdas(lamdas, m)
            ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr, lamdas, m)
        elif pn == 0:
            break
        else:
           # Take the lamdas for tilde_Theta_(pn)
            noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(pn,m+2)))
            tilde_lam = initialized_lamdas_tn(pn, noise_u, rho_e)
            # Take the gates for tilde_Theta(pn)
            Gfin_dg_axis = control_ten[m-pn].axis_names
            Gfin_axis = control_ten[-(m-pn+2)].axis_names
            tilde_ctr = control_ten[m-pn+1:-(m-pn+2)]
            M_axis = ['z\''+str(pn+1), 's'+str(pn+1)]
            M = tn.Node(proj_O, name='M', axis_names=M_axis)
            tmp_ctr_dg = np.identity(2, dtype=complex)
            for ctr_l in range(pn):
                tmp_ctr_dg = np.conj(tilde_ctr[ctr_l].tensor.T) @ tmp_ctr_dg
            tmp_ctr = tn.Node(np.conj(tmp_ctr_dg.T), name='Gfin', axis_names=Gfin_axis)
            tmp_ctr_dg = tn.Node(tmp_ctr_dg, name='Gfin_dg', axis_names=Gfin_dg_axis)
            
            tilde_ctr.insert(0, tmp_ctr_dg)
            tilde_ctr.append(tmp_ctr)
            tilde_ctr.append(M)
            
            # define the edges for tilde_Theta(pn)
            lam_ex_2 = edges_in_lamdas(tilde_lam, pn)
            ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr, tilde_lam, pn)
        # print(len(ctr_ex_2))
        # print(len(lam_ex_2))
        # print('ctr',ctr_ex_2)
        # After pop_no_contract_edg, the list will only contain the edges need to be contracted.
        pop_no_contract_edg(exclude_2, ctr_ex_2, lam_ex_2)
        tmp_theta_2 = contract_edge_list(lam_ex_2)
        tmp_theta_2 = contract_edge_list(ctr_ex_2, name='theta2_'+str(pn))
        tmp_theta_2 = tn.contract_between(tilde_ctr[i], tmp_theta_2, name='theta2_'+str(pn), allow_outer_product=True)

        tilde_theta_2 += beta * tmp_theta_2.tensor
        # ============ tilde_theta_2 is done, now a tilde_theta_1 =========================
        # tilde_theta_1 = contract_edge_list(lam_ex_1)
        # tilde_theta_1 = contract_edge_list(ctr_ex_1, name='theta1_'+str(m+1-l))
        
        # dL = (F_exp[j] - F[k,j]) * Theta_2(j+) + (F_exp[j+2] - F[k,j+2]) * Theta_1(i+1)
    



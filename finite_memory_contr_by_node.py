#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:47:54 2022

@author: sxyang
"""

import numpy as np
import random
from random import randint
import tensornetwork as tn

from utils_tn import initialized_lamdas_tn, gen_control_ten
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2
from utils_tn import rX, rY, contract_by_nodes, noise_nonM_unitary
from utils_tn import non_Markovian_unitary_map, rand_clifford_sequence_unitary_noise_list
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

m = 6
moves = 10
lr = tn.Node(0.001)

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
I = np.eye(2, dtype=complex)
rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
proj_O = np.kron(ket_0, np.conj(ket_0.T))

sys_dim = 2
bond_dim = 2
# noise_u = np.identity(sys_dim * bond_dim, dtype=complex)
noise_u = noise_nonM_unitary(m)

lamdas = initialized_lamdas_tn(m, noise_u, rho_e)

random.seed(2)
rand_clifford = []
for i in range(m):
    rand_clifford.append(clifford_list[randint(0,23)])
control_ten = gen_control_ten(rho_s, m, proj_O, rand_clifford)

# -------------------- Generate the F_exp here -------------
F_exp = np.zeros(m)
# F_exp, non-M unitary noise from Pedro's work.
for lg in range(1, m+1):
    # for i in range(sample_size):
    tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(lg, rho, noise_u, rand_clifford)
    # 
    tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
    # final_state = noise_u @ tmp_rho @ np.conj(noise_u).T
    final_state = non_Markovian_unitary_map(tmp_rho, noise_u[lg])
    f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
        # fm[i] = np.trace(proj_O @ f_sys_state).real
        
    F_exp[lg-1] = np.trace(proj_O @ f_sys_state).real
# print(F_exp)

# ----------------------------------------------------------
F = np.zeros((moves, m))
test_f = np.zeros((moves, m))
test_f_1 = np.zeros(moves)

for k in range(moves):
    for node in lamdas:
        node.fresh_edges(node.axis_names)

    for node in control_ten:
        node.fresh_edges(node.axis_names)
    
    # When updating the lams, also updating the lam_dgs.
    # i did not go to lam_0.
    i = k % (m+1)
    
    # edges for constructing Fm, m = m.
    e_edgs = edges_in_lamdas(lamdas, m)
    s_edgs = edges_btw_ctr_nois(control_ten, lamdas, m)
    
    tmp_F = contract_by_nodes(lamdas+control_ten, None, 'F')
    
    F[k,m-1] = np.real(tmp_F.tensor)
    
    # Fm from analytical expression, m = 1, ..., m-1
    for j in range(1,m+1):
        # This part has some redundent calculations, some F_pn did not change every time also they might not contribute to the cost func. But it will make things more complex.
        # Take the lamdas of sequence depth j and transform to order 2 tensor formate to calculate the analytical F.
        """
        Error here, don't know why yet.
        """
        
        noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-j,m+2)))
        analytical_F = non_Markovian_theory_Fm(proj_O, rho, I, sys_dim)
        F[k,j-1] = analytical_F.theory_Fm(j, noise_u[j-1])
        
    # L_{i,i+1} for checking correctness.
    L_edg = lamdas[i][-1] ^ lamdas[i+1][0]
    L_dangled = lamdas[i].get_all_dangling() + lamdas[i+1].get_all_dangling()
    L_axis_names = list(map(lambda edg: edg.name, L_dangled))
    L = contract_by_nodes([lamdas[i], lamdas[i+1]], L_dangled, 'L')
    L_left = L.get_all_dangling()
    
    # math: beta_{n} tilde_Theta_{n}^{i, i-1}, i = m+1, m, ..., 1
    # tilde_Theta_{n}^{i, i-1} is a function takes n as input so, drop +1 here.
    # python: beta_{pn} tilde_Theta_{pn}^{i, i+1}, i = 0, ..., m
    tilde_theta_2 = np.zeros([2]*6, dtype=complex)

    grad2 = tn.Node(np.zeros(L.shape))
    for l in range(i+1):
        pn = m-l
        exclude_2 = [i-l, i+1-l]
        beta_pn = tn.Node(F_exp[pn-1] - F[k, pn-1])
        if pn == m:
            tilde_ctr = control_ten
            tilde_lam = lamdas
            lam_ex_2 = edges_in_lamdas(tilde_lam, m)
            # lam_ex_2 = edges_in_lamdas(lamdas, m)
            ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr, lamdas, m)
            
        elif pn == 0:
             break
        else:
            '''
            something went wrong here, pn == m, i=0 is fine.
            issues: it did not always return 1 for k>1 in the test case.
            '''
            # Take the lamdas for tilde_Theta_(pn)
            noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(l, m+2)))

            tilde_lam = initialized_lamdas_tn(pn, noise_u, rho_e)
            # Take the gates for tilde_Theta(pn)
            tilde_ctr = control_ten[m-pn+1:-(m-pn+2)]
            M_axis = ['z\''+str(pn+1), 's'+str(pn+1)]
            M = tn.Node(proj_O, name='M', axis_names=M_axis)
            
            Gfin_axis = control_ten[m-pn].axis_names
            # Did not take the axis_names of Gfin_dg of control_ten,
            # since tilde_ctr is shorter than it, and the axis_names should be continuous.
            Gfin_dg_axis = control_ten[-(m-pn+2)].axis_names
            tmp_ctr = np.identity(2, dtype=complex)
            for ctr_l in range(pn):
                tmp_ctr = np.conj(tilde_ctr[ctr_l].tensor.T) @ tmp_ctr
            tmp_ctr_dg = tn.Node(np.conj(tmp_ctr.T), name='Gfin_dg', axis_names=Gfin_dg_axis)
            tmp_ctr = tn.Node(tmp_ctr, name='Gfin', axis_names=Gfin_axis)
            
            tilde_ctr.insert(0, tmp_ctr)
            tilde_ctr.append(tmp_ctr_dg)
            tilde_ctr.append(M)
            
            # define the edges for tilde_Theta(pn)
            lam_ex_2 = edges_in_lamdas(tilde_lam, pn)
            ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr, tilde_lam, pn)

        exclude_nodes_2 = [tilde_lam[exclude_2[0]]] + [tilde_lam[exclude_2[1]]]
        tilde_theta_2_ls = []
        for n in tilde_lam:
            if n not in exclude_nodes_2:
                tilde_theta_2_ls.append(n)
        tilde_theta_2_ls = tilde_theta_2_ls + tilde_ctr
        tilde_theta_2 = contract_by_nodes(tilde_theta_2_ls, None, 'til_theta2')
        tt2_name = list(map(lambda edg: edg.name, tilde_theta_2.get_all_edges()))
        tilde_theta_2.add_axis_names(tt2_name)
        tt2_reorder = list(map(lambda name: tilde_theta_2[name], L_axis_names))
        tilde_theta_2.reorder_edges(tt2_reorder)
        '''
        Since the code isn't finished, without contract all nodes, remaining some
        undangling edges would cause problem in loop.
        '''
        # con_ls = [tilde_theta_2] + exclude_nodes_2
        # test_f[k,pn-1] = contract_by_nodes(con_ls, None, 'f').tensor.real
        grad2 += beta_pn * tilde_theta_2
    grad2.add_axis_names(tilde_theta_2.axis_names)
    for x in range(len(tilde_theta_2.edges)):
        grad2[x].set_name(tilde_theta_2.axis_names[x])

    if i+2 >= 1 and m+2-(i+2) >= 3:
        # exclude_1 = [0] # always the left most node.
        beta_1 = tn.Node(F_exp[m-(i+2)-1] - F[k, m-(i+2)-1])
        # Take the lamdas for tilde_Theta_(pn)
        noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(i+1, m+2)))
    
        tilde_lam = initialized_lamdas_tn(m-(i+1), noise_u, rho_e)
        # Take the gates for tilde_Theta(m-(i+2))
        tilde_ctr = control_ten[(i+2):-((i+2)+1)]
        M_axis = ['z\''+str(m-(i+1)+1), 's'+str(m-(i+1)+1)]
        M = tn.Node(proj_O, name='M', axis_names=M_axis)
        
        Gfin_axis = control_ten[i+1].axis_names
        # Did not take the axis_names of Gfin_dg of control_ten,
        # since tilde_ctr is shorter than it, and the axis_names should be continuous.
        Gfin_dg_axis = control_ten[-((i+1)+2)].axis_names
        tmp_ctr = np.identity(2, dtype=complex)
        for ctr_l in range(int((len(tilde_ctr)-1)/2)):
            tmp_ctr = np.conj(tilde_ctr[ctr_l].tensor.T) @ tmp_ctr
        tmp_ctr_dg = tn.Node(np.conj(tmp_ctr.T), name='Gfin_dg', axis_names=Gfin_dg_axis)
        tmp_ctr = tn.Node(tmp_ctr, name='Gfin', axis_names=Gfin_axis)
        
        tilde_ctr.insert(0, tmp_ctr)
        tilde_ctr.append(tmp_ctr_dg)
        tilde_ctr.append(M)
        
        # define the edges for tilde_Theta(pn)
        lam_ex_1 = edges_in_lamdas(tilde_lam, m-(i+1))
        ctr_ex_1 = edges_btw_ctr_nois(tilde_ctr, tilde_lam, m-(i+1))
        tilde_theta_1 = contract_by_nodes(tilde_lam[1:]+tilde_ctr, None, 'tilde_theta_1')
        
        tt1_name = list(map(lambda edg: edg.name, tilde_theta_1.get_all_edges()))
        tilde_theta_1.add_axis_names(tt1_name)

        tilde_theta_1.fresh_edges(tilde_theta_1.axis_names)
        ele_inv_lam = tn.Node(1) / lamdas[i]
        for x in range(len(ele_inv_lam.edges)):
            ele_inv_lam[x].set_name(lamdas[i].axis_names[x])
        ele_inv_lam.set_name('inv')
        ele_inv_lam.add_axis_names(lamdas[i].axis_names)
        cont_edg = list(set(lamdas[i].axis_names) & set(tilde_theta_1.axis_names))[0]
        ele_inv_lam[cont_edg] ^ tilde_theta_1[cont_edg]
        # tilde1 = ele_inv_lam @ tilde_theta_1
        tilde_theta_1 = tilde_theta_1 @ ele_inv_lam
        tilde1_name = list(map(lambda edg: edg.name, tilde_theta_1.get_all_edges()))
        tilde_theta_1.add_axis_names(tilde1_name)
        grad1 = beta_1 * tilde_theta_1
        for x in range(len(tilde_theta_1.edges)):
            grad1[x].set_name(tilde_theta_1[x].name)
        grad1.add_axis_names(tilde_theta_1.axis_names)

        # test_f_1[k] = contract_by_nodes([tilde_theta_1]+[tilde_lam[0]], None, 'test_f_1').tensor.real
    testa = grad1.tensor[0,0,0,0]
    # Reordering grad1 and grad2.
    grad1_reorder = list(map(lambda name: grad1[name], grad2.axis_names))
    grad1.reorder_edges(grad1_reorder)
    testb = grad1.tensor[0,0,0,0]
    L -= lr * (grad2 + grad1)
    for x in range(len(L.edges)):
        L[x].set_name(grad1[x].name)
    L.add_axis_names(grad1.axis_names)
    
    # SVD on L
    u_prime, vh_prime, _ = tn.split_node(L, left_edges=[L[0],L[1],L[2]], right_edges=[L[3],L[4],L[5]])
    u_prime.set_name(lamdas[i].name)
    u_prime.add_axis_names(lamdas[i].axis_names)
    u_prime[-1].set_name(lamdas[i].axis_names[-1])
    vh_prime.set_name(lamdas[i+1].name)
    vh_prime.add_axis_names(lamdas[i+1].axis_names)
    vh_prime[0].set_name(lamdas[i+1].axis_names[0])
    lamdas[i] = u_prime
    lamdas[i+1] = vh_prime
# Note here, the order of test_f_1 and test_f is not the same.
# print('test_f', np.matrix.round(test_f,4).real)
# print('test_f_1', np.matrix.round(test_f_1,4).real)

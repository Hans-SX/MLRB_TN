#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:47:54 2022

@author: sxyang
"""
#%%
from signal import pause
from tracemalloc import stop
import numpy as np
import random
from random import randint
from scipy import linalg
import tensornetwork as tn
import matplotlib.pyplot as plt

from utils_tn import initialized_lamdas_tn, gen_control_ten, order2_to_4
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2, single_cliffords
from utils_tn import contract_by_nodes, noise_nonM_unitary
from utils_tn import non_Markovian_unitary_map, rand_clifford_sequence_unitary_noise_list
from nonM_analytical_expression import non_Markovian_theory_Fm


m = 6
moves = 10
lr = tn.Node(0.1)  # tn.Node(0.0001)
delta = 100 # >=1, stopping condition

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
I = np.eye(2, dtype=complex)
rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
proj_O = np.kron(ket_0, np.conj(ket_0.T))

sys_dim = 2
bond_dim = 2
init_noise = np.identity(sys_dim * bond_dim, dtype=complex)
# init_noise = noise_nonM_unitary(m)
noise_u = noise_nonM_unitary(m, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
# init_noise = noise_nonM_unitary(m, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
# noise_u = noise_nonM_unitary(m)


lamdas = initialized_lamdas_tn(m, init_noise, rho_e)

clifford_list = single_cliffords()
random.seed(2)
rand_clifford = []
sample_size = 1000
# print(rand_clifford)
for i in range(m):
    rand_clifford.append(clifford_list[randint(0,23)])
control_ten = gen_control_ten(rho_s, m, proj_O, rand_clifford)

control_lg = []
for j in range(m):
    control_lg.append(gen_control_ten(rho_s, j+1, proj_O, rand_clifford[:j+1]))

# -------------------- Generate the F_exp here -------------
# F_exp = np.ones(m) # if nois_u are identities.
F_e = np.zeros((m, sample_size))
# std_exp = np.zeros(m)
# F_exp, non-M unitary noise from Pedro's work.
for sam in range(sample_size):
    rand_clifford_sample = []
    for i in range(m):
        rand_clifford_sample.append(clifford_list[randint(0,23)])
    for lg in range(1, m+1):
        tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(lg, rho, noise_u, rand_clifford_sample)
        tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
        final_state = non_Markovian_unitary_map(tmp_rho, noise_u[lg])
        f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
        F_e[lg-1, sam] = np.trace(proj_O @ f_sys_state).real
std_exp = np.std(F_e, axis=1)
F_exp = np.mean(F_e, axis=1)
# print(F_exp)

# ----------------------------------------------------------
F = np.zeros((moves, m))
# test_f = np.zeros((moves, m))
# test_f_1 = np.zeros(moves)
eigval = []
grad2s = []
# For updating lam^dagger
complement = 2 * (m+2) - 1
#%%
for k in range(0, moves):
    for node in lamdas:
        node.fresh_edges(node.axis_names)

    for node in control_ten:
        node.fresh_edges(node.axis_names)
    
    # When updating the lams, also updating the lam_dgs.
    # i did not go to lam_0.
    i = m - (k % (m+1))

    # edges for constructing Fm, m = m.
    e_edgs = edges_in_lamdas(lamdas, m)
    s_edgs = edges_btw_ctr_nois(control_ten, lamdas, m)
    
    tmp_F = contract_by_nodes(lamdas+control_ten)
    
    # Should it natural be real?
    F[k,m-1] = np.abs(tmp_F.tensor)
    
    # Fm from analytical expression, m = 1, ..., m-1
    # for j in range(1,m+1):
    #     # This part has some redundent calculations, some F_pn did not change every time also they might not contribute to the cost func. But it will make things more complex.
    #     # Take the lamdas of sequence depth j and transform to order 2 tensor formate to calculate the analytical F.
        
    #     noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-j,m+2)))
    #     analytical_F = non_Markovian_theory_Fm(proj_O, rho, I, sys_dim)
    #     F[k,j-1] = analytical_F.theory_Fm(j, noise_u[j-1])
    '''
    The negative ASF cause by did not update lam^dagger (now fixed).
    The problem now is that ASF fluctuation severly and did not converge.
    Try to substitue this by contraction.
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for lg in range(1, m):
        noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-lg,m+2)))
        tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(lg, rho, noise_tmp, rand_clifford)
        tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
        final_state = non_Markovian_unitary_map(tmp_rho, noise_tmp[lg])
        f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
        F[k, lg-1] = np.trace(proj_O @ f_sys_state).real
    '''
    for lg in range(1,m):
        noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-lg,m+2)))
        lam_lg = initialized_lamdas_tn(lg, noise_tmp, rho_e)
        lam_lg_edg = edges_in_lamdas(lam_lg, lg)
        ctr_lg_edg = edges_btw_ctr_nois(control_lg[lg-1], lam_lg, lg)
        tmp_lg = tn.contractors.auto(lam_lg+control_lg[lg-1], None, ignore_edge_order=True)
        F[k, lg-1] = np.abs(tmp_lg.tensor)

    # L_{i,i+1} for checking correctness.
    L_edg = lamdas[i][-1] ^ lamdas[i+1][0]
    L_dangled = lamdas[i].get_all_dangling() + lamdas[i+1].get_all_dangling()
    L_axis_names = list(map(lambda edg: edg.name, L_dangled))
    L = contract_by_nodes([lamdas[i], lamdas[i+1]], L_dangled, 'L', ignore=False)
    L_left = L.get_all_dangling()
    # print(L.tensor)

    # math: beta_{n} tilde_Theta_{n}^{i, i-1}, i = m+1, m, ..., 1
    # tilde_Theta_{n}^{i, i-1} is a function takes n as input so, drop +1 here.
    # python: beta_{pn} tilde_Theta_{pn}^{i, i+1}, i = 0, ..., m
    tilde_theta_2 = np.zeros([2]*6, dtype=complex)

    grad2 = tn.Node(np.zeros(L.shape))
    beta = 0
    for l in range(i+1):
        pn = m-l
        exclude_2 = [i-l, i+1-l]
        beta_pn = tn.Node(F_exp[pn-1] - F[k, pn-1])
        if pn == m:
            tilde_ctr = control_ten
            tilde_lam = lamdas
            lam_ex_2 = edges_in_lamdas(tilde_lam, m)
            ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr, lamdas, m)
            
        elif pn == 0:
             break
        else:

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

        # con_ls = [tilde_theta_2] + exclude_nodes_2
        # test_f[k,pn-1] = contract_by_nodes(con_ls, None, 'f').tensor.real
        grad2 += beta_pn * tilde_theta_2
        beta += abs(beta_pn.tensor)
    if beta <= np.sum(std_exp) / delta:
        # for lam in lamdas:
        #     print(lam.name, np.matrix.round(lam.tensor, 4))
        print('k', k)
        print('beta', beta)
        print('std_exp', std_exp)
        print('sig', sig.tensor)
        print('allsig', allsig)
        break
    grad2.add_axis_names(tilde_theta_2.axis_names)
    for x in range(len(tilde_theta_2.edges)):
        grad2[x].set_name(tilde_theta_2.axis_names[x])

    # print(beta)
    L -= lr * grad2
    grad2s.append(grad2)
    # print(L.tensor)
    # break
    for x in range(len(L.edges)):
        L[x].set_name(grad2[x].name)
    L.add_axis_names(grad2.axis_names)

    # SVD on L
    svd_tn = True
    if svd_tn == True:
        # u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
        #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
        u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
                        right_edges=[L[3],L[4],L[5]], max_singular_values=2)
        # u_prime, vh_prime, tranc = tn.split_node(L, left_edges=[L[0],L[1],L[2]], 
                        # right_edges=[L[3],L[4],L[5]], max_singular_values=None)
        # To check the unitariry, u_prime need to put in order4_to_2,
        # while vh_prime could be reshape directly.
        # print(np.matrix.round(u_prime.tensor,10))
        # print(np.matrix.round(grad2.tensor, 8))
        # print(vh_prime.tensor)
        
    else:
        u = np.zeros((bond_dim, sys_dim, sys_dim, bond_dim), dtype=complex)
        vh = np.zeros((bond_dim, sys_dim, sys_dim, bond_dim), dtype=complex)
        eig = []
        for s1 in range(sys_dim):
            for s2 in range(sys_dim):
                for s3 in range(sys_dim):
                    for s4 in range(sys_dim):
                        u[:,s1,s2,:], s, vh[:,s3,s4,:] = np.linalg.svd(L.tensor[:,s1,s2,s3,s4,:], full_matrices=True)
                        eig.append(s)
        
    # print(u_prime.tensor)
    # L2 = tn.contractors.auto([u_prime, vh_prime], 
    #         output_edge_order=[u_prime[0], u_prime[1], u_prime[2], vh_prime[2], vh_prime[1], vh_prime[3]])
    # L2 = tn.contractors.auto([u_prime, vh_prime], 
    #         ignore_edge_order=True)
    # print(np.matrix.round(L.tensor - L2.tensor, 9))
    # print(np.matrix.round(L2.tensor, 8))
    # print('u_prime', np.matrix.round(u_prime.tensor,8))
    # print('vh_prime', np.matrix.round(vh_prime.tensor,8))
    # break
    qr = True
    if qr == True:
        if svd_tn == True:
            u = order4_to_2(u_prime.tensor)
            vh = order4_to_2(vh_prime.tensor)
        else:
            u = order4_to_2(u)
            vh = order4_to_2(vh)
        uq, ur = linalg.qr(u)
        vr, vq = linalg.rq(vh)
        uvh = uq @ np.conj(vq).T
        uvh = order2_to_4(uvh)
        uq = order2_to_4(uq)
        vq = order2_to_4(vq)
        u_prime = tn.Node(uvh)
        vh_prime = tn.Node(uvh)
    # else:
    #     u_prime = tn.Node(u)
    #     vh_prime = tn.Node(vh)
    '''
    Here u from 2 separate trials are not the same when k = 1.
    Lprime_svd (got back from u s vh) are not the same.
    But gradients and Lprime are the same.
    '''
    # if k == 1:
    #     # tmpt = tn.contractors.auto([u_prime]+[sig]+[vh_prime], ignore_edge_order=True).tensor
    #     # print(np.matrix.round(tmpt, 6))
    #     # print(np.matrix.round(grad2.tensor, 6))
    #     print(np.matrix.round(L.tensor, 6))
    #     # print(np.matrix.round(u_prime.tensor, 6))
    #     # print(allsig)
    #     break
    
    u_prime.set_name(lamdas[i].name)
    u_prime.add_axis_names(lamdas[i].axis_names)
    # u_prime[-1].set_name(lamdas[i].axis_names[-1])
    # vh_prime (e,s,e',s') -> (e,s,s',e') if u_prime = vh_prime do I need to reorder?
    # vh_prime.reorder_axes([0,1,3,2])
    vh_prime.set_name(lamdas[i+1].name)
    vh_prime.add_axis_names(lamdas[i+1].axis_names)
    # vh_prime[0].set_name(lamdas[i+1].axis_names[0])
    lamdas[i] = u_prime
    lamdas[i+1] = vh_prime
    # For updating lam^dagger
    ud_ind = complement - i
    vhd_ind = complement - (i+1)
    ud_prime = tn.Node(np.conj(u_prime.tensor).T)
    ud_prime.set_name(lamdas[ud_ind].name)
    ud_prime.add_axis_names(lamdas[ud_ind].axis_names)
    # ud_prime[-1].set_name(lamdas[ud_ind].axis_names[-1])
    vhd_prime = tn.Node(np.conj(vh_prime.tensor).T)
    vhd_prime.set_name(lamdas[vhd_ind].name)
    vhd_prime.add_axis_names(lamdas[vhd_ind].axis_names)
    # vhd_prime[0].set_name(lamdas[vhd_ind].axis_names[0])
    lamdas[complement - i] = ud_prime
    lamdas[complement - (i+1)] = vhd_prime
    # for n in range(m):
    #     print('k,m', k, n, F[k,n])

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(range(1,m+1), F_exp, s=10, c='b', marker="s", label='F_exp')
colorls = ['g','r','c','m','y','k','w']
for p in range(7):
    ax1.scatter(range(1,m+1), F[1+p,:], s=10, c=colorls[p], marker="o", label='u'+str(1+p))
plt.legend(loc='upper left')
plt.show()

#%%
# print('F_exp', F_exp)
# print(F[:10])
# print(F[:,-1])
# print(F[k,:])
# print(max(F[20:,-1]))
# print(F)
# Note here, the order of test_f_1 and test_f is not the same.
# print('test_f', np.matrix.round(test_f,4).real)
# print('test_f_1', np.matrix.round(test_f_1,4).real)
# %%
import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt

int_dis = np.zeros(100000)
for i in range(100000):
    int_dis[i] = randint(0,23)
# %%

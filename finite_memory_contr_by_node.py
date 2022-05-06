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
from scipy.stats import unitary_group
import tensornetwork as tn
import matplotlib.pyplot as plt

from utils_tn import initialized_lamdas_tn, gen_control_ten, order2_to_4
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2, single_cliffords
from utils_tn import contract_by_nodes, noise_nonM_unitary
from utils_tn import non_Markovian_unitary_map, rand_clifford_sequence_unitary_noise_list
from utils_tn import ASF_learning_plot
from nonM_analytical_expression import non_Markovian_theory_Fm


m = 20  # m=3, F[27]; m=6, F[35] closest to F_exp
# m = 20, F[26], 26min, (delta ~ 7)
moves = 50
lr = tn.Node(1)  # tn.Node(0.0001)
gamma = 1  # make the cost bigger
delta = 8 # >=1, stopping condition

sample_size = 50

random.seed(5)  # seed 2 -> gamma 5. seed 5 -> gamma 2

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
I = np.eye(2, dtype=complex)
rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
rho_s = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
rho_e = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
proj_O = np.kron(ket_0, np.conj(ket_0.T))

sys_dim = 2
bond_dim = 2
d = sys_dim * bond_dim
init_noise = np.identity(sys_dim * bond_dim, dtype=complex)
# init_noise = noise_nonM_unitary(m-1, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
# init_noise.insert(0, np.identity(4, dtype=complex))
noise_u = noise_nonM_unitary(m, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
# noise_u = np.identity(sys_dim * bond_dim, dtype=complex)
# noise_u = unitary_group.rvs(4)
# init_noise = noise_u.copy()
# init_noise.reverse()
# init_noise.insert(0, np.identity(sys_dim * bond_dim, dtype=complex))

lamdas = initialized_lamdas_tn(m, init_noise, rho_e)

clifford_list = single_cliffords()
rand_clifford = []

sam_clif = np.zeros((sample_size, m, 2, 2), dtype=complex)
control_ten = []
control_n = []
sam_n = dict()
for sam in range(sample_size):
    for i in range(m):
        sam_clif[sam, i] = clifford_list[randint(0,23)]
    control_ten.append(gen_control_ten(rho_s, m, proj_O, sam_clif[sam]))
    for j in range(m-1):
        control_n.append(gen_control_ten(rho_s, j+1, proj_O, sam_clif[sam, :j+1]))
    sam_n[sam] = control_n
    control_n = []

# -------------------- Generate the F_exp here -------------
# F_exp = np.ones(m) # if nois_u are identities.
#%%
F_e = np.zeros((m, sample_size))
# std_exp = np.zeros(m)
# F_exp, non-M unitary noise from Pedro's work.
for sam in range(sample_size):
    for n in range(1, m+1):
        tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(n, rho, noise_u, sam_clif[sam, :n])
        tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
        if type(noise_u) == type([]):
            final_state = non_Markovian_unitary_map(tmp_rho, noise_u[n-1])
        else:
            final_state = non_Markovian_unitary_map(tmp_rho, noise_u)
        f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
        F_e[n-1, sam] = np.trace(proj_O @ f_sys_state).real
std_exp = np.std(F_e, axis=1)
F_exp = np.mean(F_e, axis=1)
# print(F_exp)

# ----------------------------------------------------------
F = np.zeros((moves, m))
f = np.zeros(moves)
# test_f = np.zeros((moves, m))
# test_f_1 = np.zeros(moves)
grad2s = []
costs = []
# For updating lam^dagger
complement = 2 * (m+2) - 1
#%%
for k in range(0, moves):
    for node in lamdas:
        node.fresh_edges(node.axis_names)

    for nodes in control_ten:
        for node in nodes:
            node.fresh_edges(node.axis_names)
        
    # When updating the lams, also updating the lam_dgs.
    # i did not go to lam_1, no update on lam_0.
    # The noise is from operation, assume prepare noise is fixed.
    i = m - (k % m) - 1
    tmp_F = np.zeros(sample_size, dtype=complex)
    tmp_Fn = np.zeros((sample_size, m-1), dtype=complex)
    for sam in range(sample_size):
        # edges for constructing Fm, m = m.
        e_edgs = edges_in_lamdas(lamdas, m)
        s_edgs = edges_btw_ctr_nois(control_ten[sam], lamdas, m)
        
        tmp_F[sam] = contract_by_nodes(lamdas+control_ten[sam]).tensor
        
        for n in range(1,m):
            noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(m-n,m+2)))
            noise_tmp.reverse()
            lam_n = initialized_lamdas_tn(n, noise_tmp, rho_e)
            lam_n_edg = edges_in_lamdas(lam_n, n)
            ctr_n_edg = edges_btw_ctr_nois(sam_n[sam][n-1], lam_n, n)
            tmp_n = tn.contractors.auto(lam_n+sam_n[sam][n-1], None, ignore_edge_order=True)
            tmp_Fn[sam, n-1] = np.abs(tmp_n.tensor)
    F[k, :m-1] = np.mean(np.abs(tmp_Fn), axis=0)
    F[k,m-1] = np.mean(np.abs(tmp_F))
    # break
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

    grad2 = tn.Node(np.zeros(L.shape))
    beta = 0
    cost = 0
    for l in range(i+1):
        pn = m-l
        if pn == 0:
            break
        exclude_2 = [i-l, i+1-l]
        beta_pn = tn.Node(F_exp[pn-1] - F[k, pn-1])*gamma  # F[k, pn-1] is averaged.
        tilde_theta_2 = dict()
        tilde_ctr = dict()
        for sam in range(sample_size):
            for node in lamdas:
                node.fresh_edges(node.axis_names)
            if pn == m:
                tilde_ctr[sam] = control_ten[sam]
                tilde_lam = lamdas
                lam_ex_2 = edges_in_lamdas(tilde_lam, m)
                ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr[sam], tilde_lam, m)
                
            else:

                # Take the lamdas for tilde_Theta_(pn)
                noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor), np.arange(l, m+2)))

                tilde_lam = initialized_lamdas_tn(pn, noise_u, rho_e)
                # Take the gates for tilde_Theta(pn)
                tilde_ctr[sam] = control_ten[sam][m-pn+1:-(m-pn+2)]
                M_axis = ['z\''+str(pn+1), 's'+str(pn+1)]
                M = tn.Node(proj_O, name='M', axis_names=M_axis)
                
                Gfin_axis = control_ten[sam][m-pn].axis_names
                # Did not take the axis_names of Gfin_dg of control_ten,
                # since tilde_ctr is shorter than it, and the axis_names should be continuous.
                Gfin_dg_axis = control_ten[sam][-(m-pn+2)].axis_names
                tmp_ctr = np.identity(2, dtype=complex)
                for ctr_l in range(pn):
                    tmp_ctr = np.conj(tilde_ctr[sam][ctr_l].tensor.T) @ tmp_ctr
                tmp_ctr_dg = tn.Node(np.conj(tmp_ctr.T), name='Gfin_dg', axis_names=Gfin_dg_axis)
                tmp_ctr = tn.Node(tmp_ctr, name='Gfin', axis_names=Gfin_axis)
                
                tilde_ctr[sam].insert(0, tmp_ctr)
                tilde_ctr[sam].append(tmp_ctr_dg)
                tilde_ctr[sam].append(M)
                
                # define the edges for tilde_Theta(pn)
                lam_ex_2 = edges_in_lamdas(tilde_lam, pn)
                ctr_ex_2 = edges_btw_ctr_nois(tilde_ctr[sam], tilde_lam, pn)

            exclude_nodes_2 = [tilde_lam[exclude_2[0]]] + [tilde_lam[exclude_2[1]]]
            tilde_theta_2_ls = []
            for n in tilde_lam:
                if n not in exclude_nodes_2:
                    tilde_theta_2_ls.append(n)
            tilde_theta_2_ls = tilde_theta_2_ls + tilde_ctr[sam]
            tilde_theta_2[sam] = contract_by_nodes(tilde_theta_2_ls, None, 'til_theta2')
            tt2_name = list(map(lambda edg: edg.name, tilde_theta_2[sam].get_all_edges()))
            tilde_theta_2[sam].add_axis_names(tt2_name)
            tt2_reorder = list(map(lambda name: tilde_theta_2[sam][name], L_axis_names))
            tilde_theta_2[sam].reorder_edges(tt2_reorder)

        tmp_node = tn.Node(np.zeros(([2]*6), dtype=complex))
        for til_theta in tilde_theta_2:
            tmp_node += tilde_theta_2[til_theta]
        avg_tilde_theta_2 = tmp_node /sample_size
        # con_ls = [tilde_theta_2] + exclude_nodes_2
        # test_f[k,pn-1] = contract_by_nodes(con_ls, None, 'f').tensor.real
        grad2 += beta_pn * avg_tilde_theta_2
        beta += abs(beta_pn.tensor)/gamma
        cost += beta_pn.tensor**2
    costs.append(cost/2)
    tmpf = order4_to_2(lamdas[0].tensor)
    f[k] = (abs(np.trace(tmpf))**2 + d) / (d**2 + d)

    if beta <= np.sum(std_exp) / delta:
        # Note that if sample_size=1, std_exp=0, the condition did not work.
        # for lam in lamdas:
        #     print(lam.name, np.matrix.round(lam.tensor, 4))
        print('k', k)
        print('beta', beta)
        print('std_exp', std_exp)
        print('sig', sig.tensor)
        print('allsig', allsig)
        break
    # Axis_name and edge_name are the same for each sample.
    grad2.add_axis_names(tilde_theta_2[0].axis_names)
    for x in range(len(tilde_theta_2[0].edges)):
        grad2[x].set_name(tilde_theta_2[0].axis_names[x])

    # print(beta)
    L -= lr * grad2
    grad2s.append(grad2)
    # print(L.tensor)
    # break
    for x in range(len(L.edges)):
        L[x].set_name(grad2[x].name)
    L.add_axis_names(grad2.axis_names)

    # SVD on L
    # u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
    #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
    u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
                    right_edges=[L[3],L[4],L[5]], max_singular_values=2)
    # u_prime, vh_prime, tranc = tn.split_node(L, left_edges=[L[0],L[1],L[2]], 
                    # right_edges=[L[3],L[4],L[5]], max_singular_values=None)
        
        
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
        u = order4_to_2(u_prime.tensor)
        # vh = u_prime.tensor
        # u = order4_to_2(u)
        # vh = order4_to_2(vh)
        vh = order4_to_2(vh_prime.tensor)
        uq, ur = linalg.qr(u)
        vr, vq = linalg.rq(vh)
        uvh = uq @ np.conj(vq).T
        # f[k] = (abs(np.trace(uq))**2 + d) / (d**2 + d)
        uvh = order2_to_4(uvh)
        uq = order2_to_4(uq)
        vq = order2_to_4(vq)
        u_prime = tn.Node(uvh)
        vh_prime = tn.Node(uvh)
        # u_prime = tn.Node(uq)
        # vh_prime = tn.Node(vq)
    else:
        u_prime = tn.Node(u_prime)
        vh_prime = tn.Node(vh_prime)

    # if k == 1:
    #     # tmpt = tn.contractors.auto([u_prime]+[sig]+[vh_prime], ignore_edge_order=True).tensor
    #     # print(np.matrix.round(tmpt, 6))
    #     # print(np.matrix.round(grad2.tensor, 6))
    #     print(np.matrix.round(L.tensor, 6))
    #     # print(np.matrix.round(u_prime.tensor, 6))
    #     # print(allsig)
    #     break

    # if abs(np.sum(u_prime.tensor)) < 1e-4:
    #     print('Values too small.', k)
    #     break
    lam_up = order4_to_2(u_prime.tensor)
    # lam_up = order4_to_2(u_prime.tensor)
    lamdas = initialized_lamdas_tn(m, lam_up, rho_e)
    # lamdas = initialized_lamdas_tn(m, unitary_group.rvs(4), rho_e)

    """
    ### This part is for update only the pair and corresponding dg.
    u_prime.set_name(lamdas[i].name)
    u_prime.add_axis_names(lamdas[i].axis_names)
    # u_prime[-1].set_name(lamdas[i].axis_names[-1])
    vh_prime.set_name(lamdas[i+1].name)
    vh_prime.add_axis_names(lamdas[i+1].axis_names)
    lamdas[i] = u_prime
    lamdas[i+1] = vh_prime
    # For updating lam^dagger
    ud_ind = complement - i
    vhd_ind = complement - (i+1)
    ud_prime = tn.Node(np.conj(u_prime.tensor).T)
    ud_prime.set_name(lamdas[ud_ind].name)
    ud_prime.add_axis_names(lamdas[ud_ind].axis_names)
    vhd_prime = tn.Node(np.conj(vh_prime.tensor).T)
    vhd_prime.set_name(lamdas[vhd_ind].name)
    vhd_prime.add_axis_names(lamdas[vhd_ind].axis_names)
    lamdas[complement - i] = ud_prime
    lamdas[complement - (i+1)] = vhd_prime
    # for n in range(m):
    #     print('k,m', k, n, F[k,n])
    """
#%%
s = 6
e = moves
# ASF_learning_plot(F_exp, F, m, s, e)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(range(1,m+1), F_exp, s=10, c='b', marker="s", label='F_exp')
# colorls = ['g','r','c','m','y','k']

# pts = moves
# # for p in range(moves-6, moves):
# for p in range(pts-6, pts):
#     colorp = p % 6
#     ax1.scatter(range(1,m+1), F[p], s=10, c=colorls[colorp], marker="o", label='u'+str(p))
# plt.legend(loc='lower left')
# plt.show()

fig2, ax2 = plt.subplots(1,2)
ax2[0].plot(costs)
ax2[1].plot(f)
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

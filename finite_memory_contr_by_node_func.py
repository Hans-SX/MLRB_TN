#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:47:54 2022

@author: sxyang
"""
#%%
import numpy as np
import random
from datetime import datetime
from random import randint
from scipy import linalg
# from scipy.stats import unitary_group
import tensornetwork as tn
# import matplotlib.pyplot as plt

from utils_tn import initialized_lamdas_tn, gen_control_ten, order2_to_4
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2, single_cliffords
from utils_tn import contract_by_nodes, noise_nonM_unitary
from utils_tn import unitary_map, rand_clifford_sequence_unitary_noise_list
from utils_tn import load_plot

def estimate_noise_via_sweep(m, updates, sample_size=100, rand_seed=5, lr=tn.Node(1), delta=8, nM=True, qr=False, adam1=0.9, adam2=0.999, init_noise=None, optimizer="Adam"):

    """
    m = 20  # m=3, F[27]; m=6, F[35] closest to F_exp
    # m = 20, F[26], 26min, (delta ~ 7)
    updates = 50
    lr = tn.Node(1)  # tn.Node(0.0001)
    delta = 8 # >=1, stopping condition

    sample_size = 50
    """
    start_time = datetime.now()

    random.seed(rand_seed)

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
    if type(init_noise) == type(None):
        init_noise = np.identity(sys_dim * bond_dim, dtype=complex)
        lfile = False
    else:
        lfile = True
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
    """
    Still some bug to fix.

    all_control_ten = np.load("Clifford_m60_samp100_seed5.npz", allow_pickle=True)
    control_ten = all_control_ten["control_ten"]
    sam_n = all_control_ten["sam_n"]
    """
    # -------------------- Generate the F_exp here -------------
    #%%
    # nM = True
    if nM == True:
        F_e = np.zeros((m, sample_size))

        # F_exp, non-M unitary noise from Pedro's work.
        for sam in range(sample_size):
            for n in range(1, m+1):
                tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(n, rho, noise_u, sam_clif[sam, :n])
                tmp_rho = np.kron(I, inver_op) @ tmp_rho @ np.conj(np.kron(I, inver_op)).T
                if type(noise_u) == type([]):
                    final_state = unitary_map(tmp_rho, noise_u[n-1])
                else:
                    final_state = unitary_map(tmp_rho, noise_u)
                f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
                F_e[n-1, sam] = np.trace(proj_O @ f_sys_state).real
        std_exp = np.std(F_e, axis=1)
        F_exp = np.mean(F_e, axis=1)
        # print(F_exp)
    else:
        # Marko = np.load("Markovian_m80_amp_damp_p0.06_unitary_samp100.npz")
        Marko = np.load("Markovian_m80_p_flip_p0.06_unitary_samp100.npz")
        F_exp = Marko['F_exp']
        F_exp = F_exp[:m]
        # Somehow std_exp in Markovian case (depolar) are ~1e-16, did not figure out why. Set to 0 for the convenience.
        std_exp = Marko['std_exp']
        std_exp = std_exp[:m]
        """
        directly load data might not be a good idea.
        the sampling control tensor are different.
        but ideally, the control tensor is the average of all samples might be fine.
        """

    # ----------------------------------------------------------
    """
    Make F allow complex, but it should be zero on img.
    """
    F = np.zeros((updates, m), dtype=complex)
    agf = np.zeros(updates)

    grad2s = []
    costs = []
    all_sigs = []
    noise_ten = []
    if optimizer == "AdaGrad":
        tmppregrads = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    elif optimizer == "Adam":
        m_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
        v_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    #     mh_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    #     vh_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)

    #%%
    for k in range(0, updates):
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

        L_edg = lamdas[i][-1] ^ lamdas[i+1][0]
        L_dangled = lamdas[i].get_all_dangling() + lamdas[i+1].get_all_dangling()
        L_axis_names = list(map(lambda edg: edg.name, L_dangled))
        L = contract_by_nodes([lamdas[i], lamdas[i+1]], L_dangled, 'L', ignore=False).tensor
        L = tn.Node(L, name='L', axis_names=L_axis_names)
        
        # math: beta_{n} tilde_Theta_{n}^{i, i-1}, i = m+1, m, ..., 1
        # tilde_Theta_{n}^{i, i-1} is a function takes n as input so, drop +1 here.
        # python: beta_{pn} tilde_Theta_{pn}^{i, i+1}, i = 0, ..., m

        grad2 = tn.Node(np.zeros(L.shape))
        cost = 0
        for l in range(i+1):
            pn = m-l
            if pn == 0:
                break
            exclude_2 = [i-l, i+1-l]
            beta_pn = tn.Node(F_exp[pn-1] - F[k, pn-1])  # F[k, pn-1] is averaged.
            tilde_theta_2 = dict()
            tilde_ctr = dict()
            for sam in range(sample_size):
                for node in lamdas:
                    node.fresh_edges(node.axis_names)
                if pn == m:
                    tilde_ctr[sam] = control_ten[sam]
                    tilde_lam = lamdas
                    # Connecting edges, need not to be called.
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

            grad2 += beta_pn * avg_tilde_theta_2

        cost = sum((F[k] - F_exp)**2)
        costs.append(cost/2)
        tmpagf = order4_to_2(lamdas[0].tensor)
        # Averaged gate fidelity
        agf[k] = (abs(np.trace(tmpagf))**2 + d) / (d**2 + d)
 
        # Axis_name and edge_name are the same for each sample.
        grad2 = tn.Node(grad2.tensor, name='grad', axis_names=tilde_theta_2[0].axis_names)
        # grad2.add_axis_names(tilde_theta_2[0].axis_names)
        # for x in range(len(tilde_theta_2[0].edges)):
        #     grad2[x].set_name(tilde_theta_2[0].axis_names[x])

        if optimizer == "AdaGrad":
            tmppregrads += grad2.tensor ** 2
            tmppre = 1/np.sqrt(tmppregrads + 1e-8)
            pregrads = tn.Node(tmppre)
            L -= pregrads * lr * grad2
            grad2s.append(pregrads * grad2)
        elif optimizer == "Adam":
            m_adam = adam1 * m_adam + (1-adam1) * grad2.tensor
            v_adam = adam2 * v_adam + (1-adam2) * grad2.tensor ** 2
            mh_adam = m_adam / (1 - adam1**(k+1))
            vh_adam = v_adam / (1 - adam2**(k+1))
            mv_adam = mh_adam / (np.sqrt(vh_adam) + 1e-8)
            mv_adam = tn.Node(mv_adam)
            L -= lr * mv_adam
            grad2s.append(mv_adam)
        elif optimizer == "SGD":
            L -= lr * grad2
            grad2s.append(grad2)

        # for x in range(len(L.edges)):
        #     L[x].set_name(grad2[x].name)
        # L.add_axis_names(grad2.axis_names)

        # SVD on L
        # u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
        #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
        u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
                        right_edges=[L[3],L[4],L[5]], max_singular_values=2)
        # u_prime, vh_prime, tranc = tn.split_node(L, left_edges=[L[0],L[1],L[2]], 
        #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
            
        all_sigs.append(allsig)

        if qr == True:
            u = order4_to_2(u_prime.tensor)
            vh = order4_to_2(vh_prime.tensor)
            uq, _ = linalg.qr(u)
            _, vq = linalg.rq(vh)
            # uvh = uq @ np.conj(vq).T
            uvh = uq @ vq
            uvh = order2_to_4(uvh)
            uq = order2_to_4(uq)
            vq = order2_to_4(vq)
            u_prime = tn.Node(uvh)
            vh_prime = tn.Node(uvh)

            lam_up = order4_to_2(u_prime.tensor)
            noise_ten.append(lam_up)
            lamdas = initialized_lamdas_tn(m, lam_up, rho_e)
        elif qr == False:
            # Project u_prime/vh_prime to unitary.
            # Proj(UXVh) = UProj(X)Vh, X = USigVh.
            lu, su, ru = linalg.svd(order4_to_2(u_prime.tensor))
            lvh, svh, rvh = linalg.svd(order4_to_2(vh_prime.tensor))
            uvh = lu @ ru @ lvh @ rvh
            noise_ten.append(uvh)
            lamdas = initialized_lamdas_tn(m, uvh, rho_e)
        else:
            u_prime = tn.Node(u_prime, name=lamdas[i].name, axis_names=lamdas[i].axis_names)
            vh_prime = tn.Node(vh_prime, name=lamdas[i+1].name, axis_names=lamdas[i+1].axis_names)
            uh_prime = np.conj(u_prime.tensor.T)
            v_prime = np.conj(vh_prime.tensor.T)
            uh_prime = tn.Node(uh_prime, name=lamdas[2*(m+2)-i].name, axis_names=lamdas[2*(m+2)-i].axis_names)
            v_prime = tn.Node(v_prime, name=lamdas[2*(m+2)-(i+1)].name, axis_names=lamdas[2*(m+2)-(i+1)].axis_names)

            lamdas[i] = u_prime
            lamdas[i+1] = vh_prime
            lamdas[2*(m+2)-i] = uh_prime
            lamdas[2*(m+2)-(i+1)] = v_prime

        beta = sum(abs(F[k] - F_exp))
        if nM == True:
            if beta <= np.sum(std_exp/np.sqrt(sample_size)) / delta:
                # Note that if sample_size=1, std_exp=0, the condition did not work.
                F = F[:k+2]
                print('k', k)
                print('beta', beta)
                break
        else:
            if beta <= 0.1: 
                # for the Markovian case, obtain 0.26 from a result.
                F = F[:k+2]
                print('k', k)
                print('beta', beta)
                break

        # lamdas = initialized_lamdas_tn(m, unitary_group.rvs(4), rho_e)

        # if (k+1) % 10 == 0:
        #     print(str(k+1) + " updates finished.")


    if nM == True:
        if optimizer == "Adam":
            fname = "m"+ str(m) + "_lr" + str(lr.tensor.real) + "_adama" + str(adam1) + "_adamb" + str(adam2) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed) + "_delta" + str(delta)
        elif optimizer == "AdaGrad":
            fname = "m"+ str(m) + "_lr" + str(lr.tensor.real) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed) + "_delta" + str(delta) + "_Ada"
        else:
            fname = "m"+ str(m) + "_lr" + str(lr.tensor.real) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed) + "_delta" + str(delta)

    else:
        fname = "Markovian_m"+ str(m) + "_lr" + str(lr.tensor.real) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed) + "_delta" + str(delta)
        fname = fname + "_pflip"
        if optimizer == "Adam":
            fname = fname + "_Adam"
        elif optimizer == "AdaGrad":
            fname = fname + "_Ada"
    
    if lfile:
        fname = fname + "_load_cb"

    costs = np.asarray(costs)
    end_time = datetime.now()
    duration = end_time - start_time
    Duration = 'Duration: {}'.format(duration)
    np.savez(fname, F_exp=F_exp, std_exp=std_exp, F=F, all_sigs=all_sigs, costs=costs, grad2s=grad2s, noise_ten=noise_ten, Duration=Duration)

    
    print('Duration: {}'.format(duration))
    return F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname

    #%%
    # s = 6
    # e = updates
    # ASF_learning_plot(F_exp, F, m, s, e)

    # import matplotlib.pyplot as plt
    # fig2, ax2 = plt.subplots(1,2)
    # ax2[0].plot(costs)
    # ax2[1].plot(agf)

#%%
if __name__ == '__main__':
    # start_time = datetime.now()

    # fname = 'm60_updates20_sample100_seed5_gamma1_lr0.0103_delta1.55.npz'
    # fname = 'm60_lr1e-05_adama0.9_adamb0.999_updates80_sample100_seed5_gamma1_delta2.npz'
    # data = np.load(fname)
    # min_ind = np.where(data['costs']==min(data['costs']))[0][0]
    # init_noise = data['noise_ten'][min_ind-1]
    m = 10; updates = 40; nM = True; qr = False; rand_seed = 5; lr = tn.Node(0.001);  delta = 2; sample_size = 100
    F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM)

    data, F_exp, norm_std, F, costs = load_plot(fname, m)

    # m = 60; updates = 80; nM = True; qr = True; rand_seed = 5; lr = tn.Node(0.01); delta = 2; sample_size = 100; adam1=0.5; adam2=0.5
    # F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM, qr, adam1, adam2)


    # m = 6; updates = 70; nM = True; qr = True; rand_seed = 5; lr = tn.Node(1); delta = 10; sample_size = 100

    # F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM, qr)

    # end_time = datetime.now()
    # print('Duration: {}'.format(end_time - start_time))


# %%

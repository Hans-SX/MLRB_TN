#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:21:54 2022

@author: sxyang
"""
#%%
import sys
import numpy as np
import random
from datetime import datetime
from random import randint
from scipy import linalg
from scipy.stats import unitary_group
# from scipy.stats import unitary_group
import tensornetwork as tn
# import matplotlib.pyplot as plt

from utils_tn import initialized_lamdas_tn, gen_control_ten, order2_to_4
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2, single_cliffords
from utils_tn import contract_by_nodes, noise_nonM_unitary
from utils_tn import unitary_map, rand_clifford_sequence_unitary_noise_list
from utils_tn import load_plot_non_Markovianity, rand_clifford_sequence_markovianized_asf

from RB_numerical_toy_model_v1 import clifford_sequence_with_noise, gen_randH

from reproduce_210705403 import non_Markovian_theory_Fm

def estimate_noise_via_sweep_envq(m, updates, sample_size=100, rand_seed=5, lr=tn.Node(1), delta=8, nM=True, update_all=True, adam1=0.9, adam2=0.999, init_noise=None, optimizer="Adam", noise_model="nM", sys_dim=2, bond_dim=2, coeff=1, test=False, wb=1):

    start_time = datetime.now()

    random.seed(rand_seed)

    dim = sys_dim * bond_dim
    I = np.eye(2, dtype=complex)
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0,0] = 1
    rho_s = np.trace(rho.reshape(bond_dim, sys_dim, bond_dim, sys_dim), axis1=0, axis2=2)
    rho_e = np.trace(rho.reshape(bond_dim, sys_dim, bond_dim, sys_dim), axis1=1, axis2=3)
    proj_O = np.zeros((sys_dim, sys_dim), dtype=complex)
    proj_O[0,0] = 1

    
    if type(init_noise) == type(None):
        init_noise = np.identity(dim, dtype=complex)
        lfile = False
    elif type(init_noise) == str:
        lfile = False
    else:
        lfile = True
    # init_noise = noise_nonM_unitary(m-1, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
    # init_noise.insert(0, np.identity(4, dtype=complex))

    if noise_model == "nM":
        # noise_u = noise_nonM_unitary(m, J=1.2, hx=1.17, hy=-1.15, delta=0.05)
        noise_u = noise_nonM_unitary(m, J=1.2, hx=1.17, hy=-1.15, delta=0.1)
    elif noise_model == "randH" and nM == True:
        # Should fix one, somehow it did not use the same. (This is fixed, I think.)
        # H = np.random.random((4,4)) + 1j*np.random.random((4,4))
        # H = H @ np.conj(H.T)
        # noise_u = linalg.expm(-1j*0.05*H)
        noise_u = np.load("data/randH_seed5.npy")
        noise_model = "randH_seed5"
    
    if init_noise == "randu":
        fn_init = init_noise
        init_noise = np.load("load_data/randu_init.npz")
        if bond_dim == 4:
            init_noise = init_noise["dim8"]
        elif bond_dim == 2:
            init_noise = init_noise["dim4"]
        
    elif init_noise == "model":
        if bond_dim == 4:
            fn_init = init_noise
            init_noise = np.kron(np.identity(2), noise_u[0])
        elif bond_dim == 2:
            fn_init = init_noise
            init_noise = noise_u[0]
    else:
        fn_init = None
    # noise_u = np.identity(sys_dim * bond_dim, dtype=complex)
    # noise_u = unitary_group.rvs(4)
    # init_noise = noise_u.copy()
    # init_noise.reverse()
    # init_noise.insert(0, np.identity(sys_dim * bond_dim, dtype=complex))

    lamdas = initialized_lamdas_tn(m, init_noise, rho_e, sys_dim, bond_dim)

    
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

    # -------------------- Generate the F_exp here -------------
    #
    # nM = True
    if nM == True:
        data = np.load("load_data/")
        F_exp = data["F_exp"]
        var_exp = data["var_exp"]
        """
        F_e = np.zeros((m, sample_size))
        I_gate = np.identity(bond_dim, dtype=complex)

        # F_exp, non-M unitary noise from Pedro's work.
        for sam in range(sample_size):
            for n in range(1, m+1):
                tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(n, rho, noise_u, sam_clif[sam, :n], sys_dim, bond_dim)
                tmp_rho = np.kron(I_gate, inver_op) @ tmp_rho @ np.conj(np.kron(I_gate, inver_op)).T
                if type(noise_u) == type([]):
                    final_state = unitary_map(tmp_rho, noise_u[n-1])
                else:
                    final_state = unitary_map(tmp_rho, noise_u)
                f_sys_state = np.trace(final_state.reshape(bond_dim, sys_dim, bond_dim, sys_dim), axis1=0, axis2=2)
                F_e[n-1, sam] = np.trace(proj_O @ f_sys_state).real
        F_exp = F_e.T
        var_exp = np.var(F_e, axis=1).reshape(m, 1)
        std_exp = np.std(F_e, axis=1)
        # F_exp = np.mean(F_e, axis=1)
        """
        """
        # theory nM ASF
        nonM_theory_Fm = non_Markovian_theory_Fm(noise_u, proj_O, rho, I, sys_dim)
        # print(F_exp)
        F_exp = np.zeros(m)
        std_exp = np.zeros(m)
        for n in range(m):
            F_exp[n] = nonM_theory_Fm.theory_Fm(n)
        """
    else:
        if noise_model == "AD":
            Marko = np.load("data/Markovian_m80_amp_damp_p0.06_unitary_samp100.npz")
            F_exp = Marko['F_exp']
            F_exp = F_exp[:m]
            std_exp = Marko['std_exp']
            std_exp = std_exp[:m]
        elif noise_model == "PF":
            Marko = np.load("data/Markovian_m80_p_flip_p0.06_unitary_samp100.npz")
            F_exp = Marko['F_exp']
            F_exp = F_exp[:m]
            std_exp = Marko['std_exp']
            std_exp = std_exp[:m]
        elif noise_model == "DP":
            Marko = np.load("data/Markovian_m80_depolar_p0.06_unitary_samp100.npz")
            F_exp = Marko['F_exp']
            F_exp = F_exp[:m]
            # Somehow std_exp in Markovian case (depolar) are ~1e-16, did not figure out why. Set to 0 for the convenience.
            std_exp = Marko['std_exp']
            std_exp = std_exp[:m]
        elif noise_model == "randH":
            randH = gen_randH(0)
            noise_para = 0.1
            fm = np.zeros((sample_size, m), dtype=complex)
            for sam in range(sample_size):
                for n in range(m):
                    final_state, _ = clifford_sequence_with_noise(rho_s, sam_clif[sam, :n+1], noise_model, noise_para, rand_seed, randH)
                    fm[sam, n] = np.trace(proj_O @ final_state)
                    F_exp = np.mean(fm, axis=0)
                    std_exp = np.std(fm, axis=0)
        else:
            print("Noise model " + noise_model + " not support.")
            sys.exit(0)

        """
        directly load data might not be a good idea.
        the sampling control tensor are different.
        but ideally, the control tensor is the average of all samples might be fine.
        """

    markovianized = rand_clifford_sequence_markovianized_asf(m, rho_s, proj_O, sys_dim, bond_dim)
    if nM == True:
        markF_exp = markovianized.markovianized_asf(sam_clif, noise_u)
    else:
        markF_exp = F_exp.copy()
    # ----------------------------------------------------------
    """
    Make F allow complex, but it should be zero on img.
    """
    F = np.zeros((updates, sample_size,  m), dtype=complex)
    agf = np.zeros(updates)
    markF = np.zeros((updates, sample_size, m), dtype=complex)

    # coeff_on_cost = np.ones(m)
    # coeff_on_cost[30:] = 1/coeff
    grads = []
    costs = []
    all_sigs = []
    noise_ten = []
    nonMarkovianity = []
    if optimizer == "AdaGrad":
        tmppregrads = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    elif optimizer == "Adam":
        m_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
        v_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    #     mh_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)
    #     vh_adam = np.zeros((bond_dim, sys_dim, sys_dim, sys_dim, sys_dim, bond_dim),dtype=complex)

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
        tmp_F = np.zeros((sample_size, 1), dtype=complex)
        tmp_Fn = np.zeros((sample_size, m-1), dtype=complex)
        tmp_noise = list(map(lambda x: order4_to_2(lamdas[x].tensor, sys_dim, bond_dim), np.arange(m+1)))
        markF[k] = markovianized.markovianized_asf(sam_clif, tmp_noise)
        for sam in range(sample_size):
            # edges for constructing Fm, m = m.
            e_edgs = edges_in_lamdas(lamdas, m)
            s_edgs = edges_btw_ctr_nois(control_ten[sam], lamdas, m)
            
            tmp_F[sam] = contract_by_nodes(lamdas+control_ten[sam]).tensor

            for n in range(1,m):
                noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor, sys_dim, bond_dim), np.arange(m-n,m+2)))
                noise_tmp.reverse()
                lam_n = initialized_lamdas_tn(n, noise_tmp, rho_e, sys_dim, bond_dim)
                lam_n_edg = edges_in_lamdas(lam_n, n)
                ctr_n_edg = edges_btw_ctr_nois(sam_n[sam][n-1], lam_n, n)
                tmp_n = tn.contractors.auto(lam_n+sam_n[sam][n-1], None, ignore_edge_order=True)
                tmp_Fn[sam, n-1] = np.abs(tmp_n.tensor)
        F[k, :, :] = np.concatenate((tmp_Fn, tmp_F), axis=1)

        L_edg = lamdas[i][-1] ^ lamdas[i+1][0]
        L_dangled = lamdas[i].get_all_dangling() + lamdas[i+1].get_all_dangling()
        L_axis_names = list(map(lambda edg: edg.name, L_dangled))
        L = contract_by_nodes([lamdas[i], lamdas[i+1]], L_dangled, 'L', ignore=False).tensor
        L = tn.Node(L, name='L', axis_names=L_axis_names)
        
        # math: beta_{n} tilde_Theta_{n}^{i, i-1}, i = m+1, m, ..., 1
        # tilde_Theta_{n}^{i, i-1} is a function takes n as input so, drop +1 here.
        # python: beta_{pn} tilde_Theta_{pn}^{i, i+1}, i = 0, ..., m
        
        tilde_theta = []
        for l in range(i+1):
            pn = m-l
            if pn == 0:
                break
            exclude_2 = [i-l, i-l+1]
            # beta_pn = (F_exp[:, pn-1] - F[k, :, pn-1]) * coeff_on_cost[pn-1]  # F[k, :, pn-1] is not averaged.
            tilde_theta_2 = []
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
                    noise_u = list(map(lambda x: order4_to_2(lamdas[x].tensor, sys_dim, bond_dim), np.arange(l, m+2)))

                    tilde_lam = initialized_lamdas_tn(pn, noise_u, rho_e, sys_dim, bond_dim)
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
                tilde_theta_2.append(contract_by_nodes(tilde_theta_2_ls, None, 'til_theta2'))
                tt2_name = list(map(lambda edg: edg.name, tilde_theta_2[sam].get_all_edges()))
                tilde_theta_2[sam].add_axis_names(tt2_name)
                tt2_reorder = list(map(lambda name: tilde_theta_2[sam][name], L_axis_names))
                tilde_theta_2[sam].reorder_edges(tt2_reorder)
            
            # Use insert to put pn = 0, 1, ... in an ascending order (same as F).
            tilde_theta.insert(0, tilde_theta_2.copy())

        cost = 0
        var_model = np.var(F[k, :, :], axis=0).reshape(m, 1)
        gk = np.zeros(L.shape, dtype=complex)
        gl = np.zeros(L.shape, dtype=complex)
        gM = np.zeros(L.shape, dtype=complex)

        # beta_pn = (F_exp - F[k, :, :]) * coeff_on_cost
        beta_pn = (F_exp - F[k, :, :])
        beta_pn = np.mean(beta_pn, axis=0)
        sum_samp_tilde_theta = np.zeros(L.shape, dtype=complex)
        sum_samp_f_minus_F_tilde_theta = np.zeros(L.shape, dtype=complex)

        for n in range(i+1):
            for sam in range(sample_size):
                sum_samp_tilde_theta += tilde_theta[n][sam].tensor
                # tmp_tilde += tilde_theta[n][sam].tensor
                sum_samp_f_minus_F_tilde_theta += (F[k, sam, n] - np.mean(F[k, :, n])) * tilde_theta[n][sam].tensor
            weight = 1/var_model[n] - (var_exp[n] + beta_pn[n]**2) / var_model[n]**2
            gk += weight * sum_samp_f_minus_F_tilde_theta
            gl += beta_pn[n] * sum_samp_tilde_theta / var_model[n]**2 
            # gM += (np.mean(markF[k, :, n] - F[k, :, n])) * sum_samp_tilde_theta
        
        weight_frob = 1 / (sys_dim * bond_dim) / sample_size / wb
        g_frobenius = 2 * L.tensor * weight_frob
        grad = (gk + gl) / sample_size / m + gM / sample_size + g_frobenius

        # Not only cost, the gradient also need to apply coeff_on_cost.
        # cM = np.sum(np.mean(markF[k, :, :] - F[k, :, :], axis=0)**2 * coeff_on_cost)/2
        cM = np.sum(np.mean(markF[k, :, :] - F[k, :, :], axis=0)**2)/2
        nonMarkovianity.append(cM.real)
        kl = np.sum(np.log(var_model / var_exp) + (var_exp + beta_pn**2) / (2 * var_model) - 1/2)
        L_order2 = np.transpose(L.tensor, (0,1,3,5,2,4)).reshape(sys_dim * bond_dim * 2, sys_dim * bond_dim * 2)
        frobenius = np.trace(L_order2 @ np.conj(L_order2.T))
        # Found that KL tends to give larger values, 1e1 with random initial noise.
        # And non-Markovianity tends to give values around 1e-3. Give some weights to balance them.
        # print("kl", kl)
        # print("cM", cM)
        # cost = kl / m / sample_size + cM (* m)
        cost = kl / m / sample_size + frobenius * weight_frob
        # cost = sum((F[k] - F_exp)**2)
        costs.append(cost.real)
        tmpagf = order4_to_2(lamdas[0].tensor, sys_dim, bond_dim)
        # Averaged gate fidelity
        agf[k] = (abs(np.trace(tmpagf))**2 + dim) / (dim**2 + dim)

        # Axis_name and edge_name are the same for each sample.
        grad = tn.Node(grad, name='grad', axis_names=tilde_theta_2[0].axis_names)
        # grad2.add_axis_names(tilde_theta_2[0].axis_names)
        # for x in range(len(tilde_theta_2[0].edges)):
        #     grad2[x].set_name(tilde_theta_2[0].axis_names[x])

        if optimizer == "AdaGrad":
            tmppregrads += grad.tensor ** 2
            tmppre = 1/np.sqrt(tmppregrads + 1e-8)
            pregrads = tn.Node(tmppre)
            L -= pregrads * lr * grad
            grads.append(pregrads * grad)
        elif optimizer == "Adam":
            m_adam = adam1 * m_adam + (1-adam1) * grad.tensor
            v_adam = adam2 * v_adam + (1-adam2) * grad.tensor ** 2
            mh_adam = m_adam / (1 - adam1**(k+1))
            vh_adam = v_adam / (1 - adam2**(k+1))
            mv_adam = mh_adam / (np.sqrt(vh_adam) + 1e-8)
            mv_adam = tn.Node(mv_adam)
            L -= lr * mv_adam
            grads.append(mv_adam)
        elif optimizer == "BGD":
            L -= lr * grad
            grads.append(grad)

        # for x in range(len(L.edges)):
        #     L[x].set_name(grad2[x].name)
        # L.add_axis_names(grad2.axis_names)

        # SVD on L
        # u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
        #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
        u_prime, vh_prime, sig, allsig = tn.split_node_u_s_vh(L, left_edges=[L[0],L[1],L[2]], 
                        right_edges=[L[3],L[4],L[5]], max_singular_values=bond_dim)
        # u_prime, vh_prime, tranc = tn.split_node(L, left_edges=[L[0],L[1],L[2]], 
        #                 right_edges=[L[3],L[4],L[5]], max_singular_values=None)
            
        all_sigs.append(allsig)

        if update_all == True:
            # Project u_prime/vh_prime to unitary.
            # Proj(UXVh) = UProj(X)Vh, X = USigVh.
            lu, su, ru = linalg.svd(order4_to_2(u_prime.tensor, sys_dim, bond_dim))
            lvh, svh, rvh = linalg.svd(order4_to_2(vh_prime.tensor, sys_dim, bond_dim))
            uvh = lu @ ru @ lvh @ rvh
            noise_ten.append(uvh)
            lamdas = initialized_lamdas_tn(m, uvh, rho_e, sys_dim, bond_dim)
        else:
            lu, su, ru = linalg.svd(order4_to_2(u_prime.tensor, sys_dim, bond_dim))
            lvh, svh, rvh = linalg.svd(order4_to_2(vh_prime.tensor, sys_dim, bond_dim))
            noise_ten.append(lu @ ru)
            u_new = order2_to_4(lu @ ru, sys_dim, bond_dim)
            vh_new = order2_to_4(lvh @ rvh, sys_dim, bond_dim)
            u_prime = tn.Node(u_new, name=lamdas[i].name, axis_names=lamdas[i].axis_names)
            vh_prime = tn.Node(vh_new, name=lamdas[i+1].name, axis_names=lamdas[i+1].axis_names)
            uh_prime = np.conj(u_prime.tensor.T)
            v_prime = np.conj(vh_prime.tensor.T)
            uh_prime = tn.Node(uh_prime, name=lamdas[2*(m+2)-1-i].name, axis_names=lamdas[2*(m+2)-1-i].axis_names)
            v_prime = tn.Node(v_prime, name=lamdas[2*(m+2)-1-(i+1)].name, axis_names=lamdas[2*(m+2)-1-(i+1)].axis_names)

            lamdas[i] = u_prime
            lamdas[i+1] = vh_prime
            lamdas[2*(m+2)-1-i] = uh_prime
            lamdas[2*(m+2)-1-(i+1)] = v_prime

        beta = np.sum(abs(F[k] - F_exp))
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
        fname = "m"+ str(m) + "_dimE" + str(bond_dim) + "_lr" + str(lr.tensor.real) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed)
    else:
        fname = "Markovian_m"+ str(m) + "_dimE" + str(bond_dim) + "_lr" + str(lr.tensor.real) + "_updates" + str(updates) + "_sample" + str(sample_size) + "_seed" + str(rand_seed)
    
    if optimizer == "Adam":
        fname = fname + "_adama" + str(adam1) + "_adamb" + str(adam2) + "_Adam"
    elif optimizer == "AdaGrad":
        fname = fname + "_Ada"

    if update_all == True:
        fname += "_replace_all"
    else:
        fname += "_replace_1"

    if coeff != 1:
        fname = fname + "_wcost_" + coeff
    fname = fname + "_" + noise_model

    if lfile:
        # fname = fname + "_load_cb"
        # for m=20 replace1, lazy to make it general.
        fname = fname + "_load_1"
     
    if fn_init != None:
        fname = fname + "_init_" + fn_init + ""
    
    fname = fname + "_cost_KL_frob_wb_" + str(wb) + "_nM"
    costs = np.asarray(costs)
    end_time = datetime.now()
    duration = end_time - start_time
    Duration = 'Duration: {}'.format(duration)
    
    if test == False:
        if type(noise_u) == type([]):
            np.savez("data/" + fname, F_exp=F_exp, std_exp=std_exp, F=F, all_sigs=all_sigs, costs=costs, grads=grads, noise_ten=noise_ten, Duration=Duration, nonM=nonMarkovianity, markF=markF, markF_exp=markF_exp, init_noise=init_noise)
        else:
            np.savez("data/" + fname, F_exp=F_exp, std_exp=std_exp, F=F, all_sigs=all_sigs, costs=costs, grads=grads, noise_ten=noise_ten, Duration=Duration, noise_u=noise_u, nonM=nonMarkovianity, markF=markF, markF_exp=markF_exp, init_noise=init_noise)

    
    print('Duration: {}'.format(duration))
    return F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname, markF, markF_exp, nonMarkovianity

#%%
if __name__ == '__main__':

    # from flexible_env_qubit_model import estimate_noise_via_sweep_envq
    from scipy.stats import unitary_group
    import tensornetwork as tn
    from utils_tn import plot_inset
    
    m = 10
    lr = tn.Node(complex(0.01))
    adam1 = 0.9
    adam2 = 0.99
    optimizer = "Adam"
    nM = True
    rand_seed = 5
    updates = 20
    sample_size = 50
    update_all = True
    noise_model = "nM"
    delta = 2
    sys_dim = 2
    bond_dim = 2
    # init_noise = None
    init_noise = "randu"
    test = True
    coeff = 1

    F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname, markF, markF_exp, nonMarkovianity = estimate_noise_via_sweep_envq(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model, sys_dim, bond_dim, coeff, test)

    F_exp = np.mean(F_exp, axis=0)
    F = np.mean(F, axis=1)
    markF_exp = np.mean(markF_exp, axis=0)
    markF = np.mean(markF, axis=1)

    min_cost_ind = np.where(costs == min(costs))
    min_cost_ind = min_cost_ind[0][0]
    eval = 100 * (np.sum((np.mean(markF_exp, axis=0) - F_exp)**2 / 2) - nonMarkovianity[min_cost_ind]) / np.sum((markF_exp - F_exp)**2 / 2)
    print("min cost & min non-Markovianity & ind", costs[min_cost_ind], nonMarkovianity[min_cost_ind], min_cost_ind)
    print("(CM(exp) - CM(model)/CM(exp)) x 100% = ", eval)
    print("sum(|F[min]-F_exp|) = ", sum(abs(F[min_cost_ind] - F_exp)))
    norm_std = std_exp / np.sqrt(sample_size)
    print("Num of outside error bar", sum(abs(F[min_cost_ind] - F_exp) > norm_std))
    plot_inset(F_exp.real, norm_std.real, F.real, m, noise_model, min_cost_ind, costs, fname, "png")

    # data, F_exp, norm_std, F, costs = load_plot_non_Markovianity(fname, m, noise_model, False, sample_size)

# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:44:50 2022

Control tensor follow Eq.B20 ~ B22 in RB_with_ML_0113_22 .

@author: sxyang
"""

import numpy as np
from scipy import linalg
import tensornetwork as tn
import matplotlib.pyplot as plt

"""
The order of counting the ket is |000>, |001>, ...
since reshape is strarting from row elements.
"""

def single_cliffords():
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
    return clifford_list

def compare_statistcs(l1, l2):
    lam1 = l1.tensor
    lam2 = l2.tensor
    print('max', np.max(lam1), np.max(lam2))
    print('min', np.min(lam1), np.min(lam2))
    print('mean', np.mean(lam1), np.mean(lam2))

def order2_to_4(lamda, sys_dim=2, bond_dim=2):
    lamda = lamda.reshape(bond_dim, sys_dim, bond_dim, sys_dim)
    lamda= np.swapaxes(lamda, -2, -1)
    return lamda

def order4_to_2(lamda, sys_dim=2, bond_dim=2):
    lamda= np.swapaxes(lamda, -2, -1)
    lamda = lamda.reshape(sys_dim*bond_dim, sys_dim*bond_dim)
    return lamda

def initialized_lamdas_tn(m, noise_u, rho_e, sys_dim=2, bond_dim=2):
    # lam_edgs = []
    lamdas = []
    lamdas.append(tn.Node(rho_e, name='rho_e', axis_names=['e0', 'r0']))
    if type(noise_u) == type([]):
        # the length of noise list = m+2, and the first one could be Markovian or identity. It is the coupling from preparation rather than process. ---> It doesnt' affect anything since we only use it from identity to update noise tensor. But it did cause problem when I want to check whether ( I tensor U_nM) and (U_nM) will lead to the same ASF behavior.
        # ---> It did affect while updating lamdas, I input the new unitay to this function. Thus, lamda0 is also updated. 
        for i in range(m+2):
            # e(m+2) = r(m+2), the out most edge.
            tmp = order2_to_4(noise_u[i], sys_dim, bond_dim)
            lam = tn.Node(tmp, name='lam'+str(i), axis_names=['e'+str(i+1), 's'+str(i), 's\''+str(i),'e'+str(i)])
            lam_dg = tn.Node(np.conj(tmp.T), name='lam_dg'+str(i), axis_names=['r'+str(i), 'z'+str(i), 'z\''+str(i), 'r'+str(i+1)])
            
            lamdas.insert(0, lam)
            lamdas.append(lam_dg)
    else:
        noise_0 = np.identity(sys_dim * bond_dim, dtype=complex)
        noise_0 = order2_to_4(noise_0, sys_dim, bond_dim)
        lam = tn.Node(noise_0, name='lam'+str(0), axis_names=['e'+str(0+1), 's'+str(0), 's\''+str(0),'e'+str(0)])
        lam_dg = tn.Node(np.conj(noise_0.T), name='lam_dg'+str(0), axis_names=['r'+str(0), 'z'+str(0), 'z\''+str(0), 'r'+str(0+1)])
        lamdas.insert(0, lam)
        lamdas.append(lam_dg)
        for i in range(1, m+2):
            # e(m+2) = r(m+2), the out most edge.
            tmp = order2_to_4(noise_u, sys_dim, bond_dim)
        
            lam = tn.Node(tmp, name='lam'+str(i), axis_names=['e'+str(i+1), 's'+str(i), 's\''+str(i),'e'+str(i)])
            lam_dg = tn.Node(np.conj(tmp.T), name='lam_dg'+str(i), axis_names=['r'+str(i), 'z'+str(i), 'z\''+str(i), 'r'+str(i+1)])
            
            lamdas.insert(0, lam)
            lamdas.append(lam_dg)
            
    ind_rhoe = int((2*(m+2)+1)/2)
    # Shuffle the rho_e to the end of the list.
    lamdas.append(lamdas.pop(ind_rhoe))
    return lamdas

def gen_control_ten(rho_s, m, proj_O, rand_clifford):
    control_ten = [tn.Node(rho_s, name='rho_s', axis_names=['s\'0', 'z0'])]
    inv_m = np.identity(2, dtype=complex)
    for i in range(m):
        tmp = rand_clifford[i]
        g = tn.Node(tmp, name='G'+str(i+1), axis_names = ['s\''+str(i+1), 's'+str(i)])
        g_dg = tn.Node(np.conj(tmp.T), name='G_dg'+str(i+1), axis_names = ['z\''+str(i), 'z'+str(i+1)])
        control_ten.insert(0, g)
        control_ten.append(g_dg)
        
        # the order of the inverse matters, previously, I swaped them thus get the wrong values.
        inv_m = inv_m @ np.conj(tmp.T)
    final_g_dg = tn.Node(np.conj(inv_m.T), name='Gfin_dg', axis_names = ['z\''+str(m), 'z'+str(m+1)])
    final_g = tn.Node(inv_m, name='Gfin', axis_names = ['s\''+str(m+1), 's'+str(m)])
    M = tn.Node(proj_O, name='M', axis_names = ['z\''+str(m+1), 's'+str(m+1)])
    control_ten.insert(0, final_g)
    control_ten.append(final_g_dg)
    control_ten.append(M)
    return control_ten

def edges_in_lamdas(lams, m):
    lam_edgs = []
    ind_rhoe = int((2*(m+2)+1)/2)
    for i in range(2*(m+2)-1):
        # The following if elif are dealing with the fact we move rho_e to the end of the list for better align them with control tensors. But we keep the edge order in noise tensors.
        if i == ind_rhoe - 1:
            lam_edgs.append(lams[i][-1] ^ lams[-1][0])
            lam_edgs.append(lams[i+1][0] ^ lams[-1][1])
        # elif i == ind_rhoe:
        #     lam_edgs.append(lams[i][0] ^lams[-1][-1])
        else:
            lam_edgs.append(lams[i][-1] ^ lams[i+1][0])
    lam_edgs.insert(0, lams[0][0] ^ lams[-2][-1])
    return lam_edgs

def edges_btw_ctr_nois(control_ten, lamdas, m):
    # define the edges between control tensor and noise tensor, store them in a list of tn.edge.
    ind_rhos = int(len(control_ten)/2-1)
    edge_list = []
    for i in range(ind_rhos):
        edge_list.append(control_ten[i]['s\''+str(m+1-i)] ^ lamdas[i]['s\''+str(m+1-i)])
        edge_list.append(control_ten[i]['s'+str(m-i)] ^ lamdas[i+1]['s'+str(m-i)])
    
    edge_list.append(control_ten[ind_rhos]['s\'0'] ^ lamdas[ind_rhos]['s\'0'])
    edge_list.append(control_ten[ind_rhos]['z0'] ^ lamdas[ind_rhos+1]['z0'])
    
    for i in range(1, ind_rhos+1):
        edge_list.append(control_ten[i+ind_rhos]['z\''+str(i-1)] ^ lamdas[i+ind_rhos]['z\''+str(i-1)])
        edge_list.append(control_ten[i+ind_rhos]['z'+str(i)] ^ lamdas[i+1+ind_rhos]['z'+str(i)])
    
    # Not sure which edge of M should connect to lam_m+1 or lam_dg_m+1.
    edge_list.append(control_ten[-1]['z\''+str(m+1)] ^ lamdas[-2]['z\''+str(m+1)])
    edge_list.insert(0, control_ten[-1]['s'+str(m+1)] ^ lamdas[0]['s'+str(m+1)])

    return edge_list
    
def contract_edge_list(edg_list, name=None):
    # for i in edg_list:
    #     tensor = tn.contract(i, name=name)
    for i in range(len(edg_list)-1, -1, -1):
        tensor = tn.contract(edg_list[i], name=name)
        edg_list.pop(i)
    return tensor

def contract_by_nodes(cont_ls, out_order=None, name=None, ignore=True):
    tensor = tn.contractors.auto(cont_ls, output_edge_order=out_order, ignore_edge_order=ignore)
    if name != None:
        tensor.set_name(name)
    return tensor

def pop_no_contract_edg(exclude, ctr_edg, lam_edg):
    if type(exclude) != type([]):
        print('exclude should be a list.')
    else:
        pop_s_edgs, pop_e_edgs = not_contract_edgs(exclude)
        pop_out(lam_edg, pop_e_edgs)
        pop_out(ctr_edg, pop_s_edgs)

def not_contract_edgs(exclude_lams):
    # exclude_lams: a list of max length 2, indicates which nodes we don't contract.
    # Pop out in reverse order, so that they will not affect the order of the remainings.
    if len(exclude_lams) == 2:
        pop_e_edgs = [exclude_lams[1]+1, exclude_lams[1], exclude_lams[0]]
        pop_s_edgs = [2*exclude_lams[1]+1, 2*exclude_lams[1], 2*exclude_lams[0]+1, 2*exclude_lams[0]]
    elif len(exclude_lams) == 1:
        pop_e_edgs = [exclude_lams[0]+1, exclude_lams[0]]
        pop_s_edgs = [2*exclude_lams[0]+1, 2*exclude_lams[0]]
    else:
        print('Took at most exclude 2 nodes in this function.')
    return pop_s_edgs, pop_e_edgs

def pop_out(edg_list, pop_list):
    for i in pop_list:
        edg_list.pop(i).disconnect()

def rX(theta):
    X = np.array([[0, 1], [1, 0]])
    matrix = linalg.expm(-1j * theta * np.pi/2 * X)
    return matrix

def rY(theta):
    Y = np.array([[0, -1j], [1j, 0]])
    matrix = linalg.expm(-1j * theta * np.pi/2 * Y)
    return matrix
    
def noise_nonM_unitary(M, J=1.7, hx=1.47, hy=-1.05, delta=0.03):
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    # Z = np.array([[1, 0],[0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # ds = 2   # dim(rho_s)
    # de = 2      # General version de >= 2 is in the appendix of 2107.05403.
    # J = 1.7
    # hx = 1.47
    # hy = -1.05
    # delta = 0.03

    H = J * np.kron(X,X) + hx * (np.kron(X, I) + np.kron(I, X)) + hy * (np.kron(Y, I) + np.kron(I, Y))

    # Non-Markovian noise, assume unitary.
    # noise_u = linalg.expm(-1j*delta*H)
    noise_u = []
    # Generate the initial noise of lamda_ms + lamda_m+1, lamda_0
    for i in range(M+1):
        noise_u.append(linalg.expm(-1j*delta*H))
        # noise_u.append(np.identity(4, dtype=complex))
    return noise_u

def unitary_map(rho, noise_u):
    dim = rho.shape[0]
    noise_dim = noise_u.shape[0]
    I_noise = np.identity(int(dim/noise_dim), dtype=complex)
    return np.kron(I_noise, noise_u) @ rho @ np.conj(np.kron(I_noise, noise_u)).T

def rand_clifford_sequence_unitary_noise_list(m, rho, noise_u, rand_clifford, sys_dim=2, bond_dim=2):
    # apply unitary noise in the sequence
    # Each step has different noise as indicate by the list.
    dim = sys_dim * bond_dim
    tmp_rho = rho
    inver_op = np.eye(sys_dim)
    # lam_0 is not consider in the previous situation, adding it awkwardly here when all lams are the same.
    if type(noise_u) == type([]):
        for i in range(m):
            noise_dim = noise_u[i].shape[0]
            I_gate = np.identity(bond_dim, dtype=complex)
            I_noise = np.identity(int(dim/noise_dim), dtype=complex)
            gate = rand_clifford[i]
            tmp_rho = np.kron(I_noise, noise_u[i]) @ np.kron(I_gate, gate) @ tmp_rho @ np.conj(np.kron(I_gate, gate)).T @ np.conj(np.kron(I_noise, noise_u[i])).T
            
            inver_op = inver_op @ np.conj(gate).T
    else:
        noise_dim = noise_u.shape[0]
        I_gate = np.identity(bond_dim, dtype=complex)
        I_noise = np.identity(int(dim/noise_dim), dtype=complex)
        for i in range(m):
            gate = rand_clifford[i]
            tmp_rho = np.kron(I_noise, noise_u) @ np.kron(I_gate, gate) @ tmp_rho @ np.conj(np.kron(I_gate, gate)).T @ np.conj(np.kron(I_noise, noise_u)).T
            
            inver_op = inver_op @ np.conj(gate).T
    return tmp_rho, inver_op

def ASF_learning_plot(F_exp, norm_std, F, m, b, e):
    fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(range(1,m+1), F_exp, s=10, c='b', marker="s", label='F_exp')
    plt.errorbar(range(1,m+1), F_exp, yerr=norm_std, c='b', label='F_exp')
    colorls = ['g','r','c','m','y','k']

    # for p in range(moves-6, moves):
    for p in range(e-b, e):
        colorp = p % 6
        plt.scatter(range(1,m+1), F[p].real, s=10, c=colorls[colorp], marker="o", label='u'+str(p))
    plt.xlabel("Sequence length")
    plt.ylabel("Sequence fidelity")
    plt.legend(loc='lower left')
    plt.show()

def plot_for_read(F_exp, norm_std, F, m, noise, up_ind, fname):
    fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(range(1,m+1), F_exp, s=10, c='b', marker="s", label='F_exp')
    plt.errorbar(range(1,m+1), F_exp, yerr=norm_std, c='b', label='$\\mathcal{F}^{%s}$' % (noise))
    plt.scatter(range(1,m+1), F[up_ind].real, s=10, marker="o", c='r', label='$\\mathcal{F}_{%d}^{%s}$' % (up_ind, noise))
    plt.xlabel("Sequence length")
    plt.ylabel("Sequence fidelity")
    plt.legend(loc='lower left')
    plt.grid(True)
    # plt.savefig(fname + "_ASF.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def load_plot(fname, m, noise_model, save, samples=100):
    data = np.load('data/' + fname +'.npz')
    F_exp = np.mean(data['F_exp'], axis=0)
    # F_exp = data['F_exp']
    std_exp = data['std_exp']
    F = np.mean(data['F'], axis=1)
    # F = data['F']
    costs = data['costs']
    # noise_ten = data['noise_ten']
    # all_sigs = data['all_sigs']
    # Duration = data['Duration']
    
    min_cost_ind = np.where(costs == min(costs))
    min_cost_ind = min_cost_ind[0][0]
    print("min cost & ind", costs[min_cost_ind], min_cost_ind)
    print("sum(|F[min]-F_exp|) = ", sum(abs(F[min_cost_ind] - F_exp)))
    norm_std = std_exp / np.sqrt(samples)
    print("Num of outside error bar", sum(abs(F[min_cost_ind] - F_exp) > norm_std))
    # plot_inset_violin(F_exp.real, norm_std.real, F.real, m, noise_model, min_cost_ind, costs, fname, save)
    plot_inset(F_exp.real, norm_std.real, F.real, m, noise_model, min_cost_ind, costs, fname, save)
    # ASF_learning_plot(F_exp, norm_std, F, m, 1, min_cost_ind+1)
    # plt.plot(costs.real)
    # plt.xlabel("Number of updates")
    # plt.ylabel("Cost")
    
    return data, F_exp, norm_std, F, costs

def plot_inset(F_exp, norm_std, F, m, noise_model, up_ind, costs, fname, save):
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.5, 0.60, 0.4, 0.28]
    # left, bottom, width, height = [0.2, 0.2, 0.35, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.errorbar(range(1,m+1), F_exp, yerr=norm_std, c='b', fmt="o", ms=3, label='$\\mathcal{F}^{%s}$' % (noise_model))
    ax1.scatter(range(1,m+1), F[up_ind].real, s=10, marker="s", c='r', label='$\\mathcal{F}_{%d}^{%s}$' % (up_ind, noise_model))
    ax2.plot(costs)
    ax2.patch.set_alpha(0.7)
    ax2.set_xlabel("Number of updates")
    ax2.set_ylabel("Cost")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Sequence fidelity")
    ax1.legend(loc='lower left')
    # ax1.legend(loc='upper right')
    ax1.grid()
    if save == True:
        plt.savefig("figures/" + fname + ".pdf", format="pdf", bbox_inches="tight")
    elif save == "png":
        plt.savefig("figures/" + fname + ".png", format="png", bbox_inches="tight")
    plt.show()

def plot_inset_violin(F, m, noise, up_ind, costs, fname, save):
    """
    Finished, but in the sequence length m=60 is not obvious. Will be just a line.
    """
    if noise == "nM":
        F_e = np.load("nM_m60_samp100_Fe.npy")
    # elif noise == "PF":

    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.5, 0.60, 0.4, 0.28]
    # left, bottom, width, height = [0.2, 0.2, 0.35, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.violin(F_e)
    ax1.scatter(range(1,m+1), F[up_ind].real, s=10, marker="o", c='r', label='$\\mathcal{F}_{%d}^{%s}$' % (up_ind, noise))
    ax2.plot(costs)
    ax2.set_xlabel("Number of updates")
    ax2.set_ylabel("Cost")
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Sequence fidelity")
    ax1.legend(loc='lower left')
    # ax1.legend(loc='upper right')
    ax1.grid()
    if save == True:
        plt.savefig(fname + "_ASF.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def load_plot_non_Markovianity(fname, m, noise_model, save, samples=100):
    data = np.load('data/' + fname +'.npz')
    F_exp = np.mean(data['F_exp'], axis=0)
    std_exp = data['std_exp']
    F = np.mean(data['F'], axis=1)
    costs = data['costs']
    markF_exp = np.mean(data['markF_exp'], axis=0)
    nonMarkovianity = data['nonM']

    min_cost_ind = np.where(costs == min(costs))
    min_cost_ind = min_cost_ind[0][0]
    eval = (np.sum((markF_exp - F_exp)**2 / 2) - nonMarkovianity[min_cost_ind]) / np.sum((markF_exp - F_exp)**2 / 2)
    print("min cost & min non-Markovianity & ind", costs[min_cost_ind], nonMarkovianity[min_cost_ind], min_cost_ind)
    print("CM(exp) - CM(model) = ", eval)
    print("sum(|F[min]-F_exp|) = ", sum(abs(F[min_cost_ind] - F_exp)))
    norm_std = std_exp / np.sqrt(samples)
    print("Num of outside error bar", sum(abs(F[min_cost_ind] - F_exp) > norm_std))
    plot_inset(F_exp.real, norm_std.real, F.real, m, noise_model, min_cost_ind, costs, fname, save)

    
    return data, F_exp, norm_std, F, costs

class rand_clifford_sequence_markovianized_asf:
    def __init__(self, m, rho_s, proj_O, sys_dim=2, bond_dim=2) -> None:
        self.m = m
        self.proj_O = proj_O
        self.tmp_rho = rho_s
        self.envd = bond_dim
        self.sysd = sys_dim
        self.env_sys = [bond_dim, sys_dim, bond_dim, sys_dim]
        self.dim = sys_dim * bond_dim
        
        # self.comp_noise = np.identity(int(self.dim/noise_dim), dtype=complex)
    def _unitary_map(self, rho, unitary):
        return unitary @ rho @ np.conj(unitary).T
    
    def _trace_E(self, rho):
        return np.trace(rho.reshape(self.env_sys), axis1=0, axis2=2)

    def _rand_clifford_sequence_markovianized_list(self, n, noise_u, rand_clifford):
        # apply unitary noise in the sequence
        # Each step has different noise as indicate by the list.
        # dim = sys_dim * bond_dim
        # tmp_rho = rho_s
        inver_op = np.eye(self.sysd)
        comp_sys = np.identity(self.envd, dtype=complex) / (self.dim/self.sysd)
        # I_gate = np.identity(self.envd, dtype=complex)
        tmp_rho = self.tmp_rho
        for i in range(n):
            noise_dim = noise_u[i].shape[0]
            comp_noise = np.identity(int(self.dim/noise_dim), dtype=complex)
            tmp_rho = self._unitary_map(tmp_rho, rand_clifford[i])
            tmp_rho = self._unitary_map(np.kron(comp_sys, tmp_rho), np.kron(comp_noise, noise_u[i]))
            tmp_rho = self._trace_E(tmp_rho)
            inver_op = inver_op @ np.conj(rand_clifford[i]).T
            
        tmp_rho = self._unitary_map(tmp_rho, inver_op)
        noise_dim = noise_u[n].shape[0]
        comp_noise = np.identity(int(self.dim/noise_dim), dtype=complex)
        tmp_rho = self._unitary_map(np.kron(comp_sys, tmp_rho), np.kron(comp_noise, noise_u[n]))
        tmp_rho = self._trace_E(tmp_rho)
        return tmp_rho

    def markovianized_asf(self, sam_clif, noise_u):
        sample_size = sam_clif.shape[0]
        markF = np.zeros((sample_size, self.m))
        if type(noise_u) != type([]):
            noise_u = [noise_u] * (self.m + 1)
            
        for sam in range(sample_size):
            for n in range(self.m):
                tmp_rho = self._rand_clifford_sequence_markovianized_list(n, noise_u[:n+1], sam_clif[sam, :n+1])
                markF[sam, n] = np.trace(self.proj_O @ tmp_rho).real
        return markF
"""
# For testing rand_clifford_sequence_markovianized_asf class. Done!!!!!!!!!!!!!!!!!!!!!!!!!
import numpy as np
from utils_tn import rand_clifford_sequence_markovianized_asf, noise_nonM_unitary
from RB_numerical_toy_model_v1 import clifford_sequence_with_noise
from utils_tn import single_cliffords
from random import randint

m = 10
noise_u = noise_nonM_unitary(10, J=1.2, hx=1.17, hy=-1.15, delta=0.1)
sample_size = 100
clifford_list = single_cliffords()

sam_clif = np.zeros((sample_size, m, 2, 2), dtype=complex)
for sam in range(sample_size):
    for i in range(m):
        sam_clif[sam, i] = clifford_list[randint(0,23)]

rho_s = np.array([[1, 0], [0, 0]], dtype=complex)
proj_O = np.array([[1, 0], [0, 0]], dtype=complex)
markovianized = rand_clifford_sequence_markovianized_asf(m, rho_s, proj_O, 2, 2)
markF = markovianized.markovianized_asf(sam_clif, noise_u)
"""

"""
The following should be added to tensornetwork/network_operations.
Edit from tn.split_node().

tensornetwork/__init__.py import split_node_u_s_vh
"""
# def split_node_u_s_vh(
#     node: AbstractNode,
#     left_edges: List[Edge],
#     right_edges: List[Edge],
#     max_singular_values: Optional[int] = None,
#     max_truncation_err: Optional[float] = None,
#     relative: Optional[bool] = False,
#     left_name: Optional[Text] = None,
#     right_name: Optional[Text] = None,
#     edge_name: Optional[Text] = None,
# ) -> Tuple[AbstractNode, AbstractNode, Tensor]:
#   """
#   Edit from tensornetwork/network_operations.py.
#   To get separate u, s, vh.
#   """

#   if not hasattr(node, 'backend'):
#     raise AttributeError('Node {} of type {} has no `backend`'.format(
#         node, type(node)))

#   if node.axis_names and edge_name:
#     left_axis_names = []
#     right_axis_names = [edge_name]
#     for edge in left_edges:
#       left_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
#                              else node.axis_names[edge.axis2])
#     for edge in right_edges:
#       right_axis_names.append(node.axis_names[edge.axis1] if edge.node1 is node
#                               else node.axis_names[edge.axis2])
#     left_axis_names.append(edge_name)
#   else:
#     left_axis_names = None
#     right_axis_names = None

#   backend = node.backend
#   transp_tensor = node.tensor_from_edge_order(left_edges + right_edges)

#   u, s, vh, trun_vals = backend.svd(transp_tensor,
#                                     len(left_edges),
#                                     None, # max_singular_values
#                                     max_truncation_err,
#                                     relative=relative)
  
#   allsig = np.append(s, trun_vals)
#   sqrt_s = backend.sqrt(s)
#   u_s = backend.broadcast_right_multiplication(u, sqrt_s)
#   vh_s = backend.broadcast_left_multiplication(sqrt_s, vh)
#   # u_s = u_s[:,:,:,2:2+max_singular_values]
#   # vh_s = vh_s[2:2+max_singular_values,:,:,:]
#   u_s = u_s[:,:,:,:max_singular_values]
#   vh_s = vh_s[:max_singular_values,:,:,:]

#   if max_singular_values == None:
#     sig = np.zeros([len(s)]*2)
#   else:
#     sig = np.zeros([max_singular_values]*2)
#   # assume 1 system qubit, 2:2+max_..., max_singular_values related to num of env qubit.
#   np.fill_diagonal(sig, s[:max_singular_values])
#   mid_node = Node(sig,
#                   name='s value',
#                   backend=backend)

#   left_node = Node(u_s,
#                    name=left_name,
#                    axis_names=left_axis_names,
#                    backend=backend)

#   left_axes_order = [
#       edge.axis1 if edge.node1 is node else edge.axis2 for edge in left_edges
#   ]
#   for i, edge in enumerate(left_edges):
#     left_node.add_edge(edge, i)
#     edge.update_axis(left_axes_order[i], node, i, left_node)

#   right_node = Node(vh_s,
#                     name=right_name,
#                     axis_names=right_axis_names,
#                     backend=backend)

#   right_axes_order = [
#       edge.axis1 if edge.node1 is node else edge.axis2 for edge in right_edges
#   ]
#   for i, edge in enumerate(right_edges):
#     # i + 1 to account for the new edge.
#     right_node.add_edge(edge, i + 1)
#     edge.update_axis(right_axes_order[i], node, i + 1, right_node)

#   # connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
#   connect(left_node.edges[-1], right_node.edges[0], name=edge_name)
#   node.fresh_edges(node.axis_names)
#   return left_node, right_node, mid_node, allsig

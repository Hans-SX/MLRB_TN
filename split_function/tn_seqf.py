"""
Created on Mon Mar 14 11:31:54 2023

@author: sxyang
"""
#%%
import numpy as np
from random import randint
import tensornetwork as tn
import matplotlib.pyplot as plt

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils_tn import initialized_lamdas_tn, gen_control_ten, order2_to_4
from utils_tn import edges_btw_ctr_nois, edges_in_lamdas
from utils_tn import order4_to_2, single_cliffords
from utils_tn import contract_by_nodes, noise_nonM_unitary
from reproduce_210705403 import non_Markovian_theory_Fm, sequence_with_unitary_noise, non_Markovian_unitary_map

"""
Verifying the ASF from analytic form, simulated experiment, and tensor network model are the same.
tn_asf()
input:
num_gates -> num of gates;

noise_lambda -> noise to put in the process tensor;
sequence_gates -> a sequence of gates to from a control tensor.

output:
AvgSF -> averaged sequence fidelity
"""

#%%
class tn_rb():
    def sample_single_cliffords(self, num_gates, sample_size=200, sys_dim=2):
        self.sys_dim = sys_dim
        rho_s = np.zeros((sys_dim, sys_dim), dtype=complex)
        rho_s[0] = 1
        proj_O = rho_s

        clifford_list = single_cliffords()

        sam_clif = np.zeros((sample_size, num_gates, 2, 2), dtype=complex)
        control_ten = []
        control_n = []
        sam_n = dict()
        for sam in range(sample_size):
            for i in range(num_gates):
                sam_clif[sam, i] = clifford_list[randint(0,23)]
            control_ten.append(gen_control_ten(rho_s, num_gates, proj_O, sam_clif[sam]))
            for j in range(num_gates-1):
                control_n.append(gen_control_ten(rho_s, j+1, proj_O, sam_clif[sam, :j+1]))
            sam_n[sam] = control_n
            control_n = []
        # control_ten[sampe][gate]
        return control_ten, sam_n


    def tn_seqf(self, num_gates, noise_lambda, sequence_gates,  sam_n, bond_dim=2):
        
        rho_e = np.zeros((bond_dim, bond_dim), dtype=complex)
        rho_e[0] = 1


        if type(noise_lambda) == type([]):
            noise_lambda = noise_lambda[:num_gates+2]
        # else:
        #     noise_lambda = self.noise_lambda

        lamdas = initialized_lamdas_tn(num_gates, noise_lambda, rho_e, self.sys_dim, bond_dim)

        # edges for constructing Fm, m = m.
        e_edgs = edges_in_lamdas(lamdas, num_gates)
        s_edgs = edges_btw_ctr_nois(sequence_gates, lamdas, num_gates)
        
        SeqF_num_gates = contract_by_nodes(lamdas + sequence_gates).tensor.real

        SeqF_n_gates = np.zeros(num_gates - 1)
        for n_gates in range(1, num_gates):
            noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor, self.sys_dim, bond_dim), np.arange(num_gates-n_gates,num_gates+2)))
            noise_tmp.reverse()
            lam_n = initialized_lamdas_tn(n_gates, noise_tmp, rho_e, self.sys_dim, bond_dim)
            lam_n_edg = edges_in_lamdas(lam_n, n_gates)
            ctr_n_edg = edges_btw_ctr_nois(sam_n[n_gates-1], lam_n, n_gates)
            tmp_n = tn.contractors.auto(lam_n + sam_n[n_gates-1], None, ignore_edge_order=True)
            SeqF_n_gates[n_gates-1] = tmp_n.tensor.real
        SeqF = np.concatenate((SeqF_n_gates.reshape(num_gates-1), SeqF_num_gates.reshape(1)), axis=0)

        return SeqF



#%%
if __name__ == '__main__':
    num_gates = 30
    sample_size = 200
    bond_dim = 4

    # In the this form of non-Markovian noise, each matrix is the same after a gate. And noise_nonM_unitary will generate (num_gates + 1) noise.
    noise_lambda = np.array(noise_nonM_unitary(0, J=1.2, hx=1.17, hy=-1.15, delta=0.1))
    ex_dim = int(bond_dim/2 - 1)
    if ex_dim > 0:
        noise_ex_env = np.kron(np.identity(2**ex_dim), noise_lambda)

    # ------------ ASF from tensor network ---------
    rb_f = tn_rb()
    control_ten, sam_n = rb_f.sample_single_cliffords(num_gates, sample_size)
    SeqF = np.zeros((sample_size, num_gates))
    SeqF_ex_env = np.zeros((sample_size, num_gates))
    for samp in range(sample_size):
        SeqF[samp] = rb_f.tn_seqf(num_gates, noise_lambda, control_ten[samp], sam_n[samp], bond_dim=2)
        SeqF_ex_env[samp] = rb_f.tn_seqf(num_gates, noise_ex_env, control_ten[samp],  sam_n[samp], bond_dim=4)
    
    AvgSF = np.mean(SeqF, axis=0)
    AvgSF_ex_env = np.mean(SeqF_ex_env, axis=0)

    # ---------- ASF from exp -------------
    # fm_exp = np.zeros(sample_size)
    # Fm_exp = np.zeros(num_gates)
    # rho = np.zeros((4,4), dtype=complex)
    # rho[0] = 1
    # id = np.identity(2)
    # proj_O = rho[:2,:2]
    
    # for m in range(1, num_gates+1):
    #     for i in range(sample_size):
    #         tmp_rho, inver_op = sequence_with_unitary_noise(m, rho, noise_lambda, id)
    #         # tmp_rho = tmp_rho.reshape(4,4)
    #         """tmp_rho.shape = (4,4,1)??? why"""
    #         tmp_rho = np.kron(id, inver_op) @ tmp_rho @ np.conj(np.kron(id, inver_op)).T
    #         # final_state = noise_u @ tmp_rho @ np.conj(noise_u).T
    #         final_state = non_Markovian_unitary_map(tmp_rho, noise_lambda)
    #         f_sys_state = np.trace(final_state.reshape(2,2,2,2), axis1=0, axis2=2)
    #         fm_exp[i] = np.trace(proj_O @ f_sys_state).real
        
    #     Fm_exp[m-1] = np.average(fm_exp)

    # # -------- ASF from analytic -------------
    # nonM_theory_Fm  = non_Markovian_theory_Fm(noise_lambda, proj_O, rho, id, 2)
    # theory_Fm = np.zeros(num_gates)
    # for i in range(num_gates):
    #     theory_Fm[i] = nonM_theory_Fm.theory_Fm(i)

    # plt.plot(range(num_gates), AvgSF, 'g--', range(num_gates), AvgSF_ex_env, 'b--', range(num_gates), Fm_exp, 'k^', range(num_gates), theory_Fm, 'ro')
    plt.plot(range(num_gates), AvgSF, 'g--', range(num_gates), AvgSF_ex_env, 'b--')
    print("Finished")
#%%
"""
Created on Thu Sep 15 1:10 2022.

This verifies it did not work if average Clifford sequences first then calculate the "ASF".
It drops fast, since the averaged gates are not unitaries.
Then, the invers by taking transpose and conjuate is not a really invers.

But this is for simulating, putting gate, noise, gate,... perhaps the contraction is fine?
It did not!!
It is even worst, the F_avg in this way is very small from the begining.
"""
#%%
import numpy as np
from numpy.linalg import inv
from scipy.stats import unitary_group
from scipy import linalg
from random import randint
import matplotlib.pyplot as plt
from RB_numerical_toy_model_v1 import clifford_sequence_with_noise, gen_randH
import tensornetwork as tn
from utils_tn import single_cliffords, gen_control_ten, initialized_lamdas_tn, edges_btw_ctr_nois, edges_in_lamdas, contract_by_nodes, order4_to_2


def non_Markovian_unitary_map(rho, noise_u):
    return noise_u @ rho @ np.conj(noise_u).T



X = np.array([[0, 1],[1, 0]], dtype=complex)
Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
Z = np.array([[1, 0],[0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

ds = 2   
de = 2
J = 1.2
hx = 1.17
hy = -1.15
delta = 0.05
bond_dim = 2
sys_dim = 2

H = J * np.kron(X,X) + hx * (np.kron(X, I) + np.kron(I, X)) + hy * (np.kron(Y, I) + np.kron(I, Y))

# Non-Markovian noise, assume unitary.
noise_u = linalg.expm(-1j*delta*H)


seed = 5
np.random.seed(seed)

ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
ket_1 = np.array([0,1], dtype=complex).reshape(2,1)

rho = np.kron(np.kron(ket_0, np.conj(ket_0.T)), np.kron(ket_0, np.conj(ket_0.T)))
proj_O = np.kron(ket_0, np.conj(ket_0.T))
rho_s = proj_O
noise_model = "randH"

m = 20


lamdas = initialized_lamdas_tn(m, noise_u, rho_s)


sample_size = int(1000)

clifford_list = single_cliffords()
sam_clif = np.zeros((sample_size, m, 2, 2), dtype=complex)
control_ten = []
control_n = []
sam_n = dict()
F = np.zeros(m)
avg_seq = np.zeros((m, 2, 2), dtype=complex)

for sam in range(sample_size):
    for i in range(m):
        sam_clif[sam, i] = clifford_list[randint(0,23)]
    control_ten.append(gen_control_ten(rho_s, m, proj_O, sam_clif[sam]))
    for j in range(m-1):
        control_n.append(gen_control_ten(rho_s, j+1, proj_O, sam_clif[sam, :j+1]))
    sam_n[sam] = control_n
    control_n = []

tmp_F = np.zeros(sample_size, dtype=complex)
tmp_Fn = np.zeros((sample_size, m-1), dtype=complex)
for sam in range(sample_size):
    avg_seq += sam_clif[sam]
   # edges for constructing Fm, m = m.
    e_edgs = edges_in_lamdas(lamdas, m)
    s_edgs = edges_btw_ctr_nois(control_ten[sam], lamdas, m)
    
    tmp_F[sam] = contract_by_nodes(lamdas+control_ten[sam]).tensor
    
    for n in range(1,m):
        noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor, sys_dim, bond_dim), np.arange(m-n,m+2)))
        noise_tmp.reverse()
        lam_n = initialized_lamdas_tn(n, noise_tmp, rho_s, sys_dim, bond_dim)
        lam_n_edg = edges_in_lamdas(lam_n, n)
        ctr_n_edg = edges_btw_ctr_nois(sam_n[sam][n-1], lam_n, n)
        tmp_n = tn.contractors.auto(lam_n+sam_n[sam][n-1], None, ignore_edge_order=True)
        tmp_Fn[sam, n-1] = np.abs(tmp_n.tensor)
F[:m-1] = np.mean(tmp_Fn, axis=0)
F[m-1] = np.mean(tmp_F)


# avg_ten = []
avg_n = []
avg_seq = avg_seq / sample_size
avg_ten = gen_control_ten(rho_s, m, proj_O, avg_seq)
for j in range(m-1):
    avg_n.append(gen_control_ten(rho_s, j+1, proj_O, avg_seq[:j+1]))

e_edgs = edges_in_lamdas(lamdas, m)
s_edgs = edges_btw_ctr_nois(avg_ten, lamdas, m)

avg_F = np.zeros(m)
avg_F[-1] = contract_by_nodes(lamdas+avg_ten).tensor.real

for n in range(1,m):
    noise_tmp = list(map(lambda x: order4_to_2(lamdas[x].tensor, sys_dim, bond_dim), np.arange(m-n,m+2)))
    noise_tmp.reverse()
    lam_n = initialized_lamdas_tn(n, noise_tmp, rho_s, sys_dim, bond_dim)
    lam_n_edg = edges_in_lamdas(lam_n, n)
    ctr_n_edg = edges_btw_ctr_nois(avg_n[n-1], lam_n, n)
    tmp_n = tn.contractors.auto(lam_n+avg_n[n-1], None, ignore_edge_order=True)
    avg_F[n-1] = tmp_n.tensor.real

"""
randH = gen_randH(0)
noise_para = 0.1
fm = np.zeros((sample_size, m))
avg_seq = np.zeros((m, 2, 2), dtype=complex)
avg_F_exp = np.zeros(m)
for sam in range(sample_size):
    avg_seq += sam_clif[sam]
    for n in range(m):
        final_state, _ = clifford_sequence_with_noise(rho_s, sam_clif[sam, :n+1], noise_model, noise_para, seed, randH)
        fm[sam, n] = np.trace(proj_O @ final_state).real
        F_exp = np.mean(fm, axis=0)
        std_exp = np.std(fm, axis=0)
avg_seq = avg_seq / sample_size
for n in range(m):
    avg_final_state, _ = clifford_sequence_with_noise(rho_s, avg_seq[:n+1], noise_model, noise_para, seed, randH)
    avg_F_exp[n] = np.trace(proj_O @ avg_final_state).real
"""

# plt.plot(range(1, m+1), F_exp, 'o', label=r'$F_exp$')
# plt.plot(range(1, m+1), avg_F_exp, 's', label=r'$avg_F_exp$')
plt.plot(range(1, m+1), F, 'o', label=r'$F$')
plt.plot(range(1, m+1), avg_F, 's', label=r'$F_avg$')
plt.legend()
#%%
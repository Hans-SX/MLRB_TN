"""
Created on Mon Jun 13 14:17:54 2022

@author: sxyang
"""

from turtle import update
import numpy as np
import tensornetwork as tn
import argparse
from scipy.stats import unitary_group
import random
from utils_tn import plot_inset
# from finite_memory_contr_by_node_func import estimate_noise_via_sweep
# from flexible_env_qubit_model import estimate_noise_via_sweep_envq
# from flexible_env_qubit_model_KL_divergence import estimate_noise_via_sweep_envq
from flexible_env_qubit_model_nonMarkovianity_KL_divergence import estimate_noise_via_sweep_envq

# Initialize parser.
parser = argparse.ArgumentParser()

# Adding optionale argument.
parser.add_argument('--m', type=int, default=60)
parser.add_argument('--lr', type=complex, default=0.01)
parser.add_argument('--adam1', type=complex, default=0.9)
parser.add_argument('--adam2', type=complex, default=0.999)
parser.add_argument('--opt', type=str, default='AdaGrad')
parser.add_argument('--lfile', type=bool, default=False)
# To continue from a current best for the replace pair case,
# need some effort. The initial noise should not take the one corresponding to the min_ind.
parser.add_argument('--fname', type=str, default='./data/m10_dimE4_lr0.001_updates3000_sample200_seed5_Ada_replace_1_nM_init_randu_cost_KL_frob_wb_1_nM.npz')
parser.add_argument('--nM', default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--ups', type=int, default=50)
parser.add_argument('--samps', type=int, default=200)
parser.add_argument('--update_all', default=True)
parser.add_argument('--noise_model', type=str, default="nM")
parser.add_argument('--bond_dim', type=int, default=2)
# coeff for making (F - F_exp) smaller, m>30.
parser.add_argument('--coeff', type=int, default=1)
parser.add_argument('--test', default=False)
parser.add_argument('--init_noise', default=None)
parser.add_argument('--weight_frob', type=float, default=1)

# Read arguments from command line.
args = parser.parse_args()

m = args.m
lr = tn.Node(complex(args.lr))
adam1 = args.adam1
adam2 = args.adam2
optimizer = args.opt
lfname = args.fname
nM = args.nM
rand_seed = args.seed
updates = args.ups
sample_size = args.samps
update_all = args.update_all
noise_model = args.noise_model
bond_dim = args.bond_dim
coeff = args.coeff
test = args.test
init_noise = args.init_noise
wb = args.weight_frob

sys_dim = 2
delta = 5

# if args.init_noise == "randu":
#     init_noise = unitary_group.rvs(bond_dim * sys_dim)
# else:
#     init_noise = args.init_noise

if args.lfile:
    data = np.load(lfname)
    min_ind = np.where(data['costs']==min(data['costs']))[0][0]
    init_noise = data['noise_ten'][min_ind-1]
    # random.seed(10)
# else:
#     init_noise = None
# print('updat_all?', update_all)
# exit()

# F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model)

F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname, markF, markF_exp, nonMarkovianity = estimate_noise_via_sweep_envq(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model, sys_dim, bond_dim, coeff, test, wb)

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

print(fname)
"""
Created on Mon Jun 13 14:17:54 2022

@author: sxyang
"""

from turtle import update
import numpy as np
import tensornetwork as tn
import argparse
from finite_memory_contr_by_node_func import estimate_noise_via_sweep
# from flexible_env_qubit_model import estimate_noise_via_sweep_envq
from flexible_env_qubit_model_mean_variance import estimate_noise_via_sweep_envq

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
parser.add_argument('--fname', type=str, default='m20_lr3.0_updates2000_sample100_seed5_replace_1_nM_load_cb.npz')
parser.add_argument('--nM', default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--ups', type=int, default=50)
parser.add_argument('--samps', type=int, default=50)
parser.add_argument('--update_all', default=True)
parser.add_argument('--noise_model', type=str, default="nM")
parser.add_argument('--bond_dim', type=int, default=2)
# coeff for making (F - F_exp) smaller, m>30.
parser.add_argument('--coeff', type=int, default=1)
parser.add_argument('--test', default=False)

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

sys_dim = 2
delta = 5

if args.lfile:
    data = np.load(lfname)
    min_ind = np.where(data['costs']==min(data['costs']))[0][0]
    init_noise = data['noise_ten'][min_ind]
else:
    init_noise = None
# print('updat_all?', update_all)
# exit()

# F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model)

F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep_envq(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model, sys_dim, bond_dim, coeff, test)

print(fname)
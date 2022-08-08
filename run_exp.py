"""
Created on Mon Jun 13 14:17:54 2022

@author: sxyang
"""

from turtle import update
import numpy as np
import tensornetwork as tn
import argparse
from finite_memory_contr_by_node_func import estimate_noise_via_sweep


# Initialize parser.
parser = argparse.ArgumentParser()

# Adding optionale argument.
parser.add_argument('--m', type=int, default=60)
parser.add_argument('--step')
parser.add_argument('--adam1', type=complex, default=0.9)
parser.add_argument('--adam2', type=complex, default=0.99)
parser.add_argument('--opt', type=str, default='AdaGrad')
parser.add_argument('--lfile', type=bool, default=False)
# To continue from a current best for the replace pair case,
# need some effort. The initial noise should not take the one corresponding to the min_ind.
parser.add_argument('--fname', type=str, default='m20_lr3.0_updates2000_sample100_seed5_replace_1_nM_load_cb.npz')
parser.add_argument('--nM', default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--ups', type=int, default=50)
parser.add_argument('--samps', type=int, default=100)
parser.add_argument('--update_all', default=True)
parser.add_argument('--noise', type=str, default="nM")

# Read arguments from command line.
args = parser.parse_args()

m = args.m
lr = tn.Node(complex(args.step))
adam1 = args.adam1
adam2 = args.adam2
optimizer = args.opt
lfname = args.fname
nM = args.nM
rand_seed = args.seed
updates = args.ups
sample_size = args.samps
update_all = args.update_all
noise_model = args.noise

if args.lfile:
    data = np.load(lfname)
    min_ind = np.where(data['costs']==min(data['costs']))[0][0]
    init_noise = data['noise_ten'][min_ind]
else:
    init_noise = None
# print('updat_all?', update_all)
# exit()
delta = 2
F_exp, std_exp, F, all_sigs, costs, noise_ten, Duration, fname = estimate_noise_via_sweep(m, updates, sample_size, rand_seed, lr, delta, nM, update_all, adam1, adam2, init_noise, optimizer, noise_model)

print(fname)
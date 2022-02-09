#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:20:43 2022

Only test on cloud, my notebook has not enough ram.
There is a better version, this one only estimate the contraction time.
Another also check the correctness.

@author: sxyang
"""

import numpy as np
import tensornetwork as tn
import time

a = np.identity(2, dtype=complex)
b = np.identity(2, dtype=complex)
m = 2
n = 4*m+4 - 1
tic_make = time.perf_counter()
for i in range(n):
    b = np.tensordot(b, a, axes=0)
toc_make = time.perf_counter()
make_ten_time = toc_make-tic_make

tic_shuff = time.perf_counter()
for i in range(5):
    c = np.moveaxis(b, i, np.random.randint(i+5, n+1))
toc_shuff = time.perf_counter()
shuff_ten_time = toc_shuff-tic_shuff

tic_make_node = time.perf_counter()
B = tn.Node(b)
toc_make_node = time.perf_counter()
m_node_time = toc_make_node - tic_make_node

C = tn.Node(c)
tic_edging = time.perf_counter()
for i in range(n+1):
    B[i] ^ C[i]
toc_edging = time.perf_counter()
edging_time = toc_edging - tic_edging

tic_cont = time.perf_counter()
D = B @ C
toc_cont = time.perf_counter()
cont_time = toc_cont - tic_cont

print('make_ten_time', make_ten_time)
print('shuff_ten_time', shuff_ten_time)
print('m_node_time', m_node_time)
print('edging_time', edging_time)
print('cont_time', cont_time)

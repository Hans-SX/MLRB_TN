#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:13:14 2022

Initial ver of class control_tensor, unfinished and without tensornetwork.

@author: sxyang
"""

import numpy as np

def order2_to_4(lamda, sys_dim=2, bond_dim=2):
    lamda = lamda.reshape(sys_dim, bond_dim, sys_dim, bond_dim)
    lamda= np.swapaxes(lamda, 0, 1)
    return lamda

def order4_to_2(lamda, sys_dim=2, bond_dim=2):
    lamda= np.swapaxes(lamda, 0, 1)
    lamda = lamda.reshape(sys_dim*bond_dim, sys_dim*bond_dim)
    return lamda

def initialized_lamdas(m, rho_e, sys_dim=2, bond_dim=2):
    lamdas = []
    lamdas.append(rho_e)
    for i in range(m+2):
        # e(m+2) = r(m+2), the out most edge.
        tmp = order2_to_4(np.identity(sys_dim * bond_dim, dtype=complex))
        lamdas.insert(0, tmp)
        lamdas.append(np.conj(tmp.T))
    return lamdas

class control_tensor():
    def __init__(self, m, sys_dim):
        self.m = m
        self.dim = sys_dim
        
    def _2_kronecker_del(self, shuffled=False):
        del_2 = np.zeros([self.dim]*4)
        if shuffled == True:
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            if b==d and c==e:
                                del_2[b,c,d,e] = 1
        else:
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            if b==c and d==e:
                                del_2[b,c,d,e] = 1
        return del_2
    
    def Delta(self):
        delta = self._2_kronecker_del(True)
        del_2 = self._2_kronecker_del(True)
        for i in range(1, self.m):
            delta = np.tensordot(delta, del_2, axes=0)
            delta = np.moveaxis(delta, [-4, -3], [0, 1])
        return delta/self.dim**self.m
        
    def Alpha(self):
        del_2 = self._2_kronecker_del()*self.dim
        del_2_sf = self._2_kronecker_del(True)
        unit_alpha = (del_2 - del_2_sf)/ (self.dim * (self.dim**2 - 1))
        alpha = unit_alpha
        for i in range(1, self.m):
            alpha = np.tensordot(alpha, unit_alpha, axes=0)
            alpha = np.moveaxis(alpha, [-4, -3], [0, 1])
        
        return alpha
    
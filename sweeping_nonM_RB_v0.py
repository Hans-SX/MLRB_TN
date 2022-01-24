#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:01:48 2022

Considering 1 to m steps ASF via analytical expression.

@author: sxyang
"""

import numpy as np
import tensornetwork as tn
from utils_tn import initialized_lamdas_tn



if __name__ == "__main__":
    m = 2
    lr=1e-2
    shift = 2
    env_qbit=1    # Assume enviornment is all 0 state |00...>
    # ----------------------------------------------
    # averaging 1 gate tensor.
    rho_s = np.zeros((2,2), dtype=complex)
    rho_s[0,0] = 1
    M = rho_s
    
    bond_dim = 2 ** env_qbit
    
    rho_e = np.zeros((bond_dim, bond_dim), dtype=complex)
    rho_e[0,0] = 1
    
    lamdas = initialized_lamdas_tn(m, noise_u, rho_e)
    
    #---------- Step 1 finished. ------------
    
    
    
    
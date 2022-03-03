#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:44:50 2022

Control tensor follow Eq.B20 ~ B22 in RB_with_ML_0113_22 .

@author: sxyang
"""

import numpy as np
from scipy import linalg
from random import randint
import tensornetwork as tn

def order2_to_4(lamda, sys_dim=2, bond_dim=2):
    lamda = lamda.reshape(sys_dim, bond_dim, sys_dim, bond_dim)
    lamda= np.swapaxes(lamda, -2, -1)
    # lamda= np.swapaxes(lamda, 0, 1)
    return lamda

def order4_to_2(lamda, sys_dim=2, bond_dim=2):
    lamda= np.swapaxes(lamda, -2, -1)
    # lamda= np.swapaxes(lamda, 0, 1)
    lamda = lamda.reshape(sys_dim*bond_dim, sys_dim*bond_dim)
    return lamda

def initialized_lamdas_tn(m, noise_u, rho_e, sys_dim=2, bond_dim=2):
    # lam_edgs = []
    lamdas = []
    lamdas.append(tn.Node(rho_e, name='rho_e', axis_names=['e0', 'r0']))
    if type(noise_u) == type([]):
        for i in range(m+2):
            # e(m+2) = r(m+2), the out most edge.
            tmp = order2_to_4(noise_u[i])
            lam = tn.Node(tmp, name='lam'+str(i), axis_names=['e'+str(i+1), 's'+str(i), 's\''+str(i),'e'+str(i)])
            lam_dg = tn.Node(np.conj(tmp.T), name='lam_dg'+str(i), axis_names=['r'+str(i), 'z'+str(i), 'z\''+str(i), 'r'+str(i+1)])
            
            lamdas.insert(0, lam)
            lamdas.append(lam_dg)
    else:
        for i in range(m+2):
            # e(m+2) = r(m+2), the out most edge.
            tmp = order2_to_4(noise_u)
        
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

# def contract_by_nodes(cont_ls):
#     tensor = tn.contractors.auto(cont_ls, ignore_edge_order=True)
    

def contract_by_nodes(cont_ls, out_order, node_name):
    tensor = tn.contractors.auto(cont_ls, output_edge_order=None, ignore_edge_order=True)
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
    # Pop out in reverse order, so that they sill not affect the order of the remainings.
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
    
if __name__ == '__main__':
    test_L = L.copy()
    test_tilde = tmp_theta_2.copy()
    test_edg = []
    for i in range(6):
        test_edg.append(test_L[i] ^ test_tilde[i])
    test_f = contract_edge_list(test_edg)
    test_f
'''
class control_tensor():
    def __init__(self, m, sys_dim):
        self.m = m
        self.dim = sys_dim
        
    # def _2_kronecker_del(self):
    #     del_2 = np.zeros([self.dim]*4)
    #     for b in range(2):
    #         for c in range(2):
    #             for d in range(2):
    #                 for e in range(2):
    #                     if b==c and d==e:
    #                         del_2[b,c,d,e] = 1
    #     return del_2
    
    def _uni_del(self, lst,  n):
        # uni_del: delta_x1x2 delta_y1y2
        # There are 2 ways here to deal with subtraction or addition,
        # 1) fix the order for 2 tensors for subtract
        # 2) contract with noise tensor then substract.
        # I choose 1) setting up order for subtraction or addition.
        
        # I want the indices to go along whenever a tensor is generated, easier to track.
        uni_del_ind = [ind1+str(n), ind2+str(n), ind3+str(n), ind4+str(n)] # should be a list to fit in tn.Node().
        
        uni_del_array = np.identity(self.dim, dtype=complex)
        uni_del_array = np.tensordot(uni_del_array, uni_del_array, axes=0) # numpy.array, to do the calculation.
        return uni_del_array, uni_del_ind
        
    
    def _Delta(self):
        delta = np.identity(self.dim*2, dtype=complex).reshape([self.dim]*4) / (self.dim**self.m)

        del_2 = np.identity(self.dim*2, dtype=complex).reshape([self.dim]*4) / (self.dim**self.m)
        for i in range(1, self.m):
            # uni_del: delta_x1x2 delta_y1y2
            uni_del = 'del_s'+ str(i) + 'z' + str(i) + 'del_s'+ str(i) + '\''  + 'z' + str(i-1) + '\'' 
            # uni_axis: x1 x2 y1 y2
            uni_axis = ['s'+ str(i), 'z' + str(i), 's'+ str(i) + '\'', 'z' + str(i) + '\'']
            
            delta = tn.Node(delta, name=uni_del, axis_names=uni_axis)
            delta = np.tensordot(delta, del_2, axes=0) / (self.dim**self.m)
            delta = tn.Node(delta, name='del_')
        # ds^m or (ds^m)^m?
        delta*self.dim**self.m
        delta = tn.Node(delta, name='del_')
        return delta
        
    #--------- here ------------------
    def _subtract(self, d, n):
        # lst1, lst2 -> del1, del2; d -> coefficient; n -> 1~m
        # In the control tensor formula, we only have following 2 orders (with different number).
        lst1 = ['s', 's\'', 'z\'', 'z']
        lst2 = ['s', 'z\'', 's\'', 'z']
        _uni_del, _uni_del_ind = self._uni_del_array(lst1, n)
        # take the advantage of only exchange middle 2 indices.
        _uni_del_shuff, _uni_del_ind = self._uni_del_array(lst2, n)
        _uni_del_shuff = np.swapaxes(_uni_del_shuff, 1, 2)
        _subtract = d * _uni_del - _uni_del_shuff
        
        return _subtract
        
    def _Alpha(self):
        
        alpha_array = self._subtract()
        alpha_ind = 
        for i in range(1:m):
            _subtract = self._substract()
            aplpha_array = np.tensordot(alpha_array, _subtract, axes=0)
            
            alpha_ind = 
            
        
        return alpha_ind, alpha_array
    
    def return_array(self, Meas, rho_s):
        alpha_ind, alpha_array = self._Alpha()
        Delta_ind, Delta_array = self._Delta()
        
        # alpha and a subtraction term
        subtract_ind, subtract_array = self._subtract()
        alpha_contr_uni_del = np.tensordot(alpha_array, uni_del_for_alpha, axes=0)
        alpha_contr_uni_del = np.moveaxis()
        
        uni_del_for_Delta = self._uni_del()
        Delta_contr_uni_del = np.tensordot(Delta_array, uni_del_for_Delta, axes=0)
        Delta_contr_uni_del = np.moveaxis()
        
        main_theta = alpha_contr_uni_del + Delta_contr_uni_del
        Theta = np.tensordot(main_theta, Meas, axes=0)
        Theta = np.tensordot(Theta, rho_e, axes=0)
        Theta = np.moveaxis()
        
        # Creating tn.Node()
        
        return Theta
'''
    
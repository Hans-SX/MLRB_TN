import numpy as np
from utils_tn import rand_clifford_sequence_unitary_noise_list, noise_nonM_unitary, unitary_map

clifford_list = single_cliffords()

sam_clif = np.zeros((sample_size, m, 2, 2), dtype=complex)
control_ten = []
control_n = []
sam_n = dict()
for sam in range(sample_size):
    for i in range(m):
        sam_clif[sam, i] = clifford_list[randint(0,23)]
    control_ten.append(gen_control_ten(rho_s, m, proj_O, sam_clif[sam]))
    for j in range(m-1):
        control_n.append(gen_control_ten(rho_s, j+1, proj_O, sam_clif[sam, :j+1]))
    sam_n[sam] = control_n
    control_n = []

def sim_nonM(m, sample_size, num_q):
    sys_dim = 2
    bond_dim = 2 ** num_q
    noise_u = noise_nonM_unitary(1, J=1.2, hx=1.17, hy=-1.15, delta=0.1)
    F_e = np.zeros((m, sample_size))
    I_gate = np.identity(bond_dim, dtype=complex)

    # F_exp, non-M unitary noise from Pedro's work.
    for sam in range(sample_size):
        for n in range(1, m+1):
            tmp_rho, inver_op = rand_clifford_sequence_unitary_noise_list(n, rho, noise_u, sam_clif[sam, :n], sys_dim, bond_dim)
            tmp_rho = np.kron(I_gate, inver_op) @ tmp_rho @ np.conj(np.kron(I_gate, inver_op)).T
            if type(noise_u) == type([]):
                final_state = unitary_map(tmp_rho, noise_u[n-1])
            else:
                final_state = unitary_map(tmp_rho, noise_u)
            f_sys_state = np.trace(final_state.reshape(bond_dim, sys_dim, bond_dim, sys_dim), axis1=0, axis2=2)
            F_e[n-1, sam] = np.trace(proj_O @ f_sys_state).real
    F_exp = F_e.T
    var_exp = np.var(F_e, axis=1).reshape(m, 1)
    std_exp = np.std(F_e, axis=1)

    return F_exp, var_exp
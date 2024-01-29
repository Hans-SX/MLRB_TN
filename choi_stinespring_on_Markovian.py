
import numpy as np
from scipy.stats import unitary_group
from utils_tn import load_plot, order2_to_4


def amplitude_damping(rho_s):
    para = 0.06
    # noise channel should be the same as sum over Kraus.
    noisy_rho = 1j*np.zeros((2,2))
    K = np.zeros((2,2,2)) + 1j*np.zeros((2,2,2))
    K[0] = np.array([[1,0],[0, np.sqrt(1-para)]])
    K[1] = np.array([[0, np.sqrt(para)],[0,0]])
    # print(K[0])
    # print(K[1])
    for i in range(2):
        noisy_rho += K[i] @ rho_s @ np.conj(K[i]).T
    return noisy_rho

fname ="Markovian_m40_lr0.000101_updates100_sample100_seed5_delta2_Ada"
m = 40
noise_model = "AD"
data, F_exp, norm_std, F, costs = load_plot(fname, m, noise_model)

noise_ten = data['noise_ten']
noise = noise_ten[32]

u = unitary_group.rvs(2)
rho_s = np.zeros((2,2), dtype=complex)
rho_s[0,0] = 1
rho_e = rho_s
rho = np.kron(rho_e, u @ rho_s @ np.conj(u.T))
rho_u = u @ rho_s @ np.conj(u.T)

noise_rho = noise @ rho @ np.conj(noise.T)

ket_0 = np.array([1,0],dtype=complex).reshape(2,1)
ket_1 = np.array([0,1],dtype=complex).reshape(2,1)
ket = [ket_0, ket_1]

# Stinespring, I think the next part is the correct way.
stinespring = order2_to_4(noise_rho)
stinespring = np.trace(stinespring, axis1=0, axis2=3)
cptp = amplitude_damping(rho_u)

# From higher dimension U to Kraus representation.
K = np.zeros((2,2,2)) + 1j*np.zeros((2,2,2))
for i in range(2):
    K[i] = np.conj(ket[i].T) @ rho_u  @ ket[0]

# AD channel
para = 0.06
K_ad = np.zeros((2,2,2)) + 1j*np.zeros((2,2,2))
K_ad[0] = np.array([[1,0],[0, np.sqrt(1-para)]])
K_ad[1] = np.array([[0, np.sqrt(para)],[0,0]])

# From Kraus to Choi representation
noise_U = np.zeros((4,4), dtype=complex)
for i in range(2):
    for j in range(2):
        noise_U += np.kron(np.kron(ket[i], np.conj(ket[j].T)), amplitude_damping(np.kron(ket[i], np.conj(ket[j].T)))) 
import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import json
import os

from datetime import datetime


def calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts):
    psi_up_s = psi_s[::2]
    psi_up_s_shifted = np.roll(psi_up_s, 3)
    psi_dn_s = psi_s[1::2]
    psi_dn_s_shifted = np.roll(psi_dn_s, 3)

    up_a_terms = var_terms[:6]
    up_adg_terms = var_terms[6:12]
    dn_a_terms = var_terms[12:18]
    dn_adg_terms = var_terms[18:]

    t_psi_var = t * (sum([np.conj(psi_up) * up_a_term for psi_up, up_a_term in zip(psi_up_s_shifted, up_a_terms)])
                     + sum([psi_up * up_adg_term for psi_up, up_adg_term in zip(psi_up_s_shifted, up_adg_terms)])
                     + sum([np.conj(psi_dn) * dn_a_term for psi_dn, dn_a_term in zip(psi_dn_s_shifted, dn_a_terms)])
                     + sum([psi_dn * dn_adg_term for psi_dn, dn_adg_term in zip(psi_dn_s_shifted, dn_adg_terms)]))
    ret = t_psi_var
    ret += (t * t_term + u_term + v_term + mu * mu_term)
    ret += (t * dig_h * ((- np.real(ts[0] * (np.conj(psi_up_s[0]) * psi_up_s[3]))
                          - np.real(np.conj(ts[4]) * (np.conj(psi_up_s[1]) * psi_up_s[4]))
                          - np.real(ts[2] * (np.conj(psi_up_s[2]) * psi_up_s[5]))
                          - np.real(np.conj(ts[0]) * (np.conj(psi_up_s[3]) * psi_up_s[0]))
                          - np.real(ts[4] * (np.conj(psi_up_s[4]) * psi_up_s[1]))
                          - np.real(np.conj(ts[2]) * (np.conj(psi_up_s[5]) * psi_up_s[2]))
                          - np.real(ts[1] * (np.conj(psi_dn_s[0]) * psi_dn_s[3]))
                          - np.real(np.conj(ts[5]) * (np.conj(psi_dn_s[1]) * psi_dn_s[4]))
                          - np.real(ts[3] * (np.conj(psi_dn_s[2]) * psi_dn_s[5]))
                          - np.real(np.conj(ts[1]) * (np.conj(psi_dn_s[3]) * psi_dn_s[0]))
                          - np.real(ts[5] * (np.conj(psi_dn_s[4]) * psi_dn_s[1]))
                          - np.real(np.conj(ts[3]) * (np.conj(psi_dn_s[5]) * psi_dn_s[2])))))
    return ret


def update(h_hexa, hexagon_mf_operators, psi_s, err):

    def check(p, q):
        return abs(p-q) <= err

    # d_hex, vec_hex = np.linalg.eig(h_hexa)
    # d_hex, vec_hex = sparse.linalg.eigsh(h_hexa, which='SA')
    d_hex, vec_hex = sparse.linalg.eigs(h_hexa, which='SR')
    d_hex_min, v_hex_min = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

    Phi_s = np.array([v_hex_min.conj().T.dot(b.dot(v_hex_min)) for b in hexagon_mf_operators])
    phi_s = psi_s

    # value difference for designated order parameters with the trial solutions
    is_self_consistent = np.all(np.array([check(phi, Phi) for phi, Phi in zip(phi_s, Phi_s)]))

    avg_error = np.sum(np.absolute(phi_s - Phi_s)) / phi_s.shape[0]

    return is_self_consistent, Phi_s, avg_error, d_hex_min, v_hex_min


def load_params(file):
    with open(file) as fp:
        return json.load(fp)


def dump_result(Psi_s, Ns, params):
    path = datetime.today().strftime("%Y_%m_%d_%H:%M:%S")
    os.makedirs(path)
    os.chdir(path)
    np.save("Psi_s.npy", Psi_s)
    np.save("Ns.npy", Ns)
    sio.savemat("result.mat", {"Psi1up": Psi_s[0], "Psi1dn": Psi_s[1], "Psi2up": Psi_s[2],
                               "Psi2dn": Psi_s[3], "Psi1updn": Psi_s[14], "Psi12up": Psi_s[12],
                               "Psi12dn": Psi_s[13], "Psi1upanddn": Psi_s[18]})
    with open("params.json", 'w') as fp:
        json.dump(params, fp)

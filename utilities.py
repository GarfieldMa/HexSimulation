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

    ret = t * (np.conj(psi_up_s_shifted).dot(up_a_terms)
               + psi_up_s_shifted.dot(up_adg_terms)
               + np.conj(psi_dn_s_shifted).dot(dn_a_terms)
               + psi_dn_s_shifted.dot(dn_adg_terms))
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
    try:
        d_hex, vec_hex = sparse.linalg.eigs(h_hexa, which='SR')
    except sparse.linalg.ArpackNoConvergence:
        d_hex = sparse.linalg.ArpackNoConvergence.eigenvalues
        vec_hex = sparse.linalg.ArpackNoConvergence.eigenvectors
    _, v_hex_min = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

    v_hex_min = sparse.csr_matrix(v_hex_min).transpose()
    Phi_s = np.array([(v_hex_min.getH().dot(b.dot(v_hex_min))).data[0] for b in hexagon_mf_operators])
    phi_s = psi_s

    # value difference for designated order parameters with the trial solutions
    is_self_consistent = np.any(np.greater(np.repeat(err, 12), np.absolute(Phi_s-phi_s)))

    return is_self_consistent, Phi_s, v_hex_min


def load_params(file):
    with open(file) as fp:
        return json.load(fp)


def dump_result(Psi_s, Ns, tA, Ma, params):
    path = datetime.today().strftime("%Y_%m_%d_%H:%M:%S")
    os.makedirs(path)
    os.chdir(path)
    np.save("Psi_s.npy", Psi_s)
    np.save("Ns.npy", Ns)
    sio.savemat("result.mat", {"Psi1up": Psi_s[0], "Psi1dn": Psi_s[1], "Psi2up": Psi_s[2],
                               "Psi2dn": Psi_s[3], "Psi1updn": Psi_s[14], "Psi12up": Psi_s[12],
                               "Psi12dn": Psi_s[13], "Psi1upanddn": Psi_s[18], "tA": tA, "Ma": Ma})
    with open("params.json", 'w') as fp:
        json.dump(params, fp)

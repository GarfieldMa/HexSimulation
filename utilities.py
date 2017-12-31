import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import json
import os

from datetime import datetime


def calc_h_hexa(g, t, mu, psi_s, uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, dig_h, ts):

    # t = 0.12

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
    ret += (t * t_term + u_term + v_term + mu * mu_term + uab_term + g * g_term)
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
        d_hex, vec_hex = sparse.linalg.eigsh(h_hexa, which='SA', k=1)
    except sparse.linalg.ArpackNoConvergence:
        return False, None, None

    v_hex_min = vec_hex[:, 0]

    v_hex_min = sparse.csr_matrix(v_hex_min).transpose()
    Phi_s = np.array([(v_hex_min.getH().dot(b.dot(v_hex_min))).data[0] for b in hexagon_mf_operators])
    phi_s = psi_s

    # value difference for designated order parameters with the trial solutions
    is_self_consistent = np.any(np.greater(np.repeat(err, 12), np.absolute(Phi_s-phi_s)))

    return is_self_consistent, Phi_s, v_hex_min


def load_params(file):
    with open(file) as fp:
        return json.load(fp)


def dump_result(Psi_s, Ns, Nsquare_s, NaN, gA, Ma, EVals, EVecs, params):
    base = os.getcwd()
    path = datetime.today().strftime("%Y_%m_%d_%H%M%S")
    os.makedirs(path)
    os.chdir(path)
    np.save("Psis.npy", Psi_s)
    np.save("Ns.npy", Ns)
    np.save("Nsquares.npy", Nsquare_s)
    sio.savemat("Psis.mat", {"Psi1up": Psi_s[0], "Psi1dn": Psi_s[1], "Psi2up": Psi_s[2], "Psi2dn": Psi_s[3],
                             "Psi3up": Psi_s[4], "Psi3dn": Psi_s[5], "Psi4up": Psi_s[6], "Psi4dn": Psi_s[7],
                             "Psi5up": Psi_s[8], "Psi5dn": Psi_s[9], "Psi6up": Psi_s[10], "Psi6dn": Psi_s[11]})

    sio.savemat("Ns.mat", {"Ns1up": Ns[0], "Ns1dn": Ns[1], "Ns2up": Ns[2], "Ns2dn": Ns[3],
                           "Ns3up": Ns[4], "Ns3dn": Ns[5], "Ns4up": Ns[6], "Ns4dn": Ns[7],
                           "Ns5up": Ns[8], "Ns5dn": Ns[9], "Ns6up": Ns[10], "Ns6dn": Ns[11]})

    sio.savemat("Nsquares.mat", {"N1squareup": Nsquare_s[0], "N1squaredn": Nsquare_s[1], "N2squareup": Nsquare_s[2], "N2squaredn": Nsquare_s[3],
                                 "N3squareup": Nsquare_s[4], "N3squaredn": Nsquare_s[5], "N4squareup": Nsquare_s[6], "N4squaredn": Nsquare_s[7],
                                 "N5squareup": Nsquare_s[8], "N5squaredn": Nsquare_s[9], "N6squareup": Nsquare_s[10], "N6squaredn": Nsquare_s[11]})

    # sio.savemat("axis.mat", {"NaN": NaN, "EVals": EVals, 'EVecs': EVecs, "gA": gA, "Ma": Ma})

    sio.savemat("axis.mat", {"NaN": NaN, "gA": gA, "Ma": Ma})

    with open("params.json", 'w') as fp:
        json.dump(params, fp)

    os.chdir(base)

import numpy as np


def calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h):
    psi_up_s = psi_s[::2]

    t_psi_var = t * (psi_s[z])
    return np.eye(1)


def update(h_hexa, hexagon_mf_operators, psi_s, err):
    d_hex, vec_hex = np.linalg.eig(h_hexa)
    _, v_hex_min = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

    Phi_s = np.array([v_hex_min.dot(b.dot(v_hex_min)) for b in hexagon_mf_operators])
    phi_s = psi_s

    # value difference for designated order parameters with the trial solutions
    # TODO: should also check angles
    is_self_consistent = np.all(np.array([abs(abs(phi) - abs(Phi)) <= err for phi, Phi in zip(phi_s, Phi_s)]))

    return is_self_consistent, Phi_s

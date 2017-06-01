import numpy as np
import cmath


def calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts):
    psi_up_s = psi_s[::2]
    psi_up_s_shifted = np.roll(psi_up_s, 3)
    psi_dn_s = psi_s[1::2]
    psi_dn_s_shifted = np.roll(psi_dn_s, 3)

    up_a_terms = var_terms[::4]
    up_adg_terms = var_terms[1::4]
    dn_a_terms = var_terms[2::4]
    dn_adg_terms = var_terms[3::4]

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
        r1, t1 = cmath.polar(q)
        r2, t2 = cmath.polar(q)
        check_phase = (abs(r1 - r2) <= err)
        check_angle = (abs(t1 - t2) <= err * 0.1 or (cmath.pi <= abs(t1 - t2) <= (err * 0.1 + cmath.pi)))
        return check_phase and check_angle

    d_hex, vec_hex = np.linalg.eig(h_hexa)
    _, v_hex_min = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

    Phi_s = np.array([v_hex_min.dot(b.dot(v_hex_min)) for b in hexagon_mf_operators])
    phi_s = psi_s

    # value difference for designated order parameters with the trial solutions
    # TODO: should also check angles
    is_self_consistent = np.all(np.array([check(phi, Phi) for phi, Phi in zip(phi_s, Phi_s)]))

    return is_self_consistent, Phi_s

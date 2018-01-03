import numpy as np
import scipy.sparse as sparse
from time import time

from utilities import calc_h_hexa, update


def iterate(k, j, g, wall_time, hexagon_mf_operators,
            ts, Ma, uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, t,
            dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs, err):
    t_begin = time()
    mu = Ma.flat[j]
    # initial d_hex_min is the minimum of eigenvalue
    d_hex_min, v_hex_min, vec_hex, d_hex = 1.0e5, None, None, None
    phi_s = None

    # t_init_begin = time()
    for lp in range(0, len(Pr)):
        psi_s = np.repeat(Pr.flat[lp], 12)

        # import the 6 single-site mean-field Hamiltonians for a Honeycomb lattice
        #  with two species of Pseudospins
        h_hexa = calc_h_hexa(g, t, mu, psi_s, uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, dig_h, ts)
        # solve the Hamilton with Eigenvectors and Eigenvalues
        # python returns array of Eigenvalues and normalized Eigenvectors
        try:
            d_hex, vec_hex = sparse.linalg.eigsh(h_hexa, which='SA', k=1)
            d_hex0, v_hex0 = d_hex[0], vec_hex[:, 0]
            v_hex0 /= np.linalg.norm(v_hex0)
        except sparse.linalg.ArpackNoConvergence:
            continue

        # find phi1up(down)---the trial solution corresponding to the lowest eigenvalues of Hsite
        if d_hex0 < d_hex_min:
            d_hex_min, v_hex_min = d_hex0, v_hex0
            phi_s = psi_s

    # Values of Order parameters corresponding to the trial solution of ground state above
    # # value difference for designated order parameters with the trial solutions
    is_self_consistent, Phi_s, v_hex_min = update(h_hexa, hexagon_mf_operators, phi_s, err)

    for lp in range(0, wall_time):
        if is_self_consistent or Phi_s is None:
            break
        else:
            psi_s = Phi_s
            h_hexa = calc_h_hexa(g, t, mu, psi_s, uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, dig_h, ts)
            is_self_consistent, Phi_s, v_hex_min = update(h_hexa, hexagon_mf_operators, psi_s, err)

    if not is_self_consistent:
        print(f"    {k}, {j} iteration fail to converge", flush=True)
        NaN[j, k] = np.nan
        # Phi_s[2] = np.nan

    if Phi_s is not None:
        evals, evecs = sparse.linalg.eigsh(h_hexa, which='SA', k=10)
        args = np.argsort(evals)
        EVals[j, k] = evals[args]
        EVecs[j, k] = evecs[:, args].T

        # save the final optimal value of both order parameters£¬also save the
        # corresponding state eigenvector
        for i in range(0, 12):
            Psi_s[i][j, k] = Phi_s[i]

        # if not is_self_consistent:
        #     Psi_s[2][j, k] = np.nan

        Psi_s[12][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[0].getH().dot(hexagon_mf_operators[2].dot(v_hex_min)))).data[0]
        Psi_s[13][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[1].getH().dot(hexagon_mf_operators[3].dot(v_hex_min)))).data[0]
        Psi_s[14][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[0].getH().dot(hexagon_mf_operators[1].dot(v_hex_min)))).data[0]
        Psi_s[15][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[2].getH().dot(hexagon_mf_operators[3].dot(v_hex_min)))).data[0]
        Psi_s[16][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[0].getH().dot(hexagon_mf_operators[3].dot(v_hex_min)))).data[0]
        Psi_s[17][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[1].getH().dot(hexagon_mf_operators[2].dot(v_hex_min)))).data[0]
        Psi_s[18][j, k] = (v_hex_min.getH().dot((hexagon_mf_operators[0] + hexagon_mf_operators[1]).dot(v_hex_min))).data[0]
        Psi_s[19][j, k] = (v_hex_min.getH().dot((hexagon_mf_operators[2] + hexagon_mf_operators[3]).dot(v_hex_min))).data[0]

        for i in range(0, 12):
            Ns[i][j, k] = (v_hex_min.getH().dot(hexagon_mf_operators[i].getH().dot(hexagon_mf_operators[i].dot(v_hex_min)))).data[0]
        for i in range(0, 12):
            tmp = hexagon_mf_operators[i].getH().dot(hexagon_mf_operators[i])
            Nsquare_s[i][j, k] = (v_hex_min.getH().dot(tmp.dot(tmp.dot(v_hex_min)))).data[0]
    else:
        for i in range(0, 20):
            Psi_s[i][j, k] = np.nan
        for i in range(0, 4):
            Ns[i][j, k] = np.nan
        for i in range(4, 8):
            Ns[i][j, k] = np.nan
    print(f"{k}, {j} iteration finished in {time()-t_begin:.4} seconds with Psi1up{j,k}={Psi_s[0][j, k]}", flush=True)
    return Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs


def solves_part(hexagon_mf_operators,
                start_t, stop_t, ts, Ma,
                uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, t,
                dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs, err, wall_time):
    start, stop = len(start_t), len(stop_t)
    for k in range(start, start + stop):
        # set hopping parameter
        g = stop_t.flat[k - start]

        for j in range(0, len(Ma)):
            Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs = iterate(k, j, g, wall_time, hexagon_mf_operators,
                                                              ts, Ma, uab_term, u_term, v_term, mu_term, t_term, g_term,
                                                              var_terms, t,
                                                              dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs, err)

    return Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs


def solves(hexagon_mf_operators,
           g_a, g_b, ts, Ma,
           uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, t,
           dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs,
           err, wall_time):
    t_begin = time()
    print("Simulation begin!", flush=True)
    Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs= solves_part(hexagon_mf_operators,
                                                         [], g_a, ts, Ma,
                                                         uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, t,
                                                         dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs, err, wall_time)
    Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs = solves_part(hexagon_mf_operators,
                                                          g_a, g_b, ts, Ma,
                                                          uab_term, u_term, v_term, mu_term, t_term, g_term, var_terms, t,
                                                          dig_h, Pr, Psi_s, Ns, Nsquare_s, NaN, EVals, EVecs, err, wall_time)
    print(f"Simulation completed within {time()-t_begin:.4} seconds", flush=True)
    return {'Psi_s': Psi_s, 'Ns': Ns, 'Nsquare_s': Nsquare_s, 'NaN': NaN, 'EVals': EVals, 'EVecs': EVecs}


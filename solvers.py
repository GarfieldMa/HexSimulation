import numpy as np
from time import clock
from functools import partial
from multiprocessing import Pool

from utilities import calc_h_hexa, update


def iterate(j, k, t, wall_time,
            nmax, hexagon_mf_bases, hexagon_mf_operators,
            start_t, stop_t, tA, ts, Ma,
            u_term, v_term, mu_term, t_term, var_terms,
            dig_h, toc1, toc2, T1, T2,
            Pr, Psi_s, Ns, err):
    t_begin = clock()
    print(f"{k}, {j} iteration begin!", flush=True)
    mu = Ma.flat[j]
    # initial d_hex_min is the minimum of eigenvalue
    d_hex_min, v_hex_min = 1.0e5, None
    phi_s = None

    for lp in range(0, len(Pr)):
        psi_s = np.repeat(Pr.flat[lp], 12)

        # import the 6 single-site mean-field Hamiltonians for a Honeycomb lattice
        #  with two species of Pseudospins
        h_hexa = calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts)

        # solve the Hamilton with Eigenvectors and Eigenvalues
        # python returns array of Eigenvalues and normalized Eigenvectors
        d_hex, vec_hex = np.linalg.eig(h_hexa)
        d_hex0, v_hex0 = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

        # find phi1up(down)---the trial solution corresponding to the lowest eigenvalues of Hsite
        if d_hex0 < d_hex_min:
            d_hex_min, v_hex_min = d_hex0, v_hex0
            phi_s = psi_s.copy()

    # Values of Order parameters corresponding to the trial solution of ground state above
    # # value difference for designated order parameters with the trial solutions
    is_self_consistent, Phi_s = update(h_hexa, hexagon_mf_operators, psi_s, err)

    for _ in range(0, wall_time):
        if is_self_consistent:
            break
        else:
            psi_s = Phi_s
            h_hexa = calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts)
            is_self_consistent, Phi_s = update(h_hexa, hexagon_mf_operators, psi_s, err)

    if not is_self_consistent:
        print(f"{k}, {j} iteration fail to converge", flush=True)
        Phi_s[2] = np.nan

    # save the final optimal value of both order parameters£¬also save the
    # corresponding state eigenvector
    # ret = []
    for i in range(0, 4):
        # ret.append(Phi_s[i])
        Psi_s[i][j, k] = Phi_s[i]
    # check j, k
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[2].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[1].T.dot(hexagon_mf_operators[3].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[1].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[2].T.dot(hexagon_mf_operators[3].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[3].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot(hexagon_mf_operators[1].T.dot(hexagon_mf_operators[2].dot(v_hex_min))))
    # ret.append(v_hex_min.T.dot((hexagon_mf_operators[0] + hexagon_mf_operators[1]).dot(v_hex_min)))
    # ret.append(v_hex_min.T.dot((hexagon_mf_operators[2] + hexagon_mf_operators[3]).dot(v_hex_min)))
    Psi_s[12][j, k] = v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[2].dot(v_hex_min)))
    Psi_s[13][j, k] = v_hex_min.T.dot(hexagon_mf_operators[1].T.dot(hexagon_mf_operators[3].dot(v_hex_min)))
    Psi_s[14][j, k] = v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[1].dot(v_hex_min)))
    Psi_s[15][j, k] = v_hex_min.T.dot(hexagon_mf_operators[2].T.dot(hexagon_mf_operators[3].dot(v_hex_min)))
    Psi_s[16][j, k] = v_hex_min.T.dot(hexagon_mf_operators[0].T.dot(hexagon_mf_operators[3].dot(v_hex_min)))
    Psi_s[17][j, k] = v_hex_min.T.dot(hexagon_mf_operators[1].T.dot(hexagon_mf_operators[2].dot(v_hex_min)))
    Psi_s[18][j, k] = v_hex_min.T.dot((hexagon_mf_operators[0] + hexagon_mf_operators[1]).dot(v_hex_min))
    Psi_s[19][j, k] = v_hex_min.T.dot((hexagon_mf_operators[2] + hexagon_mf_operators[3]).dot(v_hex_min))

    for i in range(0, 4):
        # ret.append(v_hex_min.T.dot(hexagon_mf_operators[i].T.dot(hexagon_mf_operators[i].dot(v_hex_min))))
        Ns[i][j, k] = v_hex_min.T.dot(hexagon_mf_operators[i].T.dot(hexagon_mf_operators[i].dot(v_hex_min)))
    for i in range(4, 8):
        tmp = hexagon_mf_operators[i].T.dot(hexagon_mf_operators[i])
        # ret.append(v_hex_min.T.dot(tmp.dot(tmp.dot(v_hex_min))))
        Ns[i][j, k] = v_hex_min.T.dot(tmp.dot(tmp.dot(v_hex_min)))
    print(f"{j}, {k} iteration finished in {clock()-t_begin} seconds", flush=True)
    # return j, k, ret
    return Psi_s, Ns


def solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                start_t, stop_t, tA, ts, Ma,
                u_term, v_term, mu_term, t_term, var_terms,
                dig_h, toc1, toc2, T1, T2,
                Pr, Psi_s, Ns, err):
    wall_time = 100
    start, stop = len(start_t), len(stop_t)
    for k in range(start, stop):

        # set hopping parameter
        t = stop_t.flat[k - start]

        # iters_rets = Pool().map(partial(iterate, k=k, t=t, wall_time=wall_time,
        #                                 nmax=nmax, hexagon_mf_bases=hexagon_mf_bases, hexagon_mf_operators=hexagon_mf_operators,
        #                                 start_t=start_t, stop_t=stop_t, tA=tA, ts=ts, Ma=Ma,
        #                                 u_term=u_term, v_term=v_term, mu_term=mu_term, t_term=t_term, var_terms=var_terms,
        #                                 dig_h=dig_h, toc1=toc1, toc2=toc2, T1=T1, T2=T2,
        #                                 Pr=Pr, Psi_s=Psi_s, Ns=Ns, err=err), range(0, len(Ma)))
        # # iters_rets = list(iters_rets)

        for j in range(0, len(Ma)):
            Psi_s, Ns = iterate(j, k, t, wall_time,
                                nmax, hexagon_mf_bases, hexagon_mf_operators,
                                start_t, stop_t, tA, ts, Ma,
                                u_term, v_term, mu_term, t_term, var_terms,
                                dig_h, toc1, toc2, T1, T2,
                                Pr, Psi_s, Ns, err)

    return Psi_s, Ns


def solves(nmax, hexagon_mf_bases, hexagon_mf_operators,
           t_a, t_b, tA, ts, Ma,
           u_term, v_term, mu_term, t_term, var_terms,
           dig_h, toc1, toc2, T1, T2,
           Pr, Psi_s, Ns, err):

    Psi_s, Ns = solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                            [], t_a, tA, ts, Ma,
                            u_term, v_term, mu_term, t_term, var_terms,
                            dig_h, toc1, toc2, T1, T2,
                            Pr, Psi_s, Ns, err)
    Psi_s, Ns = solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                            t_a, t_b, tA, ts, Ma,
                            u_term, v_term, mu_term, t_term, var_terms,
                            dig_h, toc1, toc2, T1, T2,
                            Pr, Psi_s, Ns, err)
    return Psi_s, Ns


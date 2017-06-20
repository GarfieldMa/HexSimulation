import numpy as np
import scipy.sparse as sparse
from time import time
from functools import partial
from multiprocessing import Pool

from utilities import calc_h_hexa, update


def iterate(k, j, t, wall_time,
            nmax, hexagon_mf_bases, hexagon_mf_operators,
            start_t, stop_t, tA, ts, Ma,
            u_term, v_term, mu_term, t_term, var_terms,
            dig_h, toc1, toc2, T1, T2,
            Pr, Psi_s, Ns, err):
    t_begin = time()
    print(f"{k}, {j} iteration begin!", flush=True)
    mu = Ma.flat[j]
    # initial d_hex_min is the minimum of eigenvalue
    d_hex_min, v_hex_min = 1.0e5, None
    phi_s = None

    t_init_begin = time()
    for lp in range(0, len(Pr)):
        psi_s = np.repeat(Pr.flat[lp], 12)

        # import the 6 single-site mean-field Hamiltonians for a Honeycomb lattice
        #  with two species of Pseudospins
        h_hexa = calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts)

        # solve the Hamilton with Eigenvectors and Eigenvalues
        # python returns array of Eigenvalues and normalized Eigenvectors
        # t_svd_begin = time()
        # d_hex, vec_hex = sparse.linalg.eigsh(h_hexa, which='SA')
        d_hex, vec_hex = sparse.linalg.eigs(h_hexa, which='SR')
        # d_hex, vec_hex = np.linalg.eig(h_hexa)
        # print(f"            solve eig within {time()-t_svd_begin} seconds")
        d_hex0, v_hex0 = min(zip(d_hex, vec_hex.T), key=lambda x: x[0])

        # find phi1up(down)---the trial solution corresponding to the lowest eigenvalues of Hsite
        # if d_hex0 < d_hex_min and abs(d_hex0 - d_hex_min) > 0.01:
        if d_hex0 < d_hex_min:
            # due to the precision of floating number,
            # print(f"    update with d_hex_min={d_hex0} and v_hex_min={v_hex0}", flush=True)
            d_hex_min, v_hex_min = d_hex0, v_hex0
            phi_s = psi_s

    # Values of Order parameters corresponding to the trial solution of ground state above
    # # value difference for designated order parameters with the trial solutions
    is_self_consistent, Phi_s, avg_err, d_hex_min, v_hex_min = update(h_hexa, hexagon_mf_operators, phi_s, err)
    print(f"    Initialization Complete within {time()-t_init_begin} seconds, d_hex_min={d_hex_min}, avg_err={avg_err}", flush=True)

    for lp in range(0, wall_time):
        t_lp_begin = time()
        print(f"    {lp}-th loop", end=', ', flush=True)
        if is_self_consistent:
            print(f"    complete within {time()-t_lp_begin} seconds")
            break
        else:
            psi_s = Phi_s
            h_hexa = calc_h_hexa(t, mu, psi_s, u_term, v_term, mu_term, t_term, var_terms, dig_h, ts)
            is_self_consistent, Phi_s, avg_err, d_hex_min, v_hex_min = update(h_hexa, hexagon_mf_operators, psi_s, err)
            print(f"    complete within {time()-t_lp_begin} seconds, avg_err={avg_err}, d_hex_min={d_hex_min}")
            # print(f"    hexa={h_hexa}, is_self_consistent={is_self_consistent}", flush=True)

    if not is_self_consistent:
        print(f"    {k}, {j} iteration fail to converge", flush=True)
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
    # print(f"    Psi1up{j,k}={Psi_s[0][j, k]}", flush=True)
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
    print(f"{k}, {j} iteration finished in {time()-t_begin} seconds with Psi1up{j,k}={Psi_s[0][j, k]}", flush=True)
    # return j, k, ret
    return Psi_s, Ns


def solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                start_t, stop_t, tA, ts, Ma,
                u_term, v_term, mu_term, t_term, var_terms,
                dig_h, toc1, toc2, T1, T2,
                Pr, Psi_s, Ns, err, wall_time):
    # wall_time = 50
    start, stop = len(start_t), len(stop_t)
    for k in range(start, start + stop):

        # set hopping parameter
        t = stop_t.flat[k - start]
        # print(f"t={t}")
        # iters_rets = Pool().map(partial(iterate, k=k, t=t, wall_time=wall_time,
        #                                 nmax=nmax, hexagon_mf_bases=hexagon_mf_bases, hexagon_mf_operators=hexagon_mf_operators,
        #                                 start_t=start_t, stop_t=stop_t, tA=tA, ts=ts, Ma=Ma,
        #                                 u_term=u_term, v_term=v_term, mu_term=mu_term, t_term=t_term, var_terms=var_terms,
        #                                 dig_h=dig_h, toc1=toc1, toc2=toc2, T1=T1, T2=T2,
        #                                 Pr=Pr, Psi_s=Psi_s, Ns=Ns, err=err), range(0, len(Ma)))
        # # iters_rets = list(iters_rets)

        for j in range(0, len(Ma)):
            Psi_s, Ns = iterate(k, j, t, wall_time,
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
           Pr, Psi_s, Ns, err, wall_time):
    # print("part one")
    Psi_s, Ns = solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                            [], t_a, tA, ts, Ma,
                            u_term, v_term, mu_term, t_term, var_terms,
                            dig_h, toc1, toc2, T1, T2,
                            Pr, Psi_s, Ns, err, wall_time)
    # print("part two")
    Psi_s, Ns = solves_part(nmax, hexagon_mf_bases, hexagon_mf_operators,
                            t_a, t_b, tA, ts, Ma,
                            u_term, v_term, mu_term, t_term, var_terms,
                            dig_h, toc1, toc2, T1, T2,
                            Pr, Psi_s, Ns, err, wall_time)
    return Psi_s, Ns


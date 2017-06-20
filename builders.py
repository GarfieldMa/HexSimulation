import numpy as np
import cmath
import scipy.io as sio
from time import time


def build_hexagon_mf_basis(nmax):

    # dimension of the basis state, as well as the Hilbert space
    nn = (nmax + 1) ** 12

    # initialize matrices
    # k1up, k1down, k2up, k2down ... k6up, k6down
    hexagon_mf_bases = np.array([np.zeros((nn, 1), dtype=complex) for _ in range(0, 12)], dtype=complex)
    # TODO: should be optimized later
    count = 0
    for i0 in range(0, nmax + 1):
        for i1 in range(0, nmax + 1):
            for i2 in range(0, nmax + 1):
                for i3 in range(0, nmax + 1):
                    for i4 in range(0, nmax + 1):
                        for i5 in range(0, nmax + 1):
                            for i6 in range(0, nmax + 1):
                                for i7 in range(0, nmax + 1):
                                    for i8 in range(0, nmax + 1):
                                        for i9 in range(0, nmax + 1):
                                            for i10 in range(0, nmax + 1):
                                                for i11 in range(0, nmax + 1):
                                                    hexagon_mf_bases[0].flat[count] = i0
                                                    hexagon_mf_bases[1].flat[count] = i1
                                                    hexagon_mf_bases[2].flat[count] = i2
                                                    hexagon_mf_bases[3].flat[count] = i3
                                                    hexagon_mf_bases[4].flat[count] = i4
                                                    hexagon_mf_bases[5].flat[count] = i5
                                                    hexagon_mf_bases[6].flat[count] = i6
                                                    hexagon_mf_bases[7].flat[count] = i7
                                                    hexagon_mf_bases[8].flat[count] = i8
                                                    hexagon_mf_bases[9].flat[count] = i9
                                                    hexagon_mf_bases[10].flat[count] = i10
                                                    hexagon_mf_bases[11].flat[count] = i11
                                                    count += 1
    return hexagon_mf_bases


def build_hexagon_mf_operator(hexagon_mf_bases):

    nn = max(hexagon_mf_bases[0].shape)
    # initialize matrices
    # b1up, b1down, b2up, b2down ... b6up, b6down
    hexagon_mf_operators = np.array([np.zeros((nn, nn), dtype=complex) for _ in range(0, 12)])

    # fill up operators
    for i in range(0, nn):

        # ks = [K1up(I), K1down(I) ...]
        ks = np.array([k.flat[i] for k in hexagon_mf_bases])

        for j in range(0, nn):

            # ls = [K1up(j), K1down(j) ...]
            ls = np.array([l.flat[j] for l in hexagon_mf_bases])
            cmp_results = np.where(np.not_equal(ks, ls))[0]
            if cmp_results.shape[0] == 1:
                idx = cmp_results[0]
                if ks[idx] == ls[idx] - 1:
                    hexagon_mf_operators[idx][i, j] += cmath.sqrt(ls[idx])

    return hexagon_mf_operators


def build_u_term_mf_cluster(hexagon_mf_bases, U0):
    base_l = max(hexagon_mf_bases[0].shape)
    u_term = np.zeros((base_l, base_l), dtype=complex)

    for i in range(0, base_l):
        # diagonal interaction part of Hamiltonian
        ns = [k.flat[i] for k in hexagon_mf_bases]
        # TODO:  should be optimized later
        u_term[i, i] = (U0 / 2 * (ns[0] * (ns[0] - 1) + ns[1] * (ns[1] - 1))
                        + U0 / 2 * (ns[2] * (ns[2] - 1) + ns[3] * (ns[3] - 1))
                        + U0 / 2 * (ns[4] * (ns[4] - 1) + ns[5] * (ns[5] - 1))
                        + U0 / 2 * (ns[6] * (ns[6] - 1) + ns[7] * (ns[7] - 1))
                        + U0 / 2 * (ns[8] * (ns[8] - 1) + ns[9] * (ns[9] - 1))
                        + U0 / 2 * (ns[10] * (ns[10] - 1) + ns[11] * (ns[11] - 1)))

    return u_term


def build_v_term_mf_cluster(hexagon_mf_bases, V0):
    base_l = max(hexagon_mf_bases[0].shape)
    v_term = np.zeros((base_l, base_l), dtype=complex)

    for i in range(0, base_l):
        # diagonal interaction part of Hamiltonian
        ns = [k.flat[i] for k in hexagon_mf_bases]
        # TODO:  should be optimized later
        v_term[i, i] = V0 * (ns[0] * ns[1] + ns[2] * ns[3] + ns[4] * ns[5] + ns[6] * ns[7]
                             + ns[8] * ns[9] + ns[10] * ns[11])

    return v_term


def build_mu_term_mf_cluster(hexagon_mf_bases, MU0):
    base_l = max(hexagon_mf_bases[0].shape)
    mu_term = np.zeros((base_l, base_l), dtype=complex)

    for i in range(0, base_l):
        # diagonal interaction part of Hamiltonian
        ns = [k.flat[i] for k in hexagon_mf_bases]
        # TODO:  should be optimized later
        mu_term[i, i] = -MU0 * sum(ns)

    return mu_term


def build_t_term_mf_cluster(hexagon_mf_bases, ts):
    base_l = max(hexagon_mf_bases[0].shape)
    t_term = np.zeros((base_l, base_l), dtype=complex)
    ts_shifted = [ts[2], ts[3], ts[0], ts[1], ts[4], ts[5], ts[2], ts[3], ts[0], ts[1], ts[4], ts[5]]
    for i in range(0, base_l):
        ks = np.array([k.flat[i] for k in hexagon_mf_bases])

        # off-diagonal kinetic hopping part of Hamiltonian
        for j in range(0, base_l):
            ls = np.array([l.flat[j] for l in hexagon_mf_bases])

            cmp_results = np.where(np.not_equal(ks, ls))[0]
            if cmp_results.shape[0] == 2:
                idx0, idx1 = cmp_results[0], cmp_results[1]
                if idx0 // 2 == 0 and idx1 // 2 == 5:
                    idx0, idx1 = idx1, idx0
                if idx1 - idx0 == 2 or idx1 - idx0 == -10:
                    if ks[idx0] == ls[idx0] + 1 and ks[idx1] == ls[idx1] - 1:
                        if (idx0 // 2) % 2:
                            t_term[i, j] -= ts_shifted[idx0] * cmath.sqrt(ks[idx0] * ls[idx1])
                        else:
                            t_term[i, j] -= np.conj(ts_shifted[idx0]) * cmath.sqrt(ks[idx0] * ls[idx1])
                    if ks[idx0] == ls[idx0] - 1 and ks[idx1] == ls[idx1] + 1:
                        if (idx0 // 2) % 2:
                            t_term[i, j] -= np.conj(ts_shifted[idx0]) * cmath.sqrt(ks[idx1] * ls[idx0])
                        else:
                            t_term[i, j] -= ts_shifted[idx0] * cmath.sqrt(ks[idx1] * ls[idx0])
            #     if abs(ks[idx] - ls[idx]) == 1:
            # spin-up intra-cluster tunneling terms terms ai^{dagger}aj
            # if ns[0] == ls[0] + 1 and ns[1] == ls[1] and ns[2] == ls[2] - 1 and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[2].conjugate() * cmath.sqrt(ns[0] * ls[2])
            # if ns[0] == ls[0] - 1 and ns[1] == ls[1] and ns[2] == ls[2] + 1 and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[2] * cmath.sqrt(ns[2] * ls[0])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] + 1 and ns[3] == ls[3] and\
            #     ns[4] == ls[4] - 1 and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[0] * cmath.sqrt(ns[2] * ls[4])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] - 1 and ns[3] == ls[3] and\
            #     ns[4] == ls[4] + 1 and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[0].conjugate() * cmath.sqrt(ns[4] * ls[2])
            #
            # if ns[0] == ls[0]-1 and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10]+1 and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[4] * cmath.sqrt(ns[10] * ls[0])
            # if ns[0] == ls[0]+1 and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10]-1 and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[4].conjugate() * cmath.sqrt(ns[0] * ls[10])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10]-1 and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[0] * cmath.sqrt(ns[8] * ls[10])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10]+1 and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[0].conjugate() * cmath.sqrt(ns[10] * ls[8])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6]+1 and ns[7] == ls[7] and\
            #         ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[2] * cmath.sqrt(ns[6] * ls[8])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6]-1 and ns[7] == ls[7] and\
            #         ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[2].conjugate() * cmath.sqrt(ns[8] * ls[6])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4]+1 and ns[5] == ls[5] and ns[6] == ls[6]-1 and ns[7] == ls[7] and\
            #         ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[4] * cmath.sqrt(ns[4] * ls[6])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4]-1 and ns[5] == ls[5] and ns[6] == ls[6]+1 and ns[7] == ls[7] and\
            #         ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[4].conjugate() * cmath.sqrt(ns[6] * ls[4])
            #
            # # spin-dn intra-cluster tunneling terms ai^{dagger}aj
            # if ns[0] == ls[0] and ns[1] == ls[1]+1 and ns[2] == ls[2] and ns[3] == ls[3]-1 and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[3].conjugate() * cmath.sqrt(ns[1] * ls[3])
            # if ns[0] == ls[0] and ns[1] == ls[1]-1 and ns[2] == ls[2] and ns[3] == ls[3]+1 and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[3] * cmath.sqrt(ns[3] * ls[1])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3]+1 and\
            #     ns[4] == ls[4] and ns[5] == ls[5]-1 and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[1] * cmath.sqrt(ns[3] * ls[5])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3]-1 and\
            #     ns[4] == ls[4] and ns[5] == ls[5]+1 and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[1].conjugate() * cmath.sqrt(ns[5] * ls[3])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1]-1 and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]+1:
            #     t_terms[i, j] -= ts[5] * cmath.sqrt(ns[11] * ls[1])
            # if ns[0] == ls[0] and ns[1] == ls[1]+1 and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]-1:
            #     t_terms[i, j] -= ts[5].conjugate() * cmath.sqrt(ns[1] * ls[11])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9]+1 and ns[10] == ls[10] and ns[11] == ls[11]-1:
            #     t_terms[i, j] -= ts[1].conjugate() * cmath.sqrt(ns[9] * ls[11])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
            #         ns[8] == ls[8] and ns[9] == ls[9]-1 and ns[10] == ls[10] and ns[11] == ls[11]+1:
            #     t_terms[i, j] -= ts[1] * cmath.sqrt(ns[11] * ls[9])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7]+1 and\
            #         ns[8] == ls[8] and ns[9] == ls[9]-1 and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[3] * cmath.sqrt(ns[7] * ls[9])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7]-1 and\
            #         ns[8] == ls[8] and ns[9] == ls[9]+1 and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[3].conjugate() * cmath.sqrt(ns[9] * ls[7])
            #
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4] and ns[5] == ls[5]+1 and ns[6] == ls[6] and ns[7] == ls[7]-1 and\
            #         ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[5].conjugate() * cmath.sqrt(ns[5] * ls[7])
            # if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
            #     ns[4] == ls[4]-1 and ns[5] == ls[5]-1 and ns[6] == ls[6] and ns[7] == ls[7]+1 and\
            #         ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
            #     t_terms[i, j] -= ts[5] * cmath.sqrt(ns[7] * ls[5])

    return t_term


def build_var_terms(hexagon_mf_bases, ts):
    base_l = max(hexagon_mf_bases[0].shape)
    # print(f"    base_l={base_l}")
    var_terms = np.array([np.zeros((base_l, base_l), dtype=complex) for _ in range(0, 24)])
    # reorder ts into t1up, t3up, t2up, t1up, t3up, t2up, t1dn, t3dn, t2dn, t1dn, t3dn, t2dn
    t_factors = [ts[0], ts[4], ts[2], ts[0], ts[4], ts[2], ts[1], ts[5], ts[3], ts[1], ts[5], ts[3]]
    # t_lp_begin = time()
    for i in range(0, base_l):
        ks = np.array([k.flat[i] for k in hexagon_mf_bases])

        for j in range(0, base_l):
            ls = np.array([l.flat[j] for l in hexagon_mf_bases])

            # compare k1up, l1up ... k6dn, l6dn
            cmp_results = np.where(np.not_equal(ks, ls))[0]
            if cmp_results.shape[0] == 1:
                idx = cmp_results[0]
                if abs(ks[idx] - ls[idx]) == 1:
                    # condition for up_a_term
                    if idx % 2 == 0 and ks[idx] == ls[idx] - 1:
                        if idx % 4 == 0:
                            var_terms[idx // 2][i, j] -= t_factors[idx // 2] * cmath.sqrt(ls[idx])
                        else:
                            var_terms[idx // 2][i, j] -= (t_factors[idx // 2].conj()) * cmath.sqrt(ls[idx])
                    # condition for up_adg_term
                    if idx % 2 == 0 and ks[idx] == ls[idx] + 1:
                        if idx % 4 == 0:
                            var_terms[(idx // 2) + 6][i, j] -= (t_factors[idx // 2].conj()) * cmath.sqrt(ks[idx])
                        else:
                            var_terms[(idx // 2) + 6][i, j] -= t_factors[idx // 2] * cmath.sqrt(ks[idx])
                    # condition for dn_a_term
                    if idx % 2 != 0 and ks[idx] == ls[idx] - 1:
                        if (idx - 1) % 4 == 0:
                            var_terms[(idx // 2) + 12][i, j] -= t_factors[(idx // 2) + 6] * cmath.sqrt(ls[idx])
                        else:
                            var_terms[(idx // 2) + 12][i, j] -= (t_factors[(idx // 2) + 6].conj()) * cmath.sqrt(ls[idx])
                    # condition for dn_adg_term
                    if idx % 2 != 0 and ks[idx] == ls[idx] + 1:
                        if (idx - 1) % 4 == 0:
                            var_terms[(idx // 2) + 18][i, j] -= (t_factors[(idx // 2) + 6].conj()) * cmath.sqrt(ks[idx])
                        else:
                            var_terms[(idx // 2) + 18][i, j] -= t_factors[(idx // 2) + 6] * cmath.sqrt(ks[idx])
        # if i % 100 == 0 and i != 0:
            # print(f"        {i}-th loop completed within {time()-t_lp_begin} seconds")
            # t_lp_begin = time()

    return var_terms


def create(name, func, params):
    try:
        print(f"Loading {name} ...", end=' ', flush=True)
        ret = np.load(f"var/{name}.npy")
        print("Done!", flush=True)
        return ret
    except IOError:
        print(f"{name} not found and now building {name} ...", end=' ', flush=True)
        ret = func(*params)
        print(f"saving to file ...", end=' ', flush=True)
        np.save(f"var/{name}.npy", ret)
        # sio.savemat(f"var/{name}.mat", {name: ret})
        print("Done!", flush=True)
        return ret


def builder(nmax, t_first, t_second,
            err, n1, n2, mu_range, ma, wall_time):
    hexagon_mf_bases = create("hexagon_mf_bases", func=build_hexagon_mf_basis, params=[nmax])
    hexagon_mf_operators = create("hexagon_mf_operators", func=build_hexagon_mf_operator, params=[hexagon_mf_bases])
    # EVHexmin = []
    # range setting of hopping strength
    # t_first, t_second = 0.2, 0.4
    # ta-part1,near phase transition boundary, need to be calculated more densely
    t_a = np.linspace(0, t_first, n1)
    # tb-part2
    t_b = np.linspace(t_first, t_second, n2)
    tA = np.array([*t_a, *t_b])

    # setting tunneling terms
    # phase winding factor W
    W = 2 * cmath.pi / 3
    t0 = 1 + 0j
    t1, t2, t3 = t0, t0 * cmath.exp(1j * W), t0 * cmath.exp(-1j * W)
    t1_up, t2_up, t3_up = t1, t2, t3
    t1_dn, t2_dn, t3_dn = t1_up.conjugate(), t2_up.conjugate(), t3_up.conjugate()
    ts = np.array([t1_up, t1_dn, t2_up, t2_dn, t3_up, t3_dn])

    # setting chemical potential
    mu0 = 1
    # the range of mu, chemical potential
    Ma = np.linspace(-0.5, mu_range, ma)
    len_ma = len(Ma)

    # setting on-site interactions
    # range of on-site of the two same pseudo-spin particles
    U, U0 = 1, 1
    # range of on-site interaction of the two different pseudo-spin particles, fix V first
    V, V0 = 0.25, 0.25

    # build Hamiltonian terms
    u_term = create("u_term", func=build_u_term_mf_cluster, params=[hexagon_mf_bases, U0])
    v_term = create("v_term", func=build_v_term_mf_cluster, params=[hexagon_mf_bases, V0])
    mu_term = create("mu_term", func=build_mu_term_mf_cluster, params=[hexagon_mf_bases, mu0])
    t_term = create("t_term", func=build_t_term_mf_cluster, params=[hexagon_mf_bases, ts])
    var_terms = create("var_terms", func=build_var_terms, params=[hexagon_mf_bases, ts])

    # build other vars
    print("Building Other Terms ...", end=' ', flush=True)
    dig_h = np.identity((nmax + 1) ** 12, dtype=complex)
    toc1, toc2 = np.zeros(len_ma, dtype=complex), np.zeros(len_ma, dtype=complex)
    T1, T2 = 0, 0
    # the range of order parameters trial solution, the trial OrderParameter is Complex with Pa(i,j)=Pr*exp(i*theta)
    Pr = np.linspace(0.01, cmath.sqrt(nmax), 10)
    # tA has been handled before
    # Psi1up, Psi1dn, Psi2up, Psi2dn ... Psi6up, Psi6dn
    Psi_s = np.array([np.zeros((len_ma, n1 + n2), dtype=complex) for _ in range(0, 20)])
    # N1up, ... N2dn, N1squareup, ... N2squaredn
    Ns = np.array([np.zeros((len_ma, n1 + n2), dtype=complex) for _ in range(0, 8)])

    print("Done!", flush=True)

    return (nmax, hexagon_mf_bases, hexagon_mf_operators,
            t_a, t_b, tA, ts, Ma,
            u_term, v_term, mu_term, t_term, var_terms,
            dig_h, toc1, toc2, T1, T2,
            Pr, Psi_s, Ns, err, wall_time)

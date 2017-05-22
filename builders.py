import numpy as np
from math import sqrt, pi, exp


def build_hexagon_mf_basis(nmax):

    # dimension of the basis state, as well as the Hilbert space
    nn = (nmax + 1) ** 12

    # initialize matrices
    # k1up, k1down, k2up, k2down ... k6up, k6down
    hexagon_mf_bases = [np.zeros((nn, 1), dtype=complex) for _ in range(0, 12)]
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
    hexagon_mf_operators = [np.zeros((nn, nn), dtype=complex) for _ in range(0, 12)]

    # fill up operators
    for i in range(0, nn):

        # ks = [K1up(I), K1down(I) ...]
        # TODO: might be optimized via list comprehension
        ks = list(map(lambda x: x.flat[i], hexagon_mf_operators))

        for j in range(0, nn):

            # ls = [K1up(j), K1down(j) ...]
            ls = list(map(lambda x: x.flat[j], hexagon_mf_operators))

            # TODO: should be optimized later
            # b1
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[0][i, j] += sqrt(ls[0])
            # b2
            if ks[0] == ls[0] and ks[1] == ls[1]-1 and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[1][i, j] += sqrt(ls[1])
            # b3
            if ks[0] == ls[0] and ks[1] == ls[1] and ks[2] == ls[2]-1 and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[2][i, j] += sqrt(ls[2])
            # b4
            if ks[0] == ls[0] and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3]-1 and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[3][i, j] += sqrt(ls[3])
            # b5
            if ks[0] == ls[0] and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4]-1 and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[4][i, j] += sqrt(ls[4])
            # b6
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5]-1 and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[6][i, j] += sqrt(ls[6])
            # b7
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6]-1 and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[6][i, j] += sqrt(ls[6])
            # b8
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7]-1 and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[7][i, j] += sqrt(ls[7])
            # b9
            if ks[0] == ls[0] and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8]-1 and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[8][i, j] += sqrt(ls[8])
            # b10
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9]-1 and ks[10] == ls[10] and ks[11] == ls[11]:
                hexagon_mf_operators[9][i, j] += sqrt(ls[9])
            # b11
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10]-1 and ks[11] == ls[11]:
                hexagon_mf_operators[10][i, j] += sqrt(ls[10])
            # b12
            if ks[0] == ls[0]-1 and ks[1] == ls[1] and ks[2] == ls[2] and ks[3] == ls[3] and\
                ks[4] == ls[4] and ks[5] == ls[5] and ks[6] == ls[6] and ks[7] == ls[7] and\
                    ks[8] == ls[8] and ks[9] == ls[9] and ks[10] == ls[10] and ks[11] == ls[11]-1:
                hexagon_mf_operators[11][i, j] += sqrt(ls[11])

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
    t_terms = np.zeros((base_l, base_l), dtype=complex)

    for i in range(0, base_l):
        ns = [k.flat[i] for k in hexagon_mf_bases]

        # off-diagonal kinetic hopping part of Hamiltonian
        for j in range(0, base_l):
            ls = [l.flat[i] for l in hexagon_mf_bases]

            # spin-up intra-cluster tunneling terms terms ai^{dagger}aj
            # TODO: should be optimized later
            if ns[0] == ls[0] + 1 and ns[1] == ls[1] and ns[2] == ls[2] - 1 and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[2].conjugate() * sqrt(ns[0] * ls[2])
            if ns[0] == ls[0] - 1 and ns[1] == ls[1] and ns[2] == ls[2] + 1 and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[2] * sqrt(ns[2] * ls[0])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] + 1 and ns[3] == ls[3] and\
                ns[4] == ls[4] - 1 and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[0] * sqrt(ns[2] * ls[4])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] - 1 and ns[3] == ls[3] and\
                ns[4] == ls[4] + 1 and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[0].conjugate() * sqrt(ns[4] * ls[2])

            if ns[0] == ls[0]-1 and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10]+1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[4] * sqrt(ns[10] * ls[0])
            if ns[0] == ls[0]+1 and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10]-1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[4].conjugate() * sqrt(ns[0] * ls[10])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10]-1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[0] * sqrt(ns[8] * ls[10])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10]+1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[0].conjugate() * sqrt(ns[10] * ls[8])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6]+1 and ns[7] == ls[7] and\
                    ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[2] * sqrt(ns[6] * ls[8])
            # TODO: check formula here
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6]-1 and ns[7] == ls[7] and\
                    ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10]+1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[2].conjugate() * sqrt(ns[8] * ls[6])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4]+1 and ns[5] == ls[5] and ns[6] == ls[6]-1 and ns[7] == ls[7] and\
                    ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[4] * sqrt(ns[4] * ls[6])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4]-1 and ns[5] == ls[5] and ns[6] == ls[6]+1 and ns[7] == ls[7] and\
                    ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[4].conjugate() * sqrt(ns[6] * ls[4])

            # spin-dn intra-cluster tunneling terms ai^{dagger}aj
            if ns[0] == ls[0] and ns[1] == ls[1]+1 and ns[2] == ls[2] and ns[3] == ls[3]-1 and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[3].conjugate() * sqrt(ns[1] * ls[3])
            if ns[0] == ls[0] and ns[1] == ls[1]-1 and ns[2] == ls[2] and ns[3] == ls[3]+1 and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[3] * sqrt(ns[3] * ls[1])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3]+1 and\
                ns[4] == ls[4] and ns[5] == ls[5]-1 and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[1] * sqrt(ns[3] * ls[5])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3]-1 and\
                ns[4] == ls[4] and ns[5] == ls[5]+1 and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[1].conjugate() * sqrt(ns[5] * ls[3])

            if ns[0] == ls[0] and ns[1] == ls[1]-1 and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]+1:
                t_terms[i, j] -= ts[5] * sqrt(ns[11] * ls[1])
            if ns[0] == ls[0] and ns[1] == ls[1]+1 and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]-1:
                t_terms[i, j] -= ts[5].conjugate() * sqrt(ns[1] * ls[11])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9]+1 and ns[10] == ls[10] and ns[11] == ls[11]-1:
                t_terms[i, j] -= ts[1].conjugate() * sqrt(ns[9] * ls[11])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7] and\
                    ns[8] == ls[8] and ns[9] == ls[9]-1 and ns[10] == ls[10] and ns[11] == ls[11]+1:
                t_terms[i, j] -= ts[1] * sqrt(ns[11] * ls[9])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7]+1 and\
                    ns[8] == ls[8] and ns[9] == ls[9]-1 and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[3] * sqrt(ns[7] * ls[9])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5] and ns[6] == ls[6] and ns[7] == ls[7]-1 and\
                    ns[8] == ls[8] and ns[9] == ls[9]+1 and ns[10] == ls[10]+1 and ns[11] == ls[11]:
                t_terms[i, j] -= ts[3].conjugate() * sqrt(ns[9] * ls[7])

            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4] and ns[5] == ls[5]+1 and ns[6] == ls[6] and ns[7] == ls[7]-1 and\
                    ns[8] == ls[8]+1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[5].conjugate() * sqrt(ns[5] * ls[7])
            if ns[0] == ls[0] and ns[1] == ls[1] and ns[2] == ls[2] and ns[3] == ls[3] and\
                ns[4] == ls[4]-1 and ns[5] == ls[5]-1 and ns[6] == ls[6] and ns[7] == ls[7]+1 and\
                    ns[8] == ls[8]-1 and ns[9] == ls[9] and ns[10] == ls[10] and ns[11] == ls[11]:
                t_terms[i, j] -= ts[5] * sqrt(ns[7] * ls[5])

    return t_terms


# build f1upa, f1upad, f1dna, f1dnad ...
def build_var_terms(hexagon_mf_bases, ts):
    base_l = max(hexagon_mf_bases[0].shape)
    var_terms = [np.zeros((base_l, base_l), dtype=complex) for _ in range(0, 24)]
    # reorder ts into t1up, t3up, t2up, t1up, t3up, t2up, t1dn, t3dn, t2dn, t1dn, t3dn, t2dn
    t_factors = [ts[0], ts[4], ts[2], ts[0], ts[4], ts[2], ts[1], ts[5], ts[3], ts[1], ts[5], ts[3]]

    for i in range(0, base_l):
        ks = [k.flat[i] for k in hexagon_mf_bases]

        for j in range(0, base_l):
            ls = [l.flat[i] for l in hexagon_mf_bases]

            # compare k1up, l1up ... k6dn, l6dn
            cmp_results = [x == y for x, y in zip(ks, ls)]
            if cmp_results.count(False) == 1:
                idx = cmp_results.index(False)
                if abs(ks[idx] - ls[idx]) == 1:
                    # condition for up_a_term
                    if idx % 2 == 0 and ks[idx] == ls[idx] - 1:
                        if idx % 4 == 0:
                            var_terms[idx // 2] -= t_factors[idx // 2] * sqrt(ls[idx])
                        else:
                            var_terms[idx // 2] -= np.conj(t_factors[idx // 2]) * sqrt(ls[idx])
                    # condition for up_adg_term
                    if idx % 2 == 0 and ks[idx] == ls[idx] + 1:
                        if idx % 4 == 0:
                            var_terms[(idx // 2) + 6] -= np.conj(t_factors[idx // 2]) * sqrt(ls[idx])
                        else:
                            var_terms[(idx // 2) + 6] -= t_factors[idx // 2] * sqrt(ls[idx])
                    # condition for dn_a_term
                    if idx % 2 != 0 and ks[idx] == ls[idx] - 1:
                        if (idx - 1) % 4 == 0:
                            var_terms[(idx // 2) + 12] -= t_factors[(idx // 2) + 6] * sqrt(ls[idx])
                        else:
                            var_terms[(idx // 2) + 12] -= np.conj(t_factors[(idx // 2) + 6]) * sqrt(ls[idx])
                    # condition for dn_adg_term
                    if idx % 2 != 0 and ks[idx] == ls[idx] - 1:
                        if (idx - 1) % 4 == 0:
                            var_terms[(idx // 2) + 18] -= np.conj(t_factors[(idx // 2) + 6]) * sqrt(ls[idx])
                        else:
                            var_terms[(idx // 2) + 18] -= t_factors[(idx // 2) + 6] * sqrt(ls[idx])

    return var_terms

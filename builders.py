import numpy as np


def build_hexagon_mf_basis(nmax):

    # dimension of the basis state, as well as the Hilbert space
    nn = (nmax + 1) ** 12

    # initialize matrices
    # k1up, k1down, k2up, k2down ... k6up, k6down
    hexagon_mf_bases = [np.zeros((nn, 1), dtype=complex) for _ in range(0, 12)]
    # should be optimized later
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

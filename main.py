import numpy as np
import cmath

from builders import *
from solvers import solves

if __name__ == '__main__':
    nmax = 1
    print("Building Hexagon MF Bases ...", end=' ', flush=True)
    hexagon_mf_bases = build_hexagon_mf_basis(nmax)
    print("Done!", flush=True)

    print("Building Hexagon MF Operators ...", end=' ', flush=True)
    hexagon_mf_operators = build_hexagon_mf_operator(hexagon_mf_bases)
    print("Done!", flush=True)

    EVHexmin = []
    # range setting of hopping strength
    t_first, t_second, n1, n2 = 0.2, 0.4, 20, 20
    # ta-part1,near phase transition boundary, need to be calculated more densely
    t_a = np.linspace(0, t_first, n1)
    # tb-part2
    t_b = np.linspace(t_first, t_second, n2)
    # TODO: possible overlapping
    tA = np.array([*t_a, *t_b])

    # setting tunneling terms
    # phase winding factor W
    W = 2 * pi / 3
    t0 = 1 + 0j
    t1, t2, t3 = t0, t0 * cmath.exp(1j * W), t0 * cmath.exp(-1j * W)
    t1_up, t2_up, t3_up = t1, t2, t3
    t1_dn, t2_dn, t3_dn = t1_up.conjugate(), t2_up.conjugate(), t3_up.conjugate()
    ts = [t1_up, t1_dn, t2_up, t2_dn, t3_up, t3_dn]

    # setting chemical potential
    mu0 = 1
    mu_range = 4
    # the range of mu, chemical potential
    Ma = np.linspace(-1, mu_range, 30)
    len_ma = len(Ma)

    # setting on-site interactions
    # range of on-site of the two same pseudo-spin particles
    U, U0 = 1, 1
    # range of on-site interaction of the two different pseudo-spin particles, fix V first
    V, V0 = 1, 1

    # build Hamiltonian terms
    print("Building Hamiltonian Terms ...", end=' ', flush=True)
    u_term = build_u_term_mf_cluster(hexagon_mf_bases, U0)
    v_term = build_v_term_mf_cluster(hexagon_mf_bases, V0)
    mu_term = build_mu_term_mf_cluster(hexagon_mf_bases, mu0)
    t_term = build_t_term_mf_cluster(hexagon_mf_bases, ts)
    var_terms = build_var_terms(hexagon_mf_bases, ts)
    print("Done!", flush=True)

    # build other vars
    dig_h = np.identity((nmax + 1) ** 12, dtype=complex)
    toc1, toc2 = np.zeros(len_ma, dtype=complex), np.zeros(len_ma, dtype=complex)
    T1, T2 = 0, 0
    # the range of order parameters trial solution, the trial OrderParameter is Complex with Pa(i,j)=Pr*exp(i*theta)
    Pr = np.linspace(0.01, sqrt(nmax), 10)
    # tA has been handled before
    # Psi1up, Psi1dn, Psi2up, Psi2dn ... Psi6up, Psi6dn
    # , Psi12up, Psi12dn, Psi1updn, Psi2updn, Psi12updn, Psi12dnup,Psi1upanddn, Psi2upanddn
    psi_s = [np.zeros((len_ma, n1 + n2), dtype=complex) for _ in range(0, 20)]
    # N1up, ... N2dn, N1squareup, ... N2squaredn
    Ns = [np.zeros((len_ma, n1 + n2), dtype=complex) for _ in range(0, 8)]

    # set error for self-consistency
    err = 1.0e-5

    # solve self-consistency problem
    solves(nmax, hexagon_mf_bases, hexagon_mf_operators,
           tA, ts, Ma,
           u_term, v_term, mu_term, t_term, var_terms,
           dig_h, toc1, toc2, T1, T2,
           Pr, psi_s, Ns, err)




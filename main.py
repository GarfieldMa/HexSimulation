import numpy as np
import scipy.io as sio

from builders import builder
from solvers import solves

from scipy.misc import toimage

if __name__ == '__main__':
    (nmax, hexagon_mf_bases, hexagon_mf_operators,
     t_a, t_b, tA, ts, Ma,
     u_term, v_term, mu_term, t_term, var_terms,
     dig_h, toc1, toc2, T1, T2,
     Pr, Psi_s, Ns, err, wall_time) = builder(nmax=1, t_first=0.2, t_second=0.4,
                                              err=1.0e-5, n1=15, n2=8, mu_range=1.5, ma=12, wall_time=10)

    Psi_s, Ns = solves(nmax, hexagon_mf_bases, hexagon_mf_operators,
                       t_a, t_b, tA, ts, Ma,
                       u_term, v_term, mu_term, t_term, var_terms,
                       dig_h, toc1, toc2, T1, T2,
                       Pr, Psi_s, Ns, err, wall_time)

    np.save("result", Psi_s)
    sio.savemat("result.mat", {"Psi1up": Psi_s[0], "Psi1dn": Psi_s[1], "Psi2up": Psi_s[2],
                               "Psi2dn": Psi_s[3], "Psi1updn": Psi_s[14], "Psi12up": Psi_s[12],
                               "Psi12dn": Psi_s[13], "Psi1upanddn": Psi_s[18]})
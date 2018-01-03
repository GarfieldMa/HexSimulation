import numpy as np
import scipy.sparse as sparse
from time import time
import cmath
import os
from abc import ABCMeta, abstractmethod


class Builder(metaclass=ABCMeta):

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.term = None
        self.is_prepared = True

    def get_term(self):
        if self.is_prepared:
            return self.term.copy()
        else:
            raise Exception(f"{self.name} is not prepared")


class CachedBuilder(Builder):

    cached_path = "var"

    def __init__(self, name, shape):
        Builder.__init__(self, name, shape)
        self.coefficient_mat = None
        self.bases_mat = None
        self.bases_mat_sparsity = False
        self.is_prepared = False

    def clear(self):
        self.bases_mat = None
        self.coefficient_mat = None

    @abstractmethod
    def _build_bases(self, **kwargs):
        pass

    def _build_coefficient(self, **kwargs):
        self.coefficient_mat = np.array([1])
        return self.coefficient_mat

    @abstractmethod
    def _combine(self):
        pass

    def _save(self):
        if self.bases_mat is not None:
            cur_dir = os.getcwd()
            if not os.path.exists(CachedBuilder.cached_path):
                os.makedirs(CachedBuilder.cached_path)
            os.chdir(CachedBuilder.cached_path)
            print(f"Saving {self.name} ...", flush=True, end=' ')
            (sparse.save_npz(f"{self.name}.npz", self.bases_mat) if self.bases_mat_sparsity else
             np.save(f"{self.name}.npy", self.bases_mat))
            os.chdir(cur_dir)
            print(f"Done", flush=True)
        else:
            raise Exception(f"{self.name} is not ready for save")

    def _load(self):
        cur_dir = os.getcwd()
        if not os.path.exists(CachedBuilder.cached_path):
            os.makedirs(CachedBuilder.cached_path)
        os.chdir(CachedBuilder.cached_path)
        try:
            print(f"Loading {self.name} ...", flush=True, end=' ')
            base_mat = (sparse.load_npz(f"{self.name}.npz") if self.bases_mat_sparsity else
                        np.load(f"{self.name}.npy"))
            os.chdir(cur_dir)
            print(f"Done", flush=True)
            return base_mat
        finally:
            os.chdir(cur_dir)

    def prepare_term(self, **kwargs):
        try:
            if self.bases_mat is None:
                self.bases_mat = self._load()
        except IOError:
            print(f"Fail to load {self.name}\nTry to build ... ", flush=True, end=' ')
            self.bases_mat = self._build_bases(**kwargs)
            print("Done", flush=True)
            self._save()
        finally:
            self.coefficient_mat = self._build_coefficient(**kwargs)
            if self.bases_mat is not None:
                self.term = self._combine()
                self.is_prepared = True
                return self
            else:
                raise Exception(f"{self.name}'s preparation failed")


class DiagCachedBuilder(CachedBuilder):

    def __init__(self, name, shape, coefficient):
        CachedBuilder.__init__(self, name, shape)
        self.coefficient = coefficient

    def _combine(self):
        ret = sparse.csr_matrix(np.diagflat(np.multiply(self.coefficient, self.bases_mat)))
        assert ret.shape == self.shape
        return ret

    @abstractmethod
    def _update(self, idx, ns):
        pass

    def _init_base(self):
        self.bases_mat = np.zeros(self.shape[0], dtype=complex)
        return self.bases_mat

    def _build_coefficient(self, **kwargs):
        self.coefficient = kwargs['coefficient']
        return np.array([self.coefficient])

    def _build_bases(self, **kwargs):
        hexagon_mf_bases = kwargs['hexagon_mf_bases']
        assert hexagon_mf_bases.shape[1] == self.shape[0]
        base_l = max(hexagon_mf_bases.shape)
        self._init_base()
        for i in range(0, base_l):
            # diagonal interaction part of Hamiltonian
            ns = [k[i] for k in hexagon_mf_bases]
            self._update(i, ns)

        return self.bases_mat


class HexMFBasesBuilder(CachedBuilder):

    def __init__(self, nmax):
        CachedBuilder.__init__(self, "HexMFBases", shape=(12, (nmax + 1) ** 12))

    def _combine(self):
        return self.bases_mat

    def _build_bases(self, **kwargs):
        base_l = self.shape[1]
        self.bases_mat = np.array([[0 if j < (base_l / (2 ** (idx+1))) else 1
                                   for _ in range(0, 2 ** idx)
                                   for j in range(0, int(base_l / (2 ** idx)))]
                                  for idx in range(0, 12)], dtype=complex)
        return self.bases_mat


class HexMFOperatorsBuilder(CachedBuilder):

    def __init__(self, nmax):
        CachedBuilder.__init__(self, "HexMFOperators", shape=(12, (nmax + 1) ** 12, (nmax + 1) ** 12))

    def _combine(self):
        return self.bases_mat

    def _build_bases(self, **kwargs):

        hexagon_mf_bases = kwargs['hexagon_mf_bases']
        base_l = self.shape[1]
        _bases_mat = np.array([sparse.lil_matrix((base_l, base_l), dtype=complex) for _ in range(0, 12)])

        kss = np.repeat(hexagon_mf_bases.T, base_l).reshape(base_l, 12, base_l)
        lss = np.tile(hexagon_mf_bases, (base_l, 1, 1))

        cmp_mat = np.not_equal(kss, lss)
        condition_mat = np.array([[cmp_mat[i, :, j] if np.count_nonzero(cmp_mat[i, :, j]) == 1 else np.zeros(12)
                                   for j in range(0, base_l)] for i in range(0, base_l)])

        for i, j, k in np.argwhere(condition_mat):
            if hexagon_mf_bases[k, i] == hexagon_mf_bases[k, j] - 1:
                _bases_mat[k][i, j] = cmath.sqrt(hexagon_mf_bases[k, j])

        self.bases_mat = np.array([sparse.csr_matrix(mat) for mat in _bases_mat])
        return self.bases_mat


class UTermBuilder(DiagCachedBuilder):

    def __init__(self, nmax, u=None):
        DiagCachedBuilder.__init__(self, "UTerm", shape=(((nmax + 1) ** 12), ((nmax + 1) ** 12)),
                                   coefficient=u)

    def _combine(self):
        ret = sparse.csr_matrix(np.diagflat(np.sum(np.multiply(self.coefficient, self.bases_mat), axis=0)))
        assert ret.shape == self.shape
        return ret

    def _init_base(self):
        self.bases_mat = np.zeros((6, self.shape[0]), dtype=complex)
        return self.bases_mat

    def _update(self, idx, ns):
        for j in range(0, 6):
            self.bases_mat[j, idx] = ns[2 * j] * (ns[2 * j] - 1) + ns[2 * j + 1] * (ns[2 * j + 1] - 1)


class UABTermBuilder(DiagCachedBuilder):

    def __init__(self, nmax, delta=None):
        DiagCachedBuilder.__init__(self, "UABTerm", shape=(((nmax + 1) ** 12), ((nmax + 1) ** 12)),
                                   coefficient=delta)

    def _combine(self):
        ua, ub = self.coefficient / 2, -self.coefficient / 2
        ret = sparse.csr_matrix(np.diagflat(np.multiply(ua, self.bases_mat[0]) + np.multiply(ub, self.bases_mat[1])))
        assert ret.shape == self.shape
        return ret

    def _update(self, idx, ns):
        self.bases_mat[0, idx] = ns[0] + ns[1] + ns[4] + ns[5] + ns[8] + ns[9]
        self.bases_mat[1, idx] = ns[2] + ns[3] + ns[6] + ns[7] + ns[10] + ns[11]

    def _init_base(self):
        self.bases_mat = np.zeros((2, self.shape[0]), dtype=complex)
        return self.bases_mat


class VTermBuilder(DiagCachedBuilder):

    def __init__(self, nmax, V=None):
        DiagCachedBuilder.__init__(self, "VTerm", shape=(((nmax + 1) ** 12), ((nmax + 1) ** 12)),
                                   coefficient=V)

    def _update(self, idx, ns):
        self.bases_mat[idx] = (ns[0] * ns[1] + ns[2] * ns[3] + ns[4] * ns[5] + ns[6] * ns[7]
                               + ns[8] * ns[9] + ns[10] * ns[11])


class MUTermBuilder(DiagCachedBuilder):

    def __init__(self, nmax, MU=None):
        DiagCachedBuilder.__init__(self, "MUTerm", shape=(((nmax + 1) ** 12), ((nmax + 1) ** 12)),
                                   coefficient=MU)

    def _update(self, idx, ns):
        self.bases_mat[idx] = -np.sum(ns)


class TTermBuilder(CachedBuilder):

    def __init__(self, nmax):
        CachedBuilder.__init__(self, "TTerm", shape=((nmax + 1) ** 12, (nmax + 1) ** 12))

    def _combine(self):
        t_coefficient_mat = sparse.lil_matrix((self.shape[0], self.shape[0]), dtype=complex)
        for i, j in np.argwhere(self.bases_mat[1]):
            k = self.bases_mat[1][i, j].imag
            k_idx = np.int(np.round(k))
            t_coefficient_mat[i, j] = self.coefficient_mat[k_idx] if k > 0 else np.conj(self.coefficient_mat[-k_idx])
        return sparse.csr_matrix(t_coefficient_mat.multiply(self.bases_mat[0]))

    def _build_coefficient(self, **kwargs):
        ts = kwargs['ts']
        self.coefficient_mat = np.array([ts[2], ts[3], ts[0], ts[1], ts[4], ts[5], ts[2], ts[3], ts[0], ts[1], ts[4], ts[5]])
        return self.coefficient_mat

    def _build_bases(self, **kwargs):

        hexagon_mf_bases = kwargs['hexagon_mf_bases']
        base_l = max(hexagon_mf_bases[0].shape)
        t_term_base = np.array([sparse.lil_matrix((base_l, base_l), dtype=complex) for _ in range(0, 2)])

        kss = np.repeat(hexagon_mf_bases.T, base_l).reshape(base_l, 12, base_l)
        lss = np.tile(hexagon_mf_bases, (base_l, 1, 1))

        cmp_mat = np.not_equal(kss, lss)
        condition_mat = np.array([[cmp_mat[i, :, j] if np.count_nonzero(cmp_mat[i, :, j]) == 2 else np.zeros(12)
                                   for j in range(0, base_l)] for i in range(0, base_l)])

        arg_mat = np.argwhere(condition_mat)
        arg_mat = arg_mat.reshape((arg_mat.shape[0] // 2, 2, 3))
        for pair in arg_mat:
            i, j, k1 = pair[0, 0:3]
            k2 = pair[1, 2]
            if k1 // 2 == 0 and k2 // 2 == 5:
                k1, k2 = k2, k1
            if k2 - k1 == 2 or k2 - k1 == -10:
                if kss[i, k1, j] == lss[i, k1, j] + 1 and kss[i, k2, j] == lss[i, k2, j] - 1:
                    t_term_base[0][i, j] = np.sqrt(kss[i, k1, j] * lss[i, k2, j])
                    t_term_base[1][i, j] = complex(1, k1) if (k1 // 2) % 2 else complex(1, -k1)
                elif kss[i, k1, j] == lss[i, k1, j] - 1 and kss[i, k2, j] == lss[i, k2, j] + 1:
                    t_term_base[0][i, j] = np.sqrt(kss[i, k2, j] * lss[i, k1, j])
                    t_term_base[1][i, j] = complex(1, -k1+0.1) if (k1 // 2) % 2 else complex(1, k1+0.1)
        self.bases_mat = t_term_base

        return self.bases_mat


class GTermBuilder(CachedBuilder):

    def __init__(self, nmax):
        CachedBuilder.__init__(self, "GTerm", shape=((nmax + 1) ** 12, (nmax + 1) ** 12))

    def _combine(self):
        return sparse.csr_matrix(self.coefficient_mat.multiply(self.bases_mat[0]))

    def _build_coefficient(self, **kwargs):
        g_coefficient_mat = sparse.lil_matrix(np.ones(self.shape[0], self.shape[0]), dtype=complex)
        ww = kwargs['WW']

        for i, j in np.argwhere(self.bases_mat[1]):
            g_coefficient_mat[i, j] = self.bases_mat[1][i, j] ** ww

        self.coefficient_mat = g_coefficient_mat
        return self.coefficient_mat

    def _build_bases(self, **kwargs):

        hexagon_mf_bases = kwargs['hexagon_mf_bases']
        base_l = max(hexagon_mf_bases[0].shape)
        # g_term_base = sparse.lil_matrix((base_l, base_l), dtype=complex)
        g_term_base = np.array([sparse.lil_matrix((base_l, base_l), dtype=complex) for _ in range(0, 2)])

        # for i in range(0, base_l):
        #     for j in range(0, base_l):
        #         g_term_base[1][i, j] = 1

        kss = np.repeat(hexagon_mf_bases.T, base_l).reshape(base_l, 12, base_l)
        lss = np.tile(hexagon_mf_bases, (base_l, 1, 1))

        cmp_mat = np.not_equal(kss, lss)
        condition_mat = np.array([[cmp_mat[i, :, j] if np.count_nonzero(cmp_mat[i, :, j]) == 2 else np.zeros(12)
                                   for j in range(0, base_l)] for i in range(0, base_l)], dtype=complex)

        arg_mat = np.argwhere(condition_mat)
        arg_mat = arg_mat.reshape((arg_mat.shape[0] // 2, 2, 3))
        for pair in arg_mat:
            i, j, k1 = pair[0, 0:3]
            k2 = pair[1, 2]
            assert k2 > k1
            # k1 is the first different index, k2 is the second
            if k2 - k1 == 1 and k1 % 2 == 0:
                if kss[i, k1, j] == lss[i, k1, j] + 1 and kss[i, k2, j] == lss[i, k2, j] - 1:

                    g_term_base[0][i, j] = np.sqrt(kss[i, k1, j] * lss[i, k2, j])

                    # cases from 1up to 3up
                    if k1 in [2, 4]:
                        g_term_base[1][i, j] = np.e ** (-2j * np.pi / 3)
                    elif k1 in [8, 10]:
                        g_term_base[1][i, j] = np.e ** (2j * np.pi / 3)
                    # else:
                    #     g_term_base[1][i, j] = 1

                elif kss[i, k1, j] == lss[i, k1, j] - 1 and kss[i, k2, j] == lss[i, k2, j] + 1:

                    g_term_base[0][i, j] = np.sqrt(kss[i, k2, j] * lss[i, k1, j])

                    if k1 in [2, 4]:
                        g_term_base[1][i, j] = np.e ** (2j * np.pi / 3)
                    elif k1 in [8, 10]:
                        g_term_base[1][i, j] = np.e ** (-2j * np.pi / 3)
                    # else:
                    #     g_term_base[1][i, j] = 1

        self.bases_mat = g_term_base

        return self.bases_mat


class VarTermsBuilder(CachedBuilder):

    def __init__(self, nmax):
        CachedBuilder.__init__(self, "VarTerm", shape=(24, (nmax + 1) ** 12, (nmax + 1) ** 12))

    def _combine(self):
        t_factor_mat = np.array([sparse.lil_matrix((self.shape[1], self.shape[1]), dtype=complex) for _ in range(0, 24)])
        for i in range(0, 24):
            for j, k in np.argwhere(self.bases_mat[1][i]):
                idx = np.int(self.bases_mat[1][i][j, k].imag)
                t_factor_mat[i][j, k] = self.coefficient_mat[idx] if idx > 0 else np.conj(self.coefficient_mat[-idx])
        ret = np.array([sparse.csr_matrix(t_mat.multiply(b_mat)) for t_mat, b_mat in zip(t_factor_mat, self.bases_mat[0])])
        return ret

    def _build_coefficient(self, **kwargs):
        ts = kwargs['ts']
        self.coefficient_mat = np.array([ts[0], ts[4], ts[2], ts[0], ts[4], ts[2], ts[1], ts[5], ts[3], ts[1], ts[5], ts[3]])
        return self.coefficient_mat

    def _build_bases(self, **kwargs):

        hexagon_mf_bases = kwargs['hexagon_mf_bases']
        base_l = max(hexagon_mf_bases[0].shape)
        var_term_base = np.array([[sparse.lil_matrix((base_l, base_l), dtype=complex) for _ in range(0, 24)] for _ in range(0, 2)])
        # var_term_base = np.tile(np.repeat(sparse.lil_matrix((base_l, base_l), dtype=complex), 24), (2, 1))

        kss = np.repeat(hexagon_mf_bases.T, base_l).reshape(base_l, 12, base_l)
        lss = np.tile(hexagon_mf_bases, (base_l, 1, 1))

        cmp_mat = np.not_equal(kss, lss)
        condition_mat = np.array([[cmp_mat[i, :, j] if np.count_nonzero(cmp_mat[i, :, j]) == 1 else np.zeros(12)
                                   for j in range(0, base_l)] for i in range(0, base_l)])

        for ii, jj, kk in np.argwhere(condition_mat):

            if np.abs(kss[ii, kk, jj] - lss[ii, kk, jj]) == 1:
                # condition for up_a_term
                if kk % 2 == 0 and kss[ii, kk, jj] == lss[ii, kk, jj] - 1:
                    var_term_base[0][kk // 2][ii, jj] = np.sqrt(lss[ii, kk, jj])
                    var_term_base[1][kk // 2][ii, jj] = complex(1, -(kk // 2)) if kk % 4 else complex(1, (kk // 2))
                # condition for up_adg_term
                elif kk % 2 == 0 and kss[ii, kk, jj] == lss[ii, kk, jj] + 1:
                    var_term_base[0][(kk // 2) + 6][ii, jj] = np.sqrt(kss[ii, kk, jj])
                    var_term_base[1][(kk // 2) + 6][ii, jj] = complex(1, (kk // 2)) if kk % 4 else complex(1, -(kk // 2))
                # condition for dn_a_term
                elif kk % 2 != 0 and kss[ii, kk, jj] == lss[ii, kk, jj] - 1:
                    var_term_base[0][(kk // 2) + 12][ii, jj] = np.sqrt(lss[ii, kk, jj])
                    var_term_base[1][(kk // 2) + 12][ii, jj] = complex(1, (-((kk // 2) + 6))) if (kk - 1) % 4 else complex(1, ((kk // 2) + 6))
                # condition for dn_adg_term
                elif kk % 2 != 0 and kss[ii, kk, jj] == lss[ii, kk, jj] + 1:
                    var_term_base[0][(kk // 2) + 18][ii, jj] = np.sqrt(kss[ii, kk, jj])
                    var_term_base[1][(kk // 2) + 18][ii, jj] = complex(1, ((kk // 2) + 6)) if (kk - 1) % 4 else complex(1, (-((kk // 2) + 6)))

        self.bases_mat = var_term_base
        return self.bases_mat


class TsBuilder(Builder):

    def __init__(self, W):
        Builder.__init__(self, "Ts", shape=(6, ))
        t0 = 1 + 0j
        t1, t2, t3 = t0, t0 * cmath.exp(1j * W), t0 * cmath.exp(-1j * W)
        t1_up, t2_up, t3_up = t1, t2, t3
        t1_dn, t2_dn, t3_dn = t1_up.conjugate(), t2_up.conjugate(), t3_up.conjugate()
        self.term = np.array([t1_up, t1_dn, t2_up, t2_dn, t3_up, t3_dn])


def build(model, **kwargs):
    return model(nmax=kwargs['nmax']).prepare_term(**kwargs).get_term()


def builder(nmax, g_lower_bound, g_pivot, g_upper_bound, n1, n2,
            delta, MU, U, V, W, WW, mu_lower_bound, mu_upper_bound, ma):

    base_l = (nmax + 1) ** 12
    # non-term preparations
    # range setting of hopping strength
    # ta-part1,near phase transition boundary, need to be calculated more densely
    g_a = np.linspace(g_lower_bound, g_pivot, n1)
    # tb-part2
    g_b = np.linspace(g_pivot, g_upper_bound, n2)
    gA = np.array([*g_a, *g_b])
    # the range of mu, chemical potential
    Ma = np.linspace(mu_lower_bound, mu_upper_bound, ma)
    # build other vars
    dig_h = sparse.eye((nmax + 1) ** 12, dtype=complex, format='csr')
    # the range of order parameters trial solution, the trial OrderParameter is Complex with Pa(i,j)=Pr*exp(i*theta)
    Pr = np.linspace(0.01, cmath.sqrt(nmax), 10)
    # Psi1up, Psi1dn, Psi2up, Psi2dn ... Psi6up, Psi6dn
    Psi_s = np.tile(np.zeros((ma, n1 + n2), dtype=complex), (20, 1, 1))
    # N1up, ... N2dn, N1squareup, ... N2squaredn
    Ns = np.tile(np.zeros((ma, n1 + n2), dtype=complex), (12, 1, 1))
    Nsquare_s = np.tile(np.zeros((ma, n1 + n2), dtype=complex), (12, 1, 1))
    NaN = np.zeros((ma, n1 + n2), dtype=complex)
    # store all the eigen-vectors solved
    # Vec_s = np.tile(np.zeros((nmax+1)**12, dtype=complex), (ma, n1+n2, 1))
    EVals = np.zeros((ma, n1+n2, 10), dtype=complex)
    EVecs = np.zeros((ma, n1+n2, 10, base_l), dtype=complex)

    # ts
    ts = TsBuilder(W=W).get_term()

    # build terms
    mf_bases = build(model=HexMFBasesBuilder, nmax=nmax)
    mf_ops = build(model=HexMFOperatorsBuilder, nmax=nmax, hexagon_mf_bases=mf_bases)
    u_term = build(model=UTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, coefficient=U)
    uab_term = build(model=UABTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, coefficient=delta)
    v_term = build(model=VTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, coefficient=V)
    mu_term = build(model=MUTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, coefficient=MU)
    t_term = build(model=TTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, ts=ts)
    var_terms = build(model=VarTermsBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, ts=ts)
    g_term = build(model=GTermBuilder, nmax=nmax, hexagon_mf_bases=mf_bases, WW=WW)

    return {"hexagon_mf_operators": mf_ops,
            'g_a': g_a, 'g_b': g_b, 'gA': gA, 'ts': ts, 'Ma': Ma,
            'uab_term': uab_term, 'u_term': u_term, 'v_term': v_term, 'mu_term': mu_term, 't_term': t_term,
            'g_term': g_term, 'var_terms': var_terms,
            'dig_h': dig_h, 'Pr': Pr, 'Psi_s': Psi_s, 'Ns': Ns, 'EVals': EVals, 'EVecs': EVecs,
            'Nsquare_s': Nsquare_s, 'NaN': NaN}


if __name__ == '__main__':
    from utilities import load_params
    params = load_params("params.json")
    terms = builder(nmax=params['nmax'], g_lower_bound=params['g_lower_bound'], g_pivot=params['g_pivot'],
                    g_upper_bound=params['g_upper_bound'],
                    n1=params['n1'], n2=params['n2'], delta=params['delta'], MU=params['MU'], U=params['U'],
                    V=params['V'],  W=params['W'],
                    mu_lower_bound=params['mu_lower_bound'], mu_upper_bound=params['mu_upper_bound'], ma=params['ma'])

    cur_dir = os.getcwd()
    target = "build_result"
    if not os.path.exists(target):
        os.makedirs(target)
    os.chdir(target)
    term_list = ['uab_term', 'u_term', 'v_term', 'mu_term', 't_term', 'var_terms', 'hexagon_mf_operators']
    for term in term_list:
        if sparse.isspmatrix(terms[term]):
            sparse.save_npz(f"{term}.npz", terms[term])
        else:
            np.save(f"{term}.npy", terms[term])
    os.chdir(cur_dir)


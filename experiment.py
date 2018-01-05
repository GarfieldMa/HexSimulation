from builders import builder
from solvers import solves
from utilities import load_params, dump_result


class Experiment(object):

    def __init__(self, param_path):
        self.params = load_params(param_path)

    def run(self):
        terms = builder(nmax=self.params['nmax'], g_lower_bound=self.params['g_lower_bound'], g_pivot=self.params['g_pivot'], g_upper_bound=
                        self.params['g_upper_bound'],
                        n1=self.params['n1'], n2=self.params['n2'], delta=self.params['delta'], MU=self.params['MU'], U=
                        self.params['U'], V=self.params['V'], W=self.params['W'], WW=self.params['WW'],
                        mu_lower_bound=self.params['mu_lower_bound'], mu_upper_bound=self.params['mu_upper_bound'], ma=
                        self.params['ma'])

        result = solves(hexagon_mf_operators=terms['hexagon_mf_operators'], g_a=terms['g_a'], g_b=terms['g_b'],
                        ts=terms['ts'], Ma=terms['Ma'], uab_term=terms['uab_term'], u_term=terms['u_term'], v_term=terms['v_term'], mu_term=terms['mu_term'],
                        t_term=terms['t_term'], g_term=terms['g_term'], var_terms=terms['var_terms'], t = self.params["t"],
                        dig_h=terms['dig_h'], Pr=terms['Pr'], Psi_s=terms['Psi_s'], Ns=terms['Ns'], Nsquare_s=terms['Nsquare_s'], NaN=terms['NaN'], EVals=terms['EVals'], EVecs=terms['EVecs'],
                        err=self.params['err'], wall_time=self.params['wall_time'])

        dump_result(Psi_s=result['Psi_s'], Ns=result['Ns'], Nsquare_s=result['Nsquare_s'], NaN=result['NaN'], EVals=result['EVals'], EVecs=result['EVecs'], gA=terms['gA'], Ma=terms['Ma'], params=self.params)
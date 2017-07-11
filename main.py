from builders import builder
from solvers import solves
from utilities import load_params, dump_result


if __name__ == '__main__':
    params = load_params("params.json")
    terms = builder(nmax=params['nmax'], t_lower_bound=params['t_lower_bound'], t_pivot=params['t_pivot'], t_upper_bound=params['t_upper_bound'],
                    n1=params['n1'], n2=params['n2'], delta=params['delta'], MU=params['MU'], U=params['U'], V=params['V'],  W=params['W'],
                    mu_lower_bound=params['mu_lower_bound'], mu_upper_bound=params['mu_upper_bound'], ma=params['ma'], cached=params['cached'])
    result = solves(hexagon_mf_operators=terms['hexagon_mf_operators'], t_a=terms['t_a'], t_b=terms['t_b'],
                    ts=terms['ts'], Ma=terms['Ma'], uab_term=terms['uab_term'], u_term=terms['u_term'], v_term=terms['v_term'], mu_term=terms['mu_term'],
                    t_term=terms['t_term'], var_terms=terms['var_terms'], dig_h=terms['dig_h'], Pr=terms['Pr'],
                    Psi_s=terms['Psi_s'], Ns=terms['Ns'],Nsquare_s=terms['Nsquare_s'], Vec_s=terms['Vec_s'], err=params['err'], wall_time=params['wall_time'])
    dump_result(Psi_s=result['Psi_s'], Ns=result['Ns'], Nsquare_s=result['Nsquare_s'], Vec_s=result['Vec_s'], tA=terms['tA'], Ma=terms['Ma'], params=params)
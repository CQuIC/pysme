"""Code for cascading a leaky cavity as a source for squeezed light

    .. module:: matrix_form.py
       :synopsis: Code for cascading a leaky cavity as a source for squeezed
                  light
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import pysme.integrate as integ
import pysme.gellmann as gm
import pysme.system_builder as sb

def trunc_osc_src_SLH(n_max, E, g):
    '''SLH triple for a number-truncated cavity squeezed light source

    Approximate the cavity with the subspace spanned by the first `n_max`
    number states.

    '''
    S = np.eye(n_max + 1, dtype=np.complex)
    a = np.zeros((n_max + 1, n_max + 1), dtype=np.complex)
    for n in range(n_max):
        a[n,n+1] = np.sqrt(n + 1)
    a_dag = a.conjugate().T
    L = np.sqrt(g) * a
    H = -1.j * (E * a_dag @ a_dag - E.conjugate() * a @ a) / 2
    return {'S': S, 'L': L, 'H': H}

def sqz_trunc_osc_src_SLH(n_max, E, g):
    '''SLH triple for a squeezed-number-truncated cavity squeezed light source

    Approximate the cavity with the subspace spanned by the first `n_max`
    squeezed number states with squeezing parameters corresponding to the
    steady-state squeezing of the cavity.

    '''
    S = np.eye(n_max + 1, dtype=np.complex)
    a = np.zeros((n_max + 1, n_max + 1), dtype=np.complex)
    for n in range(n_max):
        a[n,n+1] = np.sqrt(n + 1)
    a_dag = a.conjugate().T
    mu = np.angle(E) / 2
    sinh2r = (g / np.sqrt(g**2 - 4 * np.abs(E)**2) - 1) / 2
    coshr = np.sqrt(1 + sinh2r)
    sinhr = np.sqrt(sinh2r)
    a_sq = coshr * a - np.exp(2.j * mu) * sinhr * a_dag
    L = np.sqrt(g) * a_sq
    H = -1.j * (E * a_dag @ a_dag - E.conjugate() * a @ a) / 2
    return {'S': S, 'L': L, 'H': H}

def series_SLH(SLH2, SLH1):
    S1 = SLH1['S']
    L1 = SLH1['L']
    H1 = SLH1['H']
    S2 = SLH2['S']
    L2 = SLH2['L']
    H2 = SLH2['H']
    S = np.kron(S1, S2)
    L = np.kron(np.eye(L1.shape[0], dtype=L1.dtype), L2) + np.kron(L1, S2)
    H = (np.kron(H1, np.eye(H2.shape[0], dtype=H2.dtype)) +
         np.kron(np.eye(H1.shape[0], dtype=H1.dtype), H2) +
         (np.kron(L1, L2.conjugate().T @ S2) -
          np.kron(L1.conjugate().T, S2.conjugate().T @ L2)) / 2.j)
    return {'S': S, 'L': L, 'H': H}

def make_trunc_osc_src_integrator(n_max, E, g_cavity, c_sys, H_sys):
    total_SLH = series_SLH({'S': np.eye(c_sys.shape[0], dtype=c_sys.dtype),
                            'L': c_sys, 'H': H_sys},
                           trunc_osc_src_SLH(n_max, E, g_cavity))
    return integ.UncondGaussIntegrator(total_SLH['L'], 0, 0, total_SLH['H'])

def make_sqz_trunc_osc_src_integrator(n_max, E, g_cavity, c_sys, H_sys):
    total_SLH = series_SLH({'S': np.eye(c_sys.shape[0], dtype=c_sys.dtype),
                            'L': c_sys, 'H': H_sys},
                           sqz_trunc_osc_src_SLH(n_max, E, g_cavity))
    return integ.UncondGaussIntegrator(total_SLH['L'], 0, 0, total_SLH['H'])

class SqzTruncOscSrcIntegratorFactory():
    def __init__(self, d_sys, n_max):
        self.d_sys = d_sys
        self.n_max = n_max
        self.d_total = (self.n_max + 1) * self.d_sys
        self.basis = gm.get_basis(self.d_total)
        # Only want the parts of common_dict pertaining to the basis setup
        # (which is time-consuming). Suggests I should split the basis part out
        # into a separate function (perhaps a basis object).
        zero_op = np.zeros((self.d_total, self.d_total), dtype=np.complex)
        self.basis_common_dict = sb.op_calc_setup(zero_op, 0, 0, zero_op,
                                                  self.basis[:-1])

    def make_integrator(self, c_op, E, g, H):
        total_SLH = series_SLH({'S': np.eye(c_op.shape[0], dtype=c_op.dtype),
                                'L': c_op, 'H': H},
                               sqz_trunc_osc_src_SLH(self.n_max, E, g))
        c_total = total_SLH['L']
        H_total = total_SLH['H']
        common_dict = self.basis_common_dict.copy()
        common_dict['C_vector'] = sb.vectorize(c_total, common_dict['basis'])
        common_dict['H_vector'] = sb.vectorize(H_total, common_dict['basis'])
        D_c = sb.diffusion_op(**common_dict)
        F = sb.hamiltonian_op(**common_dict)
        drift_rep = D_c + F
        return integ.UncondGaussIntegrator(c_total, 0, 0, H_total,
                                           basis=common_dict['basis'],
                                           drift_rep=drift_rep)

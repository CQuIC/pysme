"""Code for cascading a leaky cavity as a source for squeezed light

    .. module:: matrix_form.py
       :synopsis: Code for cascading a leaky cavity as a source for squeezed
                  light
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import pysme.integrate as integ

def trunc_osc_src_SLH(n_max, E, g):
    S = np.eye(n_max + 1, dtype=np.complex)
    a = np.zeros((n_max + 1, n_max + 1), dtype=np.complex)
    for n in range(n_max):
        a[n,n+1] = np.sqrt(n + 1)
    L = np.sqrt(g) * a
    H = -1.j * (E * a.T.conjugate() @ a.T.conjugate() -
				E.conjugate() * a @ a) / 2
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

def make_sqz_src_integrator(n_max, E, g_cavity, c_sys, H_sys):
    total_SLH = series_SLH({'S': np.eye(c_sys.shape[0], dtype=c_sys.dtype),
                            'L': c_sys, 'H': H_sys},
                           trunc_osc_src_SLH(n_max, E, g_cavity))
    return integ.UncondGaussIntegrator(total_SLH['L'], 0, 0, total_SLH['H'])

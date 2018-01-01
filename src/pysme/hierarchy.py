"""Code for solving hierarchies of master equations

    .. module:: hierarchy.py
       :synopsis: Code for solving hierarchies of master equations
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import pysme.matrix_form as mf

class HierarchyState(list):
    '''Hierarchy state class that supports addition and scalar multiplication

    Useful for using in an euler integration routine.

    '''
    def __add__(self, other):
        return HierarchyState([[self_component + other_component
                                for self_component, other_component
                                in zip(self_row, other_row)]
                               for self_row, other_row in zip(self, other)])

    def __rmul__(self, other):
        return HierarchyState([[other * self_component
                                for self_component in self_row]
                               for self_row in self])

def get_mn(m, n, rho_hier, m_max):
    if m <= m_max and n <= m_max:
        return rho_hier[m,n]
    else:
        return np.zeros(rho_hier[0,0].shape, dtype=rho_hier[0,0].dtype)

def rho_dot_sqz_hier(rho_hier, c, r, mu, gamma, xi, m_max):
    xi_star = xi.conjugate()
    c_dag = c.conjugate().T
    return np.array([[mf.D(c, get_mn(m, n, rho_hier, m_max)) +
                      xi_star * mf.comm(c, np.sqrt(n + 1) * np.exp(-2.j * mu) *
                      np.sinh(r) * get_mn(m, n + 1, rho_hier, m_max) +
                      np.sqrt(n) * np.cosh(r) *
                      get_mn(m, n - 1, rho_hier, m_max)) +
                      xi * mf.comm(np.sqrt(m + 1) * np.exp(2.j * mu) *
                      np.sinh(r) * get_mn(m + 1, n, rho_hier, m_max) +
                      np.sqrt(m) * np.cosh(r) *
                      get_mn(m - 1, n, rho_hier, m_max), c_dag)
                      for n in range(m_max + 1)]
                     for m in range(m_max + 1)])

def euler_integrate_sqz_hier(rho_0, c, r, mu, gamma, xi_fn, times, m_max):
    rho_hier_0 = np.array([[rho_0 if j == k
                            else np.zeros(rho_0.shape, dtype=rho_0.dtype)
                            for j in range(m_max + 1)]
                           for k in range(m_max + 1)])
    return mf.euler_integrate(rho_hier_0,
                              lambda rho_hier, t:
                              rho_dot_sqz_hier(rho_hier, c, r, mu, gamma,
                                               xi_fn(t), m_max),
                              times)

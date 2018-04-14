"""Code for solving hierarchies of master equations

    .. module:: hierarchy.py
       :synopsis: Code for solving hierarchies of master equations
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
import pysme.matrix_form as mf
import pysme.system_builder as sb
import pysme.integrate as integ
import pysme.gellmann as gm

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

class HierarchyIntegratorFactory():
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
        self.A = np.zeros((self.n_max + 1, self.n_max + 1),
                          dtype=np.complex)
        for n in range(n_max):
            self.A[n, n+1] = np.sqrt(n + 1)

    def drift_rep(self, c_tot, Asq_pl, xi_t, S, H):
# This common_dict solution is really proving to be annoying to build upon...
        common_dict = self.basis_common_dict.copy()
        c_1 = c_tot + xi_t * np.kron(S, Asq_pl)
        common_dict['C_vector'] = sb.vectorize(c_1, common_dict['basis'])
        D_1 = sb.diffusion_op(**common_dict)
        c_2 = np.kron(np.eye(self.d_sys), Asq_pl)
        common_dict['C_vector'] = sb.vectorize(c_2, common_dict['basis'])
        D_2 = np.abs(xi_t)**2 * sb.diffusion_op(**common_dict)
        h_tot = np.kron(H, np.eye(self.n_max + 1))
        common_dict['H_vector'] = sb.vectorize(h_tot, common_dict['basis'])
        F = sb.hamiltonian_op(**common_dict)
        return D_1 - D_2 + F

    def make_integrator(self, c_op, r, mu, xi_fn, S, H):
        c_tot = np.kron(c_op, np.eye(self.n_max + 1))
        Asq_pl = np.cosh(r) * self.A.T - np.sinh(r) * np.exp(2.j * mu) * self.A
        drift_rep_fn = lambda t: self.drift_rep(c_tot, Asq_pl, xi_fn(t), S, H)
        return WavepacketIntegrator(drift_rep_fn,
                                    self.basis_common_dict['basis'],
                                    self.n_max)

class WavepacketIntegrator:
    def __init__(self, drift_rep_fn, basis, n_max):
        self.Dfun = drift_rep_fn
        self.basis = basis
        self.n_max = n_max

    def a_fn(self, rho, t):
        return np.dot(self.Dfun(t), rho)

    def integrate(self, rho_0, times):
        r"""Integrate the equation for a list of times with given initial
        conditions.

        :param rho_0:   The initial state of the system
        :type rho_0:    `numpy.array`
        :param times:   A sequence of time points for which to solve for rho
        :type times:    `list(real)`
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         `Solution`

        """
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1)),
                                 self.basis).real
        vec_soln = odeint(self.a_fn, rho_0_vec, times, Dfun=self.Dfun)
        return integ.Solution(vec_soln, self.basis)

"""Code for solving hierarchies of master equations

    .. module:: hierarchy.py
       :synopsis: Code for solving hierarchies of master equations
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
import pysme.matrix_form as mf
import pysme.system_builder as sb
import pysme.sparse_system_builder as ssb
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
        self.sparse_basis = ssb.SparseBasis(self.d_total)
        self.A = np.zeros((self.n_max + 1, self.n_max + 1),
                          dtype=np.complex)
        for n in range(n_max):
            self.A[n, n+1] = np.sqrt(n + 1)

    def make_integrator(self, xi_fn, S, L, H, r, mu):
        return WavepacketUncondIntegrator(self.sparse_basis, self.n_max,
                                          self.A, xi_fn, S, L, H, r, mu)

class WavepacketUncondIntegrator:
    def __init__(self, sparse_basis, n_max, A, xi_fn, S, L, H, r, mu):
        self.basis = sparse_basis.basis.todense()
        self.n_max = n_max
        self.xi_fn = xi_fn

        I_hier = np.eye(n_max + 1, dtype=np.complex)
        L_vec = sparse_basis.vectorize(np.kron(L, I_hier))
        DL = sparse_basis.make_diff_op_matrix(L_vec)
        H_vec = sparse_basis.vectorize(np.kron(H, I_hier))
        Hcomm = sparse_basis.make_hamil_comm_matrix(H_vec)
        self.wp_ind = DL + Hcomm

        Asq_pl = np.cosh(r) * A.conj().T - np.sinh(r) * np.exp(2.j*mu) * A
        SA_vec = sparse_basis.vectorize(np.kron(S, Asq_pl))
        self.wp_re = 2 * sparse_basis.make_real_comm_matrix(SA_vec, L_vec)
        self.wp_im = 2 * sparse_basis.make_real_comm_matrix(1.j*SA_vec, L_vec)

        I_sys = np.eye(S.shape[0], dtype=np.complex)
        Asq_pl_vec = sparse_basis.vectorize(np.kron(I_sys, Asq_pl))
        S_vec = sparse_basis.vectorize(np.kron(S, I_hier))
        Asand = sparse_basis.make_real_sand_matrix(Asq_pl_vec, Asq_pl_vec)
        DS = sparse_basis.make_diff_op_matrix(S_vec)
        self.wp_abs = DS.dot(Asand)

    def a_fn(self, rho, t):
        return np.dot(self.Dfun(t), rho)

    def Dfun(self, t):
        xi_t = self.xi_fn(t)
        return (self.wp_ind + xi_t.real * self.wp_re + xi_t.imag * self.wp_im +
                np.abs(xi_t)**2 * self.wp_abs)

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

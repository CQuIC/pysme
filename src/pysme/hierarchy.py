"""Code for solving hierarchies of master equations

    .. module:: hierarchy.py
       :synopsis: Code for solving hierarchies of master equations
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import pysme.matrix_form as mf
import pysme.system_builder as sb
import pysme.sparse_system_builder as ssb
import pysme.integrate as integ
import pysme.sde as sde

def process_default_kwargs(kwargs, default_kwargs):
    """Update a default kwarg dict with user-supplied values

    """
    if kwargs is None:
        kwargs = {}
    for kwarg, value in kwargs.items():
        default_kwargs[kwarg] = value

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
                                        np.sinh(r) *
                                        get_mn(m, n + 1, rho_hier, m_max) +
                                        np.sqrt(n) * np.cosh(r) *
                                        get_mn(m, n - 1, rho_hier, m_max)) +
                      xi * mf.comm(np.sqrt(m + 1) * np.exp(2.j * mu) *
                                   np.sinh(r) *
                                   get_mn(m + 1, n, rho_hier, m_max) +
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

class HierarchySolution(integ.Solution):
    def __init__(self, vec_soln, basis, d_sys):
        super().__init__(vec_soln, basis)
        self.d_sys = d_sys
        self.d_total = int(np.sqrt(basis.shape[0]))
        self.d_hier = self.d_total // self.d_sys
        self.basis_sys = ssb.SparseBasis(d_sys).basis.todense()

    def get_phys_dual_basis(self, field_rho_0):
        return np.array([sb.dualize(np.kron(basis_el, field_rho_0),
                                    self.basis).real / sb.norm_squared(basis_el)
                         for basis_el in self.basis_sys])

    def get_expectations(self, observable, field_rho_0, idx_slice=None, hermitian=True):
        hier_obs = np.kron(observable, field_rho_0)
        return super().get_expectations(hier_obs, idx_slice=idx_slice,
                                        hermitian=hermitian)

    def get_purities(self, field_rho_0):
        phys_soln = self.get_phys_soln(field_rho_0)
        return phys_soln.get_purities()

    def get_hierarchy_expectations(self, observable, idx_slice=None, hermitian=True):
        return super().get_expectations(observable, idx_slice=idx_slice,
                                        hermitian=hermitian)

    def get_hierarchy_density_matrices(self, idx_slice=None):
        return super().get_density_matrices(idx_slice)

    def get_density_matrices(self, field_rho_0, idx_slice=None):
        phys_soln = self.get_phys_soln(field_rho_0)
        return phys_soln.get_density_matrices(idx_slice)

    def save(self, outfile):
        np.savez_compressed(outfile, vec_soln=self.vec_soln,
                            basis=self.basis, d_sys=self.d_sys)

    def get_phys_soln(self, field_rho_0):
        phys_dual_basis = self.get_phys_dual_basis(field_rho_0)
        return integ.Solution(np.einsum('jk,mk->mj', phys_dual_basis,
                                        self.vec_soln), self.basis_sys)

def load_hierarchy_solution(infile):
    loaded = np.load(infile)
    return HierarchySolution(loaded['vec_soln'], loaded['basis'],
                             loaded['d_sys'])

class HierarchyIntegratorFactory():
    def __init__(self, d_sys, n_max, sparse_basis=None):
        '''sparse_basis should be a SparseBasis object from sparse_system_builder.

        '''
        self.d_sys = d_sys
        self.n_max = n_max
        self.d_total = (self.n_max + 1) * self.d_sys
        if sparse_basis is None:
            self.sparse_basis = ssb.SparseBasis(self.d_total)
        else:
            self.sparse_basis = sparse_basis
        self.A = np.zeros((self.n_max + 1, self.n_max + 1),
                          dtype=np.complex)
        for n in range(n_max):
            self.A[n, n+1] = np.sqrt(n + 1)

    def make_uncond_integrator(self, xi_fn, S, L, H, r, mu):
        return WavepacketUncondIntegrator(self.sparse_basis, self.n_max,
                                          self.A, xi_fn, S, L, H, r, mu)

    def make_euler_hom_integrator(self, xi_fn, S, L, H, r=0, mu=0, hom_ang=0,
                                  field_state=None):
        return EulerWavepacketHomodyneIntegrator(self.sparse_basis, self.n_max,
                                                 self.A, xi_fn, S, L, H, r, mu,
                                                 hom_ang, field_state)

    def make_milstein_hom_integrator(self, xi_fn, S, L, H, r=0, mu=0, hom_ang=0,
                                     field_state=None):
        return MilsteinWavepacketHomodyneIntegrator(self.sparse_basis,
                                                    self.n_max, self.A, xi_fn,
                                                    S, L, H, r, mu, hom_ang,
                                                    field_state)

    def make_euler_jump_integrator(self, xi_fn, S, L, H, r=0, mu=0,
                                   field_state=None):
        return EulerWavepacketJumpIntegrator(self.sparse_basis, self.n_max,
                                             self.A, xi_fn, S, L, H, r, mu,
                                             field_state)

class HierarchyIntegratorFactoryExpCutoff(HierarchyIntegratorFactory):
    def __init__(self, d_sys, n_max, decay_const, sparse_basis=None):
        super().__init__(d_sys, n_max, sparse_basis=sparse_basis)
        self.A = np.zeros((self.n_max + 1, self.n_max + 1),
                          dtype=np.complex)
        for n in range(n_max):
            self.A[n, n+1] = np.sqrt(n + 1)*np.exp(-n/decay_const)

class WavepacketUncondIntegrator:
    def __init__(self, sparse_basis, n_max, A, xi_fn, S, L, H, r=0, mu=0):
        self.basis = sparse_basis.basis.todense()
        self.n_max = n_max
        self.d_sys = S.shape[0]
        self.xi_fn = xi_fn

        I_hier = np.eye(n_max + 1, dtype=np.complex)
        self.L_vec = sparse_basis.vectorize(np.kron(L, I_hier))
        DL = sparse_basis.make_diff_op_matrix(self.L_vec)
        H_vec = sparse_basis.vectorize(np.kron(H, I_hier))
        Hcomm = sparse_basis.make_hamil_comm_matrix(H_vec)
        self.wp_ind = DL + Hcomm

        self.Asq_pl = np.cosh(r) * A.conj().T - np.sinh(r) * np.exp(2.j*mu) * A
        self.SA_vec = sparse_basis.vectorize(np.kron(S, self.Asq_pl))
        self.wp_re = 2 * sparse_basis.make_real_comm_matrix(self.SA_vec,
                                                            self.L_vec)
        self.wp_im = 2 * sparse_basis.make_real_comm_matrix(1.j * self.SA_vec,
                                                            self.L_vec)

        self.I_sys = np.eye(self.d_sys, dtype=np.complex)
        Asq_pl_vec = sparse_basis.vectorize(np.kron(self.I_sys, self.Asq_pl))
        S_vec = sparse_basis.vectorize(np.kron(S, I_hier))
        self.Asand = sparse_basis.make_real_sand_matrix(Asq_pl_vec, Asq_pl_vec)
        DS = sparse_basis.make_diff_op_matrix(S_vec)
        self.wp_abs = DS.dot(self.Asand)

    def a_fn(self, rho, t):
        return np.dot(self.Dfun(t), rho)

    def Dfun(self, t):
        xi_t = self.xi_fn(t)
        return (self.wp_ind + xi_t.real * self.wp_re + xi_t.imag * self.wp_im +
                np.abs(xi_t)**2 * self.wp_abs)

    def integrate(self, rho_0, times, solve_ivp_kwargs=None):
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
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': lambda t, rho: self.Dfun(t)}
        process_default_kwargs(solve_ivp_kwargs, default_solve_ivp_kwargs)
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1,
                                                       dtype=np.complex)),
                                 self.basis).real
        ivp_soln = solve_ivp(lambda t, rho: self.a_fn(rho, t),
                             (times[0], times[-1]), rho_0_vec,
                              **default_solve_ivp_kwargs)
        return HierarchySolution(ivp_soln.y.T, self.basis, self.d_sys)

    def integrate_vec_init_cond(self, rho_0_vec, times):
        r"""Integrate the equation for a list of times with given initial
        conditions, already expressed in vectorized form.

        :param rho_0_vec:   The initial state of the system in vectorized form
        :type rho_0_vec:    `numpy.array`
        :param times:       A sequence of time points for which to solve for rho
        :type times:        `list(real)`
        :returns:           The components of the vecorized :math:`\rho` for all
                            specified times
        :rtype:             `Solution`

        """
        vec_soln = odeint(self.a_fn, rho_0_vec, times, Dfun=self.Dfun)
        return HierarchySolution(vec_soln, self.basis, self.d_sys)

    def integrate_hier_init_cond(self, rho_0_hier, times, solve_ivp_kwargs=None):
        r"""Integrate the equation for a list of times with given initial
        conditions, expressed as a full hierarchy density matrix rather than
        only a system density matrix. Handles non-hermitian initial conditions
        to facilitate applications involving the quantum regression theorem
        (this means the vectorized solutions will be complex in general).

        :param rho_0_hier:   The initial state of the system as a matrix
        :type rho_0_hier:    `numpy.array`
        :param times:        A sequence of time points for which to solve for
                             rho
        :type times:         `list(real)`
        :param method:       The integration method for
                             `scipy.integrate.solve_ivp` to use.
        :type method:        String
        :returns:            The components of the vecorized :math:`\rho` for
                             all
                             specified times
        :rtype:              `Solution`

        """
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': lambda t, rho: self.Dfun(t)}
        rho_0_vec = sb.vectorize(rho_0_hier, self.basis)
        ivp_soln = solve_ivp(lambda t, rho: self.a_fn(rho, t),
                             (times[0], times[-1]), rho_0_vec,
                             **default_solve_ivp_kwargs)
        return HierarchySolution(ivp_soln.y.T, self.basis, self.d_sys)

class EulerWavepacketHomodyneIntegrator(WavepacketUncondIntegrator):
    def __init__(self, sparse_basis, n_max, A, xi_fn, S, L, H, r=0, mu=0,
                 hom_ang=0, field_state=None):
        super().__init__(sparse_basis, n_max, A, xi_fn, S, L, H, r, mu)
        self.G_ind = sparse_basis.make_wiener_linear_matrix(
            np.exp(-1.j * hom_ang) * self.L_vec)
        self.G_re = sparse_basis.make_wiener_linear_matrix(
            np.exp(-1.j * hom_ang) * self.SA_vec)
        self.G_im = sparse_basis.make_wiener_linear_matrix(
            1.j * np.exp(-1.j * hom_ang) * self.SA_vec)
        if field_state is None:
            # If not specified, set initial field state to squeezed vacuum
            field_state = np.zeros(self.n_max + 1, dtype=np.complex)
            field_state[0] = 1
        field_state_proj = np.outer(field_state, field_state.conjugate())
        field_state_proj /= np.trace(field_state_proj).real
        trace_dual = np.real(sparse_basis.dualize(np.kron(self.I_sys,
                                                          field_state_proj)))
        self.k_T_ind = -trace_dual @ self.G_ind
        self.k_T_re = -trace_dual @ self.G_re
        self.k_T_im = -trace_dual @ self.G_im

    def k_T_t_fn(self, xi_t):
        return self.k_T_ind + xi_t.real * self.k_T_re + xi_t.imag * self.k_T_im

    def G_t_fn(self, xi_t):
        return self.G_ind + xi_t.real * self.G_re + xi_t.imag * self.G_im

    def b_fn(self, rho, t):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        G_t = self.G_t_fn(xi_t)
        return (k_T_t @ rho) * rho + G_t @ rho

    def integrate(self, rho_0, times, U1s=None):
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1,
                                                       dtype=np.complex)),
                                 self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) - 1)
        vec_soln = sde.euler(self.a_fn, self.b_fn, rho_0_vec, times, U1s)
        return HierarchySolution(vec_soln, self.basis, self.d_sys)

class MilsteinWavepacketHomodyneIntegrator(EulerWavepacketHomodyneIntegrator):
    def __init__(self, sparse_basis, n_max, A, xi_fn, S, L, H, r=0, mu=0,
                 hom_ang=0, field_state=None):
        super().__init__(sparse_basis, n_max, A, xi_fn, S, L, H, r, mu,
                         hom_ang, field_state)
        self.G2_ind = self.G_ind @ self.G_ind
        self.G2_re = self.G_re @ self.G_ind + self.G_ind @ self.G_re
        self.G2_im = self.G_im @ self.G_ind + self.G_ind @ self.G_im
        self.G2_reim = self.G_re @ self.G_im + self.G_im @ self.G_re
        self.G2_re2 = self.G_re @ self.G_re
        self.G2_im2 = self.G_im @ self.G_im
        self.k_T_G_ind = self.k_T_ind @ self.G_ind
        self.k_T_G_re = self.k_T_re @ self.G_ind + self.k_T_ind @ self.G_re
        self.k_T_G_im = self.k_T_im @ self.G_ind + self.k_T_ind @ self.G_im
        self.k_T_G_reim = self.k_T_re @ self.G_im + self.k_T_im @ self.G_re
        self.k_T_G_re2 = self.k_T_re @ self.G_re
        self.k_T_G_im2 = self.k_T_im @ self.G_im

    def G2_t_fn(self, xi_t):
        return (self.G2_ind + xi_t.real * self.G2_re + xi_t.imag * self.G2_im +
                xi_t.imag * xi_t.real * self.G2_reim +
                xi_t.real**2 * self.G2_re2 + xi_t.imag**2 * self.G2_im2)

    def k_T_G_t_fn(self, xi_t):
        return (self.k_T_G_ind + xi_t.real * self.k_T_G_re +
                xi_t.imag * self.k_T_G_im +
                xi_t.imag * xi_t.real * self.k_T_G_reim +
                xi_t.real**2 * self.k_T_G_re2 + xi_t.imag**2 * self.k_T_G_im2)

    def b_dx_b_fn(self, rho, t):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        G_t = self.G_t_fn(xi_t)
        G2_t = self.G2_t_fn(xi_t)
        k_T_G_t = self.k_T_G_t_fn(xi_t)
        return integ.b_dx_b(G2_t, k_T_G_t, G_t, k_T_t, rho)

    def integrate(self, rho_0, times, U1s=None):
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1,
                                                       dtype=np.complex)),
                                 self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) - 1)

        vec_soln = sde.milstein(self.a_fn, self.b_fn, self.b_dx_b_fn, rho_0_vec,
                                times, U1s)
        return HierarchySolution(vec_soln, self.basis, self.d_sys)

class EulerWavepacketJumpIntegrator(WavepacketUncondIntegrator):
    def __init__(self, sparse_basis, n_max, A, xi_fn, S, L, H, r=0, mu=0,
                 field_state=None):
        super().__init__(sparse_basis, n_max, A, xi_fn, S, L, H, r, mu)
        # Operators for the no-jump differential update
        self.F_ind = (self.wp_ind - sparse_basis.make_real_sand_matrix(
            self.L_vec, self.L_vec))
        LSA_vec = sparse_basis.vectorize(np.kron(L.conj().T @ S, self.Asq_pl))
        self.F_re = -sparse_basis.make_wiener_linear_matrix(LSA_vec)
        self.F_im = -sparse_basis.make_wiener_linear_matrix(1.j * LSA_vec)
        self.F_abs = -self.Asand
        # Operators for the jump update
        self.G_ind = sparse_basis.make_real_sand_matrix(self.L_vec, self.L_vec)
        self.G_re = 2 * sparse_basis.make_real_sand_matrix(self.L_vec,
                                                           self.SA_vec)
        self.G_im = 2 * sparse_basis.make_real_sand_matrix(self.L_vec,
                                                           1.j * self.SA_vec)
        self.G_abs = sparse_basis.make_real_sand_matrix(self.SA_vec,
                                                        self.SA_vec)
        # The functionals giving the jump rate
        if field_state is None:
            # If not specified, set initial field state to squeezed vacuum
            field_state = np.zeros(self.n_max + 1, dtype=np.complex)
            field_state[0] = 1
        field_state_proj = np.outer(field_state, field_state.conjugate())
        field_state_proj /= np.trace(field_state_proj).real
        trace_dual = np.real(sparse_basis.dualize(np.kron(self.I_sys,
                                                          field_state_proj)))
        self.k_T_ind = trace_dual @ self.G_ind
        self.k_T_re = trace_dual @ self.G_re
        self.k_T_im = trace_dual @ self.G_im
        self.k_T_abs = trace_dual @ self.G_abs

    def F_t_fn(self, xi_t):
        return (self.F_ind + xi_t.real * self.F_re + xi_t.imag * self.F_im
                + np.abs(xi_t)**2 * self.F_abs)

    def G_t_fn(self, xi_t):
        return (self.G_ind + xi_t.real * self.G_re + xi_t.imag * self.G_im
                + np.abs(xi_t)**2 * self.G_abs)

    def k_T_t_fn(self, xi_t):
        return (self.k_T_ind + xi_t.real * self.k_T_re +
                xi_t.imag * self.k_T_im + np.abs(xi_t)**2 * self.k_T_abs)

    def no_jump_fn(self, t, rho):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        F_t = self.F_t_fn(xi_t)
        return F_t @ rho + (k_T_t @ rho) * rho

    def no_jump_fn_tr_dec(self, t, rho):
        xi_t = self.xi_fn(t)
        F_t = self.F_t_fn(xi_t)
        return F_t @ rho

    def no_jump_tr_dec_D_fn(self, t, rho):
        xi_t = self.xi_fn(t)
        F_t = self.F_t_fn(xi_t)
        return F_t

    def jump_fn(self, t, rho):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        G_t = self.G_t_fn(xi_t)
        return G_t @ rho / (k_T_t @ rho)

    def jump_rate_fn(self, t, rho):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        return k_T_t @ rho

    def Dfun(self, t, rho):
        xi_t = self.xi_fn(t)
        k_T_t = self.k_T_t_fn(xi_t)
        return (self.F_t_fn(xi_t) +
                (k_T_t @ rho) * np.eye(rho.shape[0], dtype=k_T_t.dtype) +
                np.outer(rho, k_T_t))

    def integrate(self, rho_0, times, Us=None, return_meas_rec=False):
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1,
                                                       dtype=np.complex)),
                                 self.basis).real
        if Us is None:
            Us = np.random.uniform(size=len(times) - 1)

        vec_soln = sde.jump_euler(self.no_jump_fn, self.Dfun, self.jump_fn,
                                  self.jump_rate_fn, rho_0_vec, times, Us,
                                  return_dNs=return_meas_rec)
        if return_meas_rec:
            vec_soln, dNs = vec_soln
            return HierarchySolution(vec_soln, self.basis, self.d_sys), dNs

        return HierarchySolution(vec_soln, self.basis, self.d_sys)

    def integrate_tr_dec_no_jump(self, rho_0, times, solve_ivp_kwargs=None):
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': self.no_jump_tr_dec_D_fn}
        process_default_kwargs(solve_ivp_kwargs, default_solve_ivp_kwargs)
        rho_0_vec = sb.vectorize(np.kron(rho_0, np.eye(self.n_max + 1,
                                                       dtype=np.complex)),
                                 self.basis).real

        ivp_soln = solve_ivp(self.no_jump_fn_tr_dec, (times[0], times[-1]),
                             rho_0_vec, **default_solve_ivp_kwargs)
        return HierarchySolution(ivp_soln.y.T, self.basis, self.d_sys)

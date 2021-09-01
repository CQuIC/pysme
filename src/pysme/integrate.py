"""Integrate stochastic master equations in vectorized form.

    .. py:module:: integrate.py
       :synopsis: Integrate stochastic master equations in vectorized form.
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

from functools import partial
import itertools as it
import numpy as np
from scipy.integrate import solve_ivp
import sparse
import pysme.system_builder as sb
import pysme.sparse_system_builder as ssb
import pysme.sde as sde
import pysme.gellmann as gm

def process_default_kwargs(kwargs, default_kwargs):
    """Update a default kwarg dict with user-supplied values

    """
    if kwargs is None:
        kwargs = {}
    for kwarg, value in kwargs.items():
        default_kwargs[kwarg] = value

def b_dx_b(G2, k_T_G, G, k_T, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})` term for Milstein
    integration.

    Parameters
    ----------
    G2: numpy.array
        :math:`G^2`.
    k_T_G: numpy.array
        :math:`\vec{k}^TG`.
    G: numpy.array
        :math:`G`.
    k_T: numpy.array
        :math:`\vec{k}^T`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`\left(\vec{b}(\vec{\rho})\cdot
        \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})`.

    """
    k_rho_dot = np.dot(k_T, rho)
    return ((np.dot(k_T_G, rho) + 2*k_rho_dot**2)*rho +
            np.dot(G2 + 2*k_rho_dot*G, rho))

def b_dx_b_tr_dec(G2, rho):
    r"""Same as :func:`b_dx_b`, but for the linear differential equation.

    Because the nonlinear terms are discarded, this function requires fewer
    arguments.

    """
    return np.dot(G2, rho)

def b_dx_a(QG, k_T, Q, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})` term for stochastic
    integration.

    Parameters
    ----------
    QG: numpy.array
        :math:`QG`.
    k_T: numpy.array
        :math:`\vec{k}^T`.
    Q: numpy.array
        :math:`Q`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`\left(\vec{b}(\vec{\rho})\cdot
        \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})`.

    """
    return np.dot(QG + np.dot(k_T, rho)*Q, rho)

def b_dx_a_tr_dec(QG, rho):
    r"""Same as :func:`b_dx_a`, but for the linear differential equation.

    Because the nonlinear terms are discarded, this function requires fewer
    arguments.

    """
    return np.dot(QG, rho)

def a_dx_b(GQ, k_T, Q, k_T_Q, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`\left(\vec{a}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})` term for stochastic
    integration.

    GQ: numpy.array
        :math:`GQ`.
    k_T: numpy.array
        :math:`\vec{k}^T`.
    Q: numpy.array
        :math:`Q`.
    k_T_Q: numpy.array
        :math:`\vec{k}^TQ`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`\left(\vec{a}(\vec{\rho})\cdot
        \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})`.

    """
    return np.dot(GQ + np.dot(k_T, rho)*Q, rho) + np.dot(k_T_Q, rho)*rho

def a_dx_b_tr_dec(GQ, rho):
    r"""Same as :func:`a_dx_b`, but for the linear differential equation.

    Because the nonlinear terms are discarded, this function requires fewer
    arguments.

    """
    return np.dot(GQ, rho)

def a_dx_a(Q2, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`\left(\vec{a}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})` term for stochastic
    integration.

    Parameters
    ----------
    Q2: numpy.array
        :math:`Q^2`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`\left(\vec{a}(\vec{\rho})\cdot
        \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})`.

    """
    return np.dot(Q2, rho)

def b_dx_b_dx_b(G3, G2, G, k_T, k_T_G, k_T_G2, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)^2\vec{b}(\vec{\rho})` term for stochastic
    integration.

    Parameters
    ----------
    G3: numpy.array
        :math:`G^3`.
    G2: numpy.array
        :math:`G^2`.
    G: numpy.array
        :math:`G`.
    k_T: numpy.array
        :math:`\vec{k}^T`.
    k_T_G: numpy.array
        :math:`\vec{k}^TG`.
    k_T_G2: numpy.array
        :math:`\vec{k}^TG^2`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`\left(\vec{b}(\vec{\rho})\cdot
        \vec{\nabla}_{\vec{\rho}}\right)^2\vec{b}(\vec{\rho})`.

    """
    k_rho_dot = np.dot(k_T, rho)
    k_T_G_rho_dot = np.dot(k_T_G, rho)
    k_T_G2_rho_dot = np.dot(k_T_G2, rho)
    return (np.dot(G3 + 3*k_rho_dot*G2 + 3*(k_T_G_rho_dot + 2*k_rho_dot**2)*G,
                   rho) + (k_T_G2_rho_dot + 6*k_rho_dot*k_T_G_rho_dot +
                           6*k_rho_dot**3)*rho)

def b_dx_b_dx_b_tr_dec(G3, rho):
    r"""Same as :func:`b_dx_b_dx_b`, but for the linear differential equation.

    Because the nonlinear terms are discarded, this function requires fewer
    arguments.

    """
    return np.dot(G3, rho)

def b_b_dx_dx_b(G, k_T, k_T_G, rho):
    r"""A term in Taylor integration methods.

    Function to return the :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma
    b^\mu\hat{e}_\mu` term for stochastic integration.

    Parameters
    ----------
    G: numpy.array
        :math:`G`.
    k_T: numpy.array
        :math:`\vec{k}^T`.
    k_T_G: numpy.array
        :math:`\vec{k}^TG`.
    rho: numpy.array
        :math:`\rho`.

    Returns
    -------
    numpy.array
        :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma b^\mu\hat{e}_\mu`

    """
    k_rho_dot = np.dot(k_T, rho)
    k_T_G_rho_dot = np.dot(k_T_G, rho)
    return 2*(k_T_G_rho_dot + k_rho_dot**2)*(np.dot(G, rho) + k_rho_dot*rho)

class Solution:
    r"""Integrated solution to a differential equation.

    Packages the vectorized solution with the basis it is vectorized with
    respect to along with providing convenient functions for returning
    properties of the solution a user might care about (such as expectation
    value of an observable) without requiring the user to know anything about
    the particular representation used for numerical integration.

    """
    def __init__(self, vec_soln, basis):
        self.vec_soln = vec_soln
        self.basis = basis

    def get_expectations(self, observable, idx_slice=None, hermitian=True):
        r"""Calculate the expectation value of an observable for all times.

        Returns
        -------
        numpy.array
            The expectation values of an observable for all the calculated
            times.

        """
        # For an expectation, I want the trace with the observable. Dualize is
        # used for calculating traces with the adjoint of the operator, so I
        # need to preemptively adjoint here. This becomes important when taking
        # traces with non-hermitial observables.
        if idx_slice is None:
            idx_slice = np.s_[:]
        dual = sb.dualize(observable.conj().T, self.basis)
        if hermitian:
            dual = dual.real
        return np.dot(self.vec_soln[idx_slice], dual)

    def get_purities(self, idx_slice=None):
        r"""Calculate the purity of the state for all times.

        Returns
        -------
        numpy.array
            The purity :math:`\operatorname{Tr}[\rho^2]` at each calculated
            time.

        """
        if idx_slice is None:
            idx_slice = np.s_[:]
        if isinstance(self.basis, sparse.COO):
            basis_dual = np.array([np.trace(np.dot(op.conj().T, op)).real
                                   for op in self.basis.todense()])
        else:
            basis_dual = np.array([np.trace(np.dot(op.conj().T, op)).real
                                   for op in self.basis])
        return np.dot(self.vec_soln[idx_slice]**2, basis_dual)

    def get_density_matrices(self, idx_slice=None):
        r"""Represent the solution as a sequence of Hermitian arrays.

        Returns
        -------
        list of numpy.array
            The density matrix at each calculated time.

        """
        if idx_slice is None:
            idx_slice = np.s_[:]
        return np.tensordot(self.vec_soln[idx_slice], self.basis, ([-1], [0]))

    def get_density_matrices_slow(self):
        r"""Represent the solution as a sequence of Hermitian arrays.

        Returns
        -------
        list of numpy.array
            The density matrix at each calculated time.

        """
        return [sum([comp*op for comp, op in zip(state, self.basis)])
                for state in self.vec_soln]

    def save(self, outfile):
        np.savez_compressed(outfile, vec_soln=self.vec_soln,
                            basis=self.basis)

def load_solution(infile):
    loaded = np.load(infile)
    return Solution(loaded['vec_soln'], loaded['basis'])

class LindbladIntegrator:
    r"""Template class for Lindblad integrators.

    Defines the most basic constructor shared by all integrators of Lindblad
    ordinary and stochastic master equations.

    Parameters
    ----------
    Ls : [numpy.array]
        List-like collection of Lindblad operators
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `Ls`, and `H`.

    """
    def __init__(self, Ls, H, basis=None, drift_rep=None):
        dim = H.shape[0]
        self.basis = ssb.SparseBasis(dim, basis)

        if drift_rep is None:
            self.L_vecs = [self.basis.vectorize(L) for L in Ls]
            self.h_vec = self.basis.vectorize(H)
            self.Q = (self.basis.make_hamil_comm_matrix(self.h_vec)
                      + sum([self.basis.make_diff_op_matrix(L_vec)
                             for L_vec in self.L_vecs]))
        else:
            self.Q = drift_rep

    def a_fn(self, t, rho_vec):
        return np.dot(self.Q, rho_vec)

    def integrate(self, rho_0, times):
        raise NotImplementedError()

class UncondLindbladIntegrator(LindbladIntegrator):
    r"""Integrator for an unconditional Lindblad master equation.

    Parameters
    ----------
    Ls : list of numpy.array
        Collection of Lindblad operators
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `Ls`, and `H`.

    """
    def jac(self, t, rho_vec):
        return self.Q

    def integrate(self, rho_0, times, solve_ivp_kwargs=None):
        r"""Integrate the equation for a list of times with given initial
        conditions.

        :param rho_0:   The initial state of the system
        :type rho_0:    `numpy.array`
        :param times:   A sequence of time points for which to solve for rho
        :type times:    `list(real)`
        :param solve_ivp_kwargs:    kwargs for scipy.integrate.solve_ivp
        :type solve_ivp_kwargs:     dict
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         `Solution`

        """
        rho_0_vec = self.basis.vectorize(rho_0, dense=True).real
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': self.jac}
        process_default_kwargs(solve_ivp_kwargs, default_solve_ivp_kwargs)
        ivp_soln = solve_ivp(self.a_fn, (times[0], times[-1]), rho_0_vec,
                             **default_solve_ivp_kwargs)
        return Solution(ivp_soln.y.T, self.basis.basis.todense())

    def integrate_non_herm(self, rho_0, times, solve_ivp_kwargs=None):
        r"""Integrate the equation for a list of times with given initial
        conditions that may be non hermitian (useful for applications involving
        the quantum regression theorem).

        :param rho_0:   The initial state of the system
        :type rho_0:    `numpy.array`
        :param times:   A sequence of time points for which to solve for rho
        :type times:    `list(real)`
        :param solve_ivp_kwargs:    kwargs for scipy.integrate.solve_ivp
        :type solve_ivp_kwargs:     dict
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         `Solution`

        """
        rho_0_vec = self.basis.vectorize(rho_0, dense=True)
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': self.jac}
        process_default_kwargs(solve_ivp_kwargs, default_solve_ivp_kwargs)
        ivp_soln = solve_ivp(self.a_fn, (times[0], times[-1]), rho_0_vec,
                             **default_solve_ivp_kwargs)
        return Solution(ivp_soln.y.T, self.basis.basis.todense())

class UncondTimeDepLindInt(UncondLindbladIntegrator):
    r"""Integrator for an unconditional Lindblad master equation with
    time-dependent Hamiltonian and Lindblad operators.

    Parameters
    ----------
    Ls : list of lists of form [numpy.array, (numpy.array, callable), ...]
        Collection of Lindblad operators, each expressed as a list whose first
        element contains the constant part of the operator and whose subsequent
        elements are pairs whose first element is an operator and whose second
        element is the time-dependent coefficient of that operator.

        E.g., a Lindblad operator expressed algebraically as

        L_j = sum_m f_{j,m}(t) L_{j,m}

        is put into the list form

        [L_{j,0}, (L_{j,1}, f_{j,1}), ...]

    H : [numpy.array, (numpy.array, callable), ...]
        The plant Hamiltonian expressed as a list whose first element contains
        the constant part of the operator and whose subsequent elements are
        pairs whose first element is an operator and whose second element is the
        time-dependent coefficient of that operator.
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        time-independent evolution operator. Will save computation time if
        already known and don't need to calculate from `Ls`, and `H`. Sort of a
        holdover from the time-independent case right now. Can't imagine it
        being useful in its current state for the time-dependent case.

    """
    def __init__(self, Ls, H, basis=None, drift_rep=None):
        if type(H[0]) is not tuple:
            zero = np.zeros(H[0].shape, dtype=np.complex)
        else:
            zero = np.zeros(H[0][0].shape, dtype=np.complex)
        super().__init__([], zero, basis, drift_rep)

        self.time_dep_L_vecs = []
        self.time_dep_L_coeffs = []
        # Loop over different Lindblad operators
        for L in Ls:
            lvecs = []
            fs = []
            if type(L[0]) is not tuple:
                lvecs.append(self.basis.vectorize(L[0]))
                fs.append(lambda x: 1)
                for k in L[1:]:
                    lvecs.append(self.basis.vectorize(k[0]))
                    fs.append(k[1])
            else:
                for k in L:
                    lvecs.append(self.basis.vectorize(k[0]))
                    fs.append(k[1])
            self.time_dep_L_vecs.append(lvecs)
            self.time_dep_L_coeffs.append(fs)

        
        self.diagonal_L_supops = [[self.basis.make_real_comm_matrix(lvec, lvec)
                                   for lvec in lvecs]
                                  for lvecs in self.time_dep_L_vecs]

        self.re_off_diag_L_supops = [[self.basis.make_real_comm_matrix(lvec1, lvec2)
                                      + self.basis.make_real_comm_matrix(lvec2, lvec1)
                                      for lvec1, lvec2 in it.combinations(lvecs, 2)]
                                     for lvecs in self.time_dep_L_vecs]

        self.im_off_diag_L_supops = [[self.basis.make_real_comm_matrix(1.j*lvec1, lvec2)
                                      + self.basis.make_real_comm_matrix(lvec2, 1.j*lvec1)
                                      for lvec1, lvec2 in it.combinations(lvecs, 2)]
                                     for lvecs in self.time_dep_L_vecs]

        self.time_dep_H_supops = []
        self.time_dep_H_coeffs = []
        # Loop over different H factors
        if type(H[0]) is not tuple:
            hvec = self.basis.vectorize(H[0])
            self.time_dep_H_supops.append(self.basis.make_hamil_comm_matrix(hvec))
            self.time_dep_H_coeffs.append(lambda x: 1)
            for k in H[1:]:
                hvec = self.basis.vectorize(k[0])
                self.time_dep_H_supops.append(self.basis.make_hamil_comm_matrix(hvec))

                self.time_dep_H_coeffs.append(k[1])
        else:
            for k in H:
                hvec = self.basis.vectorize(k[0])
                self.time_dep_H_supops.append(self.basis.make_hamil_comm_matrix(hvec))
                self.time_dep_H_coeffs.append(k[1])

    def jac(self, t, rho_vec):
        # Need |f_{j,m}(t)|^2
        abs_coeffs = [[np.abs(fn(t))**2 for fn in fs] for fs in self.time_dep_L_coeffs]
        diag_supop = sum([sum([coeff*supop for coeff, supop in zip(coeff_list, supop_list)])
                          for coeff_list, supop_list in zip(abs_coeffs, self.diagonal_L_supops)])

        # Need re and im parts of f_{j,m}(t) f_{j,n}(t)^*
        re_coeffs = []
        im_coeffs = []
        for fn_list in self.time_dep_L_coeffs:
            re_coeff_list = []
            im_coeff_list = []
            for fn1, fn2 in it.combinations(fn_list, 2):
                coeff_val = fn1(t)*np.conj(fn2(t))
                re_coeff_list.append(np.real(coeff_val))
                im_coeff_list.append(np.imag(coeff_val))
            re_coeffs.append(re_coeff_list)
            im_coeffs.append(im_coeff_list)

        re_off_diag_supop = sum([sum([coeff*supop for coeff, supop in zip(coeff_list, supop_list)])
                                for coeff_list, supop_list
                                in zip(re_coeffs, self.re_off_diag_L_supops)])

        im_off_diag_supop = sum([sum([coeff*supop for coeff, supop in zip(coeff_list, supop_list)])
                                for coeff_list, supop_list
                                in zip(im_coeffs, self.im_off_diag_L_supops)])

        H_supop = sum([fn(t)*supop for fn, supop
                       in zip(self.time_dep_H_coeffs, self.time_dep_H_supops)])

        return diag_supop + re_off_diag_supop + im_off_diag_supop + H_supop

    def a_fn(self, t, rho_vec):
        return np.dot(self.jac(t, rho_vec), rho_vec)

class HomodyneLindbladIntegrator(UncondLindbladIntegrator):
    def __init__(self, Ls, H, meas_L_idx, basis=None, drift_rep=None, **kwargs):
        super().__init__(Ls, H, basis, drift_rep, **kwargs)
        L_meas_vec = self.L_vecs[meas_L_idx]
        L_meas = Ls[meas_L_idx]
        Id_vec = self.basis.vectorize(np.eye(L_meas.shape[0]))
        self.G = 2 * self.basis.make_real_sand_matrix(L_meas_vec, Id_vec)
        self.k_T = -2 * self.basis.dualize(L_meas).real

    def a_fn(self, t, rho_vec):
        return self.Q @ rho_vec

    def b_fn(self, t, rho_vec):
        return (self.k_T @ rho_vec) * rho_vec + self.G @ rho_vec

    def b_fn_tr_non_pres(self, t, rho_vec):
        return self.G @ rho_vec

    def dW_fn(self, dM, dt, t, rho_vec):
        '''Convert measurement increment dM to Wiener increment dW.

        '''
        return dM + np.dot(self.k_T, rho_vec) * dt

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        rho_0_vec = self.basis.vectorize(rho_0, dense=True).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.euler(self.a_fn, self.b_fn, rho_0_vec, times, U1s)
        return Solution(vec_soln, self.basis.basis.todense())

    def integrate_tr_non_pres(self, rho_0, times, U1s=None, U2s=None):
        rho_0_vec = self.basis.vectorize(rho_0, dense=True).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.euler(self.a_fn, self.b_fn_tr_non_pres, rho_0_vec, times,
                             U1s)
        # TODO: Having a difference between the basis stored by the Lindblad
        # integrators and that stored by the Gaussian integrators is also not
        # ideal. Eventually it would be nice for everything to be one of these
        # Lindblad integrators and have methods for constructing the Gaussian
        # versions from the relevant Gaussian parameters.
        return Solution(vec_soln, self.basis.basis.todense())

    def integrate_measurements(self, rho_0, times, dMs):
        r"""Integrate system evolution conditioned on a measurement record.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        dMs: numpy.array(len(times) - 1)
            Incremental measurement outcomes used to drive the SDE.

        Returns
        -------
        Solution
            The components of the vecorized :math:`\rho` for all specified
            times

        """
        rho_0_vec = self.basis.vectorize(rho_0, dense=True).real
        vec_soln = sde.meas_euler(self.a_fn, self.b_fn, self.dW_fn, rho_0_vec,
                                  times, dMs)
        return Solution(vec_soln, self.basis.basis.todense())

    def gen_meas_record(self, rho_0, times, U1s=None):
        r"""Simulate a measurement record.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise), returning a measurement record along
        with the trajectory.

        The incremental measurement outcomes making up the measurement record
        are related to the white noise increments and instantaneous state in
        the following way:

        .. math::

           dM_t=dW_t-\operatorname{tr}[(c+c^\dagger)\rho_t]

        Parameters
        ----------
        rho_0 : numpy.array of complex float
            The initial state of the system as a Hermitian matrix
        times : numpy.array of real float
            A sequence of time points for which to solve for rho
        U1s: numpy.array of real float
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories. ``U1s.shape`` is
            assumed to be ``(len(times) - 1,)``.

        Returns
        -------
        tuple of Solution and numpy.array
            The components of the vecorized :math:`\rho` for all specified
            times and an array of incremental measurement outcomes

        """
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        soln = self.integrate(rho_0, times, U1s)

        dts = times[1:] - times[:-1]
        dWs = np.sqrt(dts) * U1s
        tr_c_c_rs = np.array([-np.dot(self.k_T, rho_vec)
                              for rho_vec in soln.vec_soln[:-1]])
        dMs = dWs + tr_c_c_rs * dts

        return soln, dMs

class JumpLindbladIntegrator(UncondLindbladIntegrator):
    def __init__(self, Ls, H, meas_L_idx, basis=None, drift_rep=None, **kwargs):
        super().__init__(Ls, H, basis, drift_rep, **kwargs)
        L_meas_vec = self.L_vecs[meas_L_idx]
        self.G = -self.basis.make_real_sand_matrix(L_meas_vec, L_meas_vec)
        L_meas = Ls[meas_L_idx]
        self.kT = self.basis.dualize(L_meas.conj().T @ L_meas).real
        # Add the appropriate operator to convert the D operator into the
        # no-jump operator (-1/2) (L L† rho + rho L L†)
        self.lin_no_jump_op = self.Q + self.G
        # The jump operator without renormalization: L† rho L
        self.lin_jump_op = -self.G
        self.tr_fnctnl = self.basis.dualize(np.eye(L_meas.shape[0],
                                                   dtype=L_meas.dtype)).real

    def lin_no_jump_jac(self, t, rho_vec):
        return self.lin_no_jump_op

    def lin_no_jump_a_fn(self, t, rho_vec):
        return self.lin_no_jump_op @ rho_vec

    def jump_event(self, t, rho_vec, jump_threshold):
        return self.tr_fnctnl @ rho_vec - jump_threshold

    def integrate(self, rho_0, times, Us=None, return_meas_rec=False,
                  method='BDF'):
        rho_0_vec = self.basis.vectorize(rho_0, dense=True).real
        if Us is None:
            jump_thresholds = iter([])
        else:
            jump_thresholds = iter(Us)
        meas_rec = []
        start_idx = 0
        vec_soln_segments = []
        jump_occurred = True
        while jump_occurred and start_idx < len(times) - 1:
            try:
                jump_threshold = next(jump_thresholds)
            except StopIteration:
                # If no jump thresholds are provided or we run out, generate new
                # random thresholds.
                jump_threshold = np.random.uniform()
            jump = partial(self.jump_event, jump_threshold=jump_threshold)
            jump.terminal = True
            ivp_soln = solve_ivp(self.lin_no_jump_a_fn,
                                 [times[start_idx], times[-1]], rho_0_vec,
                                 t_eval=times[start_idx:], events=jump,
                                 method=method, jac=self.lin_no_jump_jac)
            traces = np.tensordot(ivp_soln.y, self.tr_fnctnl, [0, 0])
            vec_soln_segments.append(ivp_soln.y.T / traces[:,np.newaxis])
            if ivp_soln.status == 1:
                # A jump occurred
                meas_rec.append(ivp_soln.t_events[0][0])
                start_idx += len(ivp_soln.t)
                rho_0_vec = self.lin_jump_op @ ivp_soln.y.T[-1]
                rho_0_vec = rho_0_vec / (self.tr_fnctnl @ rho_0_vec)
            else:
                jump_occurred = False
        soln = Solution(np.vstack(vec_soln_segments),
                        self.basis.basis.todense())
        return (soln, meas_rec) if return_meas_rec else soln

class GaussIntegrator:
    r"""Template class for Gaussian integrators.

    Defines the most basic constructor shared by all integrators of Gaussian
    ordinary and stochastic master equations.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.

    """
    def __init__(self, c_op, M_sq, N, H, basis=None, drift_rep=None, **kwargs):
        if basis is None:
            d = c_op.shape[0]
            self.basis = gm.get_basis(d)
        else:
            self.basis = basis

        if drift_rep is None:
            self.Q = sb.construct_Q(c_op, M_sq, N, H, self.basis[:-1])
        else:
            self.Q = drift_rep

    def a_fn(self, t, rho_vec):
        return np.dot(self.Q, rho_vec)

    def integrate(self, rho_0, times):
        raise NotImplementedError()

class UncondGaussIntegrator(GaussIntegrator):
    r"""Integrator for an unconditional Gaussian master equation.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.

    """
    def jac(self, t, rho_vec):
        return self.Q

    def integrate(self, rho_0, times, method='BDF'):
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
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        ivp_soln = solve_ivp(self.a_fn, (times[0], times[-1]),
                             rho_0_vec, method=method, t_eval=times,
                             jac=self.jac)
        return Solution(ivp_soln.y.T, self.basis)

    def integrate_non_herm(self, rho_0, times, solve_ivp_kwargs=None):
        r"""Integrate the equation for a list of times with given initial
        conditions that may be non hermitian (useful for applications involving
        the quantum regression theorem).

        :param rho_0:   The initial state of the system
        :type rho_0:    `numpy.array`
        :param times:   A sequence of time points for which to solve for rho
        :type times:    `list(real)`
        :param method:  The integration method for `scipy.integrate.solve_ivp`
                        to use.
        :type method:   String
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         `Solution`

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis)
        default_solve_ivp_kwargs = {'method': 'BDF',
                                    't_eval': times,
                                    'jac': self.jac}
        process_default_kwargs(solve_ivp_kwargs, default_solve_ivp_kwargs)
        ivp_soln = solve_ivp(self.a_fn, (times[0], times[-1]),
                             rho_0_vec, **default_solve_ivp_kwargs)
        return Solution(ivp_soln.y.T, self.basis)

class Strong_0_5_HomodyneIntegrator(GaussIntegrator):
    r"""Template class for integrators of strong order >= 0.5.

    Defines the most basic constructor shared by all integrators of Gaussian
    homodyne stochastic master equations of strong order >= 0.5.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """
    def __init__(self, c_op, M_sq, N, H, basis=None, drift_rep=None,
                 diffusion_reps=None, **kwargs):
        super(Strong_0_5_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis, drift_rep,
                                                            **kwargs)

        if diffusion_reps is None:
            self.G, self.k_T = sb.construct_G_k_T(c_op, M_sq, N, H,
                                                  self.basis[:-1])
        else:
            self.G = diffusion_reps['G']
            self.k_T = diffusion_reps['k_T']

    def b_fn(self, t, rho_vec):
        return np.dot(self.k_T, rho_vec)*rho_vec + np.dot(self.G, rho_vec)

    def b_fn_tr_dec(self, t, rho_vec):
        # For use with a trace-decreasing linear integration function
        return np.dot(self.G, rho_vec)

    def dW_fn(self, dM, dt, t, rho_vec):
        '''Convert measurement increment dM to Wiener increment dW.

        '''
        return dM + np.dot(self.k_T, rho_vec) * dt

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        raise NotImplementedError()

    def integrate_tr_dec(self, rho_0, times, U1s=None, U2s=None):
        raise NotImplementedError()

    def gen_meas_record(self, rho_0, times, U1s=None):
        r"""Simulate a measurement record.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise), returning a measurement record along
        with the trajectory.

        The incremental measurement outcomes making up the measurement record
        are related to the white noise increments and instantaneous state in
        the following way:

        .. math::

           dM_t=dW_t-\operatorname{tr}[(c+c^\dagger)\rho_t]

        Parameters
        ----------
        rho_0 : numpy.array of complex float
            The initial state of the system as a Hermitian matrix
        times : numpy.array of real float
            A sequence of time points for which to solve for rho
        U1s: numpy.array of real float
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories. ``U1s.shape`` is
            assumed to be ``(len(times) - 1,)``.

        Returns
        -------
        tuple of Solution and numpy.array
            The components of the vecorized :math:`\rho` for all specified
            times and an array of incremental measurement outcomes

        """
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        soln = self.integrate(rho_0, times, U1s)

        dts = times[1:] - times[:-1]
        dWs = np.sqrt(dts) * U1s
        tr_c_c_rs = np.array([-np.dot(self.k_T, rho_vec)
                              for rho_vec in soln.vec_soln[:-1]])
        dMs = dWs + tr_c_c_rs * dts

        return soln, dMs

class Strong_1_0_HomodyneIntegrator(Strong_0_5_HomodyneIntegrator):
    r"""Template class for integrators of strong order >= 1.

    Defines the most basic constructor shared by all integrators of Gaussian
    homodyne stochastic master equations of strong order >= 1.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """
    def __init__(self, c_op, M_sq, N, H, basis=None, drift_rep=None,
                 diffusion_reps=None, **kwargs):
        super(Strong_1_0_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis, drift_rep,
                                                            diffusion_reps,
                                                            **kwargs)
        self.k_T_G = np.dot(self.k_T, self.G)
        self.G2 = np.dot(self.G, self.G)

class Strong_1_5_HomodyneIntegrator(Strong_1_0_HomodyneIntegrator):
    r"""Template class for integrators of strong order >= 1.5.

    Defines the most basic constructor shared by all integrators of Gaussian
    homodyne stochastic master equations of strong order >= 1.5.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """
    def __init__(self, c_op, M_sq, N, H, basis=None, drift_rep=None,
                 diffusion_reps=None, **kwargs):
        super(Strong_1_5_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis, drift_rep,
                                                            diffusion_reps,
                                                            **kwargs)
        self.G3 = np.dot(self.G2, self.G)
        self.Q2 = np.dot(self.Q, self.Q)
        self.QG = np.dot(self.Q, self.G)
        self.GQ = np.dot(self.G, self.Q)
        self.k_T_G2 = np.dot(self.k_T, self.G2)
        self.k_T_Q = np.dot(self.k_T, self.Q)

class EulerHomodyneIntegrator(Strong_0_5_HomodyneIntegrator):
    r"""Euler integrator for the conditional Gaussian master equation.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise).

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.euler(self.a_fn, self.b_fn, rho_0_vec, times, U1s)
        return Solution(vec_soln, self.basis)

    def integrate_tr_dec(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise). Integrates the linear equation,
        which ignores the tr[(c + cdag) rho] rho term and therefore decreases
        the trace.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.euler(self.a_fn, self.b_fn_tr_dec, rho_0_vec, times,
                             U1s)
        return Solution(vec_soln, self.basis)

    def integrate_measurements(self, rho_0, times, dMs):
        r"""Integrate system evolution conditioned on a measurement record.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        dMs: numpy.array(len(times) - 1)
            Incremental measurement outcomes used to drive the SDE.

        Returns
        -------
        Solution
            The components of the vecorized :math:`\rho` for all specified
            times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real

        vec_soln = sde.meas_euler(self.a_fn, self.b_fn, self.dW_fn, rho_0_vec,
                                  times, dMs)
        return Solution(vec_soln, self.basis)

class MilsteinHomodyneIntegrator(Strong_1_0_HomodyneIntegrator):
    r"""Milstein integrator for the conditional Gaussian master equation.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """
    def b_dx_b_fn(self, t, rho_vec):
        # TODO: May want this to be defined by the constructor to facilitate
        # numba optimization.
        return b_dx_b(self.G2, self.k_T_G, self.G, self.k_T, rho_vec)

    def b_dx_b_fn_tr_dec(self, t, rho_vec):
        # Used by trace decreasing interator
        return b_dx_b_tr_dec(self.G2, rho_vec)

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise).

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.milstein(self.a_fn, self.b_fn, self.b_dx_b_fn, rho_0_vec,
                                times, U1s)
        return Solution(vec_soln, self.basis)

    def integrate_tr_dec(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise). Integrates the linear equation,
        which ignores the tr[(c + cdag) rho] rho term and therefore decreases
        the trace.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.milstein(self.a_fn, self.b_fn_tr_dec,
                                self.b_dx_b_fn_tr_dec, rho_0_vec, times, U1s)
        return Solution(vec_soln, self.basis)

    def integrate_measurements(self, rho_0, times, dMs):
        r"""Integrate system evolution conditioned on a measurement record.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        dMs: numpy.array(len(times) - 1)
            Incremental measurement outcomes used to drive the SDE.

        Returns
        -------
        Solution
            The components of the vecorized :math:`\rho` for all specified
            times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real

        vec_soln = sde.meas_milstein(self.a_fn, self.b_fn, self.b_dx_b_fn,
                                     self.dW_fn, rho_0_vec, times, dMs)
        return Solution(vec_soln, self.basis)

class Taylor_1_5_HomodyneIntegrator(Strong_1_5_HomodyneIntegrator):
    r"""Order 1.5 Taylor ntegrator for the conditional Gaussian master equation.

    Parameters
    ----------
    c_op : numpy.array
        The coupling operator
    M_sq : complex float
        The squeezing parameter
    N : non-negative float
        The thermal parameter
    H : numpy.array
        The plant Hamiltonian
    basis : list of numpy.array, optional
        The Hermitian basis to vectorize the operators in terms of (with the
        component proportional to the identity in last place). If no basis is
        provided the generalized Gell-Mann basis will be used.
    drift_rep : numpy.array, optional
        The real matrix Q that acts on the vectorized rho as the deterministic
        evolution operator. Will save computation time if already known and
        don't need to calculate from `c_op`, `M_sq`, `N`, and `H`.
    diffusion_reps : dict of numpy.array, optional
        The real matrix G and row vector k_T that act on the vectorized rho as
        the stochastic evolution operator.  Will save computation time if
        already known and don't need to calculate from `c_op`, `M_sq`, and `N`.

    """
    def a_fn(self, rho_vec):
        return np.dot(self.Q, rho_vec)

    def b_fn(self, rho_vec):
        return np.dot(self.k_T, rho_vec)*rho_vec + np.dot(self.G, rho_vec)

    def b_fn_tr_dec(self, rho_vec):
        return np.dot(self.G, rho_vec)

    def b_dx_b_fn(self, rho_vec):
        return b_dx_b(self.G2, self.k_T_G, self.G, self.k_T, rho_vec)

    def b_dx_b_fn_tr_dec(self, rho_vec):
        return b_dx_b_tr_dec(self.G2, rho_vec)

    def b_dx_a_fn(self, rho_vec):
        return b_dx_a(self.QG, self.k_T, self.Q, rho_vec)

    def b_dx_a_fn_tr_dec(self, rho_vec):
        return b_dx_a_tr_dec(self.QG, rho_vec)

    def a_dx_b_fn(self, rho_vec):
        return a_dx_b(self.GQ, self.k_T, self.Q, self.k_T_Q, rho_vec)

    def a_dx_b_fn_tr_dec(self, rho_vec):
        return a_dx_b_tr_dec(self.GQ, rho_vec)

    def a_dx_a_fn(self, rho_vec):
        return a_dx_a(self.Q2, rho_vec)

    def b_dx_b_dx_b_fn(self, rho_vec):
        return b_dx_b_dx_b(self.G3, self.G2, self.G, self.k_T, self.k_T_G,
                           self.k_T_G2, rho_vec)

    def b_dx_b_dx_b_fn_tr_dec(self, rho_vec):
        return b_dx_b_dx_b_tr_dec(self.G3, rho_vec)

    def b_b_dx_dx_b_fn(self, rho_vec):
        return b_b_dx_dx_b(self.G, self.k_T, self.k_T_G, rho_vec)

    def b_b_dx_dx_b_fn_tr_dec(self, rho_vec):
        return 0

    def b_b_dx_dx_a_fn(self, rho_vec):
        return 0

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise).

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)
        if U2s is None:
            U2s = np.random.randn(len(times) -1)

        vec_soln = sde.time_ind_taylor_1_5(self.a_fn, self.b_fn, self.b_dx_b_fn,
                                           self.b_dx_a_fn, self.a_dx_b_fn,
                                           self.a_dx_a_fn, self.b_dx_b_dx_b_fn,
                                           self.b_b_dx_dx_b_fn,
                                           self.b_b_dx_dx_a_fn,
                                           rho_0_vec, times, U1s, U2s)
        return Solution(vec_soln, self.basis)

    def integrate_tr_dec(self, rho_0, times, U1s=None, U2s=None):
        r"""Integrate the initial value problem.

        Integrate for a sequence of times with a given initial condition (and
        optionally specified white noise). Integrates the linear equation,
        which ignores the tr[(c + cdag) rho] rho term and therefore decreases
        the trace.

        Parameters
        ----------
        rho_0: numpy.array
            The initial state of the system
        times: numpy.array
            A sequence of time points for which to solve for rho
        U1s: numpy.array(len(times) - 1)
            Samples from a standard-normal distribution used to construct
            Wiener increments :math:`\Delta W` for each time interval. Multiple
            rows may be included for independent trajectories.
        U2s: numpy.array(len(times) - 1)
            Unused, included to make the argument list uniform with
            higher-order integrators.

        Returns
        -------
        Solution
            The state of :math:`\rho` for all specified times

        """
        rho_0_vec = sb.vectorize(rho_0, self.basis).real
        if U1s is None:
            U1s = np.random.randn(len(times) -1)
        if U2s is None:
            U2s = np.random.randn(len(times) -1)

        vec_soln = sde.time_ind_taylor_1_5(self.a_fn, self.b_fn_tr_dec,
                                           self.b_dx_b_fn_tr_dec,
                                           self.b_dx_a_fn_tr_dec,
                                           self.a_dx_b_fn_tr_dec,
                                           self.a_dx_a_fn,
                                           self.b_dx_b_dx_b_fn_tr_dec,
                                           self.b_b_dx_dx_b_fn_tr_dec,
                                           self.b_b_dx_dx_a_fn,
                                           rho_0_vec, times, U1s, U2s)
        return Solution(vec_soln, self.basis)

class TrDecMilsteinHomodyneIntegrator(MilsteinHomodyneIntegrator):
    """Milstein integrator that does not preserve trace.

    Only does the linear evolution, where the decrease in trace now encodes
    the likelihood of the particular trajectory. Might be more appropriate to
    include as a particulare `integrate_tr_dec` method in preëxisting integrator
    classes.

    """
    def __init__(self, c_op, M_sq, N, H, basis=None, drift_rep=None,
                 diffusion_reps=None, **kwargs):
        super(TrDecMilsteinHomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                              basis, drift_rep,
                                                              diffusion_reps,
                                                              **kwargs)
        self.k_T = 0
        self.k_T_G = np.zeros(self.G.shape)

class IntegratorFactory:
    r"""Factory that pre-computes things for other integrators.

    A class that pre-computes some of the things in common to a family of
    integrators one wants to construct instances of.

    Parameters
    ----------
    IntClass : Class of integrator
        A class inheriting from :class:`GaussIntegrator` that you want to
        create many instances of.
    precomp_data
        Data needed by the `IntClass` constructor common across all integrators
        in the family of interest in the form that can be passed to the
        `parameter_fn` as `precomp_data`.
    parameter_fn
        Function that takes the parameters defining the instance of the family
        to be generated and the precomputed data and returns ``**kwargs`` to pass
        to the constructor of `IntClass`.

    """
    def __init__(self, IntClass, precomp_data, parameter_fn):
        self.precomp_data = precomp_data
        self.parameter_fn = parameter_fn
        self.IntClass = IntClass

    def make_integrator(self, params):
        """Create a new integrator.

        Create a new instance of `IntClass`, feeding `params` and
        `precomp_data` to the constructor.

        """
        constructor_kwargs = self.parameter_fn(params, self.precomp_data)
        return self.IntClass(**constructor_kwargs)

class QuasiMarkoff2LvlIntegrator(UncondLindbladIntegrator):
    r'''QuasiMarkoff equation for squeezed fields

    Assumes the memory time associated with the squeezing is very much less than
    the atomic decay rate but can be comparable to or even longer than the Rabi
    period. See Equations 1 through 25 in reference:

    [YB96]  Influence of squeezing bandwidths on resonance fluorescence
            G. Yeoman and S. M. Barnett, Journal of Modern Optics 43, 2037 (1996).
            https://doi.org/10.1080/09500349608232870

    Parameters
    ----------
    gamma : non-negative float
        Atomic linewidth.
    N_A : non-negative float
        The thermal parameter evaluated at atomic transition frequency. See
        the text below Eqn 14 in [YB96].
    N_Om : non-negative float
        The thermal parameter evaluated at atomic transition frequency minus
        the Rabi freqency. See the text below Eqn 14 in [YB96].
    M_A : complex float
        The squeezing parameter evaluated at atomic transition frequency. See
        the text below Eqn 14 in [YB96].
    M_Om : complex float
        The squeezing parameter evaluated at atomic transition frequency minus
        the Rabi freqency. See the text below Eqn 14 in [YB96].
    Delta_AL : float
        Atomic frequency minus carrier (laser) frequency i.e,
        Delta_AL = omega_A - omega_L.
    Omega : float
        The Rabi frequency.
    phi_L : float
        Carrier (laser) frequency phase, see Eq 4 in [YB96].
    F_A : complex float
        Eq. 18 in [YB96].
    G_A : complex float
        Eq. 19 in [YB96].

    '''
    def __init__(self, gamma, N_A, N_Om, M_A, M_Om, Delta_AL, Omega, phi_L, F_A, G_A):
        dim = 2
        basis = ssb.SparseBasis(dim)
        self.basis = basis
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex)
        sp = (sx + 1j*sy)/2
        sm = (sx - 1j*sy)/2
        # Eqn 16 and 17 in [YB96]
        Sm = sm*np.exp(-1j*phi_L)
        Sp = sp*np.exp(1j*phi_L)
        Y = 1j*sz@(Sp + Sm)
        sz_vec = self.basis.vectorize(sz)
        Sm_vec = self.basis.vectorize(Sm)
        Sp_vec = self.basis.vectorize(Sp)
        Y_vec = self.basis.vectorize(Y)
        H_vec = ((Omega/2 + 1j*F_A)*(Sp_vec + Sm_vec)
                 + (Delta_AL/2)*sz_vec + G_A*Y_vec)

        self.Q = -gamma/4*(
                (1 + N_A)*(basis.make_double_comm_matrix(Sm_vec, 1)
                           - 2*basis.make_diff_op_matrix(Sm_vec))
                + N_A*(basis.make_double_comm_matrix(Sp_vec, 1)
                       - 2*basis.make_diff_op_matrix(Sp_vec))
                + (1 + N_Om)*(-basis.make_double_comm_matrix(Sm_vec, 1)
                            - 2*basis.make_diff_op_matrix(Sm_vec))
                + N_Om*(-basis.make_double_comm_matrix(Sp_vec, 1)
                        - 2*basis.make_diff_op_matrix(Sp_vec))
                - 2*basis.make_double_comm_matrix(Sm_vec, M_A*np.exp(-2j*phi_L))
                + 2*basis.make_real_comm_matrix(M_A*np.exp(-2j*phi_L)*Sp_vec,
                                                Sp_vec)
                + 2*basis.make_real_comm_matrix(M_A.conj()*np.exp(2j*phi_L)
                                                *Sm_vec, Sm_vec)
                - 2*basis.make_double_comm_matrix(Sm_vec,
                                                  M_Om*np.exp(-2j*phi_L))
                - 2*basis.make_real_comm_matrix(M_Om*np.exp(-2j*phi_L)*Sp_vec,
                                                Sp_vec)
                - 2*basis.make_real_comm_matrix(M_Om.conj()*np.exp(2j*phi_L)
                                                *Sm_vec, Sm_vec))
        self.Q += basis.make_hamil_comm_matrix(H_vec)
        self.Q += -2*basis.make_real_sand_matrix(F_A*(Sp_vec - Sm_vec), sz_vec)
        self.Q += -2*basis.make_real_sand_matrix(G_A*(Sp_vec + Sm_vec), sz_vec)

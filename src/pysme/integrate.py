"""
.. py:module:: integrate.py
   :synopsis: Integrate stochastic master equations in vectorized form.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
import pysme.system_builder as sb
import pysme.sde as sde
import pysme.gellmann as gm

def b_dx_b(G2, k_T_G, G, k_T, rho):
    r'''Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})` term for Milstein
    integration.

    :param G2:          :math:`G^2`.
    :param k_T_G:       :math:`\vec{k}^TG`.
    :param G:           :math:`G`.
    :param k_T:         :math:`\vec{k}^T`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`\left(\vec{b}(\vec{\rho})\cdot
                        \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})`.

    '''
    k_rho_dot = np.dot(k_T, rho)
    return (np.dot(k_T_G, rho) + 2*k_rho_dot**2)*rho + \
            np.dot(G2 + 2*k_rho_dot*G, rho)

def b_dx_a(QG, k_T, Q, rho):
    r'''Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})` term for stochastic
    integration.

    :param QG:          :math:`QG`.
    :param k_T:         :math:`\vec{k}^T`.
    :param Q:           :math:`Q`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`\left(\vec{b}(\vec{\rho})\cdot
                        \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})`.

    '''
    return np.dot(QG + np.dot(k_T, rho)*Q, rho)

def a_dx_b(GQ, k_T, Q, k_T_Q, rho):
    r'''Function to return the :math:`\left(\vec{a}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})` term for stochastic
    integration.

    :param GQ:          :math:`GQ`.
    :param k_T:         :math:`\vec{k}^T`.
    :param Q:           :math:`Q`.
    :param k_T_Q:       :math:`\vec{k}^TQ`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`\left(\vec{a}(\vec{\rho})\cdot
                        \vec{\nabla}_{\vec{\rho}}\right)\vec{b}(\vec{\rho})`.

    '''
    return np.dot(GQ + np.dot(k_T, rho)*Q, rho) + np.dot(k_T_Q, rho)*rho

def a_dx_a(Q2, rho):
    r'''Function to return the :math:`\left(\vec{a}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})` term for stochastic
    integration.

    :param Q2:          :math:`Q^2`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`\left(\vec{a}(\vec{\rho})\cdot
                        \vec{\nabla}_{\vec{\rho}}\right)\vec{a}(\vec{\rho})`.

    '''
    return np.dot(Q2, rho)

def b_dx_b_dx_b(G3, G2, G, k_T, k_T_G, k_T_G2, rho):
    r'''Function to return the :math:`\left(\vec{b}(\vec{\rho})\cdot
    \vec{\nabla}_{\vec{\rho}}\right)^2\vec{b}(\vec{\rho})` term for stochastic
    integration.

    :param G3:          :math:`G^3`.
    :param G2:          :math:`G^2`.
    :param G:           :math:`G`.
    :param k_T:         :math:`\vec{k}^T`.
    :param k_T_G:       :math:`\vec{k}^TG`.
    :param k_T_G2:      :math:`\vec{k}^TG^2`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`\left(\vec{b}(\vec{\rho})\cdot
                        \vec{\nabla}_{\vec{\rho}}\right)^2\vec{b}(\vec{\rho})`.

    '''
    k_rho_dot = np.dot(k_T, rho)
    k_T_G_rho_dot = np.dot(k_T_G, rho)
    k_T_G2_rho_dot = np.dot(k_T_G2, rho)
    return (np.dot(G3 + 3*k_rho_dot*G2 + 3*(k_T_G_rho_dot + 2*k_rho_dot**2)*G,
                   rho) + (k_T_G2_rho_dot + 6*k_rho_dot*k_T_G_rho_dot +
                           6*k_rho_dot**3)*rho)

def b_b_dx_dx_b(G, k_T, k_T_G, rho):
    r'''Function to return the
    :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma b^\mu\hat{e}_\mu` term for
    stochastic integration.

    :param G:           :math:`G`.
    :param k_T:         :math:`\vec{k}^T`.
    :param k_T_G:       :math:`\vec{k}^TG`.
    :param rho:         :math:`\rho`.
    :returns:           :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma
                        b^\mu\hat{e}_\mu`

    '''
    k_rho_dot = np.dot(k_T, rho)
    k_T_G_rho_dot = np.dot(k_T_G, rho)
    return 2*(k_T_G_rho_dot + k_rho_dot**2)*(np.dot(G, rho) + k_rho_dot*rho)

class Solution:
    r'''Integrated solution to an ordinary or stochastic differential
    equation. Packages the vectorized solution with the basis
    it is vectorized with respect to along with providing convenient functions
    for returning properties of the solution a user might care about (such as
    expectation value of an observable) without requiring the user to know
    anything about the particular representation used for numerical integration.

    '''
    def __init__(self, vec_soln, basis):
        self.vec_soln = vec_soln
        self.basis = basis

    def get_expectations(self, observable):
        r'''Return the expectation values of an observable for all the
        calculated times.

        '''
        dual = np.array([comp.real
                         for comp in sb.dualize(observable, self.basis)])
        return [np.dot(dual, state)[0] for state in self.vec_soln]

class GaussIntegrator:
    r'''Template class with most basic constructor shared by all integrators
    of Gaussian ordinary and stochastic master equations.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place). If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''
    def __init__(self, c_op, M_sq, N, H, basis=None):
        if basis is None:
            d = c_op.shape[0]
            self.basis = gm.get_basis(d)
        else:
            self.basis = basis
        self.Q = (N + 1)*sb.diffusion_op(c_op, self.basis[:-1]) + \
                 N*sb.diffusion_op(c_op.conj().T, self.basis[:-1]) + \
                 sb.double_comm_op(c_op, M_sq, self.basis[:-1]) + \
                 sb.hamiltonian_op(H, self.basis[:-1])

    def integrate(self, rho_0, times):
        raise NotImplementedError()

class UncondGaussIntegrator(GaussIntegrator):
    r'''Integrator for an unconditional Gaussian master equation.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''
    def a_fn(self, rho, t):
        return np.dot(self.Q, rho)

    def Dfun(self, rho, t):
        return self.Q

    def integrate(self, rho_0, times):
        r'''Integrate the equation for a list of times with given initial
        conditions.

        :param rho_0:   The initial state of the system
        :type rho_0:    numpy.array
        :param times:   A sequence of time points for which to solve for rho
        :type times:    list(real)
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         list(numpy.array)

        '''
        rho_0_vec = [comp.real for comp in sb.vectorize(rho_0, self.basis)]
        vec_soln = odeint(self.a_fn, rho_0_vec, times, Dfun=self.Dfun)
        return Solution(vec_soln, self.basis)

class Strong_0_5_HomodyneIntegrator(GaussIntegrator):
    r'''Template class for integrators of the Gaussian homodyne stochastic
    master equation of strong order >= 0.5.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''
    def __init__(self, c_op, M_sq, N, H, basis=None):
        super(Strong_0_5_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis)
        self.G, self.k_T = sb.weiner_op(((N + M_sq.conjugate() + 1)*c_op -
                                         (N + M_sq)*c_op.conj().T)/
                                        np.sqrt(2*(M_sq.real + N) + 1),
                                        self.basis[:-1])

class Strong_1_0_HomodyneIntegrator(Strong_0_5_HomodyneIntegrator):
    r'''Template class for integrators of the Gaussian homodyne stochastic
    master equation of strong order >= 1.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''
    def __init__(self, c_op, M_sq, N, H, basis=None):
        super(Strong_1_0_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis)
        self.k_T_G = np.dot(self.k_T, self.G)
        self.G2 = np.dot(self.G, self.G)

class Strong_1_5_HomodyneIntegrator(Strong_1_0_HomodyneIntegrator):
    r'''Template class for integrators of the Gaussian homodyne stochastic
    master equation of strong order >= 1.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''
    def __init__(self, c_op, M_sq, N, H, basis=None):
        super(Strong_1_5_HomodyneIntegrator, self).__init__(c_op, M_sq, N, H,
                                                            basis)
        self.G3 = np.dot(self.G2, self.G)
        self.Q2 = np.dot(self.Q, self.Q)
        self.QG = np.dot(self.Q, self.G)
        self.GQ = np.dot(self.G, self.Q)
        self.k_T_G2 = np.dot(self.k_T, self.G2)
        self.k_T_Q = np.dot(self.k_T, self.Q)

class MilsteinHomodyneIntegrator(Strong_1_0_HomodyneIntegrator):
    r'''Integrator for the conditional Gaussian master equation that uses
    Milstein integration.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    '''

    def a_fn(self, rho, t):
        return np.dot(self.Q, rho)

    def b_fn(self, rho, t):
        return np.dot(self.k_T, rho)*rho + np.dot(self.G, rho)

    def b_dx_b_fn(self, rho, t):
        return b_dx_b(self.G2, self.k_T_G, self.G, self.k_T, rho)

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        r'''Integrate for a sequence of times with a given initial condition
        (and optionally specified white noise).

        :param rho_0:   The initial state of the system
        :type rho_0:    numpy.array
        :param times:   A sequence of time points for which to solve for rho
        :type times:    list(real)
        :param U1s:     Samples from a standard-normal distribution used to
                        construct Wiener increments :math:`\Delta W` for each
                        time interval. Multiple rows may be included for
                        independent trajectories.
        :type U1s:      numpy.array(len(times) - 1)
        :param U2s:     Unused, included to make the argument list uniform with
                        higher-order integrators.
        :type U2s:      numpy.array(len(times) - 1)
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         list(numpy.array)

        '''
        rho_0_vec = np.array([[comp.real]
                              for comp in sb.vectorize(rho_0, self.basis)])
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.milstein(self.a_fn, self.b_fn, self.b_dx_b_fn, rho_0_vec,
                                times, U1s)
        return Solution(vec_soln, self.basis)

class FaultyMilsteinHomodyneIntegrator(MilsteinHomodyneIntegrator):
    r'''Integrator included to test if grid convergence could identify an error
    I originally had in my Milstein integrator (missing a factor of 1/2 in front
    of the term that's added to the Euler scheme)

    '''

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        rho_0_vec = np.array([[comp.real]
                              for comp in vectorize(rho_0, self.basis)])
        if U1s is None:
            U1s = np.random.randn(len(times) -1)

        vec_soln = sde.faulty_milstein(self.a_fn, self.b_fn, self.b_dx_b_fn,
                                       rho_0_vec, times, U1s)
        return Solution(vec_soln, self.basis)

class Taylor_1_5_HomodyneIntegrator(Strong_1_5_HomodyneIntegrator):
    r"""Integrator for the conditional Gaussian master equation that uses
    strong order 1.5 Taylor integration.

    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param M_sq:    The squeezing parameter
    :type M_sq:     complex
    :param N:       The thermal parameter
    :type N:        positive real
    :param H:       The plant Hamiltonian
    :type H:        numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place) If no basis is provided the generalized Gell-Mann
                    basis will be used.
    :type basis:    list(numpy.array)

    """
    def a_fn(self, rho):
        return np.dot(self.Q, rho)

    def b_fn(self, rho):
        return np.dot(self.k_T, rho)*rho + np.dot(self.G, rho)

    def b_dx_b_fn(self, rho):
        return b_dx_b(self.G2, self.k_T_G, self.G, self.k_T, rho)

    def b_dx_a_fn(self, rho):
        return b_dx_a(self.QG, self.k_T, self.Q, rho)

    def a_dx_b_fn(self, rho):
        return a_dx_b(self.GQ, self.k_T, self.Q, self.k_T_Q, rho)

    def a_dx_a_fn(self, rho):
        return a_dx_a(self.Q2, rho)

    def b_dx_b_dx_b_fn(self, rho):
        return b_dx_b_dx_b(self.G3, self.G2, self.G, self.k_T, self.k_T_G,
                           self.k_T_G2, rho)

    def b_b_dx_dx_b_fn(self, rho):
        return b_b_dx_dx_b(self.G, self.k_T, self.k_T_G, rho)

    def b_b_dx_dx_a_fn(self, rho):
        return 0

    def integrate(self, rho_0, times, U1s=None, U2s=None):
        r'''Integrate for a sequence of times with a given initial condition
        (and optionally specified white noise).

        :param rho_0:   The initial state of the system
        :type rho_0:    numpy.array
        :param times:   A sequence of time points for which to solve for rho
        :type times:    list(real)
        :param U1s:     Samples from a standard-normal distribution used to
                        construct Wiener increments :math:`\Delta W` for each
                        time interval. Multiple rows may be included for
                        independent trajectories.
        :type U1s:      numpy.array(N, len(times) - 1)
        :param U2s:     Samples from a standard-normal distribution used to
                        construct multiple-Ito increments :math:`\Delta Z` for
                        each time interval. Multiple rows may be included for
                        independent trajectories.
        :type U2s:      numpy.array(N, len(times) - 1)
        :returns:       The components of the vecorized :math:`\rho` for all
                        specified times
        :rtype:         list(numpy.array)

        '''
        rho_0_vec = np.array([[comp.real]
                              for comp in sb.vectorize(rho_0, self.basis)])
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

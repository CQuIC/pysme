"""
.. py:module:: integrate.py
   :synopsis: Integrate stochastic master equations in vectorized form.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
from pysme.system_builder import *
from pysme.sde import *
from math import sqrt

def uncond_vac_integrate(rho_0, c_op, basis, times):
    r"""Integrate an unconditional vacuum master equation.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
                    (with the component proportional to the identity in last
                    place)
    :type basis:    list(numpy.array)
    :param times:   A sequence of time points for which to solve for rho
    :type times:    list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times
    :rtype:         list(numpy.array)

    """

    rho_0_vec = [comp.real for comp in vectorize(rho_0, basis)]
    diff_mat = diffusion_op(c_op, basis[:-1])
    
    return odeint(lambda rho_vec, t: np.dot(diff_mat, rho_vec), rho_0_vec,
            times, Dfun=(lambda rho_vec, t: diff_mat))

def uncond_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times):
    r"""Integrate an unconditional Gaussian master equation.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
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
                    place)
    :type basis:    list(numpy.array)
    :param times:   A sequence of time points for which to solve for rho
    :type times:    list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times
    :rtype:         list(numpy.array)

    """

    rho_0_vec = [comp.real for comp in vectorize(rho_0, basis)]
    diff_mat = (N + 1)*diffusion_op(c_op, basis[:-1]) + \
            N*diffusion_op(c_op.conj().T, basis[:-1]) + \
            double_comm_op(c_op, M_sq, basis[:-1]) + hamiltonian_op(H,
                    basis[:-1])
    
    return odeint(lambda rho_vec, t: np.dot(diff_mat, rho_vec), rho_0_vec,
            times, Dfun=(lambda rho_vec, t: diff_mat))

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
    return np.dot(GQ + np.dot(k_T, rho)*Q, rho) + np.dot(k_T_Q, rho)

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
    return (np.dot(G3 + 3*k_rho_dot*G2 + 3*(k_T_G_rho_dot + 2*k_rho_dot)*G,
                   rho) + (k_T_G2_rho_dot + 6*k_rho_dot*k_T_G_rho_dot +
                           6*k_rho_dot**3)*rho)

def homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times):
    r"""Integrate the conditional Gaussian master equation using Milstein
    integration.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
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
                    place)
    :type basis:    list(numpy.array)
    :param times:   A sequence of time points for which to solve for rho
    :type times:    list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times
    :rtype:         list(numpy.array)

    """

    rho_0_vec = np.array([[comp.real] for comp in vectorize(rho_0, basis)])
    Q = (N + 1)*diffusion_op(c_op, basis[:-1]) + \
        N*diffusion_op(c_op.conj().T, basis[:-1]) + \
        double_comm_op(c_op, M_sq, basis[:-1]) + hamiltonian_op(H, basis[:-1])
    G, k_T = weiner_op(((N + M_sq.conjugate() + 1)*c_op -
                           (N + M_sq)*c_op.conj().T)/
                           sqrt(2*(M_sq.real + N) + 1), basis[:-1])

    a_fn = lambda rho, t: np.dot(Q, rho)
    b_fn = lambda rho, t: np.dot(k_T, rho)*rho + np.dot(G, rho)
    k_T_G = np.dot(k_T, G)
    G2 = np.dot(G, G)
    b_dx_b_fn = lambda rho, t: b_dx_b(G2, k_T_G, G, k_T, rho)

    return milstein(a_fn, b_fn, b_dx_b_fn, rho_0_vec, times,
                    np.random.randn(len(times) - 1))

def homodyne_gauss_integrate_1_5(rho_0, c_op, M_sq, N, H, basis, times):
    r"""Integrate the conditional Gaussian master equation using order 1.5
    Taylor integration.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
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
                    place)
    :type basis:    list(numpy.array)
    :param times:   A sequence of time points for which to solve for rho
    :type times:    list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times
    :rtype:         list(numpy.array)

    """

    rho_0_vec = np.array([[comp.real] for comp in vectorize(rho_0, basis)])
    Q = (N + 1)*diffusion_op(c_op, basis[:-1]) + \
        N*diffusion_op(c_op.conj().T, basis[:-1]) + \
        double_comm_op(c_op, M_sq, basis[:-1]) + hamiltonian_op(H, basis[:-1])
    G, k_T = weiner_op(((N + M_sq.conjugate() + 1)*c_op -
                           (N + M_sq)*c_op.conj().T)/
                           sqrt(2*(M_sq.real + N) + 1), basis[:-1])

    a_fn = lambda rho: np.dot(Q, rho)
    b_fn = lambda rho: np.dot(k_T, rho)*rho + np.dot(G, rho)
    G2 = np.dot(G, G)
    G3 = np.dot(G2, G)
    Q2 = np.dot(Q, Q)
    QG = np.dot(Q, G)
    GQ = np.dot(G, Q)
    k_T_G = np.dot(k_T, G)
    k_T_G2 = np.dot(k_T, G2)
    k_T_Q = np.dot(k_T, Q)
    b_dx_b_fn = lambda rho: b_dx_b(G2, k_T_G, G, k_T, rho)
    b_dx_a_fn = lambda rho: b_dx_a(QG, k_T, Q, rho)
    a_dx_b_fn = lambda rho: a_dx_b(GQ, k_T, Q, k_T_Q, rho)
    a_dx_a_fn = lambda rho: a_dx_a(Q2, rho)
    b_dx_b_dx_b_fn = lambda rho: b_dx_b_dx_b(G3, G2, G, k_T, k_T_G, k_T_G2, rho)

    return time_ind_taylor_1_5(a_fn, b_fn, b_dx_b_fn, b_dx_a_fn, a_dx_b_fn,
                               a_dx_a_fn, b_dx_b_dx_b_fn, rho_0_vec, times,
                               np.random.randn(len(times) - 1),
                               np.random.randn(len(times) - 1))

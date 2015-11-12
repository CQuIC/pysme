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

def milstein_correction_fn(G_sq, k_T_prime, G, k_T, rho):
    k_rho_dot = np.dot(k_T, rho)
    return (np.dot(k_T_prime, rho) + 2*k_rho_dot**2)*rho + \
            np.dot(G_sq + 2*k_rho_dot*G, rho)

def homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times):
    r"""Integrate the conditional Gaussian master equation.

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
    a_mat = (N + 1)*diffusion_op(c_op, basis[:-1]) + \
            N*diffusion_op(c_op.conj().T, basis[:-1]) + \
            double_comm_op(c_op, M_sq, basis[:-1]) + hamiltonian_op(H,
                    basis[:-1])
    G_mat, k_T = weiner_op(((N + M_sq.conjugate() + 1)*c_op -
                           (N + M_sq)*c_op.conj().T)/
                           sqrt(2*(M_sq.real + N) + 1), basis[:-1])

    a_fn = lambda rho, t: np.dot(a_mat, rho)
    b_fn = lambda rho, t: np.dot(k_T, rho)*rho + np.dot(G_mat, rho)
    k_T_prime = np.dot(k_T, G_mat)
    G_sq = np.dot(G_mat, G_mat)
    b_dx_b_fn = lambda rho, t: milstein_correction_fn(G_sq, k_T_prime, G_mat,
                                                      k_T, rho)

    return milstein(a_fn, b_fn, b_dx_b_fn, rho_0_vec, times,
                    np.random.randn(len(times) - 1))

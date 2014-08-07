"""
.. py:module:: integrate.py
   :synopsis: Integrate stochastic master equations in vectorized form.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
from pysme.system_builder import *

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

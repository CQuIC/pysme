"""
.. module:: integrate.py
   :synopsis: Integrate stochastic master equations in vectorized form.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from scipy.integrate import odeint
from pysme.system_builder import *

def uncond_vac_integrate(rho_0, c_op, basis, times):
    """Integrate an unconditional vacuum master equation.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
    :type basis:    list(numpy.array)
    :param times:   A sequence of time points for which to solve for rho

    """
    rho_0_vec = [comp.real for comp in vectorize(rho_0, basis)]
    diff_mat = diffusion_op(c_op, basis)
    
    return odeint(lambda rho_vec, t: np.dot(diff_mat, rho_vec), rho_0_vec,
            times, Dfun=(lambda rho_vec, t: diff_mat))

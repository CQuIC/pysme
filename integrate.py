"""
.. module:: integrate.py
   :synopsis: Integrate stochastic master equations in vectorized form.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from pysme.system_builder import *

def uncond_vac_integrate(rho_0, c_op, basis, dt, steps):
    """Integrate an unconditional vacuum master equation.

    :param rho_0:   The initial state of the system
    :type rho_0:    numpy.array
    :param c_op:    The coupling operator
    :type c_op:     numpy.array
    :param basis:   The Hermitian basis to vectorize the operators in terms of
    :type basis:    list(numpy.array)
    :param dt:      The timestep size
    :type dt:       real
    :param steps:   The number of timesteps to integrate over
    :type steps:    positive integer

    """
    rho_0_vec = np.array([[comp.real] for comp in vectorize(rho_0, basis)])
    rho_vecs = [rho_0_vec]
    diff_mat = diffusion_op(c_op, basis)
    for n in range(steps):
        rho_vecs.append(rho_vecs[-1] + dt*np.dot(diff_mat, rho_vecs[-1]))

    return rho_vecs

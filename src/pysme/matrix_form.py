"""Code for manipulating expressions in matrix form

    .. module:: matrix_form.py
       :synopsis:Code for manipulating expressions in matrix form
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np

def comm(A, B):
    '''Calculate the commutator of two matrices

    '''
    return A @ B - B @ A

def D(c, rho):
    '''Calculate the application of the diffusion superoperator D[c] to rho

    '''
    c_dag = c.conjugate().T
    return c @ rho @ c_dag - (c_dag @ c @ rho + rho @ c_dag @ c) / 2

def euler_integrate(rho_0, rho_dot_fn, times):
    rhos = [rho_0]
    dts = np.diff(times)
    for dt, t in zip(dts, times[:-1]):
        rho_dot = rho_dot_fn(rhos[-1], t)
        rhos.append(rhos[-1] + dt * rho_dot)
    return rhos

def get_expectations(rhos, observable):
    return np.array([np.trace(observable @ rho).real for rho in rhos])

"""Code for using projection-operator methods

    .. module:: projector_method.py
       :synopsis: Code for using projection-operator methods.
    .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import pysme.matrix_form as mf

class CompositeState(list):
    '''Composite state class that supports addition and scalar multiplication

    Useful for using in an euler integration routine.

    '''
    def __add__(self, other):
        return CompositeState([self_component + other_component
                               for self_component, other_component
                               in zip(self, other)])

    def __rmul__(self, other):
        return CompositeState([other * self_component
                               for self_component in self])

def D_non_herm(c, rho_tilde, xi):
    r'''Diffusion-operator-like function for a non-hermitian argument

    .. math::

       \mathtt{D\_non\_herm}(c,\tilde{\rho},\xi)=c(\xi\tilde{\rho}^\dagger
       +\xi^*\tilde{\rho})c^\dagger-(\xi\tilde{\rho}^\dagger)c^\dagger c
       -c^\dagger c(\xi^*\tilde{\rho})

    '''
    c_dag = c.conjugate().T
    rho_tilde_dag = c.conjugate().T
    xi_star = xi.conjugate()
    return (c @ (xi * rho_tilde_dag + xi_star * rho_tilde) @ c_dag -
            xi * rho_tilde_dag @ c_dag @ c - xi_star * c_dag @ c @ rho_tilde)

def double_comm_non_herm(c, rho_tilde, xi, mu):
    r'''Double-commutator-like function for a non-hermitian argument

    .. math::

       \mathtt{double\_comm}(c,\tilde{\rho},\xi,\mu)
       =e^{-2i\mu}[c,[c,\xi^*\tilde{\rho}]]

    '''
    xi_star = xi.conjugate()
    return np.exp(-2.j * mu) * mf.comm(c, mf.comm(c, xi_star * rho_tilde))

def rho_dot_conv(rho_combined, c, r, mu, gamma, xi):
    '''Compute the derivative for the convolutionful master equation

    rho_combined = (rho, rho_tilde)

    '''
    c_dag = c.conjugate().T
    rho = rho_combined[0]
    rho_tilde = rho_combined[1]
    rho_tilde_dag = rho_tilde.conjugate().T
    xi_star = xi.conjugate()
    rho_dot = gamma * (mf.D(c, rho) - np.sinh(r) * np.cosh(r) *
        (double_comm_non_herm(c_dag, rho_tilde_dag, xi_star, -mu) +
            double_comm_non_herm(c, rho_tilde, xi, mu)) + np.sinh(r)**2 *
        (D_non_herm(c, rho_tilde, xi) +
            D_non_herm(c_dag, rho_tilde_dag, xi_star)))
    rho_tilde_dot = xi * rho
    return CompositeState([rho_dot, rho_tilde_dot])

def euler_integrate_conv(rho_0, c, r, mu, gamma, xi_fn, times):
    '''Integrate the convolutionful master equation with Euler integration

    '''
    rho_combined_0 = CompositeState([rho_0,
                                     np.zeros(rho_0.shape, dtype=rho_0.dtype)])
    return mf.euler_integrate(rho_combined_0,
                              lambda rho_combined, t:
                              rho_dot_conv(rho_combined, c, r, mu, gamma,
                                           xi_fn(t)),
                              times)

def rho_dot_convless(rho_combined, c, r, mu, gamma, xi):
    '''Compute the derivative for the convolutionless master equation

    rho_combined = (rho, xi_tilde)

    '''
    c_dag = c.conjugate().T
    rho = rho_combined[0]
    xi_tilde = rho_combined[1]
    xi_tilde_star = xi_tilde.conjugate()
    xi_star = xi.conjugate()
    rho_dot = gamma * (mf.D(c, rho) - np.sinh(r) * np.cosh(r) *
        (double_comm_non_herm(c_dag, xi_tilde_star * rho, xi_star, -mu) +
            double_comm_non_herm(c, xi_tilde * rho, xi, mu)) + np.sinh(r)**2 *
        (D_non_herm(c, xi_tilde * rho, xi) +
            D_non_herm(c_dag, xi_tilde_star * rho, xi_star)))
    xi_tilde_dot = xi
    return CompositeState([rho_dot, xi_tilde_dot])

def euler_integrate_convless(rho_0, c, r, mu, gamma, xi_fn, times):
    '''Integrate the convolutionless master equation with Euler integration

    '''
    rho_combined_0 = CompositeState([rho_0, 0.j])
    return mf.euler_integrate(rho_combined_0,
                              lambda rho_combined, t:
                              rho_dot_convless(rho_combined, c, r, mu, gamma,
                                               xi_fn(t)),
                              times)

"""
.. module:: sde.py
   :synopsis: Numerical integration techniques
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np

def milstein(drift, diffusion, bx_dx_b, X0, ts, dws):
    r"""Integrate a system of ordinary stochastic differential equations subject
    to scalar noise:

    .. math::

       d\vec{X}=\vec{a}(\vec{X},t)\,dt+\vec{b}(\vec{X},t)\,dW_t

    Uses the Milstein method:

    .. math::

       \vec{X}_{i+1}=\vec{X}_i+\vec{a}(\vec{X}_i,t_i)\Delta t_i+
       vec{b}(\vec{X}_i,t_i)\Delta W_i+
       \frac{1}{2}\left(\vec{b}(\vec{X}_i,t_i)\cdot\vec{\nabla}_{\vec{X}}\right)
       \vec{b}(\vec{X}_i,t_i)\left((\Delta W_i)^2-\Delta t_i\right)

    :param drift:           Computes the drift coefficient
                            :math:`\vec{a}(\vec{X},t)`
    :type drift:            callable(X, t)
    :param diffusion:       Computes the diffusion coefficient
                            :math:`\vec{b}(\vec{X},t)` at t0
    :type diffusion:        callable(X, t)
    :param bx_dx_b:         Computes the correction coefficient
                            :math:`\left(\vec{b}(\vec{X},t)\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X},t)`
    :type diffusion_prime:  callable(X, t)
    :param X0:              Initial condition on X
    :type X0:               array
    :param ts:              A sequence of time points for which to solve for y.
                            The initial value point should be the first element
                            of this sequence.
    :type ts:               array
    :param dws:             Normalized Weiner increments for each time step
                            (i.e. samples from a Gaussian distribution with mean
                            0 and variance 1).
    :type dws:              array, shape (len(t) - 1)
    :return:                Array containint the value of X for each desired
                            time in t, with the initial value `X0` in the first
                            row.
    :rtype:                 numpy.array, shape (len(t), len(X0))

    """

    dts = [tf - ti for tf, ti in zip(ts[1:], ts[:-1])]
    # Scale the Weiner increments to the time increments.
    sqrtdts = np.sqrt(dts)
    dWs = np.product(np.array([sqrtdts, dws]), axis=0)

    X = [np.array(X0)]

    for t, dt, dW in zip(ts[:-1], dts, dWs):
        X.append(X[-1] + drift(X[-1], t)*dt + diffusion(X[-1], t)*dW +
                 bx_dx_b(X[-1], t)*(dW**2 - dt))

    return X

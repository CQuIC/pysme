"""
.. module:: sde.py
   :synopsis: Numerical integration techniques
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np

def milstein(drift, diffusion, b_dx_b, X0, ts, Us):
    r"""Integrate a system of ordinary stochastic differential equations subject
    to scalar noise:

    .. math::

       d\vec{X}=\vec{a}(\vec{X},t)\,dt+\vec{b}(\vec{X},t)\,dW_t

    Uses the Milstein method:

    .. math::

       \vec{X}_{i+1}=\vec{X}_i+\vec{a}(\vec{X}_i,t_i)\Delta t_i+
       \vec{b}(\vec{X}_i,t_i)\Delta W_i+
       \frac{1}{2}\left(\vec{b}(\vec{X}_i,t_i)\cdot\vec{\nabla}_{\vec{X}}\right)
       \vec{b}(\vec{X}_i,t_i)\left((\Delta W_i)^2-\Delta t_i\right)

    where :math:`\Delta W_i=U_i\sqrt{\Delta t}`, :math:`U` being a normally
    distributed random variable with mean 0 and variance 1.

    :param drift:           Computes the drift coefficient
                            :math:`\vec{a}(\vec{X},t)`
    :type drift:            callable(X, t)
    :param diffusion:       Computes the diffusion coefficient
                            :math:`\vec{b}(\vec{X},t)`
    :type diffusion:        callable(X, t)
    :param b_dx_b:          Computes the correction coefficient
                            :math:`\left(\vec{b}(\vec{X},t)\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X},t)`
    :type b_dx_b:           callable(X, t)
    :param X0:              Initial condition on X
    :type X0:               array
    :param ts:              A sequence of time points for which to solve for X.
                            The initial value point should be the first element
                            of this sequence.
    :type ts:               array
    :param Us:              Normalized Weiner increments for each time step
                            (i.e. samples from a Gaussian distribution with mean
                            0 and variance 1).
    :type Us:               array, shape=(len(t) - 1)
    :return:                Array containing the value of X for each desired
                            time in t, with the initial value `X0` in the first
                            row.
    :rtype:                 numpy.array, shape=(len(t), len(X0))

    """

    dts = [tf - ti for tf, ti in zip(ts[1:], ts[:-1])]
    # Scale the Weiner increments to the time increments.
    sqrtdts = np.sqrt(dts)
    dWs = np.product(np.array([sqrtdts, Us]), axis=0)

    X = [np.array(X0)]

    for t, dt, dW in zip(ts[:-1], dts, dWs):
        X.append(X[-1] + drift(X[-1], t)*dt + diffusion(X[-1], t)*dW +
                 b_dx_b(X[-1], t)*(dW**2 - dt)/2)

    return X

def time_ind_taylor_1_5(drift, diffusion, b_dx_b, b_dx_a, a_dx_b, a_dx_a,
                        b_dx_b_dx_b, b_b_dx_dx_b, b_b_dx_dx_a,
                        X0, ts, U1s, U2s):
    r"""Integrate a system of ordinary stochastic differential equations with
    time-independent coefficients subject to scalar noise:

    .. math::

       d\vec{X}=\vec{a}(\vec{X})\,dt+\vec{b}(\vec{X})\,dW_t

    Uses an order 1.5 Taylor method:

    .. math::

       \begin{align}
       \vec{X}_{i+1}&=\vec{X}_i+\vec{a}(\vec{X}_i)\Delta t_i+
       \vec{b}(\vec{X}_i)\Delta W_i+
       \frac{1}{2}\left(\vec{b}(\vec{X}_i)\cdot\vec{\nabla}_{\vec{X}}
       \right)\vec{b}(\vec{X}_i)\left((\Delta W_i)^2-\Delta t_i\right)+ \\
       &\quad\left(\vec{b}(\vec{X}_i)\cdot\vec{\nabla}_{\vec{X}}
       \right)\vec{a}(\vec{X}_i)\Delta Z_i+\left(\vec{a}(\vec{X}_i)\cdot
       \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X}_i)\left(
       \Delta W_i\Delta t_i-\Delta Z_i\right)+ \\
       &\quad\frac{1}{2}\left(\vec{a}(\vec{X}_i)\cdot
       \vec{\nabla}_{\vec{X}}
       \right)\vec{a}(\vec{X}_i)\Delta t_i^2+\frac{1}{2}\left(
       \vec{b}(\vec{X}_i)\cdot\vec{\nabla}_{\vec{X}}
       \right)^2\,\vec{b}(\vec{X}_i)\left(\frac{1}{3}(\Delta W_i)^2-
       \Delta t_i\right)\Delta W_i
       \end{align}

    :param drift:           Computes the drift coefficient
                            :math:`\vec{a}(\vec{X})`
    :type drift:            callable(X)
    :param diffusion:       Computes the diffusion coefficient
                            :math:`\vec{b}(\vec{X})`
    :type diffusion:        callable(X)
    :param b_dx_b:          Computes the coefficient
                            :math:`\left(\vec{b}(\vec{X})\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X})`
    :type b_dx_b:           callable(X)
    :param b_dx_a:          Computes the coefficient
                            :math:`\left(\vec{b}(\vec{X})\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{a}(\vec{X})`
    :type b_dx_a:           callable(X)
    :param a_dx_b:          Computes the coefficient
                            :math:`\left(\vec{a}(\vec{X})\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X})`
    :type a_dx_b:           callable(X)
    :param a_dx_a:          Computes the coefficient
                            :math:`\left(\vec{a}(\vec{X})\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{a}(\vec{X})`
    :type a_dx_a:           callable(X)
    :param b_dx_b_dx_b:     Computes the coefficient
                            :math:`\left(\vec{b}(\vec{X})\cdot
                            \vec{\nabla}_{\vec{X}}\right)^2\vec{b}(\vec{X})`
    :type b_dx_b_dx_b:      callable(X)
    :param b_b_dx_dx_b:     Computes
                            :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma
                            b^\mu\hat{e}_\mu`.
    :type b_dx_b_dx_b:      callable(X)
    :param b_b_dx_dx_a:     Computes
                            :math:`b^\nu b^\sigma\partial_\nu\partial_\sigma
                            a^\mu\hat{e}_\mu`.
    :type b_dx_b_dx_a:      callable(X)
    :param X0:              Initial condition on X
    :type X0:               array
    :param ts:              A sequence of time points for which to solve for X.
                            The initial value point should be the first element
                            of this sequence.
    :type ts:               array
    :param U1s:             Normalized Weiner increments for each time step
                            (i.e. samples from a Gaussian distribution with mean
                            0 and variance 1).
    :type U1s:              array, shape=(len(t) - 1)
    :param U2s:             Normalized Weiner increments for each time step
                            (i.e. samples from a Gaussian distribution with mean
                            0 and variance 1).
    :type U2s:              array, shape=(len(t) - 1)
    :return:                Array containing the value of X for each desired
                            time in t, with the initial value `X0` in the first
                            row.
    :rtype:                 numpy.array, shape=(len(t), len(X0))

    """

    dts = ts[1:] - ts[:-1]
    sqrtdts = np.sqrt(dts)
    dWs = U1s*sqrtdts
    dZs = (U1s + U2s/np.sqrt(3))*sqrtdts*dts/2

    Xs = [np.array(X0)]

    for t, dt, dW, dZ in zip(ts[:-1], dts, dWs, dZs):
        X = Xs[-1]
        Xs.append(X + drift(X)*dt + diffusion(X)*dW + b_dx_b(X)*(dW**2 - dt)/2 +
                  b_dx_a(X)*dZ + (a_dx_b(X)+b_b_dx_dx_b(X)/2)*(dW*dt - dZ) +
                  (a_dx_a(X)+b_b_dx_dx_a(X)/2)*dt**2/2 +
                  b_dx_b_dx_b(X)*(dW**2/3 - dt)*dW/2)

    return Xs

def faulty_milstein(drift, diffusion, b_dx_b, X0, ts, Us):
    r"""Integrate a system of ordinary stochastic differential equations subject
    to scalar noise:

    .. math::

       d\vec{X}=\vec{a}(\vec{X},t)\,dt+\vec{b}(\vec{X},t)\,dW_t

    Uses a faulty Milstein method (i.e. missing the factor of 1/2 in the term
    added to Euler integration):

    .. math::

       \vec{X}_{i+1}=\vec{X}_i+\vec{a}(\vec{X}_i,t_i)\Delta t_i+
       \vec{b}(\vec{X}_i,t_i)\Delta W_i+
       \left(\vec{b}(\vec{X}_i,t_i)\cdot\vec{\nabla}_{\vec{X}}\right)
       \vec{b}(\vec{X}_i,t_i)\left((\Delta W_i)^2-\Delta t_i\right)

    where :math:`\Delta W_i=U_i\sqrt{\Delta t}`, :math:`U` being a normally
    distributed random variable with mean 0 and variance 1.

    :param drift:           Computes the drift coefficient
                            :math:`\vec{a}(\vec{X},t)`
    :type drift:            callable(X, t)
    :param diffusion:       Computes the diffusion coefficient
                            :math:`\vec{b}(\vec{X},t)`
    :type diffusion:        callable(X, t)
    :param b_dx_b:          Computes the correction coefficient
                            :math:`\left(\vec{b}(\vec{X},t)\cdot
                            \vec{\nabla}_{\vec{X}}\right)\vec{b}(\vec{X},t)`
    :type b_dx_b:           callable(X, t)
    :param X0:              Initial condition on X
    :type X0:               array
    :param ts:              A sequence of time points for which to solve for X.
                            The initial value point should be the first element
                            of this sequence.
    :type ts:               array
    :param Us:              Normalized Weiner increments for each time step
                            (i.e. samples from a Gaussian distribution with mean
                            0 and variance 1).
    :type Us:               array, shape=(len(t) - 1)
    :return:                Array containing the value of X for each desired
                            time in t, with the initial value `X0` in the first
                            row.
    :rtype:                 numpy.array, shape=(len(t), len(X0))

    """

    dts = [tf - ti for tf, ti in zip(ts[1:], ts[:-1])]
    # Scale the Weiner increments to the time increments.
    sqrtdts = np.sqrt(dts)
    dWs = np.product(np.array([sqrtdts, Us]), axis=0)

    X = [np.array(X0)]

    for t, dt, dW in zip(ts[:-1], dts, dWs):
        X.append(X[-1] + drift(X[-1], t)*dt + diffusion(X[-1], t)*dW +
                 b_dx_b(X[-1], t)*(dW**2 - dt))

    return X

'''Functions for testing convergence rates using grid convergence

'''

import numpy as np
from pysme.integrate import *

def double_increments(times, U1s, U2s=None):
    r'''Take a list of times (assumed to be evenly spaced) and standard-normal
    random variables used to define the Ito integrals on the intervals and
    return the equivalent lists for doubled time intervals. The new
    standard-normal random variables are defined in terms of the old ones by

    .. math:

       \begin{align}
       \tilde{U}_{1,n}&=\frac{U_{1,n}+U_{1,n+1}}{\sqrt{2}} \\
       \tilde{U}_{2,n}&=\frac{\sqrt{3}}{2}\frac{U_{1,n}-U_{1,n+1}}{\sqrt{2}}
                        +\frac{1}{2}\frac{U_{2,n}+U_{2,n+1}}{\sqrt{2}}
       \end{align}

    :param times:   List of evenly spaced times defining an even number of
                    time intervals.
    :type times:    numpy.array
    :param U1s:     Samples from a standard-normal distribution used to
                    construct Wiener increments :math:`\Delta W` for each time
                    interval.
    :type U1s:      numpy.array(len(times) - 1)
    :param U2s:     Samples from a standard-normal distribution used to
                    construct multiple-Ito increments :math:`\Delta Z` for each
                    time interval.
    :type U2s:      numpy.array(len(times) - 1)
    :returns:       Times sampled at half the frequency and the modified
                    standard-normal-random-variable samples for the new
                    intervals. If ``U2s=None``, only new U1s are returned.
    :rtype:         (numpy.array(len(times)//2 + 1),
                     numpy.array(len(times)//2)[, numpy.array(len(times)//2]))

    '''

    new_times = times[::2]
    even_U1s = U1s[::2]
    odd_U1s = U1s[1::2]
    new_U1s = (even_U1s + odd_U1s)/np.sqrt(2)

    if U2s is None:
        return new_times, new_U1s
    else:
        even_U2s = U2s[::2]
        odd_U2s = U2s[1::2]
        new_U2s = (np.sqrt(3)*(even_U1s - odd_U1s) +
                   even_U2s + odd_U2s)/(2*np.sqrt(2))
        return new_times, new_U1s, new_U2s

def milstein_grid_convergence(rho_0, c_op, M_sq, N, H, basis, times, Us=None):
    r"""Calculate the same trajectory for time increments :math:`\Delta t`,
    :math:`2\Delta t`, and :math:`4\Delta t` using Milstein integration.

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
    :param times:   A sequence of time points for which to solve for rho. The
                    length should be such that (len(times) - 1)%4 == 0.
    :type times:    list(real)
    :param Us:      A sequence of normalized Wiener increments (samples from a
                    normal distribution with mean 0 and variance 1). If None,
                    then this function will generate its own samples. The length
                    should be len(times) - 1.
    :type Us:       list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times, first for Milstein and then for Taylor 1.5
    :rtype:         (list(numpy.array), list(numpy.array))

    """

    increments = len(times) - 1
    if Us is None:
        Us = np.random.randn(increments)

    # Calculate times and random variables for the double and quadruple
    # intervals
    times_2, Us_2 = double_increments(times, Us)
    times_4, Us_4 = double_increments(times_2, Us_2)

    rhos = homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times, Us)
    rhos_2 = homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times_2,
                                      Us_2)
    rhos_4 = homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis, times_4,
                                      Us_4)

    return [(rhos, times), (rhos_2, times_2), (rhos_4, times_4)]

def faulty_milstein_grid_convergence(rho_0, c_op, M_sq, N, H, basis, times,
                                     Us=None):
    r"""Calculate the same trajectory for time increments :math:`\Delta t`,
    :math:`2\Delta t`, and :math:`4\Delta t` using faulty Milstein integration
    (i.e. missing the factor of 1/2 in the term added to Euler integration).

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
    :param times:   A sequence of time points for which to solve for rho. The
                    length should be such that (len(times) - 1)%4 == 0.
    :type times:    list(real)
    :param Us:      A sequence of normalized Wiener increments (samples from a
                    normal distribution with mean 0 and variance 1). If None,
                    then this function will generate its own samples. The length
                    should be len(times) - 1.
    :type Us:       list(real)
    :returns:       The components of the vecorized :math:`\rho` for all
                    specified times, first for Milstein and then for Taylor 1.5
    :rtype:         (list(numpy.array), list(numpy.array))

    """

    increments = len(times) - 1
    if Us is None:
        Us = np.random.randn(increments)

    # Calculate times and random variables for the double and quadruple
    # intervals
    times_2, Us_2 = double_increments(times, Us)
    times_4, Us_4 = double_increments(times_2, Us_2)

    rhos = faulty_homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis,
                                           times, Us)
    rhos_2 = faulty_homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis,
                                             times_2, Us_2)
    rhos_4 = faulty_homodyne_gauss_integrate(rho_0, c_op, M_sq, N, H, basis,
                                             times_4, Us_4)

    return [(rhos, times), (rhos_2, times_2), (rhos_4, times_4)]

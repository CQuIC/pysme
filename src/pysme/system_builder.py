"""
.. module:: system_builder.py
   :synopsis: Build the matrix representation of the system of coupled
              real-valued stochastic differential equations.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import itertools as it

def recur_dot(mats):
    """Perform numpy.dot on a list in a right-associative manner.

    """
    if len(mats) == 0:
        return 1
    elif len(mats) == 1:
        return mats[0]
    else:
        return np.dot(mats[0], recur_dot(mats[1:]))

def norm_squared(operator):
    """Returns the square of the `Frobenius norm
    <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_ of the
    operator.

    :param operator:    The operator for which to calculate the squared norm
    :type operator:     numpy.array
    :returns:           The square of the norm of the operator
    :return type:       Positive real

    """

    return abs(np.tensordot(operator.conj().T, operator,
               axes=[[1, 0], [0, 1]]))

def vectorize(operator, basis):
    """Vectorize an operator in a particular operator basis.

    :param operator:    The operator to vectorize
    :type operator:     numpy.array
    :param basis:       The basis to vectorize the operator in
    :type basis:        list(numpy.array)
    :returns:           The vector components
    :rtype:             numpy.array

    """
    return np.array([np.tensordot(basis_el.conj().T, operator,
                                  axes=[[1, 0], [0, 1]]) /
                     norm_squared(basis_el) for basis_el in basis])

def dualize(operator, basis):
    r'''Take an operator to its dual vectorized form in a particular operator
    basis.
    
    Designed to work in conjunction with ``vectorize`` so that, given an
    orthogonal basis :math:`\{\Lambda^m\}` where
    :math:`\operatorname{Tr}[{\Lambda^m}^\dagger\Lambda^n]\propto\delta_{mn}`,
    the dual action of an operator :math:`A` on another operator :math:`B`
    interpreted as :math:`\operatorname{Tr}[A^\dagger B]` can be easily
    calculated by ``sum([a*b for a, b in zip(dualize(A), vectorize(B))]`` (in
    other words it becomes an ordinaty dot product in this particular
    representation).

    :param operator:    The operator to vectorize
    :type operator:     numpy.array
    :param basis:       The basis to vectorize the operator in
    :type basis:        list(numpy.array)
    :returns:           The vector components
    :rtype:             numpy.array

    '''
    return np.array([np.tensordot(basis_el, operator.conj().T,
                                  axes=[[1, 0], [0, 1]])
                     for basis_el in basis])

def op_calc_setup(coupling_op, M_sq, N, H, partial_basis):
    """Handle the repeated tasks performed every time a superoperator matrix is
    computed.

    """

    # Add the identity to the end of the basis to complete it (important for
    # some tests for the identity to be the last basis element).
    basis = partial_basis + [np.eye(*partial_basis[0].shape)]

    dim = len(basis)

    # Vectorization of the coupling operator
    C_vector = vectorize(coupling_op, basis)
    H_vector = vectorize(H, basis)

    double_prods = {(i, j): np.dot(basis[i], basis[j])
                    for i, j in it.product(range(dim), repeat=2)}

    triple_prods = {(i, j, k): np.dot(basis[i], double_prods[j, k]) -
                    0.5 * np.dot(basis[k], double_prods[i, j]) -
                    0.5 * np.dot(basis[j], double_prods[k, i])
                    for j, k, i in it.product(range(dim), repeat=3)}

    # Square norm of basis elements
    basis_norms_sq = [norm_squared(basis[i]) for i in range(dim)]

    return {'dim': dim, 'C_vector': C_vector, 'double_prods': double_prods,
            'triple_prods': triple_prods, 'basis': basis, 'M_sq': M_sq, 'N': N,
            'H_vector': H_vector, 'basis_norms_sq': basis_norms_sq}

def construct_Q(coupling_op, M_sq, N, H, partial_basis):
    common_dict = op_calc_setup(coupling_op, M_sq, N, H, partial_basis)
    D_c = diffusion_op(**common_dict)
    conjugate_dict = common_dict.copy()
    conjugate_dict['C_vector'] = common_dict['C_vector'].conjugate()
    D_c_dag = diffusion_op(**conjugate_dict)
    E = double_comm_op(**common_dict)
    F = hamiltonian_op(**common_dict)

    Q = (N + 1) * D_c + N * D_c_dag + E + F

    return Q


def construct_G_k_T(c_op, M_sq, N, H, partial_basis):
    common_dict = op_calc_setup((N + M_sq.conjugate() + 1) * c_op -
                                (N + M_sq) * c_op.conj().T, M_sq, N, H,
                                partial_basis)

    G, k_T = weiner_op(**common_dict)

    return G, k_T


def diffusion_op(dim, C_vector, triple_prods, basis_norms_sq, basis, **kwargs):
    r"""Return a matrix :math:`D` such that when :math:`\rho` is vectorized the
    expression

    .. math::

       \frac{d\rho}{dt}=\mathcal{D}[c]\rho=c\rho c^\dagger-
       \frac{1}{2}(c^\dagger c\rho+\rho c^\dagger c)

    can be calculated by:

    .. math::

       \frac{d\vec{\rho}}{dt}=D\vec{\rho}

    Vectorization is done according to the order prescribed in *basis*, with the
    component proportional to identity in the last place.
    
    :param coupling_op: The operator :math:`c` in matrix form
    :type coupling_op:  numpy.array
    :param basis:       An almost complete (minus identity), Hermitian,
                        traceless, orthogonal basis for the operators (does not
                        need to be normalized).
    :type basis:        list(numpy.array)
    :returns:           The matrix :math:`D` operating on a vectorized density
                        operator
    :rtype:             numpy.array

    """
    D_matrix = np.zeros((dim, dim)) # The matrix to return

    # TODO: Write tests to catch the inappropriate use of conjugate() without T

    # Construct lists of basis elements up to the current basis element for
    # doing the sum of the non-symmetric part of each element.

    col_symm_ops = [sum([abs(C_vector[n]) ** 2 * triple_prods[n, col, n]
                         for n in range(dim)]) for col in range(dim)]
    col_non_symm_ops = [sum([C_vector[m] * C_vector[n].conjugate() *
                             triple_prods[m, col, n]
                             for m, n in it.chain(*[it.product(range(k), [k])
                                                    for k in range(dim)])])
                        for col in range(dim)]

    for row in range(dim):
        for col in range(dim):
            D_matrix[row, col] = (np.tensordot(basis[row],
                                               col_symm_ops[col] +
                                               2 * col_non_symm_ops[col],
                                               [[1, 0], [0, 1]]).real /
                                  basis_norms_sq[row])

    return D_matrix

# TODO: Formulate tests to verify correctness of this evolution.
# TODO: Fix this function to compute matrix elements as described in the
# Vectorization page in the documentation.
def double_comm_op(dim, C_vector, triple_prods, M_sq, basis_norms_sq, basis,
                   **kwargs):
    r"""Return a matrix :math:`E` such that when :math:`\rho` is vectorized the
    expression

    .. math::
    
       \frac{d\rho}{dt}=\left(\frac{M^*}{2}[c,[c,\rho]]+
       \frac{M}{2}[c^\dagger,[c^\dagger,\rho]]\right)
        
    can be calculated by:

    .. math::

       \frac{d\vec{\rho}}{dt}=E\vec{\rho}

    Vectorization is done according to the order prescribed in *basis*, with the
    component proportional to identity in the last place.
    
    :param coupling_op: The operator :math:`c` in matrix form
    :type coupling_op:  numpy.array
    :param M_sq:        Complex squeezing parameter :math:`M` defined by
                        :math:`\langle dB(t)dB(t)\rangle=Mdt`.
    :type M_sq:         complex
    :param basis:       An almost complete (minus identity), Hermitian,
                        traceless, orthogonal basis for the operators (does not
                        need to be normalized).
    :type basis:        list(numpy.array)
    :returns:           The matrix :math:`E` operating on a vectorized density
                        operator
    :rtype:             numpy.array

    """

    E_matrix = np.zeros((dim, dim)) # The matrix to return

    col_symm_ops = [sum([(M_sq.conjugate() * C_vector[n] ** 2).real *
                         (triple_prods[n, n, col] - triple_prods[n, col, n])
                         for n in range(dim)]) for col in range(dim)]
    col_non_symm_ops = [sum([(M_sq.conjugate() * C_vector[m] *
                              C_vector[n]).real *
                             (triple_prods[m, n, col] +
                              triple_prods[col, m, n] -
                              2.0 * triple_prods[m, col, n])
                             for m, n in it.chain(*[it.product(range(k), [k])
                                                    for k in range(dim)])])
                        for col in range(dim)]

    for row in range(dim):
        for col in range(dim):
            E_matrix[row, col] = 2 * (np.tensordot(basis[row],
                                                   col_symm_ops[col] +
                                                   col_non_symm_ops[col],
                                                   [[1, 0], [0, 1]]).real /
                                      basis_norms_sq[row])

    return E_matrix

def hamiltonian_op(dim, H_vector, double_prods, basis_norms_sq, basis,
                   **kwargs):
    r"""Return a matrix :math:`F` such that when :math:`\rho` is vectorized the
    expression

    .. math::

       \frac{d\rho}{dt}=-i[H,\rho]

    can be calculated by:

    .. math::

       \frac{d\vec{\rho}}{dt}=F\vec{\rho}

    Vectorization is done according to the order prescribed in *basis*, with the
    component proportional to identity in the last place.
    
    :param hamiltonian: The Hamiltonian :math:`H` in matrix form
    :type hamiltonian:  numpy.array
    :param basis:       An almost complete (minus identity), Hermitian,
                        traceless, orthogonal basis for the operators (does not
                        need to be normalized).
    :type basis:        list(numpy.array)
    :returns:           The matrix :math:`F` operating on a vectorized density
                        operator
    :rtype:             numpy.array

    """

    F_matrix = np.zeros((dim, dim)) # The matrix to return

    col_ops = [sum([H_vector[n].real * (double_prods[n, col] -
                                        double_prods[col, n])
               for n in range(dim)]) for col in range(dim)]

    for row in range(dim):
        for col in range(dim):
            F_matrix[row, col] = (np.tensordot(basis[row],
                                               col_ops[col],
                                               [[1, 0], [0, 1]]).imag /
                                  basis_norms_sq[row])

    return F_matrix

def weiner_op(dim, C_vector, double_prods, basis_norms_sq, basis, **kwargs):
    r"""Return a the matrix-vector pair :math:`(G,\vec{k})` such that when
    :math:`\rho` is vectorized the expression

    .. math::

        d\rho=dW\,\left(c\rho+\rho c^\dagger-
        \rho\operatorname{Tr}[(c+c^\dagger)\rho]\right)

    can be calculated by:

    .. math::

        d\vec{\rho}=dW\,(G+\vec{k}\cdot\vec{\rho})\vec{\rho}

    Vectorization is done according to the order prescribed in *basis*, with the
    component proportional to identity in the last place.
    
    :param coupling_op: The operator :math:`c` in matrix form
    :type coupling_op:  numpy.array
    :param basis:       An almost complete (minus identity), Hermitian,
                        traceless, orthogonal basis for the operators (does not
                        need to be normalized).
    :type basis:        list(numpy.array)
    :returns:           The matrix-vector pair :math:`(G,\vec{k})` operating on
                        a vectorized density operator (k is returned as a
                        row-vector)
    :rtype:             tuple(numpy.array)

    """

    G_matrix = np.zeros((dim, dim)) # The matrix to return
    k_vec = np.zeros(dim) # The dual vector to return

    col_ops = [sum([C_vector[n] * double_prods[n, col] for n in range(dim)])
               for col in range(dim)]

    for row in range(dim):
        k_vec[row] = -2.0 * C_vector[row].real * basis_norms_sq[row]
        for col in range(dim):
            G_matrix[row, col] = 2 * (np.tensordot(basis[row], col_ops[col],
                                      [[1, 0], [0, 1]]).real /
                                      basis_norms_sq[row])

    return G_matrix, k_vec

"""
.. module:: system_builder.py
   :synopsis: Build the matrix representation of the system of coupled
              real-valued stochastic differential equations.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
from itertools import product

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

    return abs(np.trace(np.dot(operator.conj().T, operator)))

def vectorize(operator, basis):
    """Vectorize an operator in a particular operator basis.

    :param operator:    The operator to vectorize
    :type operator:     list(numpy.array)
    :param basis:       The basis to vectorize the operator in
    :type basis:        list(numpy.array)
    :returns:           The vector components
    :rtype:             list(complex)

    """
    return [np.trace(np.dot(basis_el.conj().T, operator))/
            norm_squared(basis_el) for basis_el in basis]

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
    :type operator:     list(numpy.array)
    :param basis:       The basis to vectorize the operator in
    :type basis:        list(numpy.array)
    :returns:           The vector components
    :rtype:             list(complex)

    '''
    return [np.trace(np.dot(basis_el, operator.conj().T)) for basis_el in basis]

def op_calc_setup(coupling_op, basis):
    """Handle the repeated tasks performed every time a superoperator matrix is
    computed.

    """

    # Add the identity to the end of the basis to complete it (important for
    # some tests for the identity to be the last basis element).
    basis.append(np.eye(len(basis[0][0])))

    dim = len(basis)
    supop_matrix = np.zeros((dim, dim)) # The matrix to return

    # Vectorization of the coupling operator
    C_vector = vectorize(coupling_op, basis)
    c_op_pairs = list(zip(C_vector, basis))

    return dim, supop_matrix, C_vector, c_op_pairs

def diffusion_op(coupling_op, basis):
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

    dim, D_matrix, C_vector, c_op_pairs = op_calc_setup(coupling_op, basis)

    # TODO: Write tests to catch the inappropriate use of conjugate() without T

    # Construct lists of basis elements up to the current basis element for
    # doing the sum of the non-symmetric part of each element.
    part_c_op_pairs = [[c_op_pairs[m] for m in range(n)] for n in range(dim)]
    c_op_part_triplets = list(zip(C_vector, basis, part_c_op_pairs))
    for row in range(dim):
        # Square norm of basis element corresponding to current row
        sqnorm = norm_squared(basis[row])
        for col in range(dim):
            symm_addends = [abs(c)**2*np.trace(np.dot(basis[row], np.dot(op,
                np.dot(basis[col], op)) - 0.5*(np.dot(op, np.dot(op,
                basis[col])) + np.dot(basis[col], np.dot(op, op))))) for c, op
                in c_op_pairs]
            non_symm_addends = [c1*c2.conjugate()*np.trace(np.dot(basis[row],
                np.dot(op1, np.dot(basis[col], op2)) - 0.5*(np.dot(op2,
                np.dot(op1, basis[col])) + np.dot(basis[col],
                np.dot(op2, op1))))) for c1, op1, part_c_op_pair in
                c_op_part_triplets for c2, op2 in part_c_op_pair]
            D_matrix[row, col] = (sum(symm_addends).real + 
                                  2*sum(non_symm_addends).real)/sqnorm

    return D_matrix

# TODO: Formulate tests to verify correctness of this evolution.
# TODO: Fix this function to compute matrix elements as described in the
# Vectorization page in the documentation.
def double_comm_op(coupling_op, M_sq, basis):
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

    dim, E_matrix, C_vector, c_op_pairs = op_calc_setup(coupling_op, basis)

    # Construct lists of basis elements up to the current basis element for
    # doing the sum of the non-symmetric part of each element.
    part_c_op_pairs = [[c_op_pairs[m] for m in range(n)] for n in range(dim)]
    c_op_part_triplets = list(zip(C_vector, basis, part_c_op_pairs))
    for row in range(dim):
        sqnorm = norm_squared(basis[row])
        for col in range(dim):
            symm_addends = [(M_sq.conjugate()*c*c).real*(np.trace(
                recur_dot([basis[row], op, op, basis[col]]) -
                recur_dot([basis[row], op, basis[col], op]))).real for c, op
                in c_op_pairs]
            non_symm_addends = [(M_sq.conjugate()*c1*c2).real*(np.trace(
                np.dot(basis[row], recur_dot([op2, op1, basis[col]]) +
                       recur_dot([basis[col], op2, op1]) -
                       2*recur_dot([op2, basis[col], op1]))).real) for c1, op1,
                part_c_op_pair in c_op_part_triplets for c2, op2 in
                part_c_op_pair]
            E_matrix[row, col] = 2*(sum(symm_addends) +
                                    sum(non_symm_addends))/sqnorm

    return E_matrix

def hamiltonian_op(hamiltonian, basis):
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

    dim, F_matrix, H_vector, h_op_pairs = op_calc_setup(hamiltonian, basis)

    for row in range(dim):
        # Square norm of basis element corresponding to current row
        sqnorm = norm_squared(basis[row])
        for col in range(dim):
            addends = [h.real*(np.trace(np.dot(basis[row],
                np.dot(op, basis[col]) - np.dot(basis[col], op))).imag) for h,
                op in h_op_pairs]
            F_matrix[row, col] = sum(addends)/sqnorm

    return F_matrix

def weiner_op(coupling_op, basis):
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

    dim, G_matrix, C_vector, c_op_pairs = op_calc_setup(coupling_op, basis)
    k_vec = np.zeros(dim)

    for row in range(dim):
        sqnorm = norm_squared(basis[row])
        k_vec[row] = -2*C_vector[row].real*norm_squared(basis[row])
        for col in range(dim):
            G_addends = [(c*np.trace(recur_dot([basis[row], op,
                                                 basis[col]]))).real for c,
                          op in c_op_pairs]
            G_matrix[row, col] = 2*sum(G_addends)/sqnorm
    return G_matrix, k_vec

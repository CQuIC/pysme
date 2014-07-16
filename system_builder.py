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
    return [np.trace(np.dot(basis_el.conj().T, operator))/np.trace(
            np.dot(basis_el.conj().T, basis_el)) for basis_el in basis]

def diffusion_op(coupling_op, basis):
    r"""Return a matrix :math:`D` such that when :math:`\rho` is vectorized the
    expression

    .. math::
    
        d\rho=dt\,\mathcal{D}[c]\rho=c\rho c^\dagger-
        \frac{1}{2}(c^\dagger c\rho+\rho c^\dagger c)
        
    can be calculated by :math:`d\vec{\rho}=dt\,D\vec{\rho}`.
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

    # Add the identity to the end of the basis to complete it (important for
    # some tests for the identity to be the last basis element).
    basis.append(np.eye(len(basis[0][0])))

    dim = len(basis)
    D_matrix = np.zeros((dim, dim)) # The matrix to return

    # Vectorization of the coupling operator
    C_vector = vectorize(coupling_op, basis)
    c_op_pairs = list(zip(C_vector, basis))

    # TODO: Write tests to catch the inappropriate use of conjugate() without T

    # Construct lists of basis elements up to the current basis element for
    # doing the sum of the non-symmetric part of each element.
    part_c_op_pairs = [[c_op_pairs[m] for m in range(n)] for n in range(dim)]
    c_op_part_triplets = list(zip(C_vector, basis, part_c_op_pairs))
    for row in range(dim):
        # Square norm of basis element corresponding to current row
        sqnorm = norm_squared(basis[row])
        for col in range(dim):
            symm = sum([abs(c)**2*np.trace(np.dot(basis[row], np.dot(op,
                np.dot(basis[col], op)) - 0.5*(np.dot(op, np.dot(op,
                basis[col])) + np.dot(basis[col], np.dot(op, op))))) for c, op
                in c_op_pairs]).real/sqnorm
            non_symm = 2*sum([c1*c2.conjugate()*np.trace(np.dot(basis[row],
                np.dot(op1, np.dot(basis[col], op2)) - 0.5*(np.dot(op2,
                np.dot(op1, basis[col])) + np.dot(basis[col],
                np.dot(op2, op1))))) for c1, op1, part_c_op_pair in
                c_op_part_triplets for c2, op2 in part_c_op_pair]).real/sqnorm
            D_matrix[row, col] = symm + non_symm
    
    return D_matrix

# TODO: Formulate tests to verify correctness of this evolution.
def double_comm_op(coupling_op, M_sq, basis):
    r"""Return a matrix :math:`D` such that when :math:`\rho` is vectorized the
    expression

    .. math::
    
        d\rho=dt\,\left(\frac{M^*}{2}[c,[c,\rho]]+
        \frac{M}{2}[c^\dagger,[c^\dagger,\rho]]\right)
        
    can be calculated by :math:`d\vec{\rho}=dt\,D\vec{\rho}`.
    Vectorization is done according to the order prescribed in *basis*, with the
    component proportional to identity in the last place.
    
    :param coupling_op: The operator :math:`c` in matrix form
    :type coupling_op:  numpy.array
    :param M_sq:        Complex squeezing parameter :math:`M` defined by
                        :math:`\langle dB(t)dB(t)\rangle=Mdt`.
    :type M_sq:         Complex
    :param basis:       An almost complete (minus identity), Hermitian,
                        traceless, orthogonal basis for the operators (does not
                        need to be normalized).
    :type basis:        list(numpy.array)
    :returns:           The matrix :math:`D` operating on a vectorized density
                        operator
    :rtype:             numpy.array

    """

    # Add the identity to the end of the basis to complete it (important for
    # some tests for the identity to be the last basis element).
    basis.append(np.eye(len(basis[0][0])))

    dim = len(basis)
    D_matrix = np.zeros((dim, dim)) # The matrix to return

    # Vectorization of the coupling operator
    C_vector = vectorize(coupling_op, basis)
    c_op_pairs = list(zip(C_vector, basis))

    for row in range(dim):
        sqnorm = norm_squared(basis[row])
        for col in range(dim):
            addends = [c1*c2*(np.trace(recur_dot([basis[row], op1, op2,
                                                  basis[col]])).real - 
                              np.trace(recur_dot([basis[row], op1, basis[col],
                                                  op2]))) for (c1, op1),
                       (c2, op2) in product(c_op_pairs, c_op_pairs)]
            D_matrix[row, col] = 2*(M_sq*sum(addends)).real/sqnorm

    return D_matrix

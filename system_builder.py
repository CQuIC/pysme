"""
.. module:: system_builder.py
   :synopsis: Build the matrix representation of the system of coupled
   real-valued stochastic differential equations.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np

def vectorize(operator, basis):
    """Vectorize an operator in a particular operator basis.

    :param operator:    The operator to vectorize
    :type operator:     list(numpy.array)
    :param basis:       The basis to vectorize the operator in
    :type basis:        list(numpy.array)
    :returns:           The vector components
    :rtype:             list(complex)

    """
    return [np.trace(np.dot(basis_el.conj(), operator))/np.trace(
            np.dot(basis_el.conj(), basis_el)) for basis_el in basis]

def diffusion_op(coupling_op, basis):
    r"""Return a matrix :math:`D` such that when :math:`\rho` is vectorized
    :math:`d\rho=dt\,\mathcal{D}[c]\rho` can be calculated by
    :math:`d\overarrow{\rho}=dt\,D\overarrow{\rho}`. Vectorization is done
    according to the order prescribed in *basis*.
    
    :param coupling_op: The operator :math:`c` in matrix form
    :type coupling_op:  numpy.array
    :param basis:       A complete, Hermitian, traceless, orthogonal basis for
                        the operators 
    :type basis:        list(numpy.array)
    :returns:           The matrix :math:`D` operating on a vectorized density
                        operator
    :rtype:             numpy.array

    """

    dim = len(basis)
    D_matrix = np.zeros((dim, dim)) # The matrix to return
    # Vectorization of the coupling operator
    C_vector = vectorize(coupling_op, basis)
    c_op_pairs = list(zip(C_vector, basis))
    # Construct lists of basis elements up to the current basis element for
    # doing the sum of the non-symmetric part of each element.
    part_c_op_pairs = [[c_op_pairs[m] for m in range(n)] for n in range(dim)]
    c_op_part_triplets = list(zip(C_vector, basis, part_c_op_pairs))
    for row in range(dim):
        for col in range(dim):
            symm = 0.5*sum([abs(c)**2*np.trace(np.dot(basis[row], np.dot(op,
                np.dot(basis[col], op)) - 0.5*(np.dot(op, np.dot(op,
                basis[col])) + np.dot(basis[col], np.dot(op, op))))) for c, op
                in c_op_pairs]).real
            non_symm =sum([c1*c2.conjugate()*np.trace(np.dot(basis[row],
                np.dot(op1, np.dot(basis[col], op2)) - 0.5*(np.dot(op2,
                np.dot(op1, basis[col])) + np.dot(basis[col],
                np.dot(op2, op1))))) for c1, op1, part_c_op_pair in
                c_op_part_triplets for c2, op2 in part_c_op_pair]).real
            D_matrix[row, col] = symm + non_symm
    
    return D_matrix

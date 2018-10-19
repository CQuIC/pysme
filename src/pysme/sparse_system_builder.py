"""Construct integrators using sparse arrays.

"""

import numpy as np
import sparse
from sparse import COO

import pysme.gellmann as gm

def sparse_real(sparse_array):
    """`numpy.conjugate` returns a sparse array, but numpy.real does not, so
    use this function to get a sparse real part.

    """
    return (sparse_array + np.conj(sparse_array)) / 2

def sparse_imag(sparse_array):
    """`numpy.conjugate` returns a sparse array, but numpy.imag does not, so
    use this function to get a sparse imaginary part.

    """
    return (sparse_array - np.conj(sparse_array)) / 2.j

class SparseBasis:
    def __init__(self, dim, basis=None):
        if basis is None:
            self.dim = dim
            basis = gm.get_basis(dim)
        else:
            self.dim = basis[0].shape[0]
        self.basis = COO.from_numpy(np.array(basis))
        self.sq_norms = COO.from_numpy(np.einsum('jmn,jnm->j',
                                                 self.basis.todense(),
                                                 self.basis.todense()))
        sq_norms_inv = COO.from_numpy(1 / self.sq_norms.todense())
        self.dual = self.basis * sq_norms_inv[:,None,None]
        self.struct = sparse.tensordot(sparse.tensordot(self.basis, self.basis,
                                                        ([2], [1])),
                                       self.dual, ([1, 3], [2, 1]))
        if type(self.struct) == np.ndarray:
            # Sometimes sparse.tensordot returns numpy arrays. We want to force
            # it to be sparse, since sparse.tensordot fails when passed two
            # numpy arrays.
            self.struct = COO.from_numpy(self.struct)

    def vectorize(self, op, dense=False):
        sparse_op = COO.from_numpy(op)
        result = sparse.tensordot(self.dual, sparse_op, ([1,2], [1,0]))
        if not dense and type(result) == np.ndarray:
            # I want the result stored in a sparse format even if it isn't
            # sparse.
            result = COO.from_numpy(result)
        elif dense and type(result) == sparse.coo.COO:
            result = result.todense()
        return result

    def dualize(self, op):
        return np.conj(self.vectorize(op)) * self.sq_norms

    def matrize(self, vec):
        """Take a (sparse) vectorized operator and return it in matrix form.

        """
        return sparse.tensordot(self.basis, vec, ([0], [0]))

    def make_real_sand_matrix(self, x, y):
        r"""Make the superoperator matrix representation of

        N[X,Y](rho) = (1/2) ( X rho Y† + Y rho X† )

        In the basis {Λ_j}, N[X,Y](rho) = N(x,y)_jk rho_k Λ_j where

        N(x,y)_jk = Re[x_m (D_mlj + iF_mlj) (y*)_n (D_knl + iF_knl) ]

        Λ_j Λ_k = (D_jkl + iF_jkl) Λ_l

        x and y are vectorized representations of the operators X and Y stored
        in sparse format.

        The result is a dense matrix.

        """
        result_A = sparse.tensordot(x, self.struct, ([0], [0]))
        result_B = sparse.tensordot(np.conj(y), self.struct, ([0], [1]))
        # sparse.tensordot fails if both arguments are numpy ndarrays, so we
        # force the intermediate arrays to be sparse
        if type(result_B) == np.ndarray:
            result_B = COO.from_numpy(result_B)
        if type(result_A) == np.ndarray:
            result_A = COO.from_numpy(result_A)
        result = sparse_real(sparse.tensordot(result_A, result_B, ([0], [1])))
        # We want our result to be dense, to make things predictable from the
        # outside.
        if type(result) == sparse.coo.COO:
            result = result.todense()
        return result.real

    def make_real_comm_matrix(self, x, y):
        r"""Make the superoperator matrix representation of

        M[X,Y](rho) = (1/2) ( [X rho, Y†] + [Y, rho X†] )

        In the basis {Λ_j}, M[X,Y](rho) = M(x,y)_jk rho_k Λ_j where

        M(x,y)_jk = -2 Im[ (y*)_n F_lnj x_m (D_mkl + iF_mkl) ]

        Λ_j Λ_k = (D_jkl + iF_jkl) Λ_l

        x and y are vectorized representations of the operators X and Y stored
        in sparse format.

        The result is a dense matrix.

        """
        struct_imag = sparse_imag(self.struct)
        # sparse.tensordot fails if both arguments are numpy ndarrays, so we
        # force the intermediate arrays to be sparse
        result_A = sparse.tensordot(np.conj(y), struct_imag, ([0], [1]))
        result_B = sparse.tensordot(x, self.struct, ([0], [0]))
        if type(result_B) == np.ndarray:
            result_B = COO.from_numpy(result_B)
        if type(result_A) == np.ndarray:
            result_A = COO.from_numpy(result_A)
        result = -2 * sparse_imag(sparse.tensordot(result_A, result_B,
                                                   ([0], [1])))
        # We want our result to be dense, to make things predictable from the
        # outside.
        if type(result) == sparse.coo.COO:
            result = result.todense()
        return result.real

    def make_diff_op_matrix(self, x):
        """Make the superoperator matrix representation of

        X rho X† - (1/2) ( X† X rho + rho X† X )

        x is the vectorized representation of the operator X stored in sparse
        format.

        The result is a dense matrix.

        """
        return self.make_real_comm_matrix(x, x)

    def make_hamil_comm_matrix(self, h):
        """Make the superoperator matrix representation of

        -i[H,rho]

        h is the vectorized representation of the Hamiltonian H stored in sparse
        format.

        The result is a dense matrix.

        """
        struct_imag = sparse_imag(self.struct)
        result = 2 * sparse.tensordot(struct_imag, h, ([0], [0])).T
        if type(result) == sparse.coo.COO:
            result = result.todense()
        return result.real

    def make_double_comm_matrix(self, x, M):
        """Make the superoperator matrix representation of

        (M/2)[X†,[X†,rho]] + (M*/2)[X,[X,rho]]

        x is the vectorized representation of the operator X stored in sparse
        format.

        The result is a dense matrix.

        """
        return -(self.make_real_comm_matrix(M * np.conj(x), x) +
                 self.make_real_comm_matrix(x, M * np.conj(x)))

    def make_wiener_linear_matrix(self, x):
        Id_vec = self.vectorize(np.eye(self.dim))
        return 2 * self.make_real_sand_matrix(x, Id_vec)

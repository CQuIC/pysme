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
    def __init__(self, dim):
        self.dim = dim
        self.basis = COO.from_numpy(np.array(gm.get_basis(dim)))
        self.sq_norms = COO.from_numpy(np.einsum('jmn,jnm->j', self.basis.todense(),
                                   self.basis.todense()))
        sq_norms_inv = COO.from_numpy(1 / self.sq_norms.todense())
        self.dual = self.basis * sq_norms_inv[:,None,None]
        self.struct = sparse.tensordot(
                                sparse.tensordot(
                                    self.basis, self.basis, ([2], [1])),
                                self.dual, ([1, 3], [2, 1]))

    def vectorize(self, op):
        sparse_op = COO.from_numpy(op)
        result = sparse.tensordot(self.dual, sparse_op, ([1,2], [1,0]))
        if type(result) == np.ndarray:
            # I want the result stored in a sparse format even if it isn't
            # sparse.
            result = COO.from_numpy(result)
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

        `sparse.tensordot` might decide to return something dense, so the user
        should be aware of that.

        """
        return sparse_real(sparse.tensordot(
                                sparse.tensordot(x, self.struct,
                                                 ([0], [0])),
                                sparse.tensordot(np.conj(y), self.struct,
                                                 ([0], [1])),
                                ([0], [1])))

    def make_real_comm_matrix(self, x, y):
        r"""Make the superoperator matrix representation of

        M[X,Y](rho) = (1/2) ( [X rho, Y†] + [Y, rho X†] )

        In the basis {Λ_j}, M[X,Y](rho) = M(x,y)_jk rho_k Λ_j where

        M(x,y)_jk = -2 Im[ (y*)_n F_lnj x_m (D_mkl + iF_mkl) ]

        Λ_j Λ_k = (D_jkl + iF_jkl) Λ_l

        x and y are vectorized representations of the operators X and Y stored
        in sparse format.

        `sparse.tensordot` might decide to return something dense, so the user
        should be aware of that.

        """
        struct_imag = sparse_imag(self.struct)
        return -2 * sparse_imag(sparse.tensordot(
                                    sparse.tensordot(np.conj(y), struct_imag,
                                                     ([0], [1])),
                                    sparse.tensordot(x, self.struct,
                                                     ([0], [0])),
                                    ([0], [1])))

    def make_diff_op_matrix(self, x):
        """Make the superoperator matrix representation of

        X rho X† - (1/2) ( X† X rho + rho X† X )

        x is the vectorized representation of the operator X stored in sparse
        format.

        `sparse.tensordot` might decide to return something dense, so the user
        should be aware of that.

        """
        return self.make_real_comm_matrix(x, x)

    def make_hamil_comm_matrix(self, h):
        """Make the superoperator matrix representation of

        -i[H,rho]

        h is the vectorized representation of the Hamiltonian H stored in sparse
        format.

        `sparse.tensordot` might decide to return something dense, so the user
        should be aware of that.

        """
        struct_imag = sparse_imag(self.struct)
        return 2 * sparse.tensordot(struct_imag, h, ([0], [0])).T

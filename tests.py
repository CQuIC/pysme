from nose.tools import assert_almost_equal, assert_equal
import pysme.gellmann as gm
import pysme.gramschmidt as gs
import pysme.system_builder as sb
import numpy as np

def check_orthogonal(A, B):
    dot_prod = np.sqrt(np.trace(np.dot(A.conj().T, B)))
    assert_almost_equal(dot_prod, 0.0, 7)

def check_hermitian(A):
    non_herm = A - A.conj().T
    non_herm_norm = np.sqrt(np.trace(np.dot(non_herm.conj().T, non_herm)))
    assert_almost_equal(non_herm_norm, 0.0, 7)

def check_traceless(A):
    assert_almost_equal(np.trace(A), 0.0, 7)

def check_norm(A, norm):
    assert_almost_equal(np.sqrt(np.trace(np.dot(A.conj().T, A))), norm, 7)

def test_gellmann():
    for d in range(1, 5 + 1):
        matrices = [ [ gm.gellmann(j, k, d) for k in range(1, d + 1) ] for j in
            range(1, d + 1) ]
        for j in range(1, d + 1):
            for k in range(1, d + 1):
                check_hermitian(matrices[j - 1][k - 1])
                if j != d or k != d:
                    check_traceless(matrices[j - 1][k - 1])
                    check_norm(matrices[j - 1][k - 1], np.sqrt(2))
                else:
                    check_norm(matrices[j - 1][k - 1], np.sqrt(d))
                for jj in range(1, d + 1):
                    for kk in range(1, d + 1):
                        if jj != j or kk != k:
                            check_orthogonal(matrices[j - 1][k - 1],
                                matrices[jj - 1][kk - 1])

def check_recon(A, basis):
    A_coeffs = [ np.trace(np.dot(vect, A)) for vect in basis[0:3] ]
    A_recon = sum([ coeff*vect for coeff, vect in zip(A_coeffs, basis) ])
    diff = A - A_recon
    assert_almost_equal(np.sqrt(np.trace(np.dot(diff.conj().T, diff))), 0, 7)

def check_mat_eq(A, B):
    diff = A - B
    assert_almost_equal(np.sqrt(np.trace(np.dot(diff.conj().T, diff))), 0, 7)

def test_gramschmidt():
    test_matrices = [
        np.array([[ 5. + 0.j,  1. + 0.j],
                  [-2. + 0.j,  5. + 1.j]]),
        np.array([[ 1. + 0.j,  0. + 1.j],
                  [ 0. + 1.j, -1. + 0.j]]),
        np.array([[ 1. + 0.j,  1. + 1.j],
                  [-1. - 1.j,  1. + 0.j]]),
        np.array([[ 0. + 1.j,  1. + 0.j],
                  [ 1. + 0.j,  0. + 0.j]]),
        np.array([[ 1. + 0.j,  1. + 0.j,  0. + 0.j,  0. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  1. + 0.j,  0. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  0. + 0.j,  1. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j]]),
        ]

    for test_matrix in test_matrices:
        d = test_matrix.shape[0]
        basis = gs.orthonormalize(test_matrix)
        check_recon(test_matrix, basis)
        for m in range(len(basis)):
            if m == 0:
                check_mat_eq(basis[m], np.eye(d)/np.sqrt(d))
            else:
                check_traceless(basis[m])
            check_hermitian(basis[m])
            for n in range(len(basis)):
                if m != n:
                    check_orthogonal(basis[m], basis[n])
                else:
                    check_norm(basis[m], 1)

def basis(n):
    return [gm.gellmann(j, k, n) for j in range(1, n + 1) for k in
            range(1, n + 1)]

def check_trace_preservation():
    """Make sure that the identity component of the density operator doesn't
    evolve.
    """

    for dim in range(2, 3 + 1):
        for row in range(dim):
            for col in range(dim):
                c_op = np.zeros((dim, dim))
                D_matrix = sb.diffusion_op(c_op, basis(dim))
                for entry in D_matrix[-1]:
                    assert_equal(0, entry)

    for dim in range(2, 3 + 1):
        np.random.seed(dim)
        rand_c_op = np.random.randn(dim, dim) + 1.j*np.random.randn(dim, dim)
        D_matrix = sb.diffusion_op(rand_c_op, basis(dim))
        for entry in D_matrix[-1]:
            assert_equal(0, entry)

def check_vectorize(operators, basis):
    vectorizations = [sb.vectorize(operator, basis) for operator in operators]
    reconstructions = [sum([coeff*basis_el for coeff, basis_el in
                            zip(vectorization, basis)]) for vectorization in
                       vectorizations]
    for operator, reconstruction in zip(operators, reconstructions):
        diff = reconstruction - operator
        assert_almost_equal(np.trace(np.dot(diff.conj(), diff)), 0, 7)

def test_system_builder():
    check_trace_preservation()
    c2_operators = [np.array([[0, 1],
                              [0, 0]]),
                    np.array([[0, 0],
                              [1, 0]]),
                    np.array([[1, 0],
                              [0, -1]]),
                    np.array([[0, 1],
                              [1, 0]]),
                    np.array([[0, -1.j],
                              [1.j, 0]]),
                    np.array([[1, 2],
                              [3, 4]])]

    c3_operators = [np.array([[0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 0]]),
                    np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]]),
                    np.array([[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, -1]]),
                    np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])]
    check_vectorize(c2_operators, basis(2))
    check_vectorize(c3_operators, basis(3))

import numpy as np
from itertools import product
import gellmann as gm
from numpy import sqrt

def orthonormalize(A):
    d = max(A.shape) # Code won't currently work unless A is square.
    G = [ [ gm.gellmann(j, k, d) for k in range(1, d + 1) ] for j in
        range(1, d + 1) ]
    ordering = list(product(range(1, d + 1), range(1, d + 1)))
    hermitian = (A + A.conj().T)/2
    antiherm = (A - A.conj().T)/2.j
    hermitian_comps = [ np.trace(np.dot(hermitian, G[j - 1][k - 1]))/sqrt(2) if
        j != d or k!= d else
        np.trace(np.dot(hermitian, G[j - 1][k - 1]))/sqrt(d) for j, k in
        ordering ]
    antiherm_comps = [ np.trace(np.dot(antiherm, G[j - 1][k - 1]))/sqrt(2) if
        j != d or k!= d else
        np.trace(np.dot(antiherm, G[j - 1][k - 1]))/sqrt(d) for j, k in
        ordering ]

    # Identify the Gell-Mann matrices that have the most support on the
    # Hermitian and anti-Hermitian parts of A (other than identity) so that they
    # can be discarded prior to Gram-Schmidt orthogonalization.
    max_comps = [ max(abs(herm_comp), abs(anti_comp)) if j != d or k != d else 0
        for herm_comp, anti_comp, (j, k) in zip(hermitian_comps, antiherm_comps,
        ordering) ]

    # Ensure the identity is the first element in this list.
    ordered_max_comps = [list(enumerate(max_comps))[-1]] + \
        sorted(list(enumerate(max_comps))[0:-1], key=lambda elem: elem[1])

    discarded_indices = [ordered_max_comps[-1][0], ordered_max_comps[-2][0]]

    other_vectors = [ [ 0 if n != idx else 1 for n in range(d**2) ] for idx in
        range(d**2) if not idx in discarded_indices ]

    vector_set = np.array([other_vectors[-1], hermitian_comps, antiherm_comps] +
        other_vectors[0:-1]).T.real

    basis, R = np.linalg.qr(vector_set)
    basis = np.dot(basis, np.diag([ 1 if elem >= 0 else -1 for elem in
      np.diag(R) ]))
    basis = basis.T

    G_new = [ sum([ G[j - 1][k - 1]*coeff/sqrt(2) if j != d or k!= d else
        G[j - 1][k - 1]*coeff/sqrt(d) for coeff, (j, k) in
        zip(vect, ordering) ]) for vect in basis ]

    A_coeffs = [
        np.trace(np.dot(G_new[0], A)),
        np.trace(np.dot(G_new[1], A)),
        np.trace(np.dot(G_new[2], A)),
        ]

    A_recon = A_coeffs[0]*G_new[0] + A_coeffs[1]*G_new[1] + A_coeffs[2]*G_new[2]

    return G_new

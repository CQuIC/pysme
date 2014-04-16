"""
.. module:: gellmann.py
   :synopsis: Generate generalized Gell-Mann matrices
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

from sympy import *

def directsum(a, b):
    """Calculate the direct sum of two sympy matrices
    """

    ya, xa = a.shape
    yb, xb = b.shape
    return Matrix([[ a[(m, n)] if (m < ya and n < xa) else (0 if (m < ya or
        n < xa) else b[(m - ya, n - xa)]) for n in range(xa + xb)] for m in
        range(ya + yb)])

def H(K, d):
    """Calculate the diagonal generalized Gell-Mann matrices
    """

    if K == 1:
        return eye(d)
    elif K == d:
        return sqrt(2/(d*(d - 1)))*directsum(H(1, d - 1),
            Matrix([[1 - d]]))
    else:
        return directsum(H(K, d - 1), Matrix([[0]]))

def GellMann(K, J, d):
    r"""Calculate the generalized Gell-Mann matrix :math:`f_{k,j}^d` for
    :math:`k\neq j` or :math:`h_k^d` for :math:`k=j`.
    """

    d = S(d)

    if K < J:
        return Matrix([[ 1 if (j == J and k == K) or (k == J and j == K) else 0
            for j in range(1, d + 1) ] for k in range(1, d + 1)])
    elif K > J:
        return Matrix([[ I if (j == J and k == K) else (-I if (k == J and
            j == K) else 0) for j in range(1, d + 1) ] for k in
            range(1, d + 1)])
    else:
        return H(K, d)

def gellmann(j, k, d):
    """Returns a generalized Gell-Mann matrix of dimension d. According to the
    convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
    returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`, and
    :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`.

    :param j: First index for generalized Gell-Mann matrix
    :type j:  positive integer
    :param k: Second index for generalized Gell-Mann matrix
    :type k:  positive integer
    :param d: Dimension of the generalized Gell-Mann matrix
    :type d:  positive integer
    :returns: A genereralized Gell-Mann matrix.
    :rtype:   numpy.array

    """

    pass

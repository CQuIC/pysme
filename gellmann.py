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

def gellmann(i, j, d):
    pass

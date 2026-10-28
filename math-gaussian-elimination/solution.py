import numpy as np


def gaussian_eliminate(A, b):
    """
    A: a square (n, n) NumPy array that does not require row swaps
    b: a length-n NumPy array

    Returns:
        (U, c): U is A row-reduced to upper-triangular form, c is b
        transformed by the same row operations, such that Ux = c has
        the same solution as Ax = b.
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    c = b.astype(float).copy()

    for j in range(n):
        for i in range(j + 1, n):
            mult = U[i, j] / U[j, j]
            U[i] = U[i] - mult * U[j]
            c[i] = c[i] - mult * c[j]

    return U, c;



def back_substitute(U, c):
    """
    U: an upper-triangular (n, n) NumPy array
    c: a length-n NumPy array

    Returns:
        x, the solution to Ux = c, solved from the last row up.
    """
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
            x[i] = (c[i] - U[i, i+1 :] @ x[i+1:]) /  U[i, i]
            
    return x

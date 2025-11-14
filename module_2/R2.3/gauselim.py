import numpy as np

def gauss_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
        Naive Gaussian elimination code

        Usage:

            x = gauss_solve(A, B)
    """

    eps = np.finfo(np.float64).eps

    dett = np.linalg.det(A);
    if dett == 0:
        raise Exception("This system is unsolveable because det(A) = 0")

    b = B.copy()
    a = A.copy()

    n = len(b)
    x = np.zeros(n, dtype = np.float64)

    for j in range(n - 1):
        if abs(a[j,j]) < eps:
            raise Exception("Zero pivot encountered")

        for i in range(j+1, n):
            mult = a[i, j] / a[j, j]
            for k in range(j+1,n):
                a[i, k] = a[i, k] - mult * a[j, k]
            b[i] = b[i] - mult * b[j]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            b[i] = b[i] - a[i, j] * x[j]
        x[i] = b[i] / a[i, i]

    return x


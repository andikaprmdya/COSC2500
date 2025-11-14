"""
Direct Methods for Computing Eigenvalues
Adapted from TK1 Nomor_2 PCR implementation
"""

import numpy as np
import time
from numpy.linalg import norm

def givens_rotation(a, b):
    """
    Compute the cosine and sine for Givens rotation.

    Parameters:
    -----------
    a, b : float
        Elements to eliminate

    Returns:
    --------
    c, s : float
        Cosine and sine of rotation angle
    """
    if b == 0:
        return 1, 0
    else:
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
        return c, s


def qr_decomposition_givens(A, tol=1e-8):
    """
    QR decomposition using Givens Rotations.
    Returns the matrices Q and R.

    Parameters:
    -----------
    A : ndarray
        Matrix to decompose
    tol : float
        Tolerance for zero elements

    Returns:
    --------
    Q, R : ndarray
        Orthogonal matrix Q and upper triangular matrix R
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        for j in range(i + 1, m):
            if abs(R[j, i]) < tol:
                continue
            c, s = givens_rotation(R[i, i], R[j, i])
            G = np.eye(m)
            G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c

            R = G @ R
            Q = Q @ G.T

    return Q, R


def qr_algorithm_eigenvalues(A, max_iter=1000, tol=1e-10, return_vectors=True):
    """
    Compute eigenvalues (and optionally eigenvectors) using QR algorithm.

    This is the classic QR iteration: repeatedly factor A = QR and set A := RQ.
    The matrix converges to Schur form (upper triangular for real symmetric matrices).

    Parameters:
    -----------
    A : ndarray (n x n)
        Square matrix (preferably symmetric)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance for off-diagonal elements
    return_vectors : bool
        Whether to compute eigenvectors

    Returns:
    --------
    eigenvalues : ndarray
        Eigenvalues of A
    eigenvectors : ndarray (if return_vectors=True)
        Matrix where column i is eigenvector for eigenvalue i
    iterations : int
        Number of iterations performed
    """
    n = A.shape[0]
    A_k = A.copy()
    Q_total = np.eye(n)

    for iter_num in range(max_iter):
        # Perform QR decomposition
        Q, R = qr_decomposition_givens(A_k[:n, :n])  # Reduced form

        # Update A_k = R @ Q (similarity transformation)
        A_k = R @ Q

        if return_vectors:
            Q_total = Q_total @ Q

        # Check convergence: off-diagonal elements should be near zero
        off_diag = A_k - np.diag(np.diagonal(A_k))
        if np.allclose(off_diag, 0, atol=tol):
            break

    eigenvalues = np.diagonal(A_k)

    if return_vectors:
        # Normalize eigenvectors
        eigenvectors = Q_total.copy()
        for i in range(n):
            norm_val = norm(eigenvectors[:, i])
            if norm_val > 0:
                eigenvectors[:, i] /= norm_val
        return eigenvalues, eigenvectors, iter_num + 1
    else:
        return eigenvalues, iter_num + 1


def qr_algorithm_benchmark(A, max_iter=1000, tol=1e-10):
    """
    Benchmark QR algorithm with timing and convergence info.

    Parameters:
    -----------
    A : ndarray
        Matrix to analyze
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    results : dict
        Dictionary containing eigenvalues, eigenvectors, iterations, time, etc.
    """
    start_time = time.time()
    eigenvalues, eigenvectors, iterations = qr_algorithm_eigenvalues(
        A, max_iter=max_iter, tol=tol, return_vectors=True
    )
    elapsed_time = time.time() - start_time

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Estimate condition number
    if np.min(np.abs(eigenvalues)) > 1e-12:
        condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    else:
        condition_number = np.inf

    # Check orthogonality of eigenvectors
    orthogonality_error = np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(len(eigenvalues))))

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'iterations': iterations,
        'time': elapsed_time,
        'condition_number': condition_number,
        'orthogonality_error': orthogonality_error,
        'converged': iterations < max_iter
    }


def numpy_eig_benchmark(A):
    """
    Benchmark NumPy's built-in eigenvalue computation for comparison.

    Parameters:
    -----------
    A : ndarray
        Matrix to analyze

    Returns:
    --------
    results : dict
        Dictionary containing eigenvalues, eigenvectors, time, etc.
    """
    start_time = time.time()
    eigenvalues, eigenvectors = np.linalg.eig(A)
    elapsed_time = time.time() - start_time

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Estimate condition number
    if np.min(np.abs(eigenvalues)) > 1e-12:
        condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    else:
        condition_number = np.inf

    # Check orthogonality
    orthogonality_error = np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(len(eigenvalues))))

    return {
        'eigenvalues': eigenvalues.real,  # For symmetric matrices, should be real
        'eigenvectors': eigenvectors.real,
        'time': elapsed_time,
        'condition_number': condition_number,
        'orthogonality_error': orthogonality_error,
        'method': 'NumPy (LAPACK)'
    }


if __name__ == "__main__":
    # Test with a simple symmetric matrix
    print("Testing QR Algorithm for Eigenvalues")
    print("=" * 60)

    # Create a symmetric positive definite matrix
    np.random.seed(42)
    n = 5
    M = np.random.randn(n, n)
    A = M.T @ M  # Symmetric positive definite

    print(f"\nTest matrix (n={n}, symmetric positive definite):")
    print(A)

    # QR algorithm
    print("\n" + "-" * 60)
    print("QR Algorithm Results:")
    print("-" * 60)
    results_qr = qr_algorithm_benchmark(A)
    print(f"Eigenvalues: {results_qr['eigenvalues']}")
    print(f"Iterations: {results_qr['iterations']}")
    print(f"Time: {results_qr['time']:.6f} seconds")
    print(f"Condition number: {results_qr['condition_number']:.2e}")
    print(f"Orthogonality error: {results_qr['orthogonality_error']:.2e}")
    print(f"Converged: {results_qr['converged']}")

    # NumPy for comparison
    print("\n" + "-" * 60)
    print("NumPy (LAPACK) Results:")
    print("-" * 60)
    results_numpy = numpy_eig_benchmark(A)
    print(f"Eigenvalues: {results_numpy['eigenvalues']}")
    print(f"Time: {results_numpy['time']:.6f} seconds")
    print(f"Condition number: {results_numpy['condition_number']:.2e}")

    # Compare eigenvalues
    print("\n" + "-" * 60)
    print("Comparison:")
    print("-" * 60)
    eigenvalue_error = np.abs(results_qr['eigenvalues'] - results_numpy['eigenvalues'])
    print(f"Eigenvalue differences: {eigenvalue_error}")
    print(f"Max eigenvalue error: {np.max(eigenvalue_error):.2e}")
    print(f"QR is {results_numpy['time'] / results_qr['time']:.2f}x slower than NumPy")

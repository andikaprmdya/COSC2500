"""
Iterative Methods for Computing Eigenvalues
Implements Power iteration, Inverse Power iteration, Rayleigh quotient iteration, and Arnoldi
"""

import numpy as np
import time
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

def power_iteration(A, max_iter=1000, tol=1e-10, x0=None):
    """
    Power iteration to find the dominant eigenvalue and eigenvector.

    The power method repeatedly multiplies A by a vector and normalizes.
    Converges to the eigenvector corresponding to the largest (in magnitude) eigenvalue.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Square matrix
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    x0 : ndarray
        Initial guess (random if None)

    Returns:
    --------
    eigenvalue : float
        Dominant eigenvalue
    eigenvector : ndarray
        Corresponding eigenvector
    iterations : int
        Number of iterations
    converged : bool
        Whether the method converged
    """
    n = A.shape[0]

    # Random initial vector if not provided
    if x0 is None:
        x = np.random.randn(n)
    else:
        x = x0.copy()

    x = x / norm(x)

    eigenvalue_old = 0
    for k in range(max_iter):
        # Matrix-vector multiplication
        if issparse(A):
            y = A @ x
        else:
            y = A @ x

        # Normalize
        x_new = y / norm(y)

        # Rayleigh quotient for eigenvalue estimate
        if issparse(A):
            eigenvalue = (x_new.T @ (A @ x_new))
        else:
            eigenvalue = x_new.T @ A @ x_new

        # Check convergence
        if abs(eigenvalue - eigenvalue_old) < tol:
            return eigenvalue, x_new, k + 1, True

        eigenvalue_old = eigenvalue
        x = x_new

    return eigenvalue, x, max_iter, False


def inverse_power_iteration(A, mu=0.0, max_iter=1000, tol=1e-10, x0=None):
    """
    Inverse power iteration to find eigenvalue closest to shift mu.

    Solves (A - mu*I)^{-1} x at each iteration, converging to the eigenvalue nearest mu.
    For mu=0, finds the smallest eigenvalue.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Square matrix
    mu : float
        Shift parameter (target eigenvalue)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    x0 : ndarray
        Initial guess

    Returns:
    --------
    eigenvalue : float
        Eigenvalue closest to mu
    eigenvector : ndarray
        Corresponding eigenvector
    iterations : int
        Number of iterations
    converged : bool
        Whether converged
    """
    n = A.shape[0]

    if x0 is None:
        x = np.random.randn(n)
    else:
        x = x0.copy()

    x = x / norm(x)

    # Shifted matrix
    if issparse(A):
        A_shifted = A - mu * scipy.sparse.eye(n)
    else:
        A_shifted = A - mu * np.eye(n)

    eigenvalue_old = 0
    for k in range(max_iter):
        # Solve (A - mu*I) y = x
        if issparse(A_shifted):
            y = spsolve(A_shifted, x)
        else:
            y = np.linalg.solve(A_shifted, x)

        # Normalize
        x_new = y / norm(y)

        # Rayleigh quotient
        if issparse(A):
            eigenvalue = (x_new.T @ (A @ x_new))
        else:
            eigenvalue = x_new.T @ A @ x_new

        # Check convergence
        if abs(eigenvalue - eigenvalue_old) < tol:
            return eigenvalue, x_new, k + 1, True

        eigenvalue_old = eigenvalue
        x = x_new

    return eigenvalue, x, max_iter, False


def rayleigh_quotient_iteration(A, max_iter=1000, tol=1e-10, x0=None):
    """
    Rayleigh quotient iteration for finding an eigenvalue.

    Similar to inverse power iteration but uses the Rayleigh quotient as the shift,
    updated at each iteration. Converges cubically (very fast) near an eigenvalue.

    Parameters:
    -----------
    A : ndarray
        Square matrix
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    x0 : ndarray
        Initial guess

    Returns:
    --------
    eigenvalue : float
        Eigenvalue found
    eigenvector : ndarray
        Corresponding eigenvector
    iterations : int
        Number of iterations
    converged : bool
        Whether converged
    """
    n = A.shape[0]

    if x0 is None:
        x = np.random.randn(n)
    else:
        x = x0.copy()

    x = x / norm(x)

    # Initial Rayleigh quotient
    mu = x.T @ A @ x

    for k in range(max_iter):
        # Solve (A - mu*I) y = x
        try:
            A_shifted = A - mu * np.eye(n)
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            # Matrix is singular - we found an eigenvalue!
            return mu, x, k + 1, True

        # Normalize
        x_new = y / norm(y)

        # Update Rayleigh quotient
        mu_new = x_new.T @ A @ x_new

        # Check convergence
        if abs(mu_new - mu) < tol:
            return mu_new, x_new, k + 1, True

        mu = mu_new
        x = x_new

    return mu, x, max_iter, False


def arnoldi_iteration(A, k, v0=None):
    """
    Arnoldi iteration to build an orthonormal Krylov subspace.

    Generates an orthonormal basis for the Krylov subspace K_k(A, v0) = span{v0, Av0, ..., A^{k-1}v0}.
    The Hessenberg matrix H contains Rayleigh-Ritz approximations to eigenvalues.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Square matrix
    k : int
        Number of iterations (size of Krylov subspace)
    v0 : ndarray
        Starting vector

    Returns:
    --------
    Q : ndarray (n x k)
        Orthonormal basis for Krylov subspace
    H : ndarray (k x k)
        Upper Hessenberg matrix
    """
    n = A.shape[0]

    if v0 is None:
        v0 = np.random.randn(n)

    v0 = v0 / norm(v0)

    # Storage
    Q = np.zeros((n, k))
    H = np.zeros((k, k))

    Q[:, 0] = v0

    for j in range(k - 1):
        # Matrix-vector product
        if issparse(A):
            w = A @ Q[:, j]
        else:
            w = A @ Q[:, j]

        # Gram-Schmidt orthogonalization
        for i in range(j + 1):
            H[i, j] = Q[:, i].T @ w
            w = w - H[i, j] * Q[:, i]

        # Normalization
        H[j + 1, j] = norm(w)

        if H[j + 1, j] < 1e-12:
            # Lucky breakdown - subspace is invariant
            return Q[:, :j+1], H[:j+1, :j+1]

        Q[:, j + 1] = w / H[j + 1, j]

    # Final step
    if issparse(A):
        w = A @ Q[:, k - 1]
    else:
        w = A @ Q[:, k - 1]

    for i in range(k):
        H[i, k - 1] = Q[:, i].T @ w

    return Q, H


def arnoldi_eigenvalues(A, k, num_eigs=None):
    """
    Compute eigenvalue approximations using Arnoldi iteration.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Square matrix
    k : int
        Size of Krylov subspace (number of Arnoldi iterations)
    num_eigs : int
        Number of eigenvalues to return (default: all k)

    Returns:
    --------
    eigenvalues : ndarray
        Approximate eigenvalues (Ritz values)
    eigenvectors : ndarray
        Approximate eigenvectors (Ritz vectors)
    """
    Q, H = arnoldi_iteration(A, k)

    # Eigenvalues of H are Ritz values (approximations to eigenvalues of A)
    ritz_values, ritz_vectors_small = np.linalg.eig(H)

    # Sort by magnitude
    idx = np.argsort(np.abs(ritz_values))[::-1]
    ritz_values = ritz_values[idx]
    ritz_vectors_small = ritz_vectors_small[:, idx]

    # Project back to original space
    ritz_vectors = Q @ ritz_vectors_small

    if num_eigs is not None:
        return ritz_values[:num_eigs].real, ritz_vectors[:, :num_eigs].real
    else:
        return ritz_values.real, ritz_vectors.real


def benchmark_iterative_method(method_func, A, method_name, **kwargs):
    """
    Benchmark an iterative eigenvalue method.

    Parameters:
    -----------
    method_func : callable
        Iterative method function
    A : ndarray
        Matrix
    method_name : str
        Name for display
    **kwargs : additional arguments for method_func

    Returns:
    --------
    results : dict
        Dictionary with timing and convergence info
    """
    start_time = time.time()

    if method_name == "Arnoldi":
        eigenvalues, eigenvectors = method_func(A, **kwargs)
        elapsed_time = time.time() - start_time
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'time': elapsed_time,
            'method': method_name,
            'converged': True
        }
    else:
        eigenvalue, eigenvector, iterations, converged = method_func(A, **kwargs)
        elapsed_time = time.time() - start_time

        return {
            'eigenvalue': eigenvalue,
            'eigenvector': eigenvector,
            'iterations': iterations,
            'time': elapsed_time,
            'converged': converged,
            'method': method_name
        }


if __name__ == "__main__":
    print("Testing Iterative Eigenvalue Methods")
    print("=" * 70)

    # Create test matrix
    np.random.seed(42)
    n = 6
    M = np.random.randn(n, n)
    A = M.T @ M  # Symmetric positive definite

    print(f"\nTest matrix (n={n}, symmetric positive definite)")

    # True eigenvalues for comparison
    true_eigs = np.linalg.eigvals(A)
    true_eigs = np.sort(true_eigs)[::-1]
    print(f"True eigenvalues: {true_eigs}")

    # Power iteration (finds largest)
    print("\n" + "-" * 70)
    print("Power Iteration (largest eigenvalue):")
    print("-" * 70)
    eig, vec, iters, conv = power_iteration(A, max_iter=1000, tol=1e-10)
    print(f"Eigenvalue: {eig:.10f}")
    print(f"True largest: {true_eigs[0]:.10f}")
    print(f"Error: {abs(eig - true_eigs[0]):.2e}")
    print(f"Iterations: {iters}")
    print(f"Converged: {conv}")

    # Inverse Power iteration (finds smallest)
    print("\n" + "-" * 70)
    print("Inverse Power Iteration (smallest eigenvalue):")
    print("-" * 70)
    eig, vec, iters, conv = inverse_power_iteration(A, mu=0.0, max_iter=1000, tol=1e-10)
    print(f"Eigenvalue: {eig:.10f}")
    print(f"True smallest: {true_eigs[-1]:.10f}")
    print(f"Error: {abs(eig - true_eigs[-1]):.2e}")
    print(f"Iterations: {iters}")
    print(f"Converged: {conv}")

    # Rayleigh quotient iteration
    print("\n" + "-" * 70)
    print("Rayleigh Quotient Iteration:")
    print("-" * 70)
    eig, vec, iters, conv = rayleigh_quotient_iteration(A, max_iter=100, tol=1e-10)
    print(f"Eigenvalue found: {eig:.10f}")
    closest_true = true_eigs[np.argmin(np.abs(true_eigs - eig))]
    print(f"Closest true eigenvalue: {closest_true:.10f}")
    print(f"Error: {abs(eig - closest_true):.2e}")
    print(f"Iterations: {iters} (note: cubic convergence!)")
    print(f"Converged: {conv}")

    # Arnoldi iteration
    print("\n" + "-" * 70)
    print(f"Arnoldi Iteration (k={n-1} iterations, finds {n-1} eigenvalues):")
    print("-" * 70)
    ritz_vals, ritz_vecs = arnoldi_eigenvalues(A, k=n-1)
    print(f"Ritz values (approx eigenvalues):")
    for i, rv in enumerate(ritz_vals):
        true_val = true_eigs[i] if i < len(true_eigs) else np.nan
        error = abs(rv - true_val) if i < len(true_eigs) else np.nan
        print(f"  λ_{i+1}: {rv:.10f}  (true: {true_val:.10f}, error: {error:.2e})")

    print("\n" + "=" * 70)
    print("All iterative methods completed successfully!")
    print("=" * 70)

"""
Test Matrix Generators
Creates various types of matrices for eigenvalue testing
"""

import numpy as np
from scipy import sparse

def create_symmetric_matrix(n, seed=None, condition_number=None):
    """
    Create a symmetric matrix.

    Parameters:
    -----------
    n : int
        Matrix size
    seed : int
        Random seed for reproducibility
    condition_number : float
        Desired condition number (if None, random)

    Returns:
    --------
    A : ndarray
        Symmetric matrix
    """
    if seed is not None:
        np.random.seed(seed)

    if condition_number is not None:
        # Create matrix with specific condition number
        # Generate random eigenvalues with specified conditioning
        max_eig = 1.0
        min_eig = max_eig / condition_number
        eigenvalues = np.linspace(min_eig, max_eig, n)

        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(n, n))

        # A = Q * diag(eigenvalues) * Q^T
        A = Q @ np.diag(eigenvalues) @ Q.T
    else:
        # Random symmetric matrix
        M = np.random.randn(n, n)
        A = (M + M.T) / 2

    return A


def create_positive_definite_matrix(n, seed=None, condition_number=None):
    """
    Create a symmetric positive definite matrix.

    Parameters:
    -----------
    n : int
        Matrix size
    seed : int
        Random seed
    condition_number : float
        Desired condition number

    Returns:
    --------
    A : ndarray
        Symmetric positive definite matrix
    """
    if seed is not None:
        np.random.seed(seed)

    if condition_number is not None:
        # Create with specific condition number
        max_eig = 10.0
        min_eig = max_eig / condition_number
        eigenvalues = np.linspace(min_eig, max_eig, n)

        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(n, n))

        # A = Q * diag(eigenvalues) * Q^T
        A = Q @ np.diag(eigenvalues) @ Q.T
    else:
        # M^T M is always positive definite
        M = np.random.randn(n, n)
        A = M.T @ M

    return A


def create_tridiagonal_matrix(n, diag_value=2.0, off_diag_value=-1.0):
    """
    Create a tridiagonal matrix.

    Parameters:
    -----------
    n : int
        Matrix size
    diag_value : float
        Diagonal elements
    off_diag_value : float
        Off-diagonal elements

    Returns:
    --------
    A : ndarray
        Tridiagonal matrix
    """
    A = np.diag([diag_value] * n) + \
        np.diag([off_diag_value] * (n - 1), k=1) + \
        np.diag([off_diag_value] * (n - 1), k=-1)
    return A


def create_sparse_matrix(n, density=0.1, seed=None, symmetric=True):
    """
    Create a sparse matrix.

    Parameters:
    -----------
    n : int
        Matrix size
    density : float
        Fraction of non-zero elements (0 to 1)
    seed : int
        Random seed
    symmetric : bool
        Whether to make matrix symmetric

    Returns:
    --------
    A : scipy sparse matrix (CSR format)
        Sparse matrix
    """
    if seed is not None:
        np.random.seed(seed)

    if symmetric:
        # Create symmetric sparse matrix
        A = sparse.random(n, n, density=density, format='csr', random_state=seed)
        A = (A + A.T) / 2
    else:
        A = sparse.random(n, n, density=density, format='csr', random_state=seed)

    return A


def create_ill_conditioned_matrix(n, condition_number=1e6, seed=None):
    """
    Create an ill-conditioned matrix with specified condition number.

    Parameters:
    -----------
    n : int
        Matrix size
    condition_number : float
        Desired condition number (large = ill-conditioned)
    seed : int
        Random seed

    Returns:
    --------
    A : ndarray
        Ill-conditioned symmetric matrix
    """
    return create_positive_definite_matrix(n, seed=seed, condition_number=condition_number)


def create_clustered_eigenvalue_matrix(n, num_clusters=3, seed=None):
    """
    Create a matrix with clustered eigenvalues (challenging for iterative methods).

    Parameters:
    -----------
    n : int
        Matrix size
    num_clusters : int
        Number of eigenvalue clusters
    seed : int
        Random seed

    Returns:
    --------
    A : ndarray
        Matrix with clustered eigenvalues
    """
    if seed is not None:
        np.random.seed(seed)

    # Create clusters of eigenvalues
    eigenvalues = []
    cluster_centers = np.linspace(1, 10, num_clusters)

    for center in cluster_centers:
        cluster_size = n // num_clusters
        cluster_eigs = center + 0.01 * np.random.randn(cluster_size)
        eigenvalues.extend(cluster_eigs)

    # Fill remaining slots
    while len(eigenvalues) < n:
        eigenvalues.append(cluster_centers[0] + 0.01 * np.random.randn())

    eigenvalues = np.array(eigenvalues[:n])

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # A = Q * diag(eigenvalues) * Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T

    return A


def create_laplacian_1d(n):
    """
    Create 1D Laplacian matrix (discrete second derivative).
    Used in vibration analysis and heat equation.

    Parameters:
    -----------
    n : int
        Matrix size

    Returns:
    --------
    A : ndarray
        1D Laplacian matrix (tridiagonal with 2, -1, -1)
    """
    return create_tridiagonal_matrix(n, diag_value=2.0, off_diag_value=-1.0)


def create_laplacian_2d(nx, ny):
    """
    Create 2D Laplacian matrix (5-point stencil).

    Parameters:
    -----------
    nx, ny : int
        Grid dimensions

    Returns:
    --------
    A : sparse matrix
        2D Laplacian matrix
    """
    n = nx * ny
    diagonals = [
        -4 * np.ones(n),  # Main diagonal
        np.ones(n - 1),   # Upper diagonal
        np.ones(n - 1),   # Lower diagonal
        np.ones(n - nx),  # Upper band
        np.ones(n - nx),  # Lower band
    ]

    # Remove connections across grid boundaries
    diagonals[1][np.arange(1, n) % nx == 0] = 0
    diagonals[2][np.arange(n - 1) % nx == nx - 1] = 0

    offsets = [0, 1, -1, nx, -nx]
    A = sparse.diags(diagonals, offsets, shape=(n, n), format='csr')

    return A


def create_controlled_gap_matrix(n, gap_ratio, seed=42):
    """
    Create a positive definite matrix with controlled eigenvalue gap.

    The largest two eigenvalues have ratio lambda_2/lambda_1 = gap_ratio.
    This is useful for testing convergence of power iteration, which
    depends critically on this ratio.

    Parameters:
    -----------
    n : int
        Matrix size
    gap_ratio : float
        Ratio of second-largest to largest eigenvalue (0 < gap_ratio < 1)
        - gap_ratio close to 1.0: slow convergence (nearly degenerate)
        - gap_ratio close to 0.0: fast convergence (well-separated)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    A : ndarray
        Symmetric positive definite matrix with controlled gap

    Example:
    --------
    >>> A = create_controlled_gap_matrix(50, gap_ratio=0.95)
    >>> eigs = np.linalg.eigvals(A)
    >>> eigs = np.sort(eigs)[::-1]
    >>> print(f"Gap ratio: {eigs[1]/eigs[0]:.6f}")  # Should be ~0.95
    """
    if not (0 < gap_ratio < 1):
        raise ValueError("gap_ratio must be between 0 and 1")

    if seed is not None:
        np.random.seed(seed)

    # Create eigenvalue spectrum
    # lambda_1 = 1.0 (largest)
    # lambda_2 = gap_ratio (second largest)
    # lambda_3...lambda_n distributed from 0.1 to gap_ratio*0.9

    eigenvalues = np.zeros(n)
    eigenvalues[0] = 1.0  # Largest eigenvalue
    eigenvalues[1] = gap_ratio  # Second largest (controls convergence rate)

    if n > 2:
        # Remaining eigenvalues between 0.1 and 90% of gap
        eigenvalues[2:] = np.linspace(0.1, gap_ratio * 0.9, n-2)

    # Create random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # Construct A = Q * diag(eigenvalues) * Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T

    return A


def get_matrix_properties(A):
    """
    Analyze properties of a matrix.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Matrix to analyze

    Returns:
    --------
    properties : dict
        Dictionary with matrix properties
    """
    if sparse.issparse(A):
        is_sparse = True
        density = A.nnz / (A.shape[0] * A.shape[1])
        A_dense = A.toarray()
    else:
        is_sparse = False
        density = 1.0
        A_dense = A

    # Check symmetry
    is_symmetric = np.allclose(A_dense, A_dense.T)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A_dense)
    eigenvalues = np.sort(eigenvalues.real)[::-1]

    # Check positive definiteness
    is_positive_definite = np.all(eigenvalues > 0)

    # Condition number
    if np.min(np.abs(eigenvalues)) > 1e-12:
        condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
    else:
        condition_number = np.inf

    return {
        'size': A.shape[0],
        'is_sparse': is_sparse,
        'density': density,
        'is_symmetric': is_symmetric,
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues,
        'largest_eigenvalue': eigenvalues[0],
        'smallest_eigenvalue': eigenvalues[-1],
        'condition_number': condition_number,
        'num_eigenvalues': len(eigenvalues)
    }


if __name__ == "__main__":
    print("Test Matrix Generators")
    print("=" * 70)

    # Test various matrix types
    matrices = {
        'Symmetric': create_symmetric_matrix(5, seed=42),
        'Positive Definite': create_positive_definite_matrix(5, seed=42),
        'Tridiagonal': create_tridiagonal_matrix(5),
        'Ill-conditioned (κ=1e6)': create_ill_conditioned_matrix(5, condition_number=1e6, seed=42),
        'Clustered Eigenvalues': create_clustered_eigenvalue_matrix(5, num_clusters=2, seed=42),
        '1D Laplacian': create_laplacian_1d(5),
    }

    for name, A in matrices.items():
        print(f"\n{name} Matrix:")
        print("-" * 70)
        props = get_matrix_properties(A)
        print(f"Size: {props['size']}x{props['size']}")
        print(f"Symmetric: {props['is_symmetric']}")
        print(f"Positive Definite: {props['is_positive_definite']}")
        print(f"Condition Number: {props['condition_number']:.2e}")
        print(f"Eigenvalues: {props['eigenvalues']}")

    # Test sparse matrix
    print("\n" + "-" * 70)
    print("Sparse Matrix (n=100, density=0.05):")
    print("-" * 70)
    A_sparse = create_sparse_matrix(100, density=0.05, seed=42, symmetric=True)
    props_sparse = get_matrix_properties(A_sparse)
    print(f"Size: {props_sparse['size']}x{props_sparse['size']}")
    print(f"Sparse: {props_sparse['is_sparse']}")
    print(f"Density: {props_sparse['density']:.2%}")
    print(f"Symmetric: {props_sparse['is_symmetric']}")
    print(f"Largest eigenvalue: {props_sparse['largest_eigenvalue']:.6f}")
    print(f"Smallest eigenvalue: {props_sparse['smallest_eigenvalue']:.6f}")

    print("\n" + "=" * 70)
    print("All test matrices generated successfully!")
    print("=" * 70)

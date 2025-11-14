"""
PageRank Demo: Computing Dominant Eigenvector
Demonstrates eigenvalue methods applied to Google's PageRank algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Try to import networkx, use simple alternative if not available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not available. Using simplified graph implementation.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iterative_methods import power_iteration
from direct_methods import qr_algorithm_eigenvalues

def create_web_graph(n_pages=10, seed=42):
    """
    Create a random directed graph representing web pages and links.

    Parameters:
    -----------
    n_pages : int
        Number of web pages
    seed : int
        Random seed

    Returns:
    --------
    G : nx.DiGraph or adjacency matrix
        Directed graph (or adjacency matrix if networkx not available)
    """
    np.random.seed(seed)

    if HAS_NETWORKX:
        # Create directed graph with some structure
        G = nx.DiGraph()
        G.add_nodes_from(range(n_pages))

        # Add random edges (links between pages)
        for i in range(n_pages):
            # Each page links to 2-4 other pages
            num_links = np.random.randint(2, min(5, n_pages))
            targets = np.random.choice([j for j in range(n_pages) if j != i],
                                       size=num_links, replace=False)
            for j in targets:
                G.add_edge(i, j)

        return G
    else:
        # Create adjacency matrix directly
        A = np.zeros((n_pages, n_pages))
        for i in range(n_pages):
            num_links = np.random.randint(2, min(5, n_pages))
            targets = np.random.choice([j for j in range(n_pages) if j != i],
                                       size=num_links, replace=False)
            for j in targets:
                A[i, j] = 1
        return A


def create_google_matrix(G, damping=0.85):
    """
    Create the Google matrix for PageRank.

    The Google matrix is: M = damping * P + (1 - damping) * E
    where P is the transition matrix and E is the teleportation matrix.

    Parameters:
    -----------
    G : nx.DiGraph or adjacency matrix
        Directed graph or adjacency matrix
    damping : float
        Damping factor (typically 0.85)

    Returns:
    --------
    M : ndarray
        Google matrix
    """
    if HAS_NETWORKX and not isinstance(G, np.ndarray):
        n = len(G.nodes())
        # Adjacency matrix
        A = nx.adjacency_matrix(G).toarray().T
    else:
        # G is already an adjacency matrix
        A = G.T
        n = A.shape[0]

    # Transition matrix P
    # P[i,j] = 1/out_degree(j) if there's a link from j to i
    P = np.zeros((n, n))
    for j in range(n):
        out_degree = A[:, j].sum()
        if out_degree > 0:
            P[:, j] = A[:, j] / out_degree
        else:
            # Dangling node - distribute equally
            P[:, j] = 1.0 / n

    # Teleportation matrix E (all entries 1/n)
    E = np.ones((n, n)) / n

    # Google matrix
    M = damping * P + (1 - damping) * E

    return M


def pagerank_power_iteration(M, max_iter=1000, tol=1e-10):
    """
    Compute PageRank using power iteration.

    Parameters:
    -----------
    M : ndarray
        Google matrix
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    pagerank : ndarray
        PageRank scores (dominant eigenvector)
    iterations : int
        Number of iterations
    """
    n = M.shape[0]

    # Start with uniform distribution
    x = np.ones(n) / n

    for k in range(max_iter):
        x_new = M @ x

        # Check convergence
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new, k + 1

        x = x_new

    return x, max_iter


def pagerank_networkx(G, damping=0.85):
    """
    Compute PageRank using NetworkX's built-in function (for comparison).

    Parameters:
    -----------
    G : nx.DiGraph or adjacency matrix
        Directed graph or adjacency matrix
    damping : float
        Damping factor

    Returns:
    --------
    pagerank : dict
        PageRank scores
    """
    if HAS_NETWORKX and not isinstance(G, np.ndarray):
        return nx.pagerank(G, alpha=damping, tol=1e-10)
    else:
        # Use our own power iteration if networkx not available
        M = create_google_matrix(G, damping=damping)
        pr, _ = pagerank_power_iteration(M, max_iter=1000, tol=1e-10)
        return {i: pr[i] for i in range(len(pr))}


def visualize_pagerank(G, pagerank_scores, title="PageRank Visualization", output_file=None):
    """
    Visualize the graph with PageRank scores.

    Parameters:
    -----------
    G : nx.DiGraph or adjacency matrix
        Directed graph or adjacency matrix
    pagerank_scores : ndarray or dict
        PageRank scores for each node
    title : str
        Plot title
    output_file : str
        Output filename (if None, display only)
    """
    # Convert to dict if array
    if isinstance(pagerank_scores, np.ndarray):
        pr_dict = {i: pagerank_scores[i] for i in range(len(pagerank_scores))}
    else:
        pr_dict = pagerank_scores

    if HAS_NETWORKX and not isinstance(G, np.ndarray):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Graph visualization
        pos = nx.spring_layout(G, seed=42)
        node_sizes = [v * 10000 for v in pr_dict.values()]
        node_colors = list(pr_dict.values())

        nx.draw(G, pos, ax=ax1,
                node_size=node_sizes,
                node_color=node_colors,
                cmap='YlOrRd',
                with_labels=True,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=15,
                edge_color='gray',
                alpha=0.8)

        ax1.set_title(title, fontsize=14, fontweight='bold')
    else:
        # Just bar chart if no networkx
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    # Bar chart of PageRank scores
    nodes = list(pr_dict.keys())
    scores = list(pr_dict.values())

    ax2.bar(nodes, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Page ID', fontsize=12)
    ax2.set_ylabel('PageRank Score', fontsize=12)
    ax2.set_title('PageRank Scores', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()

    plt.close()


def experiment_damping_factor_analysis(G, damping_factors=None, output_dir=None):
    """
    Analyze how damping factor affects PageRank convergence.

    Theory: Convergence rate = alpha (damping factor)
    Higher alpha -> slower convergence but more "surf" less "teleport"

    Parameters:
    -----------
    G : graph or ndarray
        Web graph
    damping_factors : list
        Damping factors to test (default: [0.5, 0.7, 0.85, 0.9, 0.95])
    output_dir : str
        Output directory for results
    """
    import pandas as pd

    if damping_factors is None:
        damping_factors = [0.5, 0.7, 0.85, 0.9, 0.95]

    print("\n" + "=" * 70)
    print("EXPERIMENT: Damping Factor Analysis")
    print("=" * 70)

    results = {
        'damping': [],
        'iterations': [],
        'eigenvalue_gap': [],
        'top_page': [],
        'top_score': [],
        'ranking_changes': []
    }

    pr_baseline = None  # Baseline ranking for alpha=0.85

    for alpha in damping_factors:
        print(f"\nTesting alpha = {alpha:.2f}...")

        # Create Google matrix with this damping factor
        M = create_google_matrix(G, damping=alpha)

        # Run power iteration
        pr, iters = pagerank_power_iteration(M, max_iter=1000, tol=1e-10)

        # Get eigenvalue gap
        eigenvalues = np.linalg.eigvals(M)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        gap = eigenvalues[0] - eigenvalues[1]

        # Top ranked page
        top_idx = np.argmax(pr)
        top_score = pr[top_idx]

        # Compare ranking to baseline (alpha=0.85)
        if alpha == 0.85:
            pr_baseline = pr.copy()
            ranking_changes = 0
        else:
            # Count how many pages changed position in top 5
            if pr_baseline is not None:
                top5_baseline = set(np.argsort(pr_baseline)[::-1][:5])
                top5_current = set(np.argsort(pr)[::-1][:5])
                ranking_changes = len(top5_baseline.symmetric_difference(top5_current))
            else:
                ranking_changes = 0

        results['damping'].append(alpha)
        results['iterations'].append(iters)
        results['eigenvalue_gap'].append(gap)
        results['top_page'].append(top_idx)
        results['top_score'].append(top_score)
        results['ranking_changes'].append(ranking_changes)

        print(f"  Iterations: {iters}")
        print(f"  Eigenvalue gap: {gap:.6f}")
        print(f"  Top page: {top_idx} (score: {top_score:.6f})")
        print(f"  Ranking changes vs alpha=0.85: {ranking_changes} pages in top-5")

    df = pd.DataFrame(results)

    # Save CSV
    if output_dir:
        csv_file = os.path.join(output_dir, 'exp5_pagerank_damping.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: exp5_pagerank_damping.csv")

    # Generate plot
    if output_dir:
        plot_damping_analysis(df, output_dir)

    # Print summary
    print("\n" + "-" * 70)
    print("KEY FINDINGS:")
    print("-" * 70)

    # Finding 1: Convergence vs damping
    fastest_idx = df['iterations'].idxmin()
    slowest_idx = df['iterations'].idxmax()

    fastest_alpha = df.loc[fastest_idx, 'damping']
    fastest_iters = df.loc[fastest_idx, 'iterations']
    slowest_alpha = df.loc[slowest_idx, 'damping']
    slowest_iters = df.loc[slowest_idx, 'iterations']

    print(f"\n1. alpha={fastest_alpha} converged in {fastest_iters} iterations (fastest)")
    print(f"   alpha={slowest_alpha} required {slowest_iters} iterations (slowest)")
    print(f"   Ratio: {slowest_iters/fastest_iters:.1f}x more iterations")
    print(f"   Theory: Should scale as (1-alpha_slow)/(1-alpha_fast) = {(1-slowest_alpha)/(1-fastest_alpha):.1f}x")

    # Finding 2: Eigenvalue gap
    print(f"\n2. Eigenvalue gap = 1 - alpha (theory)")
    print(f"   alpha=0.5: gap={df.loc[df['damping']==0.5, 'eigenvalue_gap'].values[0]:.4f} (theory: 0.5000)")
    print(f"   alpha=0.95: gap={df.loc[df['damping']==0.95, 'eigenvalue_gap'].values[0]:.4f} (theory: 0.0500)")

    # Finding 3: Ranking stability
    max_changes = df['ranking_changes'].max()
    print(f"\n3. Ranking stability:")
    print(f"   Maximum top-5 changes vs alpha=0.85 baseline: {max_changes} pages")
    print(f"   Higher alpha values produce more similar rankings (less teleportation noise)")

    print("\n" + "=" * 70)

    return df


def plot_damping_analysis(df, output_dir):
    """Plot damping factor analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Iterations vs damping
    ax = axes[0, 0]
    ax.plot(df['damping'], df['iterations'], 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Damping Factor (alpha)', fontsize=12)
    ax.set_ylabel('Iterations to Converge', fontsize=12)
    ax.set_title('Higher Damping = Slower Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add theoretical curve: iterations ∝ 1/(1-alpha)
    alphas_theory = np.linspace(0.5, 0.95, 100)
    # Normalize to match at alpha=0.85
    ref_idx = df[df['damping'] == 0.85].index[0]
    ref_iters = df.loc[ref_idx, 'iterations']
    ref_alpha = 0.85
    iters_theory = ref_iters * (1 - ref_alpha) / (1 - alphas_theory)
    ax.plot(alphas_theory, iters_theory, 'r--', linewidth=1.5,
            label='Theory: k ∝ 1/(1-alpha)', alpha=0.7)
    ax.legend(fontsize=10)

    # Plot 2: Eigenvalue gap
    ax = axes[0, 1]
    ax.plot(df['damping'], df['eigenvalue_gap'], 'go-', linewidth=2, markersize=10, label='Observed')
    ax.plot(df['damping'], 1 - df['damping'], 'r--', linewidth=1.5, label='Theory: gap = 1-alpha', alpha=0.7)
    ax.set_xlabel('Damping Factor (alpha)', fontsize=12)
    ax.set_ylabel('Eigenvalue Gap (lambda₁ - lambda₂)', fontsize=12)
    ax.set_title('Eigenvalue Gap = 1 - alpha (Exact)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Ranking stability
    ax = axes[1, 0]
    ax.bar(df['damping'], df['ranking_changes'], color='steelblue', alpha=0.8, width=0.04)
    ax.set_xlabel('Damping Factor (alpha)', fontsize=12)
    ax.set_ylabel('Top-5 Ranking Changes\n(vs alpha=0.85 baseline)', fontsize=12)
    ax.set_title('Rankings Stabilize at Higher alpha', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Top page score
    ax = axes[1, 1]
    ax.plot(df['damping'], df['top_score'], 'mo-', linewidth=2, markersize=10)
    ax.set_xlabel('Damping Factor (alpha)', fontsize=12)
    ax.set_ylabel('Top Page Score', fontsize=12)
    ax.set_title('Higher alpha = More Concentrated Scores', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'exp5_pagerank_damping.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: exp5_pagerank_damping.png")


def main():
    print("PageRank Demo: Computing Dominant Eigenvector")
    print("=" * 70)

    # Create web graph
    n_pages = 12
    G = create_web_graph(n_pages=n_pages, seed=42)

    print(f"\nCreated web graph with {n_pages} pages")
    if HAS_NETWORKX and not isinstance(G, np.ndarray):
        print(f"Number of links: {G.number_of_edges()}")
        print(f"Average out-degree: {G.number_of_edges() / n_pages:.2f}")
    else:
        num_links = int(np.sum(G))
        print(f"Number of links: {num_links}")
        print(f"Average out-degree: {num_links / n_pages:.2f}")

    # Create Google matrix
    damping = 0.85
    M = create_google_matrix(G, damping=damping)

    print(f"\nGoogle matrix properties:")
    print(f"  Damping factor: {damping}")
    print(f"  Matrix is stochastic (columns sum to 1): {np.allclose(M.sum(axis=0), 1)}")

    # Method 1: Power iteration (our implementation)
    print("\n" + "-" * 70)
    print("Method 1: Power Iteration (Our Implementation)")
    print("-" * 70)
    pr_power, iters_power = pagerank_power_iteration(M, max_iter=1000, tol=1e-10)
    print(f"Converged in {iters_power} iterations")
    print(f"PageRank scores (top 5):")
    top_pages_power = np.argsort(pr_power)[::-1][:5]
    for i, page in enumerate(top_pages_power):
        print(f"  Rank {i+1}: Page {page} (score: {pr_power[page]:.6f})")

    # Method 2: Using power iteration from iterative_methods
    print("\n" + "-" * 70)
    print("Method 2: Power Iteration (From iterative_methods.py)")
    print("-" * 70)
    # Note: M^T because our power_iteration expects A*x = lambda*x format
    eigenvalue, eigenvector, iters, converged = power_iteration(M.T, max_iter=1000, tol=1e-10)
    pr_iter = np.abs(eigenvector) / np.sum(np.abs(eigenvector))  # Normalize
    print(f"Converged: {converged}, Iterations: {iters}")
    print(f"Dominant eigenvalue: {eigenvalue:.10f} (should be ~= 1.0)")
    print(f"PageRank scores match: {np.allclose(pr_power, pr_iter, atol=1e-6)}")

    # Method 3: NetworkX (for validation)
    print("\n" + "-" * 70)
    print("Method 3: NetworkX Built-in PageRank")
    print("-" * 70)
    pr_nx = pagerank_networkx(G, damping=damping)
    pr_nx_array = np.array([pr_nx[i] for i in range(n_pages)])
    print(f"PageRank scores match our implementation: {np.allclose(pr_power, pr_nx_array, atol=1e-6)}")
    print(f"Maximum difference: {np.max(np.abs(pr_power - pr_nx_array)):.2e}")

    # Compare top pages
    print("\n" + "-" * 70)
    print("Top 5 Most Important Pages:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Page':<6} {'Power Iter':<12} {'NetworkX':<12} {'Difference':<12}")
    print("-" * 70)
    for i, page in enumerate(top_pages_power):
        diff = abs(pr_power[page] - pr_nx_array[page])
        print(f"{i+1:<6} {page:<6} {pr_power[page]:<12.8f} {pr_nx_array[page]:<12.8f} {diff:<12.2e}")

    # Visualize
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'pagerank_demo.png')

    visualize_pagerank(G, pr_power,
                      title=f"PageRank on {n_pages}-Page Web Graph (Power Iteration)",
                      output_file=output_file)

    # Show eigenvalue spectrum
    print("\n" + "-" * 70)
    print("Eigenvalue Analysis:")
    print("-" * 70)
    eigenvalues = np.linalg.eigvals(M)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    print(f"Top 5 eigenvalues (by magnitude):")
    for i in range(min(5, len(eigenvalues))):
        print(f"  lambda_{i+1}: {eigenvalues[i]:.8f}")
    print(f"\nEigenvalue gap (lambda_1 - lambda_2): {eigenvalues[0] - eigenvalues[1]:.6f}")
    print(f"(Larger gap = faster convergence of power iteration)")

    # Run damping factor experiment
    print("\n\n" + "=" * 70)
    print("Running Damping Factor Sensitivity Analysis...")
    print("=" * 70)
    experiment_damping_factor_analysis(G, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("PageRank demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def f_r13a(x):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    mask = (x >= 1) & (x <= 2)
    y[mask] = np.exp(x[mask] - 1) + np.exp(2 - x[mask])
    return y

def composite_trapezoid(y, h):
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 points")
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def analyze_grid_alignment(N, a, b):
    x = np.linspace(a, b, N)
    hits_1 = np.any(np.abs(x - 1.0) < 1e-10)
    hits_2 = np.any(np.abs(x - 2.0) < 1e-10)
    return hits_1, hits_2

def run_r13():
    a, b = 0.0, 3.0
    I_exact = 2.0 * np.e - 2.0  
    
    N_values = np.array([11, 21, 31, 41, 51, 61, 81, 101, 161, 201, 321, 401, 641, 801, 1281, 1601])
    
    h_vals = []
    errors = []
    aligned_with_discontinuity = []
    
    print("="*70)
    print("R1.3: INTEGRATION OVER DISCONTINUITIES")
    print("="*70)
    print(f"Function: f(x) = exp(x-1)+exp(2-x) for 1≤x≤2, 0 otherwise")
    print(f"Domain: [0, 3]")
    print(f"Exact integral: I = 2e - 2 = {I_exact:.12f}")
    print("="*70)
    
    print(f"\n{'N':>6} {'h':>12} {'I_trap':>20} {'|error|':>14} {'Grid hits':>20}")
    print("-"*80)
    
    for N in N_values:
        x = np.linspace(a, b, N)
        h = (b - a) / (N - 1)
        y = f_r13a(x)
        
        I_trap = composite_trapezoid(y, h)
        err = abs(I_trap - I_exact)
        
        hits_1, hits_2 = analyze_grid_alignment(N, a, b)
        alignment_str = f"x=1:{hits_1}, x=2:{hits_2}"
        aligned_with_discontinuity.append(hits_1 and hits_2)
        
        h_vals.append(h)
        errors.append(err)
        
        print(f"{N:6d} {h:12.5e} {I_trap:20.12f} {err:14.5e} {alignment_str:>20}")
    
    h_vals = np.array(h_vals)
    errors = np.array(errors)
    
    print("-"*80)
    min_idx = np.argmin(errors)
    print(f"Min error {errors[min_idx]:.5e} at N={N_values[min_idx]}, h={h_vals[min_idx]:.5e}")
    
    aligned_indices = [i for i, aligned in enumerate(aligned_with_discontinuity) if aligned]
    non_aligned_indices = [i for i, aligned in enumerate(aligned_with_discontinuity) if not aligned]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.loglog(h_vals, errors, 'o-', markersize=6, label='All points')
    
    if aligned_indices:
        ax1.loglog(h_vals[aligned_indices], errors[aligned_indices], 'gs', 
                   markersize=8, label='Grid aligned with discontinuities', alpha=0.7)
    
    mid = len(h_vals)//2
    C1 = errors[mid] / h_vals[mid]
    C2 = errors[mid] / h_vals[mid]**2
    h_ref = np.linspace(h_vals.min(), h_vals.max(), 50)
    ax1.loglog(h_ref, C1 * h_ref, 'k--', alpha=0.5, label='O(h)')
    ax1.loglog(h_ref, C2 * h_ref**2, 'r--', alpha=0.5, label='O(h²)')
    
    ax1.invert_xaxis()
    ax1.set_xlabel('Step size h')
    ax1.set_ylabel('Absolute error')
    ax1.set_title('Trapezoid error vs step size')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend()
    
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            rate = np.log(errors[i]/errors[i-1]) / np.log(h_vals[i]/h_vals[i-1])
            rates.append(rate)
    
    if rates:
        ax2.plot(h_vals[1:], rates, 'o-', markersize=6)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Order 1')
        ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Order 2')
        ax2.set_xscale('log')
        ax2.invert_xaxis()
        ax2.set_xlabel('Step size h')
        ax2.set_ylabel('Convergence rate')
        ax2.set_title('Empirical convergence rate')
        ax2.grid(True, which='both', ls=':', alpha=0.6)
        ax2.legend()
        
        print(f"\nConvergence rate analysis:")
        print(f"  Mean rate: {np.mean(rates):.2f}")
        print(f"  Std dev: {np.std(rates):.2f}")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    
    x_fine = np.linspace(a, b, 1000)
    y_fine = f_r13a(x_fine)
    
    N_coarse = 11
    x_coarse = np.linspace(a, b, N_coarse)
    y_coarse = f_r13a(x_coarse)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x)')
    plt.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Discontinuities')
    plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
    plt.fill_between(x_fine, 0, y_fine, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function with discontinuities at x=1, x=2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_fine, y_fine, 'b-', linewidth=1, alpha=0.5, label='Actual function')
    plt.plot(x_coarse, y_coarse, 'ro-', markersize=6, label=f'Trapezoid grid (N={N_coarse})')
    plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Trapezoid approximation missing discontinuities')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("KEY OBSERVATIONS:")
    print("="*70)
    print("1. Convergence rate is approximately O(h) due to discontinuities")
    print("   (would be O(h²) for smooth functions)")
    print("2. Error oscillates depending on grid alignment with discontinuities")
    print("3. When grid points hit x=1 and x=2, error can be slightly better")
    print("4. The discontinuities limit the convergence rate from O(h²) to O(h)")
    print("="*70)

if __name__ == "__main__":  
    run_r13()
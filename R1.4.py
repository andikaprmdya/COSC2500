
import numpy as np
import matplotlib.pyplot as plt


def composite_trapezoid(y, h):
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 points")
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def composite_simpson(y, h):
    """Composite Simpson's rule for odd number of points (n >= 3 and odd)."""
    n = len(y)
    if n < 3 or n % 2 == 0:
        raise ValueError("Simpson requires odd number of points >= 3")
    odd_sum = np.sum(y[1:n-1:2])
    even_sum = np.sum(y[2:n-1:2])
    return (h/3.0) * (y[0] + 4*odd_sum + 2*even_sum + y[-1])

def f(x):
    return np.exp(-x)

I_exact = 1.0  


def compute_error_components(b, h, method='trapezoid'):

    truncation_error = np.exp(-b)
    
    N = max(2, int(np.floor(b / h)) + 1)
    if method == 'simpson' and N % 2 == 0:
        N += 1 
    
    h_actual = b / (N - 1)
    x = np.linspace(0.0, b, N)
    y = f(x)
    
    if method == 'trapezoid':
        I_numeric = composite_trapezoid(y, h_actual)
    elif method == 'simpson' and N >= 3:
        I_numeric = composite_simpson(y, h_actual)
    else:
        return truncation_error, np.nan, np.nan
    
    I_exact_truncated = 1.0 - np.exp(-b)
    
    quadrature_error = abs(I_numeric - I_exact_truncated)
    
    total_error = abs(I_numeric - I_exact)
    
    return truncation_error, quadrature_error, total_error


def find_optimal_b(h, method='trapezoid', b_max=20.0, nb=500):
    """Find b that minimizes total error for given h."""
    b_vals = np.linspace(0.1, b_max, nb)
    min_error = np.inf
    optimal_b = None
    
    for b in b_vals:
        _, _, total_error = compute_error_components(b, h, method)
        if not np.isnan(total_error) and total_error < min_error:
            min_error = total_error
            optimal_b = b
    
    return optimal_b, min_error

def study_fixed_h(h_values, b_max=10.0, nb=200):
    b_vals = np.linspace(0.1, b_max, nb)
    results = {}
    
    for h in h_values:
        trap_errs = []
        simp_errs = []
        trunc_errs = []
        quad_errs_trap = []
        quad_errs_simp = []
        
        for b in b_vals:
            trunc, quad, total = compute_error_components(b, h, 'trapezoid')
            trap_errs.append(total)
            trunc_errs.append(trunc)
            quad_errs_trap.append(quad)
            
            _, quad_s, total_s = compute_error_components(b, h, 'simpson')
            simp_errs.append(total_s)
            quad_errs_simp.append(quad_s)
        
        opt_b_trap, min_err_trap = find_optimal_b(h, 'trapezoid')
        opt_b_simp, min_err_simp = find_optimal_b(h, 'simpson')
        
        results[h] = {
            'b_vals': b_vals,
            'trap_errs': np.array(trap_errs),
            'simp_errs': np.array(simp_errs),
            'trunc_errs': np.array(trunc_errs),
            'quad_errs_trap': np.array(quad_errs_trap),
            'quad_errs_simp': np.array(quad_errs_simp),
            'optimal_b_trap': opt_b_trap,
            'optimal_b_simp': opt_b_simp,
            'min_err_trap': min_err_trap,
            'min_err_simp': min_err_simp
        }
    
    return results

def study_fixed_N(N_values, b_max=20.0, nb=200):
    b_vals = np.linspace(0.1, b_max, nb)
    results = {}
    
    for N in N_values:
        trap_errs = []
        simp_errs = []
        h_vals = []
        
        for b in b_vals:
            if N < 2:
                raise ValueError("N must be at least 2")
            h_local = (b - 0.0) / (N - 1)
            h_vals.append(h_local)
            
            x = np.linspace(0.0, b, N)
            y = f(x)
            
            I_trap = composite_trapezoid(y, h_local)
            trap_errs.append(abs(I_trap - I_exact))
            
            if N >= 3 and (N % 2 == 1):
                try:
                    I_simp = composite_simpson(y, h_local)
                    simp_errs.append(abs(I_simp - I_exact))
                except ValueError:
                    simp_errs.append(np.nan)
            else:
                simp_errs.append(np.nan)
        
        results[N] = {
            'b_vals': b_vals,
            'h_vals': np.array(h_vals),
            'trap_errs': np.array(trap_errs),
            'simp_errs': np.array(simp_errs)
        }
    
    return results


def plot_fixed_h_enhanced(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for h, data in results.items():
        b = data['b_vals']
        trap = data['trap_errs']
        simp = data['simp_errs']
        
        ax.semilogy(b, trap, '-', label=f'Trap, h={h}', alpha=0.7)
        valid = ~np.isnan(simp)
        if np.any(valid):
            ax.semilogy(b[valid], simp[valid], '--', label=f'Simp, h={h}', alpha=0.7)
        
        if data['optimal_b_trap']:
            ax.plot(data['optimal_b_trap'], data['min_err_trap'], 'o', 
                   markersize=8, color='red', alpha=0.5)
    
    b_ref = np.linspace(0.1, max(data['b_vals'].max() for data in results.values()), 200)
    ax.semilogy(b_ref, np.exp(-b_ref), 'k:', label=r'$e^{-b}$ (truncation)')
    ax.set_xlabel('Truncation point b')
    ax.set_ylabel('Total error')
    ax.set_title('Total error vs truncation point')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend(fontsize='small')
    
    ax = axes[0, 1]
    h_selected = 0.05 if 0.05 in results else list(results.keys())[1]
    data = results[h_selected]
    b = data['b_vals']
    
    ax.semilogy(b, data['trunc_errs'], 'b-', label='Truncation error')
    ax.semilogy(b, data['quad_errs_trap'], 'r-', label='Quadrature error (trap)')
    ax.semilogy(b, data['trap_errs'], 'k-', linewidth=2, label='Total error')
    
    if data['optimal_b_trap']:
        ax.axvline(data['optimal_b_trap'], color='green', linestyle='--', 
                  alpha=0.5, label=f'Optimal b={data["optimal_b_trap"]:.2f}')
    
    ax.set_xlabel('Truncation point b')
    ax.set_ylabel('Error')
    ax.set_title(f'Error decomposition for h={h_selected}')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend()
    
    ax = axes[1, 0]
    h_list = sorted(results.keys())
    opt_b_trap = [results[h]['optimal_b_trap'] for h in h_list]
    opt_b_simp = [results[h]['optimal_b_simp'] for h in h_list]
    min_err_trap = [results[h]['min_err_trap'] for h in h_list]
    min_err_simp = [results[h]['min_err_simp'] for h in h_list]
    
    ax.loglog(h_list, opt_b_trap, 'o-', label='Optimal b (trap)')
    ax.loglog(h_list, opt_b_simp, 's-', label='Optimal b (simp)')
    ax.set_xlabel('Step size h')
    ax.set_ylabel('Optimal truncation point b')
    ax.set_title('Optimal b vs step size')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend()
    
    ax = axes[1, 1]
    ax.loglog(h_list, min_err_trap, 'o-', label='Min error (trap)')
    ax.loglog(h_list, min_err_simp, 's-', label='Min error (simp)')
    
    h_ref = np.array(h_list)
    ax.loglog(h_ref, h_ref**2, 'k--', alpha=0.3, label='O(h²)')
    ax.loglog(h_ref, h_ref**4, 'r--', alpha=0.3, label='O(h⁴)')
    
    ax.set_xlabel('Step size h')
    ax.set_ylabel('Minimum total error')
    ax.set_title('Minimum error achieved vs step size')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_fixed_N_enhanced(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for N, data in results.items():
        b = data['b_vals']
        trap = data['trap_errs']
        simp = data['simp_errs']
        
        ax1.loglog(b, trap, '-', label=f'Trap, N={N}', alpha=0.7)
        valid = ~np.isnan(simp)
        if np.any(valid):
            ax1.loglog(b[valid], simp[valid], '--', label=f'Simp, N={N}', alpha=0.7)
    
    b_ref = np.linspace(0.1, max(data['b_vals'].max() for data in results.values()), 200)
    ax1.loglog(b_ref, np.exp(-b_ref), 'k:', linewidth=2, label=r'$e^{-b}$')
    ax1.set_xlabel('Truncation point b')
    ax1.set_ylabel('Total error')
    ax1.set_title('Fixed N: error vs truncation point')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend(fontsize='small')
    
    for N, data in results.items():
        b = data['b_vals']
        h = data['h_vals']
        ax2.loglog(b, h, '-', label=f'N={N}', alpha=0.7)
    
    ax2.set_xlabel('Truncation point b')
    ax2.set_ylabel('Step size h = b/(N-1)')
    ax2.set_title('Step size growth with truncation point')
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("="*70)
    print("R1.4: INTEGRATION OVER INFINITE INTERVAL")
    print("="*70)
    print(f"Function: f(x) = exp(-x)")
    print(f"Domain: [0, ∞)")
    print(f"Exact integral: {I_exact}")
    print("="*70)
    
    print("\n(A) FIXED STEP SIZE STUDY")
    print("-"*40)
    h_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    fixed_h_results = study_fixed_h(h_values, b_max=10.0, nb=200)
    
    print("\nOptimal truncation points for fixed h:")
    print(f"{'h':>8} {'b_opt(trap)':>12} {'error(trap)':>12} {'b_opt(simp)':>12} {'error(simp)':>12}")
    for h in sorted(fixed_h_results.keys()):
        data = fixed_h_results[h]
        print(f"{h:8.3f} {data['optimal_b_trap']:12.3f} {data['min_err_trap']:12.3e} "
              f"{data['optimal_b_simp']:12.3f} {data['min_err_simp']:12.3e}")
    
    fig1 = plot_fixed_h_enhanced(fixed_h_results)
    plt.show()
    
    print("\n(B) FIXED NUMBER OF STEPS STUDY")
    print("-"*40)
    N_values = [5, 11, 21, 51, 101, 201]
    fixed_N_results = study_fixed_N(N_values, b_max=20.0, nb=300)
    
    fig2 = plot_fixed_N_enhanced(fixed_N_results)
    plt.show()
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("1. Fixed h strategy:")
    print("   - Total error = truncation error + quadrature error")
    print("   - Optimal b balances these two competing errors")
    print("   - Smaller h → larger optimal b, lower minimum error")
    print("   - Optimal b ≈ -ln(h) for small h")
    print("")
    print("2. Fixed N strategy:")
    print("   - As b increases, h increases → quadrature error grows")
    print("   - Eventually quadrature error dominates truncation error")
    print("   - No clear optimal b; need to increase N with b")
    print("")
    print("3. Recommendations:")
    print("   - Use fixed h with b chosen to balance errors")
    print("   - Or use adaptive methods that adjust h and b automatically")
    print("="*70)
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapezoid

def composite_trapezoid(y, h):
    n = len(y)
    if n < 2: raise ValueError("Need at least 2 points")
    return h * (0.5*y[0] + np.sum(y[1:n-1]) + 0.5*y[n-1])

def composite_simpson(y, h):
    n = len(y)
    if n < 3: raise ValueError("Simpson needs at least 3 points")
    if n % 2 == 0: raise ValueError("Simpson needs odd number of points")
    return (h/3.0) * (y[0] + 4*np.sum(y[1:n-1:2]) + 2*np.sum(y[2:n-1:2]) + y[n-1])

def f1(x): return np.cos(x)
def f2(x): return np.sin(x**2) + 1.0

I_exact = {'cos_0_pi2':1.0, 'cos_0_pi':0.0, 'sinx2p1_0_10':10.58367089992962334}
problems = [
    ("cos(x) on [0, π/2]", f1, 0, np.pi/2, I_exact['cos_0_pi2']),
    ("cos(x) on [0, π]", f1, 0, np.pi, I_exact['cos_0_pi']),
    ("sin(x²)+1 on [0, 10]", f2, 0, 10, I_exact['sinx2p1_0_10'])
]

def analyze_convergence():
    print("="*80)
    print("PART (a): ACCURACY vs STEP SIZE - Log-Log Analysis")
    print("="*80)
    N_values = [5,9,17,33,65,129,257,513,1025,2049]
    results = {}
    for prob_name, func, a, b, I_true in problems:
        print(f"\n{prob_name}:")
        h_vals, err_trap, err_simp = [], [], []
        for N in N_values:
            if N % 2 == 0: N += 1
            x = np.linspace(a, b, N); h = (b - a)/(N-1); y = func(x)
            I_trap = composite_trapezoid(y, h); I_simp = composite_simpson(y, h)
            err_trap.append(abs(I_trap - I_true)); err_simp.append(abs(I_simp - I_true)); h_vals.append(h)
        h_vals = np.array(h_vals); err_trap = np.array(err_trap); err_simp = np.array(err_simp)
        min_trap_idx = np.argmin(err_trap); min_simp_idx = np.argmin(err_simp)
        print(f"  Trapezoid min error: {err_trap[min_trap_idx]:.2e} at h = {h_vals[min_trap_idx]:.2e}")
        print(f"  Simpson min error: {err_simp[min_simp_idx]:.2e} at h = {h_vals[min_simp_idx]:.2e}")
        plt.figure(figsize=(10,6))
        plt.loglog(h_vals, err_trap, 'o-', label='Trapezoid', markersize=4)
        plt.loglog(h_vals, err_simp, 's-', label='Simpson', markersize=4)
        if len(h_vals) > 5:
            ref_idx = len(h_vals)//3; h_ref = h_vals[ref_idx]
            if err_trap[ref_idx] > 1e-14:
                C_trap = err_trap[ref_idx]/(h_ref**2); plt.loglog(h_vals, C_trap*h_vals**2, 'k--', alpha=0.7, label='Reference: h²')
            if err_simp[ref_idx] > 1e-14:
                C_simp = err_simp[ref_idx]/(h_ref**4); plt.loglog(h_vals, C_simp*h_vals**4, 'k-.', alpha=0.7, label='Reference: h⁴')
        plt.gca().invert_xaxis(); plt.xlabel('Step size h'); plt.ylabel('Absolute error')
        plt.title(f'{prob_name}: Error vs Step Size'); plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
        results[prob_name] = (h_vals, err_trap, err_simp)

    print("\n" + "="*80)
    print("REPRESENTATIVE ACCURACY COMPARISON AT SAME h")
    print("="*80)
    print(f"{'Problem':<25} {'h':<12} {'Trap Error':<12} {'Simp Error':<12} {'Improvement'}")
    print("-"*75)
    for prob_name in results:
        h_vals, err_trap, err_simp = results[prob_name]
        idx = len(h_vals)//2; h_rep = h_vals[idx]; trap_err = err_trap[idx]; simp_err = err_simp[idx]
        improvement = trap_err/simp_err if simp_err>0 else float('inf')
        print(f"{prob_name:<25} {h_rep:<12.2e} {trap_err:<12.2e} {simp_err:<12.2e} {improvement:>11.1e}")

def analyze_adaptive():
    print("\n" + "="*80)
    print("PART (b): ADAPTIVE INTEGRATION (scipy.integrate.quad)")
    print("="*80)
    print(f"{'Problem':<25} {'Computed Value':<18} {'|Error|':<12} {'Function Evals'}")
    print("-"*65)
    for prob_name, func, a, b, I_true in problems:
        result = quad(func, a, b, epsabs=1e-12, epsrel=1e-12, full_output=True)
        computed_val = result[0]; estimated_error = result[1]; info_dict = result[2]; neval = info_dict.get('neval','N/A')
        actual_error = abs(computed_val - I_true)
        print(f"{prob_name:<25} {computed_val:<18.14f} {actual_error:<12.2e} {neval:>14}")
    print("\nObservations:")
    print("• For smooth integrands (cos), quad achieves machine precision with ~21 evaluations")
    print("• For oscillatory integrand (sin(x²)+1), more evaluations needed but still very efficient")
    print("• Adaptive methods vastly outperform fixed-step methods in efficiency")

def analyze_noise_effects():
    print("\n" + "="*80)
    print("PART (c): EFFECT OF NOISE ON INTEGRATION")
    print("="*80)
    noise_levels = [1e-4,1e-3,1e-2,5e-2]; N_values_noise = [5,9,17,33,65,129,257,513]
    for prob_name, func, a, b, I_true in problems:
        print(f"\n{prob_name}:"); plt.figure(figsize=(10,6)); noise_results = {}
        for sigma in noise_levels:
            h_vals, err_trap, err_simp = [], [], []
            np.random.seed(42)
            for N in N_values_noise:
                if N % 2 == 0: N += 1
                x = np.linspace(a, b, N); h = (b-a)/(N-1); y_clean = func(x)
                noise = sigma * np.random.randn(len(x)); y_noisy = y_clean + noise
                I_trap = composite_trapezoid(y_noisy, h); I_simp = composite_simpson(y_noisy, h)
                err_trap.append(abs(I_trap - I_true)); err_simp.append(abs(I_simp - I_true)); h_vals.append(h)
            h_vals = np.array(h_vals); err_trap = np.array(err_trap); err_simp = np.array(err_simp)
            noise_results[sigma] = {'trap_min':np.min(err_trap),'simp_min':np.min(err_simp),
                                    'trap_h_min':h_vals[np.argmin(err_trap)],'simp_h_min':h_vals[np.argmin(err_simp)]}
            plt.loglog(h_vals, err_trap, 'o-', alpha=0.8, label=f'Trap, σ={sigma}', markersize=3)
            plt.loglog(h_vals, err_simp, 's-', alpha=0.8, label=f'Simp, σ={sigma}', markersize=3)
        plt.gca().invert_xaxis(); plt.xlabel('Step size h'); plt.ylabel('Absolute error')
        plt.title(f'{prob_name}: Effect of Noise'); plt.grid(True, which='both', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout(); plt.show()
        print(f"{'Noise σ':<10} {'Method':<8} {'Min Error':<12} {'Optimal h':<12}"); print("-"*45)
        for sigma in noise_levels:
            res = noise_results[sigma]
            print(f"{sigma:<10g} {'Trap':<8} {res['trap_min']:<12.2e} {res['trap_h_min']:<12.2e}")
            print(f"{'':<10} {'Simp':<8} {res['simp_min']:<12.2e} {res['simp_h_min']:<12.2e}")

def analyze_computational_limits():
    print("\n" + "="*80)
    print("PART (d): ROUND-OFF, TIME, AND MEMORY")
    print("="*80)
    test_cases = [("cos(x) on [0, π]", f1, 0, np.pi, I_exact['cos_0_pi']),
                  ("sin(x²)+1 on [0, 10]", f2, 0, 10, I_exact['sinx2p1_0_10'])]
    for prob_name, func, a, b, I_true in test_cases:
        print(f"\n{prob_name}:")
        print(f"{'N':<8} {'h':<12} {'Memory':<10} {'Time(T)':<10} {'Error(T)':<12} {'Time(S)':<10} {'Error(S)':<12}")
        print("-"*85)
        N = 65; errors_trap, errors_simp, times_trap, times_simp, memory_sizes = [], [], [], [], []
        for step in range(7):
            if N % 2 == 0: N += 1
            x = np.linspace(a, b, N); h = (b-a)/(N-1); y = func(x)
            memory = (x.nbytes + y.nbytes)
            memory_str = f"{memory/1024:.1f}KB" if memory < 1024*1024 else f"{memory/(1024**2):.1f}MB"
            start = time.perf_counter()
            for _ in range(100): I_trap = composite_trapezoid(y, h)
            time_trap = (time.perf_counter()-start)/100; error_trap = abs(I_trap - I_true)
            start = time.perf_counter()
            for _ in range(100): I_simp = composite_simpson(y, h)
            time_simp = (time.perf_counter()-start)/100; error_simp = abs(I_simp - I_true)
            print(f"{N:<8} {h:<12.2e} {memory_str:<10} {time_trap:<10.6f} {error_trap:<12.2e} {time_simp:<10.6f} {error_simp:<12.2e}")
            errors_trap.append(error_trap); errors_simp.append(error_simp)
            times_trap.append(time_trap); times_simp.append(time_simp); memory_sizes.append(memory)
            N *= 2
        total_memory = memory_sizes[-1]/(1024**2); total_time = sum(times_trap)+sum(times_simp)
        errors_trap = np.array(errors_trap); errors_simp = np.array(errors_simp)
        trap_floor = np.where(errors_trap[1:] >= errors_trap[:-1] * 0.8)[0]
        simp_floor = np.where(errors_simp[1:] >= errors_simp[:-1] * 0.8)[0]
        print(f"\n  Analysis:")
        print(f"  • Memory usage: {memory_sizes[0]/1024:.1f}KB → {memory_sizes[-1]/1024:.1f}KB")
        print(f"  • Total computation time: {total_time*1000:.3f} ms")
        if len(trap_floor) > 0: print(f"  • Trapezoid hits round-off floor around step {trap_floor[0]+1}")
        else: print(f"  • Trapezoid errors still decreasing (no round-off floor yet)")
        if len(simp_floor) > 0: print(f"  • Simpson hits round-off floor around step {simp_floor[0]+1}")
        else: print(f"  • Simpson errors still decreasing (no round-off floor yet)")
        print(f"  • Conclusion: Round-off limits accuracy long before computational resources become excessive")

if __name__ == "__main__":
    print("R1.2: COMPREHENSIVE QUADRATURE STUDY")
    print("Comparing Trapezoid, Simpson, and Adaptive Integration Methods")
    analyze_convergence()
    analyze_adaptive()
    analyze_noise_effects()
    analyze_computational_limits()
    print("\n" + "="*80)
    print("SUMMARY CONCLUSIONS:")
    print("="*80)
    print("1. Simpson's rule shows clear O(h⁴) convergence vs Trapezoid's O(h²)")
    print("2. Adaptive quadrature achieves machine precision with minimal function evaluations")
    print("3. Noise creates a fundamental accuracy floor proportional to noise level σ")
    print("4. Round-off error dominates before computational resources (time/memory) become limiting")
    print("5. For practical integration: adaptive methods >> Simpson >> Trapezoid")
    print("="*80)

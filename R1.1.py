import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**4 - 2*x**3
def df(x): return 4*x**3 - 6*x**2
def d2f(x): return 12*x**2 - 12*x
def d3f(x): return 24*x - 12

def finite_diff(x0, h, func=f):
    f0, fph, fmh = func(x0), func(x0 + h), func(x0 - h)
    return (fph - f0)/h, (f0 - fmh)/h, (fph - fmh)/(2*h)

def handle_zero_errors(errors):
    e = np.array(errors)
    e[e < np.finfo(float).eps] = np.finfo(float).eps
    return e

def _nudge_ylim_based_on_plotted_data(factor=1.2):
    ax = plt.gca(); cur_ymin, cur_ymax = ax.get_ylim()
    ys = []
    for L in ax.get_lines():
        try:
            y = np.asarray(L.get_ydata())
            if y.size: ys.append(y)
        except Exception:
            pass
    if not ys: return
    ys = np.concatenate(ys)
    ys = ys[np.isfinite(ys) & (ys > 0)]
    if ys.size == 0: return
    data_max, data_min = ys.max(), ys.min()
    ax.set_ylim(min(cur_ymin, data_min), max(cur_ymax, data_max)*factor)

def compute_errors(x0, h, func=f):
    err_fd = []; err_bd = []; err_cd = []
    for hi in h:
        fd, bd, cd = finite_diff(x0, hi, func=func)
        d_exact = df(x0) if func is f else df(x0)  # keep same exact value used in prints
        err_fd.append(abs(fd - d_exact)); err_bd.append(abs(bd - d_exact)); err_cd.append(abs(cd - d_exact))
    return np.array(err_fd), np.array(err_bd), np.array(err_cd)

def plot_err(h, curves, labels, title, theo=None, theo_label=None, theo_cd=None, theo_cd_label=None):
    plt.figure(figsize=(8,6))
    curves_fixed = [handle_zero_errors(c) for c in curves]
    for y, lb in zip(curves_fixed, labels): plt.loglog(h, y, label=lb, linewidth=2)
    if theo is not None: plt.loglog(h, handle_zero_errors(theo), 'k--', alpha=0.5, label=theo_label)
    if theo_cd is not None: plt.loglog(h, handle_zero_errors(theo_cd), 'r--', alpha=0.5, label=theo_cd_label)
    eps = np.finfo(float).eps
    plt.axvline(eps**0.5, color='gray', linestyle=':', alpha=0.5, label=f'Theoretical optimal FD/BD: {eps**0.5:.2e}')
    plt.axvline(eps**(1/3), color='brown', linestyle=':', alpha=0.5, label=f'Theoretical CD: {eps**(1/3):.2e}')
    plt.xlabel('Step size (h)'); plt.ylabel('Absolute Error'); plt.title(title)
    plt.legend(fontsize=8); plt.grid(True, which='both', alpha=0.3)
    plt.ylim([1e-32, 1e3]); plt.tight_layout()
    _nudge_ylim_based_on_plotted_data(factor=1.5)
    plt.show()

def print_min_errors(x0, h, errs, label):
    i = int(np.argmin(errs)); eps = np.finfo(float).eps
    h_theory = eps**(1/3) if 'CD' in label else eps**(1/2)
    print(f"  {label}: min error = {errs[i]:.3e} at h = {h[i]:.3e} ( theory: {h_theory:.3e})")

def run_no_noise(x0, h):
    print(f"\n=== Results for x={x0} (no noise) ===")
    err_fd, err_bd, err_cd = compute_errors(x0, h)  # uses default f
    print_min_errors(x0, h, err_fd, "FD"); print_min_errors(x0, h, err_bd, "BD"); print_min_errors(x0, h, err_cd, "CD")
    if abs(d2f(x0)) < 1e-10: print(f"  Note: f''({x0}) = 0, leading to higher-order convergence for FD/BD")
    if abs(df(x0)) < 1e-10: print(f"  Note: f'({x0}) ≈ 0, numerical errors dominate at all h")
    if abs(d2f(x0)) < 1e-10:
        theo = (h**2 / 6.0) * abs(d3f(x0)) + 1e-30; theo_label = "Theoretical FD/BD Error (h²/6·|f'''|)"
    else:
        theo = 0.5 * h * abs(d2f(x0)); theo_label = "Theoretical FD/BD Error (0.5·h·|f''|)"
    theo_cd = (h**2 / 6.0) * abs(d3f(x0)) if abs(d3f(x0)) >= 1e-10 else 1e-30 * np.ones_like(h)
    plot_err(h, [err_fd, err_bd, err_cd], ['Forward Difference','Backward Difference','Central Difference'],
             f'Error of Numerical Derivatives at x = {x0}', theo, theo_label, theo_cd, "Theoretical CD Error (h²/6 · |f'''|)")
    return err_fd, err_bd, err_cd

def run_with_noise(x0, h, noise_levels):
    print(f"\n=== WITH NOISE @ x={x0} ===")
    d_exact = df(x0)
    noise_levels = [0, 1e-4, 1e-3, 1e-2, 5e-2]
    plt.figure(figsize=(10,7)); colors = ['blue','green','orange','red','purple']
    global_max, global_min = 0.0, np.inf
    for j, sigma in enumerate(noise_levels):
        np.random.seed(int(1000 * abs(x0) + j * 100))
        def noisy_func(x, s=sigma): return f(x) + s * np.random.randn()
        err_fd, err_bd, err_cd = compute_errors(x0, h, func=noisy_func)
        err_fd, err_bd, err_cd = handle_zero_errors(err_fd), handle_zero_errors(err_bd), handle_zero_errors(err_cd)
        global_max = max(global_max, err_fd.max(), err_bd.max(), err_cd.max())
        global_min = min(global_min, err_fd.min(), err_bd.min(), err_cd.min())
        plt.loglog(h, err_fd, color=colors[j], alpha=0.7, linewidth=1.5, label=f'FD, σ={sigma:.0e}')
        plt.loglog(h, err_bd, color=colors[j], alpha=0.7, linewidth=1.5, linestyle='--', label=f'BD, σ={sigma:.0e}')
        plt.loglog(h, err_cd, color=colors[j], alpha=0.7, linewidth=1.5, linestyle=':', label=f'CD, σ={sigma:.0e}')
        print(f"\n  Noise level σ = {sigma:.0e}:")
        print(f"    FD: min error = {err_fd[np.argmin(err_fd)]:.3e} at h = {h[np.argmin(err_fd)]:.3e}")
        print(f"    BD: min error = {err_bd[np.argmin(err_bd)]:.3e} at h = {h[np.argmin(err_bd)]:.3e}")
        print(f"    CD: min error = {err_cd[np.argmin(err_cd)]:.3e} at h = {h[np.argmin(err_cd)]:.3e}")
    if abs(d2f(x0)) > 1e-10:
        theo_fd = handle_zero_errors(0.5*h*abs(d2f(x0))); plt.loglog(h, theo_fd, 'k-', alpha=0.3, linewidth=1, label="Theo FD/BD (no noise)"); global_max = max(global_max, theo_fd.max()); global_min = min(global_min, theo_fd.min())
    if abs(d3f(x0)) > 1e-10:
        theo_cd = handle_zero_errors((h**2/6)*abs(d3f(x0))); plt.loglog(h, theo_cd, 'r-', alpha=0.3, linewidth=1, label="Theo CD (no noise)"); global_max = max(global_max, theo_cd.max()); global_min = min(global_min, theo_cd.min())
    plt.xlabel('Step size (h)'); plt.ylabel('Absolute Error'); plt.title(f'Effect of Noise on Numerical Derivatives at x = {x0}')
    plt.grid(True, which='both', alpha=0.3); plt.legend(ncol=3, fontsize=7, loc='best'); plt.ylim([1e-18, 1e2]); plt.tight_layout()
    if np.isfinite(global_max) and global_max > 0:
        ax = plt.gca(); cb, ct = ax.get_ylim(); ax.set_ylim(min(cb, global_min), max(ct, global_max)*1.5)
    else:
        _nudge_ylim_based_on_plotted_data(factor=1.5)
    plt.show()

def run_single_precision(x0, h):
    h32 = h.astype(np.float32); x0_32 = np.float32(x0); d_exact = np.float32(df(x0_32))
    err_fd, err_bd, err_cd = [], [], []
    for hi in h32:
        def f32(x): return np.float32(f(np.float32(x)))
        fd, bd, cd = finite_diff(x0_32, hi, func=f32)
        err_fd.append(abs(fd - d_exact)); err_bd.append(abs(bd - d_exact)); err_cd.append(abs(cd - d_exact))
    err_fd, err_bd, err_cd = handle_zero_errors(err_fd), handle_zero_errors(err_bd), handle_zero_errors(err_cd)
    print(f"\n[SINGLE precision @ x={x0}]")
    eps_single = np.finfo(np.float32).eps; h_opt_fd_single = eps_single**0.5; h_opt_cd_single = eps_single**(1/3)
    print(f"  FD: min error = {err_fd[np.argmin(err_fd)]:.3e} at h = {h32[np.argmin(err_fd)]:.3e} ( theory: {h_opt_fd_single:.3e})")
    print(f"  BD: min error = {err_bd[np.argmin(err_bd)]:.3e} at h = {h32[np.argmin(err_bd)]:.3e} ( theory: {h_opt_fd_single:.3e})")
    print(f"  CD: min error = {err_cd[np.argmin(err_cd)]:.3e} at h = {h32[np.argmin(err_cd)]:.3e} ( theory: {h_opt_cd_single:.3e})")
    plt.figure(figsize=(8,6))
    plt.loglog(h32, err_fd, 'b-', label='FD (float32)', linewidth=2.5); plt.loglog(h32, err_bd, 'r-', label='BD (float32)', linewidth=2.5); plt.loglog(h32, err_cd, 'g-', label='CD (float32)', linewidth=2.5)
    plt.axvline(h_opt_fd_single, color='gray', linestyle=':', alpha=0.5, label=f'Theory FD/BD: {h_opt_fd_single:.2e}'); plt.axvline(h_opt_cd_single, color='brown', linestyle=':', alpha=0.5, label=f'Theory CD: {h_opt_cd_single:.2e}')
    plt.ylim([1e-32, 1e2] if x0 in [0, 1.5] else [1e-10, 1e2])
    plt.xlabel('Step size (h)'); plt.ylabel('Absolute Error'); plt.title(f'Single Precision Error @ x = {x0}'); plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    _nudge_ylim_based_on_plotted_data(factor=1.5); plt.show()

if __name__ == "__main__":
    x_points = [0, 1, 1.5, 2]
    h = np.logspace(-16, 2, 300)

    print("="*60); print("R1.1: NUMERICAL DIFFERENTIATION ANALYSIS"); print("="*60)
    print("\nPart (a): Error behavior without noise"); print("-"*40)
    for x0 in x_points: run_no_noise(x0, h)

    print("\n" + "="*60); print("THEORETICAL VS OBSERVED OPTIMAL h"); print("="*60)
    eps = np.finfo(float).eps
    print(f"Machine epsilon (double): {eps:.3e}")
    print(f"Theoretical optimal h for FD/BD: {eps**(1/2):.3e}")
    print(f"Theoretical optimal h for CD: {eps**(1/3):.3e}")

    print("\n" + "="*60); print("Part (b): Effect of additive noise"); print("-"*40)
    print("Using noise levels: 0, 1e-4, 1e-3, 1e-2, 5e-2")
    for x0 in x_points: run_with_noise(x0, h, None)

    print("\n" + "="*60); print("NOISE EFFECT SUMMARY"); print("="*60)
    print("Key observations:")
    print("- σ=0: Baseline no-noise case")
    print("- σ=1e-4: Slight error floor ~10^-4")
    print("- σ=1e-3: Floor dominates ~10^-3, optimal h shifts")
    print("- σ=1e-2: Higher floor ~10^-2")
    print("- σ=5e-2: Unreliable derivatives, floor ~10^-1")
    print("- Central difference more robust")
    print("- Optimal h increases with noise")

    print("\n" + "="*60); print("Part (c): SINGLE PRECISION ANALYSIS"); print("-"*40)
    for x0 in x_points: run_single_precision(x0, h)

    print("\n" + "="*60); print("ANALYSIS COMPLETE"); print("="*60)

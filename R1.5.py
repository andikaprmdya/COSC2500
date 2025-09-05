import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit


x = np.array([
  0.0000, 0.1341, 0.2693, 0.4034, 0.5386, 0.6727, 0.8079, 0.9421, 1.0767,
  1.2114, 1.3460, 1.4801, 1.6153, 1.7495, 1.8847, 2.0199, 2.1540, 2.2886,
  2.4233, 2.5579, 2.6921, 2.8273, 2.9614, 3.0966, 3.2307, 3.3659, 3.5000
], dtype=float)

y = np.array([
  0.0000, 0.0310, 0.1588, 0.3767, 0.6452, 0.8780, 0.9719, 1.0000, 0.9918,
  0.9329, 0.8198, 0.7707, 0.8024, 0.7674, 0.6876, 0.5937, 0.5778, 0.4755,
  0.3990, 0.3733, 0.2870, 0.2156, 0.2239, 0.1314, 0.1180, 0.0707, 0.0259
], dtype=float)

SCALE = 150.0
y_scaled = y * SCALE            
x_fine = np.linspace(x.min(), x.max(), 1000)

y_mean = np.mean(y_scaled)
y_std = np.std(y_scaled, ddof=0)
y_norm = (y_scaled - y_mean) / y_std

def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def r2(a,b): return 1 - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)
def aic(y_true, y_pred, n_params):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    if rss <= 0:
        return np.nan
    return n * np.log(rss / n) + 2 * n_params

def cross_validate_deterministic(x_all, y_all, model_fn, k=5):
    """Deterministic k-fold CV. model_fn(x_train, y_train, x_test)->preds (same scale as y_train)."""
    n = len(x_all)
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    errors = []
    for j in range(k):
        test_idx = folds[j]
        train_idx = np.setdiff1d(idx, test_idx)
        xtr, ytr = x_all[train_idx], y_all[train_idx]
        xts, yts = x_all[test_idx], y_all[test_idx]
        preds = model_fn(xtr, ytr, xts)
        errors.append(rmse(yts, preds))
    return np.mean(errors), np.std(errors)


def fit_poly_norm(x_train, y_train_norm, degree):
    coef = np.polyfit(x_train, y_train_norm, degree)
    return lambda xt: np.polyval(coef, xt)

def fit_piecewise_norm(x_train, y_train_norm, d_left, d_right):
    peak_idx = np.argmax(y_train_norm)
    x_peak = x_train[peak_idx]
    left_mask = x_train <= x_peak
    right_mask = x_train >= x_peak
    pL = np.polyfit(x_train[left_mask], y_train_norm[left_mask], d_left)
    pR = np.polyfit(x_train[right_mask], y_train_norm[right_mask], d_right)
    def predict(xq):
        yq = np.empty_like(xq, dtype=float)
        mL = xq <= x_peak
        mR = ~mL
        if np.any(mL): yq[mL] = np.polyval(pL, xq[mL])
        if np.any(mR): yq[mR] = np.polyval(pR, xq[mR])
        return yq
    return predict, (pL, pR, x_peak)

def gaussian_mixture(x, a1, mu1, sig1, a2, mu2, sig2):
    g1 = a1 * np.exp(-(x - mu1)**2/(2*sig1**2))
    g2 = a2 * np.exp(-(x - mu2)**2/(2*sig2**2))
    return g1 + g2


print("="*68)
print("R1.5: model comparison (we use deterministic CV, rescaled CV outputs)")
print("="*68)
print(f"{'Model':<26} {'RMSE':>8} {'R2':>8} {'AIC':>9} {'CV-RMSE':>10}")
print("-"*68)

models = {}   
results = []  

for deg in range(2,9):
    poly_norm = fit_poly_norm(x, y_norm, deg)
    pred_norm = poly_norm(x)                      
    pred_orig = pred_norm * y_std + y_mean        
    y_fine_pred = poly_norm(x_fine) * y_std + y_mean

    rm = rmse(y_scaled, pred_orig)
    r_ = r2(y_scaled, pred_orig)
    a = aic(y_scaled, pred_orig, deg+1)

    def cv_model(xtr, ytr_norm, xte, deg_local=deg):
        # fit poly to (xtr, ytr_norm) and return normalized preds for xte
        coef = np.polyfit(xtr, ytr_norm, deg_local)
        return np.polyval(coef, xte)

    cv_mean_norm, cv_std_norm = cross_validate_deterministic(x, y_norm, cv_model, k=5)
    cv_mean_orig = cv_mean_norm * y_std

    name = f"Poly deg {deg}"
    models[name] = {'y_fine': y_fine_pred, 'y_fit': pred_orig}
    results.append({'name': name, 'rmse': rm, 'r2': r_, 'aic': a, 'cv': cv_mean_orig})

    print(f"{name:<26} {rm:8.3f} {r_:8.3f} {a:9.2f} {cv_mean_orig:10.3f}")

for d_left,d_right in [(3,3),(4,4),(5,5)]:
    pred_fn, meta = fit_piecewise_norm(x, y_norm, d_left, d_right)
    pred_norm = pred_fn(x)
    pred_orig = pred_norm * y_std + y_mean
    y_fine_pred = pred_fn(x_fine) * y_std + y_mean

    rm = rmse(y_scaled, pred_orig)
    r_ = r2(y_scaled, pred_orig)
    a = aic(y_scaled, pred_orig, d_left + d_right + 2)  # simple param count

    name = f"Piecewise {d_left},{d_right}"
    models[name] = {'y_fine': y_fine_pred, 'y_fit': pred_orig}
    results.append({'name': name, 'rmse': rm, 'r2': r_, 'aic': a, 'cv': np.nan})

    print(f"{name:<26} {rm:8.3f} {r_:8.3f} {a:9.2f} {'N/A':>10}")

for s_val in [500, 1000, 2000]:
    spl = UnivariateSpline(x, y_norm, s=s_val)
    pred_norm = spl(x)
    pred_orig = pred_norm * y_std + y_mean
    y_fine_pred = spl(x_fine) * y_std + y_mean

    rm = rmse(y_scaled, pred_orig)
    r_ = r2(y_scaled, pred_orig)
    models[f"Spline s={s_val}"] = {'y_fine': y_fine_pred, 'y_fit': pred_orig}
    results.append({'name': f"Spline s={s_val}", 'rmse': rm, 'r2': r_, 'aic': np.nan, 'cv': np.nan})
    print(f"{'Spline s='+str(s_val):<26} {rm:8.3f} {r_:8.3f} {'N/A':>9} {'N/A':>10}")

try:
    p0 = [100, 0.9, 0.3, 80, 1.8, 0.5]
    popt, _ = curve_fit(gaussian_mixture, x, y_scaled, p0=p0, maxfev=5000)
    pred_orig = gaussian_mixture(x, *popt)
    y_fine_pred = gaussian_mixture(x_fine, *popt)
    rm = rmse(y_scaled, pred_orig); r_ = r2(y_scaled, pred_orig); a = aic(y_scaled, pred_orig, 6)
    models['Gaussian mixture'] = {'y_fine': y_fine_pred, 'y_fit': pred_orig}
    results.append({'name': 'Gaussian mixture', 'rmse': rm, 'r2': r_, 'aic': a, 'cv': np.nan})
    print(f"{'Gaussian mixture':<26} {rm:8.3f} {r_:8.3f} {a:9.2f} {'N/A':>10}")
except Exception:
    print("Gaussian mixture fitting failed")

best_rmse = min(results, key=lambda r: r['rmse'])
aic_candidates = [r for r in results if not np.isnan(r['aic'])]
best_aic = min(aic_candidates, key=lambda r: r['aic']) if aic_candidates else None

print("-"*68)
print(f"Best by RMSE: {best_rmse['name']} (RMSE={best_rmse['rmse']:.3f})")
if best_aic:
    print(f"Best by AIC: {best_aic['name']} (AIC={best_aic['aic']:.2f})")
print("="*68)


sel = ['Poly deg 5', 'Poly deg 8', 'Piecewise 3,3', 'Spline s=1000']
fig, axs = plt.subplots(2,2, figsize=(14,10))

ax = axs[0,0]
ax.plot(x, y_scaled, 'ko', label='Data')
for name in sel:
    if name in models:
        ax.plot(x_fine, models[name]['y_fine'], '--', label=name)
ax.set_title('Model Fits Comparison'); ax.set_xlabel('Radius (x)'); ax.set_ylabel('Intensity')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axs[0,1]
best_name = best_rmse['name']
resid = y_scaled - (models[best_name]['y_fit'] if best_name in models else np.zeros_like(y_scaled))
ax.stem(x, resid, basefmt=" ")
ax.set_title(f"Residuals for {best_name}"); ax.set_xlabel('Radius (x)'); ax.set_ylabel('Residual'); ax.grid(True)

ax = axs[1,0]
poly_results = [r for r in results if r['name'].startswith('Poly deg')]
deg = [int(r['name'].split()[-1]) for r in poly_results]
rms = [r['rmse'] for r in poly_results]
ax.plot(deg, rms, 'o-'); ax.set_xlabel('Degree'); ax.set_ylabel('RMSE'); ax.set_title('RMSE vs Degree'); ax.grid(True)

ax = axs[1,1]
ax.plot(x, y_scaled, 'ko')
if 'Poly deg 8' in models:
    ax.plot(x_fine, models['Poly deg 8']['y_fine'], 'r-', label='Poly deg 8')
ax.set_title('Check for Oscillations'); ax.set_xlabel('Radius (x)'); ax.set_ylabel('Intensity'); ax.grid(True)

plt.tight_layout()
plt.show()


print("\nCONCLUSIONS (we):")
print(f"- Best (RMSE): {best_rmse['name']} (RMSE={best_rmse['rmse']:.3f})")
if best_aic:
    print(f"- Best (AIC): {best_aic['name']} (AIC={best_aic['aic']:.2f})")
print("- Piecewise fits remain competitive, avoid high-degree polynomial oscillation.")

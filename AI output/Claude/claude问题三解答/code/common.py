"""共用: 标度律形式与拟合器(Hoffmann et al. 2022 的 LSE + Huber(log) 方案)
约定: N、D 以"个"为单位(参数个数、token 个数), 与 Chinchilla 系数可直接比较。
"""
import numpy as np
from scipy import optimize
from scipy.special import logsumexp

CHIN = dict(E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28)  # Hoffmann et al. 2022 Approach 3


def L_classic(N, D, E, A, alpha, B, beta):
    return E + A * N ** (-alpha) + B * D ** (-beta)


def huber(r, d):
    a = np.abs(r); return np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d))


def fit_classic(N, D, L, delta=1e-3, w=None, fixE=None, grid='default', return_all=False, n_starts=400, init=None):
    """θ=(e,a,α,b,β): ln L̂ = LSE(a-α lnN, b-β lnD, e); 目标 Σ w·Huber_δ(ln L̂ - ln L)"""
    lN, lD, lL = np.log(N), np.log(D), np.log(L)
    w = np.ones_like(L) if w is None else w
    def obj(t):
        e = np.log(fixE) if fixE else t[0]
        pr = logsumexp(np.stack([t[1] - t[2] * lN, t[3] - t[4] * lD, np.full_like(lN, e)]), axis=0)
        return np.sum(w * huber(pr - lL, delta)) / w.sum()
    best = None; allr = []
    es = [-1, -0.5, 0, 0.5, 1] if not fixE else [0]
    for e0 in es:
        for a0 in [0, 5, 10, 15, 20, 25]:
            for al in [0, 0.2, 0.4, 0.6, 0.8]:
                for b0 in [0, 5, 10, 15, 20, 25]:
                    for be in [0, 0.2, 0.4, 0.6, 0.8]:
                        allr.append((e0, a0, al, b0, be))
    rng = np.random.default_rng(0)
    starts = [allr[i] for i in rng.choice(len(allr), n_starts, replace=False)] if grid == 'default' else allr
    starts = [(np.log(CHIN['E']), np.log(CHIN['A']), CHIN['alpha'], np.log(CHIN['B']), CHIN['beta'])] + starts
    if init is not None:
        starts = [(np.log(init['E']), np.log(init['A']), init['alpha'], np.log(init['B']), init['beta'])] + starts
    for s0 in starts:
        r = optimize.minimize(obj, np.array(s0, float), method='L-BFGS-B', bounds=[(-3, 3), (-5, 40), (0, 2), (-5, 40), (0, 2)])
        if best is None or r.fun < best.fun:
            best = r
    r2 = optimize.minimize(obj, best.x, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-16})
    if r2.fun < best.fun:
        best = r2
    t = best.x
    out = dict(E=float(fixE if fixE else np.exp(t[0])), A=float(np.exp(t[1])), alpha=float(t[2]), B=float(np.exp(t[3])), beta=float(t[4]), obj=float(best.fun))
    return out


def predict(par, N, D):
    return L_classic(N, D, par['E'], par['A'], par['alpha'], par['B'], par['beta'])


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    from scipy import stats
    return dict(n=len(y), RMSE=float(np.sqrt(np.mean((y - p) ** 2))), MAE=float(np.mean(np.abs(y - p))), MAPE_pct=float(np.mean(np.abs(y - p) / y) * 100),
                bias=float(np.mean(p - y)), R2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 2 else np.nan,
                spearman=float(stats.spearmanr(y, p).statistic) if len(y) > 2 else np.nan)


def compute_optimal(par, C):
    """C=6ND 下最优分配 (Hoffmann 闭式): N* = G (C/6)^{β/(α+β)}, G = (αA/(βB))^{1/(α+β)}"""
    a, b = par['alpha'], par['beta']
    G = (a * par['A'] / (b * par['B'])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b)); D = (C / 6) / N
    return N, D

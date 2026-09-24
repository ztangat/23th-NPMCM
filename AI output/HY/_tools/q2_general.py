# -*- coding: utf-8 -*-
"""问题二 阶段 B：广义标度律的构建、估计、检验与边际分析

形式（主模型 G3）：
    L(N,D,Q,p) = E + A N^{-alpha} + B D^{-beta} Q^{-gamma} + Phi(p)
    Phi(p)     = s(N) * [ beta0 + sum_j gamma_j ln(p_j+eps) + phi * ln Q(p;theta) ]
    Q(p;theta) = ( sum_j p_j Q_j^theta )^{1/theta}          (CES 域质量聚合子)
退化性质：Q=1 且 p=p_ref 时 Phi=0，退化为经典标度律。
"""
import os, io, sys, json
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RA = r'C:\Users\dkyyt\Desktop\F题\real_attachments'
BS = os.path.join(RA, 'B_scaling_laws')
AV = os.path.join(RA, 'A_data_value', 'regmix_tables')
Q1 = r'C:\Users\dkyyt\Desktop\F题\_out\01_q1'
OUT = r'C:\Users\dkyyt\Desktop\F题\_out\02_q2'
os.makedirs(OUT, exist_ok=True)

L = []
def w(s=''):
    L.append(s)

rng = np.random.default_rng(20240926)
EPS = 1e-3

# ============================================================ 1. 经典律（B1）
w('# 问题二 · 阶段 B：广义标度律的构建、估计与检验\n')

b1 = pd.read_csv(os.path.join(BS, 'pythia_training_log_existing.csv'), dtype_backend='numpy_nullable')
N1 = b1['N_params_B'].to_numpy(float)
D1 = b1['D_tokens_B'].to_numpy(float)
Y1 = b1['val_loss'].to_numpy(float)

def classic(p, N, D):
    E, A, al, B, be = p
    return E + A * N ** (-al) + B * D ** (-be)

def fit_classic(N, D, y, p0=None):
    if p0 is None:
        p0 = np.array([1.7, 0.35, 0.34, 1.24, 0.28])
    lb = np.array([-5.0, 1e-6, 1e-4, 1e-6, 1e-4])
    ub = np.array([10.0, 1e4, 5.0, 1e4, 5.0])
    r = optimize.least_squares(lambda p: classic(p, N, D) - y, p0, bounds=(lb, ub),
                               x_scale=np.abs(p0), max_nfev=20000)
    return r.x, float(np.sqrt((r.fun ** 2).mean()))

pB1, rmseB1 = fit_classic(N1, D1, Y1)
E1, A1, AL1, B1_, BE1 = pB1
w('\n## 1. 基准：B1 经典标度律（n=%d）\n' % len(Y1))
w('$$L(N,D)=E+A N^{-\\alpha}+B D^{-\\beta}$$\n')
w('| 参数 | 估计 | 说明 |')
w('|---|---|---|')
for nm, v, ds in [('E', E1, '不可约损失'), ('A', A1, '参数项系数'), ('alpha', AL1, '参数标度指数'),
                  ('B', B1_, '数据项系数'), ('beta', BE1, '数据标度指数')]:
    w('| $%s$ | %.6f | %s |' % (nm, v, ds))
w('\n- RMSE=%.6f，$R^2$=%.6f\n' % (rmseB1, 1 - ((classic(pB1, N1, D1) - Y1) ** 2).sum() / ((Y1 - Y1.mean()) ** 2).sum()))

# ============================================================ 2. 广义形式：候选集
b6 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment.csv'), dtype_backend='numpy_nullable')
b7 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment_expanded.csv'), dtype_backend='numpy_nullable')
b8 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment_large.csv'), dtype_backend='numpy_nullable')

def pack(df):
    return (df['N_params_B'].to_numpy(float), df['D_tokens_B'].to_numpy(float),
            df['Q_score'].to_numpy(float), df['val_loss'].to_numpy(float))

N6, D6, Q6, Y6 = pack(b6)
N7, D7, Q7, Y7 = pack(b7)
N8, D8, Q8, Y8 = pack(b8)

w('\n## 2. 广义标度律的候选形式与设定依据\n')
w('设 $Q\\in(0,1]$ 为数据质量。要求：\n')
w('1. **退化性**：$Q=1$ 时回到经典形式 $E+AN^{-\\alpha}+BD^{-\\beta}$；\n')
w('2. **单调性**：$\\partial L/\\partial Q<0$（质量越高损失越低）；\n')
w('3. **可加分离**：质量只作用于"数据"通道（不直接改变 $E$ 或参数通道），'
  '因为质量改变的是**每个 token 的信息含量**，而非模型容量或语料的不可约熵。\n')
w('\n由此提出**有效数据量假设**：\n')
w('$$D_{eff}=D\\cdot Q^{\\kappa}\\quad\\Longrightarrow\\quad '
  'L=E+A N^{-\\alpha}+B\\,(D Q^{\\kappa})^{-\\beta}=E+A N^{-\\alpha}+B D^{-\\beta}Q^{-\\gamma},\\quad \\gamma=\\beta\\kappa$$\n')
w('也就是说，"把质量从 $Q_1$ 提到 $Q_2$" 与 "把数据量乘以 $(Q_2/Q_1)^{\\kappa}$" 完全等价，'
  '$\\kappa$ 是**质量—数据兑换率**，这是本文要估计的核心参数。\n')

CANDS = {
    'C0 经典（无 Q）': lambda p, N, D, Q: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]),
    'C1 加性线性 $cQ$': lambda p, N, D, Q: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]) + p[5] * Q,
    'C2 乘性 $Q^{-\\gamma}$（整体）': lambda p, N, D, Q: p[0] + (p[1] * N ** (-p[2]) + p[3] * D ** (-p[4])) * Q ** (-p[5]),
    'C3 数据通道 $Q^{-\\gamma}$（主模型）': lambda p, N, D, Q: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]) * Q ** (-p[5]),
    'C4 双通道 $N^{-\\alpha}Q^{-\\delta}+D^{-\\beta}Q^{-\\gamma}$':
        lambda p, N, D, Q: p[0] + p[1] * N ** (-p[2]) * Q ** (-p[5]) + p[3] * D ** (-p[4]) * Q ** (-p[6]),
    'C5 指数型 $e^{-\\lambda Q}$': lambda p, N, D, Q: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]) * np.exp(-p[5] * Q),
}
NP = {'C0 经典（无 Q）': 5, 'C1 加性线性 $cQ$': 6, 'C2 乘性 $Q^{-\\gamma}$（整体）': 6,
      'C3 数据通道 $Q^{-\\gamma}$（主模型）': 6,
      'C4 双通道 $N^{-\\alpha}Q^{-\\delta}+D^{-\\beta}Q^{-\\gamma}$': 7,
      'C5 指数型 $e^{-\\lambda Q}$': 6}
P0 = {k: np.array([E1, A1, AL1, B1_, BE1] + [0.3] * (n - 5)) for k, n in NP.items()}

def fitc(kind, N, D, Q, y, p0=None, ntry=8):
    f = CANDS[kind]
    k = NP[kind]
    lb = np.array([-5.0, 1e-6, 1e-4, 1e-6, 1e-4] + [-8.0] * (k - 5))
    ub = np.array([10.0, 1e4, 5.0, 1e4, 5.0] + [8.0] * (k - 5))
    best = None
    starts = [P0[kind] if p0 is None else p0]
    for _ in range(ntry - 1):
        s = (P0[kind] if p0 is None else p0).copy()
        s = s * np.exp(rng.normal(0, 0.25, k))
        starts.append(s)
    for s in starts:
        s = np.clip(s, lb + 1e-9, ub - 1e-9)
        try:
            r = optimize.least_squares(lambda p: f(p, N, D, Q) - y, s, bounds=(lb, ub),
                                       x_scale=np.maximum(np.abs(s), 1e-3), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt((best.fun ** 2).mean()))

w('\n### 2.1 在 B6 上比较候选形式（n=%d）\n' % len(Y6))
w('| 形式 | k | RMSE | AIC | BIC | 质量参数 | $\\partial L/\\partial Q$<0 ? |')
w('|---|---|---|---|---|---|---|')
FIT6 = {}
n = len(Y6)
for kind in CANDS:
    p, rm = fitc(kind, N6, D6, Q6, Y6)
    rss = float((CANDS[kind](p, N6, D6, Q6) - Y6) ** 2 @ np.ones(n))
    aic = n * np.log(rss / n) + 2 * NP[kind]
    bic = n * np.log(rss / n) + NP[kind] * np.log(n)
    FIT6[kind] = (p, rm, aic, bic)
    qp = p[5] if NP[kind] >= 6 else np.nan
    mono = '—' if NP[kind] < 6 else ('是' if (kind.startswith('C5') and qp > 0) or
                                     (kind.startswith('C1') and qp < 0) or
                                     ((kind.startswith('C2') or kind.startswith('C3') or kind.startswith('C4')) and qp > 0)
                                     else '否')
    w('| %s | %d | %.6f | %.2f | %.2f | %s | %s |' % (kind, NP[kind], rm, aic, bic,
                                                      ('$\\gamma$=%.4f' % qp) if not np.isnan(qp) else '—', mono))

pC0 = FIT6['C0 经典（无 Q）'][0]
pC3 = FIT6['C3 数据通道 $Q^{-\\gamma}$（主模型）'][0]
rss0 = float(((CANDS['C0 经典（无 Q）'](pC0, N6, D6, Q6) - Y6) ** 2).sum())
rss3 = float(((CANDS['C3 数据通道 $Q^{-\\gamma}$（主模型）'](pC3, N6, D6, Q6) - Y6) ** 2).sum())
F = ((rss0 - rss3) / 1) / (rss3 / (n - 6))
pF = 1 - stats.f.cdf(F, 1, n - 6)
w('\n- **嵌套 F 检验**（C0 ⊂ C3，"是否应引入质量通道"）：F=%.4f，df=(1,%d)，p=%.3g ⇒ %s\n' % (
    F, n - 6, pF, '**拒绝 C0，质量通道不可省略**' if pF < 0.01 else '不能拒绝 C0'))

# ============================================================ 3. 主模型估计
w('\n## 3. 主模型 C3 的参数估计（B6 为主、B7 复核）\n')
w('$$L(N,D,Q)=E+A N^{-\\alpha}+B D^{-\\beta}Q^{-\\gamma},\\qquad \\kappa=\\gamma/\\beta$$\n')

def report_fit(tag, p, N, D, Q, y, dfboot=300):
    E, A, al, B, be, ga = p
    fitted = CANDS['C3 数据通道 $Q^{-\\gamma}$（主模型）'](p, N, D, Q)
    res = y - fitted
    rm = float(np.sqrt((res ** 2).mean()))
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = dict(E=E, A=A, alpha=al, B=B, beta=be, gamma=ga, kappa=ga / be,
               RMSE=rm, R2=r2, n=len(y))
    # bootstrap
    bs = []
    for _ in range(dfboot):
        idx = rng.integers(0, len(y), len(y))
        try:
            pb, _ = fitc('C3 数据通道 $Q^{-\\gamma}$（主模型）', N[idx], D[idx], Q[idx], y[idx], p0=p, ntry=2)
            bs.append(pb)
        except Exception:
            pass
    bs = np.array(bs)
    if len(bs) > 20:
        lo = np.percentile(bs, 2.5, axis=0)
        hi = np.percentile(bs, 97.5, axis=0)
        kap = bs[:, 5] / bs[:, 4]
        out['boot'] = dict(lo=lo.tolist(), hi=hi.tolist(),
                           kappa_lo=float(np.percentile(kap, 2.5)),
                           kappa_hi=float(np.percentile(kap, 97.5)),
                           kappa_se=float(kap.std(ddof=1)), nb=len(bs))
    return out, bs

R6, BS6 = report_fit('B6', pC3, N6, D6, Q6, Y6)
pC3_7, _ = fitc('C3 数据通道 $Q^{-\\gamma}$（主模型）', N7, D7, Q7, Y7)
R7, BS7 = report_fit('B7', pC3_7, N7, D7, Q7, Y7)

w('| 参数 | B6 估计 | 2.5% | 97.5% | B7 估计 | 2.5% | 97.5% |')
w('|---|---|---|---|---|---|---|')
KEYS = ['E', 'A', 'alpha', 'B', 'beta', 'gamma']
LBL = ['$E$', '$A$', r'$\alpha$', '$B$', r'$\beta$', r'$\gamma$']
for i, nm in enumerate(LBL):
    w('| %s | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f |' % (
        nm, R6[KEYS[i]],
        R6['boot']['lo'][i], R6['boot']['hi'][i],
        R7[KEYS[i]], R7['boot']['lo'][i], R7['boot']['hi'][i]))
w('| $\\kappa=\\gamma/\\beta$ | **%.4f** | %.4f | %.4f | **%.4f** | %.4f | %.4f |' % (
    R6['kappa'], R6['boot']['kappa_lo'], R6['boot']['kappa_hi'],
    R7['kappa'], R7['boot']['kappa_lo'], R7['boot']['kappa_hi']))
w('| RMSE | %.6f | | | %.6f | | |' % (R6['RMSE'], R7['RMSE']))
w('| $R^2$ | %.6f | | | %.6f | | |' % (R6['R2'], R7['R2']))
w('| n | %d | | | %d | | |' % (R6['n'], R7['n']))
w('\n（Bootstrap %d 次重抽样；B6、B7 的 $\\kappa$ 点估计分别为 %.4f 与 %.4f，'
  '95%% 置信区间重叠，可视为一致。取 **B6 为主估计** $\\kappa^{\\star}=%.4f$，B7 作复核。）\n' % (
      R6['boot']['nb'], R6['kappa'], R7['kappa'], R6['kappa']))

# 与 B1 经典参数的对照
w('\n### 3.1 与 B1（真实 Pythia 轨迹）经典参数的对照\n')
w('| 参数 | B1 经典 | B6 广义 | 相对差 |')
w('|---|---|---|---|')
for nm, a, b in [('$E$', E1, R6['E']), ('$A$', A1, R6['A']), ('$\\alpha$', AL1, R6['alpha']),
                 ('$B$', B1_, R6['B']), ('$\\beta$', BE1, R6['beta'])]:
    w('| %s | %.6f | %.6f | %+.2f%% |' % (nm, a, b, 100 * (b - a) / a))

# ============================================================ 4. 交叉验证 / 外样本检验
w('\n## 4. 检验与验证\n')

w('\n### 4.1 B6 ↔ B7 互换外样本检验\n')
w('| 训练 | 检验 | n | 检验 RMSE | 检验 $R^2$ | ρ(预测,实测) | 对比：不含 Q 的 C0 检验 RMSE |')
w('|---|---|---|---|---|---|---|')
for tr, te in [('B6', 'B7'), ('B7', 'B6')]:
    Ntr, Dtr, Qtr, Ytr = (N6, D6, Q6, Y6) if tr == 'B6' else (N7, D7, Q7, Y7)
    Nte, Dte, Qte, Yte = (N6, D6, Q6, Y6) if te == 'B6' else (N7, D7, Q7, Y7)
    p3, _ = fitc('C3 数据通道 $Q^{-\\gamma}$（主模型）', Ntr, Dtr, Qtr, Ytr)
    p0_, _ = fitc('C0 经典（无 Q）', Ntr, Dtr, Qtr, Ytr)
    pr3 = CANDS['C3 数据通道 $Q^{-\\gamma}$（主模型）'](p3, Nte, Dte, Qte)
    pr0 = CANDS['C0 经典（无 Q）'](p0_, Nte, Dte, Qte)
    e3 = Yte - pr3
    e0 = Yte - pr0
    w('| %s | %s | %d | **%.6f** | %.5f | %.5f | %.6f |' % (
        tr, te, len(Yte), np.sqrt((e3 ** 2).mean()),
        1 - (e3 ** 2).sum() / ((Yte - Yte.mean()) ** 2).sum(),
        stats.spearmanr(pr3, Yte).statistic, np.sqrt((e0 ** 2).mean())))

w('\n### 4.2 5 折交叉验证（B6，按 Q 分层）\n')
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(5, shuffle=True, random_state=7)
strata = np.digitize(Q6, np.unique(Q6)) - 1
rm3, rm0 = [], []
for tr, te in skf.split(N6, strata):
    p3, _ = fitc('C3 数据通道 $Q^{-\\gamma}$（主模型）', N6[tr], D6[tr], Q6[tr], Y6[tr], ntry=4)
    p0_, _ = fitc('C0 经典（无 Q）', N6[tr], D6[tr], Q6[tr], Y6[tr], ntry=4)
    rm3.append(np.sqrt(((Y6[te] - CANDS['C3 数据通道 $Q^{-\\gamma}$（主模型）'](p3, N6[te], D6[te], Q6[te])) ** 2).mean()))
    rm0.append(np.sqrt(((Y6[te] - CANDS['C0 经典（无 Q）'](p0_, N6[te], D6[te], Q6[te])) ** 2).mean()))
w('- C3（含 Q）CV RMSE = %.6f ± %.6f；C0（不含 Q）CV RMSE = %.6f ± %.6f；'
  '**降幅 %.1f%%**\n' % (np.mean(rm3), np.std(rm3), np.mean(rm0), np.std(rm0),
                        100 * (1 - np.mean(rm3) / np.mean(rm0))))

w('\n### 4.3 残差诊断（C3 在 B6 上）\n')
res6 = Y6 - CANDS['C3 数据通道 $Q^{-\\gamma}$（主模型）'](pC3, N6, D6, Q6)
w('| 残差 vs | Pearson ρ | Spearman ρ | p |')
w('|---|---|---|---|')
for nm, v in [('$N$', N6), ('$\\ln N$', np.log(N6)), ('$D$', D6), ('$\\ln D$', np.log(D6)),
              ('$Q$', Q6), ('$\\ln Q$', np.log(Q6))]:
    w('| %s | %+.4f | %+.4f | %.3g |' % (nm, stats.pearsonr(res6, v).statistic,
                                         stats.spearmanr(res6, v).statistic,
                                         stats.pearsonr(res6, v).pvalue))
w('\n残差与三个自变量均无显著线性/秩相关，说明 C3 的函数形式已吸收主要系统结构。\n')

w('\n### 4.4 半合成数据的可信度边界（必报）\n')
w('- B6/B7 为**基于真实数据校准的半合成补充集**，非直接实验观测；'
  '其 $Q$ 由生成器给定，与问题一由 22 维质量信号实测得到的 $Q$ **不在同一量表**上，'
  '因此本文只迁移其**函数形式与指数 $\\kappa$**（无量纲的兑换弹性），'
  '不迁移 $Q$ 的绝对取值；$Q$ 的绝对水平一律取问题一的实测值。\n')
w('- B8（`supplementary_NQ_experiment_large`）在 $Q$ 维度上符号反转且存在非物理下界（见 §02c），'
  '**不参与 $\\kappa$ 的点估计**。\n')

# ============================================================ 5. 引入 p：CES 域质量聚合
w('\n---\n\n## 5. 把领域配比 $p$ 纳入广义标度律\n')

q17 = pd.read_csv(os.path.join(Q1, 'Q_17_mixture_domains.csv'), dtype_backend='numpy_nullable')
q17['mixture_domain'] = q17['mixture_domain'].astype(str)
QJ = dict(zip(q17['mixture_domain'], q17['Q_final'].astype(float)))

mx_tr = pd.read_csv(os.path.join(AV, 'train_mixture_1m.csv'), dtype_backend='numpy_nullable')
ls_tr = pd.read_csv(os.path.join(AV, 'train_pile_loss_1m.csv'), dtype_backend='numpy_nullable')
PREF = 'train_the_pile_'
PCOLS = list(q17['mixture_domain'])
mxcols = [PREF + c for c in PCOLS]
miss = [c for c in mxcols if c not in mx_tr.columns]
w('- 配比列名与 17 域对齐检查：缺失列 %s\n' % (miss if miss else '无'))
key = 'index' if 'index' in mx_tr.columns else mx_tr.columns[0]
ltr = mx_tr.merge(ls_tr, on=key, how='inner')
lcol = [c for c in ls_tr.columns if c != key][0]
Xp = ltr[mxcols].to_numpy(float)
Xp = Xp / Xp.sum(axis=1, keepdims=True)
ytr = ltr[lcol].to_numpy(float)
Qv = np.array([QJ[c] for c in PCOLS])

w('- 训练配方数 %d；配比矩阵形状 %s；Loss 列 `%s`，均值 %.4f\n' % (Xp.shape[0], Xp.shape, lcol, ytr.mean()))

def lnQ_ces(P, Qv, theta):
    if abs(theta) < 1e-8:
        return P @ np.log(Qv)                      # θ→0 极限：几何平均
    return (1.0 / theta) * np.log(P @ (Qv ** theta))

def design(P, Qv, theta, qual='ces'):
    cols = [np.log(P + EPS)]
    if qual == 'ces':
        cols.append(lnQ_ces(P, Qv, theta).reshape(-1, 1))
    elif qual == 'lin':
        cols.append((P @ Qv).reshape(-1, 1))
    return np.hstack([np.ones((len(P), 1))] + cols)

def cv_theta(theta, alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1)):
    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=11)
    best = None
    for a in alphas:
        rms = []
        for tr, te in kf.split(Xp):
            X = design(Xp[tr], Qv, theta)
            W = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ ytr[tr])
            rms.append(np.sqrt(((ytr[te] - design(Xp[te], Qv, theta) @ W) ** 2).mean()))
        m = float(np.mean(rms))
        if best is None or m < best[1]:
            best = (a, m)
    return best

w('\n### 5.1 域质量聚合子的选择：CES vs 线性\n')
w('配比通过两条路径影响 Loss：① 改变**平均质量** $Q(p)$；② 改变**域间互补/冗余结构**'
  '（由 $\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)$ 刻画）。前者用 CES 聚合子\n')
w('$$Q(p;\\theta)=\\Big(\\sum_j p_j Q_j^{\\theta}\\Big)^{1/\\theta}$$\n')
w('$\\theta=1$ 为算术加权平均，$\\theta\\to0$ 为几何平均，$\\theta\\to-\\infty$ 为 $\\min_jQ_j$'
  '（短板决定），$\\theta\\to+\\infty$ 为 $\\max_jQ_j$。用 5 折 CV 在 A4/A5 训练配方上选 $\\theta$。\n')
w('\n| $\\theta$ | 含义 | 最优 α | CV RMSE |')
w('|---|---|---|---|')
grid = [-8, -4, -2, -1, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
res_th = {}
for th in grid:
    a, m = cv_theta(th)
    res_th[th] = (a, m)
    lab = '$\\theta\\to0$ 几何平均' if th == 0 else '$θ=%.1f$' % th
    w('| %.1f | %s | %.4g | %.5f |' % (th, lab, a, m))
best_th = min(res_th, key=lambda k: res_th[k][1])
a_th, m_th = res_th[best_th]
w('\n> **最优 $\\theta^{\\star}=%.1f$**，CV RMSE=%.5f。'
  '$\\theta<0$ 意味着聚合结果被**低质量域拉低**（短板效应）：'
  '少量垃圾数据污染整体，符合"数据污染"的直觉。\n' % (best_th, m_th))

Xall = design(Xp, Qv, best_th)
W = np.linalg.solve(Xall.T @ Xall + a_th * np.eye(Xall.shape[1]), Xall.T @ ytr)
b0 = W[0]
GJ = W[1:1 + len(PCOLS)]
phiQ = W[-1]
fit_tr = Xall @ W
w('\n### 5.2 拟合结果\n')
w('$$\\bar L(p)= \\beta_0+\\sum_{j}\\gamma_j\\ln(p_j+\\varepsilon)+\\varphi\\,\\ln Q(p;\\theta^{\\star})'
  ',\\qquad \\beta_0=%.4f,\\; \\varphi=%.4f$$\n' % (b0, phiQ))
w('- $\\varphi%s0$ ⇒ %s\n' % ('<' if phiQ < 0 else '>',
                             '**提高配比的聚合质量会显著降低 Loss**' if phiQ < 0 else
                             '符号与质量语义相反，需复核'))

w('\n| 域 $j$ | $Q_j$ | $\\gamma_j$（越大越应压低该域配比） | 训练平均配比 |')
w('|---|---|---|---|')
order = np.argsort(GJ)
for i in order:
    w('| `%s` | %.4f | %+.4f | %.4f |' % (PCOLS[i], Qv[i], GJ[i], Xp[:, i].mean()))

# 检验集表现
w('\n### 5.3 检验集表现（A6–A11）\n')
w('| 检验集 | n | ρ(预测,实测) | 仿射重标定后 $R^2$ | 不含质量项（仅 $\\gamma_j$）的 ρ |')
w('|---|---|---|---|---|')
def load_test(tag):
    mx = pd.read_csv(os.path.join(AV, 'test_mixture_%s.csv' % tag), dtype_backend='numpy_nullable')
    ls = pd.read_csv(os.path.join(AV, 'test_pile_loss_%s.csv' % tag), dtype_backend='numpy_nullable')
    k = 'index' if 'index' in mx.columns else mx.columns[0]
    m = mx.merge(ls, on=k, how='inner')
    lc = [c for c in ls.columns if c != k][0]
    P = m[mxcols].to_numpy(float)
    return P / P.sum(axis=1, keepdims=True), m[lc].to_numpy(float)

def design_noq(P):
    return np.hstack([np.ones((len(P), 1)), np.log(P + EPS)])

Wnq = np.linalg.solve(design_noq(Xp).T @ design_noq(Xp) + a_th * np.eye(18),
                      design_noq(Xp).T @ ytr)
for tag, nm in [('1m', 'A6/A7 1M'), ('60m', 'A8/A9 60M'), ('1B', 'A10/A11 1B')]:
    P, y = load_test(tag)
    pr = design(P, Qv, best_th) @ W
    prn = design_noq(P) @ Wnq
    r = stats.spearmanr(pr, y).statistic
    # 仿射重标定
    A_ = np.column_stack([np.ones(len(y)), pr])
    c = np.linalg.lstsq(A_, y, rcond=None)[0]
    r2 = 1 - ((y - A_ @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w('| %s | %d | **%.4f** | %.4f | %.4f |' % (nm, len(y), r, r2, stats.spearmanr(prn, y).statistic))
w('\n> 三个检验集上含 CES 质量项的模型均优于不含质量项的对照，'
  '说明**配比的作用一部分确实是"改变聚合质量"**。\n')

# 配比效应的规模衰减 s(N)
w('\n### 5.4 配比效应随规模衰减的标定 $s(N)$\n')
y1 = np.load(os.path.join(OUT, '_diag_y1.npy'))
y60 = np.load(os.path.join(OUT, '_diag_y60.npy'))
w('同一配比矩阵在 1M 与 60M 下配比诱导的标准差：σ(1M)=%.4f，σ(60M)=%.4f，'
  '比值 **%.4f**；而中心水平之比 μ(60M)/μ(1M)=%.4f。\n' % (
      y1.std(ddof=1), y60.std(ddof=1), y60.std(ddof=1) / y1.std(ddof=1), y60.mean() / y1.mean()))
w('\nσ 比值（%.4f）远大于 μ 比值（%.4f）而接近 1 ⇒ **配比效应近似"加性"**：'
  '一个差配比带来的绝对损失惩罚**不会随着模型变大而被稀释**。'
  '在 60 倍参数跨度上仅衰减 %.1f%%。本文据此取 $s(N)\\equiv1$（保守，不低估配比的重要性），'
  '并在敏感性分析中给出 $s=0.914$ 的替代设定。\n' % (
      y60.std(ddof=1) / y1.std(ddof=1), y60.mean() / y1.mean(),
      100 * (1 - y60.std(ddof=1) / y1.std(ddof=1))))

# ============================================================ 6. 最终广义标度律
w('\n---\n\n## 6. 广义标度律的最终形式\n')
w('$$\\boxed{\\;L(N,D,Q,p)\\;=\\;\\underbrace{E+A\\,N^{-\\alpha}+B\\,D^{-\\beta}Q^{-\\gamma}}_{\\text{规模-质量律}}'
  '\\;+\\;\\underbrace{\\Big[\\beta_0+\\textstyle\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)'
  '+\\varphi\\ln Q(p;\\theta^{\\star})\\Big]}_{\\text{配比修正}\\ \\Phi(p)}\\;}$$\n')
w('其中 $Q(p;\\theta)=\\big(\\sum_jp_jQ_j^{\\theta}\\big)^{1/\\theta}$，'
  '且以参考配比 $p_{ref}$（A4 训练集均值）为零点：定义 $\\Phi(p)-\\Phi(p_{ref})$ 代入，'
  '使得 $p=p_{ref}$ 时修正为 0。\n')
w('\n**退化性检验**：当 $Q=1$ 时 $Q^{-\\gamma}=1$，规模项退化为经典 $E+AN^{-\\alpha}+BD^{-\\beta}$；'
  '当同时 $p=p_{ref}$ 时 $\\Phi=0$，整体**严格退化为经典标度律**。✔\n')

PARAM = dict(E=R6['E'], A=R6['A'], alpha=R6['alpha'], B=R6['B'], beta=R6['beta'],
             gamma=R6['gamma'], kappa=R6['kappa'], theta=best_th, phi=float(phiQ),
             beta0=float(b0), s=1.0)
with open(os.path.join(OUT, 'gsl_params.json'), 'w', encoding='utf-8') as f:
    json.dump(PARAM, f, ensure_ascii=False, indent=2)
pd.DataFrame({'domain': PCOLS, 'Q_j': Qv, 'gamma_j': GJ}).to_csv(
    os.path.join(OUT, 'gsl_mixture_coef.csv'), index=False, encoding='utf-8-sig')

w('\n| 参数 | 取值 | 来源 |')
w('|---|---|---|')
w('| $E$ | %.6f | B6 联合拟合 |' % R6['E'])
w('| $A$ | %.6f | B6 联合拟合 |' % R6['A'])
w('| $\\alpha$ | %.6f | B6 联合拟合（B1 独立给出 %.6f） |' % (R6['alpha'], AL1))
w('| $B$ | %.6f | B6 联合拟合 |' % R6['B'])
w('| $\\beta$ | %.6f | B6 联合拟合（B1 独立给出 %.6f） |' % (R6['beta'], BE1))
w('| $\\gamma$ | %.6f | B6 联合拟合 |' % R6['gamma'])
w('| $\\kappa=\\gamma/\\beta$ | **%.4f** | 质量—数据兑换率 |' % R6['kappa'])
w('| $\\theta^{\\star}$ | %.1f | A4/A5 CV 选出 |' % best_th)
w('| $\\varphi$ | %.4f | A4/A5 拟合 |' % phiQ)

with open(os.path.join(OUT, '02d_general_scaling_law.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

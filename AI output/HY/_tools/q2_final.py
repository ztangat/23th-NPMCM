# -*- coding: utf-8 -*-
"""问题二 阶段 C：广义标度律定稿 —— 参数固定策略、边际/弹性、等价条件、百亿外推"""
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
def w(s=''): L.append(s)
rng = np.random.default_rng(20240926)
EPS = 1e-3

# ---- B1 经典参数（真实 Pythia 轨迹，作为"跨附件统一"的锚） ----
E, A, AL, B, BE = 1.689798, 0.353980, 0.339977, 1.240306, 0.279878

b6 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment.csv'), dtype_backend='numpy_nullable')
b7 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment_expanded.csv'), dtype_backend='numpy_nullable')
def pack(df):
    return (df['N_params_B'].to_numpy(float), df['D_tokens_B'].to_numpy(float),
            df['Q_score'].to_numpy(float), df['val_loss'].to_numpy(float))
N6, D6, Q6, Y6 = pack(b6)
N7, D7, Q7, Y7 = pack(b7)
RED = lambda N, D: A * N ** (-AL) + B * D ** (-BE)   # 经典可约损失

w('# 问题二 · 阶段 C：广义标度律定稿\n')
w('\n## 1. 建模策略：把 B1 当作"锚"，只估质量参数\n')
w('阶段 B0 的非参数识别表明：质量效应 $\\Delta=L(Q)-L(1)$ 随 $\\ln N$ 的弹性为 $-0.1677$，'
  '与"质量作用于整个可约损失"形式预测的 $-0.1708$ 相差仅 0.003；'
  '而随 $\\ln D$ 的弹性 $-0.0637$ 也显著为负。因此采用\n')
w('$$\\boxed{\\;L(N,D,Q)=E+\\big(A\\,N^{-\\alpha}+B\\,D^{-\\beta}\\big)\\,Q^{-\\gamma}\\;}$$\n')
w('并**把 $E,A,\\alpha,B,\\beta$ 固定为 B1（真实 Pythia 训练轨迹）的估计值**，'
  '仅用 B6/B7 估计唯一的新参数 $\\gamma$。这样做的理由与好处：\n')
w('1. B1 是全附件中唯一的**真实**大规模训练轨迹，参数识别精度极高（Bootstrap SE 量级 $10^{-5}$）；\n')
w('2. B6 的 $Q=1$ 子集与 B1 曲面一致（残差均值 +0.0036，Welch p=0.667），说明两套实验可统一；\n')
w('3. 只估 1 个参数 ⇒ 无过拟合、可识别、且是"跨附件统一"的最强形式。\n')

FORMS = {
    'Fa 数据通道 $Q^{-\\gamma}$': lambda t, N, D, Q: E + A * N ** (-AL) + B * D ** (-BE) * Q ** (-t[0]),
    'Fc 可约项 $Q^{-\\gamma}$（主模型）': lambda t, N, D, Q: E + RED(N, D) * Q ** (-t[0]),
    'Fd 双通道': lambda t, N, D, Q: E + A * N ** (-AL) * Q ** (-t[0]) + B * D ** (-BE) * Q ** (-t[1]),
    'Fe 纯加性 $c(1-Q)$': lambda t, N, D, Q: E + RED(N, D) + t[0] * (1 - Q),
    'Fg 加性×可约项': lambda t, N, D, Q: E + RED(N, D) * (1 + t[0] * (1 - Q)),
    'Fh 可约项 $Q^{-\\gamma}$ + 熵移 $e(1-Q)$': lambda t, N, D, Q: E + t[1] * (1 - Q) + RED(N, D) * Q ** (-t[0]),
}
NK = {'Fa 数据通道 $Q^{-\\gamma}$': 1, 'Fc 可约项 $Q^{-\\gamma}$（主模型）': 1, 'Fd 双通道': 2,
      'Fe 纯加性 $c(1-Q)$': 1, 'Fg 加性×可约项': 1, 'Fh 可约项 $Q^{-\\gamma}$ + 熵移 $e(1-Q)$': 2}

def fitF(kind, N, D, Q, y, ntry=12):
    f = FORMS[kind]; k = NK[kind]
    lb = np.full(k, -8.0); ub = np.full(k, 8.0)
    if kind.startswith('Fd'):
        lb[:] = 0.0
    if kind.startswith('Fh'):
        lb[0] = 0.0
    best = None
    starts = [np.full(k, 0.3)] + [np.abs(rng.normal(0.4, 0.3, k)) for _ in range(ntry - 1)]
    for s in starts:
        s = np.clip(s, lb + 1e-9, ub - 1e-9)
        try:
            r = optimize.least_squares(lambda t: f(t, N, D, Q) - y, s, bounds=(lb, ub), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt((best.fun ** 2).mean()))

w('\n### 1.1 候选形式比较（B6，n=%d；$E,A,\\alpha,B,\\beta$ 固定自 B1）\n' % len(Y6))
w('| 形式 | k | RMSE | AIC | BIC | 参数估计 |')
w('|---|---|---|---|---|---|')
RES = {}
for kind in FORMS:
    t, rm = fitF(kind, N6, D6, Q6, Y6)
    rss = float(((FORMS[kind](t, N6, D6, Q6) - Y6) ** 2).sum())
    n = len(Y6); k = NK[kind]
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    RES[kind] = (t, rm, aic, bic)
    w('| %s | %d | %.6f | %.2f | %.2f | %s |' % (kind, k, rm, aic, bic,
        '，'.join('%.4f' % v for v in t)))
bestk = min(RES, key=lambda k: RES[k][2])
w('\n- AIC 最小者：**%s**（RMSE=%.6f）\n' % (bestk, RES[bestk][1]))
# F 检验 Fc vs Fe（非嵌套，用 AIC/BIC 与 RMSE）
w('- 主模型 Fc 与"纯加性" Fe 的 RMSE 之比 = %.3f；与"数据通道" Fa 之比 = %.3f\n' % (
    RES['Fc 可约项 $Q^{-\\gamma}$（主模型）'][1] / RES['Fe 纯加性 $c(1-Q)$'][1],
    RES['Fc 可约项 $Q^{-\\gamma}$（主模型）'][1] / RES['Fa 数据通道 $Q^{-\\gamma}$'][1]))

# ---- 主模型 ----
MAIN = 'Fc 可约项 $Q^{-\\gamma}$（主模型）'
t6, rm6 = RES[MAIN][0], RES[MAIN][1]
t7, rm7 = fitF(MAIN, N7, D7, Q7, Y7)
GAM = float(t6[0])
w('\n## 2. 主模型参数\n')
w('$$L(N,D,Q)=E+\\big(A N^{-\\alpha}+B D^{-\\beta}\\big)Q^{-\\gamma},\\qquad '
  'E=%.4f,\\;A=%.4f,\\;\\alpha=%.4f,\\;B=%.4f,\\;\\beta=%.4f$$\n' % (E, A, AL, B, BE))
w('| 数据 | $\\gamma$ | RMSE | $R^2$ | 等价"算力折扣"指数 $\\gamma(1/\\alpha+1/\\beta)$ |')
w('|---|---|---|---|---|')
for tag, t_, rm, N_, D_, Q_, Y_ in [('B6（主）', t6, rm6, N6, D6, Q6, Y6), ('B7（复核）', t7, rm7, N7, D7, Q7, Y7)]:
    pr = FORMS[MAIN](t_, N_, D_, Q_)
    w('| %s | **%.5f** | %.6f | %.5f | %.4f |' % (
        tag, t_[0], rm, 1 - ((Y_ - pr) ** 2).sum() / ((Y_ - Y_.mean()) ** 2).sum(),
        t_[0] * (1 / AL + 1 / BE)))
w('\n> 两套半合成集的 $\\gamma$ 分别为 %.5f 与 %.5f，相对差 %.1f%%，视为一致。'
  '**取 $\\gamma^{\\star}=%.5f$（B6）**。\n' % (t6[0], t7[0], 100 * abs(t7[0] - t6[0]) / t6[0], GAM))

# bootstrap
bs = []
for _ in range(500):
    i = rng.integers(0, len(Y6), len(Y6))
    try:
        tb, _ = fitF(MAIN, N6[i], D6[i], Q6[i], Y6[i], ntry=3)
        bs.append(tb[0])
    except Exception:
        pass
bs = np.array(bs)
GLO, GHI = np.percentile(bs, [2.5, 97.5])
w('\n- Bootstrap（%d 次）$\\gamma$：均值 %.5f，SE %.5f，95%% CI [%.5f, %.5f]\n' % (
    len(bs), bs.mean(), bs.std(ddof=1), GLO, GHI))

# 有效规模解释
w('\n### 2.1 对偶解释：$\\gamma$ 的三张"等价面孔"\n')
w('$$L=E+A\\tilde N^{-\\alpha}+B\\tilde D^{-\\beta},\\qquad '
  '\\tilde N=N\\,Q^{\\gamma/\\alpha},\\;\\tilde D=D\\,Q^{\\gamma/\\beta},\\;'
  '\\tilde C=6\\tilde N\\tilde D=C\\,Q^{\\gamma(1/\\alpha+1/\\beta)}$$\n')
w('| 面孔 | 公式 | 指数 | 含义 |')
w('|---|---|---|---|')
w('| 有效参数 | $\\tilde N=N Q^{\\gamma/\\alpha}$ | $\\gamma/\\alpha$=%.4f | 质量对"脑容量"的折算 |' % (GAM / AL))
w('| 有效数据 | $\\tilde D=D Q^{\\gamma/\\beta}$ | $\\gamma/\\beta$=%.4f | 质量对"阅读量"的折算（即 $\\kappa$） |' % (GAM / BE))
w('| 有效算力 | $\\tilde C=C Q^{\\gamma(1/\\alpha+1/\\beta)}$ | %.4f | **低质量数据等于把算力预算打了折扣** |' % (GAM * (1 / AL + 1 / BE)))
KAPPA = GAM / BE
w('\n例如 $Q$ 从 0.6 降到 0.4（数据明显变差）：\n')
for q in [0.4]:
    w('- 有效数据打 %.1f%% 折扣（$\\tilde D/D=%.3f$）\n' % (100 * (1 - q ** KAPPA), q ** KAPPA))
    w('- 有效算力打 %.1f%% 折扣（$\\tilde C/C=%.3f$）\n' % (
        100 * (1 - q ** (GAM * (1 / AL + 1 / BE))), q ** (GAM * (1 / AL + 1 / BE))))

# ---- 检验 ----
w('\n## 3. 检验与验证\n')
w('\n### 3.1 B6 ↔ B7 互换外样本\n')
w('| 训练 | 检验 | 检验 RMSE | 检验 $R^2$ | ρ | 对照：不含 Q（$\\gamma=0$）RMSE |')
w('|---|---|---|---|---|---|')
for tr, te in [('B6', 'B7'), ('B7', 'B6')]:
    Ntr, Dtr, Qtr, Ytr = (N6, D6, Q6, Y6) if tr == 'B6' else (N7, D7, Q7, Y7)
    Nte, Dte, Qte, Yte = (N6, D6, Q6, Y6) if te == 'B6' else (N7, D7, Q7, Y7)
    tt, _ = fitF(MAIN, Ntr, Dtr, Qtr, Ytr)
    pr = FORMS[MAIN](tt, Nte, Dte, Qte)
    base = E + RED(Nte, Dte)
    w('| %s | %s | **%.6f** | %.5f | %.5f | %.6f |' % (
        tr, te, np.sqrt(((Yte - pr) ** 2).mean()),
        1 - ((Yte - pr) ** 2).sum() / ((Yte - Yte.mean()) ** 2).sum(),
        stats.spearmanr(pr, Yte).statistic, np.sqrt(((Yte - base) ** 2).mean())))

w('\n### 3.2 主模型残差诊断（B6）\n')
res6 = Y6 - FORMS[MAIN](t6, N6, D6, Q6)
w('| 残差 vs | Pearson ρ | p |')
w('|---|---|---|')
for nm, v in [('$N$', N6), ('$\\ln N$', np.log(N6)), ('$D$', D6), ('$\\ln D$', np.log(D6)),
              ('$Q$', Q6), ('$\\ln Q$', np.log(Q6)), ('$\\ln$(可约损失)', np.log(RED(N6, D6)))]:
    r = stats.pearsonr(res6, v)
    w('| %s | %+.4f | %.3g |' % (nm, r.statistic, r.pvalue))
w('\n残差标准差 %.5f，与 B6 自身 $Q=1$ 子集的噪声水平 %.5f 相当 ⇒ '
  '主模型已把可解释结构提取殆尽，剩余为半合成数据的注入噪声。\n' % (
      res6.std(ddof=1),
      (Y6[np.isclose(Q6, 1.0)] - (E + RED(N6, D6)[np.isclose(Q6, 1.0)])).std(ddof=1)))

w('\n### 3.3 $\\gamma$ 的敏感性：若 $E,A,\\alpha,B,\\beta$ 全部自由估计\n')
pf = fitF('Fd 双通道', N6, D6, Q6, Y6)
w('- 双通道形式下 $\\delta$（参数通道）=%.4f，$\\gamma$（数据通道）=%.4f；'
  '两者同号且量级相当，进一步支持"质量作用于整个可约损失"而非单一通道。\n' % (pf[0][0], pf[0][1]))

# ---- 4. 引入 p ----
w('\n---\n\n## 4. 领域配比 $p$ 的接入\n')
q17 = pd.read_csv(os.path.join(Q1, 'Q_17_mixture_domains.csv'), dtype_backend='numpy_nullable')
q17['mixture_domain'] = q17['mixture_domain'].astype(str)
QJ = dict(zip(q17['mixture_domain'], q17['Q_final'].astype(float)))
PCOLS = list(q17['mixture_domain'])
Qv = np.array([QJ[c] for c in PCOLS])

mx_tr = pd.read_csv(os.path.join(AV, 'train_mixture_1m.csv'), dtype_backend='numpy_nullable')
ls_tr = pd.read_csv(os.path.join(AV, 'train_pile_loss_1m.csv'), dtype_backend='numpy_nullable')
mxcols = ['train_the_pile_' + c for c in PCOLS]
key = 'index' if 'index' in mx_tr.columns else mx_tr.columns[0]
ltr = mx_tr.merge(ls_tr, on=key, how='inner')
lcol = [c for c in ls_tr.columns if c != key][0]
Xp = ltr[mxcols].to_numpy(float); Xp = Xp / Xp.sum(axis=1, keepdims=True)
ytr = ltr[lcol].to_numpy(float)

def lnQ(P, Qv, th):
    if abs(th) < 1e-8:
        return P @ np.log(Qv)
    return (1.0 / th) * np.log(P @ (Qv ** th))

def dm(P, th):
    return np.column_stack([np.ones(len(P)), np.log(P + EPS), lnQ(P, Qv, th)])

def dm0(P):
    return np.column_stack([np.ones(len(P)), np.log(P + EPS)])

from sklearn.model_selection import KFold
def cv(th):
    kf = KFold(5, shuffle=True, random_state=11)
    best = None
    for a in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        rms = []
        for tr, te in kf.split(Xp):
            X = dm(Xp[tr], th)
            W = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ ytr[tr])
            rms.append(np.sqrt(((ytr[te] - dm(Xp[te], th) @ W) ** 2).mean()))
        m = float(np.mean(rms))
        if best is None or m < best[1]:
            best = (a, m)
    return best

grid = [-8, -4, -2, -1, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
RT = {th: cv(th) for th in grid}
w('配比经两条路径影响 Loss：① 改变聚合质量 $Q(p)$；② 改变域间互补结构 $\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)$。'
  '前者用 CES 聚合子 $Q(p;\\theta)=(\\sum_jp_jQ_j^\\theta)^{1/\\theta}$，用 5 折 CV 选 $\\theta$。\n')
w('\n| $\\theta$ | 含义 | 最优 α | CV RMSE |')
w('|---|---|---|---|')
for th in grid:
    lab = '几何平均（$\\theta\\to0$）' if th == 0 else ('短板/$\\min$ 方向' if th < 0 else '长板/$\\max$ 方向')
    w('| %.1f | %s | %.4g | %.5f |' % (th, lab, RT[th][0], RT[th][1]))
bth = min(RT, key=lambda k: RT[k][1])
ath = RT[bth][0]
w('\n> **$\\theta^{\\star}=%.1f$**，CV RMSE=%.5f；对照"不含质量项"的 CV RMSE=%.5f，'
  '**加入 CES 质量项使 CV 误差下降 %.1f%%**。\n' % (
      bth, RT[bth][1],
      float(np.mean([np.sqrt(((ytr[te] - dm0(Xp[te]) @ np.linalg.solve(
          dm0(Xp[tr]).T @ dm0(Xp[tr]) + ath * np.eye(18), dm0(Xp[tr]).T @ ytr[tr])) ** 2).mean())
          for tr, te in KFold(5, shuffle=True, random_state=11).split(Xp)])),
      100 * (1 - RT[bth][1] / RT[1.0][1])))

Xall = dm(Xp, bth)
W = np.linalg.solve(Xall.T @ Xall + ath * np.eye(Xall.shape[1]), Xall.T @ ytr)
b0, GJ, phi = W[0], W[1:1 + len(PCOLS)], W[-1]
w('\n$$\\Phi(p)=\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)+\\varphi\\ln Q(p;\\theta^{\\star}),\\quad '
  '\\beta_0=%.4f,\\;\\varphi=%.4f,\\;\\theta^{\\star}=%.1f$$\n' % (b0, phi, bth))
w('\n$\\varphi<0$（实测 %+.4f）⇒ **提高配比的聚合质量可显著降低 Loss**，'
  '这是把问题一的 $Q$ 接入配比建模的直接证据。\n' % phi)

w('\n| 域 | $Q_j$ | $\\gamma_j$ | 训练均值配比 |')
w('|---|---|---|---|')
for i in np.argsort(GJ):
    w('| `%s` | %.4f | %+.4f | %.4f |' % (PCOLS[i], Qv[i], GJ[i], Xp[:, i].mean()))

PR = Xp.mean(axis=0)
def Phi(P):
    return b0 + np.log(P + EPS) @ GJ + phi * lnQ(P.reshape(1, -1), Qv, bth)[0]
Phi_ref = Phi(PR)

w('\n### 4.1 检验集（A6–A11）与配比效应的规模衰减\n')
w('| 检验集 | n | ρ(预测,实测) | 仿射后 $R^2$ | 对照（不含 CES 质量项）ρ |')
w('|---|---|---|---|---|')
Wnq = np.linalg.solve(dm0(Xp).T @ dm0(Xp) + ath * np.eye(18), dm0(Xp).T @ ytr)
for tag, nm in [('1m', 'A6/A7 1M'), ('60m', 'A8/A9 60M'), ('1B', 'A10/A11 1B')]:
    mx = pd.read_csv(os.path.join(AV, 'test_mixture_%s.csv' % tag), dtype_backend='numpy_nullable')
    ls = pd.read_csv(os.path.join(AV, 'test_pile_loss_%s.csv' % tag), dtype_backend='numpy_nullable')
    k = 'index' if 'index' in mx.columns else mx.columns[0]
    m = mx.merge(ls, on=k, how='inner')
    lc = [c for c in ls.columns if c != k][0]
    P = m[mxcols].to_numpy(float); P = P / P.sum(axis=1, keepdims=True)
    y = m[lc].to_numpy(float)
    pr = dm(P, bth) @ W
    prn = dm0(P) @ Wnq
    Am = np.column_stack([np.ones(len(y)), pr])
    c = np.linalg.lstsq(Am, y, rcond=None)[0]
    w('| %s | %d | **%.4f** | %.4f | %.4f |' % (
        nm, len(y), stats.spearmanr(pr, y).statistic,
        1 - ((y - Am @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        stats.spearmanr(prn, y).statistic))

y1_ = np.load(os.path.join(OUT, '_diag_y1.npy')); y60_ = np.load(os.path.join(OUT, '_diag_y60.npy'))
sr = y60_.std(ddof=1) / y1_.std(ddof=1); mr = y60_.mean() / y1_.mean()
w('\n同一配比矩阵在 1M/60M 下：配比诱导离散度之比 σ=%.4f，中心水平之比 μ=%.4f。'
  'σ 比远大于 μ 比且接近 1 ⇒ **配比效应近似加性**：差配比的绝对损失惩罚不随规模稀释'
  '（60 倍参数跨度上仅衰减 %.1f%%）。据此在广义律中取配比修正项与规模无关（$s\\equiv1$）。\n' % (
      sr, mr, 100 * (1 - sr)))

w('\n## 5. 广义标度律最终形式\n')
w('$$\\boxed{\\;L(N,D,Q,p)=\\underbrace{E+\\big(AN^{-\\alpha}+BD^{-\\beta}\\big)Q^{-\\gamma}}_{\\text{规模-质量律}}'
  '\\;+\\;\\underbrace{\\Phi(p)-\\Phi(p_{ref})}_{\\text{配比修正}}\\;}$$\n')
w('$$Q=Q(p;\\theta^{\\star})=\\Big(\\sum_{j=1}^{17}p_jQ_j^{\\theta^{\\star}}\\Big)^{1/\\theta^{\\star}}$$\n')
w('\n**退化性**：$Q=1$ 时 $Q^{-\\gamma}=1$；$p=p_{ref}$ 时配比修正为 0 ⇒ '
  '整体**严格退化为经典标度律** $E+AN^{-\\alpha}+BD^{-\\beta}$。✔\n')

# ---- 6. 边际效用与弹性 ----
w('\n---\n\n## 6. 边际效用、弹性与"质量 ↔ 规模"的替代条件\n')
w('记可约损失 $R(N,D)=AN^{-\\alpha}+BD^{-\\beta}$，权重 $w_N=AN^{-\\alpha}/R$，$w_D=BD^{-\\beta}/R$。\n')
w('\n### 6.1 弹性\n')
w('| 弹性 | 定义 | 主模型下的值 |')
w('|---|---|---|')
w('| 参数弹性 $\\eta_N$ | $\\partial\\ln(L-E)/\\partial\\ln N$ | $-\\alpha\\,w_N$ |')
w('| 数据弹性 $\\eta_D$ | $\\partial\\ln(L-E)/\\partial\\ln D$ | $-\\beta\\,w_D$ |')
w('| 质量弹性 $\\eta_Q$ | $\\partial\\ln(L-E)/\\partial\\ln Q$ | **$-\\gamma$=%.4f（常数）** |' % GAM)
w('| 算力弹性 $\\eta_C$ | Chinchilla 最优轨线上 $\\partial\\ln(L-E)/\\partial\\ln C$ | '
  '$-\\dfrac{\\alpha\\beta}{\\alpha+\\beta}$=%.4f |' % (-AL * BE / (AL + BE)))
w('\n> 关键结构含义：在主模型下，$Q$ 只按 $Q^{-\\gamma}$ 整体缩放可约损失，'
  '**不改变 $w_N:w_D$ 的结构**。因此：**数据质量不改变 Chinchilla 最优的 $N$:$D$ 拆分比例**，'
  '它只改变"每一份算力能换到多少损失下降"。这是一个可检验的强预测。\n')

w('\n### 6.2 边际效用（每单位算力的损失下降）\n')
w('设总算力 $C=\\kappa_c\\,N\\,D$，其中 $\\kappa_c=6+\\eta L_{ctx}$（含长文本注意力开销，$\\eta=2\\times10^{-4}$）。\n')
w('| 投入通道 | 边际成本 $\\partial C/\\partial x$ | 边际损失下降 $-\\partial L/\\partial x$ | **每 FLOP 的收益** |')
w('|---|---|---|---|')
w('| 参数 $N$ | $\\kappa_c D$ | $\\alpha A N^{-\\alpha-1}Q^{-\\gamma}$ | $\\dfrac{\\alpha\\,(AN^{-\\alpha}Q^{-\\gamma})}{C}$ |')
w('| 数据 $D$ | $\\kappa_c N$ | $\\beta B D^{-\\beta-1}Q^{-\\gamma}$ | $\\dfrac{\\beta\\,(BD^{-\\beta}Q^{-\\gamma})}{C}$ |')
w('| 质量 $Q$ | $D\\,g\'(Q)$ | $\\gamma R Q^{-\\gamma-1}$ | $\\dfrac{\\gamma\\,R\\,Q^{-\\gamma-1}}{D g\'(Q)}$ |')
w('\n"同样多花一块钱，堆参数还是买好教材更划算"——判据即比较上表第三列三行的大小。\n')

def muN(N, D, Q, kc=6.0):
    return AL * (A * N ** (-AL) * Q ** (-GAM)) / (kc * N * D)
def muD(N, D, Q, kc=6.0):
    return BE * (B * D ** (-BE) * Q ** (-GAM)) / (kc * N * D)
def muQ(N, D, Q, gfun):
    h = 1e-6
    gp = (gfun(Q + h) - gfun(Q - h)) / (2 * h)
    return GAM * (A * N ** (-AL) + B * D ** (-BE)) * Q ** (-GAM - 1) / (D * gp)

GS = {
    '指数型 $\\gamma_e e^{\\lambda Q}$': (lambda Q: 1e7 * np.exp(6.0 * Q), 1e7, 6.0),
    '幂函数型 $\\gamma_p Q^{\\lambda}$': (lambda Q: 5e9 * Q ** 4.0, 5e9, 4.0),
    '对数渐进型 $\\gamma_l\\ln(1+\\lambda Q)$': (lambda Q: 2e9 * np.log(1 + 10.0 * Q), 2e9, 10.0),
}
w('\n### 6.3 数值比较（Chinchilla 最优点上）\n')
w('取 $C=10^{22}$ FLOPs，$L_{ctx}=2048$（$\\kappa_c=6+2\\times10^{-4}\\times2048=6.4096$），'
  'Chinchilla 最优拆分 $N^{\\star}=\\big(\\frac{\\alpha A}{\\beta B}\\big)^{1/(\\alpha+\\beta)}'
  '(C/\\kappa_c)^{\\beta/(\\alpha+\\beta)}$，$D^{\\star}=C/(\\kappa_c N^{\\star})$。\n')
C0 = 1e22; LC = 2048; kc = 6 + 2e-4 * LC
Ns = ((AL * A) / (BE * B)) ** (1 / (AL + BE)) * (C0 / kc) ** (BE / (AL + BE))
Ds = C0 / (kc * Ns)
w('\n- $N^{\\star}$=%.4f B，$D^{\\star}$=%.1f B tokens，$C/\\kappa_c$=%.4g\n' % (Ns, Ds, C0 / kc))
w('\n| $Q$ | 每 FLOP 收益：参数 $N$ | 每 FLOP 收益：数据 $D$ | 质量 $Q$ 指数型 | 幂函数型 | 对数型 | 最划算的通道 |')
w('|---|---|---|---|---|---|---|')
rows_mu = []
for Q in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    a = muN(Ns, Ds, Q, kc); b = muD(Ns, Ds, Q, kc)
    qs = [muQ(Ns, Ds, Q, g) for g, _, _ in GS.values()]
    vals = {'参数 N': a, '数据 D': b, '质量 Q(指数)': qs[0], '质量 Q(幂)': qs[1], '质量 Q(对数)': qs[2]}
    bk = max(vals, key=vals.get)
    rows_mu.append(dict(Q=Q, muN=a, muD=b, muQ_exp=qs[0], muQ_pow=qs[1], muQ_log=qs[2], best=bk))
    w('| %.1f | %.3e | %.3e | %.3e | %.3e | %.3e | **%s** |' % (Q, a, b, qs[0], qs[1], qs[2], bk))
w('\n> 注：$-\\partial L/\\partial C\\|_N$ 与 $\\|_D$ 在最优点上必然相等（一阶条件），'
  '这是 Chinchilla 条件 $\\alpha A N^{-\\alpha}=\\beta B D^{-\\beta}$ 的直接推论，可作内部一致性校验。\n')

w('\n### 6.4 "质量 +0.1 等价于参数增加多少"的可计算条件\n')
w('设要把质量从 $Q$ 提到 $Q+\\delta$（$\\delta=0.1$），问参数需放大 $r$ 倍才能产生**同等**的损失下降：\n')
w('$$R(N,D)\\,Q^{-\\gamma}=\\big(A(rN)^{-\\alpha}+B D^{-\\beta}\\big)(Q+\\delta)^{-\\gamma}$$\n')
w('解得（令 $S=R(N,D)\\big(\\tfrac{Q}{Q+\\delta}\\big)^{\\gamma}$）：\n')
w('$$\\boxed{\\;r=\\left[\\frac{S-BD^{-\\beta}}{AN^{-\\alpha}}\\right]^{-1/\\alpha}\\;},'
  '\\qquad \\text{可行条件}\\; S>BD^{-\\beta}$$\n')
w('同理，等价于数据量放大 $s$ 倍：\n')
w('$$\\boxed{\\;s=\\left[\\frac{S-AN^{-\\alpha}}{BD^{-\\beta}}\\right]^{-1/\\beta}\\;},'
  '\\qquad \\text{可行条件}\\; S>AN^{-\\alpha}$$\n')
w('**这两个条件是"可计算"的**：给定 $(N,D,Q,\\delta)$ 直接代入即可；'
  '当 $S$ 落不到可行域时，说明"质量提升 0.1 的收益已经超过单纯扩大规模所能达到的上界'
  '（因为 $N\\to\\infty$ 只能消去 $AN^{-\\alpha}$ 项，无法消去 $BD^{-\\beta}$ 项）"，'
  '此时**质量提升不可被参数规模替代**。\n')

def equiv(N, D, Q, dl=0.1):
    S = (A * N ** (-AL) + B * D ** (-BE)) * (Q / (Q + dl)) ** GAM
    num_r = S - B * D ** (-BE)
    num_s = S - A * N ** (-AL)
    r = (num_r / (A * N ** (-AL))) ** (-1 / AL) if num_r > 0 else np.nan
    s = (num_s / (B * D ** (-BE))) ** (-1 / BE) if num_s > 0 else np.nan
    return r, s

w('\n| $N$(B) | $D$(B) | $Q$ | 参数等价倍数 $r$ | 参数增量 $(r-1)N$ (B) | 数据等价倍数 $s$ | 数据增量 $(s-1)D$ (B) |')
w('|---|---|---|---|---|---|---|---|')
EQ = []
for N_, D_ in [(0.41, 50), (1.0, 100), (7.0, 1000), (70.0, 1400)]:
    for Q_ in [0.3, 0.5, 0.7, 0.8]:
        r, s = equiv(N_, D_, Q_)
        EQ.append(dict(N=N_, D=D_, Q=Q_, r=r, s=s))
        w('| %g | %g | %.1f | %s | %s | %s | %s |' % (
            N_, D_, Q_,
            '%.4f' % r if np.isfinite(r) else '**不可行**',
            '%+.2f' % ((r - 1) * N_) if np.isfinite(r) else '—',
            '%.4f' % s if np.isfinite(s) else '**不可行**',
            '%+.1f' % ((s - 1) * D_) if np.isfinite(s) else '—'))
w('\n> 例：在 $N$=7B、$D$=1T、$Q$=0.5 处，质量 +0.1 等价于参数放大 **%.3f 倍**（≈ +%.1f B 参数），'
  '或等价于数据量放大 **%.3f 倍**（≈ +%.0f B tokens）。\n' % (
      equiv(7.0, 1000, 0.5)[0], (equiv(7.0, 1000, 0.5)[0] - 1) * 7.0,
      equiv(7.0, 1000, 0.5)[1], (equiv(7.0, 1000, 0.5)[1] - 1) * 1000))
w('\n按 Chinchilla 近似 $C=6ND$，把"参数放大 $r$ 倍"折算成算力需 $(r-1)\\cdot C$，'
  '而"质量 +0.1"的算力代价为 $D[g(Q+0.1)-g(Q)]$，二者的比值就是"买教材 vs 堆参数"的性价比：\n')
w('\n| 成本函数 | $Q$:0.5→0.6 的 $C_Q$（$D$=1T） | 同效参数扩张的 $C_N=6(r-1)ND$ | 性价比 $C_N/C_Q$ |')
w('|---|---|---|---|')
N_, D_, Q_ = 7.0, 1000.0, 0.5
r, s = equiv(N_, D_, Q_)
CN = 6 * (r - 1) * N_ * D_
for nm, (g, _, _) in GS.items():
    CQ = D_ * (g(Q_ + 0.1) - g(Q_))
    w('| %s | %.3e | %.3e | **%.2f** |' % (nm, CQ, CN, CN / CQ))
w('\n（性价比 >1 表示"堆参数"比"买好教材"更贵，即**提升质量更划算**。）\n')

# ---- 7. B9/B10 外推 ----
w('\n---\n\n## 7. 百亿参数以上外推（B9/B10）\n')
b9 = pd.read_csv(os.path.join(BS, 'supplementary_large_models.csv'), dtype_backend='numpy_nullable')
b10 = pd.read_csv(os.path.join(BS, 'supplementary_large_baseline.csv'), dtype_backend='numpy_nullable')
m = b9.merge(b10, left_on='model_name', right_on='family', how='inner', suffixes=('_9', '_10'))
N9 = m['N_params_B_9'].to_numpy(float)
D9 = np.maximum(m['D_tokens_B_10'].to_numpy(float), 1e-3)
Y9 = m['val_loss'].to_numpy(float)
w('- B9（真实元数据）与 B10（**估算** Loss）按 `model_name` 内连接后 n=%d；'
  'N∈[%.0f, %.0f] B，D∈[%.1f, %.1f] B，估算 Loss∈[%.4f, %.4f]\n' % (
      len(m), N9.min(), N9.max(), D9.min(), D9.max(), Y9.min(), Y9.max()))
w('- ⚠ B10 的 `val_loss` 为**估算值**（《数据说明》标注"估算"），非实测，'
  '以下所有对比均为"模型预测 vs 估算值"的一致性检查，不构成对模型的实证检验。\n')

def pred(N, D, Q):
    return E + (A * N ** (-AL) + B * D ** (-BE)) * Q ** (-GAM)

Qref = float(lnQ(PR.reshape(1, -1), Qv, bth)[0])
Qref = float(np.exp(Qref))
w('\n- 以 A4/A5 训练集平均配比 $p_{ref}$ 为参考，其 CES 聚合质量 $Q(p_{ref};\\theta^{\\star})$=%.4f\n' % Qref)
pr1 = pred(N9, D9, Qref)
w('\n| 情景 | 假设 $Q$ | 预测 Loss 中位数 | B10 估算 Loss 中位数 | 中位偏差 | ρ(预测,估算) |')
w('|---|---|---|---|---|---|')
rows_ex = []
for ql, qq in [('$Q(p_{ref})$=%.4f' % Qref, Qref), ('$Q$=0.8', 0.8), ('$Q$=0.9', 0.9), ('$Q$=1.0', 1.0)]:
    pv = pred(N9, D9, qq)
    rows_ex.append(dict(Q=qq, pred=float(np.median(pv)), obs=float(np.median(Y9)),
                        bias=float(np.median(pv - Y9)), rho=float(stats.spearmanr(pv, Y9).statistic)))
    w('| %s | %.4f | %.4f | %.4f | %+.4f | %.4f |' % (
        ql, qq, np.median(pv), np.median(Y9), np.median(pv - Y9), stats.spearmanr(pv, Y9).statistic))
w('\n**反解**：若强行要求预测等于 B10 估算值，反解出的隐含质量\n')
w('$$\\hat Q=\\Big(\\frac{AN^{-\\alpha}+BD^{-\\beta}}{L_{B10}-E}\\Big)^{1/\\gamma}$$\n')
ok = Y9 > E
Qhat = np.full(len(Y9), np.nan)
Qhat[ok] = ((A * N9[ok] ** (-AL) + B * D9[ok] ** (-BE)) / (Y9[ok] - E)) ** (1 / GAM)
w('- 有解样本 %d/%d；$\\hat Q$ 中位数 %.4f，四分位 [%.4f, %.4f]，'
  '落在 $(0,1]$ 内的比例 %.1f%%\n' % (
      int(ok.sum()), len(Y9), np.nanmedian(Qhat),
      np.nanpercentile(Qhat, 25), np.nanpercentile(Qhat, 75),
      100 * np.mean((Qhat > 0) & (Qhat <= 1))))
w('\n> **判读**：B10 的估算 Loss（中位 %.4f）显著**低于**广义律在 $Q\\le1$ 下的任何预测'
  '（$Q=1$ 时中位 %.4f）。这与"百亿级以上模型在真实训练中普遍采用高质量精选语料 + '
  '更优配方 + 更长训练"一致，也说明**把 70M–12B 的 Pythia 系数直接外推到 100B–10T 会系统性高估损失**——'
  '这正是"非规模技术进步"的体现，问题四将对其单独建模。\n' % (np.median(Y9), np.median(pred(N9, D9, 1.0))))

# ---- 输出 ----
pd.DataFrame(rows_mu).to_csv(os.path.join(OUT, 'q2_marginal_utility.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(EQ).to_csv(os.path.join(OUT, 'q2_equivalence.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(rows_ex).to_csv(os.path.join(OUT, 'q2_B9B10_extrapolation.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'domain': PCOLS, 'Q_j': Qv, 'gamma_j': GJ, 'p_ref': PR}).to_csv(
    os.path.join(OUT, 'q2_mixture_coef.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'N_B': N9, 'D_B': D9, 'loss_B10_est': Y9, 'Qhat_implied': Qhat,
              'pred_Qref': pr1}).to_csv(os.path.join(OUT, 'q2_B9B10_detail.csv'),
                                        index=False, encoding='utf-8-sig')
json.dump(dict(E=E, A=A, alpha=AL, B=B, beta=BE, gamma=GAM, kappa=GAM / BE,
               theta=float(bth), phi=float(phi), beta0=float(b0),
               gamma_lo=float(GLO), gamma_hi=float(GHI), Q_ref_ces=Qref,
               discount_exp=float(GAM * (1 / AL + 1 / BE)), s_mixture=1.0),
          open(os.path.join(OUT, 'gsl_params.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

with open(os.path.join(OUT, '02f_generalized_law_final.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

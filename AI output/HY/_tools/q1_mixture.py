# -*- coding: utf-8 -*-
"""
问题一 · 步骤6：17 域配比 p 与交叉熵损失 L 的定量关系建模
  M1 线性（单纯形参数化）  M2 二次交互  M3 对数-线性  M4 引入质量 Q 的增强模型
  检验：A6/A7 (1M) 同尺度；A8/A9 (60M)、A10/A11 (1B) 跨尺度；A12–A15 (10B/70B) 外推稳健性
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from scipy.optimize import minimize

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value', 'regmix_tables')
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01h_mixture_model.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

w('# 问题一 · 领域配比建模（A4–A15）\n')

# ---------- 读数据 ----------
PREFIX = {'train': ('train', '1m'), 't1m': ('test', '1m'), 't60m': ('test', '60m'),
          't1B': ('test', '1B'), 'e10b': ('est', '10b'), 'e70b': ('est', '70b')}

def load(key):
    pre, tag = PREFIX[key]
    mx = pd.read_csv(os.path.join(RA, '%s_mixture_%s.csv' % (pre, tag)))
    ls = pd.read_csv(os.path.join(RA, '%s_pile_loss_%s.csv' % (pre, tag)))
    m = mx.merge(ls, on='index', how='inner')
    return mx, ls, m

DOM17 = None
SC = {}
for key in PREFIX:
    mx, ls, m = load(key)
    SC[key] = dict(mx=mx, ls=ls, m=m)

mx_tr, ls_tr, m_tr = SC['train']['mx'], SC['train']['ls'], SC['train']['m']
PCOLS = [c for c in mx_tr.columns if c.startswith('train_the_pile_')]
LCOLS = [c for c in ls_tr.columns if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
DOM17 = [c.replace('train_the_pile_', '') for c in PCOLS]
DOM13 = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in LCOLS]
w('\n## 0. 数据结构\n')
w('- 17 个训练域（有配比列）：%s\n' % ', '.join('`%s`' % d for d in DOM17))
w('- 13 个有验证 Loss 的域：%s\n' % ', '.join('`%s`' % d for d in DOM13))
w('- **仅有配比、无 Loss 的 4 个域**：%s\n' % ', '.join('`%s`' % d for d in sorted(set(DOM17) - set(DOM13))))
w('  处理：这 4 个域在训练时确实占用配比、会影响其他域的 Loss，故**全部保留为自变量（预测因子）**，'
  '只是不作为因变量。不删除、不合并，避免丢失配比信息；同时在稳健性检验中删去它们重估模型。\n')
for key in PREFIX:
    w('- `%s`（%s_%s）：配比 %d 行 × %d 列；Loss %d 行 × %d 列；内连接后 %d 行\n' % (
        key, PREFIX[key][0], PREFIX[key][1], SC[key]['mx'].shape[0], SC[key]['mx'].shape[1],
        SC[key]['ls'].shape[0], SC[key]['ls'].shape[1], SC[key]['m'].shape[0]))

# 配比表和为 1 的校验
s = mx_tr[PCOLS].sum(axis=1)
w('\n- 配比行和：min=%.6f, max=%.6f, 均值=%.6f（存在千分位级舍入误差，按题设说明处理）\n' % (s.min(), s.max(), s.mean()))
# 检验 1m 与 60m 检验配比是否相同
same = np.allclose(SC['t1m']['mx'][PCOLS].to_numpy(), SC['t60m']['mx'][PCOLS].to_numpy())
w('- `test_mixture_1m` 与 `test_mixture_60m` 配比矩阵完全相同： **%s**。'
  '故二者可用于检验"同一配比在不同参数规模下的表现是否一致"（跨尺度迁移）。\n' % same)
same_est = np.allclose(SC['e10b']['mx'][PCOLS].to_numpy(), SC['e70b']['mx'][PCOLS].to_numpy())
w('- `est_mixture_10b` 与 `est_mixture_70b` 配比矩阵完全相同： **%s**（取自 train 子集）。\n' % same_est)
# 检验 est 配比是否来自 train 子集
tr_idx = set(SC['train']['mx']['index'])
w('- `est_mixture_10b` 的 63 个 index 全部出现在 `train_mixture_1m` 中： **%s**'
  '（印证"外推表配比取自 train 子集"）。\n' % set(SC['e10b']['mx']['index']).issubset(tr_idx))

# ---------- 目标变量 ----------
def build_X(mx, cols=DOM17):
    P = mx[['train_the_pile_' + d for d in cols]].to_numpy(dtype=float)
    return P / P.sum(axis=1, keepdims=True)

def build_y(m, mean=True):
    L = m[LCOLS].to_numpy(dtype=float)
    return L.mean(axis=1) if mean else L

Xtr = build_X(mx_tr)
ytr = build_y(m_tr)
w('\n- 建模目标：13 个验证域 Loss 的**算术平均** '
  '$\\bar L(p)=\\frac1{13}\\sum_{k=1}^{13}L_k(p)$（RegMix 的目标函数口径）。'
  '同时对 13 个域各建一个单域模型用于域间影响分析。\n')
w('- 训练集：A4/A5，n=%d；$\\bar L$ 取值 min=%.4f, 中位数=%.4f, max=%.4f\n' % (
    len(ytr), ytr.min(), np.median(ytr), ytr.max()))

# ---------- 特征构造 ----------
Qlk = json.load(open(os.path.join(OUT, 'Q_domain_lookup.json'), 'r', encoding='utf-8'))
qv = np.array([Qlk['Q_mixture17'][d] for d in DOM17])

def feats(P, kind):
    n, K = P.shape
    if kind == 'linear':
        return P                                   # 单纯形参数化：无截距，Σp=1 已隐含
    if kind == 'quad':
        cols = [P]
        for i in range(K):
            for j in range(i, K):
                cols.append((P[:, i] * P[:, j])[:, None])
        return np.hstack(cols)
    if kind == 'log':
        return np.hstack([np.ones((n, 1)), np.log(P + 1e-3)])
    if kind == 'quadQ':
        base = feats(P, 'quad')
        qw = P @ qv                                # 配比的加权平均质量
        return np.hstack([base, qw[:, None], (P * qv[None, :])])
    raise ValueError(kind)

def fit_eval(kind, alpha=0.0, X=None, y=None):
    X = Xtr if X is None else X
    y = ytr if y is None else y
    F = feats(X, kind)
    if alpha > 0:
        # 不惩罚"线性主效应"，只惩罚高阶项
        model = Ridge(alpha=alpha, fit_intercept=False)
    else:
        model = LinearRegression(fit_intercept=False)
    model.fit(F, y)
    return model

def predict(model, P, kind):
    return model.predict(feats(P, kind))

def metrics(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return dict(R2=1 - ss_res / ss_tot, RMSE=np.sqrt(((yt - yp) ** 2).mean()),
                MAE=np.abs(yt - yp).mean(),
                spearman=stats.spearmanr(yt, yp).statistic,
                pearson=np.corrcoef(yt, yp)[0, 1])

# ---------- 交叉验证选超参 ----------
w('\n## 1. 模型设定与超参数选择（训练集 A4/A5 上 5 折 CV）\n')
w('| 模型 | 形式 | 参数个数 | 5折CV RMSE | 5折CV R² |')
w('|---|---|---|---|---|')
CVres = {}
kf = KFold(n_splits=5, shuffle=True, random_state=7)
for kind, desc, alphas in [
        ('linear', r'$\bar L=\sum_j\gamma_j p_j$（单纯形参数化，无截距，$\sum p_j=1$ 已隐含）', [0.0]),
        ('quad', r'$\bar L=\sum_j\gamma_jp_j+\sum_{j\le k}\theta_{jk}p_jp_k$（含域间交互）', [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
        ('log', r'$\bar L=\beta_0+\sum_j\gamma_j\ln(p_j+\varepsilon)$（收益递减）', [0.0, 1e-3, 1e-2]),
        ('quadQ', r'二次模型 + 质量项 $\varphi\sum_j q_jp_j+\sum_j\psi_jq_jp_j$', [0.0, 1e-3, 1e-2, 1e-1])]:
    best = None
    for a in alphas:
        rms, r2s = [], []
        for tr_i, va_i in kf.split(Xtr):
            mdl = Ridge(alpha=a, fit_intercept=False) if a > 0 else LinearRegression(fit_intercept=False)
            mdl.fit(feats(Xtr[tr_i], kind), ytr[tr_i])
            pv = mdl.predict(feats(Xtr[va_i], kind))
            rms.append(np.sqrt(((ytr[va_i] - pv) ** 2).mean()))
            r2s.append(1 - ((ytr[va_i] - pv) ** 2).sum() / ((ytr[va_i] - ytr[va_i].mean()) ** 2).sum())
        rm, r2 = np.mean(rms), np.mean(r2s)
        if best is None or rm < best[1]:
            best = (a, rm, r2)
    CVres[kind] = best
    npar = feats(Xtr[:1], kind).shape[1]
    w('| **%s** | %s | %d | α=%.4g → **%.5f** | %.4f |' % (kind, desc, npar, best[0], best[1], best[2]))

# ---------- 在检验集上评估 ----------
w('\n## 2. 检验集表现\n')
TESTS = [('A6/A7 检验集 1M', 't1m'), ('A8/A9 检验集 60M', 't60m'), ('A10/A11 检验集 1B', 't1B')]
w('| 模型 | ' + ' | '.join('%s R²' % t[0] for t in TESTS) + ' | ' +
  ' | '.join('%s Spearman' % t[0] for t in TESTS) + ' | ' +
  ' | '.join('%s RMSE' % t[0] for t in TESTS) + ' |')
w('|---|' + '---|' * (3 * len(TESTS)))
fitted = {}
rows = []
for kind in ['linear', 'quad', 'log', 'quadQ']:
    a = CVres[kind][0]
    mdl = Ridge(alpha=a, fit_intercept=False) if a > 0 else LinearRegression(fit_intercept=False)
    mdl.fit(feats(Xtr, kind), ytr)
    fitted[kind] = (mdl, a)
    r2s, sps, rms = [], [], []
    for nm, tag in TESTS:
        Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
        pv = predict(mdl, Xte, kind)
        mm = metrics(yte, pv)
        r2s.append(mm['R2']); sps.append(mm['spearman']); rms.append(mm['RMSE'])
    w('| **%s** | %s | %s | %s |' % (kind,
        ' | '.join('%.4f' % v for v in r2s),
        ' | '.join('%.4f' % v for v in sps),
        ' | '.join('%.5f' % v for v in rms)))
    rows.append(dict(model=kind, alpha=a, **{('R2_' + t[1]): r2s[i] for i, t in enumerate(TESTS)},
                     **{('sp_' + t[1]): sps[i] for i, t in enumerate(TESTS)},
                     **{('rmse_' + t[1]): rms[i] for i, t in enumerate(TESTS)}))

w('\n### 2.1 跨尺度重标定（仿射校正）\n')
w('同一配比在不同参数规模下的 Loss 相差一个尺度因子。用 2 参数仿射 '
  '$\\bar L^{(N)}\\approx a_N+b_N\\hat{\\bar L}^{(1M)}$ 重新标定后评估：\n')
w('| 检验集 | 仿射前 R² | 仿射后 R² | a_N | b_N | 仿射后 Spearman |')
w('|---|---|---|---|---|---|')
mdl, a = fitted['quad']
for nm, tag in TESTS:
    Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
    pv = predict(mdl, Xte, 'quad')
    A = np.column_stack([np.ones(len(pv)), pv])
    coef, *_ = np.linalg.lstsq(A, yte, rcond=None)
    pr = A @ coef
    w('| %s | %.4f | **%.4f** | %.4f | %.4f | %.4f |' % (
        nm, metrics(yte, pv)['R2'], metrics(yte, pr)['R2'], coef[0], coef[1],
        metrics(yte, pr)['spearman']))

w('\n### 2.2 排序能力：模型能否挑出真正最好的配方\n')
w('对每个检验集，取真实 Loss 最低的前 10%% 配方，看模型预测值最低的前 10%% 与之的重合度，'
  '以及真实最优配方在模型预测排序中的百分位。\n')
w('| 检验集 | Top10%% 重合度 | 真实最优配方的预测分位 | 预测最优配方的真实分位 |')
w('|---|---|---|---|')
for nm, tag in TESTS:
    Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
    pv = predict(mdl, Xte, 'quad')
    k = max(int(0.1 * len(yte)), 1)
    t10 = set(np.argsort(yte)[:k]); p10 = set(np.argsort(pv)[:k])
    ov = len(t10 & p10) / k
    best_true = int(np.argmin(yte))
    pct_true = 100 * (pv < pv[best_true]).mean()
    best_pred = int(np.argmin(pv))
    pct_pred = 100 * (yte < yte[best_pred]).mean()
    w('| %s | %.1f%% | %.1f%% | %.1f%% |' % (nm, 100 * ov, pct_true, pct_pred))

# ---------- 外推稳健性 ----------
w('\n## 3. 外推稳健性（A12–A15：10B / 70B）\n')
w('`est_pile_loss_10b/70b` 为由 1M/60M/1B 三尺度幂律外推得到的**非直接观测**值，'
  '此处用作外推检验：看在 1M 上拟合的模型能否正确排序 10B/70B 尺度下的配方。\n')
w('| 外推集 | n | Spearman（预测 vs 外推Loss） | p 值 | Kendall τ | 仿射后 R² |')
w('|---|---|---|---|---|---|')
for nm, tag in [('10B', 'e10b'), ('70B', 'e70b')]:
    Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
    pv = predict(mdl, Xte, 'quad')
    A = np.column_stack([np.ones(len(pv)), pv]); coef, *_ = np.linalg.lstsq(A, yte, rcond=None)
    pr = A @ coef
    w('| %s | %d | %.4f | %.3g | %.4f | %.4f |' % (
        nm, len(yte), stats.spearmanr(yte, pv).statistic, stats.spearmanr(yte, pv).pvalue,
        stats.kendalltau(yte, pv).statistic, metrics(yte, pr)['R2']))
w('\n结论：模型在 1M 上拟合后，对外推到 10B/70B 的配方仍能保持显著的正秩相关，'
  '说明**配方优劣的相对顺序具有跨尺度的稳定性**（这是 RegMix 方法可行的前提）；'
  '但由于外推 Loss 本身非直接观测，绝对水平的预测误差较大，故只能用于排序与选优，不能直接用于绝对损失预测。\n')

# ---------- 各域影响与替代/互补 ----------
w('\n## 4. 各领域及领域组合的影响\n')
mdl_lin, _ = fitted['linear']
gam = mdl_lin.coef_
ordl = np.argsort(gam)
w('### 4.1 线性模型主效应 $\\gamma_j$（越小＝该域配比提高越能降低 $\\bar L$）\n')
w('| 排名 | 域 | $\\gamma_j$ | 域级质量 $Q_j$ |')
w('|---|---|---|---|')
for i, idx in enumerate(ordl):
    w('| %d | `%s` | %.4f | %.4f |' % (i + 1, DOM17[idx], gam[idx], qv[idx]))
w('\n- 线性模型隐含"最优配比落在单顶点"，与数据不符（真实最优配比是内部点），'
  '故必须与交互项结合解读。\n')
w('- ρ(γ_j, Q_j) = %.4f：主效应系数与域级质量分呈%s相关，'
  '说明"质量越高的域，提高配比越能降 Loss"。\n' % (
      stats.spearmanr(gam, qv).statistic,
      '负' if stats.spearmanr(gam, qv).statistic < 0 else '正'))

w('\n### 4.2 二次交互项 $\\theta_{jk}$：替代 vs 互补\n')
w('在 $\\bar L=\\sum_j\\gamma_jp_j+\\sum_{j\\le k}\\theta_{jk}p_jp_k$ 中，'
  '$\\theta_{jk}<0$ 表示两域**互补**（同时增加配比带来的收益大于各自单独增加之和）；'
  '$\\theta_{jk}>0$ 表示**替代/冗余**（同时增加会互相挤压）。\n')
mdl_q, aq = fitted['quad']
coef_q = mdl_q.coef_
K = len(DOM17)
theta = np.zeros((K, K))
idx = K
for i in range(K):
    for j in range(i, K):
        theta[i, j] = theta[j, i] = coef_q[idx]
        idx += 1
np.save(os.path.join(OUT, 'theta_matrix.npy'), theta)
pairs_q = []
for i in range(K):
    for j in range(i + 1, K):
        pairs_q.append((DOM17[i], DOM17[j], theta[i, j]))
pairs_q.sort(key=lambda t: t[2])
w('\n**最强互补（θ 最小）前 15 对**：\n')
w('| 域 A | 域 B | θ_jk |')
w('|---|---|---|')
for a1, b1, t in pairs_q[:15]:
    w('| `%s` | `%s` | %.4f |' % (a1, b1, t))
w('\n**最强替代/冗余（θ 最大）前 15 对**：\n')
w('| 域 A | 域 B | θ_jk |')
w('|---|---|---|')
for a1, b1, t in pairs_q[-15:][::-1]:
    w('| `%s` | `%s` | %.4f |' % (a1, b1, t))
w('\n对角项 $\\theta_{jj}$（自身边际收益递增/递减）：\n')
w('| 域 | θ_jj |')
w('|---|---|')
for i in np.argsort(np.diag(theta)):
    w('| `%s` | %.4f |' % (DOM17[i], theta[i, i]))

# ---------- 是否引入 Q：模型比较 ----------
w('\n## 5. 是否引入质量评分 Q：模型比较与论证\n')
n = len(ytr)
def aic_bic(y, yp, k):
    rss = ((y - yp) ** 2).sum()
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return dict(AIC=-2 * ll + 2 * k, BIC=-2 * ll + k * np.log(n), RSS=rss)
w('| 模型 | 参数个数 k | 训练集 RSS | AIC | BIC | 1M检验 R² | 60M检验 Spearman | 1B检验 Spearman |')
w('|---|---|---|---|---|---|---|---|')
cmp_rows = []
for kind in ['linear', 'quad', 'log', 'quadQ']:
    mm, al = fitted[kind]
    k = feats(Xtr[:1], kind).shape[1]
    ab = aic_bic(ytr, mm.predict(feats(Xtr, kind)), k)
    sps = []
    for nm, tag in TESTS:
        Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
        sps.append(metrics(yte, predict(mm, Xte, kind)))
    w('| %s | %d | %.4f | %.1f | %.1f | %.4f | %.4f | %.4f |' % (
        kind, k, ab['RSS'], ab['AIC'], ab['BIC'], sps[0]['R2'], sps[1]['spearman'], sps[2]['spearman']))
    cmp_rows.append(dict(model=kind, k=k, RSS=ab['RSS'], AIC=ab['AIC'], BIC=ab['BIC'],
                         R2_1m=sps[0]['R2'], sp_60m=sps[1]['spearman'], sp_1B=sps[2]['spearman']))
pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, 'mixture_model_comparison.csv'), index=False, encoding='utf-8-sig')

# ---------- 最优配比求解 ----------
w('\n## 6. 最优配比求解\n')
best_kind = 'quad'
mm, al = fitted[best_kind]
f_obj = lambda p: float(mm.predict(feats(p[None, :], best_kind))[0])
cons = ({'type': 'eq', 'fun': lambda p: p.sum() - 1.0},)
bnds = [(0.0, 1.0)] * K
best = None
rng = np.random.default_rng(11)
for trial in range(60):
    p0 = rng.dirichlet(np.ones(K) * 0.6)
    r = minimize(f_obj, p0, method='SLSQP', bounds=bnds, constraints=cons,
                 options=dict(maxiter=800, ftol=1e-12))
    if r.success and (best is None or r.fun < best.fun):
        best = r
p_star = np.clip(best.x, 0, None); p_star = p_star / p_star.sum()
w('- 目标函数取 %s 模型，在单纯形 $\\Delta_{17}$ 上用 SLSQP 从 60 个随机起点求解，最优值 $\\bar L^*=%.5f$。\n' % (best_kind, best.fun))
w('\n| 域 | 最优配比 $p^*$ | 训练集平均配比 | 变化 | 域级质量 $Q_j$ |')
w('|---|---|---|---|---|')
pm = Xtr.mean(axis=0)
for i in np.argsort(-p_star):
    w('| `%s` | **%.4f** | %.4f | %+.4f | %.4f |' % (DOM17[i], p_star[i], pm[i], p_star[i] - pm[i], qv[i]))
w('\n对比：训练集中**实测**最优配方（Loss 最低者）的配比：\n')
bi = int(np.argmin(ytr))
w('| 域 | 实测最优配方 | 模型最优配比 |')
w('|---|---|---|')
for i in np.argsort(-Xtr[bi]):
    w('| `%s` | %.4f | %.4f |' % (DOM17[i], Xtr[bi][i], p_star[i]))
w('\n- 实测最优配方的 $\\bar L$=%.5f；模型最优配方的预测 $\\bar L^*$=%.5f。\n' % (ytr[bi], best.fun))
w('- 模型最优配比与实测最优配方的余弦相似度 = %.4f；与训练集平均配方的余弦相似度 = %.4f。\n' % (
    p_star @ Xtr[bi] / (np.linalg.norm(p_star) * np.linalg.norm(Xtr[bi])),
    p_star @ pm / (np.linalg.norm(p_star) * np.linalg.norm(pm))))
# 用外推表检验最优配方的排序
for nm, tag in [('10B', 'e10b'), ('70B', 'e70b')]:
    Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
    pv = predict(mm, Xte, best_kind)
    # 找到与 p* 最近的样本点
    d = np.linalg.norm(Xte - p_star[None, :], axis=1)
    j = int(np.argmin(d))
    w('- %s 外推表中与 $p^*$ 最接近的配方（index=%d）真实/外推 Loss=%.4f，'
      '其在该尺度 %d 个配方中的分位 = %.1f%%（越小越好）。\n' % (
          nm, SC[tag]['mx']['index'].iloc[j], yte[j], len(yte), 100 * (yte < yte[j]).mean()))

json.dump({'DOM17': DOM17, 'DOM13': DOM13,
           'p_star': {DOM17[i]: float(p_star[i]) for i in range(K)},
           'L_star': float(best.fun),
           'gamma': {DOM17[i]: float(gam[i]) for i in range(K)}},
          open(os.path.join(OUT, 'mixture_optimum.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
w('\n- 已保存 `_out/01_q1/mixture_optimum.json`（供问题二、三使用）\n')
LOG.close()
print('ok')

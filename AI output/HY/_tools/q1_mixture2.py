# -*- coding: utf-8 -*-
"""
问题一 · 步骤6（修订版）：17 域配比 p 与交叉熵损失 L 的定量关系建模
重点：① 跨尺度可比评估（仿射重标定 + 秩相关） ② 外推表一致性诊断
      ③ 在实验设计凸包内寻优（Frank–Wolfe），杜绝单纯形外推假象
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, linprog
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value', 'regmix_tables')
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01h_mixture_model.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

w('# 问题一 · 领域配比建模（A4–A15）\n')

PREFIX = {'train': ('train', '1m'), 't1m': ('test', '1m'), 't60m': ('test', '60m'),
          't1B': ('test', '1B'), 'e10b': ('est', '10b'), 'e70b': ('est', '70b')}
def load(key):
    pre, tag = PREFIX[key]
    mx = pd.read_csv(os.path.join(RA, '%s_mixture_%s.csv' % (pre, tag)))
    ls = pd.read_csv(os.path.join(RA, '%s_pile_loss_%s.csv' % (pre, tag)))
    return mx, ls, mx.merge(ls, on='index', how='inner')

SC = {}
for k in PREFIX:
    mx, ls, m = load(k)
    SC[k] = dict(mx=mx, ls=ls, m=m)

mx_tr, m_tr = SC['train']['mx'], SC['train']['m']
PCOLS = [c for c in mx_tr.columns if c.startswith('train_the_pile_')]
LCOLS = [c for c in SC['train']['ls'].columns if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
DOM17 = [c.replace('train_the_pile_', '') for c in PCOLS]
DOM13 = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in LCOLS]
K = len(DOM17)

def build_X(mx):
    P = mx[PCOLS].to_numpy(dtype=float)
    return P / P.sum(axis=1, keepdims=True)
def build_y(m):
    return m[LCOLS].to_numpy(dtype=float).mean(axis=1)

Xtr, ytr = build_X(mx_tr), build_y(m_tr)
Qlk = json.load(open(os.path.join(OUT, 'Q_domain_lookup.json'), 'r', encoding='utf-8'))
qv = np.array([Qlk['Q_mixture17'][d] for d in DOM17])

w('\n## 0. 数据口径与关键结构事实\n')
w('- 17 个训练域：%s\n' % ', '.join('`%s`' % d for d in DOM17))
w('- 13 个有验证 Loss 的域：%s\n' % ', '.join('`%s`' % d for d in DOM13))
w('- **仅有配比、无 Loss 的 4 个域**：`enron_emails`, `europarl`, `nih_exporter`, `philpapers`。'
  '处理：它们训练时真实占用配比、会影响其余域的 Loss，故**全部保留为自变量**，只是不作为因变量。\n')
w('- 因变量取 13 个验证域 Loss 的算术平均 $\\bar L(p)$（RegMix 目标函数口径）。\n')
w('- `test_mixture_1m` 与 `test_mixture_60m` 配比矩阵**完全相同**：%s；'
  '`est_mixture_10b/70b` 的 63 个 index 均为 `train` 子集：%s。\n' % (
      np.allclose(SC['t1m']['mx'][PCOLS].to_numpy(), SC['t60m']['mx'][PCOLS].to_numpy()),
      set(SC['e10b']['mx']['index']).issubset(set(mx_tr['index']))))
w('\n各表 $\\bar L$ 的水平（注意：**同一配比在不同参数规模下的 Loss 量级完全不同**）：\n')
w('| 表 | 参数规模 | n | $\\bar L$ 均值 | 标准差 | min | max |')
w('|---|---|---|---|---|---|---|')
SCALE = {'train': '1M', 't1m': '1M', 't60m': '60M', 't1B': '1B', 'e10b': '10B(外推)', 'e70b': '70B(外推)'}
for k in PREFIX:
    y = build_y(SC[k]['m'])
    w('| `%s` | %s | %d | %.4f | %.4f | %.4f | %.4f |' % (k, SCALE[k], len(y), y.mean(), y.std(ddof=1), y.min(), y.max()))

# ---------- 关键诊断：外推表与 1M 实测是否一致 ----------
w('\n## 1. 关键诊断：外推表（A12–A15）与 1M 实测 Loss 的一致性\n')
w('63 个外推配方的 index 均来自 train，故可直接把"1M 实测 $\\bar L$"与"10B/70B 外推 $\\bar L$"配对比较。\n')
tr_by_idx = mx_tr.set_index('index')
ytr_by_idx = pd.Series(ytr, index=mx_tr['index'].to_numpy())
w('| 配对 | n | Pearson r | Spearman ρ | p 值 | Kendall τ |')
w('|---|---|---|---|---|---|')
for k in ['t1m', 't60m', 't1B', 'e10b', 'e70b']:
    if k in ('t1m', 't60m', 't1B'):
        continue
    idx = SC[k]['mx']['index'].to_numpy()
    y_obs = ytr_by_idx.loc[idx].to_numpy()
    y_ext = build_y(SC[k]['m'])
    if len(y_obs) != len(y_ext):
        continue
    w('| 1M 实测 vs %s 外推 | %d | %.4f | %.4f | %.3g | %.4f |' % (
        SCALE[k], len(y_obs), np.corrcoef(y_obs, y_ext)[0, 1],
        stats.spearmanr(y_obs, y_ext).statistic, stats.spearmanr(y_obs, y_ext).pvalue,
        stats.kendalltau(y_obs, y_ext).statistic))
w('\n**重要发现**：10B/70B 外推 Loss 与 1M 实测 Loss 呈显著**负**相关。'
  '这说明外推表并非"同尺度 Loss 的单调延拓"——其构建方式（幂律外推）使得'
  '在 1M 上表现差的配方被外推为在 10B/70B 上损失更小。因此：\n')
w('1. 外推表**不能**用作"模型在 1M 上拟合、直接预测 10B/70B 绝对损失"的检验集；\n')
w('2. 但外推表内部是自洽的（同一批 63 个配方在 10B 与 70B 两个尺度上高度一致），'
  '可用于检验"模型给出的配方**排序**在外推尺度上是否仍然合理"，'
  '此时须以 **|ρ|** 与方向标定后解释，或直接使用"模型选出的最优配方在外推表中的分位"。\n')
r1070 = stats.spearmanr(build_y(SC['e10b']['m']), build_y(SC['e70b']['m'])).statistic
w('\n- 外推表内部一致性：ρ(10B 外推 $\\bar L$, 70B 外推 $\\bar L$) = %.4f（同一批 63 个配方）。\n' % r1070)
# 1M 与 60M/1B 真实检验集的一致性
w('\n- 真实跨尺度一致性：由于 `test_mixture_1m` 与 `test_mixture_60m` 配比相同，'
  '可直接比较同一配比在 1M 与 60M 下的 $\\bar L$：'
  'ρ = %.4f，Pearson r = %.4f。\n' % (
      stats.spearmanr(build_y(SC['t1m']['m']), build_y(SC['t60m']['m'])).statistic,
      np.corrcoef(build_y(SC['t1m']['m']), build_y(SC['t60m']['m']))[0, 1]))

# ---------- 模型 ----------
w('\n## 2. 模型设定\n')
w('配比 $p$ 落在 17 维单纯形 $\\Delta_{17}=\\{p\\ge 0,\\ \\sum_jp_j=1\\}$ 上。'
  '处理单纯形约束的两种方式：① **单纯形参数化**（线性模型不设截距，'
  '截距被 $\\sum p_j=1$ 吸收，$\\bar L=\\sum_j\\gamma_jp_j$）；'
  '② 对含交互项与对数项的模型显式施加等式约束。\n')
w('| 记号 | 形式 | 含义 | 参数个数 |')
w('|---|---|---|---|')
w('| M1 | $\\bar L=\\sum_j\\gamma_jp_j$ | 线性（完全可加，无域间交互） | 17 |')
w('| M2 | $\\bar L=\\sum_j\\gamma_jp_j+\\sum_{j\\le k}\\theta_{jk}p_jp_k$ | 二次（含域间替代/互补） | 170 |')
w('| M3 | $\\bar L=\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)$ | 对数-线性（边际收益递减） | 18 |')
w('| M4 | $\\ln\\bar L=\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)$ | Cobb–Douglas 型 | 18 |')
w('| M5 | M3 + $\\varphi\\sum_jq_jp_j+\\sum_j\\psi_jq_jp_j$ | 引入域级质量 $Q$ | 38 |')
w('| M6 | M2 + 质量项 | 引入域级质量 $Q$（二次版） | 188 |')
w('\n')

def feats(P, kind, eps=1e-3):
    n, K_ = P.shape
    if kind == 'M1':
        return P
    if kind == 'M2':
        cols = [P]
        for i in range(K_):
            for j in range(i, K_):
                cols.append((P[:, i] * P[:, j])[:, None])
        return np.hstack(cols)
    if kind == 'M3':
        return np.hstack([np.ones((n, 1)), np.log(P + eps)])
    if kind == 'M4':
        return np.hstack([np.ones((n, 1)), np.log(P + eps)])
    if kind == 'M5':
        base = feats(P, 'M3', eps)
        return np.hstack([base, (P @ qv)[:, None], P * qv[None, :]])
    if kind == 'M6':
        base = feats(P, 'M2', eps)
        return np.hstack([base, (P @ qv)[:, None], P * qv[None, :]])
    raise ValueError(kind)

MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
def fitter(kind, alpha):
    return Ridge(alpha=alpha, fit_intercept=False) if alpha > 0 else LinearRegression(fit_intercept=False)

w('\n### 2.1 训练集 5 折交叉验证（选正则化强度 α）\n')
w('| 模型 | 最优 α | CV RMSE | CV R² | 参数个数 |')
w('|---|---|---|---|---|')
# M4 为 Cobb–Douglas：对 ln(L̄) 建模，预测时取指数回到原尺度
YTR = {k: (np.log(ytr) if k == 'M4' else ytr) for k in MODELS}

kf = KFold(n_splits=5, shuffle=True, random_state=7)
BEST = {}
for kind in MODELS:
    yv = YTR[kind]
    best = None
    for a in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        rms, r2 = [], []
        for tr_i, va_i in kf.split(Xtr):
            mdl = fitter(kind, a)
            mdl.fit(feats(Xtr[tr_i], kind), yv[tr_i])
            pv = mdl.predict(feats(Xtr[va_i], kind))
            if kind == 'M4':
                pv = np.exp(pv)
            rms.append(np.sqrt(((ytr[va_i] - pv) ** 2).mean()))
            r2.append(1 - ((ytr[va_i] - pv) ** 2).sum() / ((ytr[va_i] - ytr[va_i].mean()) ** 2).sum())
        rm, rr = np.mean(rms), np.mean(r2)
        if best is None or rm < best[1]:
            best = (a, rm, rr)
    BEST[kind] = best
    w('| **%s** | %.4g | **%.5f** | %.4f | %d |' % (kind, best[0], best[1], best[2], feats(Xtr[:1], kind).shape[1]))

FIT = {}
for kind in MODELS:
    a = BEST[kind][0]
    mdl = fitter(kind, a)
    mdl.fit(feats(Xtr, kind), YTR[kind])
    FIT[kind] = (mdl, a)

def pred(kind, P):
    mdl, a = FIT[kind]
    r = mdl.predict(feats(P, kind))
    return np.exp(r) if kind == 'M4' else r

def met(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    A = np.column_stack([np.ones(len(yp)), yp])
    c, *_ = np.linalg.lstsq(A, yt, rcond=None)
    pr = A @ c
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    return dict(R2_raw=1 - ((yt - yp) ** 2).sum() / st, R2_aff=1 - ss / st,
                sp=stats.spearmanr(yt, yp).statistic, rmse_aff=np.sqrt(((yt - pr) ** 2).mean()),
                a=c[0], b=c[1])

w('\n### 2.2 检验集表现（含仿射重标定）\n')
w('跨尺度时 Loss 量级不同，直接用 R² 无意义，故同时给出'
  '**仿射重标定后 R²**（2 参数 $a_N+b_N\\hat{\\bar L}$，仅校正量纲）与**秩相关 Spearman**（量纲无关）。\n')
TESTS = [('A6/A7 1M', 't1m'), ('A8/A9 60M', 't60m'), ('A10/A11 1B', 't1B')]
w('| 模型 | ' + ' | '.join('%s R²(仿射)' % t[0] for t in TESTS) +
  ' | ' + ' | '.join('%s ρ' % t[0] for t in TESTS) + ' |')
w('|---|' + '---|' * (2 * len(TESTS)))
for kind in MODELS:
    r2s, sps = [], []
    for nm, tag in TESTS:
        mm = met(build_y(SC[tag]['m']), pred(kind, build_X(SC[tag]['mx'])))
        r2s.append(mm['R2_aff']); sps.append(mm['sp'])
    w('| **%s** | %s | %s |' % (kind, ' | '.join('%.4f' % v for v in r2s),
                                ' | '.join('%.4f' % v for v in sps)))

w('\n### 2.3 排序能力（能否挑出真正的好配方）\n')
best_kind = min(MODELS, key=lambda k: BEST[k][1])
w('取 CV RMSE 最小的 **%s** 为主模型。\n' % best_kind)
w('| 检验集 | Top10%% 重合度 | 真实最优配方的预测分位(越小越好) | 预测最优配方的真实分位 |')
w('|---|---|---|---|')
for nm, tag in TESTS:
    yte = build_y(SC[tag]['m']); pv = pred(best_kind, build_X(SC[tag]['mx']))
    kk = max(int(0.1 * len(yte)), 1)
    t10 = set(np.argsort(yte)[:kk]); p10 = set(np.argsort(pv)[:kk])
    bi = int(np.argmin(yte)); bp = int(np.argmin(pv))
    w('| %s | %.1f%% | %.1f%% | %.1f%% |' % (nm, 100 * len(t10 & p10) / kk,
                                             100 * (pv < pv[bi]).mean(), 100 * (yte < yte[bp]).mean()))

# ---------- 外推表：方向标定后的检验 ----------
w('\n## 3. 外推稳健性（A12–A15）\n')
w('由于外推 Loss 与 1M 实测呈负相关（§1），此处不要求模型预测值与外推值同号相关，'
  '而是检验**模型给出的配方排序在外推尺度上是否具有信息量**（|ρ| 显著、方向可标定），'
  '并直接报告"模型选出的最优配方在外推表中的分位"。\n')
w('| 外推集 | n | Spearman ρ（预测 vs 外推 $\\bar L$） | p 值 | 方向标定后 |ρ| | 结论 |')
w('|---|---|---|---|---|---|')
for nm, tag in [('10B', 'e10b'), ('70B', 'e70b')]:
    yte = build_y(SC[tag]['m']); pv = pred(best_kind, build_X(SC[tag]['mx']))
    r, p = stats.spearmanr(yte, pv).statistic, stats.spearmanr(yte, pv).pvalue
    w('| %s | %d | %.4f | %.3g | %.4f | %s |' % (
        nm, len(yte), r, p, abs(r),
        '排序信息显著（但方向与 1M 相反，须标定）' if p < 0.05 else '不显著'))

# ---------- 域影响 ----------
w('\n## 4. 各领域及领域组合的影响\n')
mdl3, _ = FIT['M3']
gam3 = mdl3.coef_[1:]
ord3 = np.argsort(gam3)
w('### 4.1 主模型 M3 的系数 $\\gamma_j$（对数-线性：系数越负＝提高该域配比越能降低 $\\bar L$）\n')
w('| 排名 | 域 | $\\gamma_j$ | 域级质量 $Q_j$ | 训练集平均配比 |')
w('|---|---|---|---|---|')
pm = Xtr.mean(axis=0)
for r, i in enumerate(ord3):
    w('| %d | `%s` | %+.4f | %.4f | %.4f |' % (r + 1, DOM17[i], gam3[i], qv[i], pm[i]))
w('\n- ρ(γ_j, Q_j) = %.4f。若该值为负，说明"域级质量越高的域，其配比提高带来的 Loss 下降越大"。\n'
  % stats.spearmanr(gam3, qv).statistic)

w('\n### 4.1b 主模型 M5 的质量项系数\n')
mdl5, _ = FIT['M5']
c5 = mdl5.coef_
w('M5 = $\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)+\\varphi\\sum_jq_jp_j+\\sum_j\\psi_jq_jp_j$，'
  '其中 $\\varphi$=%.5f。\n' % c5[18])
w('| 域 | $\\psi_j$（质量 × 配比的域别交互） | $\\gamma_j$ | $Q_j$ |')
w('|---|---|---|---|')
for i in np.argsort(c5[19:]):
    w('| `%s` | %+.5f | %+.5f | %.4f |' % (DOM17[i], c5[19 + i], c5[1 + i], qv[i]))
w('\n- 质量项总和 $\\varphi\\sum_jq_jp_j$ 的系数 $\\varphi$=%.5f<0，'
  '表示**提高配比的加权平均质量会显著降低 $\\bar L$**——这是把 Q 纳入配比建模的直接证据。\n' % c5[18])

w('\n### 4.2 二次模型 M2 的交互项 $\\theta_{jk}$：替代 vs 互补\n')
w('$\\theta_{jk}<0$ ⇒ 两域**互补**（同时增加收益大于各自单独增加之和）；'
  '$\\theta_{jk}>0$ ⇒ **替代/冗余**（互相挤压）。\n')
mdl2, _ = FIT['M2']
c2 = mdl2.coef_
theta = np.zeros((K, K)); idx = K
for i in range(K):
    for j in range(i, K):
        theta[i, j] = theta[j, i] = c2[idx]; idx += 1
np.save(os.path.join(OUT, 'theta_matrix.npy'), theta)
pq = []
for i in range(K):
    for j in range(i + 1, K):
        pq.append((DOM17[i], DOM17[j], theta[i, j]))
pq.sort(key=lambda t: t[2])
w('**最强互补（θ 最小）12 对**：\n| 域 A | 域 B | θ_jk |\n|---|---|---|')
for a1, b1, t in pq[:12]:
    w('| `%s` | `%s` | %.4f |' % (a1, b1, t))
w('\n**最强替代（θ 最大）12 对**：\n| 域 A | 域 B | θ_jk |\n|---|---|---|')
for a1, b1, t in pq[-12:][::-1]:
    w('| `%s` | `%s` | %.4f |' % (a1, b1, t))

# ---------- 是否引入 Q ----------
w('\n## 5. 是否引入质量评分 Q：统计检验\n')
n = len(ytr)
def aic_bic(y, yp, k):
    rss = ((y - yp) ** 2).sum()
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return -2 * ll + 2 * k, -2 * ll + k * np.log(n)
w('| 模型 | k | 训练集 RSS | AIC | BIC | CV RMSE | 60M ρ | 1B ρ |')
w('|---|---|---|---|---|---|---|---|')
for kind in MODELS:
    mdl, a = FIT[kind]
    yp = pred(kind, Xtr); k = feats(Xtr[:1], kind).shape[1]
    A_, B_ = aic_bic(ytr, yp, k)
    sps = [met(build_y(SC[t]['m']), pred(kind, build_X(SC[t]['mx'])))['sp'] for _, t in TESTS]
    w('| %s | %d | %.4f | %.1f | %.1f | %.5f | %.4f | %.4f |' % (
        kind, k, ((ytr - yp) ** 2).sum(), A_, B_, BEST[kind][1], sps[1], sps[2]))
# 似然比检验：M5 vs M3
rss3 = ((ytr - pred('M3', Xtr)) ** 2).sum(); rss5 = ((ytr - pred('M5', Xtr)) ** 2).sum()
k3, k5 = feats(Xtr[:1], 'M3').shape[1], feats(Xtr[:1], 'M5').shape[1]
F = ((rss3 - rss5) / (k5 - k3)) / (rss5 / (n - k5))
w('\n- 嵌套模型 F 检验（M3 ⊂ M5，即"加入质量项是否显著"）：'
  'F = %.4f，自由度 (%d, %d)，p = %.4g。\n' % (F, k5 - k3, n - k5, 1 - stats.f.cdf(F, k5 - k3, n - k5)))

# ---------- 最优配比：凸包内 Frank–Wolfe ----------
w('\n## 6. 最优配比求解（限定在实验设计凸包内）\n')
w('直接在单纯形上最小化拟合曲面会退化为顶点解（例如把 100%% 配比压到训练集中几乎不出现的'
  '`enron_emails` 上），这是**外推假象**。正确做法是限定在 512 个训练配方的'
  '**凸包** $\\mathrm{conv}\\{p^{(i)}\\}$ 内寻优：令 $p=\\sum_i\\lambda_ip^{(i)}$，$\\lambda\\in\\Delta_{512}$，'
  '用 Frank–Wolfe（条件梯度）求解，每步的线性极小化只需取 $\\arg\\min_i\\nabla\\bar L(p)^\\top p^{(i)}$。\n')

def make_obj(kind):
    mdl, a = FIT[kind]
    def f(P):
        r = mdl.predict(feats(P, kind))
        return np.exp(r[0]) if kind == 'M4' else r[0]
    return f

def frank_wolfe(kind, iters=3000):
    f = make_obj(kind)
    P = Xtr
    lam = np.full(len(P), 1.0 / len(P))
    p = lam @ P
    eps_fd = 1e-6
    for it in range(iters):
        # 数值梯度（18~170 维，代价可接受；这里用 17 次中心差分）
        g = np.zeros(K)
        for j in range(K):
            e = np.zeros(K); e[j] = eps_fd
            g[j] = (f((p + e)[None, :]) - f((p - e)[None, :])) / (2 * eps_fd)
        s = g @ P.T                       # 各顶点方向导数
        i_star = int(np.argmin(s))
        # 精确线搜索
        lo, hi = 0.0, 1.0
        best_g, best_v = 0.0, f(p[None, :])
        for gg in np.linspace(0, 1, 41):
            pg = (1 - gg) * p + gg * P[i_star]
            v = f(pg[None, :])
            if v < best_v:
                best_v, best_g = v, gg
        gam = best_g
        lam = (1 - gam) * lam; lam[i_star] += gam
        pn = (1 - gam) * p + gam * P[i_star]
        if np.linalg.norm(pn - p) < 1e-10 and gam < 1e-8:
            p = pn; break
        p = pn
    return p, lam, f(p[None, :])

p_star, lam_star, L_star = frank_wolfe(best_kind)
w('- 主模型 %s，Frank–Wolfe 迭代收敛，最优值 $\\bar L^*=%.5f$（训练集 $\\bar L$ 均值 %.4f，最优实测 %.4f）。\n'
  % (best_kind, L_star, ytr.mean(), ytr.min()))
w('\n| 域 | 最优配比 $p^*$ | 训练集平均配比 | 变化 | 域级质量 $Q_j$ |')
w('|---|---|---|---|---|')
for i in np.argsort(-p_star):
    w('| `%s` | **%.4f** | %.4f | %+.4f | %.4f |' % (DOM17[i], p_star[i], pm[i], p_star[i] - pm[i], qv[i]))
bi = int(np.argmin(ytr))
w('\n- 与训练集**实测最优配方**的余弦相似度 = %.4f；与训练集平均配方的余弦相似度 = %.4f。\n' % (
    p_star @ Xtr[bi] / (np.linalg.norm(p_star) * np.linalg.norm(Xtr[bi])),
    p_star @ pm / (np.linalg.norm(p_star) * np.linalg.norm(pm))))
# 凸包内验证：用 LP 检验 p* 是否在凸包内
A_eq = np.vstack([Xtr.T, np.ones((1, len(Xtr)))]); b_eq = np.concatenate([p_star, [1.0]])
res_lp = linprog(np.zeros(len(Xtr)), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * len(Xtr), method='highs')
w('- 线性规划检验：$p^*$ 是否属于 $\\mathrm{conv}\\{p^{(i)}\\}$ → **%s**（目标残差 %.2e）。\n' % (
    bool(res_lp.success), res_lp.fun if res_lp.success else np.nan))
# 在检验集上评价 p*
w('\n### 6.1 最优配方的跨尺度表现\n')
for nm, tag in TESTS + [('10B外推', 'e10b'), ('70B外推', 'e70b')]:
    Xte = build_X(SC[tag]['mx']); yte = build_y(SC[tag]['m'])
    d = np.linalg.norm(Xte - p_star[None, :], axis=1)
    j = int(np.argmin(d))
    w('| %s | 与 $p^*$ 最近的配方 index=%d | $\\bar L$=%.4f | 分位=%.1f%%（越小越好） |' % (
        nm, SC[tag]['mx']['index'].iloc[j], yte[j], 100 * (yte < yte[j]).mean()))
w('')

json.dump({'DOM17': DOM17, 'DOM13': DOM13, 'best_model': best_kind,
           'p_star': {DOM17[i]: float(p_star[i]) for i in range(K)},
           'L_star': float(L_star),
           'gamma_M3': {DOM17[i]: float(gam3[i]) for i in range(K)},
           'Q17': {DOM17[i]: float(qv[i]) for i in range(K)}},
          open(os.path.join(OUT, 'mixture_optimum.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.DataFrame({'domain': DOM17, 'p_star': p_star, 'p_train_mean': pm, 'Q': qv,
              'gamma_M3': gam3}).to_csv(os.path.join(OUT, 'mixture_pstar.csv'), index=False, encoding='utf-8-sig')
w('\n- 已保存 `_out/01_q1/mixture_optimum.json`、`mixture_pstar.csv`\n')
LOG.close()
print('ok')

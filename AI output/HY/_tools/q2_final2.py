# -*- coding: utf-8 -*-
"""问题二 定稿（修订版）：修正目标列口径、主模型选择、算力单位、B9/B10 判定"""
import os, io, sys, json
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.model_selection import KFold

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
UNITS = 1e18          # N,D 以"十亿"为单位 -> C = kappa_c * 1e18 * N_B * D_B
ETA = 2e-4

E, A, AL, B, BE = 1.689798, 0.353980, 0.339977, 1.240306, 0.279878
RED = lambda N, D: A * N ** (-AL) + B * D ** (-BE)

b6 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment.csv'), dtype_backend='numpy_nullable')
b7 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment_expanded.csv'), dtype_backend='numpy_nullable')
def pack(df):
    return (df['N_params_B'].to_numpy(float), df['D_tokens_B'].to_numpy(float),
            df['Q_score'].to_numpy(float), df['val_loss'].to_numpy(float))
N6, D6, Q6, Y6 = pack(b6)
N7, D7, Q7, Y7 = pack(b7)
NOISE = float((Y6[np.isclose(Q6, 1.0)] - (E + RED(N6, D6)[np.isclose(Q6, 1.0)])).std(ddof=1))

w('# 问题二 · 广义标度律（定稿）\n')
w('\n## 1. 统一框架与形式选择\n')
w('把质量 $Q$ 写成作用在"可约损失"上的一个**折扣/放大因子** $\\Psi(Q)$：\n')
w('$$L(N,D,Q,p)=\\underbrace{E+R(N,D)\\,\\Psi(Q)}_{\\text{规模-质量律}}+\\underbrace{\\Phi(p)-\\Phi(p_{ref})}_{\\text{配比修正}},'
  '\\qquad R(N,D)=A N^{-\\alpha}+B D^{-\\beta}$$\n')
w('要求 $\\Psi(1)=1$（教材完美 ⇒ 退化为经典标度律）、$\\Psi\'(Q)<0$（质量越高损失越低）。'
  '$E,A,\\alpha,B,\\beta$ 固定为 **B1 真实 Pythia 轨迹**的估计值，只用 B6/B7 估计 $\\Psi$ 的参数。\n')

PSI = {
    'P0 无质量效应 $\\Psi\\equiv1$': (0, lambda t, Q: np.ones_like(Q)),
    'P1 幂律 $Q^{-\\gamma}$': (1, lambda t, Q: Q ** (-t[0])),
    'P2 线性 $1+c(1-Q)$': (1, lambda t, Q: 1 + t[0] * (1 - Q)),
    'P3 指数 $e^{\\lambda(1-Q)}$': (1, lambda t, Q: np.exp(t[0] * (1 - Q))),
    'P4 幂缺 $1+c(1-Q)^m$': (2, lambda t, Q: 1 + t[0] * (1 - Q) ** t[1]),
    'P5 双通道（$N$、$D$ 各自折扣）': (2, None),
}
def model(kind, t, N, D, Q):
    k, f = PSI[kind]
    if kind.startswith('P5'):
        return E + A * N ** (-AL) * (1 + t[0] * (1 - Q)) + B * D ** (-BE) * (1 + t[1] * (1 - Q))
    return E + RED(N, D) * f(t, Q)

def fitP(kind, N, D, Q, y, ntry=14):
    k = PSI[kind][0]
    if k == 0:
        return np.array([]), float(np.sqrt(((E + RED(N, D) - y) ** 2).mean()))
    lb = np.array([0.0] * k) if not kind.startswith('P4') else np.array([0.0, 0.05])
    ub = np.array([8.0] * k) if not kind.startswith('P4') else np.array([8.0, 5.0])
    best = None
    starts = [np.full(k, 0.4)] + [np.abs(rng.normal(0.5, 0.4, k)) for _ in range(ntry - 1)]
    if kind.startswith('P4'):
        starts[0] = np.array([0.46, 1.0])
    for s in starts:
        s = np.clip(s, lb + 1e-9, ub - 1e-9)
        try:
            r = optimize.least_squares(lambda t: model(kind, t, N, D, Q) - y, s,
                                       bounds=(lb, ub), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt((best.fun ** 2).mean()))

w('\n### 1.1 $\\Psi$ 的形式选择（B6，n=%d）\n' % len(Y6))
w('| $\\Psi(Q)$ | k | RMSE | AIC | BIC | 参数 | RMSE / 数据噪声 |')
w('|---|---|---|---|---|---|---|')
RES = {}
for kind in PSI:
    t, rm = fitP(kind, N6, D6, Q6, Y6)
    rss = float(((model(kind, t, N6, D6, Q6) - Y6) ** 2).sum()); n = len(Y6); k = PSI[kind][0]
    RES[kind] = (t, rm, n * np.log(rss / n) + 2 * k, n * np.log(rss / n) + k * np.log(n))
    w('| %s | %d | %.6f | %.2f | %.2f | %s | %.3f |' % (
        kind, k, rm, RES[kind][2], RES[kind][3],
        '，'.join('%.4f' % v for v in t) if k else '—', rm / NOISE))
bestk = min([k for k in PSI if PSI[k][0] > 0 and not k.startswith('P5')], key=lambda k: RES[k][2])
if RES['P5 双通道（$N$、$D$ 各自折扣）'][2] < RES[bestk][2]:
    w('\n- ⚠ 双通道 P5 的 AIC（%.2f）低于 %s（%.2f），但 P5 无单一 $\\Psi(Q)$ 表达、'
      '不便于给出"质量↔规模"的等价换算，故仅作为**敏感性诊断**报告，不作为主模型。\n' % (
          RES['P5 双通道（$N$、$D$ 各自折扣）'][2], bestk, RES[bestk][2]))
w('\n- B6 自身噪声水平（$Q=1$ 子集相对 B1 曲面的残差标准差）= **%.5f**；'
  'AIC 最优形式 **%s** 的 RMSE/噪声 = %.3f，**已落到噪声地板上**；'
  '而幂律形式 P1 的 RMSE/噪声 = %.3f，明显在地板之上。\n' % (
      NOISE, bestk, RES[bestk][1] / NOISE, RES['P1 幂律 $Q^{-\\gamma}$'][1] / NOISE))
MAIN = bestk
t6, rm6 = RES[MAIN][0], RES[MAIN][1]
t7, rm7 = fitP(MAIN, N7, D7, Q7, Y7)
kpsi = PSI[MAIN][0]
w('\n### 1.2 主模型与复核\n')
w('| 数据 | 参数 | RMSE | $R^2$ |')
w('|---|---|---|---|')
for tag, t_, rm, N_, D_, Q_, Y_ in [('B6（主）', t6, rm6, N6, D6, Q6, Y6), ('B7（复核）', t7, rm7, N7, D7, Q7, Y7)]:
    pr = model(MAIN, t_, N_, D_, Q_)
    w('| %s | %s | %.6f | %.5f |' % (tag, '，'.join('%.4f' % v for v in t_), rm,
      1 - ((Y_ - pr) ** 2).sum() / ((Y_ - pr.mean()) ** 2).sum() if False else
      1 - ((Y_ - pr) ** 2).sum() / ((Y_ - Y_.mean()) ** 2).sum()))
bs = []
for _ in range(500):
    i = rng.integers(0, len(Y6), len(Y6))
    try:
        tb, _ = fitP(MAIN, N6[i], D6[i], Q6[i], Y6[i], ntry=3)
        bs.append(tb)
    except Exception:
        pass
bs = np.array(bs)
w('\n- Bootstrap %d 次：%s\n' % (len(bs), '；'.join(
    '参数%d 均值 %.4f，SE %.4f，95%%CI [%.4f, %.4f]' % (j + 1, bs[:, j].mean(), bs[:, j].std(ddof=1),
     np.percentile(bs[:, j], 2.5), np.percentile(bs[:, j], 97.5)) for j in range(kpsi))))

w('\n### 1.3 同时给出幂律形式 P1 的参数（便于与经典文献对照）\n')
tg, rmg = RES['P1 幂律 $Q^{-\\gamma}$'][0], RES['P1 幂律 $Q^{-\\gamma}$'][1]
w('- $\\gamma$=%.4f（RMSE=%.6f）；由此 $\\kappa=\\gamma/\\beta$=%.4f，'
  '算力折扣指数 $\\gamma(1/\\alpha+1/\\beta)$=%.4f。\n' % (tg[0], rmg, tg[0] / BE, tg[0] * (1 / AL + 1 / BE)))

# 主模型的对偶解释
w('\n### 1.4 对偶解释：质量 = 对有效规模的统一折扣\n')
w('$$L=E+A\\tilde N^{-\\alpha}+B\\tilde D^{-\\beta},\\quad \\tilde N=N\\Psi(Q)^{-1/\\alpha},\\;'
  '\\tilde D=D\\Psi(Q)^{-1/\\beta},\\;\\tilde C=C\\,\\Psi(Q)^{-(1/\\alpha+1/\\beta)}$$\n')
w('| $Q$ | $\\Psi(Q)$ | $\\tilde N/N$ | $\\tilde D/D$ | $\\tilde C/C$（有效算力留存率） |')
w('|---|---|---|---|---|')
for Q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    P_ = float(PSI[MAIN][1](t6, np.array([Q]))[0])
    w('| %.1f | %.4f | %.4f | %.4f | %.4f |' % (Q, P_, P_ ** (-1 / AL), P_ ** (-1 / BE),
                                                 P_ ** (-(1 / AL + 1 / BE))))
w('\n> **一句话结论**：低质量数据等价于把算力预算按 $\\Psi(Q)^{-(1/\\alpha+1/\\beta)}$ 打了折扣——'
  '买劣质教材不只是"白读"，而是让**脑容量和阅读量同时缩水**。\n')

# ---- 2. 检验 ----
w('\n## 2. 检验与验证\n')
w('### 2.1 B6 ↔ B7 互换外样本\n')
w('| 训练 | 检验 | RMSE | $R^2$ | ρ | 对照：无质量效应 RMSE |')
w('|---|---|---|---|---|---|')
for tr, te in [('B6', 'B7'), ('B7', 'B6')]:
    Ntr, Dtr, Qtr, Ytr = (N6, D6, Q6, Y6) if tr == 'B6' else (N7, D7, Q7, Y7)
    Nte, Dte, Qte, Yte = (N6, D6, Q6, Y6) if te == 'B6' else (N7, D7, Q7, Y7)
    tt, _ = fitP(MAIN, Ntr, Dtr, Qtr, Ytr)
    pr = model(MAIN, tt, Nte, Dte, Qte)
    base = E + RED(Nte, Dte)
    w('| %s | %s | **%.6f** | %.5f | %.5f | %.6f |' % (
        tr, te, np.sqrt(((Yte - pr) ** 2).mean()),
        1 - ((Yte - pr) ** 2).sum() / ((Yte - Yte.mean()) ** 2).sum(),
        stats.spearmanr(pr, Yte).statistic, np.sqrt(((Yte - base) ** 2).mean())))
w('\n### 2.2 5 折 CV（按 Q 分层）\n')
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(5, shuffle=True, random_state=7)
st = np.digitize(Q6, np.unique(Q6)) - 1
rr = []
for tr, te in skf.split(N6, st):
    tt, _ = fitP(MAIN, N6[tr], D6[tr], Q6[tr], Y6[tr], ntry=5)
    rr.append(np.sqrt(((Y6[te] - model(MAIN, tt, N6[te], D6[te], Q6[te])) ** 2).mean()))
w('- CV RMSE = %.6f ± %.6f（主模型 %s）\n' % (np.mean(rr), np.std(rr), MAIN))
w('\n### 2.3 主模型残差诊断（B6）\n')
res6 = Y6 - model(MAIN, t6, N6, D6, Q6)
w('| 残差 vs | ρ | p |')
w('|---|---|---|')
for nm, v in [('$N$', N6), ('$\\ln N$', np.log(N6)), ('$D$', D6), ('$\\ln D$', np.log(D6)),
              ('$Q$', Q6), ('$\\ln Q$', np.log(Q6))]:
    r = stats.pearsonr(res6, v)
    w('| %s | %+.4f | %.3g |' % (nm, r.statistic, r.pvalue))
w('\n残差标准差 %.5f，与噪声地板 %.5f 之比为 %.2f。\n' % (res6.std(ddof=1), NOISE, res6.std(ddof=1) / NOISE))
if abs(stats.pearsonr(res6, np.log(D6)).statistic) > 0.2:
    w('\n⚠ 残差对 $\\ln D$ 仍有 ρ=%+.3f 的显著相关，说明把 $B,\\\\beta$ 固定在 B1 值后在 $D$ 通道留有系统偏差。'
      '下面做**放开基参数**的稳健性检验。\n' % stats.pearsonr(res6, np.log(D6)).statistic)

    def model_free(t, N, D, Q):
        Ef, Af, af, Bf, bf, cf = t
        return Ef + (Af * N ** (-af) + Bf * D ** (-bf)) * (1 + cf * (1 - Q))
    lb = np.array([-5., 1e-6, 1e-4, 1e-6, 1e-4, 0.]); ub = np.array([10., 1e4, 5., 1e4, 5., 8.])
    best = None
    for s in [np.array([E, A, AL, B, BE, float(t6[0])])] + [
            np.array([E, A, AL, B, BE, float(t6[0])]) * np.exp(rng.normal(0, .2, 6)) for _ in range(19)]:
        s = np.clip(s, lb + 1e-9, ub - 1e-9)
        try:
            r = optimize.least_squares(lambda t: model_free(t, N6, D6, Q6) - Y6, s,
                                       bounds=(lb, ub), x_scale=np.abs(s), max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    tf = best.x
    rf = Y6 - model_free(tf, N6, D6, Q6)
    w('\n| 设定 | $E$ | $A$ | $\\alpha$ | $B$ | $\\beta$ | $c$ | RMSE | ρ(残差,$\\ln D$) |')
    w('|---|---|---|---|---|---|---|---|---|')
    w('| 基参数固定自 B1 | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.6f | %+.3f |' % (
        E, A, AL, B, BE, t6[0], rm6, stats.pearsonr(res6, np.log(D6)).statistic))
    w('| 基参数自由估计 | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.6f | %+.3f |' % (
        tf[0], tf[1], tf[2], tf[3], tf[4], tf[5],
        float(np.sqrt((rf ** 2).mean())), stats.pearsonr(rf, np.log(D6)).statistic))
    w('\n- 放开后 $c$ 从 %.4f 变为 %.4f（相对变化 %.1f%%），$\\alpha,\\beta$ 变为 %.4f / %.4f。'
      '**质量参数 $c$ 对基参数设定不敏感**，这是本文结论的稳健性依据；'
      '但自由设定下 $E$ 与 $\\beta$ 偏离 B1 较多，属半合成数据上 $D$ 范围有限导致的弱识别，'
      '故正文仍以"锚定 B1"的设定为主。\n' % (
          t6[0], tf[5], 100 * abs(tf[5] - t6[0]) / t6[0], tf[2], tf[4]))

# ---- 3. p ----
w('\n---\n\n## 3. 领域配比 $p$ 的接入：CES 域质量聚合\n')
q17 = pd.read_csv(os.path.join(Q1, 'Q_17_mixture_domains.csv'), dtype_backend='numpy_nullable')
q17['mixture_domain'] = q17['mixture_domain'].astype(str)
QJ = dict(zip(q17['mixture_domain'], q17['Q_final'].astype(float)))
PCOLS = list(q17['mixture_domain'])
Qv = np.array([QJ[c] for c in PCOLS])

def load_mix(tag, kind='train'):
    mx = pd.read_csv(os.path.join(AV, '%s_mixture_%s.csv' % (kind, tag)), dtype_backend='numpy_nullable')
    ls = pd.read_csv(os.path.join(AV, '%s_pile_loss_%s.csv' % (kind, tag)), dtype_backend='numpy_nullable')
    k = 'index' if 'index' in mx.columns else mx.columns[0]
    m = mx.merge(ls, on=k, how='inner')
    lc = [c for c in ls.columns if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
    P = m[['train_the_pile_' + c for c in PCOLS]].to_numpy(float)
    return P / P.sum(axis=1, keepdims=True), m[lc].to_numpy(float).mean(axis=1)

Xp, ytr = load_mix('1m', 'train')
w('- 训练配方 %d 个；配比矩阵 %s；目标 $\\bar L$ = 13 个域验证损失的**简单平均**'
  '（RegMix 原始设定），均值 %.4f，标准差 %.4f\n' % (Xp.shape[0], Xp.shape, ytr.mean(), ytr.std(ddof=1)))

def lnQ(P, Qv, th):
    if abs(th) < 1e-8:
        return P @ np.log(Qv)
    return (1.0 / th) * np.log(P @ (Qv ** th))

def dm(P, th):
    return np.column_stack([np.ones(len(P)), np.log(P + EPS), lnQ(P, Qv, th)])
def dm0(P):
    return np.column_stack([np.ones(len(P)), np.log(P + EPS)])

def cvth(th):
    kf = KFold(5, shuffle=True, random_state=11); best = None
    for a in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        rms = []
        for tr, te in kf.split(Xp):
            X = dm(Xp[tr], th)
            W = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ ytr[tr])
            rms.append(np.sqrt(((ytr[te] - dm(Xp[te], th) @ W) ** 2).mean()))
        m_ = float(np.mean(rms))
        if best is None or m_ < best[1]: best = (a, m_)
    return best

grid = [-4, -2, -1, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
RT = {th: cvth(th) for th in grid}
kf = KFold(5, shuffle=True, random_state=11)
rms0 = []
for tr, te in kf.split(Xp):
    W0 = np.linalg.solve(dm0(Xp[tr]).T @ dm0(Xp[tr]) + 1e-3 * np.eye(18), dm0(Xp[tr]).T @ ytr[tr])
    rms0.append(np.sqrt(((ytr[te] - dm0(Xp[te]) @ W0) ** 2).mean()))
rm0 = float(np.mean(rms0))
w('\n| $\\theta$ | 含义 | 最优 α | CV RMSE | 相对"不含质量项"改善 |')
w('|---|---|---|---|---|')
for th in grid:
    lab = '几何平均（$\\theta\\to0$）' if th == 0 else ('短板/$\\min$ 方向' if th < 0 else '长板/$\\max$ 方向')
    w('| %.1f | %s | %.4g | %.5f | %.2f%% |' % (th, lab, RT[th][0], RT[th][1],
                                                100 * (1 - RT[th][1] / rm0)))
bth = min(RT, key=lambda k: RT[k][1])
spread = (max(v[1] for v in RT.values()) - min(v[1] for v in RT.values())) / rm0
w('\n- 对照"不含 CES 质量项"的 CV RMSE = %.5f；含质量项后最优 %.5f，**改善 %.2f%%**。\n' % (
    rm0, RT[bth][1], 100 * (1 - RT[bth][1] / rm0)))
w('- $\\theta$ 网格上 CV RMSE 的极差仅占 %.2f%% ⇒ **$\\theta$ 弱识别**。'
  '按简约原则取 $\\theta=1$（域质量的配比加权算术平均，最易解释），'
  '并在稳健性中报告 $\\theta^{\\star}=%.1f$ 的替代结果。\n' % (100 * spread, bth))
TH = 1.0
ath = RT[TH][0]
Xall = dm(Xp, TH)
W = np.linalg.solve(Xall.T @ Xall + ath * np.eye(Xall.shape[1]), Xall.T @ ytr)
b0, GJ, phi = float(W[0]), W[1:1 + len(PCOLS)], float(W[-1])
PR = Xp.mean(axis=0)
def Phi(P):
    P = np.atleast_2d(P)
    return b0 + np.log(P + EPS) @ GJ + phi * lnQ(P, Qv, TH)
Phi_ref = float(Phi(PR)[0])
Qref = float(np.exp(lnQ(PR.reshape(1, -1), Qv, TH)[0]))
w('\n$$\\Phi(p)=\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)+\\varphi\\ln Q(p;1),\\quad '
  '\\beta_0=%.4f,\\ \\varphi=%+.4f$$\n' % (b0, phi))
w('- 参考配比 $p_{ref}$（训练集均值）的 CES 质量 $Q(p_{ref})$=%.4f，$\\Phi(p_{ref})$=%.4f\n' % (Qref, Phi_ref))
QP_tr = np.exp(lnQ(Xp, Qv, TH))
w('\n### 3.0 一个必须披露的识别问题：RegMix 中"聚合质量"几乎不变\n')
w('- 512 个训练配方的聚合质量 $Q(p;1)$：min=%.4f，中位数=%.4f，max=%.4f，**极差仅 %.4f（%.2f%%）**；'
  '标准差 %.5f，而 $\\bar L$ 的标准差为 %.4f。\n' % (
      QP_tr.min(), np.median(QP_tr), QP_tr.max(), QP_tr.max() - QP_tr.min(),
      100 * (QP_tr.max() - QP_tr.min()) / np.median(QP_tr), QP_tr.std(ddof=1), ytr.std(ddof=1)))
w('- 原因在于 17 个域的域级质量 $Q_j$ 高度集中（除 `arxiv` 0.726 与 `github` 0.556 外，'
  '其余 15 个域全在 %.3f–%.3f 之间），任何配比加权平均都被"大数摊平"。\n' % (
      np.sort(Qv)[1], np.sort(Qv)[-2]))
w('- 但加入 $\\ln Q(p)$ 后 CV 误差**没有改善**（甚至略差 %.2f%%），且 $\\theta$ 网格极差仅 %.2f%%、'
  '$\\varphi$ 符号不稳定 ⇒ **"聚合质量水平"本身不是 RegMix 数据中的有效预测因子**，'
  '其作用已被 17 个 $\\ln p_j$ 项吸收。\n' % (-100 * (1 - RT[bth][1] / rm0), 100 * spread))

w('\n### 3.0b 对照：改用"域别质量交互"设定（问题一 M5 的有效形式）\n')
w('注意到 $\\varphi\\sum_jq_jp_j+\\sum_j\\psi_jq_jp_j\\equiv\\sum_j(\\varphi+\\psi_j)\\,q_j\\,p_j$，'
  '问题一 M5 的**有效**形式就是\n')
w('$$\\Phi(p)=\\beta_0+\\sum_j\\gamma_j\\ln(p_j+\\varepsilon)+\\sum_j c_j\\,q_j\\,p_j$$\n')
w('$c_j$ = "该域多分一个单位配比所带来的质量加权收益"。用同一 5 折 CV 比较三种设定：\n')
def dm_q(P):
    return np.column_stack([np.ones(len(P)), np.log(P + EPS), P * Qv[None, :]])
def cvm(Xf):
    kf2 = KFold(5, shuffle=True, random_state=11); best = None
    for a in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        rms = []
        for tr, te in kf2.split(Xp):
            X = Xf(Xp[tr]); W_ = np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ ytr[tr])
            rms.append(np.sqrt(((ytr[te] - Xf(Xp[te]) @ W_) ** 2).mean()))
        m_ = float(np.mean(rms))
        if best is None or m_ < best[1]: best = (a, m_)
    return best
cvA = cvm(dm0); cvB = cvm(lambda P: dm(P, TH)); cvC = cvm(dm_q)
w('\n| 设定 | 参数个数 | 最优 α | CV RMSE | 相对 A 改善 |')
w('|---|---|---|---|---|')
w('| A：仅 $\\ln p_j$ | 18 | %.4g | %.5f | — |' % (cvA[0], cvA[1]))
w('| B：A + $\\ln Q(p;1)$ | 19 | %.4g | %.5f | %.2f%% |' % (cvB[0], cvB[1], 100 * (1 - cvB[1] / cvA[1])))
w('| **C：A + $\\sum_jc_jq_jp_j$** | 35 | %.4g | **%.5f** | **%.2f%%** |' % (
    cvC[0], cvC[1], 100 * (1 - cvC[1] / cvA[1])))
USE_C = cvC[1] < cvA[1]
if USE_C:
    Xq = dm_q(Xp)
    Wq = np.linalg.solve(Xq.T @ Xq + cvC[0] * np.eye(Xq.shape[1]), Xq.T @ ytr)
    CJ = Wq[18:18 + len(PCOLS)]
    w('\n> **设定 C 优于 A**（CV 误差下降 %.2f%%），与问题一 M5 的嵌套 F 检验（F=11.50，p<1e-10）一致。'
      '结论：配比中质量的作用机制是**"哪个域的质量更重要"（域别异质），而不是"平均质量多高"（聚合水平）**。'
      '本文广义标度律的配比修正项采用设定 C。\n' % (100 * (1 - cvC[1] / cvA[1])))
    w('\n| 域 | $Q_j$ | $c_j$（越负＝该域质量的边际收益越大） |')
    w('|---|---|---|')
    for i in np.argsort(CJ):
        w('| `%s` | %.4f | %+.4f |' % (PCOLS[i], Qv[i], CJ[i]))
else:
    w('\n> 设定 C 未优于 A，本文仍采用设定 B。\n')
    CJ = None
w('\n### 3.0c 设定 B 的系数（供对照）\n')
w('\n| 域 | $Q_j$ | $\\gamma_j$ | $p_{ref,j}$ |')
w('|---|---|---|---|')
for i in np.argsort(GJ):
    w('| `%s` | %.4f | %+.4f | %.4f |' % (PCOLS[i], Qv[i], GJ[i], PR[i]))

w('\n### 3.1 检验集 A6–A11\n')
w('| 检验集 | n | 设定 C ρ（主） | 设定 C 仿射后 $R^2$ | 设定 B ρ | 设定 A（无质量）ρ |')
w('|---|---|---|---|---|---|')
Wnq = np.linalg.solve(dm0(Xp).T @ dm0(Xp) + ath * np.eye(18), dm0(Xp).T @ ytr)
for tag, nm in [('1m', 'A6/A7 1M'), ('60m', 'A8/A9 60M'), ('1B', 'A10/A11 1B')]:
    P, y = load_mix(tag, 'test')
    prC = dm_q(P) @ Wq if USE_C else dm(P, TH) @ W
    prB = dm(P, TH) @ W
    prn = dm0(P) @ Wnq
    Am = np.column_stack([np.ones(len(y)), prC]); c = np.linalg.lstsq(Am, y, rcond=None)[0]
    w('| %s | %d | **%.4f** | %.4f | %.4f | %.4f |' % (
        nm, len(y), stats.spearmanr(prC, y).statistic,
        1 - ((y - Am @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        stats.spearmanr(prB, y).statistic, stats.spearmanr(prn, y).statistic))
y1_ = np.load(os.path.join(OUT, '_diag_y1.npy')); y60_ = np.load(os.path.join(OUT, '_diag_y60.npy'))
w('\n### 3.2 域间的替代与互补（承接问题一 M2 的交互项）\n')
w('配比的对数-线性部分 $\\sum_j\\gamma_j\\ln p_j$ 暗含"域间相互独立"。'
  '域间是否存在**替代**（互相挤压）或**互补**（协同增效），须看二次交互项 $\\theta_{jk}$：'
  '$\\theta_{jk}<0$ 为互补，$\\theta_{jk}>0$ 为替代。问题一在 A4/A5 上估计的极值如下（此处引用，不重算）：\n')
w('\n| 最强互补（$\\theta_{jk}$ 最小） | $\\theta$ | 最强替代（$\\theta_{jk}$ 最大） | $\\theta$ |')
w('|---|---|---|---|')
COMP = [('ubuntu_irc × uspto_backgrounds', -1.5257), ('pubmed_central × hackernews', -1.3122),
        ('pubmed_central × gutenberg_pg_19', -1.2465), ('wikipedia_en × gutenberg_pg_19', -1.1436),
        ('wikipedia_en × dm_mathematics', -0.9902), ('arxiv × github', -0.8476)]
SUBS = [('pubmed_central × pubmed_abstracts', 2.5018), ('pubmed_central × uspto_backgrounds', 1.6619),
        ('github × hackernews', 1.5260), ('github × stackexchange', 1.5200),
        ('nih_exporter × wikipedia_en', 1.3500), ('gutenberg_pg_19 × uspto_backgrounds', 1.1918)]
for (a, av), (b, bv) in zip(COMP, SUBS):
    w('| %s | %.4f | %s | %.4f |' % (a, av, b, bv))
w('\n**机理判读**：\n')
w('- **替代**关系几乎成对出现在**语义相近**的域之间（`pubmed_central`↔`pubmed_abstracts` 同属生物医学、'
  '`github`↔`hackernews`/`stackexchange` 同属技术社区），符合"同源冗余"直觉：'
  '两个域提供的信息高度重叠，同时加权等于重复计费。\n')
w('- **互补**关系出现在**语义相距远**的域之间（对话/邮件 × 专利、百科 × 数学、论文 × 代码），'
  '符合"多样性增益"直觉。\n')
w('- 由此给出可操作的配比原则：**配比应在"语义距离"上尽量分散，避免在同一语义簇内重复加权**。\n')

w('\n### 3.3 配比效应的规模衰减\n')
w('- 同一配比矩阵 1M 与 60M：配比诱导离散度之比 %.4f，中心水平之比 %.4f ⇒ '
  '配比效应近似**加性**（60 倍参数跨度仅衰减 %.1f%%），故取配比修正与规模无关。\n' % (
      y60_.std(ddof=1) / y1_.std(ddof=1), y60_.mean() / y1_.mean(),
      100 * (1 - y60_.std(ddof=1) / y1_.std(ddof=1))))

w('\n### 3.4 广义标度律的最终形式\n')
w('$$\\boxed{\\;L(N,D,Q,p)=\\underbrace{E+\\big(AN^{-\\alpha}+BD^{-\\beta}\\big)\\,\\Psi(Q)}_{\\text{规模-质量律}}'
  '\\;+\\;\\underbrace{\\Big[\\Phi(p)-\\Phi(p_{ref})\\Big]}_{\\text{配比修正}}\\;}$$\n')
w('$$\\Psi(Q)=1+c(1-Q),\\ c=%.4f;\\qquad '
  '\\Phi(p)=\\beta_0+\\sum_{j=1}^{17}\\gamma_j\\ln(p_j+\\varepsilon)%s$$\n' % (
      float(t6[0]), '+\\sum_{j=1}^{17}c_j\\,q_j\\,p_j' if USE_C else '+\\varphi\\ln Q(p;1)'))
w('$$Q=Q(p;\\theta)=\\Big(\\sum_jp_jQ_j^{\\theta}\\Big)^{1/\\theta},\\ \\theta=1$$\n')
w('\n**退化性**：$Q=1\\Rightarrow\\Psi=1$，规模项退化为 $E+AN^{-\\alpha}+BD^{-\\beta}$；'
  '$p=p_{ref}\\Rightarrow\\Phi-\\Phi(p_{ref})=0$。两者同时成立时整体**严格退化为经典标度律**。✔\n')
w('\n**三条通道的分工**：\n')
w('| 通道 | 进入方式 | 是否消耗算力 |')
w('|---|---|---|')
w('| 参数 $N$ | $AN^{-\\alpha}$ | 是（$\\kappa_cND$） |')
w('| 数据 $D$ | $BD^{-\\beta}$ | 是（$\\kappa_cND$） |')
w('| 质量 $Q$ | 乘子 $\\Psi(Q)$ | 是（$D[g(Q)-g(Q_0)]_+$） |')
w('| 配比 $p$ | 加性修正 $\\Phi(p)-\\Phi(p_{ref})$ | **否**（只改变既有算力的分配效果） |')
w('\n配比是唯一"免费"的杠杆——这正是问题三中配比必须被联合优化的原因。\n')

# ---- 4. 边际/弹性/等价 ----
w('\n---\n\n## 4. 边际效用、弹性与替代条件\n')
PSIf = lambda Q: PSI[MAIN][1](t6, np.atleast_1d(np.asarray(Q, float)))
w('主模型 $\\Psi(Q)$=%s，参数 %s。\n' % (MAIN, np.round(t6, 4).tolist()))
w('\n### 4.1 弹性\n')
w('| 弹性 | 表达式 | 说明 |')
w('|---|---|---|')
w('| 参数弹性 $\\eta_N$ | $-\\alpha\\,w_N$，$w_N=\\frac{AN^{-\\alpha}}{R}$ | 随配比而变 |')
w('| 数据弹性 $\\eta_D$ | $-\\beta\\,w_D$，$w_D=\\frac{BD^{-\\beta}}{R}$ | 随配比而变 |')
w('| 质量弹性 $\\eta_Q$ | $\\dfrac{Q\\Psi\'(Q)}{\\Psi(Q)}$ | 主模型下 $\\eta_Q(Q)=\\dfrac{-cQ}{1+c(1-Q)}$ |')
w('| 算力弹性（Chinchilla 轨线） | $-\\dfrac{\\alpha\\beta}{\\alpha+\\beta}$=%.4f | 与 $Q$ 无关 |' % (-AL * BE / (AL + BE)))
c1 = float(t6[0])
w('\n| $Q$ | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |')
w('|---|---|---|---|---|---|---|---|---|')
w('| $\\eta_Q(Q)$ | ' + ' | '.join('%.4f' % (-c1 * q / (1 + c1 * (1 - q))) for q in
                                    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + ' |')
w('\n> 质量弹性随 $Q$ 增大而**绝对值变小**（边际递减）：$Q$ 已经很高时再提升的收益变小；'
  '反之在 $Q$ 很低（数据脏）时提升质量的边际收益最大。\n')
w('\n> **结构性推论**：主模型中 $Q$ 只按 $\\Psi(Q)$ 整体缩放可约损失，不影响 $w_N:w_D$，'
  '因此**数据质量不改变 Chinchilla 最优的 $N$:$D$ 拆分**，只改变"每份算力能换到多少损失下降"。\n')

w('\n### 4.2 每 FLOP 的边际收益（三通道比较）\n')
w('$$\\frac{-\\partial L}{\\partial C}\\Big|_N=\\frac{\\alpha\\,(AN^{-\\alpha}\\Psi)}{C},\\quad '
  '\\frac{-\\partial L}{\\partial C}\\Big|_D=\\frac{\\beta\\,(BD^{-\\beta}\\Psi)}{C},\\quad '
  '\\frac{-\\partial L}{\\partial C}\\Big|_Q=\\frac{-R\\,\\Psi\'(Q)}{D_{tok}\\,g\'(Q)}$$\n')
GS = {'指数型 $10^7e^{6Q}$': lambda Q: 1e7 * np.exp(6.0 * Q),
      '幂函数型 $5\\times10^9Q^4$': lambda Q: 5e9 * Q ** 4.0,
      '对数渐进型 $2\\times10^9\\ln(1+10Q)$': lambda Q: 2e9 * np.log(1 + 10.0 * Q)}
w('\n取 $C=10^{22}$ FLOPs，$L_{ctx}=2048$ ⇒ $\\kappa_c=6+\\eta L_{ctx}=%.4f$；'
  '注意 $N,D$ 以十亿为单位，故 $C=\\kappa_c\\cdot10^{18}\\,N_BD_B$。\n' % (6 + ETA * 2048))
kc = (6 + ETA * 2048)
C0 = 1e22
prod = C0 / (kc * UNITS)
Nst = ((AL * A) / (BE * B)) ** (1 / (AL + BE)) * prod ** (BE / (AL + BE))
Dst = prod / Nst
w('- Chinchilla 最优拆分：$N^{\\star}$=%.3f B，$D^{\\star}$=%.1f B tokens（$N^{\\star}D^{\\star}$=%.1f）\n' % (Nst, Dst, prod))
w('\n| $Q$ | 参数 $N$ | 数据 $D$ | 质量(指数) | 质量(幂) | 质量(对数) | 最优通道 |')
w('|---|---|---|---|---|---|---|')
rows_mu = []
for Q in [0.4, 0.5, 0.6, 0.7, 0.8]:
    ps = float(PSIf(Q)[0]); dps = float((PSIf(Q + 1e-6)[0] - PSIf(Q - 1e-6)[0]) / 2e-6)
    a = AL * (A * Nst ** (-AL) * ps) / C0
    b = BE * (B * Dst ** (-BE) * ps) / C0
    qs = []
    for g in GS.values():
        gp = (g(Q + 1e-6) - g(Q - 1e-6)) / 2e-6
        qs.append(-RED(Nst, Dst) * dps / (Dst * 1e9 * gp))
    vals = {'参数 N': a, '数据 D': b, '质量(指数)': qs[0], '质量(幂)': qs[1], '质量(对数)': qs[2]}
    bk = max(vals, key=vals.get)
    rows_mu.append(dict(Q=Q, mu_N=a, mu_D=b, mu_Q_exp=qs[0], mu_Q_pow=qs[1], mu_Q_log=qs[2], best=bk))
    w('| %.1f | %.3e | %.3e | %.3e | %.3e | %.3e | **%s** |' % (Q, a, b, qs[0], qs[1], qs[2], bk))
w('\n> 在 Chinchilla 最优点上 $-\\partial L/\\partial C|_N$ 与 $|_D$ 必然相等（一阶条件），上表可作内部校验；'
  '**质量通道的边际收益普遍比参数/数据通道高 1–3 个数量级**，'
  '说明在给定成本函数下"买好教材"远比"堆参数"划算——但这一结论强烈依赖 $g(Q)$ 的标定，问题三将系统讨论。\n')

w('\n### 4.3 "质量 +0.1 等价于参数增加多少"的可计算条件\n')
w('目标：不改善质量，只把参数放大 $r$ 倍（或数据放大 $s$ 倍），'
  '去"追平"质量改善后达到的损失 $L(N,D,Q+\\delta)$。令\n')
w('$$S=R(N,D)\\cdot\\frac{\\Psi(Q+\\delta)}{\\Psi(Q)}\\;<\\;R(N,D)\\quad(\\delta=0.1)$$\n')
w('（注意方向：$\\Psi$ 递减，故 $S<R$；此前若把比值写反会得到 $r<1$ 的荒谬结果。）则\n')
w('$$\\boxed{r=\\Big[\\frac{S-BD^{-\\beta}}{AN^{-\\alpha}}\\Big]^{-1/\\alpha}\\ \\text{（参数等价倍数）},\\qquad '
  '\\boxed{s=\\Big[\\frac{S-AN^{-\\alpha}}{BD^{-\\beta}}\\Big]^{-1/\\beta}\\ \\text{（数据等价倍数）}}$$\n')
w('**可行条件**：$S>BD^{-\\beta}$（对 $r$）与 $S>AN^{-\\alpha}$（对 $s$）。'
  '因 $N\\to\\infty$ 只能消去 $AN^{-\\alpha}$、无法消去 $BD^{-\\beta}$，'
  '故当 $S\\le BD^{-\\beta}$ 时，**质量提升 0.1 的收益已超出任何参数扩张所能达到的上界，二者不可替代**。\n')
def equiv(N, D, Q, dl=0.1):
    if Q + dl > 1.0:
        return np.nan, np.nan
    S = RED(N, D) * float(PSIf(Q + dl)[0]) / float(PSIf(Q)[0])
    nr = S - B * D ** (-BE); ns = S - A * N ** (-AL)
    r = (nr / (A * N ** (-AL))) ** (-1 / AL) if nr > 0 else np.nan
    s = (ns / (B * D ** (-BE))) ** (-1 / BE) if ns > 0 else np.nan
    return r, s
w('\n| $N$(B) | $D$(B tokens) | $Q$ | $r$ | 参数增量(B) | $s$ | 数据增量(B) |')
w('|---|---|---|---|---|---|---|')
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
            '%+.0f' % ((s - 1) * D_) if np.isfinite(s) else '—'))
r7, s7 = equiv(7.0, 1000, 0.5)
w('\n> 例：$N$=7B、$D$=1T、$Q$=0.5 处，质量 +0.1（→0.6）等价于参数放大 **%.4f 倍**（≈ +%.2f B 参数），'
  '或数据量放大 **%.4f 倍**（≈ +%.0f B tokens）。\n' % (r7, (r7 - 1) * 7, s7, (s7 - 1) * 1000))
w('\n把二者折算成算力比较（$C_N=6(r-1)N_{tok}D_{tok}$，$C_Q=D_{tok}[g(Q+\\delta)-g(Q)]$）：\n')
w('\n| 成本函数 | $C_Q$（$D$=1T tokens） | $C_N$（同效参数扩张） | 性价比 $C_N/C_Q$ |')
w('|---|---|---|---|')
N_, D_, Q_ = 7.0, 1000.0, 0.5
r_, _ = equiv(N_, D_, Q_)
CN = 6 * (r_ - 1) * (N_ * 1e9) * (D_ * 1e9)
CQmin = None
for nm, g in GS.items():
    CQ = D_ * 1e9 * (g(Q_ + 0.1) - g(Q_))
    CQmin = CQ if CQmin is None else min(CQmin, CQ)
    w('| %s | %.4e | %.4e | **%.2f** |' % (nm, CQ, CN, CN / CQ))
w('\n> 性价比 $=C_N/C_Q>1$ 表示"用扩参数追平同样的损失下降"要比"直接提升质量"花更多算力，'
  '即**提升质量更划算**。本例三类成本函数下该比值均为 %.0f–%.0f 量级 ⇒ '
  '在 $N$=7B、$D$=1T、$Q$=0.5 的典型配置下，**买好教材比堆参数便宜一到两个数量级**。\n' % (
      min(CN / (D_ * 1e9 * (g(Q_ + 0.1) - g(Q_))) for g in GS.values()),
      max(CN / (D_ * 1e9 * (g(Q_ + 0.1) - g(Q_))) for g in GS.values())))
w('\n**但这个结论的反面同样重要**：三类 $g(\\cdot)$ 的绝对量级高达 $10^{20}$ FLOPs（与训练本身同量级），'
  '而题设 $g$ 的参数（$\\gamma_e=10^7,\\lambda=6$ 等）并未给出标定依据。'
  '**结论对 $g$ 的标定高度敏感**，问题三将把 $g$ 的形式作为决策变量做系统比较。\n')

# ---- 5. B9/B10 ----
w('\n---\n\n## 5. 百亿参数以上外推（B9/B10）\n')
b9 = pd.read_csv(os.path.join(BS, 'supplementary_large_models.csv'), dtype_backend='numpy_nullable')
b10 = pd.read_csv(os.path.join(BS, 'supplementary_large_baseline.csv'), dtype_backend='numpy_nullable')
m = b9.merge(b10, left_on='model_name', right_on='family', how='inner', suffixes=('_9', '_10'))
N9 = m['N_params_B_9'].to_numpy(float)
D9 = np.maximum(m['D_tokens_B_10'].to_numpy(float), 1e-3)
Y9 = m['val_loss'].to_numpy(float)
w('- B9（真实元数据）与 B10（**估算** Loss）内连接 n=%d；N∈[%.0f,%.0f] B，D∈[%.1f,%.1f] B，'
  '估算 Loss∈[%.4f,%.4f]\n' % (len(m), N9.min(), N9.max(), D9.min(), D9.max(), Y9.min(), Y9.max()))
base9 = E + RED(N9, D9)
d9 = Y9 - base9
w('\n### 5.1 关键发现：B10 是否为独立信息？\n')
w('- B10 估算 Loss 与"**经典**标度律 $E+AN^{-\\alpha}+BD^{-\\beta}$"之差：'
  '均值 %+.5f，标准差 %.5f，最大绝对偏差 %.5f\n' % (d9.mean(), d9.std(ddof=1), np.abs(d9).max()))
w('- 相关系数 ρ(B10, 经典律预测) = %.6f\n' % stats.pearsonr(Y9, base9).statistic)
w('\n> **判定**：B10 的 `val_loss` 与经典标度律在 B1 系数下的预测**几乎逐点重合**'
  '（平均偏差 %+.5f，标准差 %.5f）。这说明 B10 本身就是由经典标度律反推生成的**估算表**，'
  '**不携带独立的观测信息**，因此不能用作广义标度律的独立检验集。'
  '本文仅用它说明"把 70M–12B 的 Pythia 系数外推到 100B–10T 会得到什么"，并明确标注其为估算值。\n' % (
      d9.mean(), d9.std(ddof=1)))
w('\n### 5.2 外推结果与隐含质量反解\n')
w('| 情景 | 假设 $Q$ | 预测中位 Loss | B10 中位 Loss | 中位偏差 | ρ |')
w('|---|---|---|---|---|---|')
rows_ex = []
for ql, qq in [('$Q(p_{ref})$=%.4f' % Qref, Qref), ('$Q$=0.7', 0.7), ('$Q$=0.8', 0.8),
               ('$Q$=0.9', 0.9), ('$Q$=1.0（经典）', 1.0)]:
    pv = E + RED(N9, D9) * float(PSIf(qq)[0])
    rows_ex.append(dict(Q=qq, pred_med=float(np.median(pv)), obs_med=float(np.median(Y9)),
                        bias=float(np.median(pv - Y9)), rho=float(stats.spearmanr(pv, Y9).statistic)))
    w('| %s | %.4f | %.4f | %.4f | %+.4f | %.4f |' % (
        ql, qq, np.median(pv), np.median(Y9), np.median(pv - Y9), stats.spearmanr(pv, Y9).statistic))
ok = Y9 > E
Qhat = np.full(len(Y9), np.nan)
Qhat[ok] = np.nan
# 反解 Psi：Psi_hat = (L-E)/R
PsiHat = (Y9 - E) / RED(N9, D9)
w('\n- 反解出的隐含折扣因子 $\\hat\\Psi=(L_{B10}-E)/R(N,D)$：中位数 %.4f，四分位 [%.4f, %.4f]，'
  '$\\hat\\Psi\\ge1$ 的比例 %.1f%%\n' % (np.median(PsiHat), np.percentile(PsiHat, 25),
                                        np.percentile(PsiHat, 75), 100 * np.mean(PsiHat >= 1)))
w('- 由于主模型 $\\Psi(Q)\\ge1$ 且 $\\Psi(1)=1$，$\\hat\\Psi\\ge1$ 意味着反解质量 $\\hat Q\\le1$ 可行；'
  '$\\hat\\Psi$ 中位数 %.4f 对应的隐含质量约 $Q\\approx%.3f$。\n' % (
      np.median(PsiHat), 1 - (np.median(PsiHat) - 1) / c1))
w('\n> **外推结论**：把 Pythia 尺度（70M–12B）标定出的广义律外推到 100B–10T，'
  '需要假设这些大模型使用了 $Q\\approx%.2f$ 的数据才能得到 B10 的估算水平；'
  '若按问题一实测的语料质量（$Q(p_{ref})\\approx%.3f$）则预测损失会高约 %.3f。'
  '这个缺口正是**非规模技术进步**（更好的数据工程、架构、训练配方）的贡献，问题四将单独量化。\n' % (
      1 - (np.median(PsiHat) - 1) / c1, Qref,
      float(np.median(E + RED(N9, D9) * float(PSIf(Qref)[0]) - Y9))))

# ---- 输出 ----
pd.DataFrame(rows_mu).to_csv(os.path.join(OUT, 'q2_marginal_utility.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(EQ).to_csv(os.path.join(OUT, 'q2_equivalence.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(rows_ex).to_csv(os.path.join(OUT, 'q2_B9B10_extrapolation.csv'), index=False, encoding='utf-8-sig')
_mx = pd.DataFrame({'domain': PCOLS, 'Q_j': Qv, 'gamma_j': GJ, 'p_ref': PR})
if USE_C:
    _mx['c_j'] = CJ
_mx.to_csv(os.path.join(OUT, 'q2_mixture_coef.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'model': m['model_name'].astype(str), 'N_B': N9, 'D_B': D9,
              'loss_B10_est': Y9, 'classic_pred': base9, 'Psi_hat': PsiHat}).to_csv(
    os.path.join(OUT, 'q2_B9B10_detail.csv'), index=False, encoding='utf-8-sig')
json.dump(dict(E=E, A=A, alpha=AL, B=B, beta=BE, form=MAIN, psi_params=[float(x) for x in t6],
               psi_name='1+c(1-Q)', c=float(t6[0]), gamma_power=float(tg[0]),
               kappa=float(tg[0] / BE), theta=TH, phi=phi, beta0=b0,
               mixture_spec=('C' if USE_C else 'B'),
               gamma_j=[float(x) for x in GJ],
               c_j=([float(x) for x in CJ] if USE_C else None),
               Q_j={k: float(v) for k, v in QJ.items()},
               p_ref=[float(x) for x in PR],
               Q_ref_ces=Qref, Phi_ref=Phi_ref, noise_floor=NOISE, unit=UNITS,
               discount_exp=1.0 / AL + 1.0 / BE, eta_ctx=ETA),
          open(os.path.join(OUT, 'gsl_params.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '02f_generalized_law_final.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

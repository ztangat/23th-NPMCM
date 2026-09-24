# -*- coding: utf-8 -*-
"""问题二 阶段 A：可检验假设的结构诊断
 A. B6/B7/B8 中 Q 的作用方向与量级（偏相关 / 分组单调性）
 B. RegMix A4-A15 中"配比效应"随模型规模的变化：加性 vs 乘性
"""
import os, io, sys
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RA = r'C:\Users\dkyyt\Desktop\F题\real_attachments'
BS = os.path.join(RA, 'B_scaling_laws')
AV = os.path.join(RA, 'A_data_value', 'regmix_tables')
OUT = r'C:\Users\dkyyt\Desktop\F题\_out\02_q2'
os.makedirs(OUT, exist_ok=True)

L = []
def w(s=''):
    L.append(s)


def partial_corr(y, x, Z):
    """y 与 x 在控制 Z（含截距）后的偏相关"""
    Z1 = np.column_stack([np.ones(len(y)), Z])
    by = np.linalg.lstsq(Z1, y, rcond=None)[0]
    bx = np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ by
    rx = x - Z1 @ bx
    r = stats.pearsonr(ry, rx)
    # 偏回归系数与其 t 值（用 x 的残差做一元回归）
    beta = float(rx @ ry / (rx @ rx))
    dof = len(y) - Z1.shape[1] - 1
    se = np.sqrt(((ry - beta * rx) ** 2).sum() / dof / (rx @ rx))
    return beta, float(r.statistic), float(r.pvalue), beta / se


w('# 问题二 · 阶段 A：跨附件统一假设的结构诊断\n')
w('本阶段回答两个决定广义标度律形式的问题：\n')
w('1. **附件 B 的半合成 N-D-Q 数据中，Q 对 Loss 的作用方向是否为"越高越好"？量级多大？**\n')
w('2. **附件 A 的配比效应 Φ(p) 随模型规模是"加性"（绝对损失差不随规模衰减）还是"乘性"（比例保持）？**\n')

# ---------------------------------------------------------------- A. Q 的方向
w('\n---\n\n## A. 附件 B 中 Q 的作用方向诊断\n')
w('方法：对每套 N-D-Q 数据，以 $(\\ln N,\\ \\ln D)$ 为控制变量，求 $\\mathrm{val\\_loss}$ 对 $Q$ 的'
  '**偏回归系数** $\\partial L/\\partial Q$ 与偏相关系数。若 $Q$ 确为"质量"，则该系数应显著为负。\n')

FILES = {
    'B6': 'supplementary_NQ_experiment.csv',
    'B7': 'supplementary_NQ_experiment_expanded.csv',
    'B8': 'supplementary_NQ_experiment_large.csv',
}
raw = {}
w('| 数据集 | 口径 | n | $\\partial L/\\partial Q$（偏回归） | t 值 | 偏相关 $\\rho$ | p | 判定 |')
w('|---|---|---|---|---|---|---|---|')
for tag, fn in FILES.items():
    df = pd.read_csv(os.path.join(BS, fn), dtype_backend='numpy_nullable')
    raw[tag] = df
    sub = {'B6': [('全部', df)], 'B7': [('全部', df)]}.get(tag)
    if sub is None:
        sub = [('全部', df)] + [('%s 子集' % k, df[df['data_type'] == k])
                                for k in df['data_type'].dropna().unique()]
    for nm, d in sub:
        if len(d) < 30:
            continue
        y = d['val_loss'].to_numpy(float)
        q = d['Q_score'].to_numpy(float)
        Z = np.column_stack([np.log(d['N_params_B'].to_numpy(float)),
                             np.log(d['D_tokens_B'].to_numpy(float))])
        b, r, p, t = partial_corr(y, q, Z)
        judge = '**符合质量语义（负）**' if b < 0 else '⚠ 方向反转（正）——与质量语义矛盾'
        w('| %s | %s | %d | %+.4f | %.2f | %+.4f | %.3g | %s |' % (tag, nm, len(d), b, t, r, p, judge))

w('\n### A.1 直接分组证据：固定 N、D 时 Loss 随 Q 的变化\n')
for tag in ['B6', 'B7', 'B8']:
    d = raw[tag]
    g = d.groupby(['N_params_B', 'D_tokens_B'])
    rows = []
    for (nn, dd), gg in g:
        if gg['Q_score'].nunique() >= 4:
            gg = gg.sort_values('Q_score')
            rho = stats.spearmanr(gg['Q_score'], gg['val_loss']).statistic
            rows.append((nn, dd, len(gg), gg['Q_score'].iloc[0], gg['val_loss'].iloc[0],
                         gg['Q_score'].iloc[-1], gg['val_loss'].iloc[-1], rho))
    rows = pd.DataFrame(rows, columns=['N_B', 'D_B', 'nQ', 'Q_min', 'L@Qmin', 'Q_max', 'L@Qmax', 'rho'])
    w('\n**%s**：%d 个 (N,D) 单元具备 ≥4 个 Q 水平；其中 ρ(Q,L)<0（质量越高损失越低）的占比 **%.1f%%**，'
      'ρ 中位数 **%+.3f**。\n' % (tag, len(rows), 100 * (rows['rho'] < 0).mean(), rows['rho'].median()))
    w('| N(B) | D(B) | Q 水平数 | Q_min | L(Q_min) | Q_max | L(Q_max) | ρ(Q,L) |')
    w('|---|---|---|---|---|---|---|---|')
    for _, r in rows.head(12).iterrows():
        w('| %g | %g | %d | %.2f | %.4f | %.2f | %.4f | %+.3f |' % (
            r['N_B'], r['D_B'], r['nQ'], r['Q_min'], r['L@Qmin'], r['Q_max'], r['L@Qmax'], r['rho']))

w('\n### A.2 B8 的可用性判定\n')
b8 = raw['B8']
w('- B8 中 `val_loss` 的下界被截断在 **%.2f**（共 %d 行 ≤0.6），而同等规模的真实模型 Loss 下界约为 1.7（见 B10）；'
  '一个 70M 参数模型在 10B token 上不可能取得 Loss<1。\n' % (
      b8['val_loss'].min(), int((b8['val_loss'] <= 0.6).sum())))
w('- B8 中 N=0.16B、D=500B 处：Q=0.05→L=%.4f，Q=0.10→L=%.4f，Q=0.20→L=%.4f，'
  '**Loss 随 Q 上升而上升**，与 B6/B7 相反。\n' % (
      b8[(b8.N_params_B == 0.16) & (b8.D_tokens_B == 500) & (b8.Q_score == 0.05)]['val_loss'].iloc[0],
      b8[(b8.N_params_B == 0.16) & (b8.D_tokens_B == 500) & (b8.Q_score == 0.1)]['val_loss'].iloc[0],
      b8[(b8.N_params_B == 0.16) & (b8.D_tokens_B == 500) & (b8.Q_score == 0.2)]['val_loss'].iloc[0]))
w('\n⇒ **结论**：B8（`supplementary_NQ_experiment_large`，含 `extrapolated` 子集）在 Q 维度上与 B6/B7 '
  '**符号相反且存在非物理的数值下界**，属"含外推"的合成外推表，'
  '按《数据说明》"半合成数据不得表述为直接实验观测"的要求，**本文不将其用于 κ 的点估计**，'
  '仅作为"外推失效"的警示样本在稳健性一节报告。κ 的估计以 B6 为主、B7 复核。\n')

# ---------------------------------------------------------------- B. 配比效应的规模依赖
w('\n---\n\n## B. 配比效应 Φ(p) 随模型规模的变化：加性还是乘性？\n')

def rd(fn):
    return pd.read_csv(os.path.join(AV, fn), dtype_backend='numpy_nullable')

mx1, ls1 = rd('test_mixture_1m.csv'), rd('test_pile_loss_1m.csv')
mx60, ls60 = rd('test_mixture_60m.csv'), rd('test_pile_loss_60m.csv')
mx1B, ls1B = rd('test_mixture_1B.csv'), rd('test_pile_loss_1B.csv')

def domcols(df):
    return [c for c in df.columns if c not in ('index', 'Unnamed: 0')]

P1 = domcols(mx1)
w('- 配比列（%d 个域）：%s\n' % (len(P1), ', '.join('`%s`' % c for c in P1)))
w('- 三个检验集规模：test_1m %d 行、test_60m %d 行、test_1B %d 行\n' % (lsx := len(ls1), len(ls60), len(ls1B)))
same60 = np.allclose(mx1[P1].to_numpy(float), mx60[P1].to_numpy(float))
w('- test_mixture_1m 与 test_mixture_60m 配比矩阵是否相同：**%s**\n' % same60)

def lossvec(ls):
    c = [x for x in ls.columns if x not in ('index', 'Unnamed: 0')]
    return ls[c].to_numpy(float).ravel()

def align(mx, ls):
    key = 'index' if 'index' in mx.columns else mx.columns[0]
    key2 = 'index' if 'index' in ls.columns else ls.columns[0]
    m = mx.merge(ls, left_on=key, right_on=key2, how='inner', suffixes=('_p', '_l'))
    lc = [x for x in ls.columns if x != key2]
    return m, m[lc].to_numpy(float).ravel()

m1, y1 = align(mx1, ls1)
m60, y60 = align(mx60, ls60)
m1B, y1B = align(mx1B, ls1B)

w('\n| 检验集 | n | 均值 | 标准差(配比带来的离散) | 极差 | 最小 | 最大 |')
w('|---|---|---|---|---|---|---|')
for nm, y in [('1M', y1), ('60M', y60), ('1B', y1B)]:
    w('| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
        nm, len(y), y.mean(), y.std(ddof=1), y.max() - y.min(), y.min(), y.max()))

sd1, sd60, sd1B = y1.std(ddof=1), y60.std(ddof=1), y1B.std(ddof=1)
w('\n- 离散度之比 σ(60M)/σ(1M) = %.4f，σ(1B)/σ(1M) = %.4f\n' % (sd60 / sd1, sd1B / sd1))
w('- 中心水平之比 μ(60M)/μ(1M) = %.4f，μ(1B)/μ(1M) = %.4f\n' % (y60.mean() / y1.mean(), y1B.mean() / y1.mean()))
w('\n**判读**：若配比效应为**乘性**（$L(p)=\\text{base}(N,D)\\cdot m(p)$），则离散度应与中心水平同比例缩放，'
  '即 σ 比 ≈ μ 比；若为**加性**（$L(p)=\\text{base}(N,D)+\\Phi(p)$），则 σ 比应 ≈ 1。\n')

# 更进一步：回归 L_60M = a + b L_1M，加性模型预测 b≈1；乘性模型预测 b≈μ60/μ1
res = []
if len(y1) == len(y60):
    b_a = stats.linregress(y1, y60)
    res.append(('60M ~ 1M', len(y1), b_a.slope, b_a.intercept, b_a.rvalue))
if len(y1) == len(y1B):
    b_b = stats.linregress(y1, y1B)
    res.append(('1B ~ 1M', len(y1), b_b.slope, b_b.intercept, b_b.rvalue))
if res:
    w('\n| 回归 | n | 斜率 b | 截距 a | r | 加性预测 b | 乘性预测 b |')
    w('|---|---|---|---|---|---|---|')
    for nm, n, sl, ic, rv in res:
        mu_ratio = {'60M ~ 1M': y60.mean() / y1.mean(), '1B ~ 1M': y1B.mean() / y1.mean()}[nm]
        w('| %s | %d | %.4f | %.4f | %.4f | 1.000 | %.4f |' % (nm, n, sl, ic, rv, mu_ratio))

w('\n### B.1 分离"水平"与"离散"：用变异系数比较\n')
w('| 检验集 | 均值 μ | 标准差 σ | 变异系数 CV=σ/μ |')
w('|---|---|---|---|')
for nm, y in [('1M', y1), ('60M', y60), ('1B', y1B)]:
    w('| %s | %.4f | %.4f | %.4f |' % (nm, y.mean(), y.std(ddof=1), y.std(ddof=1) / y.mean()))
w('\nCV 若随规模**明显下降**，说明配比效应更接近加性（绝对损失差恒定）；'
  '若 CV **基本不变**，说明接近乘性。\n')

np.save(os.path.join(OUT, '_diag_y1.npy'), y1)
np.save(os.path.join(OUT, '_diag_y60.npy'), y60)
np.save(os.path.join(OUT, '_diag_y1B.npy'), y1B)

with open(os.path.join(OUT, '02c_structure_diagnostics.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

# -*- coding: utf-8 -*-
"""问题二 阶段 B0：质量项函数形式的非参数结构识别

关键问题：质量 Q 的效应究竟挂在
  (a) 数据通道      L = E + A N^-a + B D^-b Q^-g          -> Δ 与 N 无关
  (b) 参数通道      L = E + A N^-a Q^-d + B D^-b          -> Δ 与 D 无关
  (c) 整个可约项    L = E + (A N^-a + B D^-b) Q^-g        -> Δ ∝ (A N^-a + B D^-b)
  (d) 双通道        L = E + A N^-a Q^-d + B D^-b Q^-g
  (e) 纯加性        L = E + A N^-a + B D^-b + c(1-Q)      -> Δ 与 N,D 均无关

判据：在固定 (N,D) 的单元里，比较 L(Q)-L(1) 随 N、D 的变化。
"""
import os, io, sys
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BS = r'C:\Users\dkyyt\Desktop\F题\real_attachments\B_scaling_laws'
OUT = r'C:\Users\dkyyt\Desktop\F题\_out\02_q2'
L = []
def w(s=''): L.append(s)

b6 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment.csv'), dtype_backend='numpy_nullable')
b7 = pd.read_csv(os.path.join(BS, 'supplementary_NQ_experiment_expanded.csv'), dtype_backend='numpy_nullable')

# B1 经典参数（阶段 B 已估）
E, A, AL, B, BE = 1.689798, 0.353980, 0.339977, 1.240306, 0.279878

w('# 问题二 · 阶段 B0：质量项函数形式的非参数识别\n')
w('\n固定 $E,A,\\alpha,B,\\beta$ 为 **B1 真实 Pythia 轨迹的估计值**（$E$=%.4f, $A$=%.4f, '
  '$\\alpha$=%.4f, $B$=%.4f, $\\beta$=%.4f），因为它们已在 B6 的 $Q=1$ 子集上通过一致性检验'
  '（残差均值 +0.0036，Welch p=0.667）。这样质量效应 $\\Delta$ 可被无混淆地识别。\n' % (E, A, AL, B, BE))

def prep(df):
    d = df.copy()
    d['Lbase'] = E + A * d['N_params_B'].astype(float) ** (-AL) + B * d['D_tokens_B'].astype(float) ** (-BE)
    return d

for tag, df in [('B6', b6), ('B7', b7)]:
    d = prep(df)
    w('\n---\n\n## %s（n=%d）\n' % (tag, len(d)))
    # 每个 (N,D) 单元中 L(Q=1) 与 Lbase 的差
    q1 = d[np.isclose(d['Q_score'].astype(float), 1.0)]
    w('\n### 1. 基准面核对：$Q=1$ 时 B6/B7 是否落在 B1 曲面上\n')
    r0 = q1['val_loss'].astype(float) - q1['Lbase']
    w('- $Q=1$ 子集 n=%d，残差 均值=%+.5f，标准差=%.5f，RMSE=%.5f\n' % (
        len(q1), r0.mean(), r0.std(ddof=1), np.sqrt((r0 ** 2).mean())))

    # Δ(Q) := L(Q) - L(1) 在同一 (N,D) 单元内
    w('\n### 2. 质量效应 $\\Delta(N,D;Q)=L(Q)-L(1)$ 的结构\n')
    piv = d.pivot_table(index=['N_params_B', 'D_tokens_B'], columns='Q_score', values='val_loss')
    piv = piv.dropna()
    L1 = piv[1.0]
    DL = piv.sub(L1, axis=0)
    w('| N(B) | D(B) | ' + ' | '.join('Q=%.2f' % c for c in DL.columns if c < 1.0) + ' |')
    w('|---|---|' + '---|' * (len([c for c in DL.columns if c < 1.0])))
    for (nn, dd), row in DL.iterrows():
        w('| %g | %g | ' % (nn, dd) + ' | '.join('%+.4f' % row[c] for c in DL.columns if c < 1.0) + ' |')

    # 关键回归：ln(-Δ) ~ ln N + ln D   （取 Q=0.1 列；若无则取最小 Q）
    qmin = min([c for c in DL.columns if c < 1.0])
    sub = pd.DataFrame({'N': [float(i[0]) for i in DL.index], 'D': [float(i[1]) for i in DL.index],
                        'Dl': DL[qmin].to_numpy(float)})
    sub = sub[sub['Dl'] > 1e-6]
    X = np.column_stack([np.ones(len(sub)), np.log(sub['N'].to_numpy(float)),
                         np.log(sub['D'].to_numpy(float))])
    yv = np.log(sub['Dl'])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = len(sub) - 3
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    w('\n### 3. 决定性回归：$\\ln\\Delta \\;=\\; c_0+c_N\\ln N+c_D\\ln D$（取 $Q=%.2f$，n=%d）\n' % (qmin, len(sub)))
    w('| 系数 | 估计 | 标准误 | t | p | 形式含义 |')
    w('|---|---|---|---|---|---|')
    for nm, b_, s_, lab in [('$c_N$（$\\ln N$）', beta[1], se[1], '若 ≈0：质量不挂在参数通道'),
                            ('$c_D$（$\\ln D$）', beta[2], se[2], '若 ≈0：质量不挂在数据通道')]:
        t = b_ / s_
        w('| %s | %+.4f | %.4f | %.2f | %.3g | %s |' % (nm, b_, s_, t, 2 * (1 - stats.t.cdf(abs(t), dof)), lab))
    w('\n$R^2$=%.4f\n' % (1 - (resid @ resid) / ((yv - yv.mean()) ** 2).sum()))
    w('\n**判读**：\n')
    w('- 若 $c_N\\approx 0$ 且 $c_D<0$ ⇒ 质量只作用于**数据通道**（形式 a）；\n')
    w('- 若 $c_N<0$ 且 $c_D<0$ 且二者量级相当 ⇒ 质量作用于**整个可约损失**（形式 c）；\n')
    w('- 若二者均 ≈0 ⇒ 纯加性（形式 e）。\n')

    # 与理论预测对照
    w('\n### 4. 与"整个可约项"形式 (c) 的解析对照\n')
    sub['pred_c'] = (A * sub['N'].to_numpy(float) ** (-AL) + B * sub['D'].to_numpy(float) ** (-BE))
    w('- 形式 (c) 预测 $c_N$、$c_D$ 应分别约等于 $-\\alpha\\cdot\\frac{AN^{-\\alpha}}{AN^{-\\alpha}+BD^{-\\beta}}$ '
      '与 $-\\beta\\cdot\\frac{BD^{-\\beta}}{AN^{-\\alpha}+BD^{-\\beta}}$ 的样本均值：\n')
    pN = -AL * (A * sub['N'].to_numpy(float) ** (-AL)) / sub['pred_c']
    pD = -BE * (B * sub['D'].to_numpy(float) ** (-BE)) / sub['pred_c']
    w('  预测 $c_N\\approx%.4f$（实测 %.4f），预测 $c_D\\approx%.4f$（实测 %.4f）\n' % (
        pN.mean(), beta[1], pD.mean(), beta[2]))
    w('- 形式 (a) 预测 $c_N=0,\\;c_D=-\\beta=%.4f$；形式 (e) 预测 $c_N=c_D=0$。\n' % (-BE))

with open(os.path.join(OUT, '02e_form_identification.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

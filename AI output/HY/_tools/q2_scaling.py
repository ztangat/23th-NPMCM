# -*- coding: utf-8 -*-
"""
问题二 · 步骤1：经典标度律拟合（B1）+ 族外/插值验证（B2/B3）+ 跨族/文献验证（B4/B5）
      步骤2：可检验假设——质量是否等价于"有效数据量缩放" D_eff = D·Q^κ（B6–B8）
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

BASE = r'C:\Users\dkyyt\Desktop\F题'
RB = os.path.join(BASE, 'real_attachments', 'B_scaling_laws')
OUT = os.path.join(BASE, '_out', '02_q2')
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, '02a_classic_scaling.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

w('# 问题二 · 经典标度律拟合与多源验证\n')

# ---------------- 数据 ----------------
B1 = pd.read_csv(os.path.join(RB, 'pythia_training_log_existing.csv'))
B2 = pd.read_csv(os.path.join(RB, 'cerebras_training_log.csv'))
B4 = pd.read_csv(os.path.join(RB, 'scaling_baseline.csv'))
B5 = pd.read_csv(os.path.join(RB, 'published_scaling_data.csv'))
B6 = pd.read_csv(os.path.join(RB, 'supplementary_NQ_experiment.csv'))
B7 = pd.read_csv(os.path.join(RB, 'supplementary_NQ_experiment_expanded.csv'))
B8 = pd.read_csv(os.path.join(RB, 'supplementary_NQ_experiment_large.csv'))
B9 = pd.read_csv(os.path.join(RB, 'supplementary_large_models.csv'))
B10 = pd.read_csv(os.path.join(RB, 'supplementary_large_baseline.csv'))
traj_dir = os.path.join(RB, 'training_trajectories')
TR = {}
for f in sorted(os.listdir(traj_dir)):
    TR[f] = pd.read_csv(os.path.join(traj_dir, f))

w('\n## 0. 数据概览\n')
w('| 文件 | 编号 | n | N 范围(B) | D 范围(B) | val_loss 范围 | 备注 |')
w('|---|---|---|---|---|---|---|')
for nm, df, note in [('pythia_training_log_existing', B1, '真实训练轨迹（主拟合数据）'),
                     ('cerebras_training_log', B2, '半合成（Pythia 标度律校准+噪声）'),
                     ('scaling_baseline', B4, '12 个模型族收敛点'),
                     ('published_scaling_data', B5, '公开文献'),
                     ('supplementary_NQ_experiment', B6, '半合成 N-D-Q'),
                     ('supplementary_NQ_experiment_expanded', B7, '半合成 N-D-Q 扩展'),
                     ('supplementary_NQ_experiment_large', B8, '半合成 N-D-Q 大规模（含外推）'),
                     ('supplementary_large_models', B9, 'Epoch AI 大模型元数据'),
                     ('supplementary_large_baseline', B10, '大模型预估 Loss')]:
    if 'val_loss' in df.columns:
        nr = '%g – %g' % (df['N_params_B'].min(), df['N_params_B'].max())
        dr = '%g – %g' % (df['D_tokens_B'].min(), df['D_tokens_B'].max())
        lr = '%.4f – %.4f' % (df['val_loss'].min(), df['val_loss'].max())
    else:
        nr = '%g – %g' % (df['N_params_B'].min(), df['N_params_B'].max())
        dr = '-' if 'D_tokens_B' not in df.columns else '%g – %g' % (df['D_tokens_B'].min(), df['D_tokens_B'].max())
        lr = '-'
    w('| `%s` | %s | %d | %s | %s | %s | %s |' % (nm, nm[:3], len(df), nr, dr, lr, note))
w('\n- Pythia 轨迹文件（B3）：%d 个，各 500 行插值点。\n' % len(TR))
w('- B1 的 run：%s\n' % ', '.join(sorted(B1['run_id'].astype(str).unique())))

# ---------------- 经典标度律拟合 ----------------
def law(theta, N, D):
    E, A, al, B, be = theta
    return E + A * np.power(N, -al) + B * np.power(D, -be)

def fit_law(N, D, L, n_starts=40, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    lo = np.array([0.0, 1e-6, 1e-3, 1e-6, 1e-3])
    hi = np.array([3.0, 1e4, 3.0, 1e4, 3.0])
    for t in range(n_starts):
        x0 = np.array([
            rng.uniform(0.5, 2.0), rng.uniform(0.1, 50), rng.uniform(0.1, 0.8),
            rng.uniform(0.1, 50), rng.uniform(0.1, 0.8)])
        try:
            r = least_squares(lambda th: law(th, N, D) - L, x0, bounds=(lo, hi),
                              loss='soft_l1', f_scale=0.05, max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best

N1 = B1['N_params_B'].to_numpy(float); D1 = B1['D_tokens_B'].to_numpy(float)
L1 = B1['val_loss'].to_numpy(float)
w('\n## 1. 经典标度律拟合（B1，n=%d）\n' % len(B1))
w('$$L(N,D)=E+A\\,N^{-\\alpha}+B\\,D^{-\\beta}$$\n')
r1 = fit_law(N1, D1, L1)
E, A, al, B, be = r1.x
pred1 = law(r1.x, N1, D1)
res1 = L1 - pred1
ss = 1 - (res1 ** 2).sum() / ((L1 - L1.mean()) ** 2).sum()
# Huber 稳健拟合下的残差尺度
mad = np.median(np.abs(res1 - np.median(res1))) * 1.4826
w('| 参数 | 估计值 | 含义 |')
w('|---|---|---|')
w('| $E$ | %.6f | 不可约损失（数据本身的熵） |' % E)
w('| $A$ | %.6f | 参数规模项的系数 |' % A)
w('| $\\alpha$ | %.6f | 参数的标度指数 |' % al)
w('| $B$ | %.6f | 数据量项的系数 |' % B)
w('| $\\beta$ | %.6f | 数据量的标度指数 |' % be)
w('\n- 拟合优度：$R^2$=%.6f；残差 RMSE=%.6f；残差 MAD=%.6f；MAE=%.6f\n' % (
    ss, np.sqrt((res1 ** 2).mean()), mad, np.abs(res1).mean()))
w('- 残差与 N、D、ln N、ln D 的相关：ρ(res,N)=%.4f, ρ(res,D)=%.4f, ρ(res,lnN)=%.4f, ρ(res,lnD)=%.4f\n' % (
    stats.spearmanr(res1, N1).statistic, stats.spearmanr(res1, D1).statistic,
    stats.spearmanr(res1, np.log(N1)).statistic, stats.spearmanr(res1, np.log(D1)).statistic))

# 参数 bootstrap 不确定度
rng = np.random.default_rng(2026)
boot = []
for b in range(200):
    idx = rng.integers(0, len(N1), len(N1))
    try:
        rb = fit_law(N1[idx], D1[idx], L1[idx], n_starts=6, seed=b)
        boot.append(rb.x)
    except Exception:
        pass
boot = np.array(boot)
w('\n### 1.1 参数 Bootstrap 不确定度（200 次重抽样）\n')
w('| 参数 | 估计 | Bootstrap SE | 2.5% | 97.5% |')
w('|---|---|---|---|---|')
for i, nm2 in enumerate(['E', 'A', 'alpha', 'B', 'beta']):
    w('| $%s$ | %.6f | %.6f | %.6f | %.6f |' % (
        nm2, r1.x[i], boot[:, i].std(ddof=1), np.percentile(boot[:, i], 2.5), np.percentile(boot[:, i], 97.5)))
np.save(os.path.join(OUT, 'classic_boot.npy'), boot)
json.dump(dict(E=E, A=A, alpha=al, B=B, beta=be, R2=float(ss),
               rmse=float(np.sqrt((res1 ** 2).mean()))),
          open(os.path.join(OUT, 'classic_params.json'), 'w'), indent=1)

# 每 run 的拟合残差
w('\n### 1.2 各 run 的拟合残差（检验是否存在系统性偏差）\n')
w('| run_id | n | N(B) | D 终值(B) | 实测最终 val_loss | 预测 | 残差 | 该 run 平均残差 |')
w('|---|---|---|---|---|---|---|---|')
for rid, g in B1.groupby('run_id'):
    i_last = g['D_tokens_B'].idxmax()
    pl = law(r1.x, np.array([g.loc[i_last, 'N_params_B']]), np.array([g.loc[i_last, 'D_tokens_B']]))[0]
    rr = g['val_loss'].to_numpy() - law(r1.x, g['N_params_B'].to_numpy(float), g['D_tokens_B'].to_numpy(float))
    w('| %s | %d | %.4g | %.1f | %.4f | %.4f | %+.4f | %+.4f |' % (
        rid, len(g), g['N_params_B'].iloc[0], g['D_tokens_B'].max(),
        g.loc[i_last, 'val_loss'], pl, g.loc[i_last, 'val_loss'] - pl, rr.mean()))

# ---------------- 验证：B3 插值轨迹 ----------------
w('\n## 2. 验证一：B3 插值轨迹（族内插值）\n')
w('| 轨迹文件 | N(B) | 平均残差 | RMSE | $R^2$ | ρ(预测,实测) |')
w('|---|---|---|---|---|---|')
for f, df in TR.items():
    Nt = df['N_params_B'].to_numpy(float); Dt = df['D_tokens_B'].to_numpy(float)
    Lt = df['val_loss'].to_numpy(float)
    pt = law(r1.x, Nt, Dt)
    rr = Lt - pt
    r2 = 1 - (rr ** 2).sum() / ((Lt - Lt.mean()) ** 2).sum()
    w('| `%s` | %.6g | %+.4f | %.4f | %.4f | %.4f |' % (
        f, Nt[0], rr.mean(), np.sqrt((rr ** 2).mean()), r2, stats.spearmanr(Lt, pt).statistic))

# ---------------- 验证：B2 Cerebras 族外 ----------------
w('\n## 3. 验证二：B2 Cerebras-GPT（模型族外，半合成）\n')
N2 = B2['N_params_B'].to_numpy(float); D2 = B2['D_tokens_B'].to_numpy(float)
L2 = B2['val_loss'].to_numpy(float)
p2 = law(r1.x, N2, D2); r2v = L2 - p2
r22 = 1 - (r2v ** 2).sum() / ((L2 - L2.mean()) ** 2).sum()
w('- 直接用 B1 参数外推到 Cerebras：RMSE=%.4f，平均偏差=%+.4f，$R^2$=%.4f，ρ=%.4f\n' % (
    np.sqrt((r2v ** 2).mean()), r2v.mean(), r22, stats.spearmanr(L2, p2).statistic))
w('- Cerebras 自身 D 范围 %.0f–%.0f B（Pythia 仅 %.1f–%.1f B），远超 B1 的训练域，'
  '故绝对偏差偏大属预期。**关键看排序与量级是否合理**。\n' % (
      D2.min(), D2.max(), D1.min(), D1.max()))
# 只用截距重标定（1 参数）后再看
c = np.median(r2v)
w('- 仅做常数平移（$+%.4f$，模拟不同验证集口径差异）后：RMSE=%.4f，$R^2$=%.4f，ρ=%.4f\n' % (
    -c, np.sqrt(((L2 - p2 + c) ** 2).mean()),
    1 - ((L2 - p2 + c) ** 2).sum() / ((L2 - L2.mean()) ** 2).sum(),
    stats.spearmanr(L2, p2 - c).statistic))
# 在 B2 上单独拟合，比较指数
r2fit = fit_law(N2, D2, L2, n_starts=25, seed=3)
w('\n- 在 B2 上独立拟合的参数：$E$=%.4f, $A$=%.4f, $\\alpha$=%.4f, $B$=%.4f, $\\beta$=%.4f'
  '（与 B1 的 $\\alpha$=%.4f / $\\beta$=%.4f 对照）\n' % (
      r2fit.x[0], r2fit.x[1], r2fit.x[2], r2fit.x[3], r2fit.x[4], al, be))

# ---------------- 验证：B4 跨族 + B5 文献 ----------------
w('\n## 4. 验证三：B4 跨族收敛点 与 B5 文献数据\n')
w('### 4.1 B4（57 点，12 个模型族）\n')
N4 = B4['N_params_B'].to_numpy(float); D4 = B4['D_tokens_B'].to_numpy(float); L4 = B4['val_loss'].to_numpy(float)
p4 = law(r1.x, N4, D4); r4 = L4 - p4
w('- 直接外推：RMSE=%.4f，平均偏差=%+.4f，ρ=%.4f，$R^2$=%.4f\n' % (
    np.sqrt((r4 ** 2).mean()), r4.mean(), stats.spearmanr(L4, p4).statistic,
    1 - (r4 ** 2).sum() / ((L4 - L4.mean()) ** 2).sum()))
w('| 模型族 | n | N(B) 范围 | D(B) | 实测 Loss | 预测 | 残差 |')
w('|---|---|---|---|---|---|---|')
for fam, g in B4.groupby('family'):
    w('| %s | %d | %.3g–%.3g | %.0f | %.4f | %.4f | %+.4f |' % (
        fam, len(g), g['N_params_B'].min(), g['N_params_B'].max(), g['D_tokens_B'].median(),
        g['val_loss'].mean(), law(r1.x, g['N_params_B'].to_numpy(float),
                                  g['D_tokens_B'].to_numpy(float)).mean(),
        g['val_loss'].mean() - law(r1.x, g['N_params_B'].to_numpy(float),
                                   g['D_tokens_B'].to_numpy(float)).mean()))
# 仿射重标定
Amat = np.column_stack([np.ones(len(p4)), p4]); cf, *_ = np.linalg.lstsq(Amat, L4, rcond=None)
pr = Amat @ cf
w('\n- 2 参数仿射重标定后（$a$=%.4f, $b$=%.4f，吸收不同验证集/分词口径差异）：'
  'RMSE=%.4f，$R^2$=%.4f，ρ=%.4f\n' % (
      cf[0], cf[1], np.sqrt(((L4 - pr) ** 2).mean()),
      1 - ((L4 - pr) ** 2).sum() / ((L4 - L4.mean()) ** 2).sum(), stats.spearmanr(L4, pr).statistic))

w('\n### 4.2 B5（44 点，9 个族 / 6 个文献来源）\n')
N5 = B5['N_params_B'].to_numpy(float); D5 = B5['D_tokens_B'].to_numpy(float); L5 = B5['val_loss'].to_numpy(float)
p5 = law(r1.x, N5, D5); r5 = L5 - p5
w('- 直接外推：RMSE=%.4f，平均偏差=%+.4f，ρ=%.4f，$R^2$=%.4f\n' % (
    np.sqrt((r5 ** 2).mean()), r5.mean(), stats.spearmanr(L5, p5).statistic,
    1 - (r5 ** 2).sum() / ((L5 - L5.mean()) ** 2).sum()))
w('| 来源 | n | 平均残差 | RMSE |')
w('|---|---|---|---|')
for src, g in B5.groupby('source'):
    pp = law(r1.x, g['N_params_B'].to_numpy(float), g['D_tokens_B'].to_numpy(float))
    rr = g['val_loss'].to_numpy() - pp
    w('| %s | %d | %+.4f | %.4f |' % (src, len(g), rr.mean(), np.sqrt((rr ** 2).mean())))
w('\n**判读**：B4/B5 是**不同分词器、不同验证集、不同训练配方**下的 Loss，'
  '与 Pythia 的绝对水平不可直接比较（平均偏差 %.3f~%.3f 属正常）。'
  '应看**秩相关**：B4 ρ=%.4f、B5 ρ=%.4f，均为强正相关，说明标度律的**形状**（指数 α、β）具有跨族泛化性。\n' % (
      r4.mean(), r5.mean(), stats.spearmanr(L4, p4).statistic, stats.spearmanr(L5, p5).statistic))

# ---------------- 可检验假设：质量 = 有效数据量缩放 ----------------
w('\n## 5. 可检验假设：质量是否等价于"有效数据量缩放"？\n')
w('**假设 H**：数据质量 $Q$ 的作用等价于把有效数据量放大为 $D_{eff}=D\\cdot Q^{\\kappa}$，'
  '即所有 $(N,D,Q)$ 实验点在把 $D$ 换成 $D\\cdot Q^\\kappa$ 后，'
  '应统一落到同一张经典标度律曲面上。\n')
w('检验方法：在 $\\kappa$ 的网格上，用 $(N,\\;D\\cdot Q^\\kappa)$ 重新拟合经典标度律，'
  '看残差 RMSE 关于 $\\kappa$ 的曲线是否有唯一极小值。若有，则 H 成立且 $\\kappa$ 可估。\n')

b6c = B6.dropna(subset=['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
NQ = b6c['N_params_B'].to_numpy(float); DQ = b6c['D_tokens_B'].to_numpy(float)
QQ = b6c['Q_score'].to_numpy(float); LQ = b6c['val_loss'].to_numpy(float)
w('\n- B6 规模：n=%d；N∈[%.3g, %.3g]；D∈[%.0f, %.0f]；Q 取值=%s；Loss∈[%.4f, %.4f]\n' % (
    len(b6c), NQ.min(), NQ.max(), DQ.min(), DQ.max(),
    sorted(set(np.round(QQ, 3))), LQ.min(), LQ.max()))

# 先检验：Q=1 子集是否服从 B1 的经典律
m1 = QQ >= 0.999
w('\n### 5.1 假设 H0：B6 中 Q=1 的子集应服从由 B1 拟合的经典标度律\n')
p_q1 = law(r1.x, NQ[m1], DQ[m1])
r_q1 = LQ[m1] - p_q1
w('- Q=1 子集 n=%d；残差 均值=%+.5f，标准差=%.5f，RMSE=%.5f；'
  '与 B1 自身残差标准差 %.5f 对照。\n' % (m1.sum(), r_q1.mean(), r_q1.std(ddof=1),
                                       np.sqrt((r_q1 ** 2).mean()), res1.std(ddof=1)))
w('- Welch t 检验（B6@Q=1 残差 vs B1 残差）：t=%.3f, p=%.4g\n' % stats.ttest_ind(
    r_q1, res1, equal_var=False)[:2])
w('⇒ %s\n' % ('**H0 成立**：B6 的 Q=1 子集与 B1 经典律无显著差异，两套独立实验可在同一模型中统一。'
              if abs(r_q1.mean()) < 0.05 else
              'H0 不完全成立，需引入尺度校正项。'))

w('\n### 5.2 $\\kappa$ 网格搜索（B6）\n')
rows = []
for kap in np.arange(0.0, 4.01, 0.10):
    rk = fit_law(NQ, DQ * np.power(QQ, kap), LQ, n_starts=6, seed=11)
    rm = float(np.sqrt(np.mean((law(rk.x, NQ, DQ * np.power(QQ, kap)) - LQ) ** 2)))
    rows.append((kap, rm, rk.x))
w('| $\\kappa$ | RMSE | $\\alpha$ | $\\beta$ |')
w('|---|---|---|---|')
for kap, rm, th in rows:
    mark = '**' if rm == min(r[1] for r in rows) else ''
    w('| %s%.2f%s | %.6f | %.4f | %.4f |' % (mark, kap, mark, rm, th[2], th[4]))
kbest = min(rows, key=lambda r: r[1])
w('\n- 粗搜索最优 $\\kappa$=%.2f（RMSE=%.6f）；$\\kappa=0$（即质量无作用）时 RMSE=%.6f。'
  '**降低幅度 %.1f%%** ⇒ 质量项确实必要。\n' % (
      kbest[0], kbest[1], rows[0][1], 100 * (1 - kbest[1] / rows[0][1])))

# 细化
rows2 = []
for kap in np.arange(max(0.0, kbest[0] - 0.2), kbest[0] + 0.21, 0.02):
    rk = fit_law(NQ, DQ * np.power(QQ, kap), LQ, n_starts=8, seed=21)
    rm = float(np.sqrt(np.mean((law(rk.x, NQ, DQ * np.power(QQ, kap)) - LQ) ** 2)))
    rows2.append((kap, rm, rk.x))
kb2 = min(rows2, key=lambda r: r[1])
w('- 细化搜索最优 $\\kappa^*$=%.3f（RMSE=%.6f），$\\alpha$=%.4f，$\\beta$=%.4f。\n' % (
    kb2[0], kb2[1], kb2[2][2], kb2[2][4]))
w('- 由此 $\\gamma=\\beta\\kappa$=%.4f。\n' % (kb2[2][4] * kb2[0]))

# 该假设在 B7/B8 上复核
for nm3, dfx in [('B7', B7), ('B8(仅 calibrated)', B8[B8['data_type'] == 'calibrated']),
                 ('B8(全部)', B8)]:
    d = dfx.dropna(subset=['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
    nx, dx, qx, lx = (d['N_params_B'].to_numpy(float), d['D_tokens_B'].to_numpy(float),
                      d['Q_score'].to_numpy(float), d['val_loss'].to_numpy(float))
    rr = []
    for kap in np.arange(0.0, 3.01, 0.10):
        rk = fit_law(nx, dx * np.power(qx, kap), lx, n_starts=6, seed=31)
        rr.append((kap, float(np.sqrt(np.mean((law(rk.x, nx, dx * np.power(qx, kap)) - lx) ** 2))), rk.x))
    kb = min(rr, key=lambda t: t[1])
    w('- %s：n=%d，最优 $\\kappa$=%.2f（RMSE=%.6f，$\\kappa=0$ 时 %.6f），$\\alpha$=%.4f，$\\beta$=%.4f，$\\gamma$=%.4f\n' % (
        nm3, len(d), kb[0], kb[1], rr[0][1], kb[2][2], kb[2][4], kb[2][4] * kb[0]))

json.dump(dict(kappa=float(kb2[0]), theta=list(map(float, kb2[2])),
               gamma=float(kb2[2][4] * kb2[0])),
          open(os.path.join(OUT, 'kappa_estimate.json'), 'w'), indent=1)
LOG.close()
print('ok')

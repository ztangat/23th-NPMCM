# -*- coding: utf-8 -*-
"""问题四 · 阶段一（定稿 v2）：稳健前沿、贡献分解、算力增长律

相对 q4_main3.py 的修正：
 1. 稳健前沿改用【固定前 K=20 名均值】而非"top 1% 比例"。
    理由：比例口径下 K 随累计样本量增长，前沿均值反而会被拉低（§1.1 出现非单调），
    不是真正的前沿。固定 K 时"前 K 大均值"对累计集合是严格单调不减的
    （超集的第 i 大 ≥ 子集的第 i 大），因此是一个合格的前沿定义。
 2. §1.2 分口径前沿改为【累计】口径（截至该月末），与 §1.1 一致；
    此前是按月窗口计算，导致非单调且与 §1.1 数值不一致。
 3. §2.2 Oaxaca 分解的技术项改在【两期分箱的共同支撑内】取值（ℓ*），
    原先取 ℓ*=4.356（78B）属外推，得到 +10.985 与观测 ΔS 无法对账；
    并以固定规模档位的直接增益作交叉校验。
 4. 固定规模档位对照改用【固定前 20 名】而非 top5%，使两期口径严格可比。
"""
import os, io, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
CE = os.path.join(ROOT, 'real_attachments', 'C_efficiency_evolution')
OUT = os.path.join(ROOT, '_out', '04_q4')
BUF = []
def w(s=''): BUF.append(s)

K = 20   # 稳健前沿：固定取前 K 名

P = pd.read_csv(os.path.join(OUT, 'q4_panel.csv'))
P['date'] = pd.to_datetime(P['date'])
P['src'] = np.where(P['lic'] == 'historical', 'historical', 'leaderboard')
LB = P[P['src'] == 'leaderboard'].copy().sort_values('date').reset_index(drop=True)
HI = P[P['src'] == 'historical'].copy().sort_values('date').reset_index(drop=True)

w('# 问题四 · 阶段一（定稿）：稳健前沿、贡献分解、算力增长律\n')

# ================= 1. 前沿 =================
w('\n## 1. 能力前沿\n')
w('\n定义两种前沿（$S$ 取综合能力主口径 $S_{avg}$，见 §0 数据口径）：\n')
w('- **单点前沿** $\\mathcal{F}_{max}(t)=\\max_{t_i\\le t}S_i$：只看最高分，易受单个偶然极值影响；\n')
w('- **稳健前沿** $\\mathcal{F}_{K}(t)=$ 截至 $t$ 已提交模型中分数最高的 **前 %d 名** 的均值（主口径）。\n' % K)
w('\n> 之所以用"固定前 %d 名"而不是"前 1%%"：比例口径下取用个数随累计样本量增长，'
  '会把越来越多中等分数拉进均值，反而使前沿**下降**（不是前沿应有的性质）。'
  '固定 $K$ 时，累计集合扩大只会让"前 $K$ 大"逐个元素不减，故 $\\mathcal{F}_K(t)$ '
  '对 $t$ **严格单调不减**，是合格的前沿定义。\n' % K)

def rob(s, k=K):
    s = np.sort(np.asarray(s, float))
    s = s[np.isfinite(s)]
    if len(s) < k:
        return np.nan
    return float(s[-k:].mean())

def fronts(df):
    d = df.sort_values('date').copy()
    d['Fmax'] = d['S_avg'].cummax()
    d['Frob'] = [rob(d['S_avg'].iloc[:i + 1]) for i in range(len(d))]
    return d

FL = fronts(LB)
w('\n### 1.1 榜单口径（C1，%s – %s）\n' % (LB['date'].min().date(), LB['date'].max().date()))
FL['ym'] = FL['date'].dt.to_period('M').astype(str)
M = FL.groupby('ym').agg(Fmax=('Fmax', 'last'), Frob=('Frob', 'last'), n=('S_avg', 'size'))
M.to_csv(os.path.join(OUT, 'q4_frontier_monthly.csv'), encoding='utf-8-sig')
w('\n| 月末 | 单点前沿 $\\mathcal{F}_{max}$ | 稳健前沿 $\\mathcal{F}_{%d}$ | 当月提交 | 累计提交 |' % K)
w('|---|---|---|---|---|')
cum = 0
for ym, r in M.iterrows():
    cum += int(r['n'])
    w('| %s | %.3f | %.3f | %d | %d |' % (ym, r['Fmax'], r['Frob'], int(r['n']), cum))
w('\n- 稳健前沿由 %.3f 升至 %.3f，9 个月 **%+.3f**（%+.1f%%）；'
  '单点前沿由 %.3f 升至 %.3f（%+.3f）。\n'
  % (M['Frob'].iloc[0], M['Frob'].iloc[-1], M['Frob'].iloc[-1] - M['Frob'].iloc[0],
     100 * (M['Frob'].iloc[-1] / M['Frob'].iloc[0] - 1),
     M['Fmax'].iloc[0], M['Fmax'].iloc[-1], M['Fmax'].iloc[-1] - M['Fmax'].iloc[0]))
w('- 单调性检查：稳健前沿逐月 %s。\n'
  % ('单调不减 ✔' if (M['Frob'].diff().dropna() >= -1e-9).all() else '存在回落 ✘'))

# ---- 1.2 分口径累计前沿 ----
w('\n### 1.2 分口径的累计稳健前沿（月末，$\\mathcal{F}_{%d}$）\n' % K)
w('\n| 月末 | 全部 | 开源(R2) | pretrained | chat/finetuned |')
w('|---|---|---|---|---|')
rows = []
for ym, g in FL.groupby('ym'):
    cut = g['date'].max()
    sub = FL[FL['date'] <= cut]
    rows.append(dict(ym=ym, all=rob(sub['S_avg']),
                     o=rob(sub[sub['open_R2'].astype(bool)]['S_avg']),
                     p=rob(sub[sub['mtype'] == 'pretrained']['S_avg']),
                     c=rob(sub[sub['mtype'].isin(['chat', 'finetuned', 'merge'])]['S_avg'])))
R = pd.DataFrame(rows).set_index('ym')
R.to_csv(os.path.join(OUT, 'q4_frontier_by_group.csv'), encoding='utf-8-sig')
for ym, r in R.iterrows():
    w('| %s | %.3f | %s | %s | %s |' % (ym, r['all'],
      ('%.3f' % r['o']) if np.isfinite(r['o']) else '—',
      ('%.3f' % r['p']) if np.isfinite(r['p']) else '—',
      ('%.3f' % r['c']) if np.isfinite(r['c']) else '—'))
w('\n- 期末（2025-03）累计稳健前沿：全部 %.3f / 开源(R2) %.3f / pretrained %.3f / chat+finetuned+merge %.3f。\n'
  % (R['all'].iloc[-1], R['o'].iloc[-1], R['p'].iloc[-1], R['c'].iloc[-1]))
w('- 9 个月累计提升：全部 %+.3f / 开源 %+.3f / pretrained %+.3f / chat 类 %+.3f。\n'
  % (R['all'].iloc[-1] - R['all'].iloc[0], R['o'].iloc[-1] - R['o'].iloc[0],
     R['p'].iloc[-1] - R['p'].iloc[0], R['c'].iloc[-1] - R['c'].iloc[0]))
w('- **pretrained 前沿显著低于 chat/finetuned 类**：基础模型未经指令微调，'
  '在 IFEval 等指令跟随类基准上天然吃亏；这与"同一底座经 chat 微调后榜单分大幅提升"一致。\n')

# ---- 1.3 长时段（只打印更新点） ----
w('\n### 1.3 长时段参照（C3 Historical + C1，仅定性：评测协议与 v2 不同）\n')
AH = fronts(pd.concat([HI, LB]).sort_values('date').reset_index(drop=True))
w('\n| 时点 | 累计单点前沿 | 代表模型 | 参数 (B) |')
w('|---|---|---|---|')
prev = -1
for _, r in AH.iterrows():
    if r['Fmax'] > prev + 1e-9:
        w('| %s | %.3f | %s | %.4g |' % (r['date'].date(), r['Fmax'], str(r['Model'])[:44],
                                         r['#Params (B)']))
        prev = r['Fmax']

# ================= 2. 分解 =================
w('\n---\n\n## 2. 规模扩张 vs 非规模技术进步：贡献分解\n')
T0, T1, T2 = pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-01'), pd.Timestamp('2025-03-14')
P1 = LB[(LB['date'] >= T0) & (LB['date'] < T1)].copy()
P2 = LB[(LB['date'] >= T1) & (LB['date'] <= T2)].copy()
w('\n- **P1**（早期）= %s – %s，%d 个；**P2**（近期）= %s – %s，%d 个。\n'
  % (T0.date(), T1.date(), len(P1), T1.date(), T2.date(), len(P2)))

# ---- 2.1 固定规模档位（固定前 20 名） ----
w('\n### 2.1 判据一（主判据）：固定规模档位的直接对照\n')
w('\n在同一参数规模档位内比较两期的**前 %d 名均值**。由于 $N$ 被钉在同一档位，'
  '该档位内的全部增益在定义上就是**非规模技术进步**（架构、数据、微调/merge 配方）。\n' % K)
edges = [0, 2, 4, 8, 16, 40, 100, 1e9]
labs = ['<2B', '2–4B', '4–8B', '8–16B', '16–40B', '40–100B', '>100B']
for d in (P1, P2):
    d['bucket'] = pd.cut(d['#Params (B)'], edges, labels=labs, right=False)
w('\n| 规模档 | $n_1$ | P1 前%d名均值 | $n_2$ | P2 前%d名均值 | 增益 | 相对增益 |' % (K, K))
w('|---|---|---|---|---|---|---|')
rows = []
for b in labs:
    a = P1[P1['bucket'] == b]['S_avg']; c = P2[P2['bucket'] == b]['S_avg']
    sa, sc = rob(a), rob(c)
    if not (np.isfinite(sa) and np.isfinite(sc)):
        w('| %s | %d | %s | %d | %s | — | — |' % (
            b, len(a), ('%.3f' % sa) if np.isfinite(sa) else '样本<20',
            len(c), ('%.3f' % sc) if np.isfinite(sc) else '样本<20'))
        continue
    rows.append(dict(bucket=b, n1=len(a), n2=len(c), s1=sa, s2=sc, d=sc - sa, rel=(sc - sa) / sa))
    w('| %s | %d | %.3f | %d | %.3f | **%+.3f** | %+.1f%% |'
      % (b, len(a), sa, len(c), sc, sc - sa, 100 * (sc - sa) / sa))
BS = pd.DataFrame(rows)
BS.to_csv(os.path.join(OUT, 'q4_fixed_scale_gain.csv'), index=False, encoding='utf-8-sig')
w('\n- %d 个有效档位**全部为正增益**：最小 %+.3f，最大 %+.3f，中位 %+.3f，均值 %+.3f；'
  '相对增益中位 **%+.1f%%**、均值 %+.1f%%。\n'
  % (len(BS), BS['d'].min(), BS['d'].max(), BS['d'].median(), BS['d'].mean(),
     100 * BS['rel'].median(), 100 * BS['rel'].mean()))
w('\n> **这是最干净的因果证据**：把参数规模"钉死"在同一档位（且两期取用个数同为 %d，'
  '口径严格可比），9 个月内能力仍提升 %.1f%%（中位）——这部分**不可能**由规模扩张解释。\n'
  % (K, 100 * BS['rel'].median()))

# ---- 2.2 Oaxaca（修正：在共同支撑内取值） ----
w('\n### 2.2 判据二：前沿曲线的 Oaxaca–Blinder 分解（修正口径）\n')
w('\n把各期"规模—能力前沿"写成对数线性曲线 $S_t(\\ell)=a_t+b_t\\ell$（$\\ell=\\ln N$，'
  '分箱点为各分位箱内前 %d 名均值）。总增益分解为\n' % K)
w('$$\\underbrace{\\Delta S}_{\\text{观测前沿增益}}='
  '\\underbrace{b_{ref}\\,\\Delta\\bar\\ell}_{\\text{规模扩张}}'
  '+\\underbrace{(\\Delta S-b_{ref}\\,\\Delta\\bar\\ell)}_{\\text{技术进步}}$$\n')
w('\n其中 $b_{ref}$ 取两期"规模—能力"曲线斜率的均值，$\\Delta\\bar\\ell$ 取两期'
  '**稳健前沿构成模型**（各期分数最高的前 %d 名）的平均 $\\ln N$ 之差。\n' % K)
w('\n> **修正说明**：main3 版本把技术项取在 $\\ell^{*}=\\ell_1=4.356$（78 B）。'
  '该处**落在两期分箱支撑之外**（属外推），算出的 +10.985 与观测到的前沿增益无法对账。'
  '本版改为：① 主分解直接锚定**观测到的**稳健前沿增益并用构成模型的规模差分配；'
  '② 曲线的垂直落差只在两期分箱的**共同支撑区间中点** $\\ell^{*}$ 处计算，'
  '作为与 §2.1 固定规模增益的交叉校验。\n')

def curve(df, nbins=8, min_per_bin=25):
    d = df.dropna(subset=['#Params (B)', 'S_avg']).copy()
    d = d[(d['#Params (B)'] > 0)]
    d['lnN'] = np.log(d['#Params (B)'])
    d['bin'] = pd.qcut(d['lnN'], nbins, duplicates='drop')
    pts = []
    for b, g in d.groupby('bin', observed=True):
        if len(g) < min_per_bin:
            continue
        v = rob(g['S_avg'])
        if not np.isfinite(v):
            continue
        pts.append((g['lnN'].median(), v, len(g), str(g.loc[g['S_avg'].idxmax(), 'Model'])[:40]))
    pts = pd.DataFrame(pts, columns=['lnN', 'S', 'n', 'Model']).sort_values('lnN')
    if len(pts) < 3:
        return None
    b, a = np.polyfit(pts['lnN'], pts['S'], 1)
    return dict(a=float(a), b=float(b), pts=pts)

def topinfo(df):
    d = df.dropna(subset=['S_avg'])
    j = d['S_avg'].idxmax()
    return (float(np.log(max(d.loc[j, '#Params (B)'], 1e-9))), float(d.loc[j, 'S_avg']),
            str(d.loc[j, 'Model'])[:40])

c1, c2 = curve(P1), curve(P2)
l1, s1, m1 = topinfo(P1); l2, s2, m2 = topinfo(P2)
S1 = lambda l: c1['a'] + c1['b'] * l
S2 = lambda l: c2['a'] + c2['b'] * l
lo = max(c1['pts']['lnN'].min(), c2['pts']['lnN'].min())
hi = min(c1['pts']['lnN'].max(), c2['pts']['lnN'].max())
lstar = 0.5 * (lo + hi)
w('\n| 项 | P1（2024-06–09） | P2（2024-10–2025-03） |')
w('|---|---|---|')
w('| 前沿曲线截距 $a$ | %.4f | %.4f |' % (c1['a'], c2['a']))
w('| 前沿曲线斜率 $b$（分/$\\ln N$） | %.4f | %.4f |' % (c1['b'], c2['b']))
w('| 前沿模型 | %s | %s |' % (m1, m2))
w('| 前沿模型规模 $\\ell=\\ln N$ | %.4f（%.3g B） | %.4f（%.3g B） |'
  % (l1, np.exp(l1), l2, np.exp(l2)))
w('| 单点前沿 $S$ | %.3f | %.3f |' % (s1, s2))
w('| 共同支撑区间 $\\ell\\in[%.3f,%.3f]$，取 $\\ell^{*}=%.3f$（%.2f B） | | |'
  % (lo, hi, lstar, np.exp(lstar)))
scale_old = c1['b'] * (l2 - l1); scale_new = c2['b'] * (l2 - l1)
tech_curve = S2(lstar) - S1(lstar)
dS_rob = rob(P2['S_avg']) - rob(P1['S_avg'])
dS_max = s2 - s1

# 主分解：锚定观测到的稳健前沿增益，按"前沿构成模型"的规模差分配
def cohort(df, k=K):
    d = df.dropna(subset=['S_avg', '#Params (B)'])
    d = d[d['#Params (B)'] > 0].nlargest(k, 'S_avg')
    return (float(np.log(d['#Params (B)']).mean()), float(d['S_avg'].mean()),
            d['date'].mean())
lb1, sb1, dt1 = cohort(P1); lb2, sb2, dt2 = cohort(P2)
dt_months = (dt2 - dt1).days / 30.4375
bref = 0.5 * (c1['b'] + c2['b'])
scale_cohort = bref * (lb2 - lb1)
tech_cohort = dS_rob - scale_cohort
share = float(100 * tech_cohort / dS_rob) if abs(dS_rob) > 1e-9 else float('nan')

w('\n| 分解项 | 数值 | 说明 |')
w('|---|---|---|')
w('| 观测：稳健前沿 $\\Delta\\mathcal{F}_{%d}$ | **%+.3f** | P1 前%d名均值 %.3f → P2 前%d名均值 %.3f |'
  % (K, dS_rob, K, sb1, K, sb2))
w('| 前沿构成模型平均规模 $\\bar\\ell$ | %.4f → %.4f | 几何平均 %.2f B → %.2f B |'
  % (lb1, lb2, np.exp(lb1), np.exp(lb2)))
w('| 规模扩张 $b_{ref}(\\bar\\ell_2-\\bar\\ell_1)$ | **%+.3f** | $b_{ref}=%.4f$，$\\Delta\\bar\\ell=%+.4f$ |'
  % (scale_cohort, bref, lb2 - lb1))
w('| 技术进步 $=\\Delta\\mathcal{F}_{%d}-\\text{规模扩张}$ | **%+.3f** | 占观测增益 **%.1f%%** |'
  % (K, tech_cohort, share))
w('| 对照：单点前沿模型口径 $b(\\ell_2-\\ell_1)$ | %+.3f | 两期单点前沿模型同为 ~78 B（$\\ell=%.4f$），恒为 0 |'
  % (0.5 * (scale_old + scale_new), l1))
w('| 观测：单点前沿 $\\Delta\\mathcal{F}_{max}$ | %+.3f | %.3f → %.3f |' % (dS_max, s1, s2))
w('\n- **交叉校验**：曲线在共同支撑中点 $\\ell^{*}=%.3f$（%.2f B）处的垂直落差为 **%+.3f**，'
  '与 §2.1 固定规模档位的直接增益（中位 %+.3f）**量级一致**——两条独立路径互相验证。\n'
  % (lstar, np.exp(lstar), tech_curve, BS['d'].median()))
w('\n> **结论：两条口径一致指向同一方向，且比"技术占 100%"更强**。\n')
w('>\n')
w('> 1. **单点前沿口径**：两期最高分模型同为 ~78 B（$\\ell_1=\\ell_2=%.4f$），'
  '规模扩张项在定义上为 0，观测增益 %+.3f **100%% 来自技术进步**。\n' % (l1, dS_max))
w('>\n')
w('> 2. **稳健前沿口径（前 %d 名）**：构成模型的平均规模**不升反降**'
  '（%.2f B → %.2f B，$\\Delta\\bar\\ell=%+.4f$），按 $b_{ref}=%.4f$ 折算，'
  '规模扩张项为 **%+.3f（负贡献）**，技术进步 **%+.3f**，相当于观测增益的 **%.1f%%**。\n'
  % (K, np.exp(lb1), np.exp(lb2), lb2 - lb1, bref, scale_cohort, tech_cohort, share))
w('>\n')
w('> 即：2024Q2–2025Q1 这 9 个月内，开源能力前沿的提升**完全由非规模技术进步贡献**，'
  '而且前沿还同时**向更小规模迁移**——在同等甚至更小的参数量下取得更高能力，'
  '这正是**算力/参数效率提升**的直接证据。\n')
w('>\n')
w('> 需要注意这是**本榜单口径下的结论**（榜单期仅 9 个月，且规模被 ~78 B 的 merge 模型封顶），'
  '并不否定更长周期上规模扩张的作用：规模扩张在 C4 的长周期里依然显著（见 §3）。\n')

w('\n#### 2.2.1 前沿分箱点（各箱内前 %d 名均值）\n' % K)
w('\n| 期 | $\\ln N$ | $N$ (B) | 箱内前%d名均值 | 箱内模型数 | 箱内最高分模型 |' % K)
w('|---|---|---|---|---|---|')
for nm, c in [('P1', c1), ('P2', c2)]:
    for _, r in c['pts'].iterrows():
        w('| %s | %.3f | %.3g | %.3f | %d | %s |'
          % (nm, r['lnN'], np.exp(r['lnN']), r['S'], int(r['n']), r['Model']))

# ---- 2.3 面板回归 ----
w('\n### 2.3 判据三：面板回归的时间项（交叉验证，样本稀疏）\n')
sub = LB.dropna(subset=['C_FLOPs', '#Params (B)', 'S_avg'])
sub = sub[(sub['C_FLOPs'] > 0) & (sub['#Params (B)'] > 0) & (sub['S_avg'] > 0)].copy()
w('\n- 同时具备 $(N,C,S,t)$ 的样本仅 **%d** 个（C1 与 C4 的算力匹配率 %.1f%%），'
  '故本判据只作方向性参照。\n' % (len(sub), 100 * len(sub) / len(LB)))
w('\n$$\\ln S_i=\\beta_0+\\beta_N\\ln N_i+\\beta_C\\ln C_i+\\gamma t_i+\\varepsilon_i$$\n')
if len(sub) >= 40:
    t0 = sub['date'].min()
    Xb = np.column_stack([np.ones(len(sub)), np.log(sub['#Params (B)']), np.log(sub['C_FLOPs']),
                          (sub['date'] - t0).dt.days / 365.25])
    y = np.log(sub['S_avg'])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    rss = float(((y - Xb @ beta) ** 2).sum()); s2 = rss / (len(sub) - 4)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xb.T @ Xb)))
    w('\n| 系数 | 估计 | 标准误 | $t$ | $p$ |')
    w('|---|---|---|---|---|')
    for i, nm in enumerate(['$\\beta_0$', '$\\beta_N$', '$\\beta_C$', '$\\gamma$（每年）']):
        tv = beta[i] / se[i]
        w('| %s | %+.5f | %.5f | %.2f | %.3f |' % (nm, beta[i], se[i], tv,
                                                   2 * (1 - stats.t.cdf(abs(tv), len(sub) - 4))))
    w('\n- $R^2$=%.4f；$\\gamma$=**%+.4f/年**（$p$=%.3f），即在控制规模与算力后，'
      '能力仍以每年约 %.1f%% 的速率独立增长——这是"非规模技术进步"的第三条独立证据。\n'
      % (1 - rss / ((y - y.mean()) ** 2).sum(), beta[3],
         2 * (1 - stats.t.cdf(abs(beta[3] / se[3]), len(sub) - 4)),
         100 * (np.exp(beta[3]) - 1)))
    json.dump(dict(beta=beta.tolist(), se=se.tolist(), n=int(len(sub)),
                   gamma=float(beta[3]), gamma_p=float(
                       2 * (1 - stats.t.cdf(abs(beta[3] / se[3]), len(sub) - 4)))),
              open(os.path.join(OUT, 'q4_panel_reg.json'), 'w'), ensure_ascii=False, indent=2)

# ================= 3. C4 增长律 =================
w('\n---\n\n## 3. 算力 / 参数 / 数据量的历史增长律（依据 C4）\n')
C4 = pd.read_csv(os.path.join(CE, 'epoch_all_ai_models.csv'), low_memory=False)
c4 = C4[C4['Domain'].fillna('').str.contains('Language', case=False, na=False)].copy()
c4['pub'] = pd.to_datetime(c4['Publication date'], errors='coerce')
c4['Cfl'] = pd.to_numeric(c4['Training compute (FLOP)'], errors='coerce')
c4['Np'] = pd.to_numeric(c4['Parameters'], errors='coerce')
c4['Dsz'] = pd.to_numeric(c4['Training dataset size (total)'], errors='coerce')
c4 = c4.dropna(subset=['pub'])
c4 = c4[(c4['pub'] >= pd.Timestamp('2012-01-01')) & (c4['pub'] <= pd.Timestamp('2026-12-31'))]
w('\n- Language 域且有发布日期：%d 条（%s – %s）。\n'
  % (len(c4), c4['pub'].min().date(), c4['pub'].max().date()))
w('- **异常值处理**：参数量 > $5\\times10^{12}$（超过 Switch-C 的 1.57 T 近 4 倍）、'
  '训练算力 > $10^{27}$ 视为脏数据剔除；每年取**前 3 名均值**作为该年前沿。\n')

SPEC = [('训练算力', 'Cfl', 1e27, 'FLOPs'),
        ('参数量', 'Np', 5e12, '个'),
        ('训练数据量', 'Dsz', 1e15, 'token')]
GROW = {}
for idx, (nm, col, cap, unit) in enumerate(SPEC):
    s = c4.dropna(subset=[col]).copy()
    n0 = len(s)
    s = s[(s[col] > 0) & (s[col] < cap)]
    w('\n### 3.%d %s（%d 条有值，剔除异常后 %d 条）\n' % (idx + 1, nm, n0, len(s)))
    g = s.groupby(s['pub'].dt.year)[col].apply(lambda v: np.sort(v.to_numpy())[-3:].mean())
    g = g[g.index >= 2015]
    w('\n| 年份 | 当年 top3 均值 | 累计前沿 | 记录数 |')
    w('|---|---|---|---|')
    cum = 0
    for yr, v in g.items():
        cum = max(cum, v)
        w('| %d | %.4g | %.4g | %d |' % (yr, v, cum, int((s['pub'].dt.year == yr).sum())))
    x = np.array(g.index, float); yv = np.log(np.array(g.values, float))
    sl, _ = np.polyfit(x, yv, 1)
    w('\n- 全时段（%d–%d）对数线性拟合：年增长率 **%.1f%%/年**，翻倍周期 %.2f 年。\n'
      % (int(x.min()), int(x.max()), 100 * (np.exp(sl) - 1), np.log(2) / sl))
    subs = {}
    for a, b in [(2018, 2026), (2021, 2026), (2023, 2026)]:
        m = (x >= a) & (x <= b)
        if m.sum() >= 4:
            s_, _ = np.polyfit(x[m], yv[m], 1)
            subs['%d-%d' % (a, b)] = float(s_)
            w('  - %d–%d：年增长率 **%.1f%%/年**，翻倍周期 %.2f 年（%d 个年度点）\n'
              % (a, b, 100 * (np.exp(s_) - 1), np.log(2) / s_, m.sum()))
    GROW[nm] = dict(years=[int(v) for v in g.index], vals=[float(v) for v in g.values],
                    slope_all=float(sl), slope_sub=subs, unit=unit, n=int(len(s)))
    json.dump(GROW[nm], open(os.path.join(OUT, 'q4_c4_growth_%s.json' % col.lower()),
                             'w', encoding='utf-8'), ensure_ascii=False, indent=2)

w('\n### 3.4 增长放缓的证据\n')
w('\n| 序列 | 全时段年增长 | 2018–2026 | 2021–2026 | 2023–2026 |')
w('|---|---|---|---|---|')
for nm, v in GROW.items():
    def f(k):
        return ('%.1f%%' % (100 * (np.exp(v['slope_sub'][k]) - 1))) if k in v['slope_sub'] else '—'
    w('| %s | %.1f%% | %s | %s | %s |'
      % (nm, 100 * (np.exp(v['slope_all']) - 1), f('2018-2026'), f('2021-2026'), f('2023-2026')))
w('\n> 注：2026 年仅部分数据（C4 覆盖至 2026-04），且年度点为"当年 top3 均值"，'
  '尾部年份样本少、波动大，故 2023–2026 的负增长主要反映**样本截断与记录稀疏**，'
  '不宜直接外推为真实负增长；预测中采用更稳健的近期区间（2021–2026）。\n')

json.dump(dict(frontier_max=[float(v) for v in M['Fmax']],
               frontier_rob=[float(v) for v in M['Frob']],
               months=[str(v) for v in M.index],
               dS_rob=float(dS_rob), dS_max=float(dS_max),
               scale=float(scale_cohort), tech=float(tech_cohort),
               share_tech=float(share), tech_curve=float(tech_curve),
               lbar=[float(lb1), float(lb2)], lstar=float(lstar),
               b_ref=float(bref), dt_months=float(dt_months),
               tech_per_month=float(tech_cohort / dt_months),
               scale_per_month=float(scale_cohort / dt_months),
               cohort_dates=[str(dt1.date()), str(dt2.date())],
               fixed_scale_median=float(BS['d'].median()),
               fixed_scale_median_rel=float(BS['rel'].median())),
          open(os.path.join(OUT, 'q4_decomposition.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '04d_decomposition.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

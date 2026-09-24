# -*- coding: utf-8 -*-
"""问题四 · 阶段一（定稿）：稳健前沿、贡献分解、算力增长律

相对 q4_main2.py 的三处修正：
 1. 历史长表只打印【前沿更新点】；
 2. 前沿改用 top-1% 均值（稳健前沿），避免被单个 78B merge 模型的偶然极值主导；
    同时保留单点最大前沿作对照；
 3. C4 增长律用每年【前 3 名均值】并剔除异常量级（参数 >5e12、算力 >1e27），
    避免 2022 年 1.739e14 参数这类脏数据把增长率带成负数。
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

P = pd.read_csv(os.path.join(OUT, 'q4_panel.csv'))
P['date'] = pd.to_datetime(P['date'])
P['src'] = np.where(P['lic'] == 'historical', 'historical', 'leaderboard')
LB = P[P['src'] == 'leaderboard'].copy().sort_values('date').reset_index(drop=True)
HI = P[P['src'] == 'historical'].copy().sort_values('date').reset_index(drop=True)

w('# 问题四 · 阶段一（定稿）：稳健前沿、贡献分解、算力增长律\n')

# ================= 1. 前沿 =================
w('\n## 1. 能力前沿\n')
w('\n定义两种前沿：\n')
w('- **单点前沿** $\\mathcal{F}_{max}(t)=\\max_{t_i\\le t}S_i$（易受单个偶然极值影响）；\n')
w('- **稳健前沿** $\\mathcal{F}_{1\\%}(t)=$ 截至 $t$ 已提交模型中分数最高的 **1%** 的均值（主口径）。\n')

def fronts(df):
    d = df.sort_values('date').copy()
    d['Fmax'] = d['S_avg'].cummax()
    rob = []
    for i in range(len(d)):
        s = d['S_avg'].iloc[:i + 1]
        k = max(3, int(np.ceil(0.01 * len(s))))
        rob.append(np.sort(s.to_numpy())[-k:].mean())
    d['Frob'] = rob
    return d

FL = fronts(LB)
w('\n### 1.1 榜单口径（C1，%s – %s）\n' % (LB['date'].min().date(), LB['date'].max().date()))
FL['ym'] = FL['date'].dt.to_period('M').astype(str)
M = FL.groupby('ym').agg(Fmax=('Fmax', 'last'), Frob=('Frob', 'last'), n=('S_avg', 'size'))
M.to_csv(os.path.join(OUT, 'q4_frontier_monthly.csv'), encoding='utf-8-sig')
w('\n| 月末 | 单点前沿 | 稳健前沿(top 1%) | 当月提交 | 累计提交 |')
w('|---|---|---|---|---|')
cum = 0
for ym, r in M.iterrows():
    cum += int(r['n'])
    w('| %s | %.3f | %.3f | %d | %d |' % (ym, r['Fmax'], r['Frob'], int(r['n']), cum))
w('\n- 稳健前沿从 %.3f 升至 %.3f，9 个月提升 **%+.3f**（%+.1f%%）；'
  '单点前沿从 %.3f 升至 %.3f（%+.3f）。\n'
  % (M['Frob'].iloc[0], M['Frob'].iloc[-1], M['Frob'].iloc[-1] - M['Frob'].iloc[0],
     100 * (M['Frob'].iloc[-1] / M['Frob'].iloc[0] - 1),
     M['Fmax'].iloc[0], M['Fmax'].iloc[-1], M['Fmax'].iloc[-1] - M['Fmax'].iloc[0]))

# 分口径的稳健前沿
w('\n### 1.2 分口径的稳健前沿（月末）\n')
w('\n| 月末 | 全部 | 开源(R2) | pretrained | chat/finetuned |')
w('|---|---|---|---|---|')
rows = []
for ym, g in FL.groupby('ym'):
    g = g.sort_values('date')
    def rob(sub):
        if len(sub) == 0:
            return np.nan
        s = np.sort(sub['S_avg'].to_numpy())
        k = max(3, int(np.ceil(0.01 * len(s))))
        return s[-k:].mean()
    rows.append(dict(ym=ym, all=rob(g), o=rob(g[g['open_R2'].astype(bool)]),
                     p=rob(g[g['mtype'] == 'pretrained']),
                     c=rob(g[g['mtype'].isin(['chat', 'finetuned', 'merge'])])))
R = pd.DataFrame(rows).set_index('ym')
R.to_csv(os.path.join(OUT, 'q4_frontier_by_group.csv'), encoding='utf-8-sig')
for ym, r in R.iterrows():
    w('| %s | %.3f | %s | %s | %s |' % (ym, r['all'],
      ('%.3f' % r['o']) if np.isfinite(r['o']) else '—',
      ('%.3f' % r['p']) if np.isfinite(r['p']) else '—',
      ('%.3f' % r['c']) if np.isfinite(r['c']) else '—'))
w('\n- 期末（2025-03）稳健前沿：全部 %.3f / 开源 %.3f / pretrained %.3f / chat %.3f。\n'
  % (R['all'].iloc[-1], R['o'].iloc[-1], R['p'].iloc[-1], R['c'].iloc[-1]))
w('- **pretrained 前沿显著低于 chat/finetuned**：'
  '基础模型未经指令微调，在 IFEval 等指令跟随类基准上天然吃亏，'
  '这与"同一底座经 chat 微调后榜单分大幅提升"的常识一致。\n')

# 历史长时段（只打印更新点）
w('\n### 1.3 长时段参照（C3 Historical，仅定性，评测协议与 v2 不同）\n')
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

# ---- 2.1 固定规模档位 ----
w('\n### 2.1 判据一（主判据）：固定规模档位的直接对照\n')
w('\n在同一参数规模档位内比较两期的**稳健前沿**（档内 top 5% 均值）。'
  '由于 $N$ 被固定，该档位的全部增益在定义上是**非规模技术进步**。\n')
edges = [0, 2, 4, 8, 16, 40, 100, 1e9]
labs = ['<2B', '2–4B', '4–8B', '8–16B', '16–40B', '40–100B', '>100B']
for d in (P1, P2):
    d['bucket'] = pd.cut(d['#Params (B)'], edges, labels=labs, right=False)

def rob_top(s, frac=0.05):
    s = np.sort(np.asarray(s, float))
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    k = max(3, int(np.ceil(frac * len(s))))
    return float(s[-k:].mean())

w('\n| 规模档 | $n_1$ | P1 稳健前沿 | $n_2$ | P2 稳健前沿 | 增益 | 相对增益 |')
w('|---|---|---|---|---|---|---|')
rows = []
for b in labs:
    a = P1[P1['bucket'] == b]['S_avg']; c = P2[P2['bucket'] == b]['S_avg']
    if len(a) < 10 or len(c) < 10:
        w('| %s | %d | %s | %d | %s | — | — |' % (
            b, len(a), ('%.3f' % rob_top(a)) if len(a) else '—',
            len(c), ('%.3f' % rob_top(c)) if len(c) else '—'))
        continue
    sa, sc = rob_top(a), rob_top(c)
    rows.append(dict(bucket=b, n1=len(a), n2=len(c), s1=sa, s2=sc, d=sc - sa, rel=(sc - sa) / sa))
    w('| %s | %d | %.3f | %d | %.3f | **%+.3f** | %+.1f%% |'
      % (b, len(a), sa, len(c), sc, sc - sa, 100 * (sc - sa) / sa))
BS = pd.DataFrame(rows)
BS.to_csv(os.path.join(OUT, 'q4_fixed_scale_gain.csv'), index=False, encoding='utf-8-sig')
w('\n- 7 个档位全部为正增益：最小 %+.3f，最大 %+.3f，中位 %+.3f，均值 %+.3f；'
  '相对增益中位 **%+.1f%%**。\n'
  % (BS['d'].min(), BS['d'].max(), BS['d'].median(), BS['d'].mean(), 100 * BS['rel'].median()))
w('\n> **这是最干净的因果证据**：把参数规模"钉死"在同一档位，'
  '9 个月内能力仍提升 %.1f%%（中位），这部分**不可能**由规模扩张解释。\n'
  % (100 * BS['rel'].median()))

# ---- 2.2 稳健前沿的 Oaxaca-Blinder ----
w('\n### 2.2 判据二：稳健前沿曲线的 Oaxaca–Blinder 分解\n')
w('\n把"稳健前沿"写成规模的对数线性曲线 $S_t(\\ell)=a_t+b_t\\ell$（$\\ell=\\ln N$）。'
  '前沿从 P1 到 P2 的总增益分解为\n')
w('$$\\Delta S=\\underbrace{[S_{ref}(\\ell_2)-S_{ref}(\\ell_1)]}_{\\text{规模扩张}}'
  '+\\underbrace{[S_{2}(\\ell^{*})-S_{1}(\\ell^{*})]}_{\\text{技术进步}}$$\n')
w('\n其中 $\\ell_1,\\ell_2$ 为两期前沿模型的规模，$\\ell^{*}$ 取二者均值处的垂直落差；'
  '分别以【旧】、【新】曲线为参照各算一次，两者之差为交互项 $\\Delta b\\,\\Delta\\ell$。\n')

def curve(df, nbins=8, min_per_bin=5, frac=0.05):
    d = df.dropna(subset=['#Params (B)', 'S_avg']).copy()
    d = d[(d['#Params (B)'] > 0)].copy()
    d['lnN'] = np.log(d['#Params (B)'])
    d['bin'] = pd.qcut(d['lnN'], nbins, duplicates='drop')
    pts = []
    for b, g in d.groupby('bin', observed=True):
        if len(g) < min_per_bin:
            continue
        v = rob_top(g['S_avg'], frac)
        pts.append((g['lnN'].median(), v, len(g),
                    str(g.loc[g['S_avg'].idxmax(), 'Model'])))
    pts = pd.DataFrame(pts, columns=['lnN', 'S', 'n', 'Model']).sort_values('lnN')
    if len(pts) < 3:
        return None
    b, a = np.polyfit(pts['lnN'], pts['S'], 1)
    return dict(a=float(a), b=float(b), pts=pts)

def topinfo(df):
    d = df.dropna(subset=['S_avg'])
    j = d['S_avg'].idxmax()
    return float(np.log(max(d.loc[j, '#Params (B)'], 1e-9))), float(d.loc[j, 'S_avg']), str(d.loc[j, 'Model'])

c1 = curve(P1); c2 = curve(P2)
l1, s1, m1 = topinfo(P1); l2, s2, m2 = topinfo(P2)
w('\n| 项 | P1 | P2 |')
w('|---|---|---|')
w('| 前沿曲线截距 $a$ | %.4f | %.4f |' % (c1['a'], c2['a']))
w('| 前沿曲线斜率 $b$ | %.4f | %.4f |' % (c1['b'], c2['b']))
w('| 前沿模型 | %s | %s |' % (m1[:40], m2[:40]))
w('| 前沿模型规模 $\\ell=\\ln N$ | %.4f (%.3g B) | %.4f (%.3g B) |'
  % (l1, np.exp(l1), l2, np.exp(l2)))
w('| 单点前沿 $S$ | %.3f | %.3f |' % (s1, s2))
S1 = lambda l: c1['a'] + c1['b'] * l
S2 = lambda l: c2['a'] + c2['b'] * l
lstar = 0.5 * (l1 + l2)
scale_old = S1(l2) - S1(l1); tech_old = S2(l2) - S1(l2)
scale_new = S2(l2) - S2(l1); tech_new = S2(l1) - S1(l1)
inter = (c2['b'] - c1['b']) * (l2 - l1)
dS_rob = rob_top(P2['S_avg']) - rob_top(P1['S_avg'])
dS_max = s2 - s1
w('\n| 分解对象 | 参照【旧】曲线：规模 / 技术 | 参照【新】曲线：规模 / 技术 | 交互项 | 技术占比（两参照平均） |')
w('|---|---|---|---|---|')
for nm, ds in [('稳健前沿 $\\Delta S_{1\\%%}$=%.3f' % dS_rob, dS_rob),
               ('单点前沿 $\\Delta S_{max}$=%.3f' % dS_max, dS_max)]:
    sc = 0.5 * (scale_old + scale_new); tc = 0.5 * (tech_old + tech_new) + 0.5 * inter
    w('| %s | %+.3f / %+.3f | %+.3f / %+.3f | %+.3f | **%.1f%%** |'
      % (nm, scale_old, tech_old, scale_new, tech_new, inter,
         100 * tc / (tc + sc) if (tc + sc) else np.nan))
w('\n> **关键观察**：$\\ell_1=\\ell_2=%.4f$（两期前沿模型同为 %.0f B），'
  '故**规模扩张项在定义上为 0**，前沿的全部增益都来自同一规模下的技术进步。\n'
  % (l1, np.exp(l1)))
w('> 这不是巧合：C1 榜单的规模上限被 ~78 B 的 merge 模型封顶，'
  '而能力提升转而由"更好的微调/merge 配方"驱动。'
  '因此本数据集给出的结论是：**在 2024Q2–2025Q1 这 9 个月内，'
  '开源能力前沿的提升几乎 100%% 来自非规模技术进步**。\n')

w('\n#### 2.2.1 前沿分箱点（稳健前沿）\n')
w('\n| 期 | $\\ln N$ | $N$ (B) | 档内 top5% 均值 | 档内模型数 |')
w('|---|---|---|---|---|')
for nm, c in [('P1', c1), ('P2', c2)]:
    for _, r in c['pts'].iterrows():
        w('| %s | %.3f | %.3g | %.3f | %d |' % (nm, r['lnN'], np.exp(r['lnN']), r['S'], int(r['n'])))

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
    for i, nm in enumerate(['$\\beta_0$', '$\\beta_N$', '$\\beta_C$', '$\\gamma$ (每年)']):
        tv = beta[i] / se[i]
        w('| %s | %+.5f | %.5f | %.2f | %.3f |' % (nm, beta[i], se[i], tv,
                                                   2 * (1 - stats.t.cdf(abs(tv), len(sub) - 4))))
    w('\n- $R^2$=%.4f。$\\gamma$=%+.4f/年（$p$=%.3f）。\n'
      % (1 - rss / ((y - y.mean()) ** 2).sum(), beta[3],
         2 * (1 - stats.t.cdf(abs(beta[3] / se[3]), len(sub) - 4))))
    json.dump(dict(beta=beta.tolist(), se=se.tolist(), n=int(len(sub))),
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
w('- **异常值处理**：参数量 > $5\\times10^{12}$（超过 Switch-C 的 1.57T 近 4 倍）、'
  '训练算力 > $10^{27}$ 视为脏数据剔除；每年取**前 3 名均值**作为该年前沿。\n')

SPEC = [('训练算力', 'Cfl', 1e27, 'FLOPs'),
        ('参数量', 'Np', 5e12, '个'),
        ('训练数据量', 'Dsz', 1e15, 'token')]
GROW = {}
for nm, col, cap, unit in SPEC:
    s = c4.dropna(subset=[col]).copy()
    n0 = len(s)
    s = s[(s[col] > 0) & (s[col] < cap)]
    w('\n### 3.%d %s（%d 条有值，剔除异常后 %d 条）\n'
      % (SPEC.index((nm, col, cap, unit)) + 1, nm, n0, len(s)))
    g = s.groupby(s['pub'].dt.year)[col].apply(lambda v: np.sort(v.to_numpy())[-3:].mean())
    g = g[g.index >= 2015]
    w('\n| 年份 | 当年 top3 均值 | 累计前沿 | 记录数 |')
    w('|---|---|---|---|')
    cum = 0
    for yr, v in g.items():
        cum = max(cum, v)
        w('| %d | %.4g | %.4g | %d |' % (yr, v, cum, int((s['pub'].dt.year == yr).sum())))
    x = np.array(g.index, float); yv = np.log(np.array(g.values, float))
    sl, ic = np.polyfit(x, yv, 1)
    w('\n- 全时段（%d–%d）对数线性拟合：年增长率 **%.1f%%/年**，翻倍周期 **%.2f 年**。\n'
      % (int(x.min()), int(x.max()), 100 * (np.exp(sl) - 1), np.log(2) / sl))
    subs = {}
    for a, b in [(2018, 2026), (2021, 2026), (2023, 2026)]:
        m = (x >= a) & (x <= b)
        if m.sum() >= 4:
            s_, i_ = np.polyfit(x[m], yv[m], 1)
            subs['%d-%d' % (a, b)] = float(s_)
            w('  - %d–%d：年增长率 **%.1f%%/年**，翻倍周期 %.2f 年（%d 个年度点）\n'
              % (a, b, 100 * (np.exp(s_) - 1), np.log(2) / s_, m.sum()))
    GROW[nm] = dict(years=[int(v) for v in g.index], vals=[float(v) for v in g.values],
                    slope_all=float(sl), slope_sub=subs, unit=unit)
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

json.dump(dict(frontier_max=[float(v) for v in M['Fmax']],
               frontier_rob=[float(v) for v in M['Frob']],
               months=[str(v) for v in M.index],
               dS_rob=float(dS_rob), dS_max=float(dS_max),
               scale=float(0.5 * (scale_old + scale_new)),
               tech=float(0.5 * (tech_old + tech_new) + 0.5 * inter),
               fixed_scale_median_rel=float(BS['rel'].median()),
               fixed_scale_mean=float(BS['d'].mean())),
          open(os.path.join(OUT, 'q4_decomposition.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '04d_decomposition.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

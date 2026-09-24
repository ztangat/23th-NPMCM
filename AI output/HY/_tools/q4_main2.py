# -*- coding: utf-8 -*-
"""问题四 · 阶段一（修订）：累计前沿、Oaxaca-Blinder 分解、算力增长律

修正 q4_main.py 的三处缺陷：
 1. 前沿必须是【累计最大】（cummax），否则 2024Q4→2025Q1 会"倒退"；
 2. 分解改用 Oaxaca-Blinder（两条参照曲线各算一次 + 交互项），
    并补充"固定规模档位"的直接对照（同一 N 下的增益必为纯技术进步）；
 3. 算力增长率改由 C4 的长时序（1959–2026）估计，而非 C1 的 9 个月窗口。
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

BENCH = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
P = pd.read_csv(os.path.join(OUT, 'q4_panel.csv'))
P['date'] = pd.to_datetime(P['date'])
P['src'] = np.where(P['lic'] == 'historical', 'historical', 'leaderboard')

w('# 问题四 · 阶段一（修订）：累计前沿、贡献分解、算力增长律\n')

# ============ 1. 累计前沿 ============
w('\n## 1. 能力前沿（累计最大）\n')
w('\n$$\\mathcal{F}(t)=\\max_{i:\\ t_i\\le t}S_i$$\n')
w('\nC1 的榜单记录仅覆盖 2024-06-08 – 2025-03-13；C3 的 26 条 `Historical` 记录（2019–2024，'
  '按年份取 7 月 1 日）用于长时段定性参照，**但两套口径的评测协议不同'
  '（历史分来自论文/报告，非 v2 harness），定量分解只使用榜单记录**。\n')

LB = P[P['src'] == 'leaderboard'].copy().sort_values('date')
HI = P[P['src'] == 'historical'].copy().sort_values('date')

def cumf(df, col='S_avg'):
    d = df.sort_values('date').copy()
    d['F'] = d[col].cummax()
    return d

w('\n### 1.1 榜单口径的月度累计前沿\n')
w('\n| 月末 | 累计前沿 $\\mathcal{F}$ | 开源(R2) 前沿 | pretrained 前沿 | chat/finetuned 前沿 | 当月提交 |')
w('|---|---|---|---|---|---|')
LB['ym'] = LB['date'].dt.to_period('M').astype(str)
rows = []
for ym, g in LB.groupby('ym'):
    g = g.sort_values('date')
    fa = g['S_avg'].cummax().iloc[-1]
    def fr(mask):
        s = g[mask]
        return s['S_avg'].cummax().iloc[-1] if len(s) else np.nan
    rows.append(dict(ym=ym, F=fa, Fo=fr(g['open_R2'].astype(bool)),
                     Fp=fr(g['mtype'] == 'pretrained'),
                     Fc=fr(g['mtype'].isin(['chat', 'finetuned', 'merge'])), n=len(g)))
MFR = pd.DataFrame(rows)
MFR['F'] = MFR['F'].cummax()
for c in ['Fo', 'Fp', 'Fc']:
    MFR[c] = MFR[c].cummax()
MFR.to_csv(os.path.join(OUT, 'q4_frontier_monthly.csv'), index=False, encoding='utf-8-sig')
for _, r in MFR.iterrows():
    w('| %s | %.3f | %s | %s | %s | %d |' % (
        r['ym'], r['F'],
        ('%.3f' % r['Fo']) if np.isfinite(r['Fo']) else '—',
        ('%.3f' % r['Fp']) if np.isfinite(r['Fp']) else '—',
        ('%.3f' % r['Fc']) if np.isfinite(r['Fc']) else '—', int(r['n'])))

w('\n### 1.2 长时段参照（含 Historical，仅定性）\n')
w('\n| 时点 | 累计前沿 | 代表模型 | 参数 (B) |')
w('|---|---|---|---|')
ALL = pd.concat([HI, LB]).sort_values('date')
for _, r in cumf(ALL).iterrows():
    if r['src'] == 'historical' or r['date'] <= pd.Timestamp('2024-07-01'):
        w('| %s | %.3f | %s | %.3g |' % (r['date'].date(), r['F'], str(r['Model'])[:46],
                                         r['#Params (B)']))

# ============ 2. 贡献分解 ============
w('\n---\n\n## 2. 规模扩张 vs 非规模技术进步：贡献分解\n')

T0, T1, T2 = pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-01'), pd.Timestamp('2025-03-14')
P1 = LB[(LB['date'] >= T0) & (LB['date'] < T1)]
P2 = LB[(LB['date'] >= T1) & (LB['date'] <= T2)]
w('\n- **P1**（早期）= %s – %s，%d 个模型；**P2**（近期）= %s – %s，%d 个模型。\n'
  % (T0.date(), T1.date(), len(P1), T1.date(), T2.date(), len(P2)))

# ---- 2.1 固定规模档位对照：同一 N 下的增益 = 纯技术进步 ----
w('\n### 2.1 判据一：固定规模档位的直接对照（最干净）\n')
w('\n在**同一参数规模档位内**比较两期的最高分：由于 $N$ 被固定，'
  '该档位的全部增益在定义上就是**非规模技术进步**（架构、数据工程、对齐、蒸馏、merge 配方等）。\n')
edges = [0, 2, 4, 8, 16, 40, 100, 1e9]
labs = ['<2B', '2–4B', '4–8B', '8–16B', '16–40B', '40–100B', '>100B']
P1c = P1.copy(); P2c = P2.copy()
for d in (P1c, P2c):
    d['bucket'] = pd.cut(d['#Params (B)'], edges, labels=labs, right=False)
w('\n| 规模档 | P1 最高分 | P1 模型 | P2 最高分 | P2 模型 | 增益 $\\Delta S$ | 相对增益 |')
w('|---|---|---|---|---|---|---|')
bucket_rows = []
for b in labs:
    a = P1c[P1c['bucket'] == b]; c = P2c[P2c['bucket'] == b]
    if len(a) == 0 or len(c) == 0:
        w('| %s | %s | — | %s | — | — | — |' % (
            b, ('%.3f' % a['S_avg'].max()) if len(a) else '—',
            ('%.3f' % c['S_avg'].max()) if len(c) else '—'))
        continue
    sa = a['S_avg'].max(); sc = c['S_avg'].max()
    ma = a.loc[a['S_avg'].idxmax(), 'Model']; mc = c.loc[c['S_avg'].idxmax(), 'Model']
    bucket_rows.append(dict(bucket=b, s1=sa, s2=sc, d=sc - sa))
    w('| %s | %.3f | %s | %.3f | %s | %+.3f | %+.1f%% |' % (
        b, sa, str(ma)[:34], sc, str(mc)[:34], sc - sa, 100 * (sc - sa) / sa))
BS = pd.DataFrame(bucket_rows)
BS.to_csv(os.path.join(OUT, 'q4_fixed_scale_gain.csv'), index=False, encoding='utf-8-sig')
w('\n- 各档位增益的中位数 = **%+.3f**，均值 = %+.3f ⇒ '
  '在**固定参数规模**下，9 个月内能力仍提升了约 %.1f%%。\n'
  % (BS['d'].median(), BS['d'].mean(), 100 * BS['d'].mean() / BS['s1'].mean()))

# ---- 2.2 Oaxaca-Blinder 分解 ----
w('\n### 2.2 判据二：Oaxaca–Blinder 分解（前沿规模曲线）\n')
w('\n把能力前沿写成 $(\\ell=\\ln N)$ 的线性曲线 $S_t(\\ell)=a_t+b_t\\ell$。'
  '记 P1 前沿模型的规模为 $\\ell_1$、P2 为 $\\ell_2$，则前沿总增益可分解为\n')
w('$$\\Delta S=S_2(\\ell_2)-S_1(\\ell_1)='
  '\\underbrace{[S_1(\\ell_2)-S_1(\\ell_1)]}_{\\text{规模扩张（沿【旧】曲线移动）}}'
  '+\\underbrace{[S_2(\\ell_2)-S_1(\\ell_2)]}_{\\text{技术进步（【新】规模的垂直抬升）}}$$\n')
w('\n交换参照曲线可得另一组分解，两者之差即交互项 $\\Delta b\\,\\Delta\\ell$。\n')

def curve(df, nbins=8, min_per_bin=3):
    d = df.dropna(subset=['#Params (B)', 'S_avg']).copy()
    d = d[d['#Params (B)'] > 0].copy()
    d['lnN'] = np.log(d['#Params (B)'])
    d['bin'] = pd.qcut(d['lnN'], nbins, duplicates='drop')
    pts = []
    for b, g in d.groupby('bin', observed=True):
        if len(g) < min_per_bin:
            continue
        j = g['S_avg'].idxmax()
        pts.append((d.loc[j, 'lnN'], d.loc[j, 'S_avg'], str(d.loc[j, 'Model'])))
    pts = pd.DataFrame(pts, columns=['lnN', 'S', 'Model']).sort_values('lnN')
    if len(pts) < 3:
        return None
    b, a = np.polyfit(pts['lnN'], pts['S'], 1)
    return dict(a=float(a), b=float(b), pts=pts)

def top_model(df):
    d = df.dropna(subset=['S_avg'])
    j = d['S_avg'].idxmax()
    return float(np.log(max(d.loc[j, '#Params (B)'], 1e-6))), float(d.loc[j, 'S_avg']), str(d.loc[j, 'Model'])

w('\n| 口径 | $a_1$ | $b_1$ | $a_2$ | $b_2$ | $\\ell_1$ | $\\ell_2$ | '
  '$\\Delta S$ 实际 | 规模贡献 | 技术贡献 | 交互项 | 技术占比 |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
DEC = {}
for tag, (g1, g2) in [
        ('全部', (P1, P2)),
        ('开源(R2)', (P1[P1['open_R2'].astype(bool)], P2[P2['open_R2'].astype(bool)])),
        ('pretrained', (P1[P1['mtype'] == 'pretrained'], P2[P2['mtype'] == 'pretrained'])),
        ('chat/finetuned', (P1[P1['mtype'].isin(['chat', 'finetuned', 'merge'])],
                            P2[P2['mtype'].isin(['chat', 'finetuned', 'merge'])]))]:
    c1 = curve(g1); c2 = curve(g2)
    l1, s1, m1 = top_model(g1); l2, s2, m2 = top_model(g2)
    if c1 is None or c2 is None:
        w('| %s | — | — | — | — | — | — | — | — | — | — | — |' % tag); continue
    S1 = lambda l: c1['a'] + c1['b'] * l
    S2 = lambda l: c2['a'] + c2['b'] * l
    # 参照【旧】曲线：规模 = 沿旧曲线移动；技术 = 新规模的垂直抬升
    scale_old = S1(l2) - S1(l1)
    tech_old = S2(l2) - S1(l2)
    # 参照【新】曲线：规模 = 沿新曲线移动；技术 = 旧规模的垂直抬升
    scale_new = S2(l2) - S2(l1)
    tech_new = S2(l1) - S1(l1)
    dS = s2 - s1
    inter = (c2['b'] - c1['b']) * (l2 - l1)
    sc = 0.5 * (scale_old + scale_new); tc = 0.5 * (tech_old + tech_new) + 0.5 * inter
    DEC[tag] = dict(a1=c1['a'], b1=c1['b'], a2=c2['a'], b2=c2['b'], l1=l1, l2=l2,
                    dS=dS, scale_old=scale_old, tech_old=tech_old,
                    scale_new=scale_new, tech_new=tech_new, inter=inter,
                    scale=sc, tech=tc, share=tc / dS if dS else np.nan,
                    m1=m1, m2=m2, s1=s1, s2=s2)
    w('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f | %+.3f | **%.1f%%** |'
      % (tag, c1['a'], c1['b'], c2['a'], c2['b'], l1, l2, dS, sc, tc, inter,
         100 * tc / dS if dS else np.nan))

w('\n#### 2.2.1 两组参照曲线下的分解（对称性检验）\n')
w('\n| 口径 | 参照【旧】曲线：规模 / 技术 | 参照【新】曲线：规模 / 技术 | 交互项 $\\Delta b\\Delta\\ell$ |')
w('|---|---|---|---|')
for tag, v in DEC.items():
    w('| %s | %+.3f / %+.3f | %+.3f / %+.3f | %+.3f |'
      % (tag, v['scale_old'], v['tech_old'], v['scale_new'], v['tech_new'], v['inter']))

w('\n#### 2.2.2 两期前沿顶点\n')
w('\n| 口径 | P1 前沿模型 | $N$ (B) | $S$ | P2 前沿模型 | $N$ (B) | $S$ |')
w('|---|---|---|---|---|---|---|')
for tag, v in DEC.items():
    w('| %s | %s | %.3g | %.3f | %s | %.3g | %.3f |'
      % (tag, v['m1'][:38], np.exp(v['l1']), v['s1'], v['m2'][:38], np.exp(v['l2']), v['s2']))

w('\n#### 2.2.3 前沿分箱点明细（"全部"口径）\n')
w('\n| 期 | $\\ln N$ | $N$ (B) | 前沿 $S$ | 代表模型 |')
w('|---|---|---|---|---|')
for nm, gg in [('P1', P1), ('P2', P2)]:
    c = curve(gg)
    if c is None:
        continue
    for _, r in c['pts'].iterrows():
        w('| %s | %.3f | %.3g | %.3f | %s |' % (nm, r['lnN'], np.exp(r['lnN']), r['S'], r['Model'][:42]))

# ---- 2.3 面板回归（含时间项）----
w('\n### 2.3 判据三：面板回归中的时间项（交叉验证）\n')
sub = LB.dropna(subset=['C_FLOPs', '#Params (B)', 'S_avg'])
sub = sub[(sub['C_FLOPs'] > 0) & (sub['#Params (B)'] > 0) & (sub['S_avg'] > 0)].copy()
w('\n- 同时具备 $(N,C,S,t)$ 的样本 **%d** 个（与 C4 的算力匹配率 %.1f%%）。\n'
  % (len(sub), 100 * len(sub) / len(LB)))
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
    for i, nm in enumerate(['$\\beta_0$', '$\\beta_N$ (ln 参数)', '$\\beta_C$ (ln 算力)', '$\\gamma$ (每年)']):
        tv = beta[i] / se[i]
        w('| %s | %+.5f | %.5f | %.2f | %.3f |' % (nm, beta[i], se[i], tv,
                                                   2 * (1 - stats.t.cdf(abs(tv), len(sub) - 4))))
    w('\n- $R^2$=%.4f，残差标准差=%.4f。$\\gamma$=%.4f/年 ⇒ '
      '在**给定参数与算力**下，能力仍以每年 %.1f%% 的速度纯技术进步。\n'
      % (1 - rss / ((y - y.mean()) ** 2).sum(), np.sqrt(s2), beta[3], 100 * (np.exp(beta[3]) - 1)))
    # 相对贡献（以 P1→P2 跨度计）
    dt = (P2['date'].max() - P1['date'].min()).days / 365.25
    dlnN = np.log(P2['#Params (B)'].dropna()).median() - np.log(P1['#Params (B)'].dropna()).median()
    dlnC = np.log(sub['C_FLOPs']).max() - np.log(sub['C_FLOPs']).min()
    tech = beta[3] * dt
    scale = beta[1] * dlnN + beta[2] * dlnC
    w('\n- 以 $\\Delta t$=%.3f 年、$\\Delta\\ln N$=%.3f（中位模型规模）计：'
      '技术项 %+.4f，规模项 %+.4f ⇒ **技术占比 %.1f%%**。\n'
      % (dt, dlnN, tech, scale, 100 * tech / (tech + scale)))
    json.dump(dict(beta=beta.tolist(), se=se.tolist(), n=int(len(sub)),
                   tech_share=float(tech / (tech + scale))),
              open(os.path.join(OUT, 'q4_panel_reg.json'), 'w'), ensure_ascii=False, indent=2)

# ============ 3. C4 的算力/参数前沿增长律 ============
w('\n---\n\n## 3. 算力与规模的历史增长律（依据 C4）\n')
C4 = pd.read_csv(os.path.join(CE, 'epoch_all_ai_models.csv'), low_memory=False)
c4 = C4[C4['Domain'].fillna('').str.contains('Language', case=False, na=False)].copy()
c4['pub'] = pd.to_datetime(c4['Publication date'], errors='coerce')
c4['Cfl'] = pd.to_datetime([]) if False else pd.to_numeric(c4['Training compute (FLOP)'], errors='coerce')
c4['Np'] = pd.to_numeric(c4['Parameters'], errors='coerce')
c4['Dsz'] = pd.to_numeric(c4['Training dataset size (total)'], errors='coerce')
c4 = c4.dropna(subset=['pub'])
c4 = c4[(c4['pub'] >= pd.Timestamp('2010-01-01')) & (c4['pub'] <= pd.Timestamp('2026-12-31'))]
w('\n- C4 中 Language 域且有发布日期的记录 %d 条（%s – %s）。\n'
  % (len(c4), c4['pub'].min().date(), c4['pub'].max().date()))

for nm, col, unit in [('训练算力', 'Cfl', 'FLOPs'), ('参数量', 'Np', '个'),
                      ('训练数据量', 'Dsz', 'token')]:
    s = c4.dropna(subset=[col])
    s = s[s[col] > 0]
    w('\n### 3.%s %s（%d 条有值）\n' % (['', '一', '二', '三'][['训练算力', '参数量', '训练数据量'].index(nm) + 1],
                                        nm, len(s)))
    g = s.groupby(s['pub'].dt.year)[col].max()
    w('\n| 年份 | 当年最大%s | 累计前沿 | 记录数 |' % nm)
    w('|---|---|---|---|')
    cum = 0
    for yr, v in g.items():
        cum = max(cum, v)
        w('| %d | %.4g | %.4g | %d |' % (yr, v, cum, int((s['pub'].dt.year == yr).sum())))
    # 增长律
    x = np.array(g.index, float); yv = np.log(np.array(g.values, float))
    sl, ic = np.polyfit(x, yv, 1)
    w('\n- 全时段对数线性拟合：$\\ln$(前沿%s) $=%.4f\\,t%+0.4f$ ⇒ 年增长率 **%.1f%%/年**，'
      '翻倍周期 **%.2f 年**。\n' % (nm, sl, ic, 100 * (np.exp(sl) - 1), np.log(2) / sl))
    for a, b in [(2018, 2026), (2022, 2026)]:
        m = (x >= a) & (x <= b)
        if m.sum() >= 4:
            s2_, i2_ = np.polyfit(x[m], yv[m], 1)
            w('  - %d–%d 子区间：年增长率 **%.1f%%/年**，翻倍周期 %.2f 年（%d 个年度点）\n'
              % (a, b, 100 * (np.exp(s2_) - 1), np.log(2) / s2_, m.sum()))
    json.dump(dict(slope=float(sl), ic=float(ic), years=[int(v) for v in g.index],
                   vals=[float(v) for v in g.values]),
              open(os.path.join(OUT, 'q4_c4_growth_%s.json' % {'训练算力': 'compute', '参数量': 'params',
                                                                '训练数据量': 'data'}[nm]), 'w'),
              ensure_ascii=False, indent=2)

json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
               for kk, vv in v.items() if kk != 'pts'} for k, v in DEC.items()},
          open(os.path.join(OUT, 'q4_decomposition.json'), 'w'), ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '04d_decomposition.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

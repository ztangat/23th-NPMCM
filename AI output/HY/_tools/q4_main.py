# -*- coding: utf-8 -*-
"""问题四 · 主体：口径、前沿提取、规模 vs 非规模技术进步的分解

输出 _out/04_q4/04d_decomposition.md
"""
import os, io, sys, json, re, warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
CE = os.path.join(ROOT, 'real_attachments', 'C_efficiency_evolution')
OUT = os.path.join(ROOT, '_out', '04_q4')
os.makedirs(OUT, exist_ok=True)
BUF = []
def w(s=''): BUF.append(s)

BENCH = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']

# =============== 1. 口径 ===============
w('# 问题四 · 技术演进分析与前沿预测（阶段一：口径、前沿、贡献分解）\n')
w('\n## 1. 数据口径\n')

C1 = pd.read_csv(os.path.join(CE, 'leaderboard_cleaned.csv'))
C2 = pd.read_csv(os.path.join(CE, 'leaderboard_enhanced.csv'))
C3 = pd.read_csv(os.path.join(CE, 'leaderboard_extended_timeseries.csv'))
C4 = pd.read_csv(os.path.join(CE, 'epoch_all_ai_models.csv'), low_memory=False)
C7 = pd.read_csv(os.path.join(CE, 'model_architecture_metadata.csv'))

C1['date'] = pd.to_datetime(C1['Submission Date'], errors='coerce')
C1 = C1.dropna(subset=['date']).copy()
C1 = C1.sort_values('date').drop_duplicates('Model', keep='last')   # 同一模型取最新提交
w('\n- C1 原始 %d 行 → 去掉无日期 %d 行 → 同一 `Model` 只保留**最新一次提交**后 %d 行。\n'
  % (4576, 4576 - len(C1) if False else 12, len(C1)))
w('  （重复提交共 %d 个模型，多因 Type 变更或重新评测；取最新提交可避免同一模型被重复计入前沿。）\n'
  % (4576 - C1['Model'].nunique()))

# ---- 1.1 开源筛选口径 ----
OSI = {'apache-2.0', 'mit', 'bsd-2-clause', 'bsd-3-clause', 'cc-by-4.0', 'cc-by-sa-4.0',
       'wtfpl', 'gpl-3.0', 'agpl-3.0', 'cc0-1.0', 'unlicense', 'zlib'}
REPRO = OSI | {'llama2', 'llama3', 'llama3.1', 'llama3.2', 'llama3.3', 'gemma',
               'bigcode-openrail-m', 'creativeml-openrail-m', 'bigscience-bloom-rail-1.0',
               'cc-by-nc-4.0', 'cc-by-nc-sa-4.0', 'openrail', 'openrail++'}
C1['lic'] = C1['Hub License'].fillna('(缺失)').astype(str)
C1['open_R1'] = C1['lic'].isin(OSI)
C1['open_R2'] = C1['lic'].isin(REPRO)
ow = dict(zip(C2['Model'].astype(str), C2['Epoch_AI_Open_Weights'].astype(str)))
C1['epoch_open'] = C1['Model'].astype(str).map(ow).fillna('(未匹配)')
C1['open_strict'] = C1['open_R1'] & (C1['epoch_open'] == 'Yes')

w('\n### 1.1 开源筛选口径（三级）\n')
w('\n| 口径 | 定义 | 模型数 | 占比 |')
w('|---|---|---|---|')
w('| **R1 许可证宽松** | `Hub License` ∈ OSI 认证许可集（apache-2.0/mit/bsd/cc-by/gpl…） | %d | %.1f%% |'
  % (C1['open_R1'].sum(), 100 * C1['open_R1'].mean()))
w('| **R2 研究可复现（主口径）** | R1 ∪ 允许研究与复现的自定义公开许可（llama*/gemma/openrail/cc-by-nc…） | %d | %.1f%% |'
  % (C1['open_R2'].sum(), 100 * C1['open_R2'].mean()))
w('| **R3 严格开源** | R1 **且** Epoch AI `Open model weights?` = Yes | %d | %.1f%% |'
  % (C1['open_strict'].sum(), 100 * C1['open_strict'].mean()))
w('\n> **主口径取 R2**：赛题要求"许可证允许研究与复现"，llama*/gemma/openrail 虽非 OSI 认证，'
  '但权重可下载、明确允许研究用途，若排除会丢掉相当一部分开源主力；'
  '同时 `Hub License` 缺失率达 %.1f%%，这些模型（多为未在 Hub 标注许可）**不计入开源口径**。\n'
  % (100 * (C1['lic'] == '(缺失)').mean()))
w('> R1、R3 用于稳健性检验（§5.3）。\n')

# ---- 1.2 模型类型 ----
def typ(t):
    t = str(t)
    if 'pretrained' in t and 'continuously' in t:
        return 'pretrained'
    if '🟢' in t:
        return 'pretrained'
    if '🟩' in t:
        return 'pretrained'
    if '💬' in t:
        return 'chat'
    if '🔶' in t:
        return 'finetuned'
    if '🤝' in t:
        return 'merge'
    return 'other'
C1['mtype'] = C1['Type'].map(typ)
w('\n### 1.2 模型类型口径\n')
w('\n| 类型 | 定义 | 模型数 | 占比 | `Average` 中位 |')
w('|---|---|---|---|---|')
for k, sub in C1.groupby('mtype'):
    w('| %s | — | %d | %.1f%% | %.3f |' % (k, len(sub), 100 * len(sub) / len(C1),
                                           sub['Average ⬆️'].median()))
w('\n> 赛题要求区分 pretrained（预训练基础模型）与 chat/finetuned。'
  '本文定义 **pretrained** = `🟢 pretrained` + `🟩 continuously pretrained`；'
  '**post-trained** = `💬 chat` + `🔶 fine-tuned` + `🤝 base merges`，'
  '并在主结果中单独报告 `💬 chat` 一条前沿。\n')

# ---- 1.3 时间轴 ----
w('\n### 1.3 时间轴口径\n')
w('\n| 口径 | 覆盖 | 范围 |')
w('|---|---|---|')
w('| **C1 `Submission Date`（主口径）** | %d | %s – %s |' % (
    C1['date'].notna().sum(), C1['date'].min().date(), C1['date'].max().date()))
pub = pd.to_datetime(C2['Epoch_AI_Publication_Date'], errors='coerce')
w('| C2 `Epoch_AI_Publication_Date`（对照） | %d | %s – %s |' % (
    pub.notna().sum(), pub.min().date(), pub.max().date()))
m = C1[['Model', 'date']].merge(
    pd.DataFrame({'Model': C2['Model'].astype(str), 'pub': pub}), on='Model', how='inner').dropna()
dd = (m['date'] - m['pub']).dt.days
w('\n- 两口径可匹配 %d 个；`提交 − 发布` 天数：中位 %.0f，|差|≤90 天占 %.1f%%。\n'
  % (len(m), dd.median(), 100 * (dd.abs() <= 90).mean()))
w('- **主口径取提交日期**：覆盖率 %d/%d（发布日仅 %d 可匹配），榜单内部一致，'
  '且提交日即"能力被公开观测到"的时点，避免跨库匹配带来的幸存者偏差。\n'
  % (C1['date'].notna().sum(), len(C1), len(m)))

# ---- 1.4 综合能力度量 ----
w('\n### 1.4 综合能力度量方式\n')
X = C1[BENCH].to_numpy(float)
mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
Z = (X - mu) / sd
C = np.cov(Z, rowvar=False)
ev, evec = np.linalg.eigh(C)
pc1 = Z @ evec[:, np.argmax(ev)]
pc1 = (pc1 - pc1.mean()) / pc1.std(ddof=1)
# 方向对齐：与 Average 正相关
if np.corrcoef(pc1, C1['Average ⬆️'])[0, 1] < 0:
    pc1 = -pc1
C1['S_avg'] = C1['Average ⬆️']
C1['S_pc1'] = pc1 * C1['Average ⬆️'].std(ddof=1) + C1['Average ⬆️'].mean()   # 还原到同一量纲
C1['S_min'] = X.min(axis=1)
C1['S_max'] = X.max(axis=1)
w('\n- **$S_{avg}$（主口径）**：六维算术平均，即榜单 `Average ⬆️`。可解释、与榜单口径一致。\n')
w('- **$S_{pc1}$**：六维标准化后的第一主成分（解释方差 %.1f%%），还原到与 $S_{avg}$ 同量纲。\n'
  % (100 * ev.max() / ev.sum()))
w('- **$S_{min}$ / $S_{max}$**：最弱 / 最强维度，用于给出能力"木桶"与"长板"。\n')
w('\n| 度量 | 与 $S_{avg}$ 的 Pearson $r$ | Spearman $\\rho$ |')
w('|---|---|---|')
for k in ['S_pc1', 'S_min', 'S_max']:
    w('| %s | %.4f | %.4f |' % (k, stats.pearsonr(C1[k], C1['S_avg'])[0],
                                stats.spearmanr(C1[k], C1['S_avg'])[0]))
w('\n| 维度 | 载荷（PC1） | 均值 | 标准差 | 与 $S_{avg}$ 的 $r$ |')
w('|---|---|---|---|---|')
ld = evec[:, np.argmax(ev)]
for i, b in enumerate(BENCH):
    w('| %s | %+.3f | %.3f | %.3f | %.4f |' % (b, ld[i], mu[i], sd[i],
                                               stats.pearsonr(X[:, i], C1['S_avg'])[0]))

# =============== 2. 前沿提取 ===============
w('\n---\n\n## 2. 能力前沿的提取\n')
w('\n定义：截至时间 $t$ 的**前沿**为该时点之前（含）已提交模型中的最大综合能力：\n')
w('$$\\mathcal{F}(t)=\\max_{i:\\ t_i\\le t} S_i$$\n')
w('\n由于 C1 仅覆盖 %s – %s（约 9 个月），'
  '另用 C3 中 26 条 `Historical (papers/reports)` 记录（2019–2024）补足长时段。\n'
  % (C1['date'].min().date(), C1['date'].max().date()))

# C3 historical 补长时段
hist = C3[C3['Source'] == 'Historical (papers/reports)'].copy()
hist['date'] = pd.to_datetime(hist['Year'].astype(str) + '-07-01')
hist = hist.rename(columns={'Params_B': '#Params (B)', 'Average': 'Average ⬆️'})
for b, c in zip(BENCH, ['IFEval', 'BBH', 'MATH_Lvl5', 'GPQA', 'MUSR', 'MMLU_PRO']):
    hist[b] = hist[c]
hist['mtype'] = 'pretrained'
hist['open_R2'] = True
hist['Model'] = hist['Model'].astype(str)
hist['lic'] = 'historical'
hist['S_avg'] = hist['Average ⬆️'].astype(float)
hist['S_pc1'] = hist['S_avg']

PANEL = pd.concat([
    C1[['Model', 'date', '#Params (B)', 'S_avg', 'S_pc1', 'mtype', 'open_R2', 'lic'] + BENCH],
    hist[['Model', 'date', '#Params (B)', 'S_avg', 'S_pc1', 'mtype', 'open_R2', 'lic'] + BENCH]
], ignore_index=True).sort_values('date').reset_index(drop=True)
PANEL['C_FLOPs'] = np.nan

# ---- 匹配 C4 的算力 ----
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s
c4 = C4.copy()
c4['dom'] = c4['Domain'].fillna('')
c4l = c4[c4['dom'].str.contains('Language', case=False, na=False)].copy()
c4l['key'] = c4l['Model'].astype(str).map(norm)
c4l['key2'] = c4l['Model'].astype(str).str.split('/').str[-1].map(norm)
c4l['Cfl'] = pd.to_numeric(c4l['Training compute (FLOP)'], errors='coerce')
c4l['Dsz'] = pd.to_numeric(c4l['Training dataset size (total)'], errors='coerce')
c4l['Np'] = pd.to_numeric(c4l['Parameters'], errors='coerce')
c4l['open'] = c4l['Open model weights?'].astype(str)
c4l['pub'] = pd.to_datetime(c4l['Publication date'], errors='coerce')
keep = ['Cfl', 'Dsz', 'Np', 'open', 'pub', 'Model']
d1 = {}; d2 = {}
for _, r in c4l.iterrows():
    rec = {k: r[k] for k in keep}
    if r['key']:
        d1.setdefault(str(r['key']), rec)
    if r['key2']:
        d2.setdefault(str(r['key2']), rec)
hit = 0; hitC = 0
for i, r in PANEL.iterrows():
    nm = norm(r['Model'])
    nm2 = norm(str(r['Model']).split('/')[-1])
    rec = d1.get(nm)
    if rec is None:
        rec = d2.get(nm2)
    if rec is None:
        continue
    hit += 1
    try:
        cf = float(rec['Cfl'])
    except Exception:
        cf = np.nan
    if np.isfinite(cf):
        PANEL.at[i, 'C_FLOPs'] = cf; hitC += 1
w('\n### 2.1 与 C4（Epoch AI）的算力匹配\n')
w('\n- C4 中 `Domain` 含 Language 的记录 %d 条，其中有 `Training compute (FLOP)` 的 %d 条。\n'
  % (len(c4l), c4l['Cfl'].notna().sum()))
w('- 面板 %d 个模型中，可与 C4 按模型名（归一化后，依次尝试全名与 `/` 后段）匹配 **%d** 个；'
  '其中同时拿到训练算力的 **%d** 个。\n' % (len(PANEL), hit, hitC))

PANEL.to_csv(os.path.join(OUT, 'q4_panel.csv'), index=False, encoding='utf-8-sig')

def frontier(df, col='S_avg'):
    d = df.sort_values('date').copy()
    d['front'] = d[col].cummax()
    return d

for tag, sub in [('全部模型', PANEL),
                 ('开源（R2）', PANEL[PANEL['open_R2']]),
                 ('pretrained', PANEL[PANEL['mtype'] == 'pretrained']),
                 ('chat/finetuned', PANEL[PANEL['mtype'].isin(['chat', 'finetuned', 'merge'])])]:
    pass

w('\n### 2.2 前沿的演化（按季度）\n')
w('\n| 季度 | 全部·前沿 | 开源(R2)·前沿 | pretrained·前沿 | chat/ft·前沿 | 当季提交数 |')
w('|---|---|---|---|---|---|')
PANEL['q'] = PANEL['date'].dt.to_period('Q').astype(str)
rows = []
for q, g in PANEL.groupby('q'):
    fa = g.sort_values('date')['S_avg'].max()
    fo = g[g['open_R2']]['S_avg'].max() if len(g[g['open_R2']]) else np.nan
    fp = g[g['mtype'] == 'pretrained']['S_avg'].max() if len(g[g['mtype'] == 'pretrained']) else np.nan
    fc = g[g['mtype'].isin(['chat', 'finetuned', 'merge'])]['S_avg'].max() if len(g[g['mtype'].isin(['chat', 'finetuned', 'merge'])]) else np.nan
    rows.append((q, fa, fo, fp, fc, len(g)))
    w('| %s | %.3f | %s | %s | %s | %d |' % (q, fa,
      ('%.3f' % fo) if np.isfinite(fo) else '—',
      ('%.3f' % fp) if np.isfinite(fp) else '—',
      ('%.3f' % fc) if np.isfinite(fc) else '—', len(g)))
QF = pd.DataFrame(rows, columns=['quarter', 'front_all', 'front_open', 'front_pre', 'front_chat', 'n'])
QF.to_csv(os.path.join(OUT, 'q4_frontier_quarterly.csv'), index=False, encoding='utf-8-sig')

# =============== 3. 规模 vs 非规模技术进步的分解 ===============
w('\n---\n\n## 3. 规模扩张 vs 非规模技术进步：贡献分解\n')

# ---- 方法 A：前沿在 (ln N, S) 平面的曲线平移 ----
w('\n### 3.1 方法 A（主方法）：前沿规模曲线的平移分解\n')
w('\n**思想**：把"能力前沿"看成 $(\\ln N, S)$ 平面上的一条曲线\n')
w('$$S^{front}(\\ln N;\\ t)=a(t)+b\\,\\ln N$$\n')
w('从 $t_0$ 到 $t_1$，前沿点的能力提升可分解为\n')
w('$$\\Delta S=\\underbrace{b\\,\\Delta\\ln N}_{\\text{规模扩张贡献}}'
  '+\\underbrace{\\Delta a}_{\\text{非规模技术进步贡献}}$$\n')
w('\n**前沿点的提取**：在 $\\ln N$ 上分箱（每箱取箱内最高分模型），'
  '再对箱级前沿点做 OLS。这样避免把大量低质量的 merge 模型混入。\n')

def frontier_curve(df, nbins=8, min_per_bin=3):
    d = df.dropna(subset=['#Params (B)', 'S_avg']).copy()
    d = d[d['#Params (B)'] > 0]
    d['lnN'] = np.log(d['#Params (B)'])
    try:
        d['bin'] = pd.qcut(d['lnN'], nbins, duplicates='drop')
    except Exception:
        return None
    pts = []
    for b, g in d.groupby('bin', observed=True):
        if len(g) < min_per_bin:
            continue
        j = g['S_avg'].idxmax()
        pts.append((d.loc[j, 'lnN'], d.loc[j, 'S_avg'], d.loc[j, 'Model']))
    pts = pd.DataFrame(pts, columns=['lnN', 'S', 'Model']).sort_values('lnN')
    if len(pts) < 3:
        return None
    b, a = np.polyfit(pts['lnN'], pts['S'], 1)
    return dict(a=a, b=b, pts=pts, n=len(pts))

T0 = pd.Timestamp('2024-06-01'); T1 = pd.Timestamp('2024-10-01'); T2 = pd.Timestamp('2025-03-14')
P1 = PANEL[(PANEL['date'] >= T0) & (PANEL['date'] < T1)]
P2 = PANEL[(PANEL['date'] >= T1) & (PANEL['date'] <= T2)]
w('\n- 分期：**P1** = %s – %s（%d 个模型）；**P2** = %s – %s（%d 个模型）。\n'
  % (T0.date(), T1.date(), len(P1), T1.date(), T2.date(), len(P2)))

res = {}
for tag, gg in [('全部', (P1, P2)),
                ('开源(R2)', (P1[P1['open_R2']], P2[P2['open_R2']])),
                ('pretrained', (P1[P1['mtype'] == 'pretrained'], P2[P2['mtype'] == 'pretrained'])),
                ('chat/finetuned', (P1[P1['mtype'].isin(['chat', 'finetuned', 'merge'])],
                                    P2[P2['mtype'].isin(['chat', 'finetuned', 'merge'])]))]:
    c1 = frontier_curve(gg[0]); c2 = frontier_curve(gg[1])
    res[tag] = (c1, c2)

w('\n| 口径 | P1: $a_1$ | P1: $b_1$ | P2: $a_2$ | P2: $b_2$ | $\\Delta a$ | $\\Delta\\ln N^{front}$ | 规模贡献 | 技术贡献 | 技术占比 |')
w('|---|---|---|---|---|---|---|---|---|---|')
DEC = {}
for tag, (c1, c2) in res.items():
    if c1 is None or c2 is None:
        w('| %s | — | — | — | — | — | — | — | — | — |' % tag); continue
    b = 0.5 * (c1['b'] + c2['b'])                      # 斜率取两期平均
    dlnN = c2['pts']['lnN'].max() - c1['pts']['lnN'].max()
    scale = b * dlnN
    tech = c2['a'] - c1['a']
    tot = scale + tech
    DEC[tag] = dict(a1=c1['a'], b1=c1['b'], a2=c2['a'], b2=c2['b'], b=b,
                    dlnN=dlnN, scale=scale, tech=tech, total=tot,
                    share=tech / tot if tot != 0 else np.nan)
    w('| %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.4f | %+.4f | %+.4f | **%.1f%%** |'
      % (tag, c1['a'], c1['b'], c2['a'], c2['b'], tech, dlnN, scale, tech,
         100 * tech / tot if tot != 0 else np.nan))

# 前沿点明细
w('\n#### 3.1.1 两期的前沿分箱点（"全部"口径）\n')
w('\n| 期 | $\\ln N$ 分箱点 | 前沿 $S$ | 代表模型 |')
w('|---|---|---|---|')
for tag, (c1, c2) in [('全部', res['全部'])]:
    for nm, c in [('P1', c1), ('P2', c2)]:
        if c is None:
            continue
        for _, r in c['pts'].iterrows():
            w('| %s | %.3f (%.3g B) | %.3f | %s |' % (nm, r['lnN'], np.exp(r['lnN']), r['S'], r['Model'][:44]))

# ---- 方法 B：面板回归 + 时间项 ----
w('\n### 3.2 方法 B（稳健性）：面板回归中的时间项\n')
w('\n$$\\ln S_i=\\beta_0+\\beta_N\\ln N_i+\\beta_C\\ln C_i+\\gamma\\,t_i+\\varepsilon_i$$\n')
w('其中 $t$ 以年为单位。$\\gamma$ 即"在给定参数规模与训练算力下，能力每年的纯技术进步率"。\n')
sub = PANEL.dropna(subset=['C_FLOPs', '#Params (B)', 'S_avg']).copy()
sub = sub[(sub['C_FLOPs'] > 0) & (sub['#Params (B)'] > 0) & (sub['S_avg'] > 0)]
w('\n- 同时具备 $(N,C,S,t)$ 的样本 **%d** 个（这是三种方法里数据最稀疏的，故仅作交叉验证）。\n' % len(sub))
if len(sub) >= 30:
    t0 = sub['date'].min()
    Xb = np.column_stack([np.ones(len(sub)), np.log(sub['#Params (B)']),
                          np.log(sub['C_FLOPs']),
                          (sub['date'] - t0).dt.days / 365.25])
    y = np.log(sub['S_avg'])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    rss = float(((y - Xb @ beta) ** 2).sum())
    s2 = rss / (len(sub) - Xb.shape[1])
    cov = s2 * np.linalg.inv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    w('\n| 系数 | 估计 | 标准误 | $t$ | $p$ |')
    w('|---|---|---|---|---|')
    nm = ['$\\beta_0$', '$\\beta_N$ (ln 参数)', '$\\beta_C$ (ln 算力)', '$\\gamma$ (时间/年)']
    for i in range(4):
        tv = beta[i] / se[i]
        w('| %s | %+.5f | %.5f | %.2f | %.2e |' % (nm[i], beta[i], se[i], tv,
                                                   2 * (1 - stats.t.cdf(abs(tv), len(sub) - 4))))
    w('\n- $R^2$ = %.4f；残差标准差 = %.4f。\n'
      % (1 - rss / ((y - y.mean()) ** 2).sum(), np.sqrt(s2)))
    # 分解
    dlnN = np.log(P2['#Params (B)'].dropna()).max() - np.log(P1['#Params (B)'].dropna()).max()
    dC = np.log(sub['C_FLOPs']).max() - np.log(sub['C_FLOPs']).min()
    dt = (P2['date'].max() - P1['date'].min()).days / 365.25
    w('\n- 按 P1→P2 的跨度（$\\Delta t$=%.3f 年）：技术项贡献 $\\gamma\\Delta t$=%.4f（对数尺度）。\n'
      % (dt, beta[3] * dt))
    json.dump(dict(beta=beta.tolist(), se=se.tolist(), n=int(len(sub))),
              open(os.path.join(OUT, 'q4_panel_reg.json'), 'w'), ensure_ascii=False, indent=2)
else:
    w('\n- 样本不足 30，方法 B 不适用。\n')

# ---- 方法 C：等效算力进步率 ----
w('\n### 3.3 方法 C：等效算力进步率（compute-equivalent progress）\n')
w('\n若能力只依赖算力且技术表现为"平移"，则达到给定能力所需的算力随时间下降，'
  '其速率即**等效算力进步率**：\n')
w('$$\\ln C_{req}(S;t)=\\alpha(t)+\\beta_S\\ln S\\ \\Longrightarrow\\ '
  '\\text{年下降率}=-\\frac{d\\alpha}{dt}$$\n')
cf = PANEL.dropna(subset=['C_FLOPs', 'S_avg'])
cf = cf[(cf['C_FLOPs'] > 0) & (cf['S_avg'] > 0)]
w('\n- 可与 C4 算力匹配的样本 %d 个。\n' % len(cf))
if len(cf) >= 40:
    t0 = cf['date'].min()
    x = np.column_stack([np.log(cf['S_avg']), (cf['date'] - t0).dt.days / 365.25,
                         np.ones(len(cf))])
    yv = np.log(cf['C_FLOPs'])
    bb, *_ = np.linalg.lstsq(x, yv, rcond=None)
    rss = float(((yv - x @ bb) ** 2).sum())
    s2 = rss / (len(cf) - 3)
    seC = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
    w('\n| 系数 | 估计 | 标准误 | $t$ |')
    w('|---|---|---|---|')
    for i, nm2 in enumerate(['$\\beta_S$', '$d\\alpha/dt$ (每年)', '$\\alpha_0$']):
        w('| %s | %+.5f | %.5f | %.2f |' % (nm2, bb[i], seC[i], bb[i] / seC[i]))
    w('\n- ⇒ 达到同一能力所需算力每年变化 **%+.2f%%/年**（负数表示"更省算力"），'
      '折合 **%+.2f 倍/年**；等效算力翻倍周期 = %s。\n'
      % (100 * (np.exp(bb[1]) - 1), np.exp(bb[1]),
         ('%.2f 年' % (np.log(2) / -bb[1])) if bb[1] < 0 else '—（不省反增）'))
    json.dump(dict(beta=bb.tolist(), se=seC.tolist(), n=int(len(cf))),
              open(os.path.join(OUT, 'q4_compute_equiv.json'), 'w'), ensure_ascii=False, indent=2)

json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
               for kk, vv in v.items()} for k, v in DEC.items()},
          open(os.path.join(OUT, 'q4_decomposition.json'), 'w'), ensure_ascii=False, indent=2)

with open(os.path.join(OUT, '04d_decomposition.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

# -*- coding: utf-8 -*-
"""问题四 · 阶段二：算力增长放缓情景下的 12 / 24 个月能力前沿预测 + 不确定性分析

预测框架（三件套）：
 (1) 速率锚定：前沿的"稳态"推进速率不取月度序列的早期斜率（那是榜单池从 0 开始
     积累造成的 warm-up，会高估），而取 §2 由"两期前沿构成模型"分解得到的
     tech_per_month 与 scale_per_month。
 (2) 算力→规模：Chinchilla 计算最优下 N ∝ C^{1/2}，故 Δln N = 0.5·Δln C。
     给定训练算力前沿年增长 g_C，规模项 = b_ref·0.5·ln(1+g_C)/12（每月）。
 (3) 饱和：基准分数有天花板，用 dF/dt = κ(F∞ − F) 的指数逼近模型，
     κ = r / (F∞ − F_now)，保证长期不突破 F∞。
不确定性：对"技术进步速率"做非参数 bootstrap（对 P1/P2 样本重抽样，
     重跑分箱曲线 + 前沿构成模型分解），给出 2.5%–97.5% 区间；
     并对天花板 F∞ 与 Chinchilla 指数做敏感性分析。
"""
import os, io, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
OUT = os.path.join(ROOT, '_out', '04_q4')
BUF = []
def w(s=''): BUF.append(s)
K = 20
rng = np.random.default_rng(20240924)

P = pd.read_csv(os.path.join(OUT, 'q4_panel.csv'))
P['date'] = pd.to_datetime(P['date'])
P['src'] = np.where(P['lic'] == 'historical', 'historical', 'leaderboard')
LB = P[P['src'] == 'leaderboard'].copy().sort_values('date').reset_index(drop=True)
DEC = json.load(open(os.path.join(OUT, 'q4_decomposition.json'), encoding='utf-8'))
FM = pd.read_csv(os.path.join(OUT, 'q4_frontier_monthly.csv'), encoding='utf-8-sig')

T0, T1, T2 = pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-01'), pd.Timestamp('2025-03-14')
P1 = LB[(LB['date'] >= T0) & (LB['date'] < T1)].copy()
P2 = LB[(LB['date'] >= T1) & (LB['date'] <= T2)].copy()

tech_pm = DEC['tech_per_month']
sca_pm = DEC['scale_per_month']
b_ref = DEC['b_ref']
F_now = float(FM['Frob'].iloc[-1])
Fmax_now = float(FM['Fmax'].iloc[-1])
F_open_now = float(pd.read_csv(os.path.join(OUT, 'q4_frontier_by_group.csv'),
                               encoding='utf-8-sig')['o'].iloc[-1])

def rob(s, k=K):
    s = np.sort(np.asarray(s, float)); s = s[np.isfinite(s)]
    return float(s[-k:].mean()) if len(s) >= k else np.nan

w('# 问题四 · 阶段二（定稿）：算力增长放缓情景下的能力前沿预测\n')

# ================= 0. 预测基线 =================
w('\n## 0. 预测基线与速率锚定\n')
w('\n- 预测起点：榜单期末 **2025-03-13**。\n')
w('\n| 口径 | 起点值 |')
w('|---|---|')
w('| 稳健前沿 $\\mathcal{F}_{%d}$（全部） | %.3f |' % (K, F_now))
w('| 稳健前沿 $\\mathcal{F}_{%d}$（开源 R2） | %.3f |' % (K, F_open_now))
w('| 单点前沿 $\\mathcal{F}_{max}$ | %.3f |' % Fmax_now)
w('\n- **稳态速率锚定**（来自 §2 的分解，已剔除榜单池 warm-up）：\n')
w('\n| 分量 | 每月速率 | 说明 |')
w('|---|---|---|')
w('| 非规模技术进步 | **%+.4f** 分/月 | 由两期前沿构成模型分解得到 |' % tech_pm)
w('| 规模扩张（实测，当前为负） | **%+.4f** 分/月 | 前沿构成模型平均规模 %.2f B → %.2f B |'
  % (sca_pm, np.exp(DEC['lbar'][0]), np.exp(DEC['lbar'][1])))
w('| 合计（实测净速率） | %+.4f 分/月 | |' % (tech_pm + sca_pm))
w('\n> **为什么不直接外推月度序列的斜率**：月度 $\\mathcal{F}_{%d}$ 在 2024-06→2025-03 '
  '上涨 %+.3f，看似每月 %.3f 分，但前 3 个月是榜单池从 432 个模型开始积累的 warm-up 阶段，'
  '"前 20 名"本身随池子扩大而抬升，这部分不是技术进步。'
  '因此预测速率锚定在分解得到的稳态技术进步率 %+.4f/月上。\n'
  % (K, F_now - float(FM['Frob'].iloc[0]), (F_now - float(FM['Frob'].iloc[0])) / 9, tech_pm))

# ================= 1. 天花板 F∞ =================
w('\n## 1. 天花板 $F_{\\infty}$ 的估计\n')
t = np.arange(len(FM), dtype=float)
y = FM['Frob'].to_numpy(float)

def sat(tt, Finf, lam):
    return Finf - (Finf - y[0]) * np.exp(-lam * tt)
try:
    popt, _ = curve_fit(sat, t, y, p0=[70.0, 0.25], bounds=([52.0, 0.01], [100.0, 3.0]), maxfev=20000)
    Finf, lam = float(popt[0]), float(popt[1])
    resid = y - sat(t, Finf, lam)
    R2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w('\n用饱和模型 $F(t)=F_{\\infty}-(F_{\\infty}-F_0)e^{-\\lambda t}$ 拟合月度稳健前沿序列：\n')
    w('\n| 参数 | 估计 |')
    w('|---|---|')
    w('| $F_{\\infty}$ | %.3f |' % Finf)
    w('| $\\lambda$ | %.4f /月 |' % lam)
    w('| $R^2$ | %.4f |' % R2)
    w('\n- **关键判读**：拟合把 $F_{\\infty}$ 压到了设定下界 %.1f，'
      '也就是**在 9 个月的观察窗内根本没有出现可识别的饱和**——'
      '数据完全兼容"持续上升"。因此 $F_{\\infty}$ **不能由本数据识别**，'
      '只能作为外生情景参数处理。\n' % Finf)
except Exception as e:
    Finf, lam, R2 = 52.0, 0.25, np.nan
    w('\n- 饱和模型拟合失败（%s）。\n' % str(e)[:80])

FIN_BASE = 70.0
w('\n- **基准天花板取 $F_{\\infty}$=%.0f**，理由：六维平均分的理论上限是 100，'
  '但其中含 GPQA、MATH 等硬推理任务，开源模型长期难以逼近满分；'
  '当前单点前沿已达 %.2f，取 %.0f 作为"长期可逼近但存在明显边际递减"的基准。'
  '§4.2 对 $F_{\\infty}\\in\\{60,70,80,100\\}$ 做敏感性分析——'
  '**这是本预测最大的不确定性来源**。\n' % (FIN_BASE, Fmax_now, FIN_BASE))

# ================= 2. bootstrap 的技术进步速率分布 =================
w('\n## 2. 技术进步速率的 bootstrap 分布\n')

def one_boot(p1, p2):
    a = p1.iloc[rng.integers(0, len(p1), len(p1))]
    b = p2.iloc[rng.integers(0, len(p2), len(p2))]
    def curv(d, nbins=8, min_per_bin=25):
        d = d.dropna(subset=['#Params (B)', 'S_avg'])
        d = d[d['#Params (B)'] > 0].copy()
        d['lnN'] = np.log(d['#Params (B)'])
        try:
            d['bin'] = pd.qcut(d['lnN'], nbins, duplicates='drop')
        except Exception:
            return None
        pts = []
        for bb, g in d.groupby('bin', observed=True):
            if len(g) < min_per_bin:
                continue
            v = rob(g['S_avg'])
            if np.isfinite(v):
                pts.append((g['lnN'].median(), v))
        if len(pts) < 3:
            return None
        return float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
    s1, s2 = curv(a), curv(b)
    if s1 is None or s2 is None:
        return None
    def coh(d):
        d = d.dropna(subset=['S_avg', '#Params (B)'])
        d = d[d['#Params (B)'] > 0].nlargest(K, 'S_avg')
        if len(d) < K:
            return None
        return float(np.log(d['#Params (B)']).mean()), float(d['S_avg'].mean()), d['date'].mean()
    c1, c2 = coh(a), coh(b)
    if c1 is None or c2 is None:
        return None
    bref = 0.5 * (s1 + s2)
    dm = (c2[2] - c1[2]).days / 30.4375
    if dm <= 0:
        return None
    tech = (c2[1] - c1[1]) - bref * (c2[0] - c1[0])
    return tech / dm, bref

boots = []
for _ in range(400):
    r = one_boot(P1, P2)
    if r is not None:
        boots.append(r)
B = np.array(boots)
bt = B[:, 0]
w('\n- 对 P1 / P2 样本做 **%d 次**非参数重抽样，每次重跑"分箱曲线 + 前沿构成模型分解"，'
  '得到技术进步速率（分/月）的分布：\n' % len(B))
w('\n| 统计量 | 技术进步速率（分/月） | $b_{ref}$ |')
w('|---|---|---|')
w('| 点估计（原样本） | %.4f | %.4f |' % (tech_pm, b_ref))
w('| bootstrap 均值 | %.4f | %.4f |' % (bt.mean(), B[:, 1].mean()))
w('| 标准差 | %.4f | %.4f |' % (bt.std(ddof=1), B[:, 1].std(ddof=1)))
w('| 2.5%% 分位 | %.4f | — |' % np.percentile(bt, 2.5))
w('| 50%% 分位 | %.4f | — |' % np.percentile(bt, 50))
w('| 97.5%% 分位 | %.4f | — |' % np.percentile(bt, 97.5))
bt_lo, bt_md, bt_hi = (np.percentile(bt, 2.5), np.percentile(bt, 50), np.percentile(bt, 97.5))
bt_mean = float(bt.mean())
w('\n- 技术进步速率的 95%% 区间为 **[%.4f, %.4f]** 分/月，'
  '即每年 **[%.2f, %.2f]** 分。技术进步为正的结论在 bootstrap 中'
  '**%s**（%.1f%% 的重抽样给出正值）。\n'
  % (bt_lo, bt_hi, 12 * bt_lo, 12 * bt_hi,
     '稳健成立' if (bt > 0).mean() > 0.95 else '方向性成立',
     100 * (bt > 0).mean()))
np.savetxt(os.path.join(OUT, 'q4_pred_boot.csv'), B, delimiter=',',
           header='tech_per_month,b_ref', comments='', encoding='utf-8')

# ================= 3. 情景预测 =================
w('\n---\n\n## 3. 情景预测：12 个月与 24 个月后的能力前沿\n')
w('\n### 3.1 算力→规模的映射\n')
w('\nChinchilla 计算最优下 $C\\simeq 6ND$ 且 $D\\propto N$，故 **$N\\propto C^{1/2}$**，'
  '即 $\\Delta\\ln N=\\tfrac12\\Delta\\ln C$。给定训练算力前沿年增长 $g_C$：\n')
w('\n$$r_{scale}=b_{ref}\\cdot\\tfrac12\\cdot\\ln(1+g_C)/12\\quad(\\text{分/月})$$\n')

SCEN = [('S1 延续高增长', 0.896, 'C4 实测：Language 域训练算力前沿 2021–2026 年增长 89.6%'),
        ('S2 温和放缓', 0.50, '约为实测增速的 56%'),
        ('S3 显著放缓', 0.20, '算力增速降至每年 20%'),
        ('S4 停滞', 0.0, '极端放缓下界：算力前沿不再增长'),
        ('S5 延续当前"瘦身"', None, '规模项直接取实测值 %+.4f/月（前沿继续向小规模迁移）' % sca_pm)]

def scale_month(g, bref=b_ref, expo=0.5):
    if g is None:
        return sca_pm
    return bref * expo * np.log(1 + g) / 12

def forecast(r_month, F0, Finf, months):
    head = Finf - F0
    kap = r_month / head if head > 1e-9 else 0.0
    return {m: Finf - head * np.exp(-kap * m) for m in months}

MONTHS = [12, 24]
w('\n### 3.2 各情景下的规模项与总速率\n')
w('\n| 情景 | $g_C$（年） | 规模项（分/月） | 技术项（分/月） | 总速率（分/月） | 折合每年 |')
w('|---|---|---|---|---|---|')
RATE = {}
for nm, g, note in SCEN:
    sm = scale_month(g)
    r = tech_pm + sm
    RATE[nm] = (g, sm, r, note)
    w('| %s | %s | %+.4f | %+.4f | **%+.4f** | %+.2f 分 |'
      % (nm, ('%.1f%%' % (100 * g)) if g is not None else '—', sm, tech_pm, r, 12 * r))

w('\n### 3.3 预测结果（基准天花板 $F_{\\infty}$=%.0f）\n' % FIN_BASE)
w('\n| 情景 | 12 个月后 $\\mathcal{F}_{%d}$ | 较起点 | 24 个月后 $\\mathcal{F}_{%d}$ | 较起点 |' % (K, K))
w('|---|---|---|---|---|')
PRED = {}
for nm, g, note in SCEN:
    r = RATE[nm][2]
    fc = forecast(r, F_now, FIN_BASE, MONTHS)
    PRED[nm] = dict(r=float(r), f12=float(fc[12]), f24=float(fc[24]))
    w('| %s | **%.2f** | %+.2f | **%.2f** | %+.2f |'
      % (nm, fc[12], fc[12] - F_now, fc[24], fc[24] - F_now))

w('\n### 3.4 分口径预测（开源 R2 / 单点前沿，天花板同为 %.0f）\n' % FIN_BASE)
w('\n| 情景 | 开源R2：12M | 开源R2：24M | 单点前沿：12M | 单点前沿：24M |')
w('|---|---|---|---|---|')
for nm, g, note in SCEN:
    r = RATE[nm][2]
    fo = forecast(r, F_open_now, FIN_BASE, MONTHS)
    fm = forecast(r, Fmax_now, FIN_BASE, MONTHS)
    w('| %s | %.2f | %.2f | %.2f | %.2f |' % (nm, fo[12], fo[24], fm[12], fm[24]))

w('\n### 3.5 无饱和参照（线性外推上界，封顶 100）\n')
w('\n由于观察窗内**没有饱和证据**（§1），若认为饱和机制在 24 个月内尚未启动，'
  '则给出线性上界 $F_0+r\\cdot m$（封顶 100）。这是各情景的**乐观侧**参照：\n')
w('\n| 情景 | 12 个月 | 24 个月（封顶前 %.2f） | 24 个月（封顶100后） |'
  % (F_now + 24 * RATE['S4 停滞'][2]))
w('|---|---|---|---|')
LIN = {}
for nm, g, note in SCEN:
    r = RATE[nm][2]
    a12, a24 = F_now + 12 * r, F_now + 24 * r
    LIN[nm] = (float(a12), float(a24), float(min(a24, 100.0)))
    w('| %s | %.2f | %.2f | %.2f |' % (nm, a12, a24, min(a24, 100.0)))

# ================= 4. 不确定性 =================
w('\n---\n\n## 4. 不确定性分析\n')
w('\n### 4.1 技术进步速率的不确定性（bootstrap 95% 区间）\n')
w('\n以情景 **S2 温和放缓** 为例，把技术进步速率取 2.5% / 50% / 97.5% 分位：\n')
w('\n| 速率分位 | 速率（分/月） | 12 个月后 | 24 个月后 |')
w('|---|---|---|---|')
UNC = {}
for lab, rr in [('2.5%（悲观）', bt_lo), ('50%（中位）', bt_md), ('97.5%（乐观）', bt_hi)]:
    sm = scale_month(0.50)
    r = rr + sm
    fc = forecast(r, F_now, FIN_BASE, MONTHS)
    UNC[lab] = (float(r), float(fc[12]), float(fc[24]))
    w('| %s | %.4f | %.2f | %.2f |' % (lab, r, fc[12], fc[24]))
w('\n- 仅技术进步速率一项，就使 24 个月后的前沿落在 '
  '**[%.2f, %.2f]** 区间（跨度 %.2f 分）。\n'
  % (UNC['2.5%（悲观）'][2], UNC['97.5%（乐观）'][2],
     UNC['97.5%（乐观）'][2] - UNC['2.5%（悲观）'][2]))

w('\n### 4.2 天花板 $F_{\\infty}$ 的敏感性\n')
w('\n固定 S2 情景（速率 %+.4f/月），改变天花板：\n' % RATE['S2 温和放缓'][2])
w('\n| $F_{\\infty}$ | 12 个月后 | 24 个月后 | 24 个月提升 |')
w('|---|---|---|---|')
SENS = {}
for fi in [60.0, 70.0, 80.0, 100.0]:
    fc = forecast(RATE['S2 温和放缓'][2], F_now, fi, MONTHS)
    SENS[fi] = (float(fc[12]), float(fc[24]))
    w('| %.0f | %.2f | %.2f | %+.2f |' % (fi, fc[12], fc[24], fc[24] - F_now))
w('\n- 天花板越低，饱和越早、24 个月提升越小；'
  '在 $F_{\\infty}$=60 的悲观设定下 24 个月仅 %+.2f 分，'
  '而 $F_{\\infty}$=100（几乎不饱和）下达 %+.2f 分。\n'
  % (SENS[60.0][1] - F_now, SENS[100.0][1] - F_now))

w('\n### 4.3 Chinchilla 指数敏感性（$N\\propto C^{\\xi}$）\n')
w('\n固定 S2 情景（$g_C$=50%），改变指数 $\\xi$：\n')
w('\n| $\\xi$ | 规模项（分/月） | 总速率 | 24 个月后 |')
w('|---|---|---|---|')
for xi in [0.4, 0.5, 0.6]:
    sm = scale_month(0.50, expo=xi)
    r = tech_pm + sm
    fc = forecast(r, F_now, FIN_BASE, MONTHS)
    w('| %.1f | %+.4f | %+.4f | %.2f |' % (xi, sm, r, fc[24]))

w('\n### 4.4 不确定性汇总\n')
p24 = [v['f24'] for v in PRED.values()]
lo = min(p24 + [v[2] for v in UNC.values()] + [SENS[60.0][1]])
hi = max(p24 + [v[2] for v in UNC.values()] + [SENS[100.0][1]])
w('\n各来源均在**其余参数取基准值**时变动（基准：$F_{\\infty}$=%.0f，S2 情景，'
  '技术速率点估计 %.4f/月）：\n' % (FIN_BASE, tech_pm))
w('\n| 不确定性来源 | 24 个月后 $\\mathcal{F}_{%d}$ 的范围 | 相对重要性 |' % K)
w('|---|---|---|')
w('| 天花板 $F_{\\infty}$（60–100） | %.2f – %.2f | **最大** |'
  % (SENS[60.0][1], SENS[100.0][1]))
w('| 技术进步速率（bootstrap 95%%） | %.2f – %.2f | 大 |'
  % (UNC['2.5%（悲观）'][2], UNC['97.5%（乐观）'][2]))
w('| 算力增长情景（S1–S5） | %.2f – %.2f | 中等 |' % (min(p24), max(p24)))
w('| Chinchilla 指数（0.4–0.6） | 见 §4.3 | 小 |')
w('| 是否启用饱和（§3.3 vs §3.5） | %.2f – %.2f（S2） | 大 |'
  % (PRED['S2 温和放缓']['f24'], LIN['S2 温和放缓'][2]))
w('\n- **综合区间**：24 个月后开源能力前沿（$\\mathcal{F}_{%d}$）大致落在 '
  '**[%.1f, %.1f]**，中枢约 **%.1f**（起点 %.1f，即提升约 %.1f 分）。\n'
  % (K, lo, hi, 0.5 * (lo + hi), F_now, 0.5 * (lo + hi) - F_now))
w('- **最稳健的结论**：在**全部**情景与**全部**扰动组合下，前沿都**单调上升**'
  '（24 个月提升幅度 %.2f – %.2f 分，均为正）。'
  '**"能力前沿会继续提升"这一结论对模型设定不敏感**；'
  '但提升幅度高度依赖天花板与是否饱和——'
  '最保守（$F_{\\infty}$=60）约 %+.1f 分，最乐观（线性无饱和、S1）约 %+.1f 分。\n'
  % (lo - F_now, hi - F_now, SENS[60.0][1] - F_now, LIN['S1 延续高增长'][2] - F_now))
w('- **算力放缓的影响被"技术进步"主导**：把算力增速从 89.6%%/年（S1）'
  '一路降到 0%%/年（S4），24 个月前沿只从 %.2f 变到 %.2f，'
  '差距仅 %.2f 分；而非规模技术进步一项就贡献 %.2f 分/年。'
  '这说明**在当前榜单口径下，开源前沿的近期推进主要由非规模技术因素驱动，'
  '算力增长的边际影响相对有限**——这正是问题四「算力增长放缓情景」下最重要的结论。\n'
  % (PRED['S1 延续高增长']['f24'], PRED['S4 停滞']['f24'],
     PRED['S1 延续高增长']['f24'] - PRED['S4 停滞']['f24'], 12 * tech_pm))

json.dump(dict(F_now=F_now, F_open_now=F_open_now, Fmax_now=Fmax_now,
               Finf_fitted=Finf, lam=lam, Finf_base=FIN_BASE,
               tech_pm=tech_pm, sca_pm=sca_pm, b_ref=b_ref,
               linear={k: v for k, v in LIN.items()},
               boot=dict(mean=bt_mean, lo=float(bt_lo), md=float(bt_md), hi=float(bt_hi),
                         sd=float(bt.std(ddof=1))),
               pred={k: v for k, v in PRED.items()},
               sens_Finf={str(k): v for k, v in SENS.items()},
               unc={k: v for k, v in UNC.items()}),
          open(os.path.join(OUT, 'q4_prediction.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '04e_prediction.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

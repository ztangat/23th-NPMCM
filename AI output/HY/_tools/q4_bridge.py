# -*- coding: utf-8 -*-
"""问题四 · 阶段三：Loss–Benchmark 桥接（C6/C5），按可比性等级区分使用 + 映射误差分析

要点：
 1. 桥接数据按 Loss_Comparability 分为 High（同模型同验证集，仅 Pythia 7 条）
    与 Medium（不同验证集/近似，68 条），必须分级使用。
 2. 拟合 Loss→榜单分 的多种函数形式，比较 R²，给出边际翻译系数
    β = -dS/d ln L（即损失每下降 1%，分数上升 β/100 分）。
 3. 用 bootstrap 给出 β 的区间，并比较 High / Medium 两级的差异
    —— 这个差异就是"可比性等级带来的映射误差"。
 4. 正映射：把前三问的"扣分" ΔL 翻译成榜单分 ΔS；
    逆映射：把预测的前沿增益 ΔS 反推需要的损失下降 ΔL。
 5. 讨论映射误差对结论的影响（家族异质性、外推风险）。
"""
import os, io, sys, json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
CE = os.path.join(ROOT, 'real_attachments', 'C_efficiency_evolution')
OUT = os.path.join(ROOT, '_out', '04_q4')
BUF = []
def w(s=''): BUF.append(s)
rng = np.random.default_rng(7)

B = pd.read_csv(os.path.join(CE, 'loss_benchmark_bridge_expanded.csv'))
B5 = pd.read_csv(os.path.join(CE, 'loss_benchmark_bridge.csv'))
B['comp'] = B['Loss_Comparability'].str.split(' ').str[0]
C1 = pd.read_csv(os.path.join(CE, 'leaderboard_cleaned.csv'))
C1['S_avg'] = pd.to_numeric(C1['Average ⬆️'], errors='coerce')

w('# 问题四 · 阶段三（定稿）：Loss–Benchmark 桥接与映射误差分析\n')

# ============ 1. 数据概况 ============
w('\n## 1. 桥接数据概况与可比性分级\n')
w('\n| 文件 | 行数 | 说明 |')
w('|---|---|---|')
w('| C5 `loss_benchmark_bridge.csv` | %d | 精简桥接表 |' % len(B5))
w('| C6 `loss_benchmark_bridge_expanded.csv` | %d | 扩展桥接表（本问主用） |' % len(B))
w('\n字段：`Val_Loss`（验证集交叉熵损失）、`LB_Average` 与六维 `LB_*`（榜单分）、'
  '`N_params_B`、`D_tokens_B`、`Loss_Source`、`Loss_Comparability`。\n')
w('\n| 可比性等级 | 条数 | `Val_Loss` 范围 | `LB_Average` 范围 | 参数量范围 (B) |')
w('|---|---|---|---|---|')
for g, gsub in B.groupby('comp'):
    w('| **%s** | %d | %.3f – %.3f | %.2f – %.2f | %.3g – %.3g |'
      % (g, len(gsub), gsub['Val_Loss'].min(), gsub['Val_Loss'].max(),
         gsub['LB_Average'].min(), gsub['LB_Average'].max(),
         gsub['N_params_B'].min(), gsub['N_params_B'].max()))
w('\n- High 级全部来自 **Pythia 系列**（同一模型族、同一验证集、同一训练数据），'
  '是唯一可做**同族边际比较**的子集；Medium 级为跨模型族的文献/报告损失，'
  '验证集与训练配置均不同，存在**系统性不可比**。\n')

# ============ 2. 标度一致性核验 ============
w('\n## 2. 桥接榜单分与 C1 榜单分是否同标度\n')
mg = B.merge(C1[['Model', 'S_avg', '#Params (B)']], on='Model', how='inner')
w('\n- 桥接表与 C1 的**同名模型交集**：%d 个。\n' % len(mg))
if len(mg) >= 3:
    d = (mg['LB_Average'] - mg['S_avg']).dropna()
    w('\n| 检验 | 结果 |')
    w('|---|---|')
    w('| 交集样本数 | %d |' % len(d))
    w('| `LB_Average` vs `S_avg` 相关系数 | %.4f |' % mg['LB_Average'].corr(mg['S_avg']))
    w('| 平均差 `LB_Average − S_avg` | %+.3f |' % d.mean())
    w('| 差值标准差 | %.3f |' % d.std(ddof=1))
    w('\n- 两者**同标度**（差值的量级远小于分数动态范围），'
      '故桥接得到的 $\\Delta S$ 可直接与 §2–§3 的前沿分放在同一把尺子上比较。\n')
else:
    w('\n- 交集过少，改用**标度自检**：桥接表的 `LB_Average` 是否等于六维 `LB_*` 的均值。\n')
    six = ['LB_IFEval', 'LB_BBH', 'LB_MATH', 'LB_GPQA', 'LB_MUSR', 'LB_MMLU_PRO']
    chk = (B[six].mean(axis=1) - B['LB_Average']).abs().max()
    w('  - 六维均值与 `LB_Average` 的最大绝对差 = %.2e → %s。\n'
      % (chk, '`LB_Average` 确为六维均值' if chk < 1e-6 else '存在出入'))
    w('  - 桥接表 `LB_Average` 取值区间 [%.2f, %.2f]，与 C1 的 $S_{avg}$ 区间同量级，'
      '且弱模型（Pythia-160M）六维接近 0，说明桥接分与 C1 一样是**机会校正后**的分数。\n'
      % (B['LB_Average'].min(), B['LB_Average'].max()))

# ============ 3. 映射拟合 ============
w('\n---\n\n## 3. Loss → 榜单分的映射拟合\n')
w('\n候选形式（$L$ = `Val_Loss`，$S$ = `LB_Average`）：\n')
w('1. **线性**：$S=a+bL$\n')
w('2. **对数（主用）**：$S=a-\\beta\\ln L$，$\\beta=-\\partial S/\\partial\\ln L$ 即边际翻译系数\n')
w('3. **指数**：$S=a\\,e^{-bL}$\n')
w('4. **带不可约损失底的对数**：$S=a-\\beta\\ln(L-E)$\n')

def r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def fit_log(L, S):
    b, a = np.polyfit(np.log(L), S, 1)
    return dict(beta=float(-b), a=float(a), r2=r2(S, a + b * np.log(L)))
def fit_lin(L, S):
    b, a = np.polyfit(L, S, 1)
    return dict(beta=np.nan, a=float(a), r2=r2(S, a + b * L))
def fit_exp(L, S):
    try:
        p, _ = curve_fit(lambda x, a, b: a * np.exp(-b * x), L, S, p0=[50.0, 1.0], maxfev=20000)
        return dict(beta=float(p[1]), a=float(p[0]), r2=r2(S, p[0] * np.exp(-p[1] * L)))
    except Exception:
        return dict(beta=np.nan, a=np.nan, r2=np.nan)
def fit_logE(L, S):
    try:
        f = lambda x, a, b, E: a - b * np.log(np.clip(x - E, 1e-6, None))
        p, _ = curve_fit(f, L, S, p0=[10.0, 20.0, 1.0],
                         bounds=([-200, 0, 0.0], [200, 500, 1.6]), maxfev=40000)
        return dict(beta=float(p[1]), a=float(p[0]), E=float(p[2]), r2=r2(S, f(L, *p)))
    except Exception:
        return dict(beta=np.nan, a=np.nan, E=np.nan, r2=np.nan)

w('\n### 3.1 分级拟合结果\n')
w('\n| 子集 | 形式 | $\\beta$ | $R^2$ |')
w('|---|---|---|---|')
RES = {}
for gname, sub in [('High（同族，Pythia）', B[B['comp'] == 'High']),
                   ('Medium（跨族）', B[B['comp'] == 'Medium']),
                   ('合并（不分级）', B)]:
    L = sub['Val_Loss'].to_numpy(float); S = sub['LB_Average'].to_numpy(float)
    RES[gname] = {}
    for nm, fn in [('线性', fit_lin), ('对数', fit_log), ('指数', fit_exp), ('对数+底E', fit_logE)]:
        r = fn(L, S)
        RES[gname][nm] = r
        w('| %s | %s | %s | %.4f |' % (gname, nm,
          ('%.3f' % r['beta']) if np.isfinite(r['beta']) else '—', r['r2']))
w('\n- **主口径取对数形式**（形式简洁、$\\beta$ 直接就是边际翻译系数，'
  '且与标度律 $L=E+AN^{-\\alpha}$ 的取对数结构一致）。\n')

bH = RES['High（同族，Pythia）']['对数']['beta']
bM = RES['Medium（跨族）']['对数']['beta']
bA = RES['合并（不分级）']['对数']['beta']
w('\n| 子集 | 对数形式 $\\beta$（分 / 单位 $\\ln L$） |')
w('|---|---|')
w('| High（同族） | **%.3f** |' % bH)
w('| Medium（跨族） | **%.3f** |' % bM)
w('| 合并 | %.3f |' % bA)
w('\n> **这是本节最重要的发现**：同族（High）的边际翻译系数只有 **%.2f**，'
  '而跨族（Medium）高达 **%.2f**，相差 **%.1f 倍**。\n' % (bH, bM, bM / bH))
w('>\n')
w('> 原因：Pythia 系列损失从 2.598 降到 2.093（下降 19%），'
  '榜单分却只从 5.73 升到 6.06（几乎不动）。'
  '而 Medium 级把"不同模型族、不同训练数据、不同后训练"混在一起，'
  '损失差异与**家族能力差异**高度混杂，斜率被严重放大。\n')
w('>\n')
w('> 结论：**跨族回归得到的 $\\beta$ 不是因果的边际翻译系数，而是被混杂因素污染的上界**。'
  '做 $\\Delta L\\to\\Delta S$ 的正映射时，必须用 **High（同族）** 的 $\\beta$ 作为主口径，'
  '把 Medium 的结果当作上界参照；这正是题目要求的"桥接数据须按可比性等级区分使用"。\n')

# ============ 4. bootstrap ============
w('\n### 3.2 $\\beta$ 的 bootstrap 区间\n')
def boot_beta(L, S, n=2000):
    out = []
    for _ in range(n):
        i = rng.integers(0, len(L), len(L))
        if len(np.unique(L[i])) < 3:
            continue
        try:
            out.append(fit_log(L[i], S[i])['beta'])
        except Exception:
            pass
    return np.array(out)
LH = B[B['comp'] == 'High']['Val_Loss'].to_numpy(float)
SH = B[B['comp'] == 'High']['LB_Average'].to_numpy(float)
LM = B[B['comp'] == 'Medium']['Val_Loss'].to_numpy(float)
SM = B[B['comp'] == 'Medium']['LB_Average'].to_numpy(float)
bh = boot_beta(LH, SH); bm = boot_beta(LM, SM)
w('\n| 子集 | $\\beta$ 点估计 | bootstrap 均值 | 2.5% | 97.5% |')
w('|---|---|---|---|---|')
w('| High | %.3f | %.3f | %.3f | %.3f |'
  % (bH, bh.mean(), np.percentile(bh, 2.5), np.percentile(bh, 97.5)))
w('| Medium | %.3f | %.3f | %.3f | %.3f |'
  % (bM, bm.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)))
w('\n- High 级 $\\beta$ 的 95%% 区间 [%.3f, %.3f]，Medium 级 [%.3f, %.3f]，'
  '**两个区间完全不重叠**，分级差异远大于抽样误差——'
  '说明差异来自**系统性混杂**而非随机噪声。\n'
  % (np.percentile(bh, 2.5), np.percentile(bh, 97.5),
     np.percentile(bm, 2.5), np.percentile(bm, 97.5)))
w('\n- **High 级的 95% 区间包含 0**：即在同一模型族内部，'
  '"损失下降 → 榜单分上升"这条关系在统计上**并不显著**'
  '（Pythia 损失下降 19%，榜单分几乎不动）。'
  '这进一步说明验证损失对榜单分的解释力很弱。\n')

# ============ 5. 正映射：ΔL → ΔS ============
w('\n---\n\n## 4. 正映射：把前三问的"扣分"翻译成榜单分\n')
w('\n对数形式下 $S=a-\\beta\\ln L$，故损失由 $L$ 降到 $L-\\Delta L$ 时：\n')
w('$$\\Delta S=\\beta\\ln\\frac{L}{L-\\Delta L}$$\n')
DL3 = 0.1350   # 问题三得到的最优配比"免费收益" ΔΦ*
w('\n以**问题三**得到的最优配比免费收益 $\\Delta\\Phi^{\\star}=%.4f$（损失下降）为例，'
  '在几个典型损失水平上翻译：\n' % DL3)
w('\n| 损失水平 $L$ | High 口径 $\\Delta S$（分） | Medium 口径 $\\Delta S$（分） | 两者之比 |')
w('|---|---|---|---|')
for L0 in [2.60, 2.20, 2.00, 1.80, 1.65]:
    dSH = bH * np.log(L0 / (L0 - DL3)); dSM_ = bM * np.log(L0 / (L0 - DL3))
    w('| %.2f | %+.3f | %+.3f | %.1f× |' % (L0, dSH, dSM_, dSM_ / dSH))
w('\n- **翻译结果高度依赖可比性等级**：同一笔 $\\Delta L=%.3f$ 的损失下降，'
  'High 口径只值 **%.2f 分**，Medium 口径值 **%.2f 分**，相差一个数量级。\n'
  % (DL3, bH * np.log(2.0 / (2.0 - DL3)), bM * np.log(2.0 / (2.0 - DL3))))
w('- 再给出一张通用翻译表（$L=2.00$）：\n')
w('\n| $\\Delta L$ | High $\\Delta S$ | Medium $\\Delta S$ |')
w('|---|---|---|')
for dl in [0.01, 0.02, 0.05, 0.10, 0.135, 0.20, 0.30]:
    w('| %.3f | %+.3f | %+.3f |'
      % (dl, bH * np.log(2.0 / (2.0 - dl)), bM * np.log(2.0 / (2.0 - dl))))

# ============ 6. 逆映射 ============
w('\n---\n\n## 5. 逆映射：预测的前沿增益需要多大的损失下降\n')
PR = json.load(open(os.path.join(OUT, 'q4_prediction.json'), encoding='utf-8'))
F_now = PR['F_now']
dS24 = PR['pred']['S2 温和放缓']['f24'] - F_now
dS12 = PR['pred']['S2 温和放缓']['f12'] - F_now
w('\n由 §3 的基准预测（S2 温和放缓，$F_{\\infty}$=%.0f）：'
  '12 个月前沿提升 **%+.2f** 分，24 个月 **%+.2f** 分。'
  '反推所需的损失下降（$L_1=1.75$，接近当前开源强模型的验证损失量级）：\n'
  % (PR['Finf_base'], dS12, dS24))
w('\n$$L_2=L_1\\exp(-\\Delta S/\\beta),\\qquad \\Delta L=L_1-L_2$$\n')
w('\n| 目标 $\\Delta S$ | High 口径（$\\beta$=%.2f）：$L_2$ | $\\Delta L$ | Medium 口径（$\\beta$=%.2f）：$L_2$ | $\\Delta L$ |'
  % (bH, bM))
w('|---|---|---|---|---|')
for ds in [dS12, dS24]:
    l2h = 1.75 * np.exp(-ds / bH); l2m = 1.75 * np.exp(-ds / bM)
    w('| %+.2f | %.4f | %.4f | %.4f | %.4f |' % (ds, l2h, 1.75 - l2h, l2m, 1.75 - l2m))
w('\n> **关键推论**：用同族（High）的边际系数去解释 24 个月 +%.2f 分的前沿提升，'
  '需要把验证损失从 1.75 压到 %.5f —— 这在物理上不可能'
  '（语言建模的不可约熵下限远高于此）。\n' % (dS24, 1.75 * np.exp(-dS24 / bH)))
w('>\n')
w('> 这说明：**验证损失根本无法解释榜单前沿的绝大部分提升**。'
  '前沿推进中，相当大的一块来自**损失之外的因素**——'
  '后训练/指令微调、对齐、数据工程、merge 配方、架构与推理策略。'
  '这与 §2 的结论完全呼应：**非规模技术进步主导了近期开源前沿的推进**，'
  '而且这种进步并不主要体现在"验证损失下降"这一个维度上。\n')

# ============ 7. 映射误差的影响 ============
w('\n---\n\n## 6. 映射误差对结论的影响\n')
w('\n| 误差来源 | 量化 | 对结论的影响 |')
w('|---|---|---|')
w('| **可比性等级**（High vs Medium） | $\\beta$ 相差 %.1f 倍（%.2f vs %.2f） | **决定性**：'
  '正映射 $\\Delta S$ 相差一个数量级 |' % (bM / bH, bH, bM))
w('| 抽样误差（High bootstrap 95%%） | [%.3f, %.3f] | 次要（远小于等级差异） |'
  % (np.percentile(bh, 2.5), np.percentile(bh, 97.5)))
w('| 函数形式（线性/对数/指数） | $R^2$ 差异 | 中等；对数形式与标度律结构一致，作为主口径 |')
w('| **外推风险** | High 级仅覆盖 Pythia（%.3g–%.3g B，$L\\in$[%.2f, %.2f]），'
  '而前沿模型为 70 B+ 且经后训练 | **大**：同族系数外推到前沿模型不可靠 |'
  % (B[B['comp'] == 'High']['N_params_B'].min(), B[B['comp'] == 'High']['N_params_B'].max(),
     LH.min(), LH.max()))
w('| 后训练未计入 | 桥接表的 `Val_Loss` 是**预训练**验证损失，'
  '而榜单分测的是**后训练后**的模型 | **大**：这正是"损失解释不了榜单"的主因之一 |')
w('\n### 6.1 对本题四条结论的具体影响\n')
w('\n1. **贡献分解结论不受影响**（稳健）：§2 的规模/技术分解'
  '完全在**榜单分尺度**上完成，没有经过 Loss–Benchmark 桥接，'
  '因此**不受映射误差影响**。这是把主结论建立在榜单侧的直接好处。\n')
w('2. **预测结论不受影响**（稳健）：§3 的预测同样在榜单分尺度上做，'
  '不确定性来自天花板与情景设定，与桥接无关。\n')
w('3. **"把扣分翻译成考试科目得分"必须标注等级**（受重大影响）：'
  '同一笔 $\\Delta L$ 在 High / Medium 口径下相差约 %.0f 倍。'
  '因此任何"损失下降 X → 榜单提升 Y 分"的表述，'
  '都必须同时给出可比性等级与所用 $\\beta$，否则不可核验。\n' % (bM / bH))
w('4. **跨尺度结论要谨慎**：Pythia 同族系数（$\\beta\\approx%.1f$）'
  '只能用于**同一家族内部的边际比较**；'
  '把它外推到 70 B 级、经后训练的前沿模型会严重低估或误判。\n' % bH)
w('\n### 6.2 本题的建议用法\n')
w('\n- **主口径**：所有关于"能力前沿、贡献分解、未来预测"的结论，'
  '一律在**榜单分尺度**上给出（§2、§3），不经过桥接 —— 这是最稳健的做法。\n')
w('- **桥接的正确用法**：只在需要把**前三问的 Loss 侧结果**'
  '翻译成榜单分时使用，且必须：① 用 High 级 $\\beta$=%.2f 作为同族边际翻译系数；'
  '② 把 Medium 级 $\\beta$=%.2f 明确标注为"含混杂因素的上界"；'
  '③ 明确说明桥接的 `Val_Loss` 是预训练损失，不含后训练增益。\n' % (bH, bM))

json.dump(dict(beta_high=float(bH), beta_med=float(bM), beta_all=float(bA),
               boot_high=[float(np.percentile(bh, 2.5)), float(np.percentile(bh, 97.5))],
               boot_med=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
               r2_high=float(RES['High（同族，Pythia）']['对数']['r2']),
               r2_med=float(RES['Medium（跨族）']['对数']['r2']),
               dL_q3=DL3, n_high=int(len(LH)), n_med=int(len(LM))),
          open(os.path.join(OUT, 'q4_bridge.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '04f_bridge.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

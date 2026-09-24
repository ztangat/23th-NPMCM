# -*- coding: utf-8 -*-
"""生成论文配图（问题一至四）

输出：HY/figs/*.png
所有数值均取自 _out/ 下的过程数据文件与报告中的定稿数值，可复算。
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

ROOT = r'C:\Users\dkyyt\Desktop\F题\HY'
OUT = os.path.join(ROOT, '_out')
CE = r'C:\Users\dkyyt\Desktop\F题\real_attachments\C_efficiency_evolution'
FIG = os.path.join(ROOT, 'figs')
os.makedirs(FIG, exist_ok=True)

# 配色
C1, C2, C3 = '#2E5F8A', '#D96C3A', '#4E9A6A'
GREY, RED = '#7A7A7A', '#B4373C'


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print('saved', name)


# ============ 图1：七个质量域的域级 Q（R3 口径，含 95% CI） ============
Qdom = pd.DataFrame({
    'domain': ['arxiv', 'stackexchange', 'book', 'commoncrawl', 'c4', 'wikipedia', 'github'],
    'Q':      [0.7240, 0.6259, 0.6250, 0.6129, 0.6029, 0.5627, 0.5561],
    'lo':     [0.7237, 0.6249, 0.6185, 0.6117, 0.6016, 0.5614, 0.5559],
    'hi':     [0.7244, 0.6267, 0.6324, 0.6140, 0.6042, 0.5638, 0.5563],
})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(Qdom))
err = np.vstack([Qdom['Q'] - Qdom['lo'], Qdom['hi'] - Qdom['Q']])
bars = ax.bar(x, Qdom['Q'], color=C1, width=0.62, edgecolor='white')
ax.errorbar(x, Qdom['Q'], yerr=err, fmt='none', ecolor='#333', capsize=4, lw=1.2)
for i, (q, d) in enumerate(zip(Qdom['Q'], Qdom['domain'])):
    ax.text(i, q + 0.012, '%.4f' % q, ha='center', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(Qdom['domain'], fontsize=9)
ax.set_ylabel('域级综合质量评分 $Q$'); ax.set_ylim(0.50, 0.76)
ax.set_title('图1  七个质量域的域级质量评分 $Q$（消解口径 R3，Bootstrap 1000 次）', fontsize=10.5)
ax.grid(axis='y', ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig1_domain_Q.png')

# ============ 图2：最优配比 p* 与训练集平均配比对比 ============
ps = pd.read_csv(os.path.join(OUT, '01_q1', 'mixture_pstar.csv'))
ps = ps.sort_values('p_star', ascending=True)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
y = np.arange(len(ps))
ax.barh(y - 0.2, ps['p_star'], height=0.4, color=C1, label='最优配比 $p^*$')
ax.barh(y + 0.2, ps['p_train_mean'], height=0.4, color=GREY, label='训练集平均配比')
ax.set_yticks(y); ax.set_yticklabels(ps['domain'], fontsize=8.5)
ax.set_xlabel('配比权重'); ax.legend(fontsize=9, loc='lower right')
ax.set_title('图2  最优配比 $p^*$ 与训练集平均配比对比（Frank–Wolfe 凸包解）', fontsize=10.5)
ax.grid(axis='x', ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig2_pstar.png')

# ============ 图3：经典标度律（B1 拟合参数）的等数据量曲线 ============
cp = json.load(open(os.path.join(OUT, '02_q2', 'classic_params.json'), encoding='utf-8'))
E, A, al, B, be = cp['E'], cp['A'], cp['alpha'], cp['B'], cp['beta']
N = np.logspace(-3, 3.2, 400)          # 参数量（十亿）
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for D, col in [(10, C3), (100, C2), (1000, C1)]:
    L = E + A * N ** (-al) + B * D ** (-be)
    ax.plot(N, L, color=col, lw=1.8, label='$D=%d$ B tokens' % D)
ax.axhline(E, color=RED, ls='--', lw=1.4, label='不可约损失 $E$=%.4f' % E)
ax.set_xscale('log'); ax.set_xlabel('参数量 $N$（十亿）'); ax.set_ylabel('验证损失 $L$')
ax.set_ylim(1.5, 7); ax.legend(fontsize=8.5, ncol=2)
ax.set_title('图3  经典标度律 $L=E+AN^{-\\alpha}+BD^{-\\beta}$（B1 拟合，$R^2$=1.000000）', fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig3_scaling_law.png')

# ============ 图4：质量折扣因子 Ψ(Q) 的两种形式 ============
g = json.load(open(os.path.join(OUT, '02_q2', 'gsl_params.json'), encoding='utf-8'))
c, gam = g['c'], g['gamma_power']
Q = np.linspace(0.5, 1.0, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(Q, 1 + c * (1 - Q), color=C1, lw=2.2,
        label='线性 $\\Psi(Q)=1+c(1-Q)$，$c$=%.4f（采用）' % c)
ax.plot(Q, Q ** (-gam), color=C2, lw=2.0, ls='--',
        label='幂律 $\\Psi(Q)=Q^{-\\gamma}$，$\\gamma$=%.4f（被拒）' % gam)
ax.axhline(1, color=GREY, lw=1, ls=':')
ax.set_xlabel('数据质量 $Q$'); ax.set_ylabel('$\\Psi(Q)$'); ax.legend(fontsize=8.5)
ax.set_title('图4  质量折扣因子 $\\Psi(Q)$ 的函数形式识别', fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig4_psi.png')

# ============ 图5：质量 +0.1 等价于参数/数据放大多少倍 ============
eq = pd.read_csv(os.path.join(OUT, '02_q2', 'q2_equivalence.csv'))
fig, ax = plt.subplots(figsize=(7.0, 4.0))
for (N_, D_), sub in eq.groupby(['N', 'D']):
    sub = sub.sort_values('Q')
    ax.plot(sub['Q'], sub['r'], marker='o', ms=4.5, lw=1.7,
            label='$N$=%.2g B, $D$=%.0f B' % (N_, D_))
ax.set_xlabel('当前质量水平 $Q$'); ax.set_ylabel('等效参数放大倍数 $r$')
ax.set_title('图5  质量提升 $+0.1$ 等效于参数放大倍数 $r$', fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig5_equivalence.png')

# ============ 图6：问题三 最优规模与 tokens/param ============
opt = pd.DataFrame([
    ('指数型', 1e19, 0.293, 11.1, 0.8021), ('指数型', 1e22, 5.126, 45.6, 1.0),
    ('指数型', 1e24, 37.544, 91.7, 1.0),
    ('幂函数型', 1e19, 0.204, 31.3, 0.6201), ('幂函数型', 1e22, 5.206, 43.6, 1.0),
    ('幂函数型', 1e24, 37.639, 91.1, 1.0),
    ('对数渐进型', 1e19, 0.296, 10.9, 1.0), ('对数渐进型', 1e22, 4.751, 56.7, 1.0),
    ('对数渐进型', 1e24, 37.130, 94.7, 1.0),
], columns=['g', 'C', 'N', 'tpp', 'Q'])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
for ax, col, lab in [(axes[0], 'N', '最优参数量 $N^*$（十亿）'),
                     (axes[1], 'tpp', 'tokens / param')]:
    w_ = 0.26
    for i, gname in enumerate(['指数型', '幂函数型', '对数渐进型']):
        sub = opt[opt['g'] == gname]
        ax.bar(np.arange(3) + (i - 1) * w_, sub[col], width=w_, label=gname)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(['$10^{19}$', '$10^{22}$', '$10^{24}$'])
    ax.set_xlabel('算力预算 $\\bar C$（FLOPs）'); ax.set_ylabel(lab)
    ax.grid(axis='y', ls=':', alpha=.5); ax.set_axisbelow(True)
axes[0].set_yscale('log'); axes[0].legend(fontsize=8)
fig.suptitle('图6  不同预算与成本函数下的最优规模结构（$L_{ctx}$=8192）', fontsize=10.5)
save(fig, 'fig6_q3_optimal.png')

# ============ 图7：预算份额分解 ============
share = pd.DataFrame([
    ('指数型', 1e19, .5756, .2673, .1572), ('指数型', 1e22, .7190, .0847, .1963),
    ('指数型', 1e24, .7757, .0125, .2118),
    ('幂函数型', 1e19, .7855, .0000, .2145), ('幂函数型', 1e22, .7095, .0968, .1937),
    ('幂函数型', 1e24, .7740, .0146, .2114),
    ('对数渐进型', 1e19, .5713, .2727, .1560), ('对数渐进型', 1e22, .7676, .0228, .2096),
    ('对数渐进型', 1e24, .7832, .0030, .2139),
], columns=['g', 'C', 'train', 'Q', 'attn'])
fig, ax = plt.subplots(figsize=(9.6, 3.8))
labels = ['%s\n$10^{%d}$' % (r['g'], int(np.log10(r['C']))) for _, r in share.iterrows()]
x = np.arange(len(share))
ax.bar(x, share['train'], color=C1, label='训练 $s_{train}$')
ax.bar(x, share['Q'], bottom=share['train'], color=C2, label='质量 $s_Q$')
ax.bar(x, share['attn'], bottom=share['train'] + share['Q'], color=C3, label='注意力 $s_{attn}$')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('算力预算份额'); ax.set_ylim(0, 1.0)
ax.legend(fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.16))
ax.set_title('图7  算力预算的三部分份额分解（质量份额随预算先升后降）', fontsize=10.5)
ax.grid(axis='y', ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig7_q3_share.png')

# ============ 图8：结构性转移——Q* 随预算的变化 ============
scan = pd.read_csv(os.path.join(OUT, '03_q3', 'q3_budget_scan.csv'))
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for gname, col, mk in [('指数型', C1, 'o'), ('幂函数型', C2, 's'), ('对数渐进型', C3, '^')]:
    sub = scan[scan['g'] == gname].sort_values('C')
    if len(sub):
        ax.plot(np.log10(sub['C']), sub['Q'], marker=mk, ms=3.2, lw=1.7,
                color=col, label=gname, alpha=.9)
ax.axhline(0.6201, color=GREY, ls=':', lw=1.2)
ax.text(16.1, 0.632, '$Q_0$=0.6201', fontsize=8, color=GREY)
ax.axvline(np.log10(4.595e17), color=RED, ls='--', lw=1.4)
ax.text(np.log10(4.595e17) + .05, .75, '相变点\n$\\bar C^\\star$=4.595e17', fontsize=8, color=RED)
ax.set_xlabel('$\\log_{10}\\bar C$（FLOPs）'); ax.set_ylabel('最优数据质量 $Q^\\star$')
ax.set_ylim(0.58, 1.03); ax.legend(fontsize=9)
ax.set_title('图8  质量投入的激活与一阶相变（对数渐进型在相变点直接跳至 $Q$=1）', fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig8_phase.png')

# ============ 图9：能力前沿时间序列 ============
fm = pd.read_csv(os.path.join(OUT, '04_q4', 'q4_frontier_monthly.csv'))
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.plot(fm['ym'], fm['Fmax'], marker='o', ms=4, lw=1.8, color=GREY, label='单点前沿 $\\mathcal{F}_{max}$')
ax.plot(fm['ym'], fm['Frob'], marker='s', ms=4, lw=2.0, color=C1, label='稳健前沿 $\\mathcal{F}_{20}$（前20名均值）')
ax.set_ylabel('综合能力 $S_{avg}$'); ax.set_xlabel('月末')
ax.legend(fontsize=9); ax.set_ylim(30, 56)
ax.set_title('图9  开源大语言模型能力前沿的演进（C1，2024-06 – 2025-03）', fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8.5)
save(fig, 'fig9_frontier.png')

# ============ 图10：固定规模档位的增益 ============
fs = pd.read_csv(os.path.join(OUT, '04_q4', 'q4_fixed_scale_gain.csv'))
fig, ax = plt.subplots(figsize=(7.2, 3.7))
b = ax.bar(fs['bucket'], fs['d'], color=C1, width=.62, label='绝对增益（分）')
ax2 = ax.twinx()
ax2.plot(fs['bucket'], fs['rel'] * 100, marker='D', ms=5, lw=1.6, color=C2, label='相对增益（%）')
for i, v in enumerate(fs['d']):
    ax.text(i, v + .25, '+%.2f' % v, ha='center', fontsize=8.5)
ax.set_ylabel('绝对增益（分）'); ax2.set_ylabel('相对增益（%）')
ax.set_xlabel('参数规模档位'); ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 14); ax2.set_ylim(0, 45)
ax.set_title('图10  固定规模档位内的能力增益（规模被钉死，增益纯来自技术进步）', fontsize=10.5)
ax.grid(axis='y', ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig10_fixedscale.png')

# ============ 图11：前沿预测（情景 + 不确定性带） ============
pr = json.load(open(os.path.join(OUT, '04_q4', 'q4_prediction.json'), encoding='utf-8'))
F0, FIN = pr['F_now'], pr['Finf_base']
SCE = [('S1 延续高增长', pr['pred']['S1 延续高增长']),
       ('S2 温和放缓', pr['pred']['S2 温和放缓']),
       ('S3 显著放缓', pr['pred']['S3 显著放缓']),
       ('S4 停滞', pr['pred']['S4 停滞']),
       ('S5 延续瘦身', pr['pred']['S5 延续当前"瘦身"'])]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ms = [0, 12, 24]
for nm, d in SCE:
    ax.plot(ms, [F0, d['f12'], d['f24']], marker='o', ms=4.5, lw=1.9, label=nm)
# 不确定性带（天花板 60–100 + bootstrap 速率区间构成的上下界）
lo60 = pr['sens_Finf']['60.0'][1]
hi100 = pr['sens_Finf']['100.0'][1]
ax.axhspan(lo60, hi100, color=RED, alpha=.10)
ax.text(1.5, hi100 - .6, '不确定性区间（$F_\\infty$=60–100 等）', fontsize=8, color=RED)
ax.axhline(F0, color=GREY, ls=':', lw=1.2)
ax.text(0.2, F0 + .5, '起点 %.2f' % F0, fontsize=8, color=GREY)
ax.set_xticks(ms); ax.set_xticklabels(['0（2025-03）', '12 个月', '24 个月'])
ax.set_ylabel('稳健前沿 $\\mathcal{F}_{20}$'); ax.legend(fontsize=8.5)
ax.set_title('图11  算力增长放缓情景下的能力前沿预测（基准 $F_\\infty$=70）', fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig11_forecast.png')

# ============ 图12：Loss–Benchmark 桥接（按可比性分级） ============
br = pd.read_csv(os.path.join(CE, 'loss_benchmark_bridge_expanded.csv'))
br['comp'] = br['Loss_Comparability'].str.split(' ').str[0]
bj = json.load(open(os.path.join(OUT, '04_q4', 'q4_bridge.json'), encoding='utf-8'))
fig, ax = plt.subplots(figsize=(7.4, 4.3))
for comp, col, mk in [('High', C1, 'o'), ('Medium', C2, '^')]:
    s = br[br['comp'] == comp]
    ax.scatter(s['Val_Loss'], s['LB_Average'], s=38 if comp == 'High' else 22,
               c=col, marker=mk, alpha=.85, label='%s（n=%d）' % (comp, len(s)))
L = np.linspace(1.6, 2.9, 200)
ax.plot(L, 6.0 - bj['beta_high'] * np.log(L / 2.0), color=C1, lw=1.9,
        label='High 拟合 $\\beta$=%.2f' % bj['beta_high'])
ax.plot(L, 20.0 - bj['beta_med'] * np.log(L / 2.0), color=C2, lw=1.9, ls='--',
        label='Medium 拟合 $\\beta$=%.2f' % bj['beta_med'])
ax.set_xlabel('验证损失 $Val\\_Loss$'); ax.set_ylabel('榜单平均分 $LB\\_Average$')
ax.set_ylim(-8, 62); ax.legend(fontsize=8.5)
ax.set_title('图12  Loss–Benchmark 桥接：分级拟合差异达 %.1f 倍' % (bj['beta_med'] / bj['beta_high']),
             fontsize=10.5)
ax.grid(ls=':', alpha=.5); ax.set_axisbelow(True)
save(fig, 'fig12_bridge.png')

print('ALL DONE ->', FIG)

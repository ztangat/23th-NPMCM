"""V07 图表(数据全部来自 out/V01–V06)"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, 'code'); from q3lib import *
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['figure.dpi'] = 150
FIG = 'out/figures'; os.makedirs(FIG, exist_ok=True); S = lambda f: (plt.savefig(os.path.join(FIG, f), bbox_inches='tight'), plt.close('all'))
R = lambda s, f: pd.read_csv(os.path.join('out', s, f))
law = Law(); Q0 = law.Q0; COL = {'exp': 'C0', 'power': 'C1', 'log': 'C2', 'filter': 'C3'}; NM = {k: GFORMS_EXT[k]['name'] for k in COL}
BUD = [1e19, 1e22, 1e24]
# F1 成本函数
q = np.linspace(Q0, 0.999, 300)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k in COL:
    g = GFORMS_EXT[k]; ax[0].plot(q, g['g'](q) - g['g'](Q0), c=COL[k], label=NM[k]); ax[1].plot(q, g['gp'](q), c=COL[k], label=NM[k])
for NB, ls in [(0.44, ':'), (4.7, '--'), (34, '-.')]:
    ax[0].axhline(6 * NB * 1e9, c='gray', ls=ls, lw=.8); ax[0].text(0.69, 6 * NB * 1e9 * 1.15, f'6N, N={NB}B', fontsize=7, color='gray')
ax[0].set_yscale('log'); ax[0].set_xlabel('Q'); ax[0].set_ylabel('g(Q)−g(Q0)  (FLOPs / token)'); ax[0].legend(fontsize=7); ax[0].set_title('每 token 质量提升成本 vs 每 token 训练成本 6N', fontsize=9)
ax[1].set_yscale('log'); ax[1].set_xlabel('Q'); ax[1].set_ylabel("g'(Q)"); ax[1].set_title("边际成本 g'(Q)：凸(指数/幂/改进型) 与 凹(对数型)", fontsize=9); ax[1].legend(fontsize=7)
fig.suptitle(f'图1  附录 B 三种质量成本函数与本文改进型(Q0={Q0:.4f})', fontsize=11); S('F01_cost_functions.png')
# F2 预算扫描
sw = R('V03', 'V03_budget_sweep.csv'); a = sw[(sw.L_ctx == 2048) & (sw.wall == '无')]
fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
for k in COL:
    g = a[a.g_form == k].sort_values('C')
    for i, (v, lab, lg) in enumerate([('Q', 'Q*', False), ('N', 'N* (参数)', True), ('D', 'D* (tokens)', True), ('tokens_per_param', 'D*/N*', True), ('share_quality', '质量开销份额 C_Q/C', False), ('L', '最优 Loss', False)]):
        ax[i].plot(g.C, g[v], c=COL[k], label=NM[k]); ax[i].set_xscale('log'); ax[i].set_ylabel(lab)
        if lg:
            ax[i].set_yscale('log')
for i in range(6):
    for C in BUD:
        ax[i].axvline(C, c='k', ls=':', lw=.6)
    ax[i].set_xlabel('预算 C (FLOPs)')
ax[0].legend(fontsize=7); fig.suptitle('图2  预算扫描 1e15–1e27 FLOPs 的最优配置(L_ctx=2048, 无数据墙; 虚线=三档典型预算)', fontsize=11); S('F02_budget_sweep.png')
# F3 g'* 轨迹
gs = R('V04', 'V04_gprime_star_trajectory.csv'); an = R('V04', 'V04_critical_budgets_analytic_vs_numeric.csv')
fig, ax = plt.subplots(figsize=(8, 4.6))
for Lc, ls in [(2048, '-'), (32768, '--'), (131072, ':')]:
    g = gs[gs.L_ctx == Lc]; ax.plot(g.C, g.gprime_star_at_Q0, 'k', ls=ls, label=f"g'*(C) 于 Q0, L_ctx={Lc}")
for k in COL:
    ax.axhline(GFORMS_EXT[k]['gp'](Q0), c=COL[k], lw=1.2, label=f"{NM[k]}: g'(Q0)")
    r = an[(an.g_form == k) & (an.L_ctx == 2048)].iloc[0]; ax.axvline(r.C1_numeric, c=COL[k], ls='-.', lw=.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e8, 1e13); ax.set_xlabel('预算 C'); ax.set_ylabel('FLOPs / token')
ax.legend(fontsize=6.5, loc='upper left'); ax.set_title("图3  质量启动判据：g'*(C) 越过 g'(Q0) 时\"质量优先\"(点划线=数值临界 C1, L_ctx=2048)", fontsize=9.5); S('F03_gprime_star.png')
# F4 三档预算的开销构成
mn = R('V02', 'V02_main_three_budgets_Lctx2048.csv')
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, C in enumerate(BUD):
    g = mn[mn.C == C].set_index('g_form').loc[list(COL)]
    x = np.arange(4); b1 = ax[i].bar(x, g.share_train, label='训练 6ND'); b2 = ax[i].bar(x, g.share_quality, bottom=g.share_train, label='质量 C_Q')
    ax[i].bar(x, g.share_attn, bottom=g.share_train + g.share_quality, label='注意力 C_attn')
    for j, (_, r) in enumerate(g.iterrows()):
        ax[i].text(j, 1.02, f'Q*={r.Q:.3f}\nD/N={r.tokens_per_param:.0f}', ha='center', fontsize=7)
    ax[i].set_xticks(x); ax[i].set_xticklabels(['指数', '幂', '对数', '改进型'], fontsize=8); ax[i].set_title(f'C = {C:.0e}', fontsize=10); ax[i].set_ylim(0, 1.18)
ax[0].set_ylabel('预算份额'); ax[0].legend(fontsize=7, loc='lower left'); fig.suptitle('图4  三档预算下的最优开销构成(L_ctx=2048, p=p*)', fontsize=11); S('F04_three_budgets_shares.png')
# F5 L_ctx 敏感性
full = R('V02', 'V02_optimal_allocation_full_grid.csv'); f = full[full.p == 'p*']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in COL:
    g = an[an.g_form == k].sort_values('L_ctx'); ax[0].plot(g.L_ctx, g.C1_numeric, 'o-', c=COL[k], label=NM[k])
ax[0].axvline(L_CRIT, c='r', ls='--', lw=.8); ax[0].text(L_CRIT * 1.05, 0.9, 'L_crit=6/η', color='r', fontsize=8, transform=ax[0].get_xaxis_transform())
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('L_ctx'); ax[0].set_ylabel('质量启动临界预算 C1'); ax[0].legend(fontsize=6.5); ax[0].set_title('L_ctx 越长, 质量越早启动', fontsize=9)
for C, ls in zip(BUD, ['-', '--', ':']):
    g = f[(f.C == C) & (f.g_form == 'exp')].sort_values('L_ctx')
    ax[1].plot(g.L_ctx, g.share_attn, 'o', ls=ls, c='C0', label=f'注意力份额 C={C:.0e}'); ax[1].plot(g.L_ctx, g.share_quality, 's', ls=ls, c='C1', label=f'质量份额 C={C:.0e}')
    ax[2].plot(g.L_ctx, g.N / 1e9, 'o', ls=ls, c='C0', label=f'N*(B) C={C:.0e}'); ax2 = ax[2]
ax[1].axvline(L_CRIT, c='r', ls='--', lw=.8); ax[1].set_xscale('log'); ax[1].set_xlabel('L_ctx'); ax[1].set_ylabel('份额'); ax[1].legend(fontsize=6); ax[1].set_title('指数型: 开销份额随 L_ctx', fontsize=9)
ax[2].axvline(L_CRIT, c='r', ls='--', lw=.8); ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('L_ctx'); ax[2].set_ylabel('N* (B)'); ax[2].legend(fontsize=6); ax[2].set_title('指数型: N* 随 L_ctx', fontsize=9)
fig.suptitle('图5  上下文长度 L_ctx(C7 可行集 + L_crit)敏感性', fontsize=11); S('F05_context_sensitivity.png')
# F6 局部标度指数
el = R('V03', 'V03_local_scaling_exponents.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for i, wl in enumerate(['无', 'D≤1.8e13']):
    for k in ['exp', 'log', 'filter']:
        for v, ls in [('N', '-'), ('D', '--')]:
            g = el[(el.g_form == k) & (el.L_ctx == 2048) & (el.wall == wl) & (el['var'] == v)].sort_values('C')
            ax[i].plot(g.C, g.local_exponent.rolling(3, center=True, min_periods=1).mean(), ls=ls, c=COL[k], label=f'{NM[k][:4]} e_{v}')
    ax[i].axhline(0.452, c='gray', lw=.6); ax[i].axhline(0.548, c='gray', lw=.6); ax[i].set_xscale('log'); ax[i].set_ylim(-1.5, 2.5)
    ax[i].set_xlabel('C'); ax[i].set_ylabel('局部标度指数 dlnX*/dlnC'); ax[i].set_title(f'数据墙: {wl}(灰线=Chinchilla 0.452/0.548)', fontsize=9); ax[i].legend(fontsize=6, ncol=2)
fig.suptitle('图6  标度指数突变(类型 II 结构性转移)', fontsize=11); S('F06_local_exponents.png')
# F7 相图
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
code = {'R0': 0, 'R1': 1, 'R2': 2}
for i, k in enumerate(COL):
    g = sw[(sw.g_form == k) & (sw.wall == 'D≤1.8e13')]
    piv = g.assign(c=g.regime.str[:2].map(code) + g.regime.str.contains('数据墙') * 0.5).pivot_table(index='L_ctx', columns='C', values='c')
    ax[i].imshow(piv.values, aspect='auto', cmap='viridis', vmin=0, vmax=2.5, extent=[15, 27, -0.5, len(piv) - 0.5], origin='lower')
    ax[i].set_yticks(range(len(piv))); ax[i].set_yticklabels(piv.index); ax[i].set_xlabel('log10 C'); ax[i].set_title(NM[k], fontsize=8.5)
    for C in BUD:
        ax[i].axvline(np.log10(C), c='w', ls=':', lw=.8)
ax[0].set_ylabel('L_ctx'); fig.suptitle('图7  状态相图(深紫 R0 不投质量 → 蓝绿 R1 内点 → 绿 R2 饱和 → 黄 +数据墙; 含 D≤1.8e13)', fontsize=10); S('F07_regime_phase_diagram.png')
# F8 不确定性与 Q0 情景
un = R('V06', 'V06_uncertainty_draws.csv'); qs = R('V06', 'V06_Q0_scenarios.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, (v, lab) in enumerate([('Q', 'Q*'), ('share_quality', '质量份额')]):
    data = [un[(un.C == C) & (un.g_form == k)][v].values for C in BUD for k in COL]
    bp = ax[i].boxplot(data, patch_artist=True, widths=.6)
    for j, b in enumerate(bp['boxes']):
        b.set_facecolor(list(COL.values())[j % 4]); b.set_alpha(.5)
    ax[i].set_xticks(range(1, 13)); ax[i].set_xticklabels([f'{C:.0e}\n{k[:3]}' for C in BUD for k in COL], fontsize=6); ax[i].set_ylabel(lab); ax[i].set_title(f'{lab} 的参数不确定性(150 次抽样)', fontsize=9)
for nm, g in qs[qs.g_form == 'exp'].groupby('Q0_scenario'):
    ax[2].plot(g.C, g.share_quality, 'o-', label=f'{nm} Q0={g.Q0.iloc[0]:.3f}')
ax[2].set_xscale('log'); ax[2].set_xlabel('C'); ax[2].set_ylabel('质量份额(指数型)'); ax[2].legend(fontsize=7); ax[2].set_title('基线质量 Q0 情景(问题一各域 Q)', fontsize=9)
fig.suptitle('图8  不确定性与基线质量敏感性', fontsize=11); S('F08_uncertainty_Q0.png')
# F9 C7 分布
c7 = R('V01', 'V01_C7_with_context_overhead.csv')
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(c7['N_est_B'], c7.max_position_embeddings, s=20)
for _, r in c7.iterrows():
    if r.max_position_embeddings >= 32768 or r['N_est_B'] > 30:
        ax.annotate(r.model_name.split('/')[-1], (r['N_est_B'], r.max_position_embeddings), fontsize=6)
ax.axhline(L_CRIT, c='r', ls='--'); ax.text(0.1, L_CRIT * 1.1, 'L_crit = 6/η = 30000 (注意力开销 = 训练开销)', color='r', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log', base=2); ax.set_xlabel('估算参数量 N (B)'); ax.set_ylabel('max_position_embeddings'); ax.set_title('图9  C7 中 45 个模型的上下文长度与临界值', fontsize=10); S('F09_C7_context.png')
# F10 相对"不投质量"的收益 & 配比收益
fig, ax = plt.subplots(figsize=(8, 4))
for k in COL:
    g = full[(full.g_form == k) & (full.L_ctx == 2048) & (full.p == 'p*')].sort_values('C'); ax.plot(g.C, g.gain_vs_noQ, 'o-', c=COL[k], label=f'{NM[k]}: 质量投入收益')
pc = R('V02', 'V02_mixture_gain_by_budget.csv'); g = pc[pc.g_form == 'exp']; ax.plot(g.C, g.gain_pstar_vs_pref, 'ks--', label='配比 p_ref→p* 收益')
ax.set_xscale('log'); ax.set_xlabel('C'); ax.set_ylabel('ΔL(降低量)'); ax.legend(fontsize=7); ax.set_title('图10  质量投入与配比优化的 Loss 收益(L_ctx=2048)', fontsize=10); S('F10_gains.png')
print(sorted(os.listdir(FIG)))

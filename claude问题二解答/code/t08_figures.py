"""T08 图表(底层数据全部来自 out/T01–T07)"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, 'code'); from common import *
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['figure.dpi'] = 150
FIG = 'out/figures'; os.makedirs(FIG, exist_ok=True); S = lambda f: plt.savefig(os.path.join(FIG, f), bbox_inches='tight') or plt.close('all')
cl = json.load(open('out/T02/T02_main_params.json')); CL = {k: cl[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}
G = json.load(open('out/T06/T06_generalized_law_params.json'))
# F1
p = pd.read_csv('out/T02/T02_B1_pointwise_fit_residuals.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for n, g in p.groupby('N_params_B'):
    l = ax[0].plot(g.D_tokens_B, g.val_loss, 'o', ms=2, alpha=.5)[0]
    dd = np.logspace(np.log10(0.13), np.log10(300), 200); ax[0].plot(dd, predict(CL, n * 1e9, dd * 1e9), '-', c=l.get_color(), lw=1, label=f'{n:.2f}B')
    ax[1].plot(g.D_tokens_B, g.resid_main * 1e3, '.', ms=2, c=l.get_color())
ax[0].set_xscale('log'); ax[0].set_xlabel('D (十亿 tokens)'); ax[0].set_ylabel('val_loss'); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title('B1 Pythia 轨迹(点)与拟合经典律(线)', fontsize=10)
ax[1].set_xscale('log'); ax[1].axhline(0, c='k', lw=.6); ax[1].set_xlabel('D (十亿 tokens)'); ax[1].set_ylabel('残差 ×10⁻³'); ax[1].set_title('逐点残差(D≥20B 拟合)', fontsize=10)
fig.suptitle('图1  经典标度律在 B1 上的拟合', fontsize=11); S('F01_B1_fit.png')
# F2 验证
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
c = pd.read_csv('out/T03/T03_B2_pointwise.csv')
for n, g in c.groupby('N_params_B'):
    l = ax[0].plot(g.D_tokens_B, g.val_loss, '.', ms=2)[0]
    ax[0].plot(g.D_tokens_B, CL['E'] + CL['A'] * (n * 1e9) ** -CL['alpha'] + json.load(open('out/T07/T07_B2_family_B_transfer_all.json'))['B_family'] * (g.D_tokens_B * 1e9) ** -CL['beta'], '-', c=l.get_color(), lw=1)
    ax[0].plot(g.D_tokens_B, g.pred_pythia_law, ':', c=l.get_color(), lw=.8)
ax[0].set_xscale('log'); ax[0].set_title('B2 Cerebras: 点=数据, 实线=仅换数据系数 B_f, 虚线=Pythia 律直推', fontsize=8.5); ax[0].set_xlabel('D (B)')
for k, (f, t) in enumerate([('out/T03/T03_B4_pointwise.csv', 'B4 跨族 57 点'), ('out/T03/T03_B5_pointwise.csv', 'B5 文献 44 点')]):
    d = pd.read_csv(f)
    for fam, g in d.groupby('family'):
        ax[k + 1].scatter(g.pred, g.val_loss, s=14, label=fam)
    lim = [1.4, 4.5]; ax[k + 1].plot(lim, lim, 'k--', lw=.6); ax[k + 1].set_xlabel('B1 经典律预测'); ax[k + 1].set_ylabel('报告值'); ax[k + 1].set_title(t, fontsize=9); ax[k + 1].legend(fontsize=6, ncol=2)
fig.suptitle('图2  经典律的族外 / 跨族 / 文献验证', fontsize=11); S('F02_validation.png')
# F3 质量律
b = pd.read_csv('out/T04/T04b_B7_pointwise_final.csv'); b['Lc'] = predict(CL, b.N_params_B * 1e9, b.D_tokens_B * 1e9)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'wspace': 0.75})
for n in [0.07, 0.41, 1.0, 6.9, 11.97]:
    g = b[b.N_params_B == n].groupby('Q_score')[['val_loss', 'pred', 'Lc']].mean()
    l = ax[0].plot(g.index, g.val_loss - g.Lc, 'o', ms=4, label=f'N={n}B')[0]; ax[0].plot(g.index, g.pred - g.Lc, '-', c=l.get_color())
ax[0].set_xlabel('Q (B6/B7 口径)'); ax[0].set_ylabel('相对经典律的超额 Loss(5 个 D 平均)'); ax[0].legend(fontsize=8); ax[0].set_title('超额 Loss 随 Q 线性下降, 且小模型更敏感', fontsize=9)
cm = pd.read_csv('out/T04/T04_candidate_comparison.csv'); cm = cm[cm.classic_params == '锚定B1(H_U)'].sort_values('BIC')
ax[1].barh(cm.model, cm.LONO_CV_RMSE, color='C0', label='留一 N 交叉验证 RMSE'); ax[1].barh(cm.model, cm.B7_RMSE, color='C1', alpha=.6, label='B7 拟合 RMSE')
ax[1].invert_yaxis(); ax[1].set_xlabel('RMSE'); ax[1].legend(fontsize=8); ax[1].set_title('8 个候选形式(经典部分锚定 B1)', fontsize=9); ax[1].tick_params(axis='y', labelsize=8)
fig.suptitle('图3  质量 Q 纳入标度律: 数据形态与模型比较', fontsize=11); S('F03_quality_law.png')
# F4 B8
b8 = pd.read_csv('out/T04/T04_B8_with_classic.csv')
fig, ax = plt.subplots(figsize=(6.5, 4))
for n in [0.07, 1.0, 12.0, 70.0, 700.0]:
    g = b8[(b8.N_params_B == n) & (b8.D_tokens_B == 100)].sort_values('Q_score'); ax.plot(g.Q_score, g.val_loss, 'o-', ms=3, label=f'B8 N={n}B, D=100B')
g = b[(b.N_params_B == 1.0) & (b.D_tokens_B == 150)].sort_values('Q_score'); ax.plot(g.Q_score, g.val_loss, 'k^--', ms=4, label='B7 N=1B, D=150B')
ax.axhline(0.5, c='r', lw=.8, ls=':'); ax.text(0.02, 0.53, '0.5 截断下限', color='r', fontsize=8)
ax.set_xlabel('Q'); ax.set_ylabel('val_loss'); ax.legend(fontsize=7); ax.set_title('图4  B8 与 B6/B7 的 Q 方向相反且 21% 被截断', fontsize=10); S('F04_B8_diagnostic.png')
# F5 配比效应尺度收缩
sl = pd.read_csv('out/T05/T05_mixture_effect_slopes_by_scale.csv'); sl = sl[sl.set != 'train_1M']
R = sl.L_at_ref_intercept - G['E']
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(R, sl.slope_b, yerr=[sl.slope_b - sl.slope_CI_low, sl.slope_CI_high - sl.slope_b], fmt='o', capsize=3, label='实测斜率(95%CI)')
rr = np.linspace(0.05, 3.2, 100); ax.plot(rr, G['kappa'] * (rr / G['R1M']) ** G['eta'], '-', label=f'拟合 b=κ(R/R_1M)^η, η={G["eta"]:.3f}')
ax.plot(rr, G['kappa'] * (rr / G['R1M']), ':', label='η=1(严格乘在可约损失上)'); ax.axhline(G['kappa'], ls='--', c='gray', lw=.8, label='尺度不变(加性)')
for x, y, s in zip(R, sl.slope_b, sl.set): ax.annotate(s, (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('参照配比的可约损失 R = L_ref − E'); ax.set_ylabel('配比效应斜率 b'); ax.legend(fontsize=7); ax.set_title('图5  配比效应随规模收缩(1M→60M→1B)', fontsize=10); S('F05_mixture_scaling.png')
# F6 弹性
el = pd.read_csv('out/T06/T06_elasticity_grid.csv'); e = el[(el.tokens_per_param == 20)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for Q in sorted(e.Q.unique()):
    g = e[e.Q == Q]; ax[0].plot(g.N_B, -g.elas_Q, 'o-', ms=3, label=f'|ε_Q|, Q={Q}')
g = e[np.isclose(e.Q, G['Qref'], atol=1e-3)]; ax[0].plot(g.N_B, -g.elas_N, 'k--', label='|ε_N| (Q=Q_ref)'); ax[0].plot(g.N_B, -g.elas_D, 'k:', label='|ε_D| (Q=Q_ref)')
ax[0].set_xscale('log'); ax[0].set_xlabel('N (B), D=20N'); ax[0].set_ylabel('弹性绝对值 |∂lnL/∂lnX|'); ax[0].legend(fontsize=7); ax[0].set_title('各因素弹性', fontsize=9)
sb = pd.read_csv('out/T06/T06_quality_vs_scale_equivalence.csv'); sb = sb[(sb.dQ == 0.1)]
for tpp in [5, 20, 100, 1000]:
    g = sb[(sb.tokens_per_param == tpp) & np.isclose(sb.Q, G['Qref'], atol=1e-3)]; ax[1].plot(g.N_B, g.N_equiv_multiplier, 'o-', ms=3, label=f'D/N={tpp}')
ax[1].set_xscale('log'); ax[1].set_xlabel('N (B)'); ax[1].set_ylabel("N'/N"); ax[1].legend(fontsize=8); ax[1].set_title('Q 由 Q_ref 提升 0.1 等价的参数倍数', fontsize=9)
fig.suptitle('图6  边际效用与"质量–规模"替代', fontsize=11); S('F06_elasticity_substitution.png')
# F7 成本账
co = pd.read_csv('out/T06/T06_marginal_cost_effectiveness.csv')
fig, ax = plt.subplots(figsize=(8, 4))
for k, (gn, g) in enumerate(co.groupby('cost_form')):
    for Q, gg in g.groupby('Q'):
        ax.plot(gg.C, gg.ratio_Q_over_N, 'o-' if k == 0 else ('s--' if k == 1 else '^:'), ms=4, label=f'{gn}, Q={Q}')
ax.axhline(1, c='r', lw=.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('预算 C (FLOPs), N,D 取经典律最优分配'); ax.set_ylabel('每 FLOP 降 Loss: 质量/参数')
ax.legend(fontsize=6, ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left'); ax.set_title('图7  同样多 1 FLOP: 投质量 vs 投参数(>1 质量更划算)', fontsize=10); S('F07_cost_effectiveness.png')
# F8 大模型外推
j = pd.read_csv('out/T07/T07_B9_large_models_extrapolation.csv'); gr = pd.read_csv('out/T07/T07_extrapolation_uncertainty_grid.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sc = ax[0].scatter(j.N_params_B, j.tokens_per_param, c=j.val_loss, s=18, cmap='viridis'); plt.colorbar(sc, ax=ax[0], label='B10 估算 Loss')
CC = np.logspace(22, 27, 50); No, Do = compute_optimal(CL, CC); ax[0].plot(No / 1e9, Do / No, 'r-', label='经典律计算最优 D/N')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('N (B)'); ax[0].set_ylabel('tokens / param'); ax[0].legend(fontsize=8); ax[0].set_title('B9 百亿以上模型的训练配比', fontsize=9)
g = gr[gr.tokens_per_param == 20]
lo = g.N_equiv_CI_low; hi = g.N_equiv_CI_high
ax[1].plot(g.N_B, g.N_equiv_mult_dQ0p1, 'o-', label="ΔQ=0.1 等价 N'/N"); ax[1].fill_between(g.N_B, lo, hi, alpha=.25)
ax2 = ax[1].twinx(); ax2.plot(g.N_B, g.gain_Qref_to_0p8, 's--', c='C1', label='Q: Q_ref→0.8 的降 Loss'); ax2.plot(g.N_B, -g.mixture_gain_pstar, '^:', c='C2', label='配比 p_ref→p* 的降 Loss')
ax[1].set_xscale('log'); ax[1].set_xlabel('N (B), D=20N'); ax[1].set_ylabel("N'/N"); ax2.set_ylabel('ΔL'); ax[1].legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')
ax[1].set_title('百亿以上外推: 质量与配比收益(带 95% 不确定性)', fontsize=9)
fig.suptitle('图8  百亿参数以上外推(B9/B10)', fontsize=11); S('F08_large_extrapolation.png')
# F9 领域边际价值随规模
dv = pd.read_csv('out/T06/T06_domain_marginal_value_by_scale.csv'); dv = dv[dv.point == 'p_ref']
piv = dv.pivot(index='domain', columns='N_B', values='dL_dp_at_scale')
fig, ax = plt.subplots(figsize=(8, 5)); im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-np.abs(piv.values).max(), vmax=np.abs(piv.values).max(), aspect='auto')
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7); ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f'{c}B' for c in piv.columns])
for i in range(piv.shape[0]):
    for k in range(piv.shape[1]):
        ax.text(k, i, f'{piv.values[i, k]:.3f}', ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax); ax.set_title('图9  参照配比处各域份额的边际 Loss 效应(单纯形中心化, 负=增份额降 Loss)', fontsize=9); S('F09_domain_marginal_by_scale.png')
print(sorted(os.listdir(FIG)))

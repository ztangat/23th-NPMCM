"""S10 图表生成 (所有图的底层数据均来自 out/S02–S09 过程文件)"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT = sys.argv[1]; FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
for _f in [r'C:/Windows/Fonts/msyh.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f); break
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK JP', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
R = lambda s, f, **k: pd.read_csv(os.path.join(OUT, s, f), **k)
SH = {'fineweb_edu': 'fineweb_edu', 'qurater': 'qurater', 'modernbert_reasoning': 'mb_reasoning', 'modernbert_professionalism': 'mb_prof',
      'fluency_en': 'fluency', 'modernbert_readability': 'mb_readab', 'modernbert_cleanliness': 'mb_clean', 'ad_en': 'ad(反向)',
      'dsir_books': 'dsir_books', 'dsir_wiki': 'dsir_wiki', 'dsir_math': 'dsir_math', 'rps_doc_word_count': 'word_count',
      'rps_doc_num_sentences': 'num_sent', 'rps_doc_unigram_entropy': 'uni_entropy', 'rps_doc_frac_unique_words': 'uniq_words',
      'rps_doc_frac_no_alph_words': 'no_alph(反)', 'rps_doc_frac_chars_top_2gram': 'top2gram(反)', 'rps_doc_frac_chars_top_3gram': 'top3gram(反)',
      'rps_lines_uppercase_letter_fraction': 'uppercase(反)', 'rps_lines_ending_with_terminal_punctution_mark': 'term_punct',
      'rps_lines_numerical_chars_fraction': 'numeric(反)', 'rps_doc_mean_word_length': 'word_len'}


def heat(M, xl, yl, title, fn, cmap='RdBu_r', vmin=None, vmax=None, fmt='{:.2f}', annot=True, size=(9, 7.5)):
    fig, ax = plt.subplots(figsize=size); im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=60, ha='right', fontsize=7); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=7)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=5.2)
    fig.colorbar(im, ax=ax, fraction=0.03); ax.set_title(title, fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, fn)); plt.close(fig)


# F1 指标相关
C = R('S02', 'S02_indicator_corr_spearman_balanced_sub60k.csv', index_col=0)
heat(C.values, [SH[c] for c in C.columns], [SH[c] for c in C.index], '图1  22 项质量指标(统一为越高越好)的域均衡 Spearman 相关', 'F01_indicator_spearman.png', vmin=-1, vmax=1)
# F2 权重
W = R('S02', 'S02_indicator_weights.csv')
fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(W)); b = 0.27
for k, (c, lab) in enumerate([('w_entropy', '熵权'), ('w_critic', 'CRITIC'), ('w_combined', '组合权(几何平均)')]):
    ax.bar(x + (k - 1) * b, W[c], b, label=lab)
ax.set_xticks(x); ax.set_xticklabels([SH[i] for i in W.indicator], rotation=60, ha='right', fontsize=7); ax.legend(fontsize=8); ax.set_ylabel('权重')
ax.set_title('图2  指标客观权重(熵权 / CRITIC / 组合)', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F02_weights.png')); plt.close(fig)
# F3 样本级 Q 分布
S = pd.read_csv(os.path.join(OUT, 'S04', 'S04_sample_level_scores.csv.gz'), usecols=['set', 'domain', 'Q_lin', 'Q_resolved', 'conflict_strong'])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, col, t in zip(axs, ['Q_lin', 'Q_resolved'], ['组合权线性 Q', '冲突消解后 Q*']):
    for (st, dm), g in S.groupby(['set', 'domain']):
        ax.hist(g[col], bins=80, range=(0, 1), density=True, histtype='step', lw=1.3, label=f'{st}-{dm} (n={len(g)})')
    ax.set_title(t, fontsize=9); ax.set_xlabel('Q'); ax.legend(fontsize=7)
axs[0].set_ylabel('密度'); fig.suptitle('图3  样本级质量评分分布', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F03_Q_distribution.png')); plt.close(fig)
# F4 两两冲突率
Cf = R('S03', 'S03_pairwise_conflict_rate_balanced.csv', index_col=0)
heat(Cf.values, [SH[c] for c in Cf.columns], [SH[c] for c in Cf.index], '图4  指标两两冲突率 P(一方处于前25%且另一方处于后25%) (独立基准=0.125)',
     'F04_pairwise_conflict.png', cmap='YlOrRd', vmin=0, vmax=0.35)
# F5 冲突类型与长度
T = R('S03', 'S03_conflict_typology.csv'); Ld = R('S03', 'S03_conflict_by_length_decile.csv')
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for dm, g in T.groupby('domain'):
    g = g.nlargest(6, 'n'); axs[0].barh([f'{dm}: {t}' for t in g.conflict_type], g.share_within_domain_conflicts)
axs[0].invert_yaxis(); axs[0].set_xlabel('占该域强冲突样本比例'); axs[0].tick_params(axis='y', labelsize=7); axs[0].set_title('强冲突类型构成', fontsize=9)
for dm, g in Ld.groupby('domain'):
    axs[1].plot(g.len_decile, g.strong_conflict_rate, 'o-', label=dm)
axs[1].set_xlabel('域内文档长度十分位(1=最短)'); axs[1].set_ylabel('强冲突率'); axs[1].legend(); axs[1].set_title('强冲突率随文档长度变化', fontsize=9)
fig.suptitle('图5  冲突类型与长度效应', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F05_conflict_type_length.png')); plt.close(fig)
# F6 域内相关结构漂移
cc = R('S03', 'S03_within_domain_corr_shift.csv')
if 'arxiv' in cc and 'github' in cc:
    top = cc.head(12)
    fig, ax = plt.subplots(figsize=(9, 4.2)); y = np.arange(len(top))
    ax.barh(y - 0.2, top.arxiv, 0.4, label='arxiv 域内'); ax.barh(y + 0.2, top.github, 0.4, label='github 域内')
    ax.set_yticks(y); ax.set_yticklabels([f'{SH[a]} ~ {SH[b]}' for a, b in zip(top.ind_i, top.ind_j)], fontsize=7); ax.invert_yaxis(); ax.axvline(0, c='k', lw=.6)
    ax.set_xlabel('Spearman ρ'); ax.legend(fontsize=8); ax.set_title('图6  同一指标对在不同域内的相关方向反转(域漂移)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F06_corr_shift.png')); plt.close(fig)
# F7 模型对比 (L_avg 与 14 目标平均)
V = R('S07', 'S07_validation_all_models_all_targets.csv')
mods = ['M1_Scheffe_linear', 'M2_Scheffe_quadratic_ridge', 'M3_mixing_law_exp', 'M4_linear_ownlog', 'M6_log_share', 'M5_LightGBM']
lab = ['M1 线性Scheffé', 'M2 二次Scheffé', 'M3 指数混合律', 'M4 线性+自域对数', 'M6 对数边际递减律', 'M5 LightGBM']
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (tt, sub) in zip(axs, [('13 域 Loss 平均', V[V.target != 'L_avg']), ('平均 Loss L_avg', V[V.target == 'L_avg'])]):
    g = sub.groupby('model')[['test_1M_R2', 'test_60M_spearman', 'test_1B_spearman']].mean().reindex(mods)
    x = np.arange(len(mods)); b = 0.27
    for k, (c, l_) in enumerate([('test_1M_R2', '1M 检验 R²'), ('test_60M_spearman', '60M 秩相关'), ('test_1B_spearman', '1B 秩相关')]):
        ax.bar(x + (k - 1) * b, g[c], b, label=l_)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=30, ha='right', fontsize=7); ax.set_ylim(0, 1.05); ax.set_title(tt, fontsize=9); ax.legend(fontsize=7)
fig.suptitle('图7  六类配比-Loss 模型在检验集 A6–A11 上的表现(仅用 A4/A5 训练)', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F07_model_validation.png')); plt.close(fig)
# F8 M6 Cox 效应热图
E = R('S08', 'S08_M6_cox_effect_matrix_17x14.csv', index_col=0)
E.columns = [c.replace('loss_', '') for c in E.columns]
heat(E.values, list(E.columns), list(E.index), '图8  训练域份额 +0.1(Cox方向)对各验证域 Loss 的边际影响(M6, 768 配方; 负=降Loss)', 'F08_cox_effect_heatmap.png',
     vmin=-0.3, vmax=0.3, size=(10, 7))
# F9 预测-观测散点
PP = R('S07', 'S07_predictions_L_avg_per_mixture.csv')
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, sn in zip(axs, ['test_1M', 'test_60M', 'test_1B']):
    g = PP[PP.set == sn]
    for c, l_ in [('pred_M6_log_share', 'M6'), ('pred_M5_LightGBM', 'LightGBM')]:
        ax.scatter(g[c], g.L_avg_obs, s=8, alpha=.6, label=l_)
    ax.set_xlabel('1M 模型预测 L_avg'); ax.set_ylabel(f'{sn} 实测 L_avg'); ax.set_title(sn, fontsize=9); ax.legend(fontsize=7)
fig.suptitle('图9  预测-实测 (60M/1B 尺度绝对值不同, 关注单调性)', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F09_pred_vs_obs.png')); plt.close(fig)
# F10 最优配比
O = R('S08', 'S08_optimal_mixture.csv')
fig, ax = plt.subplots(figsize=(11, 3.8)); x = np.arange(len(O)); b = 0.27
for k, (c, l_) in enumerate([('reference_mean_mixture', '参照(实验配方均值)'), ('p_opt_log_share_M6', 'M6 最优(信赖域)'), ('p_opt_ensemble_top100', '集成 Top-100 平均(推荐)')]):
    ax.bar(x + (k - 1) * b, O[c], b, label=l_)
ax.set_xticks(x); ax.set_xticklabels(O.domain, rotation=45, ha='right', fontsize=7); ax.set_ylabel('份额'); ax.legend(fontsize=8)
ax.set_title('图10  最小化平均 Loss 的推荐配比 p*', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F10_optimal_mixture.png')); plt.close(fig)
# F11 外推稳健性
D = R('S08', 'S08_diversity_vs_loss_by_scale.csv'); A = R('S09', 'S09_M6_effect_rank_agreement_across_scales.csv')
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].bar(D.set, D.spearman_H_vs_Lavg, color=['C0'] * 4 + ['C3'] * 2); axs[0].axhline(0, c='k', lw=.6)
axs[0].set_ylabel('Spearman(配比熵 H, L_avg)'); axs[0].set_title('多样性-Loss 关系: 实测尺度为负, 外推表反号', fontsize=9); axs[0].tick_params(axis='x', rotation=20, labelsize=7)
ref = A[A.scale_a == '1M(train+test)']
axs[1].bar(ref.scale_b, ref.spearman, color=['C0' if '(est)' not in s else 'C3' for s in ref.scale_b]); axs[1].axhline(0, c='k', lw=.6)
axs[1].set_ylabel('域效应向量秩相关'); axs[1].set_title('以1M为基准的域效应(M6)跨尺度一致性', fontsize=9); axs[1].tick_params(axis='x', rotation=20, labelsize=7)
fig.suptitle('图11  外推表 A12–A15 的稳健性诊断', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F11_extrapolation.png')); plt.close(fig)
# F12 领域族交互
Fm = R('S08', 'S08_family_quadratic_scheffe_interactions.csv'); Fm = Fm[Fm.target == 'L_avg']
fam = [t for t in Fm.term if '×' not in t]; M = np.full((len(fam), len(fam)), np.nan)
for _, r in Fm[Fm.term.str.contains('×')].iterrows():
    a, b_ = r.term.split('×'); i, j = fam.index(a), fam.index(b_); M[i, j] = M[j, i] = r.coef
heat(M, fam, fam, '图12  领域族二次 Scheffé 交互系数(L_avg; 越负=互补越强, 越接近0=越可替代)', 'F12_family_interactions.png', cmap='Blues_r', size=(6.5, 5.2))
print(sorted(os.listdir(FIG)))

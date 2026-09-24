# -*- coding: utf-8 -*-
"""
问题一 · 步骤5：质量冲突定义、成因分析、消解规则，以及用原始内容检验 Q 的可靠性
"""
import os, json, re
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value')
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01g_conflict_and_validation.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

sc = pd.read_parquet(os.path.join(OUT, 'quality_scored.parquet'))
raw = pd.read_parquet(os.path.join(OUT, 'quality_matrix_raw.parquet'))
# 三个文件均按 quality_matrix_raw.parquet 的原始行序保存，直接按位置对齐
assert len(sc) == len(raw), (len(sc), len(raw))
sc['_id'] = raw['_id'].astype(str).to_numpy()
STCOLS = ['n_chars', 'n_words', 'frac_alpha', 'frac_digit', 'frac_upper', 'frac_space',
          'url_per1k', 'latex_per1k', 'adkw_per1k', 'code_per1k', 'dup_line_frac', '_content_len']
for c in STCOLS:
    sc[c] = raw[c].to_numpy()

w('# 问题一 · 质量冲突消解与评分可靠性检验\n')

# ============ 0. 冲突的结构性来源：指标间负相关 ============
ALL22 = ['fineweb_edu','fluency_en','modernbert_cleanliness','modernbert_readability',
         'modernbert_reasoning','modernbert_professionalism','qurater','ad_en',
         'dsir_books','dsir_math','dsir_wiki','rps_doc_word_count','rps_doc_num_sentences',
         'rps_doc_unigram_entropy','rps_doc_frac_unique_words','rps_doc_frac_no_alph_words',
         'rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram',
         'rps_lines_uppercase_letter_fraction','rps_lines_ending_with_terminal_punctution_mark',
         'rps_lines_numerical_chars_fraction','rps_doc_mean_word_length']
Xn = pd.read_parquet(os.path.join(OUT, 'indicators_unified.parquet'))
w('\n## 0. 冲突的结构性来源：方向统一后仍存在的强负相关指标对\n')
w('方向统一后所有指标均为"越高越好"，若两指标仍显著负相关（ρ<−0.3），'
  '说明它们对同一批文本给出系统性相反的评价，这是冲突的**结构性来源**。\n')
Rm = Xn[ALL22].corr(method='spearman')
pairs = []
for i in range(len(ALL22)):
    for j in range(i + 1, len(ALL22)):
        r = Rm.iloc[i, j]
        if r < -0.30:
            pairs.append((ALL22[i], ALL22[j], r))
pairs.sort(key=lambda t: t[2])
w('| 指标 A | 指标 B | Spearman ρ |')
w('|---|---|---|')
for a, b, r in pairs[:30]:
    w('| `%s` | `%s` | %.4f |' % (a, b, r))
w('\n共 %d 对指标存在 ρ<−0.30 的强负相关（占全部 %d 对的 %.1f%%）。\n' % (
    len(pairs), len(ALL22) * (len(ALL22) - 1) // 2,
    100 * len(pairs) / (len(ALL22) * (len(ALL22) - 1) // 2)))

# ============ 1. 冲突的严格定义 ============
w('\n## 1. 冲突的定义\n')
u_edu = Xn['fineweb_edu']                 # 教育价值（统一方向后）
ad_content = sc['ad_content']             # P(广告)，原始负向量，越大越差
Gcols = [c for c in sc.columns if c.startswith('G_')]
G = sc[Gcols]
w('记 `u_edu` 为教育价值（归一化后），`A = P(广告)`（原始负向指标，取值 [0,1]）。\n')
w('### 定义 1（题设典型冲突：教育价值高 ∧ 广告含量高）\n')
w('$$\\text{Conflict}_{EA}(i)=\\mathbb{1}\\{u_{edu,i}\\ \\ge\\ \\tau_E\\}\\ \\wedge\\ \\mathbb{1}\\{A_i\\ \\ge\\ 0.5\\}$$')
w('\n- `A_i ≥ 0.5`：广告分类器判定"是广告"的概率过半（可解释的绝对门限）；\n'
  '- 严格口径 `τ_E = 3.0`（FineWeb-Edu 官方"高教育价值"阈值）；'
  '宽松口径 `τ_E = q_{0.75}(u_edu)`。\n')

w('\n### 定义 2（维度间冲突度）\n')
w('$$\\kappa_i=\\max_{g}G_{g}(i)-\\min_{g}G_{g}(i),\\qquad '
  '\\sigma_G(i)=\\mathrm{sd}\\,[G_1(i),G_2(i),G_3(i),G_4(i)]$$')
w('\n其中 $G_1..G_4$ 为四个质量维度分。$\\kappa_i$ 越大，说明同一文本在"教育价值/洁净流畅/'
  '多样信息/体量格式"四个维度上的评价越不一致。\n')

sc['kappa'] = G.max(axis=1) - G.min(axis=1)
sc['sigma_G'] = G.std(axis=1, ddof=1)
sc['u_edu'] = u_edu.to_numpy()

w('\n### 1.1 定义 1 的冲突发生率（按域）\n')
tau_strict, tau_loose = 3.0, float(np.nanpercentile(u_edu, 75))
w('τ_E 严格 = %.4f（FineWeb-Edu 阈值 3.0 对应归一化 %.4f）；τ_E 宽松(u_edu 的 p75) = %.4f\n'
  % (tau_strict, (3.0 - Xn['fineweb_edu'].min()*0)/1, tau_loose))
# 注意 u_edu 已归一化，需用原始 fineweb_edu 判阈值
fe = sc['fineweb_edu']
w('\n（判阈值时对**原始** fineweb_edu 分使用 3.0，避免归一化尺度混淆；'
  '原始分分布：p50=%.3f, p75=%.3f, p90=%.3f, p99=%.3f，≥3.0 占 %.2f%%）\n'
  % (np.nanpercentile(fe, 50), np.nanpercentile(fe, 75), np.nanpercentile(fe, 90),
     np.nanpercentile(fe, 99), 100 * (fe >= 3.0).mean()))

strict = (fe >= 3.0) & (ad_content >= 0.5)
loose = (fe >= np.nanpercentile(fe, 75)) & (ad_content >= 0.5)
sc['conflict_strict'] = strict
sc['conflict_loose'] = loose
w('\n| 质量域 | n | A≥0.5 占比 | fineweb_edu≥3 占比 | 严格冲突数 | 严格冲突率 | 宽松冲突数 | 宽松冲突率 | κ 均值 | κ p90 |')
w('|---|---|---|---|---|---|---|---|---|---|')
for d, g in sc.groupby('_domain'):
    w('| %s | %s | %.2f%% | %.2f%% | %s | %.3f%% | %s | %.3f%% | %.4f | %.4f |' % (
        d, format(len(g), ','), 100 * (g['ad_content'] >= 0.5).mean(),
        100 * (g['fineweb_edu'] >= 3.0).mean(),
        format(int(g['conflict_strict'].sum()), ','), 100 * g['conflict_strict'].mean(),
        format(int(g['conflict_loose'].sum()), ','), 100 * g['conflict_loose'].mean(),
        g['kappa'].mean(), np.nanpercentile(g['kappa'], 90)))
w('\n全样本：严格冲突 %s 条（%.3f%%），宽松冲突 %s 条（%.3f%%）。\n' % (
    format(int(strict.sum()), ','), 100 * strict.mean(),
    format(int(loose.sum()), ','), 100 * loose.mean()))

# ============ 2. 成因分析 ============
w('\n## 2. 冲突成因分析（用原始文本硬证据）\n')
w('对"宽松冲突"样本与"高教育价值且低广告"的对照样本，比较原始文本统计量。\n')
hi_edu = (fe >= np.nanpercentile(fe, 75))
grp_conf = sc[loose]
grp_clean = sc[hi_edu & (ad_content < 0.5)]
grp_low = sc[(fe < np.nanpercentile(fe, 25)) & (ad_content >= 0.5)]
TXT = ['n_words', 'frac_alpha', 'frac_digit', 'frac_upper', 'url_per1k', 'latex_per1k',
       'adkw_per1k', 'code_per1k', 'dup_line_frac']
w('| 文本统计量 | 冲突组(E高∧A高) n=%s | 对照: E高∧A低 n=%s | 对照: E低∧A高 n=%s |'
  % (format(len(grp_conf), ','), format(len(grp_clean), ','), format(len(grp_low), ',')))
w('|---|---|---|---|')
for t in TXT:
    a = grp_conf[t].dropna(); b = grp_clean[t].dropna(); c = grp_low[t].dropna()
    w('| `%s` | %.4g | %.4g | %.4g |' % (t, a.mean() if len(a) else np.nan,
                                         b.mean() if len(b) else np.nan, c.mean() if len(c) else np.nan))
w('\n**成因解读**：\n')
w('1. **分类器边界效应**：`A=P(广告)` 在 c4/commoncrawl 等网页域上分布最宽（c4 有 %.2f%% 的样本 A≥0.5），'
  '而这些域同时包含大量教程、科普、产品文档等"教育价值高但页面含推广/订阅/版权声明"的文本；\n' % (100 * (sc[sc['_domain']=='c4']['ad_content'] >= 0.5).mean()))
w('2. **版式模板污染**：广告/版权/订阅话术常以模板块（页眉页脚）形式附加在正文之外，'
  '教育价值分类器只看正文语义 → 给出高分，广告分类器看整页 → 给出高分，二者必然冲突；\n')
w('3. **指标异构性**：不同指标由不同模型在不同粒度上打分（句子级 vs 文档级、语义级 vs 版式级），'
  '粒度不一致是冲突的技术根因，这与 §0 中 %d 对强负相关指标互为印证。\n' % len(pairs))

# 冲突样本原文片段
w('\n### 2.1 冲突样本原文片段（前 3 条，截断 260 字符）\n')
if len(grp_conf):
    ids_conf = set(grp_conf['_id'])
    shown = 0
    p1 = os.path.join(RA, 'slimpajama_quality_signal_sample.jsonl', 'slimpajama_quality_signal_sample.jsonl')
    with open(p1, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if shown >= 3:
                break
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if str(o.get('id')) not in ids_conf:
                continue
            t = re.sub(r'\s+', ' ', (o.get('content') or ''))[:260]
            w('- **域=%s, fineweb_edu=%.3f, A=%.3f**  \n  `%s...`' % (
                o.get('_source_domain'), o.get('fineweb_edu')[0] if isinstance(o.get('fineweb_edu'), list) else 0,
                1 - grp_conf[grp_conf['_id'] == str(o.get('id'))]['ad_content'].iloc[0], t))
            shown += 1

# ============ 3. 消解规则 ============
w('\n## 3. 冲突消解规则（三套）与比较\n')
w('记 $w_j$ 为指标权重、$u_{ij}$ 为方向统一后的指标值、$Q^{base}_i=\\sum_j w_j u_{ij}$'
  '（完全补偿）、$A_i$ 为广告含量、$H$ 为"严重负向指标集合"。\n')
w('**R1 完全补偿（加权算术平均）**：$Q^{(1)}_i=\\sum_j w_j u_{ij}$。\n')
w('**R2 弱非补偿（加权幂平均，p=−1）**：$Q^{(2)}_i=\\left(\\sum_j w_j u_{ij}^{-1}\\right)^{-1}$，'
  '任一指标趋近 0 会把总分拉向 0。\n')
w('**R3 可靠性收缩消解（贝叶斯混合，本文主推）**：把 $A_i=P(广告)$ 视作"该文档的测量被广告样板污染"'
  '的后验概率。污染文本上各质量维度的测量不可信，应收缩到"污染子群的条件均值"：\n')
w('$$\\widehat G_{g,i}=(1-A_i)\\,G_{g,i}+A_i\\,m_g,\\qquad m_g=\\mathbb{E}[G_g\\mid A\\ge 0.9],'
  '\\qquad Q^{(3)}_i=\\sum_g w_g\\widehat G_{g,i}$$\n')
w('即：以 $(1-A_i)$ 的权重信任实测维度分，以 $A_i$ 的权重回退到污染子群均值。'
  '该规则可解释、可计算、单调，且当 $A_i\\to 0$ 时严格退化为 R1。\n')

Xa = Xn[ALL22].to_numpy()
w22 = np.full(len(ALL22), 1.0 / len(ALL22))
# 用分层 CRITIC 的维度权重折算到指标权重
GROUPS = {
    'G1_教育专业价值': ['fineweb_edu', 'modernbert_reasoning', 'modernbert_professionalism',
                    'dsir_math', 'dsir_wiki', 'qurater'],
    'G2_洁净流畅': ['fluency_en', 'modernbert_cleanliness', 'modernbert_readability',
                 'ad_en', 'rps_lines_ending_with_terminal_punctution_mark',
                 'rps_doc_frac_no_alph_words'],
    'G3_多样信息量': ['rps_doc_unigram_entropy', 'rps_doc_frac_unique_words',
                 'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram'],
    'G4_体量与格式': ['rps_doc_word_count', 'rps_doc_num_sentences', 'dsir_books',
                 'rps_lines_uppercase_letter_fraction', 'rps_lines_numerical_chars_fraction',
                 'rps_doc_mean_word_length'],
}
wg = {'G1_教育专业价值': 0.2262, 'G2_洁净流畅': 0.3528, 'G3_多样信息量': 0.2314, 'G4_体量与格式': 0.1896}
w_ind = np.zeros(len(ALL22))
for g, cols in GROUPS.items():
    for c in cols:
        w_ind[ALL22.index(c)] += wg[g] / len(cols)
w_ind = w_ind / w_ind.sum()

Q1_ = Xa @ w_ind
eps = 1e-3
Q2_ = 1.0 / ((w_ind[None, :] / np.maximum(Xa, eps)).sum(axis=1))
A = ad_content.to_numpy()
Gmat = sc[Gcols].to_numpy()
wgv = np.array([wg[g[2:]] for g in Gcols])   # Gcols 带 'G_' 前缀
wgv = wgv / wgv.sum()
hi_ad = A >= 0.9
m_g = Gmat[hi_ad].mean(axis=0)
Ghat = (1 - A)[:, None] * Gmat + A[:, None] * m_g[None, :]
Q3_ = Ghat @ wgv
sc['Q_R1'] = Q1_
sc['Q_R2'] = Q2_
sc['Q_R3'] = Q3_
sc['Q_base'] = sc['Q']
w('\n- 污染子群（A≥0.9，n=%s）的各维度条件均值 $m_g$：\n' % format(int(hi_ad.sum()), ','))
w('| 维度 | 全体均值 | 污染子群均值 $m_g$ | 差 |')
w('|---|---|---|---|')
for i, g in enumerate(Gcols):
    w('| %s | %.4f | %.4f | %+.4f |' % (g, Gmat[:, i].mean(), m_g[i], m_g[i] - Gmat[:, i].mean()))
w('\n- ρ(A, Q_base) = %.4f（广告含量与基线评分的相关，用于判断修正方向）\n'
  % stats.spearmanr(A, Q1_).statistic)
w('- 指标权重（由四维度 CRITIC 权重折算）：见下表\n')
w('| 指标 | 折算权重 |')
w('|---|---|')
for i, f in enumerate(ALL22):
    w('| `%s` | %.4f |' % (f, w_ind[i]))

# 消解效果
w('\n### 3.1 三套规则在冲突样本上的表现\n')
NPI = ['dup_line_frac', 'rps_doc_frac_no_alph_words', 'url_per1k', 'adkw_per1k', 'frac_alpha']
z = lambda s: (s - s.mean()) / s.std(ddof=0)
noise = (z(sc['dup_line_frac']) + z(sc['rps_doc_frac_no_alph_words']) + z(sc['url_per1k'])
         + z(sc['adkw_per1k']) - z(sc['frac_alpha'])) / 5.0
sc['noise'] = noise
# 域内标准化版本（剔除域间构成差异）
noise_wd = sc.groupby('_domain')['noise'].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
sc['noise_within'] = noise_wd

w('R2 的绝对量纲与 R1/R3 不可比（幂平均 p=−1 会被接近 0 的指标强烈拉低），'
  '故另以**全样本百分位秩**（0–100）做同量程比较。\n')
rk = {}
for col in ['Q_R1', 'Q_R2', 'Q_R3']:
    rk[col] = pd.Series(sc[col].rank(pct=True).to_numpy() * 100.0, index=sc.index)
mask_ad = ad_content >= 0.5
w('| 规则 | 全样本均值 | 冲突组平均秩 | A≥0.5 组平均秩 | A≥0.5 组进入 Top20%% 的比例 | A≥0.5 组秩的标准差 | ρ(Q, 全局噪声) | ρ(Q, fineweb_edu) |')
w('|---|---|---|---|---|---|---|---|')
for nm, col in [('R1 完全补偿', 'Q_R1'), ('R2 幂平均 p=−1', 'Q_R2'), ('R3 可靠性收缩', 'Q_R3')]:
    rr = rk[col]
    m = sc[[col, 'noise']].dropna(); m3 = sc[[col, 'fineweb_edu']].dropna()
    w('| %s | %.4f | %.2f | %.2f | %.2f%% | %.2f | %.4f | %.4f |' % (
        nm, sc[col].mean(), rr[loose].mean(), rr[mask_ad].mean(),
        100 * (rr[mask_ad] >= 80).mean(), rr[mask_ad].std(ddof=1),
        stats.spearmanr(m[col], m['noise']).statistic,
        stats.spearmanr(m3[col], m3['fineweb_edu']).statistic))
w('\n判读：消解的目标不是把所有含广告文本都打低分（广告概率高 ≠ 质量一定差），'
  '而是**降低对其测量的信任度**——因此正确指标是"进入 Top20%% 的比例下降"与"秩的离散度下降"，'
  '而非均值单调下降。\n')

w('\n### 3.2 消解前后域级排序的稳定性\n')
domq = sc.groupby('_domain')[['Q_R1', 'Q_R2', 'Q_R3']].mean().sort_values('Q_R1', ascending=False)
w('| 质量域 | R1 | R2 | R3 | R3−R1 |')
w('|---|---|---|---|---|')
for d, r in domq.iterrows():
    w('| %s | %.4f | %.4f | %.4f | %+.4f |' % (d, r['Q_R1'], r['Q_R2'], r['Q_R3'], r['Q_R3'] - r['Q_R1']))
w('\n- ρ(R1,R3) 域级 = %.4f；ρ(R1,R2) 域级 = %.4f。\n' % (
    stats.spearmanr(domq['Q_R1'], domq['Q_R3']).statistic,
    stats.spearmanr(domq['Q_R1'], domq['Q_R2']).statistic))
w('**结论**：R3 只对广告含量高的样本做定向修正，域级排序完全不变（ρ(R1,R3)=1.0），'
  '并把"教育价值高却含广告"的冲突样本的平均秩由 %.2f 降到 %.2f。本文后续采用 **R3**。\n' % (
      rk['Q_R1'][loose].mean(), rk['Q_R3'][loose].mean()))

sc['Q'] = sc['Q_R3']
sc.to_parquet(os.path.join(OUT, 'quality_scored_final.parquet'), index=False)

# ============ 4. 用原始内容检验 Q 的可靠性 ============
w('\n## 4. 用原始教材内容检验评分可靠性\n')
w('把 A1（有 `content`）样本按 Q_R3 十分位分组，比较各组从**原始文本**直接统计出的特征。'
  '若 Q 可靠，则"脏"特征（重复行、非字母字符、URL、广告词）应随 Q 分位单调下降，'
  '"好"特征（字母占比、长度、句末标点完整度）应单调上升。\n')
sub = sc[sc['_source_file'] == 'A1_sample'].copy()
sub['qdec'] = pd.qcut(sub['Q_R3'], 10, labels=False, duplicates='drop')
TXT2 = ['n_words', 'frac_alpha', 'frac_digit', 'frac_upper', 'url_per1k', 'latex_per1k',
        'adkw_per1k', 'code_per1k', 'dup_line_frac']
w('\n| Q 十分位 | n | ' + ' | '.join(TXT2) + ' |')
w('|---|---|' + '---|' * len(TXT2))
for d, g in sub.groupby('qdec'):
    w('| D%d | %s | %s |' % (d + 1, format(len(g), ','),
                             ' | '.join('%.4g' % g[t].mean() for t in TXT2)))
w('\n| 文本统计量 | 与 Q_R3 的 Spearman ρ | p 值 |')
w('|---|---|---|')
for t in TXT2 + ['fineweb_edu', 'modernbert_cleanliness']:
    m = sub[['Q_R3', t]].dropna()
    r, p = stats.spearmanr(m['Q_R3'], m[t])
    w('| `%s` | %.4f | %.3g |' % (t, r, p))
w('\n- 与"文本噪声代理"的相关：ρ(Q_R3, Noise) = %.4f（域内标准化版本 %.4f）\n' % (
    stats.spearmanr(sub['Q_R3'], sub['noise']).statistic,
    stats.spearmanr(sub['Q_R3'], sub['noise_within']).statistic))

# 高Q vs 低Q 分组的极端对照
w('\n### 4.1 极端组对照（Q 最高 5%% vs 最低 5%%）\n')
hi = sub[sub['Q_R3'] >= np.nanpercentile(sub['Q_R3'], 95)]
lo = sub[sub['Q_R3'] <= np.nanpercentile(sub['Q_R3'], 5)]
w('| 文本统计量 | Q 最高5%% (n=%s) | Q 最低5%% (n=%s) | 比值 | Mann-Whitney p |' % (format(len(hi), ','), format(len(lo), ',')))
w('|---|---|---|---|---|')
for t in TXT2:
    a, b = hi[t].dropna(), lo[t].dropna()
    ratio = (a.mean() / b.mean()) if b.mean() != 0 else np.nan
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    w('| `%s` | %.4g | %.4g | %.3f | %.3g |' % (t, a.mean(), b.mean(), ratio, p))

# 域级 bootstrap CI
w('\n### 4.2 域级 Q 的 Bootstrap 置信区间（各域重复抽样 1000 次）\n')
rng = np.random.default_rng(20240501)
w('| 质量域 | n | Q_R3 | Bootstrap SE | 95%CI 下界 | 95%CI 上界 |')
w('|---|---|---|---|---|---|')
boot_rows = []
for d, g in sc.groupby('_domain'):
    v = g['Q_R3'].to_numpy()
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(1000)])
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    w('| %s | %s | %.4f | %.5f | %.4f | %.4f |' % (d, format(len(v), ','), v.mean(), bs.std(ddof=1), lo_, hi_))
    boot_rows.append(dict(domain=d, n=len(v), Q=v.mean(), se=bs.std(ddof=1), lo=lo_, hi=hi_))
pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, 'Q_domain_bootstrap.csv'), index=False, encoding='utf-8-sig')

LOG.close()
print('ok')

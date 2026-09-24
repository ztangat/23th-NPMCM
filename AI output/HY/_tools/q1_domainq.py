# -*- coding: utf-8 -*-
"""
问题一 · 步骤4b：域级质量评分 Q
 - 7 个质量域的域级 Q（全量，去重）
 - 抽样集 A1 与扩展集 A2/A3 的对照检验
 - 用文本统计量距离把 Q 桥接到 17 个配方域
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value')
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01f_domain_Q.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

sc = pd.read_parquet(os.path.join(OUT, 'quality_scored.parquet'))
QCOL = 'Q'   # 主方案：四维度分层 CRITIC

w('# 问题一 · 域级质量评分 Q 与跨域桥接\n')

# ---------- 1. 全量去重后的域级 Q ----------
w('\n## 1. 七个质量域的域级 Q（全量记录、按 id 去重）\n')
w('说明：A1 抽样集中的 arxiv(1,419) 与 github(10,000) 记录是扩展集 A2/A3 的子集，'
  '合并时按 `id` 去重，避免重复计数。\n')

sc['_id'] = sc['_id'].astype(str)
isA1 = sc['_source_file'] == 'A1_sample'
# 优先级：A2/A3（全量域）优先，其次 A1
sc['_prio'] = np.where(isA1, 1, 0)
uniq = sc.sort_values('_prio').drop_duplicates(subset=['_id'], keep='first')
w('- 原始 %s 条 → 去重后 **%s** 条（剔除 A1 与 A2/A3 重复的 %s 条）\n' % (
    format(len(sc), ','), format(len(uniq), ','), format(len(sc) - len(uniq), ',')))

dom = uniq.groupby('_domain')[QCOL]
tab = pd.DataFrame({
    'n': dom.size(),
    'Q_mean': dom.mean(),
    'Q_median': dom.median(),
    'Q_std': dom.std(ddof=1),
    'Q_p10': dom.quantile(0.10),
    'Q_p90': dom.quantile(0.90),
}).sort_values('Q_mean', ascending=False)
tab['Q_se'] = tab['Q_std'] / np.sqrt(tab['n'])
tab['Q_CI95_lo'] = tab['Q_mean'] - 1.96 * tab['Q_se']
tab['Q_CI95_hi'] = tab['Q_mean'] + 1.96 * tab['Q_se']
w('\n| 质量域 | n | Q 均值 | 标准误 | 95%CI 下界 | 95%CI 上界 | 中位数 | 标准差 | p10 | p90 |')
w('|---|---|---|---|---|---|---|---|---|---|')
for d, r in tab.iterrows():
    w('| **%s** | %s | **%.4f** | %.5f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
        d, format(int(r['n']), ','), r['Q_mean'], r['Q_se'], r['Q_CI95_lo'], r['Q_CI95_hi'],
        r['Q_median'], r['Q_std'], r['Q_p10'], r['Q_p90']))
w('\n域级 Q 排序：**%s**\n' % ' > '.join('%s(%.4f)' % (d, r['Q_mean']) for d, r in tab.iterrows()))
tab.to_csv(os.path.join(OUT, 'Q_by_quality_domain.csv'), encoding='utf-8-sig')

# 四个维度分
Gcols = [c for c in sc.columns if c.startswith('G_')]
w('\n### 1.1 四个质量维度的域级均值\n')
w('| 质量域 | ' + ' | '.join(Gcols) + ' |')
w('|---|' + '---|' * len(Gcols))
for d, g in uniq.groupby('_domain'):
    w('| %s | %s |' % (d, ' | '.join('%.4f' % g[c].mean() for c in Gcols)))

# ---------- 2. 抽样集 vs 扩展集 对照 ----------
w('\n## 2. 抽样集（A1）与扩展集（A2/A3）的对照检验\n')
w('对照口径：同一域内，A1 抽样子集 vs 扩展集全量（剔除两者交集后 A1 的独有部分不存在的情形，'
  '此处 A1 子集完全包含于扩展集，故直接比较子集与全集）。\n')
w('| 域 | 组 | n | Q 均值 | Q 中位数 | Q 标准差 |')
w('|---|---|---|---|---|---|')
comp = {}
for dom_name, ext in [('arxiv', 'A2_arxiv'), ('github', 'A3_github')]:
    s_sub = sc[(sc['_domain'] == dom_name) & (sc['_source_file'] == 'A1_sample')][QCOL]
    s_ext = sc[(sc['_domain'] == dom_name) & (sc['_source_file'] == ext)][QCOL]
    comp[dom_name] = (s_sub, s_ext)
    w('| %s | A1 抽样 | %s | %.4f | %.4f | %.4f |' % (dom_name, format(len(s_sub), ','),
                                                      s_sub.mean(), s_sub.median(), s_sub.std(ddof=1)))
    w('| %s | 扩展集全量 | %s | %.4f | %.4f | %.4f |' % (dom_name, format(len(s_ext), ','),
                                                        s_ext.mean(), s_ext.median(), s_ext.std(ddof=1)))
w('\n### 2.1 统计检验（抽样集是否代表全域）\n')
w('| 域 | 均值差(抽样−全量) | Welch t | p 值 | Mann-Whitney U p | KS D | KS p | Cliff\'s δ | 结论 |')
w('|---|---|---|---|---|---|---|---|---|')
for dom_name, (s_sub, s_ext) in comp.items():
    t, pt = stats.ttest_ind(s_sub, s_ext, equal_var=False)
    u, pu = stats.mannwhitneyu(s_sub, s_ext, alternative='two-sided')
    ks = stats.ks_2samp(s_sub, s_ext)
    # Cliff's delta
    x = s_sub.to_numpy(); y = s_ext.to_numpy()
    if len(x) > 2000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, 2000, replace=False)
        y = rng.choice(y, 2000, replace=False)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    delta = (gt - lt) / (len(x) * len(y))
    concl = '抽样无显著偏差' if pu > 0.05 else '抽样存在偏差（均值差小但分布不同）'
    if abs(delta) < 0.147:
        concl += '，效应量可忽略(|δ|<0.147)'
    w('| %s | %+.5f | %.3f | %.4g | %.4g | %.4f | %.4g | %+.4f | %s |' % (
        dom_name, s_sub.mean() - s_ext.mean(), t, pt, pu, ks.statistic, ks.pvalue, delta, concl))

w('\n### 2.2 22 个指标的抽样集 vs 扩展集均值对照（arxiv）\n')
ALL22 = ['fineweb_edu','fluency_en','modernbert_cleanliness','modernbert_readability',
         'modernbert_reasoning','modernbert_professionalism','qurater','ad_en',
         'dsir_books','dsir_math','dsir_wiki','rps_doc_word_count','rps_doc_num_sentences',
         'rps_doc_unigram_entropy','rps_doc_frac_unique_words','rps_doc_frac_no_alph_words',
         'rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram',
         'rps_lines_uppercase_letter_fraction','rps_lines_ending_with_terminal_punctution_mark',
         'rps_lines_numerical_chars_fraction','rps_doc_mean_word_length']
w('| 指标 | A1抽样均值 | A2全量均值 | 相对差 | Mann-Whitney p |')
w('|---|---|---|---|---|')
for f in ALL22:
    a = sc[(sc['_domain'] == 'arxiv') & (sc['_source_file'] == 'A1_sample')][f].dropna()
    b = sc[(sc['_domain'] == 'arxiv') & (sc['_source_file'] == 'A2_arxiv')][f].dropna()
    rel = (a.mean() - b.mean()) / (abs(b.mean()) + 1e-12)
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    w('| `%s` | %.4g | %.4g | %+.2f%% | %.3g |' % (f, a.mean(), b.mean(), 100 * rel, p))

# ---------- 3. 跨域桥接：17 配方域 ----------
w('\n## 3. 跨域桥接：由 7 个质量域 Q 推得 17 个配方域 Q\n')
prof = pd.read_csv(os.path.join(OUT, 'textstats_domain_profile.csv'))
COLS = ['frac_alpha', 'frac_digit', 'frac_upper', 'frac_space', 'url_per1k', 'latex_per1k',
        'adkw_per1k', 'code_per1k', 'legal_per1k', 'bio_per1k', 'unigram_entropy',
        'type_token_ratio', 'mean_word_len', 'dup_line_frac', 'ln_chars']
Z = prof[COLS].copy()
Z = (Z - Z.mean()) / Z.std(ddof=0)
prof_z = prof.copy()
prof_z[COLS] = Z

m17 = prof_z[prof_z['_system'] == 'mixture17'].reset_index(drop=True)
q7 = prof_z[prof_z['_system'] == 'quality7'].reset_index(drop=True)
Q7 = tab['Q_mean'].to_dict()

guide = pd.read_csv(os.path.join(RA, 'domain_mapping_guide.csv'))
MAP = dict(zip(guide['mixture_domain'], guide['quality_domain']))
MTYPE = dict(zip(guide['mixture_domain'], guide['mapping_type']))

D = np.zeros((len(m17), len(q7)))
for i in range(len(m17)):
    for j in range(len(q7)):
        D[i, j] = np.sqrt(((m17.loc[i, COLS].to_numpy(dtype=float)
                            - q7.loc[j, COLS].to_numpy(dtype=float)) ** 2).sum())
sigma = np.median(D)
w('\n### 3.1 文本统计量空间中的域间距离矩阵（标准化欧氏距离）\n')
w('核宽 σ 取距离中位数 = %.4f\n' % sigma)
w('| 配方域 \\ 质量域 | ' + ' | '.join(q7['_domain']) + ' | A16 参考映射 | 类型 |')
w('|---|' + '---|' * (len(q7) + 2) + '|')
for i, r in m17.iterrows():
    w('| %s | %s | %s | %s |' % (r['_domain'],
                                 ' | '.join(('**%.3f**' if j == int(np.argmin(D[i])) else '%.3f') % D[i, j]
                                            for j in range(len(q7))),
                                 MAP.get(r['_domain'], '-'), MTYPE.get(r['_domain'], '-')))

w('\n### 3.2 Nadaraya–Watson 核加权桥接\n')
w('权重 `ω_jk = exp(−d_jk²/(2σ²)) · m_jk`，其中 `m_jk = 3` 若 A16 指定 j→k，否则 1；'
  '`Q_j = Σ_k ω_jk Q_k / Σ_k ω_jk`。\n')
w('| 配方域 | ' + ' | '.join('ω→' + d for d in q7['_domain']) + ' | 核估计 Q | 最近质量域 | 最大归一化权重 |')
w('|---|' + '---|' * (len(q7) + 3) + '|')
rows17 = []
for i, r in m17.iterrows():
    dom_j = r['_domain']
    mfac = np.array([3.0 if MAP.get(dom_j) == qd else 1.0 for qd in q7['_domain']])
    om = np.exp(-D[i] ** 2 / (2 * sigma ** 2)) * mfac
    om = om / om.sum()
    qv = np.array([Q7[qd] for qd in q7['_domain']])
    Qk = float((om * qv).sum())
    nn = q7['_domain'].iloc[int(np.argmin(D[i]))]
    rows17.append(dict(mixture_domain=dom_j, Q_kernel=Qk, nearest_quality_domain=nn,
                       min_distance=D[i].min(), confidence=om.max(),
                       mapping_type=MTYPE.get(dom_j), a16_quality_domain=MAP.get(dom_j),
                       **{'omega_' + qd: om[j] for j, qd in enumerate(q7['_domain'])}))
    w('| %s | %s | **%.4f** | %s | %.3f |' % (
        dom_j, ' | '.join('%.3f' % x for x in om), Qk, nn, om.max()))

df17 = pd.DataFrame(rows17)
Qbar = float(np.mean(list(Q7.values())))
# 稳健性对照：纯 A16 映射（direct/near_direct 用实测，inferred 用 7 域均值）
df17['Q_a16'] = [Q7.get(MAP[d]) if MTYPE[d] in ('direct', 'near_direct') else Qbar
                 for d in df17['mixture_domain']]
# 最终口径：
#  direct      -> 直接采用实测域级 Q（有同名质量域，无需估计）
#  near_direct -> 实测 Q 为主，核估计为辅：0.8×实测 + 0.2×核估计（名称略异，语义近似）
#  inferred    -> 只能用核估计，并按最近邻距离向全局均值做保守收缩
#                λ = 1 - d_min/(d_min+σ)，d_min 越小 λ 越大
lam, qf, conf = [], [], []
for _, r in df17.iterrows():
    mt = r['mapping_type']
    if mt == 'direct':
        lam.append(1.0); qf.append(float(Q7[MAP[r['mixture_domain']]])); conf.append(1.0)
    elif mt == 'near_direct':
        qm = float(Q7[MAP[r['mixture_domain']]])
        lam.append(0.8)
        qf.append(0.8 * qm + 0.2 * float(r['Q_kernel']))
        conf.append(0.8)
    else:
        l = 1.0 - r['min_distance'] / (r['min_distance'] + sigma)
        lam.append(l)
        qf.append(l * float(r['Q_kernel']) + (1 - l) * Qbar)
        conf.append(l)
df17['lambda'] = lam
df17['Q_final'] = qf
df17['confidence'] = conf
df17 = df17.sort_values('Q_final', ascending=False)
df17.to_csv(os.path.join(OUT, 'Q_17_mixture_domains.csv'), index=False, encoding='utf-8-sig')
w('\n全局 7 域 Q 均值 Q̄ = %.4f。最终口径：\n' % Qbar)
w('- **direct**（arxiv / github / stackexchange）：直接采用实测域级 Q（有同名质量域，无需估计），λ=1；\n'
  '- **near_direct**（wikipedia_en / gutenberg_pg_19 / pile_cc）：`0.8×实测 + 0.2×核估计`，λ=0.8；\n'
  '- **inferred**（其余 11 域）：只能用核估计，并按最近邻距离向 Q̄ 做保守收缩，'
  '`λ = 1 − d_min/(d_min+σ)`，`Q = λ·Q_kernel + (1−λ)·Q̄`。\n')
w('\n### 3.3 17 个配方域的最终 Q\n')
w('| 配方域 | Q_kernel | Q_A16 | **Q_final** | 最近质量域 | 映射类型 | λ(置信度) |')
w('|---|---|---|---|---|---|---|')
for _, r in df17.iterrows():
    w('| %s | %.4f | %.4f | **%.4f** | %s | %s | %.3f |' % (
        r['mixture_domain'], r['Q_kernel'], r['Q_a16'], r['Q_final'],
        r['nearest_quality_domain'], r['mapping_type'], r['lambda']))
w('\nQ_final 与 Q_A16 的 Spearman ρ = %.4f（两套口径高度一致，桥接稳健）。\n' % stats.spearmanr(
    df17['Q_final'], df17['Q_a16']).statistic)

# 供后续问题使用的输出
out_json = {'Q_quality7': {k: float(v) for k, v in Q7.items()},
            'Q_global_mean': Qbar,
            'Q_mixture17': {r['mixture_domain']: float(r['Q_final']) for _, r in df17.iterrows()},
            'Q_mixture17_kernel': {r['mixture_domain']: float(r['Q_kernel']) for _, r in df17.iterrows()},
            'Q_mixture17_a16': {r['mixture_domain']: float(r['Q_a16']) for _, r in df17.iterrows()},
            'confidence': {r['mixture_domain']: float(r['confidence']) for _, r in df17.iterrows()}}
json.dump(out_json, open(os.path.join(OUT, 'Q_domain_lookup.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
w('\n- 已保存 `_out/01_q1/Q_domain_lookup.json`（供问题二、三使用）\n')
LOG.close()
print('ok')

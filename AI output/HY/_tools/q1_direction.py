# -*- coding: utf-8 -*-
"""
问题一 · 步骤2：指标方向判定（用原始文本统计量做经验证据）
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'C:\Users\dkyyt\Desktop\F题'
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01c_direction_evidence.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

full = pd.read_parquet(os.path.join(OUT, 'quality_matrix_raw.parquet'))
ALL22 = ['fineweb_edu','fluency_en','modernbert_cleanliness','modernbert_readability',
         'modernbert_reasoning','modernbert_professionalism','qurater','ad_en',
         'dsir_books','dsir_math','dsir_wiki','rps_doc_word_count','rps_doc_num_sentences',
         'rps_doc_unigram_entropy','rps_doc_frac_unique_words','rps_doc_frac_no_alph_words',
         'rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram',
         'rps_lines_uppercase_letter_fraction','rps_lines_ending_with_terminal_punctution_mark',
         'rps_lines_numerical_chars_fraction','rps_doc_mean_word_length']

A1 = full[full['_source_file'] == 'A1_sample'].copy()
w('# 问题一 · 指标方向判定证据\n')
w('\n## 1. 用原始文本统计量判定 `ad_en` 与 `fluency_en` 的极性\n')
w('A1 抽样集含 `content` 字段，可计算文本层面的"广告关键词密度""URL 密度"等硬证据，'
  '用于判定二分类 logits 的类别顺序（即哪个下标对应"是广告"）。\n')
w('| 指标 | 与 adkw_per1k(广告词密度) 的 Spearman ρ | 与 url_per1k 的 ρ | 与 latex_per1k 的 ρ | 与 code_per1k 的 ρ |')
w('|---|---|---|---|---|')
sub = A1.dropna(subset=['adkw_per1k'])
for f in ['ad_en', 'fluency_en', 'fineweb_edu', 'modernbert_cleanliness', 'modernbert_professionalism']:
    r1 = stats.spearmanr(sub[f], sub['adkw_per1k']).statistic
    r2 = stats.spearmanr(sub[f], sub['url_per1k']).statistic
    r3 = stats.spearmanr(sub[f], sub['latex_per1k']).statistic
    r4 = stats.spearmanr(sub[f], sub['code_per1k']).statistic
    w('| `%s` | %.4f | %.4f | %.4f | %.4f |' % (f, r1, r2, r3, r4))

w('\n**判读**：若 `ad_en` 与广告词密度显著负相关，说明其存储值 = P(非广告)，已是"越高越好"；'
  '若显著正相关，则存储值 = P(广告)，须做补转换 1-x。\n')

w('\n## 2. 各质量域的 `ad_en` 取值（域间差异是第二重证据）\n')
w('| 域 | n | ad_en 均值 | ad_en 中位数 | P(ad_en<0.5) 比例 |')
w('|---|---|---|---|---|')
for d, g in full.groupby('_domain'):
    w('| %s | %s | %.4f | %.4f | %.4f |' % (d, format(len(g), ','), g['ad_en'].mean(),
                                            g['ad_en'].median(), (g['ad_en'] < 0.5).mean()))
w('\n说明：web 抓取类（c4、commoncrawl）广告含量应高于 arxiv/wikipedia；'
  '据此可反推 ad_en 的类别顺序。\n')

w('\n## 3. 22 指标与文本硬证据的 Spearman 相关（A1，n=%s）\n' % format(len(sub), ','))
TEXT = ['frac_alpha', 'frac_digit', 'frac_upper', 'url_per1k', 'latex_per1k',
        'adkw_per1k', 'code_per1k', 'dup_line_frac', 'n_words']
w('| 指标 | ' + ' | '.join(TEXT) + ' |')
w('|---|' + '---|' * len(TEXT))
for f in ALL22:
    vals = []
    for t in TEXT:
        m = sub[[f, t]].dropna()
        vals.append('%.4f' % stats.spearmanr(m[f], m[t]).statistic)
    w('| `%s` | %s |' % (f, ' | '.join(vals)))

w('\n## 4. 域级均值（辅助判定方向：优质域应"教育价值/专业/洁净"更高）\n')
w('| 指标 | ' + ' | '.join(sorted(full['_domain'].unique())) + ' |')
w('|---|' + '---|' * full['_domain'].nunique())
doms = sorted(full['_domain'].unique())
for f in ALL22:
    vals = []
    for d in doms:
        g = full[full['_domain'] == d]
        vals.append('%.3g' % g[f].mean())
    w('| `%s` | %s |' % (f, ' | '.join(vals)))

LOG.close()
print('ok')

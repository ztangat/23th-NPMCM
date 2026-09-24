# -*- coding: utf-8 -*-
"""探测质量信号字段语义：列表型字段长度、取值范围、域分布"""
import os, json, lzma
import numpy as np

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value')
OUT = os.path.join(BASE, '_out', '01_q1')
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, '01a_field_probe.md'), 'w', encoding='utf-8')

def w(s=''):
    LOG.write(str(s) + '\n')

LIST_FIELDS = ['fineweb_edu', 'fluency_en', 'modernbert_cleanliness', 'modernbert_readability',
               'modernbert_reasoning', 'modernbert_professionalism', 'qurater', 'ad_en']
SCALAR_FIELDS = ['dsir_books', 'dsir_math', 'dsir_wiki', 'rps_doc_word_count', 'rps_doc_num_sentences',
                 'rps_doc_unigram_entropy', 'rps_doc_frac_unique_words', 'rps_doc_frac_no_alph_words',
                 'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram',
                 'rps_lines_uppercase_letter_fraction', 'rps_lines_ending_with_terminal_punctution_mark',
                 'rps_lines_numerical_chars_fraction', 'rps_doc_mean_word_length']

files = [
    ('A1_sample', os.path.join(RA, 'slimpajama_quality_signal_sample.jsonl', 'slimpajama_quality_signal_sample.jsonl'), True),
    ('A2_arxiv', os.path.join(RA, 'slimpajama_quality_extended', 'arxiv_part-6777d8857c6e-000486.jsonl', 'arxiv_part-6777d8857c6e-000486.jsonl'), False),
    ('A3_github', os.path.join(RA, 'slimpajama_quality_extended', 'github_part-6777d8857c6e-000275.jsonl', 'github_part-6777d8857c6e-000275.jsonl'), False),
]

N_PROBE = 4000
for name, p, has_domain in files:
    w('\n## %s\n' % name)
    lens = {f: [] for f in LIST_FIELDS}
    vals = {f: [] for f in SCALAR_FIELDS}
    domains = {}
    lf_first = {}
    n = 0
    with open(p, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            if has_domain:
                d = o.get('_source_domain', '?')
                domains[d] = domains.get(d, 0) + 1
            for f in LIST_FIELDS:
                v = o.get(f)
                if v is None:
                    lens[f].append(-1)
                    continue
                if isinstance(v, list):
                    lens[f].append(len(v))
                    if f not in lf_first:
                        lf_first[f] = v
                else:
                    lens[f].append(0)
            for f in SCALAR_FIELDS:
                v = o.get(f)
                try:
                    vals[f].append(float(v))
                except Exception:
                    pass
            if n >= N_PROBE:
                break
    w('- 探测记录数: %d' % n)
    if has_domain:
        w('- `_source_domain` 分布: %s' % json.dumps(domains, ensure_ascii=False))
    w('\n列表型字段长度（唯一值计数）:')
    for f in LIST_FIELDS:
        a = np.array(lens[f])
        u, c = np.unique(a, return_counts=True)
        w('  - `%s`: %s' % (f, dict(zip(u.tolist(), c.tolist()))))
        if f in lf_first:
            w('     首值样例: %s' % str([round(x, 4) for x in lf_first[f]]))
    w('\n标量字段统计（前%s条）:' % N_PROBE)
    w('\n| 字段 | n | min | p1 | p25 | p50 | p75 | p99 | max | mean |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for f in SCALAR_FIELDS:
        a = np.array(vals[f], dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            w('| `%s` | 0 | | | | | | | | |' % f)
            continue
        w('| `%s` | %d | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |' % (
            f, len(a), a.min(), np.percentile(a, 1), np.percentile(a, 25), np.percentile(a, 50),
            np.percentile(a, 75), np.percentile(a, 99), a.max(), a.mean()))

# 收集列表字段原始值的全局取值范围（抽样）
w('\n## 列表型字段取值全域抽样（每个文件取更多样本）\n')
for name, p, has_domain in files:
    agg = {f: [] for f in LIST_FIELDS}
    n = 0
    with open(p, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            if n % 20 == 0:  # 每20条取1条
                for f in LIST_FIELDS:
                    v = o.get(f)
                    if isinstance(v, list):
                        agg[f].extend(v)
            if n >= 60000:
                break
    w('\n### %s (扫描 %d 条, 抽样 %d 条)' % (name, n, n // 20))
    w('| 字段 | n值 | min | p5 | p50 | p95 | max |')
    w('|---|---|---|---|---|---|---|')
    for f in LIST_FIELDS:
        a = np.array(agg[f], dtype=float)
        if len(a) == 0:
            continue
        w('| `%s` | %d | %.4g | %.4g | %.4g | %.4g | %.4g |' % (
            f, len(a), a.min(), np.percentile(a, 5), np.percentile(a, 50), np.percentile(a, 95), a.max()))

LOG.close()
print('ok')

# -*- coding: utf-8 -*-
"""
问题一 · 步骤4a：计算两套域体系在"文本表层统计量"空间的坐标
- 17 个配方域：A18 regmix_domain_sample.jsonl（138,034 条原始文本）
- 7 个质量域：A1 抽样集的 content 字段
用于以定量方式建立 17 域 ↔ 7 域 的跨域关联（替代主观指派）
"""
import os, json, re, math
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value')
OUT = os.path.join(BASE, '_out', '01_q1')

RE_URL = re.compile(r'https?://|www\.', re.I)
RE_LATEX = re.compile(r'\\[a-zA-Z]+\{|\\begin\{|\$[^$]+\$')
RE_ADKW = re.compile(r'\b(buy now|click here|subscribe|sign up|free trial|limited offer|'
                    r'add to cart|shop now|best price|discount|copyright|all rights reserved)\b', re.I)
RE_CODE = re.compile(r'(def |class |import |#include|public static|<\?php|function\s*\()')
RE_WORD = re.compile(r"[A-Za-z']+")
RE_LEGAL = re.compile(r'\b(whereas|hereby|pursuant|plaintiff|defendant|claim\s+\d|'
                      r'patent|embodiment|herein|thereof)\b', re.I)
RE_BIO = re.compile(r'\b(abstract|patients|methods|results|conclusions?|et al\.|'
                    r'doi:|pubmed|protein|gene|cell)\b', re.I)

def doc_stats(txt):
    n = len(txt)
    if n == 0:
        return None
    alpha = sum(c.isalpha() for c in txt)
    digit = sum(c.isdigit() for c in txt)
    upper = sum(c.isupper() for c in txt)
    space = sum(c.isspace() for c in txt)
    words = RE_WORD.findall(txt)
    nw = max(len(words), 1)
    from collections import Counter
    c = Counter(w.lower() for w in words)
    tot = sum(c.values())
    H = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    lines = txt.split('\n')
    nl = max(len(lines), 1)
    dup = 1.0 - len(set(l.strip() for l in lines)) / nl
    return dict(
        n_chars=n, n_words=len(words),
        frac_alpha=alpha / n, frac_digit=digit / n, frac_upper=upper / max(alpha, 1),
        frac_space=space / n,
        url_per1k=len(RE_URL.findall(txt)) * 1000.0 / n,
        latex_per1k=len(RE_LATEX.findall(txt)) * 1000.0 / n,
        adkw_per1k=len(RE_ADKW.findall(txt)) * 1000.0 / n,
        code_per1k=len(RE_CODE.findall(txt)) * 1000.0 / n,
        legal_per1k=len(RE_LEGAL.findall(txt)) * 1000.0 / n,
        bio_per1k=len(RE_BIO.findall(txt)) * 1000.0 / n,
        unigram_entropy=H, type_token_ratio=len(c) / nw,
        mean_word_len=np.mean([len(w) for w in words]) if words else 0,
        dup_line_frac=dup,
    )

rows = []
# ---- A18：17 个配方域 ----
p18 = os.path.join(RA, 'regmix_domain_sample.jsonl', 'regmix_domain_sample.jsonl')
n18 = 0
with open(p18, 'rt', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        t = o.get('text')
        if not isinstance(t, str) or len(t) < 50:
            continue
        s = doc_stats(t)
        if s is None:
            continue
        s['_domain'] = o.get('_source_domain')
        s['_system'] = 'mixture17'
        rows.append(s)
        n18 += 1
        if n18 % 30000 == 0:
            pass

# ---- A1：7 个质量域 ----
full = pd.read_parquet(os.path.join(OUT, 'quality_matrix_raw.parquet'))
A1 = full[full['_source_file'] == 'A1_sample']
p1 = os.path.join(RA, 'slimpajama_quality_signal_sample.jsonl', 'slimpajama_quality_signal_sample.jsonl')
need = set(A1['_id'].astype(str))
n1 = 0
with open(p1, 'rt', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        t = o.get('content')
        if not isinstance(t, str) or len(t) < 50:
            continue
        s = doc_stats(t)
        if s is None:
            continue
        s['_domain'] = o.get('_source_domain')
        s['_system'] = 'quality7'
        rows.append(s)
        n1 += 1

df = pd.DataFrame(rows)
df.to_parquet(os.path.join(OUT, 'textstats_by_domain_raw.parquet'), index=False)

# 域级聚合
STATS = ['frac_alpha', 'frac_digit', 'frac_upper', 'frac_space', 'url_per1k',
         'latex_per1k', 'adkw_per1k', 'code_per1k', 'legal_per1k', 'bio_per1k',
         'unigram_entropy', 'type_token_ratio', 'mean_word_len', 'dup_line_frac',
         'n_chars', 'n_words']
agg = df.groupby(['_system', '_domain']).agg(
    n=('n_chars', 'size'), **{s: (s, 'mean') for s in STATS}).reset_index()
agg['ln_chars'] = np.log1p(agg['n_chars'])
agg['ln_words'] = np.log1p(agg['n_words'])
agg.to_csv(os.path.join(OUT, 'textstats_domain_profile.csv'), index=False, encoding='utf-8-sig')

log = open(os.path.join(OUT, '01e_textstats_bridge.md'), 'w', encoding='utf-8')
def w(s=''):
    log.write(str(s) + '\n')
w('# 跨域关联：17 配方域 vs 7 质量域 的文本表层统计量画像\n')
w('\n- A18（配方域原始文本）计入 %s 条；A1（质量域原始文本）计入 %s 条\n' % (format(n18, ','), format(n1, ',')))
COLS = ['frac_alpha', 'frac_digit', 'frac_upper', 'frac_space', 'url_per1k', 'latex_per1k',
        'adkw_per1k', 'code_per1k', 'legal_per1k', 'bio_per1k', 'unigram_entropy',
        'type_token_ratio', 'mean_word_len', 'dup_line_frac', 'ln_chars']
w('\n| 体系 | 域 | n | ' + ' | '.join(COLS) + ' |')
w('|---|---|---|' + '---|' * len(COLS))
for _, r in agg.iterrows():
    w('| %s | %s | %s | %s |' % (r['_system'], r['_domain'], format(int(r['n']), ','),
                                 ' | '.join('%.4g' % r[c] for c in COLS)))
log.close()
print('ok')

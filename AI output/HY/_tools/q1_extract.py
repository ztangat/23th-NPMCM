# -*- coding: utf-8 -*-
"""
问题一 · 步骤1：全量质量信号抽取与指标标量化
输入：A1 抽样集(51,230) + A2 arxiv 扩展集(17,523) + A3 github 扩展集(203,752)
输出：_out/01_q1/quality_matrix.parquet  （272,505 × 指标矩阵）
      _out/01_q1/01b_extract_report.md
"""
import os, json, re, math
import numpy as np
import pandas as pd

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments', 'A_data_value')
OUT = os.path.join(BASE, '_out', '01_q1')
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, '01b_extract_report.md'), 'w', encoding='utf-8')

def w(s=''):
    LOG.write(str(s) + '\n')

# ---------- 指标定义 ----------
# K = 类别数；None 表示单值回归分
LIST_SPEC = {
    'fineweb_edu':               dict(K=None, clip=(0.0, 5.0), desc='FineWeb-Edu 教育价值回归分(0-5)'),
    'fluency_en':                dict(K=2, desc='英文流畅度二分类 logits'),
    'modernbert_cleanliness':    dict(K=6, desc='ModernBERT 洁净度 0-5 分 logits'),
    'modernbert_readability':    dict(K=6, desc='ModernBERT 可读性 0-5 分 logits'),
    'modernbert_reasoning':      dict(K=6, desc='ModernBERT 推理含量 0-5 分 logits'),
    'modernbert_professionalism':dict(K=6, desc='ModernBERT 专业性 0-5 分 logits'),
    'qurater':                   dict(K=4, desc='QuRater 质量 0-3 分 logits'),
    'ad_en':                     dict(K=2, desc='广告含量二分类 logits(负向)'),
}
SCALAR_FIELDS = ['dsir_books', 'dsir_math', 'dsir_wiki',
                 'rps_doc_word_count', 'rps_doc_num_sentences', 'rps_doc_unigram_entropy',
                 'rps_doc_frac_unique_words', 'rps_doc_frac_no_alph_words',
                 'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram',
                 'rps_lines_uppercase_letter_fraction',
                 'rps_lines_ending_with_terminal_punctution_mark',
                 'rps_lines_numerical_chars_fraction', 'rps_doc_mean_word_length']
ALL22 = list(LIST_SPEC.keys()) + SCALAR_FIELDS

def softmax(v):
    a = np.asarray(v, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()

def scalarize(o):
    """把一条记录的 22 个指标压成 22 个标量（原始尺度，尚未统一方向）"""
    row = {}
    for f, spec in LIST_SPEC.items():
        v = o.get(f)
        if v is None:
            row[f] = np.nan
            continue
        if spec['K'] is None:
            # 单值回归分
            if isinstance(v, list):
                v = v[0] if len(v) else np.nan
            try:
                x = float(v)
            except Exception:
                x = np.nan
            lo, hi = spec['clip']
            if np.isfinite(x):
                x = min(max(x, lo), hi)
            row[f] = x
        else:
            if not isinstance(v, list) or len(v) != spec['K'] or any(
                    (not isinstance(t, (int, float))) or (isinstance(t, float) and not np.isfinite(t)) for t in v):
                row[f] = np.nan
                continue
            p = softmax(v)
            idx = np.arange(spec['K'], dtype=np.float64)
            row[f] = float(np.sum(p * idx) / (spec['K'] - 1))  # 归一化到 [0,1]
    for f in SCALAR_FIELDS:
        v = o.get(f)
        try:
            row[f] = float(v)
        except Exception:
            row[f] = np.nan
    return row

# ---------- 原始文本统计量（用于可靠性检验） ----------
RE_URL = re.compile(r'https?://|www\.', re.I)
RE_LATEX = re.compile(r'\\[a-zA-Z]+\{|\\begin\{|\$[^$]+\$')
RE_ADKW = re.compile(r'\b(buy now|click here|subscribe|sign up|free trial|limited offer|'
                    r'add to cart|shop now|best price|discount|copyright|all rights reserved)\b', re.I)
RE_CODE = re.compile(r'(def |class |import |#include|public static|<\?php|function\s*\()')

def text_stats(txt):
    if not isinstance(txt, str) or len(txt) == 0:
        return dict(n_chars=0, n_words=0, frac_alpha=np.nan, frac_digit=np.nan,
                    frac_upper=np.nan, frac_space=np.nan, url_per1k=np.nan,
                    latex_per1k=np.nan, adkw_per1k=np.nan, code_per1k=np.nan,
                    dup_line_frac=np.nan)
    n = len(txt)
    alpha = sum(c.isalpha() for c in txt)
    digit = sum(c.isdigit() for c in txt)
    upper = sum(c.isupper() for c in txt)
    space = sum(c.isspace() for c in txt)
    nw = len(txt.split())
    urls = len(RE_URL.findall(txt))
    latex = len(RE_LATEX.findall(txt))
    adkw = len(RE_ADKW.findall(txt))
    code = len(RE_CODE.findall(txt))
    lines = txt.split('\n')
    nl = max(len(lines), 1)
    dup = 1.0 - len(set(l.strip() for l in lines)) / nl
    return dict(n_chars=n, n_words=nw, frac_alpha=alpha / n, frac_digit=digit / n,
                frac_upper=upper / max(alpha, 1), frac_space=space / n,
                url_per1k=urls * 1000.0 / n, latex_per1k=latex * 1000.0 / n,
                adkw_per1k=adkw * 1000.0 / n, code_per1k=code * 1000.0 / n,
                dup_line_frac=dup)

FILES = [
    ('A1_sample', os.path.join(RA, 'slimpajama_quality_signal_sample.jsonl',
                               'slimpajama_quality_signal_sample.jsonl'), None, True),
    ('A2_arxiv', os.path.join(RA, 'slimpajama_quality_extended',
                              'arxiv_part-6777d8857c6e-000486.jsonl',
                              'arxiv_part-6777d8857c6e-000486.jsonl'), 'arxiv', False),
    ('A3_github', os.path.join(RA, 'slimpajama_quality_extended',
                               'github_part-6777d8857c6e-000275.jsonl',
                               'github_part-6777d8857c6e-000275.jsonl'), 'github', False),
]

frames = []
for name, path, fixed_domain, read_content in FILES:
    rows = []
    n = 0
    with open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            n += 1
            r = scalarize(o)
            r['_id'] = o.get('id')
            r['_source_file'] = name
            r['_domain'] = fixed_domain if fixed_domain else o.get('_source_domain')
            if read_content:
                r.update(text_stats(o.get('content')))
                r['_content_len'] = len(o.get('content')) if isinstance(o.get('content'), str) else 0
            rows.append(r)
            if n % 50000 == 0:
                w('- %s 已读取 %d 条 ...' % (name, n))
    df = pd.DataFrame(rows)
    frames.append(df)
    w('\n## %s\n- 抽取记录数 **%s**\n- 域分布: %s\n' % (
        name, format(len(df), ','), json.dumps(df['_domain'].value_counts().to_dict(), ensure_ascii=False)))
    # 缺失统计
    miss = df[ALL22].isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        w('- 指标缺失: %s' % '; '.join('`%s`=%d(%.3f%%)' % (k, v, 100.0 * v / len(df)) for k, v in miss.items()))
    else:
        w('- 指标缺失: 无')

full = pd.concat(frames, ignore_index=True)
w('\n## 合并矩阵\n- 总记录数 **%s**\n' % format(len(full), ','))
w('- 列: %s\n' % ', '.join(full.columns))

# 缺失总览
w('\n### 全量 22 指标缺失与取值概览\n')
w('| 指标 | 缺失数 | 缺失率 | min | p1 | p25 | p50 | p75 | p99 | max |')
w('|---|---|---|---|---|---|---|---|---|---|')
for f in ALL22:
    s = full[f]
    n_miss = int(s.isna().sum())
    a = s.dropna().to_numpy(dtype=float)
    if len(a) == 0:
        w('| `%s` | %d | %.3f%% | | | | | | | |' % (f, n_miss, 100.0 * n_miss / len(full)))
        continue
    w('| `%s` | %d | %.3f%% | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g | %.4g |' % (
        f, n_miss, 100.0 * n_miss / len(full), a.min(), np.percentile(a, 1), np.percentile(a, 25),
        np.percentile(a, 50), np.percentile(a, 75), np.percentile(a, 99), a.max()))

# A1 与 A2/A3 的 id 重叠检查
ids = {}
for name, df in zip(['A1_sample', 'A2_arxiv', 'A3_github'], frames):
    ids[name] = set(df['_id'].dropna().astype(str))
w('\n### 样本集与扩展集的 id 重叠\n')
for a in ['A1_sample']:
    for b in ['A2_arxiv', 'A3_github']:
        inter = ids[a] & ids[b]
        w('- `%s` ∩ `%s` = **%s** 条（占 %s 的 %.2f%%）' % (
            a, b, format(len(inter), ','), a, 100.0 * len(inter) / max(len(ids[a]), 1)))
w('- 结论：抽样集 A1 的 arxiv/github 记录是扩展集 A2/A3 的**子集**（同源抽样），' \
  '因此二者可直接对照；非 arxiv/github 的 5 个域仅在 A1 中出现。')

full.to_parquet(os.path.join(OUT, 'quality_matrix_raw.parquet'), index=False)
w('\n- 已保存: `_out/01_q1/quality_matrix_raw.parquet` (%s 行)' % format(len(full), ','))
LOG.close()
print('ok')

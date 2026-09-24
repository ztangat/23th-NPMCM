"""S01 流式读取质量信号 JSONL(.xz)，把 8 个列表型指标压缩为标量，输出样本级原始标量表。
支持 A1(抽样集, 含 content/_source_domain) 与 A2/A3(扩展集, 域名由文件名推断)。
列表→标量规则(依据 Meta-rater / QuRating / FineWeb-Edu 模型输出结构):
  modernbert_{cleanliness,readability,reasoning,professionalism}: 6 个 logits(对应 0–5 分) → softmax 期望分 E[k]∈[0,5]
  fineweb_edu: 长度 1 的回归分 → 取该值
  ad_en / fluency_en: 2 个 logits → softmax, 保存第 2 类概率 p1; 哪一类为"广告/流畅"在 S02 用 arxiv 锚定检验
  qurater: 4 个维度分(writing_style, required_expertise, facts_trivia, educational_value) → 等权均值, 并保留4个分量
另记录: 列表长度异常/缺失/非有限值计数 → 预处理日志。
content 字段(仅 A1)只在线提取统计量与前 300 字符摘录, 不整体入库。
用法: python s01_parse_quality.py <数据目录> <输出目录>
"""
import lzma, json, sys, os, math, re, glob
import numpy as np, pandas as pd

SCALAR = ['dsir_books', 'dsir_wiki', 'dsir_math', 'rps_doc_word_count', 'rps_doc_num_sentences',
          'rps_doc_unigram_entropy', 'rps_doc_frac_unique_words', 'rps_doc_frac_no_alph_words',
          'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram', 'rps_lines_uppercase_letter_fraction',
          'rps_lines_ending_with_terminal_punctution_mark', 'rps_lines_numerical_chars_fraction',
          'rps_doc_mean_word_length']
LIST6 = ['modernbert_cleanliness', 'modernbert_readability', 'modernbert_reasoning', 'modernbert_professionalism']
QR = ['qr_writing_style', 'qr_required_expertise', 'qr_facts_trivia', 'qr_educational_value']


def softmax(v):
    v = np.asarray(v, float); e = np.exp(v - v.max()); return e / e.sum()


def fnum(v):
    try:
        v = float(v); return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def parse(path, domain_from_name=None, keep_text=False):
    rows = []
    log = {'file': os.path.basename(path), 'n_lines_parsed': 0, 'bad_json': 0, 'missing': {}, 'bad_list_len': {}}
    op = lzma.open if path.endswith('.xz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                log['bad_json'] += 1; continue
            log['n_lines_parsed'] += 1
            o = {'id': r.get('id'), 'domain': r.get('_source_domain') or domain_from_name,
                 'source_file': log['file']}
            for k in SCALAR:
                v = fnum(r.get(k))
                if v != v:
                    log['missing'][k] = log['missing'].get(k, 0) + 1
                o[k] = v
            for k in LIST6:
                v = r.get(k)
                if not isinstance(v, list) or len(v) != 6:
                    log['bad_list_len'][k] = log['bad_list_len'].get(k, 0) + 1; o[k] = np.nan; continue
                o[k] = float((softmax(v) * np.arange(6)).sum())
            v = r.get('fineweb_edu')
            o['fineweb_edu'] = fnum(v[0]) if isinstance(v, list) and v else fnum(v)
            for k in ['ad_en', 'fluency_en']:
                v = r.get(k)
                if isinstance(v, list) and len(v) == 2:
                    o[k + '_p1'] = float(softmax(v)[1])
                else:
                    log['bad_list_len'][k] = log['bad_list_len'].get(k, 0) + 1; o[k + '_p1'] = np.nan
            v = r.get('qurater')
            if isinstance(v, list) and len(v) == 4:
                for j, nm in enumerate(QR):
                    o[nm] = fnum(v[j])
                o['qurater'] = float(np.nanmean([fnum(x) for x in v]))
            else:
                log['bad_list_len']['qurater'] = log['bad_list_len'].get('qurater', 0) + 1; o['qurater'] = np.nan
            c = r.get('content')
            if isinstance(c, str):
                o['n_chars'] = len(c); o['n_lines'] = c.count('\n') + 1
                o['code_sym_frac'] = sum(c.count(s) for s in '{}();=<>[]#') / max(len(c), 1)
                o['url_count'] = len(re.findall(r'https?://', c))
                if keep_text:
                    o['excerpt'] = c[:300].replace('\n', ' ')
            rows.append(o)
    return pd.DataFrame(rows), log


if __name__ == '__main__':
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    jobs = []
    a1 = [p for p in sorted(glob.glob(os.path.join(data, 'slimpajama_quality_signal_sample.jsonl*'))) if os.path.isfile(p)]
    if a1:
        jobs.append(('A1', a1[0], None))
    for p in sorted(g for g in glob.glob(os.path.join(data, '*_part-*.jsonl*')) if os.path.isfile(g)):
        b = os.path.basename(p)
        jobs.append(('A2' if b.startswith('arxiv') else 'A3', p, b.split('_part')[0]))
    logs, frames = [], []
    for tag, p, dom in jobs:
        df, log = parse(p, dom, keep_text=(tag == 'A1'))
        df['set'] = tag; frames.append(df); log['set'] = tag; logs.append(log)
        print(tag, p, len(df), flush=True)
    allq = pd.concat(frames, ignore_index=True)
    allq.to_csv(os.path.join(out, 'S01_sample_level_raw_scalars.csv.gz'), index=False, compression='gzip')
    json.dump(logs, open(os.path.join(out, 'S01_parse_log.json'), 'w'), ensure_ascii=False, indent=1)
    print(allq.groupby(['set', 'domain']).size())

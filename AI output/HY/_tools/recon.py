# -*- coding: utf-8 -*-
"""阶段0：数据勘察。输出所有文件的结构快照到 _out/00_recon/"""
import os, sys, json, io, csv, lzma, gzip
import numpy as np
import pandas as pd

BASE = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(BASE, 'real_attachments')
OUT = os.path.join(BASE, '_out', '00_recon')
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, '00_data_reconnaissance.md'), 'w', encoding='utf-8')

def w(s=''):
    LOG.write(str(s) + '\n')

w('# 阶段0 数据勘察报告\n')
w('工作目录: `%s`\n' % BASE)

# ---------- 1. 目录树与体积 ----------
w('\n## 1. 目录结构与体积\n')
tree_rows = []
def walk(d, depth=0, maxdepth=3):
    try:
        ents = sorted(os.listdir(d))
    except Exception as e:
        return
    dirs = [e for e in ents if os.path.isdir(os.path.join(d, e))]
    files = [e for e in ents if os.path.isfile(os.path.join(d, e))]
    tot = 0
    for f in files:
        try:
            tot += os.path.getsize(os.path.join(d, f))
        except Exception:
            pass
    tree_rows.append((depth, d.replace(BASE, '.'), len(dirs), len(files), tot))
    if depth < maxdepth:
        for dd in dirs:
            walk(os.path.join(d, dd), depth+1, maxdepth)

walk(RA, 0, 2)
w('| 层级 | 目录 | 子目录数 | 文件数 | 文件总体积(bytes) |')
w('|---|---|---|---|---|')
for depth, d, nd, nf, tot in tree_rows:
    w('| %d | `%s` | %d | %d | %s |' % (depth, d, nd, nf, format(tot, ',')))

# ---------- 2. CSV 全量勘察 ----------
w('\n## 2. CSV 文件勘察\n')
csv_files = []
for root, dirs, files in os.walk(RA):
    for f in files:
        if f.lower().endswith('.csv'):
            csv_files.append(os.path.join(root, f))
csv_files.sort()

for p in csv_files:
    rel = os.path.relpath(p, RA).replace('\\', '/')
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        w('\n### `%s`\n读取失败: %s\n' % (rel, e))
        continue
    w('\n### `%s`\n' % rel)
    w('- 形状: **%d 行 × %d 列**  文件大小 %s bytes' % (df.shape[0], df.shape[1], format(os.path.getsize(p), ',')))
    w('- 列名: %s' % ', '.join('`%s`' % c for c in df.columns))
    # 缺失
    na = df.isna().sum()
    na = na[na > 0]
    if len(na):
        w('- 缺失: ' + '; '.join('`%s`=%d' % (k, v) for k, v in na.items()))
    else:
        w('- 缺失: 无')
    # 数值列描述
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        desc = num.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
        desc = desc.round(6)
        w('\n| 列 | count | mean | std | min | 50% | max |')
        w('|---|---|---|---|---|---|---|')
        for idx, r in desc.iterrows():
            w('| `%s` | %s | %s | %s | %s | %s | %s |' % (
                idx,
                ('' if pd.isna(r['count']) else '%.0f' % r['count']),
                ('' if pd.isna(r['mean']) else '%.6g' % r['mean']),
                ('' if pd.isna(r['std']) else '%.6g' % r['std']),
                ('' if pd.isna(r['min']) else '%.6g' % r['min']),
                ('' if pd.isna(r['50%']) else '%.6g' % r['50%']),
                ('' if pd.isna(r['max']) else '%.6g' % r['max'])))
    # 对象列取值样例
    obj = df.select_dtypes(exclude=[np.number])
    if obj.shape[1] > 0 and df.shape[0] > 0:
        w('\n对象列取值样例:')
        for c in obj.columns:
            vc = obj[c].astype(str).value_counts()
            w('  - `%s`: %d 个不同取值; top5 = %s' % (
                c, len(vc), ', '.join('%s(%d)' % (k[:40], v) for k, v in vc.head(5).items())))
    # 保存 schema
    with open(os.path.join(OUT, 'schema_' + rel.replace('/', '__') + '.json'), 'w', encoding='utf-8') as f:
        json.dump({'path': rel, 'shape': list(df.shape),
                   'columns': list(df.columns),
                   'dtypes': {c: str(t) for c, t in df.dtypes.items()},
                   'na_counts': {k: int(v) for k, v in df.isna().sum().items()}},
                  f, ensure_ascii=False, indent=1)

# ---------- 3. JSONL.XZ 勘察 ----------
w('\n## 3. JSONL(.xz) 文件勘察\n')
xz_files = []
for root, dirs, files in os.walk(RA):
    for f in files:
        if f.endswith('.jsonl.xz') or f.endswith('.jsonl'):
            xz_files.append(os.path.join(root, f))
xz_files.sort()

for p in xz_files:
    rel = os.path.relpath(p, RA).replace('\\', '/')
    op = lzma.open if p.endswith('.xz') else open
    w('\n### `%s`  (%s bytes)' % (rel, format(os.path.getsize(p), ',')))
    n = 0
    keys = None
    sample = []
    try:
        with op(p, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if keys is None:
                    keys = list(o.keys())
                if n < 2:
                    sample.append(o)
                n += 1
    except Exception as e:
        w('- 遍历失败: %s' % e)
        continue
    w('- 记录数: **%s**' % format(n, ','))
    w('- 字段数: %d' % len(keys))
    w('- 字段: %s' % ', '.join('`%s`' % k for k in keys))
    if sample:
        o = sample[0]
        w('- 首条记录字段类型与示例值:')
        for k in keys:
            v = o.get(k)
            t = type(v).__name__
            sv = str(v)
            if len(sv) > 120:
                sv = sv[:120] + '...(截断)'
            w('  - `%s` : %s = %s' % (k, t, sv))
    # 存 schema
    with open(os.path.join(OUT, 'schema_' + rel.replace('/', '__') + '.json'), 'w', encoding='utf-8') as f:
        json.dump({'path': rel, 'n_records': n, 'fields': keys}, f, ensure_ascii=False, indent=1)

# ---------- 4. 其他文件 ----------
w('\n## 4. 其他文件\n')
for root, dirs, files in os.walk(RA):
    for f in files:
        if not (f.endswith('.csv') or f.endswith('.jsonl.xz') or f.endswith('.jsonl')):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, RA).replace('\\', '/')
            w('- `%s`  %s bytes' % (rel, format(os.path.getsize(p), ',')))

LOG.close()
print('done')

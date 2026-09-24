# -*- coding: utf-8 -*-
"""勘察 B8 大规模 NQ 半合成集的异质性：为何 kappa 网格搜索给出 0"""
import os, io, sys, json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RA = r'C:\Users\dkyyt\Desktop\F题\real_attachments\B_scaling_laws'
OUT = r'C:\Users\dkyyt\Desktop\F题\_out\02_q2'
os.makedirs(OUT, exist_ok=True)

lines = []
def w(s=''):
    lines.append(s)

for fn in ['supplementary_NQ_experiment.csv',
           'supplementary_NQ_experiment_expanded.csv',
           'supplementary_NQ_experiment_large.csv',
           'supplementary_large_models.csv',
           'supplementary_large_baseline.csv',
           'open_model_family_metadata.csv',
           'pythia_checkpoint_index.csv']:
    p = os.path.join(RA, fn)
    df = pd.read_csv(p, dtype_backend='numpy_nullable')
    w('\n## %s  n=%d\n' % (fn, len(df)))
    w('列：%s\n' % ', '.join('`%s`' % c for c in df.columns))
    # 逐列画像
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype(str)
            vc = s.value_counts(dropna=False)
            if len(vc) <= 25:
                w('- `%s`（object，%d 类）：%s' % (c, len(vc),
                    '；'.join('%s=%d' % (k, v) for k, v in vc.items())))
            else:
                w('- `%s`（object，%d 类，示例 %s）' % (c, len(vc), list(vc.index[:5])))
        else:
            w('- `%s`：min=%s，25%%=%s，中位数=%s，75%%=%s，max=%s，缺失=%d' % (
                c, s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max(), int(s.isna().sum())))
    w('')

# B8 特殊：寻找可能的子集标记
b8 = pd.read_csv(os.path.join(RA, 'supplementary_NQ_experiment_large.csv'), dtype_backend='numpy_nullable')
w('\n### B8 分类列交叉\n')
obj = [c for c in b8.columns if not pd.api.types.is_numeric_dtype(b8[c])]
w('object 列：%s\n' % obj)
for c in obj:
    w('\n按 `%s` 分组的 val_loss 统计：\n' % c)
    g = b8.groupby(c)['val_loss'].agg(['count', 'mean', 'min', 'max'])
    w(g.to_string())
    w('')

# 是否有极端行
w('\n### B8 极端值检查\n')
w('val_loss < 1.0 的行数：%d\n' % int((b8['val_loss'] < 1.0).sum()))
if (b8['val_loss'] < 1.0).any():
    w(b8[b8['val_loss'] < 1.0].head(30).to_string())
    w('')
w('val_loss 分布分位：%s\n' % np.round(np.percentile(b8['val_loss'].dropna(),
      [0, 1, 5, 25, 50, 75, 95, 99, 100]), 4).tolist())

with open(os.path.join(OUT, '02b_B8_probe.md'), 'w', encoding='utf-8') as f:
    f.write('# 问题二 · B6–B10 半合成/大模型数据结构勘察\n')
    f.write('\n'.join(lines))
print('ok')

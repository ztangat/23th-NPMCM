# -*- coding: utf-8 -*-
"""诊断：C8 聚合值 与 C1 汇总值的系统性偏移来源（修正键值用反的 bug）"""
import os, io, sys, json, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'C:\Users\dkyyt\Desktop\F题'
CE = os.path.join(ROOT, 'real_attachments', 'C_efficiency_evolution')
OUT = os.path.join(ROOT, '_out', '04_q4')
DR = os.path.join(CE, 'detailed_results')
BUF = []
def w(s=''): BUF.append(s)

C1 = pd.read_csv(os.path.join(CE, 'leaderboard_cleaned.csv'))
# dir -> Model（键是目录名）
d2m = {}
for m in C1['Model'].astype(str):
    d2m.setdefault(m.replace('/', '_'), m)

w('# 诊断：C8 聚合与 C1 汇总的系统性偏移\n')
w('\n- C1 共 %d 行 / %d 个唯一 `Model`（同一模型多次提交，需去重）。\n'
  % (len(C1), C1['Model'].nunique()))
d = C1[C1['Model'].duplicated(keep=False)].sort_values('Model')
w('\n### 重复提交示例（同一 `Model` 多行，按 Type/日期区分）\n')
w('\n| Model | #Params | 提交日期 | Type | Average |')
w('|---|---|---|---|---|')
for _, r in d.head(10).iterrows():
    w('| %s | %s | %s | %s | %.4f |' % (r['Model'], r['#Params (B)'],
                                        r['Submission Date'], r['Type'], r['Average ⬆️']))

def gval(gr, res, tk):
    """从 groups（值可能是 dict）或 results 取标量 acc_norm"""
    v = gr.get(tk)
    if isinstance(v, dict):
        v = v.get('acc_norm,none', v.get('acc,none'))
    if not isinstance(v, (int, float)):
        v = res.get(tk, {}).get('acc_norm,none')
    return float(v) * 100.0 if isinstance(v, (int, float)) else np.nan


def load(dname):
    fs = sorted(glob.glob(os.path.join(DR, dname, '*.json')))
    for f in reversed(fs):
        try:
            return json.load(open(f, encoding='utf-8')), len(fs)
        except Exception:
            continue
    return None, len(fs)

tests = ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct',
         'mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-2-9b-it',
         '0-hero/Matter-0.2-7B-DPO', 'Qwen/Qwen2.5-72B-Instruct',
         'microsoft/Phi-3-mini-4k-instruct']
w('\n## 1. 逐个模型的对照（BBH）\n')
w('\n| 模型 | 目录存在 | C1 `BBH` | `results[leaderboard_bbh].acc_norm` | `groups[leaderboard_bbh]` | '
  '子任务均值 | 子任务数 | `n-shot` | JSON 数 |')
w('|---|---|---|---|---|---|---|---|---|')
for m in tests:
    dname = m.replace('/', '_')
    ok = os.path.isdir(os.path.join(DR, dname))
    if not ok:
        w('| %s | 否 | — | — | — | — | — | — | — |' % m); continue
    rec, nf = load(dname)
    if rec is None:
        w('| %s | 是 | — | — | — | — | — | — | %d |' % (m, nf)); continue
    res = rec.get('results', {}); gr = rec.get('groups', {})
    rb = res.get('leaderboard_bbh', {}).get('acc_norm,none')
    gb = gr.get('leaderboard_bbh')
    if isinstance(gb, dict):
        gb = gb.get('acc_norm,none', gb.get('acc,none'))
    subs = [t for t in res if t.startswith('leaderboard_bbh_')]
    ms = np.mean([res[t].get('acc_norm,none') for t in subs
                  if isinstance(res[t].get('acc_norm,none'), (int, float))])
    c1v = C1.loc[C1['Model'] == m, 'BBH']
    c1v = float(c1v.iloc[0]) if len(c1v) else np.nan
    w('| %s | 是 | %.4f | %s | %s | %.4f | %d | %s | %d |' % (
        m, c1v,
        ('%.4f' % (rb * 100)) if isinstance(rb, (int, float)) else '—',
        ('%.4f' % (gb * 100)) if isinstance(gb, (int, float)) else '—',
        ms * 100, len(subs), str(rec.get('n-shot')), nf))

PAIRS = [('IFEval', 'leaderboard_ifeval'), ('BBH', 'leaderboard_bbh'),
         ('MATH Lvl 5', 'leaderboard_math_hard'), ('GPQA', 'leaderboard_gpqa'),
         ('MUSR', 'leaderboard_musr'), ('MMLU-PRO', 'leaderboard_mmlu_pro')]

w('\n## 2. 六维逐一对照（同上批模型）\n')
w('\n| 模型 | 维度 | C1 | `results.acc_norm` | `groups` |')
w('|---|---|---|---|---|')
for m in tests[:6]:
    dname = m.replace('/', '_')
    if not os.path.isdir(os.path.join(DR, dname)):
        continue
    rec, _ = load(dname)
    if rec is None:
        continue
    res = rec.get('results', {}); gr = rec.get('groups', {})
    for c1c, tk in PAIRS:
        c1v = C1.loc[C1['Model'] == m, c1c]
        c1v = float(c1v.iloc[0]) if len(c1v) else np.nan
        rv = res.get(tk, {}).get('acc_norm,none')
        gv = gr.get(tk)
        if isinstance(gv, dict):
            gv = gv.get('acc_norm,none', gv.get('acc,none'))
        w('| %s | %s | %.4f | %s | %s |' % (
            m, c1c, c1v,
            ('%.4f' % (rv * 100)) if isinstance(rv, (int, float)) else '—',
            ('%.4f' % (gv * 100)) if isinstance(gv, (int, float)) else '—'))

# ---- 全样本回归 ----
w('\n## 3. 全样本：C1 值（纵轴）对 C8 聚合值（横轴）的回归\n')
C1u = C1.drop_duplicates('Model', keep='first')
rows = []
for dname in sorted(os.listdir(DR)):
    m = d2m.get(dname)
    if m is None:
        continue
    rec, _ = load(dname)
    if rec is None:
        continue
    res = rec.get('results', {}); gr = rec.get('groups', {})
    row = {'Model': m}
    for c1c, tk in PAIRS:
        row['C8_' + c1c] = gval(gr, res, tk)
    rows.append(row)
A = pd.DataFrame(rows).merge(C1u, on='Model', how='inner')
w('\n- 可比对 %d 个模型（C1 去重后 %d 行）。\n' % (len(A), len(C1u)))
w('\n| 维度 | $n$ | 斜率 $b$ | 截距 $a$ | $R^2$ | C1 均值 | C8 均值 | 均值差 |')
w('|---|---|---|---|---|---|---|---|')
for c1c, tk in PAIRS:
    x = A['C8_' + c1c]; y = A[c1c]
    ok = x.notna() & y.notna()
    if ok.sum() < 50:
        w('| %s | %d | — | — | — | — | — | — |' % (c1c, ok.sum())); continue
    b, a = np.polyfit(x[ok], y[ok], 1)
    pr = np.corrcoef(x[ok], y[ok])[0, 1]
    w('| %s | %d | %.4f | %+.4f | %.4f | %.3f | %.3f | %+.3f |' % (
        c1c, ok.sum(), b, a, pr ** 2, y[ok].mean(), x[ok].mean(),
        y[ok].mean() - x[ok].mean()))

# ---- BBH 子任务清单 ----
w('\n## 4. BBH 子任务清单与"24 项 vs 27 项"检验\n')
dname = 'Qwen_Qwen2.5-7B-Instruct'
rec, _ = load(dname)
if rec:
    res = rec['results']; gr = rec.get('groups', {})
    subs = sorted([t for t in res if t.startswith('leaderboard_bbh_')])
    w('\n- 该模型 JSON 中 `leaderboard_bbh_*` 子任务共 **%d** 项（官方 BBH 为 27 项）。\n' % len(subs))
    w('\n| # | 子任务 | acc_norm |')
    w('|---|---|---|')
    for i, t in enumerate(subs, 1):
        v = res[t].get('acc_norm,none')
        w('| %d | `%s` | %s |' % (i, t, ('%.4f' % v) if isinstance(v, (int, float)) else '—'))
    ms = np.mean([res[t]['acc_norm,none'] for t in subs])
    gbv = gr.get('leaderboard_bbh')
    if isinstance(gbv, dict):
        gbv = gbv.get('acc_norm,none', gbv.get('acc,none'))
    w('\n- 子任务均值 = %.4f；`groups[leaderboard_bbh]` = %s；'
      '`results[leaderboard_bbh]` = %s；C1 `BBH` = %.4f\n'
      % (ms,
         ('%.4f' % gbv) if isinstance(gbv, (int, float)) else '—',
         ('%.4f' % res['leaderboard_bbh']['acc_norm,none']) if 'leaderboard_bbh' in res else '—',
         float(C1u.loc[C1u['Model'] == 'Qwen/Qwen2.5-7B-Instruct', 'BBH'].iloc[0])))
    # 官方 27 项缺了哪 3 项？
    w('\n- `group_subtasks[leaderboard_bbh]` 声明的子任务数 = %d\n'
      % len(rec.get('group_subtasks', {}).get('leaderboard_bbh', [])))

with open(os.path.join(OUT, '04c_c8_offset_diagnosis.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

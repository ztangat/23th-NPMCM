# -*- coding: utf-8 -*-
"""问题四 · C8 逐任务聚合（修正版）

修正两点：
 1. 目录名 -> Model 的还原：C1 的 Model 名把非字母数字之外的 '/' 替换为 '_'，
    故正确逆映射是建立 {Model.replace('/','_'): Model} 字典后精确查表，
    而非把 '_' 全替换成 '/'（会破坏模型名中原有的下划线）。
 2. 六维归属：直接采用 JSON 的 `group_subtasks` 字段（权威分组定义），
    而不是靠任务名前缀猜。
"""
import os, io, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
CE = os.path.join(ROOT, 'real_attachments', 'C_efficiency_evolution')
OUT = os.path.join(ROOT, '_out', '04_q4')
os.makedirs(OUT, exist_ok=True)
BUF = []
def w(s=''): BUF.append(s)

w('# 问题四 · C8 逐任务聚合（修正版）\n')

DR = os.path.join(CE, 'detailed_results')
C1 = pd.read_csv(os.path.join(CE, 'leaderboard_cleaned.csv'))

# ---- 1. 目录名 -> Model 的精确逆映射 ----
sanit = {}
for m in C1['Model'].astype(str):
    sanit[m.replace('/', '_')] = m
    sanit[m.replace('/', '_').replace('.', '_')] = m
dirs = sorted(os.listdir(DR))
map_ok = sum(1 for d in dirs if d in sanit)
w('\n## 1. 目录名 → 模型名的还原\n')
w('\n- 目录总数 %d；用 `{Model.replace("/","_")}` 精确查表可还原 **%d** 个（%.1f%%）。\n'
  % (len(dirs), map_ok, 100 * map_ok / len(dirs)))
w('- 无法还原的 %d 个目录（C1 中无对应记录）示例：%s\n'
  % (len(dirs) - map_ok,
     ', '.join(d for d in dirs if d not in sanit)[:3] or '—'))

# ---- 2. 先探查 group_subtasks ----
w('\n## 2. 分组定义（取自 JSON 的 `group_subtasks`）\n')
gs_union = {}
grp_union = {}
njson = 0
for d in dirs[:400]:
    fs = sorted(glob.glob(os.path.join(DR, d, '*.json')))
    for f in reversed(fs[:1]):
        try:
            r = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        njson += 1
        for k, v in (r.get('group_subtasks') or {}).items():
            gs_union[k] = list(v) if isinstance(v, list) else []
        for k in (r.get('groups') or {}):
            grp_union[k] = grp_union.get(k, 0) + 1
        break
w('\n- 抽样 %d 个 JSON。`groups` 键的并集（出现次数）：\n' % njson)
w('\n| 分组 | 出现次数 | 子任务数 |')
w('|---|---|---|')
for k, c in sorted(grp_union.items(), key=lambda kv: -kv[1]):
    w('| `%s` | %d | %d |' % (k, c, len(gs_union.get(k, []))))

# ---- 3. 全量解析：模型 × 任务 矩阵 ----
w('\n## 3. 全量解析\n')
rows = []
per_task = {}
nbad = 0
missing_group = 0
for d in dirs:
    fs = sorted(glob.glob(os.path.join(DR, d, '*.json')))
    rec = None
    for f in reversed(fs):
        try:
            rec = json.load(open(f, encoding='utf-8'))
            break
        except Exception:
            nbad += 1
    if rec is None:
        continue
    res = rec.get('results', {}) or {}
    rec_row = {'_dir': d, '_model': sanit.get(d, '')}
    for t, v in res.items():
        if not isinstance(v, dict):
            continue
        val = None
        for k in ['acc_norm,none', 'acc,none', 'exact_match,none']:
            if k in v and isinstance(v[k], (int, float)):
                val = float(v[k]); break
        if val is None:
            cands = [float(v[k]) for k in v if isinstance(v[k], (int, float))
                     and 'stderr' not in k]
            val = float(np.mean(cands)) if cands else None
        if val is not None:
            rec_row[t] = val * 100.0
    # 分组聚合：用 groups 里的现成值（若有），否则用 group_subtasks 均值
    gr = rec.get('groups') or {}
    gsub = rec.get('group_subtasks') or {}
    for gname in ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
                  'leaderboard_gpqa', 'leaderboard_musr', 'leaderboard_mmlu_pro',
                  'leaderboard']:
        if gname in gr and isinstance(gr[gname], (int, float)):
            rec_row['G_' + gname] = float(gr[gname]) * 100.0
        elif gname in gsub:
            vs = [rec_row.get(t) for t in gsub[gname] if rec_row.get(t) is not None]
            rec_row['G_' + gname] = float(np.mean(vs)) if vs else np.nan
        else:
            missing_group += 1
    rows.append(rec_row)
    for t in res:
        per_task[t] = per_task.get(t, 0) + 1

M = pd.DataFrame(rows)
w('\n- 成功解析 %d 个目录；跳过损坏 JSON %d 个；任务名总数 %d。\n' % (len(M), nbad, len(per_task)))

task_cols = [c for c in M.columns if not c.startswith('_') and not c.startswith('G_')]
w('\n### 3.1 全部 %d 个任务的覆盖情况\n' % len(task_cols))
w('\n| 任务 | 覆盖模型数 | 缺失率 |')
w('|---|---|---|')
cov = M[task_cols].notna().sum().sort_values(ascending=False)
for t in cov.index:
    w('| `%s` | %d | %.1f%% |' % (t, cov[t], 100 * (1 - cov[t] / len(M))))

M.to_csv(os.path.join(OUT, 'q4_c8_task_matrix.csv'), index=False, encoding='utf-8-sig')
w('\n- 模型 × 任务矩阵已存 `q4_c8_task_matrix.csv`（%d 行 × %d 列）。\n' % M.shape)

# ---- 4. 与 C1 六维汇总值对照 ----
w('\n## 4. 聚合结果与 C1 六维汇总的对照\n')
GNAME = {'IFEval': 'G_leaderboard_ifeval', 'BBH': 'G_leaderboard_bbh',
         'MATH Lvl 5': 'G_leaderboard_math_hard', 'GPQA': 'G_leaderboard_gpqa',
         'MUSR': 'G_leaderboard_musr', 'MMLU-PRO': 'G_leaderboard_mmlu_pro'}
mm = M[M['_model'] != ''].merge(C1, left_on='_model', right_on='Model', how='inner')
w('\n- 可匹配 %d 条（占已解析目录 %.1f%%）。\n' % (len(mm), 100 * len(mm) / len(M)))
w('\n| 维度 | 可比对数 | Pearson $r$ | Spearman $\\rho$ | 平均绝对差 | 中位绝对差 |')
w('|---|---|---|---|---|---|')
from scipy import stats
for a, g in GNAME.items():
    if g not in mm.columns:
        w('| %s | — | — | — | — | — |' % a); continue
    sub = mm[[a, g]].dropna()
    if len(sub) < 30:
        w('| %s | %d | — | — | — | — |' % (a, len(sub))); continue
    w('| %s | %d | %.4f | %.4f | %.3f | %.3f |' % (
        a, len(sub), stats.pearsonr(sub[a], sub[g])[0], stats.spearmanr(sub[a], sub[g])[0],
        float(np.abs(sub[a] - sub[g]).mean()), float(np.abs(sub[a] - sub[g]).median())))
# 总体
if 'G_leaderboard' in mm.columns:
    sub = mm[['Average ⬆️', 'G_leaderboard']].dropna()
    if len(sub) > 30:
        w('| **Average ↔ `leaderboard` 组** | %d | %.4f | %.4f | %.3f | %.3f |' % (
            len(sub), stats.pearsonr(sub['Average ⬆️'], sub['G_leaderboard'])[0],
            stats.spearmanr(sub['Average ⬆️'], sub['G_leaderboard'])[0],
            float(np.abs(sub['Average ⬆️'] - sub['G_leaderboard']).mean()),
            float(np.abs(sub['Average ⬆️'] - sub['G_leaderboard']).median())))

# 逐任务与所属分组的一致性（抽查 BBH 27 子任务）
w('\n### 4.1 逐任务 → 分组聚合的自洽性（以 BBH 27 子任务为例）\n')
bbh_tasks = [t for t in task_cols if t.startswith('leaderboard_bbh_')]
if len(bbh_tasks) and 'G_leaderboard_bbh' in mm.columns:
    sub = mm.dropna(subset=['G_leaderboard_bbh'])
    man = sub[bbh_tasks].mean(axis=1, skipna=True)
    w('\n| 口径 | 可比对数 | Pearson $r$ | 平均绝对差 |')
    w('|---|---|---|---|')
    w('| C8 子任务均值 vs C8 `groups` 值 | %d | %.4f | %.3f |'
      % (len(sub), stats.pearsonr(man, sub['G_leaderboard_bbh'])[0],
         float(np.abs(man - sub['G_leaderboard_bbh']).mean())))
    w('| C8 子任务均值 vs C1 `BBH` | %d | %.4f | %.3f |'
      % (len(sub), stats.pearsonr(man, sub['BBH'])[0], float(np.abs(man - sub['BBH']).mean())))

with open(os.path.join(OUT, '04b_c8_aggregation.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

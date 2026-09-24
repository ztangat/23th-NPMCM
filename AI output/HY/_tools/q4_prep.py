# -*- coding: utf-8 -*-
"""问题四 · 阶段0：数据口径勘察与 C8 逐任务聚合

产出 _out/04_q4/04a_data_audit.md 与若干过程 CSV
"""
import os, io, sys, json, glob, re, warnings
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

w('# 问题四 · 阶段0：数据口径勘察与逐任务聚合\n')

# ==================== C1 / C2 / C3 ====================
w('\n## 1. 排行榜数据（C1 / C2 / C3）的字段画像\n')
C1 = pd.read_csv(os.path.join(CE, 'leaderboard_cleaned.csv'))
C2 = pd.read_csv(os.path.join(CE, 'leaderboard_enhanced.csv'))
C3 = pd.read_csv(os.path.join(CE, 'leaderboard_extended_timeseries.csv'))

BENCH = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
w('\n| 文件 | 行数 | 列数 | 关键列 |')
w('|---|---|---|---|')
for nm, df in [('C1 leaderboard_cleaned', C1), ('C2 leaderboard_enhanced', C2),
               ('C3 leaderboard_extended_timeseries', C3)]:
    w('| %s | %d | %d | %s |' % (nm, len(df), df.shape[1], ', '.join(list(df.columns)[:8]) + ' …'))

w('\n### 1.1 模型类型 `Type`（C1）\n')
w('\n| 类型 | 数量 | 占比 | `Average` 中位数 |')
w('|---|---|---|---|')
for t, sub in C1.groupby('Type'):
    w('| %s | %d | %.1f%% | %.3f |' % (t, len(sub), 100 * len(sub) / len(C1),
                                       sub['Average ⬆️'].median()))

w('\n### 1.2 许可证 `Hub License`（C1，前 20）\n')
w('\n| 许可证 | 数量 | 占比 |')
w('|---|---|---|')
vc = C1['Hub License'].fillna('(缺失)').value_counts()
for k, v in vc.head(20).items():
    w('| %s | %d | %.1f%% |' % (k, v, 100 * v / len(C1)))
w('\n（共 %d 种不同取值）\n' % len(vc))

w('\n### 1.3 时间轴口径：提交日期 vs 发布日期\n')
C1['Submission Date'] = pd.to_datetime(C1['Submission Date'], errors='coerce')
C2['Epoch_AI_Publication_Date'] = pd.to_datetime(C2['Epoch_AI_Publication_Date'], errors='coerce')
w('\n| 口径 | 非空 | 最早 | 最晚 | 中位 |')
w('|---|---|---|---|---|')
for nm, s in [('C1 `Submission Date`', C1['Submission Date']),
              ('C2 `Epoch_AI_Publication_Date`', C2['Epoch_AI_Publication_Date'])]:
    w('| %s | %d | %s | %s | %s |' % (nm, s.notna().sum(), s.min().date(), s.max().date(),
                                      s.median().date()))
m = C1[['Model', 'Submission Date']].merge(
    C2[['Model', 'Epoch_AI_Publication_Date']], on='Model', how='inner')
m = m.dropna()
d = (m['Submission Date'] - m['Epoch_AI_Publication_Date']).dt.days
w('\n两口径可匹配的模型 %d 个；`提交日期 − 发布日期` 的天数差：'
  '均值 %.1f，中位 %.1f，5%%/95%% 分位 %.0f / %.0f，|差|≤90 天占比 %.1f%%。\n'
  % (len(m), d.mean(), d.median(), d.quantile(0.05), d.quantile(0.95),
     100 * (d.abs() <= 90).mean()))
w('\n> **口径选择**：以 `Submission Date`（提交日期）为时间轴主口径。理由：① 全部 %d 行均有值（发布日期仅 %d 行可匹配）；'
  '② 榜单口径内部一致，避免跨库匹配的幸存者偏差；③ 与发布日期中位差仅 %.0f 天，结论不敏感。\n'
  % (C1['Submission Date'].notna().sum(), len(m), d.median()))

w('\n### 1.4 C2 的 `Epoch_AI_Open_Weights` 与 C1 许可证的交叉\n')
w('\n| Epoch 开源权重 | 数量 | 占比 |')
w('|---|---|---|')
for k, v in C2['Epoch_AI_Open_Weights'].fillna('(未匹配)').value_counts().items():
    w('| %s | %d | %.1f%% |' % (k, v, 100 * v / len(C2)))

w('\n### 1.5 C3 时序数据\n')
w('\n| `Source` | 数量 | 年份范围 |')
w('|---|---|---|')
for k, sub in C3.groupby('Source'):
    w('| %s | %d | %d–%d |' % (k, len(sub), sub['Year'].min(), sub['Year'].max()))

# ==================== C4 ====================
w('\n---\n\n## 2. Epoch AI 全模型元数据（C4）\n')
C4 = pd.read_csv(os.path.join(CE, 'epoch_all_ai_models.csv'), low_memory=False)
w('\n- 共 %d 行、%d 列。\n' % (C4.shape[0], C4.shape[1]))
key4 = ['Parameters', 'Training compute (FLOP)', 'Training dataset size (total)',
        'Publication date', 'Open model weights?', 'Model accessibility',
        'Training compute lower bound', 'Training compute upper bound',
        'Organization', 'Domain', 'Country (of organization)', 'Frontier model']
w('\n| 字段 | 非空 | 非空率 | 示例/范围 |')
w('|---|---|---|---|')
for k in key4:
    if k not in C4.columns:
        w('| %s | — | — | 字段不存在 |' % k); continue
    s = C4[k]
    n = s.notna().sum()
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        ex = ('min=%.3g, 中位=%.3g, max=%.3g' % (v.min(), v.median(), v.max())) if len(v) else '—'
    else:
        ex = ', '.join(map(str, s.dropna().astype(str).unique()[:5]))
    w('| `%s` | %d | %.1f%% | %s |' % (k, n, 100 * n / len(C4), ex))

w('\n### 2.1 `Domain`（领域）与 `Open model weights?`\n')
w('\n| 领域 | 数量 |')
w('|---|---|')
for k, v in C4['Domain'].fillna('(缺失)').value_counts().head(10).items():
    w('| %s | %d |' % (k, v))

# 语言模型子集 + 有算力的记录
lang = C4[C4['Domain'].fillna('').str.contains('Language', case=False, na=False)].copy()
lang['Pub'] = pd.to_datetime(lang['Publication date'], errors='coerce')
lang['Cfl'] = pd.to_numeric(lang['Training compute (FLOP)'], errors='coerce')
lang['Np'] = pd.to_numeric(lang['Parameters'], errors='coerce')
w('\n- `Domain` 含 Language 的记录 %d 条；其中\n' % len(lang))
w('  - 有 `Publication date`：%d 条（%s – %s）\n'
  % (lang['Pub'].notna().sum(), lang['Pub'].min().date(), lang['Pub'].max().date()))
w('  - 有 `Training compute (FLOP)`：%d 条\n' % lang['Cfl'].notna().sum())
w('  - 同时有日期+算力+参数：%d 条\n'
  % (lang[['Pub', 'Cfl', 'Np']].notna().all(axis=1).sum()))

# ==================== C6 桥接 ====================
w('\n---\n\n## 3. Loss–Benchmark 桥接（C5 / C6）\n')
C5 = pd.read_csv(os.path.join(CE, 'loss_benchmark_bridge.csv'))
C6 = pd.read_csv(os.path.join(CE, 'loss_benchmark_bridge_expanded.csv'))
w('\n| 文件 | 行数 | 说明 |')
w('|---|---|---|')
w('| C5 `loss_benchmark_bridge.csv` | %d | 基础桥接 |' % len(C5))
w('| C6 `loss_benchmark_bridge_expanded.csv` | %d | 扩展桥接（主用） |' % len(C6))
w('\n### 3.1 可比性等级分布\n')
w('\n| `Loss_Comparability` | C5 | C6 |')
w('|---|---|---|')
for k in sorted(set(C5['Loss_Comparability'].dropna()) | set(C6['Loss_Comparability'].dropna())):
    w('| %s | %d | %d |' % (k, int((C5['Loss_Comparability'] == k).sum()),
                            int((C6['Loss_Comparability'] == k).sum())))
w('\n### 3.2 C6 全量明细\n')
cols = ['Model', 'N_params_B', 'D_tokens_B', 'Val_Loss', 'LB_Average',
        'LB_IFEval', 'LB_BBH', 'LB_MATH', 'LB_GPQA', 'LB_MUSR', 'LB_MMLU_PRO',
        'Loss_Comparability']
w('\n| ' + ' | '.join(cols) + ' |')
w('|' + '---|' * len(cols))
for _, r in C6.iterrows():
    w('| %s | %.4g | %.4g | %.4f | %.4f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |' % (
        r['Model'][:52], r['N_params_B'], r['D_tokens_B'], r['Val_Loss'], r['LB_Average'],
        r['LB_IFEval'], r['LB_BBH'], r['LB_MATH'], r['LB_GPQA'], r['LB_MUSR'],
        r['LB_MMLU_PRO'], str(r['Loss_Comparability'])[:34]))

# ==================== C7 ====================
w('\n---\n\n## 4. 架构元数据（C7，问题四用作上下文/规模对照）\n')
C7 = pd.read_csv(os.path.join(CE, 'model_architecture_metadata.csv'))
w('\n- %d 行，列：%s\n' % (len(C7), ', '.join(map(str, C7.columns))))

# ==================== C8 逐任务聚合 ====================
w('\n---\n\n## 5. C8 逐任务聚合（1,863 目录 / 1,958 JSON）\n')
DR = os.path.join(CE, 'detailed_results')
dirs = sorted(os.listdir(DR))
w('\n- 目录数 %d，JSON 文件总数 %d\n'
  % (len(dirs), sum(len(glob.glob(os.path.join(DR, d, '*.json'))) for d in dirs)))

# 先看一个 JSON 的顶层结构
sample = None
for d in dirs:
    fs = sorted(glob.glob(os.path.join(DR, d, '*.json')))
    if fs:
        try:
            sample = json.load(open(fs[0], encoding='utf-8'))
            break
        except Exception:
            continue
w('\n- JSON 顶层键：%s\n' % ', '.join(list(sample.keys())))
if 'groups' in sample:
    w('- `groups` 键（前 12）：%s\n' % ', '.join(list(sample['groups'].keys())[:12]))
if 'config' in sample:
    w('- `config` 键（前 12）：%s\n' % ', '.join(list(sample['config'].keys())[:12]))
if 'results' in sample:
    w('- `results` 任务数：%d；前 8 个：%s\n'
      % (len(sample['results']), ', '.join(list(sample['results'].keys())[:8])))

# 全量聚合
rows = []
nbad = 0
ndirs_ok = 0
task_set = {}
for di, d in enumerate(dirs):
    fs = sorted(glob.glob(os.path.join(DR, d, '*.json')))
    rec = None
    for f in reversed(fs):          # 取最新可解析记录
        try:
            rec = json.load(open(f, encoding='utf-8'))
            break
        except Exception:
            nbad += 1
    if rec is None:
        continue
    ndirs_ok += 1
    res = rec.get('results', {}) or {}
    per = {}
    for t, v in res.items():
        if not isinstance(v, dict):
            continue
        # 主指标优先级：acc_norm,none > acc,none > exact_match,none
        val = None
        for k in ['acc_norm,none', 'acc,none', 'exact_match,none']:
            if k in v and isinstance(v[k], (int, float)):
                val = float(v[k]); break
        if val is None:
            cands = [float(v[k]) for k in v if isinstance(v[k], (int, float))
                     and not k.endswith('_stderr,none') and 'stderr' not in k]
            val = float(np.mean(cands)) if cands else None
        if val is not None:
            per[t] = val
            task_set[t] = task_set.get(t, 0) + 1
    # 六大类分组（用于与 C1 的六维对照）
    def grp(pref):
        vs = [v for k, v in per.items() if k.startswith(pref)]
        return float(np.mean(vs)) * 100 if vs else np.nan
    rows.append(dict(model_dir=d, n_tasks=len(per),
                     mean_all=float(np.mean(list(per.values()))) * 100 if per else np.nan,
                     ifeval=grp('leaderboard_ifeval'),
                     bbh=grp('leaderboard_bbh'),
                     math=grp('leaderboard_math'),
                     gpqa=grp('leaderboard_gpqa'),
                     musr=grp('leaderboard_musr'),
                     mmlu_pro=grp('leaderboard_mmlu_pro') if any(
                         k.startswith('leaderboard_mmlu_pro') for k in per) else grp('leaderboard_mmlu')))
AGG = pd.DataFrame(rows)
AGG.to_csv(os.path.join(OUT, 'q4_c8_task_aggregate.csv'), index=False, encoding='utf-8-sig')
w('\n- 成功解析目录 %d 个；损坏/截断 JSON %d 个（跳过）；任务名总数 %d\n'
  % (ndirs_ok, nbad, len(task_set)))
w('- 聚合口径：每个目录取**最新一个可解析** JSON；'
  '任务主指标优先级 `acc_norm,none` > `acc,none` > `exact_match,none`；'
  '组内任务取算术平均后 ×100。\n')

w('\n### 5.1 出现频次最高的 25 个任务（按覆盖模型数）\n')
w('\n| 任务 | 覆盖模型数 |')
w('|---|---|')
for t, c in sorted(task_set.items(), key=lambda kv: -kv[1])[:25]:
    w('| `%s` | %d |' % (t, c))

w('\n### 5.2 六大类聚合与 C1 汇总值的对照\n')
# 目录名 -> Model 名（C1 的 Model 含 '/'）
AGG['Model'] = AGG['model_dir'].str.replace('__', '/', regex=False).str.replace('_', '/', regex=False)
mm = C1.merge(AGG, on='Model', how='inner', suffixes=('', '_c8'))
w('\n- 目录名与 C1 `Model` 直接可匹配 %d 个（C1 共 %d）。\n' % (len(mm), len(C1)))
w('\n| 维度 | 可比对数 | Pearson $r$ | Spearman $\\rho$ | 平均绝对差 |')
w('|---|---|---|---|---|')
from scipy import stats
for a, b in [('IFEval', 'ifeval'), ('BBH', 'bbh'), ('MATH Lvl 5', 'math'),
             ('GPQA', 'gpqa'), ('MUSR', 'musr'), ('MMLU-PRO', 'mmlu_pro')]:
    sub = mm[[a, b]].dropna()
    if len(sub) < 10:
        w('| %s | %d | — | — | — |' % (a, len(sub))); continue
    r = stats.pearsonr(sub[a], sub[b])[0]
    rho = stats.spearmanr(sub[a], sub[b])[0]
    w('| %s | %d | %.4f | %.4f | %.3f |' % (a, len(sub), r, rho,
                                            float(np.abs(sub[a] - sub[b]).mean())))

with open(os.path.join(OUT, '04a_data_audit.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(BUF))
print('ok')

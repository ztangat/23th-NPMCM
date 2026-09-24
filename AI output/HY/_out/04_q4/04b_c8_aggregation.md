# 问题四 · C8 逐任务聚合（修正版）


## 1. 目录名 → 模型名的还原


- 目录总数 1863；用 `{Model.replace("/","_")}` 精确查表可还原 **1861** 个（99.9%）。

- 无法还原的 2 个目录（C1 中无对应记录）示例：Dre


## 2. 分组定义（取自 JSON 的 `group_subtasks`）


- 抽样 400 个 JSON。`groups` 键的并集（出现次数）：


| 分组 | 出现次数 | 子任务数 |
|---|---|---|
| `leaderboard` | 400 | 6 |
| `leaderboard_bbh` | 400 | 24 |
| `leaderboard_gpqa` | 400 | 3 |
| `leaderboard_math_hard` | 400 | 7 |
| `leaderboard_musr` | 400 | 3 |

## 3. 全量解析


- 成功解析 1860 个目录；跳过损坏 JSON 4 个；任务名总数 45。


### 3.1 全部 45 个任务的覆盖情况


| 任务 | 覆盖模型数 | 缺失率 |
|---|---|---|
| `leaderboard_bbh_boolean_expressions` | 1860 | 0.0% |
| `leaderboard_bbh_date_understanding` | 1860 | 0.0% |
| `leaderboard_bbh_causal_judgement` | 1860 | 0.0% |
| `leaderboard_bbh_formal_fallacies` | 1860 | 0.0% |
| `leaderboard_bbh_disambiguation_qa` | 1860 | 0.0% |
| `leaderboard_bbh_logical_deduction_five_objects` | 1860 | 0.0% |
| `leaderboard_bbh_logical_deduction_seven_objects` | 1860 | 0.0% |
| `leaderboard_bbh_geometric_shapes` | 1860 | 0.0% |
| `leaderboard_bbh_hyperbaton` | 1860 | 0.0% |
| `leaderboard_bbh_logical_deduction_three_objects` | 1860 | 0.0% |
| `leaderboard_bbh_movie_recommendation` | 1860 | 0.0% |
| `leaderboard_bbh_object_counting` | 1860 | 0.0% |
| `leaderboard_bbh_navigate` | 1860 | 0.0% |
| `leaderboard_bbh_temporal_sequences` | 1860 | 0.0% |
| `leaderboard_bbh_tracking_shuffled_objects_five_objects` | 1860 | 0.0% |
| `leaderboard_bbh_penguins_in_a_table` | 1860 | 0.0% |
| `leaderboard_bbh_reasoning_about_colored_objects` | 1860 | 0.0% |
| `leaderboard_bbh_ruin_names` | 1860 | 0.0% |
| `leaderboard_bbh_salient_translation_error_detection` | 1860 | 0.0% |
| `leaderboard_bbh_snarks` | 1860 | 0.0% |
| `leaderboard_bbh_sports_understanding` | 1860 | 0.0% |
| `leaderboard_bbh_web_of_lies` | 1860 | 0.0% |
| `leaderboard_bbh_tracking_shuffled_objects_three_objects` | 1860 | 0.0% |
| `leaderboard_bbh_tracking_shuffled_objects_seven_objects` | 1860 | 0.0% |
| `leaderboard_bbh` | 1858 | 0.1% |
| `leaderboard_math_intermediate_algebra_hard` | 1857 | 0.2% |
| `leaderboard_math_geometry_hard` | 1857 | 0.2% |
| `leaderboard_math_hard` | 1857 | 0.2% |
| `leaderboard_math_algebra_hard` | 1857 | 0.2% |
| `leaderboard_math_prealgebra_hard` | 1857 | 0.2% |
| `leaderboard_math_num_theory_hard` | 1857 | 0.2% |
| `leaderboard_math_precalculus_hard` | 1857 | 0.2% |
| `leaderboard_math_counting_and_prob_hard` | 1857 | 0.2% |
| `leaderboard_musr_team_allocation` | 1856 | 0.2% |
| `leaderboard_ifeval` | 1856 | 0.2% |
| `leaderboard_gpqa_diamond` | 1856 | 0.2% |
| `leaderboard_gpqa_extended` | 1856 | 0.2% |
| `leaderboard_musr_object_placements` | 1856 | 0.2% |
| `leaderboard_gpqa_main` | 1856 | 0.2% |
| `leaderboard_mmlu_pro` | 1856 | 0.2% |
| `leaderboard_musr_murder_mysteries` | 1856 | 0.2% |
| `leaderboard` | 1854 | 0.3% |
| `leaderboard_gpqa` | 1854 | 0.3% |
| `leaderboard_musr` | 1854 | 0.3% |
| `leaderboard_arc_challenge` | 85 | 95.4% |

- 模型 × 任务矩阵已存 `q4_c8_task_matrix.csv`（1860 行 × 52 列）。


## 4. 聚合结果与 C1 六维汇总的对照


- 可匹配 1897 条（占已解析目录 102.0%）。


| 维度 | 可比对数 | Pearson $r$ | Spearman $\rho$ | 平均绝对差 | 中位绝对差 |
|---|---|---|---|---|---|
| IFEval | — | — | — | — | — |
| BBH | 1897 | 0.9979 | 0.9971 | 21.282 | 21.180 |
| MATH Lvl 5 | 1893 | 0.8043 | 0.8107 | 4.431 | 1.923 |
| GPQA | 1893 | 0.9806 | 0.9809 | 23.298 | 23.498 |
| MUSR | 1893 | 0.9732 | 0.9769 | 30.814 | 30.917 |
| MMLU-PRO | — | — | — | — | — |
| **Average ↔ `leaderboard` 组** | 1893 | 0.9839 | 0.9871 | 13.621 | 14.020 |

### 4.1 逐任务 → 分组聚合的自洽性（以 BBH 27 子任务为例）


| 口径 | 可比对数 | Pearson $r$ | 平均绝对差 |
|---|---|---|---|
| C8 子任务均值 vs C8 `groups` 值 | 1897 | 1.0000 | 0.000 |
| C8 子任务均值 vs C1 `BBH` | 1897 | 0.9979 | 21.282 |
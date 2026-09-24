# 诊断：C8 聚合与 C1 汇总的系统性偏移


- C1 共 4576 行 / 4497 个唯一 `Model`（同一模型多次提交，需去重）。


### 重复提交示例（同一 `Model` 多行，按 Type/日期区分）


| Model | #Params | 提交日期 | Type | Average |
|---|---|---|---|---|
| AtAndDev/Qwen2.5-1.5B-continuous-learnt | 1.544 | 2024-10-13 | 🔶 fine-tuned on domain-specific datasets | 16.5185 |
| AtAndDev/Qwen2.5-1.5B-continuous-learnt | 1.544 | 2024-10-18 | 💬 chat models (RLHF, DPO, IFT, ...) | 17.4836 |
| BoltMonkey/NeuralDaredevil-SuperNova-Lite-7B-DARETIES-abliterated | 8.03 | 2024-10-10 | 🤝 base merges and moerges | 27.7766 |
| BoltMonkey/NeuralDaredevil-SuperNova-Lite-7B-DARETIES-abliterated | 8.03 | 2024-10-01 | 🔶 fine-tuned on domain-specific datasets | 21.3455 |
| Columbia-NLP/LION-Gemma-2b-dpo-v1.0 | 2.506 | 2024-07-04 | 💬 chat models (RLHF, DPO, IFT, ...) | 11.4840 |
| Columbia-NLP/LION-Gemma-2b-dpo-v1.0 | 2.506 | 2024-07-04 | 💬 chat models (RLHF, DPO, IFT, ...) | 11.2621 |
| Daemontatox/AetherTOT | 10.67 | 2024-12-28 | 🔶 fine-tuned on domain-specific datasets | 23.1788 |
| Daemontatox/AetherTOT | 10.67 | 2024-12-28 | 🌸 multimodal | 22.8747 |
| Daemontatox/DocumentCogito | 10.67 | 2025-01-16 | 🌸 multimodal | 24.2204 |
| Daemontatox/DocumentCogito | 10.67 | 2025-03-09 | 🌸 multimodal | 29.1082 |

## 1. 逐个模型的对照（BBH）


| 模型 | 目录存在 | C1 `BBH` | `results[leaderboard_bbh].acc_norm` | `groups[leaderboard_bbh]` | 子任务均值 | 子任务数 | `n-shot` | JSON 数 |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 是 | 34.8921 | 53.6886 | 53.6886 | 53.9633 | 24 | {'leaderboard': 0, 'leaderboard_bbh': 3, 'leaderboard_bbh_boolean_expressions': 3, 'leaderboard_bbh_causal_judgement': 3, 'leaderboard_bbh_date_understanding': 3, 'leaderboard_bbh_disambiguation_qa': 3, 'leaderboard_bbh_formal_fallacies': 3, 'leaderboard_bbh_geometric_shapes': 3, 'leaderboard_bbh_hyperbaton': 3, 'leaderboard_bbh_logical_deduction_five_objects': 3, 'leaderboard_bbh_logical_deduction_seven_objects': 3, 'leaderboard_bbh_logical_deduction_three_objects': 3, 'leaderboard_bbh_movie_recommendation': 3, 'leaderboard_bbh_navigate': 3, 'leaderboard_bbh_object_counting': 3, 'leaderboard_bbh_penguins_in_a_table': 3, 'leaderboard_bbh_reasoning_about_colored_objects': 3, 'leaderboard_bbh_ruin_names': 3, 'leaderboard_bbh_salient_translation_error_detection': 3, 'leaderboard_bbh_snarks': 3, 'leaderboard_bbh_sports_understanding': 3, 'leaderboard_bbh_temporal_sequences': 3, 'leaderboard_bbh_tracking_shuffled_objects_five_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_seven_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_three_objects': 3, 'leaderboard_bbh_web_of_lies': 3, 'leaderboard_gpqa': 0, 'leaderboard_gpqa_diamond': 0, 'leaderboard_gpqa_extended': 0, 'leaderboard_gpqa_main': 0, 'leaderboard_ifeval': 0, 'leaderboard_math_algebra_hard': 4, 'leaderboard_math_counting_and_prob_hard': 4, 'leaderboard_math_geometry_hard': 4, 'leaderboard_math_hard': 4, 'leaderboard_math_intermediate_algebra_hard': 4, 'leaderboard_math_num_theory_hard': 4, 'leaderboard_math_prealgebra_hard': 4, 'leaderboard_math_precalculus_hard': 4, 'leaderboard_mmlu_pro': 5, 'leaderboard_musr': 0, 'leaderboard_musr_murder_mysteries': 0, 'leaderboard_musr_object_placements': 0, 'leaderboard_musr_team_allocation': 0} | 1 |
| meta-llama/Meta-Llama-3-8B-Instruct | 是 | 28.2449 | 49.7136 | 49.7136 | 49.8871 | 24 | {'leaderboard': 5, 'leaderboard_arc_challenge': 5, 'leaderboard_bbh': 3, 'leaderboard_bbh_boolean_expressions': 3, 'leaderboard_bbh_causal_judgement': 3, 'leaderboard_bbh_date_understanding': 3, 'leaderboard_bbh_disambiguation_qa': 3, 'leaderboard_bbh_formal_fallacies': 3, 'leaderboard_bbh_geometric_shapes': 3, 'leaderboard_bbh_hyperbaton': 3, 'leaderboard_bbh_logical_deduction_five_objects': 3, 'leaderboard_bbh_logical_deduction_seven_objects': 3, 'leaderboard_bbh_logical_deduction_three_objects': 3, 'leaderboard_bbh_movie_recommendation': 3, 'leaderboard_bbh_navigate': 3, 'leaderboard_bbh_object_counting': 3, 'leaderboard_bbh_penguins_in_a_table': 3, 'leaderboard_bbh_reasoning_about_colored_objects': 3, 'leaderboard_bbh_ruin_names': 3, 'leaderboard_bbh_salient_translation_error_detection': 3, 'leaderboard_bbh_snarks': 3, 'leaderboard_bbh_sports_understanding': 3, 'leaderboard_bbh_temporal_sequences': 3, 'leaderboard_bbh_tracking_shuffled_objects_five_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_seven_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_three_objects': 3, 'leaderboard_bbh_web_of_lies': 3, 'leaderboard_gpqa': 0, 'leaderboard_gpqa_diamond': 0, 'leaderboard_gpqa_extended': 0, 'leaderboard_gpqa_main': 0, 'leaderboard_ifeval': 0, 'leaderboard_math_algebra_hard': 4, 'leaderboard_math_counting_and_prob_hard': 4, 'leaderboard_math_geometry_hard': 4, 'leaderboard_math_hard': 4, 'leaderboard_math_intermediate_algebra_hard': 4, 'leaderboard_math_num_theory_hard': 4, 'leaderboard_math_prealgebra_hard': 4, 'leaderboard_math_precalculus_hard': 4, 'leaderboard_mmlu_pro': 5, 'leaderboard_musr': 0, 'leaderboard_musr_murder_mysteries': 0, 'leaderboard_musr_object_placements': 0, 'leaderboard_musr_team_allocation': 0} | 1 |
| mistralai/Mistral-7B-Instruct-v0.2 | 否 | — | — | — | — | — | — | — |
| google/gemma-2-9b-it | 否 | — | — | — | — | — | — | — |
| 0-hero/Matter-0.2-7B-DPO | 是 | 10.0555 | 35.8445 | 35.8445 | 35.9625 | 24 | {'leaderboard': 0, 'leaderboard_bbh': 3, 'leaderboard_bbh_boolean_expressions': 3, 'leaderboard_bbh_causal_judgement': 3, 'leaderboard_bbh_date_understanding': 3, 'leaderboard_bbh_disambiguation_qa': 3, 'leaderboard_bbh_formal_fallacies': 3, 'leaderboard_bbh_geometric_shapes': 3, 'leaderboard_bbh_hyperbaton': 3, 'leaderboard_bbh_logical_deduction_five_objects': 3, 'leaderboard_bbh_logical_deduction_seven_objects': 3, 'leaderboard_bbh_logical_deduction_three_objects': 3, 'leaderboard_bbh_movie_recommendation': 3, 'leaderboard_bbh_navigate': 3, 'leaderboard_bbh_object_counting': 3, 'leaderboard_bbh_penguins_in_a_table': 3, 'leaderboard_bbh_reasoning_about_colored_objects': 3, 'leaderboard_bbh_ruin_names': 3, 'leaderboard_bbh_salient_translation_error_detection': 3, 'leaderboard_bbh_snarks': 3, 'leaderboard_bbh_sports_understanding': 3, 'leaderboard_bbh_temporal_sequences': 3, 'leaderboard_bbh_tracking_shuffled_objects_five_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_seven_objects': 3, 'leaderboard_bbh_tracking_shuffled_objects_three_objects': 3, 'leaderboard_bbh_web_of_lies': 3, 'leaderboard_gpqa': 0, 'leaderboard_gpqa_diamond': 0, 'leaderboard_gpqa_extended': 0, 'leaderboard_gpqa_main': 0, 'leaderboard_ifeval': 0, 'leaderboard_math_algebra_hard': 4, 'leaderboard_math_counting_and_prob_hard': 4, 'leaderboard_math_geometry_hard': 4, 'leaderboard_math_hard': 4, 'leaderboard_math_intermediate_algebra_hard': 4, 'leaderboard_math_num_theory_hard': 4, 'leaderboard_math_prealgebra_hard': 4, 'leaderboard_math_precalculus_hard': 4, 'leaderboard_mmlu_pro': 5, 'leaderboard_musr': 0, 'leaderboard_musr_murder_mysteries': 0, 'leaderboard_musr_object_placements': 0, 'leaderboard_musr_team_allocation': 0} | 1 |
| Qwen/Qwen2.5-72B-Instruct | 否 | — | — | — | — | — | — | — |
| microsoft/Phi-3-mini-4k-instruct | 否 | — | — | — | — | — | — | — |

## 2. 六维逐一对照（同上批模型）


| 模型 | 维度 | C1 | `results.acc_norm` | `groups` |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | IFEval | 75.8525 | — | — |
| Qwen/Qwen2.5-7B-Instruct | BBH | 34.8921 | 53.6886 | 53.6886 |
| Qwen/Qwen2.5-7B-Instruct | MATH Lvl 5 | 50.0000 | — | — |
| Qwen/Qwen2.5-7B-Instruct | GPQA | 5.4810 | 30.0336 | 30.0336 |
| Qwen/Qwen2.5-7B-Instruct | MUSR | 8.4539 | 40.2116 | 40.2116 |
| Qwen/Qwen2.5-7B-Instruct | MMLU-PRO | 36.5211 | — | — |
| meta-llama/Meta-Llama-3-8B-Instruct | IFEval | 74.0840 | — | — |
| meta-llama/Meta-Llama-3-8B-Instruct | BBH | 28.2449 | 49.7136 | 49.7136 |
| meta-llama/Meta-Llama-3-8B-Instruct | MATH Lvl 5 | 8.6858 | — | — |
| meta-llama/Meta-Llama-3-8B-Instruct | GPQA | 1.2304 | 25.9228 | 25.9228 |
| meta-llama/Meta-Llama-3-8B-Instruct | MUSR | 1.6029 | 35.5820 | 35.5820 |
| meta-llama/Meta-Llama-3-8B-Instruct | MMLU-PRO | 29.6044 | — | — |
| 0-hero/Matter-0.2-7B-DPO | IFEval | 33.0279 | — | — |
| 0-hero/Matter-0.2-7B-DPO | BBH | 10.0555 | 35.8445 | 35.8445 |
| 0-hero/Matter-0.2-7B-DPO | MATH Lvl 5 | 1.4350 | — | — |
| 0-hero/Matter-0.2-7B-DPO | GPQA | 1.2304 | 25.9228 | 25.9228 |
| 0-hero/Matter-0.2-7B-DPO | MUSR | 5.8719 | 38.0952 | 38.0952 |
| 0-hero/Matter-0.2-7B-DPO | MMLU-PRO | 1.8174 | — | — |

## 3. 全样本：C1 值（纵轴）对 C8 聚合值（横轴）的回归


- 可比对 1858 个模型（C1 去重后 4497 行）。


| 维度 | $n$ | 斜率 $b$ | 截距 $a$ | $R^2$ | C1 均值 | C8 均值 | 均值差 |
|---|---|---|---|---|---|---|---|
| IFEval | 0 | — | — | — | — | — | — |
| BBH | 1856 | 1.3233 | -36.4210 | 0.9967 | 26.422 | 47.490 | -21.068 |
| MATH Lvl 5 | 0 | — | — | — | — | — | — |
| GPQA | 1852 | 1.2948 | -32.1560 | 0.9796 | 6.436 | 29.806 | -23.369 |
| MUSR | 1852 | 1.2399 | -40.3096 | 0.9505 | 9.229 | 39.955 | -30.726 |
| MMLU-PRO | 0 | — | — | — | — | — | — |

## 4. BBH 子任务清单与"24 项 vs 27 项"检验


- 该模型 JSON 中 `leaderboard_bbh_*` 子任务共 **24** 项（官方 BBH 为 27 项）。


| # | 子任务 | acc_norm |
|---|---|---|
| 1 | `leaderboard_bbh_boolean_expressions` | 0.8440 |
| 2 | `leaderboard_bbh_causal_judgement` | 0.5348 |
| 3 | `leaderboard_bbh_date_understanding` | 0.5680 |
| 4 | `leaderboard_bbh_disambiguation_qa` | 0.6160 |
| 5 | `leaderboard_bbh_formal_fallacies` | 0.6000 |
| 6 | `leaderboard_bbh_geometric_shapes` | 0.5160 |
| 7 | `leaderboard_bbh_hyperbaton` | 0.5200 |
| 8 | `leaderboard_bbh_logical_deduction_five_objects` | 0.4920 |
| 9 | `leaderboard_bbh_logical_deduction_seven_objects` | 0.4680 |
| 10 | `leaderboard_bbh_logical_deduction_three_objects` | 0.7640 |
| 11 | `leaderboard_bbh_movie_recommendation` | 0.5600 |
| 12 | `leaderboard_bbh_navigate` | 0.6840 |
| 13 | `leaderboard_bbh_object_counting` | 0.3400 |
| 14 | `leaderboard_bbh_penguins_in_a_table` | 0.5822 |
| 15 | `leaderboard_bbh_reasoning_about_colored_objects` | 0.6480 |
| 16 | `leaderboard_bbh_ruin_names` | 0.5760 |
| 17 | `leaderboard_bbh_salient_translation_error_detection` | 0.5560 |
| 18 | `leaderboard_bbh_snarks` | 0.7022 |
| 19 | `leaderboard_bbh_sports_understanding` | 0.7320 |
| 20 | `leaderboard_bbh_temporal_sequences` | 0.5480 |
| 21 | `leaderboard_bbh_tracking_shuffled_objects_five_objects` | 0.1640 |
| 22 | `leaderboard_bbh_tracking_shuffled_objects_seven_objects` | 0.1360 |
| 23 | `leaderboard_bbh_tracking_shuffled_objects_three_objects` | 0.2360 |
| 24 | `leaderboard_bbh_web_of_lies` | 0.5640 |

- 子任务均值 = 0.5396；`groups[leaderboard_bbh]` = 0.5369；`results[leaderboard_bbh]` = 0.5369；C1 `BBH` = 34.8921


- `group_subtasks[leaderboard_bbh]` 声明的子任务数 = 24

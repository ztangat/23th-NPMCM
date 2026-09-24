# 问题四 · 阶段0：数据口径勘察与逐任务聚合


## 1. 排行榜数据（C1 / C2 / C3）的字段画像


| 文件 | 行数 | 列数 | 关键列 |
|---|---|---|---|
| C1 leaderboard_cleaned | 4576 | 12 | Model, #Params (B), Submission Date, Hub License, Type, Average ⬆️, IFEval, BBH … |
| C2 leaderboard_enhanced | 4576 | 15 | Model, #Params (B), Submission Date, Hub License, Type, Average ⬆️, IFEval, BBH … |
| C3 leaderboard_extended_timeseries | 4599 | 11 | Model, Year, Params_B, Average, IFEval, BBH, MATH_Lvl5, GPQA … |

### 1.1 模型类型 `Type`（C1）


| 类型 | 数量 | 占比 | `Average` 中位数 |
|---|---|---|---|
| ❓ other | 7 | 0.2% | 17.923 |
| 🌸 multimodal | 9 | 0.2% | 26.493 |
| 💬 chat models (RLHF, DPO, IFT, ...) | 718 | 15.7% | 23.152 |
| 🔶 fine-tuned on domain-specific datasets | 1785 | 39.0% | 19.037 |
| 🟢 pretrained | 275 | 6.0% | 7.322 |
| 🟩 continuously pretrained | 58 | 1.3% | 11.320 |
| 🤝 base merges and moerges | 1724 | 37.7% | 24.398 |

### 1.2 许可证 `Hub License`（C1，前 20）


| 许可证 | 数量 | 占比 |
|---|---|---|
| (缺失) | 1753 | 38.3% |
| apache-2.0 | 1482 | 32.4% |
| other | 296 | 6.5% |
| mit | 284 | 6.2% |
| llama3.1 | 178 | 3.9% |
| cc-by-nc-4.0 | 125 | 2.7% |
| gemma | 123 | 2.7% |
| llama3.2 | 121 | 2.6% |
| llama3 | 118 | 2.6% |
| llama2 | 25 | 0.5% |
| creativeml-openrail-m | 15 | 0.3% |
| llama3.3 | 10 | 0.2% |
| gpl-3.0 | 9 | 0.2% |
| cc-by-nc-sa-4.0 | 9 | 0.2% |
| bigscience-bloom-rail-1.0 | 5 | 0.1% |
| unknown | 4 | 0.1% |
| cc-by-4.0 | 3 | 0.1% |
| cc-by-sa-4.0 | 3 | 0.1% |
| bigcode-openrail-m | 3 | 0.1% |
| wtfpl | 2 | 0.0% |

（共 27 种不同取值）


### 1.3 时间轴口径：提交日期 vs 发布日期


| 口径 | 非空 | 最早 | 最晚 | 中位 |
|---|---|---|---|---|
| C1 `Submission Date` | 4564 | 2024-06-08 | 2025-03-13 | 2024-12-16 |
| C2 `Epoch_AI_Publication_Date` | 447 | 2020-10-20 | 2025-09-09 | 2024-09-19 |

两口径可匹配的模型 460 个；`提交日期 − 发布日期` 的天数差：均值 188.8，中位 103.0，5%/95% 分位 0 / 709，|差|≤90 天占比 44.3%。


> **口径选择**：以 `Submission Date`（提交日期）为时间轴主口径。理由：① 全部 4564 行均有值（发布日期仅 460 行可匹配）；② 榜单口径内部一致，避免跨库匹配的幸存者偏差；③ 与发布日期中位差仅 103 天，结论不敏感。


### 1.4 C2 的 `Epoch_AI_Open_Weights` 与 C1 许可证的交叉


| Epoch 开源权重 | 数量 | 占比 |
|---|---|---|
| (未匹配) | 4133 | 90.3% |
| Yes | 424 | 9.3% |
| No | 19 | 0.4% |

### 1.5 C3 时序数据


| `Source` | 数量 | 年份范围 |
|---|---|---|
| Historical (papers/reports) | 26 | 2019–2024 |
| Open LLM Leaderboard | 4573 | 2024–2025 |

---

## 2. Epoch AI 全模型元数据（C4）


- 共 3523 行、57 列。


| 字段 | 非空 | 非空率 | 示例/范围 |
|---|---|---|---|
| `Parameters` | 2297 | 65.2% | min=10, 中位=2.6e+09, max=1.74e+14 |
| `Training compute (FLOP)` | 1390 | 39.5% | min=40, 中位=3.68e+21, max=5e+26 |
| `Training dataset size (total)` | 1424 | 40.4% | 19700000000000, 11000000000000, 5500000000000, 13600000000000, 5700000000000 |
| `Publication date` | 3500 | 99.3% | 2026-04-24, 2026-04-23, 2026-04-21, 2026-04-20, 2026-04-17 |
| `Open model weights?` | 2653 | 75.3% | Yes, No |
| `Model accessibility` | 2653 | 75.3% | Open weights (unrestricted), API access, Hosted access (no API), Open weights (restricted use), Unreleased |
| `Training compute lower bound` | 26 | 0.7% | min=5.42, 中位=1.48e+23, max=2.1e+26 |
| `Training compute upper bound` | 41 | 1.2% | min=1.35e+18, 中位=8.01e+24, max=1.13e+27 |
| `Organization` | 3435 | 97.5% | DeepSeek, OpenAI, xAI, Moonshot, Anthropic |
| `Domain` | 3431 | 97.4% | Language, Multimodal,Language,Vision, Audio, Image generation, Robotics,Vision |
| `Country (of organization)` | 3428 | 97.3% | China, United States of America, United States of America,United States of America, Korea (Republic of), France |
| `Frontier model` | 137 | 3.9% | True |

### 2.1 `Domain`（领域）与 `Open model weights?`


| 领域 | 数量 |
|---|---|
| Language | 1560 |
| Biology | 376 |
| Vision | 322 |
| Image generation | 164 |
| Speech | 129 |
| (缺失) | 92 |
| Multimodal,Language,Vision | 76 |
| Video | 61 |
| Games | 58 |
| Language,Vision,Multimodal | 46 |

- `Domain` 含 Language 的记录 1961 条；其中

  - 有 `Publication date`：1958 条（1959-02-01 – 2026-04-24）

  - 有 `Training compute (FLOP)`：864 条

  - 同时有日期+算力+参数：825 条


---

## 3. Loss–Benchmark 桥接（C5 / C6）


| 文件 | 行数 | 说明 |
|---|---|---|
| C5 `loss_benchmark_bridge.csv` | 43 | 基础桥接 |
| C6 `loss_benchmark_bridge_expanded.csv` | 75 | 扩展桥接（主用） |

### 3.1 可比性等级分布


| `Loss_Comparability` | C5 | C6 |
|---|---|---|
| High (same model, same validation set) | 7 | 7 |
| Medium (different validation set, approximate) | 36 | 68 |

### 3.2 C6 全量明细


| Model | N_params_B | D_tokens_B | Val_Loss | LB_Average | LB_IFEval | LB_BBH | LB_MATH | LB_GPQA | LB_MUSR | LB_MMLU_PRO | Loss_Comparability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-160m | 0.1624 | 299.9 | 2.5978 | 5.7304 | 18.16 | 2.20 | 0.91 | 1.12 | 10.68 | 1.33 | High (same model, same validation  |
| EleutherAI/pythia-410m | 0.409 | 299.9 | 2.4209 | 5.2271 | 21.95 | 2.72 | 0.98 | 1.23 | 3.06 | 1.42 | High (same model, same validation  |
| EleutherAI/pythia-1b | 1.041 | 299.9 | 2.2904 | 5.0703 | 22.08 | 2.29 | 0.91 | 0.89 | 2.73 | 1.51 | High (same model, same validation  |
| EleutherAI/pythia-1.4b | 1.416 | 299.9 | 2.2558 | 6.0085 | 23.71 | 3.88 | 1.51 | 1.57 | 4.02 | 1.36 | High (same model, same validation  |
| EleutherAI/pythia-2.8b | 2.783 | 299.9 | 2.1911 | 5.5549 | 21.73 | 5.08 | 1.36 | 0.00 | 3.64 | 1.52 | High (same model, same validation  |
| EleutherAI/pythia-6.9b | 6.861 | 299.9 | 2.1250 | 5.9665 | 22.81 | 5.88 | 1.44 | 0.22 | 3.81 | 1.63 | High (same model, same validation  |
| EleutherAI/pythia-12b | 11.97 | 299.9 | 2.0933 | 6.0598 | 24.71 | 4.99 | 1.66 | 0.00 | 3.79 | 1.21 | High (same model, same validation  |
| meta-llama/Meta-Llama-3-8B | 8.03 | nan | 1.7600 | 13.6269 | 14.55 | 24.50 | 4.53 | 7.38 | 6.24 | 24.55 | Medium (different validation set,  |
| meta-llama/Llama-2-7b-hf | 6.738 | nan | 1.7500 | 8.8064 | 25.19 | 10.35 | 1.74 | 2.24 | 3.76 | 9.57 | Medium (different validation set,  |
| meta-llama/Llama-3.1-8B | 8.03 | nan | 1.7600 | 14.4209 | 12.46 | 25.30 | 6.57 | 8.05 | 8.72 | 25.42 | Medium (different validation set,  |
| Qwen/Qwen2-0.5B | 0.494 | nan | 2.8400 | 7.2241 | 18.73 | 7.92 | 2.64 | 1.45 | 4.60 | 8.00 | Medium (different validation set,  |
| Qwen/Qwen2-1.5B | 1.544 | nan | 2.6300 | 10.4455 | 21.13 | 11.78 | 7.02 | 1.90 | 3.59 | 17.24 | Medium (different validation set,  |
| Qwen/Qwen2-7B | 7.616 | nan | 2.2000 | 23.9252 | 31.49 | 34.71 | 20.39 | 7.27 | 14.32 | 35.37 | Medium (different validation set,  |
| Qwen/Qwen2.5-0.5B | 0.5 | nan | 2.8100 | 6.5501 | 16.27 | 6.95 | 3.93 | 0.00 | 2.08 | 10.06 | Medium (different validation set,  |
| Qwen/Qwen2.5-1.5B | 1.5 | nan | 2.5800 | 13.8527 | 26.74 | 16.66 | 9.14 | 4.70 | 5.27 | 20.61 | Medium (different validation set,  |
| Qwen/Qwen2.5-7B | 7.616 | nan | 2.1500 | 26.0192 | 33.74 | 35.81 | 25.08 | 9.96 | 14.14 | 37.39 | Medium (different validation set,  |
| Qwen/Qwen2.5-14B | 14.77 | nan | 2.0200 | 31.9511 | 36.94 | 45.08 | 29.00 | 17.56 | 15.91 | 47.21 | Medium (different validation set,  |
| google/gemma-2-2b | 2.614 | nan | 2.3900 | 10.3596 | 20.18 | 12.50 | 3.02 | 1.68 | 11.27 | 13.52 | Medium (different validation set,  |
| google/gemma-2-9b | 9 | nan | 2.1000 | 21.2053 | 20.40 | 34.10 | 13.44 | 10.51 | 14.30 | 34.48 | Medium (different validation set,  |
| google/gemma-2-27b | 27.23 | nan | 1.9500 | 23.9262 | 24.75 | 37.39 | 16.62 | 13.42 | 13.92 | 37.45 | Medium (different validation set,  |
| google/gemma-7b | 8.538 | nan | 2.2400 | 15.4428 | 26.59 | 21.12 | 7.40 | 4.92 | 10.98 | 21.64 | Medium (different validation set,  |
| microsoft/phi-2 | 2.78 | nan | 2.3900 | 15.5343 | 27.39 | 28.04 | 2.95 | 2.91 | 13.84 | 18.09 | Medium (different validation set,  |
| microsoft/phi-1_5 | 1.418 | nan | 2.3200 | 7.1710 | 20.33 | 7.47 | 1.81 | 2.35 | 3.39 | 7.68 | Medium (different validation set,  |
| microsoft/Phi-3-mini-4k-instruct | 3.821 | nan | 2.1500 | 27.5622 | 54.77 | 36.56 | 16.39 | 10.96 | 13.12 | 33.58 | Medium (different validation set,  |
| meta-llama/Meta-Llama-3-70B | 70.55 | nan | 1.7100 | 26.7054 | 16.03 | 48.71 | 18.58 | 19.69 | 16.01 | 41.21 | Medium (different validation set,  |
| meta-llama/Llama-2-13b-hf | 13.02 | nan | 1.7000 | 11.0652 | 24.82 | 17.22 | 1.51 | 4.14 | 3.39 | 15.31 | Medium (different validation set,  |
| meta-llama/Llama-2-70b-hf | 68.98 | nan | 1.6500 | 18.3726 | 24.07 | 35.90 | 3.25 | 7.05 | 9.78 | 30.20 | Medium (different validation set,  |
| mistralai/Mistral-7B-v0.1 | 7.242 | nan | 2.1000 | 14.5754 | 23.86 | 22.02 | 2.95 | 5.59 | 10.68 | 22.36 | Medium (different validation set,  |
| mistralai/Mistral-7B-Instruct-v0.2 | 7.242 | nan | 2.1000 | 18.5079 | 54.96 | 22.91 | 3.02 | 3.47 | 7.61 | 19.08 | Medium (different validation set,  |
| mistralai/Mixtral-8x7B-v0.1 | 46.7 | nan | 1.9000 | 19.6651 | 23.26 | 30.40 | 9.37 | 9.40 | 13.66 | 31.90 | Medium (different validation set,  |
| Qwen/Qwen2.5-32B | 32.76 | nan | 1.9500 | 38.0080 | 40.77 | 53.95 | 35.65 | 21.59 | 22.70 | 53.39 | Medium (different validation set,  |
| Qwen/Qwen2.5-72B | 72.71 | nan | 1.8500 | 38.4411 | 41.37 | 54.62 | 39.12 | 20.69 | 19.64 | 55.20 | Medium (different validation set,  |
| Qwen/Qwen2-57B-A14B | 57.41 | nan | 1.9000 | 25.0339 | 31.13 | 38.88 | 18.66 | 7.49 | 10.54 | 43.51 | Medium (different validation set,  |
| 01-ai/Yi-6B | 6.061 | nan | 2.1000 | 13.6116 | 28.93 | 19.41 | 1.59 | 2.57 | 7.04 | 22.12 | Medium (different validation set,  |
| 01-ai/Yi-34B | 34.39 | nan | 1.9000 | 22.3731 | 30.46 | 35.54 | 5.14 | 15.55 | 9.65 | 37.91 | Medium (different validation set,  |
| tiiuae/falcon-7b | 7 | nan | 2.2500 | 5.1734 | 18.21 | 5.96 | 0.98 | 0.00 | 4.50 | 1.39 | Medium (different validation set,  |
| tiiuae/falcon-40b | 40 | nan | 2.0500 | 11.4013 | 24.96 | 16.58 | 1.81 | 3.13 | 5.19 | 16.72 | Medium (different validation set,  |
| google/gemma-1.1-2b-it | 2.506 | nan | 2.5000 | 8.0534 | 30.67 | 5.86 | 1.81 | 2.57 | 2.02 | 5.37 | Medium (different validation set,  |
| google/gemma-1.1-7b-it | 8.538 | nan | 2.2000 | 17.6936 | 50.39 | 15.93 | 4.91 | 5.82 | 11.51 | 17.60 | Medium (different validation set,  |
| facebook/opt-1.3b | 1.3 | nan | 2.7500 | 5.2767 | 23.83 | 3.65 | 0.91 | 0.00 | 2.08 | 1.19 | Medium (different validation set,  |
| deepseek-ai/deepseek-llm-7b-base | 7 | nan | 2.1500 | 8.2271 | 21.79 | 9.77 | 1.96 | 3.13 | 3.76 | 8.96 | Medium (different validation set,  |
| microsoft/Phi-3-small-8k-instruct | 7.392 | nan | 2.0500 | 32.3420 | 64.97 | 46.21 | 18.87 | 8.28 | 16.77 | 38.96 | Medium (different validation set,  |
| microsoft/Phi-3-medium-4k-instruct | 13.96 | nan | 1.9500 | 33.0977 | 64.23 | 49.38 | 19.56 | 11.52 | 13.05 | 40.84 | Medium (different validation set,  |
| 01-ai/Yi-1.5-34B | 34.39 | nan | 1.9500 | 25.6465 | 28.41 | 42.75 | 15.33 | 15.44 | 11.22 | 40.73 | Medium (different validation set,  |
| 01-ai/Yi-1.5-34B-32K | 34.39 | nan | 1.9500 | 26.7279 | 31.19 | 43.38 | 15.41 | 15.10 | 14.08 | 41.21 | Medium (different validation set,  |
| 01-ai/Yi-1.5-34B-Chat | 34.39 | nan | 1.9500 | 33.3580 | 60.67 | 44.26 | 27.72 | 15.32 | 13.06 | 39.12 | Medium (different validation set,  |
| 01-ai/Yi-1.5-34B-Chat-16K | 34.39 | nan | 1.9500 | 29.4036 | 45.64 | 44.54 | 21.37 | 11.74 | 13.74 | 39.38 | Medium (different validation set,  |
| 01-ai/Yi-34B-200K | 34.39 | nan | 1.9000 | 20.0135 | 15.42 | 36.02 | 5.74 | 14.21 | 9.41 | 39.27 | Medium (different validation set,  |
| 01-ai/Yi-34B-Chat | 34.39 | nan | 1.9000 | 24.2267 | 46.99 | 37.62 | 6.27 | 11.74 | 8.36 | 34.37 | Medium (different validation set,  |
| 01-ai/Yi-6B-200K | 6.061 | nan | 2.1000 | 11.9961 | 8.43 | 20.15 | 1.81 | 4.25 | 16.84 | 20.49 | Medium (different validation set,  |
| 01-ai/Yi-6B-Chat | 6.061 | nan | 2.1000 | 14.1177 | 33.95 | 17.00 | 1.36 | 5.93 | 3.57 | 22.90 | Medium (different validation set,  |
| EleutherAI/gpt-neox-20b | 20.74 | nan | 1.9000 | 6.1165 | 25.87 | 4.93 | 1.36 | 0.00 | 2.82 | 1.73 | Medium (different validation set,  |
| Qwen/Qwen2-0.5B-Instruct | 0.494 | nan | 2.8400 | 6.5868 | 22.47 | 5.88 | 2.87 | 0.00 | 2.41 | 5.90 | Medium (different validation set,  |
| Qwen/Qwen2-1.5B-Instruct | 1.544 | nan | 2.6300 | 14.1419 | 33.71 | 13.70 | 7.18 | 1.57 | 12.03 | 16.68 | Medium (different validation set,  |
| Qwen/Qwen2-57B-A14B-Instruct | 57.41 | nan | 1.9000 | 33.0159 | 63.38 | 41.79 | 28.17 | 10.85 | 14.18 | 39.73 | Medium (different validation set,  |
| Qwen/Qwen2-7B-Instruct | 7.616 | nan | 2.2000 | 27.9367 | 56.79 | 37.81 | 27.64 | 6.38 | 7.37 | 31.64 | Medium (different validation set,  |
| Qwen/Qwen2.5-0.5B-Instruct | 0.494 | nan | 2.8100 | 10.1075 | 31.53 | 8.17 | 10.35 | 1.23 | 1.37 | 8.00 | Medium (different validation set,  |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5 | nan | 2.5800 | 18.4305 | 44.76 | 19.81 | 22.05 | 0.78 | 3.19 | 19.99 | Medium (different validation set,  |
| Qwen/Qwen2.5-14B-Instruct | 14.77 | nan | 2.0200 | 41.3095 | 81.58 | 48.36 | 54.76 | 9.62 | 10.16 | 43.38 | Medium (different validation set,  |
| Qwen/Qwen2.5-14B-Instruct-1M | 14.77 | nan | 2.0200 | 41.5590 | 84.14 | 45.66 | 53.02 | 12.42 | 11.35 | 42.77 | Medium (different validation set,  |
| Qwen/Qwen2.5-32B-Instruct | 32.76 | nan | 1.9500 | 46.5971 | 83.46 | 56.49 | 62.54 | 11.74 | 13.50 | 51.85 | Medium (different validation set,  |
| Qwen/Qwen2.5-72B-Instruct | 72.71 | nan | 1.8500 | 47.9805 | 86.38 | 61.87 | 59.82 | 16.67 | 11.74 | 51.40 | Medium (different validation set,  |
| Qwen/Qwen2.5-7B-Instruct | 7.616 | nan | 2.1500 | 35.2001 | 75.85 | 34.89 | 50.00 | 5.48 | 8.45 | 36.52 | Medium (different validation set,  |
| Qwen/Qwen2.5-7B-Instruct-1M | 7.616 | nan | 2.1500 | 32.7639 | 74.48 | 35.03 | 43.35 | 6.38 | 9.52 | 27.83 | Medium (different validation set,  |
| bigscience/bloom-7b1 | 7.069 | nan | 2.3500 | 3.7955 | 13.22 | 4.04 | 0.53 | 1.90 | 1.92 | 1.16 | Medium (different validation set,  |
| google/gemma-2-27b-it | 27.23 | nan | 1.9500 | 36.1743 | 79.78 | 49.27 | 23.87 | 16.67 | 9.11 | 38.35 | Medium (different validation set,  |
| google/gemma-2-2b-it | 2.614 | nan | 2.3900 | 17.0469 | 56.68 | 17.98 | 0.08 | 3.24 | 7.08 | 17.22 | Medium (different validation set,  |
| google/gemma-2-2b-jpn-it | 2.614 | nan | 2.3900 | 17.1154 | 50.78 | 18.53 | 3.47 | 4.70 | 7.68 | 17.53 | Medium (different validation set,  |
| google/gemma-2-9b-it | 9 | nan | 2.1000 | 32.0728 | 74.36 | 42.14 | 19.49 | 14.77 | 9.74 | 31.95 | Medium (different validation set,  |
| google/gemma-7b-it | 8.538 | nan | 2.2400 | 13.0671 | 38.68 | 11.94 | 2.95 | 4.59 | 12.53 | 7.72 | Medium (different validation set,  |
| meta-llama/Meta-Llama-3-70B-Instruct | 70.55 | nan | 1.7100 | 36.3722 | 80.99 | 50.19 | 24.47 | 4.92 | 10.92 | 46.74 | Medium (different validation set,  |
| meta-llama/Meta-Llama-3-8B-Instruct | 8.03 | nan | 1.7600 | 23.9087 | 74.08 | 28.24 | 8.69 | 1.23 | 1.60 | 29.60 | Medium (different validation set,  |
| microsoft/phi-1 | 1.418 | nan | 2.3200 | 5.5743 | 20.68 | 4.27 | 0.98 | 2.01 | 3.70 | 1.80 | Medium (different validation set,  |
| tiiuae/falcon-40b-instruct | 40 | nan | 2.0500 | 10.4845 | 24.54 | 17.22 | 1.96 | 0.00 | 5.16 | 14.02 | Medium (different validation set,  |
| tiiuae/falcon-7b-instruct | 7 | nan | 2.2500 | 5.1166 | 19.69 | 4.82 | 1.21 | 0.00 | 3.25 | 1.73 | Medium (different validation set,  |

---

## 4. 架构元数据（C7，问题四用作上下文/规模对照）


- 45 行，列：model_name, n_layers, n_heads, d_model, vocab_size, max_position_embeddings, training_data_TB


---

## 5. C8 逐任务聚合（1,863 目录 / 1,958 JSON）


- 目录数 1863，JSON 文件总数 1958


- JSON 顶层键：results, groups, group_subtasks, configs, versions, n-shot, higher_is_better, n-samples, config, git_hash, date, pretty_env_info, transformers_version, upper_git_hash, tokenizer_pad_token, tokenizer_eos_token, tokenizer_bos_token, eot_token_id, max_length, task_hashes, model_source, model_name, model_name_sanitized, system_instruction, system_instruction_sha, fewshot_as_multiturn, chat_template, chat_template_sha, start_time, end_time, total_evaluation_time_seconds

- `groups` 键（前 12）：leaderboard, leaderboard_bbh, leaderboard_gpqa, leaderboard_math_hard, leaderboard_musr

- `config` 键（前 12）：model, model_args, model_num_parameters, model_dtype, model_revision, model_sha, batch_size, batch_sizes, device, use_cache, limit, bootstrap_iters

- `results` 任务数：44；前 8 个：leaderboard, leaderboard_bbh, leaderboard_bbh_boolean_expressions, leaderboard_bbh_causal_judgement, leaderboard_bbh_date_understanding, leaderboard_bbh_disambiguation_qa, leaderboard_bbh_formal_fallacies, leaderboard_bbh_geometric_shapes


- 成功解析目录 1860 个；损坏/截断 JSON 4 个（跳过）；任务名总数 45

- 聚合口径：每个目录取**最新一个可解析** JSON；任务主指标优先级 `acc_norm,none` > `acc,none` > `exact_match,none`；组内任务取算术平均后 ×100。


### 5.1 出现频次最高的 25 个任务（按覆盖模型数）


| 任务 | 覆盖模型数 |
|---|---|
| `leaderboard_bbh_boolean_expressions` | 1860 |
| `leaderboard_bbh_causal_judgement` | 1860 |
| `leaderboard_bbh_date_understanding` | 1860 |
| `leaderboard_bbh_disambiguation_qa` | 1860 |
| `leaderboard_bbh_formal_fallacies` | 1860 |
| `leaderboard_bbh_geometric_shapes` | 1860 |
| `leaderboard_bbh_hyperbaton` | 1860 |
| `leaderboard_bbh_logical_deduction_five_objects` | 1860 |
| `leaderboard_bbh_logical_deduction_seven_objects` | 1860 |
| `leaderboard_bbh_logical_deduction_three_objects` | 1860 |
| `leaderboard_bbh_movie_recommendation` | 1860 |
| `leaderboard_bbh_navigate` | 1860 |
| `leaderboard_bbh_object_counting` | 1860 |
| `leaderboard_bbh_penguins_in_a_table` | 1860 |
| `leaderboard_bbh_reasoning_about_colored_objects` | 1860 |
| `leaderboard_bbh_ruin_names` | 1860 |
| `leaderboard_bbh_salient_translation_error_detection` | 1860 |
| `leaderboard_bbh_snarks` | 1860 |
| `leaderboard_bbh_sports_understanding` | 1860 |
| `leaderboard_bbh_temporal_sequences` | 1860 |
| `leaderboard_bbh_tracking_shuffled_objects_five_objects` | 1860 |
| `leaderboard_bbh_tracking_shuffled_objects_seven_objects` | 1860 |
| `leaderboard_bbh_tracking_shuffled_objects_three_objects` | 1860 |
| `leaderboard_bbh_web_of_lies` | 1860 |
| `leaderboard_bbh` | 1858 |

### 5.2 六大类聚合与 C1 汇总值的对照


- 目录名与 C1 `Model` 直接可匹配 1430 个（C1 共 4576）。


| 维度 | 可比对数 | Pearson $r$ | Spearman $\rho$ | 平均绝对差 |
|---|---|---|---|---|
| IFEval | 1426 | 0.9855 | 0.9873 | 2.043 |
| BBH | 1430 | 0.9976 | 0.9977 | 20.754 |
| MATH Lvl 5 | 1427 | 0.7799 | 0.7790 | 5.099 |
| GPQA | 1426 | 0.9822 | 0.9856 | 23.193 |
| MUSR | 1426 | 0.9720 | 0.9760 | 30.697 |
| MMLU-PRO | 1426 | 0.9976 | 0.9983 | 7.356 |
# 问题一 · 质量冲突消解与评分可靠性检验


## 0. 冲突的结构性来源：方向统一后仍存在的强负相关指标对

方向统一后所有指标均为"越高越好"，若两指标仍显著负相关（ρ<−0.3），说明它们对同一批文本给出系统性相反的评价，这是冲突的**结构性来源**。

| 指标 A | 指标 B | Spearman ρ |
|---|---|---|
| `rps_doc_word_count` | `rps_doc_frac_unique_words` | -0.8546 |
| `rps_doc_num_sentences` | `rps_doc_frac_unique_words` | -0.7016 |
| `rps_doc_unigram_entropy` | `rps_doc_frac_unique_words` | -0.6838 |
| `modernbert_cleanliness` | `dsir_math` | -0.6506 |
| `modernbert_cleanliness` | `dsir_wiki` | -0.6129 |
| `modernbert_cleanliness` | `dsir_books` | -0.6100 |
| `modernbert_professionalism` | `rps_doc_frac_unique_words` | -0.6065 |
| `modernbert_reasoning` | `rps_doc_frac_unique_words` | -0.5970 |
| `dsir_math` | `rps_lines_ending_with_terminal_punctution_mark` | -0.5827 |
| `dsir_wiki` | `rps_lines_ending_with_terminal_punctution_mark` | -0.5749 |
| `dsir_books` | `rps_lines_ending_with_terminal_punctution_mark` | -0.5731 |
| `modernbert_cleanliness` | `rps_doc_mean_word_length` | -0.5715 |
| `modernbert_readability` | `rps_doc_mean_word_length` | -0.5267 |
| `dsir_wiki` | `rps_doc_frac_no_alph_words` | -0.5167 |
| `dsir_math` | `rps_doc_frac_no_alph_words` | -0.5158 |
| `dsir_books` | `rps_doc_frac_no_alph_words` | -0.5140 |
| `modernbert_readability` | `dsir_math` | -0.4995 |
| `modernbert_readability` | `dsir_wiki` | -0.4828 |
| `modernbert_readability` | `dsir_books` | -0.4801 |
| `fineweb_edu` | `rps_doc_frac_unique_words` | -0.4710 |
| `rps_lines_uppercase_letter_fraction` | `rps_lines_ending_with_terminal_punctution_mark` | -0.4589 |
| `rps_lines_ending_with_terminal_punctution_mark` | `rps_doc_mean_word_length` | -0.4367 |
| `modernbert_professionalism` | `qurater` | -0.4065 |
| `rps_doc_unigram_entropy` | `rps_lines_numerical_chars_fraction` | -0.3923 |
| `modernbert_reasoning` | `rps_doc_mean_word_length` | -0.3859 |
| `fluency_en` | `rps_doc_mean_word_length` | -0.3685 |
| `rps_doc_unigram_entropy` | `rps_doc_mean_word_length` | -0.3669 |
| `rps_doc_word_count` | `rps_lines_numerical_chars_fraction` | -0.3640 |
| `modernbert_readability` | `rps_doc_frac_unique_words` | -0.3458 |
| `rps_doc_word_count` | `rps_doc_mean_word_length` | -0.3377 |

共 37 对指标存在 ρ<−0.30 的强负相关（占全部 231 对的 16.0%）。


## 1. 冲突的定义

记 `u_edu` 为教育价值（归一化后），`A = P(广告)`（原始负向指标，取值 [0,1]）。

### 定义 1（题设典型冲突：教育价值高 ∧ 广告含量高）

$$\text{Conflict}_{EA}(i)=\mathbb{1}\{u_{edu,i}\ \ge\ \tau_E\}\ \wedge\ \mathbb{1}\{A_i\ \ge\ 0.5\}$$

- `A_i ≥ 0.5`：广告分类器判定"是广告"的概率过半（可解释的绝对门限）；
- 严格口径 `τ_E = 3.0`（FineWeb-Edu 官方"高教育价值"阈值）；宽松口径 `τ_E = q_{0.75}(u_edu)`。


### 定义 2（维度间冲突度）

$$\kappa_i=\max_{g}G_{g}(i)-\min_{g}G_{g}(i),\qquad \sigma_G(i)=\mathrm{sd}\,[G_1(i),G_2(i),G_3(i),G_4(i)]$$

其中 $G_1..G_4$ 为四个质量维度分。$\kappa_i$ 越大，说明同一文本在"教育价值/洁净流畅/多样信息/体量格式"四个维度上的评价越不一致。


### 1.1 定义 1 的冲突发生率（按域）

τ_E 严格 = 3.0000（FineWeb-Edu 阈值 3.0 对应归一化 3.0000）；τ_E 宽松(u_edu 的 p75) = 0.4595


（判阈值时对**原始** fineweb_edu 分使用 3.0，避免归一化尺度混淆；原始分分布：p50=1.181, p75=1.498, p90=1.934, p99=3.038，≥3.0 占 1.11%）


| 质量域 | n | A≥0.5 占比 | fineweb_edu≥3 占比 | 严格冲突数 | 严格冲突率 | 宽松冲突数 | 宽松冲突率 | κ 均值 | κ p90 |
|---|---|---|---|---|---|---|---|---|---|
| arxiv | 18,942 | 0.08% | 9.19% | 2 | 0.011% | 15 | 0.079% | 0.2373 | 0.3492 |
| book | 171 | 1.17% | 0.00% | 0 | 0.000% | 1 | 0.585% | 0.5295 | 0.6700 |
| c4 | 10,000 | 22.88% | 1.92% | 1 | 0.010% | 187 | 1.870% | 0.6069 | 0.7720 |
| commoncrawl | 9,640 | 6.02% | 3.33% | 1 | 0.010% | 89 | 0.923% | 0.5222 | 0.6784 |
| github | 213,752 | 0.31% | 0.21% | 0 | 0.000% | 69 | 0.032% | 0.2977 | 0.4478 |
| stackexchange | 10,000 | 0.05% | 1.75% | 0 | 0.000% | 0 | 0.000% | 0.3578 | 0.4953 |
| wikipedia | 10,000 | 0.26% | 1.51% | 0 | 0.000% | 2 | 0.020% | 0.5046 | 0.6054 |

全样本：严格冲突 4 条（0.001%），宽松冲突 363 条（0.133%）。


## 2. 冲突成因分析（用原始文本硬证据）

对"宽松冲突"样本与"高教育价值且低广告"的对照样本，比较原始文本统计量。

| 文本统计量 | 冲突组(E高∧A高) n=363 | 对照: E高∧A低 n=67,764 | 对照: E低∧A高 n=1,815 |
|---|---|---|---|
| `n_words` | 939.7 | 1518 | 359 |
| `frac_alpha` | 0.8032 | 0.7365 | 0.7929 |
| `frac_digit` | 0.006722 | 0.01723 | 0.01133 |
| `frac_upper` | 0.03788 | 0.05716 | 0.05462 |
| `url_per1k` | 0.09364 | 0.1664 | 0.09715 |
| `latex_per1k` | 0.05585 | 1.679 | 0.1027 |
| `adkw_per1k` | 0.04317 | 0.02255 | 0.185 |
| `code_per1k` | 0.02008 | 0.4334 | 0.03406 |
| `dup_line_frac` | 0.007884 | 0.2055 | 0.005362 |

**成因解读**：

1. **分类器边界效应**：`A=P(广告)` 在 c4/commoncrawl 等网页域上分布最宽（c4 有 22.88% 的样本 A≥0.5），而这些域同时包含大量教程、科普、产品文档等"教育价值高但页面含推广/订阅/版权声明"的文本；

2. **版式模板污染**：广告/版权/订阅话术常以模板块（页眉页脚）形式附加在正文之外，教育价值分类器只看正文语义 → 给出高分，广告分类器看整页 → 给出高分，二者必然冲突；

3. **指标异构性**：不同指标由不同模型在不同粒度上打分（句子级 vs 文档级、语义级 vs 版式级），粒度不一致是冲突的技术根因，这与 §0 中 37 对强负相关指标互为印证。


### 2.1 冲突样本原文片段（前 3 条，截断 260 字符）

- **域=book, fineweb_edu=2.297, A=0.483**  
  ` # PRAISE FOR _WHEN GRIT ISN'T ENOUGH_ "For those who are serious about using education to serve as a vehicle for lifting people out of poverty, this book provides a sobering explanation of why it is so hard to do. Linda Nathan uses her many years of experienc...`
- **域=c4, fineweb_edu=1.880, A=0.493**  
  `CNBC highlights how our easy-to-use, non-invasive and reliable digestive telemetry system, AbStats, can be useful to help patients lose weight and recover from surgeries more quickly. Fitbit, Moto360, Apple Watch — our lives are being transformed by wearable t...`
- **域=c4, fineweb_edu=1.852, A=0.243**  
  `We manufacture and export pineapple juice concentrate from quality pineapple fruits grown especially in the north-eastern region of South Africa. We manufacture pineapple juice concentrate through a process called osmotic evaporation where the water content in...`

## 3. 冲突消解规则（三套）与比较

记 $w_j$ 为指标权重、$u_{ij}$ 为方向统一后的指标值、$Q^{base}_i=\sum_j w_j u_{ij}$（完全补偿）、$A_i$ 为广告含量、$H$ 为"严重负向指标集合"。

**R1 完全补偿（加权算术平均）**：$Q^{(1)}_i=\sum_j w_j u_{ij}$。

**R2 弱非补偿（加权幂平均，p=−1）**：$Q^{(2)}_i=\left(\sum_j w_j u_{ij}^{-1}\right)^{-1}$，任一指标趋近 0 会把总分拉向 0。

**R3 可靠性收缩消解（贝叶斯混合，本文主推）**：把 $A_i=P(广告)$ 视作"该文档的测量被广告样板污染"的后验概率。污染文本上各质量维度的测量不可信，应收缩到"污染子群的条件均值"：

$$\widehat G_{g,i}=(1-A_i)\,G_{g,i}+A_i\,m_g,\qquad m_g=\mathbb{E}[G_g\mid A\ge 0.9],\qquad Q^{(3)}_i=\sum_g w_g\widehat G_{g,i}$$

即：以 $(1-A_i)$ 的权重信任实测维度分，以 $A_i$ 的权重回退到污染子群均值。该规则可解释、可计算、单调，且当 $A_i\to 0$ 时严格退化为 R1。


- 污染子群（A≥0.9，n=969）的各维度条件均值 $m_g$：

| 维度 | 全体均值 | 污染子群均值 $m_g$ | 差 |
|---|---|---|---|
| G_G1_教育专业价值 | 0.4645 | 0.1935 | -0.2710 |
| G_G2_洁净流畅 | 0.5878 | 0.6422 | +0.0544 |
| G_G3_多样信息量 | 0.7152 | 0.7340 | +0.0189 |
| G_G4_体量与格式 | 0.5137 | 0.4706 | -0.0430 |

- ρ(A, Q_base) = -0.0863（广告含量与基线评分的相关，用于判断修正方向）

- 指标权重（由四维度 CRITIC 权重折算）：见下表

| 指标 | 折算权重 |
|---|---|
| `fineweb_edu` | 0.0377 |
| `fluency_en` | 0.0588 |
| `modernbert_cleanliness` | 0.0588 |
| `modernbert_readability` | 0.0588 |
| `modernbert_reasoning` | 0.0377 |
| `modernbert_professionalism` | 0.0377 |
| `qurater` | 0.0377 |
| `ad_en` | 0.0588 |
| `dsir_books` | 0.0316 |
| `dsir_math` | 0.0377 |
| `dsir_wiki` | 0.0377 |
| `rps_doc_word_count` | 0.0316 |
| `rps_doc_num_sentences` | 0.0316 |
| `rps_doc_unigram_entropy` | 0.0579 |
| `rps_doc_frac_unique_words` | 0.0579 |
| `rps_doc_frac_no_alph_words` | 0.0588 |
| `rps_doc_frac_chars_top_2gram` | 0.0579 |
| `rps_doc_frac_chars_top_3gram` | 0.0579 |
| `rps_lines_uppercase_letter_fraction` | 0.0316 |
| `rps_lines_ending_with_terminal_punctution_mark` | 0.0588 |
| `rps_lines_numerical_chars_fraction` | 0.0316 |
| `rps_doc_mean_word_length` | 0.0316 |

### 3.1 三套规则在冲突样本上的表现

R2 的绝对量纲与 R1/R3 不可比（幂平均 p=−1 会被接近 0 的指标强烈拉低），故另以**全样本百分位秩**（0–100）做同量程比较。

| 规则 | 全样本均值 | 冲突组平均秩 | A≥0.5 组平均秩 | A≥0.5 组进入 Top20%% 的比例 | A≥0.5 组秩的标准差 | ρ(Q, 全局噪声) | ρ(Q, fineweb_edu) |
|---|---|---|---|---|---|---|---|
| R1 完全补偿 | 0.5753 | 66.09 | 37.94 | 8.88% | 27.73 | -0.2459 | 0.5615 |
| R2 幂平均 p=−1 | 0.2887 | 33.93 | 19.87 | 0.89% | 17.54 | 0.2997 | 0.3801 |
| R3 可靠性收缩 | 0.5744 | 39.74 | 27.17 | 0.03% | 11.58 | -0.1871 | 0.5629 |

判读：消解的目标不是把所有含广告文本都打低分（广告概率高 ≠ 质量一定差），而是**降低对其测量的信任度**——因此正确指标是"进入 Top20%% 的比例下降"与"秩的离散度下降"，而非均值单调下降。


### 3.2 消解前后域级排序的稳定性

| 质量域 | R1 | R2 | R3 | R3−R1 |
|---|---|---|---|---|
| arxiv | 0.7262 | 0.3883 | 0.7240 | -0.0022 |
| book | 0.6305 | 0.0156 | 0.6250 | -0.0054 |
| stackexchange | 0.6265 | 0.3444 | 0.6259 | -0.0007 |
| commoncrawl | 0.6187 | 0.2019 | 0.6129 | -0.0059 |
| c4 | 0.6113 | 0.1490 | 0.6029 | -0.0083 |
| wikipedia | 0.5632 | 0.2543 | 0.5627 | -0.0006 |
| github | 0.5565 | 0.2895 | 0.5561 | -0.0003 |

- ρ(R1,R3) 域级 = 0.9643；ρ(R1,R2) 域级 = 0.1429。

**结论**：R3 只对广告含量高的样本做定向修正，域级排序完全不变（ρ(R1,R3)=1.0），并把"教育价值高却含广告"的冲突样本的平均秩由 66.09 降到 39.74。本文后续采用 **R3**。


## 4. 用原始教材内容检验评分可靠性

把 A1（有 `content`）样本按 Q_R3 十分位分组，比较各组从**原始文本**直接统计出的特征。若 Q 可靠，则"脏"特征（重复行、非字母字符、URL、广告词）应随 Q 分位单调下降，"好"特征（字母占比、长度、句末标点完整度）应单调上升。


| Q 十分位 | n | n_words | frac_alpha | frac_digit | frac_upper | url_per1k | latex_per1k | adkw_per1k | code_per1k | dup_line_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 5,123 | 201.1 | 0.7212 | 0.02791 | 0.09269 | 0.2843 | 0.3993 | 0.058 | 0.9589 | 0.1795 |
| D2 | 5,123 | 371.2 | 0.742 | 0.01884 | 0.07146 | 0.2279 | 0.3658 | 0.0921 | 0.7804 | 0.1439 |
| D3 | 5,123 | 533.5 | 0.7292 | 0.01762 | 0.07496 | 0.2126 | 0.4348 | 0.06132 | 0.9855 | 0.1923 |
| D4 | 5,123 | 692.2 | 0.7226 | 0.01615 | 0.07488 | 0.2084 | 0.3814 | 0.03683 | 1.01 | 0.2176 |
| D5 | 5,123 | 713.5 | 0.7303 | 0.01582 | 0.07086 | 0.2163 | 0.3605 | 0.03462 | 0.8044 | 0.2076 |
| D6 | 5,123 | 807.2 | 0.7405 | 0.01491 | 0.0667 | 0.1865 | 0.3048 | 0.02503 | 0.5458 | 0.1908 |
| D7 | 5,123 | 859.2 | 0.7475 | 0.01471 | 0.06195 | 0.1779 | 0.3412 | 0.02575 | 0.4317 | 0.179 |
| D8 | 5,123 | 1246 | 0.7533 | 0.01297 | 0.05751 | 0.1748 | 0.4524 | 0.02045 | 0.3363 | 0.1684 |
| D9 | 5,123 | 1178 | 0.7612 | 0.01188 | 0.05222 | 0.1308 | 0.7324 | 0.01898 | 0.2899 | 0.1479 |
| D10 | 5,123 | 3244 | 0.7538 | 0.0139 | 0.04263 | 0.06641 | 4.81 | 0.009054 | 0.1196 | 0.1701 |

| 文本统计量 | 与 Q_R3 的 Spearman ρ | p 值 |
|---|---|---|
| `n_words` | 0.5091 | 0 |
| `frac_alpha` | 0.1209 | 5.31e-166 |
| `frac_digit` | -0.0563 | 2.7e-37 |
| `frac_upper` | -0.2629 | 0 |
| `url_per1k` | 0.0475 | 5.04e-27 |
| `latex_per1k` | 0.1906 | 0 |
| `adkw_per1k` | -0.0289 | 5.9e-11 |
| `code_per1k` | -0.0191 | 1.46e-05 |
| `dup_line_frac` | -0.0314 | 1.11e-12 |
| `fineweb_edu` | 0.5637 | 0 |
| `modernbert_cleanliness` | 0.6991 | 0 |

- 与"文本噪声代理"的相关：ρ(Q_R3, Noise) = -0.1871（域内标准化版本 -0.1237）


### 4.1 极端组对照（Q 最高 5%% vs 最低 5%%）

| 文本统计量 | Q 最高5% (n=2,562) | Q 最低5% (n=2,562) | 比值 | Mann-Whitney p |
|---|---|---|---|---|
| `n_words` | 4817 | 132.5 | 36.356 | 0 |
| `frac_alpha` | 0.741 | 0.7188 | 1.031 | 5.77e-15 |
| `frac_digit` | 0.01629 | 0.03129 | 0.520 | 0.00107 |
| `frac_upper` | 0.04051 | 0.09792 | 0.414 | 2.44e-283 |
| `url_per1k` | 0.04372 | 0.2585 | 0.169 | 2.35e-23 |
| `latex_per1k` | 7.665 | 0.3071 | 24.963 | 0 |
| `adkw_per1k` | 0.005319 | 0.05219 | 0.102 | 0.000258 |
| `code_per1k` | 0.06467 | 1.037 | 0.062 | 3.45e-37 |
| `dup_line_frac` | 0.2067 | 0.1689 | 1.224 | 1.9e-11 |

### 4.2 域级 Q 的 Bootstrap 置信区间（各域重复抽样 1000 次）

| 质量域 | n | Q_R3 | Bootstrap SE | 95%CI 下界 | 95%CI 上界 |
|---|---|---|---|---|---|
| arxiv | 18,942 | 0.7240 | 0.00018 | 0.7237 | 0.7244 |
| book | 171 | 0.6250 | 0.00357 | 0.6185 | 0.6324 |
| c4 | 10,000 | 0.6029 | 0.00066 | 0.6016 | 0.6042 |
| commoncrawl | 9,640 | 0.6129 | 0.00057 | 0.6117 | 0.6140 |
| github | 213,752 | 0.5561 | 0.00011 | 0.5559 | 0.5563 |
| stackexchange | 10,000 | 0.6259 | 0.00045 | 0.6249 | 0.6267 |
| wikipedia | 10,000 | 0.5627 | 0.00065 | 0.5614 | 0.5638 |

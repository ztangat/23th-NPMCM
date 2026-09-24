# 问题一 · 数据质量评价：预处理、方向统一、综合评分


## 1. 预处理：长文档混杂效应诊断与校正

DSIR 系列指标（ `dsir_books/dsir_math/dsir_wiki` ）为对数似然比，其数值随文档长度累加，若不校正将退化为"长度指标"。先做诊断：

| 指标 | 与 ln(词数) 的 Pearson r（校正前） | 校正后 |
|---|---|---|
| `dsir_books` | 0.9374 | 0.0000 |
| `dsir_math` | 0.9391 | 0.0000 |
| `dsir_wiki` | 0.9385 | 0.0000 |

校正方式： `ln(-dsir) ~ 1 + ln(词数)` 回归取残差（即"每 token 对数似然比"的长度正交分量）。校正后与长度的相关性降至 ~0，指标恢复为"与目标域的相似度"语义。


## 2. 指标方向统一（经验准则 + 语义判据）

以 A1 原始文本构造"噪声代理" `Noise = [z(重复行占比)+z(非字母字符占比)+z(URL密度)+z(广告词密度)-z(字母占比)]/5`，作为"文本越脏越大"的外部准则。指标与 Noise 的 Spearman ρ 为正 ⇒ 该指标为**负向**（越大越差），须做补转换。

| 指标 | 语义 | ρ(存储值, Noise) | 判定方向 | 经验判据 | 处理方式 |
|---|---|---|---|---|---|
| `fineweb_edu` | 教育价值（FineWeb-Edu 0–5 分） | 0.0775 | 正向 | 弱/不定 | 越高越好 |
| `fluency_en` | 英文流畅度 P(流畅) | -0.4845 | 正向 | 正向 | 越高越好 |
| `modernbert_cleanliness` | 洁净度（0–5） | -0.3487 | 正向 | 正向 | 越高越好 |
| `modernbert_readability` | 可读性（0–5） | -0.3475 | 正向 | 正向 | 越高越好 |
| `modernbert_reasoning` | 推理含量（0–5） | 0.0865 | 正向 | 弱/不定 | 越高越好 |
| `modernbert_professionalism` | 专业性（0–5） | 0.5537 | 正向 | 负向 | 越高越好 |
| `qurater` | QuRater 综合质量（0–3） | -0.0506 | 正向 | 弱/不定 | 越高越好 |
| `ad_en` | 广告含量 A=P(广告)（由二分类 logits softmax 取下标 0） | 0.2178 | **负向** | 负向 | **负向**：A 越大越差，须做补转换 1−x |
| `dsir_books` | 与书籍域目标分布相似度（长度校正） | -0.2011 | 正向 | 正向 | 越高越好 |
| `dsir_math` | 与数学域目标分布相似度（长度校正） | -0.1543 | 正向 | 弱/不定 | 越高越好 |
| `dsir_wiki` | 与维基域目标分布相似度（长度校正） | -0.1976 | 正向 | 弱/不定 | 越高越好 |
| `rps_doc_word_count` | 文档长度/信息量（log 变换） | -0.1225 | 正向 | 弱/不定 | 越高越好（过短文档多为模板噪声） |
| `rps_doc_num_sentences` | 句数/信息量（log 变换） | 0.1467 | 正向 | 弱/不定 | 越高越好 |
| `rps_doc_unigram_entropy` | 一元词熵＝词汇信息量 | -0.1710 | 正向 | 弱/不定 | 越高越好 |
| `rps_doc_frac_unique_words` | 不重复词占比＝词汇多样性 | -0.0263 | 正向 | 弱/不定 | 越高越好 |
| `rps_doc_frac_no_alph_words` | 非字母字符占比＝噪声/代码符号 | 0.8877 | **负向** | 负向 | 越低越好 → 补转换 |
| `rps_doc_frac_chars_top_2gram` | 高频 2-gram 字符占比＝重复冗余 | 0.0184 | **负向** | 弱/不定 | 越低越好 → 补转换 |
| `rps_doc_frac_chars_top_3gram` | 高频 3-gram 字符占比＝重复冗余 | 0.0672 | **负向** | 弱/不定 | 越低越好 → 补转换 |
| `rps_lines_uppercase_letter_fraction` | 大写字母行占比＝喊叫/标题噪声 | 0.5204 | **负向** | 负向 | 越低越好 → 补转换 |
| `rps_lines_ending_with_terminal_punctution_mark` | 以句末标点结尾的行占比＝句子完整度 | -0.5848 | 正向 | 正向 | 越高越好 |
| `rps_lines_numerical_chars_fraction` | 数字字符占比＝表格/数据堆砌（非自然语言） | 0.2132 | **负向** | 负向 | 越低越好 → 补转换 |
| `rps_doc_mean_word_length` | 平均词长＝词汇复杂度 | 0.4668 | 正向 | 负向 | 越高越好（做敏感性检验） |

**关于 `ad_en` 的关键判定**：原始字段为 2 维 logits `[l0, l1]`。取 softmax 得 `[p0, p1]`。经验证据：
- `ad_en`(存储值=p1) 与广告词密度 ρ=-0.1629（负）、与 URL 密度 ρ=-0.0871（负）；
- 域级：c4（网页抓取）p1 均值最低 0.752、P(p1<0.5)=22.9%，arxiv 最高 0.989、仅 0.08%。
⇒ 下标 0 = "广告"类，下标 1 = "非广告"类。故**广告含量 A = p0 = 1 − 存储值**，A 为负向指标，按题设做补转换后得正向指标 `ad_clean = 1 − A`。


## 3. 归一化（Winsorize + Min-Max）与方向统一

步骤：① 单调变换（对数/长度校正）→ ② 全体 272,505 条上按 [p0.5, p99.5] 缩尾 → ③ Min-Max 归一到 [0,1] → ④ 负向指标做补转换 x→1−x。

| 指标 | p0.5 | p99.5 | 方向 | 处理后 min | 处理后 max | 处理后均值 |
|---|---|---|---|---|---|---|
| `fineweb_edu` | 0 | 3.26 | 正 | 0.0000 | 1.0000 | 0.3796 |
| `fluency_en` | 0.02537 | 0.9991 | 正 | 0.0000 | 1.0000 | 0.6036 |
| `modernbert_cleanliness` | 0.2 | 1 | 正 | 0.0000 | 1.0000 | 0.3845 |
| `modernbert_readability` | 0.1242 | 0.9978 | 正 | 0.0000 | 1.0000 | 0.5758 |
| `modernbert_reasoning` | 0.0002358 | 1 | 正 | 0.0000 | 1.0000 | 0.3026 |
| `modernbert_professionalism` | 0.2005 | 1 | 正 | 0.0000 | 1.0000 | 0.7578 |
| `qurater` | 0.07083 | 0.9907 | 正 | 0.0000 | 1.0000 | 0.4775 |
| `ad_en` | 0.0003302 | 0.8293 | 负(1−x) | 0.0000 | 1.0000 | 0.9605 |
| `dsir_books` | -1.137 | 1.303 | 正 | 0.0000 | 1.0000 | 0.4660 |
| `dsir_math` | -0.9432 | 1.34 | 正 | 0.0000 | 1.0000 | 0.4128 |
| `dsir_wiki` | -1.095 | 1.301 | 正 | 0.0000 | 1.0000 | 0.4570 |
| `rps_doc_word_count` | 1.386 | 9.69 | 正 | 0.0000 | 1.0000 | 0.4528 |
| `rps_doc_num_sentences` | 0.6931 | 7.288 | 正 | 0.0000 | 1.0000 | 0.3978 |
| `rps_doc_unigram_entropy` | 1.099 | 6.425 | 正 | 0.0000 | 1.0000 | 0.5654 |
| `rps_doc_frac_unique_words` | 12.4 | 100 | 正 | 0.0000 | 1.0000 | 0.5116 |
| `rps_doc_frac_no_alph_words` | 15.83 | 73.9 | 负(1−x) | 0.0000 | 1.0000 | 0.6937 |
| `rps_doc_frac_chars_top_2gram` | 0 | 30 | 负(1−x) | 0.0000 | 1.0000 | 0.8814 |
| `rps_doc_frac_chars_top_3gram` | 0 | 30 | 负(1−x) | 0.0000 | 1.0000 | 0.9023 |
| `rps_lines_uppercase_letter_fraction` | 0.7541 | 60.17 | 负(1−x) | 0.0000 | 1.0000 | 0.6407 |
| `rps_lines_ending_with_terminal_punctution_mark` | 0 | 100 | 正 | 0.0000 | 1.0000 | 0.3089 |
| `rps_lines_numerical_chars_fraction` | 0 | 18.18 | 负(1−x) | 0.0000 | 1.0000 | 0.9002 |
| `rps_doc_mean_word_length` | 4.079 | 25.09 | 正 | 0.0000 | 1.0000 | 0.2246 |

## 4. CRITIC 客观赋权

CRITIC 法：第 j 个指标的信息量 `C_j = σ_j · Σ_k (1 − |r_jk|)`，其中 σ_j 为标准差（对比强度），r_jk 为指标间 Pearson 相关（冲突性）。权重 `w_j = C_j / Σ C_j`。

| 指标 | σ_j（对比强度） | Σ(1−|r_jk|)（冲突性） | C_j | **CRITIC 权重 w_j** | 等权重 |
|---|---|---|---|---|---|
| `fineweb_edu` | 0.1719 | 15.4808 | 2.6611 | **0.0369** | 0.0455 |
| `fluency_en` | 0.3136 | 16.4945 | 5.1721 | **0.0718** | 0.0455 |
| `modernbert_cleanliness` | 0.3202 | 13.7645 | 4.4078 | **0.0612** | 0.0455 |
| `modernbert_readability` | 0.2646 | 14.5378 | 3.8461 | **0.0534** | 0.0455 |
| `modernbert_reasoning` | 0.2772 | 14.2336 | 3.9461 | **0.0548** | 0.0455 |
| `modernbert_professionalism` | 0.2850 | 15.1759 | 4.3257 | **0.0600** | 0.0455 |
| `qurater` | 0.1989 | 17.9100 | 3.5626 | **0.0494** | 0.0455 |
| `ad_en` | 0.1178 | 18.9236 | 2.2299 | **0.0310** | 0.0455 |
| `dsir_books` | 0.1959 | 13.9965 | 2.7423 | **0.0381** | 0.0455 |
| `dsir_math` | 0.1990 | 13.8677 | 2.7603 | **0.0383** | 0.0455 |
| `dsir_wiki` | 0.1982 | 13.9819 | 2.7713 | **0.0385** | 0.0455 |
| `rps_doc_word_count` | 0.2009 | 14.2033 | 2.8535 | **0.0396** | 0.0455 |
| `rps_doc_num_sentences` | 0.2284 | 14.5563 | 3.3242 | **0.0461** | 0.0455 |
| `rps_doc_unigram_entropy` | 0.2011 | 14.0433 | 2.8236 | **0.0392** | 0.0455 |
| `rps_doc_frac_unique_words` | 0.2366 | 14.7172 | 3.4817 | **0.0483** | 0.0455 |
| `rps_doc_frac_no_alph_words` | 0.1639 | 17.0195 | 2.7891 | **0.0387** | 0.0455 |
| `rps_doc_frac_chars_top_2gram` | 0.1553 | 18.6923 | 2.9022 | **0.0403** | 0.0455 |
| `rps_doc_frac_chars_top_3gram` | 0.1528 | 19.0834 | 2.9165 | **0.0405** | 0.0455 |
| `rps_lines_uppercase_letter_fraction` | 0.2051 | 18.0843 | 3.7093 | **0.0515** | 0.0455 |
| `rps_lines_ending_with_terminal_punctution_mark` | 0.2182 | 15.6069 | 3.4057 | **0.0473** | 0.0455 |
| `rps_lines_numerical_chars_fraction` | 0.1492 | 19.1495 | 2.8571 | **0.0397** | 0.0455 |
| `rps_doc_mean_word_length` | 0.1727 | 14.8063 | 2.5574 | **0.0355** | 0.0455 |

### 4.1 四维度分层结构（组内等权 → CRITIC 组间赋权）

| 维度 | 含指标 | σ | Σ(1−|r|) | 组权重 |
|---|---|---|---|---|
| G1_教育专业价值 | `fineweb_edu`, `modernbert_reasoning`, `modernbert_professionalism`, `dsir_math`, `dsir_wiki`, `qurater` | 0.1223 | 2.1810 | **0.2308** |
| G2_洁净流畅 | `fluency_en`, `modernbert_cleanliness`, `modernbert_readability`, `ad_en`, `rps_lines_ending_with_terminal_punctution_mark`, `rps_doc_frac_no_alph_words` | 0.1529 | 2.6389 | **0.3492** |
| G3_多样信息量 | `rps_doc_unigram_entropy`, `rps_doc_frac_unique_words`, `rps_doc_frac_chars_top_2gram`, `rps_doc_frac_chars_top_3gram` | 0.0948 | 2.8219 | **0.2314** |
| G4_体量与格式 | `rps_doc_word_count`, `rps_doc_num_sentences`, `dsir_books`, `rps_lines_uppercase_letter_fraction`, `rps_lines_numerical_chars_fraction`, `rps_doc_mean_word_length` | 0.1043 | 2.0896 | **0.1886** |

### 4.2 PCA 特征值（检验"质量"是否为多维结构）

| 主成分 | 特征值 | 方差贡献率 | 累计贡献率 |
|---|---|---|---|
| PC1 | 0.3462 | 0.3347 | 0.3347 |
| PC2 | 0.2459 | 0.2378 | 0.5725 |
| PC3 | 0.0746 | 0.0721 | 0.6446 |
| PC4 | 0.0561 | 0.0542 | 0.6988 |
| PC5 | 0.0444 | 0.0429 | 0.7417 |
| PC6 | 0.0384 | 0.0371 | 0.7788 |
| PC7 | 0.0359 | 0.0347 | 0.8135 |
| PC8 | 0.0348 | 0.0336 | 0.8472 |

前 4 个主成分累计解释 69.88% 的方差，说明 22 维质量信号确实呈多维结构，采用"分维度合成"优于直接 22 维加权平均。


## 5. 三种合成方案的秩相关一致性

| 方案对 | Spearman ρ | Pearson r |
|---|---|---|
| Q_CRITIC vs Q_HIER | 0.9729 | 0.9803 |
| Q_CRITIC vs Q_EQ | 0.9765 | 0.9841 |
| Q_HIER vs Q_EQ | 0.9414 | 0.9595 |

外部效度（以广泛使用的 FineWeb-Edu 教育价值分作为外部准则）:

- ρ(Q_CRITIC, fineweb_edu) = 0.5957
- ρ(Q_HIER, fineweb_edu)   = 0.5646
- ρ(Q_EQ, fineweb_edu)     = 0.5948

- 已保存 `_out/01_q1/quality_scored.parquet`（含 Q_critic / Q_hier / Q_eq 与四个维度分）


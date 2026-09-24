- A1_sample 已读取 50000 条 ...

## A1_sample
- 抽取记录数 **51,230**
- 域分布: {"c4": 10000, "github": 10000, "stackexchange": 10000, "wikipedia": 10000, "commoncrawl": 9640, "arxiv": 1419, "book": 171}

- 指标缺失: `modernbert_reasoning`=13(0.025%); `modernbert_professionalism`=5(0.010%)

## A2_arxiv
- 抽取记录数 **17,523**
- 域分布: {"arxiv": 17523}

- 指标缺失: 无
- A3_github 已读取 50000 条 ...
- A3_github 已读取 100000 条 ...
- A3_github 已读取 150000 条 ...
- A3_github 已读取 200000 条 ...

## A3_github
- 抽取记录数 **203,752**
- 域分布: {"github": 203752}

- 指标缺失: `modernbert_professionalism`=1(0.000%)

## 合并矩阵
- 总记录数 **272,505**

- 列: fineweb_edu, fluency_en, modernbert_cleanliness, modernbert_readability, modernbert_reasoning, modernbert_professionalism, qurater, ad_en, dsir_books, dsir_math, dsir_wiki, rps_doc_word_count, rps_doc_num_sentences, rps_doc_unigram_entropy, rps_doc_frac_unique_words, rps_doc_frac_no_alph_words, rps_doc_frac_chars_top_2gram, rps_doc_frac_chars_top_3gram, rps_lines_uppercase_letter_fraction, rps_lines_ending_with_terminal_punctution_mark, rps_lines_numerical_chars_fraction, rps_doc_mean_word_length, _id, _source_file, _domain, n_chars, n_words, frac_alpha, frac_digit, frac_upper, frac_space, url_per1k, latex_per1k, adkw_per1k, code_per1k, dup_line_frac, _content_len


### 全量 22 指标缺失与取值概览

| 指标 | 缺失数 | 缺失率 | min | p1 | p25 | p50 | p75 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| `fineweb_edu` | 0 | 0.000% | 0 | 0.0298 | 0.9045 | 1.181 | 1.498 | 3.038 | 4.694 |
| `fluency_en` | 0 | 0.000% | 0.001976 | 0.03782 | 0.3483 | 0.6542 | 0.9065 | 0.9989 | 0.9995 |
| `modernbert_cleanliness` | 0 | 0.000% | 0.2 | 0.2 | 0.3714 | 0.4014 | 0.7311 | 1 | 1 |
| `modernbert_readability` | 0 | 0.000% | 0.006056 | 0.1593 | 0.4602 | 0.633 | 0.8153 | 0.9958 | 0.9999 |
| `modernbert_reasoning` | 13 | 0.005% | 1.078e-05 | 0.0004604 | 0.1161 | 0.2188 | 0.3733 | 1 | 1 |
| `modernbert_professionalism` | 6 | 0.002% | 0.2 | 0.202 | 0.7547 | 0.8557 | 0.9977 | 1 | 1 |
| `qurater` | 0 | 0.000% | 0.000196 | 0.1562 | 0.3733 | 0.4472 | 0.607 | 0.9831 | 0.9999 |
| `ad_en` | 0 | 0.000% | 0.0007932 | 0.3966 | 0.9814 | 0.9952 | 0.9984 | 0.9996 | 0.9998 |
| `dsir_books` | 0 | 0.000% | -2.026e+06 | -1.027e+05 | -5702 | -2395 | -1075 | -121 | -16.82 |
| `dsir_math` | 0 | 0.000% | -1.754e+06 | -7.178e+04 | -5392 | -2302 | -1034 | -116.5 | -15.06 |
| `dsir_wiki` | 0 | 0.000% | -1.966e+06 | -1.027e+05 | -5737 | -2410 | -1080 | -120.8 | -17.66 |
| `rps_doc_word_count` | 0 | 0.000% | 1 | 4 | 53 | 152 | 446 | 1.2e+04 | 5.527e+05 |
| `rps_doc_num_sentences` | 0 | 0.000% | 1 | 1 | 8 | 24 | 74 | 1012 | 3.045e+04 |
| `rps_doc_unigram_entropy` | 0 | 0.000% | 0 | 1.386 | 3.395 | 4.134 | 4.84 | 6.296 | 7.847 |
| `rps_doc_frac_unique_words` | 0 | 0.000% | 1.217 | 14.98 | 42.61 | 56.9 | 71.43 | 100 | 100 |
| `rps_doc_frac_no_alph_words` | 0 | 0.000% | 4.348 | 17.05 | 26.51 | 33.16 | 39.57 | 59.64 | 100 |
| `rps_doc_frac_chars_top_2gram` | 0 | 0.000% | 0 | 0 | 0.8251 | 2 | 5 | 24 | 194 |
| `rps_doc_frac_chars_top_3gram` | 0 | 0.000% | 0 | 0 | 0 | 1 | 4 | 24 | 284 |
| `rps_lines_uppercase_letter_fraction` | 0 | 0.000% | 0 | 1.363 | 13.54 | 20.77 | 29.14 | 55.34 | 96.49 |
| `rps_lines_ending_with_terminal_punctution_mark` | 0 | 0.000% | 0 | 0 | 16.95 | 25 | 38.38 | 100 | 100 |
| `rps_lines_numerical_chars_fraction` | 0 | 0.000% | 0 | 0 | 0.08692 | 0.8167 | 2.347 | 13.77 | 85.68 |
| `rps_doc_mean_word_length` | 0 | 0.000% | 1.927 | 4.233 | 6.033 | 8.057 | 10.53 | 22.28 | 87 |

### 样本集与扩展集的 id 重叠

- `A1_sample` ∩ `A2_arxiv` = **1,419** 条（占 A1_sample 的 2.77%）
- `A1_sample` ∩ `A3_github` = **10,000** 条（占 A1_sample 的 19.52%）
- 结论：抽样集 A1 的 arxiv/github 记录是扩展集 A2/A3 的**子集**（同源抽样），因此二者可直接对照；非 arxiv/github 的 5 个域仅在 A1 中出现。

- 已保存: `_out/01_q1/quality_matrix_raw.parquet` (272,505 行)

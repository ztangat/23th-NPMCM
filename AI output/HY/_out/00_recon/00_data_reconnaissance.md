# 阶段0 数据勘察报告

工作目录: `C:\Users\dkyyt\Desktop\F题`


## 1. 目录结构与体积

| 层级 | 目录 | 子目录数 | 文件数 | 文件总体积(bytes) |
|---|---|---|---|---|
| 0 | `.\real_attachments` | 3 | 2 | 287,746 |
| 1 | `.\real_attachments\A_data_value` | 4 | 2 | 2,561 |
| 2 | `.\real_attachments\A_data_value\regmix_domain_sample.jsonl` | 0 | 1 | 714,013,556 |
| 2 | `.\real_attachments\A_data_value\regmix_tables` | 0 | 12 | 380,578 |
| 2 | `.\real_attachments\A_data_value\slimpajama_quality_extended` | 2 | 0 | 0 |
| 2 | `.\real_attachments\A_data_value\slimpajama_quality_signal_sample.jsonl` | 0 | 1 | 433,080,377 |
| 1 | `.\real_attachments\B_scaling_laws` | 1 | 11 | 454,399 |
| 2 | `.\real_attachments\B_scaling_laws\training_trajectories` | 0 | 8 | 112,320 |
| 1 | `.\real_attachments\C_efficiency_evolution` | 9 | 7 | 9,579,721 |
| 2 | `.\real_attachments\C_efficiency_evolution\data` | 0 | 1 | 1,109,997 |
| 2 | `.\real_attachments\C_efficiency_evolution\detailed_results` | 1863 | 0 | 0 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_1.4b_eval_details` | 0 | 1 | 63,450 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_12b_eval_details` | 0 | 1 | 72,327 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_160m_eval_details` | 0 | 1 | 72,028 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_2.8b_eval_details` | 0 | 1 | 72,145 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_410m_eval_details` | 0 | 1 | 72,065 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_6.9b_eval_details` | 0 | 1 | 72,470 |
| 2 | `.\real_attachments\C_efficiency_evolution\pythia_eval_details` | 0 | 1 | 63,185 |

## 2. CSV 文件勘察


### `A_data_value/domain_mapping_guide.csv`

- 形状: **17 行 × 4 列**  文件大小 1,296 bytes
- 列名: `mixture_domain`, `quality_domain`, `mapping_type`, `note`
- 缺失: 无

对象列取值样例:
  - `mixture_domain`: 17 个不同取值; top5 = arxiv(1), github(1), stackexchange(1), wikipedia_en(1), gutenberg_pg_19(1)
  - `quality_domain`: 7 个不同取值; top5 = (none)(11), arxiv(1), github(1), stackexchange(1), wikipedia(1)
  - `mapping_type`: 3 个不同取值; top5 = inferred(11), direct(3), near_direct(3)
  - `note`: 5 个不同取值; top5 = 无同名质量域，需自行建立关联(11), 配方域名与质量信号域名一致(3), 名称略有差异，语义对应(1), 书籍类文本，可参考 book 域质量信号(1), 网页抓取类文本，可参考 commoncrawl 域质量信号(1)

### `A_data_value/regmix_domain_summary.csv`

- 形状: **17 行 × 5 列**  文件大小 1,265 bytes
- 列名: `domain`, `source_path`, `sample_rows`, `sample_bytes_requested`, `avg_text_chars`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `sample_rows` | 17 | 8119.65 | 10419.9 | 16 | 1922 | 30378 |
| `sample_bytes_requested` | 17 | 8.38861e+07 | 0 | 8.38861e+07 | 8.38861e+07 | 8.38861e+07 |
| `avg_text_chars` | 17 | 50767.8 | 101215 | 1330.45 | 5180.6 | 381180 |

对象列取值样例:
  - `domain`: 17 个不同取值; top5 = arxiv(1), dm_mathematics(1), enron_emails(1), europarl(1), freelaw(1)
  - `source_path`: 17 个不同取值; top5 = valid/arxiv-0-93015664.jsonl(1), valid/dm_mathematics-0-34933789.jsonl(1), valid/enron_emails-0-74209619.jsonl(1), valid/europarl-0-10512041.jsonl(1), valid/freelaw-0-47906234.jsonl(1)

### `A_data_value/regmix_tables/est_mixture_10b.csv`

- 形状: **63 行 × 18 列**  文件大小 6,113 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 63 | 32 | 18.3303 | 1 | 32 | 63 |
| `train_the_pile_arxiv` | 63 | 0.123841 | 0.223277 | 0 | 0.008 | 0.904 |
| `train_the_pile_freelaw` | 63 | 0.131302 | 0.238517 | 0 | 0.009 | 0.944 |
| `train_the_pile_nih_exporter` | 63 | 0.008587 | 0.014008 | 0 | 0 | 0.05 |
| `train_the_pile_pubmed_central` | 63 | 0.121254 | 0.198424 | 0 | 0.007 | 0.828 |
| `train_the_pile_wikipedia_en` | 63 | 0.067667 | 0.130443 | 0 | 0.003 | 0.59 |
| `train_the_pile_dm_mathematics` | 63 | 0.01619 | 0.042903 | 0 | 0 | 0.222 |
| `train_the_pile_github` | 63 | 0.080222 | 0.175034 | 0 | 0.007 | 0.783 |
| `train_the_pile_philpapers` | 63 | 0.004016 | 0.008605 | 0 | 0 | 0.055 |
| `train_the_pile_stackexchange` | 63 | 0.092365 | 0.143607 | 0 | 0.013 | 0.552 |
| `train_the_pile_enron_emails` | 63 | 0.001063 | 0.003379 | 0 | 0 | 0.019 |
| `train_the_pile_gutenberg_pg_19` | 63 | 0.063143 | 0.103164 | 0 | 0.009 | 0.4 |
| `train_the_pile_pile_cc` | 63 | 0.116238 | 0.222014 | 0 | 0.006 | 0.86 |
| `train_the_pile_ubuntu_irc` | 63 | 0.012873 | 0.026621 | 0 | 0 | 0.148 |
| `train_the_pile_europarl` | 63 | 0.013079 | 0.026539 | 0 | 0.001 | 0.114 |
| `train_the_pile_hackernews` | 63 | 0.014365 | 0.030633 | 0 | 0 | 0.119 |
| `train_the_pile_pubmed_abstracts` | 63 | 0.05273 | 0.100263 | 0 | 0.003 | 0.546 |
| `train_the_pile_uspto_backgrounds` | 63 | 0.080937 | 0.135623 | 0 | 0.008 | 0.673 |

### `A_data_value/regmix_tables/est_mixture_70b.csv`

- 形状: **63 行 × 18 列**  文件大小 6,113 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 63 | 32 | 18.3303 | 1 | 32 | 63 |
| `train_the_pile_arxiv` | 63 | 0.123841 | 0.223277 | 0 | 0.008 | 0.904 |
| `train_the_pile_freelaw` | 63 | 0.131302 | 0.238517 | 0 | 0.009 | 0.944 |
| `train_the_pile_nih_exporter` | 63 | 0.008587 | 0.014008 | 0 | 0 | 0.05 |
| `train_the_pile_pubmed_central` | 63 | 0.121254 | 0.198424 | 0 | 0.007 | 0.828 |
| `train_the_pile_wikipedia_en` | 63 | 0.067667 | 0.130443 | 0 | 0.003 | 0.59 |
| `train_the_pile_dm_mathematics` | 63 | 0.01619 | 0.042903 | 0 | 0 | 0.222 |
| `train_the_pile_github` | 63 | 0.080222 | 0.175034 | 0 | 0.007 | 0.783 |
| `train_the_pile_philpapers` | 63 | 0.004016 | 0.008605 | 0 | 0 | 0.055 |
| `train_the_pile_stackexchange` | 63 | 0.092365 | 0.143607 | 0 | 0.013 | 0.552 |
| `train_the_pile_enron_emails` | 63 | 0.001063 | 0.003379 | 0 | 0 | 0.019 |
| `train_the_pile_gutenberg_pg_19` | 63 | 0.063143 | 0.103164 | 0 | 0.009 | 0.4 |
| `train_the_pile_pile_cc` | 63 | 0.116238 | 0.222014 | 0 | 0.006 | 0.86 |
| `train_the_pile_ubuntu_irc` | 63 | 0.012873 | 0.026621 | 0 | 0 | 0.148 |
| `train_the_pile_europarl` | 63 | 0.013079 | 0.026539 | 0 | 0.001 | 0.114 |
| `train_the_pile_hackernews` | 63 | 0.014365 | 0.030633 | 0 | 0 | 0.119 |
| `train_the_pile_pubmed_abstracts` | 63 | 0.05273 | 0.100263 | 0 | 0.003 | 0.546 |
| `train_the_pile_uspto_backgrounds` | 63 | 0.080937 | 0.135623 | 0 | 0.008 | 0.673 |

### `A_data_value/regmix_tables/est_pile_loss_10b.csv`

- 形状: **63 行 × 14 列**  文件大小 6,467 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 63 | 32 | 18.3303 | 1 | 32 | 63 |
| `metric/the_pile_arxiv_val_loss` | 63 | 1.6989 | 0.30822 | 1.0486 | 1.6872 | 2.3615 |
| `metric/the_pile_freelaw_val_loss` | 63 | 1.74564 | 0.310529 | 0.9749 | 1.7367 | 2.4423 |
| `metric/the_pile_pubmed_central_val_loss` | 63 | 1.72454 | 0.251945 | 1.0114 | 1.7411 | 2.1906 |
| `metric/the_pile_wikipedia_en_val_loss` | 63 | 1.94799 | 0.232262 | 1.3918 | 1.9293 | 2.5116 |
| `metric/the_pile_dm_mathematics_val_loss` | 63 | 1.34691 | 0.426823 | 0.6258 | 1.4993 | 1.9448 |
| `metric/the_pile_github_val_loss` | 63 | 1.78786 | 0.274899 | 1.1153 | 1.7985 | 2.2485 |
| `metric/the_pile_stackexchange_val_loss` | 63 | 1.75641 | 0.190592 | 1.3223 | 1.793 | 2.0749 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 63 | 1.80729 | 0.215572 | 1.3043 | 1.8293 | 2.2052 |
| `metric/the_pile_pile_cc_val_loss` | 63 | 1.9299 | 0.211936 | 1.4675 | 1.9455 | 2.4014 |
| `metric/the_pile_ubuntu_irc_val_loss` | 63 | 2.25492 | 0.312352 | 1.6397 | 2.2451 | 2.9153 |
| `metric/the_pile_hackernews_val_loss` | 63 | 1.82216 | 0.173056 | 1.4418 | 1.8149 | 2.2718 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 63 | 1.81945 | 0.216561 | 1.2101 | 1.8226 | 2.2223 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 63 | 1.72501 | 0.22586 | 1.1278 | 1.733 | 2.1819 |

### `A_data_value/regmix_tables/est_pile_loss_70b.csv`

- 形状: **63 行 × 14 列**  文件大小 6,467 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 63 | 32 | 18.3303 | 1 | 32 | 63 |
| `metric/the_pile_arxiv_val_loss` | 63 | 1.33547 | 0.241198 | 0.8006 | 1.3279 | 1.7909 |
| `metric/the_pile_freelaw_val_loss` | 63 | 1.37555 | 0.262296 | 0.7261 | 1.3691 | 1.9874 |
| `metric/the_pile_pubmed_central_val_loss` | 63 | 1.35674 | 0.20583 | 0.7569 | 1.3716 | 1.7529 |
| `metric/the_pile_wikipedia_en_val_loss` | 63 | 1.53443 | 0.209974 | 1.0555 | 1.5357 | 2.0438 |
| `metric/the_pile_dm_mathematics_val_loss` | 63 | 1.05512 | 0.325749 | 0.4964 | 1.122 | 1.5436 |
| `metric/the_pile_github_val_loss` | 63 | 1.40569 | 0.218623 | 0.8524 | 1.43 | 1.8061 |
| `metric/the_pile_stackexchange_val_loss` | 63 | 1.38114 | 0.156417 | 1.0106 | 1.3731 | 1.6667 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 63 | 1.42368 | 0.193861 | 0.981 | 1.4283 | 1.7654 |
| `metric/the_pile_pile_cc_val_loss` | 63 | 1.52048 | 0.197076 | 1.0931 | 1.5365 | 1.9606 |
| `metric/the_pile_ubuntu_irc_val_loss` | 63 | 1.77386 | 0.258528 | 1.2435 | 1.7133 | 2.353 |
| `metric/the_pile_hackernews_val_loss` | 63 | 1.43486 | 0.162581 | 1.0934 | 1.4448 | 1.8547 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 63 | 1.43311 | 0.193814 | 0.9056 | 1.4365 | 1.7891 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 63 | 1.35896 | 0.199656 | 0.8482 | 1.3719 | 1.7756 |

### `A_data_value/regmix_tables/test_mixture_1B.csv`

- 形状: **64 行 × 18 列**  文件大小 6,393 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 64 | 31.5 | 18.619 | 0 | 31.5 | 63 |
| `train_the_pile_arxiv` | 64 | 0.143656 | 0.114893 | 0 | 0.107 | 0.501 |
| `train_the_pile_freelaw` | 64 | 0.070703 | 0.062957 | 0.002 | 0.052 | 0.276 |
| `train_the_pile_nih_exporter` | 64 | 0.006359 | 0.03211 | 0 | 0 | 0.253 |
| `train_the_pile_pubmed_central` | 64 | 0.173516 | 0.101247 | 0.039 | 0.16 | 0.501 |
| `train_the_pile_wikipedia_en` | 64 | 0.052828 | 0.061878 | 0 | 0.027 | 0.287 |
| `train_the_pile_dm_mathematics` | 64 | 0.019016 | 0.033451 | 0 | 0.001 | 0.174 |
| `train_the_pile_github` | 64 | 0.097844 | 0.09051 | 0.006 | 0.07 | 0.482 |
| `train_the_pile_philpapers` | 64 | 0.004234 | 0.012616 | 0 | 0 | 0.069 |
| `train_the_pile_stackexchange` | 64 | 0.071375 | 0.073969 | 0 | 0.056 | 0.408 |
| `train_the_pile_enron_emails` | 64 | 0.000156 | 0.00113 | 0 | 0 | 0.009 |
| `train_the_pile_gutenberg_pg_19` | 64 | 0.029 | 0.059682 | 0 | 0.002 | 0.265 |
| `train_the_pile_pile_cc` | 64 | 0.227734 | 0.132132 | 0.006 | 0.2135 | 0.618 |
| `train_the_pile_ubuntu_irc` | 64 | 0.017406 | 0.047143 | 0 | 0 | 0.296 |
| `train_the_pile_europarl` | 64 | 0.006984 | 0.022836 | 0 | 0 | 0.117 |
| `train_the_pile_hackernews` | 64 | 0.002891 | 0.007035 | 0 | 0 | 0.034 |
| `train_the_pile_pubmed_abstracts` | 64 | 0.030859 | 0.05602 | 0 | 0.0105 | 0.351 |
| `train_the_pile_uspto_backgrounds` | 64 | 0.045156 | 0.060684 | 0 | 0.0245 | 0.307 |

### `A_data_value/regmix_tables/test_mixture_1m.csv`

- 形状: **256 行 × 18 列**  文件大小 23,537 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 256 | 128.5 | 74.045 | 1 | 128.5 | 256 |
| `train_the_pile_arxiv` | 256 | 0.102352 | 0.186883 | 0 | 0.012 | 0.961 |
| `train_the_pile_freelaw` | 256 | 0.074941 | 0.158863 | 0 | 0.006 | 0.955 |
| `train_the_pile_nih_exporter` | 256 | 0.005891 | 0.011779 | 0 | 0 | 0.055 |
| `train_the_pile_pubmed_central` | 256 | 0.123832 | 0.203494 | 0 | 0.012 | 0.901 |
| `train_the_pile_wikipedia_en` | 256 | 0.080242 | 0.162661 | 0 | 0.002 | 0.762 |
| `train_the_pile_dm_mathematics` | 256 | 0.029719 | 0.053454 | 0 | 0.001 | 0.239 |
| `train_the_pile_github` | 256 | 0.12441 | 0.211336 | 0 | 0.014 | 0.988 |
| `train_the_pile_philpapers` | 256 | 0.005258 | 0.010996 | 0 | 0 | 0.055 |
| `train_the_pile_stackexchange` | 256 | 0.100898 | 0.178934 | 0 | 0.01 | 0.9 |
| `train_the_pile_enron_emails` | 256 | 0.002355 | 0.005258 | 0 | 0 | 0.025 |
| `train_the_pile_gutenberg_pg_19` | 256 | 0.051117 | 0.085476 | 0 | 0.002 | 0.403 |
| `train_the_pile_pile_cc` | 256 | 0.10998 | 0.201001 | 0 | 0.01 | 0.956 |
| `train_the_pile_ubuntu_irc` | 256 | 0.020348 | 0.039979 | 0 | 0.001 | 0.177 |
| `train_the_pile_europarl` | 256 | 0.014098 | 0.025808 | 0 | 0.001 | 0.117 |
| `train_the_pile_hackernews` | 256 | 0.011684 | 0.022308 | 0 | 0 | 0.116 |
| `train_the_pile_pubmed_abstracts` | 256 | 0.06798 | 0.124139 | 0 | 0.002 | 0.561 |
| `train_the_pile_uspto_backgrounds` | 256 | 0.074855 | 0.135815 | 0 | 0.005 | 0.695 |

### `A_data_value/regmix_tables/test_mixture_60m.csv`

- 形状: **256 行 × 18 列**  文件大小 23,537 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 256 | 128.5 | 74.045 | 1 | 128.5 | 256 |
| `train_the_pile_arxiv` | 256 | 0.102352 | 0.186883 | 0 | 0.012 | 0.961 |
| `train_the_pile_freelaw` | 256 | 0.074941 | 0.158863 | 0 | 0.006 | 0.955 |
| `train_the_pile_nih_exporter` | 256 | 0.005891 | 0.011779 | 0 | 0 | 0.055 |
| `train_the_pile_pubmed_central` | 256 | 0.123832 | 0.203494 | 0 | 0.012 | 0.901 |
| `train_the_pile_wikipedia_en` | 256 | 0.080242 | 0.162661 | 0 | 0.002 | 0.762 |
| `train_the_pile_dm_mathematics` | 256 | 0.029719 | 0.053454 | 0 | 0.001 | 0.239 |
| `train_the_pile_github` | 256 | 0.12441 | 0.211336 | 0 | 0.014 | 0.988 |
| `train_the_pile_philpapers` | 256 | 0.005258 | 0.010996 | 0 | 0 | 0.055 |
| `train_the_pile_stackexchange` | 256 | 0.100898 | 0.178934 | 0 | 0.01 | 0.9 |
| `train_the_pile_enron_emails` | 256 | 0.002355 | 0.005258 | 0 | 0 | 0.025 |
| `train_the_pile_gutenberg_pg_19` | 256 | 0.051117 | 0.085476 | 0 | 0.002 | 0.403 |
| `train_the_pile_pile_cc` | 256 | 0.10998 | 0.201001 | 0 | 0.01 | 0.956 |
| `train_the_pile_ubuntu_irc` | 256 | 0.020348 | 0.039979 | 0 | 0.001 | 0.177 |
| `train_the_pile_europarl` | 256 | 0.014098 | 0.025808 | 0 | 0.001 | 0.117 |
| `train_the_pile_hackernews` | 256 | 0.011684 | 0.022308 | 0 | 0 | 0.116 |
| `train_the_pile_pubmed_abstracts` | 256 | 0.06798 | 0.124139 | 0 | 0.002 | 0.561 |
| `train_the_pile_uspto_backgrounds` | 256 | 0.074855 | 0.135815 | 0 | 0.005 | 0.695 |

### `A_data_value/regmix_tables/test_pile_loss_1B.csv`

- 形状: **64 行 × 14 列**  文件大小 10,639 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 64 | 31.5 | 18.619 | 0 | 31.5 | 63 |
| `metric/the_pile_arxiv_val_loss` | 64 | 1.79439 | 0.105746 | 1.61814 | 1.77936 | 2.27112 |
| `metric/the_pile_freelaw_val_loss` | 64 | 1.98232 | 0.139643 | 1.74451 | 1.95904 | 2.37832 |
| `metric/the_pile_pubmed_central_val_loss` | 64 | 1.76114 | 0.052817 | 1.6468 | 1.75805 | 1.86556 |
| `metric/the_pile_wikipedia_en_val_loss` | 64 | 2.50874 | 0.15947 | 2.23587 | 2.47135 | 2.94439 |
| `metric/the_pile_dm_mathematics_val_loss` | 64 | 1.5509 | 0.29962 | 1.18099 | 1.45952 | 2.18295 |
| `metric/the_pile_github_val_loss` | 64 | 1.05553 | 0.12098 | 0.831067 | 1.03829 | 1.39828 |
| `metric/the_pile_stackexchange_val_loss` | 64 | 1.85619 | 0.130215 | 1.62764 | 1.82617 | 2.14996 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 64 | 2.81289 | 0.243901 | 2.29944 | 2.86536 | 3.44735 |
| `metric/the_pile_pile_cc_val_loss` | 64 | 2.98311 | 0.101561 | 2.81712 | 2.96662 | 3.34033 |
| `metric/the_pile_ubuntu_irc_val_loss` | 64 | 2.64322 | 0.426159 | 1.76512 | 2.73064 | 3.22581 |
| `metric/the_pile_hackernews_val_loss` | 64 | 3.14909 | 0.198438 | 2.67541 | 3.23729 | 3.45748 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 64 | 2.46184 | 0.113984 | 2.18347 | 2.47638 | 2.66312 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 64 | 2.37306 | 0.141571 | 2.05079 | 2.34728 | 2.6628 |

### `A_data_value/regmix_tables/test_pile_loss_1m.csv`

- 形状: **256 行 × 14 列**  文件大小 61,175 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 256 | 128.5 | 74.045 | 1 | 128.5 | 256 |
| `metric/the_pile_arxiv_val_loss` | 256 | 4.87895 | 0.781106 | 3.63838 | 4.82008 | 7.14247 |
| `metric/the_pile_freelaw_val_loss` | 256 | 5.31598 | 0.738443 | 3.8608 | 5.38445 | 6.86255 |
| `metric/the_pile_pubmed_central_val_loss` | 256 | 5.08813 | 0.837817 | 3.86257 | 5.04207 | 6.9573 |
| `metric/the_pile_wikipedia_en_val_loss` | 256 | 5.81566 | 0.547442 | 4.72303 | 5.80595 | 7.167 |
| `metric/the_pile_dm_mathematics_val_loss` | 256 | 3.75906 | 1.5364 | 1.83446 | 3.59864 | 6.26415 |
| `metric/the_pile_github_val_loss` | 256 | 5.07373 | 0.923088 | 3.69359 | 4.88082 | 7.50993 |
| `metric/the_pile_stackexchange_val_loss` | 256 | 5.11964 | 0.665346 | 3.9877 | 5.03301 | 6.95217 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 256 | 5.42923 | 0.449531 | 4.66402 | 5.36813 | 6.72441 |
| `metric/the_pile_pile_cc_val_loss` | 256 | 5.7337 | 0.321124 | 5.12717 | 5.75656 | 6.68006 |
| `metric/the_pile_ubuntu_irc_val_loss` | 256 | 6.48999 | 0.99937 | 4.69872 | 6.86716 | 8.19113 |
| `metric/the_pile_hackernews_val_loss` | 256 | 5.39041 | 0.336951 | 4.69831 | 5.38991 | 6.30706 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 256 | 5.41484 | 0.571426 | 4.44542 | 5.3648 | 7.11874 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 256 | 5.16036 | 0.511084 | 4.1189 | 5.27171 | 6.43317 |

### `A_data_value/regmix_tables/test_pile_loss_60m.csv`

- 形状: **256 行 × 14 列**  文件大小 61,827 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 256 | 128.5 | 74.045 | 1 | 128.5 | 256 |
| `metric/the_pile_arxiv_val_loss` | 256 | 3.28535 | 0.571019 | 2.42863 | 3.22571 | 5.25372 |
| `metric/the_pile_freelaw_val_loss` | 256 | 3.97294 | 0.645432 | 2.74088 | 4.01096 | 5.40812 |
| `metric/the_pile_pubmed_central_val_loss` | 256 | 3.4243 | 0.569907 | 2.59455 | 3.37071 | 5.10087 |
| `metric/the_pile_wikipedia_en_val_loss` | 256 | 4.46335 | 0.501681 | 3.49935 | 4.4601 | 5.70851 |
| `metric/the_pile_dm_mathematics_val_loss` | 256 | 2.45802 | 0.769755 | 1.49795 | 2.27115 | 4.7007 |
| `metric/the_pile_github_val_loss` | 256 | 2.63412 | 0.687512 | 1.67279 | 2.44862 | 5.61734 |
| `metric/the_pile_stackexchange_val_loss` | 256 | 3.49648 | 0.5574 | 2.62021 | 3.41483 | 5.57184 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 256 | 4.46037 | 0.385637 | 3.78037 | 4.42701 | 5.58006 |
| `metric/the_pile_pile_cc_val_loss` | 256 | 4.66879 | 0.314442 | 4.10011 | 4.68933 | 5.61656 |
| `metric/the_pile_ubuntu_irc_val_loss` | 256 | 4.56353 | 0.703991 | 3.29724 | 4.77173 | 6.6632 |
| `metric/the_pile_hackernews_val_loss` | 256 | 4.41407 | 0.308637 | 3.79759 | 4.41224 | 5.30389 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 256 | 4.07257 | 0.548085 | 3.19796 | 3.99443 | 5.78925 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 256 | 3.78394 | 0.433108 | 2.94383 | 3.85669 | 5.01165 |

### `A_data_value/regmix_tables/train_mixture_1m.csv`

- 形状: **512 行 × 18 列**  文件大小 46,412 bytes
- 列名: `index`, `train_the_pile_arxiv`, `train_the_pile_freelaw`, `train_the_pile_nih_exporter`, `train_the_pile_pubmed_central`, `train_the_pile_wikipedia_en`, `train_the_pile_dm_mathematics`, `train_the_pile_github`, `train_the_pile_philpapers`, `train_the_pile_stackexchange`, `train_the_pile_enron_emails`, `train_the_pile_gutenberg_pg_19`, `train_the_pile_pile_cc`, `train_the_pile_ubuntu_irc`, `train_the_pile_europarl`, `train_the_pile_hackernews`, `train_the_pile_pubmed_abstracts`, `train_the_pile_uspto_backgrounds`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 512 | 256.5 | 147.946 | 1 | 256.5 | 512 |
| `train_the_pile_arxiv` | 512 | 0.113004 | 0.200918 | 0 | 0.006 | 0.904 |
| `train_the_pile_freelaw` | 512 | 0.096809 | 0.188638 | 0 | 0.005 | 0.993 |
| `train_the_pile_nih_exporter` | 512 | 0.005889 | 0.011985 | 0 | 0 | 0.058 |
| `train_the_pile_pubmed_central` | 512 | 0.113729 | 0.205466 | 0 | 0.012 | 0.999 |
| `train_the_pile_wikipedia_en` | 512 | 0.075078 | 0.146487 | 0 | 0.002 | 0.708 |
| `train_the_pile_dm_mathematics` | 512 | 0.025461 | 0.051505 | 0 | 0.001 | 0.235 |
| `train_the_pile_github` | 512 | 0.110689 | 0.209058 | 0 | 0.006 | 0.991 |
| `train_the_pile_philpapers` | 512 | 0.005486 | 0.011391 | 0 | 0 | 0.055 |
| `train_the_pile_stackexchange` | 512 | 0.099424 | 0.188104 | 0 | 0.005 | 0.998 |
| `train_the_pile_enron_emails` | 512 | 0.002211 | 0.005077 | 0 | 0 | 0.026 |
| `train_the_pile_gutenberg_pg_19` | 512 | 0.046912 | 0.087038 | 0 | 0.002 | 0.404 |
| `train_the_pile_pile_cc` | 512 | 0.119482 | 0.21297 | 0 | 0.011 | 0.995 |
| `train_the_pile_ubuntu_irc` | 512 | 0.0196 | 0.038898 | 0 | 0.001 | 0.176 |
| `train_the_pile_europarl` | 512 | 0.012736 | 0.024476 | 0 | 0 | 0.117 |
| `train_the_pile_hackernews` | 512 | 0.011648 | 0.025346 | 0 | 0 | 0.12 |
| `train_the_pile_pubmed_abstracts` | 512 | 0.066174 | 0.126362 | 0 | 0.002 | 0.576 |
| `train_the_pile_uspto_backgrounds` | 512 | 0.075549 | 0.135083 | 0 | 0.003 | 0.678 |

### `A_data_value/regmix_tables/train_pile_loss_1m.csv`

- 形状: **512 行 × 14 列**  文件大小 121,898 bytes
- 列名: `index`, `metric/the_pile_arxiv_val_loss`, `metric/the_pile_freelaw_val_loss`, `metric/the_pile_pubmed_central_val_loss`, `metric/the_pile_wikipedia_en_val_loss`, `metric/the_pile_dm_mathematics_val_loss`, `metric/the_pile_github_val_loss`, `metric/the_pile_stackexchange_val_loss`, `metric/the_pile_gutenberg_pg_19_val_loss`, `metric/the_pile_pile_cc_val_loss`, `metric/the_pile_ubuntu_irc_val_loss`, `metric/the_pile_hackernews_val_loss`, `metric/the_pile_pubmed_abstracts_val_loss`, `metric/the_pile_uspto_backgrounds_val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `index` | 512 | 256.5 | 147.946 | 1 | 256.5 | 512 |
| `metric/the_pile_arxiv_val_loss` | 512 | 5.04406 | 1.01493 | 3.63811 | 4.94545 | 8.60601 |
| `metric/the_pile_freelaw_val_loss` | 512 | 5.28257 | 0.784427 | 3.85284 | 5.32755 | 6.9868 |
| `metric/the_pile_pubmed_central_val_loss` | 512 | 5.17593 | 0.85307 | 3.83993 | 5.02284 | 7.71324 |
| `metric/the_pile_wikipedia_en_val_loss` | 512 | 5.8109 | 0.529394 | 4.75348 | 5.81216 | 7.34504 |
| `metric/the_pile_dm_mathematics_val_loss` | 512 | 3.89358 | 1.53484 | 1.80644 | 3.84645 | 6.51411 |
| `metric/the_pile_github_val_loss` | 512 | 5.27952 | 1.05675 | 3.68792 | 5.08732 | 8.63397 |
| `metric/the_pile_stackexchange_val_loss` | 512 | 5.24585 | 0.776327 | 3.94625 | 5.07223 | 7.55932 |
| `metric/the_pile_gutenberg_pg_19_val_loss` | 512 | 5.44576 | 0.435517 | 4.63933 | 5.40067 | 6.68374 |
| `metric/the_pile_pile_cc_val_loss` | 512 | 5.72779 | 0.320739 | 5.08213 | 5.73154 | 6.64493 |
| `metric/the_pile_ubuntu_irc_val_loss` | 512 | 6.55349 | 1.02864 | 4.75665 | 6.83647 | 8.71559 |
| `metric/the_pile_hackernews_val_loss` | 512 | 5.40372 | 0.342676 | 4.72536 | 5.3842 | 6.50586 |
| `metric/the_pile_pubmed_abstracts_val_loss` | 512 | 5.43956 | 0.585213 | 4.42264 | 5.41111 | 7.74306 |
| `metric/the_pile_uspto_backgrounds_val_loss` | 512 | 5.17781 | 0.53137 | 4.12255 | 5.34949 | 6.91101 |

### `B_scaling_laws/cerebras_training_log.csv`

- 形状: **1029 行 × 15 列**  文件大小 89,643 bytes
- 列名: `run_id`, `N_params_B`, `D_tokens_B`, `C_FLOPs_1e21`, `steps`, `batch_tokens_M`, `lr`, `wd`, `precision`, `gpu_days`, `step_time_ms`, `train_loss`, `val_loss`, `ppl`, `grad_norm_avg`
- 缺失: `gpu_days`=1029; `step_time_ms`=1029; `grad_norm_avg`=1029

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 1029 | 3.52243 | 4.41711 | 0.111 | 1.3 | 13 |
| `D_tokens_B` | 1029 | 1059.86 | 607.044 | 14 | 1061 | 2050 |
| `C_FLOPs_1e21` | 1029 | 22.3998 | 34.8155 | 0.009548 | 6.27042 | 159.9 |
| `steps` | 1029 | 74000 | 42454.9 | 1000 | 74000 | 147000 |
| `batch_tokens_M` | 1029 | 14.3 | 0 | 14.3 | 14.3 | 14.3 |
| `lr` | 1029 | 0.0006 | 0 | 0.0006 | 0.0006 | 0.0006 |
| `wd` | 1029 | 0.1 | 0 | 0.1 | 0.1 | 0.1 |
| `gpu_days` | 0 |  |  |  |  |  |
| `step_time_ms` | 0 |  |  |  |  |  |
| `train_loss` | 1029 | 3.46574 | 0.504714 | 2.8505 | 3.3481 | 6.504 |
| `val_loss` | 1029 | 3.44515 | 0.504682 | 2.8354 | 3.3305 | 6.4979 |
| `ppl` | 1029 | 38.2401 | 45.7064 | 17.04 | 27.95 | 663.77 |
| `grad_norm_avg` | 0 |  |  |  |  |  |

对象列取值样例:
  - `run_id`: 7 个不同取值; top5 = cerebras_0.111B(147), cerebras_0.256B(147), cerebras_0.59B(147), cerebras_1.3B(147), cerebras_2.7B(147)
  - `precision`: 1 个不同取值; top5 = bf16(1029)

### `B_scaling_laws/open_model_family_metadata.csv`

- 形状: **18 行 × 8 列**  文件大小 1,975 bytes
- 列名: `family`, `model_repo`, `config_bytes`, `readme_bytes`, `weight_file_count`, `weight_total_bytes`, `source_url`, `error`
- 缺失: `error`=18

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `config_bytes` | 18 | 603.333 | 291.344 | 359 | 569.5 | 1322 |
| `readme_bytes` | 18 | 13503.7 | 1686.66 | 10560 | 13569 | 17697 |
| `weight_file_count` | 18 | 2 | 1.23669 | 1 | 2 | 6 |
| `weight_total_bytes` | 18 | 1.45964e+10 | 1.88572e+10 | 5.63039e+07 | 5.61187e+09 | 5.51048e+10 |
| `error` | 0 |  |  |  |  |  |

对象列取值样例:
  - `family`: 3 个不同取值; top5 = pythia(9), cerebras_gpt(7), olmo(2)
  - `model_repo`: 18 个不同取值; top5 = EleutherAI/pythia-14m(1), EleutherAI/pythia-70m(1), EleutherAI/pythia-160m(1), EleutherAI/pythia-410m(1), EleutherAI/pythia-1b(1)
  - `source_url`: 18 个不同取值; top5 = https://huggingface.co/EleutherAI/pythia(1), https://huggingface.co/EleutherAI/pythia(1), https://huggingface.co/EleutherAI/pythia(1), https://huggingface.co/EleutherAI/pythia(1), https://huggingface.co/EleutherAI/pythia(1)

### `B_scaling_laws/published_scaling_data.csv`

- 形状: **44 行 × 6 列**  文件大小 1,967 bytes
- 列名: `family`, `N_params_B`, `D_tokens_B`, `val_loss`, `source`, `is_converged`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 44 | 44.8433 | 96.7723 | 0.125 | 7.05 | 540 |
| `D_tokens_B` | 44 | 516.795 | 526.863 | 5 | 300 | 2000 |
| `val_loss` | 44 | 2.20441 | 0.385067 | 1.473 | 2.189 | 3.015 |
| `is_converged` | 44 | 1 | 0 | 1 | 1 | 1 |

对象列取值样例:
  - `family`: 9 个不同取值; top5 = Chinchilla(9), GPT-3(8), Gopher(7), OPT(5), LLaMA(4)
  - `source`: 6 个不同取值; top5 = Hoffmann et al. 2022(16), Kaplan et al. 2020(9), Touvron et al. 2023(7), Zhang et al. 2022(5), Scao et al. 2022(4)

### `B_scaling_laws/pythia_checkpoint_index.csv`

- 形状: **1386 行 × 5 列**  文件大小 117,698 bytes
- 列名: `model_repo`, `model_size`, `step`, `branch`, `commit`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `step` | 1386 | 66863.8 | 43893.4 | 0 | 66500 | 143000 |

对象列取值样例:
  - `model_repo`: 9 个不同取值; top5 = EleutherAI/pythia-14m(154), EleutherAI/pythia-70m(154), EleutherAI/pythia-160m(154), EleutherAI/pythia-410m(154), EleutherAI/pythia-1b(154)
  - `model_size`: 9 个不同取值; top5 = 14m(154), 70m(154), 160m(154), 410m(154), 1b(154)
  - `branch`: 154 个不同取值; top5 = step142000(9), step141000(9), step140000(9), step139000(9), step138000(9)
  - `commit`: 1384 个不同取值; top5 = 2a259cdd96a4beb1cdf467512e3904197345f6a9(3), cca39af689139d049aedf10206e932638f19679a(1), 37d03d13381a010ede721945b18170e2f61b0f95(1), a5891531bf82c6a04eeea84c2bb5b69da15c7c4b(1), d1fd0ae9d7cd2a98fb8e792c993e601cf17f910b(1)

### `B_scaling_laws/pythia_training_log_existing.csv`

- 形状: **1176 行 × 15 列**  文件大小 111,459 bytes
- 列名: `run_id`, `N_params_B`, `D_tokens_B`, `C_FLOPs_1e21`, `steps`, `batch_tokens_M`, `lr`, `wd`, `precision`, `gpu_days`, `step_time_ms`, `train_loss`, `val_loss`, `ppl`, `grad_norm_avg`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `run_id` | 1176 | 620 | 355.552 | 8 | 620 | 1232 |
| `N_params_B` | 1176 | 3.08859 | 3.95408 | 0.070542 | 1.22853 | 11.9658 |
| `D_tokens_B` | 1176 | 146.9 | 88.8626 | 0.134 | 146.801 | 299.893 |
| `C_FLOPs_1e21` | 1176 | 2.72228 | 4.39303 | 0.0001 | 0.7321 | 21.5308 |
| `steps` | 1176 | 70047.3 | 42373 | 64 | 70000 | 143000 |
| `batch_tokens_M` | 1176 | 2.09986 | 0.001159 | 2.09 | 2.1 | 2.1 |
| `lr` | 1176 | 0.000246 | 4.1e-05 | 0.00019 | 0.000239 | 0.000318 |
| `wd` | 1176 | 0.055477 | 0.025014 | 0.01 | 0.056 | 0.1 |
| `gpu_days` | 1176 | 100.988 | 162.966 | 0 | 27.15 | 798.7 |
| `step_time_ms` | 1176 | 532.212 | 397.602 | 100.4 | 417.7 | 1307.4 |
| `train_loss` | 1176 | 2.35416 | 0.329258 | 1.943 | 2.2634 | 4.4729 |
| `val_loss` | 1176 | 2.47683 | 0.342208 | 2.0933 | 2.37425 | 4.7388 |
| `ppl` | 1176 | 12.8696 | 7.58549 | 8.11 | 10.745 | 114.3 |
| `grad_norm_avg` | 1176 | 1.58691 | 2.45352 | 0.392 | 1.0095 | 25.788 |

对象列取值样例:
  - `precision`: 2 个不同取值; top5 = bf16(797), fp16(379)

### `B_scaling_laws/scaling_baseline.csv`

- 形状: **57 行 × 5 列**  文件大小 1,421 bytes
- 列名: `family`, `N_params_B`, `D_tokens_B`, `val_loss`, `is_converged`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 57 | 11.8312 | 18.6811 | 0.07 | 4.2 | 72 |
| `D_tokens_B` | 57 | 3084.84 | 5818.54 | 6 | 1084 | 33739 |
| `val_loss` | 57 | 2.41185 | 0.469511 | 1.71 | 2.3258 | 4.3629 |
| `is_converged` | 57 | 1 | 0 | 1 | 1 | 1 |

对象列取值样例:
  - `family`: 12 个不同取值; top5 = Pythia(8), Qwen2(7), OPT(7), Cerebras-GPT(7), Phi(6)

### `B_scaling_laws/supplementary_NQ_experiment.csv`

- 形状: **360 行 × 5 列**  文件大小 13,670 bytes
- 列名: `experiment_id`, `N_params_B`, `D_tokens_B`, `Q_score`, `val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 360 | 3.11222 | 3.79482 | 0.07 | 1 | 11.97 |
| `D_tokens_B` | 360 | 222 | 214.157 | 10 | 150 | 600 |
| `Q_score` | 360 | 0.5375 | 0.31642 | 0.1 | 0.5 | 1 |
| `val_loss` | 360 | 2.62098 | 0.345536 | 2.0161 | 2.5783 | 3.7216 |

对象列取值样例:
  - `experiment_id`: 360 个不同取值; top5 = N0.07_D0010_Q0.1(1), N0.07_D0010_Q0.2(1), N0.07_D0010_Q0.3(1), N0.07_D0010_Q0.4(1), N0.07_D0010_Q0.6(1)

### `B_scaling_laws/supplementary_NQ_experiment_expanded.csv`

- 形状: **450 行 × 5 列**  文件大小 17,154 bytes
- 列名: `experiment_id`, `N_params_B`, `D_tokens_B`, `Q_score`, `val_loss`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 450 | 3.11222 | 3.79377 | 0.07 | 1 | 11.97 |
| `D_tokens_B` | 450 | 222 | 214.098 | 10 | 150 | 600 |
| `Q_score` | 450 | 0.55 | 0.287548 | 0.1 | 0.55 | 1 |
| `val_loss` | 450 | 2.61653 | 0.341019 | 2.0161 | 2.5741 | 3.7216 |

对象列取值样例:
  - `experiment_id`: 450 个不同取值; top5 = N0.07_D0010_Q0.1(1), N0.07_D0010_Q0.2(1), N0.07_D0010_Q0.3(1), N0.07_D0010_Q0.4(1), N0.07_D0010_Q0.6(1)

### `B_scaling_laws/supplementary_NQ_experiment_large.csv`

- 形状: **1704 行 × 6 列**  文件大小 81,268 bytes
- 列名: `experiment_id`, `N_params_B`, `D_tokens_B`, `Q_score`, `val_loss`, `data_type`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 1704 | 89.9908 | 185.154 | 0.07 | 6.9 | 700 |
| `D_tokens_B` | 1704 | 429.613 | 617.959 | 5 | 125 | 2000 |
| `Q_score` | 1704 | 0.541901 | 0.321054 | 0.05 | 0.55 | 1 |
| `val_loss` | 1704 | 1.33686 | 0.697518 | 0.5 | 1.2965 | 3.3659 |

对象列取值样例:
  - `experiment_id`: 1704 个不同取值; top5 = N0.07_D0005_Q0.05(1), N0.07_D0005_Q0.1(1), N0.07_D0005_Q0.2(1), N0.07_D0005_Q0.3(1), N0.07_D0005_Q0.4(1)
  - `data_type`: 2 个不同取值; top5 = calibrated(984), extrapolated(720)

### `B_scaling_laws/supplementary_large_baseline.csv`

- 形状: **128 行 × 5 列**  文件大小 4,744 bytes
- 列名: `family`, `N_params_B`, `D_tokens_B`, `val_loss`, `is_converged`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 128 | 559.741 | 1009.24 | 100 | 285 | 10000 |
| `D_tokens_B` | 128 | 6237.15 | 9275.94 | 0.1 | 1634 | 36000 |
| `val_loss` | 128 | 2.00786 | 0.398888 | 1.7838 | 1.90605 | 4.0812 |
| `is_converged` | 128 | 1 | 0 | 1 | 1 | 1 |

对象列取值样例:
  - `family`: 128 个不同取值; top5 = Ling-flash-base-2.0-20T(1), FaXin(1), PLaMo-100B(1), Yi-Large(1), Hunyuan(1)

### `B_scaling_laws/supplementary_large_models.csv`

- 形状: **132 行 × 8 列**  文件大小 13,400 bytes
- 列名: `model_name`, `N_params_B`, `D_tokens_B`, `FLOPs`, `publication_date`, `organization`, `accessibility`, `country`
- 缺失: `FLOPs`=11; `organization`=1; `accessibility`=2; `country`=1

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 132 | 551.529 | 995.243 | 100 | 285 | 10000 |
| `D_tokens_B` | 132 | 6048.15 | 9196.07 | 0 | 1549.5 | 36000 |
| `FLOPs` | 121 | 1.114e+25 | 5.52834e+25 | 1.449e+18 | 1.8e+24 | 5e+26 |

对象列取值样例:
  - `model_name`: 132 个不同取值; top5 = Ling-flash-base-2.0-20T(1), FaXin(1), PLaMo-100B(1), Yi-Large(1), Hunyuan(1)
  - `publication_date`: 114 个不同取值; top5 = 2025-08-05(3), 2025-04-05(3), 2025-02-18(3), 2024-02-23(3), 2022-10-20(3)
  - `organization`: 67 个不同取值; top5 = Google(10), DeepSeek(10), Meta AI(9), Alibaba(7), OpenAI(5)
  - `accessibility`: 6 个不同取值; top5 = Open weights (unrestricted)(37), Unreleased(35), Open weights (restricted use)(22), API access(21), Hosted access (no API)(8)
  - `country`: 21 个不同取值; top5 = China(54), United States of America(41), China,China(9), Korea (Republic of)(5), United States of America,United States o(4)

### `B_scaling_laws/training_trajectories/pythia_0.070542B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 0.070542 | 0 | 0.070542 | 0.070542 | 0.070542 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 3.45622 | 0.540199 | 2.8127 | 3.30045 | 4.7385 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_0.162405B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 0.162405 | 0 | 0.162405 | 0.162405 | 0.162405 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 3.24129 | 0.54013 | 2.5959 | 3.08625 | 4.5229 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_0.409009B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 0.409009 | 0 | 0.409009 | 0.409009 | 0.409009 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 3.06398 | 0.540235 | 2.4196 | 2.908 | 4.3484 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_1.040867B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 1.04087 | 0 | 1.04087 | 1.04087 | 1.04087 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 2.93333 | 0.539906 | 2.2923 | 2.7782 | 4.2162 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_1.416184B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 1.41618 | 0 | 1.41618 | 1.41618 | 1.41618 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 2.89893 | 0.540261 | 2.2523 | 2.74415 | 4.1845 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_11.965825B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,540 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 11.9658 | 0 | 11.9658 | 11.9658 | 11.9658 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 2.73679 | 0.540062 | 2.0936 | 2.58125 | 4.0229 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_2.782831B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 14,040 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 2.78283 | 0 | 2.78283 | 2.78283 | 2.78283 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 2.83432 | 0.540098 | 2.1897 | 2.6787 | 4.1185 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `B_scaling_laws/training_trajectories/pythia_6.86104B_trajectory.csv`

- 形状: **500 行 × 5 列**  文件大小 13,540 bytes
- 列名: `N_params_B`, `D_tokens_B`, `val_loss`, `step`, `interpolated`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 500 | 6.86104 | 0 | 6.86104 | 6.86104 | 6.86104 |
| `D_tokens_B` | 500 | 39.0855 | 66.255 | 0.134 | 6.339 | 299.893 |
| `val_loss` | 500 | 2.76817 | 0.540132 | 2.1224 | 2.61305 | 4.0498 |
| `step` | 500 | 18.278 | 31.6606 | 0 | 3 | 143 |
| `interpolated` | 500 | 1 | 0 | 1 | 1 | 1 |

### `C_efficiency_evolution/epoch_all_ai_models.csv`

- 形状: **3523 行 × 57 列**  文件大小 6,671,403 bytes
- 列名: `Model`, `Domain`, `Task`, `Organization`, `Authors`, `Publication date`, `Reference`, `Link`, `Citations`, `Notability criteria`, `Notability criteria notes`, `Parameters`, `Parameters notes`, `Training compute (FLOP)`, `Training compute notes`, `Training dataset size (total)`, `Dataset size notes`, `Training time (hours)`, `Training time notes`, `Training hardware`, `Approach`, `Confidence`, `Abstract`, `Epochs`, `WikiText and Penn Treebank data`, `Model accessibility`, `Country (of organization)`, `Base model`, `Finetune compute (FLOP)`, `Finetune compute notes`, `Hardware quantity`, `Hardware utilization (MFU)`, `Last modified`, `Training cloud compute vendor`, `Training data center`, `Archived links`, `Batch size`, `Batch size notes`, `Organization categorization`, `Foundation model`, `Training compute lower bound`, `Training compute upper bound`, `Training chip-hours`, `Training code accessibility`, `Accessibility notes`, `Possibly over 1e23 FLOP`, `Training compute cost (2023 USD)`, `Utilization notes`, `Numerical format`, `Frontier model`, `Training power draw (W)`, `Training compute estimation method`, `Hugging Face developer id`, `Post-training compute (FLOP)`, `Post-training compute notes`, `Hardware utilization (HFU)`, `Open model weights?`
- 缺失: `Domain`=92; `Task`=122; `Organization`=88; `Authors`=842; `Publication date`=23; `Reference`=174; `Link`=44; `Citations`=2060; `Notability criteria`=2481; `Notability criteria notes`=2682; `Parameters`=1226; `Parameters notes`=1662; `Training compute (FLOP)`=2133; `Training compute notes`=1886; `Training dataset size (total)`=2099; `Dataset size notes`=1804; `Training time (hours)`=2973; `Training time notes`=2918; `Training hardware`=2343; `Approach`=3211; `Confidence`=197; `Abstract`=540; `Epochs`=2720; `WikiText and Penn Treebank data`=3118; `Model accessibility`=870; `Country (of organization)`=95; `Base model`=2839; `Finetune compute (FLOP)`=3263; `Finetune compute notes`=3218; `Hardware quantity`=2680; `Hardware utilization (MFU)`=3462; `Training cloud compute vendor`=3470; `Training data center`=3457; `Archived links`=3498; `Batch size`=3272; `Batch size notes`=3326; `Organization categorization`=107; `Foundation model`=3445; `Training compute lower bound`=3497; `Training compute upper bound`=3482; `Training chip-hours`=3308; `Training code accessibility`=1120; `Accessibility notes`=1773; `Possibly over 1e23 FLOP`=3002; `Training compute cost (2023 USD)`=3299; `Utilization notes`=3435; `Numerical format`=3161; `Frontier model`=3386; `Training power draw (W)`=2752; `Training compute estimation method`=2083; `Hugging Face developer id`=2888; `Post-training compute (FLOP)`=3522; `Post-training compute notes`=3522; `Hardware utilization (HFU)`=3498; `Open model weights?`=870

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `Citations` | 1463 | 4319.68 | 15058.5 | 0 | 471 | 215518 |
| `Parameters` | 2297 | 1.76165e+11 | 4.19994e+12 | 10 | 2.6e+09 | 1.739e+14 |
| `Training compute (FLOP)` | 1390 | 1.64829e+24 | 1.9601e+25 | 40 | 3.67968e+21 | 5e+26 |
| `Training time (hours)` | 550 | 457.93 | 877.733 | 0.1 | 144 | 9022.8 |
| `Epochs` | 803 | 564.744 | 7213.48 | 0 | 16 | 191400 |
| `Finetune compute (FLOP)` | 260 | 8.10985e+22 | 3.58972e+23 | 0 | 9.14488e+20 | 2.78256e+24 |
| `Hardware quantity` | 843 | 974.372 | 7962.02 | 1 | 32 | 200000 |
| `Hardware utilization (MFU)` | 61 | 0.371536 | 0.111028 | 0.171 | 0.3649 | 0.56 |
| `Batch size` | 251 | 5.64082e+06 | 1.62283e+07 | 1 | 1.04858e+06 | 1.28e+08 |
| `Training compute lower bound` | 26 | 9.904e+24 | 4.10514e+25 | 5.423 | 1.476e+23 | 2.1e+26 |
| `Training compute upper bound` | 41 | 5.65428e+25 | 1.79855e+26 | 1.35e+18 | 8.01e+24 | 1.13e+27 |
| `Training chip-hours` | 215 | 1.68198e+06 | 1.06002e+07 | 1 | 20700 | 1.32e+08 |
| `Training compute cost (2023 USD)` | 224 | 6.01529e+06 | 3.76265e+07 | 3.90458 | 27006.5 | 3.87843e+08 |
| `Training power draw (W)` | 771 | 777472 | 4.87088e+06 | 75.8259 | 25605.7 | 1.09949e+08 |
| `Post-training compute (FLOP)` | 1 | 9.4e+22 |  | 9.4e+22 | 9.4e+22 | 9.4e+22 |
| `Hardware utilization (HFU)` | 25 | 0.490876 | 0.148841 | 0.2359 | 0.4935 | 0.927 |

对象列取值样例:
  - `Model`: 3518 个不同取值; top5 = Gemini 3.1 Pro(2), GLM-5(2), SAM 3(2), Eurus-2-7B-PRIME(2), Tulu 3 (Tülu 3) 70B(2)
  - `Domain`: 173 个不同取值; top5 = Language(1560), Biology(376), Vision(322), Image generation(164), Speech(129)
  - `Task`: 922 个不同取值; top5 = Language modeling(317), Language modeling/generation(181), Language modeling/generation,Question an(164), Image classification(127), Image generation,Text-to-image(67)
  - `Organization`: 1313 个不同取值; top5 = OpenAI(119), Alibaba(104), Google DeepMind(100), Google(91), Meta AI(72)
  - `Authors`: 2086 个不同取值; top5 = Qwen Team(16), Gemini Team(12), Susan Zhang∗ , Stephen Roller∗ , Naman G(12), Sho Takase, Jun Suzuki, Masaaki Nagata(10), Shaojie Bai, J. Zico Kolter, Vladlen Kol(9)
  - `Publication date`: 1695 个不同取值; top5 = 2024-09-24(16), 2024-09-19(16), 2024-05-15(14), 2024-12-19(12), 2022-06-21(12)
  - `Reference`: 2657 个不同取值; top5 = CAC registry(48), OPT: Open Pre-trained Transformer Langua(13), Qwen3: Think Deeper, Act Faster(10), DeepSeek-R1: Incentivizing Reasoning Cap(9), Qwen2.5: A Party of Foundation Models!(8)
  - `Link`: 2850 个不同取值; top5 = https://arxiv.org/abs/2205.01068(14), https://qwenlm.github.io/blog/qwen3/(10), https://www.volcengine.com/docs/6360/126(9), https://arxiv.org/abs/2412.13702(8), https://arxiv.org/abs/2304.01373(8)
  - `Notability criteria`: 39 个不同取值; top5 = SOTA improvement(471), Highly cited(180), Historical significance(63), Highly cited,SOTA improvement(62), Training cost(57)
  - `Notability criteria notes`: 782 个不同取值; top5 = Outperforms GPT-4 Turbo and other models(5), "The resulting ESM-2 model family signif(5), Major Alibaba release(4), Top10 recent paper from Sebastian Sartor(4), Model has been open-sourced and frequent(4)
  - `Parameters notes`: 1454 个不同取值; top5 = 7B(43), 70B(23), Table 1(23), 32B(20), 8B(19)
  - `Training compute notes`: 1575 个不同取值; top5 = 
(14), Table D.1
https://arxiv.org/abs/2005.141(8), Flagship model from a leading developer (7), Training compute estimated to be 3.8e25 (5), "We trained our proposed network on 128 (4)
  - `Training dataset size (total)`: 832 个不同取值; top5 = 103000000(62), 929000(41), 2000000(33), 1280000(28), 2000000000000(27)
  - `Dataset size notes`: 1548 个不同取值; top5 = 36T(10), "We train all models for 299,892,736,000(7), 300b, per table d.1(7), "In terms of Qwen2.5, the language model(6), Table 2(6)
  - `Training time notes`: 564 个不同取值; top5 = Table 6(5), Table 7(3), Our final Cambrian-1 models are trained (3), "Training TeleChat took one month (inclu(3), Like its predecessors, Claude 3 models e(3)
  - `Training hardware`: 143 个不同取值; top5 = NVIDIA A100(214), NVIDIA H100 SXM5 80GB(141), NVIDIA V100(125), NVIDIA A100 SXM4 80 GB(80), Google TPU v3(76)
  - `Approach`: 7 个不同取值; top5 = Self-supervised learning(169), Supervised(95), Reinforcement learning(29), Unsupervised(15), Supervised fine-tuning (SFT)(2)
  - `Confidence`: 4 个不同取值; top5 = Confident(1782), Unknown(792), Likely(560), Speculative(192)
  - `Abstract`: 2466 个不同取值; top5 = Large language models, which are often t(12), How do large language models (LLMs) deve(8), Recent work has demonstrated substantial(8), This paper introduces Typhoon 2, a serie(7), In the field of artificial intelligence,(7)
  - `WikiText and Penn Treebank data`: 403 个不同取值; top5 = DEQ-TrellisNet(2), WeNet (WT2)(2), LLaMA-7B (LoRA finetuned)(1), LLaMA-65B (LoRA finetuned)(1), LLaMA-13B (LoRA finetuned)(1)
  - `Model accessibility`: 6 个不同取值; top5 = Unreleased(828), Open weights (unrestricted)(801), API access(378), Open weights (restricted use)(299), Open weights (non-commercial)(225)
  - `Country (of organization)`: 381 个不同取值; top5 = United States of America(1169), China(567), United States of America,United States o(218), United Kingdom of Great Britain and Nort(131), China,China(71)
  - `Base model`: 343 个不同取值; top5 = Llama 2-70B(20), Llama 2-7B(17), Llama 2-13B(16), LLaMA-7B(14), Llama 3.1-8B(13)
  - `Finetune compute notes`: 294 个不同取值; top5 = Definitely a new model, not a GPT-4 fine(5), 6 FLOP / token / parameter * 7*10^9 para(3), 
(2), 5530000000 * 7000000000 * 6 = 2.323e20(2), Not enough info to estimate. GPU time gi(2)
  - `Last modified`: 250 个不同取值; top5 = 2025-11-28 11:53:44+00:00(1519), 2026-02-11 19:26:27+00:00(454), 2025-11-28 11:52:13+00:00(360), 2025-11-28 11:53:59+00:00(156), 2026-02-11 19:23:45+00:00(60)
  - `Training cloud compute vendor`: 12 个不同取值; top5 = Amazon Web Services(14), Google Cloud(8), Microsoft(8), IBM(6), Cerebras(4)
  - `Training data center`: 34 个不同取值; top5 = There is no paper to reference, no infor(12), AWS US East(8), The paper does not mention any hardware,(5), IBM's super computing cluster, Blue Vela(4), Meta’s Research Super Cluster(4)
  - `Archived links`: 24 个不同取值; top5 = Yi-34B(2), TransformerXL + FWL(1), mT0-13B(1), TransformerXL + PowerSGD + L-Greco(1), ADP-FAIRSEQ+NGRAMRES(1)
  - `Batch size notes`: 153 个不同取值; top5 = not listed(18), Table 1(10), Table 9(5), Not listed(4),  0.5M, per table 2.1(3)
  - `Organization categorization`: 126 个不同取值; top5 = Industry(1754), Academia(529), Academia,Academia(187), Industry,Academia(159), Academia,Industry(123)
  - `Foundation model`: 1 个不同取值; top5 = True(78)
  - `Training code accessibility`: 4 个不同取值; top5 = Unreleased(1655), Open source(593), Open (non-commercial)(131), Open (restricted use)(24)
  - `Accessibility notes`: 1548 个不同取值; top5 = Apache 2.0(16), apache 2.0(9), MIT license(7), Modified Apache 2.0 license on https://h(6), Llama 2 license. can't use outputs to tr(6)
  - `Possibly over 1e23 FLOP`: 1 个不同取值; top5 = True(521)
  - `Utilization notes`: 84 个不同取值; top5 = Table 4: Training metric comparison

"Ca(2), "We demonstrate its effectiveness by tra(2), Per https://github.com/jzhang38/TinyLlam(2), CANNOT VERIFY, LIKELY HFU, PAYWALLED. 
((2), Abstract says: "We achieve 47% Model FLO(1)
  - `Numerical format`: 7 个不同取值; top5 = BF16(123), FP32(117), FP16(93), FP8(25), TF32(2)
  - `Frontier model`: 1 个不同取值; top5 = True(137)
  - `Training compute estimation method`: 38 个不同取值; top5 = Operation counting(639), Hardware(433), Operation counting,Hardware(82), Hardware,Operation counting(77), Reported(54)
  - `Hugging Face developer id`: 193 个不同取值; top5 = Qwen(64), nvidia(34), deepseek-ai(31), google(27), mistralai(26)
  - `Post-training compute notes`: 1 个不同取值; top5 = Section 4 gives detail about the post-tr(1)
  - `Open model weights?`: 2 个不同取值; top5 = No(1328), Yes(1325)

### `C_efficiency_evolution/leaderboard_cleaned.csv`

- 形状: **4576 行 × 12 列**  文件大小 1,011,558 bytes
- 列名: `Model`, `#Params (B)`, `Submission Date`, `Hub License`, `Type`, `Average ⬆️`, `IFEval`, `BBH`, `MATH Lvl 5`, `GPQA`, `MUSR`, `MMLU-PRO`
- 缺失: `#Params (B)`=3; `Submission Date`=12; `Hub License`=1753

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `#Params (B)` | 4573 | 11.2451 | 14.6628 | 0 | 8.03 | 140.63 |
| `Average ⬆️` | 4576 | 21.8149 | 10.8021 | 0.737851 | 21.9504 | 52.0814 |
| `IFEval` | 4576 | 45.5845 | 20.4875 | 0 | 45.1167 | 89.9758 |
| `BBH` | 4576 | 27.6475 | 15.224 | 0.251527 | 29.4905 | 76.7 |
| `MATH Lvl 5` | 4576 | 15.5505 | 14.6261 | 0 | 10.8006 | 71.4502 |
| `GPQA` | 4576 | 6.7176 | 5.08511 | 0 | 5.92841 | 29.4183 |
| `MUSR` | 4576 | 9.98397 | 5.83787 | 0 | 10.016 | 38.688 |
| `MMLU-PRO` | 4576 | 25.4052 | 14.2661 | 0 | 27.1895 | 70.0336 |

对象列取值样例:
  - `Model`: 4497 个不同取值; top5 = AtAndDev/Qwen2.5-1.5B-continuous-learnt(2), BoltMonkey/NeuralDaredevil-SuperNova-Lit(2), Columbia-NLP/LION-Gemma-2b-dpo-v1.0(2), Daemontatox/AetherTOT(2), Daemontatox/DocumentCogito(2)
  - `Submission Date`: 262 个不同取值; top5 = 2024-06-26(174), 2024-06-12(151), 2025-02-02(67), 2025-02-27(64), 2024-12-29(63)
  - `Hub License`: 26 个不同取值; top5 = apache-2.0(1482), other(296), mit(284), llama3.1(178), cc-by-nc-4.0(125)
  - `Type`: 7 个不同取值; top5 = 🔶 fine-tuned on domain-specific datasets(1785), 🤝 base merges and moerges(1724), 💬 chat models (RLHF, DPO, IFT, ...)(718), 🟢 pretrained(275), 🟩 continuously pretrained(58)

### `C_efficiency_evolution/leaderboard_enhanced.csv`

- 形状: **4576 行 × 15 列**  文件大小 1,036,749 bytes
- 列名: `Model`, `#Params (B)`, `Submission Date`, `Hub License`, `Type`, `Average ⬆️`, `IFEval`, `BBH`, `MATH Lvl 5`, `GPQA`, `MUSR`, `MMLU-PRO`, `Epoch_AI_Publication_Date`, `Epoch_AI_Organization`, `Epoch_AI_Open_Weights`
- 缺失: `#Params (B)`=3; `Submission Date`=12; `Hub License`=1753; `Epoch_AI_Publication_Date`=4129; `Epoch_AI_Organization`=4133; `Epoch_AI_Open_Weights`=4133

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `#Params (B)` | 4573 | 11.2451 | 14.6628 | 0 | 8.03 | 140.63 |
| `Average ⬆️` | 4576 | 21.8149 | 10.8021 | 0.737851 | 21.9504 | 52.0814 |
| `IFEval` | 4576 | 45.5845 | 20.4875 | 0 | 45.1167 | 89.9758 |
| `BBH` | 4576 | 27.6475 | 15.224 | 0.251527 | 29.4905 | 76.7 |
| `MATH Lvl 5` | 4576 | 15.5505 | 14.6261 | 0 | 10.8006 | 71.4502 |
| `GPQA` | 4576 | 6.7176 | 5.08511 | 0 | 5.92841 | 29.4183 |
| `MUSR` | 4576 | 9.98397 | 5.83787 | 0 | 10.016 | 38.688 |
| `MMLU-PRO` | 4576 | 25.4052 | 14.2661 | 0 | 27.1895 | 70.0336 |

对象列取值样例:
  - `Model`: 4497 个不同取值; top5 = AtAndDev/Qwen2.5-1.5B-continuous-learnt(2), BoltMonkey/NeuralDaredevil-SuperNova-Lit(2), Columbia-NLP/LION-Gemma-2b-dpo-v1.0(2), Daemontatox/AetherTOT(2), Daemontatox/DocumentCogito(2)
  - `Submission Date`: 262 个不同取值; top5 = 2024-06-26(174), 2024-06-12(151), 2025-02-02(67), 2025-02-27(64), 2024-12-29(63)
  - `Hub License`: 26 个不同取值; top5 = apache-2.0(1482), other(296), mit(284), llama3.1(178), cc-by-nc-4.0(125)
  - `Type`: 7 个不同取值; top5 = 🔶 fine-tuned on domain-specific datasets(1785), 🤝 base merges and moerges(1724), 💬 chat models (RLHF, DPO, IFT, ...)(718), 🟢 pretrained(275), 🟩 continuously pretrained(58)
  - `Epoch_AI_Publication_Date`: 69 个不同取值; top5 = 2024-09-19(151), 2024-12-12(40), 2024-04-18(26), 2024-06-07(24), 2025-01-22(24)
  - `Epoch_AI_Organization`: 45 个不同取值; top5 = Alibaba(199), Microsoft Research(43), DeepSeek(33), Meta AI(32), Stanford University(19)
  - `Epoch_AI_Open_Weights`: 2 个不同取值; top5 = Yes(424), No(19)

### `C_efficiency_evolution/leaderboard_extended_timeseries.csv`

- 形状: **4599 行 × 11 列**  文件大小 826,723 bytes
- 列名: `Model`, `Year`, `Params_B`, `Average`, `IFEval`, `BBH`, `MATH_Lvl5`, `GPQA`, `MUSR`, `MMLU_PRO`, `Source`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `Year` | 4599 | 2024.4 | 0.521347 | 2019 | 2024 | 2025 |
| `Params_B` | 4599 | 11.3857 | 15.2838 | 0 | 8.03 | 176 |
| `Average` | 4599 | 21.7725 | 10.8285 | 0.74 | 21.91 | 52.08 |
| `IFEval` | 4599 | 45.4059 | 20.5941 | 0 | 45.008 | 89.9758 |
| `BBH` | 4599 | 27.537 | 15.2944 | 0 | 29.3984 | 76.7 |
| `MATH_Lvl5` | 4599 | 15.4749 | 14.627 | 0 | 10.7251 | 71.4502 |
| `GPQA` | 4599 | 6.6808 | 5.09557 | 0 | 5.92841 | 29.4183 |
| `MUSR` | 4599 | 9.93687 | 5.86032 | 0 | 9.9638 | 38.688 |
| `MMLU_PRO` | 4599 | 25.2928 | 14.35 | 0 | 27.111 | 70.0336 |

对象列取值样例:
  - `Model`: 4520 个不同取值; top5 = AtAndDev/Qwen2.5-1.5B-continuous-learnt(2), BoltMonkey/NeuralDaredevil-SuperNova-Lit(2), Columbia-NLP/LION-Gemma-2b-dpo-v1.0(2), Daemontatox/AetherTOT(2), Daemontatox/DocumentCogito(2)
  - `Source`: 2 个不同取值; top5 = Open LLM Leaderboard(4573), Historical (papers/reports)(26)

### `C_efficiency_evolution/loss_benchmark_bridge.csv`

- 形状: **43 行 × 13 列**  文件大小 11,807 bytes
- 列名: `Model`, `N_params_B`, `D_tokens_B`, `Val_Loss`, `LB_Average`, `LB_IFEval`, `LB_BBH`, `LB_MATH`, `LB_GPQA`, `LB_MUSR`, `LB_MMLU_PRO`, `Loss_Source`, `Loss_Comparability`
- 缺失: `D_tokens_B`=36

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 43 | 14.9217 | 20.214 | 0.162405 | 7.242 | 72.706 |
| `D_tokens_B` | 7 | 299.893 | 0 | 299.893 | 299.893 | 299.893 |
| `Val_Loss` | 43 | 2.15754 | 0.302344 | 1.65 | 2.125 | 2.84 |
| `LB_Average` | 43 | 15.9364 | 9.73921 | 5.07027 | 13.8527 | 38.4411 |
| `LB_IFEval` | 43 | 28.5005 | 12.5646 | 12.4598 | 24.7148 | 64.9665 |
| `LB_BBH` | 43 | 22.2368 | 16.0427 | 2.19883 | 19.4085 | 54.6151 |
| `LB_MATH` | 43 | 8.60647 | 9.94361 | 0.906344 | 3.24773 | 39.1239 |
| `LB_GPQA` | 43 | 6.13912 | 5.96814 | 0 | 4.1387 | 21.5884 |
| `LB_MUSR` | 43 | 8.85474 | 5.39492 | 2.02448 | 8.7151 | 22.6958 |
| `LB_MMLU_PRO` | 43 | 21.2808 | 15.9868 | 1.18942 | 19.0769 | 55.2028 |

对象列取值样例:
  - `Model`: 43 个不同取值; top5 = EleutherAI/pythia-160m(1), EleutherAI/pythia-410m(1), EleutherAI/pythia-1b(1), EleutherAI/pythia-1.4b(1), EleutherAI/pythia-2.8b(1)
  - `Loss_Source`: 19 个不同取值; top5 = Pythia training log (Attachment B, final(7), Qwen2.5 Technical Report (Alibaba, 2024)(6), Qwen2 Technical Report (Alibaba, 2024), (4), Gemma 2 Technical Report (Google, 2024),(3), Gemma Technical Report (Google, 2024), f(3)
  - `Loss_Comparability`: 2 个不同取值; top5 = Medium (different validation set, approx(36), High (same model, same validation set)(7)

### `C_efficiency_evolution/loss_benchmark_bridge_expanded.csv`

- 形状: **75 行 × 13 列**  文件大小 19,124 bytes
- 列名: `Model`, `N_params_B`, `D_tokens_B`, `Val_Loss`, `LB_Average`, `LB_IFEval`, `LB_BBH`, `LB_MATH`, `LB_GPQA`, `LB_MUSR`, `LB_MMLU_PRO`, `Loss_Source`, `Loss_Comparability`
- 缺失: `D_tokens_B`=68

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `N_params_B` | 75 | 17.1225 | 20.2248 | 0.162405 | 7.616 | 72.706 |
| `D_tokens_B` | 7 | 299.893 | 0 | 299.893 | 299.893 | 299.893 |
| `Val_Loss` | 75 | 2.14779 | 0.292937 | 1.65 | 2.1 | 2.84 |
| `LB_Average` | 75 | 19.1097 | 11.7268 | 3.79551 | 17.0469 | 47.9805 |
| `LB_IFEval` | 75 | 37.2548 | 21.2804 | 8.43307 | 28.4117 | 86.3838 |
| `LB_BBH` | 75 | 25.399 | 16.9261 | 2.19883 | 22.0183 | 61.8733 |
| `LB_MATH` | 75 | 13.0119 | 15.5776 | 0.075529 | 5.74018 | 62.5378 |
| `LB_GPQA` | 75 | 6.59806 | 5.88454 | 0 | 4.9217 | 21.5884 |
| `LB_MUSR` | 75 | 8.69089 | 4.9698 | 1.37344 | 9.11276 | 22.6958 |
| `LB_MMLU_PRO` | 75 | 23.7037 | 16.0672 | 1.16172 | 22.1243 | 55.2028 |

对象列取值样例:
  - `Model`: 75 个不同取值; top5 = EleutherAI/pythia-160m(1), EleutherAI/pythia-410m(1), EleutherAI/pythia-1b(1), EleutherAI/pythia-1.4b(1), EleutherAI/pythia-2.8b(1)
  - `Loss_Source`: 31 个不同取值; top5 = Qwen2.5 report(8), Pythia training log (Attachment B, final(7), Qwen2.5 Technical Report (Alibaba, 2024)(6), Qwen2 Technical Report (Alibaba, 2024), (4), Yi-1.5 report(4)
  - `Loss_Comparability`: 2 个不同取值; top5 = Medium (different validation set, approx(68), High (same model, same validation set)(7)

### `C_efficiency_evolution/model_architecture_metadata.csv`

- 形状: **45 行 × 7 列**  文件大小 2,357 bytes
- 列名: `model_name`, `n_layers`, `n_heads`, `d_model`, `vocab_size`, `max_position_embeddings`, `training_data_TB`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `n_layers` | 45 | 34.3111 | 16.4701 | 6 | 32 | 80 |
| `n_heads` | 45 | 34.5111 | 33.9174 | 8 | 32 | 232 |
| `d_model` | 45 | 3628.09 | 2089.84 | 512 | 4096 | 8192 |
| `vocab_size` | 45 | 94804 | 67388.1 | 32000 | 51200 | 256128 |
| `max_position_embeddings` | 45 | 13744.4 | 21934.7 | 2048 | 4096 | 131072 |
| `training_data_TB` | 45 | 6.43222 | 6.43266 | 0.3 | 2.05 | 18 |

对象列取值样例:
  - `model_name`: 45 个不同取值; top5 = EleutherAI/pythia-70m(1), EleutherAI/pythia-160m(1), EleutherAI/pythia-410m(1), EleutherAI/pythia-1b(1), EleutherAI/pythia-1.4b(1)

### `attachment_size_summary.csv`

- 形状: **2012 行 × 4 列**  文件大小 277,728 bytes
- 列名: `file`, `bytes`, `MB`, `GB`
- 缺失: 无

| 列 | count | mean | std | min | 50% | max |
|---|---|---|---|---|---|---|
| `bytes` | 2012 | 547628 | 1.30487e+07 | 1265 | 117156 | 5.50914e+08 |
| `MB` | 2012 | 0.547685 | 13.0487 | 0.001 | 0.117 | 550.914 |
| `GB` | 2012 | 0.000548 | 0.013049 | 1e-06 | 0.000117 | 0.550914 |

对象列取值样例:
  - `file`: 2012 个不同取值; top5 = A_data_value/domain_mapping_guide.csv(1), A_data_value/regmix_domain_sample.jsonl.(1), A_data_value/regmix_domain_summary.csv(1), A_data_value/regmix_tables/est_mixture_1(1), A_data_value/regmix_tables/est_mixture_7(1)

## 3. JSONL(.xz) 文件勘察


### `A_data_value/regmix_domain_sample.jsonl/regmix_domain_sample.jsonl`  (714,013,556 bytes)
- 记录数: **138,034**
- 字段数: 3
- 字段: `text`, `_source_domain`, `_source_path`
- 首条记录字段类型与示例值:
  - `text` : str = ---
abstract: 'The role of the spatial structure of a turbulent flow in enhancing particle collision rates in suspension...(截断)
  - `_source_domain` : str = arxiv
  - `_source_path` : str = valid/arxiv-0-93015664.jsonl

### `A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl/arxiv_part-6777d8857c6e-000486.jsonl`  (22,073,732 bytes)
- 记录数: **17,523**
- 字段数: 24
- 字段: `id`, `dsir_books`, `fluency_en`, `rps_lines_ending_with_terminal_punctution_mark`, `modernbert_cleanliness`, `qurater`, `rps_doc_num_sentences`, `rps_doc_word_count`, `ad_en`, `rps_doc_frac_no_alph_words`, `modernbert_reasoning`, `rps_doc_frac_chars_top_2gram`, `rps_lines_uppercase_letter_fraction`, `rps_doc_frac_unique_words`, `rps_lines_numerical_chars_fraction`, `fineweb_edu`, `dsir_math`, `rps_doc_mean_word_length`, `dsir_wiki`, `rps_doc_frac_chars_top_3gram`, `rps_doc_unigram_entropy`, `modernbert_professionalism`, `modernbert_readability`, `sub_path`
- 首条记录字段类型与示例值:
  - `id` : str = BkiUdvE4eIfiUWiFdbiw
  - `dsir_books` : float = -1961.1743289
  - `fluency_en` : list = [-0.396484375, 0.446533203125]
  - `rps_lines_ending_with_terminal_punctution_mark` : float = 12.5
  - `modernbert_cleanliness` : list = [-8.703125, 1.4111328125, 9.984375, 1.9501953125, 0.95751953125, -10.9609375]
  - `qurater` : list = [-0.83837890625, 5.6796875, 0.158203125, 1.4248046875]
  - `rps_doc_num_sentences` : int = 10
  - `rps_doc_word_count` : float = 83.0
  - `ad_en` : list = [-2.732421875, 3.697265625]
  - `rps_doc_frac_no_alph_words` : float = 28.957055214723923
  - `modernbert_reasoning` : list = [6.38671875, 8.125, 3.447265625, -3.646484375, -5.8671875, -6.44921875]
  - `rps_doc_frac_chars_top_2gram` : float = 12.060301507537687
  - `rps_lines_uppercase_letter_fraction` : float = 16.542599272378048
  - `rps_doc_frac_unique_words` : float = 42.168674698795186
  - `rps_lines_numerical_chars_fraction` : float = 1.6677602909446716
  - `fineweb_edu` : list = [1.3297353982925415]
  - `dsir_math` : float = -1629.94474532
  - `rps_doc_mean_word_length` : float = 7.192771084337349
  - `dsir_wiki` : float = -1934.91548054
  - `rps_doc_frac_chars_top_3gram` : float = 8.040201005025125
  - `rps_doc_unigram_entropy` : float = 3.39185707
  - `modernbert_professionalism` : list = [-4.96875, -3.70703125, -1.28515625, 0.7763671875, 3.55859375, 2.96484375]
  - `modernbert_readability` : list = [-5.640625, -2.0625, -1.189453125, 0.01424407958984375, 3.388671875, 1.3349609375]
  - `sub_path` : str = 

### `A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl/github_part-6777d8857c6e-000275.jsonl`  (249,790,037 bytes)
- 记录数: **203,752**
- 字段数: 24
- 字段: `id`, `modernbert_reasoning`, `dsir_books`, `modernbert_readability`, `fineweb_edu`, `rps_doc_unigram_entropy`, `modernbert_cleanliness`, `rps_doc_frac_chars_top_3gram`, `rps_doc_frac_chars_top_2gram`, `rps_doc_frac_no_alph_words`, `modernbert_professionalism`, `rps_lines_ending_with_terminal_punctution_mark`, `dsir_math`, `rps_doc_frac_unique_words`, `rps_doc_num_sentences`, `rps_lines_numerical_chars_fraction`, `ad_en`, `dsir_wiki`, `rps_doc_word_count`, `rps_doc_mean_word_length`, `rps_lines_uppercase_letter_fraction`, `qurater`, `fluency_en`, `sub_path`
- 首条记录字段类型与示例值:
  - `id` : str = BkiUbIY5ixsDMKtEntrM
  - `modernbert_reasoning` : list = [7.21484375, 5.13671875, 1.7216796875, -4.203125, -4.37890625, -4.59375]
  - `dsir_books` : float = -1510.36861888
  - `modernbert_readability` : list = [-3.10546875, 3.19140625, 3.830078125, 1.3935546875, 1.55859375, -4.97265625]
  - `fineweb_edu` : list = [0.9449735283851624]
  - `rps_doc_unigram_entropy` : float = 3.75736973
  - `modernbert_cleanliness` : list = [-1.408203125, 17.40625, 7.18359375, -4.85546875, -6.578125, -8.3203125]
  - `rps_doc_frac_chars_top_3gram` : int = 0
  - `rps_doc_frac_chars_top_2gram` : int = 0
  - `rps_doc_frac_no_alph_words` : float = 33.167825223435955
  - `modernbert_professionalism` : list = [-4.0234375, -3.41015625, -1.943359375, 0.00202178955078125, 4.99609375, 3.47265625]
  - `rps_lines_ending_with_terminal_punctution_mark` : float = 4.545454545454546
  - `dsir_math` : float = -1405.6938557
  - `rps_doc_frac_unique_words` : float = 75.75757575757575
  - `rps_doc_num_sentences` : int = 24
  - `rps_lines_numerical_chars_fraction` : float = 2.6037200549804296
  - `ad_en` : list = [-2.525390625, 3.291015625]
  - `dsir_wiki` : float = -1520.9007771
  - `rps_doc_word_count` : float = 66.0
  - `rps_doc_mean_word_length` : float = 11.227272727272727
  - `rps_lines_uppercase_letter_fraction` : float = 8.267011554977238
  - `qurater` : list = [-3.701171875, 0.144287109375, -3.98828125, -2.7109375]
  - `fluency_en` : list = [0.0097808837890625, 0.49951171875]
  - `sub_path` : str = 

### `A_data_value/slimpajama_quality_signal_sample.jsonl/slimpajama_quality_signal_sample.jsonl`  (433,080,377 bytes)
- 记录数: **51,230**
- 字段数: 27
- 字段: `id`, `content`, `dsir_books`, `fluency_en`, `rps_lines_ending_with_terminal_punctution_mark`, `modernbert_cleanliness`, `qurater`, `rps_doc_num_sentences`, `rps_doc_word_count`, `ad_en`, `rps_doc_frac_no_alph_words`, `modernbert_reasoning`, `rps_doc_frac_chars_top_2gram`, `rps_lines_uppercase_letter_fraction`, `rps_doc_frac_unique_words`, `rps_lines_numerical_chars_fraction`, `fineweb_edu`, `dsir_math`, `rps_doc_mean_word_length`, `dsir_wiki`, `rps_doc_frac_chars_top_3gram`, `rps_doc_unigram_entropy`, `modernbert_professionalism`, `modernbert_readability`, `sub_path`, `_source_domain`, `_source_path`
- 首条记录字段类型与示例值:
  - `id` : str = BkiUdvE4eIfiUWiFdbiw
  - `content` : str = \subsection{Isolated S(Se)-edge band in MoSe$_{2}$, WS$_{2}$ and WSe$_{2}$
from DFT calculations}

\begin{figure}[tbph]
...(截断)
  - `dsir_books` : float = -1961.1743289
  - `fluency_en` : list = [-0.396484375, 0.446533203125]
  - `rps_lines_ending_with_terminal_punctution_mark` : float = 12.5
  - `modernbert_cleanliness` : list = [-8.703125, 1.4111328125, 9.984375, 1.9501953125, 0.95751953125, -10.9609375]
  - `qurater` : list = [-0.83837890625, 5.6796875, 0.158203125, 1.4248046875]
  - `rps_doc_num_sentences` : int = 10
  - `rps_doc_word_count` : float = 83.0
  - `ad_en` : list = [-2.732421875, 3.697265625]
  - `rps_doc_frac_no_alph_words` : float = 28.957055214723923
  - `modernbert_reasoning` : list = [6.38671875, 8.125, 3.447265625, -3.646484375, -5.8671875, -6.44921875]
  - `rps_doc_frac_chars_top_2gram` : float = 12.060301507537687
  - `rps_lines_uppercase_letter_fraction` : float = 16.542599272378048
  - `rps_doc_frac_unique_words` : float = 42.168674698795186
  - `rps_lines_numerical_chars_fraction` : float = 1.6677602909446716
  - `fineweb_edu` : list = [1.3297353982925415]
  - `dsir_math` : float = -1629.94474532
  - `rps_doc_mean_word_length` : float = 7.192771084337349
  - `dsir_wiki` : float = -1934.91548054
  - `rps_doc_frac_chars_top_3gram` : float = 8.040201005025125
  - `rps_doc_unigram_entropy` : float = 3.39185707
  - `modernbert_professionalism` : list = [-4.96875, -3.70703125, -1.28515625, 0.7763671875, 3.55859375, 2.96484375]
  - `modernbert_readability` : list = [-5.640625, -2.0625, -1.189453125, 0.01424407958984375, 3.388671875, 1.3349609375]
  - `sub_path` : str = 
  - `_source_domain` : str = arxiv
  - `_source_path` : str = arxiv/part-6777d8857c6e-000486.jsonl

## 4. 其他文件

- `source_manifest.json`  10,018 bytes
- `C_efficiency_evolution/data/train-00000-of-00001.parquet`  1,109,997 bytes
- `C_efficiency_evolution/detailed_results/0-hero_Matter-0.2-7B-DPO/results_2024-08-07T13-21-02.903091.json`  119,298 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-34B/results_2024-06-19T03-56-15.142347.json`  120,071 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-34B-32K/results_2024-06-17T17-59-09.162276.json`  120,058 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-34B-Chat/results_2024-06-17T00-49-43.171619.json`  121,683 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-34B-Chat-16K/results_2024-07-17T11-40-22.183397.json`  119,088 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-6B/results_2024-06-17T14-53-45.170909.json`  120,069 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-6B-Chat/results_2024-06-17T10-38-31.759281.json`  120,578 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-9B/results_2024-06-16T17-52-34.155462.json`  120,030 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-9B-32K/results_2024-06-16T17-46-59.477573.json`  120,091 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-9B-Chat/results_2024-06-17T09-46-28.414421.json`  120,577 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-1.5-9B-Chat-16K/results_2024-06-16T18-02-27.668102.json`  120,581 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-34B/results_2024-06-17T01-51-43.500501.json`  121,150 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-34B-200K/results_2024-06-17T01-54-37.398814.json`  121,159 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-34B-Chat/results_2025-02-13T18-27-04.338360.json`  121,534 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-6B/results_2024-06-16T17-47-09.271300.json`  120,064 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-6B-200K/results_2024-06-16T17-48-55.414532.json`  120,053 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-6B-Chat/results_2024-06-16T17-51-39.036011.json`  120,311 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-9B/results_2024-06-17T09-47-58.683900.json`  120,022 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-9B-200K/results_2024-06-16T17-43-23.132103.json`  120,066 bytes
- `C_efficiency_evolution/detailed_results/01-ai_Yi-Coder-9B-Chat/results_2024-09-14T00-02-12.867980.json`  119,111 bytes
- `C_efficiency_evolution/detailed_results/1-800-LLMs_Qwen-2.5-14B-Hindi/results_2025-02-06T06-30-20.641880.json`  114,650 bytes
- `C_efficiency_evolution/detailed_results/1-800-LLMs_Qwen-2.5-14B-Hindi-Custom-Instruct/results_2025-02-06T11-27-58.645224.json`  114,673 bytes
- `C_efficiency_evolution/detailed_results/1024m_PHI-4-Hindi/results_2025-02-06T05-43-08.878637.json`  114,627 bytes
- `C_efficiency_evolution/detailed_results/1024m_QWEN-14B-B100/results_2025-02-06T10-49-32.966803.json`  116,937 bytes
- `C_efficiency_evolution/detailed_results/152334H_miqu-1-70b-sf/results_2024-07-26T02-32-43.573766.json`  118,562 bytes
- `C_efficiency_evolution/detailed_results/1TuanPham_T-VisStar-7B-v0.1/results_2024-09-21T10-25-05.082341.json`  119,470 bytes
- `C_efficiency_evolution/detailed_results/1TuanPham_T-VisStar-v0.1/results_2024-09-21T08-53-27.413933.json`  119,462 bytes
- `C_efficiency_evolution/detailed_results/3rd-Degree-Burn_L-3.1-Science-Writer-8B/results_2024-11-20T00-17-49.945335.json`  114,718 bytes
- `C_efficiency_evolution/detailed_results/3rd-Degree-Burn_Llama-3.1-8B-Squareroot/results_2024-10-10T16-33-29.207911.json`  119,053 bytes
- `C_efficiency_evolution/detailed_results/3rd-Degree-Burn_Llama-3.1-8B-Squareroot-v1/results_2024-11-10T09-13-13.695538.json`  115,193 bytes
- `C_efficiency_evolution/detailed_results/3rd-Degree-Burn_Llama-Squared-8B/results_2024-10-08T16-24-46.765820.json`  123,434 bytes
- `C_efficiency_evolution/detailed_results/4season_final_model_test_v2/results_2024-07-25T15-13-01.854583.json`  118,558 bytes
- `C_efficiency_evolution/detailed_results/AALF_FuseChat-Llama-3.1-8B-Instruct-preview/results_2024-11-20T12-47-00.610130.json`  115,155 bytes
- `C_efficiency_evolution/detailed_results/AALF_FuseChat-Llama-3.1-8B-SFT-preview/results_2024-11-21T13-45-14.605333.json`  115,136 bytes
- `C_efficiency_evolution/detailed_results/AALF_gemma-2-27b-it-SimPO-37K/results_2024-09-01T02-53-57.422115.json`  118,890 bytes
- `C_efficiency_evolution/detailed_results/AALF_gemma-2-27b-it-SimPO-37K-100steps/results_2024-09-27T18-43-00.611233.json`  118,897 bytes
- `C_efficiency_evolution/detailed_results/Aashraf995_Creative-7B-nerd/results_2024-12-18T03-17-53.359820.json`  114,617 bytes
- `C_efficiency_evolution/detailed_results/Aashraf995_Gemma-Evo-10B/results_2024-12-18T04-35-43.057636.json`  114,588 bytes
- `C_efficiency_evolution/detailed_results/Aashraf995_Qwen-Evo-7B/results_2024-12-18T03-20-24.239014.json`  114,632 bytes
- `C_efficiency_evolution/detailed_results/Aashraf995_QwenStock-14B/results_2024-12-18T04-15-47.313168.json`  114,623 bytes
- `C_efficiency_evolution/detailed_results/abacusai_bigyi-15b/results_2024-09-18T06-37-06.680694.json`  118,586 bytes
- `C_efficiency_evolution/detailed_results/abacusai_Liberated-Qwen1.5-14B/results_2024-09-05T12-34-33.126455.json`  119,293 bytes
- `C_efficiency_evolution/detailed_results/AbacusResearch_Jallabi-34B/results_2024-07-30T05-45-04.496708.json`  118,529 bytes
- `C_efficiency_evolution/detailed_results/abideen_MedPhi-4-14B-v1/results_2025-01-13T23-03-42.989646.json`  115,070 bytes
- `C_efficiency_evolution/detailed_results/AELLM_gemma-2-aeria-infinity-9b/results_2024-10-09T11-48-07.657590.json`  118,840 bytes
- `C_efficiency_evolution/detailed_results/AELLM_gemma-2-lyco-infinity-9b/results_2024-10-10T12-58-52.149709.json`  118,871 bytes
- `C_efficiency_evolution/detailed_results/agentlans_Llama3.1-8B-drill/results_2024-12-27T23-35-02.383650.json`  115,080 bytes
- `C_efficiency_evolution/detailed_results/agentlans_Llama3.1-Daredevilish/results_2025-01-23T10-50-23.011610.json`  115,074 bytes
- `C_efficiency_evolution/detailed_results/agentlans_Qwen2.5-0.5B-Instruct-CrashCourse-dropout/results_2025-01-01T06-48-50.303492.json`  117,231 bytes
- `C_efficiency_evolution/detailed_results/AGI-0_Art-v0-3B/results_2025-01-18T12-23-40.344038.json`  115,381 bytes
- `C_efficiency_evolution/detailed_results/AGI-0_Artificium-llama3.1-8B-001/results_2024-09-11T00-52-31.543315.json`  123,514 bytes
- `C_efficiency_evolution/detailed_results/AGI-0_smartllama3.1-8B-001/results_2024-11-25T22-15-59.091478.json`  114,657 bytes
- `C_efficiency_evolution/detailed_results/Ahdoot_StructuredThinker-v0.3-MoreStructure/results_2025-01-01T03-42-11.275880.json`  114,702 bytes
- `C_efficiency_evolution/detailed_results/Ahdoot_Test_StealthThinker/results_2025-01-06T23-06-48.071737.json`  114,651 bytes
- `C_efficiency_evolution/detailed_results/AI-MO_NuminaMath-7B-CoT/results_2024-09-11T22-04-44.864778.json`  119,043 bytes
- `C_efficiency_evolution/detailed_results/AI-MO_NuminaMath-7B-TIR/results_2024-07-25T13-40-31.863736.json`  118,583 bytes
- `C_efficiency_evolution/detailed_results/AI-Sweden-Models_gpt-sw3-40b/results_2024-07-26T18-58-19.331750.json`  118,546 bytes
- `C_efficiency_evolution/detailed_results/AI-Sweden-Models_Llama-3-8B-instruct/results_2024-07-22T12-09-41.340187.json`  119,019 bytes
- `C_efficiency_evolution/detailed_results/AI4free_Dhanishtha/results_2025-02-23T21-34-08.816420.json`  115,849 bytes
- `C_efficiency_evolution/detailed_results/AI4free_t2/results_2025-02-27T13-24-19.191994.json`  115,825 bytes
- `C_efficiency_evolution/detailed_results/AicoresSecurity_Cybernet-Sec-3B-R1-V0/results_2025-02-27T08-26-21.525046.json`  117,633 bytes
- `C_efficiency_evolution/detailed_results/AicoresSecurity_Cybernet-Sec-3B-R1-V0-Coder/results_2025-02-27T08-25-30.321172.json`  117,654 bytes
- `C_efficiency_evolution/detailed_results/AicoresSecurity_Cybernet-Sec-3B-R1-V1/results_2025-03-07T07-38-49.658744.json`  117,643 bytes
- `C_efficiency_evolution/detailed_results/AicoresSecurity_Cybernet-Sec-3B-R1-V1.1/results_2025-03-07T20-20-00.411476.json`  117,636 bytes
- `C_efficiency_evolution/detailed_results/AIDC-AI_Marco-o1/results_2025-02-01T05-51-39.702980.json`  115,423 bytes
- `C_efficiency_evolution/detailed_results/akhadangi_Llama3.2.1B.0.01-First/results_2025-03-10T11-47-38.291006.json`  113,576 bytes
- `C_efficiency_evolution/detailed_results/akhadangi_Llama3.2.1B.0.1-First/results_2025-03-10T11-38-49.477216.json`  113,581 bytes
- `C_efficiency_evolution/detailed_results/akhadangi_Llama3.2.1B.BaseFiT/results_2025-03-10T11-54-29.662487.json`  113,539 bytes
- `C_efficiency_evolution/detailed_results/Alepach_notHumpback-M0/results_2025-01-06T21-45-58.081166.json`  114,964 bytes
- `C_efficiency_evolution/detailed_results/Alepach_notHumpback-M1/results_2025-01-06T21-45-56.042287.json`  114,996 bytes
- `C_efficiency_evolution/detailed_results/Alepach_notHumpback-M1-v2/results_2025-01-20T06-48-43.444170.json`  115,246 bytes
- `C_efficiency_evolution/detailed_results/Alibaba-NLP_gte-Qwen2-7B-instruct/results_2024-08-12T13-03-04.687518.json`  118,897 bytes
- `C_efficiency_evolution/detailed_results/allenai_Llama-3.1-Tulu-3-70B/results_2024-11-28T10-27-03.067415.json`  115,203 bytes
- `C_efficiency_evolution/detailed_results/allenai_Llama-3.1-Tulu-3-8B-DPO/results_2024-11-22T10-05-59.254446.json`  115,244 bytes
- `C_efficiency_evolution/detailed_results/allenai_Llama-3.1-Tulu-3-8B-RM/results_2024-11-22T10-23-40.491308.json`  114,877 bytes
- `C_efficiency_evolution/detailed_results/allenai_Llama-3.1-Tulu-3-8B-SFT/results_2024-11-22T10-12-59.705361.json`  115,243 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Gemma2Slerp1-2.6B/results_2024-12-06T19-34-34.054813.json`  114,631 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Gemma2Slerp1-27B/results_2024-12-06T21-29-35.095408.json`  114,638 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Gemma2Slerp2-27B/results_2024-12-06T21-42-19.850532.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Gemma2Slerp4-27B/results_2024-12-06T21-42-08.193662.json`  114,631 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_GemmaSlerp5-10B/results_2024-11-22T21-11-53.068712.json`  114,627 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_GemmaStock1-27B/results_2024-12-06T21-32-03.119459.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Llama3.1-60B/results_2024-10-11T14-34-09.525837.json`  118,269 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Mistralmash2-7B-s/results_2024-09-03T06-57-24.156090.json`  118,605 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_MistralPhi3-11B/results_2024-09-03T06-29-04.715113.json`  118,228 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Phi3mash1-17B-pass/results_2024-09-03T06-44-10.031807.json`  118,256 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Qwen2.5-42B-AGI/results_2024-10-22T00-42-05.989075.json`  118,270 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Qwen2.5-7B-task3/results_2024-11-06T18-32-30.118491.json`  114,659 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Qwen2.5-7B-task4/results_2024-11-06T18-31-18.397337.json`  114,669 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_QwenSlerp12-7B/results_2024-11-22T20-03-47.750767.json`  114,638 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Qwenslerp3-14B/results_2024-10-21T20-32-17.784584.json`  118,594 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Qwenslerp3-7B/results_2024-11-06T18-37-23.836740.json`  114,655 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_QwenSlerp4-14B/results_2024-12-06T19-49-12.629259.json`  114,641 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_QwenSlerp6-14B/results_2024-12-06T19-39-20.271977.json`  114,647 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_QwenStock1-14B/results_2024-12-06T20-08-10.157704.json`  114,625 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_QwenStock2-14B/results_2024-12-06T20-07-24.006733.json`  114,628 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_ROGERphi-7B-slerp/results_2024-08-13T08-08-29.741245.json`  118,606 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Rombos-LLM-V2.5-Qwen-42b/results_2024-10-29T06-38-37.491206.json`  118,285 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Yi-1.5-34B/results_2024-10-29T01-57-59.713822.json`  118,214 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Yislerp-34B/results_2024-09-21T13-56-11.768462.json`  118,584 bytes
- `C_efficiency_evolution/detailed_results/allknowingroger_Yislerp2-34B/results_2024-09-27T23-49-27.470228.json`  118,575 bytes
- `C_efficiency_evolution/detailed_results/allura-org_Mistral-Small-24b-Sertraline-0304/results_2025-03-05T17-04-59.040521.json`  113,957 bytes
- `C_efficiency_evolution/detailed_results/Alsebay_Qwen2.5-7B-test-novelist/results_2024-12-18T02-28-41.543232.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/Amaorynho_BBAI2006/results_2025-02-27T22-01-12.890348.json`  113,109 bytes
- `C_efficiency_evolution/detailed_results/Amaorynho_BBAI270V4/results_2025-02-27T08-30-09.873664.json`  116,049 bytes
- `C_efficiency_evolution/detailed_results/Amaorynho_BBAIIFEV1/results_2025-03-01T19-40-25.427409.json`  114,032 bytes
- `C_efficiency_evolution/detailed_results/Amaorynho_BBAI_375/results_2025-02-27T12-12-55.675096.json`  113,108 bytes
- `C_efficiency_evolution/detailed_results/amazon_MegaBeam-Mistral-7B-300k/results_2024-10-07T13-52-28.283899.json`  119,126 bytes
- `C_efficiency_evolution/detailed_results/Amu_t1-1.5B/results_2025-03-08T04-40-03.345654.json`  116,096 bytes
- `C_efficiency_evolution/detailed_results/Amu_t1-3B/results_2025-03-13T16-46-20.235537.json`  116,141 bytes
- `C_efficiency_evolution/detailed_results/anakin87_gemma-2b-orpo/results_2024-07-29T09-00-07.856925.json`  118,939 bytes
- `C_efficiency_evolution/detailed_results/arcee-ai_Llama-3.1-SuperNova-Lite/results_2024-09-18T15-23-41.201451.json`  119,049 bytes
- `C_efficiency_evolution/detailed_results/ArliAI_ArliAI-RPMax-12B-v1.1/results_2024-09-14T01-35-22.892458.json`  122,777 bytes
- `C_efficiency_evolution/detailed_results/ArliAI_Llama-3.1-8B-ArliAI-RPMax-v1.1/results_2024-09-19T08-16-03.312956.json`  119,081 bytes
- `C_efficiency_evolution/detailed_results/Arthur-LAGACHERIE_Precis-1B-Instruct/results_2025-01-20T02-26-42.196441.json`  118,588 bytes
- `C_efficiency_evolution/detailed_results/Artples_L-MChat-7b/results_2024-08-21T15-54-00.632996.json`  118,895 bytes
- `C_efficiency_evolution/detailed_results/Artples_L-MChat-Small/results_2024-08-12T20-08-53.082655.json`  118,844 bytes
- `C_efficiency_evolution/detailed_results/Aryanne_QwentileSwap/results_2025-03-14T11-42-48.673718.json`  116,142 bytes
- `C_efficiency_evolution/detailed_results/Aryanne_SHBA/results_2025-01-14T02-47-19.535563.json`  115,006 bytes
- `C_efficiency_evolution/detailed_results/Aryanne_SuperHeart/results_2024-09-23T11-59-42.090754.json`  118,588 bytes
- `C_efficiency_evolution/detailed_results/asharsha30_LLAMA_Harsha_8_B_ORDP_10k/results_2024-12-02T04-58-31.780769.json`  114,956 bytes
- `C_efficiency_evolution/detailed_results/AtAndDev_Qwen2.5-1.5B-continuous-learnt/results_2024-10-15T04-55-44.027283.json`  121,258 bytes
- `C_efficiency_evolution/detailed_results/Ateron_Glowing-Forest-12B/results_2025-03-06T15-03-45.762050.json`  113,462 bytes
- `C_efficiency_evolution/detailed_results/Ateron_Lotus-Magpic/results_2025-03-05T13-12-09.385184.json`  117,658 bytes
- `C_efficiency_evolution/detailed_results/Ateron_Way_of_MagPicaro/results_2025-03-05T15-31-39.956586.json`  113,460 bytes
- `C_efficiency_evolution/detailed_results/AuraIndustries_Aura-4B/results_2024-12-18T03-06-02.963485.json`  115,028 bytes
- `C_efficiency_evolution/detailed_results/AuraIndustries_Aura-8B/results_2024-12-11T04-50-16.483325.json`  115,066 bytes
- `C_efficiency_evolution/detailed_results/AuraIndustries_Aura-MoE-2x4B/results_2024-12-18T04-41-59.014444.json`  115,008 bytes
- `C_efficiency_evolution/detailed_results/AuraIndustries_Aura-MoE-2x4B-v2/results_2024-12-18T04-58-06.054592.json`  115,023 bytes
- `C_efficiency_evolution/detailed_results/Aurel9_testmerge-7b/results_2024-11-20T00-59-08.345134.json`  114,598 bytes
- `C_efficiency_evolution/detailed_results/awnr_Mistral-7B-v0.1-signtensors-1-over-4/results_2024-07-22T12-11-38.418833.json`  117,939 bytes
- `C_efficiency_evolution/detailed_results/axolotl-ai-co_romulus-mistral-nemo-12b-simpo/results_2024-07-26T17-54-00.090079.json`  118,921 bytes
- `C_efficiency_evolution/detailed_results/Ayush-Singh_Llama1B-sft-2/results_2025-02-05T09-53-28.796940.json`  114,296 bytes
- `C_efficiency_evolution/detailed_results/Azure99_blossom-v5-32b/results_2024-09-27T16-03-40.225448.json`  118,966 bytes
- `C_efficiency_evolution/detailed_results/Azure99_blossom-v5-llama3-8b/results_2024-09-21T11-45-04.971156.json`  118,979 bytes
- `C_efficiency_evolution/detailed_results/Azure99_blossom-v5.1-34b/results_2024-07-17T02-52-19.756120.json`  119,026 bytes
- `C_efficiency_evolution/detailed_results/Azure99_blossom-v5.1-9b/results_2024-07-17T00-49-16.822666.json`  118,993 bytes
- `C_efficiency_evolution/detailed_results/Azure99_Blossom-V6-14B/results_2025-02-01T09-24-58.712862.json`  115,071 bytes
- `C_efficiency_evolution/detailed_results/Azure99_Blossom-V6-7B/results_2025-02-01T08-37-59.351105.json`  115,093 bytes
- `C_efficiency_evolution/detailed_results/Ba2han_Llama-Phi-3_DoRA/results_2024-07-29T08-29-23.036311.json`  119,067 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Gemma2-9B-IT-Simpo-Infinity-Preference/results_2024-09-03T10-12-22.221077.json`  118,986 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0613-Llama3-70B/results_2024-07-25T07-35-25.760033.json`  119,061 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0613-Mistral-7B/results_2024-07-22T10-30-55.104090.json`  118,894 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0625-Llama3-70B/results_2024-09-03T00-17-17.935598.json`  119,070 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0625-Llama3-8B/results_2024-07-25T14-21-16.697446.json`  119,040 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0625-Mistral-7B/results_2024-08-12T18-08-03.104883.json`  118,916 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0625-Qwen2-7B/results_2024-08-12T18-24-56.268433.json`  118,995 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-3M-0625-Yi-1.5-9B/results_2024-08-21T05-35-19.299638.json`  119,163 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-7M-0729-Llama3_1-8B/results_2024-08-12T13-18-16.514134.json`  119,109 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-7M-0729-mistral-7B/results_2024-08-12T13-10-51.363675.json`  118,929 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-7M-Gen-Llama3_1-70B/results_2024-09-29T01-05-56.315539.json`  119,064 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-7M-Gen-Llama3_1-8B/results_2024-09-02T15-50-11.409871.json`  119,078 bytes
- `C_efficiency_evolution/detailed_results/BAAI_Infinity-Instruct-7M-Gen-mistral-7B/results_2024-09-02T15-46-52.517392.json`  118,887 bytes
- `C_efficiency_evolution/detailed_results/BAAI_OPI-Llama-3.1-8B-Instruct/results_2024-09-23T11-57-00.029469.json`  118,698 bytes
- `C_efficiency_evolution/detailed_results/Baptiste-HUVELLE-10_LeTriomphant2.2_ECE_iLAB/results_2025-03-01T08-30-05.978173.json`  113,567 bytes
- `C_efficiency_evolution/detailed_results/bedio_DistMerge_Llama-3.1-8B-Instruct/results_2024-09-12T00-20-16.819235.json`  118,648 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_Meta-Llama-3-8Bee/results_2024-07-25T12-24-55.860310.json`  118,658 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_smol_llama-101M-GQA/results_2024-08-13T08-16-04.970628.json`  118,260 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_smol_llama-220M-GQA/results_2024-06-26T23-24-10.372392.json`  117,943 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_smol_llama-220M-GQA-fineweb_edu/results_2024-07-18T11-09-45.118921.json`  118,259 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_smol_llama-220M-openhermes/results_2024-09-21T20-51-56.948604.json`  118,261 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_tFINE-900m-e16-d32-flan/results_2024-09-14T06-01-16.457735.json`  118,249 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_tFINE-900m-e16-d32-flan-infinity-instruct-7m-T2T_en-1024/results_2024-09-14T09-26-54.566432.json`  118,361 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_tFINE-900m-e16-d32-instruct_2e/results_2024-09-23T16-45-27.487413.json`  118,286 bytes
- `C_efficiency_evolution/detailed_results/BEE-spoke-data_tFINE-900m-instruct-orpo/results_2024-09-23T17-46-03.770289.json`  118,496 bytes
- `C_efficiency_evolution/detailed_results/BenevolenceMessiah_Qwen2.5-72B-2x-Instruct-TIES-v1.0/results_2024-11-25T10-42-51.480482.json`  117,318 bytes
- `C_efficiency_evolution/detailed_results/BenevolenceMessiah_Yi-Coder-9B-Chat-Instruct-TIES-MoE-v1.0/results_2024-09-28T01-29-46.479228.json`  118,712 bytes
- `C_efficiency_evolution/detailed_results/bigscience_bloom-1b1/results_2024-06-16T21-45-03.222205.json`  119,924 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Bloslain-8B-v0.2/results_2024-11-20T00-14-46.384603.json`  114,675 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_llama-3-luminous-merged/results_2024-10-11T13-21-44.421484.json`  118,657 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_llama-3.1-8B-Galore-openassistant-guanaco/results_2024-10-19T11-26-56.879492.json`  118,727 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Llama-3.1-8B-OpenO1-SFT-v0.1/results_2024-12-29T05-50-35.952715.json`  119,568 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Llama-3.1-8B-pythonic-passthrough-merge/results_2024-11-06T21-15-20.894828.json`  114,699 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Neos-Gemma-2-9b/results_2024-11-11T12-49-01.846333.json`  115,258 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Neos-Llama-3.1-8B/results_2024-11-12T13-25-16.816634.json`  114,915 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Neos-Llama-3.1-base/results_2024-11-11T14-31-42.592878.json`  114,744 bytes
- `C_efficiency_evolution/detailed_results/BlackBeenie_Neos-Phi-3-14B-v0.1/results_2024-11-27T15-11-29.345795.json`  114,928 bytes
- `C_efficiency_evolution/detailed_results/Bllossom_llama-3.2-Korean-Bllossom-AICA-5B/results_2024-12-18T04-48-36.945827.json`  119,914 bytes
- `C_efficiency_evolution/detailed_results/BoltMonkey_DreadMix/results_2024-10-15T10-14-27.298886.json`  119,037 bytes
- `C_efficiency_evolution/detailed_results/BoltMonkey_NeuralDaredevil-SuperNova-Lite-7B-DARETIES-abliterated/results_2024-10-02T01-12-39.953335.json`  118,725 bytes
- `C_efficiency_evolution/detailed_results/BoltMonkey_SuperNeuralDreadDevil-8b/results_2024-10-15T04-58-27.394066.json`  119,047 bytes
- `C_efficiency_evolution/detailed_results/bosonai_Higgs-Llama-3-70B/results_2024-10-06T02-29-41.562725.json`  119,058 bytes
- `C_efficiency_evolution/detailed_results/braindao_DeepSeek-R1-Distill-Qwen-1.5B-Blunt/results_2025-02-23T21-30-24.799148.json`  115,925 bytes
- `C_efficiency_evolution/detailed_results/braindao_DeepSeek-R1-Distill-Qwen-14B-Blunt/results_2025-03-03T22-24-01.916394.json`  115,932 bytes
- `C_efficiency_evolution/detailed_results/braindao_DeepSeek-R1-Distill-Qwen-14B-Blunt-Uncensored/results_2025-03-06T14-15-38.238783.json`  115,952 bytes
- `C_efficiency_evolution/detailed_results/braindao_DeepSeek-R1-Distill-Qwen-14B-Blunt-Uncensored-Blunt/results_2025-03-03T22-23-13.808613.json`  115,987 bytes
- `C_efficiency_evolution/detailed_results/braindao_DeepSeek-R1-Distill-Qwen-7B-Reflective/results_2025-03-03T20-24-38.352295.json`  115,932 bytes
- `C_efficiency_evolution/detailed_results/braindao_Qwen2.5-14B/results_2025-03-06T09-37-30.014109.json`  116,084 bytes
- `C_efficiency_evolution/detailed_results/BrainWave-ML_llama3.2-3B-maths-orpo/results_2024-10-25T15-23-53.121910.json`  118,273 bytes
- `C_efficiency_evolution/detailed_results/BramVanroy_fietje-2/results_2024-10-28T15-01-12.007604.json`  118,519 bytes
- `C_efficiency_evolution/detailed_results/BramVanroy_fietje-2-chat/results_2024-10-28T15-30-42.669861.json`  118,805 bytes
- `C_efficiency_evolution/detailed_results/BramVanroy_fietje-2-instruct/results_2024-10-29T06-19-03.359209.json`  118,791 bytes
- `C_efficiency_evolution/detailed_results/BramVanroy_GEITje-7B-ultra/results_2024-10-28T15-46-03.761866.json`  118,980 bytes
- `C_efficiency_evolution/detailed_results/BSC-LT_salamandra-7b/results_2024-11-22T08-16-09.604171.json`  114,253 bytes
- `C_efficiency_evolution/detailed_results/BSC-LT_salamandra-7b-instruct/results_2024-11-22T08-58-14.996318.json`  115,708 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_FuseCyberMix-Qwen-2.5-7B-Instruct/results_2024-12-20T09-17-49.550984.json`  117,122 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Gemma2-9B-TitanFusion/results_2025-02-19T15-04-16.854720.json`  114,166 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_HyperLlama-3.1-8B/results_2024-09-07T00-47-53.210821.json`  123,511 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.1-8B-TitanFusion-v3/results_2024-09-27T16-43-14.678628.json`  118,651 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-Deep-Test/results_2025-01-01T18-12-28.548987.json`  118,387 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-Della/results_2025-01-20T06-12-21.388884.json`  118,764 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-Long-Think/results_2024-10-24T00-00-00.000000.json`  122,726 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-Mix-Skill/results_2024-10-28T14-55-48.361579.json`  122,670 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-ProdigyPlus/results_2024-10-25T19-54-51.799388.json`  122,719 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Llama-3.2-3B-ToxicKod/results_2025-03-10T22-39-13.540337.json`  117,610 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-3.5-mini-TitanFusion-0.1/results_2024-10-15T16-43-12.313024.json`  119,126 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-Model-Stock-v4/results_2025-01-25T10-05-56.730634.json`  115,162 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-ReasoningRP/results_2025-01-28T13-39-38.811621.json`  115,160 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-RP-v0/results_2025-01-14T02-53-40.099872.json`  115,061 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-RR-Shoup/results_2025-02-02T12-54-45.987972.json`  115,106 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-Sce-exp-v0.1/results_2025-02-07T16-42-39.879750.json`  115,141 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-Stock-Ex/results_2025-01-16T19-58-42.857768.json`  115,156 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-Stock-RP/results_2025-01-13T18-58-31.199413.json`  115,135 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Phi-4-Trim-Exp1/results_2025-02-18T15-28-01.718723.json`  113,920 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen-2.5-7B-Deep-Stock-v1/results_2025-01-25T12-15-50.445689.json`  117,262 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen-2.5-7B-Deep-Stock-v5/results_2025-02-18T17-48-11.735090.json`  116,138 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen-2.5-7B-Exp-Sce/results_2025-02-18T20-49-55.995211.json`  116,110 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen-2.5-7b-S1k/results_2025-02-24T22-40-58.147391.json`  116,181 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-1.5B-Model-Stock/results_2025-02-28T19-58-44.405602.json`  115,871 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-3B-Model-Stock-v2/results_2025-01-15T21-40-54.951808.json`  117,001 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-3B-Model-Stock-v3.2/results_2025-02-26T09-09-35.582526.json`  116,219 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-3B-Model-Stock-v4.1/results_2025-02-28T20-09-24.661001.json`  116,236 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-3B-RP-Mix/results_2024-10-23T17-42-13.231593.json`  121,261 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-Instruct-Fusion/results_2024-11-06T17-58-12.254186.json`  117,265 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-MixStock-Sce-V0.3/results_2025-02-19T14-26-51.712601.json`  116,163 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-MixStock-V0.1/results_2025-02-05T19-27-09.819898.json`  117,278 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-R1-Bespoke-Stock/results_2025-01-25T06-59-40.273507.json`  117,024 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-R1-Bespoke-Task/results_2025-01-25T06-55-46.740522.json`  117,025 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Qwen2.5-7B-Sky-R1-Mini/results_2025-02-19T15-06-20.141651.json`  113,549 bytes
- `C_efficiency_evolution/detailed_results/bunnycore_Smol-Llama-3.2-3B/results_2024-12-29T21-03-54.900088.json`  118,721 bytes
- `C_efficiency_evolution/detailed_results/byroneverson_Mistral-Small-Instruct-2409-abliterated/results_2024-10-15T19-32-32.570391.json`  122,854 bytes
- `C_efficiency_evolution/detailed_results/byroneverson_Yi-1.5-9B-Chat-abliterated/results_2024-09-18T14-49-52.560572.json`  119,139 bytes
- `C_efficiency_evolution/detailed_results/CarrotAI_Llama-3.2-Rabbit-Ko-3B-Instruct/results_2024-12-20T07-08-53.216233.json`  115,194 bytes
- `C_efficiency_evolution/detailed_results/CarrotAI_Llama-3.2-Rabbit-Ko-3B-Instruct-2412/results_2024-12-19T09-37-44.312285.json`  114,713 bytes
- `C_efficiency_evolution/detailed_results/carsenk_phi3.5_mini_exp_825_uncensored/results_2024-09-02T23-26-41.313457.json`  119,154 bytes
- `C_efficiency_evolution/detailed_results/Casual-Autopsy_L3-Umbral-Mind-RP-v2.0-8B/results_2024-07-22T09-49-57.500293.json`  119,098 bytes
- `C_efficiency_evolution/detailed_results/cat-searcher_gemma-2-9b-it-sppo-iter-1/results_2024-08-13T03-41-28.069452.json`  118,936 bytes
- `C_efficiency_evolution/detailed_results/CausalLM_14B/results_2024-06-16T22-41-11.628824.json`  120,061 bytes
- `C_efficiency_evolution/detailed_results/CausalLM_34b-beta/results_2024-08-13T04-12-29.067058.json`  118,597 bytes
- `C_efficiency_evolution/detailed_results/CausalLM_preview-1-hf/results_2025-01-26T19-50-38.868150.json`  114,967 bytes
- `C_efficiency_evolution/detailed_results/cckm_tinymistral_950m/results_2025-01-13T21-36-28.245477.json`  114,440 bytes
- `C_efficiency_evolution/detailed_results/Changgil_K2S3-14b-v0.2/results_2024-08-12T22-44-05.278865.json`  118,607 bytes
- `C_efficiency_evolution/detailed_results/Changgil_K2S3-v0.1/results_2025-02-13T18-27-04.338360.json`  118,528 bytes
- `C_efficiency_evolution/detailed_results/chujiezheng_Mistral7B-PairRM-SPPO-ExPO/results_2024-09-22T08-23-37.171869.json`  119,059 bytes
- `C_efficiency_evolution/detailed_results/ClaudioItaly_Albacus/results_2024-09-14T01-42-34.001195.json`  118,559 bytes
- `C_efficiency_evolution/detailed_results/ClaudioItaly_Book-Gut12B/results_2024-09-18T16-10-32.498210.json`  118,545 bytes
- `C_efficiency_evolution/detailed_results/ClaudioItaly_Evolutionstory-7B-v2.2/results_2024-09-03T07-38-49.543556.json`  118,591 bytes
- `C_efficiency_evolution/detailed_results/ClaudioItaly_intelligence-cod-rag-7b-v3/results_2024-12-02T20-56-56.981425.json`  115,371 bytes
- `C_efficiency_evolution/detailed_results/cloudyu_Llama-3-70Bx2-MOE/results_2024-07-27T07-27-59.261852.json`  118,619 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9-llama3-8b/results_2024-06-16T19-05-03.780835.json`  121,599 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9.1-llama-3-70b/results_2024-10-05T20-41-16.262145.json`  119,060 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9.1-yi-1.5-9b/results_2024-08-12T11-27-37.056736.json`  119,024 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9.3-mistral-7B-32k/results_2024-07-22T09-19-12.455984.json`  119,021 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9.4-gemma2-2b/results_2024-09-01T05-44-15.827438.json`  118,976 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_dolphin-2.9.4-llama3.1-8b/results_2024-08-31T20-29-37.223117.json`  118,896 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_Dolphin3.0-Qwen2.5-0.5B/results_2025-03-03T13-08-49.788475.json`  116,188 bytes
- `C_efficiency_evolution/detailed_results/cognitivecomputations_Dolphin3.0-R1-Mistral-24B/results_2025-02-07T18-14-36.451324.json`  115,039 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_aya-23-35B/results_2024-06-17T22-14-18.057565.json`  129,037 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_aya-23-8B/results_2024-06-17T11-49-39.129219.json`  129,098 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_aya-expanse-32b/results_2024-10-24T23-13-12.725496.json`  116,057 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_aya-expanse-8b/results_2024-10-24T16-15-03.662470.json`  116,069 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_c4ai-command-r-plus/results_2024-06-20T10-16-23.151914.json`  129,112 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_c4ai-command-r-plus-08-2024/results_2024-09-20T05-28-13.938100.json`  119,890 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_c4ai-command-r-v01/results_2024-06-17T20-00-09.158840.json`  128,748 bytes
- `C_efficiency_evolution/detailed_results/CohereForAI_c4ai-command-r7b-12-2024/results_2024-12-20T13-37-10.543380.json`  127,828 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-Gemma-2b-dpo-v1.0/results_2024-08-13T04-55-19.543716.json`  119,072 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-Gemma-2b-odpo-v1.0/results_2024-08-13T02-35-20.190532.json`  119,069 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-Gemma-2b-sft-v1.0/results_2024-08-13T04-56-06.975578.json`  119,025 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-LLaMA-3-8b-dpo-v1.0/results_2024-08-13T04-36-21.322067.json`  118,953 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0/results_2024-08-13T04-44-27.328884.json`  118,972 bytes
- `C_efficiency_evolution/detailed_results/Columbia-NLP_LION-LLaMA-3-8b-sft-v1.0/results_2024-08-13T04-49-26.292140.json`  118,967 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_huihui-ai-abliterated-Qwen2.5-32B-Inst-BaseMerge-TIES/results_2024-12-07T17-16-26.181747.json`  117,009 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_huihui-ai-abliteratedV2-Qwen2.5-14B-Inst-BaseMerge-TIES/results_2024-12-07T06-29-43.941929.json`  117,028 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_Josiefied-abliteratedV4-Qwen2.5-14B-Inst-BaseMerge-TIES/results_2024-12-07T05-30-29.259046.json`  117,026 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_Rombos-Qwen2.5-7B-Inst-BaseMerge-TIES/results_2024-10-29T15-00-49.989293.json`  120,929 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_YiSM-blossom5.1-34B-SLERP/results_2024-09-02T18-34-45.787681.json`  119,137 bytes
- `C_efficiency_evolution/detailed_results/CombinHorizon_zetasepic-abliteratedV2-Qwen2.5-32B-Inst-BaseMerge-TIES/results_2024-12-21T01-37-15.623991.json`  117,021 bytes
- `C_efficiency_evolution/detailed_results/ContactDoctor_Bio-Medical-3B-CoT-012025/results_2025-01-15T09-22-59.977970.json`  114,688 bytes
- `C_efficiency_evolution/detailed_results/ContactDoctor_Bio-Medical-Llama-3-8B/results_2024-12-24T06-22-56.971273.json`  114,629 bytes
- `C_efficiency_evolution/detailed_results/CoolSpring_Qwen2-0.5B-Abyme/results_2024-09-06T09-06-54.964156.json`  118,894 bytes
- `C_efficiency_evolution/detailed_results/CoolSpring_Qwen2-0.5B-Abyme-merge2/results_2024-07-30T04-00-30.669462.json`  118,957 bytes
- `C_efficiency_evolution/detailed_results/CoolSpring_Qwen2-0.5B-Abyme-merge3/results_2024-07-30T03-41-02.675646.json`  118,991 bytes
- `C_efficiency_evolution/detailed_results/Corianas_llama-3-reactor/results_2024-08-12T17-59-58.747351.json`  118,639 bytes
- `C_efficiency_evolution/detailed_results/Corianas_Neural-Mistral-7B/results_2024-12-06T14-35-18.646555.json`  115,111 bytes
- `C_efficiency_evolution/detailed_results/Corianas_Quokka_2.7b/results_2024-12-11T18-32-14.809182.json`  114,488 bytes
- `C_efficiency_evolution/detailed_results/CortexLM_btlm-7b-base-v0.2/results_2024-08-12T22-07-03.175931.json`  118,664 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_merge_model_20250308_2/results_2025-03-08T15-11-05.567757.json`  113,497 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_merge_model_20250308_3/results_2025-03-08T15-15-23.298099.json`  113,518 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_merge_model_20250308_4/results_2025-03-08T15-21-43.946714.json`  113,486 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_SCE-2-24B/results_2025-02-05T01-14-41.928805.json`  116,267 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_SCE-3-24B/results_2025-02-05T01-07-55.745567.json`  116,295 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_T.E-8.1/results_2024-09-29T18-25-58.538578.json`  119,246 bytes
- `C_efficiency_evolution/detailed_results/Cran-May_tempmotacilla-cinerea-0308/results_2025-03-08T15-27-36.469214.json`  113,503 bytes
- `C_efficiency_evolution/detailed_results/CreitinGameplays_Llama-3.1-8B-R1-v0.1/results_2025-03-11T00-12-57.650499.json`  118,491 bytes
- `C_efficiency_evolution/detailed_results/cstr_llama3.1-8b-spaetzle-v90/results_2024-09-16T04-03-46.320177.json`  123,513 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Broca/results_2024-12-23T18-32-51.550970.json`  114,621 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Brocav3/results_2024-12-24T00-40-55.160630.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Brocav6/results_2024-12-24T00-45-48.593643.json`  114,644 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Brocav7/results_2024-12-24T00-39-18.307055.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-BrocaV9/results_2025-01-14T03-27-24.176624.json`  114,639 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Emerged/results_2024-12-20T00-16-33.822279.json`  114,635 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Emergedv3/results_2024-12-22T02-13-24.444781.json`  114,631 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-FinalMerge/results_2024-12-24T01-43-26.836129.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Hyper/results_2025-01-20T07-31-35.276448.json`  114,634 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Hyperionv3/results_2025-01-19T05-37-04.733936.json`  114,646 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Hyperionv4/results_2025-01-20T19-39-31.032174.json`  114,637 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Hyperionv5/results_2025-01-20T19-35-29.527144.json`  114,646 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-HyperMarck-dl/results_2025-02-25T03-04-33.550164.json`  113,510 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-MegaMerge-pt2/results_2024-10-25T17-28-29.915496.json`  118,614 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-MergeStock/results_2024-10-24T00-00-00.000000.json`  118,630 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-partialmergept1/results_2025-01-19T05-48-12.721431.json`  114,672 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-ReasoningMerge/results_2025-02-24T23-23-47.851116.json`  113,523 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Ultimav2/results_2025-02-05T01-39-46.826724.json`  114,638 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Unity/results_2024-12-22T02-13-12.144491.json`  114,635 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Wernicke/results_2024-10-23T19-04-06.984034.json`  118,609 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Wernicke-SFT/results_2024-11-20T02-17-53.848498.json`  117,263 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Wernicke-SLERP/results_2024-10-25T21-08-24.240304.json`  121,319 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwen2.5-14B-Wernickev3/results_2024-12-20T00-20-51.798524.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwenfinity-2.5-14B/results_2024-12-23T20-53-54.936287.json`  114,648 bytes
- `C_efficiency_evolution/detailed_results/CultriX_Qwestion-14B/results_2024-11-23T06-19-01.826337.json`  114,601 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14B/results_2024-11-20T10-20-14.065091.json`  114,623 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14B-EvolMerge/results_2024-11-27T11-01-35.329151.json`  114,641 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14B-EvolMergev1/results_2024-11-27T12-57-19.390079.json`  114,657 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14B-v5/results_2024-11-20T00-54-32.999873.json`  114,624 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14Bv1/results_2024-11-27T10-50-57.059584.json`  114,617 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14Bv2/results_2024-12-08T17-38-34.281942.json`  114,630 bytes
- `C_efficiency_evolution/detailed_results/CultriX_SeQwence-14Bv3/results_2024-11-27T21-18-32.079945.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_Llama-PLLuM-8B-base/results_2025-02-28T14-13-32.792972.json`  113,580 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_Llama-PLLuM-8B-chat/results_2025-02-28T13-53-42.976676.json`  114,698 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_PLLuM-12B-base/results_2025-02-28T14-26-10.371596.json`  113,461 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_PLLuM-12B-chat/results_2025-02-28T14-04-50.916092.json`  114,587 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_PLLuM-12B-nc-base/results_2025-02-28T14-13-34.734587.json`  113,477 bytes
- `C_efficiency_evolution/detailed_results/CYFRAGOVPL_PLLuM-12B-nc-chat/results_2025-02-28T14-03-56.285441.json`  114,591 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_AetherDrake-SFT/results_2024-12-25T20-08-51.776455.json`  114,674 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_AetherSett/results_2024-12-30T14-23-42.325851.json`  114,616 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_AetherTOT/results_2024-12-28T17-49-09.639117.json`  114,645 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_AetherUncensored/results_2025-01-13T09-52-18.924092.json`  114,685 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Cogito-MIS/results_2025-02-18T21-15-33.420905.json`  113,957 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_CogitoDistil/results_2025-01-23T10-51-52.863988.json`  117,372 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_CogitoZ/results_2025-01-08T09-03-00.911977.json`  114,623 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_CogitoZ14/results_2025-01-08T03-34-10.130534.json`  117,304 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_DocumentCogito/results_2025-01-16T02-47-58.653318.json`  114,669 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Llama3.3-70B-CogniLink/results_2025-01-14T12-59-48.589114.json`  114,665 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Llama_cot/results_2025-03-09T04-46-23.117014.json`  118,701 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_MawaredT1/results_2025-01-06T10-36-21.706918.json`  114,612 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_mini-Cogito-R1/results_2025-02-24T16-55-04.757962.json`  113,566 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_mini_Pathfinder/results_2025-01-20T21-31-43.950385.json`  117,390 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Mini_QwQ/results_2025-01-16T15-58-03.581284.json`  114,606 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_NemoR/results_2024-12-31T20-41-25.756485.json`  114,597 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_PathfinderAI/results_2024-12-25T22-39-30.466377.json`  114,630 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_PathFinderAI2.0/results_2024-12-30T20-53-06.571859.json`  114,645 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_PathFinderAi3.0/results_2024-12-31T20-18-44.352719.json`  114,647 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Phi-4-COT/results_2025-01-14T02-51-00.520335.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_PixelParse_AI/results_2024-12-29T04-31-58.660338.json`  114,680 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_RA2.0/results_2025-01-01T18-18-02.159312.json`  114,589 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_RA_Reasoner/results_2024-12-25T22-17-44.601562.json`  114,625 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_RA_Reasoner2.0/results_2024-12-29T18-48-19.037398.json`  114,640 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_ReasonTest/results_2024-12-31T20-06-33.065277.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Research_PathfinderAI/results_2025-02-24T17-46-32.815615.json`  115,909 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_SphinX/results_2024-12-31T20-16-02.475993.json`  114,606 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Sphinx2.0/results_2024-12-30T15-58-05.484286.json`  117,330 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_TinySphinx/results_2024-12-31T19-47-39.007434.json`  114,276 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_TinySphinx2.0/results_2024-12-31T19-44-34.193223.json`  114,658 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Zirel-7B-Math/results_2025-02-28T23-12-53.666730.json`  116,163 bytes
- `C_efficiency_evolution/detailed_results/Daemontatox_Zirel_1.5/results_2025-03-04T12-01-01.254825.json`  116,180 bytes
- `C_efficiency_evolution/detailed_results/Dampfinchen_Llama-3.1-8B-Ultra-Instruct/results_2024-09-02T16-31-28.146511.json`  123,534 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-10b/results_2025-01-06T22-44-16.806373.json`  114,608 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-14b-Phi-3-medium-ORPO/results_2024-08-12T10-50-02.852147.json`  118,686 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-14b-phi-4/results_2025-01-26T19-24-19.048500.json`  114,663 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-14b-phi-4-v2/results_2025-02-05T01-03-56.203278.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-3b-GRPO/results_2025-02-24T21-04-06.645485.json`  113,515 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-Llama3-8b-ORPO/results_2024-07-30T05-19-20.813390.json`  119,084 bytes
- `C_efficiency_evolution/detailed_results/Danielbrdz_Barcenas-R1-Qwen-1.5b/results_2025-01-26T19-17-19.771431.json`  114,729 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_12b-mn-dans-reasoning-test-2/results_2025-03-07T23-17-49.989635.json`  113,901 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_12b-mn-dans-reasoning-test-3/results_2025-03-10T00-55-26.687847.json`  113,918 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Dans-Instruct-CoreCurriculum-12b-ChatML/results_2024-09-05T05-25-31.840107.json`  118,698 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Dans-Instruct-Mix-8b-ChatML/results_2024-09-16T04-36-47.830370.json`  118,701 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Dans-Instruct-Mix-8b-ChatML-V0.1.0/results_2024-09-21T05-00-02.509113.json`  118,732 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Dans-Instruct-Mix-8b-ChatML-V0.1.1/results_2024-09-23T12-04-51.145255.json`  118,723 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Dans-Instruct-Mix-8b-ChatML-V0.2.0/results_2024-09-30T09-20-31.787829.json`  119,044 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_mistral-7b-test-merged/results_2024-11-27T06-47-09.799669.json`  114,983 bytes
- `C_efficiency_evolution/detailed_results/Dans-DiscountModels_Mistral-7b-v0.3-Test-E0.7/results_2024-11-17T23-39-37.800317.json`  115,010 bytes
- `C_efficiency_evolution/detailed_results/Darkknight535_OpenCrystal-12B-L3/results_2024-09-03T01-29-38.043440.json`  118,662 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepHermes-3-Llama-3-8B-Preview-16.5B-Brainstorm/results_2025-03-10T04-29-31.276484.json`  113,612 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-BlackRoot-R1-Distill-Llama-3.1-8B/results_2025-03-10T03-34-45.491804.json`  113,621 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-Grand-Horror-SMB-R1-Distill-Llama-3.1-16B/results_2025-03-10T05-04-21.696996.json`  113,666 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-MOE-4X8B-R1-Distill-Llama-3.1-Deep-Thinker-Uncensored-24B/results_2025-03-10T04-45-27.408239.json`  113,706 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-MOE-4X8B-R1-Distill-Llama-3.1-Mad-Scientist-24B/results_2025-03-10T04-48-15.064174.json`  113,693 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-R1-Distill-Qwen-25.5B-Brainstorm/results_2025-03-10T06-34-12.025015.json`  113,638 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepSeek-V2-Grand-Horror-SMB-R1-Distill-Llama-3.1-Uncensored-16.5B/results_2025-03-10T05-06-15.045531.json`  113,703 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_DeepThought-MOE-8X3B-R1-Llama-3.2-Reasoning-18B/results_2025-03-10T05-04-45.792108.json`  113,626 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Gemma-The-Writer-9B/results_2025-01-14T03-35-46.382052.json`  114,974 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Gemma-The-Writer-DEADLINE-10B/results_2025-01-14T04-00-14.444095.json`  115,318 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Gemma-The-Writer-J.GutenBerg-10B/results_2025-01-14T03-56-59.713142.json`  115,358 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Gemma-The-Writer-Mighty-Sword-9B/results_2025-01-14T03-13-25.874400.json`  114,999 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Gemma-The-Writer-N-Restless-Quill-10B-Uncensored/results_2025-01-14T03-26-20.349500.json`  115,173 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-Dark-Planet-8B/results_2024-09-14T01-22-26.219344.json`  118,622 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-DARKEST-PLANET-16.5B/results_2025-01-14T03-40-51.254747.json`  115,129 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-Jamet-12.2B-MK.V-Blackroot-Instruct/results_2024-09-11T18-30-00.046087.json`  118,688 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-Lumimaid-12.2B-v0.1-OAS-Instruct/results_2024-09-14T01-49-45.517798.json`  118,679 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-SMB-Instruct-12.2B-F32/results_2024-09-14T01-44-45.928610.json`  118,631 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-Stheno-Maid-Blackroot-Grand-HORROR-16B/results_2024-09-10T01-41-23.314602.json`  118,703 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3-Stheno-v3.2-12.2B-Instruct/results_2024-09-14T01-49-00.730479.json`  118,651 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3.1-Dark-Planet-SpinFire-Uncensored-8B/results_2025-01-13T16-26-31.175383.json`  119,612 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_L3.1-MOE-2X8B-Deepseek-DeepHermes-e32-uncensored-abliterated-13.7B/results_2025-03-10T03-23-45.418230.json`  113,710 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Qwen2.5-MOE-2X1.5B-DeepSeek-Uncensored-Censored-4B/results_2025-03-10T03-21-01.285602.json`  113,666 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Qwen2.5-MOE-2X7B-DeepSeek-Abliterated-Censored-19B/results_2025-03-10T04-01-27.449634.json`  113,676 bytes
- `C_efficiency_evolution/detailed_results/DavidAU_Qwen2.5-MOE-6x1.5B-DeepSeek-Reasoning-e32/results_2025-03-10T04-10-10.425547.json`  113,642 bytes
- `C_efficiency_evolution/detailed_results/Davidsv_SUONG-1/results_2025-02-24T16-35-20.128773.json`  113,101 bytes
- `C_efficiency_evolution/detailed_results/DavieLion_Llama-3.2-1B-SPIN-iter0/results_2024-12-29T02-42-14.122225.json`  114,314 bytes
- `C_efficiency_evolution/detailed_results/DavieLion_Llama-3.2-1B-SPIN-iter1/results_2024-12-29T07-27-12.916238.json`  114,337 bytes
- `C_efficiency_evolution/detailed_results/DavieLion_Llama-3.2-1B-SPIN-iter2/results_2024-12-29T09-28-09.319125.json`  114,327 bytes
- `C_efficiency_evolution/detailed_results/DavieLion_Llama-3.2-1B-SPIN-iter3/results_2024-12-29T06-27-51.383742.json`  114,327 bytes
- `C_efficiency_evolution/detailed_results/DavieLion_Lllma-3.2-1B/results_2024-12-27T07-42-34.986242.json`  114,297 bytes
- `C_efficiency_evolution/detailed_results/DebateLabKIT_Llama-3.1-Argunaut-1-8B-SFT/results_2025-01-02T23-42-32.363726.json`  119,603 bytes
- `C_efficiency_evolution/detailed_results/Deci_DeciLM-7B/results_2024-06-16T21-49-06.071553.json`  120,064 bytes
- `C_efficiency_evolution/detailed_results/Deci_DeciLM-7B-instruct/results_2024-06-16T21-50-32.589302.json`  120,465 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_causal_gpt2/results_2024-10-17T06-14-28.690757.json`  118,489 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_d2nwg_causal_gpt2/results_2024-10-18T01-36-00.225326.json`  118,515 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_d2nwg_causal_gpt2_v1/results_2024-10-18T17-39-39.780270.json`  118,496 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_d2nwg_Llama-3.1-8B-Instruct-v0.0/results_2024-09-12T00-17-43.652795.json`  119,083 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_Explore_Llama-3.1-8B-Inst/results_2024-09-21T17-34-15.250414.json`  123,531 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_Explore_Llama-3.2-1B-Inst/results_2024-10-07T07-57-21.365937.json`  122,732 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_Explore_Llama-3.2-1B-Inst_v0/results_2024-10-08T11-14-35.125813.json`  122,683 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_Explore_Llama-3.2-1B-Inst_v1/results_2024-10-09T10-26-35.079919.json`  122,685 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_Explore_Llama-3.2-1B-Inst_v1.1/results_2024-10-09T10-29-37.972649.json`  122,722 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_ldm_soup_Llama-3.1-8B-Inst/results_2024-09-17T16-53-14.508118.json`  119,060 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_ldm_soup_Llama-3.1-8B-Instruct-v0.0/results_2024-09-16T00-55-20.303513.json`  119,079 bytes
- `C_efficiency_evolution/detailed_results/DeepAutoAI_ldm_soup_Llama-3.1-8B-Instruct-v0.1/results_2024-09-16T11-07-45.732075.json`  119,077 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Lexora-Lite-3B/results_2024-10-20T17-04-44.753677.json`  121,272 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Lexora-Lite-3B_v2/results_2025-02-26T01-20-11.246594.json`  116,123 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Lexora-Medium-7B/results_2024-09-24T13-30-06.561197.json`  121,231 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Llama-3-8b-Ita/results_2024-08-13T04-33-17.994268.json`  119,094 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Llama-3.1-8b-ITA/results_2024-10-24T00-00-00.000000.json`  119,038 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Llama-3.1-8b-ITA/results_2025-02-13T18-27-04.338360.json`  118,636 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Llama-3.1-Distilled/results_2024-10-25T16-30-54.674107.json`  119,014 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_mergekit-ties-okvgjfz/results_2024-10-25T16-48-46.916487.json`  120,943 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2-1.5B-Ita/results_2025-02-28T20-55-59.716664.json`  113,871 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2-1.5B-Ita_v2/results_2025-03-06T07-46-27.142642.json`  113,927 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2-1.5B-Ita_v3/results_2025-03-06T09-07-19.003105.json`  113,900 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2-1.5B-Ita_v5/results_2025-03-10T13-52-00.461153.json`  113,913 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2-1.5B-Ita_v6/results_2025-03-10T16-42-48.688447.json`  116,163 bytes
- `C_efficiency_evolution/detailed_results/DeepMount00_Qwen2.5-7B-Instruct-MathCoder/results_2024-10-25T16-48-55.371157.json`  120,968 bytes
- `C_efficiency_evolution/detailed_results/deepseek-ai_deepseek-llm-7b-base/results_2024-06-17T16-08-00.757449.json`  120,034 bytes
- `C_efficiency_evolution/detailed_results/deepseek-ai_deepseek-moe-16b-base/results_2024-06-17T15-10-41.581210.json`  120,104 bytes
- `C_efficiency_evolution/detailed_results/deepseek-ai_DeepSeek-R1-Distill-Llama-8B/results_2025-01-20T21-34-30.383569.json`  116,688 bytes
- `C_efficiency_evolution/detailed_results/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/results_2025-01-20T22-52-38.383912.json`  116,706 bytes
- `C_efficiency_evolution/detailed_results/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/results_2025-01-20T22-51-25.465135.json`  116,708 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Baldur-8B/results_2024-10-06T23-26-24.662364.json`  118,639 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Control-8B/results_2024-11-25T22-58-23.311876.json`  115,206 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Control-8B-V1.1/results_2024-11-25T22-59-39.146282.json`  115,231 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Darkens-8B/results_2024-10-06T23-43-28.378749.json`  118,571 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Henbane-7b-attempt2/results_2024-10-11T04-26-22.219154.json`  118,976 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Odin-9B/results_2024-10-07T06-28-49.718610.json`  118,564 bytes
- `C_efficiency_evolution/detailed_results/Delta-Vector_Tor-8B/results_2024-10-07T05-32-42.968657.json`  118,572 bytes
- `C_efficiency_evolution/detailed_results/DevQuasar_DevQuasar-R1-Uncensored-Llama-8B/results_2025-02-24T21-08-11.864665.json`  113,605 bytes
- `C_efficiency_evolution/detailed_results/dnhkng_RYS-Llama-3-8B-Instruct/results_2024-08-12T19-09-35.366165.json`  119,125 bytes
- `C_efficiency_evolution/detailed_results/dnhkng_RYS-Llama-3-Large-Instruct/results_2024-08-13T00-58-36.953051.json`  119,110 bytes
- `C_efficiency_evolution/detailed_results/dnhkng_RYS-Llama-3.1-8B-Instruct/results_2024-09-02T22-11-07.496089.json`  123,521 bytes
- `C_efficiency_evolution/detailed_results/Dongwei_DeepSeek-R1-Distill-Qwen-7B-GRPO/results_2025-02-05T21-39-58.605765.json`  116,712 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_L3-8B-R1-WolfCore/results_2025-02-28T08-19-11.374461.json`  113,521 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_L3-8B-R1-WolfCore-V1.5-test/results_2025-03-01T06-02-01.732045.json`  113,558 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_L3-8B-WolfCore/results_2025-02-28T15-09-48.824639.json`  113,528 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MiniusLight-24B/results_2025-03-04T03-48-41.345537.json`  113,498 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MiniusLight-24B-test/results_2025-03-04T04-36-04.245444.json`  113,432 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MiniusLight-24B-v1b-test/results_2025-03-04T05-07-42.130665.json`  113,535 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MiniusLight-24B-v1c-test/results_2025-03-04T19-39-54.628918.json`  113,500 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MiniusLight-24B-v1d-test/results_2025-03-07T08-10-57.040104.json`  113,474 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-FoxFrame-test/results_2025-02-06T06-23-45.550962.json`  114,625 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-FoxFrame2-test/results_2025-02-18T20-23-38.255325.json`  113,490 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-FoxFrame3-test/results_2025-02-18T20-33-24.161176.json`  113,505 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Kakigori/results_2025-01-29T06-05-49.129207.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-LilithFrame/results_2025-01-29T06-14-21.361820.json`  114,651 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-LilithFrame-Experiment-2/results_2025-01-29T09-38-43.090670.json`  114,691 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-LilithFrame-Experiment-3/results_2025-01-30T03-11-48.420023.json`  114,659 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-LilithFrame-Experiment-4/results_2025-02-01T07-44-15.589128.json`  114,678 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-GreenSnake/results_2025-01-27T04-06-45.582696.json`  114,661 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-Nocturne/results_2025-03-09T01-56-11.380620.json`  113,524 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-Orochi/results_2025-01-28T04-52-13.925053.json`  114,666 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-Orochi-v2-Experiment/results_2025-01-28T07-33-29.878696.json`  114,551 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-Orochi-v3-Experiment/results_2025-01-28T08-45-35.014563.json`  114,702 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-Orochi-v4-Experiment/results_2025-01-28T11-41-49.522671.json`  114,632 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-WhiteSnake/results_2025-01-27T04-00-36.616405.json`  114,657 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-WhiteSnake-v2-Experiment-1/results_2025-01-29T01-43-28.326883.json`  114,678 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-WhiteSnake-v2-Experiment-2/results_2025-01-29T02-00-09.948353.json`  114,611 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-WhiteSnake-v2-Experiment-3/results_2025-01-29T01-52-03.291596.json`  114,628 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Mimicore-WhiteSnake-v2-Experiment-4/results_2025-01-29T02-06-49.274720.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-Unleashed-Twilight/results_2025-02-24T16-40-04.762595.json`  113,520 bytes
- `C_efficiency_evolution/detailed_results/DoppelReflEx_MN-12B-WolFrame/results_2025-02-01T13-32-06.626014.json`  114,639 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Again-8B-Model_Stock/results_2024-12-18T05-38-04.604621.json`  115,076 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Alita99-8B-LINEAR/results_2024-11-26T09-13-36.210728.json`  115,066 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_AnotherTest/results_2025-01-29T02-35-35.598992.json`  115,127 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire-8B-model_stock/results_2024-09-18T14-56-50.576279.json`  119,080 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_1.3-8B_model-stock/results_2024-11-06T17-49-07.924455.json`  119,569 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2-8B-Model_Stock/results_2025-01-20T06-20-55.838072.json`  115,154 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,173 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2.1-8B-Model_Stock/results_2025-01-20T18-49-22.014883.json`  115,159 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2.1-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,176 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2_ALT-8B-Model_Stock/results_2025-01-20T06-38-55.168149.json`  115,144 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2_ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,177 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2_ALT_ROW-8B-Model_Stock/results_2025-01-20T06-42-43.305849.json`  115,153 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V2_ALT_ROW-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,186 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V3-8B-Model_Stock/results_2025-01-21T22-15-56.442048.json`  115,127 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V3-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,149 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V4-8B-Model_Stock/results_2025-01-23T10-39-04.936674.json`  119,575 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V4-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,595 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V4_ALT-8B-Model_Stock/results_2025-01-23T10-44-32.891431.json`  119,580 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aspire_V4_ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,598 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Asymmetric_Linearity-8B-Model_Stock/results_2024-12-20T14-48-42.157149.json`  115,114 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Asymmetric_Linearity-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,134 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LINEAR/results_2024-09-26T03-22-46.018360.json`  119,075 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  119,092 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LORABLATED/results_2024-09-29T12-49-37.441170.json`  119,090 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LORABLATED/results_2025-02-13T18-27-04.338360.json`  119,107 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LORABLATED_ALT/results_2024-09-29T12-49-24.284544.json`  119,127 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Aurora_faustus-8B-LORABLATED_ALT/results_2025-02-13T18-27-04.338360.json`  119,129 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Autumn_Dawn-8B-LINEAR/results_2025-02-01T17-06-48.582022.json`  115,085 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Autumn_Dawn-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,104 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel-8B-LINEAR/results_2024-11-10T09-44-59.097945.json`  115,103 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,123 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel-8B-Model_Stock/results_2024-12-29T19-00-29.560316.json`  115,077 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,083 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V2-8B-Model_Stock/results_2025-01-28T19-42-28.211701.json`  115,082 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,104 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V2_ALT-8B-Model_Stock/results_2025-01-28T19-38-58.798579.json`  115,097 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V2_ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,119 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V3-8B-Model_Stock/results_2025-01-13T09-16-19.036411.json`  115,136 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BaeZel_V3-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,160 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Blunt_Edge-8B-SLERP/results_2025-01-18T02-10-03.731222.json`  115,125 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Blunt_Edge-8B-SLERP/results_2025-02-13T18-27-04.338360.json`  115,146 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BulkUp/results_2025-01-27T17-16-56.005778.json`  114,737 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_BulkUp/results_2025-02-13T18-27-04.338360.json`  114,757 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Cadence-8B-LINEAR/results_2025-02-25T20-04-11.601267.json`  113,955 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Caelid-8B-Model_Stock/results_2025-01-14T22-38-17.427185.json`  115,126 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Caelid-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,144 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Casuar-9B-Model_Stock/results_2024-12-21T13-44-19.211575.json`  115,154 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Casuar-9B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,235 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Condensed_Milk-8B-Model_Stock/results_2024-11-27T14-59-58.204848.json`  115,121 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Condensed_Milk-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,141 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_CoolerCoder-8B-LINEAR/results_2024-11-21T12-51-12.654562.json`  115,044 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_CoolerCoder-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,047 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Damasteel-8B-LINEAR/results_2024-11-06T17-50-50.122250.json`  119,554 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Damasteel-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  119,576 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Dearly_Beloved-8B-TIES/results_2024-11-22T17-19-32.707651.json`  115,242 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Dearly_Beloved-8B-TIES/results_2025-02-13T18-27-04.338360.json`  115,263 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Decayed-8B-LINEAR/results_2025-02-25T20-09-22.458828.json`  113,980 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative-8B-Model_Stock/results_2025-01-06T12-39-47.373298.json`  115,140 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,159 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V2-8B-Model_Stock/results_2025-01-07T17-53-46.866278.json`  115,123 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,142 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V2_ALT-8B-Model_Stock/results_2025-01-07T17-31-57.123136.json`  115,159 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V2_ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,179 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V3-8B-Model_Stock/results_2025-01-08T04-38-58.510524.json`  115,130 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Derivative_V3-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,163 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Elusive_Dragon_Heart-8B-LINEAR/results_2024-12-18T21-57-14.616096.json`  115,107 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Elusive_Dragon_Heart-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,111 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Emu_Eggs-9B-Model_Stock/results_2024-10-19T00-21-04.701620.json`  119,157 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Emu_Eggs-9B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,221 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Eunoia_Vespera-8B-LINEAR/results_2024-09-23T15-34-53.615095.json`  119,028 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Eunoia_Vespera-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  119,048 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_felix_dies-mistral-7B-model_stock/results_2024-10-04T20-17-02.592191.json`  118,633 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_felix_dies-mistral-7B-model_stock/results_2025-02-13T18-27-04.338360.json`  118,654 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Fu_sion_HA-8B-SLERP/results_2025-01-17T16-08-50.966537.json`  115,132 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Fu_sion_HA-8B-SLERP/results_2025-02-13T18-27-04.338360.json`  115,154 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_hakuchido-8B-MODEL_STOCK/results_2025-02-27T14-30-59.823930.json`  118,405 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Happy_New_Year-8B-Model_Stock/results_2024-12-31T13-47-19.587479.json`  115,137 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Happy_New_Year-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,154 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Heart_Stolen-8B-Model_Stock/results_2024-09-12T08-41-24.834372.json`  119,094 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Heart_Stolen-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,100 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Heart_Stolen-ALT-8B-Model_Stock/results_2024-09-14T01-53-28.879983.json`  119,295 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Heart_Stolen-ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,316 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Here_We_Go_Again-8B-SLERP/results_2024-12-31T20-17-33.938464.json`  115,153 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Here_We_Go_Again-8B-SLERP/results_2025-02-13T18-27-04.338360.json`  115,172 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_HOT_STINKING_GARBAGE/results_2025-02-03T12-25-44.299932.json`  115,058 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_HOT_STINKING_GARBAGE/results_2025-02-13T18-27-04.338360.json`  115,112 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Howdy-8B-LINEAR/results_2025-01-16T00-49-15.558585.json`  115,110 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Howdy-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,133 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_H_the_eighth-8B-LINEAR/results_2025-03-06T15-42-24.897731.json`  113,988 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ichor-8B-Model_Stock/results_2025-03-03T05-09-40.415004.json`  113,980 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ichor_1.1-8B-Model_Stock/results_2025-03-06T05-28-40.569766.json`  118,434 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Incidental-8B-Model_Stock/results_2025-01-14T00-36-17.315142.json`  115,125 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Incidental-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,145 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_inexpertus-8B-Model_Stock/results_2025-03-07T04-36-48.980204.json`  113,985 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_inexpertus_1.1-8B-LINEAR/results_2025-03-07T19-46-05.126898.json`  113,990 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_inexpertus_1.2-8B-LINEAR/results_2025-03-08T03-04-56.665051.json`  114,002 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Irina-8B-model_stock/results_2024-09-03T09-15-33.576548.json`  119,069 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Irina-8B-model_stock/results_2025-02-13T18-27-04.338360.json`  119,087 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Kindling-8B-Model_Stock/results_2025-01-26T23-31-44.510400.json`  115,105 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Kindling-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,141 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_L3.1-BaeZel-8B-Della/results_2024-11-18T07-24-11.728796.json`  115,126 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_L3.1-BaeZel-8B-Della/results_2025-02-13T18-27-04.338360.json`  115,145 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Laughing_Stock-8B-Model_Stock/results_2025-01-27T00-42-41.347898.json`  119,575 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Laughing_Stock-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,607 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Lava_Lamp-8B-SLERP/results_2025-01-18T02-12-44.217651.json`  115,025 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Lava_Lamp-8B-SLERP/results_2025-02-13T18-27-04.338360.json`  115,046 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_LemonP-8B-Model_Stock/results_2025-01-20T01-14-35.867731.json`  115,080 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_LemonP-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,100 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Lydia_of_Whiterun-8B-LINEAR/results_2025-02-26T06-05-04.586560.json`  113,999 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Machroom-3B-model_stock/results_2024-07-25T13-13-48.957746.json`  119,087 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Matryoshka-8B-LINEAR/results_2024-12-02T06-08-23.925788.json`  115,080 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Matryoshka-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,100 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Mercury_In_Retrograde-8b-Model-Stock/results_2024-12-03T20-06-26.383676.json`  115,120 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Mercury_In_Retrograde-8b-Model-Stock/results_2025-02-13T18-27-04.338360.json`  115,139 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_mergekit-nuslerp-nqzkedi/results_2025-01-29T01-28-19.123593.json`  115,147 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_mergekit-nuslerp-nqzkedi/results_2025-02-13T18-27-04.338360.json`  115,155 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy-8B-Model_Stock/results_2025-01-26T15-15-45.681745.json`  115,088 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,109 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy_ALT-8B-Model_Stock/results_2025-01-26T15-43-02.183510.json`  119,562 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy_ALT-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,569 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy_V2-8B-Model_Stock/results_2025-01-26T19-23-51.006257.json`  119,585 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minthy_V2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,607 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Minus_Penus-8B-Model_Stock/results_2025-03-07T16-10-05.679943.json`  113,999 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Morphing-8B-Model_Stock/results_2025-01-14T22-41-03.161406.json`  119,590 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Nother_One-8B-Model_Stock/results_2025-01-20T00-59-37.057005.json`  115,123 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Nother_One-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,141 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Not_Even_My_Final_Form-8B-Model_Stock/results_2025-01-14T22-41-30.063351.json`  119,625 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Not_Even_My_Final_Form-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,658 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Noxis-8B-LINEAR/results_2025-03-13T18-41-06.692706.json`  113,965 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Nullsworn-12B-LINEAR/results_2025-03-10T22-33-42.048008.json`  113,860 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Nwah-8B-Model_Stock/results_2025-02-27T09-24-09.065204.json`  113,983 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Oh_Boy-8B-LINEAR/results_2025-01-29T18-18-41.280595.json`  115,118 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Oh_Boy-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  115,136 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ONeil-model_stock-8B/results_2024-07-25T13-28-14.297521.json`  119,082 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ONeil-model_stock-8B/results_2025-02-13T18-27-04.338360.json`  119,101 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_OrangeJ-8B-Model_Stock/results_2025-01-22T00-09-52.098773.json`  115,073 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_OrangeJ-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,109 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Promissum_Mane-8B-LINEAR/results_2024-10-02T02-01-55.174584.json`  119,094 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Promissum_Mane-8B-LINEAR/results_2025-02-13T18-27-04.338360.json`  119,113 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Promissum_Mane-8B-LINEAR-lorablated/results_2024-10-02T02-26-56.406581.json`  119,108 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Promissum_Mane-8B-LINEAR-lorablated/results_2025-02-13T18-27-04.338360.json`  119,130 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_remember_to_breathe-8b-Model-Stock/results_2024-12-06T05-58-31.123091.json`  118,787 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_remember_to_breathe-8b-Model-Stock/results_2025-02-13T18-27-04.338360.json`  118,803 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_RPMash-8B-Model_Stock/results_2025-01-22T03-44-41.852688.json`  115,139 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_RPMash-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,158 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_RPMash_V3-8B-Model_Stock/results_2025-01-25T01-39-00.017938.json`  119,576 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_RPMash_V3-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,581 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Rusted_Gold-8B-LINEAR/results_2025-03-07T18-29-34.818596.json`  113,987 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Rusted_Platinum-8B-LINEAR/results_2025-03-06T22-38-56.474512.json`  113,993 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Rusted_Platinum-8B-Model_Stock/results_2025-03-06T20-43-28.942442.json`  114,003 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sellen-8B-model_stock/results_2024-09-02T17-46-47.033755.json`  119,279 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sellen-8B-model_stock/results_2025-02-13T18-27-04.338360.json`  119,299 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Something-8B-Model_Stock/results_2025-01-15T09-22-05.164431.json`  115,094 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Something-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,116 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Spring_Dusk-8B-SCE/results_2025-02-02T15-28-36.327598.json`  115,120 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Spring_Dusk-8B-SCE/results_2025-02-13T18-27-04.338360.json`  115,139 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Dawn-8B-SCE/results_2025-02-02T07-15-27.779608.json`  115,119 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Dawn-8B-SCE/results_2025-02-13T18-27-04.338360.json`  115,140 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Dusk-8B-TIES/results_2025-02-05T02-25-28.574510.json`  115,140 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Dusk-8B-TIES/results_2025-02-13T18-27-04.338360.json`  115,161 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Rain-8B-SCE/results_2025-02-03T11-17-55.442228.json`  115,123 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Rain-8B-SCE/results_2025-02-13T18-27-04.338360.json`  115,140 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Rain-8B-TIES/results_2025-02-03T09-55-50.970325.json`  115,126 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Summer_Rain-8B-TIES/results_2025-02-13T18-27-04.338360.json`  115,143 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sun-8B-Model_Stock/results_2025-01-27T15-46-13.564485.json`  115,072 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sun-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,092 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sweetened_Condensed_Milk-8B-Model_Stock/results_2024-11-27T23-51-11.357784.json`  118,808 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Sweetened_Condensed_Milk-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  118,828 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_test/results_2025-01-27T19-47-15.817165.json`  115,083 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_test/results_2025-02-13T18-27-04.338360.json`  114,899 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST02-Ignore/results_2025-02-05T16-24-42.862094.json`  115,106 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST02-Ignore/results_2025-02-13T18-27-04.338360.json`  115,124 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST03-ignore/results_2025-02-05T21-02-00.315240.json`  115,081 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST03-ignore/results_2025-02-13T18-27-04.338360.json`  115,099 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST06-ignore/results_2025-02-07T16-33-43.140590.json`  118,727 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST06-ignore/results_2025-02-13T18-27-04.338360.json`  118,748 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST07-ignore/results_2025-02-07T17-51-49.400710.json`  115,076 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST07-ignore/results_2025-02-13T18-27-04.338360.json`  115,094 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST08-ignore/results_2025-02-07T17-56-09.870835.json`  115,087 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_TEST08-ignore/results_2025-02-13T18-27-04.338360.json`  115,107 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_tests_pending-do_not_use_yet/results_2025-02-04T23-47-56.162324.json`  115,142 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_tests_pending-do_not_use_yet/results_2025-02-13T18-27-04.338360.json`  115,163 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_test_ALT/results_2025-01-28T22-34-17.675705.json`  114,757 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_test_ALT/results_2025-02-13T18-27-04.338360.json`  114,906 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Trinas_Nectar-8B-model_stock/results_2024-09-02T14-06-48.716186.json`  119,099 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Trinas_Nectar-8B-model_stock/results_2025-02-13T18-27-04.338360.json`  119,121 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_UNTESTED-VENN_1.2-8B-Model_Stock/results_2024-12-27T12-12-30.735647.json`  114,698 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_UNTESTED-VENN_1.2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  114,717 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_VENN_1.2-8B-Model_Stock/results_2024-12-28T01-33-18.984448.json`  115,090 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_VENN_1.2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,112 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Wannabe-8B-Model_Stock/results_2025-01-28T14-17-48.746318.json`  115,115 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Wannabe-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,135 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_What_A_Thrill-8B-Model_Stock/results_2025-01-16T13-47-28.929851.json`  119,582 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_What_A_Thrill-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  119,605 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter-8B-SCE/results_2025-02-03T22-31-58.226839.json`  115,087 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter-8B-SCE/results_2025-02-13T18-27-04.338360.json`  115,108 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter_Dawn-8B-TIES/results_2025-02-01T16-17-05.641334.json`  115,145 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter_Dawn-8B-TIES/results_2025-02-13T18-27-04.338360.json`  65,536 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter_Dusk-8B-TIES/results_2025-02-01T15-20-14.885144.json`  115,148 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter_Dusk-8B-TIES/results_2025-02-13T18-27-04.338360.json`  115,167 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Winter_Night-8B-Model_Stock/results_2025-02-18T19-05-03.458954.json`  113,998 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_WIP-Acacia-8B-Model_Stock/results_2024-11-29T21-25-50.060860.json`  115,093 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_WIP-Acacia-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,112 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_WIP_Damascus-8B-TIES/results_2024-10-30T03-24-17.255548.json`  119,062 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_WIP_Damascus-8B-TIES/results_2025-02-13T18-27-04.338360.json`  119,097 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Yafune-8B-Model_Stock/results_2025-02-24T16-41-29.051993.json`  113,986 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Yearn_V3-8B-Model_Stock/results_2025-01-25T07-47-31.767590.json`  115,108 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Yearn_V3-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,142 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Zelus-8B-Model_Stock/results_2025-01-15T14-06-07.680636.json`  115,074 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Zelus-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,094 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Zelus_V2-8B-Model_Stock/results_2025-01-15T23-22-25.558353.json`  115,136 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_Zelus_V2-8B-Model_Stock/results_2025-02-13T18-27-04.338360.json`  115,153 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ZEUS-8B-V17-Abliterated_ALT/results_2025-01-18T16-39-02.419840.json`  119,578 bytes
- `C_efficiency_evolution/detailed_results/DreadPoor_ZEUS-8B-V17-Abliterated_ALT/results_2025-02-13T18-27-04.338360.json`  119,597 bytes
- `C_efficiency_evolution/detailed_results/DRXD1000_Atlas-7B/results_2024-12-11T04-58-40.091093.json`  114,779 bytes
- `C_efficiency_evolution/detailed_results/DRXD1000_Phoenix-7B/results_2024-12-11T05-51-08.434177.json`  114,813 bytes
- `C_efficiency_evolution/detailed_results/DUAL-GPO_zephyr-7b-ipo-0k-15k-i1/results_2024-09-23T12-42-02.224196.json`  118,583 bytes
- `C_efficiency_evolution/detailed_results/duyhv1411_Llama-3.2-3B-en-vi/results_2025-03-06T15-14-49.975526.json`  117,765 bytes
- `C_efficiency_evolution/detailed_results/dwikitheduck_gemma-2-2b-id-inst/results_2024-11-25T00-09-52.794752.json`  115,166 bytes
- `C_efficiency_evolution/detailed_results/DZgas_GIGABATEMAN-7B/results_2024-09-18T07-24-33.032428.json`  118,553 bytes
- `C_efficiency_evolution/detailed_results/Edgerunners_meta-llama-3-8b-instruct-hf-ortho-baukit-34fail-3000total-bf16/results_2025-02-01T09-23-19.367481.json`  115,260 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_coolqwen-3b-it/results_2025-01-06T11-19-50.562735.json`  116,975 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_Gemma2-9B-it-psy10k-mental_health/results_2024-08-13T06-16-47.307542.json`  119,301 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_Gemma2-9b-it-train6/results_2024-08-13T06-07-44.550913.json`  119,236 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_qwen2.5-test-32b-it/results_2025-02-03T22-53-55.289898.json`  117,336 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_QwenQwen2.5-7B-IT/results_2025-02-01T10-51-49.455261.json`  116,972 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_rufalcon3-3b-it/results_2025-01-06T10-46-25.510674.json`  115,251 bytes
- `C_efficiency_evolution/detailed_results/ehristoforu_ruphi-4b/results_2025-01-06T10-48-51.382397.json`  114,779 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_gpt-j-6b/results_2025-02-13T18-27-04.338360.json`  120,059 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_gpt-neo-1.3B/results_2025-02-13T18-27-04.338360.json`  120,029 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_gpt-neo-125m/results_2025-02-13T18-27-04.338360.json`  118,499 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_gpt-neo-2.7B/results_2025-02-13T18-27-04.338360.json`  120,002 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_gpt-neox-20b/results_2025-02-13T18-27-04.338360.json`  120,044 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-1.4b/results_2025-02-13T18-27-04.338360.json`  114,592 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-12b/results_2025-02-13T18-27-04.338360.json`  120,080 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-160m/results_2025-02-13T18-27-04.338360.json`  120,014 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-1b/results_2025-01-27T10-10-36.877664.json`  114,511 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-2.8b/results_2025-02-13T18-27-04.338360.json`  120,005 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-410m/results_2025-02-13T18-27-04.338360.json`  120,027 bytes
- `C_efficiency_evolution/detailed_results/EleutherAI_pythia-6.9b/results_2024-06-16T22-34-46.256992.json`  120,068 bytes
- `C_efficiency_evolution/detailed_results/ell44ot_gemma-2b-def/results_2024-11-28T07-07-27.154882.json`  114,585 bytes
- `C_efficiency_evolution/detailed_results/Enno-Ai_EnnoAi-Pro-French-Llama-3-8B-v0.4/results_2024-07-22T09-45-37.259005.json`  118,822 bytes
- `C_efficiency_evolution/detailed_results/Enno-Ai_EnnoAi-Pro-Llama-3-8B/results_2025-02-13T18-27-04.338360.json`  118,832 bytes
- `C_efficiency_evolution/detailed_results/Enno-Ai_EnnoAi-Pro-Llama-3-8B-v0.3/results_2025-02-13T18-27-04.338360.json`  118,883 bytes
- `C_efficiency_evolution/detailed_results/Enno-Ai_EnnoAi-Pro-Llama-3.1-8B-v0.9/results_2025-02-13T18-27-04.338360.json`  118,883 bytes
- `C_efficiency_evolution/detailed_results/EnnoAi_EnnoAi-Pro-Llama-3.1-8B-v1.0/results_2024-09-10T15-36-07.744412.json`  118,752 bytes
- `C_efficiency_evolution/detailed_results/Epiculous_Azure_Dusk-v0.2/results_2025-02-13T18-27-04.338360.json`  118,884 bytes
- `C_efficiency_evolution/detailed_results/Epiculous_Crimson_Dawn-v0.2/results_2025-02-13T18-27-04.338360.json`  118,939 bytes
- `C_efficiency_evolution/detailed_results/Epiculous_NovaSpark/results_2025-02-13T18-27-04.338360.json`  119,021 bytes
- `C_efficiency_evolution/detailed_results/Epiculous_Violet_Twilight-v0.2/results_2024-10-24T00-00-00.000000.json`  118,930 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Athene-codegemma-2-7b-it-alpaca-v1.2/results_2024-09-02T14-11-04.110306.json`  118,655 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-12B-v1.2/results_2025-02-13T18-27-04.338360.json`  118,626 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1-8B-Philos/results_2025-02-13T18-27-04.338360.json`  118,722 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.01-8B-Philos/results_2025-02-13T18-27-04.338360.json`  118,713 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.03-8B-Philos/results_2025-02-13T18-27-04.338360.json`  118,718 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.04-8B-Philos/results_2024-09-14T01-13-51.381155.json`  118,693 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.06-8B-Philos-dpo/results_2024-09-14T01-20-38.820444.json`  118,715 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.07-8B-Philos-Math/results_2025-02-13T18-27-04.338360.json`  118,758 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.08-8B-C-R1-KTO-Reflection/results_2025-02-13T18-27-04.338360.json`  118,751 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Alpaca-Llama3.1.08-8B-Philos-C-R1/results_2024-09-14T02-25-21.957551.json`  118,710 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Llama-3.1-8B-Philos-Reflection/results_2025-02-13T18-27-04.338360.json`  118,734 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-MathMistral-Nemo-Base-2407-v2dpo/results_2024-09-02T16-12-06.252706.json`  119,587 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-math/results_2024-10-15T15-17-27.675353.json`  118,778 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.005-128K-code-COT/results_2025-02-13T18-27-04.338360.json`  118,785 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI2_Fireball-Phi-3-medium-4k-inst-Philos/results_2024-09-21T11-05-45.666210.json`  119,034 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Alpaca-Llama3.1-8B/results_2024-09-13T23-51-55.057359.json`  118,637 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Athena-gemma-2-2b-it/results_2025-02-13T18-27-04.338360.json`  118,534 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Athena-gemma-2-2b-it-Philos/results_2025-02-13T18-27-04.338360.json`  119,176 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Athene-codegemma-2-7b-it-alpaca-v1.3/results_2024-09-12T00-22-14.676123.json`  118,658 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_DeepPhi-3.5-mini-instruct/results_2025-02-28T03-49-49.665483.json`  113,465 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_DeepThinkers-Phi4/results_2025-03-01T23-25-47.678589.json`  114,032 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_FineLlama3.1-8B-Instruct/results_2024-09-06T12-17-19.771656.json`  118,690 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-12B/results_2025-02-13T18-27-04.338360.json`  118,553 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-12B-v1.13a-philosophers/results_2024-09-10T07-39-57.330192.json`  118,635 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Alpaca-Llama-3.1-8B-Philos-DPO-200/results_2024-09-16T11-50-18.226636.json`  118,724 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Alpaca-Llama3.1.07-8B-Philos-Math-KTO-beta/results_2025-02-13T18-27-04.338360.json`  123,638 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Alpaca-Llama3.1.08-8B-Philos-C-R2/results_2025-02-13T18-27-04.338360.json`  118,702 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-0.001-128K-auto/results_2025-02-13T18-27-04.338360.json`  114,781 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K/results_2025-02-13T18-27-04.338360.json`  118,769 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code/results_2024-10-07T10-02-11.286074.json`  118,761 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds/results_2025-02-13T18-27-04.338360.json`  119,709 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto/results_2024-10-29T22-09-32.822876.json`  128,446 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.004-128K-code-COT/results_2024-10-11T07-28-43.923029.json`  118,768 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.004-128K-code-ds-auto/results_2025-02-13T18-27-04.338360.json`  124,513 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.1-8B-Instruct-Math/results_2024-09-24T02-54-03.656817.json`  118,668 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Meta-Llama-3.2-8B-Instruct-agent-003-128k-code-DPO/results_2024-10-09T10-38-32.448780.json`  118,765 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-Mistral-Nemo-Base-2407-v1-DPO2/results_2025-02-13T18-27-04.338360.json`  118,651 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Fireball-R1-Llama-3.1-8B/results_2025-02-19T13-46-24.133657.json`  123,239 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Llama-3.2-3B-Agent007-Coder/results_2025-02-13T18-27-04.338360.json`  118,673 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Mistral-Nemo-Instruct-12B-Philosophy-Math/results_2025-02-13T18-27-04.338360.json`  118,695 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_OpenReasoner-Llama-3.2-3B-rs1.0/results_2025-02-18T16-15-11.292536.json`  123,272 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Polypsyche-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto-divergent/results_2024-12-20T17-48-24.829103.json`  124,525 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Polypsyche-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto-Empathy/results_2025-02-13T18-27-04.338360.json`  124,548 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Polypsyche-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto-Logic/results_2024-12-20T17-44-58.771039.json`  124,522 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.1-CoT-RE1-NMT/results_2025-01-29T06-39-48.331687.json`  119,569 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.1-CoT-RE1-NMT-V2-ORPO/results_2025-02-13T18-27-04.338360.json`  114,762 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.2-1B-Instruct-v1.2/results_2025-02-13T18-27-04.338360.json`  119,532 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.2-1B-Instruct-v1.3/results_2025-02-13T18-27-04.338360.json`  119,561 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.2-3B-Math-Instruct-RE1/results_2025-02-13T18-27-04.338360.json`  119,529 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_Reasoning-Llama-3.2-3B-Math-Instruct-RE1-ORPO/results_2025-02-13T18-27-04.338360.json`  119,634 bytes
- `C_efficiency_evolution/detailed_results/EpistemeAI_ReasoningCore-3B-T1_1/results_2025-02-13T18-27-04.338360.json`  119,561 bytes
- `C_efficiency_evolution/detailed_results/Eric111_CatunaMayo/results_2025-02-13T18-27-04.338360.json`  118,560 bytes
- `C_efficiency_evolution/detailed_results/Eric111_CatunaMayo-DPO/results_2025-02-13T18-27-04.338360.json`  118,569 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Chocolatine-3B-Instruct-DPO-Revised-Ties/results_2025-02-13T18-27-04.338360.json`  118,681 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Chocolatine-3B-Instruct-DPO-Revised-Ties-v2/results_2025-02-13T18-27-04.338360.json`  118,684 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Herplete-LLM-Llama-3.1-8b/results_2025-02-13T18-27-04.338360.json`  118,656 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Herplete-LLM-Llama-3.1-8b-Ties/results_2024-10-17T17-17-33.503999.json`  118,630 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Qwen2.5-7B-della-test/results_2025-02-13T18-27-04.338360.json`  117,124 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Qwen2.5-Coder-7B-Instruct-Ties/results_2025-02-13T18-27-04.338360.json`  118,648 bytes
- `C_efficiency_evolution/detailed_results/Etherll_Replete-LLM-V3-Llama-3.1-8b/results_2025-02-13T18-27-04.338360.json`  119,344 bytes
- `C_efficiency_evolution/detailed_results/Etherll_SuperHermes/results_2025-02-13T18-27-04.338360.json`  118,621 bytes
- `C_efficiency_evolution/detailed_results/Eurdem_Defne-llama3.1-8B/results_2024-08-12T18-16-51.931979.json`  119,052 bytes
- `C_efficiency_evolution/detailed_results/EVA-UNIT-01_EVA-Qwen2.5-14B-v0.2/results_2025-02-13T18-27-04.338360.json`  114,632 bytes
- `C_efficiency_evolution/detailed_results/EVA-UNIT-01_EVA-Qwen2.5-72B-v0.2/results_2024-11-28T01-23-33.473911.json`  114,963 bytes
- `C_efficiency_evolution/detailed_results/ewre324_Thinker-Qwen2.5-0.5B-Instruct-Reasoning/results_2025-01-08T03-30-46.210771.json`  114,628 bytes
- `C_efficiency_evolution/detailed_results/facebook_opt-1.3b/results_2024-06-17T13-53-58.344286.json`  119,985 bytes
- `C_efficiency_evolution/detailed_results/facebook_opt-30b/results_2024-06-17T17-08-46.250569.json`  120,029 bytes
- `C_efficiency_evolution/detailed_results/failspy_Meta-Llama-3-70B-Instruct-abliterated-v3.5/results_2024-09-02T23-54-59.562583.json`  119,133 bytes
- `C_efficiency_evolution/detailed_results/failspy_Phi-3-medium-4k-instruct-abliterated-v3/results_2024-07-30T04-20-55.435914.json`  118,972 bytes
- `C_efficiency_evolution/detailed_results/FallenMerick_Chewy-Lemon-Cookie-11B/results_2025-02-13T18-27-04.338360.json`  118,666 bytes
- `C_efficiency_evolution/detailed_results/Felladrin_Llama-160M-Chat-v1/results_2024-07-25T09-54-15.757000.json`  118,489 bytes
- `C_efficiency_evolution/detailed_results/Felladrin_Minueza-32M-UltraChat/results_2025-02-13T18-27-04.338360.json`  118,539 bytes
- `C_efficiency_evolution/detailed_results/fhai50032_Unaligned-Thinker-PHI-4/results_2025-01-17T21-33-15.880671.json`  114,665 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_Chocolatine-Fusion-14B/results_2025-02-13T18-27-04.338360.json`  49,152 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_L3-8B/results_2025-02-13T18-27-04.338360.json`  119,523 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_Phi-4-RRStock/results_2025-02-05T16-09-19.318994.json`  114,607 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_Q-Small-3B/results_2025-02-13T18-27-04.338360.json`  117,276 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_QwQ-Buddy-32B-Alpha/results_2025-02-13T18-27-04.338360.json`  114,651 bytes
- `C_efficiency_evolution/detailed_results/FINGU-AI_RomboUltima-32B/results_2025-02-13T18-27-04.338360.json`  114,632 bytes
- `C_efficiency_evolution/detailed_results/flammenai_Mahou-1.2a-mistral-7B/results_2024-08-13T08-47-02.760508.json`  118,627 bytes
- `C_efficiency_evolution/detailed_results/flammenai_Mahou-1.5-mistral-nemo-12B/results_2024-10-07T04-22-59.176979.json`  118,967 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_100k_fineweb_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit/results_2025-02-13T18-27-04.338360.json`  117,326 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_10k_continued_pretraining_Phi-3-mini-4k-instruct_Unsloth_merged_16bit/results_2024-11-22T17-10-30.745797.json`  115,278 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_10k_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit/results_2024-11-25T18-35-35.437059.json`  117,136 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_40k_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit/results_2024-11-25T20-12-20.428213.json`  117,138 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_83k_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit/results_2025-02-13T18-27-04.338360.json`  117,251 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1000k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,434 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1000k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,523 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1000k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,514 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1200k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,440 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1200k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,452 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1200k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,527 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1400k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,467 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1400k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,537 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_1400k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,514 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_200k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,507 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_200k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,490 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_400k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,466 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_400k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,514 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_400k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,496 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_600k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,424 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_600k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,510 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_600k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,510 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_800k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,437 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_800k_fineweb_uncovai_human_removed/results_2025-02-13T18-27-04.338360.json`  114,540 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2-135M_pretrained_800k_fineweb_uncovai_selected/results_2025-02-13T18-27-04.338360.json`  114,516 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_smollm2_pretrained_200k_fineweb/results_2025-02-13T18-27-04.338360.json`  114,413 bytes
- `C_efficiency_evolution/detailed_results/FlofloB_test_continued_pretraining_Phi-3-mini-4k-instruct_Unsloth_merged_16bit/results_2024-11-21T23-44-40.262072.json`  115,258 bytes
- `C_efficiency_evolution/detailed_results/fluently-lm_Llama-TI-8B/results_2024-12-07T20-23-25.091439.json`  114,666 bytes
- `C_efficiency_evolution/detailed_results/fluently-lm_Llama-TI-8B-Instruct/results_2025-01-16T20-38-11.058635.json`  119,563 bytes
- `C_efficiency_evolution/detailed_results/fluently-sets_FalconThink3-10B-IT/results_2024-12-29T19-40-47.934216.json`  116,590 bytes
- `C_efficiency_evolution/detailed_results/freewheelin_free-evo-qwen72b-v0.8-re/results_2024-10-05T14-32-52.988483.json`  118,630 bytes
- `C_efficiency_evolution/detailed_results/FuJhen_ft-openhermes-25-mistral-7b-irca-dpo-pairs/results_2024-09-14T01-13-21.040681.json`  118,739 bytes
- `C_efficiency_evolution/detailed_results/FuJhen_mistral-instruct-7B-DPO/results_2024-09-14T01-09-08.984631.json`  122,811 bytes
- `C_efficiency_evolution/detailed_results/FuJhen_mistral_7b_v0.1_structedData_e2e/results_2024-09-18T13-35-58.119143.json`  118,478 bytes
- `C_efficiency_evolution/detailed_results/FuJhen_mistral_7b_v0.1_structedData_viggo/results_2025-02-13T18-27-04.338360.json`  118,620 bytes
- `C_efficiency_evolution/detailed_results/fulim_FineLlama-3.1-8B/results_2024-12-18T04-47-10.406336.json`  114,678 bytes
- `C_efficiency_evolution/detailed_results/FuseAI_FuseChat-7B-v2.0/results_2025-02-13T18-27-04.338360.json`  114,639 bytes
- `C_efficiency_evolution/detailed_results/FuseAI_FuseChat-Llama-3.1-8B-Instruct/results_2025-01-07T07-01-34.828380.json`  115,122 bytes
- `C_efficiency_evolution/detailed_results/FuseAI_FuseChat-Llama-3.2-3B-Instruct/results_2025-02-18T20-26-21.852409.json`  114,001 bytes
- `C_efficiency_evolution/detailed_results/FuseAI_FuseChat-Qwen-2.5-7B-Instruct/results_2025-02-13T18-27-04.338360.json`  117,133 bytes
- `C_efficiency_evolution/detailed_results/GalrionSoftworks_MagnusIntellectus-12B-v1/results_2025-02-13T18-27-04.338360.json`  119,115 bytes
- `C_efficiency_evolution/detailed_results/GalrionSoftworks_MN-LooseCannon-12B-v1/results_2025-02-13T18-27-04.338360.json`  118,934 bytes
- `C_efficiency_evolution/detailed_results/gaverfraxz_Meta-Llama-3.1-8B-Instruct-HalfAbliterated-DELLA/results_2024-09-28T01-30-12.053735.json`  118,711 bytes
- `C_efficiency_evolution/detailed_results/gaverfraxz_Meta-Llama-3.1-8B-Instruct-HalfAbliterated-TIES/results_2024-09-19T08-18-53.778310.json`  119,119 bytes
- `C_efficiency_evolution/detailed_results/gbueno86_Brinebreath-Llama-3.1-70B/results_2024-09-03T03-48-42.670969.json`  119,055 bytes
- `C_efficiency_evolution/detailed_results/GenVRadmin_AryaBhatta-GemmaOrca-2-Merged/results_2025-02-06T10-54-15.198935.json`  114,567 bytes
- `C_efficiency_evolution/detailed_results/GenVRadmin_AryaBhatta-GemmaOrca-Merged/results_2025-02-06T10-43-35.410744.json`  114,545 bytes
- `C_efficiency_evolution/detailed_results/GenVRadmin_AryaBhatta-GemmaUltra-Merged/results_2025-02-13T18-27-04.338360.json`  114,622 bytes
- `C_efficiency_evolution/detailed_results/GenVRadmin_llama38bGenZ_Vikas-Merged/results_2025-02-06T10-43-30.868879.json`  114,644 bytes
- `C_efficiency_evolution/detailed_results/gmonsoon_gemma2-9b-sahabatai-v1-instruct-BaseTIES/results_2024-11-20T02-23-13.828344.json`  115,344 bytes
- `C_efficiency_evolution/detailed_results/godlikehhd_ifd_new_correct_all_sample_2500_qwen/results_2024-12-31T09-00-40.891610.json`  114,702 bytes
- `C_efficiency_evolution/detailed_results/godlikehhd_ifd_new_qwen_2500/results_2024-12-31T04-09-10.782723.json`  114,647 bytes
- `C_efficiency_evolution/detailed_results/godlikehhd_qwen-2.5-1.5b-cherry/results_2024-12-29T15-12-23.640073.json`  114,649 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_j.o.s.i.e.v4o-1.5b-dpo-stage1-v1/results_2025-02-13T18-27-04.338360.json`  122,945 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_josie-3b-v6.0/results_2025-02-13T18-27-04.338360.json`  119,073 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_josie-7b-v6.0/results_2025-02-13T18-27-04.338360.json`  117,629 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_josie-7b-v6.0-step2000/results_2025-02-13T18-27-04.338360.json`  117,656 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1/results_2025-02-13T18-27-04.338360.json`  119,343 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1/results_2024-09-28T10-58-22.921064.json`  122,836 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v2/results_2024-09-28T10-55-49.897632.json`  122,818 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v3/results_2024-09-28T10-55-18.813510.json`  122,799 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-14B-Instruct-abliterated-v4/results_2025-02-13T18-27-04.338360.json`  122,723 bytes
- `C_efficiency_evolution/detailed_results/Goekdeniz-Guelmez_Josiefied-Qwen2.5-7B-Instruct-abliterated-v2/results_2025-02-13T18-27-04.338360.json`  121,943 bytes
- `C_efficiency_evolution/detailed_results/google_codegemma-1.1-2b/results_2024-08-13T09-01-51.651456.json`  118,440 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-1.1-2b-it/results_2024-06-17T09-00-56.334963.json`  120,597 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-2-27b-it/results_2024-08-26T14-00-19.311199.json`  119,067 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-2-2b/results_2024-07-29T16-38-17.664954.json`  118,490 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-2-2b-it/results_2024-07-29T16-25-50.870458.json`  118,968 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-2-2b-jpn-it/results_2024-10-11T13-51-38.420715.json`  118,544 bytes
- `C_efficiency_evolution/detailed_results/google_gemma-7b-it/results_2024-06-16T15-52-57.979737.json`  77,814 bytes
- `C_efficiency_evolution/detailed_results/google_recurrentgemma-9b/results_2024-07-04T22-34-03.508539.json`  118,254 bytes
- `C_efficiency_evolution/detailed_results/google_recurrentgemma-9b-it/results_2024-07-05T20-12-40.169257.json`  118,914 bytes
- `C_efficiency_evolution/detailed_results/GoToCompany_gemma2-9b-cpt-sahabatai-v1-instruct/results_2024-11-21T16-06-20.952173.json`  114,686 bytes
- `C_efficiency_evolution/detailed_results/GoToCompany_llama3-8b-cpt-sahabatai-v1-instruct/results_2025-02-13T18-27-04.338360.json`  115,204 bytes
- `C_efficiency_evolution/detailed_results/GreenNode_GreenNode-small-9B-it/results_2025-02-13T18-27-04.338360.json`  119,031 bytes
- `C_efficiency_evolution/detailed_results/grimjim_DeepSauerHuatuoSkywork-R1-o1-Llama-3.1-8B/results_2025-02-01T09-55-26.563110.json`  114,736 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Gigantes-v2-gemma2-9b-it/results_2024-12-29T16-40-54.211241.json`  114,568 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Gigantes-v3-gemma2-9b-it/results_2024-12-29T22-04-51.345542.json`  114,580 bytes
- `C_efficiency_evolution/detailed_results/grimjim_HuatuoSkywork-o1-Llama-3.1-8B/results_2025-01-06T12-09-54.318043.json`  114,692 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Llama-3-Instruct-8B-SPPO-Iter3-SimPO-merge/results_2024-08-12T16-20-28.765484.json`  118,731 bytes
- `C_efficiency_evolution/detailed_results/grimjim_llama-3-Nephilim-v2-8B/results_2024-09-18T15-30-41.248227.json`  118,637 bytes
- `C_efficiency_evolution/detailed_results/grimjim_llama-3-Nephilim-v2.1-8B/results_2024-09-18T15-31-10.573478.json`  118,649 bytes
- `C_efficiency_evolution/detailed_results/grimjim_llama-3-Nephilim-v3-8B/results_2024-09-02T21-29-15.009684.json`  118,618 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Llama-3.1-Bonsaikraft-8B-Instruct/results_2025-01-25T04-57-46.509667.json`  114,704 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Llama-Nephilim-Metamorphosis-v2-8B/results_2025-01-02T22-56-26.602076.json`  114,721 bytes
- `C_efficiency_evolution/detailed_results/grimjim_Magot-v2-Gemma2-8k-9B/results_2024-12-30T14-09-02.992490.json`  114,265 bytes
- `C_efficiency_evolution/detailed_results/GritLM_GritLM-7B-KTO/results_2025-02-13T18-27-04.338360.json`  118,983 bytes
- `C_efficiency_evolution/detailed_results/GritLM_GritLM-8x7B-KTO/results_2025-02-13T18-27-04.338360.json`  119,002 bytes
- `C_efficiency_evolution/detailed_results/Groq_Llama-3-Groq-8B-Tool-Use/results_2025-02-13T18-27-04.338360.json`  115,139 bytes
- `C_efficiency_evolution/detailed_results/Gryphe_Pantheon-RP-1.0-8b-Llama-3/results_2025-02-13T18-27-04.338360.json`  119,012 bytes
- `C_efficiency_evolution/detailed_results/Gryphe_Pantheon-RP-1.5-12b-Nemo/results_2025-02-13T18-27-04.338360.json`  118,990 bytes
- `C_efficiency_evolution/detailed_results/Gryphe_Pantheon-RP-1.6-12b-Nemo/results_2025-02-13T18-27-04.338360.json`  118,936 bytes
- `C_efficiency_evolution/detailed_results/Gryphe_Pantheon-RP-1.6-12b-Nemo-KTO/results_2025-02-13T18-27-04.338360.json`  119,275 bytes
- `C_efficiency_evolution/detailed_results/Gryphe_Pantheon-RP-Pure-1.6.2-22b-Small/results_2025-02-13T18-27-04.338360.json`  122,836 bytes
- `C_efficiency_evolution/detailed_results/Gunulhona_Gemma-Ko-Merge/results_2024-10-29T06-01-37.065452.json`  119,075 bytes
- `C_efficiency_evolution/detailed_results/Gunulhona_Gemma-Ko-Merge-PEFT/results_2025-02-13T18-27-04.338360.json`  118,890 bytes
- `C_efficiency_evolution/detailed_results/gz987_qwen2.5-7b-cabs-v0.4/results_2025-02-18T15-56-13.630751.json`  116,189 bytes
- `C_efficiency_evolution/detailed_results/Hastagaras_Llama-3.1-Jamet-8B-MK.I/results_2024-11-19T23-48-47.605970.json`  115,122 bytes
- `C_efficiency_evolution/detailed_results/Hastagaras_Zabuza-8B-Llama-3.1/results_2024-11-06T19-01-58.072743.json`  115,095 bytes
- `C_efficiency_evolution/detailed_results/HelpingAI_Cipher-20B/results_2025-02-13T18-27-04.338360.json`  117,467 bytes
- `C_efficiency_evolution/detailed_results/HelpingAI_Priya-10B/results_2025-02-13T18-27-04.338360.json`  119,268 bytes
- `C_efficiency_evolution/detailed_results/HelpingAI_Priya-3B/results_2025-02-13T18-27-04.338360.json`  118,182 bytes
- `C_efficiency_evolution/detailed_results/HeraiHench_Double-Down-Qwen-Math-7B/results_2025-02-13T18-27-04.338360.json`  114,380 bytes
- `C_efficiency_evolution/detailed_results/HeraiHench_Marge-Qwen-Math-7B/results_2025-02-13T18-27-04.338360.json`  114,421 bytes
- `C_efficiency_evolution/detailed_results/HeraiHench_Phi-4-slerp-ReasoningRP-14B/results_2025-01-29T13-31-01.353343.json`  114,357 bytes
- `C_efficiency_evolution/detailed_results/HiroseKoichi_Llama-Salad-4x8B-V3/results_2025-02-13T18-27-04.338360.json`  119,108 bytes
- `C_efficiency_evolution/detailed_results/HoangHa_Pensez-Llama3.1-8B/results_2025-02-24T17-22-21.367738.json`  115,066 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_FalconSlerp2-7B/results_2024-12-23T14-32-27.002312.json`  114,591 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_FalconSlerp3-7B/results_2024-12-23T14-31-56.117330.json`  114,609 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_Gemma2atlas-27B/results_2024-12-02T06-39-45.019030.json`  114,629 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_Gemma2Crono-27B/results_2024-12-04T04-36-27.482016.json`  114,622 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_Llama-Hermes-slerp-8B/results_2024-12-23T15-03-34.457004.json`  114,646 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_LlamaStock-8B/results_2025-01-08T14-58-46.780812.json`  114,669 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_Qwen2.5-HomerSlerp-7B/results_2024-12-07T17-49-11.339082.json`  114,643 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenSlerp-14B/results_2025-01-08T15-42-43.646675.json`  114,630 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenSlerp-7B/results_2024-12-23T12-45-04.291866.json`  114,647 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenSlerp2-14B/results_2025-01-08T15-19-32.893685.json`  114,631 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenSparse-7B/results_2025-01-08T14-15-37.706847.json`  114,292 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenStock-1.7B/results_2025-01-08T14-37-07.779979.json`  114,661 bytes
- `C_efficiency_evolution/detailed_results/hotmailuser_QwenStock1-14B/results_2025-01-08T15-38-22.540908.json`  114,621 bytes
- `C_efficiency_evolution/detailed_results/HPAI-BSC_Llama3-Aloe-8B-Alpha/results_2024-10-29T15-05-18.647108.json`  119,082 bytes
- `C_efficiency_evolution/detailed_results/HPAI-BSC_Llama3.1-Aloe-Beta-8B/results_2024-11-06T18-55-21.201805.json`  114,678 bytes
- `C_efficiency_evolution/detailed_results/HPAI-BSC_Qwen2.5-Aloe-Beta-7B/results_2024-12-18T05-23-22.292212.json`  117,255 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceH4_zephyr-7b-alpha/results_2025-02-13T18-27-04.338360.json`  78,759 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceH4_zephyr-7b-beta/results_2025-02-13T18-27-04.338360.json`  120,588 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceH4_zephyr-7b-gemma-v0.1/results_2024-06-16T15-58-33.412754.json`  77,627 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceH4_zephyr-orpo-141b-A35b-v0.1/results_2025-02-13T18-27-04.338360.json`  120,777 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-1.7B/results_2025-02-13T18-27-04.338360.json`  118,634 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-1.7B-Instruct/results_2025-02-13T18-27-04.338360.json`  118,662 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-135M/results_2025-02-13T18-27-04.338360.json`  118,625 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-135M-Instruct/results_2025-02-13T18-27-04.338360.json`  118,620 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-360M/results_2025-02-13T18-27-04.338360.json`  118,570 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM-360M-Instruct/results_2025-02-13T18-27-04.338360.json`  118,663 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-1.7B/results_2025-02-13T18-27-04.338360.json`  114,633 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-1.7B-Instruct/results_2025-02-13T18-27-04.338360.json`  115,067 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-135M/results_2025-02-13T18-27-04.338360.json`  112,059 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-135M-Instruct/results_2025-02-13T18-27-04.338360.json`  114,594 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-360M/results_2025-02-13T18-27-04.338360.json`  114,619 bytes
- `C_efficiency_evolution/detailed_results/HuggingFaceTB_SmolLM2-360M-Instruct/results_2025-02-13T18-27-04.338360.json`  115,066 bytes
- `C_efficiency_evolution/detailed_results/huihui-ai_DeepSeek-R1-Distill-Qwen-14B-abliterated-v2/results_2025-01-29T05-41-09.046383.json`  116,733 bytes
- `C_efficiency_evolution/detailed_results/HumanLLMs_Humanish-LLama3-8B-Instruct/results_2025-02-13T18-27-04.338360.json`  119,226 bytes
- `C_efficiency_evolution/detailed_results/HumanLLMs_Humanish-Mistral-Nemo-Instruct-2407/results_2024-10-07T02-41-47.679146.json`  119,162 bytes
- `C_efficiency_evolution/detailed_results/HumanLLMs_Humanish-Qwen2.5-7B-Instruct/results_2025-02-13T18-27-04.338360.json`  118,760 bytes
- `C_efficiency_evolution/detailed_results/IDEA-CCNL_Ziya-LLaMA-13B-v1/results_2025-02-13T18-27-04.338360.json`  120,863 bytes
- `C_efficiency_evolution/detailed_results/ifable_gemma-2-Ifable-9B/results_2024-09-27T17-11-40.930590.json`  118,498 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama31_8B_en_emo_v4/results_2025-02-28T20-05-25.513225.json`  118,471 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama32_1B_en_emo_v1/results_2025-01-14T15-27-35.797200.json`  119,362 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama32_3B_en_emo_2000_stp/results_2025-03-07T12-22-43.575886.json`  118,389 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama32_3B_en_emo_300_stp/results_2025-03-07T10-26-09.297087.json`  118,380 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama32_3B_en_emo_5000_stp/results_2025-03-08T13-11-39.019377.json`  118,405 bytes
- `C_efficiency_evolution/detailed_results/iFaz_llama32_3B_en_emo_v3/results_2025-02-28T06-09-36.993032.json`  118,394 bytes
- `C_efficiency_evolution/detailed_results/IlyaGusev_gemma-2-2b-it-abliterated/results_2025-02-13T18-27-04.338360.json`  115,231 bytes
- `C_efficiency_evolution/detailed_results/IlyaGusev_gemma-2-9b-it-abliterated/results_2025-02-13T18-27-04.338360.json`  114,967 bytes
- `C_efficiency_evolution/detailed_results/Infinirc_Infinirc-Llama3-8B-2G-Release-v1.0/results_2025-02-13T18-27-04.338360.json`  118,701 bytes
- `C_efficiency_evolution/detailed_results/INSAIT-Institute_BgGPT-Gemma-2-27B-IT-v1.0/results_2025-02-13T18-27-04.338360.json`  114,660 bytes
- `C_efficiency_evolution/detailed_results/Intel_neural-chat-7b-v3/results_2025-02-13T18-27-04.338360.json`  120,105 bytes
- `C_efficiency_evolution/detailed_results/Intel_neural-chat-7b-v3-1/results_2025-02-13T18-27-04.338360.json`  120,080 bytes
- `C_efficiency_evolution/detailed_results/Intel_neural-chat-7b-v3-2/results_2025-02-13T18-27-04.338360.json`  120,103 bytes
- `C_efficiency_evolution/detailed_results/Intel_neural-chat-7b-v3-3/results_2025-02-13T18-27-04.338360.json`  49,152 bytes
- `C_efficiency_evolution/detailed_results/IntervitensInc_internlm2_5-20b-llamafied/results_2024-11-11T12-07-29.151211.json`  114,640 bytes
- `C_efficiency_evolution/detailed_results/Invalid-Null_PeiYangMe-0.5/results_2025-02-13T18-27-04.338360.json`  114,308 bytes
- `C_efficiency_evolution/detailed_results/Invalid-Null_PeiYangMe-0.7/results_2025-02-13T18-27-04.338360.json`  114,573 bytes
- `C_efficiency_evolution/detailed_results/Isaak-Carter_Josiefied-Qwen2.5-7B-Instruct-abliterated/results_2024-09-21T14-45-46.127361.json`  121,142 bytes
- `C_efficiency_evolution/detailed_results/Isaak-Carter_Josiefied-Qwen2.5-7B-Instruct-abliterated-v2/results_2025-02-13T18-27-04.338360.json`  121,139 bytes
- `C_efficiency_evolution/detailed_results/Isaak-Carter_JOSIEv4o-8b-stage1-v4/results_2025-02-13T18-27-04.338360.json`  118,730 bytes
- `C_efficiency_evolution/detailed_results/J-LAB_Thynk_orpo/results_2025-02-13T18-27-04.338360.json`  118,597 bytes
- `C_efficiency_evolution/detailed_results/JackFram_llama-160m/results_2025-02-13T18-27-04.338360.json`  114,394 bytes
- `C_efficiency_evolution/detailed_results/JackFram_llama-68m/results_2025-02-13T18-27-04.338360.json`  114,373 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Casual-Magnum-34B/results_2025-02-13T18-27-04.338360.json`  118,598 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Inf-Silent-Kunoichi-v0.1-2x7B/results_2025-02-13T18-27-04.338360.json`  118,653 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Inf-Silent-Kunoichi-v0.2-2x7B/results_2025-02-13T18-27-04.338360.json`  118,657 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Proto-Athena-4x7B/results_2025-02-13T18-27-04.338360.json`  118,608 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Proto-Athena-v0.2-4x7B/results_2025-02-13T18-27-04.338360.json`  118,616 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Proto-Harpy-Blazing-Light-v0.1-2x7B/results_2025-02-13T18-27-04.338360.json`  118,657 bytes
- `C_efficiency_evolution/detailed_results/Jacoby746_Proto-Harpy-Spark-v0.1-7B/results_2025-02-13T18-27-04.338360.json`  118,621 bytes
- `C_efficiency_evolution/detailed_results/jayasuryajsk_Qwen2.5-3B-reasoner/results_2025-02-18T19-05-53.560318.json`  113,508 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-DPO-1epoch/results_2025-02-13T18-27-04.338360.json`  115,055 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-DPO-5epoch/results_2024-12-26T05-35-35.448012.json`  114,998 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-eDPO-1epoch/results_2025-02-13T18-27-04.338360.json`  115,096 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-eDPO-5epoch/results_2024-12-26T05-25-52.840251.json`  115,036 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-IRPO-1epoch/results_2024-12-26T08-46-55.267826.json`  115,048 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen-0.5B-IRPO-5epoch/results_2025-02-13T18-27-04.338360.json`  115,028 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-Instruct-SFT/results_2025-02-13T18-27-04.338360.json`  117,393 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-Instruct-SFT-DPO-1epoch_v1/results_2024-12-26T20-30-29.551698.json`  117,420 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-Instruct-SFT-IRPO-1epoch_v1/results_2025-02-13T18-27-04.338360.json`  117,364 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-Instruct-SFT-MDPO-1epoch_v1/results_2024-12-26T20-27-02.598498.json`  117,393 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT/results_2025-02-13T18-27-04.338360.json`  114,676 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-4/results_2024-12-28T09-28-08.864023.json`  114,484 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-4-2ep/results_2025-02-13T18-27-04.338360.json`  114,465 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-4-3ep/results_2024-12-29T15-29-47.615567.json`  114,541 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-4-5ep/results_2025-02-13T18-27-04.338360.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-5/results_2025-02-13T18-27-04.338360.json`  114,612 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-5-2ep/results_2024-12-30T00-38-42.524746.json`  114,561 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-5-3ep/results_2024-12-29T15-09-58.644085.json`  114,570 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-1e-5-5ep/results_2025-02-13T18-27-04.338360.json`  114,629 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-4/results_2024-12-28T09-33-58.487922.json`  114,307 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-4-2ep/results_2024-12-30T07-52-34.968240.json`  114,304 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-4-3ep/results_2024-12-29T15-33-00.016651.json`  114,508 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-4-5ep/results_2024-12-30T05-17-07.829474.json`  114,571 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5/results_2025-02-13T18-27-04.338360.json`  114,607 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep/results_2024-12-30T00-39-36.534132.json`  114,496 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_3e-7-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,102 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-6-1ep_0alp_5lam/results_2025-01-06T11-15-18.885108.json`  117,225 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-6-2ep_0alp_5lam/results_2025-01-06T11-17-26.136733.json`  117,006 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-6-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,137 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-7-1ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,280 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-7-2ep_0alp_5lam/results_2025-01-03T00-17-30.079517.json`  116,989 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPOP_5e-7-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,270 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_1e-6-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,288 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_1e-6-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,329 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_1e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,304 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_1e-7-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,295 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_1e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,149 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_2e-6-1ep_0alp_0lam/results_2025-01-02T01-33-30.695249.json`  117,254 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_2e-6-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,291 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_2e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,297 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_3e-6-1ep_0alp_0lam/results_2025-01-02T01-42-47.105367.json`  117,168 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_3e-6-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,300 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_3e-6-3ep_0alp_0lam/results_2025-01-03T06-59-06.233458.json`  117,178 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_3e-7-3ep_0alp_0lam/results_2025-01-08T03-28-31.813523.json`  117,180 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-6-1ep_0alp_0lam/results_2025-01-02T01-51-21.319158.json`  117,240 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-6-2ep_0alp_0lam/results_2025-01-02T01-55-00.773602.json`  117,191 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,145 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-7_1ep_0alp_0lam/results_2025-01-01T02-45-59.875419.json`  117,175 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-7_2ep_0alp_0lam/results_2025-01-01T01-15-16.839837.json`  117,006 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_5e-7_3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,310 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_7e-7_1ep_0alp_0lam/results_2025-01-01T02-41-24.803003.json`  117,002 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_7e-7_2ep_0alp_0lam/results_2025-01-01T02-36-55.162456.json`  117,173 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-DPO_7e-7_3ep_0alp_0lam/results_2025-01-01T02-37-22.197996.json`  117,180 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_1e-7-1ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,153 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_1e-7-2ep_1alp_0lam/results_2025-01-07T11-36-11.134717.json`  117,181 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_1e-7-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,160 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_3e-7-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,147 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-6-1ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,283 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-6-2ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,149 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-6-3ep_1alp_0lam/results_2025-01-06T12-01-03.647111.json`  117,188 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-7-1ep_1alp_0lam/results_2025-01-02T23-19-17.617577.json`  117,001 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-7-2ep_1alp_0lam/results_2025-01-02T22-57-37.160585.json`  117,006 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-IRPO_5e-7-3ep_1alp_0lam/results_2025-01-02T23-09-51.306955.json`  116,984 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_0.5_1e-7-1ep_0alp_0lam/results_2025-01-07T09-49-10.164062.json`  117,143 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_0.5_1e-7-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,301 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_0.5_1e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,325 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_1e-6-3ep_0alp_0lam/results_2025-01-01T04-51-16.407163.json`  117,214 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_1e-6_1ep_0alp_0lam/results_2025-01-01T04-36-32.829954.json`  117,168 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_1e-6_2ep_0alp_0lam/results_2025-01-01T04-32-33.474966.json`  117,188 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_2e-6-3ep_0alp_0lam/results_2025-01-01T04-56-12.225886.json`  117,004 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_2e-6_1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,335 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_2e-6_2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,305 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_3e-6-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,272 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_3e-6-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,150 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_3e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,274 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-6-1ep_0alp_0lam/results_2025-01-02T01-15-44.734990.json`  117,225 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-6-2ep_0alp_0lam/results_2025-01-02T01-13-13.802530.json`  116,989 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,148 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,317 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-7_1ep_0alp_0lam/results_2024-12-31T08-23-05.016642.json`  117,194 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_5e-7_2ep_0alp_0lam/results_2024-12-31T08-20-06.028753.json`  117,006 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_7e-7-3ep_0alp_0lam/results_2025-01-01T02-50-51.686412.json`  117,003 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_7e-7_1ep_0alp_0lam/results_2025-01-01T03-01-18.107162.json`  117,173 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-2ep-MDPO_7e-7_2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,330 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-3ep/results_2025-02-13T18-27-04.338360.json`  114,619 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep/results_2024-12-30T07-10-54.556076.json`  114,570 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_5e-7_3ep_0alp_0lam/results_2024-12-31T02-00-42.876223.json`  117,218 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_5e-7_3ep_0alp_0lam_1ep/results_2024-12-31T02-18-06.890327.json`  117,225 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_5e-7_3ep_0alp_0lam_2ep/results_2025-02-13T18-27-04.338360.json`  117,289 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_7e-7_3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,271 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_7e-7_3ep_0alp_0lam_1ep/results_2025-02-13T18-27-04.338360.json`  117,148 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-2e-5-5ep-MDPO_7e-7_3ep_0alp_0lam_2ep/results_2025-02-13T18-27-04.338360.json`  117,140 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-5e-5/results_2025-02-13T18-27-04.338360.json`  114,446 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-5e-5-2ep/results_2024-12-30T00-39-05.683860.json`  114,507 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-5e-5-3ep/results_2024-12-29T15-05-41.185341.json`  114,481 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-5e-5-5ep/results_2025-02-13T18-27-04.338360.json`  114,630 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-7e-5/results_2025-02-13T18-27-04.338360.json`  114,418 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-7e-5-2ep/results_2025-02-13T18-27-04.338360.json`  114,462 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-7e-5-3ep/results_2025-02-13T18-27-04.338360.json`  114,600 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-7e-5-5ep/results_2025-02-13T18-27-04.338360.json`  114,484 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-DPO-1epoch_v1/results_2025-02-13T18-27-04.338360.json`  117,237 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen2.5-0.5B-SFT-MDPO-1epoch_v1/results_2025-02-13T18-27-04.338360.json`  117,300 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-cDPO_5e-7-3ep_0vpo_const_0.1/results_2025-02-24T22-44-46.252795.json`  116,202 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-cDPO_5e-7-3ep_0vpo_const_0.3/results_2025-02-18T17-55-04.123975.json`  116,180 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_1e-6-3ep_0alp_5lam/results_2025-01-14T02-50-52.528378.json`  116,940 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_1e-7-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,202 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-6-1ep_0alp_5lam/results_2025-01-09T03-58-26.030948.json`  117,233 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-6-2ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,246 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-6-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,244 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-7-1ep_0alp_5lam/results_2025-01-15T05-30-28.203370.json`  117,139 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-7-2ep_0alp_5lam/results_2025-01-15T05-21-45.094032.json`  117,113 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_3e-7-3ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,220 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_5e-7-1ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,096 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_5e-7-2ep_0alp_5lam/results_2025-02-13T18-27-04.338360.json`  117,232 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPOP_5e-7-3ep_0alp_5lam/results_2025-01-09T03-09-40.868283.json`  117,136 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_1e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,110 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_1e-7-3ep_0alp_0lam/results_2025-01-13T18-49-20.675984.json`  116,949 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-6-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,260 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-6-2ep_0alp_0lam/results_2025-01-09T03-35-02.133914.json`  117,156 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-6-3ep_0alp_0lam/results_2025-01-07T23-53-04.755874.json`  117,202 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-7-1ep_0alp_0lam/results_2025-01-15T09-12-32.791967.json`  117,125 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-7-2ep_0alp_0lam/results_2025-01-15T09-11-14.790371.json`  116,945 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_3e-7-3ep_0alp_0lam/results_2025-01-15T09-23-39.196530.json`  116,943 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_5e-7-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,081 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_5e-7-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,053 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-DPO_5e-7-3ep_0alp_0lam/results_2025-01-08T03-45-44.132699.json`  117,126 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IPO_5e-7-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,101 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IPO_5e-7-3ep_0alp_0lam/results_2025-01-26T04-48-22.284565.json`  116,955 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_1e-6-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,096 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_1e-7-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,216 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_3e-6-1ep_1alp_0lam/results_2025-01-09T04-15-28.904556.json`  117,124 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_3e-6-2ep_1alp_0lam/results_2025-01-09T04-04-45.876414.json`  116,951 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_3e-6-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,263 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_3e-7-1ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,072 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_3e-7-3ep_1alp_0lam/results_2025-01-15T09-24-06.937216.json`  116,946 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_5e-7-1ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,093 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_5e-7-2ep_1alp_0lam/results_2025-01-09T03-26-15.034504.json`  116,945 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-IRPO_5e-7-3ep_1alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,077 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.1_3e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,244 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.1_5e-7-3ep_0alp_0lam/results_2025-01-21T07-13-13.514593.json`  117,139 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.3_3e-6-3ep_0alp_0lam/results_2025-01-09T04-48-35.371892.json`  116,962 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.3_5e-7-3ep_0alp_0lam/results_2025-01-09T04-44-39.692037.json`  117,165 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_1e-5-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,242 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_3e-7-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,257 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_3e-7-2ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,272 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_3e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,242 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_4e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,103 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_6e-6-3ep_0alp_0lam/results_2025-01-13T15-45-01.205704.json`  117,147 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_7e-6-3ep_0alp_0lam/results_2025-01-13T17-57-25.419354.json`  116,954 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.5_7e-7-3ep_0alp_0lam/results_2025-01-14T01-52-42.526523.json`  117,123 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.7_3e-6-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,088 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.7_5e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,081 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-MDPO_0.9_5e-7-3ep_0alp_0lam/results_2025-01-13T17-52-55.439417.json`  117,146 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-rDPO_3e-6-1ep_0vpo_const_0.1/results_2025-02-27T10-38-23.212605.json`  116,201 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-rDPO_5e-7-3ep_0vpo_const_0.3/results_2025-02-18T18-02-25.746599.json`  116,166 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VDPO_5e-7-1ep_0alp_0lam/results_2025-01-25T09-34-19.451187.json`  117,167 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VDPO_5e-7-1ep_10vpo_const/results_2025-02-25T00-37-19.926047.json`  116,158 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VDPO_5e-7-1ep_3vpo_const/results_2025-02-25T00-38-16.358599.json`  116,135 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VDPO_5e-7-3ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,229 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VIPO_5e-7-1ep_0alp_0lam/results_2025-02-13T18-27-04.338360.json`  117,102 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VIPO_5e-7-1ep_30vpo_const/results_2025-02-25T03-20-46.678124.json`  116,193 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VIPO_5e-7-1ep_3vpo_const/results_2025-02-24T17-43-57.079384.json`  116,183 bytes
- `C_efficiency_evolution/detailed_results/JayHyeon_Qwen_0.5-VIPO_5e-7-3ep_0alp_0lam/results_2025-01-26T01-45-44.854883.json`  116,927 bytes
- `C_efficiency_evolution/detailed_results/jebish7_gemma-2-9b-it/results_2025-02-06T20-01-35.038625.json`  114,537 bytes
- `C_efficiency_evolution/detailed_results/jebish7_Llama-3-Nanda-10B-Chat/results_2025-02-18T16-16-25.862303.json`  113,575 bytes
- `C_efficiency_evolution/detailed_results/jebish7_Llama-3.1-8B-Instruct/results_2025-02-06T12-17-02.420246.json`  114,655 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-minperplexity-2/results_2024-12-03T20-09-20.636372.json`  115,248 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v0.9/results_2024-11-13T03-54-51.081772.json`  114,690 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v1.0/results_2024-11-14T12-55-04.681817.json`  117,433 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v1.1/results_2024-11-10T05-48-20.418763.json`  117,602 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v1.2/results_2024-11-10T05-48-59.875913.json`  114,691 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v1.3/results_2024-11-10T05-52-36.690886.json`  114,689 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-nerd-uncensored-v1.5/results_2024-11-20T01-39-50.321433.json`  114,689 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-olm-v1.2/results_2025-01-01T22-43-57.626039.json`  114,646 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-olm-v1.3/results_2025-01-18T04-41-41.249302.json`  114,638 bytes
- `C_efficiency_evolution/detailed_results/jeffmeloy_Qwen2.5-7B-olm-v1.5/results_2025-02-27T09-43-53.906040.json`  113,528 bytes
- `C_efficiency_evolution/detailed_results/jiangxinyang-shanda_Homer-LLama3-8B/results_2024-11-10T09-58-15.357580.json`  115,348 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-bert-f1-beta10-gamma0.3-lr1.0e-6-1minus-rerun/results_2025-02-13T18-27-04.338360.json`  119,278 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-bert_f1-beta10-gamma0.3-lr1.0e-6-scale-log/results_2024-09-23T10-55-46.640885.json`  119,253 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-bert_p-beta10-gamma0.3-lr1.0e-6-scale-log/results_2025-02-13T18-27-04.338360.json`  119,250 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-bleu-beta0.1-no-length-scale-gamma0.4/results_2024-09-09T07-20-40.829244.json`  119,121 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-rouge2-beta10-1minus-gamma0.3-rerun/results_2025-02-13T18-27-04.338360.json`  119,244 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-rouge2-beta10-gamma0.3-lr1.0e-6-scale-log/results_2025-02-13T18-27-04.338360.json`  119,231 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_llama-3-8b-instruct-gapo-v2-rougeL-beta10-gamma0.3-lr1.0e-6-scale-log/results_2024-09-23T10-57-23.954689.json`  119,215 bytes
- `C_efficiency_evolution/detailed_results/Jimmy19991222_Llama-3-Instruct-8B-SimPO-v0.2/results_2025-02-13T18-27-04.338360.json`  119,152 bytes
- `C_efficiency_evolution/detailed_results/jlzhou_Qwen2.5-3B-Infinity-Instruct-0625/results_2025-02-07T16-35-38.330845.json`  117,231 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.1-gamma-0.01/results_2024-07-22T11-05-35.403491.json`  118,736 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.1-gamma-0.1/results_2024-07-25T22-26-52.820949.json`  118,730 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.5-gamma-0.01/results_2024-07-18T12-27-23.979317.json`  118,375 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.5-gamma-0.1/results_2024-07-25T17-44-56.707499.json`  118,728 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.7-gamma-0.01/results_2024-07-22T11-15-24.947259.json`  118,366 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs-density-0.7-gamma-0.1/results_2024-07-22T10-53-11.821902.json`  118,698 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.1-gamma-0.01/results_2024-07-22T10-53-56.880012.json`  118,743 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.5-gamma-0.01/results_2024-07-26T05-39-04.636359.json`  118,651 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.5-gamma-0.1/results_2024-07-25T17-23-23.363454.json`  118,718 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.7-gamma-0.01/results_2024-07-25T21-50-59.490636.json`  118,635 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.9-gamma-0.01/results_2024-07-25T21-10-19.778206.json`  118,385 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_breadcrumbs_ties-density-0.9-gamma-0.1/results_2024-07-22T11-17-36.167672.json`  118,737 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_linear/results_2024-07-26T05-49-40.770735.json`  118,651 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_ties-density-0.1/results_2024-07-26T04-03-34.116430.json`  118,683 bytes
- `C_efficiency_evolution/detailed_results/johnsutor_Llama-3-8B-Instruct_ties-density-0.7/results_2024-07-26T03-41-46.435458.json`  118,677 bytes
- `C_efficiency_evolution/detailed_results/Joseph717171_Hermes-3-Llama-3.1-8B_TIES_with_Base_Embeds_Initialized_to_Special_Instruct_Toks_dtypeF32/results_2024-10-29T05-27-54.968559.json`  119,241 bytes
- `C_efficiency_evolution/detailed_results/Joseph717171_Llama-3.1-SuperNova-8B-Lite_TIES_with_Base/results_2025-02-13T18-27-04.338360.json`  119,110 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_Cinder-Phi-2-V1-F16-gguf/results_2025-02-13T18-27-04.338360.json`  118,698 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_Differential-Attention-Liquid-Metal-Tinyllama/results_2025-02-13T18-27-04.338360.json`  114,936 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_TinyLlama-Cinder-Agent-v1/results_2025-02-13T18-27-04.338360.json`  119,071 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_Tinyllama-STEM-Cinder-Agent-v1/results_2025-02-13T18-27-04.338360.json`  115,078 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_TinyLlama-v1.1-Cinders-World/results_2025-02-13T18-27-04.338360.json`  119,045 bytes
- `C_efficiency_evolution/detailed_results/Josephgflowers_TinyLlama_v1.1_math_code-world-test-1/results_2025-02-13T18-27-04.338360.json`  118,657 bytes
- `C_efficiency_evolution/detailed_results/JungZoona_T3Q-Qwen2.5-14B-Instruct-1M-e3/results_2025-03-13T11-19-17.541040.json`  116,131 bytes
- `C_efficiency_evolution/detailed_results/JungZoona_T3Q-qwen2.5-14b-v1.0-e3/results_2025-03-13T18-54-06.038301.json`  116,110 bytes
- `C_efficiency_evolution/detailed_results/Junhoee_Qwen-Megumin/results_2025-02-13T18-27-04.338360.json`  117,103 bytes
- `C_efficiency_evolution/detailed_results/kayfour_T3Q-Qwen2.5-7B-it-KOR-Safe/results_2025-02-24T23-12-05.080927.json`  113,522 bytes
- `C_efficiency_evolution/detailed_results/keeeeenw_MicroLlama/results_2024-09-16T04-12-27.701485.json`  118,180 bytes
- `C_efficiency_evolution/detailed_results/kevin009_llamaRAGdrama/results_2024-08-13T01-41-28.966642.json`  120,223 bytes
- `C_efficiency_evolution/detailed_results/Khetterman_DarkAtom-12B-v3/results_2025-02-13T18-27-04.338360.json`  118,843 bytes
- `C_efficiency_evolution/detailed_results/Khetterman_Kosmos-8B-v1/results_2025-02-13T18-27-04.338360.json`  114,669 bytes
- `C_efficiency_evolution/detailed_results/khoantap_llama-3-8b-stock-merge/results_2025-01-21T20-42-27.642353.json`  114,663 bytes
- `C_efficiency_evolution/detailed_results/khoantap_llama-linear-0.5-0.5-1-merge/results_2025-01-21T16-21-35.674922.json`  114,697 bytes
- `C_efficiency_evolution/detailed_results/khoantap_llama-linear-0.5-1-0.5-merge/results_2025-01-21T16-53-28.419235.json`  114,699 bytes
- `C_efficiency_evolution/detailed_results/khoantap_llama-slerp-merge/results_2025-01-21T16-55-28.551903.json`  114,676 bytes
- `C_efficiency_evolution/detailed_results/khulaifi95_Llama-3.1-8B-Reason-Blend-888k/results_2024-12-27T08-01-29.233950.json`  119,608 bytes
- `C_efficiency_evolution/detailed_results/Kimargin_GPT-NEO-1.3B-wiki/results_2025-02-13T18-27-04.338360.json`  118,625 bytes
- `C_efficiency_evolution/detailed_results/KingNish_qwen-1b-continued/results_2025-03-07T09-37-53.465122.json`  113,418 bytes
- `C_efficiency_evolution/detailed_results/KingNish_qwen-1b-continued-v2.1/results_2025-03-08T10-12-52.903270.json`  113,440 bytes
- `C_efficiency_evolution/detailed_results/KingNish_qwen-1b-continued-v2.2/results_2025-03-09T05-05-28.063314.json`  113,487 bytes
- `C_efficiency_evolution/detailed_results/KingNish_Qwen2.5-0.5b-Test-ft/results_2024-09-29T12-44-22.226822.json`  118,540 bytes
- `C_efficiency_evolution/detailed_results/KingNish_Reasoning-Llama-3b-v0.1/results_2024-10-26T03-53-11.772801.json`  119,111 bytes
- `C_efficiency_evolution/detailed_results/kms7530_chemeng_llama-3-8b-Instruct-bnb-4bit_24_1_100_1/results_2024-10-15T06-58-05.271446.json`  119,173 bytes
- `C_efficiency_evolution/detailed_results/kms7530_chemeng_qwen-math-7b_24_1_100_1_nonmath/results_2024-11-22T02-26-03.950276.json`  117,407 bytes
- `C_efficiency_evolution/detailed_results/Kquant03_CognitiveFusion2-4x7B-BF16/results_2025-02-13T18-27-04.338360.json`  120,270 bytes
- `C_efficiency_evolution/detailed_results/Kquant03_L3-Pneuma-8B/results_2025-02-13T18-27-04.338360.json`  118,653 bytes
- `C_efficiency_evolution/detailed_results/Krystalan_DRT-o1-14B/results_2025-02-13T18-27-04.338360.json`  114,628 bytes
- `C_efficiency_evolution/detailed_results/Krystalan_DRT-o1-7B/results_2025-02-13T18-27-04.338360.json`  114,655 bytes
- `C_efficiency_evolution/detailed_results/KSU-HW-SEC_Llama3-70b-SVA-FT-1415/results_2024-09-14T11-15-52.495807.json`  118,603 bytes
- `C_efficiency_evolution/detailed_results/KSU-HW-SEC_Llama3-70b-SVA-FT-500/results_2025-02-13T18-27-04.338360.json`  118,656 bytes
- `C_efficiency_evolution/detailed_results/KSU-HW-SEC_Llama3-70b-SVA-FT-final/results_2024-09-14T11-20-10.542367.json`  118,604 bytes
- `C_efficiency_evolution/detailed_results/KSU-HW-SEC_Llama3.1-70b-SVA-FT-1000step/results_2024-09-14T13-17-42.155979.json`  118,638 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralExperiment-7b-MagicCoder-v7.5/results_2025-02-13T18-27-04.338360.json`  120,304 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralLLaMa-3-8b-DT-v0.1/results_2024-09-18T14-44-45.615531.json`  118,640 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralLLaMa-3-8b-ORPO-v0.3/results_2025-02-13T18-27-04.338360.json`  118,886 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralSynthesis-7B-v0.1/results_2025-02-13T18-27-04.338360.json`  118,619 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralSynthesis-7B-v0.3/results_2025-02-13T18-27-04.338360.json`  118,642 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_NeuralSynthesis-7b-v0.4-slerp/results_2025-02-13T18-27-04.338360.json`  118,632 bytes
- `C_efficiency_evolution/detailed_results/Kukedlc_Qwen-2.5-7b-Spanish-o1-CoT/results_2024-12-04T02-31-08.583409.json`  114,667 bytes
- `C_efficiency_evolution/detailed_results/Kumar955_Hemanth-llm/results_2025-02-13T18-27-04.338360.json`  118,567 bytes
- `C_efficiency_evolution/detailed_results/L-RAGE_3_PRYMMAL-ECE-7B-SLERP-V1/results_2025-02-13T18-27-04.338360.json`  45,805 bytes
- `C_efficiency_evolution/detailed_results/Lambent_qwen2.5-reinstruct-alternate-lumen-14B/results_2024-09-28T11-34-19.772653.json`  118,658 bytes
- `C_efficiency_evolution/detailed_results/Langboat_Mengzi3-8B-Chat/results_2025-02-13T18-27-04.338360.json`  119,193 bytes
- `C_efficiency_evolution/detailed_results/langgptai_Qwen-las-v0.1/results_2024-09-14T05-01-27.615839.json`  118,966 bytes
- `C_efficiency_evolution/detailed_results/langgptai_qwen1.5-7b-chat-sa-v0.1/results_2024-09-14T04-40-26.591950.json`  119,031 bytes
- `C_efficiency_evolution/detailed_results/lars1234_Mistral-Small-24B-Instruct-2501-writer/results_2025-03-07T11-12-43.340060.json`  113,534 bytes
- `C_efficiency_evolution/detailed_results/LEESM_llama-2-7b-hf-lora-oki100p/results_2024-11-10T03-46-21.542307.json`  114,586 bytes
- `C_efficiency_evolution/detailed_results/LEESM_llama-2-7b-hf-lora-oki10p/results_2024-11-10T03-44-42.609036.json`  114,661 bytes
- `C_efficiency_evolution/detailed_results/LEESM_llama-3-8b-bnb-4b-kowiki231101/results_2025-02-13T18-27-04.338360.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/LEESM_llama-3-Korean-Bllossom-8B-trexlab-oki10p/results_2025-02-13T18-27-04.338360.json`  114,711 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-9B/results_2024-09-03T02-22-52.690100.json`  118,457 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-Advanced-9B/results_2024-10-02T04-48-56.751637.json`  118,417 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-v3j-9B/results_2024-10-09T12-11-13.469254.json`  118,367 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-v4-Advanced-9B/results_2024-10-15T17-45-28.399685.json`  118,615 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-v4b-9B/results_2024-10-23T19-06-41.963096.json`  118,588 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Gemma-2-Ataraxy-v4c-9B/results_2024-10-16T19-20-43.025260.json`  118,579 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_llama-3-NeuralMahou-8b/results_2024-08-13T08-55-19.395372.json`  118,924 bytes
- `C_efficiency_evolution/detailed_results/lemon07r_Llama-3-RedMagic4-8B/results_2024-07-25T13-29-35.436402.json`  118,890 bytes
- `C_efficiency_evolution/detailed_results/LenguajeNaturalAI_leniachat-gemma-2b-v0/results_2024-09-03T08-54-22.794959.json`  119,084 bytes
- `C_efficiency_evolution/detailed_results/LenguajeNaturalAI_leniachat-qwen2-1.5B-v0/results_2024-09-30T10-56-03.778577.json`  119,004 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_CheckPoint_A/results_2025-02-13T18-27-04.338360.json`  115,549 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_CheckPoint_B/results_2025-02-13T18-27-04.338360.json`  115,543 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_CheckPoint_C/results_2025-02-13T18-27-04.338360.json`  115,560 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_LCARS_AI_001/results_2025-02-13T18-27-04.338360.json`  118,536 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_LCARS_AI_1x4_003_SuperAI/results_2025-02-13T18-27-04.338360.json`  118,612 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_LCARS_AI_StarTrek_Computer/results_2025-02-13T18-27-04.338360.json`  118,662 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_LCARS_TOP_SCORE/results_2025-02-13T18-27-04.338360.json`  118,619 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_Mixtral_AI_SwahiliTron_7b/results_2025-02-13T18-27-04.338360.json`  120,216 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWebAI_Human_AGI/results_2025-02-13T18-27-04.338360.json`  118,588 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWebAI_Human_AGI_001/results_2025-02-13T18-27-04.338360.json`  114,681 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_CyberTron_Ultra_7b/results_2025-02-13T18-27-04.338360.json`  118,604 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAGI_001_M2/results_2025-02-13T18-27-04.338360.json`  114,653 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAGI_002/results_2025-02-13T18-27-04.338360.json`  114,685 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_001/results_2025-02-13T18-27-04.338360.json`  118,585 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_006/results_2025-02-13T18-27-04.338360.json`  114,592 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_007/results_2025-02-13T18-27-04.338360.json`  114,693 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_009_CHAT/results_2025-02-13T18-27-04.338360.json`  114,673 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_010_CHAT/results_2025-02-13T18-27-04.338360.json`  114,663 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_011_INSTRUCT/results_2025-02-13T18-27-04.338360.json`  114,680 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_011_INSTRUCT_ML/results_2025-02-13T18-27-04.338360.json`  114,713 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_011_INSTRUCT_ML_r1/results_2025-02-13T18-27-04.338360.json`  114,721 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_012_INSTRUCT_IA/results_2025-02-13T18-27-04.338360.json`  114,689 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_012_INSTRUCT_MX/results_2025-02-13T18-27-04.338360.json`  114,679 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_012_INSTRUCT_XA/results_2025-02-13T18-27-04.338360.json`  114,706 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_RP/results_2025-02-13T18-27-04.338360.json`  114,639 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_AI_HumanAI_TextVision/results_2025-02-13T18-27-04.338360.json`  114,656 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_HumanAI_M1/results_2025-02-13T18-27-04.338360.json`  118,626 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_HumanAI_M2/results_2025-02-13T18-27-04.338360.json`  118,636 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer_SpydazWeb_HumanAI_M3/results_2025-02-13T18-27-04.338360.json`  118,571 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_12/results_2025-02-13T18-27-04.338360.json`  118,544 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_14/results_2025-02-13T18-27-04.338360.json`  119,072 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_BIBLE_002/results_2025-02-13T18-27-04.338360.json`  118,597 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_ChatML_002/results_2025-02-13T18-27-04.338360.json`  119,308 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_ChatQA/results_2025-02-13T18-27-04.338360.json`  118,401 bytes
- `C_efficiency_evolution/detailed_results/LeroyDyer__Spydaz_Web_AI_ChatQA_003/results_2025-02-13T18-27-04.338360.json`  118,476 bytes
- `C_efficiency_evolution/detailed_results/LGAI-EXAONE_EXAONE-3.0-7.8B-Instruct/results_2025-02-13T18-27-04.338360.json`  118,947 bytes
- `C_efficiency_evolution/detailed_results/LGAI-EXAONE_EXAONE-3.5-2.4B-Instruct/results_2025-02-13T18-27-04.338360.json`  115,062 bytes
- `C_efficiency_evolution/detailed_results/LGAI-EXAONE_EXAONE-3.5-32B-Instruct/results_2025-02-13T18-27-04.338360.json`  115,043 bytes
- `C_efficiency_evolution/detailed_results/LGAI-EXAONE_EXAONE-3.5-7.8B-Instruct/results_2025-02-13T18-27-04.338360.json`  115,041 bytes
- `C_efficiency_evolution/detailed_results/lightblue_suzume-llama-3-8B-multilingual/results_2024-08-12T10-41-49.479020.json`  119,113 bytes
- `C_efficiency_evolution/detailed_results/lightblue_suzume-llama-3-8B-multilingual-orpo-borda-half/results_2024-07-25T14-17-45.167820.json`  119,162 bytes
- `C_efficiency_evolution/detailed_results/lightblue_suzume-llama-3-8B-multilingual-orpo-borda-top25/results_2024-07-25T14-18-43.997470.json`  119,161 bytes
- `C_efficiency_evolution/detailed_results/LightningRodLabs_Flashlight-v1.0/results_2025-02-13T18-27-04.338360.json`  115,195 bytes
- `C_efficiency_evolution/detailed_results/LightningRodLabs_Flashlight-v1.1/results_2025-02-13T18-27-04.338360.json`  115,116 bytes
- `C_efficiency_evolution/detailed_results/LightningRodLabs_Flashlight-v1.2/results_2025-02-13T18-27-04.338360.json`  117,095 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-2B-SLERP-V1/results_2025-02-13T18-27-04.338360.json`  114,655 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-2B-SLERP-V2/results_2025-02-13T18-27-04.338360.json`  114,628 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-7B-SLERP/results_2025-02-13T18-27-04.338360.json`  112,158 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-7B-SLERP-V1/results_2025-02-13T18-27-04.338360.json`  118,315 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-7B-SLERP-V2/results_2025-02-13T18-27-04.338360.json`  118,315 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_2_PRYMMAL-ECE-7B-SLERP-V3/results_2025-02-13T18-27-04.338360.json`  118,373 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_PRYMMAL-ECE-1B-SLERP-V1/results_2025-02-13T18-27-04.338360.json`  118,606 bytes
- `C_efficiency_evolution/detailed_results/Lil-R_PRYMMAL-ECE-7B-SLERP-V8/results_2025-02-13T18-27-04.338360.json`  118,326 bytes
- `C_efficiency_evolution/detailed_results/LilRg_10PRYMMAL-3B-slerp/results_2025-02-13T18-27-04.338360.json`  118,622 bytes
- `C_efficiency_evolution/detailed_results/LilRg_ECE-1B-merge-PRYMMAL/results_2025-02-13T18-27-04.338360.json`  118,640 bytes
- `C_efficiency_evolution/detailed_results/LilRg_ECE_Finetunning/results_2025-02-13T18-27-04.338360.json`  118,646 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-6B-slerp/results_2025-02-13T18-27-04.338360.json`  118,222 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-ECE-7B-SLERP-V3/results_2025-02-13T18-27-04.338360.json`  118,365 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-ECE-7B-SLERP-V4/results_2025-02-13T18-27-04.338360.json`  118,364 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-ECE-7B-SLERP-V5/results_2025-02-13T18-27-04.338360.json`  118,360 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-ECE-7B-SLERP-V6/results_2025-02-13T18-27-04.338360.json`  118,357 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-ECE-7B-SLERP-V7/results_2025-02-13T18-27-04.338360.json`  118,365 bytes
- `C_efficiency_evolution/detailed_results/LilRg_PRYMMAL-slerp-Merge/results_2025-02-13T18-27-04.338360.json`  118,608 bytes
- `C_efficiency_evolution/detailed_results/LimYeri_CodeMind-Llama3-8B-unsloth_v2-merged/results_2025-02-13T18-27-04.338360.json`  119,161 bytes
- `C_efficiency_evolution/detailed_results/LimYeri_CodeMind-Llama3-8B-unsloth_v3-merged/results_2025-02-13T18-27-04.338360.json`  119,168 bytes
- `C_efficiency_evolution/detailed_results/LimYeri_CodeMind-Llama3-8B-unsloth_v4-one-DPO-merged/results_2024-09-03T08-41-46.953089.json`  119,159 bytes
- `C_efficiency_evolution/detailed_results/LimYeri_CodeMind-Llama3-8B-unsloth_v4-one-merged/results_2025-02-13T18-27-04.338360.json`  118,738 bytes
- `C_efficiency_evolution/detailed_results/LimYeri_CodeMind-Llama3.1-8B-unsloth-merged/results_2025-02-13T18-27-04.338360.json`  123,562 bytes
- `C_efficiency_evolution/detailed_results/lkoenig_BBAI_200_Gemma/results_2025-02-27T11-24-52.773192.json`  113,128 bytes
- `C_efficiency_evolution/detailed_results/lkoenig_BBAI_456_QwenKoen/results_2025-03-01T16-19-14.615825.json`  113,509 bytes
- `C_efficiency_evolution/detailed_results/lkoenig_BBAI_7B_KoenQwenDyan/results_2025-03-08T20-01-53.125995.json`  113,513 bytes
- `C_efficiency_evolution/detailed_results/lkoenig_BBAI_7B_Qwen2.5koen/results_2025-03-01T17-02-53.917814.json`  113,497 bytes
- `C_efficiency_evolution/detailed_results/lkoenig_BBAI_7B_QwenDyanKoenLo/results_2025-03-08T19-57-59.849642.json`  113,515 bytes
- `C_efficiency_evolution/detailed_results/LLM360_K2/results_2025-02-13T18-27-04.338360.json`  118,516 bytes
- `C_efficiency_evolution/detailed_results/LLM360_K2-Chat/results_2025-02-13T18-27-04.338360.json`  120,475 bytes
- `C_efficiency_evolution/detailed_results/LLM4Binary_llm4decompile-1.3b-v2/results_2025-02-13T18-27-04.338360.json`  114,762 bytes
- `C_efficiency_evolution/detailed_results/llmat_Mistral-v0.3-7B-ORPO/results_2024-08-12T12-37-49.752603.json`  118,830 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_CollectiveLM-Falcon-3-7B/results_2025-02-13T18-27-04.338360.json`  115,363 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_Hercules-6.0-Llama-3.1-8B/results_2024-09-26T10-11-45.568903.json`  119,312 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_Hercules-6.1-Llama-3.1-8B/results_2025-02-13T18-27-04.338360.json`  119,330 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_Llama-3-NeuralHercules-5.0-8B/results_2024-08-13T01-27-58.645604.json`  119,382 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_Llama-3-Yggdrasil-2.0-8B/results_2024-08-13T01-24-48.518321.json`  119,372 bytes
- `C_efficiency_evolution/detailed_results/Locutusque_TinyMistral-248M-v2.5/results_2024-09-16T04-23-52.850271.json`  118,285 bytes
- `C_efficiency_evolution/detailed_results/Luni_StarDust-12b-v1/results_2025-02-13T18-27-04.338360.json`  118,922 bytes
- `C_efficiency_evolution/detailed_results/Luni_StarDust-12b-v2/results_2025-02-13T18-27-04.338360.json`  118,931 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v3/results_2025-02-24T23-30-56.918941.json`  113,540 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v5/results_2025-02-25T00-00-30.869764.json`  116,216 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v6/results_2025-02-26T07-03-41.908628.json`  113,531 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v7-rebase/results_2025-02-28T21-38-57.260500.json`  113,524 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v8/results_2025-02-28T15-01-10.974713.json`  113,538 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v8.5/results_2025-02-28T23-25-38.441600.json`  113,549 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v8.6/results_2025-03-01T02-45-12.512000.json`  113,551 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v8.7/results_2025-03-01T08-45-31.657703.json`  113,538 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v8.9/results_2025-03-06T11-00-17.588392.json`  113,543 bytes
- `C_efficiency_evolution/detailed_results/Lunzima_NQLSG-Qwen2.5-14B-MegaFusion-v9-stock/results_2025-03-07T07-26-34.728949.json`  113,552 bytes
- `C_efficiency_evolution/detailed_results/Lyte_Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3/results_2025-02-13T18-27-04.338360.json`  123,584 bytes
- `C_efficiency_evolution/detailed_results/Lyte_Llama-3.2-1B-Instruct-COT-RL-Expriement1-EP04/results_2024-09-26T03-14-13.908783.json`  122,757 bytes
- `C_efficiency_evolution/detailed_results/Lyte_Llama-3.2-3B-Overthinker/results_2024-10-18T15-15-40.654668.json`  119,223 bytes
- `C_efficiency_evolution/detailed_results/M4-ai_TinyMistral-248M-v3/results_2024-10-18T08-51-04.559101.json`  118,217 bytes
- `C_efficiency_evolution/detailed_results/magnifi_Phi3_intent_v56_3_w_unknown_5_lr_0.002/results_2025-03-10T21-31-33.519419.json`  113,729 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3-8B-Magpie-Align-SFT-v0.1/results_2024-09-18T06-08-40.741828.json`  119,119 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3-8B-Magpie-Align-SFT-v0.3/results_2024-08-13T01-20-48.160312.json`  119,172 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3-8B-Magpie-Align-v0.1/results_2025-02-13T18-27-04.338360.json`  119,141 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3-8B-Magpie-Align-v0.3/results_2025-02-13T18-27-04.338360.json`  119,169 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3.1-8B-Magpie-Align-SFT-v0.1/results_2025-02-13T18-27-04.338360.json`  119,160 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_Llama-3.1-8B-Magpie-Align-v0.1/results_2024-09-18T06-41-31.872133.json`  118,776 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_MagpieLM-8B-Chat-v0.1/results_2025-02-13T18-27-04.338360.json`  118,980 bytes
- `C_efficiency_evolution/detailed_results/Magpie-Align_MagpieLM-8B-SFT-v0.1/results_2025-02-13T18-27-04.338360.json`  119,184 bytes
- `C_efficiency_evolution/detailed_results/maldv_badger-lambda-llama-3-8b/results_2024-07-18T22-08-35.167678.json`  119,055 bytes
- `C_efficiency_evolution/detailed_results/ManoloPueblo_ContentCuisine_1-7B-slerp/results_2025-02-13T18-27-04.338360.json`  114,674 bytes
- `C_efficiency_evolution/detailed_results/ManoloPueblo_LLM_MERGE_CC2/results_2025-02-13T18-27-04.338360.json`  114,600 bytes
- `C_efficiency_evolution/detailed_results/ManoloPueblo_LLM_MERGE_CC3/results_2025-02-13T18-27-04.338360.json`  114,622 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-MST/results_2025-03-09T12-03-06.802478.json`  116,224 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-MST-v1.3/results_2025-03-09T12-51-36.770537.json`  116,160 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-Preview/results_2025-03-07T02-14-20.812691.json`  116,144 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-RP-v1.4-1M/results_2025-03-10T18-54-33.706076.json`  116,145 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-v1.1/results_2025-03-08T14-26-26.531812.json`  116,117 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Hush-Qwen2.5-7B-v1.3/results_2025-03-08T18-53-52.937772.json`  116,127 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Qwen2.5-7B-Preview/results_2025-03-09T15-22-35.247692.json`  116,105 bytes
- `C_efficiency_evolution/detailed_results/marcuscedricridia_Yell-Qwen2.5-7B-Preview-v1.1/results_2025-03-08T08-51-49.160851.json`  116,179 bytes
- `C_efficiency_evolution/detailed_results/MarinaraSpaghetti_Nemomix-v4.0-12B/results_2025-02-13T18-27-04.338360.json`  118,924 bytes
- `C_efficiency_evolution/detailed_results/MarinaraSpaghetti_NemoReRemix-12B/results_2025-02-13T18-27-04.338360.json`  118,608 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_general3B-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  118,650 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_general3Bv2-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  114,688 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_lareneg1_78B-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  118,675 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_lareneg3B-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  114,716 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_lareneg3Bv2-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  114,697 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_MiniMathExpert-2_61B-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  118,671 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_MiniQwenMathExpert-ECE-PRYMMAL-Martial/results_2025-02-13T18-27-04.338360.json`  118,690 bytes
- `C_efficiency_evolution/detailed_results/Marsouuu_MistralBase-4x7B-MoE-ECE-PRYMMAL-Martial/results_2024-10-04T02-05-16.789490.json`  118,583 bytes
- `C_efficiency_evolution/detailed_results/mattshumer_Reflection-Llama-3.1-70B/results_2024-09-09T18-34-06.815867.json`  118,637 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-llama3.1-70b/results_2025-02-13T18-27-04.338360.json`  115,068 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-phi3-4b/results_2024-07-18T11-08-51.380541.json`  118,891 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-phi3.5-4b/results_2024-09-03T09-41-32.172139.json`  119,069 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-qwen2-72b/results_2024-07-18T06-00-38.574165.json`  118,977 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,985 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-qwen2.5-72b/results_2024-10-05T22-25-27.320099.json`  118,888 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.1-rys-78b/results_2025-02-13T18-27-04.338360.json`  118,958 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-llama3-70b/results_2024-07-17T08-13-13.687245.json`  118,972 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-llama3.1-70b/results_2025-02-13T18-27-04.338360.json`  119,130 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-phi3-4b/results_2024-07-18T09-39-19.713557.json`  118,905 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-qwen2-72b/results_2025-02-13T18-27-04.338360.json`  118,984 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,993 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-qwen2.5-72b/results_2025-02-13T18-27-04.338360.json`  118,971 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.2-rys-78b/results_2025-02-13T18-27-04.338360.json`  118,929 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-llama3-70b/results_2025-02-13T18-27-04.338360.json`  119,015 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-llama3.1-70b/results_2025-02-13T18-27-04.338360.json`  119,136 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-phi3-4b/results_2025-02-13T18-27-04.338360.json`  118,615 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-qwen2-72b/results_2024-09-28T03-12-45.476266.json`  118,922 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,989 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.3-rys-78b/results_2025-02-13T18-27-04.338360.json`  118,928 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.4-llama3-70b/results_2024-07-18T09-03-39.762026.json`  118,990 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.4-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,998 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.4-rys-78b/results_2025-02-13T18-27-04.338360.json`  118,978 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.5-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,981 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.6-qwen2-7b/results_2025-02-13T18-27-04.338360.json`  118,979 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-2.7-qwen2-7b/results_2024-09-19T06-16-21.919709.json`  118,975 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.1-baguette-3b/results_2025-02-13T18-27-04.338360.json`  115,046 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.1-instruct-3b/results_2025-02-13T18-27-04.338360.json`  115,047 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.1-instruct-78b/results_2025-02-13T18-27-04.338360.json`  115,002 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.1-llamaloi-3b/results_2024-11-10T07-21-06.518071.json`  118,761 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.2-baguette-3b/results_2025-02-13T18-27-04.338360.json`  115,021 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.2-instruct-3b/results_2025-02-13T18-27-04.338360.json`  117,365 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.2-instruct-78b/results_2025-02-13T18-27-04.338360.json`  115,010 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.3-baguette-3b/results_2025-02-13T18-27-04.338360.json`  117,129 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_calme-3.3-instruct-3b/results_2025-02-13T18-27-04.338360.json`  117,145 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Calme-4x7B-MoE-v0.1/results_2025-02-13T18-27-04.338360.json`  118,645 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Calme-4x7B-MoE-v0.2/results_2025-02-13T18-27-04.338360.json`  118,653 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Llama-3-70B-Instruct-v0.1/results_2024-10-05T14-21-45.746802.json`  119,018 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Llama-3-8B-Instruct-v0.10/results_2024-07-18T10-57-13.925740.json`  119,167 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Llama-3-8B-Instruct-v0.8/results_2024-07-04T03-47-33.210245.json`  118,740 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Llama-3-8B-Instruct-v0.9/results_2025-02-13T18-27-04.338360.json`  119,245 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Qwen1.5-MoE-A2.7B-Wikihow/results_2025-02-13T18-27-04.338360.json`  119,009 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Qwen2-7B-Instruct-v0.1/results_2024-07-29T18-41-30.681992.json`  118,626 bytes
- `C_efficiency_evolution/detailed_results/MaziyarPanahi_Qwen2-7B-Instruct-v0.8/results_2025-02-13T18-27-04.338360.json`  118,654 bytes
- `C_efficiency_evolution/detailed_results/meditsolutions_Llama-3.2-SUN-1B-chat/results_2024-11-06T17-51-53.110040.json`  118,775 bytes
- `C_efficiency_evolution/detailed_results/meditsolutions_Llama-3.2-SUN-2.4B-checkpoint-34800/results_2024-10-05T13-51-49.717401.json`  122,805 bytes
- `C_efficiency_evolution/detailed_results/meditsolutions_Llama-3.2-SUN-2.5B-chat/results_2024-10-24T00-00-00.000000.json`  122,698 bytes
- `C_efficiency_evolution/detailed_results/MEscriva_ECE-PRYMMAL-0.5B-FT-V5-MUSR-Mathis/results_2025-02-13T18-27-04.338360.json`  114,673 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-2-13b-hf/results_2024-06-17T11-35-38.123827.json`  120,078 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-2-70b-chat-hf/results_2024-06-17T09-38-51.358362.json`  121,970 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-2-70b-hf/results_2024-06-17T14-10-55.365075.json`  121,146 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-2-7b-chat-hf/results_2024-06-17T15-57-01.774754.json`  120,995 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-3.1-70B/results_2024-07-18T20-24-37.562222.json`  118,565 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-3.1-70B-Instruct/results_2024-07-19T19-54-46.793254.json`  118,937 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-3.1-8B-Instruct/results_2025-02-06T09-49-35.437423.json`  114,657 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-3.2-1B/results_2024-09-23T09-45-37.626278.json`  118,419 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Llama-3.2-3B-Instruct/results_2024-09-23T09-53-31.373398.json`  118,983 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Meta-Llama-3-70B/results_2024-06-17T10-04-04.013818.json`  121,180 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Meta-Llama-3-70B-Instruct/results_2024-06-19T08-22-58.348428.json`  120,553 bytes
- `C_efficiency_evolution/detailed_results/meta-llama_Meta-Llama-3-8B-Instruct/results_2024-06-17T11-25-17.966751.json`  120,556 bytes
- `C_efficiency_evolution/detailed_results/microsoft_phi-1/results_2024-06-21T13-03-02.514754.json`  120,026 bytes
- `C_efficiency_evolution/detailed_results/microsoft_phi-1_5/results_2024-06-17T10-46-09.252711.json`  120,001 bytes
- `C_efficiency_evolution/detailed_results/microsoft_Phi-3-medium-128k-instruct/results_2024-09-02T07-32-20.832197.json`  118,902 bytes
- `C_efficiency_evolution/detailed_results/microsoft_Phi-3-medium-4k-instruct/results_2024-06-17T10-38-50.763081.json`  120,433 bytes
- `C_efficiency_evolution/detailed_results/microsoft_Phi-3-mini-128k-instruct/results_2024-06-25T07-10-31.918768.json`  120,422 bytes
- `C_efficiency_evolution/detailed_results/microsoft_phi-4/results_2025-01-08T16-41-36.968588.json`  114,608 bytes
- `C_efficiency_evolution/detailed_results/microsoft_Phi-4-mini-instruct/results_2025-02-28T12-41-07.003145.json`  114,023 bytes
- `C_efficiency_evolution/detailed_results/migtissera_Tess-3-Mistral-Nemo-12B/results_2024-09-03T12-01-51.063796.json`  119,013 bytes
- `C_efficiency_evolution/detailed_results/Minami-su_Amara-o1-7B-Qwen/results_2025-02-13T18-27-04.338360.json`  117,156 bytes
- `C_efficiency_evolution/detailed_results/Minami-su_Amara-o2-7B-Qwen/results_2025-02-13T18-27-04.338360.json`  117,187 bytes
- `C_efficiency_evolution/detailed_results/Minami-su_test-7B-00/results_2025-02-13T18-27-04.338360.json`  117,318 bytes
- `C_efficiency_evolution/detailed_results/Minami-su_test-7B-01/results_2025-02-13T18-27-04.338360.json`  117,310 bytes
- `C_efficiency_evolution/detailed_results/Minami-su_test-v2-7B-00/results_2025-02-13T18-27-04.338360.json`  117,172 bytes
- `C_efficiency_evolution/detailed_results/minghaowu_Qwen1.5-1.8B-OpenHermes-2.5/results_2024-09-18T13-54-12.323880.json`  118,741 bytes
- `C_efficiency_evolution/detailed_results/mistral-community_Mixtral-8x22B-v0.1/results_2025-02-13T18-27-04.338360.json`  121,475 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Codestral-22B-v0.1/results_2024-09-28T11-41-42.167336.json`  119,698 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-7B-Instruct-v0.1/results_2024-06-17T09-16-22.960059.json`  120,658 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-7B-Instruct-v0.3/results_2024-06-17T14-37-59.603167.json`  120,650 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-7B-v0.3/results_2024-06-16T16-00-54.042738.json`  77,185 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-Large-Instruct-2411/results_2024-11-20T12-25-36.264064.json`  115,044 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-Nemo-Instruct-2407/results_2024-07-19T11-53-04.051251.json`  119,593 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mistral-Small-Instruct-2409/results_2024-09-19T10-04-43.281334.json`  122,728 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mixtral-8x22B-Instruct-v0.1/results_2024-06-26T15-18-07.434042.json`  121,385 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mixtral-8x22B-v0.1/results_2024-06-24T22-38-47.314637.json`  120,103 bytes
- `C_efficiency_evolution/detailed_results/mistralai_Mixtral-8x7B-v0.1/results_2024-06-17T17-56-38.031288.json`  120,075 bytes
- `C_efficiency_evolution/detailed_results/mkurman_llama-3.2-MEDIT-3B-o1/results_2025-01-08T00-40-26.314883.json`  118,739 bytes
- `C_efficiency_evolution/detailed_results/mkurman_phi-4-MedIT-11B-exp-1/results_2025-01-13T20-26-24.993942.json`  115,117 bytes
- `C_efficiency_evolution/detailed_results/mkxu_llama-3-8b-po1/results_2024-11-30T06-55-29.252571.json`  114,635 bytes
- `C_efficiency_evolution/detailed_results/mlabonne_BigQwen2.5-52B-Instruct/results_2024-09-29T02-46-54.207954.json`  121,288 bytes
- `C_efficiency_evolution/detailed_results/mlabonne_ChimeraLlama-3-8B-v2/results_2024-09-02T14-04-32.528223.json`  118,641 bytes
- `C_efficiency_evolution/detailed_results/mlabonne_OrpoLlama-3-8B/results_2024-06-17T09-44-13.592676.json`  120,346 bytes
- `C_efficiency_evolution/detailed_results/mlabonne_phixtral-2x2_8/results_2024-06-17T09-44-36.496962.json`  120,411 bytes
- `C_efficiency_evolution/detailed_results/MLP-KTLim_llama-3-Korean-Bllossom-8B/results_2025-02-13T18-27-04.338360.json`  119,194 bytes
- `C_efficiency_evolution/detailed_results/mlx-community_Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-float32/results_2025-01-08T03-58-22.402554.json`  119,241 bytes
- `C_efficiency_evolution/detailed_results/mlx-community_Mistral-Small-24B-Instruct-2501-bf16/results_2025-02-07T06-30-17.181165.json`  114,662 bytes
- `C_efficiency_evolution/detailed_results/mmnga_Llama-3-70B-japanese-suzume-vector-v0.1/results_2024-09-29T00-30-29.453759.json`  119,121 bytes
- `C_efficiency_evolution/detailed_results/mobiuslabsgmbh_DeepSeek-R1-ReDistill-Llama3-8B-v1.1/results_2025-02-18T17-01-07.249399.json`  115,948 bytes
- `C_efficiency_evolution/detailed_results/mobiuslabsgmbh_DeepSeek-R1-ReDistill-Qwen-7B-v1.1/results_2025-02-18T17-47-32.642827.json`  115,943 bytes
- `C_efficiency_evolution/detailed_results/ModelCloud_Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1/results_2025-01-10T11-16-08.038199.json`  118,806 bytes
- `C_efficiency_evolution/detailed_results/ModelSpace_GemmaX2-28-9B-v0.1/results_2025-02-24T20-52-51.514794.json`  113,515 bytes
- `C_efficiency_evolution/detailed_results/monsterapi_gemma-2-2b-LoRA-MonsterInstruct/results_2024-08-13T06-06-06.413253.json`  119,267 bytes
- `C_efficiency_evolution/detailed_results/monsterapi_Llama-3_1-8B-Instruct-orca-ORPO/results_2024-09-13T23-58-00.534893.json`  123,220 bytes
- `C_efficiency_evolution/detailed_results/MTSAIR_Cotype-Nano/results_2025-02-13T18-27-04.338360.json`  117,177 bytes
- `C_efficiency_evolution/detailed_results/MTSAIR_MultiVerse_70B/results_2025-02-13T18-27-04.338360.json`  118,604 bytes
- `C_efficiency_evolution/detailed_results/mukaj_Llama-3.1-Hawkish-8B/results_2024-12-18T08-18-26.443505.json`  119,503 bytes
- `C_efficiency_evolution/detailed_results/NAPS-ai_naps-gemma-2-27b-v0.1.0/results_2024-11-11T10-10-02.008201.json`  113,960 bytes
- `C_efficiency_evolution/detailed_results/NAPS-ai_naps-llama-3_1-instruct-v0.5.0/results_2024-09-30T03-13-20.847203.json`  123,548 bytes
- `C_efficiency_evolution/detailed_results/natong19_Mistral-Nemo-Instruct-2407-abliterated/results_2024-09-23T16-03-15.478274.json`  122,686 bytes
- `C_efficiency_evolution/detailed_results/natong19_Qwen2-7B-Instruct-abliterated/results_2024-07-30T07-08-14.647283.json`  119,019 bytes
- `C_efficiency_evolution/detailed_results/Naveenpoliasetty_llama3-8B-V2/results_2024-07-25T01-07-18.059419.json`  118,622 bytes
- `C_efficiency_evolution/detailed_results/nazimali_Mistral-Nemo-Kurdish-Instruct/results_2024-10-15T18-24-20.801908.json`  122,701 bytes
- `C_efficiency_evolution/detailed_results/NbAiLab_nb-llama-3.1-8B-Instruct/results_2024-12-11T04-48-05.360067.json`  119,526 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Dumpling-Qwen2.5-1.5B/results_2025-01-29T14-03-52.354879.json`  117,256 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Dumpling-Qwen2.5-14B/results_2025-02-19T15-17-34.318477.json`  116,124 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_gemma2-gutenberg-27B/results_2024-09-30T00-44-40.185741.json`  114,316 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_gemma2-gutenberg-9B/results_2024-08-12T22-31-28.823844.json`  118,527 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Gemma2-Gutenberg-Doppel-9B/results_2024-10-02T02-10-48.497344.json`  118,560 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Gutensuppe-mistral-nemo-12B/results_2024-09-03T13-15-04.026140.json`  118,591 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Hermes2-Gutenberg2-Mistral-7B/results_2024-10-02T01-22-35.817848.json`  118,617 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Llama-3.1-Nemotron-lorablated-70B/results_2024-10-18T18-21-36.549778.json`  123,464 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mahou-1.5-mistral-nemo-12B-lorablated/results_2024-10-19T14-15-30.943516.json`  119,012 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Gutenberg-Doppel-7B-FFT/results_2024-11-19T23-58-30.543865.json`  115,800 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_mistral-nemo-cc-12B/results_2024-09-14T18-04-27.947372.json`  118,540 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_mistral-nemo-gutenberg-12B-v3/results_2024-09-03T12-12-12.650925.json`  118,594 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_mistral-nemo-gutenberg2-12B-test/results_2024-09-25T02-09-49.175236.json`  118,576 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_mistral-nemo-kartoffel-12B/results_2025-01-13T21-02-32.803522.json`  115,028 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Nemo-Moderne-12B-FFT-experimental/results_2024-11-26T14-21-45.562933.json`  115,091 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Nemo-Prism-12B/results_2024-11-12T14-48-16.155912.json`  115,011 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Nemo-Prism-12B-v2/results_2024-11-26T14-01-51.878225.json`  115,006 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_mistral-nemo-wissenschaft-12B/results_2024-09-02T22-18-50.596772.json`  122,665 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Small-Drummer-22B/results_2024-10-02T02-03-59.903701.json`  118,586 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Mistral-Small-Gutenberg-Doppel-22B/results_2024-09-25T13-19-50.574919.json`  118,626 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Qwen2.5-Gutenberg-Doppel-14B/results_2024-11-11T15-47-45.295181.json`  117,013 bytes
- `C_efficiency_evolution/detailed_results/nbeerbower_Stella-mistral-nemo-12B-v2/results_2024-09-14T18-16-38.368634.json`  118,611 bytes
- `C_efficiency_evolution/detailed_results/NCSOFT_Llama-VARCO-8B-Instruct/results_2024-12-21T17-40-41.330448.json`  115,085 bytes
- `C_efficiency_evolution/detailed_results/necva_IE-cont-Llama3.1-8B/results_2025-03-06T07-54-31.179464.json`  113,133 bytes
- `C_efficiency_evolution/detailed_results/Nekochu_Llama-3.1-8B-German-ORPO/results_2024-09-24T17-07-03.925941.json`  118,270 bytes
- `C_efficiency_evolution/detailed_results/neopolita_jessi-v0.1-bf16-falcon3-7b-instruct/results_2025-01-08T12-08-21.824117.json`  116,627 bytes
- `C_efficiency_evolution/detailed_results/neopolita_jessi-v0.1-qwen2.5-7b-instruct/results_2025-01-08T15-19-06.068963.json`  114,683 bytes
- `C_efficiency_evolution/detailed_results/neopolita_jessi-v0.2-falcon3-10b-instruct/results_2025-01-21T23-37-09.659225.json`  116,924 bytes
- `C_efficiency_evolution/detailed_results/neopolita_jessi-v0.2-falcon3-7b-instruct/results_2025-01-08T21-05-35.811465.json`  114,627 bytes
- `C_efficiency_evolution/detailed_results/netcat420_DeepSeek-R1-MFANN-TIES-unretrained-7b/results_2025-01-22T04-15-19.603999.json`  117,375 bytes
- `C_efficiency_evolution/detailed_results/netcat420_MFANN-Llama3.1-Abliterated-Slerp-TIES/results_2024-10-29T15-47-55.490741.json`  118,686 bytes
- `C_efficiency_evolution/detailed_results/netcat420_MFANN-Llama3.1-Abliterated-SLERP-TIES-V3/results_2024-11-26T08-19-33.493259.json`  114,733 bytes
- `C_efficiency_evolution/detailed_results/netcat420_MFANN-Llama3.1-Abliterated-Slerp-V3.2/results_2024-10-29T16-33-43.166964.json`  118,690 bytes
- `C_efficiency_evolution/detailed_results/netcat420_MFANN-llama3.1-abliterated-v2/results_2024-10-07T21-53-39.588918.json`  118,656 bytes
- `C_efficiency_evolution/detailed_results/netcat420_MFANN-phigments-slerp-V3.3/results_2025-02-07T13-31-54.238006.json`  114,648 bytes
- `C_efficiency_evolution/detailed_results/netcat420_Qwen2.5-Coder-Scholar-7B-Abliterated-MFANN/results_2025-01-01T06-39-22.095522.json`  117,405 bytes
- `C_efficiency_evolution/detailed_results/netcat420_Qwen2.5-DeepSeek-R1-MFANN-Slerp-7b/results_2025-01-23T15-12-53.603377.json`  117,255 bytes
- `C_efficiency_evolution/detailed_results/netcat420_qwen2.5-MFANN-7b-SLERPv1.1/results_2025-02-04T21-31-25.930696.json`  117,347 bytes
- `C_efficiency_evolution/detailed_results/newsbang_Homer-v1.0-Qwen2.5-72B/results_2024-12-26T12-15-06.766575.json`  114,654 bytes
- `C_efficiency_evolution/detailed_results/newsbang_Homer-v1.0-Qwen2.5-7B/results_2024-12-04T09-06-52.510223.json`  114,664 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DeepDive_3_R1_Prev_v1.0/results_2025-02-27T15-36-48.522598.json`  114,137 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DoberWild_v2.02/results_2025-02-28T10-15-57.545121.json`  113,923 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DobHerWild_R1_v1.1R/results_2025-02-27T18-15-40.531386.json`  118,482 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DodoWild_v2.01/results_2025-02-28T00-40-44.676114.json`  118,439 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DodoWild_v2.02/results_2025-02-28T10-12-03.740197.json`  117,583 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_DodoWild_v2.03/results_2025-02-28T11-23-06.488083.json`  114,112 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Dolermed_R1_V1.01/results_2025-02-28T10-08-59.607397.json`  117,616 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Dolermed_V1.01/results_2025-02-28T01-33-57.969777.json`  113,897 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Dolerstormed_V1.04/results_2025-03-01T05-53-20.770746.json`  114,113 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Hermedive_R1_V1.01/results_2025-02-28T01-44-22.380618.json`  113,924 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Mediver_V1.01/results_2025-02-27T22-32-16.425943.json`  113,694 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Smarteaz_0.2_R1/results_2025-02-27T14-51-34.008698.json`  115,890 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.1_8b_Smarteaz_V1.01/results_2025-02-27T21-57-29.777655.json`  113,958 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_1b_AquaSyn_0.1/results_2025-02-24T19-23-39.554871.json`  113,577 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_1b_Odyssea_V1/results_2025-02-24T19-38-32.014384.json`  117,570 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_1b_Odyssea_V1.01/results_2025-02-27T06-52-42.473198.json`  117,575 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_1b_SunOrca_V1/results_2025-03-06T02-41-14.753353.json`  114,092 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_1b_Sydonia_0.1/results_2025-02-24T19-31-30.175948.json`  113,575 bytes
- `C_efficiency_evolution/detailed_results/Nexesenex_Llama_3.2_3b_Kermes_v2/results_2025-02-18T15-08-36.633055.json`  113,885 bytes
- `C_efficiency_evolution/detailed_results/nhyha_merge_Qwen2.5-7B-Instruct_20241023_0314/results_2024-11-06T18-12-26.688718.json`  114,686 bytes
- `C_efficiency_evolution/detailed_results/nhyha_N3N_gemma-2-9b-it_20241110_2026/results_2024-11-12T08-54-24.495501.json`  115,313 bytes
- `C_efficiency_evolution/detailed_results/NikolaSigmoid_AceMath-1.5B-Instruct-dolphin-r1-200/results_2025-03-06T03-21-29.997810.json`  113,215 bytes
- `C_efficiency_evolution/detailed_results/NikolaSigmoid_DeepSeek-R1-Distill-Qwen-1.5B-500/results_2025-03-04T17-04-50.228012.json`  113,255 bytes
- `C_efficiency_evolution/detailed_results/NikolaSigmoid_phi-4-14b/results_2025-02-28T06-10-09.004119.json`  113,499 bytes
- `C_efficiency_evolution/detailed_results/nisten_franqwenstein-35b/results_2024-10-05T10-03-58.318391.json`  118,602 bytes
- `C_efficiency_evolution/detailed_results/nisten_tqwendo-36b/results_2024-12-22T23-45-54.604790.json`  114,609 bytes
- `C_efficiency_evolution/detailed_results/nlpguy_Mistral-NeMo-Minitron-Upscale-v3/results_2024-10-05T08-21-18.309431.json`  118,254 bytes
- `C_efficiency_evolution/detailed_results/noname0202_gemma-2-2b-it-ties/results_2025-01-29T05-25-09.210151.json`  115,132 bytes
- `C_efficiency_evolution/detailed_results/noname0202_gemma-2-9b-sft-jp-en-zh-v1/results_2025-01-08T04-41-53.656589.json`  115,233 bytes
- `C_efficiency_evolution/detailed_results/noname0202_gemma-2-9b-sft-jp-en-zh-v2/results_2025-01-07T20-57-33.217458.json`  115,286 bytes
- `C_efficiency_evolution/detailed_results/noname0202_llama-math-1b-r16-0to512tokens-test/results_2025-01-25T01-09-20.202486.json`  118,758 bytes
- `C_efficiency_evolution/detailed_results/noname0202_llama-math-1b-r32-0to512tokens-test/results_2025-01-24T17-05-57.480718.json`  118,799 bytes
- `C_efficiency_evolution/detailed_results/noname0202_llama-math-1b-r32-test/results_2025-01-24T02-10-47.083582.json`  118,759 bytes
- `C_efficiency_evolution/detailed_results/NotASI_FineTome-Llama3.2-3B-1002/results_2024-10-05T07-19-46.651384.json`  128,340 bytes
- `C_efficiency_evolution/detailed_results/NotASI_FineTome-v1.5-Llama3.2-1B-1007/results_2024-10-07T14-10-32.000639.json`  122,653 bytes
- `C_efficiency_evolution/detailed_results/NotASI_FineTome-v1.5-Llama3.2-3B-1007/results_2024-10-07T14-36-28.367357.json`  122,712 bytes
- `C_efficiency_evolution/detailed_results/notbdq_Qwen2.5-14B-Instruct-1M-GRPO-Reasoning/results_2025-02-06T03-51-24.710492.json`  117,153 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_DeepHermes-3-Mistral-24B-Preview/results_2025-03-13T18-57-43.884131.json`  114,107 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_Hermes-2-Pro-Llama-3-8B/results_2024-06-16T17-24-34.509503.json`  120,349 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_Hermes-2-Pro-Mistral-7B/results_2024-06-16T17-23-03.425460.json`  120,359 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_Hermes-3-Llama-3.1-70B/results_2024-08-28T23-42-29.435150.json`  118,956 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_Nous-Hermes-2-Mistral-7B-DPO/results_2024-06-17T09-36-01.123340.json`  120,417 bytes
- `C_efficiency_evolution/detailed_results/NousResearch_Yarn-Llama-2-13b-128k/results_2024-06-26T13-35-58.419685.json`  120,092 bytes
- `C_efficiency_evolution/detailed_results/nvidia_Mistral-NeMo-Minitron-8B-Base/results_2024-08-22T14-14-50.709024.json`  118,536 bytes
- `C_efficiency_evolution/detailed_results/oopere_Llama-FinSent-S/results_2025-02-18T20-22-42.735112.json`  113,557 bytes
- `C_efficiency_evolution/detailed_results/oopere_pruned20-llama-1b/results_2024-11-18T11-38-47.380381.json`  114,573 bytes
- `C_efficiency_evolution/detailed_results/oopere_pruned20-llama-3.2-3b/results_2024-12-18T02-55-41.663089.json`  114,564 bytes
- `C_efficiency_evolution/detailed_results/oopere_pruned40-llama-1b/results_2024-11-26T09-36-00.519182.json`  114,487 bytes
- `C_efficiency_evolution/detailed_results/oopere_pruned40-llama-3.2-1B/results_2025-02-24T17-19-19.767214.json`  113,460 bytes
- `C_efficiency_evolution/detailed_results/oopere_pruned60-llama-1b/results_2024-11-25T21-20-53.829333.json`  114,293 bytes
- `C_efficiency_evolution/detailed_results/Open-Orca_Mistral-7B-OpenOrca/results_2024-06-16T17-32-32.802384.json`  120,422 bytes
- `C_efficiency_evolution/detailed_results/OpenBuddy_openbuddy-llama3-70b-v21.2-32k/results_2024-10-15T17-12-00.837354.json`  114,993 bytes
- `C_efficiency_evolution/detailed_results/OpenBuddy_openbuddy-llama3-8b-v21.1-8k/results_2024-08-12T12-02-09.854304.json`  118,947 bytes
- `C_efficiency_evolution/detailed_results/OpenBuddy_openbuddy-llama3.1-70b-v22.1-131k/results_2024-09-14T13-07-38.561128.json`  118,873 bytes
- `C_efficiency_evolution/detailed_results/OpenBuddy_openbuddy-llama3.1-8b-v22.3-131k/results_2024-10-02T02-14-42.968052.json`  118,927 bytes
- `C_efficiency_evolution/detailed_results/OpenBuddy_openbuddy-yi1.5-34b-v21.3-32k/results_2024-09-03T10-53-41.358514.json`  118,902 bytes
- `C_efficiency_evolution/detailed_results/OpenLeecher_llama3-8b-lima/results_2024-10-02T01-58-49.544644.json`  119,327 bytes
- `C_efficiency_evolution/detailed_results/OpenScholar_Llama-3.1_OpenScholar-8B/results_2024-12-03T20-18-02.470239.json`  115,044 bytes
- `C_efficiency_evolution/detailed_results/orai-nlp_Llama-eus-8B/results_2024-09-30T09-13-52.100084.json`  118,632 bytes
- `C_efficiency_evolution/detailed_results/Orenguteng_Llama-3.1-8B-Lexi-Uncensored-V2/results_2024-09-02T17-57-37.858974.json`  123,543 bytes
- `C_efficiency_evolution/detailed_results/Orion-zhen_phi-4-abliterated/results_2024-12-20T14-55-57.931598.json`  114,613 bytes
- `C_efficiency_evolution/detailed_results/pankajmathur_orca_mini_phi-4/results_2025-01-22T03-35-35.081545.json`  115,180 bytes
- `C_efficiency_evolution/detailed_results/PJMixers-Dev_LLaMa-3.1-Instruct-Interleaved-Zeroed-13B/results_2024-12-18T10-25-04.014332.json`  119,633 bytes
- `C_efficiency_evolution/detailed_results/PJMixers-Dev_LLaMa-3.1-RomboTiesTest-8B/results_2024-12-20T03-32-25.776285.json`  119,589 bytes
- `C_efficiency_evolution/detailed_results/PJMixers-Dev_LLaMa-3.1-RomboTiesTest2-8B/results_2024-12-21T06-01-29.891396.json`  119,589 bytes
- `C_efficiency_evolution/detailed_results/PJMixers-Dev_LLaMa-3.2-Instruct-JankMix-v0.2-SFT-3B/results_2024-10-16T08-59-35.330807.json`  122,775 bytes
- `C_efficiency_evolution/detailed_results/PJMixers-Dev_LLaMa-3.2-Instruct-JankMixBread-v0.1-3B/results_2024-10-12T20-37-49.184725.json`  122,754 bytes
- `C_efficiency_evolution/detailed_results/PJMixers_LLaMa-3-CursedStock-v2.0-8B/results_2024-07-18T22-31-46.079432.json`  119,103 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_gemma-2-9b-it-DPO/results_2024-09-20T04-06-40.646930.json`  118,937 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_gemma-2-9b-it-SimPO/results_2024-08-12T11-27-16.600007.json`  118,979 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-8B-ProLong-64k-Base/results_2024-09-18T14-45-43.303270.json`  118,626 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-8B-ProLong-64k-Instruct/results_2024-09-18T14-44-03.560285.json`  118,668 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-DPO/results_2024-10-07T05-21-18.044555.json`  119,078 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-IPO/results_2024-10-07T05-40-23.259485.json`  119,031 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-KTO/results_2024-10-07T05-49-48.712363.json`  118,979 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-ORPO/results_2024-10-07T05-31-32.253469.json`  119,113 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-RDPO/results_2024-10-07T05-55-40.287392.json`  119,088 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-RRHF/results_2024-10-07T05-47-57.293676.json`  119,129 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Base-8B-SFT-SLiC-HF/results_2024-10-07T06-16-48.580292.json`  119,147 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Instruct-8B-CPO-v0.2/results_2024-09-29T00-09-50.771451.json`  119,096 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Instruct-8B-DPO/results_2024-09-29T01-09-15.396608.json`  119,053 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Instruct-8B-RDPO/results_2024-09-29T00-03-19.447694.json`  119,093 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Instruct-8B-RRHF-v0.2/results_2024-10-07T05-52-18.234908.json`  119,131 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Llama-3-Instruct-8B-SimPO/results_2024-09-28T23-44-35.094681.json`  119,108 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Base-SFT-CPO/results_2024-10-07T05-44-33.093002.json`  119,106 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Base-SFT-DPO/results_2024-10-07T05-17-08.148062.json`  119,074 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Base-SFT-RDPO/results_2024-10-07T05-48-52.432475.json`  119,036 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Base-SFT-SimPO/results_2024-09-21T10-30-35.004520.json`  118,964 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Base-SFT-SLiC-HF/results_2024-10-07T05-56-14.688403.json`  119,109 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Instruct-DPO/results_2024-10-07T05-16-04.705100.json`  119,208 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Instruct-IPO/results_2024-10-07T05-39-53.799634.json`  119,278 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Instruct-KTO/results_2024-10-07T05-54-00.828203.json`  119,178 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Mistral-7B-Instruct-SLiC-HF/results_2024-10-16T17-55-37.345668.json`  119,282 bytes
- `C_efficiency_evolution/detailed_results/princeton-nlp_Sheared-LLaMA-1.3B/results_2024-07-30T07-27-05.879406.json`  118,539 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Galactic-Qwen-14B-Exp2/results_2025-03-13T12-44-48.264475.json`  113,523 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Llama-3.1-8B-Open-SFT/results_2025-01-13T20-41-48.426106.json`  114,691 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Llama-3.2-6B-AlgoCode/results_2025-01-13T20-08-17.772449.json`  114,694 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Llama-Express.1-Math/results_2025-01-25T08-10-21.262789.json`  119,556 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Magellanic-Qwen-25B-R999/results_2025-03-05T16-27-25.331314.json`  113,445 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Phi-4-Empathetic/results_2025-01-13T19-33-26.604717.json`  114,662 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Phi-4-Math-IO/results_2025-01-13T19-05-26.192612.json`  114,641 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Phi-4-o1/results_2025-01-13T10-17-40.927521.json`  114,626 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Phi4-Super/results_2025-01-23T15-21-28.283497.json`  114,630 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Primal-Opus-14B-Optimus-v1/results_2025-02-05T03-05-18.841754.json`  114,680 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Primal-Opus-14B-Optimus-v2/results_2025-02-27T21-05-40.220144.json`  113,498 bytes
- `C_efficiency_evolution/detailed_results/prithivMLmods_Qwen2.5-14B-DeepSeek-R1-1M/results_2025-01-30T04-27-35.041591.json`  114,641 bytes
- `C_efficiency_evolution/detailed_results/qingy2019_LLaMa_3.2_3B_Catalysts/results_2024-10-29T15-22-23.624392.json`  118,616 bytes
- `C_efficiency_evolution/detailed_results/qingy2019_OpenMath2-Llama3.1-8B/results_2024-11-23T05-44-15.106083.json`  114,696 bytes
- `C_efficiency_evolution/detailed_results/qingy2019_Qwen2.5-Ultimate-14B-Instruct/results_2024-12-04T01-04-16.559323.json`  117,335 bytes
- `C_efficiency_evolution/detailed_results/qingy2024_Qwen2.5-4B/results_2025-01-16T20-58-17.085742.json`  114,591 bytes
- `C_efficiency_evolution/detailed_results/qingy2024_Qwen2.5-Coder-Draft-1.5B-Instruct/results_2025-02-01T05-41-05.748637.json`  117,194 bytes
- `C_efficiency_evolution/detailed_results/qingy2024_Qwen2.5-Math-14B-Instruct-Preview/results_2024-12-11T02-56-45.077518.json`  117,025 bytes
- `C_efficiency_evolution/detailed_results/qingy2024_Qwen2.6-14B-Instruct/results_2024-12-05T10-23-22.933498.json`  114,650 bytes
- `C_efficiency_evolution/detailed_results/qingy2024_QwEnlarge-16B-Instruct/results_2025-03-07T00-57-57.081072.json`  116,208 bytes
- `C_efficiency_evolution/detailed_results/qq8933_OpenLongCoT-Base-Gemma2-2B/results_2024-11-12T14-30-23.544874.json`  114,811 bytes
- `C_efficiency_evolution/detailed_results/Quazim0t0_graphite-14b-sce/results_2025-02-05T15-47-09.281719.json`  115,157 bytes
- `C_efficiency_evolution/detailed_results/Quazim0t0_Lo-Phi-14b/results_2025-03-09T16-24-52.723278.json`  113,976 bytes
- `C_efficiency_evolution/detailed_results/Quazim0t0_Phi4.Turn.R1Distill.16bit/results_2025-02-01T01-30-01.897993.json`  115,164 bytes
- `C_efficiency_evolution/detailed_results/Quazim0t0_Phi4Basis-14B-sce/results_2025-02-04T08-19-51.830517.json`  115,152 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-0.5B/results_2024-06-16T18-29-53.027471.json`  121,083 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-1.8B/results_2024-06-16T18-43-23.280850.json`  121,079 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-110B/results_2024-06-25T12-07-08.632481.json`  120,034 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-14B-Chat/results_2024-06-16T18-40-14.610140.json`  121,196 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-32B/results_2024-06-26T06-37-24.381705.json`  120,033 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-32B-Chat/results_2024-06-25T21-18-30.710867.json`  120,458 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-4B-Chat/results_2024-06-16T18-29-56.853847.json`  121,555 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen1.5-MoE-A2.7B/results_2024-06-17T00-13-15.397461.json`  121,035 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-0.5B-Instruct/results_2024-06-17T10-54-45.875940.json`  120,422 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-1.5B-Instruct/results_2024-06-17T09-51-37.025715.json`  120,472 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-57B-A14B/results_2024-07-04T08-36-42.751894.json`  118,257 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-7B/results_2024-06-16T18-53-46.134214.json`  121,138 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-7B-Instruct/results_2024-06-16T18-34-11.511342.json`  121,568 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-Math-72B-Instruct/results_2024-09-21T18-32-21.222955.json`  118,964 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-VL-72B-Instruct/results_2024-10-22T04-20-31.971941.json`  115,751 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2-VL-7B-Instruct/results_2024-10-21T10-26-59.679844.json`  115,767 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-1.5B-Instruct/results_2024-09-19T16-22-58.240552.json`  121,113 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-14B-Instruct/results_2024-09-19T08-51-03.564190.json`  120,925 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-32B-Instruct/results_2024-09-19T12-58-54.557932.json`  120,861 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-3B-Instruct/results_2024-09-19T10-05-30.339220.json`  120,858 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-72B/results_2024-09-20T03-02-35.170648.json`  118,495 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-7B-Instruct/results_2024-09-19T08-24-07.576642.json`  120,925 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-Math-72B-Instruct/results_2024-09-28T23-18-59.974937.json`  121,310 bytes
- `C_efficiency_evolution/detailed_results/Qwen_Qwen2.5-Math-7B/results_2024-09-27T23-55-14.959617.json`  121,282 bytes
- `C_efficiency_evolution/detailed_results/Qwen_QwQ-32B-Preview/results_2024-11-29T18-47-19.440839.json`  117,404 bytes
- `C_efficiency_evolution/detailed_results/rasyosef_Phi-1_5-Instruct-v0.1/results_2024-08-12T09-47-31.671049.json`  118,565 bytes
- `C_efficiency_evolution/detailed_results/rasyosef_phi-2-instruct-apo/results_2024-09-18T08-11-57.792002.json`  118,506 bytes
- `C_efficiency_evolution/detailed_results/recoilme_Gemma-2-Ataraxy-Gemmasutra-9B-slerp/results_2024-09-14T00-39-37.734233.json`  119,155 bytes
- `C_efficiency_evolution/detailed_results/Replete-AI_Llama3-8B-Instruct-Replete-Adapted/results_2024-08-13T01-22-15.449243.json`  119,177 bytes
- `C_efficiency_evolution/detailed_results/RESMPDEV_Qwen2-Wukong-0.5B/results_2024-08-12T22-32-49.035473.json`  118,645 bytes
- `C_efficiency_evolution/detailed_results/rhysjones_phi-2-orange-v2/results_2024-08-13T01-27-07.339858.json`  118,538 bytes
- `C_efficiency_evolution/detailed_results/riaz_FineLlama-3.1-8B/results_2024-10-12T19-57-08.713841.json`  119,541 bytes
- `C_efficiency_evolution/detailed_results/RLHFlow_ArmoRM-Llama3-8B-v0.1/results_2024-10-08T16-02-16.474038.json`  118,620 bytes
- `C_efficiency_evolution/detailed_results/Rombo-Org_Rombo-LLM-V2.5-Qwen-7b/results_2025-01-14T09-46-40.725016.json`  116,993 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_Rombos-Coder-V2.5-Qwen-14b/results_2025-02-06T11-46-45.885563.json`  117,340 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_Rombos-LLM-V2.5-Qwen-0.5b/results_2024-09-29T12-43-27.493254.json`  118,535 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_Rombos-LLM-V2.5-Qwen-14b/results_2024-09-29T13-44-29.673248.json`  118,625 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_Rombos-LLM-V2.5-Qwen-32b/results_2024-10-10T04-14-28.109800.json`  114,671 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_Rombos-LLM-V2.5.1-Qwen-3b/results_2024-10-08T16-02-31.462708.json`  118,627 bytes
- `C_efficiency_evolution/detailed_results/rombodawg_rombos_Replete-Coder-Llama3-8B/results_2024-10-15T15-33-39.386678.json`  119,616 bytes
- `C_efficiency_evolution/detailed_results/rsh345_mistral-ft-optimized-1218-NeuralHermes-2.5-Mistral-7B/results_2025-01-27T23-02-49.046436.json`  114,716 bytes
- `C_efficiency_evolution/detailed_results/sabersalehk_Llama3-001-300/results_2024-12-03T20-07-28.344986.json`  114,663 bytes
- `C_efficiency_evolution/detailed_results/sabersalehk_Llama3_01_300/results_2024-12-03T20-08-26.893164.json`  114,683 bytes
- `C_efficiency_evolution/detailed_results/sabersaleh_Llama2-7B-SimPO/results_2024-12-02T04-10-10.920662.json`  114,558 bytes
- `C_efficiency_evolution/detailed_results/sabersaleh_Llama3/results_2024-11-30T01-19-25.401264.json`  114,648 bytes
- `C_efficiency_evolution/detailed_results/SaisExperiments_Gemma-2-2B-Stheno-Filtered/results_2024-10-08T11-23-44.995228.json`  118,635 bytes
- `C_efficiency_evolution/detailed_results/SaisExperiments_RightSheep-Llama3.2-3B/results_2025-01-25T13-41-22.073426.json`  114,692 bytes
- `C_efficiency_evolution/detailed_results/Sakalti_Llama3.2-3B-Uranus-1/results_2025-01-09T11-57-15.735520.json`  114,683 bytes
- `C_efficiency_evolution/detailed_results/Sakalti_Phi3.5-Comets-3.8B/results_2024-12-21T09-18-24.309772.json`  114,303 bytes
- `C_efficiency_evolution/detailed_results/Sakalti_Qwen2.5-1B-Instruct/results_2025-01-18T07-20-14.657826.json`  114,291 bytes
- `C_efficiency_evolution/detailed_results/SentientAGI_Dobby-Mini-Leashed-Llama-3.1-8B/results_2025-02-05T00-54-44.743677.json`  119,578 bytes
- `C_efficiency_evolution/detailed_results/SentientAGI_Dobby-Mini-Unhinged-Llama-3.1-8B/results_2025-02-04T08-18-57.668288.json`  119,601 bytes
- `C_efficiency_evolution/detailed_results/sequelbox_Llama3.1-70B-PlumChat/results_2024-11-28T10-33-34.389076.json`  114,653 bytes
- `C_efficiency_evolution/detailed_results/sequelbox_Llama3.1-8B-MOTH/results_2024-09-11T21-30-25.104353.json`  123,502 bytes
- `C_efficiency_evolution/detailed_results/sequelbox_Llama3.1-8B-PlumCode/results_2024-10-04T01-24-56.330833.json`  118,589 bytes
- `C_efficiency_evolution/detailed_results/sequelbox_Llama3.1-8B-PlumMath/results_2024-10-04T01-24-04.196002.json`  118,622 bytes
- `C_efficiency_evolution/detailed_results/sethuiyer_Llama-3.1-8B-Experimental-1208-Instruct/results_2025-01-18T17-10-57.107525.json`  115,135 bytes
- `C_efficiency_evolution/detailed_results/sethuiyer_Llamaverse-3.1-8B-Instruct/results_2025-01-14T13-12-45.227965.json`  115,117 bytes
- `C_efficiency_evolution/detailed_results/sethuiyer_Qwen2.5-7B-Anvita/results_2024-10-27T11-40-06.834908.json`  119,293 bytes
- `C_efficiency_evolution/detailed_results/shastraai_Shastra-LLAMA2-Math-Commonsense-SFT/results_2024-10-27T21-01-52.917860.json`  118,653 bytes
- `C_efficiency_evolution/detailed_results/shivank21_mistral_dpo_self/results_2025-02-05T16-15-31.122108.json`  115,789 bytes
- `C_efficiency_evolution/detailed_results/Sicarius-Prototyping_Brainy_LLAMA/results_2024-12-05T09-38-14.561375.json`  114,679 bytes
- `C_efficiency_evolution/detailed_results/SicariusSicariiStuff_LLAMA-3_8B_Unaligned_BETA/results_2024-10-18T16-55-14.789774.json`  118,642 bytes
- `C_efficiency_evolution/detailed_results/SicariusSicariiStuff_Phi-Line_14B/results_2025-02-19T15-13-18.540442.json`  113,878 bytes
- `C_efficiency_evolution/detailed_results/SicariusSicariiStuff_Qwen2.5-14B_Uncensored/results_2024-09-23T16-39-42.907470.json`  118,620 bytes
- `C_efficiency_evolution/detailed_results/SicariusSicariiStuff_Qwen2.5-14B_Uncensored_Instruct/results_2024-09-23T18-19-07.561604.json`  121,257 bytes
- `C_efficiency_evolution/detailed_results/siqi00_Mistral-7B-DFT/results_2025-02-06T03-27-45.600986.json`  115,078 bytes
- `C_efficiency_evolution/detailed_results/skymizer_Llama2-7b-sft-chat-custom-template-dpo/results_2024-07-22T09-51-58.414875.json`  118,631 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Llama3.2-1B-lora-epoch5/results_2024-12-27T23-22-01.016739.json`  114,722 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Llama3.2-1B-lora-v2-epoch3/results_2024-12-27T14-47-03.733196.json`  114,734 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Llama3.2-3B-lora-epoch1/results_2024-12-27T19-38-23.716546.json`  114,711 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Qwen2.5-3B-Instruct/results_2025-01-09T05-49-19.366368.json`  114,649 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Qwen2.5-7B-Instruct-SFT-step-15000/results_2025-02-01T12-03-40.542422.json`  114,688 bytes
- `C_efficiency_evolution/detailed_results/SkyOrbis_SKY-Ko-Qwen2.5-7B-Instruct-SFT-step-5000/results_2025-02-05T00-10-46.826517.json`  114,696 bytes
- `C_efficiency_evolution/detailed_results/someon98_qwen-CoMa-0.5b/results_2024-12-29T16-17-57.730788.json`  114,301 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwen2.5-14B-Vimarckoso-v2/results_2024-12-26T12-16-22.404204.json`  114,696 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwen2.5-14B-Vimarckoso-v3/results_2024-12-27T12-07-58.633030.json`  114,684 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwen2.5-14B-Vimarckoso-v3-IF-Variant/results_2024-12-28T09-42-11.707881.json`  114,729 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwen2.5-7B-Gordion-v0.1-Prose/results_2025-02-18T19-58-57.959022.json`  113,570 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentessential-14B-v1/results_2025-02-18T15-25-19.264487.json`  113,509 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentinuum-14B-v1/results_2024-12-21T18-45-12.892338.json`  114,666 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentinuum-14B-v2/results_2024-12-22T02-16-47.237976.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentinuum-14B-v6/results_2024-12-22T14-21-58.905643.json`  114,654 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentinuum-14B-v7/results_2024-12-22T18-57-51.290456.json`  114,655 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwentinuum-14B-v8/results_2024-12-22T19-01-02.783885.json`  114,663 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v0.6-004-model_stock/results_2025-01-03T08-49-30.636647.json`  114,721 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v3/results_2024-12-21T06-57-25.308738.json`  114,677 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v3-Prose/results_2024-12-21T04-28-21.825696.json`  114,691 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v3-Reason/results_2024-12-21T06-54-07.153436.json`  114,689 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v8/results_2025-01-16T01-39-52.248754.json`  114,652 bytes
- `C_efficiency_evolution/detailed_results/sometimesanotion_Qwenvergence-14B-v9/results_2025-01-17T15-31-48.086833.json`  114,656 bytes
- `C_efficiency_evolution/detailed_results/Sorawiz_Gemma-9B-Base/results_2025-02-25T02-06-43.469016.json`  114,135 bytes
- `C_efficiency_evolution/detailed_results/Sorawiz_Gemma-Creative-9B-Base/results_2025-02-25T02-11-01.948666.json`  114,155 bytes
- `C_efficiency_evolution/detailed_results/Sourjayon_DeepSeek-R1-ForumNXT/results_2025-02-04T23-50-28.463200.json`  114,692 bytes
- `C_efficiency_evolution/detailed_results/spmurrayzzz_Mistral-Syndicate-7B/results_2024-07-25T14-22-01.504229.json`  118,541 bytes
- `C_efficiency_evolution/detailed_results/ssmits_Qwen2.5-95B-Instruct/results_2024-10-06T06-27-47.292101.json`  121,298 bytes
- `C_efficiency_evolution/detailed_results/suayptalha_DeepSeek-R1-Distill-Llama-3B/results_2025-02-24T20-06-57.063798.json`  114,107 bytes
- `C_efficiency_evolution/detailed_results/suayptalha_Komodo-Llama-3.2-3B-v2-fp16/results_2024-11-20T01-03-58.085488.json`  118,789 bytes
- `C_efficiency_evolution/detailed_results/sumink_bbhqwen/results_2025-02-25T21-41-00.885835.json`  113,467 bytes
- `C_efficiency_evolution/detailed_results/sumink_bbhqwen2/results_2025-02-26T02-23-52.551465.json`  113,409 bytes
- `C_efficiency_evolution/detailed_results/sumink_bbhqwen4/results_2025-02-26T02-24-49.960421.json`  113,411 bytes
- `C_efficiency_evolution/detailed_results/sumink_bbhqwen5/results_2025-02-26T03-28-13.192529.json`  113,328 bytes
- `C_efficiency_evolution/detailed_results/sumink_bbhqwen6/results_2025-02-26T02-59-59.806428.json`  113,298 bytes
- `C_efficiency_evolution/detailed_results/sumink_Qwenftmodel/results_2024-12-05T12-23-41.024721.json`  114,611 bytes
- `C_efficiency_evolution/detailed_results/Supichi_BBAI_135_Gemma/results_2025-02-27T11-18-58.504826.json`  113,116 bytes
- `C_efficiency_evolution/detailed_results/Syed-Hasan-8503_Phi-3-mini-4K-instruct-cpo-simpo/results_2024-08-27T22-39-22.007233.json`  118,973 bytes
- `C_efficiency_evolution/detailed_results/synergetic_FrankenQwen2.5-14B/results_2024-12-02T04-24-32.113248.json`  116,986 bytes
- `C_efficiency_evolution/detailed_results/T145_Meta-Llama-3.1-8B-Instruct-TIES/results_2024-12-23T12-02-34.528121.json`  119,587 bytes
- `C_efficiency_evolution/detailed_results/T145_qwen-2.5-3B-merge-test/results_2024-11-18T11-53-58.718004.json`  117,261 bytes
- `C_efficiency_evolution/detailed_results/tangledgroup_tangled-llama-pints-1.5b-v0.1-instruct/results_2024-09-03T07-42-32.532079.json`  118,845 bytes
- `C_efficiency_evolution/detailed_results/tanliboy_lambda-qwen2.5-14b-dpo-test/results_2024-09-27T22-59-32.436873.json`  120,954 bytes
- `C_efficiency_evolution/detailed_results/tanliboy_lambda-qwen2.5-32b-dpo-test/results_2024-10-07T14-01-59.393600.json`  117,045 bytes
- `C_efficiency_evolution/detailed_results/Tarek07_Thalassic-Alpha-LLaMa-70B/results_2025-01-28T19-49-48.861442.json`  114,695 bytes
- `C_efficiency_evolution/detailed_results/Telugu-LLM-Labs_Indic-gemma-2b-finetuned-sft-Navarasa-2.0/results_2025-02-06T10-10-57.836787.json`  114,703 bytes
- `C_efficiency_evolution/detailed_results/Telugu-LLM-Labs_Indic-gemma-7b-finetuned-sft-Navarasa-2.0/results_2025-02-06T10-00-03.236662.json`  114,700 bytes
- `C_efficiency_evolution/detailed_results/tensopolis_falcon3-10b-tensopolis-v1/results_2025-02-01T06-40-26.814340.json`  114,667 bytes
- `C_efficiency_evolution/detailed_results/tensopolis_mistral-small-2501-tensopolis-v1/results_2025-02-18T18-47-42.969024.json`  115,224 bytes
- `C_efficiency_evolution/detailed_results/tensopolis_phi-4-tensopolis-v1/results_2025-02-18T14-54-38.286634.json`  114,037 bytes
- `C_efficiency_evolution/detailed_results/tensopolis_qwen2.5-14b-tensopolis-v1/results_2025-02-01T13-12-37.238511.json`  117,338 bytes
- `C_efficiency_evolution/detailed_results/tensopolis_qwen2.5-7b-tensopolis-v1/results_2025-02-25T00-48-31.801446.json`  116,223 bytes
- `C_efficiency_evolution/detailed_results/TheDrummer_Gemmasutra-9B-v1/results_2024-09-20T02-41-16.900874.json`  118,581 bytes
- `C_efficiency_evolution/detailed_results/TheDrummer_Tiger-Gemma-9B-v2/results_2025-01-08T03-45-37.327645.json`  115,092 bytes
- `C_efficiency_evolution/detailed_results/theo77186_Qwen2.5-Coder-7B-Instruct-20241106/results_2025-01-19T17-54-18.644052.json`  117,340 bytes
- `C_efficiency_evolution/detailed_results/theprint_CleverBoi-Llama-3.1-8B-Instruct/results_2024-09-14T02-44-59.748963.json`  118,708 bytes
- `C_efficiency_evolution/detailed_results/theprint_CleverBoi-Llama-3.1-8B-v2/results_2024-09-23T16-37-17.327998.json`  118,681 bytes
- `C_efficiency_evolution/detailed_results/theprint_ReWiz-Llama-3.1-8B-v2/results_2024-11-06T19-12-21.680195.json`  114,727 bytes
- `C_efficiency_evolution/detailed_results/TheTsar1209_qwen-carpmuscle-r-v0.3/results_2024-10-23T18-42-27.012961.json`  119,523 bytes
- `C_efficiency_evolution/detailed_results/TheTsar1209_qwen-carpmuscle-v0.1/results_2024-10-11T00-59-37.703224.json`  119,498 bytes
- `C_efficiency_evolution/detailed_results/thomas-yanxin_XinYuan-Qwen2-1_5B/results_2024-09-18T07-53-12.933479.json`  119,319 bytes
- `C_efficiency_evolution/detailed_results/tianyil1_MistralForCausalLM_Cal_DPO/results_2025-01-25T10-12-42.974648.json`  115,089 bytes
- `C_efficiency_evolution/detailed_results/TIGER-Lab_AceCoder-Qwen2.5-Coder-7B-Base-Rule/results_2025-03-07T08-59-54.639763.json`  116,275 bytes
- `C_efficiency_evolution/detailed_results/TIGER-Lab_Qwen2.5-Math-7B-CFT/results_2025-03-07T07-37-24.019712.json`  116,211 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_falcon-11B/results_2024-06-17T17-56-25.833143.json`  120,041 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_falcon-40b/results_2024-06-20T06-09-03.551737.json`  119,979 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_falcon-mamba-7b/results_2024-07-23T11-58-56.964566.json`  118,612 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_Falcon3-10B-Base/results_2024-12-12T21-20-43.803477.json`  114,624 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_Falcon3-1B-Instruct/results_2024-12-16T10-36-28.445098.json`  115,111 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_Falcon3-3B-Base/results_2024-12-13T11-06-08.865143.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_Falcon3-7B-Instruct/results_2024-12-16T13-16-31.959114.json`  115,176 bytes
- `C_efficiency_evolution/detailed_results/tiiuae_Falcon3-Mamba-7B-Instruct/results_2024-12-13T11-42-41.018624.json`  114,914 bytes
- `C_efficiency_evolution/detailed_results/tinycompany_ShawtyIsBad-bgem3/results_2025-03-08T08-32-58.921643.json`  113,505 bytes
- `C_efficiency_evolution/detailed_results/tinycompany_ShawtyIsBad-e5-large/results_2025-03-08T08-45-33.614567.json`  113,502 bytes
- `C_efficiency_evolution/detailed_results/tinycompany_ShawtyIsBad-nomic1.5/results_2025-03-08T17-51-05.875611.json`  113,499 bytes
- `C_efficiency_evolution/detailed_results/TinyLlama_TinyLlama-1.1B-Chat-v0.1/results_2024-12-02T17-54-49.516141.json`  114,276 bytes
- `C_efficiency_evolution/detailed_results/TinyLlama_TinyLlama-1.1B-Chat-v1.0/results_2024-08-07T01-00-11.175718.json`  118,598 bytes
- `C_efficiency_evolution/detailed_results/TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T/results_2024-11-27T22-50-39.774473.json`  114,613 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Dolphin3-Llama3.2-Smart/results_2025-01-27T14-48-04.135846.json`  114,574 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_DS-Distilled-Hermes-Llama-3.1/results_2025-01-27T15-23-30.338588.json`  114,709 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_DSR1-Distill-Llama-Lit-8B/results_2025-02-25T00-07-41.414817.json`  113,614 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_DSR1-Distill-Qwen-7B-RP/results_2025-02-24T23-56-33.577219.json`  113,597 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Llama3.1-Allades-Lit-8b/results_2025-02-24T23-59-58.348172.json`  113,432 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Mistral-Small-24b-Harmony/results_2025-02-25T03-44-29.665726.json`  113,509 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Phi-4-AbliteratedRP/results_2025-01-27T14-22-32.819976.json`  114,644 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Phi4-RP-o1/results_2025-02-01T01-52-07.473628.json`  114,643 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Phi4-RP-o1-Ablit/results_2025-02-25T02-45-03.072006.json`  113,499 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_Porpoise-R1-Llama3.2-3b/results_2025-02-18T19-24-07.519824.json`  113,509 bytes
- `C_efficiency_evolution/detailed_results/Triangle104_RomboHermes3-R1-Llama3.2-3b/results_2025-02-18T14-53-48.172245.json`  113,613 bytes
- `C_efficiency_evolution/detailed_results/TTTXXX01_Mistral-7B-Base-SimPO2-5e-7/results_2024-09-03T07-49-40.799646.json`  119,058 bytes
- `C_efficiency_evolution/detailed_results/tugstugi_Qwen2.5-7B-Instruct-QwQ-v0.1/results_2025-01-20T01-06-51.096821.json`  117,441 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Gemma-2-9B-It-SPPO-Iter2/results_2024-08-13T00-42-58.730890.json`  118,903 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Gemma-2-9B-It-SPPO-Iter3/results_2024-08-12T11-41-39.699261.json`  118,922 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Llama-3-Instruct-8B-SPPO-Iter3/results_2024-06-27T00-39-40.520034.json`  118,785 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Mistral7B-PairRM-SPPO/results_2024-09-27T19-16-54.672314.json`  119,129 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Mistral7B-PairRM-SPPO-Iter2/results_2024-08-12T23-18-02.723037.json`  119,187 bytes
- `C_efficiency_evolution/detailed_results/UCLA-AGI_Mistral7B-PairRM-SPPO-Iter3/results_2024-08-12T23-19-55.089024.json`  119,149 bytes
- `C_efficiency_evolution/detailed_results/UKzExecution_LlamaExecutor-8B-3.0.5/results_2024-08-13T05-48-33.833062.json`  123,500 bytes
- `C_efficiency_evolution/detailed_results/unsloth_Llama-3.2-1B-Instruct-no-system-message/results_2025-01-24T03-48-55.154490.json`  115,326 bytes
- `C_efficiency_evolution/detailed_results/unsloth_phi-4-unsloth-bnb-4bit/results_2025-01-09T10-28-37.423992.json`  115,165 bytes
- `C_efficiency_evolution/detailed_results/uukuguy_speechless-code-mistral-7b-v1.0/results_2024-07-22T12-34-36.172307.json`  118,619 bytes
- `C_efficiency_evolution/detailed_results/uukuguy_speechless-instruct-mistral-7b-v0.2/results_2024-08-13T03-33-29.481718.json`  118,672 bytes
- `C_efficiency_evolution/detailed_results/uukuguy_speechless-llama2-hermes-orca-platypus-wizardlm-13b/results_2024-08-13T04-12-53.046407.json`  118,685 bytes
- `C_efficiency_evolution/detailed_results/v000000_Qwen2.5-14B-Gutenberg-1e-Delta/results_2024-09-28T11-15-58.328773.json`  120,957 bytes
- `C_efficiency_evolution/detailed_results/v000000_Qwen2.5-Lumen-14B/results_2024-10-02T02-58-23.331254.json`  120,930 bytes
- `C_efficiency_evolution/detailed_results/VAGOsolutions_Llama-3-SauerkrautLM-70b-Instruct/results_2024-07-21T07-14-53.671049.json`  119,126 bytes
- `C_efficiency_evolution/detailed_results/VAGOsolutions_Llama-3-SauerkrautLM-8b-Instruct/results_2024-07-25T10-20-19.611472.json`  119,091 bytes
- `C_efficiency_evolution/detailed_results/VAGOsolutions_Llama-3.1-SauerkrautLM-8b-Instruct/results_2024-07-30T04-09-12.174878.json`  119,093 bytes
- `C_efficiency_evolution/detailed_results/VAGOsolutions_SauerkrautLM-gemma-2-9b-it/results_2024-09-02T17-46-00.346913.json`  119,113 bytes
- `C_efficiency_evolution/detailed_results/VAGOsolutions_SauerkrautLM-Gemma-2b/results_2024-07-18T11-45-39.813638.json`  118,889 bytes
- `C_efficiency_evolution/detailed_results/ValiantLabs_Llama3.1-70B-ShiningValiant2/results_2024-10-31T06-27-05.966661.json`  118,644 bytes
- `C_efficiency_evolution/detailed_results/ValiantLabs_Llama3.1-8B-Fireplace2/results_2024-07-25T21-07-29.515450.json`  119,057 bytes
- `C_efficiency_evolution/detailed_results/ValiantLabs_Llama3.2-3B-Esper2/results_2024-10-10T08-27-15.528214.json`  118,612 bytes
- `C_efficiency_evolution/detailed_results/vhab10_llama-3-8b-merged-linear/results_2024-09-26T16-47-03.367890.json`  119,097 bytes
- `C_efficiency_evolution/detailed_results/vhab10_Llama-3.1-8B-Base-Instruct-SLERP/results_2024-10-04T14-02-43.313749.json`  118,657 bytes
- `C_efficiency_evolution/detailed_results/vhab10_Llama-3.2-Instruct-3B-TIES/results_2024-11-23T13-02-17.445185.json`  114,704 bytes
- `C_efficiency_evolution/detailed_results/vicgalle_Configurable-Hermes-2-Pro-Llama-3-8B/results_2024-07-31T15-11-17.909344.json`  118,954 bytes
- `C_efficiency_evolution/detailed_results/vicgalle_Humanish-RP-Llama-3.1-8B/results_2024-08-12T18-10-33.666915.json`  119,041 bytes
- `C_efficiency_evolution/detailed_results/vicgalle_Roleplay-Llama-3-8B/results_2024-06-28T08-38-22.786427.json`  118,739 bytes
- `C_efficiency_evolution/detailed_results/VIRNECT_llama-3-Korean-8B-r-v-0.1/results_2024-09-14T02-33-08.958613.json`  119,178 bytes
- `C_efficiency_evolution/detailed_results/vonjack_Qwen2.5-Coder-0.5B-Merged/results_2024-11-20T00-06-17.841413.json`  116,913 bytes
- `C_efficiency_evolution/detailed_results/wave-on-discord_qwent-7b/results_2024-10-01T03-14-19.789003.json`  118,262 bytes
- `C_efficiency_evolution/detailed_results/weathermanj_Menda-3b-Optim-100/results_2025-03-10T22-28-20.086194.json`  116,203 bytes
- `C_efficiency_evolution/detailed_results/Weyaxi_Einstein-v6.1-developed-by-Weyaxi-Llama3-8B/results_2024-07-18T10-30-14.113211.json`  119,032 bytes
- `C_efficiency_evolution/detailed_results/Weyaxi_Einstein-v6.1-Llama3-8B/results_2024-07-17T10-00-42.383986.json`  118,991 bytes
- `C_efficiency_evolution/detailed_results/Weyaxi_Einstein-v7-Qwen2-7B/results_2024-07-18T09-58-17.163893.json`  118,968 bytes
- `C_efficiency_evolution/detailed_results/win10_EVA-Norns-Qwen2.5-v0.1/results_2024-11-19T23-36-26.139733.json`  115,001 bytes
- `C_efficiency_evolution/detailed_results/win10_Norns-Qwen2.5-12B/results_2024-11-20T02-10-23.946831.json`  114,900 bytes
- `C_efficiency_evolution/detailed_results/win10_Norns-Qwen2.5-7B/results_2024-11-19T23-36-10.410545.json`  114,976 bytes
- `C_efficiency_evolution/detailed_results/win10_Qwen2.5-2B-Instruct/results_2024-12-20T23-57-48.525040.json`  114,503 bytes
- `C_efficiency_evolution/detailed_results/wzhouad_gemma-2-9b-it-WPO-HB/results_2025-01-08T04-51-06.266732.json`  114,656 bytes
- `C_efficiency_evolution/detailed_results/x0000001_Deepseek-Lumen-R1-Qwen2.5-14B/results_2025-01-30T04-29-30.938714.json`  116,858 bytes
- `C_efficiency_evolution/detailed_results/Xiaojian9992024_Llama3.2-1B-THREADRIPPER/results_2025-02-03T12-11-47.982597.json`  118,807 bytes
- `C_efficiency_evolution/detailed_results/Xiaojian9992024_Llama3.2-1B-THREADRIPPER-v0.2/results_2025-02-05T00-27-48.058628.json`  118,813 bytes
- `C_efficiency_evolution/detailed_results/Xiaojian9992024_Qwen2.5-THREADRIPPER-Medium-Censored/results_2025-02-18T17-17-02.948632.json`  116,254 bytes
- `C_efficiency_evolution/detailed_results/Xiaojian9992024_Qwen2.5-Ultra-1.5B-25.02-Exp/results_2025-02-25T00-14-30.453157.json`  116,701 bytes
- `C_efficiency_evolution/detailed_results/xinchen9_Llama3.1_8B_Instruct_CoT/results_2024-09-18T04-49-40.558012.json`  118,621 bytes
- `C_efficiency_evolution/detailed_results/xinchen9_Llama3.1_CoT_V1/results_2024-09-10T17-43-01.266913.json`  118,619 bytes
- `C_efficiency_evolution/detailed_results/xinchen9_Mistral-7B-CoT/results_2024-09-19T06-54-45.305377.json`  118,580 bytes
- `C_efficiency_evolution/detailed_results/xkp24_Llama-3-8B-Instruct-SPPO-Iter2_gp_2b-table/results_2024-09-30T08-02-09.476119.json`  119,135 bytes
- `C_efficiency_evolution/detailed_results/xkp24_Llama-3-8B-Instruct-SPPO-Iter2_gp_8b-table/results_2024-09-30T08-55-19.449352.json`  119,142 bytes
- `C_efficiency_evolution/detailed_results/xkp24_Llama-3-8B-Instruct-SPPO-score-Iter2_gp_8b-table-0.002/results_2024-10-02T02-36-39.927047.json`  119,166 bytes
- `C_efficiency_evolution/detailed_results/xukp20_Llama-3-8B-Instruct-SPPO-Iter3_bt_2b-table/results_2024-09-29T03-19-27.914131.json`  119,141 bytes
- `C_efficiency_evolution/detailed_results/xukp20_Llama-3-8B-Instruct-SPPO-Iter3_bt_8b-table/results_2024-09-29T02-50-37.872205.json`  119,128 bytes
- `C_efficiency_evolution/detailed_results/xukp20_Llama-3-8B-Instruct-SPPO-Iter3_gp_2b-table/results_2024-09-29T03-03-23.153786.json`  119,126 bytes
- `C_efficiency_evolution/detailed_results/yam-peleg_Hebrew-Mistral-7B/results_2024-07-29T09-03-40.276470.json`  118,596 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-Iter1_bt_2b-table/results_2024-09-30T03-09-25.381758.json`  119,102 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-Iter1_bt_8b-table/results_2024-09-30T03-04-53.482652.json`  119,115 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-Iter1_gp_2b-table/results_2024-09-29T15-12-59.528765.json`  119,137 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-Iter1_gp_8b-table/results_2024-09-29T15-24-35.214199.json`  119,137 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-score-Iter1_bt_2b-table-0.001/results_2024-09-30T03-05-19.684483.json`  119,168 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-score-Iter1_gp_2b-table-0.001/results_2024-09-29T15-18-45.245227.json`  119,169 bytes
- `C_efficiency_evolution/detailed_results/yfzp_Llama-3-8B-Instruct-SPPO-score-Iter1_gp_8b-table-0.002/results_2024-09-29T15-31-50.382131.json`  119,160 bytes
- `C_efficiency_evolution/detailed_results/yifAI_Llama-3-8B-Instruct-SPPO-score-Iter3_gp_8b-table-0.002/results_2024-09-30T03-03-28.510156.json`  119,164 bytes
- `C_efficiency_evolution/detailed_results/ymcki_gemma-2-2b-jpn-it-abliterated-17/results_2024-10-17T11-26-10.721815.json`  118,597 bytes
- `C_efficiency_evolution/detailed_results/ymcki_gemma-2-2b-jpn-it-abliterated-17-18-24/results_2024-11-06T19-05-49.169139.json`  114,954 bytes
- `C_efficiency_evolution/detailed_results/ymcki_gemma-2-2b-jpn-it-abliterated-17-ORPO-alpaca/results_2024-10-27T13-22-32.303741.json`  118,756 bytes
- `C_efficiency_evolution/detailed_results/ymcki_gemma-2-2b-ORPO-jpn-it-abliterated-18/results_2024-10-30T19-59-00.170912.json`  118,569 bytes
- `C_efficiency_evolution/detailed_results/ymcki_gemma-2-2b-ORPO-jpn-it-abliterated-18-merge/results_2024-10-30T17-06-58.119904.json`  118,900 bytes
- `C_efficiency_evolution/detailed_results/Youlln_ECE-Qwen0.5B-FT-V2/results_2024-10-12T07-11-52.657109.json`  118,620 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-0505/results_2025-01-27T16-50-10.808851.json`  114,639 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-0805/results_2025-01-27T10-13-49.522292.json`  114,640 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-1010/results_2025-01-25T10-29-09.986638.json`  114,657 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-latest/results_2025-02-04T03-18-38.619840.json`  114,654 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-latest-V2/results_2025-02-24T23-24-51.650002.json`  116,130 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-SCE/results_2025-02-01T13-00-06.843853.json`  114,666 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-V4/results_2025-03-03T07-41-39.178588.json`  116,100 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-14B-YOYO-V4-p2/results_2025-03-01T12-31-25.964204.json`  116,183 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_Qwen2.5-7B-it-restore/results_2025-03-10T05-50-55.816942.json`  116,136 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_ZYH-LLM-Qwen2.5-14B/results_2025-02-05T07-57-59.664680.json`  114,636 bytes
- `C_efficiency_evolution/detailed_results/YOYO-AI_ZYH-LLM-Qwen2.5-14B-V2/results_2025-02-18T21-18-55.492202.json`  113,513 bytes
- `C_efficiency_evolution/detailed_results/Yuma42_Llama3.1-IgneousIguana-8B/results_2025-03-10T16-47-02.867519.json`  113,951 bytes
- `C_efficiency_evolution/detailed_results/yuvraj17_Llama3-8B-abliterated-Spectrum-slerp/results_2024-09-23T18-22-53.409708.json`  118,693 bytes
- `C_efficiency_evolution/detailed_results/yuvraj17_Llama3-8B-SuperNova-Spectrum-dare_ties/results_2024-09-27T19-05-22.624706.json`  118,707 bytes
- `C_efficiency_evolution/detailed_results/yuvraj17_Llama3-8B-SuperNova-Spectrum-Hermes-DPO/results_2024-10-01T08-33-11.491722.json`  118,972 bytes
- `C_efficiency_evolution/detailed_results/zake7749_gemma-2-2b-it-chinese-kyara-dpo/results_2024-10-17T04-10-42.957748.json`  118,611 bytes
- `C_efficiency_evolution/detailed_results/zelk12_gemma-2-S2MTM-9B/results_2024-12-12T18-13-21.682997.json`  115,102 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen1-gemma-2-9B/results_2024-10-23T21-24-17.413275.json`  119,200 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen2-gemma-2-9B/results_2024-11-10T22-32-55.379788.json`  115,239 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen2-GI-gemma-2-9B/results_2024-11-29T22-30-55.978828.json`  115,285 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen3-gemma-2-9B/results_2024-11-30T14-08-57.787789.json`  115,265 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen6-gemma-2-9B/results_2025-01-24T01-42-30.885051.json`  114,918 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Gen7-gemma-2-9B/results_2025-02-25T02-04-11.408754.json`  113,871 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Merge-gemma-2-9B/results_2024-10-23T19-38-04.039402.json`  119,211 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Merge2-MU-gemma-2-MTg2MT1g2-9B/results_2024-11-29T22-29-59.697100.json`  115,309 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Merge4-gemma-2-9B/results_2024-12-21T19-08-08.651796.json`  115,249 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT-Merge6-gemma-2-9B/results_2025-02-25T03-01-55.229984.json`  114,146 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT1-gemma-2-9B/results_2024-10-17T12-20-18.137349.json`  119,199 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT1-Gen2-gemma-2-9B/results_2024-11-11T20-13-38.140139.json`  115,256 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT1-Gen5-gemma-2-9B/results_2024-12-25T20-34-21.650551.json`  115,172 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen1-gemma-2-9B/results_2024-10-27T22-45-53.785140.json`  119,213 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen2-gemma-2-9B/results_2024-11-12T18-32-55.388652.json`  115,277 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen3-gemma-2-9B/results_2024-12-05T10-21-35.003790.json`  115,270 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen4-gemma-2-9B/results_2024-12-18T05-23-35.675576.json`  115,234 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen5-gemma-2-9B/results_2024-12-25T21-16-19.023125.json`  115,199 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Gen6-gemma-2-9B/results_2025-02-06T03-58-58.215439.json`  114,925 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT2-Max-Merge_02012025163610-gemma-2-9B/results_2025-01-08T05-03-03.850406.json`  115,319 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-gemma-2-9B/results_2024-10-17T17-43-50.543791.json`  119,212 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen1-gemma-2-9B/results_2024-10-29T01-41-50.029786.json`  119,154 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen2-gemma-2-9B/results_2024-11-21T18-43-33.559212.json`  115,168 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen3-gemma-2-9B/results_2024-12-08T00-57-24.501650.json`  115,248 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen4-gemma-2-9B/results_2024-12-18T05-51-16.035979.json`  115,260 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen5-gemma-2-9B/results_2024-12-26T15-15-01.178166.json`  115,244 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen5-gemma-2-9B_v1/results_2024-12-28T00-24-09.037062.json`  115,263 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Gen6-gemma-2-9B/results_2025-02-25T02-19-01.013692.json`  113,876 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT3-Max-Merge_02012025163610-gemma-2-9B/results_2025-01-14T02-07-53.374345.json`  115,322 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT4-gemma-2-9B/results_2024-10-21T19-43-45.509063.json`  119,193 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT4-Gen2-gemma-2-9B/results_2024-11-22T19-52-54.972355.json`  115,276 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT4-Gen3-gemma-2-9B/results_2024-12-08T13-20-29.805102.json`  115,269 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT4-Gen5-gemma-2-9B/results_2024-12-28T22-03-36.293690.json`  115,247 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT5-gemma-2-9B/results_2024-10-21T21-26-23.941227.json`  119,212 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT5-Gen1-gemma-2-9B/results_2024-11-06T18-23-09.006291.json`  115,273 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT5-Gen2-gemma-2-9B/results_2024-11-23T14-41-47.446807.json`  115,237 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MT5-Gen3-gemma-2-9B/results_2024-12-08T19-48-01.131041.json`  115,227 bytes
- `C_efficiency_evolution/detailed_results/zelk12_MTM-Merge-gemma-2-9B/results_2025-01-01T19-29-02.979642.json`  115,239 bytes
- `C_efficiency_evolution/detailed_results/zelk12_recoilme-gemma-2-Gutenberg-Doppel-9B-v0.1/results_2024-10-08T20-36-56.622261.json`  119,251 bytes
- `C_efficiency_evolution/detailed_results/zelk12_recoilme-gemma-2-Ifable-9B-v0.1/results_2024-10-08T20-39-10.045354.json`  119,256 bytes
- `C_efficiency_evolution/detailed_results/zelk12_recoilme-gemma-2-psy10k-mental_healt-9B-v0.1/results_2024-10-09T13-20-24.603938.json`  119,294 bytes
- `C_efficiency_evolution/detailed_results/zelk12_Rv0.4DMv1t0.25-gemma-2-9B/results_2024-12-31T17-18-49.680300.json`  115,287 bytes
- `C_efficiency_evolution/detailed_results/zelk12_Test01012025155054t0.5_gemma-2/results_2025-01-02T01-12-02.762534.json`  114,953 bytes
- `C_efficiency_evolution/detailed_results/ZeroXClem_Llama-3.1-8B-AthenaSky-MegaMix/results_2025-03-12T15-57-21.995435.json`  118,439 bytes
- `C_efficiency_evolution/detailed_results/ZeroXClem_Llama-3.1-8B-SuperNova-EtherealHermes/results_2025-03-12T16-02-34.311760.json`  114,125 bytes
- `C_efficiency_evolution/detailed_results/ZeroXClem_Qwen-2.5-Aether-SlerpFusion-7B/results_2024-11-21T13-44-53.382689.json`  117,292 bytes
- `C_efficiency_evolution/detailed_results/ZeroXClem_Qwen2.5-7B-HomerAnvita-NerdMix/results_2024-11-22T00-34-20.371295.json`  117,349 bytes
- `C_efficiency_evolution/detailed_results/zetasepic_Qwen2.5-32B-Instruct-abliterated-v2/results_2025-01-11T00-05-13.596997.json`  117,025 bytes
- `C_efficiency_evolution/detailed_results/zetasepic_Qwen2.5-72B-Instruct-abliterated/results_2024-11-12T12-05-07.437154.json`  114,687 bytes
- `C_efficiency_evolution/pythia_1.4b_eval_details/README.md`  63,450 bytes
- `C_efficiency_evolution/pythia_12b_eval_details/README.md`  72,327 bytes
- `C_efficiency_evolution/pythia_160m_eval_details/README.md`  72,028 bytes
- `C_efficiency_evolution/pythia_2.8b_eval_details/README.md`  72,145 bytes
- `C_efficiency_evolution/pythia_410m_eval_details/README.md`  72,065 bytes
- `C_efficiency_evolution/pythia_6.9b_eval_details/README.md`  72,470 bytes
- `C_efficiency_evolution/pythia_eval_details/README.md`  63,185 bytes

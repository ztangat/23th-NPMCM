# -*- coding: utf-8 -*-
"""
common.py —— 问题一公共工具库
统一：路径、日志、数据读取、指标名录、域映射、结果落盘
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
ROOT = r"C:/Users/dkyyt/Desktop/F题"
ATT = os.path.join(ROOT, "real_attachments")
ATT_A = os.path.join(ATT, "A_data_value")
OUT = os.path.join(ROOT, "deepseek")
DATA = os.path.join(OUT, "data")
RESULTS = os.path.join(OUT, "results")
FIG = os.path.join(RESULTS, "figures")
TABLES = os.path.join(RESULTS, "tables")
LOGS = os.path.join(OUT, "logs")
REPORTS = os.path.join(OUT, "reports")
for d in (DATA, RESULTS, FIG, TABLES, LOGS, REPORTS):
    os.makedirs(d, exist_ok=True)

# 质量信号文件
F_SAMPLE = os.path.join(ATT_A, "slimpajama_quality_signal_sample.jsonl",
                        "slimpajama_quality_signal_sample.jsonl")
F_ARXIV = os.path.join(ATT_A, "slimpajama_quality_extended",
                       "arxiv_part-6777d8857c6e-000486.jsonl",
                       "arxiv_part-6777d8857c6e-000486.jsonl")
F_GITHUB = os.path.join(ATT_A, "slimpajama_quality_extended",
                        "github_part-6777d8857c6e-000275.jsonl",
                        "github_part-6777d8857c6e-000275.jsonl")
F_DOMAINSAMPLE = os.path.join(ATT_A, "regmix_domain_sample.jsonl", "regmix_domain_sample.jsonl")
MIX_DIR = os.path.join(ATT_A, "regmix_tables")
F_MAPGUIDE = os.path.join(ATT_A, "domain_mapping_guide.csv")
F_DOMSUMMARY = os.path.join(ATT_A, "regmix_domain_summary.csv")

# ---------------- 日志 ----------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()
    def flush(self):
        self.stdout.flush()
        self.f.flush()

def start_log(name):
    p = os.path.join(LOGS, f"{name}.log")
    sys.stdout = Tee(p)
    print(f"[START] {name} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return p

def end_log(name):
    print(f"[END] {name} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = sys.stdout.stdout

# ---------------- 指标名录 ----------------
# 22 个质量指标字段
SCALAR_METRICS = [
    "dsir_books", "dsir_wiki", "dsir_math",
    "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
# 8 个列表型指标（需压缩为标量）
LIST_METRICS = [
    "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism",
    "qurater", "ad_en", "fineweb_edu",
]
ALL_METRICS = SCALAR_METRICS + LIST_METRICS

# 指标方向（+1：原始值越高越好；-1：原始值越低越好；0：需按信息量/中性处理）
# 依据：dsir_* 为 DSIR 重要性权重（相对分布的对数比，越高代表越贴近高价值语料），因此越高越好
# fluency_en/modernbert_*/qurater/fineweb_edu 为分类器打分（对数几率型），越高越好
# ad_en 为广告含量分类器，越高代表广告越多 → 越低越好
# rps_* 为文本统计特征，其"好"的方向需结合语义：unique_words/entropy 越高越好；
#   no_alph_words / frac_chars_top_2gram / frac_chars_top_3gram / uppercase_letter_fraction
#   / numerical_chars_fraction 越高代表噪声或重复越多 → 越低越好；
#   word_count / num_sentences 为长度类，过短不好但过长不等价于好 → 采用区间型处理
#   mean_word_length 过长过短都不好 → 区间型
#   ending_with_terminal_punctution_mark 越高越好（句子完整）
METRIC_DIRECTION = {
    # 越高越好
    "dsir_books": +1, "dsir_wiki": +1, "dsir_math": +1,
    "fluency_en": +1, "modernbert_cleanliness": +1, "modernbert_readability": +1,
    "modernbert_reasoning": +1, "modernbert_professionalism": +1,
    "qurater": +1, "fineweb_edu": +1,
    "rps_doc_unigram_entropy": +1, "rps_doc_frac_unique_words": +1,
    "rps_lines_ending_with_terminal_punctution_mark": +1,
    # 越低越好（负向指标）
    "ad_en": -1,
    "rps_doc_frac_no_alph_words": -1,
    "rps_doc_frac_chars_top_2gram": -1,
    "rps_doc_frac_chars_top_3gram": -1,
    "rps_lines_uppercase_letter_fraction": -1,
    "rps_lines_numerical_chars_fraction": -1,
    # 区间型（长度/词长类）
    "rps_doc_word_count": 0,
    "rps_doc_num_sentences": 0,
    "rps_doc_mean_word_length": 0,
}

METRIC_CN = {
    "fineweb_edu": "FineWeb教育价值",
    "fluency_en": "英语流畅度",
    "modernbert_cleanliness": "ModernBERT干净度",
    "modernbert_readability": "ModernBERT可读性",
    "modernbert_reasoning": "ModernBERT推理性",
    "modernbert_professionalism": "ModernBERT专业性",
    "dsir_books": "DSIR图书相似度",
    "dsir_wiki": "DSIR百科相似度",
    "dsir_math": "DSIR数学相似度",
    "qurater": "QuRater质量等级",
    "ad_en": "英文广告含量",
    "rps_doc_word_count": "文档词数",
    "rps_doc_num_sentences": "文档句数",
    "rps_doc_unigram_entropy": "一元词熵",
    "rps_doc_frac_unique_words": "唯一词占比",
    "rps_doc_frac_no_alph_words": "非字母词占比",
    "rps_doc_frac_chars_top_2gram": "最高频2-gram字符占比",
    "rps_doc_frac_chars_top_3gram": "最高频3-gram字符占比",
    "rps_lines_uppercase_letter_fraction": "大写字母行占比",
    "rps_lines_ending_with_terminal_punctution_mark": "句末标点行占比",
    "rps_lines_numerical_chars_fraction": "数字字符行占比",
    "rps_doc_mean_word_length": "平均词长",
}

# 指标族分组（用于冲突成因分析）
METRIC_FAMILY = {
    "dsir_books": "DSIR重要性", "dsir_wiki": "DSIR重要性", "dsir_math": "DSIR重要性",
    "fineweb_edu": "分类器打分", "fluency_en": "分类器打分",
    "modernbert_cleanliness": "分类器打分", "modernbert_readability": "分类器打分",
    "modernbert_reasoning": "分类器打分", "modernbert_professionalism": "分类器打分",
    "qurater": "分类器打分", "ad_en": "分类器打分",
    "rps_doc_word_count": "文本统计", "rps_doc_num_sentences": "文本统计",
    "rps_doc_unigram_entropy": "文本统计", "rps_doc_frac_unique_words": "文本统计",
    "rps_doc_frac_no_alph_words": "文本统计", "rps_doc_frac_chars_top_2gram": "文本统计",
    "rps_doc_frac_chars_top_3gram": "文本统计",
    "rps_lines_uppercase_letter_fraction": "文本统计",
    "rps_lines_ending_with_terminal_punctution_mark": "文本统计",
    "rps_lines_numerical_chars_fraction": "文本统计",
    "rps_doc_mean_word_length": "文本统计",
}

# 17 个配方域
MIXTURE_DOMAINS = [
    "arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
    "dm_mathematics", "github", "philpapers", "stackexchange", "enron_emails",
    "gutenberg_pg_19", "pile_cc", "ubuntu_irc", "europarl", "hackernews",
    "pubmed_abstracts", "uspto_backgrounds",
]
# 有 Loss 列的 13 个域
LOSS_DOMAINS = [
    "arxiv", "freelaw", "pubmed_central", "wikipedia_en", "dm_mathematics",
    "github", "stackexchange", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
    "hackernews", "pubmed_abstracts", "uspto_backgrounds",
]
# 质量信号覆盖的 7 个域
QUALITY_DOMAINS = ["arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"]

# ---------------- 工具函数 ----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_np_conv)

def _np_conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def md_table(df, floatfmt="%.4f"):
    """把 DataFrame 转为 markdown 表格"""
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].map(lambda x: (floatfmt % x) if pd.notna(x) else "")
    head = "| " + " | ".join(str(c) for c in df2.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df2.columns)) + "|"
    rows = []
    for _, r in df2.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join([head, sep] + rows)

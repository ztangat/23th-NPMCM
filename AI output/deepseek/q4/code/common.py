# -*- coding: utf-8 -*-
"""
common.py —— 问题四公共工具库

问题四：技术演进分析与前沿预测

【核心任务】
  1) 分离并量化"规模扩张"与"非规模技术进步"对能力提升的贡献占比
  2) 在算力增长放缓情景下，预测未来 12/24 个月开源大模型能力前沿边界 + 不确定性
  3) Loss–Benchmark 桥接（衔接前三问的交叉熵损失结论）

【数据接口（附件 C）】
  C1 leaderboard_cleaned.csv              排行榜（4576 行，6 benchmark + Average）
  C2 leaderboard_enhanced.csv             增强（+Epoch AI 发布日期/组织/开源权重）
  C3 leaderboard_extended_timeseries.csv  时序（含 2019-2022 历史模型）
  C4 epoch_all_ai_models.csv              全模型元数据（3523 行，57 字段，含算力/数据量）
  C5 loss_benchmark_bridge.csv            桥接（43 条）
  C6 loss_benchmark_bridge_expanded.csv   桥接扩展（75 条，含可比性等级）
  C7 model_architecture_metadata.csv      架构元数据
  C8 detailed_results/                    逐任务评测 JSON（1863 目录）

【口径约定（赛题要求说明）】
  · 综合能力度量：Open LLM Leaderboard 的 Average ⬆️（6 项 benchmark 归一化均值）
      6 项：IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO
  · 开源筛选口径：Hub License 明确 + Epoch_AI_Open_Weights=Yes（首选）；备选放宽到
      非闭源（Type 非 proprietary / 有 Hub License）
  · 模型类型：pretrained（🟢/🟩）vs chat/finetuned（💬 / 🔶 / 🤝）
  · 时间轴口径：主用 Submission Date（提交日期，作为"能力可用时点"）；
      辅以 Epoch_AI_Publication_Date（发布日期）做稳健性检验
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
ROOT = r"C:/Users/dkyyt/Desktop/F题"
Q1 = os.path.join(ROOT, "deepseek", "q1")
Q2 = os.path.join(ROOT, "deepseek", "q2")
Q3 = os.path.join(ROOT, "deepseek", "q3")
Q4 = os.path.join(ROOT, "deepseek", "q4")
C_DIR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution")

CODE = os.path.join(Q4, "code")
DATA = os.path.join(Q4, "data")
LOGS = os.path.join(Q4, "logs")
REPORTS = os.path.join(Q4, "reports")
RESULTS = os.path.join(Q4, "results")
TABLES = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")

for _p in (CODE, DATA, LOGS, REPORTS, TABLES, FIG):
    os.makedirs(_p, exist_ok=True)

# ---------------- 核心口径 ----------------
BENCH_COLS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
BENCH_COLS_ALT = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]  # C3 口径
AVG_COL = "Average ⬆️"

# 开源许可能够研究与复现的（白名单）
OS_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc-by-sa-4.0",
               "cc-by-nc-4.0", "llama2", "llama3", "llama3.1", "llama3.2",
               "gemma", "qwen", "openrail", "creativeml-openrail-m", "other",
               "cc-by-nc-sa-4.0", "gpl-3.0", "bigscience-bloom-rail-1.0",
               "openrail++", "cc-by-nc-nd-4.0", "bsd-2-clause", "mpl-2.0"}

# 模型类型归并
def classify_type(type_str):
    if not isinstance(type_str, str):
        return "unknown"
    if "🟢" in type_str or "🟩" in type_str:
        return "pretrained"
    if "💬" in type_str or "🔶" in type_str or "🤝" in type_str:
        return "chat_finetuned"
    return "other"


# ---------------- 数据读取 ----------------
def load_C1():
    d = pd.read_csv(os.path.join(C_DIR, "leaderboard_cleaned.csv"))
    d["submission_dt"] = pd.to_datetime(d["Submission Date"], errors="coerce")
    d["type_grp"] = d["Type"].map(classify_type)
    d["open_license"] = d["Hub License"].astype(str).str.lower().isin(OS_LICENSES)
    d["has_license"] = d["Hub License"].notna()
    return d


def load_C2():
    d = pd.read_csv(os.path.join(C_DIR, "leaderboard_enhanced.csv"))
    d["submission_dt"] = pd.to_datetime(d["Submission Date"], errors="coerce")
    d["pub_dt"] = pd.to_datetime(d["Epoch_AI_Publication_Date"], errors="coerce")
    d["type_grp"] = d["Type"].map(classify_type)
    d["open_weights"] = d["Epoch_AI_Open_Weights"].astype(str).str.lower().eq("yes")
    d["open_license"] = d["Hub License"].astype(str).str.lower().isin(OS_LICENSES)
    return d


def load_C3():
    return pd.read_csv(os.path.join(C_DIR, "leaderboard_extended_timeseries.csv"))


def load_C4():
    return pd.read_csv(os.path.join(C_DIR, "epoch_all_ai_models.csv"), low_memory=False)


def load_bridge(expanded=True):
    fn = "loss_benchmark_bridge_expanded.csv" if expanded else "loss_benchmark_bridge.csv"
    return pd.read_csv(os.path.join(C_DIR, fn))


def load_C7():
    return pd.read_csv(os.path.join(C_DIR, "model_architecture_metadata.csv"))


def c8_model_dirs():
    base = os.path.join(C_DIR, "detailed_results")
    return [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]


# ---------------- 日志与保存 ----------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.f.write(s)

    def flush(self):
        self.stdout.flush(); self.f.flush()


_cur_tee = None


def start_log(name):
    global _cur_tee
    _cur_tee = Tee(os.path.join(LOGS, f"{name}.log"))
    sys.stdout = _cur_tee
    print(f"[START] {name} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")


def end_log(name):
    global _cur_tee
    print(f"[END] {name} @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    if _cur_tee:
        sys.stdout = _cur_tee.stdout
        _cur_tee.f.close()
        _cur_tee = None


def save_json(obj, path):
    def _d(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_d)
    print(f"  [save] {path}")

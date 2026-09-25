# -*- coding: utf-8 -*-
"""
common.py —— 问题二公共工具库

定义路径、附件 B 数据字典、可信度分层、Q 读取接口（来自问题一）、日志与保存工具。

问题二数据：附件 B（B_scaling_laws），12 个文件。
问题一输出：质量向量 q（q1/data/s4_domain_Q.npz）—— 通过 Q 引入跨问联系。
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
ROOT = r"C:/Users/dkyyt/Desktop/F题"
Q1 = os.path.join(ROOT, "deepseek", "q1")          # 问题一产出
Q2 = os.path.join(ROOT, "deepseek", "q2")          # 问题二产出（本问）
B_DIR = os.path.join(ROOT, "real_attachments", "B_scaling_laws")
A_DIR = os.path.join(ROOT, "real_attachments", "A_data_value")

CODE = os.path.join(Q2, "code")
DATA = os.path.join(Q2, "data")
LOGS = os.path.join(Q2, "logs")
REPORTS = os.path.join(Q2, "reports")
RESULTS = os.path.join(Q2, "results")
TABLES = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")

for _p in (CODE, DATA, LOGS, REPORTS, TABLES, FIG):
    os.makedirs(_p, exist_ok=True)

# ---------------- 附件 B 数据字典 ----------------
# 编号 -> (文件名, 规模描述, 可信度, 用途)
B_FILES = {
    "B1":  ("pythia_training_log_existing.csv",           "1176 行×15",  "真实",   "主拟合"),
    "B2":  ("cerebras_training_log.csv",                  "1029 行×15",  "半合成", "族外验证"),
    "B3":  ("training_trajectories",                      "8×500",       "插值",   "轨迹验证"),
    "B4":  ("scaling_baseline.csv",                       "57 行×5",     "真实",   "跨族验证"),
    "B5":  ("published_scaling_data.csv",                 "44 行×6",     "真实",   "文献验证"),
    "B6":  ("supplementary_NQ_experiment.csv",            "360 行×5",    "半合成", "质量Q基础"),
    "B7":  ("supplementary_NQ_experiment_expanded.csv",   "450 行×5",    "半合成", "质量Q扩展"),
    "B8":  ("supplementary_NQ_experiment_large.csv",      "1704 行×6",   "半合成", "质量Q大规模"),
    "B9":  ("supplementary_large_models.csv",             "132 行×8",    "真实",   "大模型参数"),
    "B10": ("supplementary_large_baseline.csv",           "128 行×5",    "估算",   "大模型外推"),
    "B11": ("open_model_family_metadata.csv",             "18 行×8",     "真实",   "辅助"),
    "B12": ("pythia_checkpoint_index.csv",                "1386 行×5",   "真实",   "辅助"),
}

# 可信度分级（用于加权与标注）
CREDIBILITY_ORDER = {"真实": 3, "插值": 2, "半合成": 1, "估算": 0}

# 附件 B 主要标量列
COLS_NDL = ["N_params_B", "D_tokens_B", "val_loss"]
COLS_NDQL = ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]


def pathB(key):
    fn = B_FILES[key][0]
    return os.path.join(B_DIR, fn)


def loadB(key):
    """载入附件 B 的某个 CSV 为 DataFrame（B3 返回目录内合并）"""
    p = pathB(key)
    if key == "B3":
        frames = []
        for f in sorted(os.listdir(p)):
            if f.endswith(".csv"):
                d = pd.read_csv(os.path.join(p, f))
                d["traj_file"] = f
                frames.append(d)
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(p)


# ---------------- 问题一输出接口 ----------------
def load_q_domains():
    """读取问题一 17 配方域质量 q。返回 (domains, Q, Q_sd, levels)"""
    z = np.load(os.path.join(Q1, "data", "s4_domain_Q.npz"), allow_pickle=True)
    return z["domains"], z["Q"], z["Q_sd"], z["levels"]


def load_q_mix_loss():
    """读取问题一配比模型关键结果（供配比项设计）"""
    z = np.load(os.path.join(Q1, "data", "s5_pred_train.npy"))
    return z


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
    p = os.path.join(LOGS, f"{name}.log")
    _cur_tee = Tee(p)
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


def md_table(df, floatfmt=".4f"):
    """DataFrame → markdown 表"""
    return df.to_markdown(index=False, floatfmt=floatfmt)

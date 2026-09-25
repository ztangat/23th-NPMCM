# -*- coding: utf-8 -*-
"""
common.py —— 问题三公共工具库

问题三：算力约束下的多维资源联合优化与结构性转移

【赛题给出的成本结构】（附录 B + 正文）
  总算力预算 C（考察 10^19 / 10^22 / 10^24 FLOPs 三档，可扩展）
  1) 基础训练开销:      C_train = 6·N·D
                        （N 为参数个数，D 为训练 token 数；B 单位须乘 1e9）
  2) 数据质量提升开销:  C_Q = D·[g(Q) - g(Q_0)]_+      （增量成本形式）
                        三族 g(Q)（附录 B.1）:
                          指数型:     g(Q) = γ·e^{λQ},       γ=1e7,  λ=6.0
                          幂函数型:   g(Q) = γ·Q^λ,         γ=5e9,  λ=4.0
                          对数渐进型: g(Q) = γ·ln(1+λQ),    γ=2e9,  λ=10.0
                        Q_0 为基线质量（取自附件 A / 问题一），Q ∈ (0,1]
  3) 长文本注意力开销:  C_attn = η·N·D·L_ctx      η=2e-4
                        L_ctx 为上下文窗口长度（外生给定，依据 C7）

【约束】C_train + C_Q + C_attn ≤ C
【临界值】L_ctx_crit = 6/η = 30000（使 C_attn = C_train）

【目标】min L(N, D, Q)  —— 用问题二的广义标度律
【衔接】Q ← 问题一 17 域质量 q 与配比 p（Q = Q_mix = Σ p_j q_j）
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
ROOT = r"C:/Users/dkyyt/Desktop/F题"
Q1 = os.path.join(ROOT, "deepseek", "q1")
Q2 = os.path.join(ROOT, "deepseek", "q2")
Q3 = os.path.join(ROOT, "deepseek", "q3")
C_DIR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution")

CODE = os.path.join(Q3, "code")
DATA = os.path.join(Q3, "data")
LOGS = os.path.join(Q3, "logs")
REPORTS = os.path.join(Q3, "reports")
RESULTS = os.path.join(Q3, "results")
TABLES = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")

for _p in (CODE, DATA, LOGS, REPORTS, TABLES, FIG):
    os.makedirs(_p, exist_ok=True)

# ---------------- 赛题常量 ----------------
ETA = 2e-4                 # 注意力开销系数
L_CTX_CRIT = 6.0 / ETA     # = 30000，C_attn = C_train 的临界上下文长度

# 附件 C7 给出的可行上下文长度（max_position_embeddings 唯一值）
L_CTX_FEASIBLE = [2048, 4096, 8192, 32768, 131072]

# 三档典型算力预算（FLOPs，赛题建议）
BUDGETS = {"低(1e19)": 1e19, "中(1e22)": 1e22, "高(1e24)": 1e24}

# 质量成本函数族（附录 B.1）
QUALITY_COST_FAMILIES = {
    "指数型":     {"g": lambda Q, g=1e7, l=6.0: g * np.exp(l * Q),          "gamma": 1e7, "lam": 6.0,
                   "note": "g(Q)=γ·e^{λQ}, γ=1e7, λ=6.0"},
    "幂函数型":   {"g": lambda Q, g=5e9, l=4.0: g * Q ** l,                 "gamma": 5e9, "lam": 4.0,
                   "note": "g(Q)=γ·Q^λ, γ=5e9, λ=4.0"},
    "对数渐进型": {"g": lambda Q, g=2e9, l=10.0: g * np.log1p(l * Q),       "gamma": 2e9, "lam": 10.0,
                   "note": "g(Q)=γ·ln(1+λQ), γ=2e9, λ=10.0"},
}


def g_of(family, Q):
    return QUALITY_COST_FAMILIES[family]["g"](Q)


def unit_check():
    """单位换乘：N,D 以 B 计 → 绝对个数需 ×1e9；C_train = 6·(N·1e9)·(D·1e9) = 6·N·D·1e18"""
    return 1e18


# ---------------- 前两问接口 ----------------
def load_q1_quality():
    """问题一：17 配方域质量 q 与映射层级"""
    z = np.load(os.path.join(Q1, "data", "s4_domain_Q.npz"), allow_pickle=True)
    return [str(x) for x in z["domains"]], z["Q"].astype(float), z["Q_sd"].astype(float), z["levels"]


def load_q1_mixture_weights():
    """问题一：真实配比权重（SlimPajama 风格，13 域有值）"""
    w = {"arxiv": 0.0493, "freelaw": 0.0598, "pubmed_central": 0.0586, "wikipedia_en": 0.0890,
         "dm_mathematics": 0.0338, "github": 0.0484, "stackexchange": 0.0699,
         "gutenberg_pg_19": 0.1047, "pile_cc": 0.1307, "ubuntu_irc": 0.0524,
         "hackernews": 0.1352, "pubmed_abstracts": 0.0816, "uspto_backgrounds": 0.0863}
    return w


def load_q2_scaling():
    """问题二：广义标度律参数（M4 整体质量弹性）与经典参数"""
    with open(os.path.join(Q2, "data", "s2c_final_params.json"), encoding="utf-8") as f:
        g = json.load(f)
    with open(os.path.join(Q2, "data", "s1_classic_params.json"), encoding="utf-8") as f:
        c = json.load(f)
    return g, c


def load_q3_arch():
    """附件 C7：模型架构元数据（上下文长度来源）"""
    return pd.read_csv(os.path.join(C_DIR, "model_architecture_metadata.csv"))


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

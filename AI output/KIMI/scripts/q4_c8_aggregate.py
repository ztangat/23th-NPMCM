# -*- coding: utf-8 -*-
"""问题四(前置): C8 detailed_results 逐任务聚合分析 (必用要求)"""
import os, json, glob
import numpy as np
import pandas as pd

ROOT = r"C:/Users/dkyyt/Desktop/F题"
DR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution", "detailed_results")
OUT = os.path.join(ROOT, "outputs", "q4_evolution")
os.makedirs(OUT, exist_ok=True)

SIX = {"leaderboard_bbh": "BBH", "leaderboard_mmlu_pro": "MMLU-PRO", "leaderboard_ifeval": "IFEval",
       "leaderboard_math_hard": "MATH Lvl 5", "leaderboard_gpqa": "GPQA", "leaderboard_musr": "MUSR"}

def pick_acc(v):
    """按任务选取主指标: 优先 acc, 其次 acc_norm, exact_match; IFEval用strict双级均值"""
    if not isinstance(v, dict):
        return np.nan
    for k in ["acc,none", "acc_norm,none", "exact_match,none"]:
        val = v.get(k)
        if isinstance(val, (int, float)):
            return val
    p, i = v.get("prompt_level_strict_acc,none"), v.get("inst_level_strict_acc,none")
    if isinstance(p, (int, float)) and isinstance(i, (int, float)):
        return (p + i) / 2
    if isinstance(p, (int, float)):
        return p
    return np.nan

rows, task_rows, bad = [], [], []
dirs = sorted(os.listdir(DR))
for di, d in enumerate(dirs):
    dd = os.path.join(DR, d)
    if not os.path.isdir(dd):
        continue
    files = sorted(glob.glob(os.path.join(dd, "*.json")))
    rec = None
    for f in reversed(files):  # 取最新可解析
        try:
            with open(f, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
            break
        except Exception:
            bad.append(f)
            continue
    if rec is None:
        bad.append(dd)
        continue
    res = rec.get("results", {})
    model = d.replace("_", "/", 1)
    r = {"model_dir": d}
    # 组级得分
    for k, v in res.items():
        acc = pick_acc(v)
        if k in SIX:
            r[SIX[k]] = acc * 100 if acc == acc else np.nan
        elif k.startswith("leaderboard_") and k != "leaderboard":
            task_rows.append({"model_dir": d, "task": k, "acc": acc})
    rows.append(r)
    if (di + 1) % 400 == 0:
        print(f"processed {di+1}/{len(dirs)}")

lb = pd.DataFrame(rows)
tasks = pd.DataFrame(task_rows)
lb.to_csv(os.path.join(OUT, "c8_model_sixdim.csv"), index=False)
tasks.to_csv(os.path.join(OUT, "c8_model_task_scores.csv"), index=False)
print("models:", len(lb), "task records:", len(tasks), "bad:", len(bad))
pd.DataFrame({"bad_entry": bad}).to_csv(os.path.join(OUT, "c8_unparseable.csv"), index=False)

# 六维完整性
comp = lb[["BBH","MMLU-PRO","IFEval","MATH Lvl 5","GPQA","MUSR"]].notna().all(axis=1).sum()
print("六维完整模型数:", comp)

# ---- 逐任务聚合分析1: 任务区分度(跨模型标准差, 覆盖率>=1000模型的任务) ----
tc = tasks.dropna()
cov = tc.groupby("task")["model_dir"].nunique()
tc = tc[tc.task.isin(cov[cov >= 500].index)]
disc = tc.groupby("task")["acc"].agg(["count", "mean", "std"]).sort_values("std", ascending=False)
disc["mean_pct"] = disc["mean"] * 100
disc["std_pct"] = disc["std"] * 100
disc.to_csv(os.path.join(OUT, "c8_task_discrimination.csv"))
print(disc.head(15).round(3).to_string())

# ---- 逐任务聚合分析2: 与C1汇总表的一致性验证(BBH子任务均值 vs C1 BBH列) ----
c1 = pd.read_csv(os.path.join(ROOT, "real_attachments", "C_efficiency_evolution", "leaderboard_cleaned.csv"))
bbh_sub = tc[tc.task.str.startswith("leaderboard_bbh_")].groupby("model_dir")["acc"].mean() * 100
c1m = c1.copy()
c1m["model_dir"] = c1m["Model"].str.replace("/", "_", n=1)
chk = pd.DataFrame({"bbh_from_subtasks": bbh_sub}).join(c1m.set_index("model_dir")[["BBH"]], how="inner")
chk["abs_diff"] = (chk.bbh_from_subtasks - chk.BBH).abs()
chk.to_csv(os.path.join(OUT, "c8_vs_c1_bbh_check.csv"))
print("BBH一致性: n={}, 平均|差|={:.3f}, 中位|差|={:.3f}, >1分占比={:.4f}".format(
    len(chk), chk.abs_diff.mean(), chk.abs_diff.median(), (chk.abs_diff > 1).mean()))

# ---- 逐任务聚合分析3: 任务相关结构(任务间Spearman相关, 抽样300模型) ----
piv = tc.pivot_table(index="model_dir", columns="task", values="acc")
piv = piv.dropna(axis=1, thresh=int(0.8 * len(piv)))
corr = piv.corr(method="spearman", min_periods=300)
corr.to_csv(os.path.join(OUT, "c8_task_correlation.csv"))
print("任务矩阵:", piv.shape)
print("DONE_C8")

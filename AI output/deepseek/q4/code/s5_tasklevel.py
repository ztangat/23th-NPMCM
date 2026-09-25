# -*- coding: utf-8 -*-
"""
s5_tasklevel.py —— 问题四 阶段5：C8 逐任务（per-task）聚合分析

赛题背景：
  问题四要求"利用 C6/C5 建立 Loss–Benchmark 桥接映射，并讨论映射误差的影响"，
  且能力度量建立在 6 项 benchmark 之上。C8（detailed_results）提供了**逐任务**
  （含 BBH 的 25 个子任务）的原始正确率与标准误，可支撑：
    (1) 复核/校准能力度量口径（Average 的构造）；
    (2) 识别"饱和任务"与"瓶颈任务"，解释为何能力天花板不可无限外推；
    (3) 逐任务的"前沿上移"结构，为 s3/s4 的效率故事提供微观证据。

【输出】
  · 逐模型 × 6 主 benchmark 的分数表（含 stderr）
  · BBH 25 子任务 × 模型的矩阵
  · 任务难度/区分度/饱和度的统计
  · 前沿模型在各任务上的表现画像（瓶颈识别）
"""
import os, sys, json, time, warnings, glob
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# 6 项主 benchmark 在 C8 中的键与取值字段
MAIN_TASKS = {
    "IFEval":      ("leaderboard_ifeval",      "prompt_level_strict_acc,none"),
    "BBH":         ("leaderboard_bbh",         "acc_norm,none"),
    "MATH Lvl 5":  ("leaderboard_math_hard",   "exact_match,none"),
    "GPQA":        ("leaderboard_gpqa_main",   "acc_norm,none"),
    "MUSR":        ("leaderboard_musr",        "acc_norm,none"),
    "MMLU-PRO":    ("leaderboard_mmlu_pro",    "acc,none"),
}
MAIN_STDERR = {
    "IFEval":      ("leaderboard_ifeval",      "prompt_level_strict_acc_stderr,none"),
    "BBH":         ("leaderboard_bbh",         "acc_norm_stderr,none"),
    "MATH Lvl 5":  ("leaderboard_math_hard",   "exact_match_stderr,none"),
    "GPQA":        ("leaderboard_gpqa_main",   "acc_norm_stderr,none"),
    "MUSR":        ("leaderboard_musr",        "acc_norm_stderr,none"),
    "MMLU-PRO":    ("leaderboard_mmlu_pro",    "acc_stderr,none"),
}


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    return np.nan


def parse_one(fp):
    try:
        j = json.load(open(fp, encoding="utf-8"))
    except Exception:
        return None
    res = j.get("results", {})
    row = {"model": j.get("model_name_sanitized") or j.get("model_name"),
           "file": os.path.basename(fp), "ts": j.get("date")}
    for name, (kk, field) in MAIN_TASKS.items():
        d = res.get(kk, {})
        row[name] = _num(d.get(field))
        sk, sf = MAIN_STDERR[name]
        row[name + "_se"] = _num(res.get(sk, {}).get(sf))
    # BBH 子任务
    bbh_sub = {}
    for k, v in res.items():
        if k.startswith("leaderboard_bbh_"):
            sub = k[len("leaderboard_bbh_"):]
            bbh_sub[sub] = _num(v.get("acc_norm,none"))
    row["_bbh_sub"] = bbh_sub
    return row


def main():
    start_log("s5_tasklevel")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段5：C8 逐任务聚合分析")
    print("=" * 78)

    dirs = c8_model_dirs()
    print(f"\n   C8 模型目录数 = {len(dirs)}")
    rows = []
    for i, p in enumerate(dirs):
        fs = [f for f in os.listdir(p) if f.endswith(".json")]
        if not fs:
            continue
        fp = os.path.join(p, fs[0])
        r = parse_one(fp)
        if r is not None:
            rows.append(r)
        if (i + 1) % 300 == 0:
            print(f"     已解析 {i+1}/{len(dirs)} ...")
    print(f"   成功解析模型数 = {len(rows)}")

    # ---------- 主表 ----------
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    main_cols = list(MAIN_TASKS.keys())
    df_out = df[["model", "date"] + main_cols + [c + "_se" for c in main_cols]].copy()
    # 用 C8 六项均值（0–1）作为复核口径
    df["C8_mean6"] = df[main_cols].mean(axis=1, skipna=True)
    df_out["C8_mean6"] = df["C8_mean6"].values
    df_out.to_csv(os.path.join(TABLES, "s5_C8逐模型主表.csv"), index=False, encoding="utf-8-sig")
    print(f"\n   [表] 逐模型主表已存（{len(df_out)} 行 × {len(df_out.columns)} 列）")

    # ---------- 逐任务统计 ----------
    print("\n1. 各 benchmark 的分布、饱和度与区分度")
    print("-" * 78)
    print(f"   {'任务':<12s}{'n':>6s}{'均值':>8s}{'中位':>8s}{'P95':>8s}{'最大':>8s}{'std':>8s}{'饱和度':>8s}")
    stat = {}
    for c in main_cols:
        s = df[c].dropna()
        if not len(s):
            continue
        # 饱和度：P95/max_ceiling（用 1.0 作理论上限）
        sat = float(s.quantile(0.95))
        disc = float(s.std())
        stat[c] = {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
                   "p95": float(s.quantile(.95)), "max": float(s.max()),
                   "std": float(s.std()), "p99": float(s.quantile(.99)),
                   "saturation_p95": sat}
        print(f"   {c:<12s}{len(s):>6d}{s.mean():>8.3f}{s.median():>8.3f}"
              f"{s.quantile(.95):>8.3f}{s.max():>8.3f}{s.std():>8.3f}{sat:>8.3f}")
    bottleneck = min(stat.items(), key=lambda kv: kv[1]["p95"])
    most_sat = max(stat.items(), key=lambda kv: kv[1]["p95"])
    most_disc = max(stat.items(), key=lambda kv: kv[1]["std"])
    print(f"""
   判读：{bottleneck[0]} 的 P95 上界最低（{bottleneck[1]['p95']:.3f}），是最**瓶颈**的任务 ——
        它直接压低 Average 的天花板；
        {most_sat[0]} 的 P95 已到 {most_sat[1]['p95']:.3f}，最**易饱和**；
        {most_disc[0]} 的组间标准差最大（{most_disc[1]['std']:.3f}），区分度最强。
        ⇒ 这正是能力天花板（s4 的 F_inf）受 benchmark 难度硬约束的微观原因。
""")

    # ---------- 逐任务 × 时间（前沿上移的微观证据） ----------
    print("2. 逐任务前沿（P95）随时间上移（效率进步的微观证据）")
    print("-" * 78)
    df["ym"] = df["dt"].dt.to_period("M")
    task_time = {}
    for c in main_cols:
        g = df.dropna(subset=[c, "ym"]).groupby("ym")[c]
        ser = g.quantile(0.95)
        task_time[c] = ser
    tt = pd.DataFrame(task_time)
    tt.index = tt.index.astype(str)
    print(tt.round(3).to_string())
    tt.to_csv(os.path.join(TABLES, "s5_逐任务前沿轨迹.csv"), encoding="utf-8-sig")

    print("\n   各任务前沿 P95 的首末变化（早3月 → 末3月）：")
    for c in main_cols:
        s = tt[c].dropna()
        if len(s) < 6:
            continue
        k = max(2, len(s)//3)
        early = s.iloc[:k].median(); late = s.iloc[-k:].median()
        print(f"     {c:<12s} {early:.3f} → {late:.3f}   Δ={late-early:+.3f} "
              f"({(late-early)/max(1e-9,early)*100:+.1f}%)")

    # ---------- BBH 子任务矩阵 ----------
    print("\n3. BBH 25 子任务分析")
    print("-" * 78)
    subs = set()
    for r in rows:
        subs |= set(r["_bbh_sub"].keys())
    subs = sorted(subs)
    print(f"   子任务数 = {len(subs)}")
    sub_mat = []
    for r in rows:
        rowd = {"model": r["model"]}
        for s in subs:
            rowd[s] = r["_bbh_sub"].get(s, np.nan)
        sub_mat.append(rowd)
    sm = pd.DataFrame(sub_mat)
    sm.to_csv(os.path.join(TABLES, "s5_BBH子任务矩阵.csv"), index=False, encoding="utf-8-sig")
    print(f"   [表] BBH 子任务矩阵已存（{len(sm)} × {len(sub_mat[0])}）")

    sm_cols = [c for c in sm.columns if c != "model"]
    print(f"\n   {'子任务':<40s}{'n':>6s}{'均值':>8s}{'P95':>8s}{'最大':>8s}")
    sub_stat = {}
    for c in sm_cols:
        s = sm[c].dropna()
        if len(s) < 50:
            continue
        sub_stat[c] = {"n": int(len(s)), "mean": float(s.mean()),
                       "p95": float(s.quantile(.95)), "max": float(s.max())}
    ordered = sorted(sub_stat.items(), key=lambda kv: kv[1]["mean"])
    for c, v in ordered:
        print(f"   {c:<40s}{v['n']:>6d}{v['mean']:>8.3f}{v['p95']:>8.3f}{v['max']:>8.3f}")
    show = pd.DataFrame([{"subtask": c, **v} for c, v in ordered])
    show.to_csv(os.path.join(TABLES, "s5_BBH子任务统计.csv"), index=False, encoding="utf-8-sig")

    print(f"""
   判读：BBH 子任务中，{ordered[0][0]}（最低均值 {ordered[0][1]['mean']:.3f}）
        与 {ordered[-1][0]}（最高均值 {ordered[-1][1]['mean']:.3f}）差异巨大，
        说明**同一 benchmark 内部任务难度极不均衡** —— 这是"Average 作为
        单一能力度量"的一个内生误差源（映射误差讨论，T4）。
""")

    # ---------- 前沿模型任务画像（瓶颈识别） ----------
    print("4. 前沿模型的任务画像（Top-20 均值 vs 全体均值，识别瓶颈）")
    print("-" * 78)
    top = df.nlargest(20, "C8_mean6")
    prof = pd.DataFrame({
        "task": main_cols,
        "front_top20": [top[c].mean() for c in main_cols],
        "all_mean": [df[c].mean() for c in main_cols],
        "gap": [top[c].mean() - df[c].mean() for c in main_cols],
        "P95_all": [df[c].quantile(.95) for c in main_cols],
    })
    prof["headroom"] = 1.0 - prof["front_top20"]
    prof = prof.sort_values("headroom", ascending=False)
    print(prof.round(3).to_string(index=False))
    prof.to_csv(os.path.join(TABLES, "s5_前沿任务画像.csv"), index=False, encoding="utf-8-sig")
    print(f"""
   判读：前沿模型在 {prof.iloc[0]['task']} 上仍有最大提升空间
        （headroom={prof.iloc[0]['headroom']:.3f}），这是**未来前沿上行的主要来源**；
        而 {prof.iloc[-1]['task']} 已接近饱和（headroom={prof.iloc[-1]['headroom']:.3f}），
        继续提升需靠新 benchmark 或更难的数据。
""")

    out = {
        "n_models": int(len(df)),
        "bench_stat": stat,
        "bbh_sub_stat": sub_stat,
        "frontier_profile": prof.to_dict(orient="records"),
        "task_frontier_traj": {c: {str(k): (float(v) if pd.notna(v) else None)
                                   for k, v in tt[c].items()} for c in main_cols},
    }
    save_json(out, os.path.join(DATA, "s5_tasklevel.json"))

    print(f"\n[OK] s5_tasklevel 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s5_tasklevel")


if __name__ == "__main__":
    main()

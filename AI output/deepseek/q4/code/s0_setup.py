# -*- coding: utf-8 -*-
"""
s0_setup.py —— 问题四 阶段0：问题界定、开源口径与能力度量设计

本阶段内容：
  1) 解析赛题要求，列出问题四的四个子任务与四类口径说明
  2) 探索 C1/C2/C3/C4/C5/C6/C8 结构，确定"开源前沿"筛选口径
  3) 确定综合能力度量（Average ⬆️）与其 6 项 benchmark 组成
  4) 分析时间轴、模型类型、参数规模、开源权重等分布
  5) 建立"前沿"定义（按时间分箱的 P95/P99 分位 + 开源子集）
  6) 输出口径说明表与基础统计

产出：
  data/s0_setup.json
  results/tables/s0_*.csv
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    start_log("s0_setup")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段0：问题界定、开源口径与能力度量设计")
    print("=" * 78)

    # ------------------------------------------------------------
    print("\n1. 问题四子任务与口径设计（依赛题）")
    print("-" * 78)
    print("""
  【子任务】
    (T1) 建立动力学/因果模型，分离"规模扩张"与"非规模技术进步"的贡献占比
    (T2) 算力增长放缓情景下，预测未来 12/24 个月开源模型能力前沿边界
    (T3) 给出不确定性分析
    (T4) 利用 C6/C5 建立 Loss–Benchmark 桥接映射，讨论映射误差影响

  【四类口径（赛题要求明确说明）】
    (K1) 综合能力度量：采用 Open LLM Leaderboard 的 Average ⬆️
         = 6 项 benchmark 的归一化平均：IFEval/BBH/MATH Lvl 5/GPQA/MUSR/MMLU-PRO
    (K2) 开源筛选口径：Main = 有明确 Hub License 且 Epoch_AI_Open_Weights=Yes
         Alt  = 有 Hub License（宽松，便于大样本）
    (K3) 模型类型：pretrained（🟢/🟩） vs chat_finetuned（💬/🔶/🤝）
    (K4) 时间轴：Main = Submission Date（提交日期，能力可用时点）
         Alt  = Epoch_AI_Publication_Date（发布日期）
""")

    # ------------------------------------------------------------
    print("\n2. C1 排行榜结构（4576 行）")
    print("-" * 78)
    c1 = load_C1()
    print(f"   行数={len(c1)}  列={list(c1.columns[:12])}")
    print(f"   提交日期范围: {c1['submission_dt'].min().date()} → {c1['submission_dt'].max().date()}")
    print(f"   跨度 {(c1['submission_dt'].max()-c1['submission_dt'].min()).days} 天")
    print(f"\n   模型类型分布：")
    for k, v in c1["type_grp"].value_counts().items():
        print(f"     {k:<16s}: {v:>5d} ({v/len(c1)*100:5.1f}%)")
    print(f"\n   许可证是否明确: {c1['has_license'].sum()}/{len(c1)} "
          f"({c1['has_license'].mean()*100:.1f}%)")
    print(f"   许可在白名单(可复现): {c1['open_license'].sum()} "
          f"({c1['open_license'].mean()*100:.1f}%)")
    print(f"\n   Average 分布: min={c1[AVG_COL].min():.3f} "
          f"median={c1[AVG_COL].median():.3f} max={c1[AVG_COL].max():.3f}")
    print(f"   参数量分布: min={c1['#Params (B)'].min():.3f} "
          f"median={c1['#Params (B)'].median():.3f} max={c1['#Params (B)'].max():.3f}")

    # 时间分箱统计
    c1["ym"] = c1["submission_dt"].dt.to_period("M")
    mm = c1.groupby("ym").agg(n=(AVG_COL, "size"), avg_max=(AVG_COL, "max"),
                              avg_p95=(AVG_COL, lambda s: s.quantile(0.95)),
                              avg_med=(AVG_COL, "median")).reset_index()
    mm["ym"] = mm["ym"].astype(str)
    print("\n   按月分箱（前沿 = max/P95）表（前 6 行）：")
    print(mm.head(6).to_string(index=False))
    mm.to_csv(os.path.join(TABLES, "s0_月度前沿统计.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    print("\n3. C2 增强：开源权重与发布日期字段覆盖率")
    print("-" * 78)
    c2 = load_C2()
    print(f"   行数={len(c2)}")
    print(f"   Epoch_AI_Open_Weights 非空: {c2['Epoch_AI_Open_Weights'].notna().sum()} "
          f"({c2['Epoch_AI_Open_Weights'].notna().mean()*100:.1f}%)")
    ow = c2["Epoch_AI_Open_Weights"].value_counts(dropna=False)
    print(f"   Open_Weights 取值: {dict(ow)}")
    print(f"   Epoch_AI_Publication_Date 非空: {c2['pub_dt'].notna().sum()} "
          f"({c2['pub_dt'].notna().mean()*100:.1f}%)")
    print(f"   Epoch_AI_Organization 非空: {c2['Epoch_AI_Organization'].notna().sum()}")

    # ------------------------------------------------------------
    print("\n4. C3 时序（含 2019-2022 历史模型）")
    print("-" * 78)
    c3 = load_C3()
    print(f"   行数={len(c3)}  年份范围 {c3['Year'].min()}→{c3['Year'].max()}")
    print(f"   来源: {dict(c3['Source'].value_counts())}")
    hist = c3[c3["Source"].str.contains("Historical")]
    print(f"   历史模型 {len(hist)} 个: {list(hist['Model'].head(10))}")

    # ------------------------------------------------------------
    print("\n5. C4 宏观元数据（含算力/数据量）")
    print("-" * 78)
    c4 = load_C4()
    print(f"   行数={len(c4)}  列数={len(c4.columns)}")
    keyc = ["Training compute (FLOP)", "Training dataset size (total)",
            "Parameters", "Publication date", "Open model weights?",
            "Model accessibility", "Organization"]
    for c in keyc:
        nn = c4[c].notna().sum()
        print(f"   {c:<34s} 非空 {nn:>5d} ({nn/len(c4)*100:5.1f}%)")
    # 语言领域子集（与本问相关）
    lang = c4[c4["Domain"].astype(str).str.contains("Language", na=False)]
    print(f"\n   Language 域子集: {len(lang)} 行")
    lang_open = lang[lang["Open model weights?"].astype(str).eq("Yes")]
    print(f"   其中开源权重: {len(lang_open)} 行")
    print(f"   含 Training compute 的开源语言模型: "
          f"{lang_open['Training compute (FLOP)'].notna().sum()}")

    # ------------------------------------------------------------
    print("\n6. C6 桥接数据（Loss ↔ Benchmark）")
    print("-" * 78)
    b = load_bridge(True)
    print(f"   行数={len(b)}")
    print(f"   可比性等级: {dict(b['Loss_Comparability'].value_counts())}")
    print(f"   N 范围 [{b['N_params_B'].min():.3f}, {b['N_params_B'].max():.2f}] B")
    print(f"   Val_Loss 范围 [{b['Val_Loss'].min():.4f}, {b['Val_Loss'].max():.4f}]")
    print(f"   LB_Average 范围 [{b['LB_Average'].min():.3f}, {b['LB_Average'].max():.3f}]")
    r = b[["Val_Loss", "LB_Average"]].corr().iloc[0, 1]
    print(f"   corr(Val_Loss, LB_Average) = {r:.4f}  （负相关，符合预期）")
    rl = np.corrcoef(b["Val_Loss"], np.log(b["LB_Average"]))[0, 1]
    print(f"   corr(Val_Loss, ln LB_Average) = {rl:.4f}")

    # ------------------------------------------------------------
    print("\n7. C8 逐任务评测（抽样确认结构）")
    print("-" * 78)
    dirs = c8_model_dirs()
    print(f"   模型目录数 = {len(dirs)}")
    import glob as _g
    sample = dirs[0]
    js = _g.glob(os.path.join(sample, "*.json"))
    if js:
        d = json.load(open(js[0], encoding="utf-8"))
        ks = list(d.get("results", {}).keys())
        print(f"   样本 {os.path.basename(sample)}: {len(ks)} 个任务键")
        fams = sorted(set(k.replace("leaderboard_", "").split("_")[0] for k in ks))
        print(f"   任务族: {fams}")
        n_bbh = len([k for k in ks if "bbh" in k])
        print(f"   其中 BBH 子任务数 = {n_bbh}（可用于逐任务聚合）")

    # ------------------------------------------------------------
    print("\n8. 口径汇总")
    print("-" * 78)
    print(f"""
   · 能力度量 y = Average ⬆️（0-100 尺度，6 项 benchmark 归一化均值）
   · 主分析样本：C1 中 type 明确、有许可、有提交日期的记录
   · 前沿定义：
       F_max(t) = 时间窗 t 内 Average 的最大值（最强模型）
       F_p99(t) = 99 分位（抗异常值）
       F_p95(t) = 95 分位（稳健前沿）
   · 开源前沿：额外要求 open_license=True
   · 规模变量：x = log10(#Params (B))；或 C4 的 log10(Training compute)
   · 时间变量：τ = (submission_dt - t0).days / 365.25
""")

    out = {
        "subtasks": ["T1 规模/非规模贡献分解", "T2 前沿边界预测",
                     "T3 不确定性分析", "T4 Loss-Benchmark桥接"],
        "metrics": {"capability": AVG_COL, "benchmarks": BENCH_COLS},
        "open_criteria": {"main": "Hub License 白名单 + Epoch Open_Weights=Yes",
                          "alt": "Hub License 白名单"},
        "type_criteria": {"pretrained": "🟢/🟩", "chat_finetuned": "💬/🔶/🤝"},
        "time_criteria": {"main": "Submission Date", "alt": "Epoch Publication Date"},
        "frontier_defs": {"F_max": "窗口内最大值", "F_p99": "99分位", "F_p95": "95分位"},
        "c1": {"n": len(c1),
               "date_min": str(c1["submission_dt"].min().date()),
               "date_max": str(c1["submission_dt"].max().date()),
               "type_dist": {k: int(v) for k, v in c1["type_grp"].value_counts().items()},
               "n_open_license": int(c1["open_license"].sum()),
               "avg_desc": {k: float(v) for k, v in c1[AVG_COL].describe().items()}},
        "c2": {"n": len(c2), "n_open_weights_nonnull": int(c2["Epoch_AI_Open_Weights"].notna().sum()),
               "n_pubdate_nonnull": int(c2["pub_dt"].notna().sum())},
        "c4": {"n": len(c4),
               "n_language": int(len(lang)),
               "n_lang_open": int(len(lang_open))},
        "bridge": {"n": len(b),
                   "corr_lin": float(r), "corr_log": float(rl),
                   "comparability": {k: int(v) for k, v in b["Loss_Comparability"].value_counts().items()}},
        "c8": {"n_dirs": len(dirs), "task_families": fams, "n_bbh_subtasks": int(n_bbh)},
    }
    save_json(out, os.path.join(DATA, "s0_setup.json"))

    print(f"\n[OK] s0_setup 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s0_setup")


if __name__ == "__main__":
    main()

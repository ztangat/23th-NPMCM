# -*- coding: utf-8 -*-
"""
s0_explore.py —— 问题二 阶段0：附件 B 全量探查与预处理

任务：
  1. 逐文件统计行数、字段、类型、缺失、范围
  2. 校验单位口径（N_params_B / D_tokens_B 以 10^9 计；C_FLOPs_1e21 以 10^21 计）
  3. 校验 C = 6ND 一致性
  4. 建立数据字典与可信度分层
  5. 提取可用于标度律拟合的 (N, D, L[, Q]) 有效点

产出：data/s0_explore.json, s0_字段统计.csv, s0_单位一致性.csv, s0_拟合点概览.csv
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import matplotlib
matplotlib.use("Agg")


def describe(df, name):
    rows = []
    for c in df.columns:
        s = df[c]
        d = {"数据集": name, "字段": c, "类型": str(s.dtype),
             "非空": int(s.notna().sum()), "缺失": int(s.isna().sum()),
             "缺失率": round(float(s.isna().mean()), 4)}
        if pd.api.types.is_numeric_dtype(s):
            d.update({"最小": float(np.nanmin(s)) if s.notna().any() else None,
                      "中位": float(np.nanmedian(s)) if s.notna().any() else None,
                      "最大": float(np.nanmax(s)) if s.notna().any() else None,
                      "均值": float(np.nanmean(s)) if s.notna().any() else None})
        else:
            uniq = s.dropna().unique()[:5]
            d.update({"取值示例": "|".join(map(str, uniq))})
        rows.append(d)
    return rows


def main():
    start_log("s0_explore")
    print("="*78)
    print("问题二 阶段0：附件 B 全量探查")
    print("="*78)

    summary = {}
    stat_rows = []

    for key in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12"]:
        fn, size, cred, use = B_FILES[key]
        df = loadB(key)
        print(f"\n--- {key}: {fn} ({size}, {cred}, {use}) ---")
        print(f"    shape = {df.shape}")
        print(f"    columns = {list(df.columns)}")
        stat_rows += describe(df, key)
        summary[key] = {"file": fn, "shape": list(df.shape), "credibility": cred,
                        "usage": use, "columns": list(df.columns)}
        # 数值列范围摘要
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                print(f"      {c:<18s} min={np.nanmin(df[c]):<12.4g} "
                      f"med={np.nanmedian(df[c]):<12.4g} max={np.nanmax(df[c]):<12.4g}")

    pd.DataFrame(stat_rows).to_csv(os.path.join(TABLES, "s0_字段统计.csv"),
                                   index=False, encoding="utf-8-sig")

    # ---------- 单位一致性校验 ----------
    print("\n" + "="*78)
    print("单位一致性校验：C_FLOPs_1e21 ≈ 6 * N_params_B * D_tokens_B * 1e-3")
    print("  （因 N,D 以 10^9 计，6ND = 6*N*1e9*D*1e9 = 6*N*D*1e18 FLOPs = 6*N*D*1e-3 * 1e21）")
    print("="*78)
    unit_rows = []
    for key in ["B1", "B2"]:
        df = loadB(key)
        if "C_FLOPs_1e21" in df.columns:
            c_pred = 6 * df["N_params_B"] * df["D_tokens_B"] * 1e-3
            ratio = df["C_FLOPs_1e21"] / c_pred.replace(0, np.nan)
            print(f"  {key}: 比值中位={np.nanmedian(ratio):.4f} "
                  f"均值={np.nanmean(ratio):.4f} [{np.nanmin(ratio):.3f},{np.nanmax(ratio):.3f}]")
            unit_rows.append({"数据集": key, "比值中位": float(np.nanmedian(ratio)),
                              "比值均值": float(np.nanmean(ratio)),
                              "比值min": float(np.nanmin(ratio)), "比值max": float(np.nanmax(ratio))})
    pd.DataFrame(unit_rows).to_csv(os.path.join(TABLES, "s0_单位一致性.csv"),
                                   index=False, encoding="utf-8-sig")

    # ---------- 拟合点概览 ----------
    print("\n" + "="*78)
    print("可用于标度律拟合的点（按数据源）")
    print("="*78)
    ov = []
    for key in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]:
        df = loadB(key)
        has_NDL = all(c in df.columns for c in COLS_NDL)
        has_Q = "Q_score" in df.columns
        n = len(df)
        nmax = float(df["N_params_B"].max()) if "N_params_B" in df.columns else np.nan
        nmin = float(df["N_params_B"].min()) if "N_params_B" in df.columns else np.nan
        dmax = float(df["D_tokens_B"].max()) if "D_tokens_B" in df.columns else np.nan
        cred = B_FILES[key][2]
        ov.append({"编号": key, "文件": B_FILES[key][0], "可信度": cred, "行数": n,
                   "含N/D/L": has_NDL, "含Q": has_Q,
                   "N范围_B": f"[{nmin:.2f},{nmax:.1f}]",
                   "D最大_B": f"{dmax:.1f}" if np.isfinite(dmax) else "-"})
    dfov = pd.DataFrame(ov)
    print(dfov.to_string(index=False))
    dfov.to_csv(os.path.join(TABLES, "s0_拟合点概览.csv"), index=False, encoding="utf-8-sig")

    # ---------- Q_score 分布（B6/B7/B8） ----------
    print("\n" + "="*78)
    print("质量 Q_score 分布（B6/B7/B8，唯一含质量的数据）")
    print("="*78)
    q_stat = []
    for key in ["B6", "B7", "B8"]:
        df = loadB(key)
        Q = df["Q_score"]
        print(f"  {key}: n={len(Q)}  Q∈[{Q.min():.4f},{Q.max():.4f}] "
              f"均值={Q.mean():.4f} 唯一值={Q.nunique()}")
        print(f"        分辨率 ΔQ={np.diff(np.sort(Q.unique()))[:5]} ...")
        q_stat.append({"编号": key, "n": len(Q), "Q_min": float(Q.min()),
                       "Q_max": float(Q.max()), "Q_mean": float(Q.mean()),
                       "Q_unique": int(Q.nunique())})
        if "data_type" in df.columns:
            print(f"        data_type 分布: {df['data_type'].value_counts().to_dict()}")
            q_stat[-1]["data_type"] = str(df["data_type"].value_counts().to_dict())
    save_json(q_stat, os.path.join(DATA, "s0_q_score_stat.json"))

    # ---------- 与问题一的衔接：17 域质量 q ----------
    print("\n" + "="*78)
    print("与问题一的衔接：读取 17 配方域质量 q（q1/data/s4_domain_Q.npz）")
    print("="*78)
    try:
        doms, Q, Qsd, lev = load_q_domains()
        print(f"  成功读取 {len(doms)} 个域")
        qmap = {str(d): float(Q[i]) for i, d in enumerate(doms)}
        print(f"  q 范围 [{min(qmap.values()):.4f}, {max(qmap.values()):.4f}]")
        print(f"  示例: {dict(list(qmap.items())[:5])}")
        summary["q1_domains"] = qmap
    except Exception as e:
        print(f"  [警告] 读取问题一 q 失败: {e}")

    save_json(summary, os.path.join(DATA, "s0_explore.json"))
    print("\n[OK] s0_explore 完成")
    end_log("s0_explore")


if __name__ == "__main__":
    main()

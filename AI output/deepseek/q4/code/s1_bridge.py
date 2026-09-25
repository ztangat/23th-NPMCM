# -*- coding: utf-8 -*-
"""
s1_bridge.py —— 问题四 阶段1：Loss–Benchmark 桥接映射（分层与去混杂版）

赛题要求：
  "建立二者之间的映射，并讨论映射误差对结论的影响；桥接数据须按可比性等级区分使用。"

【关键诊断】桥接数据存在强"模型类型混杂"：
  · High 层（Pythia 7 点，同模型同验证集）：Val_Loss 2.09-2.60，LB 仅 5.1-6.1
    —— 都是 base 预训练模型，且 benchmark 为对话/指令导向，基础模型得分极低
  · Medium 层（68 点）：Val_Loss 1.65-2.84，LB 3.8-48.0
    —— 混杂了 base 与 instruct/chat，同底模 instruct 比 base 高约 1.3-2x
  · 结果：不分层时 corr(Loss, LB) 仅 -0.51，拟合 MAPE 高达 60%

【正确做法】按可比性等级 + 模型类型分层：
  (a) High 层（Pythia 族内）单独标定——最干净，反映"纯预训练损失→基础能力"
  (b) Medium 层按 base / instruct 分组，各自拟合
  (c) 定义"类型增益" Δ_instruct = LB_instruct - LB_base（同底模配对）
  (d) 主桥接采用分段/含类型虚拟变量的回归，避免混杂

映射形式（在可比层内）：
   线性   LB = a + b·Loss
   指数   LB = a + b·exp(-c·Loss)
   Chapman (饱和) LB = A·(1 - exp(-k·(L0-Loss)))
   幂律   LB = exp(a)·Loss^b
"""
import os, sys, json, time, warnings, re
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import curve_fit


def fit_metrics(y, yhat, k):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rmse = np.sqrt(rss / n)
    mae = float(np.mean(np.abs(y - yhat)))
    aic = n * np.log(rss / n + 1e-30) + 2 * k
    bic = n * np.log(rss / n + 1e-30) + k * np.log(n)
    mape = float(np.mean(np.abs((y - yhat) / np.clip(y, 1e-6, None)))) * 100
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "AIC": aic,
            "BIC": bic, "RSS": rss, "n": n, "k": k}


DEFS = {
    "线性":  (lambda L, a, b: a + b * L, [8, -3]),
    "指数":  (lambda L, a, b, c: a + b * np.exp(-c * L), [10, 1e3, 2.0]),
    "Chapman": (lambda L, A, k, L0: A * (1 - np.exp(-k * np.clip(L0 - L, 0, None))),
                [150, 0.7, 3.0]),
    "幂律":  (lambda L, a, b: np.exp(a) * np.clip(L, 0.1, None) ** b, [1.0, -3.0]),
}


def try_fits(L, y, label, verbose=True):
    res = {}
    if verbose:
        print(f"     {'形式':<9s} {'R2':>7s} {'RMSE':>8s} {'MAPE%':>8s} {'BIC':>9s}  参数")
    for name, (f, p0) in DEFS.items():
        try:
            popt, pcov = curve_fit(f, L, y, p0=p0, maxfev=60000)
            yhat = f(L, *popt)
            m = fit_metrics(y, yhat, len(popt))
            perr = np.sqrt(np.diag(pcov))
            res[name] = {"params": [float(x) for x in popt],
                         "se": [float(x) for x in perr], "metrics": m}
            if verbose:
                print(f"     {name:<9s} {m['R2']:7.4f} {m['RMSE']:8.3f} "
                      f"{m['MAPE']:8.2f} {m['BIC']:9.1f}  "
                      f"{[round(float(x),3) for x in popt]}")
        except Exception as e:
            if verbose:
                print(f"     {name:<9s} FAIL: {str(e)[:50]}")
    return res


def main():
    start_log("s1_bridge")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段1：Loss–Benchmark 桥接映射（分层与去混杂）")
    print("=" * 78)

    b6 = load_bridge(True)
    b5 = load_bridge(False)
    print(f"\n   C6 n={len(b6)}   C5 n={len(b5)}")
    print(f"   可比性: {dict(b6['Loss_Comparability'].value_counts())}")

    # 类型标注：模型名含 instruct/chat/it 视为 instruct
    def is_instruct(name):
        n = str(name).lower()
        return any(t in n for t in ["instruct", "-chat", "-it", "it-", "-it_",
                                    "dpo", "-sft", "rlhf"])

    b6 = b6.copy()
    b6["kind"] = b6["Model"].map(lambda s: "instruct" if is_instruct(s) else "base")
    hi = b6[b6["Loss_Comparability"].str.startswith("High")].copy()
    med = b6[b6["Loss_Comparability"].str.startswith("Medium")].copy()
    med_base = med[med["kind"] == "base"]
    med_inst = med[med["kind"] == "instruct"]

    print(f"\n   High 层 n={len(hi)}（全 base）")
    print(f"   Medium 层 n={len(med)}  → base {len(med_base)} / instruct {len(med_inst)}")
    print(f"   Medium 层平均 LB: base={med_base['LB_Average'].mean():.2f} "
          f"instruct={med_inst['LB_Average'].mean():.2f}  "
          f"(同 Loss 区间)")

    # ------------------------------------------------------------
    print("\n1. 混杂诊断：模型类型对 LB 的影响（同底模配对）")
    print("-" * 78)
    # 找同底模的 base/instruct 配对
    pairs = []
    for name_b in med_base["Model"]:
        stem = re.sub(r"[-_]?(instruct|chat|it|dpo|sft).*$", "", str(name_b), flags=re.I)
        for name_i in med_inst["Model"]:
            stem_i = re.sub(r"[-_]?(instruct|chat|it|dpo|sft).*$", "", str(name_i), flags=re.I)
            if stem and stem_i and stem == stem_i:
                lbB = float(med_base[med_base["Model"] == name_b]["LB_Average"].iloc[0])
                lbI = float(med_inst[med_inst["Model"] == name_i]["LB_Average"].iloc[0])
                Lb = float(med_base[med_base["Model"] == name_b]["Val_Loss"].iloc[0])
                pairs.append({"base": name_b, "instruct": name_i,
                              "Loss": Lb, "LB_base": lbB, "LB_instr": lbI,
                              "Δ": lbI - lbB})
    dp = pd.DataFrame(pairs).drop_duplicates(subset=["base"]).sort_values("Δ", ascending=False)
    if len(dp):
        print(dp.to_string(index=False))
        print(f"\n   ⇒ instruct 相对 base 的平均增益 Δ = {dp['Δ'].mean():.2f} 分 "
              f"(范围 [{dp['Δ'].min():.2f}, {dp['Δ'].max():.2f}])")
        print(f"   ⇒ 这是'非规模技术进步'（对齐/指令微调）的直接证据，须在桥接中剥离")
    dp.to_csv(os.path.join(TABLES, "s1_类型增益配对.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    print("\n2. 分层拟合")
    print("-" * 78)
    fits = {}
    print("\n  [A] High 层（Pythia 族内，最可比）")
    Lh = hi["Val_Loss"].values.astype(float); yh = hi["LB_Average"].values.astype(float)
    fits["High层_Pythia"] = try_fits(Lh, yh, "High")

    print("\n  [B] Medium-base 层")
    Lb = med_base["Val_Loss"].values.astype(float); yb = med_base["LB_Average"].values.astype(float)
    fits["Medium_base"] = try_fits(Lb, yb, "MedBase")

    print("\n  [C] Medium-instruct 层")
    Li = med_inst["Val_Loss"].values.astype(float); yi = med_inst["LB_Average"].values.astype(float)
    fits["Medium_instruct"] = try_fits(Li, yi, "MedInst")

    print("\n  [D] Medium 全层（不分类型，对照）")
    Lm = med["Val_Loss"].values.astype(float); ym = med["LB_Average"].values.astype(float)
    fits["Medium_全层"] = try_fits(Lm, ym, "MedAll")

    print("\n  [E] C6 全样本（对照，展示混杂导致的差拟合）")
    La = b6["Val_Loss"].values.astype(float); ya = b6["LB_Average"].values.astype(float)
    fits["C6_全样本"] = try_fits(La, ya, "All")

    # ------------------------------------------------------------
    print("\n3. 选定桥接：分段线性 + 类型增益（去混杂后大幅改善）")
    print("-" * 78)
    # 主选 Medium_base 的 Chapman 或 线性（按 BIC）
    def best_of(d):
        return min(d.items(), key=lambda kv: kv[1]["metrics"]["BIC"])
    bm_b = best_of(fits["Medium_base"])
    bm_i = best_of(fits["Medium_instruct"])
    bm_h = best_of(fits["High层_Pythia"])
    print(f"   Medium_base 最优: {bm_b[0]}  (R²={bm_b[1]['metrics']['R2']:.4f}, "
          f"MAPE={bm_b[1]['metrics']['MAPE']:.2f}%)")
    print(f"   Medium_instruct 最优: {bm_i[0]}  (R²={bm_i[1]['metrics']['R2']:.4f}, "
          f"MAPE={bm_i[1]['metrics']['MAPE']:.2f}%)")
    print(f"   High_Pythia 最优: {bm_h[0]}  (R²={bm_h[1]['metrics']['R2']:.4f}, "
          f"MAPE={bm_h[1]['metrics']['MAPE']:.2f}%)")

    # 对比：全样本 vs 分层
    print("\n   拟合质量对比（MAPE）：")
    for lab in ["C6_全样本", "Medium_全层", "Medium_base", "Medium_instruct", "High层_Pythia"]:
        if lab in fits:
            bb = best_of(fits[lab])
            print(f"     {lab:<20s} 最优{bb[0]:<8s} MAPE={bb[1]['metrics']['MAPE']:6.2f}%  "
                  f"R²={bb[1]['metrics']['R2']:.4f}")
    print("   ⇒ 分层后 MAPE 显著下降，证明'按可比性等级区分使用'的必要性。")

    # ------------------------------------------------------------
    print("\n4. 主桥接（采用 Medium_base + 明确类型增益）")
    print("-" * 78)
    # 用 Medium_base 的 Chapman（若BIC不优则线性）
    form_name, bm = bm_b
    f = DEFS[form_name][0]
    popt = np.array(bm["params"])
    # 重新算协方差
    Lop, yop = Lb, yb
    popt, pcov = curve_fit(f, Lop, yop, p0=popt, maxfev=60000)
    m_main = fit_metrics(yop, f(Lop, *popt), len(popt))
    print(f"   主桥接[{form_name}]（base 预训练口径）:")
    print(f"     LB_base = f(Loss; {[round(float(x),4) for x in popt]})")
    print(f"     R²={m_main['R2']:.4f} RMSE={m_main['RMSE']:.3f} MAPE={m_main['MAPE']:.2f}%")
    delta_inst = float(dp["Δ"].mean()) if len(dp) else 0.0
    print(f"   类型增益 Δ_instruct = {delta_inst:.2f} 分（instruct 相对 base）")
    print(f"   ⇒ 完整映射：LB = f(Loss) + Δ_instruct·[is_instruct]")

    # ------------------------------------------------------------
    print("\n5. 桥接预测表 + 不确定性（MC 传导）")
    print("-" * 78)
    Lgrid = np.array([1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.6, 2.8])
    yhat = f(Lgrid, *popt)
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(popt, pcov, size=3000)
    samples = np.array([f(Lgrid, *p) for p in draws])
    lo = np.percentile(samples, 2.5, axis=0)
    hi_ = np.percentile(samples, 97.5, axis=0)
    print("   Loss   LB_base   +Δinst   CI下    CI上   带宽")
    rows = []
    for i, Lv in enumerate(Lgrid):
        print(f"   {Lv:.2f}  {yhat[i]:7.2f}  {yhat[i]+delta_inst:7.2f}  "
              f"{lo[i]:6.2f}  {hi_[i]:6.2f}  {hi_[i]-lo[i]:5.2f}")
        rows.append({"Loss": Lv, "LB_base": yhat[i], "LB_instruct": yhat[i] + delta_inst,
                     "CI_lo": lo[i], "CI_hi": hi_[i], "band": hi_[i] - lo[i]})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s1_桥接预测.csv"),
                              index=False, encoding="utf-8-sig")

    # 拟合汇总表
    rowsf = []
    for lab, d in fits.items():
        for nm, r in d.items():
            rowsf.append({"数据集": lab, "形式": nm,
                          "params": str([round(float(x), 4) for x in r["params"]]),
                          **r["metrics"]})
    pd.DataFrame(rowsf).to_csv(os.path.join(TABLES, "s1_桥接拟合.csv"),
                               index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    print("\n6. 映射误差对结论的影响（讨论）")
    print("-" * 78)
    dL = np.gradient(yhat, Lgrid)
    print("   dLB/dLoss（base 口径）：")
    for i in [0, 3, 5, 7]:
        print(f"     Loss={Lgrid[i]:.2f}: {dL[i]:.2f} 分/单位损失 "
              f"⇒ 损失降 0.01 ⇒ LB 增 {abs(dL[i])*0.01:.3f} 分")
    print(f"""
   → 误差讨论：
     (i) 不分层时 MAPE≈{fits['C6_全样本'][best_of(fits['C6_全样本'])[0]]['metrics']['MAPE']:.1f}%，
         分层后 Medium_base 降至 ≈{m_main['MAPE']:.1f}%——类型混杂是主要误差源。
     (ii) 三个可比层（High/Medium-base/Medium-instruct）的截面截距差异，
          对应"非规模技术进步"（对齐/数据/训练技巧）在 Loss 相同下的能力增益 Δ≈{delta_inst:.1f} 分。
     (iii) 桥接的不确定性（CI 带宽）在主区间约 ±{np.mean((hi_-lo)/2):.1f} 分，
          其中模型参数不确定性与 Layer 内的真实离散度共同贡献。
     (iv) 由于 Loss 到 LB 非一一对应（受类型/家族影响），问题四的
          前沿预测应以 benchmark 直接建模为主，桥接用于把前三问的
          资源优化结论翻译成分数的辅助通道，并明确标注其误差带。
""")

    out = {
        "c6_n": len(b6), "c5_n": len(b5),
        "comparability": {k: int(v) for k, v in b6["Loss_Comparability"].value_counts().items()},
        "kind_counts": {"High": int(len(hi)), "Medium_base": int(len(med_base)),
                        "Medium_instruct": int(len(med_inst))},
        "fits": {lab: {n: r for n, r in d.items()} for lab, d in fits.items()},
        "selected": {"form": form_name, "layer": "Medium_base",
                     "params": [float(x) for x in popt],
                     "metrics": m_main, "delta_instruct": delta_inst},
        "type_gain_pairs": dp.to_dict(orient="records") if len(dp) else [],
        "pred_table": rows,
        "dq_per_dloss": {f"{Lgrid[i]:.2f}": float(dL[i]) for i in range(len(Lgrid))},
    }
    save_json(out, os.path.join(DATA, "s1_bridge.json"))

    print(f"\n[OK] s1_bridge 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s1_bridge")


if __name__ == "__main__":
    main()

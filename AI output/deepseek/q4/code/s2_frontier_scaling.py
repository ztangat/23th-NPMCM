# -*- coding: utf-8 -*-
"""
s2_frontier_scaling.py —— 问题四 阶段2：能力前沿与规模标度律

本阶段内容：
  1) 在 C1/C2 上定义"能力前沿"（按时间窗的 max / P95 / P99）
  2) 建立能力关于规模 x=log10(N) 的标度律：
       线性    y = a + b·x
       log-log y = exp(a)·N^b     （幂律，检验是否饱和）
       饱和型  y = A - B·N^(-β)    （Chapman/幂律饱和，符合能力上限）
  3) 区分 pretrained vs chat_finetuned（同规模不同能力）
  4) 加入时间 τ 的二维模型 y = f(x, τ)
  5) 用时序 C3 看跨年演化
  6) 计算规模弹性 dLB/dlogN

【数据要点】
  · C1 中 pretrained 仅 7.3%，chat 92.4%——类型对能力影响巨大（见 s1 的 Δ=5.5）
  · 因此规模律须分类型拟合，或加入类型虚拟变量
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import curve_fit


def fit_metrics(y, yhat, k):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return {"R2": r2, "RMSE": float(np.sqrt(rss / n)),
            "MAE": float(np.mean(np.abs(y - yhat))),
            "MAPE": float(np.mean(np.abs((y - yhat) / np.clip(y, 1e-6, None)))) * 100,
            "AIC": n * np.log(rss / n + 1e-30) + 2 * k,
            "BIC": n * np.log(rss / n + 1e-30) + k * np.log(n), "n": n, "k": k}


DEFS = {
    "线性(x)":    (lambda x, a, b: a + b * x, [20, 8]),
    "幂律":       (lambda x, a, b: np.exp(a) * np.power(10.0, b * x), [1.0, 0.5]),
    "饱和幂":     (lambda x, A, B, beta: A - B * np.power(10.0, -beta * x), [100, 100, 0.3]),
    "对数":       (lambda x, a, b: a + b * np.log10(np.clip(np.power(10.0, x), 0.1, None)), [20, 8]),
}


def fit_group(x, y, label, verbose=True):
    res = {}
    if verbose:
        print(f"     {'形式':<10s} {'R2':>7s} {'RMSE':>8s} {'MAPE%':>8s} {'BIC':>9s}")
    for name, (f, p0) in DEFS.items():
        try:
            popt, pcov = curve_fit(f, x, y, p0=p0, maxfev=60000)
            yhat = f(x, *popt)
            m = fit_metrics(y, yhat, len(popt))
            perr = np.sqrt(np.diag(pcov))
            res[name] = {"params": [float(v) for v in popt],
                         "se": [float(v) for v in perr], "metrics": m}
            if verbose:
                print(f"     {name:<10s} {m['R2']:7.4f} {m['RMSE']:8.3f} "
                      f"{m['MAPE']:8.2f} {m['BIC']:9.1f}  "
                      f"{[round(float(v),3) for v in popt]}")
        except Exception as e:
            if verbose:
                print(f"     {name:<10s} FAIL {str(e)[:40]}")
    return res


def main():
    start_log("s2_frontier_scaling")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段2：能力前沿与规模标度律")
    print("=" * 78)

    c1 = load_C1()
    c2 = load_C2()

    # ------------------------------------------------------------
    print("\n1. 能力前沿的时间演化（C1）")
    print("-" * 78)
    d = c1.dropna(subset=["submission_dt", AVG_COL, "#Params (B)"]).copy()
    d = d[d["#Params (B)"] > 0]
    d["x"] = np.log10(d["#Params (B)"])
    d["t"] = (d["submission_dt"] - d["submission_dt"].min()).dt.days / 365.25
    print(f"   有效样本 n={len(d)}（有日期/能力/参数）")
    print(f"   未验证：{len(c1)-len(d)} 行（缺日期或参数或能力）")

    # 按半月/月分箱
    d["ym"] = d["submission_dt"].dt.to_period("M")
    fr = []
    for ym, g in d.groupby("ym"):
        fr.append({"ym": str(ym), "n": len(g),
                   "F_max": g[AVG_COL].max(),
                   "F_p99": g[AVG_COL].quantile(0.99),
                   "F_p95": g[AVG_COL].quantile(0.95),
                   "F_med": g[AVG_COL].median(),
                   "N_med": g["#Params (B)"].median(),
                   "t": g["t"].mean()})
    fr = pd.DataFrame(fr).sort_values("ym").reset_index(drop=True)
    print(f"\n   月度前沿（前 3 / 后 3）：")
    print(fr.head(3)[["ym", "n", "F_max", "F_p95", "F_med"]].to_string(index=False))
    print("     ...")
    print(fr.tail(3)[["ym", "n", "F_max", "F_p95", "F_med"]].to_string(index=False))
    fr.to_csv(os.path.join(TABLES, "s2_月度前沿.csv"), index=False, encoding="utf-8-sig")

    # 前沿增长
    print(f"\n   前沿增长：F_max 从 {fr['F_max'].iloc[0]:.2f} → {fr['F_max'].iloc[-1]:.2f} "
          f"(+{fr['F_max'].iloc[-1]-fr['F_max'].iloc[0]:.2f})")
    print(f"             F_p95 从 {fr['F_p95'].iloc[0]:.2f} → {fr['F_p95'].iloc[-1]:.2f} "
          f"(+{fr['F_p95'].iloc[-1]-fr['F_p95'].iloc[0]:.2f})")

    # ------------------------------------------------------------
    print("\n2. 规模标度律（全样本，含类型混杂）")
    print("-" * 78)
    print("   [全部]")
    fits_all = fit_group(d["x"].values, d[AVG_COL].values, "all")

    print("\n   [仅 pretrained]")
    dp = d[d["type_grp"] == "pretrained"]
    print(f"     n={len(dp)}  x范围[{dp['x'].min():.2f},{dp['x'].max():.2f}]")
    fits_pre = fit_group(dp["x"].values, dp[AVG_COL].values, "pre")

    print("\n   [仅 chat_finetuned]")
    dc = d[d["type_grp"] == "chat_finetuned"]
    print(f"     n={len(dc)}  x范围[{dc['x'].min():.2f},{dc['x'].max():.2f}]")
    fits_chat = fit_group(dc["x"].values, dc[AVG_COL].values, "chat")

    # ------------------------------------------------------------
    print("\n3. 前沿规模律：只用最强模型（分位数回归思路）")
    print("-" * 78)
    # 在每个参数规模 bin 内取 P95，再看 x 与 frontier 关系
    d["xbin"] = pd.cut(d["x"], bins=np.arange(0, 2.3, 0.2))
    fb = d.groupby("xbin", observed=True).agg(
        x_mid=("x", "mean"), n=(AVG_COL, "size"),
        y_p95=(AVG_COL, lambda s: s.quantile(0.95)),
        y_max=(AVG_COL, "max"), y_med=(AVG_COL, "median")).reset_index()
    print("   参数规模 bin 内的前沿：")
    print(fb[["x_mid", "n", "y_med", "y_p95", "y_max"]].round(2).to_string(index=False))
    fb.to_csv(os.path.join(TABLES, "s2_规模bin前沿.csv"), index=False, encoding="utf-8-sig")

    print("\n   [规模-bin P95 拟合]")
    fbf = fb.dropna()
    fits_frontier = fit_group(fbf["x_mid"].values, fbf["y_p95"].values, "frontier")

    # ------------------------------------------------------------
    print("\n4. 二维模型：能力 ~ 规模 + 时间（分离静/动态）")
    print("-" * 78)
    # y = a + b·x + c·τ（线性双因子）
    X = np.column_stack([np.ones(len(d)), d["x"].values, d["t"].values])
    y = d[AVG_COL].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    m2 = fit_metrics(y, yhat, 3)
    print(f"   y = {coef[0]:.3f} + {coef[1]:.3f}·log10(N) + {coef[2]:.3f}·τ")
    print(f"      R²={m2['R2']:.4f} RMSE={m2['RMSE']:.3f}")
    print(f"   ⇒ 规模弹性 b={coef[1]:.3f} 分/十倍参数（静态）")
    print(f"   ⇒ 时间趋势 c={coef[2]:.3f} 分/年（同规模下的时间增益=非规模进步）")

    # 分类型二维
    for lab, gg in [("pretrained", dp), ("chat", dc)]:
        Xg = np.column_stack([np.ones(len(gg)), gg["x"].values, gg["t"].values])
        yg = gg[AVG_COL].values
        cg, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
        mg = fit_metrics(yg, Xg @ cg, 3)
        print(f"   [{lab}] y = {cg[0]:.3f} + {cg[1]:.3f}·log10(N) + {cg[2]:.3f}·τ  "
              f"(R²={mg['R2']:.4f})")

    # ------------------------------------------------------------
    print("\n5. 跨年时序（C3，含 2019 历史）")
    print("-" * 78)
    c3 = load_C3()
    c3f = c3.dropna(subset=["Average", "Params_B"]).copy()
    c3f = c3f[c3f["Params_B"] > 0]
    yr = c3f.groupby("Year").agg(n=("Average", "size"), y_max=("Average", "max"),
                                 y_p95=("Average", lambda s: s.quantile(0.95)),
                                 N_max=("Params_B", "max")).reset_index()
    print(yr.round(2).to_string(index=False))
    yr.to_csv(os.path.join(TABLES, "s2_跨年前沿.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    print("\n6. 规模弹性汇总")
    print("-" * 78)
    print("   各群体/形式的规模弹性（每 +1 个数量级参数带来分）：")
    for lab, fis in [("全样本", fits_all), ("pretrained", fits_pre),
                     ("chat", fits_chat), ("前沿P95", fits_frontier)]:
        b = fis["线性(x)"]["metrics"]
        ps = fis["线性(x)"]["params"]
        print(f"     {lab:<12s}: b={ps[1]:+.3f} 分/decade  (线性 R²={b['R2']:.3f})")
    print("""
   → 关键观察：
     · pretrained 与 chat 的规模律斜率不同：chat 因对齐增益整体上移
     · 二维模型的"时间项"系数 c 即"同规模下的年化能力增益"，
       它与规模项 b 共同构成对能力增长的分解基础（见 s3）。
     · 幂律 vs 饱和型的 BIC 对比可判断是否存在"能力饱和"。
""")

    out = {
        "n_valid": len(d),
        "frontier_monthly": fr.to_dict(orient="records"),
        "sizebin_frontier": fb.round(4).to_dict(orient="records"),
        "fits": {"all": fits_all, "pretrained": fits_pre, "chat": fits_chat,
                 "frontier_p95": fits_frontier},
        "two_factor_all": {"coef": [float(c) for c in coef], "metrics": m2,
                           "interp": f"y={coef[0]:.3f}+{coef[1]:.3f}*log10N+{coef[2]:.3f}*tau"},
        "yearly": yr.to_dict(orient="records"),
    }
    save_json(out, os.path.join(DATA, "s2_frontier.json"))

    print(f"\n[OK] s2_frontier_scaling 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s2_frontier_scaling")


if __name__ == "__main__":
    main()

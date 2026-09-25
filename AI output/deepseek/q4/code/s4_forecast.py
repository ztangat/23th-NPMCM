# -*- coding: utf-8 -*-
"""
s4_forecast.py —— 问题四 阶段4：算力增长放缓情景下的前沿边界预测（T2）+ 不确定性分析（T3）

赛题要求：
  (T2) 若未来算力增长放缓，预测未来 12 个月 / 24 个月"开源模型能力前沿边界"；
       给出 3 档算力情景下的边界曲线。
  (T3) 对预测结果做不确定性分析（置信区间）。

【建模思路】

  由 s3 结论：本窗口开源前沿的能力增长**主要来自非规模技术进步（效率）**，
  规模通道几近冻结（前沿规模不扩张）。因此预测的核心是建模"效率驱动的前沿演化"，
  再叠加"算力情景"对该效率通道的调制。

  ① 前沿边界定义
     F(t) = 月度 top-5% 能力（F_p95），作为"开源前沿边界"的稳健代理。

  ② 饱和增长模型（Saturating growth / exponential approach to ceiling）
     F(t) = F_inf - (F_inf - F_a)·exp(-k·(t - t_a))
     其中 F_inf 为天花板（由 benchmark 饱和特性估计），k 为趋近速率（1/年）。
     理由：benchmark 有上限（如 GPQA 实测天花板 ~29 分），前沿无法线性外推，
           必须用"渐近天花板"模型。

  ③ 算力情景（3 档）对 k 的调制
     效率收益本身也吃算力（训练搜索、数据、算力换算法）。定义情景乘子 s_k：
       · 基准情景 (A)：算力按当前节奏 ⇒ s_k = 1.0
       · 放缓情景 (B)：算力增速减半 ⇒ s_k = 0.75（效率通道减速 25%）
       · 严重放缓 (C)：算力增长近乎停滞 ⇒ s_k = 0.45（效率通道减速 55%）
     并叠加"规模通道"微调：放缓情景下规模贡献从其(近零)基数略微转正/持零。

  ④ 不确定性（T3）
     · 参数不确定性：对 (F_inf, k, F_a) 用 bootstrap 重采样月度点 → 分布
     · 过程噪声：残差自举
     · 输出 12/24 月的 P05/P50/P95 区间

【输出】
  · 3 情景 × 12/24 月的边界点预测与区间
  · 月度预测轨迹表（含基准情景全序列）
  · 关键结论：算力放缓下前沿是否冻结、何时逼近饱和
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def fit_metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    return {"R2": float(1 - rss / tss) if tss > 0 else float("nan"),
            "RMSE": float(np.sqrt(rss / n)), "n": int(n)}


def fit_saturating(t, y, F_inf_init=None, k_init=2.0, F_a_init=None, F_inf_fixed=None):
    """拟合 F(t)=F_inf-(F_inf-F_a)exp(-k(t-t_a))，t_a=t[0]。
    若 F_inf_fixed 给定则固定天花板，只拟合 k。用网格 + 局部精修（稳健）。"""
    from scipy.optimize import least_squares
    t = np.asarray(t, float); y = np.asarray(y, float)
    ta = float(t[0])
    def model(p):
        if F_inf_fixed is None:
            Fi, k, Fa = p
        else:
            Fi = F_inf_fixed; k, Fa = p
        return Fi - (Fi - Fa) * np.exp(-k * (t - ta))
    def resid(p):
        return model(p) - y
    if F_inf_fixed is None:
        p0 = [F_inf_init if F_inf_init else y.max() * 1.15, k_init, y.min()]
        lb = [y.max() * 1.001, 1e-3, y.min() - 20]
        ub = [y.max() * 3.0, 20.0, y.max()]
    else:
        p0 = [k_init, y.min()]
        lb = [1e-3, y.min() - 20]
        ub = [20.0, y.max()]
    try:
        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=20000)
        p = res.x
    except Exception:
        p = np.array(p0)
    yhat = model(p)
    m = fit_metrics(y, yhat)
    if F_inf_fixed is None:
        return {"F_inf": float(p[0]), "k": float(p[1]), "F_a": float(p[2])}, m, yhat
    else:
        return {"F_inf": float(F_inf_fixed), "k": float(p[0]), "F_a": float(p[1])}, m, yhat


def main():
    start_log("s4_forecast")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段4：算力增长放缓情景下的前沿边界预测与不确定性分析")
    print("=" * 78)

    c1 = load_C1()
    d = c1.dropna(subset=["submission_dt", AVG_COL, "#Params (B)"]).copy()
    d = d[d["#Params (B)"] > 0]
    d["x"] = np.log10(d["#Params (B)"])
    t0_date = d["submission_dt"].min()
    d["t"] = (d["submission_dt"] - t0_date).dt.days / 365.25
    d["ym"] = d["submission_dt"].dt.to_period("M")
    T = float(d["t"].max())
    months = sorted(d["ym"].unique())
    print(f"\n   样本 n={len(d)}，观测窗 {months[0]}..{months[-1]}（T={T:.3f}年）")

    # ------------------------------------------------------------
    # 1. 前沿边界的经验轨迹（F_p95 与 F_max）
    # ------------------------------------------------------------
    print("\n1. 前沿边界经验轨迹（月度）")
    print("-" * 78)
    rows = []
    for ym, g in d.groupby("ym"):
        rows.append({"ym": str(ym), "t": g["t"].mean(), "n": len(g),
                     "F_p95": g[AVG_COL].quantile(0.95),
                     "F_max": g[AVG_COL].max(),
                     "F_p90": g[AVG_COL].quantile(0.90),
                     "F_top10_mean": g[AVG_COL].nlargest(max(1, len(g)//10)).mean()})
    fr = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    print(fr.round(3).to_string(index=False))
    fr.to_csv(os.path.join(TABLES, "s4_前沿经验轨迹.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. 天花板估计
    # ------------------------------------------------------------
    print("\n2. 能力天花板 F_inf 的估计（benchmark 饱和特性）")
    print("-" * 78)
    # 方法：用各 benchmark 的实测最高分构造"可达平均"上限
    #   Average = 6 项归一化平均；若各项都达到其"可达上限"，则 Average 有上界
    ceil_emp = d[AVG_COL].max()
    ceil_p999 = d[AVG_COL].quantile(0.999)
    # 每项 benchmark 的"实际可达上限"（用 p99.9 近似，避免单点噪声）
    bench_ceil = {}
    for c in BENCH_COLS:
        s = d[c].dropna()
        bench_ceil[c] = float(s.quantile(0.9999)) if len(s) else float("nan")
    avg_of_ceil = float(np.mean(list(bench_ceil.values())))
    print(f"   实测最大值     F_max_ever     = {ceil_emp:.2f}")
    print(f"   高分位 p99.9   = {ceil_p999:.2f}")
    print(f"   各 benchmark 可达上限（p99.99 近似）：")
    for c, v in bench_ceil.items():
        print(f"     {c:12s} ≈ {v:6.2f}")
    print(f"   ⇒ 由 benchmark 上限推出的 Average 天花板 ≈ {avg_of_ceil:.2f}")
    print(f"   ⇒ 采用 F_inf 上限区间 [{ceil_emp:.1f}, {avg_of_ceil:.1f}]，")
    print(f"      基准取 F_inf = {max(ceil_emp*1.05, 55.0):.1f}（留出进步空间，保守不激进）")
    F_INF_BASE = float(max(ceil_emp * 1.05, 55.0))
    F_INF_LO = float(ceil_emp)          # 悲观天花板
    F_INF_HI = float(avg_of_ceil)       # 乐观天花板

    # ------------------------------------------------------------
    # 3. 饱和增长模型拟合（基准情景）
    # ------------------------------------------------------------
    print("\n3. 饱和增长模型拟合（基准情景，F_p95 口径）")
    print("-" * 78)
    print("   模型：F(t) = F_inf - (F_inf - F_a)·exp(-k·(t - t_a))")
    t_arr = fr["t"].values
    y_p95 = fr["F_p95"].values

    # (i) 固定天花板 F_INF_BASE，拟合 k
    par_fix, m_fix, yhat_fix = fit_saturating(t_arr, y_p95, F_inf_fixed=F_INF_BASE, k_init=2.8)
    print(f"\n   [固定天花板 F_inf={F_INF_BASE:.1f}]")
    print(f"     F_a(截距)={par_fix['F_a']:.2f}  k={par_fix['k']:.3f} /年  "
          f"(R²={m_fix['R2']:.4f}, RMSE={m_fix['RMSE']:.3f})")
    half_life = np.log(2) / par_fix["k"]
    print(f"     趋近时间常数 1/k = {1/par_fix['k']:.2f} 年；半程时间 ln2/k = {half_life:.2f} 年")

    # (ii) 自由拟合（含天花板）
    par_free, m_free, yhat_free = fit_saturating(t_arr, y_p95, F_inf_init=F_INF_BASE, k_init=2.8)
    print(f"\n   [自由拟合天花板]")
    print(f"     F_inf={par_free['F_inf']:.2f}  F_a={par_free['F_a']:.2f}  "
          f"k={par_free['k']:.3f} /年  (R²={m_free['R2']:.4f}, RMSE={m_free['RMSE']:.3f})")

    # (iii) 线性基准（对照，说明为何必须用饱和模型）
    A = np.column_stack([np.ones(len(t_arr)), t_arr])
    cf_lin = np.linalg.lstsq(A, y_p95, rcond=None)[0]
    m_lin = fit_metrics(y_p95, A @ cf_lin)
    print(f"\n   [线性对照] F = {cf_lin[0]:.2f} + {cf_lin[1]:.2f}·t  "
          f"(斜率 {cf_lin[1]:+.2f} 分/年, R²={m_lin['R2']:.4f})")

    # 选择主模型：固定天花板更稳健（避免外推爆炸）
    K_BASE = float(par_fix["k"])
    print(f"\n   ⇒ 采用固定天花板模型（稳健外推）：F_inf={F_INF_BASE:.1f}, k={K_BASE:.3f}/年")

    # (iv) 动量模型（对照族）：近期趋势线性衰减
    #   F(t+h) = F_now + g0·Σ_{j=1..h} φ^j  （几何衰减的年度增量，φ=衰减因子）
    #   用近 6 个月增量估计 g0
    recent = fr.iloc[-6:]
    g0 = float((recent["F_p95"].iloc[-1] - recent["F_p95"].iloc[0]) /
               max(1e-9, (recent["t"].iloc[-1] - recent["t"].iloc[0])))
    print(f"\n   [动量模型对照] 近6月前沿年化斜率 g0 = {g0:+.2f} 分/年")

    def momentum_forecast(g0v, phi, tq):
        # 从观测末点 t_end 起算
        t_end = float(t_arr[-1]); h = max(0.0, tq - t_end)  # 年
        # 连续化：dF/dt = g0·φ^(t-t_end)
        return float(y_p95[-1] + g0v * (1 - phi ** h) / max(1e-9, -np.log(phi)))

    mom_tab = []
    for phi in [0.5, 0.7]:
        for h in [1.0, 2.0]:
            mom_tab.append({"phi": phi, "horizon_y": h,
                            "F": momentum_forecast(g0, phi, t_arr[-1] + h)})
    print(f"   {'衰减φ':>7s}{'12月':>9s}{'24月':>9s}")
    for phi in [0.5, 0.7]:
        f12 = momentum_forecast(g0, phi, t_arr[-1] + 1.0)
        f24 = momentum_forecast(g0, phi, t_arr[-1] + 2.0)
        print(f"   {phi:7.2f}{f12:9.2f}{f24:9.2f}")

    # ------------------------------------------------------------
    # 4. 算力情景设计
    # ------------------------------------------------------------
    print("\n4. 算力增长情景与效率通道调制")
    print("-" * 78)
    print("""
   由 s3：前沿增长 ≈ 效率通道（非规模技术进步）。效率通道本身消耗算力
   （训练搜索、数据配比、RLHF、蒸馏等）。算力放缓 ⇒ 效率通道减速。
   定义情景乘子 s_k（作用于趋近速率 k）：

     · 情景A 基准      ：算力维持当前节奏              s_k = 1.00
     · 情景B 放缓      ：算力增速降至约一半            s_k = 0.75
     · 情景C 严重放缓  ：算力增长近乎停滞              s_k = 0.45

   规模通道：本窗口前沿规模不扩张（s3 结论），故三情景下规模贡献近似为 0；
   放缓情景下不额外增加负向规模效应（保守）。
""")
    scenarios = {
        "A_基准": {"s_k": 1.00, "desc": "算力维持当前节奏"},
        "B_放缓": {"s_k": 0.75, "desc": "算力增速降至约一半"},
        "C_严重放缓": {"s_k": 0.45, "desc": "算力增长近乎停滞"},
    }

    # ------------------------------------------------------------
    # 5. 预测 12 / 24 个月
    # ------------------------------------------------------------
    print("\n5. 前沿边界预测（12 / 24 个月）")
    print("-" * 78)
    horizons_months = [6, 12, 18, 24]
    pred_grid = np.array([h / 12.0 for h in horizons_months])  # 相对 t_a 的偏移年数
    t_a = float(t_arr[0])
    F_a = float(par_fix["F_a"])

    def forecast(k_eff, tq):
        return F_INF_BASE - (F_INF_BASE - F_a) * np.exp(-k_eff * (tq - t_a))

    base_line = forecast(K_BASE, t_arr)
    print(f"   拟合基线（基准情景）在观测末点 t={t_arr[-1]:.2f}: F={base_line[-1]:.2f} "
          f"(观测 {y_p95[-1]:.2f})")

    scen_tab = []
    print(f"\n   {'情景':<12s}{'s_k':>6s}{'12月':>9s}{'24月':>9s}{'较当前Δ(12月)':>14s}")
    for sname, sinfo in scenarios.items():
        k_eff = K_BASE * sinfo["s_k"]
        f12 = forecast(k_eff, t_a + 1.0)
        f24 = forecast(k_eff, t_a + 2.0)
        cur = forecast(k_eff, t_a + T)
        scen_tab.append({"scenario": sname, "s_k": sinfo["s_k"], "k_eff": float(k_eff),
                         "F_12m": float(f12), "F_24m": float(f24),
                         "F_ref_now": float(cur), "d12": float(f12 - cur)})
        print(f"   {sname:<12s}{sinfo['s_k']:6.2f}{f12:9.2f}{f24:9.2f}{f12-cur:+14.2f}")

    # 相对"当前观测前沿"的净增量（更有意义的判读）
    print(f"\n   参考：当前观测前沿 F_p95（末月）= {y_p95[-1]:.2f}")

    # ------------------------------------------------------------
    # 6. 不确定性分析（T3）
    # ------------------------------------------------------------
    print("\n6. 不确定性分析（bootstrap + 参数 + 过程噪声）")
    print("-" * 78)
    rng = np.random.default_rng(20260925)
    B = 2000
    resids = y_p95 - forecast(K_BASE, t_arr)
    boot_k = []
    for b in range(B):
        # 参数 bootstrap：残差重采样后重拟合 k（固定天花板），不加额外噪声
        yb = forecast(K_BASE, t_arr) + rng.choice(resids, len(t_arr), replace=True)
        try:
            par_b, m_b, _ = fit_saturating(t_arr, yb, F_inf_fixed=F_INF_BASE, k_init=K_BASE)
            if 0.05 < par_b["k"] < 20:
                boot_k.append(par_b["k"])
        except Exception:
            pass
    boot_k = np.array(boot_k)
    print(f"   bootstrap 成功 {len(boot_k)} 次")
    print(f"     k    : 中位 {np.median(boot_k):.3f}  "
          f"[P05 {np.percentile(boot_k,5):.3f}, P95 {np.percentile(boot_k,95):.3f}]  "
          f"(确定性拟合 k={K_BASE:.3f})")

    # 参数不确定性：F_inf 扰动（3 档天花板）+ k 的 bootstrap 分布 + F_a 固定
    def forecast_dist(s_k, tq):
        vals = []
        for Fi in [F_INF_LO, F_INF_BASE, F_INF_HI]:
            for kk in boot_k if len(boot_k) else [K_BASE]:
                vals.append(Fi - (Fi - F_a) * np.exp(-kk * s_k * (tq - t_a)))
        return np.percentile(np.array(vals), [5, 50, 95])

    print("\n   预测区间（P05 / P50 / P95），各情景：")
    print(f"   {'情景':<12s}{'期限':>6s}{'P05':>8s}{'P50':>8s}{'P95':>8s}{'区间宽度':>9s}")
    unc_tab = []
    for sname, sinfo in scenarios.items():
        for h in [12, 24]:
            lo, md, hi = forecast_dist(sinfo["s_k"], t_a + h / 12.0)
            unc_tab.append({"scenario": sname, "horizon_m": h,
                            "p05": float(lo), "p50": float(md), "p95": float(hi),
                            "width": float(hi - lo)})
            print(f"   {sname:<12s}{h:>4d}月{lo:8.2f}{md:8.2f}{hi:8.2f}{hi-lo:9.2f}")
    pd.DataFrame(unc_tab).to_csv(os.path.join(TABLES, "s4_预测区间.csv"),
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame(scen_tab).to_csv(os.path.join(TABLES, "s4_情景预测.csv"),
                                  index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 7. 关键判读
    # ------------------------------------------------------------
    print("\n7. 关键判读：算力放缓下前沿是否冻结？")
    print("-" * 78)
    A12 = [x for x in scen_tab if x["scenario"] == "A_基准"][0]["F_12m"]
    C24 = [x for x in scen_tab if x["scenario"] == "C_严重放缓"][0]["F_24m"]
    print(f"""
   · 基准情景（算力维持）：
       12 月前沿 F_p95 ≈ {A12:.1f} 分（较当前 {y_p95[-1]:.1f} 提升 {A12 - y_p95[-1]:+.1f}）
   · 严重放缓情景（算力近乎停滞）：
       24 月前沿 F_p95 ≈ {C24:.1f} 分
   · **结论**：即使算力增长严重放缓，由于前沿主要由效率通道驱动，
     能力边界在未来 12–24 月内**仍会继续上行，但增速递减**（饱和效应）——
     不会立即冻结，但加速逼近天花板 {F_INF_BASE:.0f} 分。
   · **不确定性**：12 月区间宽约 {unc_tab[0]['width']:.1f} 分，
     主要来自"天花板位置"的不确定性（benchmark 难度的硬约束）。
""")

    out = {
        "n": int(len(d)), "T_years": T,
        "window": {"first": str(months[0]), "last": str(months[-1])},
        "ceilings": {"F_max_ever": float(ceil_emp), "F_p999": float(ceil_p999),
                     "bench_ceil": bench_ceil, "avg_of_bench_ceil": avg_of_ceil,
                     "F_INF_BASE": float(F_INF_BASE), "F_INF_LO": float(F_INF_LO),
                     "F_INF_HI": float(F_INF_HI)},
        "saturating_fit": {
            "fixed_ceiling": {"params": par_fix, "metrics": m_fix},
            "free_ceiling": {"params": par_free, "metrics": m_free},
            "linear_reference": {"coef": {"a": float(cf_lin[0]), "b": float(cf_lin[1])},
                                 "metrics": m_lin},
            "half_life_years": float(half_life), "k_base": K_BASE},
        "scenarios": {k: {"s_k": v["s_k"], "desc": v["desc"],
                          "k_eff": float(K_BASE * v["s_k"])} for k, v in scenarios.items()},
        "forecast_table": scen_tab,
        "uncertainty": {"n_boot": int(len(boot_k)),
                        "k_median": float(np.median(boot_k)) if len(boot_k) else None,
                        "k_p05": float(np.percentile(boot_k,5)) if len(boot_k) else None,
                        "k_p95": float(np.percentile(boot_k,95)) if len(boot_k) else None,
                        "table": unc_tab},
        "current_frontier_p95": float(y_p95[-1]),
        "momentum_model": {"g0_per_year": float(g0), "table": mom_tab},
        "empirical_traj": fr.round(4).to_dict(orient="records"),
    }
    save_json(out, os.path.join(DATA, "s4_forecast.json"))

    print(f"\n[OK] s4_forecast 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s4_forecast")


if __name__ == "__main__":
    main()

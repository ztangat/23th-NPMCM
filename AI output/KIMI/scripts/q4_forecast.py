# -*- coding: utf-8 -*-
"""问题四(预测修正): 2022-2025年度前沿 + C2月度前沿 双序列预测"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:/Users/dkyyt/Desktop/F题"
CDIR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution")
OUT = os.path.join(ROOT, "outputs", "q4_evolution")

# 年度前沿 (top3, 2022-2025)
fn = pd.read_csv(os.path.join(OUT, "frontier_top3_by_year.csv"))
fa = fn[fn.Year >= 2022].copy()
yrs, fvs = fa.Year.values.astype(float), fa.frontier_score.values

# 月度前沿 (C2, 宽口径开源, top5均值)
c2 = pd.read_csv(os.path.join(CDIR, "leaderboard_enhanced.csv"))
c2["date"] = pd.to_datetime(c2["Submission Date"], errors="coerce")
c2 = c2.dropna(subset=["date", "Average ⬆️"])
OSI = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc"}
OPEN = OSI | {"llama2", "llama3", "llama3.1", "llama3.2", "gemma", "creativeml-openrail-m"}
c2o = c2[c2["Hub License"].isin(OPEN)].copy()
c2o["ym"] = c2o.date.dt.to_period("M").astype(str)
fm = c2o.sort_values("Average ⬆️", ascending=False).groupby("ym").head(5) \
        .groupby("ym").agg(f5=("Average ⬆️", "mean"), n=("Average ⬆️", "size")).reset_index()
fm["t"] = fm.ym.str[:4].astype(int) + (fm.ym.str[5:7].astype(int) - 0.5) / 12
fm.to_csv(os.path.join(OUT, "frontier_by_month.csv"), index=False)
print(fm.round(3).to_string())

def forecast(yrs, fvs, label, log_time=False):
    x = np.log(yrs - 2020) if log_time else yrs
    sl, ic, r, p, se = stats.linregress(x, fvs)
    resid = fvs - (ic + sl * x)
    rng = np.random.default_rng(0)
    out = {}
    for xf, hm in [(2026.0, 12), (2027.0, 24)]:
        xt = np.log(xf - 2020) if log_time else xf
        boot = []
        for _ in range(3000):
            yb = fvs + rng.choice(resid, size=len(fvs), replace=True)
            s_, i_, *_ = stats.linregress(x, yb)
            boot.append(i_ + s_ * xt)
        out[hm] = {"point": ic + sl * xt, "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)}
    print(f"{label}: slope={sl:.3f}, R={r:.3f}, p={p:.4f}")
    return out

rows = []
r1 = forecast(yrs, fvs, "年度top3线性(2022-2025)")
r2 = forecast(yrs, fvs, "年度top3对数减速", log_time=True)
r3 = forecast(fm.t.values, fm.f5.values, "月度top5线性(2024.6-2025.3)")
for hm in [12, 24]:
    rows.append({"horizon_months": hm,
                 "annual_linear_point": r1[hm]["point"], "annual_linear_PI": f"[{r1[hm]['lo']:.1f}, {r1[hm]['hi']:.1f}]",
                 "annual_slowdown_point": r2[hm]["point"], "annual_slowdown_PI": f"[{r2[hm]['lo']:.1f}, {r2[hm]['hi']:.1f}]",
                 "monthly_linear_point": r3[hm]["point"], "monthly_linear_PI": f"[{r3[hm]['lo']:.1f}, {r3[hm]['hi']:.1f}]"})
fc = pd.DataFrame(rows)
fc.to_csv(os.path.join(OUT, "frontier_forecast_final.csv"), index=False)
print(fc.round(2).to_string())

# 前沿参数量事实
fa2 = fa.copy()
print("前沿参数logN(2022-2025):", fa2.mean_logN.round(3).tolist(), " → 规模未扩张甚至收缩")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(fn.Year, fn.frontier_score, "o-", color="#AAB7B8", label="年度top3(全部年份)")
ax.plot(yrs, fvs, "o-", color="#2E86C1", lw=2, label="年度top3(2022-2025, 用于趋势)")
sl, ic, r, p, se = stats.linregress(yrs, fvs)
xx = np.linspace(2022, 2027, 100)
ax.plot(xx, ic + sl * xx, "k:", lw=1.2, label=f"线性趋势 {sl:.1f}分/年")
tl = np.log(yrs - 2020); sl2, ic2, *_ = stats.linregress(tl, fvs)
ax.plot(xx, ic2 + sl2 * np.log(xx - 2020), "g--", lw=1.2, label="减速情景(对数时间)")
for hm, xf, col in [(12, 2026.0, "#C0392B"), (24, 2027.0, "#C0392B")]:
    ax.errorbar([xf], [r1[hm]["point"]], yerr=[[r1[hm]["point"] - r1[hm]["lo"]], [r1[hm]["hi"] - r1[hm]["point"]]],
                fmt="D", color=col, capsize=7, markersize=9)
    ax.errorbar([xf + 0.05], [r2[hm]["point"]], yerr=[[r2[hm]["point"] - r2[hm]["lo"]], [r2[hm]["hi"] - r2[hm]["point"]]],
                fmt="s", color="#28B463", capsize=7, markersize=8)
ax.plot([], [], "D", color="#C0392B", label="线性情景预测(95%PI)")
ax.plot([], [], "s", color="#28B463", label="减速情景预测(95%PI)")
ax.set_xlabel("年份"); ax.set_ylabel("前沿能力得分(6维平均)")
ax.set_title("开源模型能力前沿: 12/24个月双情景预测")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_frontier_forecast.png"), dpi=150); plt.close()
print("DONE_Q4_FORECAST")

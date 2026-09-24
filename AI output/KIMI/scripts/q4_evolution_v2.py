# -*- coding: utf-8 -*-
"""
问题四(修正): 分解用C3长窗口为主 + C2稳健性; 前沿=年度top3均值; 双情景预测
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:/Users/dkyyt/Desktop/F题"
CDIR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution")
OUT = os.path.join(ROOT, "outputs", "q4_evolution")

c3 = pd.read_csv(os.path.join(CDIR, "leaderboard_extended_timeseries.csv"))
c3 = c3.dropna(subset=["Year", "Average", "Params_B"])
c3 = c3[c3.Params_B > 0.05].copy()
c3["logN"] = np.log10(c3.Params_B * 1e9)
print("C3 n =", len(c3), "年份:", sorted(c3.Year.unique()))
print(c3.groupby("Year").agg(n=("Average", "size"), max=("Average", "max")).to_string())

# ---- 1. 主分解: Average = b0 + b1*logN + b2*(Year-2022) ----
def ols(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    sigma2 = np.sum((y - pred)**2) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, se, r2

X = np.column_stack([np.ones(len(c3)), c3.logN, c3.Year - 2022])
beta, se, r2 = ols(X, c3.Average.values)
dec = pd.DataFrame({"term": ["const", "log10(N)", "year(tech)"], "coef": beta, "se": se, "t": beta / se})
dec.to_csv(os.path.join(OUT, "decomposition_c3_main.csv"), index=False)
print(dec.round(4).to_string(), "R2=", round(r2, 4))

# 前沿规模增速: 每年top3模型的平均logN
top3 = c3.sort_values("Average", ascending=False).groupby("Year").head(3)
fn = top3.groupby("Year").agg(frontier_score=("Average", "mean"), mean_logN=("logN", "mean")).reset_index()
fn.to_csv(os.path.join(OUT, "frontier_top3_by_year.csv"), index=False)
print(fn.round(3).to_string())
recent = fn[fn.Year >= 2022]
dlogN_year = np.diff(recent.mean_logN).mean()
print(f"前沿参数增速(2022起): {dlogN_year:.3f} dex/年 (≈{10**dlogN_year:.2f}×/年)")

# 分解 (以2022-2025年 frontier 年均提升为总收益)
dscore_year = np.diff(recent.frontier_score).mean()
scale_gain = beta[1] * dlogN_year
tech_gain = beta[2]
total = scale_gain + tech_gain
dec2 = pd.DataFrame([
    {"component": "规模扩张(参数增长)", "annual_points": scale_gain, "share": scale_gain / total},
    {"component": "非规模技术进步", "annual_points": tech_gain, "share": tech_gain / total},
    {"component": "合计(模型预测)", "annual_points": total, "share": 1.0},
    {"component": "前沿实际年均提升(2022-2025)", "annual_points": dscore_year, "share": np.nan}])
dec2.to_csv(os.path.join(OUT, "contribution_decomposition.csv"), index=False)
print(dec2.round(3).to_string())

# ---- 2. 稳健性: C2短窗回归 (含is_chat) ----
c2 = pd.read_csv(os.path.join(CDIR, "leaderboard_enhanced.csv"))
c2["date"] = pd.to_datetime(c2["Submission Date"], errors="coerce")
c2 = c2.dropna(subset=["date", "Average ⬆️", "#Params (B)"])
c2 = c2[c2["#Params (B)"] > 0.05].copy()
c2["logN"] = np.log10(c2["#Params (B)"] * 1e9)
c2["year_frac"] = c2.date.dt.year + (c2.date.dt.dayofyear - 1) / 365.25
c2["is_chat"] = c2["Type"].str.contains("chat", case=False, na=False).astype(float)
X2 = np.column_stack([np.ones(len(c2)), c2.logN, c2.year_frac - 2024.5, c2.is_chat])
b2, se2, r22 = ols(X2, c2["Average ⬆️"].values)
rob = pd.DataFrame({"term": ["const", "log10(N)", "year(tech)", "is_chat"], "coef": b2, "se": se2, "t": b2 / se2})
rob.to_csv(os.path.join(OUT, "decomposition_c2_robustness.csv"), index=False)
print(rob.round(4).to_string(), "R2=", round(r22))

# ---- 3. 前沿预测: 年度top3 (2021-2025) 线性 + 减速两情景 ----
f = fn[fn.Year >= 2021].copy()
yrs, fvs = f.Year.values.astype(float), f.frontier_score.values
# 情景A: 线性持续
slA, icA, rA, pA, _ = stats.linregress(yrs, fvs)
# 情景B: 算力放缓 -> 对数时间(减速): score = a + b*ln(year-2020)
tl = np.log(yrs - 2020)
slB, icB, rB, pB, _ = stats.linregress(tl, fvs)
rng = np.random.default_rng(0)
def boot_pi(fit_fn, xf, n=3000):
    resid = fvs - fit_fn(sl=None, x=yrs)
    out = []
    for _ in range(n):
        yb = fvs + rng.choice(resid, size=len(fvs), replace=True)
        out.append(fit_fn2(yb, xf))
    return np.percentile(out, [2.5, 97.5])
rows = []
residA = fvs - (icA + slA * yrs)
predsA = {2026: [], 2027: []}
residB = fvs - (icB + slB * np.log(yrs - 2020))
predsB = {2026: [], 2027: []}
for _ in range(3000):
    yb = fvs + rng.choice(residA, size=len(fvs), replace=True)
    s_, i_, *_ = stats.linregress(yrs, yb)
    for xf in [2026, 2027]: predsA[xf].append(i_ + s_ * xf)
    yb2 = fvs + rng.choice(residB, size=len(fvs), replace=True)
    s2_, i2_, *_ = stats.linregress(np.log(yrs - 2020), yb2)
    for xf in [2026, 2027]: predsB[xf].append(i2_ + s2_ * np.log(xf - 2020))
for xf, label in [(2026, "+12个月(2026)"), (2027, "+24个月(2027)")]:
    rows.append({"horizon": label, "scenario_A_linear_point": icA + slA * xf,
                 "A_pi_low": np.percentile(predsA[xf], 2.5), "A_pi_high": np.percentile(predsA[xf], 97.5),
                 "scenario_B_slowdown_point": icB + slB * np.log(xf - 2020),
                 "B_pi_low": np.percentile(predsB[xf], 2.5), "B_pi_high": np.percentile(predsB[xf], 97.5)})
fc = pd.DataFrame(rows)
fc.to_csv(os.path.join(OUT, "frontier_forecast.csv"), index=False)
print(f"情景A(线性): slope={slA:.2f}分/年 R={rA:.3f}; 情景B(减速): R={rB:.3f}")
print(fc.round(2).to_string())

# ---- 4. 桥接补充: All拟合RMSE ----
c6 = pd.read_csv(os.path.join(CDIR, "loss_benchmark_bridge_expanded.csv"))
xl, yv = np.log(c6.Val_Loss.values), c6.LB_Average.values
sll, icl, rl, pl, _ = stats.linregress(xl, yv)
rmse_all = float(np.sqrt(np.mean((yv - (icl + sll * xl))**2)))
with open(os.path.join(OUT, "bridge_fit_summary.json"), "w") as fp:
    json.dump({"log_fit": "LB = %.3f + %.3f * ln(Val_Loss)" % (icl, sll),
               "R": rl, "RMSE_points": rmse_all, "n": len(c6)}, fp, indent=2, ensure_ascii=False)
print(f"桥接: LB = {icl:.2f} {sll:+.2f}·ln(Loss), R={rl:.3f}, RMSE={rmse_all:.2f}分")

# ---- 5. 图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(c3.logN, c3.Average, c=c3.Year, s=12, alpha=0.5, cmap="viridis")
xx = np.linspace(c3.logN.min(), c3.logN.max(), 50)
for yy, c in [(2021, "#7D3C98"), (2023, "#2E86C1"), (2025, "#C0392B")]:
    axes[0].plot(xx, beta[0] + beta[1] * xx + beta[2] * (yy - 2022), color=c, label=f"{yy}年")
axes[0].set_xlabel("log10(参数量)"); axes[0].set_ylabel("综合能力得分")
axes[0].set_title("同规模能力随年份上移(=非规模技术进步)")
axes[0].legend(); plt.colorbar(sc, ax=axes[0], label="年份")
axes[1].bar(["规模扩张\n(参数增长)", "非规模技术进步"], [scale_gain, tech_gain], color=["#2E86C1", "#E67E22"])
axes[1].set_ylabel("年均得分提升(分/年)")
axes[1].set_title(f"能力增长分解(2022-2025): 规模{scale_gain/total:.0%} vs 技术{tech_gain/total:.0%}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_scale_vs_tech.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(fn.Year, fn.frontier_score, "o-", color="#2E86C1", label="年度前沿(top3均值)")
xf = np.array([2026, 2027])
ax.plot(np.append(yrs, xf), icA + slA * np.append(yrs, xf), "k:", lw=1, label="情景A: 线性持续")
xxl = np.linspace(2021, 2027, 100)
ax.plot(xxl, icB + slB * np.log(xxl - 2020), "g--", lw=1, label="情景B: 算力放缓(减速)")
ax.errorbar(xf, [icA + slA * 2026, icA + slA * 2027],
            yerr=[[icA + slA * 2026 - fc.iloc[0].A_pi_low, icA + slA * 2027 - fc.iloc[1].A_pi_low],
                  [fc.iloc[0].A_pi_high - icA - slA * 2026, fc.iloc[1].A_pi_high - icA - slA * 2027]],
            fmt="D", color="#C0392B", capsize=6, markersize=8, label="情景A预测(95%PI)")
ax.errorbar(xf + 0.06, fc.scenario_B_slowdown_point,
            yerr=[fc.scenario_B_slowdown_point - fc.B_pi_low, fc.B_pi_high - fc.scenario_B_slowdown_point],
            fmt="s", color="#28B463", capsize=6, markersize=7, label="情景B预测(95%PI)")
ax.set_xlabel("年份"); ax.set_ylabel("前沿能力得分"); ax.set_title("开源模型能力前沿: 12/24个月双情景预测")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_frontier_forecast.png"), dpi=150); plt.close()
print("DONE_Q4_V2")

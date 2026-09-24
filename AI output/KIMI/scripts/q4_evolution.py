# -*- coding: utf-8 -*-
"""
问题四(主分析): 规模扩张 vs 非规模技术进步分解 + Loss-Benchmark桥接 + 前沿预测
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:/Users/dkyyt/Desktop/F题"
CDIR = os.path.join(ROOT, "real_attachments", "C_efficiency_evolution")
OUT = os.path.join(ROOT, "outputs", "q4_evolution")
os.makedirs(OUT, exist_ok=True)

c2 = pd.read_csv(os.path.join(CDIR, "leaderboard_enhanced.csv"))
c2["date"] = pd.to_datetime(c2["Submission Date"], errors="coerce")
c2 = c2.dropna(subset=["date", "Average ⬆️", "#Params (B)"])
c2["year_frac"] = c2["date"].dt.year + (c2["date"].dt.dayofyear - 1) / 365.25

# ---- 口径定义 ----
# 开源(宽口径: 开放权重可研究复现): OSI许可 或 开放权重社区许可
OSI = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "afl-3.0", "gpl-3.0", "lgpl-3.0"}
OPEN_COMMUNITY = OSI | {"llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4",
                        "gemma", "creativeml-openrail-m", "bigscience-openrail-m", "openrail",
                        "apache-2.0", "qwen", "tongyi-qianwen", "deepseek"}
c2["open_wide"] = c2["Hub License"].isin(OPEN_COMMUNITY)
c2["open_strict"] = c2["Hub License"].isin(OSI)
c2["is_pretrained"] = c2["Type"].str.contains("pretrained", case=False, na=False) & \
                      ~c2["Type"].str.contains("continuously", case=False, na=False)
c2["is_chat"] = c2["Type"].str.contains("chat", case=False, na=False)
print("宽口径开源:", c2.open_wide.sum(), "严口径:", c2.open_strict.sum(),
      "pretrained:", c2.is_pretrained.sum(), "chat:", c2.is_chat.sum())

# ---- C4匹配(算力/数据量/开源权重字段) ----
c4 = pd.read_csv(os.path.join(CDIR, "epoch_all_ai_models.csv"))
c4["key"] = c4["Model"].astype(str).str.lower().str.split("/").str[-1].str.replace(r"[^a-z0-9.\-]", "", regex=True)
c2["key"] = c2["Model"].astype(str).str.lower().str.split("/").str[-1].str.replace(r"[^a-z0-9.\-]", "", regex=True)
c4u = c4.dropna(subset=["Training compute (FLOP)"]).drop_duplicates("key")
mg = c2.merge(c4u[["key", "Training compute (FLOP)", "Training dataset size (total)", "Open model weights?",
                   "Publication date"]], on="key", how="left")
matched = mg["Training compute (FLOP)"].notna().sum()
print(f"C4算力匹配: {matched}/{len(mg)}")
mg["logC"] = np.log10(mg["Training compute (FLOP)"])
mg.to_csv(os.path.join(OUT, "c2_c4_merged.csv"), index=False)

# ---- 1. 规模 vs 非规模技术进步分解 ----
# 回归: Average = b0 + b1*logC + b2*(year-2024.5) + b3*is_chat
d = mg.dropna(subset=["logC"]).copy()
d = d[(d["Training compute (FLOP)"] > 1e18)]
X = np.column_stack([np.ones(len(d)), d.logC, d.year_frac - 2024.5, d.is_chat.astype(float)])
y = d["Average ⬆️"].values
beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
# 系数标准误
sigma2 = np.sum((y - pred)**2) / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
decomp = pd.DataFrame({"term": ["const", "log10(compute)", "year(tech)", "is_chat"],
                       "coef": beta, "se": se, "t": beta / se})
decomp.to_csv(os.path.join(OUT, "scale_tech_regression.csv"), index=False)
print(decomp.round(4).to_string(), "\nR2=", round(r2, 4), "n=", len(d))

# 年均收益分解: 规模贡献 = b1 * 年均ΔlogC; 技术贡献 = b2
dd = d.copy()
dd["year"] = dd.date.dt.year
g = dd.groupby("year").agg(mean_logC=("logC", "mean"), max_logC=("logC", "max"), n=("logC", "size"))
print(g.round(3).to_string())
# 用frontier算力增速: 2024 vs 2025 top decile logC
dC_year = g["mean_logC"].diff().mean()
scale_gain = beta[1] * dC_year
tech_gain = beta[2]
dec_share = pd.DataFrame([
    {"component": "规模扩张贡献", "annual_gain_points": scale_gain, "share": scale_gain / (scale_gain + tech_gain)},
    {"component": "非规模技术进步贡献", "annual_gain_points": tech_gain, "share": tech_gain / (scale_gain + tech_gain)}])
dec_share.to_csv(os.path.join(OUT, "contribution_decomposition.csv"), index=False)
print(dec_share.round(4).to_string())

# 稳健性: 仅用params作规模代理在全样本上分解
d2 = mg.dropna(subset=["#Params (B)"]).copy()
d2 = d2[d2["#Params (B)"] > 0.05]
d2["logN"] = np.log10(d2["#Params (B)"] * 1e9)
X2 = np.column_stack([np.ones(len(d2)), d2.logN, d2.year_frac - 2024.5, d2.is_chat.astype(float)])
y2 = d2["Average ⬆️"].values
b2_, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
p2 = X2 @ b2_
r2b = 1 - np.sum((y2 - p2)**2) / np.sum((y2 - y2.mean())**2)
pd.DataFrame({"term": ["const", "log10(N)", "year(tech)", "is_chat"], "coef": b2_}).to_csv(
    os.path.join(OUT, "scale_tech_regression_params_proxy.csv"), index=False)
print("参数量代理回归: logN系数={:.3f}, 年技术进步={:.3f}分/年, R2={:.4f}, n={}".format(b2_[1], b2_[2], r2b, len(d2)))

# ---- 2. 前沿模型追踪 (C3 2019-2025 + C2 季度前沿) ----
c3 = pd.read_csv(os.path.join(CDIR, "leaderboard_extended_timeseries.csv"))
c3 = c3.dropna(subset=["Year", "Average"])
front_year = c3.groupby("Year")["Average"].max().reset_index()
front_year.columns = ["year", "frontier_avg"]
front_year.to_csv(os.path.join(OUT, "frontier_by_year.csv"), index=False)
print(front_year.to_string())

# 季度前沿 (C2, 宽口径开源)
mg["quarter"] = mg.date.dt.to_period("Q").astype(str)
fq = mg[mg.open_wide].groupby("quarter").agg(frontier=("Average ⬆️", "max"), n=("Average ⬆️", "size"),
    q95=("Average ⬆️", lambda s: s.quantile(0.95))).reset_index()
fq["t"] = fq.quarter.str[:4].astype(int) + (fq.quarter.str[-1].astype(int) - 1) / 4 + 0.125
fq.to_csv(os.path.join(OUT, "frontier_by_quarter.csv"), index=False)
print(fq.round(3).to_string())

# ---- 3. 前沿预测 (12/24个月) ----
# 模型: frontier_avg ~ year 线性(年度粒度, 2019-2025), 加上季度前沿校验
yrs = front_year.year.values.astype(float)
fvs = front_year.frontier_avg.values
sl, ic, rv, pv, se_sl = stats.linregress(yrs, fvs)
# bootstrap预测区间
rng = np.random.default_rng(0)
preds_12, preds_24 = [], []
resid = fvs - (ic + sl * yrs)
for _ in range(2000):
    yb = fvs + rng.choice(resid, size=len(fvs), replace=True)
    slb, icb, *_ = stats.linregress(yrs, yb)
    preds_12.append(icb + slb * 2026)
    preds_24.append(icb + slb * 2027)
fc = pd.DataFrame([
    {"horizon": "+12个月(2026)", "point": ic + sl * 2026, "pi_low": np.percentile(preds_12, 2.5),
     "pi_high": np.percentile(preds_12, 97.5)},
    {"horizon": "+24个月(2027)", "point": ic + sl * 2027, "pi_low": np.percentile(preds_24, 2.5),
     "pi_high": np.percentile(preds_24, 97.5)}])
fc.to_csv(os.path.join(OUT, "frontier_forecast.csv"), index=False)
print("年度前沿趋势: slope={:.3f} 分/年 (p={:.2e}), R={:.3f}".format(sl, pv, rv))
print(fc.round(2).to_string())

# 季度粒度稳健性
t_, f_ = fq.t.values, fq.frontier.values
slq, icq, rq, pq, _ = stats.linregress(t_, f_)
print("季度前沿趋势: slope={:.3f} 分/年 (p={:.2e}), R={:.3f}".format(slq, pq, rq))

# ---- 4. Loss-Benchmark桥接 (C6, 分可比性) ----
c6 = pd.read_csv(os.path.join(CDIR, "loss_benchmark_bridge_expanded.csv"))
c6["high"] = c6.Loss_Comparability.str.startswith("High")
bridge_rows = []
for subset_name, sub in [("High(Pythia同分布)", c6[c6.high]), ("Medium(跨技术报告)", c6[~c6.high]), ("All", c6)]:
    x, yv = sub.Val_Loss.values, sub.LB_Average.values
    # 线性 + 对数两种
    slx, icx, rx, px, _ = stats.linregress(x, yv)
    xl = np.log(x)
    sll, icl, rl, pl, _ = stats.linregress(xl, yv)
    bridge_rows.append({"subset": subset_name, "n": len(sub),
                        "linear_slope": slx, "linear_R": rx,
                        "log_slope": sll, "log_intercept": icl, "log_R": rl})
bridge = pd.DataFrame(bridge_rows)
bridge.to_csv(os.path.join(OUT, "loss_benchmark_bridge_fit.csv"), index=False)
print(bridge.round(4).to_string())

# 桥接误差对结论的影响: 用High子集留一法RMSE
sub = c6[c6.high]
errs = []
for i in range(len(sub)):
    tr = sub.drop(sub.index[i])
    slx, icx, *_ = stats.linregress(tr.Val_Loss, tr.LB_Average)
    errs.append(sub.iloc[i].LB_Average - (icx + slx * sub.iloc[i].Val_Loss))
loo_rmse = float(np.sqrt(np.mean(np.array(errs)**2)))
with open(os.path.join(OUT, "bridge_loo_error.json"), "w") as f:
    json.dump({"high_subset_LOO_RMSE_points": loo_rmse}, f, indent=2)
print("桥接LOO RMSE =", round(loo_rmse, 3), "分")

# ---- 5. 图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(d.logC, d["Average ⬆️"], c=d.year_frac, s=10, alpha=0.5, cmap="viridis")
xx = np.linspace(d.logC.min(), d.logC.max(), 50)
for yy, c in [(2024.0, "#2E86C1"), (2025.0, "#C0392B")]:
    axes[0].plot(xx, beta[0] + beta[1] * xx + beta[2] * (yy - 2024.5), color=c,
                 label=f"{int(yy)}年水平线")
axes[0].set_xlabel("log10(训练算力 FLOP)"); axes[0].set_ylabel("综合能力得分")
axes[0].set_title("算力-能力关系随年份上移(=非规模技术进步)")
axes[0].legend(); plt.colorbar(sc, ax=axes[0], label="年份")
axes[1].bar(["规模扩张", "非规模技术进步"], [scale_gain, tech_gain], color=["#2E86C1", "#E67E22"])
axes[1].set_ylabel("年均得分提升(分/年)"); axes[1].set_title("能力增长分解")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_scale_vs_tech.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(front_year.year, front_year.frontier_avg, "o-", color="#2E86C1", label="年度前沿(C3)")
ax.plot(fq.t, fq.frontier, "s--", color="#E67E22", alpha=0.7, label="季度前沿(C2,开源)")
xf = np.array([2026, 2027])
ax.errorbar(xf, fc.point, yerr=[fc.point - fc.pi_low, fc.pi_high - fc.pi_low],
            fmt="D", color="#C0392B", capsize=6, markersize=8, label="预测(95%PI)")
ax.plot(np.append(yrs, xf), ic + sl * np.append(yrs, xf), "k:", lw=1)
ax.set_xlabel("年份"); ax.set_ylabel("前沿能力得分(6维平均)")
ax.set_title("开源模型能力前沿与12/24个月预测")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_frontier_forecast.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(c6[c6.high].Val_Loss, c6[c6.high].LB_Average, s=50, color="#28B463",
           label=f"High可比 (n={c6.high.sum()})", zorder=3)
ax.scatter(c6[~c6.high].Val_Loss, c6[~c6.high].LB_Average, s=18, alpha=0.5,
           color="#7D3C98", label=f"Medium可比 (n={(~c6.high).sum()})")
xs = np.linspace(c6.Val_Loss.min(), c6.Val_Loss.max(), 100)
bh = bridge[bridge.subset == "All"].iloc[0]
ax.plot(xs, bh.log_intercept + bh.log_slope * np.log(xs), "r--",
        label=f"对数拟合 R={bh.log_R:.3f}")
ax.set_xlabel("Val Loss"); ax.set_ylabel("Leaderboard Average")
ax.set_title("Loss–Benchmark 桥接映射"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_bridge.png"), dpi=150); plt.close()
print("DONE_Q4_MAIN")

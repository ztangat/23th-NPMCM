# -*- coding: utf-8 -*-
"""
s6_figures.py —— 问题四 阶段6：图表产出

产出 8 张核心图（results/figures/）：
  fig1  数据总览：C1 模型规模-能力散点 + 类型/时间彩色
  fig2  规模-能力-时间二维模型（规模弹性 vs 时间趋势）
  fig3  贡献分解（规模 vs 非规模）—— 多方法对比
  fig4  固定规模分箱前沿随时间的上移（效率进步核心证据）
  fig5  规模分布迁移（全体 vs 前沿）
  fig6  前沿边界预测（3 情景 + 不确定性带）
  fig7  逐任务前沿上移（C8，6 benchmark）
  fig8  瓶颈与 headroom 画像（前沿 Top20 vs 全体）
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

# 中国习惯：涨=红，跌=绿
RED = "#d62728"; GREEN = "#2ca02c"; BLUE = "#1f77b4"; ORANGE = "#ff7f0e"
PURPLE = "#9467bd"; GRAY = "#7f7f7f"


def load_json(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))


def main():
    start_log("s6_figures")
    t0 = time.time()
    print("=" * 78)
    print("问题四 阶段6：图表产出")
    print("=" * 78)

    c1 = load_C1()
    d = c1.dropna(subset=["submission_dt", AVG_COL, "#Params (B)"]).copy()
    d = d[d["#Params (B)"] > 0]
    d["x"] = np.log10(d["#Params (B)"])
    t0d = d["submission_dt"].min()
    d["t"] = (d["submission_dt"] - t0d).dt.days / 365.25
    d["ym"] = d["submission_dt"].dt.to_period("M")

    s3 = load_json("s3_decomposition.json")
    s4 = load_json("s4_forecast.json")
    s5 = load_json("s5_tasklevel.json")

    # ============ fig1 数据总览 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    for gname, col, mk in [("pretrained", BLUE, "o"), ("chat_finetuned", ORANGE, "o")]:
        g = d[d["type_grp"] == gname]
        ax.scatter(g["#Params (B)"], g[AVG_COL], s=7, alpha=0.35, c=col,
                   label=f"{gname} (n={len(g)})", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("参数量 N (B, log)"); ax.set_ylabel("能力 Average ⬆️")
    ax.set_title("(a) C1 模型规模-能力散点（按类型着色）", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(d["#Params (B)"], d[AVG_COL], s=8, c=d["t"], cmap="viridis",
                    alpha=0.55, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("参数量 N (B, log)"); ax.set_ylabel("能力 Average ⬆️")
    ax.set_title("(b) 同上，按提交时间着色（越黄越晚）", fontsize=12)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("时间 (年，相对首月)", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_数据总览.png")); plt.close(fig)
    print("   [图] fig1_数据总览.png")

    # ============ fig2 二维模型 ============
    b2 = s3["baseline_2d_model"]
    a_, b_, c_ = b2["a"], b2["b"], b2["c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    xg = np.linspace(d["x"].min(), d["x"].max(), 100)
    for tqv, col in [(0.0, BLUE), (0.25, ORANGE), (0.5, GREEN), (0.75, RED)]:
        ax.plot(xg, a_ + b_ * xg + c_ * tqv, color=col, lw=2,
                label=f"τ={tqv:.2f}年")
    ax.scatter(d["x"], d[AVG_COL], s=5, alpha=0.15, c=GRAY, edgecolors="none")
    ax.set_xlabel("log10(N)"); ax.set_ylabel("能力 Average ⬆️")
    ax.set_title(f"(a) 二维模型 y={a_:.2f}+{b_:.2f}·logN+{c_:.2f}·τ\n"
                 f"(R²={b2['metrics']['R2']:.3f})", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    lay = s3.get("by_type", {})
    names = list(lay.keys()); bs = [lay[k]["b"] for k in names]; cs = [lay[k]["c"] for k in names]
    xs = np.arange(len(names)); w = 0.35
    ax.bar(xs - w/2, bs, w, color=BLUE, label="规模弹性 b (分/decade)")
    ax.bar(xs + w/2, cs, w, color=ORANGE, label="时间趋势 c (分/年)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=10)
    ax.set_title("(b) 分层拟合：不同类型模型的规模弹性 vs 时间趋势", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_二维模型.png")); plt.close(fig)
    print("   [图] fig2_二维模型.png")

    # ============ fig3 贡献分解 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    te = s3["tech_estimates"]
    labels = ["固定规模前沿\n上移(加权)", "固定规模\nOLS中位", "回归时间项 c", "同规模早晚\n配对"]
    keys = ["A_fixed_scale_weighted", "A_ols_median", "regression_c", "pair_tech"]
    vals = [te[k] for k in keys]
    bars = ax.bar(range(len(vals)), vals, color=ORANGE, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(np.median(vals), color=RED, ls="--", lw=1.5,
               label=f"中位综合 = {np.median(vals):.2f} 分/年")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("非规模技术进步率 (分/年)")
    ax.set_title("(a) 四条独立路径给出的'非规模技术进步率'", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    summ = s3["summary"]
    gs = summ["g_scale_front"]; gtf = summ["tech_median"]
    gsa = summ["g_scale_all"]
    g_obs = summ["g_obs"]
    cats = ["前沿规模口径\n(主结论)", "全体规模口径\n(对照)"]
    sc_v = [gs, gsa]; tc_v = [gtf, gtf]
    xs = np.arange(2); w = 0.35
    b1 = ax.bar(xs - w/2, sc_v, w, color=RED, label="规模扩张贡献")
    b2 = ax.bar(xs + w/2, tc_v, w, color=ORANGE, label="非规模技术进步")
    ax.axhline(g_obs, color=BLUE, ls=":", lw=1.6, label=f"观测前沿增速 = {g_obs:.2f}")
    ax.axhline(0, color="k", lw=0.8)
    for bs_ in (b1, b2):
        for r in bs_:
            h = r.get_height()
            ax.text(r.get_x()+r.get_width()/2, h + (0.2 if h >= 0 else -0.9),
                    f"{h:+.1f}", ha="center", fontsize=9.5)
    ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("年化贡献 (分/年)")
    ax.set_title("(b) 两力分解：规模 vs 非规模（分/年）", fontsize=12)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_贡献分解.png")); plt.close(fig)
    print("   [图] fig3_贡献分解.png")

    # ============ fig4 固定规模前沿上移 ============
    chA = s3["channelA_fixed_scale"]["per_bin"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    # 重算分箱逐月前沿
    bins = [0, 2, 4, 8, 16, 32, 1e9]
    blabels = ["[0,2)B", "[2,4)B", "[4,8)B", "[8,16)B", "[16,32)B", "[32,80+]B"]
    d["bin"] = pd.cut(d["#Params (B)"], bins=bins, labels=blabels, right=False)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(blabels)))
    for i, bl in enumerate(blabels):
        g = d[d["bin"] == bl]
        if len(g) < 10:
            continue
        ser = g.groupby("ym")[AVG_COL].quantile(0.90)
        ts = g.groupby("ym")["t"].mean()
        ser.index = ser.index.astype(str)
        ax.plot(ts.values, ser.values, "o-", color=cmap[i], lw=1.8, ms=4,
                label=f"{bl} (n={len(g)})")
    ax.set_xlabel("时间 τ (年)"); ax.set_ylabel("该规模档位内 前沿 F_p90")
    ax.set_title("固定规模分箱内的能力前沿随时间上移\n（＝非规模技术进步的纯净度量）", fontsize=12.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_固定规模前沿上移.png")); plt.close(fig)
    print("   [图] fig4_固定规模前沿上移.png")

    # ============ fig5 规模分布迁移 ============
    ds = pd.DataFrame(s3["scale_dist_monthly"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ax.plot(ds["t"], 10**ds["x_mean"], "o-", color=BLUE, label="全体样本 规模均值")
    frd = pd.DataFrame(s3["frontier_monthly"])
    ax.plot(frd["t"], 10**frd["x_front_med"], "s-", color=RED, label="前沿(P95) 中位规模")
    ax.set_yscale("log")
    ax.set_xlabel("时间 τ (年)"); ax.set_ylabel("参数量 N (B, log)")
    ax.set_title("(a) 规模轨迹：并未扩张，反而收缩", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    # 规模分布箱线（早 vs 晚）
    months = sorted(d["ym"].unique())
    early = d[d["ym"].isin(months[:3])]["x"]; late = d[d["ym"].isin(months[-3:])]["x"]
    ax.hist(early, bins=40, alpha=0.55, color=BLUE, label="首3月", density=True)
    ax.hist(late, bins=40, alpha=0.55, color=RED, label="末3月", density=True)
    ax.set_xlabel("log10(N)"); ax.set_ylabel("密度")
    ax.set_title("(b) 规模分布：未右移（甚至左移）", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_规模分布迁移.png")); plt.close(fig)
    print("   [图] fig5_规模分布迁移.png")

    # ============ fig6 前沿边界预测 ============
    fig, ax = plt.subplots(figsize=(10, 6))
    emp = pd.DataFrame(s4["empirical_traj"])
    ax.plot(emp["t"], emp["F_p95"], "ko-", lw=2, ms=5, label="观测前沿 F_p95")
    ax.plot(emp["t"], emp["F_max"], "o", color=GRAY, ms=4, alpha=0.6, label="月度最强 F_max")
    # 拟合段 + 预测段
    fit = s4["saturating_fit"]["fixed_ceiling"]["params"]
    Fi = s4["ceilings"]["F_INF_BASE"]; k = s4["saturating_fit"]["k_base"]
    ta = emp["t"].values[0]; Fa = fit["F_a"]
    tgrid = np.linspace(emp["t"].min(), emp["t"].max() + 2.0, 200)
    scen_colors = {"A_基准": BLUE, "B_放缓": ORANGE, "C_严重放缓": RED}
    for sname, sinfo in s4["scenarios"].items():
        ke = sinfo["k_eff"]
        yg = Fi - (Fi - Fa) * np.exp(-ke * (tgrid - ta))
        ax.plot(tgrid, yg, color=scen_colors[sname], lw=2, ls="--",
                label=f"{sname} (s_k={sinfo['s_k']:.2f})")
    # 不确定性带（基准情景）
    unc = pd.DataFrame(s4["uncertainty"]["table"])
    for _, r in unc[unc["scenario"] == "A_基准"].iterrows():
        th = emp["t"].max() + r["horizon_m"]/12.0
        ax.fill_between([th-0.05, th+0.05], [r["p05"]]*2, [r["p95"]]*2,
                        color=BLUE, alpha=0.2)
        ax.errorbar(th, r["p50"], yerr=[[r["p50"]-r["p05"]], [r["p95"]-r["p50"]]],
                    fmt="D", color=BLUE, ms=6, capsize=4, lw=1.5)
    ax.axvline(emp["t"].max(), color="gray", ls=":", lw=1.2)
    ax.text(emp["t"].max()+0.02, 30, "预测起点", fontsize=9, color="gray", rotation=90)
    ax.axhline(Fi, color=GREEN, ls=":", lw=1.4, label=f"天花板 F_inf={Fi:.0f}")
    ax.set_xlabel("时间 τ (年)"); ax.set_ylabel("前沿边界 F_p95")
    ax.set_title("开源模型能力前沿边界预测（3 算力情景 + 90% 不确定性区间）", fontsize=12.5)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_前沿边界预测.png")); plt.close(fig)
    print("   [图] fig6_前沿边界预测.png")

    # ============ fig7 逐任务前沿上移 ============
    tft = s5["task_frontier_traj"]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(tft)))
    for i, (task, ser) in enumerate(tft.items()):
        items = sorted([(k, v) for k, v in ser.items() if v is not None])
        xs_ = list(range(len(items))); ys_ = [v for _, v in items]
        ax.plot(xs_, ys_, "o-", color=cmap[i], lw=1.8, ms=4, label=task)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], fontsize=7.5, rotation=45, ha="right")
    ax.set_ylabel("各 benchmark 前沿 P95")
    ax.set_title("逐 benchmark 前沿上移（C8 微观证据）\nMATH Lvl 5 暴涨、IFEval/GPQA 饱和", fontsize=12.5)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_逐任务前沿上移.png")); plt.close(fig)
    print("   [图] fig7_逐任务前沿上移.png")

    # ============ fig8 瓶颈画像 ============
    prof = pd.DataFrame(s5["frontier_profile"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    xs = np.arange(len(prof)); w = 0.35
    ax.bar(xs - w/2, prof["front_top20"], w, color=RED, label="前沿 Top20 均值")
    ax.bar(xs + w/2, prof["all_mean"], w, color=BLUE, label="全体均值")
    ax.set_xticks(xs); ax.set_xticklabels(prof["task"], fontsize=9)
    ax.set_ylabel("正确率 (0–1)")
    ax.set_title("(a) 各 benchmark：前沿 vs 全体", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    pf = prof.sort_values("headroom", ascending=True)
    ax.barh(pf["task"], pf["headroom"], color=ORANGE, alpha=0.85)
    for i, v in enumerate(pf["headroom"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_xlabel("剩余提升空间 Headroom = 1 − 前沿Top20")
    ax.set_title("(b) 瓶颈识别：谁还有空间？", fontsize=12)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_瓶颈画像.png")); plt.close(fig)
    print("   [图] fig8_瓶颈画像.png")

    print(f"\n[OK] s6_figures 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s6_figures")


if __name__ == "__main__":
    main()

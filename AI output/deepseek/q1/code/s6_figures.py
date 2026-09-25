# -*- coding: utf-8 -*-
"""
s6_figures.py —— 阶段6：问题一全部图表产出

产出图表（results/figures/）：
  fig1_质量分布与域得分.png        各数据集/域的综合质量 Q 分布与排序
  fig2_权重与一致性.png            四种赋权方案权重对比 + 一致性热图
  fig3_冲突诊断.png                族间冲突 Dfam 分布 + 与 Q 的关系
  fig4_域映射.png                  7 质量域 → 17 配方域 映射与迁移质量
  fig5_配比损失拟合.png            s5 各数据集 预测 vs 实际（含 R²）
  fig6_自效应热图.png              13域 × 6规模 自效应相关系数热图
  fig7_规模依赖性.png              幅度收缩比 + 域结构指纹相关
  fig8_域级建模.png                13 域在各规模上的 R² 柱状图
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches

C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"
KEYS_ORDER = ["train", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
SCALE_LABEL = {"train": "1M(训练)", "test_1m": "1M(同规模)", "test_60m": "60M",
               "test_1B": "1B", "est_10b": "10B", "est_70b": "70B"}


def fig1():
    z = np.load(os.path.join(DATA, "s2_scores_sample.npz"), allow_pickle=True)
    Q = z["Q"]; dom = z["DOM"]
    s2 = json.load(open(os.path.join(DATA, "s2_summary.json")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 左：Q 分布
    ax = axes[0]
    ax.hist(Q, bins=60, color=C_MAIN, alpha=0.8, edgecolor="white")
    ax.axvline(Q.mean(), color=C_ALT, ls="--", lw=2, label=f"均值={Q.mean():.4f}")
    ax.set_xlabel("综合质量得分 Q"); ax.set_ylabel("样本数")
    ax.set_title("(a) SlimPajama 样本综合质量分布 (n=51,230)", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    # 右：域排序
    ax = axes[1]
    uniq = sorted(set(dom))
    means = [Q[dom == d].mean() for d in uniq]
    order = np.argsort(means)
    ax.barh([uniq[i] for i in order], [means[i] for i in order],
            color=[C_G if m > Q.mean() else C_ALT for m in np.array(means)[order]], alpha=0.85)
    ax.axvline(Q.mean(), color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("域内平均质量 Q"); ax.set_title("(b) 各数据集质量排序", fontsize=12)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_质量分布与域得分.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig2():
    z = np.load(os.path.join(DATA, "s2_weights.npz"), allow_pickle=True)
    s2 = json.load(open(os.path.join(DATA, "s2_summary.json")))
    cons = json.load(open(os.path.join(DATA, "s2_consistency.json")))
    schemes = ["w_ew", "w_ewm", "w_critic", "w_pca", "w_combo"]
    labels = ["等权", "熵权", "CRITIC", "PCA", "组合"]
    W = np.array([z[s] for s in schemes])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    x = np.arange(len(ALL_METRICS)); wdt = 0.15
    for j, (w, lb) in enumerate(zip(W, labels)):
        ax.bar(x + j * wdt, w, wdt, label=lb, alpha=0.85)
    ax.set_xticks(x + 2 * wdt)
    ax.set_xticklabels([METRIC_CN.get(m, m) for m in ALL_METRICS], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("权重"); ax.set_title("(a) 五种赋权方案下 22 指标权重", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    # 右：一致性
    ax = axes[1]
    vals = [cons.get("kendall_W", np.nan)]
    names = ["Kendall W"]
    sm = cons.get("spearman_matrix", None)
    if sm is not None:
        try:
            sma = np.array(sm, dtype=float)
            if sma.ndim == 2:
                off = sma[np.triu_indices_from(sma, k=1)]
                vals.append(float(np.mean(off)))
                names.append("方案间 Spearman 均值")
        except Exception:
            pass
    bars = ax.bar(names, vals, color=[C_MAIN, C_G][:len(vals)], alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=11)
    ax.axhline(0.8, color=C_ALT, ls="--", lw=1.5, label="W=0.8 强一致阈值")
    ax.set_ylim(0, 1.12); ax.set_ylabel("系数")
    ax.set_title("(b) 赋权方案排序一致性检验", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_权重与一致性.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig3():
    z = np.load(os.path.join(DATA, "s3_conflict_sample.npz"), allow_pickle=True)
    s3 = json.load(open(os.path.join(DATA, "s3_summary.json")))
    Dfam = z["Dfam"]
    Q = np.load(os.path.join(DATA, "s2_scores_sample.npz"), allow_pickle=True)["Q"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(Dfam, bins=60, color=C_O, alpha=0.85, edgecolor="white")
    thr = np.percentile(Dfam, 90)
    ax.axvline(thr, color=C_ALT, ls="--", lw=2, label=f"冲突阈值(p90)={thr:.3f}")
    ax.set_xlabel("族间冲突度 D_fam"); ax.set_ylabel("样本数")
    ax.set_title(f"(a) 质量族间冲突分布 (冲突率={s3.get('conflict_rate_sample',np.nan):.1%})", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.scatter(Dfam, Q, s=3, alpha=0.25, color=C_P)
    r = np.corrcoef(Dfam, Q)[0, 1]
    ax.set_xlabel("族间冲突度 D_fam"); ax.set_ylabel("综合质量 Q")
    ax.set_title(f"(b) 冲突度 vs 综合质量 (r={r:+.3f})", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_冲突诊断.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig4():
    z = np.load(os.path.join(DATA, "s4_domain_Q.npz"), allow_pickle=True)
    dom = z["domains"]; Q = z["Q"]; lev = z["levels"]; meth = z["methods"]
    order = np.argsort(Q)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = {"L1": C_G, "L2": C_O, "L3": C_ALT, "direct": C_G, "near_direct": C_O, "inferred": C_ALT}
    colors = [cmap.get(str(m), C_MAIN) for m in np.array(meth)[order]]
    bars = ax.barh([dom[i] for i in order], [Q[i] for i in order], color=colors, alpha=0.88)
    for b, i in zip(bars, order):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{Q[i]:.3f}", va="center", fontsize=9)
    ax.set_xlabel("映射后域质量 Q"); ax.set_xlim(0, max(Q) * 1.15)
    ax.set_title("17 个 The Pile 配方域的质量得分（问题一输出，供问题二/三使用）", fontsize=13)
    handles = [mpatches.Patch(color=cmap["L1"], label="direct 直接匹配"),
               mpatches.Patch(color=cmap["L2"], label="near_direct 近似匹配"),
               mpatches.Patch(color=cmap["L3"], label="inferred 推断")]
    ax.legend(handles=handles, loc="lower right"); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_域映射.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig5():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ev = json.load(open(os.path.join(DATA, "s5_summary.json")))["eval"]
    evd = {e["数据集"]: e for e in ev}
    for ax, k in zip(axes.ravel(), KEYS_ORDER):
        d = np.load(os.path.join(DATA, f"s5_pred_{k}.npy"))
        y, p = d[:, 0], d[:, 1]
        r2v = evd[k]["R²"]; sp = evd[k]["Spearman"]
        col = C_G if r2v > 0.6 else (C_O if r2v > 0 else C_ALT)
        ax.scatter(y, p, s=12, alpha=0.5, color=col)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_title(f"{SCALE_LABEL[k]}  R²={r2v:.3f}  ρ={sp:.3f}", fontsize=11)
        ax.set_xlabel("实际 y"); ax.set_ylabel("预测 y"); ax.grid(alpha=0.3)
    plt.suptitle("配比→损失 模型预测 vs 实际（双重中心化加权损失 y）", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_配比损失拟合.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig6():
    df = pd.read_csv(os.path.join(TABLES, "s5_域自效应相关.csv"))
    M = df[KEYS_ORDER].values
    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(KEYS_ORDER)))
    ax.set_xticklabels([SCALE_LABEL[k] for k in KEYS_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["域"])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if abs(M[i, j]) > 0.5 else "black")
    plt.colorbar(im, label="corr(log p_i, Lzz_i)")
    ax.set_title("同域自效应：13 域 × 6 模型规模\n（负=本域数据量增加降低本域相对损失）", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_自效应热图.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig7():
    s5 = json.load(open(os.path.join(DATA, "s5_summary.json")))
    ratio = s5["scale_amplitude_ratio_70B_over_1M"]
    fpc = s5["fingerprint_corr"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ds = list(ratio.keys())
    vals = [ratio[d] for d in ds]
    order = np.argsort(vals)
    ax.barh([ds[i] for i in order], [vals[i] for i in order], color=C_T, alpha=0.85)
    ax.axvline(np.mean(vals), color=C_ALT, ls="--", lw=1.5,
               label=f"均值={np.mean(vals):.3f}")
    ax.set_xlabel("sd(70B) / sd(1M)  相对损失幅度比")
    ax.set_title("(a) 逐域相对损失幅度的规模收缩", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    ax = axes[1]
    ks = KEYS_ORDER
    cvals = [fpc[k] for k in ks]
    cols = [C_G if v > 0.95 else (C_O if v > 0.7 else C_ALT) for v in cvals]
    bars = ax.bar([SCALE_LABEL[k] for k in ks], cvals, color=cols, alpha=0.88)
    for b, v in zip(bars, cvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(0.95, color=C_G, ls="--", lw=1.2, alpha=0.7, label="0.95 结构保持阈值")
    ax.set_ylabel("域难度指纹与训练集的相关"); ax.set_ylim(0, 1.1)
    ax.set_title("(b) 各规模域结构指纹一致性", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_规模依赖性.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig8():
    df = pd.read_csv(os.path.join(TABLES, "s5_域级建模结果.csv"))
    ks = ["test_1m_R²", "test_60m_R²", "test_1B_R²", "est_10b_R²", "est_70b_R²"]
    labels = ["1M", "60M", "1B", "10B", "70B"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(df)); wdt = 0.16
    for j, (c, lb) in enumerate(zip(ks, labels)):
        vals = df[c].values
        cols = [C_G if v > 0.5 else (C_O if v > 0 else C_ALT) for v in vals]
        ax.bar(x + j * wdt, vals, wdt, color=cols, alpha=0.9, label=lb, edgecolor="white", lw=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + 2 * wdt); ax.set_xticklabels(df["损失域"], rotation=45, ha="right")
    ax.set_ylabel("域级模型 R²"); ax.set_ylim(-0.4, 1.05)
    ax.set_title("13 个损失域的配比-损失模型在各模型规模上的泛化能力", fontsize=13)
    ax.legend(ncol=5); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig8_域级建模.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    start_log("s6_figures")
    os.makedirs(FIG, exist_ok=True)
    for f, nm in [(fig1, "质量分布"), (fig2, "权重一致性"), (fig3, "冲突诊断"),
                  (fig4, "域映射"), (fig5, "配比损失拟合"), (fig6, "自效应热图"),
                  (fig7, "规模依赖"), (fig8, "域级建模")]:
        try:
            f(); print(f"  [OK] {nm}")
        except Exception as e:
            print(f"  [ERR] {nm}: {e}")
    print("[OK] s6_figures 完成")
    end_log("s6_figures")


if __name__ == "__main__":
    main()

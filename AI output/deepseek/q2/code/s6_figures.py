# -*- coding: utf-8 -*-
"""
s6_figures.py —— 问题二 全部图表产出

产出（results/figures/）：
  fig1_经典标度律.png        B1 拟合：预测vs实际 + 损失等值线 + 残差
  fig2_族间验证.png           B1主拟合→B2/B3/B4/B5 直接预测 vs 族内标定
  fig3_Q方向诊断.png          B6/B7/B8 的 L vs Q 方向对比 + B8 截断
  fig4_广义律拟合.png         最终形式 M4 的预测vs实际 + 残差分布
  fig5_Q弹性.png              Q 的等效数据量倍数 + 不同规模下 dL/dQ
  fig6_弹性与替代.png         η_N/η_D/η_Q + Q↔D 替代率
  fig7_外推与置信带.png       100B-10T 损失外推（多Q情景 + 95%CI）
  fig8_质量红利.png           同目标损失的算力节省 + 反问题资源需求
  fig9_不确定性分解.png       各参数对预测方差的贡献
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

C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"


def _load(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))


def fig1():
    z = np.load(os.path.join(DATA, "s1_preds.npz"), allow_pickle=True)
    N, D, L = z["b1_N"], z["b1_D"], z["b1_L"]
    P = z["b1_pred"]; p = z["p_add"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 预测 vs 实际
    ax = axes[0]
    ax.scatter(L, P, s=8, c=N, cmap="viridis", alpha=0.7)
    lim = [min(L.min(), P.min()) - 0.05, max(L.max(), P.max()) + 0.05]
    ax.plot(lim, lim, "r--", lw=1.5, label="y=x")
    ax.set_xlabel("实际 val_loss"); ax.set_ylabel("预测 val_loss")
    ax.set_title("(a) B1 拟合：预测 vs 实际 (n=1176)", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    # (b) 损失等值线（N-D 平面）
    ax = axes[1]
    ng = np.logspace(np.log10(0.05), np.log10(15), 120)
    dg = np.logspace(np.log10(2), np.log10(1300), 120)
    NG, DG = np.meshgrid(ng, dg)
    LG = p[0] + p[1] * NG ** (-p[2]) + p[3] * DG ** (-p[4])
    cs = ax.contourf(NG, DG, LG, levels=20, cmap="RdYlGn_r", alpha=0.9)
    ax.scatter(N, D, s=6, c="k", alpha=0.35, label="B1 实测点")
    plt.colorbar(cs, ax=ax, label="val_loss")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("参数量 N (B)"); ax.set_ylabel("token 数 D (B)")
    ax.set_title("(b) 损失等值线 (N–D 平面)", fontsize=12)
    ax.legend(fontsize=8)
    # (c) 残差
    ax = axes[2]
    r = L - P
    ax.hist(r, bins=60, color=C_MAIN, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("残差 (实际-预测)"); ax.set_ylabel("频数")
    ax.set_title(f"(c) 残差分布 (std={r.std():.2e})", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_经典标度律.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig2():
    s1 = _load("s1_classic_params.json")
    val = pd.DataFrame(s1["validation"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dsets = ["B1_pythia", "B2_cerebras", "B3_traj", "B4_baseline", "B5_published"]
    # (a) R² 对比
    ax = axes[0]
    x = np.arange(len(dsets)); w = 0.26
    for j, (fm, lb, c) in enumerate([("加性(直接预测)", "加性·直接", C_ALT),
                                     ("乘积(直接预测)", "乘积·直接", C_O),
                                     ("加性(族内标定)", "加性·族内标定", C_G)]):
        r2 = [val[(val.数据集 == d) & (val.形式 == fm)]["R2"].values[0] for d in dsets]
        ax.bar(x + j * w, r2, w, label=lb, color=c, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w); ax.set_xticklabels(dsets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("R²"); ax.set_title("(a) 跨数据集验证：三种预测方式 R²", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    # (b) 留一规模 CV
    ax = axes[1]
    cv = pd.read_csv(os.path.join(TABLES, "s1_留一规模CV.csv"))
    ax.plot(cv.留出规模_B, cv.R2, "o-", color=C_MAIN, lw=2, ms=7, label="留出 R²")
    ax.set_xscale("log")
    ax.set_xlabel("留出的模型规模 (B)"); ax.set_ylabel("R²")
    ax.set_title("(b) 留一规模交叉验证 (B1, 8 折)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend()
    for _, r_ in cv.iterrows():
        ax.annotate(f"α={r_.alpha_fold:.3f}\nβ={r_.beta_fold:.3f}",
                    (r_.留出规模_B, r_.R2), fontsize=7, xytext=(0, -18),
                    textcoords="offset points", ha="center", color=C_G)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_族间验证.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig3():
    b6, b7, b8 = loadB("B6"), loadB("B7"), loadB("B8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 全局散点 L vs lnQ
    ax = axes[0]
    for d, lb, c in [(b6, "B6", C_MAIN), (b7, "B7", C_G), (b8, "B8", C_ALT)]:
        ax.scatter(np.log(d.Q_score), d.val_loss, s=5, alpha=0.35, color=c, label=lb)
    ax.set_xlabel("ln(Q_score)"); ax.set_ylabel("val_loss")
    ax.set_title("(a) 全局：L 与 lnQ 的方向", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    # (b) 切片示例：B7 单调下降 vs B8 单调上升
    ax = axes[1]
    s7 = b7[(b7.N_params_B == 1.0) & (b7.D_tokens_B == 150)].sort_values("Q_score")
    ax.plot(s7.Q_score, s7.val_loss, "o-", color=C_G, lw=2, ms=5, label="B7: N=1B,D=150B (质量口径)")
    s8 = b8[(b8.N_params_B == 1.0) & (b8.D_tokens_B == 100)].sort_values("Q_score")
    ax.plot(s8.Q_score, s8.val_loss, "s-", color=C_ALT, lw=2, ms=5, label="B8: N=1B,D=100B (反转口径)")
    ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="B8 硬下截断 0.5")
    ax.set_xlabel("Q_score"); ax.set_ylabel("val_loss")
    ax.set_title("(b) 同规模切片：B7 下降 / B8 上升", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (c) B8 截断诊断：按 Q 分箱
    ax = axes[2]
    t = pd.read_csv(os.path.join(TABLES, "s2b_B8截断诊断.csv"))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.n, 0.4, color=C_MAIN, alpha=0.85, label="总点数")
    ax.bar(x + 0.2, t.floor数, 0.4, color=C_ALT, alpha=0.85, label="截断点数")
    ax.set_xticks(x); ax.set_xticklabels(t.Q_score, fontsize=9)
    ax.set_xlabel("Q 分箱"); ax.set_ylabel("点数")
    ax.set_title("(c) B8 val_loss≤0.5 截断按 Q 分布", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_Q方向诊断.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig4():
    z = np.load(os.path.join(DATA, "s2c_preds.npz"), allow_pickle=True)
    N, D, Q, L, P = z["b7_N"], z["b7_D"], z["b7_Q"], z["b7_L"], z["b7_pred"]
    s2c = _load("s2c_final_params.json")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 预测 vs 实际（按 Q 着色）
    ax = axes[0]
    sc = ax.scatter(L, P, s=10, c=Q, cmap="plasma", alpha=0.8)
    lim = [min(L.min(), P.min()) - .05, max(L.max(), P.max()) + .05]
    ax.plot(lim, lim, "r--", lw=1.5)
    plt.colorbar(sc, ax=ax, label="Q")
    ax.set_xlabel("实际 val_loss"); ax.set_ylabel("预测 val_loss")
    ax.set_title("(a) 广义律 M4 拟合 (B7, n=450)", fontsize=12)
    ax.grid(alpha=0.3)
    # (b) 残差 vs Q
    ax = axes[1]
    r = P - L
    ax.scatter(Q, r, s=10, c=N, cmap="viridis", alpha=0.8)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Q"); ax.set_ylabel("残差 (预测-实际)")
    ax.set_title("(b) 残差 vs Q（无明显系统偏差）", fontsize=12)
    ax.grid(alpha=0.3)
    # (c) 形式对比 BIC
    ax = axes[2]
    fs = pd.DataFrame(s2c["form_selection"]).sort_values("BIC")
    ax.barh(fs.形式, -fs.BIC, color=C_MAIN, alpha=0.85)
    ax.set_xlabel("−BIC（越大越好）")
    ax.set_title("(c) 候选形式 BIC 对比", fontsize=12)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_广义律拟合.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig5():
    s2c = _load("s2c_final_params.json")
    pf = s2c["params_B7"]
    E, a, al, b, be, g = pf["E"], pf["a"], pf["alpha"], pf["b"], pf["beta"], pf["g"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    # (a) 等效数据量倍数
    ax = axes[0]
    qq = np.linspace(0.05, 1.0, 100)
    ax.plot(qq, qq ** (-g / be), color=C_MAIN, lw=2.5, label=f"Q^(-γ/β), γ={g:.4f}, β={be:.4f}")
    for qv in [0.2, 0.5, 0.8]:
        ax.axvline(qv, color="gray", ls=":", lw=1)
        ax.plot(qv, qv ** (-g / be), "o", color=C_ALT, ms=8)
        ax.annotate(f"×{qv**(-g/be):.1f}", (qv, qv ** (-g / be)), fontsize=9,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("数据质量 Q"); ax.set_ylabel("等效数据量倍数 (相对 Q=1)")
    ax.set_title("(a) 质量→数据量当量换算", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscat = ax.set_yscale("log")
    # (b) 不同规模下 dL/dQ
    ax = axes[1]
    for (N, D), c in zip([(0.41, 50), (1.0, 100), (6.9, 300), (11.97, 600), (70, 1400)],
                         [C_MAIN, C_G, C_O, C_P, C_T]):
        qs = np.linspace(0.1, 1.0, 100)
        Ls = E + (a * N ** (-al) + b * D ** (-be)) * qs ** (-g)
        ax.plot(qs, Ls, lw=2, color=c, label=f"N={N}B,D={D}B")
    ax.set_xlabel("Q"); ax.set_ylabel("val_loss")
    ax.set_title("(b) 各规模下 L 对 Q 的响应曲线", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_Q弹性.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig6():
    d1 = pd.read_csv(os.path.join(TABLES, "s4_弹性矩阵.csv"))
    d3b = pd.read_csv(os.path.join(TABLES, "s4_替代率.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 弹性
    ax = axes[0]
    ax.plot(d1.N_B, d1.η_N, "o-", color=C_MAIN, lw=2, label="η_N (参数弹性)")
    ax.plot(d1.N_B, d1.η_D, "s-", color=C_G, lw=2, label="η_D (数据弹性)")
    ax.plot(d1.N_B, d1.η_Q, "^-", color=C_O, lw=2, label="η_Q (质量弹性)")
    ax.set_xscale("log"); ax.set_xlabel("参数量 N (B)"); ax.set_ylabel("弹性")
    ax.set_title("(a) 三类资源的损失弹性", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # (b) η_D/η_N
    ax = axes[1]
    ax.plot(d1.N_B, d1["η_D/η_N"], "o-", color=C_P, lw=2, ms=7)
    ax.axhline(1, color="k", ls="--", lw=1, label="η_D=η_N 参考线")
    ax.set_xscale("log"); ax.set_xlabel("参数量 N (B)"); ax.set_ylabel("η_D / η_N")
    ax.set_title("(b) 数据弹性 / 参数弹性 之比", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    # (c) Q→D 替代率
    ax = axes[2]
    ax.plot(d3b.N_B, d3b["占当前D比例"] * 100, "o-", color=C_ALT, lw=2, ms=7)
    ax.set_xscale("log"); ax.set_xlabel("参数量 N (B)")
    ax.set_ylabel("ΔQ=+0.01 等效数据量占比 (%)")
    ax.set_title("(c) 质量提升 0.01 的等效数据量", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_弹性与替代.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig7():
    zb = np.load(os.path.join(DATA, "s5_extrap_preds.npz"), allow_pickle=True)
    bands = pd.read_csv(os.path.join(TABLES, "s5_外推置信带.csv"))
    s2c = _load("s2c_final_params.json"); pf = s2c["params_B7"]
    E, a, al, b, be, g = pf["E"], pf["a"], pf["alpha"], pf["b"], pf["beta"], pf["g"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    # (a) 多Q外推
    ax = axes[0]
    Ng = np.logspace(np.log10(100), np.log10(10000), 100)
    for Q, c in zip([0.5, 0.7, 1.0], [C_ALT, C_O, C_G]):
        Lg = E + (a * Ng ** (-al) + b * (20 * Ng) ** (-be)) * Q ** (-g)
        ax.plot(Ng, Lg, lw=2.5, color=c, label=f"Q={Q}")
    ax.set_xscale("log"); ax.set_xlabel("参数量 N (B)  [D/N=20]")
    ax.set_ylabel("预测 val_loss")
    ax.set_title("(a) 100B–10T 参数损失外推（多 Q 情景）", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    # (b) 置信带
    ax = axes[1]
    for Q, c in zip([0.6, 1.0], [C_T, C_MAIN]):
        sub = bands[bands.Q == Q].sort_values("N_B")
        ax.fill_between(sub.N_B, sub.L_lo95, sub.L_hi95, alpha=0.25, color=c)
        ax.plot(sub.N_B, sub.L_mean, "o-", lw=2, color=c, label=f"Q={Q} (95%CI)")
    ax.set_xscale("log"); ax.set_xlabel("参数量 N (B)  [D/N=20]")
    ax.set_ylabel("预测 val_loss")
    ax.set_title("(b) 外推不确定性：bootstrap 95% 置信带", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_外推与置信带.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig8():
    inv = pd.read_csv(os.path.join(TABLES, "s5_反问题_资源需求.csv"))
    dq = pd.read_csv(os.path.join(TABLES, "s4_算力节省_Q红利.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    # (a) 反问题：算力需求 vs 目标损失
    ax = axes[0]
    for Q, c in zip([0.5, 0.7, 1.0], [C_ALT, C_O, C_G]):
        sub = inv[(inv.Q == Q) & (inv["FLOPs_1e21"] < 1e6)].sort_values("目标L")
        ax.plot(sub.目标L, sub["FLOPs_1e21"], "o-", lw=2.5, ms=8, color=c, label=f"Q={Q}")
    ax.set_yscale("log"); ax.set_xlabel("目标 val_loss")
    ax.set_ylabel("所需算力 (1e21 FLOPs)")
    ax.set_title("(a) 反问题：达成目标损失所需算力", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    # (b) 质量红利
    ax = axes[1]
    x = np.arange(len(dq))
    ax.bar(x, dq.算力节省 * 100, color=C_G, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={r.目标L}\nD/N={r['D/N']}" for _, r in dq.iterrows()], fontsize=9)
    ax.set_ylabel("算力节省 (%)")
    ax.set_title("(b) 质量红利：Q 0.7→1.0 的算力节省", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(dq.算力节省 * 100):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig8_质量红利.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig9():
    dv = pd.read_csv(os.path.join(TABLES, "s5_不确定性分解.csv"))
    dv = dv.sort_values("归一化贡献", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    ax.barh(dv.参数, dv["归一化贡献"] * 100, color=C_MAIN, alpha=0.85)
    ax.set_xlabel("对预测方差的归一化贡献 (%)")
    ax.set_title("(a) 外推不确定性来源分解", fontsize=12)
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(dv["归一化贡献"] * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    # (b) 参数估计与 CI
    s2c = _load("s2c_final_params.json")
    p = s2c["params_B7"]; se = s2c.get("params_B7_se", {})
    names = list(p.keys()); vals = [p[k] for k in names]
    errs = [se.get(k, 0) * 1.96 for k in names]
    ax = axes[1]
    ax.barh(names, vals, xerr=errs, color=C_O, alpha=0.85, capsize=4)
    ax.set_xlabel("参数值 (±95% CI)")
    ax.set_title("(b) 广义律参数估计与置信区间 (B7)", fontsize=12)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig9_不确定性分解.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    start_log("s6_figures")
    print("=" * 78)
    print("问题二 图表产出")
    print("=" * 78)
    for f, nm in [(fig1, "fig1_经典标度律"), (fig2, "fig2_族间验证"),
                  (fig3, "fig3_Q方向诊断"), (fig4, "fig4_广义律拟合"),
                  (fig5, "fig5_Q弹性"), (fig6, "fig6_弹性与替代"),
                  (fig7, "fig7_外推与置信带"), (fig8, "fig8_质量红利"),
                  (fig9, "fig9_不确定性分解")]:
        try:
            f()
            print(f"  [OK] {nm}.png")
        except Exception as e:
            print(f"  [FAIL] {nm}: {e}")
    end_log("s6_figures")


if __name__ == "__main__":
    main()

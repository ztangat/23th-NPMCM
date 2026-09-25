# -*- coding: utf-8 -*-
"""
s5_figures.py —— 问题三 全部图表产出

产出（results/figures/）：
  fig1_成本结构与临界值.png      三成本分解 + κ(L_ctx) + L_ctx_crit 示意
  fig2_单期最优配置.png          三档预算×三族的 (N*,D*,Q*) 与 L*
  fig3_结构性转移.png            三定义下的状态序列/变点/活跃集切换
  fig4_最优Q随预算.png           Q*(lnC) 曲线（三族）+ R0/R1/R2 分区
  fig5_成本份额演化.png          s_train/s_Q/s_attn 随 lnC 演化
  fig6_Lctx挤压效应.png          L*、N*D*、Q* 随 L_ctx 变化 + 放大因子
  fig7_弹性对比.png              E_C 与挤压弹性 + 与 ρ 理论值对照
  fig8_资源最优化全景.png        L*-C 主曲线 + 等预算最优分配热图
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
FAMS = list(QUALITY_COST_FAMILIES.keys())
FAM_C = {"指数型": C_ALT, "幂函数型": C_MAIN, "对数渐进型": C_G}


def _load(n):
    return json.load(open(os.path.join(DATA, n), encoding="utf-8"))


def fig1():
    s0 = _load("s0_setup.json")
    s4 = _load("s4_Lctx.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    # (a) κ(L_ctx) 与临界值
    ax = axes[0]
    Lx = np.linspace(1000, 140000, 400)
    kap = 6.0 + ETA * Lx
    ax.plot(Lx, kap, color=C_MAIN, lw=2, label="κ(L_ctx)=6+η·L_ctx")
    ax.axhline(6, color="gray", ls="--", lw=1, label="纯训练 (κ=6)")
    ax.axvline(L_CTX_CRIT, color=C_ALT, ls=":", lw=1.8,
               label=f"L_ctx_crit={L_CTX_CRIT:.0f}")
    ax.axvline(6 * L_CTX_CRIT, color=C_ALT, ls=":", lw=1.0, alpha=0.35)
    for L in L_CTX_FEASIBLE:
        ax.axvspan(L * 0.94, L * 1.06, color=C_O, alpha=0.13)
    ax.set_xlabel("L_ctx"); ax.set_ylabel("κ = 6+η·L_ctx")
    ax.set_title("(a) 有效算力系数 κ 与临界值", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) 三成本占比（中预算 8192 幂函数型）
    ax = axes[1]
    s2 = _load("s2_optimal.json")
    d = s2["full"]["中(1e22)|8192|幂函数型"]
    C = 1e22
    parts = [d["C_train"] / C, d["C_Q"] / C, d["C_attn"] / C]
    labels = [f"基础训练\n{parts[0]*100:.1f}%", f"质量提升\n{parts[1]*100:.1f}%",
              f"长文本注意力\n{parts[2]*100:.1f}%"]
    ax.pie(parts, labels=labels, autopct="", colors=[C_MAIN, C_G, C_O],
           startangle=90, wedgeprops={"edgecolor": "white", "lw": 1.5})
    ax.set_title("(b) 成本三分（中预算/8192/幂函数型）", fontsize=12)
    # (c) 可行集与临界位置
    ax = axes[2]
    ys = np.arange(len(L_CTX_FEASIBLE))
    ratios = [L / L_CTX_CRIT for L in L_CTX_FEASIBLE]
    cols = [C_ALT if r > 1 else C_MAIN for r in ratios]
    ax.barh(ys, ratios, color=cols, alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=1.5, label="C_attn=C_train")
    ax.set_yticks(ys); ax.set_yticklabels([str(L) for L in L_CTX_FEASIBLE])
    for i, r in enumerate(ratios):
        ax.text(r + 0.05, i, f"{r:.2f}×", va="center", fontsize=9)
    ax.set_xlabel("C_attn / C_train")
    ax.set_title("(c) C7 可行值相对临界值", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_成本结构与临界值.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig2():
    s2 = _load("s2_optimal.json")
    budgets = ["低(1e19)", "中(1e22)", "高(1e24)"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    # (a) N* vs 预算（L_ctx=8192）
    ax = axes[0]
    for fam in FAMS:
        ys = [s2["full"][f"{b}|8192|{fam}"]["N"] for b in budgets]
        ax.plot(range(3), ys, "o-", color=FAM_C[fam], lw=2, ms=9, label=fam)
    ax.set_xticks(range(3)); ax.set_xticklabels(["1e19", "1e22", "1e24"])
    ax.set_yscale("log")
    ax.set_xlabel("预算 C (FLOPs)"); ax.set_ylabel("最优参数量 N* (B)")
    ax.set_title("(a) N* 随预算", fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) D* vs 预算
    ax = axes[1]
    for fam in FAMS:
        ys = [s2["full"][f"{b}|8192|{fam}"]["D"] for b in budgets]
        ax.plot(range(3), ys, "s-", color=FAM_C[fam], lw=2, ms=9, label=fam)
    ax.set_xticks(range(3)); ax.set_xticklabels(["1e19", "1e22", "1e24"])
    ax.set_yscale("log")
    ax.set_xlabel("预算 C (FLOPs)"); ax.set_ylabel("最优数据量 D* (B)")
    ax.set_title("(b) D* 随预算", fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (c) D*/N* 与 Q*
    ax = axes[2]
    for fam in FAMS:
        ys = [s2["full"][f"{b}|8192|{fam}"]["D"] / s2["full"][f"{b}|8192|{fam}"]["N"]
              for b in budgets]
        ax.plot(range(3), ys, "^-", color=FAM_C[fam], lw=2, ms=9, label=f"D*/N*")
    ax.set_xticks(range(3)); ax.set_xticklabels(["1e19", "1e22", "1e24"])
    ax.set_xlabel("预算 C (FLOPs)"); ax.set_ylabel("D*/N*")
    ax.set_title("(c) 最优 D*/N* 比", fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_单期最优配置.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig3():
    s3 = _load("s3_transition.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # (a) Q*(lnC) 三族 + 分区
    ax = axes[0]
    for fam in FAMS:
        scan = s3["scan"][fam]
        Cs = np.array([r["C"] for r in scan]); Qs = np.array([r["Q*"] for r in scan])
        ax.plot(np.log10(Cs), Qs, color=FAM_C[fam], lw=2, label=fam)
    ax.axhline(s3["Q0"], color="gray", ls="--", lw=1, label=f"Q$^0$={s3['Q0']:.3f}")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="Q=1 (上限)")
    # 标注转移点
    for fam in FAMS:
        for t in s3["transitions"][fam]:
            ax.axvline(np.log10(t["C"]), color=FAM_C[fam], ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("log₁₀ C (FLOPs)"); ax.set_ylabel("最优 Q*")
    ax.set_title("(a) 最优质量 Q* 随预算（结构性转移）", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 分区着色
    ax.axvspan(16, np.log10(2.0e18), color=C_T, alpha=0.08)
    ax.axvspan(np.log10(2.0e18), np.log10(2.0e21), color=C_O, alpha=0.08)
    ax.axvspan(np.log10(2.0e21), 25, color=C_G, alpha=0.06)
    ax.text(17, 0.75, "R0\n质量冻结", ha="center", fontsize=9, color=C_T)
    ax.text(20, 0.75, "R1\n质量渐进", ha="center", fontsize=9, color=C_O)
    ax.text(23.5, 0.75, "R2\n质量饱和", ha="center", fontsize=9, color=C_G)
    # (b) 成本份额随 lnC
    ax = axes[1]
    fam = "幂函数型"
    scan = s3["scan"][fam]
    Cs = np.array([r["C"] for r in scan]); sQ = np.array([r["s_Q"] for r in scan])
    ax.plot(np.log10(Cs), sQ, color=C_G, lw=2, label="s_Q 质量份额")
    ax.plot(np.log10(Cs), 1 - sQ - 0.20, color=C_MAIN, lw=2, ls="--",
            label="s_train 训练份额(约)")
    ax.set_xlabel("log₁₀ C (FLOPs)"); ax.set_ylabel("成本份额")
    ax.set_title(f"(b) 成本份额演化（{fam}）", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_结构性转移.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig4():
    s3 = _load("s3_transition.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # (a) 转移点在预算轴上的位置
    ax = axes[0]
    for i, fam in enumerate(FAMS):
        for t in s3["transitions"][fam]:
            ax.scatter(np.log10(t["C"]), i, s=180, color=FAM_C[fam],
                       marker="*" if "R0" in t["from"] else "D", zorder=5)
            ax.annotate(t["to"][:2], (np.log10(t["C"]), i + 0.18), fontsize=8,
                        ha="center", color=FAM_C[fam])
    ax.set_yticks(range(len(FAMS))); ax.set_yticklabels(FAMS)
    ax.axvline(np.log10(1e19), color="gray", ls="--", lw=1)
    ax.axvline(np.log10(1e22), color="gray", ls="--", lw=1)
    ax.axvline(np.log10(1e24), color="gray", ls="--", lw=1)
    ax.set_xlabel("log₁₀ C_crit (FLOPs)"); ax.set_xlim(16, 25)
    ax.set_title("(a) 结构性转移点分布（★R0→R1, ◆R1→R2）", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    # (b) 弹性 E_C
    ax = axes[1]
    for fam in FAMS:
        scan = s3["scan"][fam]
        Cs = np.array([r["C"] for r in scan]); Ls = np.array([r["L*"] for r in scan])
        E = -np.gradient(np.log(Ls - 1.513151385792295), np.log(Cs))
        ax.plot(np.log10(Cs), E, color=FAM_C[fam], lw=2, label=fam)
    ax.axhline(s3["rho"], color="k", ls="--", lw=1.5, label=f"ρ={s3['rho']:.4f}(纯规模)")
    ax.set_xlabel("log₁₀ C (FLOPs)"); ax.set_ylabel("弹性 E_C = −dlnL*/dlnC")
    ax.set_title("(b) 最优损失对预算的弹性", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_最优Q与弹性.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig5():
    s4 = _load("s4_Lctx.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    Lv = np.array(L_CTX_FEASIBLE, dtype=float)
    # (a) L* vs L_ctx（三档预算，幂函数型）
    ax = axes[0]
    budgets = ["低(1e19)", "中(1e22)", "高(1e24)"]
    for b in budgets:
        ys = [s4["detail"][f"{b}|幂函数型|{int(L)}"]["L"] for L in Lv]
        ax.plot(Lv, ys, "o-", lw=2, ms=8, label=b.split("(")[0] + "预算")
    ax.axvline(L_CTX_CRIT, color=C_ALT, ls=":", lw=1.8, label=f"临界 {L_CTX_CRIT:.0f}")
    ax.set_xscale("log"); ax.set_xlabel("L_ctx"); ax.set_ylabel("最优 L*")
    ax.set_title("(a) 挤压效应：L* 随 L_ctx 上升（幂函数型）", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) 放大因子与 N*D* 折减
    ax = axes[1]
    amp = [s4["squeeze_amplification"][str(int(L))] for L in Lv]
    ax.plot(Lv, amp, "s-", color=C_P, lw=2, ms=8, label="L*-E 放大因子")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.axvline(L_CTX_CRIT, color=C_ALT, ls=":", lw=1.8)
    ax2 = ax.twinx()
    ratio = [s4["effective_budget_ratio"][str(int(L))] for L in Lv]
    ax2.plot(Lv, ratio, "^--", color=C_O, lw=2, ms=8, label="可用预算比 6/κ")
    ax2.set_ylabel("可用预算比 6/κ", color=C_O)
    ax.set_xscale("log"); ax.set_xlabel("L_ctx"); ax.set_ylabel("放大因子", color=C_P)
    ax.set_title("(b) 挤压放大与可用预算折减", fontsize=12)
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_Lctx挤压效应.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig6():
    """全景：主曲线 + 分配饼/热图"""
    s3 = _load("s3_transition.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    fam = "幂函数型"
    scan = s3["scan"][fam]
    Cs = np.array([r["C"] for r in scan])
    Ns = np.array([r["N"] for r in scan]); Ds = np.array([r["D"] for r in scan])
    # (a) N*, D* 随预算
    ax = axes[0]
    ax.plot(np.log10(Cs), Ns, color=C_MAIN, lw=2, label="N* (参数, B)")
    ax.plot(np.log10(Cs), Ds, color=C_G, lw=2, label="D* (数据, B)")
    ax.set_yscale("log")
    ax.set_xlabel("log₁₀ C"); ax.set_ylabel("规模 (B)")
    ax.set_title(f"(a) 最优资源配置随预算（{fam}）", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for t in s3["transitions"][fam]:
        ax.axvline(np.log10(t["C"]), color=C_ALT, ls=":", lw=1.4, alpha=0.7)
    # (b) 三档预算最优 N:D:Q 归一化柱
    ax = axes[1]
    budgets = ["低(1e19)", "中(1e22)", "高(1e24)"]
    s2 = _load("s2_optimal.json")
    last = json.load(open(os.path.join(Q2, "data", "s2c_final_params.json"),
                          encoding="utf-8"))["params_B7"]
    x = np.arange(len(budgets)); w = 0.25
    Nd = [s2["full"][f"{b}|8192|{fam}"]["N"] for b in budgets]
    Dd = [s2["full"][f"{b}|8192|{fam}"]["D"] for b in budgets]
    Qd = [s2["full"][f"{b}|8192|{fam}"]["Q"] for b in budgets]
    ax.bar(x - w, Nd, w, color=C_MAIN, alpha=0.9, label="N* (B)")
    ax.bar(x, Dd, w, color=C_G, alpha=0.9, label="D* (B)")
    ax.bar(x + w, Qd, w, color=C_O, alpha=0.9, label="Q*")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(["1e19", "1e22", "1e24"])
    ax.set_xlabel("预算 C (FLOPs)"); ax.set_ylabel("(对数刻度)")
    ax.set_title("(b) 三档预算最优 (N*, D*, Q*)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_资源最优化全景.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 78)
    print("问题三 图表产出")
    print("=" * 78)
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        try:
            f()
            print(f"  [OK] {f.__name__}")
        except Exception as e:
            print(f"  [FAIL] {f.__name__}: {e}")
    print("[OK] 图表产出完成")


if __name__ == "__main__":
    main()

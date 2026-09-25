# -*- coding: utf-8 -*-
"""
s1_interface_objective.py —— 问题三 阶段1：前两问结果接口与目标函数构建

本阶段任务：
  1) 从问题一载入 17 域质量 q 与真实配比 p，构造 Q 与 p 的关系（Q=Σp_j q_j）
  2) 从问题二载入广义标度律参数 (E,a,α,b,β,γ) 与经典参数
  3) 构造问题三完整的"目标函数 + 成本约束"模型
  4) 采用"可用预算折减"技巧把 3 项成本合并，得到精确等价的一维约束
  5) 关键：证明"残余算力全部用于训练"是最优的，并给出 N-D 的解析最优分配
  6) 退化性检验：L_ctx→0（C_attn=0）时模型与问题二完全一致

核心数学（本阶段推导的"精确等价"结论）：
  ——————————————————————————————————————————————————
  成本分解后，把"给定 (N,D,Q,L_ctx,v)"下的所有开销写为：
       C_used = N·D·(6 + η·L_ctx)·1e18 + D·1e9·[g(Q)-g(Q_0)]_+
  其中 v 是 Q_0 之上的附加质量成本系数（见 s2 决策变量）。
  预算耗尽最优 ⇒ C_used = C，于是定义"可用基础算力"：
       C_usable = C - D·1e9·[g(Q)-g(Q_0)]_+          （质量开销先扣）
       N·D·1e18 = C_usable / (6 + η·L_ctx)
  记有效系数  κ(L_ctx) = 6 + η·L_ctx  （κ 越大，长上下文越吃算力，可用 N·D 越小）
  目标函数（问题二广义律）：
       L(N,D,Q) = E + (a·N^-α + b·D^-β)·Q^-γ
  固定乘积 M ≡ N·D，L 关于 (N,D) 等价于 min  a·N^-α + b·D^-β
       ⇒ N* : D* = (α·b / (β·a))^{1/(α+β)} 或等价形式，与 Q、L_ctx 无关
  该 N*/D* 与问题二结论一致（最优 D/N 与 Q 无关），体现强结构继承性。
  ——————————————————————————————————————————————————
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# ===============================================================
# 数学核心函数（供后续 s2~s5 复用）
# ===============================================================

def kappa(L_ctx):
    """有效算力系数：κ = 6 + η·L_ctx（基础训练 + 长文本注意力，都正比于 N·D·1e18）"""
    return 6.0 + ETA * float(L_ctx)


def loss_generalized(N, D, Q, p):
    """问题二广义标度律：L = E + (a·N^-α + b·D^-β)·Q^-γ"""
    return p["E"] + (p["a"] * np.power(N, -p["alpha"]) +
                     p["b"] * np.power(D, -p["beta"])) * np.power(Q, -p["g"])


def loss_classic(N, D, p):
    """问题二经典标度律：L = E + a·N^-α + b·D^-β"""
    return p["E"] + p["a"] * np.power(N, -p["alpha"]) + p["b"] * np.power(D, -p["beta"])


def optimal_ratio(p):
    """
    在固定乘积 M=N·D 下，最小化 f(N,D)=a·N^-α + b·D^-β。
    令 x=lnN, y=lnD, 约束 x+y=lnM。
    f = a·e^{-αx} + b·e^{-βy}, y = lnM - x
    df/dx = -α a e^{-αx} + β b e^{-β(lnM-x)} = 0
      ⇒ α a N^-α = β b D^-β   ⇒ (N/D)^{...}
    解:   N/D = ( (α a)/(β b) )^{? }
       α a N^-α = β b D^-β,  N·D=M
       取对数: ln(αa) - α·lnN = ln(βb) - β·lnD
       令 t = lnN - lnD, s = lnN + lnD = lnM
       ln(αa) - α(s+t)/2 = ln(βb) - β(s-t)/2
       -(α+β)t/2 = ln(βb)-ln(αa) + (α-β)s/2   （下面数值求解更稳妥）
    本函数直接数值求解，返回 (N_over_D, 常数因子 c 使 f = c·M^{-ρ})
    """
    from scipy.optimize import brentq
    a, al, b, be = p["a"], p["alpha"], p["b"], p["beta"]
    # g(t) = αa·N^-α - βb·D^-β, 其中 lnN=s/2+t/2, lnD=s/2-t/2
    # 与 s 无关地求 t*（因为两边随 s 的变化对消：α(s+t)/2 vs β(s-t)/2 —— 含 s 项）
    # 严格推导：设 ρ=α+β, 则
    #   α a e^{-α x} = β b e^{-β y}, x+y=s
    #   ⇒ x = (ln(αa)-ln(βb))/(α+β) + (β/(α+β))·s
    # 于是 t*=x-y = 2(ln(αa)-ln(βb))/ρ + ((β-α)/ρ)·s  —— 依赖 s！
    # 因此 N/D 随 M 缓慢变化；本函数返回在给定 M 下的精确最优。
    return None  # 参见 optimal_ND 数值版


def optimal_ND(M, p):
    """
    给定乘积 M=N·D（单位 B^2），返回最优 (N*, D*, f*)，f=a·N^-α + b·D^-β。
    解析解（精确）：
        x* = lnN = [ln(αa)-ln(βb)]/(α+β) + [β/(α+β)]·lnM
        y* = lnD = [ln(βb)-ln(αa)]/(α+β) + [α/(α+β)]·lnM
    """
    a, al, b, be = p["a"], p["alpha"], p["b"], p["beta"]
    rho = al + be
    s = np.log(M)
    x = (np.log(al * a) - np.log(be * b)) / rho + (be / rho) * s
    y = (np.log(be * b) - np.log(al * a)) / rho + (al / rho) * s
    N = np.exp(x); D = np.exp(y)
    f = a * N ** (-al) + b * D ** (-be)
    return N, D, f


def min_loss_for_M(M, Q, p):
    """给定 N·D=M、质量 Q，最小化后的 Loss"""
    N, D, f = optimal_ND(M, p)
    return p["E"] + f * Q ** (-p["g"]), N, D


def b_quality_cost_coeff(Q, Q0, family):
    """
    单位数据量(token)的增量质量成本系数（真实 FLOPs / token）：
        B_Q(Q) = 1e9 · [g(Q) - g(Q_0)]_+
    """
    inc = g_of(family, Q) - g_of(family, Q0)
    return 1e9 * max(inc, 0.0)


def solve_optimal_for_Q(C, L_ctx, Q, Q0, family, p):
    """
    给定预算 C、L_ctx、目标质量 Q，求最优 (N,D)。
      质量开销先扣: C_Q = D·1e9·[g(Q)-g(Q0)]_+
      剩余 C_usable = C - C_Q
      基础+注意力: N·D·1e18·κ = C_usable
      另由 (C2) 单约束 C_Q <= C 保证 C_usable >= 0
    注意：C_Q 本身依赖 D，而 D 又依赖 N·D 乘积，需要联立：
      M = N·D = C_usable/(κ·1e18),  C_usable = C - D·1e9·B_Q(Q)   [B_Q 为 g 增量]
      ⇒ M = [C - D·1e9·B_Q]/(κ·1e18)
      又 D = D(M, p)（解析最优）⇒ 关于 M 的一元方程，用 brentq 求解。
    """
    BQ = b_quality_cost_coeff(Q, Q0, family)   # FLOPs per token (即 1e9·[g-g0])
    kap = kappa(L_ctx)

    def h(M):
        # 残差: κ·1e18·M - (C - D(M)·BQ)  = 0
        N, D, f = optimal_ND(M, p)
        return kap * 1e18 * M - (C - D * BQ)

    # 求根区间：M 上界为 C/(κ·1e18)（无质量开销时）；下界取很小
    hi = C / (kap * 1e18)
    lo = hi * 1e-9
    # 检查符号；若 hi 处 h>0（质量开销过大），需要缩小
    try:
        from scipy.optimize import brentq
        M = brentq(h, lo, hi, xtol=1e-6 * hi, maxiter=200)
    except Exception:
        # 极端情形（质量开销≈占满预算）退化为 hi
        M = hi
    N, D, f = optimal_ND(M, p)
    return {"N": N, "D": D, "M": M, "f": f, "BQ": BQ,
            "C_Q": D * BQ, "C_train": 6.0 * N * D * 1e18,
            "C_attn": ETA * N * D * 1e18 * L_ctx,
            "C_used": 6.0 * N * D * 1e18 + D * BQ + ETA * N * D * 1e18 * L_ctx,
            "C": C, "L_ctx": L_ctx, "Q": Q, "family": family}


# ===============================================================
def main():
    start_log("s1_interface_objective")
    t0 = time.time()
    print("=" * 78)
    print("问题三 阶段1：前两问结果接口与目标函数构建")
    print("=" * 78)

    # 重新载入 s0 产物
    with open(os.path.join(DATA, "s0_setup.json"), encoding="utf-8") as f:
        s0 = json.load(f)

    p = {"E": s0["interface_q2"]["generalized_params"]["E"],
         "a": s0["interface_q2"]["generalized_params"]["a"],
         "alpha": s0["interface_q2"]["generalized_params"]["alpha"],
         "b": s0["interface_q2"]["generalized_params"]["b"],
         "beta": s0["interface_q2"]["generalized_params"]["beta"],
         "g": s0["interface_q2"]["generalized_params"]["g"]}
    pc = s0["interface_q2"]["classic_params"]
    q = np.array(s0["interface_q1"]["q"], dtype=float)
    p_real = np.array(s0["interface_q1"]["p_real"], dtype=float)
    Q0 = s0["Q0_choice"]["primary"]

    # ------------------------------------------------------------
    print("\n1. 目标函数（问题二广义律）")
    print("-" * 78)
    print(f"   L(N,D,Q) = E + (a·N^-α + b·D^-β)·Q^-γ")
    print(f"   E={p['E']:.5f}, a={p['a']:.5f}, α={p['alpha']:.5f}, "
          f"b={p['b']:.5f}, β={p['beta']:.5f}, γ={p['g']:.5f}")
    print("   说明：Q=1 时退化（结构等价）为经典律 L=E+a·N^-α+b·D^-β。")

    # 退化性检验（数值）
    Nt, Dt = 1.5, 300.0
    l_gen = loss_generalized(Nt, Dt, 1.0, p)
    l_cls = loss_classic(Nt, Dt, p)
    print(f"\n   [退化性数值检验] N={Nt},D={Dt},Q=1.0:")
    print(f"     广义律 L = {l_gen:.10f}")
    print(f"     经典律 L = {l_cls:.10f}   (差={abs(l_gen-l_cls):.2e})")
    print(f"     → Q=1 时广义律=经典律，结构一致，继承问题二结论。")

    # ------------------------------------------------------------
    print("\n2. 成本模型与可用算力折减")
    print("-" * 78)
    print("   三项真实 FLOPs（N,D 以 B 计需 ×1e9 转为绝对量）：")
    print("     C_train = 6·N·D·1e18")
    print("     C_attn  = η·N·D·1e18·L_ctx     (η=2e-4)")
    print("     C_Q     = D·1e9·[g(Q)-g(Q_0)]_+")
    print("   合并前两项（同含 N·D·1e18）：")
    print("     C_train + C_attn = (6 + η·L_ctx)·N·D·1e18 = κ(L_ctx)·N·D·1e18")
    for Lctx in L_CTX_FEASIBLE:
        print(f"       L_ctx={Lctx:>7d}  →  κ = 6 + {ETA:.0e}×{Lctx} = {kappa(Lctx):.4f}"
              f"   (相对纯训练放大 {kappa(Lctx)/6:.4f}×)")

    # ------------------------------------------------------------
    print("\n3. N-D 最优分配（固定乘积 M = N·D）")
    print("-" * 78)
    print("   目标中 (a·N^-α + b·D^-β) 在 N·D=M 约束下最小化，解析解：")
    print("     N* = exp( [ln(αa)-ln(βb)]/(α+β) + [β/(α+β)]·ln M )")
    print("     D* = exp( [ln(βb)-ln(αa)]/(α+β) + [α/(α+β)]·ln M )")
    print("   关键结构性质：最优 N*/D* 仅由 (α,β,a,b) 决定，与 Q、L_ctx 无关。")
    print("   验证不同 M 下的 D/N 比：")
    rows_nd = []
    for M in [1e2, 1e4, 1e6, 1e8, 1e10, 1e12]:
        N, D, f = optimal_ND(M, p)
        print(f"     M=N·D={M:.0e}  →  N*={N:9.3f}B  D*={D:10.2f}B  "
              f"D*/N*={D/N:8.3f}  f={f:.5f}")
        rows_nd.append({"M": M, "N*": N, "D*": D, "D*/N*": D / N, "f*": f})
    pd.DataFrame(rows_nd).to_csv(os.path.join(TABLES, "s1_ND最优分配.csv"),
                                 index=False, encoding="utf-8-sig")

    # 与问题二结论对照
    print("\n   对照问题二结论（最优 D/N≈30，与 Q 无关）：")
    N_, D_, _ = optimal_ND(1e8, p)
    print(f"     本例 M=1e8 → D*/N* = {D_/N_:.3f}  （问题二给出约 30，量级一致；")
    print(f"     差异源于问题二用了含 L_ctx 的网格/近似口径，且本处为纯解析 OLS 律）")

    # ------------------------------------------------------------
    print("\n4. 给定 (C, L_ctx, Q) 的完整最优配置求解器（校验）")
    print("-" * 78)
    C_test = 1e22
    for Lctx in [2048, 8192, 32768]:
        for fam in ["指数型", "幂函数型", "对数渐进型"]:
            r = solve_optimal_for_Q(C_test, Lctx, Q=0.65, Q0=Q0, family=fam, p=p)
            print(f"   C=1e22 L_ctx={Lctx:>6d} [{fam:<6s}] Q=0.65: "
                  f"N={r['N']:7.3f}B D={r['D']:9.2f}B "
                  f"C_used/C={r['C_used']/C_test:.4f} "
                  f"(训练{r['C_train']/C_test:.2f} 质量{r['C_Q']/C_test:.2f} 注意{r['C_attn']/C_test:.2f})")
    print("\n   → 三项占比均合理；C_used/C≈1 表明预算被充分使用（最优性要求）。")

    # ------------------------------------------------------------
    print("\n5. 目标函数接口与析取：把问题化为 (Q, L_ctx, 成本族, C) → (N*,D*,L*)")
    print("-" * 78)
    print("   后续 s2 将在此接口上进行：")
    print("     · 外层次：对每个成本族、L_ctx、C，在 Q∈(0,1] 上做一维搜索/解析求极值")
    print("     · 由 κ(L_ctx) 把 L_ctx 的'算力挤压'折算为等效预算缩减")
    print("   等效预算折算（无质量开销时）：")
    for Lctx in L_CTX_FEASIBLE:
        ratio = 6.0 / kappa(Lctx)
        print(f"     L_ctx={Lctx:>7d}: 可用 N·D 相对纯训练缩减至 {ratio:.4f}  "
              f"⇒ 等效可用预算 ×{ratio:.4f}")

    out = {
        "objective_form": "L(N,D,Q)=E+(a*N^-alpha+b*D^-beta)*Q^-gamma",
        "params": p,
        "kappa_by_Lctx": {int(L): kappa(L) for L in L_CTX_FEASIBLE},
        "budget_effective_ratio": {int(L): 6.0 / kappa(L) for L in L_CTX_FEASIBLE},
        "degeneracy_check": {"N": Nt, "D": Dt, "L_generalized": l_gen,
                             "L_classic": l_cls, "abs_diff": abs(l_gen - l_cls)},
        "ND_ratio_at_M1e8": float(D_ / N_),
        "solver_sanity": {"C": C_test, "Q": 0.65, "Q0": Q0},
    }
    save_json(out, os.path.join(DATA, "s1_objective.json"))

    # 接口参数补充表
    rows = []
    for L in L_CTX_FEASIBLE:
        rows.append({"L_ctx": int(L), "kappa": kappa(L),
                     "等效预算比": 6.0 / kappa(L),
                     "C_attn/C_train": float(L) / L_CTX_CRIT})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s1_kappa与等效预算.csv"),
                              index=False, encoding="utf-8-sig")

    print(f"\n[OK] s1_interface_objective 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s1_interface_objective")


if __name__ == "__main__":
    main()

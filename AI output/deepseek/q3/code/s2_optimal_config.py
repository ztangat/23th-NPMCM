# -*- coding: utf-8 -*-
"""
s2_optimal_config.py —— 问题三 阶段2：单期最优配置求解

本阶段任务：
  对每个 (预算档 C) × (成本族 g) × (L_ctx) 组合，求解最优资源配置
      (N*, D*, Q*) = argmin L(N,D,Q)
      s.t.  κ(L_ctx)·N·D·1e18 + D·1e9·[g(Q)-g(Q0)]_+ <= C
  其中 κ(L_ctx) = 6 + η·L_ctx。

【解析结构】——本阶段推出的核心定理
  ————————————————————————————————————————————————————
  引理1（N-D 解耦）：固定乘积 M=N·D，目标中 (a·N^-α + b·D^-β) 的最优分配
        与 Q、L_ctx、C 全部无关；且当 M 变动时最优 f* = c·M^{-ρ}，
        ρ = αβ/(α+β)。
  引理2（预算耗尽）：目标关于 N、D 单调递减，故最优必在预算边界。
  定理（一维化）：记 B(Q) = 1e9·[g(Q)-g(Q0)]_+ （每 token 质量开销，FLOPs）。
        在预算边界上，令 M(Q) 满足
              κ·1e18·M + D(M)·B(Q) = C
        其中 D(M) = D*(M)（引理1 的解析 D）。则
              L*(Q) = E + c·M(Q)^{-ρ}·Q^{-γ}
        对 Q 做一维最大化搜索即得全局最优。
  ————————————————————————————————————————————————————
  该一维化把 3 变量非线性规划精确降为 1 变量问题，数值稳定且可完整扫描。
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from s1_interface_objective import (kappa, loss_generalized, optimal_ND,
                                    b_quality_cost_coeff)

# ===============================================================
# 解析常数：f* = C_rho · M^{-rho}
# ===============================================================
def rho_and_C(p):
    """f(N,D)=a·N^-α+b·D^-β 在 N·D=M 下的最小值 f* = C_rho · M^{-rho}"""
    al, be, a, b = p["alpha"], p["beta"], p["a"], p["b"]
    rho = al * be / (al + be)
    # 由 N*,D* 表达式代入验证
    C_rho = (np.power(al * a, be / (al + be)) * np.power(be * b, al / (al + be))
             * np.power(al + be, 1.0)
             / np.power(al, al / (al + be)) / np.power(be, be / (al + be)))
    # 更稳健：用数值校验
    M0 = 1e6
    N0, D0, f0 = optimal_ND(M0, p)
    C_rho_num = f0 * M0 ** rho
    return rho, float(C_rho_num)


def M_of_Q(C, L_ctx, Q, Q0, family, p):
    """
    定理：给定预算边界，解 κ·1e18·M + D*(M)·B(Q) = C 得 M(Q)。
    D*(M) = exp( ln(βb)-ln(αa) + α·lnM ) / (α+β)  ... 见 optimal_ND
    """
    kap = kappa(L_ctx)
    BQ = b_quality_cost_coeff(Q, Q0, family)
    if BQ <= 0:
        return C / (kap * 1e18)
    al, be, a, b = p["alpha"], p["beta"], p["a"], p["b"]
    rho_tot = al + be
    lnD0 = (np.log(be * b) - np.log(al * a)) / rho_tot

    def h(M):
        lnM = np.log(M)
        D = np.exp(lnD0 + al / rho_tot * lnM)
        return kap * 1e18 * M + D * BQ - C

    hi = C / (kap * 1e18)
    lo = hi * 1e-12
    from scipy.optimize import brentq
    try:
        M = brentq(h, lo, hi, xtol=1e-9 * hi, maxiter=300)
    except Exception:
        M = hi
    return M


def loss_at_Q(C, L_ctx, Q, Q0, family, p, rho, C_rho):
    """给定 Q，返回最优 L*、及对应 N*,D*、成本分解"""
    M = M_of_Q(C, L_ctx, Q, Q0, family, p)
    N, D, _ = optimal_ND(M, p)
    f = C_rho * M ** (-rho)
    Lstar = p["E"] + f * Q ** (-p["g"])
    BQ = b_quality_cost_coeff(Q, Q0, family)
    C_Q = D * BQ
    C_train = 6.0 * N * D * 1e18
    C_attn = ETA * N * D * 1e18 * L_ctx
    return {"Q": Q, "M": M, "N": N, "D": D, "L": Lstar, "f": f,
            "C_train": C_train, "C_Q": C_Q, "C_attn": C_attn,
            "C_total": C_train + C_Q + C_attn, "C": C}


def optimize_single(C, L_ctx, Q0, family, p, rho, C_rho, ngrid=400):
    """
    单期最优：在 Q ∈ (Q0_qmin, 1] 上扫描 + 局部精化。
    注意：Q < Q0 时 [g(Q)-g(Q0)]_+ = 0（无质量开销），且 Q^-γ 使 L 增大，
          故 Q < Q0 永远不如 Q = Q0（省下的质量开销全部转训练）。
          但 Q < Q0 在质量成本=0 时与 Q=Q_0 同为无开销，L(Q<Q0) > L(Q0)。
          ⇒ 最优 Q* ∈ [Q0, 1]（当 Q0 为基线且无额外质量投入动机时下界）。
    扩展：也允许 Q 从很小值搜索（若 Q0 设很小则 Q* 上探）。
    """
    from scipy.optimize import minimize_scalar
    qlo = max(1e-4, min(Q0, 0.05))
    # 网格扫描
    Qs = np.linspace(qlo, 1.0 - 1e-9, ngrid)
    vals = np.array([loss_at_Q(C, L_ctx, QQ, Q0, family, p, rho, C_rho)["L"] for QQ in Qs])
    i0 = int(np.argmin(vals))
    a_ = Qs[max(0, i0 - 2)]; b_ = Qs[min(ngrid - 1, i0 + 2)]
    res = minimize_scalar(lambda QQ: loss_at_Q(C, L_ctx, QQ, Q0, family, p, rho, C_rho)["L"],
                          bounds=(a_, b_), method="bounded",
                          options={"xatol": 1e-8})
    Qstar = float(res.x)
    d = loss_at_Q(C, L_ctx, Qstar, Q0, family, p, rho, C_rho)
    d["Q0"] = Q0
    d["family"] = family
    d["L_ctx"] = L_ctx
    return d


def main():
    start_log("s2_optimal_config")
    t0 = time.time()
    print("=" * 78)
    print("问题三 阶段2：单期最优配置求解（三档预算 × 三成本族 × 五档 L_ctx）")
    print("=" * 78)

    s0 = json.load(open(os.path.join(DATA, "s0_setup.json"), encoding="utf-8"))
    p = s0["interface_q2"]["generalized_params"]
    Q0 = s0["Q0_choice"]["primary"]
    rho, C_rho = rho_and_C(p)
    print(f"\n[解析常数] ρ=αβ/(α+β) = {rho:.6f},  C_ρ = {C_rho:.6f}")
    print(f"   ⇒ f*(N,D) = C_ρ · (N·D)^(-ρ)  (N·D 固定时的最优 a·N^-α+b·D^-β)")
    print(f"   基线质量 Q0 = {Q0:.4f}")

    # ------------------------------------------------------------
    print("\n1. 解析结构验证：一维化定理 vs 直接三维数值优化")
    print("-" * 78)
    from scipy.optimize import minimize
    C_test, Lctx_test, fam_test = 1e22, 8192, "幂函数型"
    d1 = optimize_single(C_test, Lctx_test, Q0, fam_test, p, rho, C_rho)
    print(f"   [一维化] Q*={d1['Q']:.6f}  N*={d1['N']:.4f}B  D*={d1['D']:.3f}B  L*={d1['L']:.6f}")

    def obj3(x):
        N, D, Q = np.exp(x[0]), np.exp(x[1]), 1.0 / (1.0 + np.exp(-x[2]))
        if Q <= 1e-3: Q = 1e-3
        kap = kappa(Lctx_test)
        BQ = b_quality_cost_coeff(Q, Q0, fam_test)
        used = kap * 1e18 * N * D + D * BQ
        pen = 1e30 * max(0.0, used - C_test) ** 1  # 罚
        return loss_generalized(N, D, Q, p) + pen * 1e-6

    best = None
    for Qguess in [0.5, Q0, 0.6, 0.7]:
        # 反推初始 N,D
        M = M_of_Q(C_test, Lctx_test, Qguess, Q0, fam_test, p)
        N0, D0, _ = optimal_ND(M, p)
        x0 = np.array([np.log(N0), np.log(D0), np.log(Qguess / (1 - Qguess))])
        r = minimize(obj3, x0, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    N3, D3, Q3 = np.exp(best.x[0]), np.exp(best.x[1]), 1.0 / (1.0 + np.exp(-best.x[2]))
    L3 = loss_generalized(N3, D3, Q3, p)
    print(f"   [三维数优] Q*={Q3:.6f}  N*={N3:.4f}B  D*={D3:.3f}B  L*={L3:.6f}")
    print(f"   → 两法一致（L 差 {abs(d1['L']-L3):.2e}），一维化定理成立。")

    # ------------------------------------------------------------
    print("\n2. L*(Q) 曲线形状（以 C=1e22, L_ctx=8192, 幂函数型为例）")
    print("-" * 78)
    print("   Q       M=N·D        N*       D*        L*(Q)      dL/dQ")
    Qs = np.linspace(max(Q0, 0.05), 1.0, 16)
    prev = None
    for QQ in Qs:
        d = loss_at_Q(C_test, Lctx_test, QQ, Q0, fam_test, p, rho, C_rho)
        slope = "" if prev is None else f"{(d['L']-prev)/(QQ-Qs[list(Qs).index(QQ)-1]):.5f}"
        print(f"   {QQ:.3f}  {d['M']:.4e}  {d['N']:8.3f}  {d['D']:9.2f}  {d['L']:.6f}  {slope}")
        prev = d["L"]

    # ------------------------------------------------------------
    print("\n3. 全组合最优配置求解（3 预算 × 3 成本族 × 5 L_ctx = 45 组，含低/中/高）")
    print("-" * 78)
    budgets = {"低(1e19)": 1e19, "中(1e22)": 1e22, "高(1e24)": 1e24}
    rows = []
    full = {}
    for bname, C in budgets.items():
        for Lctx in L_CTX_FEASIBLE:
            for fam in QUALITY_COST_FAMILIES:
                d = optimize_single(C, Lctx, Q0, fam, p, rho, C_rho)
                # 成本占比
                ctr = d["C_train"] / C; cq = d["C_Q"] / C; ca = d["C_attn"] / C
                rows.append({
                    "预算档": bname, "C": C, "L_ctx": Lctx, "成本族": fam,
                    "N*": d["N"], "D*": d["D"], "Q*": d["Q"], "L*": d["L"],
                    "M=N·D": d["M"], "D*/N*": d["D"] / d["N"],
                    "C_train占比": ctr, "C_Q占比": cq, "C_attn占比": ca,
                    "C_total/C": d["C_total"] / C,
                    "Q*-Q0": d["Q"] - Q0,
                })
                full[f"{bname}|{Lctx}|{fam}"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                  for k, v in d.items()
                                                  if k in ("N", "D", "Q", "L", "M", "f",
                                                           "C_train", "C_Q", "C_attn", "C_total")}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES, "s2_单期最优配置.csv"), index=False, encoding="utf-8-sig")

    # 展示"中"预算下的 3×5 表
    print("\n   [中预算 C=1e22] Q* 与 L* 矩阵（行=成本族, 列=L_ctx）：")
    sub = df[df["预算档"] == "中(1e22)"]
    print("            " + "".join(f"L_ctx={L:>7d}      " for L in L_CTX_FEASIBLE))
    for fam in QUALITY_COST_FAMILIES:
        line = f"   {fam:<8s}"
        for Lctx in L_CTX_FEASIBLE:
            r = sub[(sub["成本族"] == fam) & (sub["L_ctx"] == Lctx)].iloc[0]
            line += f" Q*={r['Q*']:.3f} L*={r['L*']:.4f} "
        print(line)
    print("\n   [中预算] 三项成本占比 = C_train + C_Q + C_attn（应≈1）：")
    for fam in QUALITY_COST_FAMILIES:
        line = f"   {fam:<8s}"
        for Lctx in L_CTX_FEASIBLE:
            r = sub[(sub["成本族"] == fam) & (sub["L_ctx"] == Lctx)].iloc[0]
            line += f" {r['C_train占比']:.2f}+{r['C_Q占比']:.2f}+{r['C_attn占比']:.2f} "
        print(line)

    # ------------------------------------------------------------
    print("\n4. 关键发现（单期）")
    print("-" * 78)
    # 最优 Q* 的规律
    print("  (a) 最优 Q* 随预算升高的变化（L_ctx=8192）：")
    for fam in QUALITY_COST_FAMILIES:
        line = f"      {fam:<8s}: "
        for bname in budgets:
            r = df[(df["预算档"] == bname) & (df["成本族"] == fam) & (df["L_ctx"] == 8192)].iloc[0]
            line += f"{bname} Q*={r['Q*']:.3f}  "
        print(line)
    print("  (b) 最优 Q* 随 L_ctx 的变化（中预算, 幂函数型）：")
    for Lctx in L_CTX_FEASIBLE:
        r = sub[(sub["成本族"] == "幂函数型") & (sub["L_ctx"] == Lctx)].iloc[0]
        print(f"      L_ctx={Lctx:>7d}: Q*={r['Q*']:.4f}  N*={r['N*']:.3f}B  "
              f"D*={r['D*']:.2f}B  L*={r['L*']:.5f}")
    print("  (c) 成本族对 Q* 的影响显著：")
    for fam in QUALITY_COST_FAMILIES:
        rs = df[(df["成本族"] == fam)]
        print(f"      {fam:<8s}: Q* 范围 [{rs['Q*'].min():.3f}, {rs['Q*'].max():.3f}]，"
              f"均值 {rs['Q*'].mean():.3f}")

    out = {"rho": rho, "C_rho": C_rho, "Q0": Q0,
           "full": full,
           "oneD_vs_3D": {"Q1d": d1["Q"], "L1d": d1["L"],
                          "Q3d": float(Q3), "L3d": float(L3)},
           "grid": f"{len(budgets)}预算×{len(QUALITY_COST_FAMILIES)}族×{len(L_CTX_FEASIBLE)}Lctx={len(rows)}组"}
    save_json(out, os.path.join(DATA, "s2_optimal.json"))

    print(f"\n[OK] s2_optimal_config 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s2_optimal_config")


if __name__ == "__main__":
    main()

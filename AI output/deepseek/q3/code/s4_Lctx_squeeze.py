# -*- coding: utf-8 -*-
"""
s4_Lctx_squeeze.py —— 问题三 阶段4：上下文长度挤压效应与临界值敏感性

赛题要求：
  "Lctx 的可行取值须依据 C7，须解析给出使注意力开销与训练开销相当的临界值
   Lctxcrit=6/η 并在其可行取值上完成敏感性分析，不得仅凭主观假设给定。"

本阶段内容：
  1) 解析推导 L_ctx_crit = 6/η = 30000（并给出其与 N,D 无关的证明）
  2) 在 C7 可行集 {2048,4096,8192,32768,131072} 上做敏感性分析
  3) 量化"挤压效应"：L_ctx 增大 → κ=6+ηL_ctx 增大 → 可用 N·D 缩小 → L* 上升
  4) 定义"挤压弹性"：-dlnL*/dlnL_ctx，并用 κ 折算等效预算损失
  5) 考察 L_ctx 对最优 Q*、N*、D*、成本份额的影响
  6) 阈值敏感性：改变 η，观察临界值与实际可行集的相对位置变化
  7) 关键：L_ctx 是否触发"结构性转移"（结合 s3 的 R0/R1/R2 定义）
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from s1_interface_objective import kappa, b_quality_cost_coeff
from s2_optimal_config import rho_and_C, optimize_single, loss_at_Q, M_of_Q
from s3_structural_transition import classify_regime


def main():
    start_log("s4_Lctx_squeeze")
    t0 = time.time()
    print("=" * 78)
    print("问题三 阶段4：上下文长度挤压效应与临界值敏感性")
    print("=" * 78)

    s0 = json.load(open(os.path.join(DATA, "s0_setup.json"), encoding="utf-8"))
    p = s0["interface_q2"]["generalized_params"]
    Q0 = s0["Q0_choice"]["primary"]
    rho, C_rho = rho_and_C(p)

    # ------------------------------------------------------------
    print("\n1. 临界上下文长度 L_ctx_crit 的解析推导")
    print("-" * 78)
    print("""
   设 C_attn = η·N·D·L_ctx,  C_train = 6·N·D。
   令两者相当（比值=1）：
        η·N·D·L_ctx = 6·N·D
   两边同时约去 N·D（恒正）：
        L_ctx_crit = 6/η
   代入 η = 2×10^-4：
        L_ctx_crit = 6 / 2e-4 = 30000  ← 精确值
   ★ 结论：L_ctx_crit 与 N、D、Q、C 均无关，是纯由成本系数决定的常数。
     原因：C_attn 与 C_train 都正比于同一个 N·D，比值为纯 L_ctx 的函数。
""")
    print(f"   L_ctx_crit = {L_CTX_CRIT:.0f}（精确）")
    print(f"   验证：η={ETA}, 6/η = {6.0/ETA:.1f}")

    # 与 C7 可行集对照
    print("\n   C7 可行集与临界值的相对位置：")
    for L in L_CTX_FEASIBLE:
        ratio = L / L_CTX_CRIT
        state = "低于临界" if ratio < 0.95 else ("≈临界" if ratio < 1.05 else "高于临界")
        print(f"     L_ctx={L:>7d}:  C_attn/C_train = {ratio:.4f}  [{state}]")
    print("   → 可行集中 32768（比 1.092）刚好越过临界；131072（比 4.369）远超临界。")

    # ------------------------------------------------------------
    print("\n2. 挤压效应的量化：κ(L_ctx) 与等效预算折减")
    print("-" * 78)
    print("   κ(L_ctx) = 6 + η·L_ctx   （N·D·1e18 的有效系数）")
    print("   等效可用预算 = C/κ × 6  （相对纯训练的可用比例 = 6/κ）")
    rows = []
    for L in L_CTX_FEASIBLE:
        k_ = kappa(L)
        rows.append({"L_ctx": L, "kappa": k_, "等效预算比": 6.0 / k_,
                     "C_attn/C_train": L / L_CTX_CRIT,
                     "可用N·D折减%": (1 - 6.0 / k_) * 100})
        print(f"     L_ctx={L:>7d}: κ={k_:8.4f}  可用预算比={6.0/k_:.4f}  "
              f"（N·D 折减 {(1-6.0/k_)*100:5.2f}%）")
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s4_挤压_κ与等效预算.csv"),
                              index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    print("\n3. 三档预算 × 五档 L_ctx 的敏感性（三成本族）")
    print("-" * 78)
    budgets = {"低(1e19)": 1e19, "中(1e22)": 1e22, "高(1e24)": 1e24}
    rows = []
    detail = {}
    for bname, C in budgets.items():
        for fam in QUALITY_COST_FAMILIES:
            for L in L_CTX_FEASIBLE:
                d = optimize_single(C, L, Q0, fam, p, rho, C_rho)
                reg = classify_regime(d["Q"], Q0)
                rows.append({
                    "预算档": bname, "C": C, "成本族": fam, "L_ctx": L,
                    "N*": d["N"], "D*": d["D"], "Q*": d["Q"], "L*": d["L"],
                    "状态": reg,
                    "C_train占比": d["C_train"] / C, "C_Q占比": d["C_Q"] / C,
                    "C_attn占比": d["C_attn"] / C,
                })
                detail[f"{bname}|{fam}|{L}"] = {
                    "N": d["N"], "D": d["D"], "Q": d["Q"], "L": d["L"],
                    "C_train": d["C_train"], "C_Q": d["C_Q"], "C_attn": d["C_attn"]}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES, "s4_Lctx敏感性.csv"), index=False, encoding="utf-8-sig")

    print("\n   [L* 随 L_ctx 的变化] （行=预算档×成本族, 列=L_ctx）")
    print(f"   {'组合':<22s}" + "".join(f"{L:>10d}" for L in L_CTX_FEASIBLE))
    for bname in budgets:
        for fam in QUALITY_COST_FAMILIES:
            sub = df[(df["预算档"] == bname) & (df["成本族"] == fam)]
            line = f"   {bname[:2]}-{fam:<8s}" if False else f"   {bname.split('(')[0]}-{fam:<9s}"
            for L in L_CTX_FEASIBLE:
                r = sub[sub["L_ctx"] == L].iloc[0]
                line += f"{r['L*']:>10.4f}"
            print(line)

    # ------------------------------------------------------------
    print("\n4. 挤压弹性：-dlnL*/dlnL_ctx")
    print("-" * 78)
    for bname in budgets:
        for fam in QUALITY_COST_FAMILIES:
            sub = df[(df["预算档"] == bname) & (df["成本族"] == fam)].sort_values("L_ctx")
            Lv = sub["L_ctx"].values.astype(float)
            Lst = sub["L*"].values.astype(float)
            # 与理论：无质量开销时 L*-E ∝ κ^{-ρ}，故 dlnL/dlnκ = -ρ·(κ/(L-E))
            # 挤压弹性 (对 L_ctx): 
            elas = -np.gradient(np.log(Lst - p["E"]), np.log(Lv))
            print(f"   {bname.split('(')[0]}-{fam:<9s}: "
                  f"挤压弹性范围 [{elas.min():.4f}, {elas.max():.4f}]，"
                  f"L* 从 {Lst[0]:.4f}→{Lst[-1]:.4f} (增 {100*(Lst[-1]/Lst[0]-1):.2f}%)")

    # ------------------------------------------------------------
    print("\n5. L_ctx 对最优 Q* 的影响（是否触发结构性转移）")
    print("-" * 78)
    for bname in budgets:
        for fam in QUALITY_COST_FAMILIES:
            sub = df[(df["预算档"] == bname) & (df["成本族"] == fam)].sort_values("L_ctx")
            regs = list(sub["状态"].values)
            qs = list(sub["Q*"].values)
            changed = len(set(regs)) > 1
            print(f"   {bname.split('(')[0]}-{fam:<9s}: Q*={['%.3f'%q for q in qs]} "
                  f"状态={[r[:2] for r in regs]}"
                  + ("  ★随L_ctx发生模式转移" if changed else ""))

    # ------------------------------------------------------------
    print("\n6. 阈值敏感性：η 变化对临界值的影响")
    print("-" * 78)
    print("   L_ctx_crit(η) = 6/η")
    etas = [1e-4, 2e-4, 4e-4, 1e-3]
    rows_eta = []
    for e in etas:
        crit = 6.0 / e
        # 哪些可行值超临界
        over = [L for L in L_CTX_FEASIBLE if L > crit]
        print(f"     η={e:.1e}: L_ctx_crit={crit:9.0f}  超临界的可行值={over if over else '无'}")
        rows_eta.append({"eta": e, "L_ctx_crit": crit, "超临界可行值": str(over)})
    pd.DataFrame(rows_eta).to_csv(os.path.join(TABLES, "s4_阈值敏感性_eta.csv"),
                                  index=False, encoding="utf-8-sig")
    print("   → η 越小，临界值越大，长上下文的'挤压'越轻；η 越大则越早触发挤压。")

    # ------------------------------------------------------------
    print("\n7. 解析公式：无质量开销时的挤压放大律")
    print("-" * 78)
    print("""
   无质量开销（Q=Q0）时，最优 L* - E = C_ρ · M^{-ρ}，其中 M = C/(κ·1e18)。
   故：  L*(L_ctx) - E = C_ρ · (C/1e18)^{-ρ} · κ(L_ctx)^{-ρ}
     ⇒ 相对放大因子 = (κ(L_ctx)/6)^{-ρ}
    ρ={rho:.4f} 时，各行内的放大因子：
""")
    for L in L_CTX_FEASIBLE:
        fac = (kappa(L) / 6.0) ** (-rho)
        amp = 1.0 / fac
        print(f"     L_ctx={L:>7d}: (κ/6)^-ρ = {fac:.5f}  ⇒ L*-E 放大 {amp:.5f}×")
    print("   → 由于 ρ≈0.121 很小，L_ctx 增大对 L* 的放大是'温和但确定'的。")

    out = {
        "L_ctx_crit": float(L_CTX_CRIT),
        "derivation": "C_attn = C_train ⇒ η·N·D·L_ctx = 6·N·D ⇒ L_ctx_crit = 6/η = 30000",
        "kappa_by_Lctx": {int(L): kappa(L) for L in L_CTX_FEASIBLE},
        "effective_budget_ratio": {int(L): 6.0 / kappa(L) for L in L_CTX_FEASIBLE},
        "squeeze_amplification": {int(L): float((kappa(L) / 6.0) ** (-rho))
                                  for L in L_CTX_FEASIBLE},
        "eta_sensitivity": rows_eta,
        "detail": detail,
        "rho": rho,
    }
    save_json(out, os.path.join(DATA, "s4_Lctx.json"))

    print(f"\n[OK] s4_Lctx_squeeze 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s4_Lctx_squeeze")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
s3_structural_transition.py —— 问题三 阶段3：结构性转移的定义、识别与预算扫描

赛题核心要求：
  "当预算 C 跨越上述不同量级时，资源分配是否出现结构性转移（即最优策略发生
   质的变化），并给出明确的数学定义与识别方法。"

本阶段给出三套互补的"结构性转移"数学定义与识别方法：

  【定义 A｜资源分配模式的质变（离散类别）】
    把最优策略按"质量投入状态"离散分类：
      R0 质量冻结态：Q* = Q0（C_Q/C ≈ 0），算力全给规模/长度
      R1 质量渐进态：Q0 < Q* < 1（内点解），质量随预算平滑抬升
      R2 质量饱和态：Q* = 1（边界解），质量已用尽上限
    结构性转移 = 相邻预算量级间类别标签发生改变（R_i ≠ R_j）。
    识别：C 的对数扫描 + 类别标注 + 变点检测。

  【定义 B｜份额的结构性反转（连续量的变点）】
    定义三个成本份额 s_train, s_Q, s_attn ∈ [0,1]。
    结构性转移 = 存在某一份额在预算量级间出现 排序反转 或 斜率突变：
      · 排序反转：argmax_j s_j(C) 在此区间发生变化
      · 斜率突变：d(份额)/d(lnC) 在该点前后符号改变或跳变超过阈值
    识别：拟合 lnC—份额 曲线的分段线性模型，用变点检测（BIC 最小）定位。

  【定义 C｜弹性的结构性切换（数学严格版）】
    最优解对预算的弹性
        E_C = -d ln L* / d ln C
    结构性转移 = E_C(lnC) 的导数（即二阶弹性）发生符号改变，
      或 E_C 跨越理论阈值（如从"规模主导"(E_C≈ρ) 跳到"质量主导"）。
    更严格：定义 Hessian / KKT 活跃约束集合 A(C) = {约束 | 在最优解处取等号}。
    "质的变化" = 活跃约束集合 A(C) 的基数或组成发生改变。
    识别：逐点求活跃集，记录 A(C) 的切换点。

本阶段输出：
  · 三个定义下的转移点（C_crit）识别结果
  · lnC 细扫描表（含 Q*, s_j, E_C, A(C)）
  · 与赛题"低收入顾温饱→中收入重教育→高收入重健康"的类比对照
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from s1_interface_objective import kappa, b_quality_cost_coeff
from s2_optimal_config import rho_and_C, optimize_single, loss_at_Q

# ===============================================================
def classify_regime(Qstar, Q0, eps_bound=1e-3, eps_frozen=1e-3):
    """定义 A：把最优解离散分类"""
    if abs(Qstar - 1.0) < eps_bound:
        return "R2_质量饱和态"
    if Qstar <= Q0 + eps_frozen:
        return "R0_质量冻结态"
    return "R1_质量渐进态"


def cost_shares(d):
    """三份额（总和≈1）"""
    tot = d["C_total"]
    return d["C_train"] / tot, d["C_Q"] / tot, d["C_attn"] / tot


def active_set(d, C, tol=1e-6):
    """定义 C：KKT 活跃约束集合"""
    s = set()
    # 预算约束几乎总活跃（最优在边界）
    if abs(d["C_total"] - C) / C < 1e-4:
        s.add("budget")
    # 质量上限约束 Q<=1
    if abs(d["Q"] - 1.0) < 1e-3:
        s.add("Q<=1")
    # 质量冻结约束 Q>=Q0
    if abs(d["Q"] - d["Q0"]) < 1e-3:
        s.add("Q>=Q0")
    return s


def find_change_points(x, y, min_seg=4):
    """
    分段线性变点检测（BIC 最小）：返回最优变点索引列表。
    x = lnC, y = 某个份额或 Q*。
    """
    n = len(x)
    def sse(i, j):
        xi, yi = x[i:j], y[i:j]
        if len(xi) < 2:
            return 0.0
        A = np.vstack([xi, np.ones_like(xi)]).T
        coef, *_ = np.linalg.lstsq(A, yi, rcond=None)
        pred = A @ coef
        return float(np.sum((yi - pred) ** 2))
    best = None
    # 单变点
    for k in range(min_seg, n - min_seg + 1):
        bic = n * np.log((sse(0, k) + sse(k, n)) / n + 1e-30) + 2 * 2 * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, [k])
    # 双变点
    best2 = None
    for k1 in range(min_seg, n - 2 * min_seg + 1):
        for k2 in range(k1 + min_seg, n - min_seg + 1):
            bic = n * np.log((sse(0, k1) + sse(k1, k2) + sse(k2, n)) / n + 1e-30) + 3 * 2 * np.log(n)
            if best2 is None or bic < best2[0]:
                best2 = (bic, [k1, k2])
    if best2 and best2[0] < best[0]:
        return best2[1], "2变点"
    return best[1], "1变点"


def main():
    start_log("s3_structural_transition")
    t0 = time.time()
    print("=" * 78)
    print("问题三 阶段3：结构性转移的定义、识别与预算扫描")
    print("=" * 78)

    s0 = json.load(open(os.path.join(DATA, "s0_setup.json"), encoding="utf-8"))
    p = s0["interface_q2"]["generalized_params"]
    Q0 = s0["Q0_choice"]["primary"]
    rho, C_rho = rho_and_C(p)

    print("\n【结构性转移的数学定义】")
    print("-" * 78)
    print("""
  定义 A（模式质变，离散）：
      状态空间 Ω = {R0_质量冻结态, R1_质量渐进态, R2_质量饱和态}
      转移事件 = 相邻预算量级处 Ω 标签变化。
  定义 B（份额变点，连续）：
      三份额 s_train(C), s_Q(C), s_attn(C)。
      (i) 排序反转: argmax_j s_j 改变;
      (ii) 斜率突变: d s_j / d lnC 变号或跳变 > τ。
  定义 C（活跃约束集切换，严格）：
      A(C) = KKT 活跃约束集合。质变 = A(C) 的组成/基数改变。
      等价判据：最优解结构（内点↔边界）切换。
""")

    # ------------------------------------------------------------
    print("\n1. 定义 A：模式质变 —— 对数预算精细扫描（L_ctx=8192）")
    print("-" * 78)
    Lctx_main = 8192
    Cgrid = np.logspace(16, 25, 181)
    regime_tables = {}
    for fam in QUALITY_COST_FAMILIES:
        regs = []
        for C in Cgrid:
            d = optimize_single(C, Lctx_main, Q0, fam, p, rho, C_rho)
            regs.append((C, d["Q"], classify_regime(d["Q"], Q0), d["L"],
                         d["C_Q"] / C, d["N"], d["D"]))
        regime_tables[fam] = regs
        # 找转移点
        print(f"\n  [{fam}] 状态序列与转移点：")
        prev = None
        trans = []
        for (C, Qstar, reg, L, cq, N, D) in regs:
            if prev is not None and reg != prev:
                trans.append((prev, reg, C))
                print(f"     ★ 转移 @ C={C:.3e}: {prev} → {reg}  (Q*={Qstar:.4f})")
            prev = reg
        print(f"     总转移次数 = {len(trans)}")

    # ------------------------------------------------------------
    print("\n2. 定义 B：份额与 Q* 的变点检测（L_ctx=8192）")
    print("-" * 78)
    lnC = np.log(Cgrid)
    for fam in QUALITY_COST_FAMILIES:
        regs = regime_tables[fam]
        Qarr = np.array([r[1] for r in regs])
        sQ = np.array([r[4] for r in regs])            # C_Q 份额
        sattn = np.array([1.0 - r[4] - (r[6] * 6.0e18 / r[0]) for r in regs])  # 近似
        # 更准确：直接从数据算 train 份额
        stra = np.array([(r[6] * (r[5] * 6.0e18)) / r[0] for r in regs])  # N*D*6e18/C
        # Q* 变点
        cps_q, mode_q = find_change_points(lnC, Qarr)
        print(f"\n  [{fam}]")
        print(f"     Q*(lnC) 变点: {[round(float(lnC[k]),2) for k in cps_q]} "
              f"(对应 C≈{[f'{Cgrid[k]:.2e}' for k in cps_q]}) [{mode_q}]")
        cps_s, mode_s = find_change_points(lnC, sQ)
        print(f"     s_Q(lnC) 变点: C≈{[f'{Cgrid[k]:.2e}' for k in cps_s]} [{mode_s}]")

    # ------------------------------------------------------------
    print("\n3. 定义 C：KKT 活跃约束集切换")
    print("-" * 78)
    for fam in QUALITY_COST_FAMILIES:
        print(f"\n  [{fam}] 活跃集 A(C) 随预算的切换：")
        prevA = None
        for C in Cgrid[::6]:
            d = optimize_single(C, Lctx_main, Q0, fam, p, rho, C_rho)
            A = active_set(d, C)
            key = frozenset(A)
            if prevA is not None and key != prevA:
                print(f"     ★ A(C) 切换 @ C={C:.3e}: "
                      f"{{{','.join(sorted(prevA)) or '∅'}}} → {{{','.join(sorted(A)) or '∅'}}}")
            prevA = key
        print(f"     末态 A = {{{','.join(sorted(A)) or '∅'}}}")

    # ------------------------------------------------------------
    print("\n4. 定义 C 的定量版：最优解对预算的弹性 E_C = -dlnL*/dlnC")
    print("-" * 78)
    print("   E_C 理论：若纯规模主导，L*-E ∝ C^{-ρ} ⇒ E_C = ρ = %.4f" % rho)
    print("             若质量主导，E_C 应偏离 ρ。转移点处 E_C 斜率突变。")
    for fam in QUALITY_COST_FAMILIES:
        regs = regime_tables[fam]
        Larr = np.array([r[3] for r in regs])
        Cc = np.array([r[0] for r in regs])
        # 数值弹性
        Earr = -np.gradient(np.log(Larr - p["E"]), np.log(Cc))
        # 找 E_C 突变（二阶）
        d2 = np.gradient(Earr, np.log(Cc))
        # 突变点：|d2| 局部极大
        idx = []
        for i in range(3, len(d2) - 3):
            if abs(d2[i]) == max(abs(d2[i - 3:i + 4])) and abs(d2[i]) > 0.005:
                idx.append(i)
        print(f"\n  [{fam}] E_C ∈ [{Earr.min():.4f}, {Earr.max():.4f}] 均值 {Earr.mean():.4f}")
        print(f"     E_C 突变点 @ C≈{[f'{Cc[i]:.2e}' for i in idx]}")

    # ------------------------------------------------------------
    print("\n5. 三定义一致性汇总（核心转移点）")
    print("-" * 78)
    # 以定义 A 的转移点为准
    summary = {}
    for fam in QUALITY_COST_FAMILIES:
        regs = regime_tables[fam]
        prev = None
        trans = []
        for (C, Qstar, reg, L, cq, N, D) in regs:
            if prev is not None and reg != prev:
                trans.append({"from": prev, "to": reg, "C": C, "Q*": Qstar})
            prev = reg
        summary[fam] = trans
        print(f"  [{fam}]")
        for t in trans:
            print(f"     C_crit ≈ {t['C']:.3e}:  {t['from']} → {t['to']}  (Q*={t['Q*']:.4f})")
        if not trans:
            print("     （在本扫描范围内无模式转移）")

    # ------------------------------------------------------------
    print("\n6. 与赛题类比的对照")
    print("-" * 78)
    print("""
   赛题类比："低收入顾温饱 → 中等收入重教育 → 高收入重健康"
   本问对应："低预算顾规模 → 中预算重质量 → 高预算质量饱和后回归规模/长度"

   · R0 质量冻结态 ≈ 顾温饱：预算太少，先把基础训练做大（C_Q≈0）
   · R1 质量渐进态 ≈ 重教育：有余力则逐步加码数据质量（内点 Q*）
   · R2 质量饱和态 ≈ 重健康：质量封顶（Q*=1），转而把算力投向参数/长度
   → 这正是一个典型的"结构性转移"：最优策略的'重心'发生质变。
""")

    # 保存
    full_scan = {}
    for fam in QUALITY_COST_FAMILIES:
        regs = regime_tables[fam]
        full_scan[fam] = [{"C": float(r[0]), "Q*": float(r[1]), "regime": r[2],
                           "L*": float(r[3]), "s_Q": float(r[4]),
                           "N": float(r[5]), "D": float(r[6])} for r in regs]
    out = {"regime_definition": {
        "R0": "质量冻结态 Q*=Q0",
        "R1": "质量渐进态 Q0<Q*<1",
        "R2": "质量饱和态 Q*=1"},
        "rho": rho, "Q0": Q0, "L_ctx": Lctx_main,
        "transitions": summary,
        "scan": full_scan,
        "grid": {"C_min": float(Cgrid[0]), "C_max": float(Cgrid[-1]), "n": len(Cgrid)}}
    save_json(out, os.path.join(DATA, "s3_transition.json"))

    # 转移点表
    rows = []
    for fam, ts in summary.items():
        for t in ts:
            rows.append({"成本族": fam, "C_crit": t["C"], "从": t["from"],
                         "到": t["to"], "Q*@转移": t["Q*"]})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s3_结构性转移点.csv"),
                              index=False, encoding="utf-8-sig")

    print(f"\n[OK] s3_structural_transition 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s3_structural_transition")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
s0_setup.py —— 问题三 阶段0：问题界定、成本函数建模与前两问接口

本阶段内容：
  1) 解析赛题成本结构，写出完整优化问题
  2) 解析推导临界上下文长度 L_ctx_crit = 6/η = 30000
  3) 从附件 C7 确定 L_ctx 的可行集
  4) 从问题一、二接入接口：质量 q/p、标度律参数
  5) 校验三族质量成本函数的量级与形状
  6) 输出问题界定文档所需的所有基础量

产出：
  data/s0_setup.json
  results/tables/s0_成本函数量级.csv, s0_Lctx可行集.csv, s0_接口参数.csv
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    start_log("s0_setup")
    t0 = time.time()
    print("=" * 78)
    print("问题三 阶段0：问题界定与成本函数建模")
    print("=" * 78)

    # ------------------------------------------------------------
    # 1. 优化问题陈述
    # ------------------------------------------------------------
    print("\n1. 优化问题（数学陈述）")
    print("-" * 78)
    print("""
  决策变量:  N (参数量, B),  D (数据量, B token),  Q (数据质量),  p (17域配比)
  外生参数:  L_ctx (上下文长度, 由 C7 给定),  Q_0 (基线质量),  C (总算力预算)

  目标:      min  L(N,D,Q) = E + (a·N^-α + b·D^-β)·Q^-γ      [问题二广义律]
  约束:
     (C1) 6·N·D·1e18                     <= C     [基础训练]
     (C2) D·1e9·[g(Q)-g(Q_0)]_+          <= C     [质量提升]
     (C3) η·N·D·1e18·L_ctx               <= C     [长文本注意力]
     (C1)+(C2)+(C3) 合计                 <= C     [总预算]
     0 < Q <= 1,  N>0,  D>0,  p ∈ Δ^16 (单纯形)

  注意: N,D 以 B 计，绝对量须 ×1e9；故 6ND 的真实 FLOPs = 6·N·D·1e18。
""")

    # ------------------------------------------------------------
    # 2. 临界上下文长度解析推导
    # ------------------------------------------------------------
    print("2. 临界上下文长度 L_ctx_crit（解析推导）")
    print("-" * 78)
    print("  由 C_attn = η·N·D·L_ctx 与 C_train = 6·N·D 相等：")
    print("      η·N·D·L_ctx = 6·N·D  ⇒  L_ctx_crit = 6/η")
    print(f"  代入 η={ETA:.1e}:  L_ctx_crit = 6/{ETA:.1e} = {L_CTX_CRIT:.0f}")
    print(f"\n  → 当 L_ctx < {L_CTX_CRIT:.0f} 时，注意力开销 < 基础训练开销；")
    print(f"     当 L_ctx > {L_CTX_CRIT:.0f} 时，注意力开销 > 基础训练开销（算力被长上下文吃掉）。")
    print(f"     注意 L_ctx_crit={L_CTX_CRIT:.0f} 与 N,D 无关（因两者都正比于 N·D）——这是重要的结构性质。")

    # ------------------------------------------------------------
    # 3. L_ctx 可行集（依据 C7）
    # ------------------------------------------------------------
    print("\n3. L_ctx 可行集（依据附件 C7）")
    print("-" * 78)
    arch = load_q3_arch()
    vals = sorted(arch.max_position_embeddings.dropna().unique())
    print(f"  C7 记录 {len(arch)} 个模型的 max_position_embeddings，唯一取值：")
    cnt = arch.max_position_embeddings.value_counts().sort_index()
    for v, c in cnt.items():
        ratio = float(v) / L_CTX_CRIT
        flag = "≈临界" if 0.8 < ratio < 1.25 else ("<临界" if ratio < 1 else ">临界")
        print(f"    L_ctx = {int(v):>7d}  (n={c:>2d})   注意力/训练开销比 = {ratio:.3f}  {flag}")
    print(f"\n  → 可行集 = {L_CTX_FEASIBLE}")
    print(f"     其中 32768 > L_ctx_crit={L_CTX_CRIT:.0f} 略微超过临界；131072 远超临界（比 4.4 倍）")
    dctx = pd.DataFrame([{"L_ctx": int(v), "n_models": int(c),
                          "C_attn/C_train": float(v) / L_CTX_CRIT,
                          "是否超临界": "是" if v > L_CTX_CRIT else "否"}
                         for v, c in cnt.items()])
    dctx.to_csv(os.path.join(TABLES, "s0_Lctx可行集.csv"), index=False, encoding="utf-8-sig")
    print(f"     解析临界值 L_ctx_crit = {L_CTX_CRIT:.0f} 落入可行集相邻档位 (8192, 32768) 之间。")

    # ------------------------------------------------------------
    # 4. 前两问接口
    # ------------------------------------------------------------
    print("\n4. 前两问结果接口")
    print("-" * 78)
    dom, q, qsd, levels = load_q1_quality()
    wmix = load_q1_mixture_weights()
    print(f"  [问题一] 17 配方域质量 q：范围 [{q.min():.4f},{q.max():.4f}]，均值 {q.mean():.4f}")
    print(f"          最高: {dom[int(np.argmax(q))]} = {q.max():.4f}")
    print(f"          最低: {dom[int(np.argmin(q))]} = {q.min():.4f}")
    # 真实配比下的 Q_mix
    p_real = np.zeros(len(dom))
    for i, d in enumerate(dom):
        p_real[i] = wmix.get(d, 0.0)
    p_real = p_real / p_real.sum() if p_real.sum() > 0 else np.ones(len(dom)) / len(dom)
    Q_mix = float(np.dot(p_real, q))
    print(f"          Q_mix(真实配比) = Σp_j q_j = {Q_mix:.4f}")
    print(f"          Q_mix(均匀) = {q.mean():.4f}")

    s2c, s1 = load_q2_scaling()
    pf = s2c["params_B7"]
    print(f"\n  [问题二] 广义律 M4: L = E + (a·N^-α + b·D^-β)·Q^-γ")
    for k, v in pf.items():
        print(f"          {k:<7s} = {v:.5f}")
    print(f"          形式={s2c['final_form']}, B7 R²={s2c['metrics_B7']['R2']:.4f}")
    cl = s1["additive"]["params"]
    print(f"  [问题二] 经典律(B1): E={cl['E']:.4f} a={cl['a']:.4f} α={cl['alpha']:.4f} "
          f"b={cl['b']:.4f} β={cl['beta']:.4f}")

    # 基线质量 Q_0 的确定
    print(f"\n  基线质量 Q_0 的确定：")
    print(f"     (a) 取问题一真实配比下的 Q_mix = {Q_mix:.4f}（推荐，可解释为'现状）")
    print(f"     (b) 取均匀配比 = {q.mean():.4f}")
    print(f"     (c) 取问题二 B10 实测最佳拟合 Q = 0.40（工业界大模型现状）")
    print(f"     → 主分析取 Q_0 = {Q_mix:.4f}（问题一真实配比），敏感性分析覆盖 0.40~0.58")

    # ------------------------------------------------------------
    # 5. 三族质量成本函数的量级与形状
    # ------------------------------------------------------------
    print("\n5. 三族质量成本函数 g(Q) 量级与形状")
    print("-" * 78)
    rows = []
    for fam in QUALITY_COST_FAMILIES:
        spec = QUALITY_COST_FAMILIES[fam]
        print(f"  [{fam}] {spec['note']}")
        line = "     "
        for Qv in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            gv = g_of(fam, Qv)
            line += f"g({Qv})={gv:.3e}  "
            rows.append({"成本族": fam, "Q": Qv, "g(Q)": gv,
                         "增量g(Q)-g(Q0=0.5621)": gv - g_of(fam, Q_mix)})
        print(line)
    dg = pd.DataFrame(rows)
    dg.to_csv(os.path.join(TABLES, "s0_成本函数量级.csv"), index=False, encoding="utf-8-sig")
    print("\n  → 关键观察：")
    print("     · 指数型 γ=1e7 最小，但 λ=6.0 使 g 对 Q 极敏感（Q:0.1→1.0 涨 200 倍）")
    print("     · 幂函数型 γ=5e9 中等，λ=4.0 敏感性适中")
    print("     · 对数渐进型 γ=2e9，λ=10.0 在低 Q 区快速上升后趋于平缓（渐进饱和）")
    print("     · 三族在 Q→1 附近的边际成本 dg/dQ 差异巨大，将直接决定最优 Q* 的高低")

    # 计算各族的边际成本 dg/dQ
    print("\n  各族的边际质量成本 dg/dQ：")
    marg = []
    for fam in QUALITY_COST_FAMILIES:
        spec = QUALITY_COST_FAMILIES[fam]
        line = f"  [{fam:<8s}] "
        for Qv in [0.2, 0.5, 0.8, 1.0]:
            dQv = 1e-5
            dg_ = (g_of(fam, Qv + dQv) - g_of(fam, Qv - dQv)) / (2 * dQv)
            line += f"dQ/dQ({Qv})={dg_:.3e}  "
            marg.append({"成本族": fam, "Q": Qv, "dg/dQ": dg_})
        print(line)
    pd.DataFrame(marg).to_csv(os.path.join(TABLES, "s0_边际质量成本.csv"),
                              index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 6. 接口参数汇总保存
    # ------------------------------------------------------------
    out = {
        "problem_definition": {
            "objective": "min L(N,D,Q) = E + (a*N^-alpha + b*D^-beta)*Q^-gamma",
            "constraints": {
                "C1_train": "6*N*D*1e18 <= C",
                "C2_quality": "D*1e9*[g(Q)-g(Q0)]_+ <= C",
                "C3_attn": "eta*N*D*1e18*L_ctx <= C",
                "total": "C_train + C_Q + C_attn <= C",
                "simplex": "sum_j p_j = 1, p_j >= 0",
                "ranges": "0 < Q <= 1, N > 0, D > 0",
            },
        },
        "constants": {"eta": ETA, "L_ctx_crit": float(L_CTX_CRIT),
                      "unit_factor": 1e18, "budgets": BUDGETS,
                      "L_ctx_feasible": L_CTX_FEASIBLE},
        "quality_cost_families": {k: {"note": v["note"], "gamma": v["gamma"], "lam": v["lam"]}
                                  for k, v in QUALITY_COST_FAMILIES.items()},
        "interface_q1": {"domains": dom, "q": q.tolist(), "levels": [str(x) for x in levels],
                         "Q_mix_real": Q_mix, "Q_mix_uniform": float(q.mean()),
                         "p_real": p_real.tolist()},
        "interface_q2": {"generalized_params": pf, "classic_params": cl,
                         "form": s2c["final_form"], "R2_B7": s2c["metrics_B7"]["R2"]},
        "Q0_choice": {"primary": Q_mix, "range": [0.40, 0.58],
                      "rationale": "主用问题一真实配比下的 Q_mix；敏感性覆盖工业界实测 0.40~0.58"},
        "L_ctx_crit_derivation": "C_attn = C_train ⇒ eta·N·D·L_ctx = 6·N·D ⇒ L_ctx_crit = 6/eta = 30000",
    }
    save_json(out, os.path.join(DATA, "s0_setup.json"))

    # 接口参数表
    rowsi = []
    rowsi.append({"来源": "问题一", "量": "q 均值", "值": float(q.mean())})
    rowsi.append({"来源": "问题一", "量": "Q_mix(真实配比)", "值": Q_mix})
    for k, v in pf.items():
        rowsi.append({"来源": "问题二", "量": f"广义律 {k}", "值": v})
    for k, v in cl.items():
        rowsi.append({"来源": "问题二", "量": f"经典律 {k}", "值": v})
    rowsi.append({"来源": "赛题", "量": "η", "值": ETA})
    rowsi.append({"来源": "赛题", "量": "L_ctx_crit", "值": L_CTX_CRIT})
    pd.DataFrame(rowsi).to_csv(os.path.join(TABLES, "s0_接口参数.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[OK] s0_setup 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s0_setup")


if __name__ == "__main__":
    main()

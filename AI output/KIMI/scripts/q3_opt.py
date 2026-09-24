# -*- coding: utf-8 -*-
"""
问题三：算力约束下 (N,D,Q) 联合优化 + 结构性转移分析
成本: C = 6ND + D*[g(Q)-g(Q0)]+ + eta*N*D*Lctx <= B
目标: min L = E + A/N^a + B_*m(p*)/(D^b*(Q/Q0)^gq)
预算: 1e19 / 1e22 / 1e24 FLOPs; Lctx in C7可行集 {2048,4096,8192,32768,131072}
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = r"C:/Users/dkyyt/Desktop/F题"
OUT = os.path.join(ROOT, "outputs", "q3_optimization")
os.makedirs(OUT, exist_ok=True)

# ---- 问题二输出参数 ----
law = json.load(open(os.path.join(ROOT, "outputs", "q2_scaling", "final_generalized_law.json")))
E, A, AL, Bc, BE, GQ, Q0 = law["E"], law["A"], law["alpha"], law["B"], law["beta"], law["gamma_Q"], law["Q0_baseline"]
ETA = 2e-4
L_CRIT = 6.0 / ETA  # 30000
GAMMA1e9 = 1e9

# ---- 附录B成本函数 g(Q) (单位: FLOPs/token) ----
G_FORMS = {
    "exponential": (lambda Q: 1e7 * np.exp(6.0 * Q), "g(Q)=1e7·e^{6Q}"),
    "power":       (lambda Q: 5e9 * Q**4.0,          "g(Q)=5e9·Q^4"),
    "logarithmic": (lambda Q: 2e9 * np.log(1 + 10.0 * Q), "g(Q)=2e9·ln(1+10Q)"),
}
for nm, (g, _) in G_FORMS.items():
    print(f"{nm}: g(Q0={Q0})={g(Q0):.3e}, g(0.8)={g(0.8):.3e}, g(1.0)={g(1.0):.3e}")

def loss(NDQ):
    N, D, Q = NDQ
    Nb, Db = N / 1e9, D / 1e9  # 标度律参数以"十亿"为单位拟合
    return E + A / Nb**AL + Bc / (Db**BE * (Q / Q0)**GQ)

def cost_parts(NDQ, Lctx, g):
    N, D, Q = NDQ
    ctr = 6 * N * D
    cq = D * max(g(Q) - g(Q0), 0.0)
    ca = ETA * N * D * Lctx
    return ctr, cq, ca

def solve(Bbudget, Lctx, g, nstarts=12, seed=0):
    rng = np.random.default_rng(seed)
    def obj(v):
        return loss((np.exp(v[0]), np.exp(v[1]), v[2]))
    def con(v):
        N, D, Q = np.exp(v[0]), np.exp(v[1]), v[2]
        c = sum(cost_parts((N, D, Q), Lctx, g))
        return Bbudget - c
    best = None
    def consider(r):
        nonlocal best
        N, D, Q = np.exp(r.x[0]), np.exp(r.x[1]), r.x[2]
        if con(r.x) >= -1e-3 * Bbudget and (best is None or r.fun < best.fun):
            best = r
    for k in range(nstarts):
        v0 = [np.log(10**rng.uniform(7, 11)), np.log(10**rng.uniform(9, 12)), rng.uniform(Q0, 1.0)]
        r = minimize(obj, v0, method="SLSQP",
                     bounds=[(np.log(1e7), np.log(1e12)), (np.log(1e8), np.log(1e13)), (Q0, 1.0)],
                     constraints=[{"type": "ineq", "fun": con}],
                     options={"maxiter": 2000, "ftol": 1e-12})
        consider(r)
    if best is None:  # 兜底: 纯Chinchilla网格搜索
        for lN in np.linspace(7, 12, 40):
            for lD in np.linspace(8, 13, 40):
                for Q in [Q0, 0.7, 0.85, 1.0]:
                    v = np.array([lN * np.log(10), lD * np.log(10), Q])
                    if con(v) >= 0 and (best is None or obj(v) < best.fun):
                        class R: pass
                        best = R(); best.x = v; best.fun = obj(v)
    N, D, Q = np.exp(best.x[0]), np.exp(best.x[1]), best.x[2]
    ctr, cq, ca = cost_parts((N, D, Q), Lctx, g)
    return {"N_B": N / 1e9, "D_B": D / 1e9, "Q": Q, "loss": best.fun,
            "C_train": ctr, "C_quality": cq, "C_attn": ca, "C_total": ctr + cq + ca,
            "share_train": ctr / (ctr + cq + ca), "share_quality": cq / (ctr + cq + ca),
            "share_attn": ca / (ctr + cq + ca)}

BUDGETS = [1e19, 1e22, 1e24]
LCTXS = [2048, 4096, 8192, 32768, 131072]

# ---- 1. 主求解: 3预算 x 3成本形式 (Lctx=2048 基准) ----
rows = []
for Bb in BUDGETS:
    for fnm, (g, gdesc) in G_FORMS.items():
        r = solve(Bb, 2048, g)
        rows.append({"budget": Bb, "cost_form": fnm, "Lctx": 2048, **r})
main = pd.DataFrame(rows)
main.to_csv(os.path.join(OUT, "optimal_allocations.csv"), index=False)
print(main[["budget","cost_form","N_B","D_B","Q","loss","share_train","share_quality","share_attn"]].round(4).to_string())

# ---- 2. 对照: 不投资质量(Q=Q0固定) ----
rows2 = []
for Bb in BUDGETS:
    g = G_FORMS["exponential"][0]
    r = solve(Bb, 2048, g)
    # Q固定: 退化为Chinchilla分配
    def obj2(v):
        N, D = np.exp(v[0]), np.exp(v[1])
        return loss((N, D, Q0))
    def con2(v):
        N, D = np.exp(v[0]), np.exp(v[1])
        return Bb - (6 * N * D + ETA * N * D * 2048)
    best2 = None
    rng = np.random.default_rng(0)
    for k in range(8):
        v0 = [np.log(10**rng.uniform(7, 11)), np.log(10**rng.uniform(9, 12))]
        rr = minimize(obj2, v0, method="SLSQP",
                      bounds=[(np.log(1e7), np.log(1e12)), (np.log(1e8), np.log(1e13))],
                      constraints=[{"type": "ineq", "fun": con2}], options={"maxiter": 2000})
        if con2(rr.x) >= -1e-3 * Bb and (best2 is None or rr.fun < best2.fun):
            best2 = rr
    if best2 is None:
        for lN in np.linspace(7, 12, 40):
            for lD in np.linspace(8, 13, 40):
                v = np.array([lN * np.log(10), lD * np.log(10)])
                if con2(v) >= 0 and (best2 is None or obj2(v) < best2.fun):
                    class R2: pass
                    best2 = R2(); best2.x = v; best2.fun = obj2(v)
    best = best2
    N, D = np.exp(best.x[0]), np.exp(best.x[1])
    rows2.append({"budget": Bb, "scenario": "Q_fixed_baseline", "N_B": N/1e9, "D_B": D/1e9,
                  "Q": Q0, "loss": best.fun})
    rows2.append({"budget": Bb, "scenario": "Q_optimized(exp)", "N_B": r["N_B"], "D_B": r["D_B"],
                  "Q": r["Q"], "loss": r["loss"]})
cmp_q = pd.DataFrame(rows2)
cmp_q.to_csv(os.path.join(OUT, "quality_investment_value.csv"), index=False)
print(cmp_q.round(4).to_string())

# ---- 3. Lctx 敏感性 (B=1e22, 指数型) ----
rows3 = []
g = G_FORMS["exponential"][0]
for lc in LCTXS:
    r = solve(1e22, lc, g)
    rows3.append({"budget": 1e22, "Lctx": lc, "attn_over_train": ETA * lc / 6.0, **r})
sens = pd.DataFrame(rows3)
sens.to_csv(os.path.join(OUT, "lctx_sensitivity.csv"), index=False)
print(sens[["Lctx","attn_over_train","N_B","D_B","Q","loss","share_attn"]].round(4).to_string())

# ---- 4. 预算扫描: 结构性转移识别 (指数型, Lctx=2048) ----
Bs = np.logspace(18, 26, 33)
sweep = []
for Bb in Bs:
    r = solve(Bb, 2048, g)
    sweep.append({"budget": Bb, **r})
sw = pd.DataFrame(sweep)
sw.to_csv(os.path.join(OUT, "budget_sweep.csv"), index=False)
# 结构性转移定义与识别:
# (T1) 主导成本项切换(argmax share变化); (T2) Q*触及上界1(边界激活); (T3) 份额穿越10%/50%阈值
trans = []
prev = None
for i in range(1, len(sw)):
    a, b = sw.iloc[i-1], sw.iloc[i]
    argmax_a = np.argmax([a.share_train, a.share_quality, a.share_attn])
    argmax_b = np.argmax([b.share_train, b.share_quality, b.share_attn])
    if argmax_a != argmax_b:
        trans.append({"type": "T1_主导项切换", "from_budget": a.budget, "to_budget": b.budget,
                      "detail": f"{['train','quality','attn'][argmax_a]} -> {['train','quality','attn'][argmax_b]}"})
    if a.Q < 0.999 and b.Q >= 0.999:
        trans.append({"type": "T2_Q触及上界", "from_budget": a.budget, "to_budget": b.budget, "detail": "Q*=1 边界激活"})
    for th in [0.001, 0.10, 0.50]:
        if (a.share_quality - th) * (b.share_quality - th) < 0:
            trans.append({"type": f"T3_质量份额穿越{th:.1%}" if th < 0.1 else f"T3_质量份额穿越{th:.0%}",
                          "from_budget": a.budget, "to_budget": b.budget,
                          "detail": f"{a.share_quality:.4f} -> {b.share_quality:.4f}"})
trans_df = pd.DataFrame(trans)
trans_df.to_csv(os.path.join(OUT, "structural_transitions.csv"), index=False)
print(trans_df.to_string() if len(trans) else "扫描区间内未检测到转移")
print(sw[["budget","N_B","D_B","Q","loss","share_train","share_quality","share_attn"]].round(4).to_string())

# ---- 5. 成本形式对比 (B=1e22) ----
print("\n成本形式对最优解的影响 (B=1e22):")
print(main[main.budget == 1e22][["cost_form","N_B","D_B","Q","loss","share_quality"]].round(4).to_string())

# ---- 6. 图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axes[0].loglog(sw.budget, sw.N_B, "o-", label="N* (B)", color="#2E86C1")
axes[0].loglog(sw.budget, sw.D_B, "s-", label="D* (B)", color="#28B463")
axes[0].set_xlabel("预算 B (FLOPs)"); axes[0].set_ylabel("十亿"); axes[0].legend()
axes[0].set_title("最优规模随预算增长")
axes[1].semilogx(sw.budget, sw.Q, "o-", color="#E67E22")
axes[1].set_xlabel("预算 B (FLOPs)"); axes[1].set_ylabel("Q*"); axes[1].set_title("最优质量投入")
axes[1].axhline(Q0, color="gray", ls="--", lw=1, label=f"基线 Q0={Q0}")
axes[1].legend()
axes[2].loglog(sw.budget, sw.share_train, "o-", label="训练份额", color="#2E86C1")
axes[2].loglog(sw.budget, sw.share_quality, "s-", label="质量份额", color="#E67E22")
axes[2].loglog(sw.budget, sw.share_attn, "^-", label="注意力份额", color="#7D3C98")
axes[2].set_xlabel("预算 B (FLOPs)"); axes[2].set_ylabel("预算份额"); axes[2].legend()
axes[2].set_title("预算结构 (Lctx=2048)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_budget_sweep.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(main[main.budget == 1e22])); w = 0.25
mm = main[main.budget == 1e22]
ax.bar(x - w, mm.share_train, w, label="训练", color="#2E86C1")
ax.bar(x, mm.share_quality, w, label="质量", color="#E67E22")
ax.bar(x + w, mm.share_attn, w, label="注意力", color="#7D3C98")
ax.set_xticks(x); ax.set_xticklabels(mm.cost_form)
ax.set_title("成本函数形式对预算结构的影响 (B=1e22)"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_costform.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sens.Lctx, sens.loss, "o-", color="#C0392B")
ax.axvline(L_CRIT, color="k", ls="--", label=f"L*={L_CRIT:.0f} (C_attn=C_train)")
ax.set_xscale("log"); ax.set_xlabel("L_ctx"); ax.set_ylabel("最优Loss")
ax.set_title("上下文长度对可达Loss的影响 (B=1e22)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_lctx.png"), dpi=150); plt.close()
print("DONE_Q3")

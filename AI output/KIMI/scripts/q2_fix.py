# -*- coding: utf-8 -*-
"""
问题二(修正与定稿)：
1. B1完美拟合诊断 2. B2族外单参数再校准 3. B8方向反转诊断(可信度边界)
4. 最终广义标度律: B1锚定 + B6/B7定质量指数 + 配比惩罚m(p)
5. 弹性/替代条件 6. 图与下游参数文件
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

ROOT = r"C:/Users/dkyyt/Desktop/F题"
B = os.path.join(ROOT, "real_attachments", "B_scaling_laws")
OUT = os.path.join(ROOT, "outputs", "q2_scaling")

pyt = pd.read_csv(os.path.join(B, "pythia_training_log_existing.csv"))
cer = pd.read_csv(os.path.join(B, "cerebras_training_log.csv"))
nq6 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment.csv"))
nq7 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_expanded.csv"))
nq8 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_large.csv"))

# ---------- 1. B1 完美拟合诊断 ----------
pc = {"E": 1.6898, "A": 0.3540, "alpha": 0.3400, "B": 1.2403, "beta": 0.2799}
pr1 = pc["E"] + pc["A"]/pyt.N_params_B.values**pc["alpha"] + pc["B"]/pyt.D_tokens_B.values**pc["beta"]
res = pyt.val_loss.values - pr1
diag = {"max_abs_resid": float(np.abs(res).max()), "std_resid": float(res.std()),
        "rel_resid_p99": float(np.percentile(np.abs(res)/pyt.val_loss.values, 99))}
with open(os.path.join(OUT, "b1_fit_diagnosis.json"), "w") as f:
    json.dump(diag, f, indent=2)
print("B1残差诊断:", diag)

# ---------- 2. B2 族外验证与单参数再校准 ----------
pr2 = pc["E"] + pc["A"]/cer.N_params_B.values**pc["alpha"] + pc["B"]/cer.D_tokens_B.values**pc["beta"]
L2 = cer.val_loss.values
# 再校准: 仅拟合加性族常数 delta, 即 L = classic + delta
delta = float(np.mean(L2 - pr2))
pr2c = pr2 + delta
r2_raw = 1 - np.sum((L2-pr2)**2)/np.sum((L2-L2.mean())**2)
r2_cal = 1 - np.sum((L2-pr2c)**2)/np.sum((L2-L2.mean())**2)
cal = pd.DataFrame([{"dataset":"B2_Cerebras","R2_raw":r2_raw,"R2_after_1param_calib":r2_cal,
                     "family_offset_delta":delta,"spearman_raw":spearmanr(pr2,L2)[0]}])
cal.to_csv(os.path.join(OUT, "b2_family_recalibration.csv"), index=False)
print(cal.round(4).to_string())

# ---------- 3. B8 方向反转诊断 ----------
rows8 = []
for (n_, d_), g8 in nq8.groupby(["N_params_B","D_tokens_B"]):
    g8 = g8.sort_values("Q_score")
    if len(g8) >= 4:
        rho = spearmanr(g8.Q_score, g8.val_loss)[0]
        rows8.append({"N": n_, "D": d_, "spearman_Q_L_B8": rho, "n_Q": len(g8),
                      "type": g8.data_type.iloc[0]})
b8dir = pd.DataFrame(rows8)
b8dir.to_csv(os.path.join(OUT, "b8_direction_diagnosis.csv"), index=False)
print("B8各(N,D)网格点Q-L相关: 均值={:.3f}, >0占比={:.2f}".format(
    b8dir.spearman_Q_L_B8.mean(), (b8dir.spearman_Q_L_B8 > 0).mean()))
rows6 = []
for (n_, d_), g6 in pd.concat([nq6, nq7]).groupby(["N_params_B","D_tokens_B"]):
    g6 = g6.sort_values("Q_score")
    if len(g6) >= 4:
        rows6.append({"N": n_, "D": d_, "spearman_Q_L_B67": spearmanr(g6.Q_score, g6.val_loss)[0]})
b67dir = pd.DataFrame(rows6)
b67dir.to_csv(os.path.join(OUT, "b67_direction_check.csv"), index=False)
print("B6/B7各网格点Q-L相关: 均值={:.3f}, <0占比={:.2f}".format(
    b67dir.spearman_Q_L_B67.mean(), (b67dir.spearman_Q_L_B67 < 0).mean()))

# ---------- 4. F1形式在 B6 拟合 / B7 验证 (质量指数 gamma) ----------
def f1_resid(theta, N, D, Q, L):
    E, A, a, Bb, b, g = np.exp(theta[0]), np.exp(theta[1]), theta[2], np.exp(theta[3]), theta[4], theta[5]
    return np.log(E + A/N**a + Bb/(D**b * Q**g)) - np.log(L)

def fit_f1(N, D, Q, L, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    lo = [-10,-20,0.01,-20,0.01,0.001]; hi = [5,20,2,20,2,3]
    for _ in range(30):
        x0 = np.array([0.0, 0.0, 0.3, 0.0, 0.2, 0.1]) + rng.normal(0, 0.4, 6)
        x0 = np.clip(x0, lo, hi)
        r = least_squares(f1_resid, x0, args=(N, D, Q, L), loss="soft_l1", f_scale=0.1,
                          bounds=(lo, hi), max_nfev=30000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x

x6 = fit_f1(nq6.N_params_B.values, nq6.D_tokens_B.values, nq6.Q_score.values, nq6.val_loss.values)
E6, A6, a6, B6_, b6, g6 = np.exp(x6[0]), np.exp(x6[1]), x6[2], np.exp(x6[3]), x6[4], x6[5]
def f1_pred(x, N, D, Q):
    E, A, a, Bb, b, g = np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), x[4], x[5]
    return E + A/N**a + Bb/(D**b * Q**g)
pr7 = f1_pred(x6, nq7.N_params_B.values, nq7.D_tokens_B.values, nq7.Q_score.values)
r2_7 = 1 - np.sum((nq7.val_loss.values-pr7)**2)/np.sum((nq7.val_loss.values-nq7.val_loss.mean())**2)
print(f"F1(B6拟合): E={E6:.4f} A={A6:.4f} a={a6:.4f} B={B6_:.4f} b={b6:.4f} gamma={g6:.4f}; B7验证R2={r2_7:.4f}")

# B8单独拟合(展示符号反转)
x8 = fit_f1(nq8.N_params_B.values, nq8.D_tokens_B.values, nq8.Q_score.values, nq8.val_loss.values, seed=3)
g8 = x8[5]
print(f"B8单独拟合的gamma={g8:.4f} (符号与B6相反: {np.sign(g8)!=np.sign(g6)})")

# ---------- 5. 配比惩罚范围 ----------
tr = pd.read_csv(os.path.join(ROOT, "real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv"))
trl = pd.read_csv(os.path.join(ROOT, "real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv"))
mg = tr.merge(trl, on="index")
losscols = [c for c in mg.columns if c.startswith("metric/")]
ml = mg[losscols].mean(axis=1)
mix_pen = {"m_min(best_observed)": float(ml.min()/ml.min()), "m_uniform": 4.6976/4.7220,
           "m_worst_observed": float(ml.max()/ml.min()),
           "loss_min": float(ml.min()), "loss_max": float(ml.max())}
with open(os.path.join(OUT, "mixture_penalty_range.json"), "w") as f:
    json.dump(mix_pen, f, indent=2)
print("配比惩罚范围:", mix_pen)

# ---------- 6. 最终广义标度律(B1锚定 + gamma移植) ----------
# L(N,D,Q,p) = E + A/N^a + B*m(p) / (D^b * (Q/Q0)^g)
# 口径: B1的Pythia数据对应基线质量Q0=0.55(Pile域Q均值), m(p*)=1
final = {"E": pc["E"], "A": pc["A"], "alpha": pc["alpha"], "B": pc["B"], "beta": pc["beta"],
         "gamma_Q": float(g6), "Q0_baseline": 0.55,
         "note": "L = E + A/N^a + B*m(p)/(D^b*(Q/Q0)^g); m(p*)=1, m(worst)=mixture_penalty_range",
         "m_worst": mix_pen["m_worst_observed"], "Deff_exponent_Q": float(g6/pc["beta"])}
with open(os.path.join(OUT, "final_generalized_law.json"), "w") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print("最终广义标度律:", final)

# ---------- 7. 弹性与替代分析(最终参数) ----------
E, A, a, Bb, b, g = pc["E"], pc["A"], pc["alpha"], pc["B"], pc["beta"], float(g6)
Q0 = 0.55
refs = [(0.4, 8, 0.55), (1.0, 20, 0.55), (7.0, 140, 0.55), (12.0, 300, 0.55), (70.0, 1400, 0.55)]
rows = []
for N0, D0, Qv in refs:
    P_ = A/N0**a
    S = Bb/(D0**b * (Qv/Q0)**g)
    L0 = E + P_ + S
    eN, eD, eQ = -a*P_/L0, -b*S/L0, -g*S/L0
    # 替代: ΔL=0 -> a*P_*ΔN/N = g*S*ΔQ/Q -> ΔN/N = (g*S)/(a*P_) * ΔQ/Q
    ratio = (g*S)/(a*P_)
    # Q+0.1 (相对Q0 +18.18%) 等价的N相对增幅
    dN_rel = ratio * (0.1/Q0)
    rows.append({"N_B": N0, "D_B": D0, "Q": Qv, "L_pred": L0,
                 "share_E": E/L0, "share_param": P_/L0, "share_data": S/L0,
                 "elasticity_N": eN, "elasticity_D": eD, "elasticity_Q": eQ,
                 "sub_ratio_dNoverN_per_dQoverQ": ratio,
                 "Q_plus_0.1_equiv_N_growth_pct": dN_rel*100})
el = pd.DataFrame(rows)
el.to_csv(os.path.join(OUT, "elasticity_analysis_final.csv"), index=False)
print(el.round(4).to_string())

# 边际效用: 每增加1%投入的Loss下降 (在N=1B,D=20B,Q=0.55)
N0, D0 = 1.0, 20.0
P_ = A/N0**a; S = Bb/(D0**b)
dL_dN_1pct = -a*P_*0.01; dL_dD_1pct = -b*S*0.01; dL_dQ_1pct = -g*S*0.01
print(f"N=1B,D=20B,Q=0.55: 各因素+1% -> ΔL: N {dL_dN_1pct:.5f}, D {dL_dD_1pct:.5f}, Q {dL_dQ_1pct:.5f}")

# ---------- 8. 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7, 5))
pr7all = f1_pred(x6, nq7.N_params_B.values, nq7.D_tokens_B.values, nq7.Q_score.values)
ax.scatter(nq7.val_loss, pr7all, s=14, alpha=0.6, color="#2E86C1", label="B7 验证 (R²=%.3f)" % r2_7)
pr6all = f1_pred(x6, nq6.N_params_B.values, nq6.D_tokens_B.values, nq6.Q_score.values)
ax.scatter(nq6.val_loss, pr6all, s=10, alpha=0.4, color="#28B463", label="B6 拟合")
ax.plot([2, 3.8], [2, 3.8], "k--", lw=1)
ax.set_xlabel("观测 val_loss"); ax.set_ylabel("预测 val_loss")
ax.set_title("广义标度律(F1): B6拟合/B7验证"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_f1_validation.png"), dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
sub6 = nq6[(nq6.N_params_B == 0.07) & (nq6.D_tokens_B == 10)].sort_values("Q_score")
axes[0].plot(sub6.Q_score, sub6.val_loss, "o-", color="#28B463")
axes[0].set_title("B6/B7: Q↑ → Loss↓ (正常方向)\nN=0.07B, D=10B")
axes[0].set_xlabel("Q"); axes[0].set_ylabel("val_loss")
sub8 = nq8[(nq8.N_params_B == 0.07) & (nq8.D_tokens_B == 5)].sort_values("Q_score")
axes[1].plot(sub8.Q_score, sub8.val_loss, "s-", color="#C0392B")
axes[1].set_title("B8: Q↑ → Loss↑ (方向反转, 判为不可信)\nN=0.07B, D=5B")
axes[1].set_xlabel("Q"); axes[1].set_ylabel("val_loss")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_b8_inversion.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(el)); w = 0.25
ax.bar(x-w, -el["elasticity_N"], w, label="|e_N| 参数", color="#2E86C1")
ax.bar(x,   -el["elasticity_D"], w, label="|e_D| 数据", color="#28B463")
ax.bar(x+w, -el["elasticity_Q"], w, label="|e_Q| 质量", color="#E67E22")
ax.set_xticks(x); ax.set_xticklabels([f"N={r:g}B" for r in el["N_B"]])
ax.set_ylabel("弹性绝对值"); ax.set_title("最终广义标度律: 各因素弹性(参考点D=20N)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_elasticity_final.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 5))
dd = np.logspace(0, 3, 200)
for Qv, c in [(0.3, "#C0392B"), (0.55, "#E67E22"), (0.8, "#28B463"), (1.0, "#2E86C1")]:
    ax.plot(dd, E + A/1.0**a + Bb/(dd**b * (Qv/Q0)**g), color=c, label=f"Q={Qv}")
ax.set_xscale("log"); ax.set_xlabel("D (十亿tokens)"); ax.set_ylabel("L")
ax.set_title("最终广义标度律 (N=1B): 质量提升等效于数据量右移"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_final_surface.png"), dpi=150); plt.close()
print("DONE_Q2_FIX")

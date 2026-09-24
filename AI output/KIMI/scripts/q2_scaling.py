# -*- coding: utf-8 -*-
"""
问题二：经典标度律拟合 + 纳入Q与p的广义标度律 + 弹性分析
数据: B1-B10
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

ROOT = r"C:/Users/dkyyt/Desktop/F题"
B = os.path.join(ROOT, "real_attachments", "B_scaling_laws")
OUT = os.path.join(ROOT, "outputs", "q2_scaling")
os.makedirs(OUT, exist_ok=True)

pyt = pd.read_csv(os.path.join(B, "pythia_training_log_existing.csv"))
cer = pd.read_csv(os.path.join(B, "cerebras_training_log.csv"))
base = pd.read_csv(os.path.join(B, "scaling_baseline.csv"))
pub = pd.read_csv(os.path.join(B, "published_scaling_data.csv"))
nq6 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment.csv"))
nq7 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_expanded.csv"))
nq8 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_large.csv"))
bigM = pd.read_csv(os.path.join(B, "supplementary_large_models.csv"))
bigL = pd.read_csv(os.path.join(B, "supplementary_large_baseline.csv"))

print("B1 Pythia:", pyt.shape, "N范围", pyt.N_params_B.min(), "-", pyt.N_params_B.max())
print("B6/B7/B8:", nq6.shape, nq7.shape, nq8.shape)

# ============ 1. 经典标度律 L = E + A/N^a + B/D^b (B1主拟合) ============
def classic_resid(theta, N, D, L):
    E, A, a, Bb, b = np.exp(theta[0]), np.exp(theta[1]), theta[2], np.exp(theta[3]), theta[4]
    return np.log(E + A / N**a + Bb / D**b) - np.log(L)

def fit_classic(N, D, L, seed=0):
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(30):
        x0 = np.array([np.log(1.0)+rng.normal(0,1), np.log(100)+rng.normal(0,2),
                       0.3+rng.normal(0,0.2), np.log(100)+rng.normal(0,2), 0.3+rng.normal(0,0.2)])
        x0 = np.clip(x0, [-10,-20,0.01,-20,0.01], [5,20,2,20,2])
        r = least_squares(classic_resid, x0, args=(N, D, L), loss="soft_l1", f_scale=0.1,
                          bounds=([-10,-20,0.01,-20,0.01],[5,20,2,20,2]), max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    t = best.x
    return {"E": np.exp(t[0]), "A": np.exp(t[1]), "alpha": t[2], "B": np.exp(t[3]), "beta": t[4],
            "cost": best.cost}

N1, D1, L1 = pyt.N_params_B.values, pyt.D_tokens_B.values, pyt.val_loss.values
pc = fit_classic(N1, D1, L1)
print("经典标度律参数(B1):", {k: round(v, 4) for k, v in pc.items()})

def classic_pred(p, N, D):
    return p["E"] + p["A"]/N**p["alpha"] + p["B"]/D**p["beta"]

# Bootstrap置信区间
boot = []
rng = np.random.default_rng(1)
n = len(L1)
for k in range(200):
    idx = rng.integers(0, n, n)
    try:
        pb = fit_classic(N1[idx], D1[idx], L1[idx], seed=k+1)
        boot.append([pb["E"], pb["A"], pb["alpha"], pb["B"], pb["beta"]])
    except Exception:
        pass
boot = np.array(boot)
ci = pd.DataFrame({"param": ["E","A","alpha","B","beta"],
    "estimate": [pc["E"],pc["A"],pc["alpha"],pc["B"],pc["beta"]],
    "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0)})
ci.to_csv(os.path.join(OUT, "classic_params_bootstrap.csv"), index=False)
print(ci.round(4).to_string())

# ============ 2. 经典律验证: B2族外 / B3轨迹 / B4跨族 / B5文献 ============
def evalset(name, N, D, L, p):
    pr = classic_pred(p, N, D)
    ss = 1 - np.sum((L-pr)**2)/np.sum((L-L.mean())**2)
    rmse = float(np.sqrt(np.mean((L-pr)**2)))
    rho = spearmanr(pr, L)[0]
    return {"dataset": name, "n": len(L), "R2": ss, "RMSE": rmse, "spearman": rho,
            "mean_obs": float(L.mean()), "mean_pred": float(pr.mean())}

val_rows = [evalset("B1_Pythia(拟合)", N1, D1, L1, pc),
            evalset("B2_Cerebras(族外)", cer.N_params_B.values, cer.D_tokens_B.values, cer.val_loss.values, pc),
            evalset("B4_跨族收敛点", base.N_params_B.values, base.D_tokens_B.values, base.val_loss.values, pc),
            evalset("B5_文献基准", pub.N_params_B.values, pub.D_tokens_B.values, pub.val_loss.values, pc)]
# B3 轨迹: 8个文件逐模型R2
traj_dir = os.path.join(B, "training_trajectories")
for f in sorted(os.listdir(traj_dir)):
    t = pd.read_csv(os.path.join(traj_dir, f))
    val_rows.append(evalset("B3_"+f.replace("_trajectory.csv",""), t.N_params_B.values, t.D_tokens_B.values, t.val_loss.values, pc))
val = pd.DataFrame(val_rows)
val.to_csv(os.path.join(OUT, "classic_validation.csv"), index=False)
print(val.round(4).to_string())

# B9/B10 大模型外推
bm = bigM.merge(bigL, left_on="model_name", right_on="family", how="inner", suffixes=("","_est"))
print("B9∩B10:", bm.shape)
pr_big = classic_pred(pc, bm.N_params_B.values, bm.D_tokens_B.values)
big_eval = evalset("B10_大模型估算(100B+)", bm.N_params_B.values, bm.D_tokens_B.values, bm.val_loss.values, pc)
big_df = pd.DataFrame([big_eval])
big_df.to_csv(os.path.join(OUT, "large_model_extrapolation.csv"), index=False)
print(big_df.round(4).to_string())

# ============ 3. 广义标度律: 纳入Q (B6拟合, B7验证, B8外推) ============
# 候选形式 (Q在[0,1], Q0=参考质量; Q=1完美时数据项退化为经典形式):
# F1: L = E + A/N^a + B/(D^b * Q^g)        有效数据 D_eff = D * Q^(g/b)
# F2: L = E + A/N^a + B/D^b + G/Q^g        加性质量项
# F3: L = E + A/N^a + B/(D*(1+l*Q))^b      线性有效数据
def gen_resid_factory(form):
    def f(theta, N, D, Q, L):
        E, A, a, Bb, b = np.exp(theta[0]), np.exp(theta[1]), theta[2], np.exp(theta[3]), theta[4]
        if form == "F1":
            g = theta[5]
            return np.log(E + A/N**a + Bb/(D**b * Q**g)) - np.log(L)
        if form == "F2":
            G, g = np.exp(theta[5]), theta[6]
            return np.log(E + A/N**a + Bb/D**b + G/Q**g) - np.log(L)
        if form == "F3":
            l = np.exp(theta[5])
            return np.log(E + A/N**a + Bb/(D*(1+l*Q))**b) - np.log(L)
    return f

def fit_gen(form, N, D, Q, L, seed=0):
    f = gen_resid_factory(form)
    k = {"F1": 6, "F2": 7, "F3": 6}[form]
    lo = [-10,-20,0.01,-20,0.01] + [-5]*(k-5)
    hi = [5,20,2,20,2] + [10]*(k-5)
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(25):
        x0 = np.array([np.log(1.0), np.log(100), 0.3, np.log(100), 0.3] + [0.5]*(k-5)) \
             + rng.normal(0, 0.5, k)
        x0 = np.clip(x0, lo, hi)
        r = least_squares(f, x0, args=(N, D, Q, L), loss="soft_l1", f_scale=0.1,
                          bounds=(lo, hi), max_nfev=30000)
        if best is None or r.cost < best.cost:
            best = r
    return best

Nq, Dq, Qq, Lq = nq6.N_params_B.values, nq6.D_tokens_B.values, nq6.Q_score.values, nq6.val_loss.values
fits, cmp_rows = {}, []
for form in ["F1", "F2", "F3"]:
    r = fit_gen(form, Nq, Dq, Qq, Lq)
    fits[form] = r
    npts, kp = len(Lq), len(r.x)
    aic = 2*kp + npts*np.log(2*r.cost/npts)  # cost = 0.5*sum(resid^2)
    cmp_rows.append({"form": form, "params": np.round(r.x, 4).tolist(), "cost": r.cost, "AIC": aic})
cmpf = pd.DataFrame(cmp_rows)
cmpf.to_csv(os.path.join(OUT, "generalized_form_comparison.csv"), index=False)
print(cmpf.to_string())

best_form = cmpf.sort_values("AIC").iloc[0]["form"]
print("最优形式:", best_form)
rb = fits[best_form]

def gen_pred(form, x, N, D, Q):
    E, A, a, Bb, b = np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), x[4]
    if form == "F1": return E + A/N**a + Bb/(D**b * Q**x[5])
    if form == "F2": return E + A/N**a + Bb/D**b + np.exp(x[5])/Q**x[6]
    if form == "F3": return E + A/N**a + Bb/(D*(1+np.exp(x[5])*Q))**b

def eval_gen(name, df):
    pr = gen_pred(best_form, rb.x, df.N_params_B.values, df.D_tokens_B.values, df.Q_score.values)
    L = df.val_loss.values
    return {"dataset": name, "n": len(L),
            "R2": 1-np.sum((L-pr)**2)/np.sum((L-L.mean())**2),
            "RMSE": float(np.sqrt(np.mean((L-pr)**2))),
            "spearman": spearmanr(pr, L)[0]}
gev = pd.DataFrame([eval_gen("B6(拟合)", nq6), eval_gen("B7(验证)", nq7), eval_gen("B8(大规模外推)", nq8)])
gev.to_csv(os.path.join(OUT, "generalized_validation.csv"), index=False)
print(gev.round(4).to_string())

# 广义律参数导出(可读形式)
E_, A_, a_, B_, b_ = np.exp(rb.x[0]), np.exp(rb.x[1]), rb.x[2], np.exp(rb.x[3]), rb.x[4]
extra = rb.x[5:]
params_out = {"form": best_form, "E": E_, "A": A_, "alpha": a_, "B": B_, "beta": b_}
if best_form == "F1":
    params_out["gamma_Q"] = float(extra[0])
    params_out["Deff_exponent"] = float(extra[0]/b_)  # D_eff = D*Q^(g/b)
elif best_form == "F2":
    params_out["G"] = float(np.exp(extra[0])); params_out["gamma_Q"] = float(extra[1])
else:
    params_out["lambda_Q"] = float(np.exp(extra[0]))
with open(os.path.join(OUT, "generalized_params.json"), "w") as f:
    json.dump(params_out, f, indent=2, ensure_ascii=False)
print(params_out)

# ============ 4. 纳入配比p: 有效数据统一形式 ============
# 由问题一GBM配比模型给出配比惩罚 m(p) = L_gbm(p)/L_gbm(p*) >= 1
# 在固定(N,D,Q)下, 数据项乘m(p) 等价于 D_eff = D / m(p)^(1/b)
# 计算几个代表性配比的m(p)供问题三使用
import pickle
sys_path = os.path.join(ROOT, "outputs", "q1_mixture")
cand = pd.read_csv(os.path.join(sys_path, "optimal_mixture_gbm.csv"))
# 重新加载GBM太贵, 用保存的预测: candidate_pred_loss.csv 中 gbm_opt=4.722, uniform=4.6976, train_mean=4.6839
# 直接用数值:
Lstar = 4.7220
m_tbl = pd.DataFrame({
    "mixture": ["p_optimal", "uniform", "train_mean", "best_observed_train"],
    "loss_pred_gbm": [4.7220, 4.6976, 4.6839, 4.7596]})
m_tbl["m_penalty"] = m_tbl["loss_pred_gbm"] / Lstar
m_tbl["Deff_factor"] = m_tbl["m_penalty"] ** (-1.0/b_)
m_tbl.to_csv(os.path.join(OUT, "mixture_penalty_factors.csv"), index=False)
print(m_tbl.round(4).to_string())

# ============ 5. 弹性与边际效用分析 ============
# 在参考点 (N0, D0, Q0) 处:
# 数据项 S = B/D^b (经典) 或 B/(D^b Q^g); 参数项 P_ = A/N^a
refs = [(0.4, 8, 0.55), (1.0, 20, 0.55), (7.0, 140, 0.55), (12.0, 300, 0.55)]
el_rows = []
for N0, D0, Q0 in refs:
    P_ = A_/N0**a_
    if best_form == "F1":
        g_ = extra[0]
        S = B_/(D0**b_ * Q0**g_)
        dQ_factor = g_/Q0
    else:
        S = B_/D0**b_; dQ_factor = np.nan
    L0 = E_ + P_ + S
    # 弹性: e_N = -a_*P_/L0, e_D = -b_*S/L0, e_Q = -g_*S/L0
    el_rows.append({"N_B": N0, "D_B": D0, "Q": Q0, "L": L0, "irreducible_share": E_/L0,
                    "param_share": P_/L0, "data_share": S/L0,
                    "elasticity_N": -a_*P_/L0, "elasticity_D": -b_*S/L0,
                    "elasticity_Q": (-dQ_factor*S/L0) if best_form=="F1" else np.nan,
                    # 替代条件: ΔQ/Q 与 ΔN/N 使ΔL=0 -> ΔN/N = (g*S)/(a*P_) * ΔQ/Q
                    "sub_ratio_N_over_Q": (extra[0]*S)/(a_*P_) if best_form=="F1" else np.nan})
el = pd.DataFrame(el_rows)
el.to_csv(os.path.join(OUT, "elasticity_analysis.csv"), index=False)
print(el.round(4).to_string())

# 质量提升0.1 等价参数增长倍数 (Q: 0.55->0.65, 即相对+18.2%)
for r in el_rows:
    if best_form == "F1":
        rel_Q = 0.1/0.55
        rel_N = r["sub_ratio_N_over_Q"] * rel_Q
        print(f"N={r['N_B']}B: Q+0.1 等价于 N 增加 {rel_N*100:.1f}% (N×{1+rel_N:.2f})")

# ============ 6. 图 ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7.5, 5))
for Nfix, c in [(0.07, "#1f77b4"), (0.41, "#2ca02c"), (1.04, "#ff7f0e"), (12.0, "#d62728")]:
    sub = pyt[np.abs(pyt.N_params_B - Nfix) < 0.05].sort_values("D_tokens_B")
    if len(sub) == 0: continue
    dd = np.linspace(sub.D_tokens_B.min(), sub.D_tokens_B.max(), 100)
    ax.scatter(sub.D_tokens_B, sub.val_loss, s=6, color=c, alpha=0.4)
    ax.plot(dd, classic_pred(pc, np.full_like(dd, Nfix), dd), color=c,
            label=f"N={Nfix}B")
ax.set_xscale("log"); ax.set_xlabel("D (十亿tokens)"); ax.set_ylabel("val_loss")
ax.set_title("经典标度律拟合 (B1 Pythia)"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_classic_fit.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for nm, df, mk in [("B2 Cerebras", cer, "s"), ("B4 跨族", base, "^"), ("B5 文献", pub, "D")]:
    pr = classic_pred(pc, df.N_params_B.values, df.D_tokens_B.values)
    ax.scatter(df.val_loss, pr, s=22, alpha=0.7, marker=mk, label=nm)
lo, hi = 1.5, 5.5
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("观测 val_loss"); ax.set_ylabel("预测 val_loss")
ax.set_title("经典律: 族外/跨族/文献验证"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_classic_validation.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 5))
if best_form == "F1":
    for Qfix, c in [(0.3, "#d62728"), (0.55, "#ff7f0e"), (0.8, "#2ca02c"), (1.0, "#1f77b4")]:
        sub = nq8[np.abs(nq8.Q_score - Qfix) < 0.03]
        if len(sub):
            ax.scatter(sub.D_tokens_B, sub.val_loss, s=5, color=c, alpha=0.3)
        dd = np.logspace(0, 3.5, 100)
        ax.plot(dd, gen_pred("F1", rb.x, np.full_like(dd, 1.0), dd, np.full_like(dd, Qfix)),
                color=c, label=f"Q={Qfix}")
    ax.set_xscale("log")
ax.set_xlabel("D (十亿tokens)"); ax.set_ylabel("val_loss")
ax.set_title("广义标度律(N=1B): 质量Q对Loss曲面的影响"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_quality_surface.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(el)); w = 0.25
ax.bar(x-w, -el["elasticity_N"], w, label="|e_N| 参数弹性", color="#2E86C1")
ax.bar(x,   -el["elasticity_D"], w, label="|e_D| 数据弹性", color="#28B463")
if best_form == "F1":
    ax.bar(x+w, -el["elasticity_Q"], w, label="|e_Q| 质量弹性", color="#E67E22")
ax.set_xticks(x); ax.set_xticklabels([f"N={r}B" for r in el["N_B"]])
ax.set_ylabel("弹性绝对值"); ax.set_title("各因素弹性对比(参考点D=20N, Q=0.55)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_elasticity.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(bm.N_params_B, bm.val_loss, s=22, alpha=0.6, label="B10 估算Loss", color="#8E44AD")
ax.scatter(bm.N_params_B, pr_big, s=22, alpha=0.6, label="经典律外推", color="#E67E22")
ax.set_xscale("log"); ax.set_xlabel("N (B)"); ax.set_ylabel("val_loss")
ax.set_title("百亿参数以上外推 vs B10估算"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_large_extrapolation.png"), dpi=150); plt.close()

print("DONE_Q2")

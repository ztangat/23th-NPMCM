# -*- coding: utf-8 -*-
"""
问题一(下·修正)：GBM配比模型寻优 + 跨尺度一致性 + 外推稳健性补充
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = r"C:/Users/dkyyt/Desktop/F题"
TAB = os.path.join(ROOT, "real_attachments", "A_data_value", "regmix_tables")
OUT = os.path.join(ROOT, "outputs", "q1_mixture")

DOMAINS17 = ["arxiv","freelaw","nih_exporter","pubmed_central","wikipedia_en","dm_mathematics",
             "github","philpapers","stackexchange","enron_emails","gutenberg_pg_19","pile_cc",
             "ubuntu_irc","europarl","hackernews","pubmed_abstracts","uspto_backgrounds"]
MIXCOLS = [f"train_the_pile_{d}" for d in DOMAINS17]
LOSS_DOMAINS = ["arxiv","freelaw","pubmed_central","wikipedia_en","dm_mathematics","github",
                "stackexchange","gutenberg_pg_19","pile_cc","ubuntu_irc","hackernews",
                "pubmed_abstracts","uspto_backgrounds"]
LOSSCOLS = [f"metric/the_pile_{d}_val_loss" for d in LOSS_DOMAINS]

def load_pair(mix_file, loss_file):
    m = pd.read_csv(os.path.join(TAB, mix_file))
    l = pd.read_csv(os.path.join(TAB, loss_file))
    return m.merge(l, on="index", validate="one_to_one")

train = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
t1m   = load_pair("test_mixture_1m.csv", "test_pile_loss_1m.csv")
t60m  = load_pair("test_mixture_60m.csv", "test_pile_loss_60m.csv")
t1b   = load_pair("test_mixture_1B.csv", "test_pile_loss_1B.csv")
e10b  = load_pair("est_mixture_10b.csv", "est_pile_loss_10b.csv")
e70b  = load_pair("est_mixture_70b.csv", "est_pile_loss_70b.csv")

Xtr, Ytr = train[MIXCOLS].values, train[LOSSCOLS].values

# ---------- 1. GBM 集成模型(13目标) ----------
gbms = []
for j in range(13):
    g = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08, random_state=0)
    g.fit(Xtr, Ytr[:, j])
    gbms.append(g)

def pred_mean_gbm(P):
    P = np.atleast_2d(P)
    return np.mean([g.predict(P) for g in gbms], axis=0)

# ---------- 2. 基于GBM的配比寻优(SLSQP多起点, 数值梯度) ----------
def solve_opt(cap=None, nstarts=24, seed=1):
    rng = np.random.default_rng(seed)
    bounds = [(0.0, cap if cap else 1.0)] * 17
    cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
    best = None
    for k in range(nstarts):
        x0 = rng.dirichlet(np.ones(17))
        if cap: x0 = np.minimum(x0, cap); x0 /= x0.sum()
        r = minimize(lambda p: pred_mean_gbm(p)[0], x0, method="SLSQP",
                     bounds=bounds, constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    p = best.x.copy(); p[p < 1e-4] = 0; p /= p.sum()
    return p, best.fun

p_gbm, f_gbm = solve_opt(cap=None)
p_gbm_cap, f_gbm_cap = solve_opt(cap=0.30)
print("GBM最优(无上限) mean_loss_pred =", round(f_gbm, 4))
print("GBM最优(上限0.30) mean_loss_pred =", round(f_gbm_cap, 4))

p_uni = np.ones(17) / 17
p_best_obs = train.loc[train[LOSSCOLS].mean(axis=1).idxmin(), MIXCOLS].values.astype(float)
p_mean = train[MIXCOLS].mean().values.astype(float)

cand = pd.DataFrame({"domain": DOMAINS17,
                     "p_gbm_opt": p_gbm, "p_gbm_opt_cap30": p_gbm_cap,
                     "p_uniform": p_uni, "p_best_observed_train": p_best_obs, "p_train_mean": p_mean})
cand.to_csv(os.path.join(OUT, "optimal_mixture_gbm.csv"), index=False)
print(cand.round(4).to_string())

pred_tbl = pd.DataFrame({
    "candidate": ["gbm_opt", "gbm_opt_cap30", "uniform", "best_observed_train", "train_mean"],
    "pred_mean_loss_GBM": [pred_mean_gbm(p)[0] for p in [p_gbm, p_gbm_cap, p_uni, p_best_obs, p_mean]],
})
pred_tbl.to_csv(os.path.join(OUT, "candidate_pred_loss.csv"), index=False)
print(pred_tbl.round(4).to_string())

# ---------- 3. 候选配比在检验/外推集的近邻实测 ----------
def nn_eval(df, p):
    X = df[MIXCOLS].values
    y = df[LOSSCOLS].values.mean(axis=1)
    d = np.linalg.norm(X - p, axis=1)
    i = int(np.argmin(d))
    return {"nearest_index": int(df["index"].iloc[i]), "dist": round(float(d[i]), 4),
            "actual_mean_loss": round(float(y[i]), 4),
            "loss_rank_pct": round(float((y < y[i]).mean()), 4)}
rows = []
for nm, df in [("test_1M", t1m), ("test_60M", t60m), ("test_1B", t1b), ("est_10B", e10b), ("est_70B", e70b)]:
    for cn, p in [("gbm_opt", p_gbm), ("gbm_opt_cap30", p_gbm_cap), ("uniform", p_uni),
                  ("best_observed_train", p_best_obs)]:
        rows.append({"eval_set": nm, "candidate": cn, **nn_eval(df, p)})
nnr = pd.DataFrame(rows)
nnr.to_csv(os.path.join(OUT, "robustness_eval_gbm.csv"), index=False)
print(nnr.to_string())

# ---------- 4. 跨尺度一致性: z-score合并模型 ----------
frames = []
for nm, df in [("train_1M", train), ("test_1M", t1m), ("test_60M", t60m), ("test_1B", t1b)]:
    Y = df[LOSSCOLS].values
    Z = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    frames.append(pd.DataFrame({"scale": nm, "z_mean_loss": Z.mean(axis=1),
                                **{d: df[c].values for d, c in zip(DOMAINS17, MIXCOLS)}}))
alld = pd.concat(frames, ignore_index=True)
tr = alld[alld["scale"] == "train_1M"]
gz = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08, random_state=0)
gz.fit(tr[DOMAINS17].values, tr["z_mean_loss"].values)
cons_rows = []
for nm in ["test_1M", "test_60M", "test_1B"]:
    sub = alld[alld["scale"] == nm]
    pr = gz.predict(sub[DOMAINS17].values)
    rho, pv = spearmanr(pr, sub["z_mean_loss"].values)
    cons_rows.append({"eval_set": nm, "spearman": rho, "p_value": pv, "n": len(sub)})
cons_df = pd.DataFrame(cons_rows)
cons_df.to_csv(os.path.join(OUT, "cross_scale_consistency.csv"), index=False)
print(cons_df.round(4).to_string())

# ---------- 5. 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5))
od = cand.sort_values("p_gbm_opt", ascending=False)
x = np.arange(17); w = 0.35
ax.bar(x - w/2, od["p_gbm_opt"], w, label="GBM最优(无上限)", color="#2E86C1")
ax.bar(x + w/2, od["p_gbm_opt_cap30"], w, label="GBM最优(单域≤30%)", color="#E67E22")
ax.set_xticks(x); ax.set_xticklabels(od["domain"], rotation=45, ha="right")
ax.axhline(1/17, color="gray", ls="--", lw=1, label="均匀配比")
ax.set_title("最优领域配比(GBM模型, 最小化13域平均Loss)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_optimal_mixture_gbm.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
pr = np.array([pred_mean_gbm(X) for X in [t1m[MIXCOLS].values]])[0]
true = t1m[LOSSCOLS].values.mean(axis=1)
ax.scatter(true, pr, s=14, alpha=0.6, color="#28B463")
lo, hi = min(true.min(), pr.min()), max(true.max(), pr.max())
ax.plot([lo, hi], [lo, hi], "r--")
r2 = r2_score(true, pr)
ax.set_xlabel("真实平均Loss"); ax.set_ylabel("GBM预测平均Loss")
ax.set_title(f"检验集test_1M: GBM预测 vs 真实 (R²={r2:.3f})")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_gbm_pred_vs_true_1m.png"), dpi=150); plt.close()
print("test_1M GBM mean-loss R2 =", round(r2, 4))
print("DONE_Q1_MIXTURE_V2")

# -*- coding: utf-8 -*-
"""
问题一(下)：领域配比建模 (RegMix 配方实验 A4-A15)
目标: 建立17域配比p -> 13域验证Loss的定量模型, 检验集验证, 外推稳健性
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = r"C:/Users/dkyyt/Desktop/F题"
TAB = os.path.join(ROOT, "real_attachments", "A_data_value", "regmix_tables")
OUT = os.path.join(ROOT, "outputs", "q1_mixture")
Q1Q = os.path.join(ROOT, "outputs", "q1_quality")
os.makedirs(OUT, exist_ok=True)

DOMAINS17 = ["arxiv","freelaw","nih_exporter","pubmed_central","wikipedia_en","dm_mathematics",
             "github","philpapers","stackexchange","enron_emails","gutenberg_pg_19","pile_cc",
             "ubuntu_irc","europarl","hackernews","pubmed_abstracts","uspto_backgrounds"]
MIXCOLS = [f"train_the_pile_{d}" for d in DOMAINS17]
LOSS_DOMAINS = ["arxiv","freelaw","pubmed_central","wikipedia_en","dm_mathematics","github",
                "stackexchange","gutenberg_pg_19","pile_cc","ubuntu_irc","hackernews",
                "pubmed_abstracts","uspto_backgrounds"]  # 13个有Loss的域
LOSSCOLS = [f"metric/the_pile_{d}_val_loss" for d in LOSS_DOMAINS]
NOLOSS = ["nih_exporter","philpapers","enron_emails","europarl"]

def load_pair(mix_file, loss_file):
    m = pd.read_csv(os.path.join(TAB, mix_file))
    l = pd.read_csv(os.path.join(TAB, loss_file))
    df = m.merge(l, on="index", validate="one_to_one")
    return df

train = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
t1m   = load_pair("test_mixture_1m.csv", "test_pile_loss_1m.csv")
t60m  = load_pair("test_mixture_60m.csv", "test_pile_loss_60m.csv")
t1b   = load_pair("test_mixture_1B.csv", "test_pile_loss_1B.csv")
e10b  = load_pair("est_mixture_10b.csv", "est_pile_loss_10b.csv")
e70b  = load_pair("est_mixture_70b.csv", "est_pile_loss_70b.csv")
print({k: v.shape for k, v in [("train",train),("t1m",t1m),("t60m",t60m),("t1b",t1b),("e10b",e10b),("e70b",e70b)]})

# 单纯形检验
sums = train[MIXCOLS].sum(axis=1)
print("train配比和: min={:.6f} max={:.6f}".format(sums.min(), sums.max()))
with open(os.path.join(OUT,"simplex_check.txt"),"w") as f:
    for nm, df in [("train",train),("t1m",t1m),("t60m",t60m),("t1b",t1b),("e10b",e10b),("e70b",e70b)]:
        s = df[MIXCOLS].sum(axis=1)
        f.write(f"{nm}: n={len(df)}, 配比和 min={s.min():.6f}, max={s.max():.6f}, "
                f"负分量数={(df[MIXCOLS]<0).sum().sum()}\n")

Xtr = train[MIXCOLS].values
Ytr = train[LOSSCOLS].values
train["mean_loss"] = Ytr.mean(axis=1)

# ---------- 质量评分引入(论证性试验) ----------
# 用A16映射: direct/near_direct域取问题一领域级Q(熵权), 其余域取全样本均值
domq = pd.read_csv(os.path.join(Q1Q, "domain_Q_summary.csv"))
qmap_raw = {"arxiv":"arxiv","github":"github","stackexchange":"stackexchange",
            "wikipedia_en":"wikipedia","gutenberg_pg_19":"book","pile_cc":"commoncrawl"}
qz = dict(zip(domq["domain"], domq["Q_primary"]))
Q17 = {}
qvals = [qz[v] for v in qmap_raw.values() if v in qz]
qdefault = float(np.mean(qvals))
for d in DOMAINS17:
    if d in qmap_raw and qmap_raw[d] in qz:
        Q17[d] = qz[qmap_raw[d]]
    else:
        Q17[d] = qdefault
Q17_vec = np.array([Q17[d] for d in DOMAINS17])
pd.DataFrame({"domain":DOMAINS17,"Q_used":Q17_vec,
              "source":["direct/near_direct" if d in qmap_raw else "default(均值)" for d in DOMAINS17]}
             ).to_csv(os.path.join(OUT,"domain_Q_for_mixture.csv"), index=False)
print("域级Q:", {d: round(Q17[d],3) for d in DOMAINS17})

# 特征方案: F0=原始17维; F1=+质量调整项 p_i*Q_i (17维附加 -> 用p*Q向量与p拼接会共线, 改为附加标量特征 p·Q)
def feats(X, withQ=False):
    if not withQ: return X
    qdot = (X * Q17_vec).sum(axis=1, keepdims=True)  # 混合数据的加权平均质量
    return np.hstack([X, qdot])

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def fit_eval(withQ, alpha=1.0):
    """对13个Loss各自训练Ridge, 返回各检验集R2/RMSE + 模型"""
    models, rows = [], []
    for j, ld in enumerate(LOSS_DOMAINS):
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(feats(Xtr, withQ), Ytr[:, j])
        models.append(mdl)
        rec = {"target": ld, "withQ": withQ, "train_R2": mdl.score(feats(Xtr, withQ), Ytr[:, j])}
        for nm, df in [("test_1M", t1m), ("test_60M", t60m), ("test_1B", t1b)]:
            Xp = df[MIXCOLS].values; yp = df[LOSSCOLS].values[:, j]
            pr = mdl.predict(feats(Xp, withQ))
            rec[f"{nm}_R2"] = r2_score(yp, pr)
            rec[f"{nm}_RMSE"] = float(np.sqrt(np.mean((yp-pr)**2)))
        # 交叉验证
        cv = KFold(5, shuffle=True, random_state=0)
        r2s = []
        for tri, tei in cv.split(Xtr):
            mm = Ridge(alpha=alpha).fit(feats(Xtr[tri], withQ), Ytr[tri, j])
            r2s.append(r2_score(Ytr[tei, j], mm.predict(feats(Xtr[tei], withQ))))
        rec["cv5_R2"] = float(np.mean(r2s))
        rows.append(rec)
    return models, pd.DataFrame(rows)

m0, ev0 = fit_eval(False)
m1, ev1 = fit_eval(True)
ev = pd.concat([ev0, ev1])
ev.to_csv(os.path.join(OUT, "regression_validation.csv"), index=False)
print(ev.round(3).to_string())

# 系数表(以无Q模型为准)
coef = pd.DataFrame({"domain": DOMAINS17})
for j, ld in enumerate(LOSS_DOMAINS):
    coef[ld] = m0[j].coef_
coef.to_csv(os.path.join(OUT, "ridge_coefficients.csv"), index=False)

# ---------- 非线性对照: 梯度提升 ----------
from sklearn.ensemble import HistGradientBoostingRegressor
gb_rows = []
for j, ld in enumerate(LOSS_DOMAINS):
    g = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08, random_state=0)
    g.fit(Xtr, Ytr[:, j])
    rec = {"target": ld, "train_R2": g.score(Xtr, Ytr[:, j])}
    for nm, df in [("test_1M", t1m), ("test_60M", t60m), ("test_1B", t1b)]:
        pr = g.predict(df[MIXCOLS].values)
        rec[f"{nm}_R2"] = r2_score(df[LOSSCOLS].values[:, j], pr)
    gb_rows.append(rec)
gb = pd.DataFrame(gb_rows)
gb.to_csv(os.path.join(OUT, "gbm_validation.csv"), index=False)
print(gb.round(3).to_string())

# ---------- 最优配比求解 (目标: 13域平均Loss最小, SLSQP多起点) ----------
def predict_mean_loss(p, models, withQ=False):
    p = np.asarray(p).reshape(1, -1)
    return float(np.mean([mdl.predict(feats(p, withQ))[0] for mdl in models]))

bounds = [(0.0, 1.0)] * 17
cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
best = None
rng = np.random.default_rng(42)
for k in range(40):
    x0 = rng.dirichlet(np.ones(17))
    r = minimize(lambda p: predict_mean_loss(p, m0), x0, method="SLSQP", bounds=bounds, constraints=cons,
                 options={"maxiter": 500, "ftol": 1e-10})
    if best is None or r.fun < best.fun:
        best = r
p_opt = best.x
p_opt[p_opt < 1e-4] = 0; p_opt = p_opt / p_opt.sum()
opt_df = pd.DataFrame({"domain": DOMAINS17, "p_optimal": p_opt})
# 对照: 均匀配比 & 训练集中实测最优 & 训练集平均配比
p_uni = np.ones(17)/17
p_best_obs = train.loc[train["mean_loss"].idxmin(), MIXCOLS].values.astype(float)
p_mean = train[MIXCOLS].mean().values.astype(float)
opt_df["p_uniform"] = p_uni
opt_df["p_best_observed_train"] = p_best_obs
opt_df["p_train_mean"] = p_mean
opt_df.to_csv(os.path.join(OUT, "optimal_mixture.csv"), index=False)
print(opt_df.round(4).to_string())

# ---------- 用检验集与外推表评估"最优配比"的稳健性 ----------
def eval_mixture_rank(df, p_list, names):
    """把候选配比"代入"检验表: 找检验集中与候选配比最近的配方, 比较其真实mean_loss"""
    X = df[MIXCOLS].values
    y = df[LOSSCOLS].values.mean(axis=1)
    res = {}
    for p, nm in zip(p_list, names):
        d = np.linalg.norm(X - p, axis=1)
        i = int(np.argmin(d))
        res[nm] = {"nearest_idx": int(df["index"].iloc[i]), "dist": float(d[i]), "mean_loss": float(y[i]),
                   "rank_pct": float((y < y[i]).mean())}
    res["test_median"] = {"nearest_idx": -1, "dist": np.nan, "mean_loss": float(np.median(y)), "rank_pct": 0.5}
    return res

rob_rows = []
for nm, df in [("test_1M", t1m), ("test_60M", t60m), ("test_1B", t1b), ("est_10B", e10b), ("est_70B", e70b)]:
    r = eval_mixture_rank(df, [p_opt, p_uni, p_best_obs, p_mean],
                          ["optimized_1M", "uniform", "best_observed_train", "train_mean"])
    for k, v in r.items():
        rob_rows.append({"eval_set": nm, "candidate": k, **v})
rob = pd.DataFrame(rob_rows)
rob.to_csv(os.path.join(OUT, "robustness_eval.csv"), index=False)
print(rob.round(4).to_string())

# 外推稳健性: 1M模型预测排序 vs est_10B/70B真实排序的Spearman相关
ext_rows = []
for nm, df in [("test_60M", t60m), ("test_1B", t1b), ("est_10B", e10b), ("est_70B", e70b)]:
    Xp, Yp = df[MIXCOLS].values, df[LOSSCOLS].values
    pred = np.mean([mdl.predict(feats(Xp, False)) for mdl in m0], axis=0)
    true = Yp.mean(axis=1)
    rho, pv = spearmanr(pred, true)
    ext_rows.append({"eval_set": nm, "spearman_pred_vs_true_meanloss": rho, "p_value": pv,
                     "n": len(df)})
ext_df = pd.DataFrame(ext_rows)
ext_df.to_csv(os.path.join(OUT, "extrapolation_rank_stability.csv"), index=False)
print(ext_df.round(4).to_string())

# 域贡献分析: 系数解释 + 单域消融(将某域置0重归一化后预测mean_loss变化)
abl = []
base = predict_mean_loss(p_uni, m0)
for i, d in enumerate(DOMAINS17):
    p2 = p_uni.copy(); p2[i] = 0; p2 = p2/p2.sum()
    p3 = np.zeros(17); p3[i] = 1.0
    abl.append({"domain": d, "dloss_remove_from_uniform": predict_mean_loss(p2, m0) - base,
                "dloss_pure_domain": predict_mean_loss(p3, m0) - base})
abl = pd.DataFrame(abl)
abl.to_csv(os.path.join(OUT, "domain_ablation.csv"), index=False)
print(abl.round(4).to_string())

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9,5))
od = opt_df.sort_values("p_optimal", ascending=False)
ax.bar(od["domain"], od["p_optimal"], color="#2E86C1", label="优化配比")
ax.bar(od["domain"], od["p_uniform"], color="gray", alpha=0.4, label="均匀配比")
ax.set_title("最优领域配比(最小化13域平均Loss, 1M模型)")
plt.xticks(rotation=45, ha="right"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_optimal_mixture.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
Xp = t1m[MIXCOLS].values
pred = np.mean([mdl.predict(feats(Xp, False)) for mdl in m0], axis=0)
true = t1m[LOSSCOLS].values.mean(axis=1)
ax.scatter(true, pred, s=14, alpha=0.6, color="#28B463")
lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
ax.plot([lo,hi],[lo,hi],"r--")
ax.set_xlabel("真实平均Loss"); ax.set_ylabel("预测平均Loss")
ax.set_title("检验集 test_1M: 预测 vs 真实")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pred_vs_true_1m.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9,5))
c = coef.set_index("domain").mean(axis=1).sort_values()
ax.barh(c.index, c.values, color=["#C0392B" if v<0 else "#7D3C98" for v in c.values])
ax.axvline(0, color="k", lw=0.8)
ax.set_title("各域配比对Loss的平均边际系数(Ridge, 负=降低Loss)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_coef.png"), dpi=150); plt.close()

print("DONE_Q1_MIXTURE")

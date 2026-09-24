# -*- coding: utf-8 -*-
"""配比寻优的随机搜索验证: SLSQP解是否接近GBM目标的全局最优"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = r"C:/Users/dkyyt/Desktop/F题"
TAB = os.path.join(ROOT, "real_attachments", "A_data_value", "regmix_tables")
OUT = os.path.join(ROOT, "outputs", "q1_mixture")
DOMAINS17 = ["arxiv","freelaw","nih_exporter","pubmed_central","wikipedia_en","dm_mathematics",
             "github","philpapers","stackexchange","enron_emails","gutenberg_pg_19","pile_cc",
             "ubuntu_irc","europarl","hackernews","pubmed_abstracts","uspto_backgrounds"]
MIXCOLS = [f"train_the_pile_{d}" for d in DOMAINS17]
LOSSCOLS = [f"metric/the_pile_{d}_val_loss" for d in ["arxiv","freelaw","pubmed_central","wikipedia_en",
    "dm_mathematics","github","stackexchange","gutenberg_pg_19","pile_cc","ubuntu_irc","hackernews",
    "pubmed_abstracts","uspto_backgrounds"]]

train = pd.read_csv(os.path.join(TAB,"train_mixture_1m.csv")).merge(
        pd.read_csv(os.path.join(TAB,"train_pile_loss_1m.csv")), on="index")
Xtr, Ytr = train[MIXCOLS].values, train[LOSSCOLS].values
gbms = [HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08,
        random_state=0).fit(Xtr, Ytr[:, j]) for j in range(13)]
def pred(P): return np.mean([g.predict(np.atleast_2d(P)) for g in gbms], axis=0)

rng = np.random.default_rng(7)
R = rng.dirichlet(np.ones(17), size=20000)
pr = pred(R)
best_i = int(np.argmin(pr))
cand = pd.read_csv(os.path.join(OUT, "optimal_mixture_gbm.csv"))
p_gbm = cand["p_gbm_opt"].values; p_uni = np.ones(17)/17

print("随机搜索20000点: best pred = {:.4f}".format(pr[best_i]))
print("SLSQP-GBM解 pred = {:.4f} (在随机点中的分位 {:.4f})".format(
    pred(p_gbm)[0], (pr < pred(p_gbm)[0]).mean()))
print("均匀配比 pred = {:.4f} (分位 {:.4f})".format(pred(p_uni)[0], (pr < pred(p_uni)[0]).mean()))
print("随机最优与SLSQP解的距离 = {:.4f}".format(np.linalg.norm(R[best_i]-p_gbm)))

ver = pd.DataFrame({
    "candidate": ["slsqp_gbm_opt","uniform","random_best_of_20000"],
    "gbm_pred": [pred(p_gbm)[0], pred(p_uni)[0], pr[best_i]],
    "pct_among_20000_random": [(pr < pred(p_gbm)[0]).mean(), (pr < pred(p_uni)[0]).mean(), 0.0]})
ver.to_csv(os.path.join(OUT,"optimization_verification.csv"), index=False)
rand_best = pd.DataFrame({"domain": DOMAINS17, "p_random_best": R[best_i]})
rand_best.to_csv(os.path.join(OUT,"random_search_best.csv"), index=False)
print(ver.round(4).to_string())
print("DONE_VERIFY")

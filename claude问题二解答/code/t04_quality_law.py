"""T04 把质量 Q 纳入标度律: 候选形式比较(B6 训练 / B7 新增 Q 水平检验 / 分组交叉验证), 锚定检验, 自助 CI; B8 诊断
所有候选在 Q=1 处退化为经典律 L(N,D)。
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import optimize, stats
sys.path.insert(0, 'code'); from common import *
O = 'out/T04'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f)
rng = np.random.default_rng(11)
main = json.load(open('out/T02/T02_main_params.json')); C0 = {k: main[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}
b6 = pd.read_csv('data/B/supplementary_NQ_experiment.csv'); b7 = pd.read_csv('data/B/supplementary_NQ_experiment_expanded.csv')
b7new = b7[~b7.experiment_id.isin(b6.experiment_id)]
for d in (b6, b7, b7new):
    d['N'] = d.N_params_B * 1e9; d['D'] = d.D_tokens_B * 1e9

# ---- 候选形式 (θc: 经典 5 参, θq: 质量参数)
def terms(N, D, c): return c[1] * N ** (-c[2]), c[3] * D ** (-c[4])
MODELS = {
    'Q0 经典(忽略Q)':           (0, lambda N, D, Q, c, q: c[0] + sum(terms(N, D, c))),
    'Q1 有效数据 D·Q^κ':         (1, lambda N, D, Q, c, q: c[0] + c[1] * N ** -c[2] + c[3] * (D * Q ** q[0]) ** -c[4]),
    'Q2 有效参数 N·Q^κ':         (1, lambda N, D, Q, c, q: c[0] + c[1] * (N * Q ** q[0]) ** -c[2] + c[3] * D ** -c[4]),
    'Q3 不可约项 E+e(1-Q)^γ':    (2, lambda N, D, Q, c, q: c[0] + q[0] * (1 - Q) ** q[1] + sum(terms(N, D, c))),
    'Q4 可约项幂乘子 Q^-κ':       (1, lambda N, D, Q, c, q: c[0] + Q ** (-q[0]) * sum(terms(N, D, c))),
    'Q5 可约项乘子 1+c(1-Q)^γ':  (2, lambda N, D, Q, c, q: c[0] + (1 + q[0] * (1 - Q) ** q[1]) * sum(terms(N, D, c))),
    'Q6 分项乘子(N项/D项各自)':   (3, lambda N, D, Q, c, q: c[0] + (1 + q[0] * (1 - Q) ** q[2]) * c[1] * N ** -c[2] + (1 + q[1] * (1 - Q) ** q[2]) * c[3] * D ** -c[4]),
    'Q7 总损失乘子 L·(1+c(1-Q)^γ)': (2, lambda N, D, Q, c, q: (1 + q[0] * (1 - Q) ** q[1]) * (c[0] + sum(terms(N, D, c)))),
}
Q0INIT = {1: [0.5], 2: [0.4, 1.0], 3: [0.4, 0.4, 1.0]}
QB = {1: ([-5], [5]), 2: ([-5, 0.05], [5, 5]), 3: ([-5, -5, 0.05], [5, 5, 5])}
cvec0 = np.array([C0['E'], C0['A'], C0['alpha'], C0['B'], C0['beta']])


def fit(df, name, anchored=True):
    k, f = MODELS[name]; N, D, Q, L = df.N.values, df.D.values, df.Q_score.values, df.val_loss.values
    if anchored:  # 经典 5 参固定为 B1 主拟合 → 跨源统一假设 H_U
        if k == 0:
            return dict(c=cvec0, q=np.array([])), f(N, D, Q, cvec0, [])
        best = None
        for s in [Q0INIT[k], [x * 2 for x in Q0INIT[k]], [x * .5 for x in Q0INIT[k]]]:
            r = optimize.least_squares(lambda q: f(N, D, Q, cvec0, q) - L, s, bounds=QB[k])
            if best is None or r.cost < best.cost:
                best = r
        return dict(c=cvec0, q=best.x), f(N, D, Q, cvec0, best.x)
    # 自由: 经典参数也放开(在对数尺度参数化 A,B)
    def unpack(t): return np.array([t[0], np.exp(t[1]), t[2], np.exp(t[3]), t[4]]), t[5:]
    t0 = np.r_[cvec0[0], np.log(cvec0[1]), cvec0[2], np.log(cvec0[3]), cvec0[4], Q0INIT.get(k, [])]
    lb = np.r_[0, -5, 0.01, -5, 0.01, QB[k][0] if k else []]; ub = np.r_[5, 40, 1.5, 40, 1.5, QB[k][1] if k else []]
    best = None
    for jit in range(6):
        s = t0 if jit == 0 else t0 * (1 + 0.2 * rng.standard_normal(len(t0)))
        s = np.clip(s, lb + 1e-6, ub - 1e-6)
        r = optimize.least_squares(lambda t: f(N, D, Q, *unpack(t)) - L, s, bounds=(lb, ub), max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    c, q = unpack(best.x); return dict(c=c, q=q), f(N, D, Q, c, q)


def pred(name, fitres, df):
    return MODELS[name][1](df.N.values, df.D.values, df.Q_score.values, fitres['c'], fitres['q'])


def ic(y, p, k):
    n = len(y); rss = np.sum((y - p) ** 2); return n * np.log(rss / n) + 2 * k, n * np.log(rss / n) + k * np.log(n)

rows = []; fitted = {}
for anch in [True, False]:
    for nm in MODELS:
        k = MODELS[nm][0] + (0 if anch else 5)
        fr6, p6 = fit(b6, nm, anch); fr7, p7 = fit(b7, nm, anch); fitted[(nm, anch)] = fr7
        pnew = pred(nm, fr6, b7new)
        aic, bic = ic(b7.val_loss.values, p7, k)
        # 留一 N 分组交叉验证(B7)
        cvp = np.zeros(len(b7))
        for n in b7.N_params_B.unique():
            tr = b7[b7.N_params_B != n]; te = b7.N_params_B == n
            fr, _ = fit(tr, nm, anch); cvp[te.values] = pred(nm, fr, b7[te])
        # 留一 Q 水平
        cvq = np.zeros(len(b7))
        for qv in b7.Q_score.unique():
            tr = b7[b7.Q_score != qv]; te = b7.Q_score == qv
            fr, _ = fit(tr, nm, anch); cvq[te.values] = pred(nm, fr, b7[te])
        rows.append(dict(model=nm, classic_params='锚定B1(H_U)' if anch else '自由', k=k, q_params=np.round(fr7['q'], 5).tolist(),
                         classic_fitted=np.round(fr7['c'], 4).tolist(), B7_RMSE=metrics(b7.val_loss, p7)['RMSE'], B7_R2=metrics(b7.val_loss, p7)['R2'],
                         AIC=aic, BIC=bic, B6fit_to_B7new_RMSE=metrics(b7new.val_loss, pnew)['RMSE'], B6fit_to_B7new_R2=metrics(b7new.val_loss, pnew)['R2'],
                         LONO_CV_RMSE=metrics(b7.val_loss, cvp)['RMSE'], LOQO_CV_RMSE=metrics(b7.val_loss, cvq)['RMSE']))
        print(nm, anch, rows[-1]['B7_RMSE'], rows[-1]['LONO_CV_RMSE'], flush=True)
CMP = pd.DataFrame(rows); CMP['dBIC_vs_best'] = CMP.BIC - CMP.BIC.min(); CMP.to_csv(P('T04_candidate_comparison.csv'), index=False)
# 锚定 vs 自由: 嵌套 F 检验(同一形式)
ft = []
for nm in MODELS:
    a = CMP[(CMP.model == nm) & (CMP.classic_params == '锚定B1(H_U)')].iloc[0]; b = CMP[(CMP.model == nm) & (CMP.classic_params == '自由')].iloc[0]
    n = len(b7); rss_a = a.B7_RMSE ** 2 * n; rss_b = b.B7_RMSE ** 2 * n; df1 = 5; df2 = n - b.k
    F = ((rss_a - rss_b) / df1) / (rss_b / df2); ft.append(dict(model=nm, RSS_anchored=rss_a, RSS_free=rss_b, F=F, p_value=1 - stats.f.cdf(F, df1, df2)))
pd.DataFrame(ft).to_csv(P('T04_anchor_F_test.csv'), index=False)
# 选定模型: 锚定组中 BIC 最小者
anc = CMP[CMP.classic_params == '锚定B1(H_U)'].sort_values('BIC'); BEST = anc.iloc[0].model
fb = fitted[(BEST, True)]
b7['pred_best'] = pred(BEST, fb, b7); b7['resid_best'] = b7.val_loss - b7.pred_best
b7.to_csv(P('T04_B7_pointwise_best.csv'), index=False)
b7.groupby('Q_score').resid_best.agg(['mean', 'std', 'size']).to_csv(P('T04_resid_by_Q.csv'))
b7.groupby('N_params_B').resid_best.agg(['mean', 'std', 'size']).to_csv(P('T04_resid_by_N.csv'))
b7.groupby('D_tokens_B').resid_best.agg(['mean', 'std', 'size']).to_csv(P('T04_resid_by_D.csv'))
sw = stats.shapiro(b7.resid_best); json.dump(dict(best_model=BEST, q=fb['q'].tolist(), shapiro_W=float(sw.statistic), shapiro_p=float(sw.pvalue),
                                                  resid_sd=float(b7.resid_best.std()), n=len(b7)), open(P('T04_best_model.json'), 'w'), ensure_ascii=False, indent=1)
# 自助 CI (残差自助, 锚定)
bsq = []
for b in range(300):
    yb = b7.pred_best.values + rng.choice(b7.resid_best.values, len(b7), replace=True)
    d_ = b7.assign(val_loss=yb); fr, _ = fit(d_, BEST, True); bsq.append(fr['q'])
BQ = pd.DataFrame(bsq, columns=[f'q{i}' for i in range(len(fb['q']))]); BQ.to_csv(P('T04_best_bootstrap_q.csv'), index=False)
BQ.quantile([0.025, 0.5, 0.975]).to_csv(P('T04_best_bootstrap_CI.csv'))
# 同时对自由版给出经典参数, 用于检验与 B1 的一致
# ---- B8 诊断
b8 = pd.read_csv('data/B/supplementary_NQ_experiment_large.csv'); b8['N'] = b8.N_params_B * 1e9; b8['D'] = b8.D_tokens_B * 1e9
b8['L_classic'] = predict(C0, b8.N.values, b8.D.values); b8['censored'] = b8.val_loss <= 0.5 + 1e-9
q1 = b8[b8.Q_score == 1.0]
diag = dict(n=len(b8), censored_share=float(b8.censored.mean()), Q1_rows=int(len(q1)), Q1_vs_classic=metrics(q1.val_loss, q1.L_classic),
            spearman_Q_vs_loss_uncensored=float(stats.spearmanr(b8[~b8.censored].Q_score, b8[~b8.censored].val_loss - b8[~b8.censored].L_classic).statistic))
# 线性插值表示: L8 = Q·L_classic + (1-Q)·h(N,D), 在未截断点上逐 (N,D) 估计 h
hh = []
for (n, d_), g in b8[~b8.censored].groupby(['N_params_B', 'D_tokens_B']):
    if len(g) < 3:
        continue
    y = g.val_loss - g.Q_score * g.L_classic; x = 1 - g.Q_score
    h = float(np.sum(x * y) / np.sum(x * x)); fitv = g.Q_score * g.L_classic + (1 - g.Q_score) * h
    hh.append(dict(N_B=n, D_B=d_, n_uncensored=len(g), h=h, rmse=float(np.sqrt(np.mean((g.val_loss - fitv) ** 2))), L_classic=float(g.L_classic.iloc[0])))
H = pd.DataFrame(hh); H.to_csv(P('T04_B8_linear_mixture_representation.csv'), index=False)
diag['linear_repr_median_rmse'] = float(H.rmse.median()); diag['h_range'] = [float(H.h.min()), float(H.h.max())]
json.dump(diag, open(P('T04_B8_diagnostics.json'), 'w'), ensure_ascii=False, indent=1)
b8.to_csv(P('T04_B8_with_classic.csv'), index=False)
print(CMP.sort_values(['classic_params', 'BIC']).round(4).to_string()); print(pd.DataFrame(ft).round(4)); print(open(P('T04_best_model.json')).read())
print(BQ.quantile([0.025, 0.5, 0.975])); print(json.dumps(diag, ensure_ascii=False, indent=1)); print(H.describe().round(3))

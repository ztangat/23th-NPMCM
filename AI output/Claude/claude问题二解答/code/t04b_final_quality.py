"""T04b 最终质量形式(γ=1 线性乘子) + 跨源锚定层级检验 + 质量参数自助 CI
L = E + A[1+c_N(1-Q)]N^-α + B[1+c_D(1-Q)]D^-β   (B6/B7 口径 Q, Q=1 与 B1 同口径)
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import optimize, stats
sys.path.insert(0, 'code'); from common import *
O = 'out/T04'; P = lambda f: os.path.join(O, f); rng = np.random.default_rng(5)
m = json.load(open('out/T02/T02_main_params.json')); c0 = np.array([m['E'], m['A'], m['alpha'], m['B'], m['beta']])
b7 = pd.read_csv('data/B/supplementary_NQ_experiment_expanded.csv'); b6 = pd.read_csv('data/B/supplementary_NQ_experiment.csv')
new = b7[~b7.experiment_id.isin(b6.experiment_id)]
def F(d, E, A, al, B, be, cN, cD):
    N = d.N_params_B.values * 1e9; D = d.D_tokens_B.values * 1e9; q = 1 - d.Q_score.values
    return E + A * (1 + cN * q) * N ** -al + B * (1 + cD * q) * D ** -be
LEVELS = {'H_U0 经典5参全锚定B1': [], 'H_U1 仅放开E': [0], 'H_U2 放开E,A,B(指数α,β锚定)': [0, 1, 3], 'H_U3 全放开': [0, 1, 2, 3, 4]}
def fit(d, free):
    def unpack(t):
        c = c0.copy()
        for i, j in enumerate(free):
            c[j] = t[i] if j not in (1, 3) else np.exp(t[i])
        return c, t[len(free):]
    t0 = [c0[j] if j not in (1, 3) else np.log(c0[j]) for j in free] + [0.5, 0.35]
    r = optimize.least_squares(lambda t: F(d, *unpack(t)[0], *unpack(t)[1]) - d.val_loss.values, t0, max_nfev=50000)
    c, q = unpack(r.x); return c, q, r
rows = []; store = {}
for nm, free in LEVELS.items():
    c, q, r = fit(b7, free); pr = F(b7, *c, *q); n = len(b7); k = len(free) + 2; rss = np.sum((b7.val_loss - pr) ** 2)
    c6, q6, _ = fit(b6, free); pn = F(new, *c6, *q6)
    cv = np.zeros(n)
    for nn in b7.N_params_B.unique():
        te = (b7.N_params_B == nn).values; cc, qq, _ = fit(b7[~te], free); cv[te] = F(b7[te], *cc, *qq)
    rows.append(dict(level=nm, k=k, E=c[0], A=c[1], alpha=c[2], B=c[3], beta=c[4], c_N=q[0], c_D=q[1], RSS=rss, RMSE=np.sqrt(rss / n),
                     BIC=n * np.log(rss / n) + k * np.log(n), B6fit_B7new_RMSE=metrics(new.val_loss, pn)['RMSE'], LONO_RMSE=metrics(b7.val_loss, cv)['RMSE']))
    store[nm] = (c, q)
LV = pd.DataFrame(rows); LV['dBIC'] = LV.BIC - LV.BIC.min()
# 与 H_U0 的嵌套 F 检验
base = LV.iloc[0]
LV['F_vs_H_U0'] = [np.nan] + [((base.RSS - r.RSS) / (r.k - base.k)) / (r.RSS / (len(b7) - r.k)) for _, r in LV.iloc[1:].iterrows()]
LV['p_vs_H_U0'] = [np.nan] + [1 - stats.f.cdf(f, r.k - base.k, len(b7) - r.k) for f, (_, r) in zip(LV.F_vs_H_U0.iloc[1:], LV.iloc[1:].iterrows())]
LV.to_csv(P('T04b_anchor_levels_linear_Q.csv'), index=False)
# 主结果: H_U0(题目要求经典律以 B1 为主拟合; 质量参数在 B1 口径上估计); 自助 CI
c, q = store['H_U0 经典5参全锚定B1']; pr = F(b7, *c, *q); res = b7.val_loss.values - pr
bs = []
for b in range(500):
    d_ = b7.assign(val_loss=pr + rng.choice(res, len(res)))
    _, qq, _ = fit(d_, []); bs.append(qq)
# 簇自助(按 N 与 D 单元重抽)
cells = b7.groupby(['N_params_B', 'D_tokens_B']).groups; keys = list(cells)
for b in range(300):
    pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]; d_ = pd.concat([b7.loc[cells[k]] for k in pick])
    _, qq, _ = fit(d_, []); bs.append(qq)
BS = pd.DataFrame(bs, columns=['c_N', 'c_D']); BS['kind'] = ['residual'] * 500 + ['cell_cluster'] * 300
BS.to_csv(P('T04b_bootstrap_cN_cD.csv'), index=False)
CI = BS.groupby('kind')[['c_N', 'c_D']].quantile([.025, .5, .975]); CI.to_csv(P('T04b_bootstrap_CI.csv'))
# c_N 与 c_D 是否相等(质量是否对参数项/数据项同等作用)
diff = BS[BS.kind == 'cell_cluster'].eval('c_N - c_D')
json.dump(dict(model='L=E+A[1+c_N(1-Q)]N^-α+B[1+c_D(1-Q)]D^-β', classic_from_B1=dict(zip(['E', 'A', 'alpha', 'B', 'beta'], c.tolist())),
               c_N=float(q[0]), c_D=float(q[1]), resid_sd=float(res.std()), RMSE=float(np.sqrt(np.mean(res ** 2))),
               cN_minus_cD_CI95=[float(diff.quantile(.025)), float(diff.quantile(.975))], n=len(b7)), open(P('T04b_final_quality_law.json'), 'w'), ensure_ascii=False, indent=1)
b7.assign(pred=pr, resid=res).to_csv(P('T04b_B7_pointwise_final.csv'), index=False)
print(LV.round(5).to_string()); print(CI.round(4)); print(open(P('T04b_final_quality_law.json')).read())

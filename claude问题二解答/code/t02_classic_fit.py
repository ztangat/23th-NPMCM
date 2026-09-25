"""T02 经典标度律 L(N,D)=E+A N^-α+B D^-β 以 B1(Pythia) 为主拟合数据
- 子集敏感性: D 下限 {1,5,10,20,50}B / 仅终点 / 全部
- 留一模型(LOMO)交叉验证; D 时间外推(≤150B 拟合, >150B 检验)
- 不确定性: 模型簇自助(200) + 检查点块自助(200)
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'code'); from common import *
O = 'out/T02'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f)
rng = np.random.default_rng(2026)
p = pd.read_csv('data/B/pythia_training_log_existing.csv')
p['N'] = p.N_params_B * 1e9; p['D'] = p.D_tokens_B * 1e9
DMIN_MAIN = 20.0

rows = []
subsets = {f'D>={d}B': p[p.D_tokens_B >= d] for d in [0, 1, 5, 10, 20, 50]}
subsets['final_only(8点)'] = p.sort_values('D_tokens_B').groupby('N_params_B').tail(1)
fits = {}
for nm, s in subsets.items():
    par = fit_classic(s.N.values, s.D.values, s.val_loss.values)
    if nm.startswith('final'):
        # 8 点 5 参数欠定: 固定 β 为主拟合值再报告(下方回填)
        pass
    fits[nm] = par
    m = metrics(s.val_loss, predict(par, s.N.values, s.D.values))
    mall = metrics(p[p.D_tokens_B >= DMIN_MAIN].val_loss, predict(par, p[p.D_tokens_B >= DMIN_MAIN].N.values, p[p.D_tokens_B >= DMIN_MAIN].D.values))
    rows.append(dict(subset=nm, n=len(s), **{k: par[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}, huber_obj=par['obj'], in_RMSE=m['RMSE'], in_MAPE_pct=m['MAPE_pct'],
                     R2=m['R2'], RMSE_on_D10plus=mall['RMSE']))
SENS = pd.DataFrame(rows); SENS.to_csv(P('T02_subset_sensitivity.csv'), index=False)
main = fits['D>=20B']
S = p[p.D_tokens_B >= DMIN_MAIN].copy()
S['pred'] = predict(main, S.N.values, S.D.values); S['resid'] = S.val_loss - S.pred
p['pred_main'] = predict(main, p.N.values, p.D.values); p['resid_main'] = p.val_loss - p.pred_main
p[['run_id', 'N_params_B', 'D_tokens_B', 'val_loss', 'pred_main', 'resid_main']].to_csv(P('T02_B1_pointwise_fit_residuals.csv'), index=False)
S.groupby('N_params_B').agg(n=('resid', 'size'), mean_resid=('resid', 'mean'), rmse=('resid', lambda r: np.sqrt(np.mean(r ** 2))),
                            max_abs=('resid', lambda r: np.abs(r).max()), resid_first=('resid', 'first'), resid_last=('resid', 'last')).to_csv(P('T02_residuals_by_model.csv'))
S.assign(Dbin=pd.cut(S.D_tokens_B, [10, 25, 50, 100, 150, 200, 250, 301])).groupby('Dbin', observed=True).resid.agg(['size', 'mean', 'std']).to_csv(P('T02_residuals_by_D_bin.csv'))

# LOMO
lomo = []; lomo_pred = []
for n in sorted(S.N_params_B.unique()):
    tr, te = S[S.N_params_B != n], S[S.N_params_B == n]
    par = fit_classic(tr.N.values, tr.D.values, tr.val_loss.values, n_starts=30, init=main)
    pr = predict(par, te.N.values, te.D.values)
    lomo.append(dict(held_out_N_B=n, **metrics(te.val_loss, pr), alpha=par['alpha'], beta=par['beta'], E=par['E'],
                     extrapolation=('向外(最小)' if n == S.N_params_B.min() else ('向外(最大)' if n == S.N_params_B.max() else '内插'))))
    lomo_pred.append(te[['N_params_B', 'D_tokens_B', 'val_loss']].assign(pred_lomo=pr))
LO = pd.DataFrame(lomo); LO.to_csv(P('T02_LOMO_cv.csv'), index=False); pd.concat(lomo_pred).to_csv(P('T02_LOMO_predictions.csv'), index=False)
# D 外推
tr, te = S[S.D_tokens_B <= 150], S[S.D_tokens_B > 150]
parT = fit_classic(tr.N.values, tr.D.values, tr.val_loss.values)
TX = dict(train='20≤D≤150B', test='D>150B', **{k: parT[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}, **metrics(te.val_loss, predict(parT, te.N.values, te.D.values)))
json.dump(TX, open(P('T02_time_extrapolation.json'), 'w'), indent=1)
# 自助
Ns = sorted(S.N_params_B.unique()); bs = []
for b in range(200):
    pick = rng.choice(Ns, len(Ns), replace=True)
    s = pd.concat([S[S.N_params_B == n] for n in pick])
    if s.N_params_B.nunique() < 3:
        continue
    par = fit_classic(s.N.values, s.D.values, s.val_loss.values, n_starts=8, init=main); bs.append(dict(kind='model_cluster', **par))
blocks = pd.cut(S.D_tokens_B, 14, labels=False)
for b in range(200):
    pick = rng.choice(14, 14, replace=True); s = pd.concat([S[blocks == k] for k in pick])
    par = fit_classic(s.N.values, s.D.values, s.val_loss.values, n_starts=8, init=main); bs.append(dict(kind='D_block', **par))
BS = pd.DataFrame(bs); BS.to_csv(P('T02_bootstrap_params.csv'), index=False)
ci = BS.groupby('kind')[['E', 'A', 'alpha', 'B', 'beta']].quantile([0.025, 0.5, 0.975]).reset_index().rename(columns={'level_1': 'quantile'})
ci.to_csv(P('T02_bootstrap_CI.csv'), index=False)
# 与 Chinchilla 比较
comp = pd.DataFrame([dict(source='本文 B1 主拟合(D≥20B)', **{k: main[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}),
                     dict(source='Hoffmann et al. 2022 (Approach 3)', **CHIN)])
comp['a_opt=β/(α+β) (N*∝C^a)'] = comp.beta / (comp.alpha + comp.beta); comp['b_opt=α/(α+β) (D*∝C^b)'] = comp.alpha / (comp.alpha + comp.beta)
comp.to_csv(P('T02_compare_chinchilla.csv'), index=False)
co = []
for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25]:
    for nm, par in [('本文', main), ('Chinchilla', CHIN)]:
        N_, D_ = compute_optimal(par, C); co.append(dict(C_FLOPs=C, law=nm, N_opt_B=N_ / 1e9, D_opt_B=D_ / 1e9, tokens_per_param=D_ / N_, L_opt=predict(par, N_, D_)))
pd.DataFrame(co).to_csv(P('T02_compute_optimal_allocation.csv'), index=False)
json.dump({'main_subset': 'D>=20B', 'n': int(len(S)), **main, 'in_sample': metrics(S.val_loss, S.pred)}, open(P('T02_main_params.json'), 'w'), indent=1)
print(SENS.round(4).to_string()); print(LO.round(4).to_string()); print(TX); print(ci.round(4).to_string()); print(comp.round(4).to_string())

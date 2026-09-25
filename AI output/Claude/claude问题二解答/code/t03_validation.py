"""T03 经典标度律的族外/插值/跨族/文献验证 与 B10 生成律识别
B3: 插值轨迹验证; B2: Cerebras 族外(半合成); B4: 12 族收敛点; B5: 文献点
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, 'code'); from common import *
O = 'out/T03'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f)
main = json.load(open('out/T02/T02_main_params.json'))
par = {k: main[k] for k in ['E', 'A', 'alpha', 'B', 'beta']}
res = []
# B1 与 Chinchilla 原参数的逐点吻合度(识别 B1 的生成口径)
p = pd.read_csv('data/B/pythia_training_log_existing.csv')
pc = predict(CHIN, p.N_params_B.values * 1e9, p.D_tokens_B.values * 1e9)
rel = (p.val_loss - pc) / p.val_loss
json.dump(dict(n=len(p), median_abs_rel_err_pct=float(np.median(np.abs(rel)) * 100), max_abs_rel_err_pct=float(np.max(np.abs(rel)) * 100),
               rmse=float(np.sqrt(np.mean((p.val_loss - pc) ** 2)))), open(P('T03_B1_vs_Chinchilla_published_params.json'), 'w'), indent=1)
# ---- B3
tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data/B/training_trajectories/*.csv'))])
tr['pred'] = predict(par, tr.N_params_B.values * 1e9, tr.D_tokens_B.values * 1e9)
t3 = tr.groupby('N_params_B').apply(lambda g: pd.Series(metrics(g.val_loss, g.pred)), include_groups=False).reset_index()
t3.to_csv(P('T03_B3_trajectory_metrics_by_model.csv'), index=False)
res.append(dict(dataset='B3 插值轨迹(4000)', **metrics(tr.val_loss, tr.pred)))
tr[tr.D_tokens_B >= 20].pipe(lambda g: res.append(dict(dataset='B3 插值轨迹 D≥20B', **metrics(g.val_loss, g.pred))))
tr.to_csv(P('T03_B3_pointwise.csv'), index=False)
# ---- B2 Cerebras
c = pd.read_csv('data/B/cerebras_training_log.csv')
c['pred_pythia_law'] = predict(par, c.N_params_B.values * 1e9, c.D_tokens_B.values * 1e9)
res.append(dict(dataset='B2 Cerebras 直接外推(Pythia 律)', **metrics(c.val_loss, c.pred_pythia_law)))
# 族偏移(仅平移 E)
dE = float(np.mean(c.val_loss - c.pred_pythia_law)); c['pred_offset'] = c.pred_pythia_law + dE
res.append(dict(dataset=f'B2 Cerebras + 族常数偏移 ΔE={dE:.3f}', **metrics(c.val_loss, c.pred_offset)))
# 族内重拟合
cp = fit_classic(c.N_params_B.values * 1e9, c.D_tokens_B.values * 1e9, c.val_loss.values)
c['pred_refit'] = predict(cp, c.N_params_B.values * 1e9, c.D_tokens_B.values * 1e9)
res.append(dict(dataset='B2 Cerebras 族内重拟合(参照)', **metrics(c.val_loss, c.pred_refit)))
# 留一模型: 用 Pythia 律 + 由其余 6 个 Cerebras 模型估计的 ΔE 预测留出模型
lo = []
for n in sorted(c.N_params_B.unique()):
    trn, te = c[c.N_params_B != n], c[c.N_params_B == n]
    d_ = float(np.mean(trn.val_loss - trn.pred_pythia_law)); lo.append(dict(held_out_N_B=n, dE=d_, **metrics(te.val_loss, te.pred_pythia_law + d_)))
pd.DataFrame(lo).to_csv(P('T03_B2_leave_one_model_out_offset.csv'), index=False)
pd.DataFrame([dict(source='Pythia(B1) 主拟合', **par), dict(source='Cerebras(B2) 族内重拟合', **{k: cp[k] for k in par})]).to_csv(P('T03_B2_param_compare.csv'), index=False)
c['resid_offset'] = c.val_loss - c.pred_offset
c.groupby('N_params_B').agg(rmse_direct=('pred_pythia_law', lambda s: np.nan), mean_resid_offset=('resid_offset', 'mean'), sd_resid_offset=('resid_offset', 'std')).to_csv(P('T03_B2_resid_by_model.csv'))
c[['run_id', 'N_params_B', 'D_tokens_B', 'val_loss', 'pred_pythia_law', 'pred_offset', 'pred_refit']].to_csv(P('T03_B2_pointwise.csv'), index=False)
# ---- B4 / B5
def fam_eval(df, name):
    df = df.copy(); df['pred'] = predict(par, df.N_params_B.values * 1e9, df.D_tokens_B.values * 1e9); df['resid'] = df.val_loss - df.pred
    res.append(dict(dataset=f'{name} 直接预测', **metrics(df.val_loss, df.pred)))
    # 族随机效应: 每族常数偏移 δ_f (留一点交叉验证)
    cvp = []
    for i in df.index:
        fam = df.loc[i, 'family']; oth = df[(df.family == fam) & (df.index != i)]
        d_ = oth.resid.mean() if len(oth) else df.drop(i).resid.mean()
        cvp.append(df.loc[i, 'pred'] + d_)
    df['pred_family_offset_LOO'] = cvp
    res.append(dict(dataset=f'{name} + 族偏移(留一)', **metrics(df.val_loss, df.pred_family_offset_LOO)))
    fam = df.groupby('family').agg(n=('resid', 'size'), mean_resid=('resid', 'mean'), sd_resid=('resid', 'std'),
                                   spearman_within=('resid', lambda s: np.nan)).reset_index()
    for i, r in fam.iterrows():
        g = df[df.family == r.family]
        fam.loc[i, 'spearman_within'] = stats.spearmanr(g.val_loss, g.pred).statistic if len(g) >= 3 else np.nan
        fam.loc[i, 'rmse_after_offset'] = np.sqrt(np.mean((g.resid - g.resid.mean()) ** 2))
    return df, fam
b4 = pd.read_csv('data/B/scaling_baseline.csv'); b5 = pd.read_csv('data/B/published_scaling_data.csv')
d4, f4 = fam_eval(b4, 'B4 跨族(57)'); d4.to_csv(P('T03_B4_pointwise.csv'), index=False); f4.to_csv(P('T03_B4_by_family.csv'), index=False)
d4b, _ = fam_eval(b4[b4.family != 'Pythia'], 'B4 跨族(剔除 Pythia 近邻 49)')
d5, f5 = fam_eval(b5, 'B5 文献(44)'); d5.to_csv(P('T03_B5_pointwise.csv'), index=False); f5.to_csv(P('T03_B5_by_family.csv'), index=False)
# 各族/全体重拟合 α, β (以检验指数普适性; 固定 E 与 A/B 自由)
refit = []
for nm, df in [('B4 全体', b4), ('B5 全体', b5)]:
    q = fit_classic(df.N_params_B.values * 1e9, df.D_tokens_B.values * 1e9, df.val_loss.values)
    refit.append(dict(data=nm, n=len(df), **{k: q[k] for k in par}, RMSE=metrics(df.val_loss, predict(q, df.N_params_B.values * 1e9, df.D_tokens_B.values * 1e9))['RMSE']))
pd.DataFrame(refit).to_csv(P('T03_B4_B5_refit_params.csv'), index=False)
# ---- B10 识别
lb = pd.read_csv('data/B/supplementary_large_baseline.csv')
lb['pred_main'] = predict(par, lb.N_params_B.values * 1e9, lb.D_tokens_B.values * 1e9)
res.append(dict(dataset='B10 估算 Loss vs 本文律', **metrics(lb.val_loss, lb.pred_main)))
q10 = fit_classic(lb.N_params_B.values * 1e9, lb.D_tokens_B.values * 1e9, lb.val_loss.values)
json.dump(dict(B10_refit={k: q10[k] for k in par}, rmse_refit=metrics(lb.val_loss, predict(q10, lb.N_params_B.values * 1e9, lb.D_tokens_B.values * 1e9))['RMSE']),
          open(P('T03_B10_generator_identification.json'), 'w'), indent=1)
lb.to_csv(P('T03_B10_pointwise.csv'), index=False)
R = pd.DataFrame(res); R.to_csv(P('T03_validation_summary.csv'), index=False)
print(R.round(4).to_string()); print(f4.round(3).to_string()); print(f5.round(3).to_string()); print(pd.read_csv(P('T03_B4_B5_refit_params.csv')).round(4))
print(open(P('T03_B10_generator_identification.json')).read()); print(open(P('T03_B1_vs_Chinchilla_published_params.json')).read())
print(pd.read_csv(P('T03_B2_param_compare.csv')).round(4)); print(pd.DataFrame(lo).round(4))

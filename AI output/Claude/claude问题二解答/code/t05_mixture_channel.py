"""T05 把问题一的配比 p 与质量 Q 接入标度律
(1) 配比效应的尺度依赖: 以问题一 M6(仅 A4/A5 训练)的预测 Φ(p) 为自变量, 在 1M/60M/1B 实测 L_avg 上回归, 检验"乘在可约损失上"假设 H_mult
(2) 质量口径桥接: B6 口径 Q_B = Q_A / Q_ref (B6 在 Q=1 处与 B1 一致 → B 口径以 Pile 自然配比数据为 1)
(3) 跨域质量通道检验 H_Qp: 配比带来的 Q_mix 变化是否解释 M6 残差
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0, 'code'); from common import *
O = 'out/T05'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f); rng = np.random.default_rng(3)
Q1 = 'data/q1_out'
DOM = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en', 'dm_mathematics', 'github', 'philpapers', 'stackexchange', 'enron_emails',
       'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc', 'europarl', 'hackernews', 'pubmed_abstracts', 'uspto_backgrounds']
E0 = json.load(open('out/T02/T02_main_params.json'))['E']
co = pd.read_csv(f'{Q1}/S08/S08_M6_cox_effect_plus0.1_bootstrap.csv'); co = co[co.target == 'L_avg'].set_index('domain').reindex(DOM)
beta, gam, eps = co.beta_linear.values, co.gamma_log.values, float(co.eps.iloc[0])
M6 = lambda M: M @ beta + np.log(M + eps) @ gam
pref = pd.read_csv(f'{Q1}/S08/S08_reference_mixture_mean.csv', index_col=0).iloc[:, 0].reindex(DOM).values
opt = pd.read_csv(f'{Q1}/S08/S08_optimal_mixture.csv').set_index('domain').reindex(DOM)
qmap = pd.read_csv(f'{Q1}/S09/S09_domain_Q_mapping_17.csv').set_index('mixture_domain').reindex(DOM)
Qd = qmap.Q_shrunk.values
PR = pd.read_csv(f'{Q1}/S07/S07_predictions_L_avg_per_mixture.csv')
sets = {}
for s in ['train_1M', 'test_1M', 'test_60M', 'test_1B']:
    d = pd.read_csv(f'{Q1}/S06/S06_clean_{s}.csv'); d = d.merge(PR[PR.set == s][['index', 'pred_M6_log_share', 'pred_M5_LightGBM']], on='index')
    d['Phi'] = d.pred_M6_log_share; d['Qmix'] = d[DOM].values @ Qd; d['H'] = -(d[DOM].values * np.log(d[DOM].values + 1e-12)).sum(1); sets[s] = d
Nnom = {'train_1M': 1e6, 'test_1M': 1e6, 'test_60M': 6e7, 'test_1B': 1e9}
M6ref = float(M6(pref[None])[0])
# (1) 各尺度斜率
sl = []
for s, d in sets.items():
    X = d.Phi.values - M6ref; y = d.L_avg.values
    b, a = np.polyfit(X, y, 1); bb = [np.polyfit(X[i], y[i], 1)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    sl.append(dict(set=s, N_nominal=Nnom[s], n=len(d), L_mean=y.mean(), L_at_ref_intercept=a, slope_b=b, slope_CI_low=np.percentile(bb, 2.5), slope_CI_high=np.percentile(bb, 97.5),
                   pearson=stats.pearsonr(X, y).statistic, spearman=stats.spearmanr(X, y).statistic))
SL = pd.DataFrame(sl)
# H_mult: b_s = (L_ref,s - E)/(L_ref,1M - E); 以 test_1M 为基准, 由 60M 与 1B 各自反解 E
base = SL[SL.set == 'test_1M'].iloc[0]
def E_solve(r):
    # r.slope/base.slope = (a_s - E)/(a_1M - E) → E = (a_s - k a_1M)/(1 - k)
    k = r.slope_b / base.slope_b; return (r.L_at_ref_intercept - k * base.L_at_ref_intercept) / (1 - k) if abs(1 - k) > 1e-9 else np.nan
SL['E_implied_by_H_mult'] = [E_solve(r) for _, r in SL.iterrows()]
SL['pred_slope_H_mult_E0'] = (SL.L_at_ref_intercept - E0) / (base.L_at_ref_intercept - E0) * base.slope_b
SL['pred_slope_additive'] = base.slope_b
SL.to_csv(P('T05_mixture_effect_slopes_by_scale.csv'), index=False)
# 幂衰减替代: b_s = b_1M (N/1e6)^-δ
dl = [np.log(r.slope_b / base.slope_b) / -np.log(r.N_nominal / 1e6) for _, r in SL.iterrows() if r.N_nominal > 1e6]
# 以 H_mult 预测 60M/1B 的 L_avg (不使用该尺度 Loss 的斜率, 只用该尺度参照配方水平 = 截距)
val = []
for s in ['test_60M', 'test_1B']:
    d = sets[s]; a = SL[SL.set == s].L_at_ref_intercept.iloc[0]
    for nm, k in [('H_mult(E=B1的E)', (a - E0) / (base.L_at_ref_intercept - E0)), ('加性(尺度不变)', 1.0), ('仅截距(忽略配比)', 0.0)]:
        pr = a + k * base.slope_b * (d.Phi.values - M6ref)
        val.append(dict(set=s, model=nm, scale_factor=k, **metrics(d.L_avg, pr)))
VAL = pd.DataFrame(val); VAL.to_csv(P('T05_H_mult_validation.csv'), index=False)
# Ψ(p): 相对配比惩罚
def Psi(M, E=E0): return (M6(M) - M6ref) / (M6ref - E) * base.slope_b
mixes = {'reference_mean(≈Pile 自然配比)': pref, 'Q1 推荐 p*(集成Top100)': opt.p_opt_ensemble_top100.values, 'Q1 M6 最优(信赖域)': opt.p_opt_log_share_M6.values,
         '均匀 1/17': np.full(17, 1 / 17), '单一 pile_cc': np.eye(17)[DOM.index('pile_cc')], '单一 github': np.eye(17)[DOM.index('github')],
         '学术为主(arxiv+pubmed_central 各 0.5)': 0.5 * np.eye(17)[0] + 0.5 * np.eye(17)[3]}
Qref = float(pref @ Qd)
mx = []
for nm, pp in mixes.items():
    pp = np.asarray(pp) / np.sum(pp); qm = float(pp @ Qd)
    mx.append(dict(mixture=nm, Psi=float(Psi(pp[None])[0]), M6_L_1M=float(M6(pp[None])[0]), Q_mix_A=qm, Q_B=qm / Qref, entropy=float(-(pp * np.log(pp + 1e-12)).sum())))
MX = pd.DataFrame(mx); MX.to_csv(P('T05_mixture_Psi_and_Q.csv'), index=False)
# 各实验配方的 Ψ 与 Q_mix (过程数据)
pd.concat([d[['index', 'L_avg', 'Phi', 'Qmix', 'H']].assign(set=s, Psi=Psi(d[DOM].values), Q_B=d.Qmix / Qref) for s, d in sets.items()]).to_csv(P('T05_per_mixture_Psi_Qmix.csv'), index=False)
# (3) 跨域质量通道检验: Q_mix 是否解释 M6 残差 / 与 Ψ 的偏相关
hq = []
for s, d in sets.items():
    r = d.L_avg - (SL[SL.set == s].L_at_ref_intercept.iloc[0] + SL[SL.set == s].slope_b.iloc[0] * (d.Phi - M6ref))
    hq.append(dict(set=s, spearman_Qmix_vs_M6resid=stats.spearmanr(d.Qmix, r).statistic, p=stats.spearmanr(d.Qmix, r).pvalue,
                   spearman_Qmix_vs_L=stats.spearmanr(d.Qmix, d.L_avg).statistic, spearman_Qmix_vs_Phi=stats.spearmanr(d.Qmix, d.Phi).statistic,
                   Qmix_min=d.Qmix.min(), Qmix_max=d.Qmix.max(), QB_min=d.Qmix.min() / Qref, QB_max=d.Qmix.max() / Qref))
HQ = pd.DataFrame(hq); HQ.to_csv(P('T05_cross_domain_quality_channel_test.csv'), index=False)
json.dump(dict(Q_ref_A=Qref, M6_ref=M6ref, base_slope_1M=float(base.slope_b), E0=E0, delta_power_decay=[float(x) for x in dl],
               bridge='Q_B = Q_A / Q_ref (B6/B7 在 Q=1 处与 B1 一致, 故 B 口径以 Pythia/Pile 数据为 1)'), open(P('T05_bridge_constants.json'), 'w'), ensure_ascii=False, indent=1)
qmap.assign(Q_B=qmap.Q_shrunk / Qref).to_csv(P('T05_domain_Q_A_and_Q_B.csv'))
print(SL.round(4).to_string()); print(VAL.round(4).to_string()); print(MX.round(4).to_string()); print(HQ.round(4).to_string()); print(open(P('T05_bridge_constants.json')).read())

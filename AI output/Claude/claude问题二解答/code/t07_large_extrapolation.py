"""T07 百亿参数以上外推(B9/B10) + 不确定性传播 + B8(Q=1 切片)与 B2 族外"数据系数迁移"补充检验"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import optimize, stats
sys.path.insert(0, 'code'); from common import *
O = 'out/T07'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f); rng = np.random.default_rng(9)
G = json.load(open('out/T06/T06_generalized_law_params.json')); cl = json.load(open('out/T02/T02_main_params.json'))
E, A1, B1, al, be, Qref, b1, eta, R1M = G['E'], G['A1'], G['B1'], G['alpha'], G['beta'], G['Qref'], G['kappa'], G['eta'], G['R1M']
cN0, cD0 = G['c_N_Bscale'], G['c_D_Bscale']
CL = dict(E=cl['E'], A=cl['A'], alpha=cl['alpha'], B=cl['B'], beta=cl['beta'])
BS = pd.read_csv('out/T04/T04b_bootstrap_cN_cD.csv'); BS = BS[BS.kind == 'cell_cluster']
fam_sd = pd.read_csv('out/T03/T03_validation_summary.csv')
struct_sd = float(fam_sd[fam_sd.dataset.str.contains('B5 文献') & fam_sd.dataset.str.contains('族偏移')].RMSE.iloc[0])
psi = pd.read_csv('out/T05/T05_mixture_Psi_and_Q.csv').set_index('mixture'); dM6_star = float(psi.loc['Q1 推荐 p*(集成Top100)', 'M6_L_1M'] - psi.loc['reference_mean(≈Pile 自然配比)', 'M6_L_1M'])

def law(N, D, Q, cN, cD, Qr=Qref, dM6=0.0, et=eta):
    mN1 = 1 + cN * (1 - 1 / Qr); mD1 = 1 + cD * (1 - 1 / Qr); a1 = CL['A'] * mN1; bb1 = CL['B'] * mD1
    hN = (cN / Qr) / mN1; hD = (cD / Qr) / mD1
    base = CL['E'] + a1 * (1 + hN * (1 - Q)) * N ** -al + bb1 * (1 + hD * (1 - Q)) * D ** -be
    return base + b1 * dM6 * ((base - CL['E']) / R1M) ** et, a1, hN
def eqN(N, D, Q, cN, cD, Qr=Qref, dq=0.1):
    L0, a1, hN = law(N, D, Q, cN, cD, Qr); L1, _, _ = law(N, D, min(Q + dq, 1), cN, cD, Qr)
    r = (L0 - L1) / (a1 * (1 + hN * (1 - Q)) * N ** -al); return (1 - r) ** (-1 / al) if r < 1 else np.inf
# ---- B9/B10
j = pd.read_csv('out/T01/T01_B9_B10_joined.csv'); j = j[j.usable_LLM_row].copy()
j['N'] = j.N_params_B * 1e9; j['D'] = j.D_tokens_B * 1e9; j['C6ND'] = 6 * j.N * j.D
j['L_B1_classic'] = predict(CL, j.N.values, j.D.values)
j['L_Qref_pref'] = [law(n, d, Qref, cN0, cD0)[0] for n, d in zip(j.N, j.D)]
j['L_Q0p8_pref'] = [law(n, d, 0.8, cN0, cD0)[0] for n, d in zip(j.N, j.D)]
j['L_Qref_pstar'] = [law(n, d, Qref, cN0, cD0, dM6=dM6_star)[0] for n, d in zip(j.N, j.D)]
j['N_opt_at_same_C_B'] = [compute_optimal(CL, c)[0] / 1e9 for c in j.C6ND]
j['overtrain_ratio_tokens_per_param_vs_opt'] = j.tokens_per_param / [compute_optimal(CL, c)[1] / compute_optimal(CL, c)[0] for c in j.C6ND]
j['N_equiv_mult_dQ0p1'] = [eqN(n, d, Qref, cN0, cD0) for n, d in zip(j.N, j.D)]
qs = []
for _, r in j.iterrows():
    sam = [eqN(r.N, r.D, Qref, a, b, Qr=Qref * (1 + 0.03 * rng.standard_normal())) for a, b in BS[['c_N', 'c_D']].sample(200, random_state=1).values]
    qs.append(np.percentile(sam, [2.5, 97.5]))
j['N_equiv_mult_CI_low'] = [q[0] for q in qs]; j['N_equiv_mult_CI_high'] = [q[1] for q in qs]
j['diff_B10_minus_B1law'] = j.val_loss - j.L_B1_classic
j.to_csv(P('T07_B9_large_models_extrapolation.csv'), index=False)
summ = dict(n_rows=int(len(j)), B10_vs_B1law=metrics(j.val_loss, j.L_B1_classic), tokens_per_param_median=float(j.tokens_per_param.median()),
            share_over_trained_gt1=float((j.overtrain_ratio_tokens_per_param_vs_opt > 1).mean()), N_equiv_mult_median=float(j.N_equiv_mult_dQ0p1.median()),
            gain_from_Q0p8_median=float((j.L_Qref_pref - j.L_Q0p8_pref).median()), gain_from_pstar_median=float((j.L_Qref_pref - j.L_Qref_pstar).median()),
            struct_sd_cross_family=struct_sd)
json.dump(summ, open(P('T07_B9_summary.json'), 'w'), ensure_ascii=False, indent=1)
# ---- 规模梯度上的不确定性带(20 tokens/param 与 B9 中位 tokens/param)
grid = []
tpp_med = float(j.tokens_per_param.median())
for NB in [0.1, 1, 10, 100, 300, 1000, 3000, 10000]:
    for tpp in [20, tpp_med]:
        N = NB * 1e9; D = tpp * N
        smp = BS[['c_N', 'c_D']].sample(300, replace=True, random_state=2).values
        et_s = rng.uniform(G['eta_from_60M'], G['eta_from_1B'], 300); qr_s = Qref * (1 + 0.03 * rng.standard_normal(300))
        dl_q = [law(N, D, qr, a, b, qr)[0] - law(N, D, 0.8, a, b, qr)[0] for (a, b), qr in zip(smp, qr_s)]
        dl_p = [law(N, D, Qref, cN0, cD0, dM6=dM6_star, et=e_)[0] - law(N, D, Qref, cN0, cD0)[0] for e_ in et_s]
        em = [eqN(N, D, qr, a, b, qr) for (a, b), qr in zip(smp, qr_s)]
        L0 = law(N, D, Qref, cN0, cD0)[0]
        grid.append(dict(N_B=NB, tokens_per_param=round(tpp, 1), L_at_Qref=L0, L_band_struct_low=L0 - 1.96 * struct_sd, L_band_struct_high=L0 + 1.96 * struct_sd,
                         gain_Qref_to_0p8=np.median(dl_q), gain_CI_low=float(np.percentile(dl_q, 2.5)), gain_CI_high=float(np.percentile(dl_q, 97.5)),
                         mixture_gain_pstar=np.median(dl_p), mixture_gain_CI_low=float(np.percentile(dl_p, 2.5)), mixture_gain_CI_high=float(np.percentile(dl_p, 97.5)),
                         N_equiv_mult_dQ0p1=np.median(em), N_equiv_CI_low=float(np.percentile(em, 2.5)), N_equiv_CI_high=float(np.percentile(em, 97.5))))
GR = pd.DataFrame(grid); GR.to_csv(P('T07_extrapolation_uncertainty_grid.csv'), index=False)
# ---- B8 Q=1 切片: 校准段(N≤12B) vs 外推段(N≥20B) 与经典律
b8 = pd.read_csv('out/T04/T04_B8_with_classic.csv'); q1 = b8[b8.Q_score == 1.0]
b8r = [dict(segment=seg, **metrics(g.val_loss, g.L_classic)) for seg, g in q1.groupby('data_type')]
pd.DataFrame(b8r).to_csv(P('T07_B8_Q1_slice_vs_classic.csv'), index=False)
# ---- B2 族外: 共享 E,A,α,β, 仅族数据系数 B_f (1 参) 的迁移(留一模型)
c = pd.read_csv('data/B/cerebras_training_log.csv'); c['N'] = c.N_params_B * 1e9; c['D'] = c.D_tokens_B * 1e9
def fitB(d):
    r = optimize.least_squares(lambda t: CL['E'] + CL['A'] * d.N.values ** -CL['alpha'] + np.exp(t[0]) * d.D.values ** -CL['beta'] - d.val_loss.values, [np.log(CL['B'])])
    return np.exp(r.x[0])
lo = []
for n in sorted(c.N_params_B.unique()):
    tr, te = c[c.N_params_B != n], c[c.N_params_B == n]; Bf = fitB(tr)
    pr = CL['E'] + CL['A'] * te.N.values ** -CL['alpha'] + Bf * te.D.values ** -CL['beta']
    lo.append(dict(held_out_N_B=n, B_family=Bf, B_ratio_vs_B1=Bf / CL['B'], implied_data_efficiency=(Bf / CL['B']) ** (-1 / CL['beta']), **metrics(te.val_loss, pr)))
LB = pd.DataFrame(lo); LB.to_csv(P('T07_B2_family_B_transfer_LOMO.csv'), index=False)
Ball = fitB(c); pr = CL['E'] + CL['A'] * c.N.values ** -CL['alpha'] + Ball * c.D.values ** -CL['beta']
json.dump(dict(B_family=Ball, ratio=Ball / CL['B'], data_efficiency_factor=(Ball / CL['B']) ** (-1 / CL['beta']), all=metrics(c.val_loss, pr)), open(P('T07_B2_family_B_transfer_all.json'), 'w'), indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1)); print(GR.round(4).to_string()); print(pd.DataFrame(b8r).round(4)); print(LB.round(4).to_string())
print(j[['model_name', 'N_params_B', 'D_tokens_B', 'tokens_per_param', 'val_loss', 'L_B1_classic', 'L_Qref_pref', 'N_equiv_mult_dQ0p1', 'overtrain_ratio_tokens_per_param_vs_opt']].head(12).round(3).to_string())

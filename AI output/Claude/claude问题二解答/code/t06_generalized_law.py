"""T06 广义标度律 L(N,D,Q,p) 的组装、边际效用与弹性、质量-规模替代条件、成本账、领域替代/互补
主形式(A 口径 Q∈(0,1], Q=1 为完美数据):
  L = E + A1[1+ĉN(1-Q)]N^-α + B1[1+ĉD(1-Q)]D^-β + ΔL_p
  ΔL_p = κ·[M6(p)-M6(p_ref)]·(R/R_1M)^η ,  R = L(N,D,Q,p_ref) - E
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import optimize
sys.path.insert(0, 'code'); from common import *
O = 'out/T06'; os.makedirs(O, exist_ok=True); P = lambda f: os.path.join(O, f)
Q1 = 'data/q1_out'
cl = json.load(open('out/T02/T02_main_params.json')); ql = json.load(open('out/T04/T04b_final_quality_law.json')); br = json.load(open('out/T05/T05_bridge_constants.json'))
SL = pd.read_csv('out/T05/T05_mixture_effect_slopes_by_scale.csv').set_index('set')
E, A, al, B, be = cl['E'], cl['A'], cl['alpha'], cl['B'], cl['beta']; cN, cD = ql['c_N'], ql['c_D']; Qref = br['Q_ref_A']
# ---- 口径换算: B 口径(c_N,c_D; Q_B=Q_A/Qref) → A 口径 Q=1 完美
mN1 = 1 + cN * (1 - 1 / Qref); mD1 = 1 + cD * (1 - 1 / Qref)
A1, B1 = A * mN1, B * mD1; hN = (cN / Qref) / mN1; hD = (cD / Qref) / mD1
# ---- 配比通道: η 由 60M、1B 的斜率收缩估计 (b_s/b_1M = (R_s/R_1M)^η)
b1 = SL.loc['test_1M', 'slope_b']; R1M = SL.loc['test_1M', 'L_at_ref_intercept'] - E
pts = [(np.log((SL.loc[s, 'L_at_ref_intercept'] - E) / R1M), np.log(SL.loc[s, 'slope_b'] / b1)) for s in ['test_60M', 'test_1B']]
eta = float(sum(x * y for x, y in pts) / sum(x * x for x, y in pts)); eta_60 = pts[0][1] / pts[0][0]; eta_1B = pts[1][1] / pts[1][0]
DOM = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en', 'dm_mathematics', 'github', 'philpapers', 'stackexchange', 'enron_emails',
       'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc', 'europarl', 'hackernews', 'pubmed_abstracts', 'uspto_backgrounds']
co = pd.read_csv(f'{Q1}/S08/S08_M6_cox_effect_plus0.1_bootstrap.csv'); co = co[co.target == 'L_avg'].set_index('domain').reindex(DOM)
bet, gam, eps = co.beta_linear.values, co.gamma_log.values, float(co.eps.iloc[0])
M6 = lambda p: float(np.asarray(p) @ bet + np.log(np.asarray(p) + eps) @ gam)
pref = pd.read_csv(f'{Q1}/S08/S08_reference_mixture_mean.csv', index_col=0).iloc[:, 0].reindex(DOM).values
pstar = pd.read_csv(f'{Q1}/S08/S08_optimal_mixture.csv').set_index('domain').reindex(DOM).p_opt_ensemble_top100.values
M6ref = M6(pref)
PAR = dict(E=E, A1=A1, B1=B1, alpha=al, beta=be, hN=hN, hD=hD, Qref=Qref, kappa=b1, eta=eta, R1M=R1M, M6ref=M6ref)
json.dump(dict(PAR, eta_from_60M=eta_60, eta_from_1B=eta_1B, mN1=mN1, mD1=mD1, c_N_Bscale=cN, c_D_Bscale=cD,
               formula='L = E + A1[1+hN(1-Q)]N^-α + B1[1+hD(1-Q)]D^-β + κ[M6(p)-M6(p_ref)]((L_ref-E)/R1M)^η'),
          open(P('T06_generalized_law_params.json'), 'w'), ensure_ascii=False, indent=1)


def L_full(N, D, Q=None, p=None):
    Q = Qref if Q is None else Q
    base = E + A1 * (1 + hN * (1 - Q)) * N ** -al + B1 * (1 + hD * (1 - Q)) * D ** -be
    if p is None:
        return base
    R = base - E
    return base + b1 * (M6(p) - M6ref) * (R / R1M) ** eta


# 退化性与一致性检查
chk = []
for n_, d_ in [(7e7, 3e11), (1e9, 2e10), (1.2e10, 3e11), (7e10, 1.4e12)]:
    chk.append(dict(N=n_, D=d_, L_B1_classic=predict(dict(E=E, A=A, alpha=al, B=B, beta=be), n_, d_), L_full_at_Qref_pref=L_full(n_, d_, Qref, pref),
                    L_full_Q1_pref=L_full(n_, d_, 1.0, pref), L_classic_perfect=E + A1 * n_ ** -al + B1 * d_ ** -be))
pd.DataFrame(chk).to_csv(P('T06_degeneracy_checks.csv'), index=False)

# ---- 弹性与边际效用 (数值微分, p=p_ref)
def grads(N, D, Q, p=None):
    h = 1e-4; f = lambda n, d, q: L_full(n, d, q, p)
    L0 = f(N, D, Q)
    dN = (f(N * (1 + h), D, Q) - f(N * (1 - h), D, Q)) / (2 * h * N)
    dD = (f(N, D * (1 + h), Q) - f(N, D * (1 - h), Q)) / (2 * h * D)
    dQ = (f(N, D, min(Q + h, 1.0)) - f(N, D, Q - h)) / ((min(Q + h, 1.0) - (Q - h)))
    return L0, dN, dD, dQ
rows = []
for NB in [0.1, 1, 10, 70, 100, 1000]:
    for tpp in [20, 200]:
        for Q in [0.5, Qref, 0.8, 0.9, 1.0]:
            N = NB * 1e9; D = tpp * N; L0, dN, dD, dQ = grads(N, D, Q)
            rows.append(dict(N_B=NB, tokens_per_param=tpp, D_B=D / 1e9, Q=round(Q, 4), L=L0, dL_dlnN=dN * N, dL_dlnD=dD * D, dL_dQ=dQ,
                             elas_N=dN * N / L0, elas_D=dD * D / L0, elas_Q=dQ * Q / L0,
                             share_reducible_N=A1 * (1 + hN * (1 - Q)) * N ** -al / (L0 - E), share_reducible_D=B1 * (1 + hD * (1 - Q)) * D ** -be / (L0 - E),
                             dL_per_0p1Q=L_full(N, D, min(Q + 0.1, 1.0)) - L0 if Q < 1 else np.nan))
EL = pd.DataFrame(rows); EL.to_csv(P('T06_elasticity_grid.csv'), index=False)

# ---- 质量-规模替代: ΔQ=0.1 等价于参数量/数据量增加多少 (闭式)
def eq_N(N, D, Q, dq=0.1):
    Q2 = min(Q + dq, 1.0); gain = L_full(N, D, Q) - L_full(N, D, Q2)
    aQ = A1 * (1 + hN * (1 - Q)); termN = aQ * N ** -al
    ratio = gain / termN
    if ratio >= 1:
        return np.inf, ratio, gain
    return (1 - ratio) ** (-1 / al), ratio, gain  # N'/N
def eq_D(N, D, Q, dq=0.1):
    Q2 = min(Q + dq, 1.0); gain = L_full(N, D, Q) - L_full(N, D, Q2)
    bQ = B1 * (1 + hD * (1 - Q)); termD = bQ * D ** -be; ratio = gain / termD
    return (np.inf if ratio >= 1 else (1 - ratio) ** (-1 / be)), ratio, gain
sub = []
for NB in [0.07, 0.4, 1, 7, 12, 70, 175, 1000]:
    for tpp in [5, 20, 100, 1000]:
        for Q, dq in [(0.5, 0.1), (Qref, 0.1), (0.8, 0.1), (0.9, 0.1), (0.5, 0.3), (0.5, 0.5), (Qref, 0.3)]:
            N = NB * 1e9; D = tpp * N
            rN, xN, g = eq_N(N, D, Q, dq); rD, xD, _ = eq_D(N, D, Q, dq)
            sub.append(dict(N_B=NB, tokens_per_param=tpp, Q=round(Q, 4), dQ=dq, dL_gain=g, gain_over_Nterm=xN, N_equiv_multiplier=rN,
                            extra_params_B=(rN - 1) * NB if np.isfinite(rN) else np.inf, substitutable_by_N=bool(xN < 1),
                            gain_over_Dterm=xD, D_equiv_multiplier=rD, substitutable_by_D=bool(xD < 1)))
SUB = pd.DataFrame(sub); SUB.to_csv(P('T06_quality_vs_scale_equivalence.csv'), index=False)
# 可替代临界条件: gain < A1 aQ N^-α  ⇔  N^-α(A1 aQ - 0.1 hN A1) > 0.1 hD B1 D^-β(Q≤0.9)
# 化简: 0.1 hN/(1+hN(1-Q)) + 0.1 hD B1 D^-β /(A1(1+hN(1-Q)) N^-α) < 1  → 临界 tokens/param
crit = []
for Q in [0.5, Qref, 0.8, 0.9]:
    aQ = 1 + hN * (1 - Q); left = 1 - 0.1 * hN / aQ
    for NB in [0.07, 1, 12, 70, 175, 1000, 10000]:
        N = NB * 1e9
        # 需 0.1 hD B1 D^-β < left·A1 aQ N^-α → D > [0.1 hD B1/(left A1 aQ N^-α)]^(1/β)
        Dc = (0.1 * hD * B1 / (left * A1 * aQ * N ** -al)) ** (1 / be)
        crit.append(dict(Q=round(Q, 4), N_B=NB, D_critical_B=Dc / 1e9, tokens_per_param_critical=Dc / N))
pd.DataFrame(crit).to_csv(P('T06_substitution_critical_D.csv'), index=False)

# ---- 成本账: 每增加 1 FLOP, 投向参数(N↑, D 固定)与投向质量(附录 B 的 g(Q))各降多少 Loss
G = {'指数型 γe^{λQ}': (lambda q: 1e7 * np.exp(6 * q), lambda q: 1e7 * 6 * np.exp(6 * q)),
     '幂函数型 γQ^λ': (lambda q: 5e9 * q ** 4, lambda q: 5e9 * 4 * q ** 3),
     '对数渐进型 γln(1+λQ)': (lambda q: 2e9 * np.log(1 + 10 * q), lambda q: 2e9 * 10 / (1 + 10 * q))}
cost = []
for C in [1e19, 1e22, 1e24]:
    Nc, Dc = compute_optimal(dict(E=E, A=A, alpha=al, B=B, beta=be), C)
    for Q in [Qref, 0.8, 0.9]:
        L0, dN, dD, dQ = grads(Nc, Dc, Q)
        mN = -dN / (6 * Dc)  # 每 FLOP 通过扩大 N 的 Loss 降幅
        mD = -dD / (6 * Nc)
        for gn, (g, gp) in G.items():
            mQ = -dQ / (Dc * gp(Q))  # C_Q = D[g(Q)-g(Q0)] → dC/dQ = D g'(Q)
            cost.append(dict(C=C, N_opt_B=Nc / 1e9, D_opt_B=Dc / 1e9, Q=round(Q, 4), cost_form=gn, dL_per_FLOP_via_N=mN, dL_per_FLOP_via_D=mD,
                             dL_per_FLOP_via_Q=mQ, ratio_Q_over_N=mQ / mN, better='质量' if mQ > mN else '参数',
                             g_prime=gp(Q), FLOPs_per_token_for_0p01Q=gp(Q) * 0.01))
CO = pd.DataFrame(cost); CO.to_csv(P('T06_marginal_cost_effectiveness.csv'), index=False)
# 质量/参数等效的成本临界: mQ = mN ⇔ g'(Q) = (∂L/∂Q)/(∂L/∂N)·6 → 给出临界 g'(Q)
crit_g = []
for C in [1e19, 1e22, 1e24]:
    Nc, Dc = compute_optimal(dict(E=E, A=A, alpha=al, B=B, beta=be), C)
    for Q in [Qref, 0.8, 0.9]:
        L0, dN, dD, dQ = grads(Nc, Dc, Q); crit_g.append(dict(C=C, Q=round(Q, 4), gprime_critical_FLOPs_per_token=6 * dQ / dN))
pd.DataFrame(crit_g).to_csv(P('T06_critical_quality_cost_gprime.csv'), index=False)

# ---- 领域替代/互补: 在单纯形上沿"以 i 换 j"方向 e_i-e_j 的曲率与互换容忍度
# M6 沿 e_i-e_j 的二阶导 κ_ij = -γ_i/(p_i+ε)^2 - γ_j/(p_j+ε)^2 (1M 口径); κ>0 → 互换有凸性惩罚(二者互补、宜并存); κ≤0 → 无惩罚(可替代)
# 容忍度 t_ij = sqrt(2·τ/κ_ij): 在 p 处把 t 份额由 j 换成 i(忽略一阶项)使 Loss 上升 τ 所需份额; 大尺度下曲率乘以尺度因子 s(N)
TAU = 0.005
sig = []
for i, j in itertools.combinations(range(17), 2):
    for nm, p in [('p_ref', pref), ('p*', pstar)]:
        k_ = -gam[i] / (p[i] + eps) ** 2 - gam[j] / (p[j] + eps) ** 2
        g1 = (bet[i] + gam[i] / (p[i] + eps)) - (bet[j] + gam[j] / (p[j] + eps))
        row = dict(point=nm, domain_i=DOM[i], domain_j=DOM[j], p_i=p[i], p_j=p[j], first_order_i_minus_j=g1, curvature_1M=k_,
                   relation=('互补(互换受凸性惩罚)' if k_ > 0 else '可替代(无凸性惩罚)'))
        for NB in [0.001, 1, 70, 1000]:
            N = NB * 1e9; R = L_full(N, 20 * N, Qref) - E; sc = b1 * (R / R1M) ** eta
            row[f'swap_tolerance_share_at_{NB}B'] = np.sqrt(2 * TAU / (k_ * sc)) if k_ > 0 else np.inf
        sig.append(row)
SG = pd.DataFrame(sig); SG.to_csv(P('T06_domain_pair_curvature_substitution.csv'), index=False)
# 各域边际价值(沿单纯形的方向导数, 相对均值)及其随规模的衰减
gr = []
for nm, p in [('p_ref', pref), ('p*', pstar)]:
    g_ = np.array([bet[i] + gam[i] / (p[i] + eps) for i in range(17)]); g_c = g_ - np.sum(p * g_)  # 投影到单纯形切空间(以 p 加权中心化)
    for NB, tpp in [(0.001, 1000), (1, 20), (70, 20), (1000, 20)]:
        N = NB * 1e9; D = N * tpp; R = L_full(N, D, Qref) - E; scale = b1 * (R / R1M) ** eta
        for i in range(17):
            gr.append(dict(point=nm, N_B=NB, domain=DOM[i], share=p[i], dL_dp_centered_1M_units=g_c[i], dL_dp_at_scale=g_c[i] * scale, scale_factor=scale))
pd.DataFrame(gr).to_csv(P('T06_domain_marginal_value_by_scale.csv'), index=False)
# 领域族交互(问题一)在不同规模下的绝对量级
fam = pd.read_csv(f'{Q1}/S08/S08_family_quadratic_scheffe_interactions.csv'); fam = fam[(fam.target == 'L_avg') & fam.term.str.contains('×')].copy()
for NB in [0.001, 1, 70, 1000]:
    N = NB * 1e9; R = L_full(N, 20 * N, Qref) - E; fam[f'coef_at_{NB}B'] = fam.coef * (R / R1M) ** eta
fam.to_csv(P('T06_family_interactions_scaled.csv'), index=False)
print(json.dumps(PAR, indent=1)); print('eta60,1B', eta_60, eta_1B)
print(pd.DataFrame(chk).round(4)); print(EL[EL.tokens_per_param == 20].round(4).to_string()); print(SUB[(SUB.tokens_per_param == 20)&(SUB.dQ==0.1)].round(3).to_string()); print(SUB[SUB.dQ>0.1].groupby(['Q','dQ']).substitutable_by_N.mean())
print(CO.round(8).to_string()); print(pd.DataFrame(crit_g)); print(pd.DataFrame(crit).round(2).to_string())
print(SG[SG.point == 'p*'].sort_values('curvature_1M', ascending=False).round(4).head(8).to_string()); print(SG[SG.point == 'p*'].sort_values('curvature_1M').round(4).head(6).to_string())

"""问题三主分析(v2)
V01 C7 与上下文长度可行集、临界值 L_crit=6/η、数据墙锚点
V02 三档预算 × 4 种成本函数 × 6 个 L_ctx × 2 种配比: 最优配置 + 独立求解器核验 + 基线对照
V03 1e15–1e27 预算扫描: 约束集(状态)、局部标度指数、边际优先序、预算份额 → 结构性转移识别
V04 临界预算的解析判据与数值二分核对
V05 KKT 核验; 配比 p 与预算无关的支配性论证(数值)
V06 敏感性与不确定性: c_N,c_D 自助 / η / Q_ref; Q0 情景; 成本参数 ±50%; 数据墙
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import optimize, stats
sys.path.insert(0, 'code'); from q3lib import *
from common import CHIN, compute_optimal
O = 'out'; P = lambda s, f: (os.makedirs(os.path.join(O, s), exist_ok=True), os.path.join(O, s, f))[1]
rng = np.random.default_rng(2026)
law = Law(); FORMS = ['exp', 'power', 'log', 'filter']
# ================================================= V01
c7 = pd.read_csv('data/model_architecture_metadata.csv')
c7['N_est_B'] = (12 * c7.n_layers * c7.d_model ** 2 + c7.vocab_size * c7.d_model) / 1e9
c7['L_over_Lcrit'] = c7.max_position_embeddings / L_CRIT
c7['k(L)=6(1+L/Lcrit)'] = 6 * (1 + c7.L_over_Lcrit)
c7['attn_share_in_train+attn'] = c7.max_position_embeddings / (L_CRIT + c7.max_position_embeddings)
c7['training_tokens(T)'] = c7.training_data_TB
c7.to_csv(P('V01', 'V01_C7_with_context_overhead.csv'), index=False)
LSET = sorted(int(x) for x in c7.max_position_embeddings.unique())
fq = c7.groupby('max_position_embeddings').agg(n_models=('model_name', 'size'), 最早代表=('model_name', lambda s: '; '.join(s.str.split('/').str[-1].head(3))),
                                              训练tokens_T_median=('training_data_TB', 'median')).reset_index()
fq['L/L_crit'] = fq.max_position_embeddings / L_CRIT; fq['k(L)/6'] = 1 + fq['L/L_crit']; fq['C_attn/C_train'] = fq['L/L_crit']
fq.to_csv(P('V01', 'V01_feasible_Lctx_set.csv'), index=False)
DWALL = float(c7.training_data_TB.max()) * 1e12
json.dump(dict(eta=ETA_ATT, L_crit=L_CRIT, derivation='C_attn = C_train ⇔ η·N·D·L = 6·N·D ⇔ L_crit = 6/η = 30000', feasible_L=LSET,
               n_models=int(len(c7)), share_models_L_ge_Lcrit=float((c7.max_position_embeddings >= L_CRIT).mean()),
               data_wall_tokens=DWALL, data_wall_source='C7 training_data_TB 最大值(Qwen2.5 系列 18T tokens)'),
          open(P('V01', 'V01_critical_context.json'), 'w'), ensure_ascii=False, indent=1)
LGRID = LSET + [int(L_CRIT)]
# ================================================= V02
BUD = [1e19, 1e22, 1e24]; rows = []
for C, gf, Lc, pn in itertools.product(BUD, FORMS, LGRID, ['p*', 'p_ref']):
    lw = Law(dM6=DM6[pn]); s = solve(lw, C, Lc, gf); s0 = solve(lw, C, Lc, gf, Qfix=lw.Q0)
    ch = solve_check(lw, C, Lc, gf) if Lc in (2048, 32768) else None
    s.update(p=pn, regime=regime(s), cost_form=GFORMS_EXT[gf]['name'], N_noQ=s0['N'], D_noQ=s0['D'], L_noQ=s0['L'], gain_vs_noQ=s0['L'] - s['L'],
             tpp_noQ=s0['D'] / s0['N'], check_L=(ch['L'] if ch else np.nan), check_absdiff_L=(abs(ch['L'] - s['L']) if ch else np.nan),
             check_reldiff_N=(abs(ch['N'] / s['N'] - 1) if ch else np.nan))
    rows.append(s)
M = pd.DataFrame(rows); M.to_csv(P('V02', 'V02_optimal_allocation_full_grid.csv'), index=False)
main = M[(M.L_ctx == 2048) & (M.p == 'p*')].copy(); main.to_csv(P('V02', 'V02_main_three_budgets_Lctx2048.csv'), index=False)
# 与"不同配比"对照
pc = M[(M.L_ctx == 2048)].pivot_table(index=['C', 'g_form'], columns='p', values='L').reset_index(); pc['gain_pstar_vs_pref'] = pc['p_ref'] - pc['p*']
pc.to_csv(P('V02', 'V02_mixture_gain_by_budget.csv'), index=False)
# ================================================= V03
Cs = np.logspace(15, 27, 121); sw = []
for gf in FORMS:
    for Lc in [2048, 8192, 32768, 131072]:
        for wall in [None, DWALL]:
            for C in Cs:
                s = solve(law, C, Lc, gf, Dmax=wall); s.update(regime=regime(s), wall=('D≤1.8e13' if wall else '无'))
                dN, dD, dQ = law.grad(s['N'], s['D'], s['Q']); k = 6 * (1 + Lc / L_CRIT); g = GFORMS_EXT[gf]
                dg = max(g['g'](s['Q']) - g['g'](law.Q0), 0)
                s.update(mu_N=-dN / (k * s['D']), mu_D=-dD / (k * s['N'] + dg), mu_Q=-dQ / (s['D'] * g['gp'](s['Q'])))  # 每 FLOP 边际降 Loss
                sw.append(s)
SW = pd.DataFrame(sw); SW['lnN'] = np.log(SW.N); SW['lnD'] = np.log(SW.D); SW.to_csv(P('V03', 'V03_budget_sweep.csv'), index=False)
tr = []
for (gf, Lc, wl), g in SW.groupby(['g_form', 'L_ctx', 'wall']):
    g = g.sort_values('C'); prev = None
    for _, r in g.iterrows():
        if prev is not None and r.regime != prev.regime:
            tr.append(dict(g_form=gf, L_ctx=Lc, wall=wl, type='I 约束集切换', from_state=prev.regime, to_state=r.regime, C_low=prev.C, C_high=r.C))
        prev = r
# 类型 III: 边际优先序(在"不投质量"的最优分配处, 质量 vs 参数 每 FLOP 收益谁大)
for gf in FORMS:
    for Lc in [2048, 8192, 32768, 131072]:
        k = 6 * (1 + Lc / L_CRIT); prev = None
        for C in Cs:
            s0 = solve(law, C, Lc, gf, Qfix=law.Q0); dN, dD, dQ = law.grad(s0['N'], s0['D'], law.Q0)
            muN = -dN / (k * s0['D']); muQ = -dQ / (s0['D'] * GFORMS_EXT[gf]['gp'](law.Q0)); order = '质量优先' if muQ > muN else '规模优先'
            if prev and order != prev[0]:
                tr.append(dict(g_form=gf, L_ctx=Lc, wall='无', type='III 边际优先序反转', from_state=prev[0], to_state=order, C_low=prev[1], C_high=C))
            prev = (order, C)
TR = pd.DataFrame(tr); TR.to_csv(P('V03', 'V03_structural_transitions_detected.csv'), index=False)
# 类型 II: 局部标度指数 e_X = dlnX*/dlnC 与分段幂律断点(BIC)
el = []
for (gf, Lc, wl), g in SW.groupby(['g_form', 'L_ctx', 'wall']):
    g = g.sort_values('C'); x = np.log(g.C.values)
    for v in ['N', 'D', 'tokens_per_param']:
        d = np.gradient(np.log(g[v].values), x)
        for C, di in zip(g.C.values, d):
            el.append(dict(g_form=gf, L_ctx=Lc, wall=wl, C=C, var=v, local_exponent=di))
EL = pd.DataFrame(el); EL.to_csv(P('V03', 'V03_local_scaling_exponents.csv'), index=False)
def segfit(x, y, kmax=3):
    best = None
    for k in range(kmax + 1):
        for bps in itertools.combinations(range(6, len(x) - 6, 3), k):
            X = np.column_stack([np.ones_like(x), x] + [np.maximum(0, x - x[b]) for b in bps])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); rss = np.sum((y - X @ beta) ** 2) + 1e-30
            bic = len(x) * np.log(rss / len(x)) + (2 + 2 * k) * np.log(len(x))
            if best is None or bic < best[0]:
                best = (bic, [float(x[b]) for b in bps], beta.tolist())
    return best
bp = []
for (gf, Lc, wl), g in SW.groupby(['g_form', 'L_ctx', 'wall']):
    if Lc not in (2048, 131072):
        continue
    g = g.sort_values('C').iloc[::2]; x = np.log10(g.C.values)
    for v in ['N', 'D', 'share_quality']:
        y = np.log10(g[v].values) if v in ('N', 'D') else g[v].values
        bic, bps, beta = segfit(x, y, 2)
        bp.append(dict(g_form=gf, L_ctx=Lc, wall=wl, var=v, n_breaks=len(bps), breakpoints_log10C=[round(b, 2) for b in bps], base_slope=beta[1],
                       slope_changes=[round(b, 4) for b in beta[2:]]))
pd.DataFrame(bp).to_csv(P('V03', 'V03_segmented_powerlaw_breakpoints.csv'), index=False)
# 预算份额的"同位性"检验: 若结构不变, 份额向量应与 C 无关 → 报告各区段份额均值/极差
SW['decade'] = np.floor(np.log10(SW.C)).astype(int)
SW[(SW.L_ctx == 2048) & (SW.wall == '无')].groupby(['g_form', 'decade'])[['share_train', 'share_quality', 'share_attn', 'Q', 'tokens_per_param']].mean().to_csv(P('V03', 'V03_budget_shares_by_decade.csv'))
# ================================================= V04 解析临界预算
def ratio_gstar(C, Lc, Q):
    """在 Q 固定时的最优 (N,D) 处, 计算 g'* = k·(∂L/∂Q)/(∂L/∂N)·(1/D)·D —— 即使 μ_Q=μ_N 的 g'(Q)"""
    k = 6 * (1 + Lc / L_CRIT); s = solve(law, C, Lc, 'exp', Qfix=Q); dN, dD, dQ = law.grad(s['N'], s['D'], Q)
    return k * dQ / dN   # 注意: μ_Q = -dQ/(D g'), μ_N = -dN/(k D) → 相等时 g' = k dQ/dN
an = []
for gf in FORMS:
    for Lc in LGRID:
        gp = GFORMS_EXT[gf]['gp']
        f1 = lambda lc: np.log(ratio_gstar(np.exp(lc), Lc, law.Q0)) - np.log(gp(law.Q0))
        try:
            C1 = float(np.exp(optimize.brentq(f1, np.log(1e13), np.log(1e28), xtol=1e-6)))
        except ValueError:
            C1 = np.nan
        Qtop = 0.9999 if gf == 'filter' else 1.0
        f2 = lambda lc: np.log(ratio_gstar(np.exp(lc), Lc, Qtop)) - np.log(gp(Qtop))
        try:
            C2 = float(np.exp(optimize.brentq(f2, np.log(1e13), np.log(1e28), xtol=1e-6)))
        except ValueError:
            C2 = np.nan
        an.append(dict(g_form=gf, L_ctx=Lc, gprime_Q0=gp(law.Q0), gprime_Qtop=gp(Qtop), C1_analytic_Q_leaves_Q0=C1, C2_analytic_Q_reaches_top=C2))
AN = pd.DataFrame(an)
# 数值二分核对
def Qstar(C, gf, Lc): return solve(law, C, Lc, gf, nD=200, nQ=161)['Q']
nb = []
for _, r in AN.iterrows():
    out = {}
    for tag, cond in [('C1_numeric', lambda q: q > law.Q0 + 1e-3), ('C2_numeric', lambda q: q >= (0.999 if r.g_form != 'filter' else 0.9985))]:
        lo, hi = 1e13, 1e28
        if not cond(Qstar(hi, r.g_form, r.L_ctx)):
            out[tag] = np.nan; continue
        if cond(Qstar(lo, r.g_form, r.L_ctx)):
            out[tag] = lo; continue
        for _ in range(45):
            mid = np.sqrt(lo * hi)
            if cond(Qstar(mid, r.g_form, r.L_ctx)):
                hi = mid
            else:
                lo = mid
        out[tag] = hi
    nb.append(out)
AN = pd.concat([AN, pd.DataFrame(nb)], axis=1)
AN['C1_rel_diff'] = AN.C1_numeric / AN.C1_analytic_Q_leaves_Q0 - 1
AN.to_csv(P('V04', 'V04_critical_budgets_analytic_vs_numeric.csv'), index=False)
# g'* 随预算的轨迹(解释为何质量先于规模)
gs = []
for C in np.logspace(15, 27, 49):
    for Lc in [2048, 32768, 131072]:
        gs.append(dict(C=C, L_ctx=Lc, gprime_star_at_Q0=ratio_gstar(C, Lc, law.Q0), gprime_star_at_Q1=ratio_gstar(C, Lc, 1.0)))
pd.DataFrame(gs).to_csv(P('V04', 'V04_gprime_star_trajectory.csv'), index=False)
# ================================================= V05 KKT
kk = []
for C in BUD:
    for gf in FORMS:
        s = solve(law, C, 2048, gf); N, D, Q = s['N'], s['D'], s['Q']; dN, dD, dQ = law.grad(N, D, Q); k = 6 * (1 + 2048 / L_CRIT); g = GFORMS_EXT[gf]
        dg = max(g['g'](Q) - g['g'](law.Q0), 0); lamN = -dN / (k * D); lamD = -dD / (k * N + dg); lamQ = -dQ / (D * g['gp'](Q))
        top = 0.9999 if gf == 'filter' else 1.0
        kk.append(dict(C=C, g_form=gf, N=N, D=D, Q=Q, lambda_N=lamN, lambda_D=lamD, lambda_Q=lamQ, rel_gap_N_D=abs(lamN - lamD) / lamN, lamQ_over_lamN=lamQ / lamN,
                       budget_slack=1 - cost(N, D, Q, 2048, gf, law.Q0) / C,
                       Q_status=('内点(应有 λ_Q≈λ_N)' if law.Q0 + 1e-4 < Q < top - 1e-4 else ('上界(应有 λ_Q≥λ_N)' if Q >= top - 1e-4 else '下界(应有 λ_Q≤λ_N)'))))
pd.DataFrame(kk).to_csv(P('V05', 'V05_KKT_check.csv'), index=False)
# p 的支配性: 对若干 (N,D,Q), L 对 ΔM6 单调递增 → argmin_p 与预算无关; 数值: 各预算下四种候选配比的排序
cand = {'p_ref': 0.0, 'p*(Q1集成)': DM6['p*'], 'p_M6opt(信赖域)': DM6['p_M6opt']}
_mx = pd.read_csv('data/q2_out/T05/T05_mixture_Psi_and_Q.csv').set_index('mixture'); ref = _mx.loc['reference_mean(≈Pile 自然配比)', 'M6_L_1M']
for nm in ['均匀 1/17', '单一 pile_cc', '单一 github']:
    cand[nm] = float(_mx.loc[nm, 'M6_L_1M'] - ref)
pr = []
for C in [1e17, 1e19, 1e22, 1e24, 1e26]:
    for nm, dm in cand.items():
        s = solve(Law(dM6=dm), C, 2048, 'exp'); pr.append(dict(C=C, mixture=nm, dM6=dm, L=s['L'], N=s['N'], D=s['D'], Q=s['Q']))
PR = pd.DataFrame(pr); PR['rank_within_C'] = PR.groupby('C').L.rank(); PR.to_csv(P('V05', 'V05_mixture_dominance_by_budget.csv'), index=False)
# ================================================= V06 敏感性与不确定性
bs = pd.read_csv('data/q2_out/T04/T04b_bootstrap_cN_cD.csv'); bs = bs[bs.kind == 'cell_cluster'].sample(150, random_state=1)
unc = []
for (cN, cD), et, qr in zip(bs[['c_N', 'c_D']].values, rng.uniform(G['eta_from_60M'], G['eta_from_1B'], len(bs)), G['Qref'] * (1 + 0.03 * rng.standard_normal(len(bs)))):
    Pp = with_c(cN, cD, qr); Pp['eta'] = et; lw = Law(Pp)
    for C in BUD:
        for gf in FORMS:
            s = solve(lw, C, 2048, gf, nD=140, nQ=101); unc.append(dict(cN=cN, cD=cD, eta=et, Qref=qr, C=C, g_form=gf, N=s['N'], D=s['D'], Q=s['Q'], L=s['L'],
                                                                         share_quality=s['share_quality'], tpp=s['tokens_per_param'], regime=regime(s)))
UN = pd.DataFrame(unc); UN.to_csv(P('V06', 'V06_uncertainty_draws.csv'), index=False)
UN.groupby(['C', 'g_form'])[['N', 'D', 'Q', 'L', 'share_quality', 'tpp']].quantile([.025, .5, .975]).to_csv(P('V06', 'V06_uncertainty_quantiles.csv'))
UN.groupby(['C', 'g_form']).regime.agg(lambda s: s.value_counts(normalize=True).round(3).to_dict()).to_csv(P('V06', 'V06_regime_stability.csv'))
q1q = pd.read_csv('data/q1_out/S04/S04_domain_level_Q_all_methods.csv'); q1q = q1q[(q1q.score == 'Q_resolved') & (q1q.set == 'A1')].set_index('domain').token_weighted_mean
q0s = []
for nm, q0 in [('github 域(最低)', float(q1q['github'])), ('wikipedia 域', float(q1q['wikipedia'])), ('Pile 自然配比 Q_ref', G['Qref']), ('arxiv 域(最高)', float(q1q['arxiv']))]:
    lw = Law(Q0=q0)
    for C in BUD:
        for gf in FORMS:
            s = solve(lw, C, 2048, gf); s.update(Q0_scenario=nm, Q0=q0, regime=regime(s)); q0s.append(s)
pd.DataFrame(q0s).to_csv(P('V06', 'V06_Q0_scenarios.csv'), index=False)
# 成本参数 ±50% (γ 与 λ) 对临界预算 C1 的影响(指数型/幂型/对数型)
cp = []
base_par = {'exp': (1e7, 6.0), 'power': (5e9, 4.0), 'log': (2e9, 10.0)}
for gf, (ga, la) in base_par.items():
    for fg, fl in [(1, 1), (0.5, 1), (2, 1), (1, 0.8), (1, 1.2)]:
        ga2, la2 = ga * fg, la * fl
        gp = {'exp': lambda q: ga2 * la2 * np.exp(la2 * q), 'power': lambda q: ga2 * la2 * q ** (la2 - 1), 'log': lambda q: ga2 * la2 / (1 + la2 * q)}[gf]
        f1 = lambda lc: np.log(ratio_gstar(np.exp(lc), 2048, law.Q0)) - np.log(gp(law.Q0))
        try:
            C1 = float(np.exp(optimize.brentq(f1, np.log(1e12), np.log(1e30))))
        except ValueError:
            C1 = np.nan
        cp.append(dict(g_form=gf, gamma_factor=fg, lambda_factor=fl, gamma=ga2, lam=la2, gprime_Q0=gp(law.Q0), C1_Q_leaves_Q0=C1))
pd.DataFrame(cp).to_csv(P('V06', 'V06_cost_parameter_sensitivity.csv'), index=False)
print(main[['C', 'g_form', 'N', 'D', 'Q', 'L', 'tokens_per_param', 'share_train', 'share_quality', 'share_attn', 'regime', 'L_noQ', 'gain_vs_noQ', 'check_absdiff_L']].to_string())
print(TR.to_string()); print(AN.to_string()); print(pd.DataFrame(cp).to_string())

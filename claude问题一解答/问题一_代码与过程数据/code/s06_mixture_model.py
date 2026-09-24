"""S06–S09 领域配比 p 与交叉熵损失的定量关系 (A4–A15), 检验(A6–A11), 外推稳健性(A12–A15), 质量Q接入(A16)
用法: python s06_mixture_model.py <regmix_tables目录> <A16 mapping csv> <S04域级Q csv> <输出目录>
"""
import os, sys, json, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
import lightgbm as lgb
warnings.filterwarnings('ignore')
TB, MAPF, QF, OUT = sys.argv[1:5]
for s in ['S06', 'S07', 'S08', 'S09']:
    os.makedirs(os.path.join(OUT, s), exist_ok=True)
P = lambda s, f: os.path.join(OUT, s, f)
rng = np.random.default_rng(7)

DOM = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en', 'dm_mathematics', 'github', 'philpapers',
       'stackexchange', 'enron_emails', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc', 'europarl', 'hackernews',
       'pubmed_abstracts', 'uspto_backgrounds']
VALN = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics', 'github', 'stackexchange', 'gutenberg_pg_19',
       'pile_cc', 'ubuntu_irc', 'hackernews', 'pubmed_abstracts', 'uspto_backgrounds']
VAL = ['loss_' + x for x in VALN]
NOLOSS = [x for x in DOM if x not in VALN]
TARGETS = VAL + ['L_avg']


def load(mix, loss):
    m = pd.read_csv(os.path.join(TB, mix)); l = pd.read_csv(os.path.join(TB, loss))
    m = m.rename(columns=lambda c: c.replace('train_the_pile_', '')); l = l.rename(columns=lambda c: c.replace('metric/the_pile_', '').replace('_val_loss', '')).rename(columns=lambda c: c if c == 'index' else 'loss_' + c)
    df = m.merge(l, on='index', how='inner', validate='1:1')
    raw_sum = df[DOM].sum(axis=1)
    df[DOM] = df[DOM].div(raw_sum, axis=0)  # 单纯形: 舍入误差 → 重新归一化
    df['raw_sum'] = raw_sum; df['L_avg'] = df[VAL].mean(axis=1)
    return df


S = {'train_1M': load('train_mixture_1m.csv', 'train_pile_loss_1m.csv'),
     'test_1M': load('test_mixture_1m.csv', 'test_pile_loss_1m.csv'),
     'test_60M': load('test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
     'test_1B': load('test_mixture_1B.csv', 'test_pile_loss_1B.csv'),
     'est_10B': load('est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
     'est_70B': load('est_mixture_70b.csv', 'est_pile_loss_70b.csv')}
audit = []
for k, df in S.items():
    audit.append(dict(set=k, n=len(df), raw_sum_min=df.raw_sum.min(), raw_sum_max=df.raw_sum.max(), n_zero_share=int((df[DOM] == 0).sum().sum()),
                      zero_share_frac=float((df[DOM] == 0).values.mean()), L_avg_mean=df.L_avg.mean(), L_avg_sd=df.L_avg.std(),
                      L_avg_min=df.L_avg.min(), L_avg_max=df.L_avg.max(), any_nan=bool(df[DOM + VAL].isna().any().any())))
pd.DataFrame(audit).to_csv(P('S06', 'S06_data_audit.csv'), index=False)
for k, df in S.items():
    df.to_csv(P('S06', f'S06_clean_{k}.csv'), index=False)
# 各数据集配比描述
desc = pd.concat({k: df[DOM].describe().T[['mean', 'std', 'min', '50%', 'max']] for k, df in S.items()}, axis=1)
desc['zero_frac_train'] = (S['train_1M'][DOM] == 0).mean(); desc.to_csv(P('S06', 'S06_mixture_descriptives.csv'))
# 各 Loss 与自身域配比的相关 (探索)
tr = S['train_1M']
exp_rows = []
for v in VAL:
    for dname in DOM:
        exp_rows.append(dict(val_loss=v.replace('loss_', ''), train_domain=dname, spearman=stats.spearmanr(tr[dname], tr[v]).statistic))
pd.DataFrame(exp_rows).pivot(index='train_domain', columns='val_loss', values='spearman').to_csv(P('S06', 'S06_spearman_share_vs_loss_train.csv'))

# ============================================================== 模型
PAIRS = list(itertools.combinations(range(17), 2))


def f_lin(M): return M
def f_quad(M): return np.column_stack([M] + [M[:, i] * M[:, j] for i, j in PAIRS])


class Scheffe:
    """Scheffé 混合模型: 无截距 (Σp=1 已吸收截距), 线性或二次(ridge)"""
    def __init__(self, kind='lin', alpha=0.0):
        self.kind, self.alpha = kind, alpha
    def fit(self, M, y):
        X = f_lin(M) if self.kind == 'lin' else f_quad(M)
        self.m = (LinearRegression(fit_intercept=False) if self.alpha == 0 else Ridge(alpha=self.alpha, fit_intercept=False)).fit(X, y); return self
    def predict(self, M):
        return self.m.predict(f_lin(M) if self.kind == 'lin' else f_quad(M))


class MixLaw:
    """数据混合律 (Ye et al., 2024 形式): L = c + exp(Σ t_i p_i)  (k 被 Σp=1 吸收); 非线性最小二乘"""
    def fit(self, M, y):
        c0 = 0.8 * y.min()
        t0 = LinearRegression(fit_intercept=False).fit(M, np.log(y - c0)).coef_
        f = lambda th: th[0] + np.exp(M @ th[1:]) - y
        r = optimize.least_squares(f, np.r_[c0, t0], bounds=(np.r_[-np.inf, [-np.inf] * 17], np.r_[y.min() - 1e-6, [np.inf] * 17]))
        self.th = r.x; return self
    def predict(self, M): return self.th[0] + np.exp(M @ self.th[1:])


class OwnLog:
    """线性 + 自身域对数边际递减项: L_j = Σ β_i p_i + g ln(p_j + ε) (仅域目标)"""
    def __init__(self, j, eps): self.j, self.eps = j, eps
    def X(self, M): return np.column_stack([M, np.log(M[:, self.j] + self.eps)])
    def fit(self, M, y): self.m = LinearRegression(fit_intercept=False).fit(self.X(M), y); return self
    def predict(self, M): return self.m.predict(self.X(M))


class LogShare:
    """对数边际递减配比律: L = Σ β_i p_i + Σ γ_i ln(p_i + ε)  (34 参数; γ_i<0 表示该域份额的边际降Loss作用递减)"""
    def __init__(self, eps=1e-3, alpha=0.0): self.eps, self.alpha = eps, alpha
    def X(self, M): return np.column_stack([M, np.log(M + self.eps)])
    def fit(self, M, y):
        self.m = (LinearRegression(fit_intercept=False) if self.alpha == 0 else Ridge(alpha=self.alpha, fit_intercept=False)).fit(self.X(M), y); return self
    def predict(self, M): return self.m.predict(self.X(M))


class GBM:
    def fit(self, M, y):
        self.m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0).fit(M, y); return self
    def predict(self, M): return self.m.predict(M)


def cv_score(make, M, y, k=5):
    pr = np.zeros_like(y)
    for tri, tei in KFold(k, shuffle=True, random_state=0).split(M):
        pr[tei] = make().fit(M[tri], y[tri]).predict(M[tei])
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def metrics(y, p):
    return dict(R2=1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), RMSE=np.sqrt(((y - p) ** 2).mean()), MAE=np.abs(y - p).mean(),
                pearson=stats.pearsonr(y, p).statistic, spearman=stats.spearmanr(y, p).statistic, kendall=stats.kendalltau(y, p).statistic)


def recal_cv(y, p):
    """跨尺度: 1M 模型预测值与大尺度真实 Loss 量纲不同 → 2 折交叉的仿射校准 y≈a+b·p 后的 R²"""
    idx = rng.permutation(len(y)); h = len(y) // 2; out = np.zeros_like(y)
    for a_, b_ in [(idx[:h], idx[h:]), (idx[h:], idx[:h])]:
        co = np.polyfit(p[a_], y[a_], 1); out[b_] = np.polyval(co, p[b_])
    return 1 - ((y - out) ** 2).sum() / ((y - y.mean()) ** 2).sum()


Mtr = tr[DOM].values
# ridge α 与 ε 选择 (5 折 CV, 仅用训练集)
sel = []
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]
best_alpha = {}
for t in TARGETS:
    y = tr[t].values; sc = {a: cv_score(lambda a=a: Scheffe('quad', a), Mtr, y) for a in alphas}
    best_alpha[t] = max(sc, key=sc.get)
    for a, v in sc.items():
        sel.append(dict(target=t, model='quad_ridge', hyper=a, cv_R2=v))
best_eps = {}
for t in VAL:
    y = tr[t].values; j = DOM.index(t.replace('loss_', '')); sc = {e: cv_score(lambda e=e: OwnLog(j, e), Mtr, y) for e in [1e-4, 1e-3, 1e-2, 3e-2]}
    best_eps[t] = max(sc, key=sc.get)
    for e, v in sc.items():
        sel.append(dict(target=t, model='lin_ownlog', hyper=e, cv_R2=v))
best_eps6 = {}
for t in TARGETS:
    y = tr[t].values; sc = {e: cv_score(lambda e=e: LogShare(e), Mtr, y) for e in [1e-4, 1e-3, 3e-3, 1e-2, 3e-2]}
    best_eps6[t] = max(sc, key=sc.get)
    for e, v in sc.items():
        sel.append(dict(target=t, model='log_share', hyper=e, cv_R2=v))
pd.DataFrame(sel).to_csv(P('S07', 'S07_hyperparameter_cv.csv'), index=False)

MODELS = {
    'M1_Scheffe_linear': lambda t: Scheffe('lin'),
    'M2_Scheffe_quadratic_ridge': lambda t: Scheffe('quad', best_alpha[t]),
    'M3_mixing_law_exp': lambda t: MixLaw(),
    'M4_linear_ownlog': lambda t: OwnLog(DOM.index(t.replace('loss_', '')), best_eps[t]) if t in VAL else None,
    'M5_LightGBM': lambda t: GBM(),
    'M6_log_share': lambda t: LogShare(best_eps6[t]),
}
rows, preds = [], {}
for t in TARGETS:
    y = tr[t].values
    for mn, mk in MODELS.items():
        mdl = mk(t)
        if mdl is None:
            continue
        mdl.fit(Mtr, y); preds[(mn, t)] = mdl
        r = dict(target=t, model=mn, n_params={'M1_Scheffe_linear': 17, 'M2_Scheffe_quadratic_ridge': 153, 'M3_mixing_law_exp': 18,
                                               'M4_linear_ownlog': 18, 'M5_LightGBM': np.nan, 'M6_log_share': 34}[mn],
                 train_R2=metrics(y, mdl.predict(Mtr))['R2'], cv5_R2=cv_score(lambda: mk(t), Mtr, y))
        for sn in ['test_1M', 'test_60M', 'test_1B']:
            te = S[sn]; yt = te[t].values; pt = mdl.predict(te[DOM].values); m_ = metrics(yt, pt)
            if sn == 'test_1M':
                r.update({f'{sn}_{k}': v for k, v in m_.items()})
            else:
                r.update({f'{sn}_spearman': m_['spearman'], f'{sn}_kendall': m_['kendall'], f'{sn}_pearson': m_['pearson'], f'{sn}_recal_R2_cv2': recal_cv(yt, pt)})
        rows.append(r)
VR = pd.DataFrame(rows); VR.to_csv(P('S07', 'S07_validation_all_models_all_targets.csv'), index=False)
summ = VR.groupby('model')[[c for c in VR.columns if c not in ['target', 'model', 'n_params']]].mean()
summ.to_csv(P('S07', 'S07_validation_summary_mean_over_targets.csv'))
# 逐样本预测 (L_avg, 所有检验集) 过程数据
pp = []
for sn in ['train_1M', 'test_1M', 'test_60M', 'test_1B', 'est_10B', 'est_70B']:
    te = S[sn]; row = pd.DataFrame({'set': sn, 'index': te['index'], 'L_avg_obs': te.L_avg})
    for mn in MODELS:
        if (mn, 'L_avg') in preds:
            row['pred_' + mn] = preds[(mn, 'L_avg')].predict(te[DOM].values)
    pp.append(row)
pd.concat(pp).to_csv(P('S07', 'S07_predictions_L_avg_per_mixture.csv'), index=False)
# 选模: 在 1M 检验 R² 与跨尺度秩相关上综合最优的"主模型"
# Top-k 命中: 预测最优 k 个配方是否确为真实最优
topk = []
for sn in ['test_1M', 'test_60M', 'test_1B']:
    te = S[sn]; y = te.L_avg.values
    for mn in MODELS:
        if (mn, 'L_avg') not in preds:
            continue
        p = preds[(mn, 'L_avg')].predict(te[DOM].values)
        for k in [5, 10, 20]:
            k2 = min(k, len(y) // 3)
            topk.append(dict(set=sn, model=mn, k=k2, precision_at_k=len(set(np.argsort(p)[:k2]) & set(np.argsort(y)[:k2])) / k2,
                             true_rank_of_pred_best=int(stats.rankdata(y)[np.argmin(p)])))
pd.DataFrame(topk).to_csv(P('S07', 'S07_topk_hit_L_avg.csv'), index=False)

# ============================================================== 领域效应 (S08): 训练+1M检验合并(768)重估
comb = pd.concat([S['train_1M'], S['test_1M']], ignore_index=True); Mc = comb[DOM].values
s_ref = Mc.mean(0)  # 参照配比: 实验配方均值(近似自然 token 分布)
pd.Series(s_ref, index=DOM).to_csv(P('S08', 'S08_reference_mixture_mean.csv'))


def cox_effect(beta, s, delta=0.1):
    Ls = s @ beta
    return delta * (beta - Ls) / (1 - s)


eff_rows, coef_rows = [], []
B = 300
for t in TARGETS:
    y = comb[t].values; beta = LinearRegression(fit_intercept=False).fit(Mc, y).coef_
    bs = np.array([LinearRegression(fit_intercept=False).fit(Mc[ix], y[ix]).coef_ for ix in (rng.integers(0, len(y), len(y)) for _ in range(B))])
    ce = cox_effect(beta, s_ref); ceb = np.array([cox_effect(b, s_ref) for b in bs])
    for i, dn in enumerate(DOM):
        coef_rows.append(dict(target=t, domain=dn, beta=beta[i], beta_ci_low=np.percentile(bs[:, i], 2.5), beta_ci_high=np.percentile(bs[:, i], 97.5)))
        eff_rows.append(dict(target=t, domain=dn, cox_effect_plus0_1=ce[i], ci_low=np.percentile(ceb[:, i], 2.5), ci_high=np.percentile(ceb[:, i], 97.5),
                             significant=bool(np.percentile(ceb[:, i], 2.5) > 0 or np.percentile(ceb[:, i], 97.5) < 0)))
pd.DataFrame(coef_rows).to_csv(P('S08', 'S08_scheffe_linear_coefficients_bootstrap.csv'), index=False)
EF = pd.DataFrame(eff_rows); EF.to_csv(P('S08', 'S08_cox_effect_plus0.1_bootstrap.csv'), index=False)
EF.pivot(index='domain', columns='target', values='cox_effect_plus0_1').reindex(DOM).to_csv(P('S08', 'S08_cox_effect_matrix_17x14.csv'))
# 主解释模型 M6 (对数边际递减律) 的 Cox 方向有限差分效应: E_i = f(s + Δ(e_i - s)/(1 - s_i)) - f(s), Δ=0.1
def cox_fd(model, s, delta=0.1):
    base = model.predict(s[None])[0]; out = []
    for i in range(17):
        x = s + delta * (np.eye(17)[i] - s) / (1 - s[i]); out.append(model.predict(x[None])[0] - base)
    return np.array(out)
eff6 = []
for t in TARGETS:
    y = comb[t].values; m6 = LogShare(best_eps6[t]).fit(Mc, y); ce = cox_fd(m6, s_ref)
    ceb = np.array([cox_fd(LogShare(best_eps6[t]).fit(Mc[ix], y[ix]), s_ref) for ix in (rng.integers(0, len(y), len(y)) for _ in range(200))])
    g6 = m6.m.coef_[17:]
    for i, dn in enumerate(DOM):
        eff6.append(dict(target=t, domain=dn, cox_effect_plus0_1=ce[i], ci_low=np.percentile(ceb[:, i], 2.5), ci_high=np.percentile(ceb[:, i], 97.5),
                         significant=bool(np.percentile(ceb[:, i], 2.5) > 0 or np.percentile(ceb[:, i], 97.5) < 0),
                         beta_linear=m6.m.coef_[i], gamma_log=g6[i], eps=best_eps6[t]))
EF6 = pd.DataFrame(eff6); EF6.to_csv(P('S08', 'S08_M6_cox_effect_plus0.1_bootstrap.csv'), index=False)
EF6.pivot(index='domain', columns='target', values='cox_effect_plus0_1').reindex(DOM).to_csv(P('S08', 'S08_M6_cox_effect_matrix_17x14.csv'))
EF6.pivot(index='domain', columns='target', values='gamma_log').reindex(DOM).to_csv(P('S08', 'S08_M6_gamma_log_matrix_17x14.csv'))
# LightGBM SHAP (L_avg): 各域平均 |贡献|
g = GBM().fit(Mc, comb.L_avg.values); sh = g.m.predict(Mc, pred_contrib=True)[:, :-1]
shp = pd.DataFrame({'domain': DOM, 'mean_abs_shap': np.abs(sh).mean(0),
                    'shap_share_corr': [stats.spearmanr(Mc[:, i], sh[:, i]).statistic for i in range(17)]}).sort_values('mean_abs_shap', ascending=False)
shp.to_csv(P('S08', 'S08_lightgbm_shap_importance_L_avg.csv'), index=False)
# 领域族层面的二次 Scheffé (组合效应/互补-替代)
FAM = {'学术': ['arxiv', 'pubmed_central', 'pubmed_abstracts', 'nih_exporter', 'philpapers'],
       '代码数学': ['github', 'dm_mathematics'],
       '网页通用': ['pile_cc', 'wikipedia_en'],
       '问答对话': ['stackexchange', 'hackernews', 'ubuntu_irc', 'enron_emails'],
       '法律专利': ['freelaw', 'uspto_backgrounds'],
       '书籍议会': ['gutenberg_pg_19', 'europarl']}
FN = list(FAM)
Fc = np.column_stack([comb[FAM[f]].sum(axis=1) for f in FN])
FP = list(itertools.combinations(range(len(FN)), 2))
fq = lambda F: np.column_stack([F] + [F[:, i] * F[:, j] for i, j in FP])
fam_rows = []
for t in ['L_avg', 'loss_pile_cc', 'loss_github', 'loss_arxiv', 'loss_wikipedia_en']:
    y = comb[t].values; mdl = LinearRegression(fit_intercept=False).fit(fq(Fc), y)
    bsc = np.array([LinearRegression(fit_intercept=False).fit(fq(Fc[ix]), y[ix]).coef_ for ix in (rng.integers(0, len(y), len(y)) for _ in range(B))])
    names = FN + [f'{FN[i]}×{FN[j]}' for i, j in FP]
    r2 = metrics(y, mdl.predict(fq(Fc)))['R2']
    # 检验集泛化 (组模型只在 train 上拟合)
    Ftr = np.column_stack([tr[FAM[f]].sum(axis=1) for f in FN]); mtr = LinearRegression(fit_intercept=False).fit(fq(Ftr), tr[t].values)
    te = S['test_1M']; Fte = np.column_stack([te[FAM[f]].sum(axis=1) for f in FN]); r2te = metrics(te[t].values, mtr.predict(fq(Fte)))['R2']
    for k, nm in enumerate(names):
        lo_, hi_ = np.percentile(bsc[:, k], [2.5, 97.5])
        fam_rows.append(dict(target=t, term=nm, coef=mdl.coef_[k], ci_low=lo_, ci_high=hi_, significant=bool(lo_ > 0 or hi_ < 0),
                             interpretation=('' if k < len(FN) else ('互补(协同降Loss)' if hi_ < 0 else ('拮抗/替代(升Loss)' if lo_ > 0 else '不显著'))),
                             in_sample_R2=r2, test1M_R2_train_only=r2te))
pd.DataFrame(fam_rows).to_csv(P('S08', 'S08_family_quadratic_scheffe_interactions.csv'), index=False)
pd.DataFrame([(f, x) for f, xs in FAM.items() for x in xs], columns=['family', 'domain']).to_csv(P('S08', 'S08_family_definition.csv'), index=False)
# 17 域二次 ridge 交互项 (L_avg) 的 bootstrap 稳定性
al = best_alpha['L_avg']; y = comb.L_avg.values
cq = Scheffe('quad', al).fit(Mc, y).m.coef_
bq = np.array([Scheffe('quad', al).fit(Mc[ix], y[ix]).m.coef_ for ix in (rng.integers(0, len(y), len(y)) for _ in range(200))])
qi = []
for k, (i, j) in enumerate(PAIRS):
    lo_, hi_ = np.percentile(bq[:, 17 + k], [2.5, 97.5])
    qi.append(dict(domain_i=DOM[i], domain_j=DOM[j], coef=cq[17 + k], ci_low=lo_, ci_high=hi_, significant=bool(lo_ > 0 or hi_ < 0),
                   co_occurrence_both_gt_1pct=float(((Mc[:, i] > .01) & (Mc[:, j] > .01)).mean())))
pd.DataFrame(qi).sort_values('coef').to_csv(P('S08', 'S08_domain_pair_interactions_ridge_L_avg.csv'), index=False)

# 多样性(配比熵)与 Loss 的关系, 跨尺度
dv = []
for sn, df in S.items():
    H = -(df[DOM].values * np.log(df[DOM].values + 1e-12)).sum(1); nz = (df[DOM].values == 0).sum(1)
    dv.append(dict(set=sn, n=len(df), H_mean=H.mean(), spearman_H_vs_Lavg=stats.spearmanr(H, df.L_avg).statistic,
                   spearman_nzero_vs_Lavg=stats.spearmanr(nz, df.L_avg).statistic))
pd.DataFrame(dv).to_csv(P('S08', 'S08_diversity_vs_loss_by_scale.csv'), index=False)

# ============================================================== 最优配比 (S08)
pmax = np.maximum(Mc.max(0), 1e-3)  # 信赖域: 不超过实验中出现过的最大份额
qmodel = Scheffe('quad', best_alpha['L_avg']).fit(Mc, comb.L_avg.values)
mlaw = MixLaw().fit(Mc, comb.L_avg.values)
m6full = LogShare(best_eps6['L_avg']).fit(Mc, comb.L_avg.values)
gbm_full = GBM().fit(Mc, comb.L_avg.values)
ens = lambda M: (m6full.predict(M) + gbm_full.predict(M)) / 2  # 集成: 检验集表现最好的平滑可解释模型 M6 + 非参数 GBM


def opt_smooth(fun, starts=30):
    best = None
    cons = [{'type': 'eq', 'fun': lambda p: p.sum() - 1}]
    for s0 in [s_ref] + [rng.dirichlet(np.ones(17)) * 0 + Mc[rng.integers(len(Mc))] for _ in range(starts)]:
        s0 = np.minimum(s0, pmax); s0 = s0 / s0.sum()
        r = optimize.minimize(lambda p: fun(p[None])[0], s0, method='SLSQP', bounds=[(0, u) for u in pmax], constraints=cons,
                              options={'maxiter': 500, 'ftol': 1e-10})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return best.x
p_quad = opt_smooth(qmodel.predict); p_law = opt_smooth(m6full.predict)
# RegMix 式: 在实验配方的凸组合邻域内大规模采样 → 集成模型预测 → 取最优 100 个平均
K = 100000; A = rng.dirichlet(np.full(len(Mc), 0.05), size=K // 10)
cand = np.vstack([A @ Mc] + [rng.dirichlet(np.maximum(s_ref * 30, 0.05), size=K // 10 * 9)])
cand = cand[(cand <= pmax + 1e-12).all(1)]
pc = ens(cand); top = np.argsort(pc)[:100]; p_ens = cand[top].mean(0); p_ens /= p_ens.sum()
opt = pd.DataFrame({'domain': DOM, 'reference_mean_mixture': s_ref, 'trust_region_pmax': pmax, 'p_opt_quadratic': p_quad,
                    'p_opt_log_share_M6': p_law, 'p_opt_ensemble_top100': p_ens})
opt.to_csv(P('S08', 'S08_optimal_mixture.csv'), index=False)
po = []
for nm in ['reference_mean_mixture', 'p_opt_quadratic', 'p_opt_log_share_M6', 'p_opt_ensemble_top100']:
    v = opt[nm].values[None]
    po.append(dict(mixture=nm, pred_quad=qmodel.predict(v)[0], pred_M6=m6full.predict(v)[0], pred_gbm=gbm_full.predict(v)[0], pred_ensemble=ens(v)[0]))
for sn in ['train_1M', 'test_1M']:
    po.append(dict(mixture=f'best_observed_{sn}', pred_ensemble=S[sn].L_avg.min()))
pd.DataFrame(po).to_csv(P('S08', 'S08_optimal_mixture_predicted_loss.csv'), index=False)
# 与各尺度真实最优配方的一致性
cons_rows = []
for sn in ['test_1M', 'test_60M', 'test_1B', 'est_10B', 'est_70B']:
    te = S[sn]; best10 = te.nsmallest(10, 'L_avg')[DOM].mean().values
    for nm in ['p_opt_quadratic', 'p_opt_log_share_M6', 'p_opt_ensemble_top100']:
        cons_rows.append(dict(scale=sn, optimum=nm, L1_to_top10_mean=np.abs(opt[nm].values - best10).sum(),
                              L1_ref_to_top10=np.abs(s_ref - best10).sum(), spearman_with_top10=stats.spearmanr(opt[nm].values, best10).statistic))
pd.DataFrame(cons_rows).to_csv(P('S08', 'S08_optimum_vs_observed_top10_by_scale.csv'), index=False)

# ============================================================== 外推稳健性 (S09)
e10, e70 = S['est_10B'], S['est_70B']
trsub = tr.set_index('index').loc[e10['index']].reset_index()
ex = []
for v in TARGETS:
    ex.append(dict(target=v, mean_1M_train_subset=trsub[v].mean(), mean_1B_test=S['test_1B'][v].mean(), mean_est10B=e10[v].mean(),
                   mean_est70B=e70[v].mean(), ratio_70B_over_10B_mean=(e70[v] / e10[v]).mean(), ratio_70B_over_10B_cv=(e70[v] / e10[v]).std() / (e70[v] / e10[v]).mean(),
                   spearman_10B_vs_70B=stats.spearmanr(e10[v], e70[v]).statistic, spearman_1M_vs_10B=stats.spearmanr(trsub[v], e10[v]).statistic,
                   spearman_1M_vs_70B=stats.spearmanr(trsub[v], e70[v]).statistic,
                   est10B_exceeds_1B_mean=bool(e10[v].mean() > S['test_1B'][v].mean())))
EX = pd.DataFrame(ex); EX.to_csv(P('S09', 'S09_extrapolation_consistency.csv'), index=False)
# 各尺度重估线性 Scheffé 域效应并比较
scales = {'1M(train)': S['train_1M'], '1M(train+test)': comb, '60M(test)': S['test_60M'], '1B(test)': S['test_1B'], '10B(est)': e10, '70B(est)': e70}
ce_sc = {}
for sn, df in scales.items():
    M = df[DOM].values; y = df.L_avg.values
    rd = Ridge(alpha=1e-3 if len(df) > 100 else 1e-2, fit_intercept=False).fit(M, y)
    ce_sc[sn] = cox_effect(rd.coef_, M.mean(0))
CE = pd.DataFrame(ce_sc, index=DOM); CE.to_csv(P('S09', 'S09_cox_effect_L_avg_by_scale.csv'))
ce6 = {}
for sn, df in scales.items():
    M = df[DOM].values; mdl = LogShare(best_eps6['L_avg'], alpha=1e-4 if len(df) > 100 else 1e-2).fit(M, df.L_avg.values)
    ce6[sn] = cox_fd(mdl, M.mean(0))
CE6 = pd.DataFrame(ce6, index=DOM); CE6.to_csv(P('S09', 'S09_M6_cox_effect_L_avg_by_scale.csv'))
cmp6 = []
for a_, b_ in itertools.combinations(CE6.columns, 2):
    cmp6.append(dict(scale_a=a_, scale_b=b_, spearman=stats.spearmanr(CE6[a_], CE6[b_]).statistic, sign_agreement=float((np.sign(CE6[a_]) == np.sign(CE6[b_])).mean())))
pd.DataFrame(cmp6).to_csv(P('S09', 'S09_M6_effect_rank_agreement_across_scales.csv'), index=False)
cmp = []
for a_, b_ in itertools.combinations(CE.columns, 2):
    cmp.append(dict(scale_a=a_, scale_b=b_, spearman=stats.spearmanr(CE[a_], CE[b_]).statistic, sign_agreement=float((np.sign(CE[a_]) == np.sign(CE[b_])).mean())))
pd.DataFrame(cmp).to_csv(P('S09', 'S09_effect_rank_agreement_across_scales.csv'), index=False)
# 1M 模型对外推表的排序能力
xr = []
for sn in ['est_10B', 'est_70B']:
    te = S[sn]
    for mn in MODELS:
        if (mn, 'L_avg') in preds:
            p = preds[(mn, 'L_avg')].predict(te[DOM].values)
            xr.append(dict(set=sn, model=mn, spearman=stats.spearmanr(te.L_avg, p).statistic, kendall=stats.kendalltau(te.L_avg, p).statistic))
pd.DataFrame(xr).to_csv(P('S09', 'S09_1M_models_rank_on_est_tables.csv'), index=False)
# 在各尺度数据上重求最优配比 (线性 ridge + 信赖域), 比较稳定性
po_sc = {}
for sn, df in scales.items():
    M = df[DOM].values; rd = Ridge(alpha=1e-3 if len(df) > 100 else 1e-2, fit_intercept=False).fit(M, df.L_avg.values)
    b = rd.coef_; order = np.argsort(b); p = np.zeros(17); rem = 1.0
    for i in order:  # 线性目标在盒约束单纯形上的最优解: 按系数从小到大贪心填满上界
        take = min(pmax[i], rem); p[i] = take; rem -= take
        if rem <= 1e-12:
            break
    po_sc[sn] = p
PO = pd.DataFrame(po_sc, index=DOM); PO.to_csv(P('S09', 'S09_linear_optimum_by_scale.csv'))
pd.DataFrame([(a_, b_, np.abs(PO[a_] - PO[b_]).sum()) for a_, b_ in itertools.combinations(PO.columns, 2)],
             columns=['scale_a', 'scale_b', 'L1_distance']).to_csv(P('S09', 'S09_linear_optimum_L1_distance.csv'), index=False)

# ============================================================== 质量 Q 的接入 (S09)
mp = pd.read_csv(MAPF)
INF = {'pubmed_central': ('arxiv', 0.6, '学术全文'), 'philpapers': ('arxiv', 0.6, '学术论文'), 'pubmed_abstracts': ('arxiv', 0.5, '学术摘要'),
       'nih_exporter': ('arxiv', 0.5, '科研项目摘要'), 'dm_mathematics': ('arxiv', 0.3, '数学符号/题目文本'),
       'hackernews': ('stackexchange', 0.4, '技术社区讨论'), 'ubuntu_irc': ('stackexchange', 0.4, '技术问答对话'),
       'enron_emails': ('commoncrawl', 0.3, '非正式通信文本'), 'freelaw': ('wikipedia', 0.3, '正式说明性长文本'),
       'uspto_backgrounds': ('wikipedia', 0.3, '正式技术说明文本'), 'europarl': ('wikipedia', 0.3, '正式议会记录')}
CONF = {'direct': 1.0, 'near_direct': 0.8}
dq = pd.read_csv(QF); dq = dq[dq.score == 'Q_resolved'].groupby('domain').token_weighted_mean.mean()
qbar = dq.mean() if len(dq) else np.nan
qm = []
for _, r in mp.iterrows():
    md = r.mixture_domain
    if r.mapping_type in CONF:
        qd, c, why = r.quality_domain, CONF[r.mapping_type], r.note
    else:
        qd, c, why = INF[md]
    q_raw = dq.get(qd, np.nan)
    qm.append(dict(mixture_domain=md, quality_domain=qd, mapping_type=r.mapping_type, confidence=c, rationale=why, Q_quality_domain=q_raw,
                   Q_shrunk=(c * q_raw + (1 - c) * qbar) if q_raw == q_raw else np.nan, Q_available=bool(q_raw == q_raw)))
QM = pd.DataFrame(qm); QM.to_csv(P('S09', 'S09_domain_Q_mapping_17.csv'), index=False)
# 可识别性论证的数值证据: Q_mix(p)=Σ p_i Q_i 是 p 的线性函数 → 在 Scheffé 线性模型中完全共线
avail = QM.Q_available.values
qi_ = {'n_domains_with_Q': int(avail.sum()), 'domains_with_Q': QM.mixture_domain[avail].tolist()}
if avail.sum() >= 2:
    qv = QM.Q_shrunk.fillna(0).values
    for sn in ['train_1M', 'test_1M']:
        S[sn]['Q_mix_partial'] = S[sn][DOM].values @ qv
        S[sn]['covered_share'] = S[sn][DOM].values @ avail.astype(float)
    X1 = np.column_stack([Mtr, S['train_1M'].Q_mix_partial.values])
    qi_['rank_of_[p,Q_mix]'] = int(np.linalg.matrix_rank(X1)); qi_['n_columns'] = X1.shape[1]
    # 结构化低维模型: L = a + b·Q_mix/覆盖份额 + c·H(p) + d·覆盖份额
    def feats(df):
        H = -(df[DOM].values * np.log(df[DOM].values + 1e-12)).sum(1)
        cs = df.covered_share.values; qn = np.where(cs > 0, df.Q_mix_partial.values / np.maximum(cs, 1e-9), 0)
        return np.column_stack([np.ones(len(df)), qn * cs, H, cs])
    lr = LinearRegression(fit_intercept=False).fit(feats(S['train_1M']), S['train_1M'].L_avg.values)
    qi_['structured_model_coef[a,Q·cov,H,cov]'] = lr.coef_.tolist()
    qi_['structured_model_test1M_R2'] = metrics(S['test_1M'].L_avg.values, lr.predict(feats(S['test_1M'])))['R2']
    # 嵌套消融: 检验 Q 在"多样性 H + 覆盖份额"之上是否带来增量 (训练集拟合, 1M 检验集评估; 另报 60M/1B 秩相关)
    F_tr, F_te = feats(S['train_1M']), feats(S['test_1M'])
    abl = []
    for nm, cols in {'a+H': [0, 2], 'a+H+cov': [0, 2, 3], 'a+Q·cov': [0, 1], 'a+Q·cov+cov': [0, 1, 3], 'a+Q·cov+H+cov (full)': [0, 1, 2, 3]}.items():
        m_ = LinearRegression(fit_intercept=False).fit(F_tr[:, cols], S['train_1M'].L_avg.values)
        r_ = dict(model=nm, n_params=len(cols), train_R2=metrics(S['train_1M'].L_avg.values, m_.predict(F_tr[:, cols]))['R2'],
                  test1M_R2=metrics(S['test_1M'].L_avg.values, m_.predict(F_te[:, cols]))['R2'], coef=np.round(m_.coef_, 4).tolist())
        for sn in ['test_60M', 'test_1B']:
            dfx = S[sn].copy(); dfx['Q_mix_partial'] = dfx[DOM].values @ qv; dfx['covered_share'] = dfx[DOM].values @ avail.astype(float)
            r_[sn + '_spearman'] = stats.spearmanr(dfx.L_avg, feats(dfx)[:, cols] @ m_.coef_).statistic
        abl.append(r_)
    pd.DataFrame(abl).to_csv(P('S09', 'S09_quality_structured_ablation.csv'), index=False)
    # 域效应与 Q 的关系
    ce_avg = EF6[EF6.target == 'L_avg'].set_index('domain').cox_effect_plus0_1
    jj = QM[QM.Q_available].set_index('mixture_domain')
    qi_['cox_effect_vs_Q'] = {k: dict(Q=float(jj.loc[k, 'Q_shrunk']), cox_effect=float(ce_avg[k])) for k in jj.index}
    pd.concat([S['train_1M'][['index', 'Q_mix_partial', 'covered_share']].assign(set='train_1M'),
               S['test_1M'][['index', 'Q_mix_partial', 'covered_share']].assign(set='test_1M')]).to_csv(P('S09', 'S09_mixture_level_Qmix.csv'), index=False)
json.dump(qi_, open(P('S09', 'S09_quality_integration_tests.json'), 'w'), ensure_ascii=False, indent=1, default=float)
json.dump({'best_ridge_alpha': best_alpha, 'best_ownlog_eps': best_eps, 'best_logshare_eps': best_eps6}, open(P('S07', 'S07_selected_hyperparameters.json'), 'w'), indent=1)
print(summ.round(4).T.to_string())

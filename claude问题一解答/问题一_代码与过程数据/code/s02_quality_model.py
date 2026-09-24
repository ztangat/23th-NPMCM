"""S02–S05 数据质量评价 / 冲突识别 / 冲突消解 / 可靠性检验
输入: S01_sample_level_raw_scalars.csv.gz (A1 若存在则自动纳入; 当前环境为 A2+A3)
所有统计量(分位数、均值、方差、相关)均采用"域均衡权重" w=1/n_domain, 避免样本量最大的 github 支配归一化与赋权。
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

IN = sys.argv[1]; OUT = sys.argv[2]
for s in ['S02', 'S03', 'S04', 'S05']:
    os.makedirs(os.path.join(OUT, s), exist_ok=True)
P = lambda s, f: os.path.join(OUT, s, f)
rng = np.random.default_rng(20260924)

d = pd.read_csv(IN)
d['domain'] = d['domain'].astype(str)
log = {'n_total': int(len(d)), 'by_set_domain': d.groupby(['set', 'domain']).size().astype(int).reset_index().values.tolist()}

# ---------------------------------------------------------------- 指标规格表
SPEC = [  # name, raw column, type, dimension, 说明
    ('fineweb_edu', 'fineweb_edu', 'pos', 'D1语义价值', 'FineWeb-Edu 教育价值回归分'),
    ('qurater', 'qurater', 'pos', 'D1语义价值', 'QuRating 4 维分等权均值'),
    ('modernbert_reasoning', 'modernbert_reasoning', 'pos', 'D1语义价值', '6类logits→期望分(0–5)'),
    ('modernbert_professionalism', 'modernbert_professionalism', 'pos', 'D1语义价值', '6类logits→期望分(0–5)'),
    ('fluency_en', 'fluency_en_p1', 'pos', 'D2语言规范', '2类logits→P(流畅), 类别方向由arxiv锚定'),
    ('modernbert_readability', 'modernbert_readability', 'pos', 'D2语言规范', '6类logits→期望分(0–5)'),
    ('modernbert_cleanliness', 'modernbert_cleanliness', 'pos', 'D2语言规范', '6类logits→期望分(0–5)'),
    ('ad_en', 'ad_prob', 'neg', 'D2语言规范', '2类logits→P(广告)=1−p1, 负向, 补转换 1−x'),
    ('dsir_books', 'dsir_books_pw', 'pos', 'D3参考域相似', 'DSIR 对数重要性(总和)/词数'),
    ('dsir_wiki', 'dsir_wiki_pw', 'pos', 'D3参考域相似', 'DSIR 对数重要性(总和)/词数'),
    ('dsir_math', 'dsir_math_pw', 'pos', 'D3参考域相似', 'DSIR 对数重要性(总和)/词数'),
    ('rps_doc_word_count', 'log_word_count', 'interval', 'D4统计启发式', 'log词数, 适度型, 理想区间[50,1e5]词(Gopher规则)'),
    ('rps_doc_num_sentences', 'log_num_sentences', 'interval', 'D4统计启发式', 'log句数, 下限型, 理想≥3句'),
    ('rps_doc_unigram_entropy', 'rps_doc_unigram_entropy', 'pos', 'D4统计启发式', '一元词熵, 词汇多样性'),
    ('rps_doc_frac_unique_words', 'uniq_heaps_resid', 'pos', 'D4统计启发式', '去长度效应: log唯一词占比对log词数回归残差(Heaps律)'),
    ('rps_doc_frac_no_alph_words', 'rps_doc_frac_no_alph_words', 'neg', 'D4统计启发式', '无字母词占比(%)'),
    ('rps_doc_frac_chars_top_2gram', 'top2_clip', 'neg', 'D4统计启发式', '高频2-gram字符占比(%), 截断至[0,100]'),
    ('rps_doc_frac_chars_top_3gram', 'top3_clip', 'neg', 'D4统计启发式', '高频3-gram字符占比(%), 截断至[0,100]'),
    ('rps_lines_uppercase_letter_fraction', 'rps_lines_uppercase_letter_fraction', 'neg', 'D4统计启发式', '大写字母占比(%)'),
    ('rps_lines_ending_with_terminal_punctution_mark', 'rps_lines_ending_with_terminal_punctution_mark', 'pos', 'D4统计启发式', '以终止标点结尾行占比(%)'),
    ('rps_lines_numerical_chars_fraction', 'rps_lines_numerical_chars_fraction', 'neg', 'D4统计启发式', '数字字符占比(%)'),
    ('rps_doc_mean_word_length', 'rps_doc_mean_word_length', 'interval', 'D4统计启发式', '平均词长, 适度型, 理想区间[3,10](Gopher规则)'),
]
IND = [s[0] for s in SPEC]; DIMS = sorted(set(s[3] for s in SPEC))
INTERVAL = {'rps_doc_word_count': (np.log(50), np.log(1e5)), 'rps_doc_num_sentences': (np.log(3), np.inf),
            'rps_doc_mean_word_length': (3.0, 10.0)}

# ---------------------------------------------------------------- 预处理
pre = {}
# 方向锚定检验: arxiv 为学术论文, 绝大多数应"非广告、流畅"
arx = d[d.domain == 'arxiv']
pre['anchor_ad_en_p1_mean_arxiv'] = float(arx.ad_en_p1.mean())
pre['anchor_fluency_en_p1_mean_arxiv'] = float(arx.fluency_en_p1.mean())
d['ad_prob'] = 1 - d['ad_en_p1']  # p1 在 arxiv 上均值≈0.99 → 第2类=非广告 → P(广告)=p0
for k in ['books', 'wiki', 'math']:
    d[f'dsir_{k}_pw'] = d[f'dsir_{k}'] / d['rps_doc_word_count'].clip(lower=1)
d['log_word_count'] = np.log(d['rps_doc_word_count'].clip(lower=1))
d['log_num_sentences'] = np.log(d['rps_doc_num_sentences'].clip(lower=1))
pre['top2_gt100'] = int((d.rps_doc_frac_chars_top_2gram > 100).sum()); pre['top3_gt100'] = int((d.rps_doc_frac_chars_top_3gram > 100).sum())
d['top2_clip'] = d.rps_doc_frac_chars_top_2gram.clip(0, 100); d['top3_clip'] = d.rps_doc_frac_chars_top_3gram.clip(0, 100)
# Heaps 律去长度: log(frac_unique) = a + b log(word_count) + e
m = (d.rps_doc_frac_unique_words > 0) & (d.rps_doc_word_count > 0)
X = d.loc[m, 'log_word_count']; Y = np.log(d.loc[m, 'rps_doc_frac_unique_words'])
b, a = np.polyfit(X, Y, 1); pre['heaps_fit'] = {'a': float(a), 'b': float(b), 'implied_heaps_beta': float(1 + b)}
d['uniq_heaps_resid'] = np.log(d.rps_doc_frac_unique_words.clip(lower=1e-6)) - (a + b * d.log_word_count)
# 缺失: 按域中位数插补
miss = {}
for s in SPEC:
    c = s[1]; n = int(d[c].isna().sum())
    if n:
        miss[c] = n; d[c] = d.groupby('domain')[c].transform(lambda v: v.fillna(v.median()))
pre['imputed_missing'] = miss
# 域均衡权重
nd = d.domain.value_counts(); d['w_bal'] = d.domain.map(1.0 / nd / len(nd))


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w); c = (c - 0.5 * w) / c[-1]
    return np.interp(q, c, x)


def wecdf(x, w):
    o = np.argsort(x, kind='mergesort'); xs, ws = x[o], w[o]; c = np.cumsum(ws) / ws.sum()
    # 并列值取同一分位(取右端), 保证 rank 与方向一致
    r = np.empty_like(c); u, idx, inv = np.unique(xs, return_index=True, return_inverse=True)
    last = np.r_[idx[1:] - 1, len(xs) - 1]; r = c[last][inv]
    out = np.empty_like(r); out[o] = r; return out


W = d.w_bal.values
norm_rows = []
for name, col, typ, dim, note in SPEC:
    x = d[col].values.astype(float)
    lo, hi = wquantile(x, W, 0.01), wquantile(x, W, 0.99)
    xc = np.clip(x, lo, hi)
    if typ == 'pos':
        z = (xc - lo) / (hi - lo)
    elif typ == 'neg':
        z = 1 - (xc - lo) / (hi - lo)
    else:
        a_, b_ = INTERVAL[name]
        z = np.ones_like(xc)
        if a_ > lo:
            z = np.where(xc < a_, (xc - lo) / (a_ - lo), z)
        if np.isfinite(b_) and b_ < hi:
            z = np.where(xc > b_, (hi - xc) / (hi - b_), z)
        z = np.clip(z, 0, 1)
    d['z_' + name] = z
    norm_rows.append(dict(indicator=name, raw_column=col, type=typ, dimension=dim, note=note, winsor_P01=lo, winsor_P99=hi,
                          n_clipped_low=int((x < lo).sum()), n_clipped_high=int((x > hi).sum()),
                          z_mean_balanced=float(np.average(z, weights=W))))
pd.DataFrame(norm_rows).to_csv(P('S02', 'S02_indicator_spec_and_normalization.csv'), index=False)
Z = d[['z_' + i for i in IND]].values

# ---------------------------------------------------------------- 赋权
def wcorr(M, w):
    mu = np.average(M, axis=0, weights=w); C = (M - mu); cov = (C * w[:, None]).T @ C / w.sum()
    sd = np.sqrt(np.diag(cov)); return cov / np.outer(sd, sd), sd

R, sd = wcorr(Z, W)
# 熵权法
Pm = (Z + 1e-12) * W[:, None]; Pm = Pm / Pm.sum(0)
# 熵权法(样本按域均衡加权): e_j = -Σ p ln p / ln n
wn = W / W.sum(); n_eff = 1 / np.sum(wn ** 2)
e = -(Pm * np.log(Pm)).sum(0) / np.log(len(W))
w_ewm = (1 - e) / (1 - e).sum()
# CRITIC
Cj = sd * (1 - R).sum(1); w_cri = Cj / Cj.sum()
# 组合: 最小相对信息熵 w = sqrt(w1 w2)/Σ
w_cmb = np.sqrt(w_ewm * w_cri); w_cmb /= w_cmb.sum()
# 结构等权(维度等权, 维内等权)
dim_of = np.array([s[3] for s in SPEC]); w_eq = np.array([1 / len(DIMS) / (dim_of == dm).sum() for dm in dim_of])
WT = pd.DataFrame({'indicator': IND, 'dimension': dim_of, 'type': [s[2] for s in SPEC], 'sd_balanced': sd,
                   'sum_1_minus_r': (1 - R).sum(1), 'entropy_e': e, 'w_entropy': w_ewm, 'w_critic': w_cri,
                   'w_combined': w_cmb, 'w_structural_equal': w_eq})
WT.to_csv(P('S02', 'S02_indicator_weights.csv'), index=False)
pd.DataFrame(R, index=IND, columns=IND).to_csv(P('S02', 'S02_indicator_corr_pearson_balanced.csv'))
# 维度划分的数据检验: 层次聚类 (距离 1-|rho|)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
rs = stats.spearmanr(Z[rng.choice(len(Z), min(60000, len(Z)), replace=False, p=wn)]).statistic
pd.DataFrame(rs, index=IND, columns=IND).to_csv(P('S02', 'S02_indicator_corr_spearman_balanced_sub60k.csv'))
Dm = 1 - np.abs(rs); np.fill_diagonal(Dm, 0); Lk = linkage(squareform(Dm, checks=False), 'average')
pd.DataFrame({'indicator': IND, 'dimension_by_theory': dim_of, 'cluster_k4': fcluster(Lk, 4, 'maxclust'),
              'cluster_k6': fcluster(Lk, 6, 'maxclust')}).to_csv(P('S02', 'S02_indicator_clustering_check.csv'), index=False)

# ---------------------------------------------------------------- 样本级 Q
d['Q_lin'] = Z @ w_cmb
d['Q_equal'] = Z @ w_eq
# TOPSIS (同权重)
V = Z * w_cmb; Dp = np.sqrt(((V - w_cmb) ** 2).sum(1)); Dn = np.sqrt((V ** 2).sum(1)); d['Q_topsis'] = Dn / (Dp + Dn)
# PCA 第一主成分 (域均衡标准化)
mu = np.average(Z, axis=0, weights=W); Zs = (Z - mu) / sd
ev, evec = np.linalg.eigh(R); v1 = evec[:, -1] * np.sign(evec[:, -1].sum())
d['Q_pca1'] = Zs @ v1
pd.DataFrame({'indicator': IND, 'pc1_loading': v1}).assign(explained=ev[-1] / ev.sum()).to_csv(P('S02', 'S02_pca_pc1_loadings.csv'), index=False)

# ---------------------------------------------------------------- 冲突 (S03)
Rk = np.column_stack([wecdf(Z[:, j], W) for j in range(len(IND))])  # 域均衡百分位秩, 已统一越高越好
hi = Rk >= 0.75; lo = Rk <= 0.25
Cmat = np.zeros((len(IND), len(IND)))
for j in range(len(IND)):
    Cmat[j] = ((hi[:, [j]] & lo) | (lo[:, [j]] & hi)).astype(float).T @ W / W.sum()
np.fill_diagonal(Cmat, np.nan)
pd.DataFrame(Cmat, index=IND, columns=IND).to_csv(P('S03', 'S03_pairwise_conflict_rate_balanced.csv'))
pairs = []
for i in range(len(IND)):
    for j in range(i + 1, len(IND)):
        row = dict(ind_i=IND[i], ind_j=IND[j], dim_i=dim_of[i], dim_j=dim_of[j], conflict_rate_balanced=Cmat[i, j],
                   lift_vs_independence=Cmat[i, j] / 0.125, spearman=rs[i, j])
        for dm in nd.index:
            mk = (d.domain == dm).values
            row['rate_' + dm] = float(((hi[mk, i] & lo[mk, j]) | (lo[mk, i] & hi[mk, j])).mean())
        pairs.append(row)
PR = pd.DataFrame(pairs).sort_values('conflict_rate_balanced', ascending=False)
PR.to_csv(P('S03', 'S03_pairwise_conflict_ranked.csv'), index=False)
# 维度分与样本冲突度
for dm in DIMS:
    cols = [j for j in range(len(IND)) if dim_of[j] == dm]
    ww = w_cmb[cols] / w_cmb[cols].sum()
    d['D_' + dm] = wecdf(Rk[:, cols] @ ww, W)
Dcols = ['D_' + dm for dm in DIMS]; DD = d[Dcols].values
d['kappa_range'] = DD.max(1) - DD.min(1); d['kappa_sd'] = DD.std(1)
d['conflict_strong'] = (DD.max(1) >= 0.75) & (DD.min(1) <= 0.25)
d['conflict_type'] = np.where(d.conflict_strong, [f'{DIMS[a]}高/{DIMS[b]}低' for a, b in zip(DD.argmax(1), DD.argmin(1))], '无强冲突')
# 题面示例: 教育价值高且广告概率高
fe = IND.index('fineweb_edu'); ad = IND.index('ad_en')
d['conflict_edu_high_ad_high'] = (Rk[:, fe] >= 0.75) & (Rk[:, ad] <= 0.25)  # ad 已反向: 低秩=广告概率高
# 指标层冲突计数: 同一样本中处于对立四分位的指标对数
d['n_conflict_pairs'] = sum(((hi[:, i] & lo[:, j]) | (lo[:, i] & hi[:, j])).astype(int)
                            for i in range(len(IND)) for j in range(i + 1, len(IND)))
g = d.groupby(['set', 'domain'])
rate = g.agg(n=('Q_lin', 'size'), strong_conflict_rate=('conflict_strong', 'mean'), kappa_range_mean=('kappa_range', 'mean'),
             kappa_range_p90=('kappa_range', lambda v: v.quantile(.9)), n_conflict_pairs_mean=('n_conflict_pairs', 'mean'),
             edu_high_ad_high_rate=('conflict_edu_high_ad_high', 'mean')).reset_index()
rate.to_csv(P('S03', 'S03_conflict_rate_by_domain.csv'), index=False)
ct = d[d.conflict_strong].groupby(['domain', 'conflict_type']).size().rename('n').reset_index()
ct['share_within_domain_conflicts'] = ct.n / ct.groupby('domain').n.transform('sum')
ct.sort_values(['domain', 'n'], ascending=[True, False]).to_csv(P('S03', 'S03_conflict_typology.csv'), index=False)
# 成因: 长度十分位 × 冲突率; logistic 回归
d['len_decile'] = d.groupby('domain').log_word_count.transform(lambda v: pd.qcut(v.rank(method='first'), 10, labels=False) + 1)
d.groupby(['domain', 'len_decile']).agg(n=('conflict_strong', 'size'), strong_conflict_rate=('conflict_strong', 'mean'),
                                         kappa_range_mean=('kappa_range', 'mean'), median_words=('rps_doc_word_count', 'median')
                                         ).reset_index().to_csv(P('S03', 'S03_conflict_by_length_decile.csv'), index=False)
import statsmodels.api as sm
feat = pd.DataFrame({'log_words_std': (d.log_word_count - d.log_word_count.mean()) / d.log_word_count.std(),
                     'nonalpha_std': (d.rps_doc_frac_no_alph_words - d.rps_doc_frac_no_alph_words.mean()) / d.rps_doc_frac_no_alph_words.std(),
                     'numeric_std': (d.rps_lines_numerical_chars_fraction - d.rps_lines_numerical_chars_fraction.mean()) / d.rps_lines_numerical_chars_fraction.std(),
                     'wordlen_std': (d.rps_doc_mean_word_length - d.rps_doc_mean_word_length.mean()) / d.rps_doc_mean_word_length.std()})
for dm in list(nd.index)[1:]:
    feat['dom_' + dm] = (d.domain == dm).astype(float)
feat = sm.add_constant(feat)
res = sm.Logit(d.conflict_strong.astype(float).values, feat.values).fit(disp=0, maxiter=200)
ci = res.conf_int()
pd.DataFrame({'term': feat.columns, 'coef': res.params, 'odds_ratio': np.exp(res.params), 'OR_low95': np.exp(ci[:, 0]),
              'OR_high95': np.exp(ci[:, 1]), 'p_value': res.pvalues}).to_csv(P('S03', 'S03_conflict_cause_logit.csv'), index=False)
json.dump({'baseline_domain': nd.index[0], 'pseudo_R2': float(res.prsquared), 'n': int(res.nobs)},
          open(P('S03', 'S03_conflict_cause_logit_meta.json'), 'w'), ensure_ascii=False, indent=1)
# 域内相关结构差异 (成因: 域漂移导致模型打分器语义失配)
cc = []
for dm in nd.index:
    mk = (d.domain == dm).values; sub = rng.choice(np.where(mk)[0], min(40000, mk.sum()), replace=False)
    rr = stats.spearmanr(Z[sub]).statistic
    pd.DataFrame(rr, index=IND, columns=IND).to_csv(P('S03', f'S03_spearman_within_{dm}.csv'))
    for i in range(len(IND)):
        for j in range(i + 1, len(IND)):
            cc.append(dict(domain=dm, ind_i=IND[i], ind_j=IND[j], rho=rr[i, j]))
cc = pd.DataFrame(cc).pivot_table(index=['ind_i', 'ind_j'], columns='domain', values='rho').reset_index()
if {'arxiv', 'github'} <= set(cc.columns):
    cc['rho_gap_arxiv_minus_github'] = cc['arxiv'] - cc['github']
    cc = cc.reindex(cc.rho_gap_arxiv_minus_github.abs().sort_values(ascending=False).index)
cc.to_csv(P('S03', 'S03_within_domain_corr_shift.csv'), index=False)

# ---------------------------------------------------------------- 冲突消解 (S04)
# (1) 域自适应可信度: 指标 j 在域 d 内与"留一共识"(其余指标加权百分位均值)的 Spearman 相关 ρ_jd; τ=(1+ρ)/2 ∈[0,1]
#     (ρ=1 完全一致→1; ρ=0 正交(可能测量独立的有效侧面, 保留半权)→0.5; ρ=-1 完全矛盾→0)
# (2) 维内按 w_cmb·τ 加权 → 维度分; (3) 维间按 ω_k=Σ维内 w_cmb 加权均值, 并减去 λ·维度离散度 (悲观消解)
LAM = 0.5
trust = []
Qres = np.zeros(len(d)); Dres = np.zeros((len(d), len(DIMS)))
for dm in nd.index:
    mk = np.where((d.domain == dm).values)[0]
    Rk_d = np.column_stack([wecdf(Z[mk, j], np.ones(len(mk))) for j in range(len(IND))])
    sub = rng.choice(len(mk), min(40000, len(mk)), replace=False)
    tau = np.zeros(len(IND))
    for j in range(len(IND)):
        oth = [k for k in range(len(IND)) if k != j]
        cons = Rk_d[sub][:, oth] @ (w_cmb[oth] / w_cmb[oth].sum())
        tau[j] = stats.spearmanr(Rk_d[sub, j], cons).statistic
    tp = (1 + tau) / 2
    for j in range(len(IND)):
        trust.append(dict(domain=dm, indicator=IND[j], dimension=dim_of[j], rho_with_loo_consensus=tau[j], trust=tp[j],
                          w_base=w_cmb[j]))
    for k, dmn in enumerate(DIMS):
        cols = [j for j in range(len(IND)) if dim_of[j] == dmn]
        ww = w_cmb[cols] * tp[cols]
        if ww.sum() <= 0:
            ww = w_cmb[cols]
        Dres[mk, k] = Z[np.ix_(mk, cols)] @ (ww / ww.sum())
TR = pd.DataFrame(trust)
TR['w_adapted'] = TR.w_base * TR.trust; TR['w_adapted'] /= TR.groupby('domain').w_adapted.transform('sum')
TR.to_csv(P('S04', 'S04_domain_adaptive_trust_weights.csv'), index=False)
omega = np.array([w_cmb[dim_of == dmn].sum() for dmn in DIMS])
Qmean = Dres @ omega; Qdisp = Dres.std(1)
d['Q_resolved'] = np.clip(Qmean - LAM * Qdisp, 0, 1)
d['Q_resolved_lam0'] = Qmean
for k, dmn in enumerate(DIMS):
    d['Dz_' + dmn] = Dres[:, k]
# 稳健性: 留一指标 (LOIO) 秩稳定性; λ 敏感性
sub = rng.choice(len(d), min(50000, len(d)), replace=False, p=wn)
loio = []
for j in range(len(IND)):
    keep = [k for k in range(len(IND)) if k != j]
    ql = Z[sub][:, keep] @ (w_cmb[keep] / w_cmb[keep].sum())
    # resolved 版本的留一: 重新计算该维 (用全局 trust 均值近似)
    Dj = Dres[sub].copy(); k = DIMS.index(dim_of[j]); cols = [c for c in range(len(IND)) if dim_of[c] == dim_of[j] and c != j]
    if cols:
        tbar = TR[TR.indicator.isin([IND[c] for c in cols])].groupby('indicator').trust.mean().reindex([IND[c] for c in cols]).values
        ww = w_cmb[cols] * tbar; Dj[:, k] = Z[sub][:, cols] @ (ww / ww.sum())
    qr = np.clip(Dj @ omega - LAM * Dj.std(1), 0, 1)
    loio.append(dict(dropped=IND[j], spearman_Qlin=stats.spearmanr(d.Q_lin.values[sub], ql).statistic,
                     spearman_Qresolved=stats.spearmanr(d.Q_resolved.values[sub], qr).statistic))
LO = pd.DataFrame(loio); LO.to_csv(P('S04', 'S04_leave_one_indicator_out_stability.csv'), index=False)
lam_rows = []
for lam in [0, 0.25, 0.5, 0.75, 1.0]:
    q = np.clip(Qmean - lam * Qdisp, 0, 1)
    row = dict(lambda_=lam, spearman_vs_lam05=stats.spearmanr(q[sub], d.Q_resolved.values[sub]).statistic)
    for dm in nd.index:
        mk = (d.domain == dm).values; row['Q_mean_' + dm] = float(q[mk].mean())
    lam_rows.append(row)
pd.DataFrame(lam_rows).to_csv(P('S04', 'S04_lambda_sensitivity.csv'), index=False)

# ---------------------------------------------------------------- 聚合到域 (S02/S04)
def agg(gdf, col):
    x = gdf[col].values; wc = gdf.rps_doc_word_count.clip(lower=1).values
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(300)]
    return pd.Series({'n_docs': len(x), 'mean_doc': x.mean(), 'ci95_low': np.percentile(bs, 2.5), 'ci95_high': np.percentile(bs, 97.5),
                      'median_doc': np.median(x), 'trimmed_mean_10': stats.trim_mean(x, 0.1),
                      'token_weighted_mean': np.average(x, weights=wc), 'P10': np.percentile(x, 10), 'P90': np.percentile(x, 90),
                      'share_Q_ge_0.6': (x >= 0.6).mean()})
rows = []
for col in ['Q_lin', 'Q_equal', 'Q_topsis', 'Q_resolved', 'Q_resolved_lam0']:
    t = d.groupby(['set', 'domain']).apply(agg, col=col, include_groups=False).reset_index(); t.insert(0, 'score', col); rows.append(t)
DQ = pd.concat(rows); DQ.to_csv(P('S04', 'S04_domain_level_Q_all_methods.csv'), index=False)
# 维度分与原始指标的域均值
d.groupby(['set', 'domain'])[['z_' + i for i in IND] + ['D_' + x for x in DIMS] + ['Dz_' + x for x in DIMS]].mean().T.to_csv(
    P('S02', 'S02_domain_mean_normalized_indicators.csv'))
d.groupby(['set', 'domain'])[[s[1] for s in SPEC]].agg(['mean', 'median']).T.to_csv(P('S02', 'S02_domain_raw_indicator_mean_median.csv'))
# 方法间一致性
subd = d.iloc[sub]
cons = subd[['Q_lin', 'Q_equal', 'Q_topsis', 'Q_pca1', 'Q_resolved']].corr(method='spearman')
cons.to_csv(P('S04', 'S04_method_rank_consistency_spearman.csv'))

# ---------------------------------------------------------------- 可靠性 (S05)
rel = {}
# 分半信度: 各维内随机对半分指标 200 次
sh = []
for _ in range(200):
    A, B = [], []
    for dmn in DIMS:
        cols = np.where(dim_of == dmn)[0].copy(); rng.shuffle(cols); A += list(cols[::2]); B += list(cols[1::2])
    qa = Z[sub][:, A] @ (w_cmb[A] / w_cmb[A].sum()); qb = Z[sub][:, B] @ (w_cmb[B] / w_cmb[B].sum())
    r = stats.spearmanr(qa, qb).statistic; sh.append(2 * r / (1 + r))
rel['split_half_spearman_brown_mean'] = float(np.mean(sh)); rel['split_half_P5_P95'] = [float(np.percentile(sh, 5)), float(np.percentile(sh, 95))]
# Cronbach alpha (标准化, 域均衡)
k = len(IND); rbar = (R.sum() - k) / (k * (k - 1)); rel['cronbach_alpha_std'] = float(k * rbar / (1 + (k - 1) * rbar))
# 域内信度
for dm in nd.index:
    mk = np.where((d.domain == dm).values)[0]; s2 = rng.choice(mk, min(30000, len(mk)), replace=False)
    Rd = np.corrcoef(Z[s2].T); Rd = np.nan_to_num(Rd); rb = (Rd.sum() - k) / (k * (k - 1))
    rel[f'cronbach_alpha_std_{dm}'] = float(k * rb / (1 + (k - 1) * rb))
json.dump(rel, open(P('S05', 'S05_reliability.json'), 'w'), ensure_ascii=False, indent=1)
# 若含 A1 content: 原文校验
if 'n_chars' in d.columns and d['n_chars'].notna().any():
    a1 = d[d.set == 'A1'].copy()
    a1[['Q_lin', 'Q_resolved', 'code_sym_frac', 'url_count', 'n_chars']].corr('spearman').to_csv(P('S05', 'S05_A1_Q_vs_content_features.csv'))
    ex = []
    for dm, gg in a1.groupby('domain'):
        for tag, part in [('top', gg.nlargest(15, 'Q_resolved')), ('bottom', gg.nsmallest(15, 'Q_resolved')),
                          ('strong_conflict', gg[gg.conflict_strong].sample(min(15, gg.conflict_strong.sum()), random_state=1) if gg.conflict_strong.sum() else gg.head(0))]:
            ex.append(part.assign(group=tag)[['domain', 'group', 'id', 'Q_lin', 'Q_resolved', 'conflict_type', 'excerpt']])
    pd.concat(ex).to_csv(P('S05', 'S05_A1_text_excerpts_for_manual_check.csv'), index=False)


# 决策层面的消解效果: 按 Q 选取前 30% 数据时, 强冲突样本的入选率 / 两种评分的入选重合度
sel = []
for dm in nd.index:
    mk = (d.domain == dm).values; gg = d[mk]
    for col in ['Q_lin', 'Q_resolved']:
        thr = gg[col].quantile(0.7); top = gg[col] >= thr
        sel.append(dict(domain=dm, score=col, top30_threshold=thr, conflict_rate_in_top30=gg.conflict_strong[top].mean(),
                        conflict_rate_overall=gg.conflict_strong.mean(), kappa_range_mean_top30=gg.kappa_range[top].mean()))
    t1 = gg.Q_lin >= gg.Q_lin.quantile(.7); t2 = gg.Q_resolved >= gg.Q_resolved.quantile(.7)
    sel.append(dict(domain=dm, score='overlap_top30_Jaccard', top30_threshold=np.nan, conflict_rate_in_top30=(t1 & t2).sum() / (t1 | t2).sum()))
pd.DataFrame(sel).to_csv(P('S04', 'S04_selection_effect_top30.csv'), index=False)
# 抽样集(A1)与扩展集(A2/A3)同域对照 (A1 存在时)
if 'A1' in set(d.set):
    comp = []
    for dm in sorted(set(d.domain[d.set == 'A1']) & set(d.domain[d.set != 'A1'])):
        a = d[(d.set == 'A1') & (d.domain == dm)]; b = d[(d.set != 'A1') & (d.domain == dm)]
        for col in ['Q_lin', 'Q_resolved']:
            comp.append(dict(domain=dm, score=col, n_A1=len(a), n_ext=len(b), mean_A1=a[col].mean(), mean_ext=b[col].mean(),
                             diff=a[col].mean() - b[col].mean(), KS=stats.ks_2samp(a[col], b[col]).statistic,
                             KS_p=stats.ks_2samp(a[col], b[col]).pvalue, conflict_A1=a.conflict_strong.mean(), conflict_ext=b.conflict_strong.mean()))
    pd.DataFrame(comp).to_csv(P('S05', 'S05_A1_vs_extended_same_domain.csv'), index=False)

# ---------------------------------------------------------------- 保存样本级结果
keep = ['set', 'domain', 'id', 'rps_doc_word_count'] + ['z_' + i for i in IND] + ['D_' + x for x in DIMS] + \
       ['Dz_' + x for x in DIMS] + ['Q_lin', 'Q_equal', 'Q_topsis', 'Q_pca1', 'Q_resolved', 'Q_resolved_lam0', 'kappa_range',
                                     'kappa_sd', 'conflict_strong', 'conflict_type', 'conflict_edu_high_ad_high', 'n_conflict_pairs']
d[keep].to_csv(P('S04', 'S04_sample_level_scores.csv.gz'), index=False, compression='gzip', float_format='%.5g')
pre['n_eff_balanced'] = float(n_eff)
json.dump({'log': log, 'preprocess': pre}, open(P('S02', 'S02_preprocess_log.json'), 'w'), ensure_ascii=False, indent=1)
print(WT.round(4).to_string()); print(DQ[DQ.score.isin(['Q_lin', 'Q_resolved'])].round(4).to_string())
print(rate.round(4).to_string()); print(LO.round(4).to_string()); print(rel); print(cons.round(3))

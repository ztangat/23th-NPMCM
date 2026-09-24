# -*- coding: utf-8 -*-
"""
问题一 · 步骤3：方向统一 + 归一化 + 综合质量评分 Q + 冲突分析 + 原始内容检验
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'C:\Users\dkyyt\Desktop\F题'
OUT = os.path.join(BASE, '_out', '01_q1')
LOG = open(os.path.join(OUT, '01d_scoring_report.md'), 'w', encoding='utf-8')
def w(s=''):
    LOG.write(str(s) + '\n')

full = pd.read_parquet(os.path.join(OUT, 'quality_matrix_raw.parquet'))

GROUP_EDU = ['fineweb_edu', 'modernbert_reasoning', 'modernbert_professionalism',
             'dsir_math', 'dsir_wiki', 'qurater']
ALL22 = ['fineweb_edu','fluency_en','modernbert_cleanliness','modernbert_readability',
         'modernbert_reasoning','modernbert_professionalism','qurater','ad_en',
         'dsir_books','dsir_math','dsir_wiki','rps_doc_word_count','rps_doc_num_sentences',
         'rps_doc_unigram_entropy','rps_doc_frac_unique_words','rps_doc_frac_no_alph_words',
         'rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram',
         'rps_lines_uppercase_letter_fraction','rps_lines_ending_with_terminal_punctution_mark',
         'rps_lines_numerical_chars_fraction','rps_doc_mean_word_length']

w('# 问题一 · 数据质量评价：预处理、方向统一、综合评分\n')

# ================= 1. 长度混杂诊断与校正 =================
w('\n## 1. 预处理：长文档混杂效应诊断与校正\n')
w('DSIR 系列指标（ `dsir_books/dsir_math/dsir_wiki` ）为对数似然比，其数值随文档长度累加，'
  '若不校正将退化为"长度指标"。先做诊断：\n')
lw = np.log(full['rps_doc_word_count'].clip(lower=1))
w('| 指标 | 与 ln(词数) 的 Pearson r（校正前） | 校正后 |')
w('|---|---|---|')
resid_dsir = {}
for f in ['dsir_books', 'dsir_math', 'dsir_wiki']:
    y = np.log(-full[f].clip(upper=-1e-6))
    r0 = np.corrcoef(y, lw)[0, 1]
    # 用 ln(词数) 做一元回归后取残差
    X = np.column_stack([np.ones(len(lw)), lw])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    res = y.to_numpy() - X @ beta
    resid_dsir[f] = res
    r1 = np.corrcoef(res, lw)[0, 1]
    w('| `%s` | %.4f | %.4f |' % (f, r0, r1))
w('\n校正方式： `ln(-dsir) ~ 1 + ln(词数)` 回归取残差（即"每 token 对数似然比"的长度正交分量）。'
  '校正后与长度的相关性降至 ~0，指标恢复为"与目标域的相似度"语义。\n')

# ================= 2. 方向定义 =================
# 经验方向判定：以原始文本噪声代理为外部准则
A1 = full[full['_source_file'] == 'A1_sample'].copy()
def zscore(s):
    s = pd.to_numeric(s, errors='coerce')
    return (s - s.mean()) / s.std(ddof=0)
noise = (zscore(A1['dup_line_frac']) + zscore(A1['rps_doc_frac_no_alph_words'])
         + zscore(A1['url_per1k']) + zscore(A1['adkw_per1k'])
         - zscore(A1['frac_alpha'])) / 5.0

w('\n## 2. 指标方向统一（经验准则 + 语义判据）\n')
w('以 A1 原始文本构造"噪声代理" `Noise = [z(重复行占比)+z(非字母字符占比)+z(URL密度)+z(广告词密度)-z(字母占比)]/5`，'
  '作为"文本越脏越大"的外部准则。指标与 Noise 的 Spearman ρ 为正 ⇒ 该指标为**负向**（越大越差），须做补转换。\n')
w('| 指标 | 语义 | ρ(存储值, Noise) | 判定方向 | 经验判据 | 处理方式 |')
w('|---|---|---|---|---|---|')
rho_noise = {}
for f in ALL22:
    m = pd.DataFrame({'x': A1[f], 'n': noise}).dropna()
    r = stats.spearmanr(m['x'], m['n']).statistic
    rho_noise[f] = r

# 人工语义判据（与经验准则互相印证）；+1 正向 -1 负向
SEMANTIC = {
    'fineweb_edu': (+1, '教育价值（FineWeb-Edu 0–5 分）', '越高越好'),
    'fluency_en': (+1, '英文流畅度 P(流畅)', '越高越好'),
    'modernbert_cleanliness': (+1, '洁净度（0–5）', '越高越好'),
    'modernbert_readability': (+1, '可读性（0–5）', '越高越好'),
    'modernbert_reasoning': (+1, '推理含量（0–5）', '越高越好'),
    'modernbert_professionalism': (+1, '专业性（0–5）', '越高越好'),
    'qurater': (+1, 'QuRater 综合质量（0–3）', '越高越好'),
    'ad_en': (-1, '广告含量 A=P(广告)（由二分类 logits softmax 取下标 0）',
              '**负向**：A 越大越差，须做补转换 1−x'),
    'dsir_books': (+1, '与书籍域目标分布相似度（长度校正）', '越高越好'),
    'dsir_math': (+1, '与数学域目标分布相似度（长度校正）', '越高越好'),
    'dsir_wiki': (+1, '与维基域目标分布相似度（长度校正）', '越高越好'),
    'rps_doc_word_count': (+1, '文档长度/信息量（log 变换）', '越高越好（过短文档多为模板噪声）'),
    'rps_doc_num_sentences': (+1, '句数/信息量（log 变换）', '越高越好'),
    'rps_doc_unigram_entropy': (+1, '一元词熵＝词汇信息量', '越高越好'),
    'rps_doc_frac_unique_words': (+1, '不重复词占比＝词汇多样性', '越高越好'),
    'rps_doc_frac_no_alph_words': (-1, '非字母字符占比＝噪声/代码符号', '越低越好 → 补转换'),
    'rps_doc_frac_chars_top_2gram': (-1, '高频 2-gram 字符占比＝重复冗余', '越低越好 → 补转换'),
    'rps_doc_frac_chars_top_3gram': (-1, '高频 3-gram 字符占比＝重复冗余', '越低越好 → 补转换'),
    'rps_lines_uppercase_letter_fraction': (-1, '大写字母行占比＝喊叫/标题噪声', '越低越好 → 补转换'),
    'rps_lines_ending_with_terminal_punctution_mark': (+1, '以句末标点结尾的行占比＝句子完整度', '越高越好'),
    'rps_lines_numerical_chars_fraction': (-1, '数字字符占比＝表格/数据堆砌（非自然语言）', '越低越好 → 补转换'),
    'rps_doc_mean_word_length': (+1, '平均词长＝词汇复杂度', '越高越好（做敏感性检验）'),
}
for f in ALL22:
    d, sem, how = SEMANTIC[f]
    # 经验判据：ρ(指标存储值, Noise)>0 ⇒ 该存储值越大越"脏" ⇒ 负向
    emp = '负向' if rho_noise[f] > 0.2 else ('正向' if rho_noise[f] < -0.2 else '弱/不定')
    w('| `%s` | %s | %.4f | %s | %s | %s |' % (f, sem, rho_noise[f],
                                               '正向' if d > 0 else '**负向**', emp, how))

w('\n**关于 `ad_en` 的关键判定**：原始字段为 2 维 logits `[l0, l1]`。'
  '取 softmax 得 `[p0, p1]`。经验证据：\n'
  '- `ad_en`(存储值=p1) 与广告词密度 ρ=%.4f（负）、与 URL 密度 ρ=%.4f（负）；\n'
  '- 域级：c4（网页抓取）p1 均值最低 0.752、P(p1<0.5)=22.9%%，arxiv 最高 0.989、仅 0.08%%。\n'
  '⇒ 下标 0 = "广告"类，下标 1 = "非广告"类。故**广告含量 A = p0 = 1 − 存储值**，'
  'A 为负向指标，按题设做补转换后得正向指标 `ad_clean = 1 − A`。\n' % (-0.1629, -0.0871))

# ================= 3. 变换 + Winsorize + Min-Max + 方向统一 =================
w('\n## 3. 归一化（Winsorize + Min-Max）与方向统一\n')
w('步骤：① 单调变换（对数/长度校正）→ ② 全体 272,505 条上按 [p0.5, p99.5] 缩尾 → '
  '③ Min-Max 归一到 [0,1] → ④ 负向指标做补转换 x→1−x。\n')

X = pd.DataFrame(index=full.index)
for f in ALL22:
    if f in resid_dsir:
        X[f] = resid_dsir[f]
    elif f in ('rps_doc_word_count', 'rps_doc_num_sentences'):
        X[f] = np.log1p(full[f].clip(lower=0))
    else:
        X[f] = full[f].astype(float)

# 广告含量（负向原始量）单独保留
ad_content = 1.0 - full['ad_en'].astype(float)   # p0 = P(广告)
X['ad_en'] = ad_content                          # 覆盖为"广告含量"，方向为负

X = X.replace([np.inf, -np.inf], np.nan)
STATS = {}
Xn = pd.DataFrame(index=full.index)
for f in ALL22:
    s = X[f]
    lo, hi = np.nanpercentile(s, 0.5), np.nanpercentile(s, 99.5)
    med = np.nanmedian(s)
    sw = s.clip(lo, hi).fillna(med)
    mn, mx = sw.min(), sw.max()
    xn = (sw - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=full.index)
    d = SEMANTIC[f][0]
    Xn[f] = xn if d > 0 else (1.0 - xn)
    STATS[f] = dict(p05=lo, p995=hi, min=mn, max=mx, direction=d, median_impute=med)

w('| 指标 | p0.5 | p99.5 | 方向 | 处理后 min | 处理后 max | 处理后均值 |')
w('|---|---|---|---|---|---|---|')
for f in ALL22:
    st = STATS[f]
    w('| `%s` | %.4g | %.4g | %s | %.4f | %.4f | %.4f |' % (
        f, st['p05'], st['p995'], '正' if st['direction'] > 0 else '负(1−x)',
        Xn[f].min(), Xn[f].max(), Xn[f].mean()))

Xn.to_parquet(os.path.join(OUT, 'indicators_unified.parquet'))
pd.Series(ad_content).to_frame('ad_content_raw').to_parquet(os.path.join(OUT, 'ad_content_raw.parquet'))

# ================= 4. CRITIC 客观赋权 =================
w('\n## 4. CRITIC 客观赋权\n')
w('CRITIC 法：第 j 个指标的信息量 `C_j = σ_j · Σ_k (1 − |r_jk|)`，其中 σ_j 为标准差'
  '（对比强度），r_jk 为指标间 Pearson 相关（冲突性）。权重 `w_j = C_j / Σ C_j`。\n')
Xm = Xn[ALL22]
sd = Xm.std(ddof=0).to_numpy()
R = np.corrcoef(Xm.to_numpy().T)
conflict = (1.0 - np.abs(R)).sum(axis=1)
C = sd * conflict
wgt = C / C.sum()
wgt_eq = np.full(len(ALL22), 1.0 / len(ALL22))

w('| 指标 | σ_j（对比强度） | Σ(1−|r_jk|)（冲突性） | C_j | **CRITIC 权重 w_j** | 等权重 |')
w('|---|---|---|---|---|---|')
for i, f in enumerate(ALL22):
    w('| `%s` | %.4f | %.4f | %.4f | **%.4f** | %.4f |' % (f, sd[i], conflict[i], C[i], wgt[i], wgt_eq[i]))

Q_crit = (Xm.to_numpy() * wgt).sum(axis=1)
Q_eq = (Xm.to_numpy() * wgt_eq).sum(axis=1)

# 分层版：4 个语义维度 → PCA 验证 + 组内均值
GROUPS = {
    'G1_教育专业价值': ['fineweb_edu', 'modernbert_reasoning', 'modernbert_professionalism',
                    'dsir_math', 'dsir_wiki', 'qurater'],
    'G2_洁净流畅': ['fluency_en', 'modernbert_cleanliness', 'modernbert_readability',
                 'ad_en', 'rps_lines_ending_with_terminal_punctution_mark',
                 'rps_doc_frac_no_alph_words'],
    'G3_多样信息量': ['rps_doc_unigram_entropy', 'rps_doc_frac_unique_words',
                 'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram'],
    'G4_体量与格式': ['rps_doc_word_count', 'rps_doc_num_sentences', 'dsir_books',
                 'rps_lines_uppercase_letter_fraction', 'rps_lines_numerical_chars_fraction',
                 'rps_doc_mean_word_length'],
}
S = pd.DataFrame(index=full.index)
for g, cols in GROUPS.items():
    sub = Xn[cols]
    S[g] = sub.mean(axis=1)
sdg = S.std(ddof=0).to_numpy()
Rg = np.corrcoef(S.to_numpy().T)
cg = (1.0 - np.abs(Rg)).sum(axis=1)
Cg = sdg * cg
wg = Cg / Cg.sum()
w('\n### 4.1 四维度分层结构（组内等权 → CRITIC 组间赋权）\n')
w('| 维度 | 含指标 | σ | Σ(1−|r|) | 组权重 |')
w('|---|---|---|---|---|')
for i, g in enumerate(S.columns):
    w('| %s | %s | %.4f | %.4f | **%.4f** |' % (g, ', '.join('`%s`' % c for c in GROUPS[g]),
                                                sdg[i], cg[i], wg[i]))
Q_hier = (S.to_numpy() * wg).sum(axis=1)

# PCA 检验维度划分
from numpy.linalg import eigh
Xc = (Xm - Xm.mean())
cov = np.cov(Xc.to_numpy().T)
evals, evecs = eigh(cov)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
w('\n### 4.2 PCA 特征值（检验"质量"是否为多维结构）\n')
w('| 主成分 | 特征值 | 方差贡献率 | 累计贡献率 |')
w('|---|---|---|---|')
tot = evals.sum()
cum = 0
for i in range(8):
    cum += evals[i] / tot
    w('| PC%d | %.4f | %.4f | %.4f |' % (i + 1, evals[i], evals[i] / tot, cum))
w('\n前 4 个主成分累计解释 %.2f%% 的方差，说明 22 维质量信号确实呈多维结构，'
  '采用"分维度合成"优于直接 22 维加权平均。\n' % (100 * (evals[:4] / tot).sum()))

# ================= 5. 三种 Q 的一致性 =================
w('\n## 5. 三种合成方案的秩相关一致性\n')
w('| 方案对 | Spearman ρ | Pearson r |')
w('|---|---|---|')
for a, b, na, nb in [('Q_CRITIC', 'Q_HIER', Q_crit, Q_hier),
                     ('Q_CRITIC', 'Q_EQ', Q_crit, Q_eq),
                     ('Q_HIER', 'Q_EQ', Q_hier, Q_eq)]:
    w('| %s vs %s | %.4f | %.4f |' % (a, b, stats.spearmanr(a_, b_).statistic if False else
                                      stats.spearmanr(na, nb).statistic, np.corrcoef(na, nb)[0, 1]))
# 外部效度：与 fineweb_edu 的秩相关
w('\n外部效度（以广泛使用的 FineWeb-Edu 教育价值分作为外部准则）:\n')
w('- ρ(Q_CRITIC, fineweb_edu) = %.4f' % stats.spearmanr(Q_crit, full['fineweb_edu']).statistic)
w('- ρ(Q_HIER, fineweb_edu)   = %.4f' % stats.spearmanr(Q_hier, full['fineweb_edu']).statistic)
w('- ρ(Q_EQ, fineweb_edu)     = %.4f' % stats.spearmanr(Q_eq, full['fineweb_edu']).statistic)

res = full.copy()
res['Q_critic'] = Q_crit
res['Q_hier'] = Q_hier
res['Q_eq'] = Q_eq
res['Q'] = Q_hier     # 主方案：分层 CRITIC
for g in GROUPS:
    res['G_' + g] = S[g]
res['ad_content'] = ad_content
res.to_parquet(os.path.join(OUT, 'quality_scored.parquet'), index=False)
w('\n- 已保存 `_out/01_q1/quality_scored.parquet`（含 Q_critic / Q_hier / Q_eq 与四个维度分）\n')
LOG.close()
print('ok')

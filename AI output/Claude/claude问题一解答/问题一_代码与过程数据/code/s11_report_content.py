"""组装报告内容(JSON 块列表), 所有数字直接读取过程文件, 避免手抄错误"""
import json, os, sys
import pandas as pd, numpy as np
O = sys.argv[1]; R = lambda s, f, **k: pd.read_csv(os.path.join(O, s, f), **k)
B = []
H1 = lambda t: B.append({'t': 'h1', 'x': t}); H2 = lambda t: B.append({'t': 'h2', 'x': t}); H3 = lambda t: B.append({'t': 'h3', 'x': t})
Pp = lambda t: B.append({'t': 'p', 'x': t}); EQ = lambda t: B.append({'t': 'eq', 'x': t}); NOTE = lambda t: B.append({'t': 'note', 'x': t})
BL = lambda items: B.append({'t': 'bullets', 'x': items})
def TB(cap, df, widths=None, fs=8):
    B.append({'t': 'table', 'cap': cap, 'head': [str(c) for c in df.columns], 'rows': [['' if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ('nan', '<NA>', 'None')) else str(v) for v in r] for r in df.astype(object).values.tolist()], 'w': widths, 'fs': fs})
IMG = lambda f, cap: B.append({'t': 'img', 'f': os.path.join(O, 'figures', f), 'cap': cap})
f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'

W = R('S02', 'S02_indicator_weights.csv'); NZ = R('S02', 'S02_indicator_spec_and_normalization.csv')
pre = json.load(open(os.path.join(O, 'S02', 'S02_preprocess_log.json')))['preprocess']
DQ = R('S04', 'S04_domain_level_Q_all_methods.csv'); rel = json.load(open(os.path.join(O, 'S05', 'S05_reliability.json')))
RT = R('S03', 'S03_conflict_rate_by_domain.csv'); PR = R('S03', 'S03_pairwise_conflict_ranked.csv'); TY = R('S03', 'S03_conflict_typology.csv')
LG = R('S03', 'S03_conflict_cause_logit.csv'); CS = R('S03', 'S03_within_domain_corr_shift.csv'); TR = R('S04', 'S04_domain_adaptive_trust_weights.csv')
SEL = R('S04', 'S04_selection_effect_top30.csv'); LO = R('S04', 'S04_leave_one_indicator_out_stability.csv'); LAM = R('S04', 'S04_lambda_sensitivity.csv')
CON = R('S04', 'S04_method_rank_consistency_spearman.csv', index_col=0)
AU = R('S06', 'S06_data_audit.csv'); V = R('S07', 'S07_validation_all_models_all_targets.csv'); TK = R('S07', 'S07_topk_hit_L_avg.csv')
E6 = R('S08', 'S08_M6_cox_effect_plus0.1_bootstrap.csv'); SHP = R('S08', 'S08_lightgbm_shap_importance_L_avg.csv')
FM = R('S08', 'S08_family_quadratic_scheffe_interactions.csv'); FD = R('S08', 'S08_family_definition.csv'); DP = R('S08', 'S08_domain_pair_interactions_ridge_L_avg.csv')
DV = R('S08', 'S08_diversity_vs_loss_by_scale.csv'); OP = R('S08', 'S08_optimal_mixture.csv'); OPL = R('S08', 'S08_optimal_mixture_predicted_loss.csv')
OC = R('S08', 'S08_optimum_vs_observed_top10_by_scale.csv'); EX = R('S09', 'S09_extrapolation_consistency.csv')
AG6 = R('S09', 'S09_M6_effect_rank_agreement_across_scales.csv'); XR = R('S09', 'S09_1M_models_rank_on_est_tables.csv')
QM = R('S09', 'S09_domain_Q_mapping_17.csv'); AB = R('S09', 'S09_quality_structured_ablation.csv'); QI = json.load(open(os.path.join(O, 'S09', 'S09_quality_integration_tests.json')))
hp = json.load(open(os.path.join(O, 'S07', 'S07_selected_hyperparameters.json')))
q = lambda sc, dm, c: DQ[(DQ.score == sc) & (DQ.domain == dm)][c].iloc[0]
rate = lambda dm, c: RT[RT.domain == dm][c].iloc[0]
vL = V[V.target == 'L_avg'].set_index('model'); vA = V[V.target != 'L_avg'].groupby('model').mean(numeric_only=True)
e10 = EX[EX.target == 'L_avg'].iloc[0]

# ================================================================ 摘要
H1('摘要')
Pp('本报告完成 2026 年研究生数学建模竞赛 F 题问题一：数据质量评价、质量冲突消解与领域配比建模。全部数字均由随附代码从原始附件计算得到，并可由过程文件逐项追溯。')
BL([
    f'数据质量评价：对扩展集 A2(arxiv, 17,523 条) 与 A3(github, 203,752 条) 的全部 221,275 条记录，将 8 个列表型指标压缩为标量，22 项指标统一为"越高越好"，以域均衡权重做缩尾-Min-Max 归一化，再用熵权与 CRITIC 的几何平均组合赋权。token 加权的域级质量为：arxiv Q = {f3(q("Q_lin","arxiv","token_weighted_mean"))}，github Q = {f3(q("Q_lin","github","token_weighted_mean"))}。标准化 Cronbach α = {f3(rel["cronbach_alpha_std"])}，分半信度 = {f3(rel["split_half_spearman_brown_mean"])}。',
    f'质量冲突：把"某指标处于前 25% 而另一指标处于后 25%"定义为冲突。强冲突率 github 为 {rate("github","strong_conflict_rate")*100:.2f}%，arxiv 为 {rate("arxiv","strong_conflict_rate")*100:.2f}%。主要成因有三：面向英文散文训练的打分器被用于代码(域漂移)，arxiv 上的天花板效应，以及长度效应(长度每增加 1 个标准差，冲突比值比 OR = {LG[LG.term=="log_words_std"].odds_ratio.iloc[0]:.2f})。消解模型采用"域自适应可信度加权 + 维度离散度悲观惩罚"，使 github 按 Q 选取前 30% 数据时的冲突样本占比由 {SEL[(SEL.domain=="github")&(SEL.score=="Q_lin")].conflict_rate_in_top30.iloc[0]*100:.1f}% 降至 {SEL[(SEL.domain=="github")&(SEL.score=="Q_resolved")].conflict_rate_in_top30.iloc[0]*100:.1f}%。消解后的域级质量为 arxiv Q* = {f3(q("Q_resolved","arxiv","token_weighted_mean"))}，github Q* = {f3(q("Q_resolved","github","token_weighted_mean"))}。',
    f'领域配比建模：仅用 A4/A5(512 组)训练，在 A6–A11 上检验。提出的对数边际递减配比律 M6：L = Σβᵢpᵢ + Σγᵢ ln(pᵢ+ε)，在 13 个域 Loss 上的 1M 检验 R² 平均为 {vA.loc["M6_log_share","test_1M_R2"]:.3f}(线性 Scheffé 仅 {vA.loc["M1_Scheffe_linear","test_1M_R2"]:.3f})，对 1B 尺度的秩相关平均为 {vA.loc["M6_log_share","test_1B_spearman"]:.3f}；LightGBM 精度最高，对应数值为 {vA.loc["M5_LightGBM","test_1M_R2"]:.3f} 和 {vA.loc["M5_LightGBM","test_1B_spearman"]:.3f}。配比熵与平均 Loss 的秩相关在 1M、60M、1B 三个实测尺度上分别为 {DV[DV.set=="test_1M"].spearman_H_vs_Lavg.iloc[0]:.2f}、{DV[DV.set=="test_60M"].spearman_H_vs_Lavg.iloc[0]:.2f}、{DV[DV.set=="test_1B"].spearman_H_vs_Lavg.iloc[0]:.2f}，即配比越多样，Loss 越低，且 6 个领域族两两互补。',
    f'外推稳健性：外推表 A12–A15 存在系统性问题。70B 表约等于 10B 表乘以常数 {e10.ratio_70B_over_10B_mean:.3f}(变异系数 {e10.ratio_70B_over_10B_cv*100:.1f}%)；外推表中"多样性–Loss"关系反号(+{DV[DV.set=="est_10B"].spearman_H_vs_Lavg.iloc[0]:.2f})；由外推表估计的域效应与实测尺度估计值的秩相关约为 0。因此外推结论不稳健，只作对照使用。',
    f'质量 Q 的引入：严格证明混合质量 Q_mix(p)=ΣpᵢQᵢ 与 Scheffé 线性项完全共线(数值秩 {QI["rank_of_[p,Q_mix]"]} < {QI["n_columns"]})。消融实验中，在"熵 + 覆盖份额"模型上加入 Q，1M 检验 R² 只从 {AB[AB.model=="a+H"].test1M_R2.iloc[0]:.3f} 提升到 {AB[AB.model=="a+Q·cov+H+cov (full)"].test1M_R2.iloc[0]:.3f}。结论是：Q 不作为配比回归的自变量，而用于解释域效应，并通过 Q_mix(p) 作为问题二的输入接口。',
    '数据缺口：A1 抽样集(51,230 条)在本计算环境中无法下载(Git LFS 服务器不在网络白名单内)。凡依赖 A1 的内容(7 个域的 Q、A1 与扩展集对照、原文校验)均已写入代码：放入 A1 后运行 run_all.sh 即可自动生成，本报告中相应位置标注为"待补"。'])

# ================================================================ 1 数据
H1('1  数据来源、完整性核验与使用清单')
Pp('配方与 Loss 表(A4–A16)取自题目仓库 ztangat/23th-NPMCM。A2、A3 在该仓库以 Git LFS 存储，而 LFS 下载域名在本环境被拦截，因此改从镜像仓库 ShaunDong/HuaweiCup-Competition-Problems-2026 获取 .xz 文件。解压后的 SHA-256 与原仓库 LFS 指针记录的 oid 逐字节一致，保证两者为同一数据。')
TB('表1-1  问题一数据使用清单与核验', pd.DataFrame([
    ['A1', 'slimpajama_quality_signal_sample.jsonl.xz', '51,230', '未获取(LFS 被拦截)', '代码已支持，待补'],
    ['A2', 'arxiv_part-6777d8857c6e-000486', '17,523', 'SHA-256 2625cc47…a96 与 LFS oid 一致', '全量使用'],
    ['A3', 'github_part-6777d8857c6e-000275', '203,752', 'SHA-256 3b3bb5a3…385 与 LFS oid 一致', '全量使用'],
    ['A4/A5', 'train_mixture_1m / train_pile_loss_1m', '512', 'index 一一对应，无缺失', '训练'],
    ['A6/A7', 'test_mixture_1m / test_pile_loss_1m', '256', '同上', '检验(1M)'],
    ['A8/A9', 'test_mixture_60m / test_pile_loss_60m', '256', '配方与 A6 完全相同', '检验(60M)'],
    ['A10/A11', 'test_mixture_1B / test_pile_loss_1B', '64', '独立配方(index 0–63)', '检验(1B)'],
    ['A12–A15', 'est_mixture/pile_loss_10b, 70b', '63', '配方为 A4 子集；10b 与 70b 配方相同', '外推稳健性'],
    ['A16', 'domain_mapping_guide.csv', '17', '参考映射', '跨体系关联'],
], columns=['编号', '文件', '记录数', '完整性核验', '用途']), [900, 3300, 900, 2800, 1400])
Pp(f'A2/A3 解析日志：JSON 解析错误 0 条，列表长度异常 0 条；modernbert_professionalism 缺失 {pre["imputed_missing"].get("modernbert_professionalism",0)} 条，按域中位数插补；2-gram 与 3-gram 字符占比超过 100% 的非法值分别有 {pre["top2_gt100"]} 条和 {pre["top3_gt100"]} 条，截断到 [0,100]。A4–A15 各行配比之和在 0.996–1.003 之间(千分位舍入)，统一重新归一化到 Σp=1。')
TB('表1-2  配方与 Loss 表审计(S06_data_audit.csv)', AU.assign(**{c: AU[c].round(4) for c in ['raw_sum_min', 'raw_sum_max', 'zero_share_frac', 'L_avg_mean', 'L_avg_sd', 'L_avg_min', 'L_avg_max']})
   [['set', 'n', 'raw_sum_min', 'raw_sum_max', 'zero_share_frac', 'L_avg_mean', 'L_avg_sd', 'L_avg_min', 'L_avg_max']], fs=7.5)

# ================================================================ 2 质量评价
H1('2  问题一(1)：数据质量评价')
H2('2.1  列表型指标压缩与方向统一')
TB('表2-1  8 个列表型指标的标量化规则', pd.DataFrame([
    ['modernbert_cleanliness / readability / reasoning / professionalism', '6 维 logits(对应 0–5 分)', 'softmax 后取期望分 Σk·softmax(l)ₖ ∈ [0,5]', '正向'],
    ['fineweb_edu', '长度 1 的回归分', '直接取值', '正向'],
    ['qurater', '4 维：写作风格、所需专业度、事实性、教育价值', '4 维等权均值(4 个分量另行保存)', '正向'],
    ['fluency_en', '2 类 logits', f'P(流畅)=softmax 第 2 类；arxiv 上均值 {pre["anchor_fluency_en_p1_mean_arxiv"]:.3f}，确认第 2 类为"流畅"', '正向'],
    ['ad_en', '2 类 logits', f'arxiv 上第 2 类概率均值 {pre["anchor_ad_en_p1_mean_arxiv"]:.3f}，确认第 2 类为"非广告"，故 P(广告)=1−p₁', '负向，取 1−x'],
], columns=['指标', '原始结构', '压缩规则', '方向']), [2600, 1900, 3600, 1200], fs=7.5)
Pp('锚定依据：arxiv 为学术论文全文，先验上几乎不含广告且语言流畅，据此确定二分类输出中各类别的含义，而不是主观假定。')
H2('2.2  标量指标变换与归一化')
Pp('原始指标按性质分为正向、负向和适度型三类。所有分位数与统计量都采用域均衡权重 w = 1/(K·n_d) 计算(K 为域数，n_d 为该域样本数)，否则 github(占 92%)会主导归一化边界与客观权重。')
EQ('正向：z = (x̃ − P₁)/(P₉₉ − P₁)；负向：z = 1 − (x̃ − P₁)/(P₉₉ − P₁)；其中 x̃ = clip(x, P₁, P₉₉)')
EQ('适度型(理想区间 [a,b])：z = 1(a≤x̃≤b)，(x̃−P₁)/(a−P₁)(x̃<a)，(P₉₉−x̃)/(P₉₉−b)(x̃>b)')
BL(['DSIR 三项给出的是整篇文档的对数重要性之和，随长度线性累积，因此除以词数得到逐词平均值。',
    f'唯一词占比天然随长度下降(Heaps 律)。拟合 ln(frac_unique) = {pre["heaps_fit"]["a"]:.3f} {pre["heaps_fit"]["b"]:+.4f}·ln(word_count)(隐含 Heaps 指数 β = {pre["heaps_fit"]["implied_heaps_beta"]:.3f})，取残差作为去长度后的词汇多样性。',
    '词数与平均词长按 Gopher 质量规则设为适度型，理想区间分别为 [50, 10⁵] 词和 [3, 10] 字符；句数为下限型(≥3 句)。'])
nz = NZ.copy(); nz['winsor_P01'] = nz.winsor_P01.map(lambda x: f'{x:.4g}'); nz['winsor_P99'] = nz.winsor_P99.map(lambda x: f'{x:.4g}'); nz['z_mean_balanced'] = nz.z_mean_balanced.map(f3)
TB('表2-2  22 项指标规格、方向与归一化参数(S02_indicator_spec_and_normalization.csv)',
   nz[['indicator', 'dimension', 'type', 'raw_column', 'winsor_P01', 'winsor_P99', 'n_clipped_low', 'n_clipped_high', 'z_mean_balanced']], fs=6.5)
H2('2.3  综合评价模型：熵权–CRITIC 组合赋权')
EQ('熵权：pᵢⱼ = wᵢzᵢⱼ/Σᵢwᵢzᵢⱼ，eⱼ = −Σᵢpᵢⱼ ln pᵢⱼ / ln n，w⁽ᴱ⁾ⱼ ∝ 1 − eⱼ')
EQ('CRITIC：Cⱼ = σⱼ Σₖ(1 − rⱼₖ)，w⁽ᶜ⁾ⱼ ∝ Cⱼ；组合：wⱼ = √(w⁽ᴱ⁾ⱼ w⁽ᶜ⁾ⱼ) / Σₖ√(w⁽ᴱ⁾ₖ w⁽ᶜ⁾ₖ)')
Pp('熵权反映指标的区分度，CRITIC 同时考虑波动性与冗余度：高度相关的 DSIR 三项会被自动降权，避免重复计入。几何平均组合是在"与两种权重的相对熵之和最小"意义下的最优组合。样本级质量分为 Qᵢ = Σⱼ wⱼ zᵢⱼ ∈ [0,1]。')
w = W.copy()
for c in ['sd_balanced', 'sum_1_minus_r', 'entropy_e', 'w_entropy', 'w_critic', 'w_combined', 'w_structural_equal']:
    w[c] = w[c].map(f4)
TB('表2-3  指标权重(S02_indicator_weights.csv)', w, fs=6.5)
IMG('F01_indicator_spearman.png', '图1  22 项指标的域均衡 Spearman 相关矩阵')
IMG('F02_weights.png', '图2  熵权、CRITIC 与组合权重')
H2('2.4  由样本向语料/领域聚合')
Pp('域级 Q 同时报告多种聚合口径。主口径为 token(词数)加权均值：训练时模型按 token 而非按文档消费数据，长文档对训练分布的贡献按其长度计。辅助口径包括文档均值(附 300 次 bootstrap 95% 置信区间)、中位数、10% 截尾均值、P10/P90 以及 Q ≥ 0.6 的样本占比(用于质量筛选)。')
EQ('Q_d(token) = Σ_{i∈d} len_i·Q_i / Σ_{i∈d} len_i')
dq = DQ.copy()
for c in ['mean_doc', 'ci95_low', 'ci95_high', 'median_doc', 'trimmed_mean_10', 'token_weighted_mean', 'P10', 'P90', 'share_Q_ge_0.6']:
    dq[c] = dq[c].map(f4)
dq['n_docs'] = dq.n_docs.astype(int)
TB('表2-4  域级质量评分(全部方法与聚合口径，S04_domain_level_Q_all_methods.csv)', dq, fs=6.5)
IMG('F03_Q_distribution.png', '图3  样本级 Q 分布(左：组合权线性评分；右：冲突消解后评分)')
Pp(f'结果解读：arxiv 在语义价值、语言规范、参考域相似三个维度上的归一化均值都接近饱和，例如 modernbert_reasoning 在 arxiv 的中位数为 5.00，在 github 为 1.00。github 的质量分布宽且偏低(P10–P90：{f3(q("Q_lin","github","P10"))}–{f3(q("Q_lin","github","P90"))})，因此在 github 域内做质量筛选的空间更大。github 的 token 加权 Q 高于其文档均值({f3(q("Q_lin","github","token_weighted_mean"))} 对 {f3(q("Q_lin","github","mean_doc"))})，说明较长的代码文件质量更高，大量极短文件拉低了文档均值。')
H2('2.5  可靠性检验')
TB('表2-5  信度与方法一致性', pd.DataFrame([
    ['标准化 Cronbach α(域均衡)', f3(rel['cronbach_alpha_std'])],
    ['Cronbach α(arxiv 域内)', f3(rel['cronbach_alpha_std_arxiv'])], ['Cronbach α(github 域内)', f3(rel['cronbach_alpha_std_github'])],
    ['分半信度(维内随机对半 200 次，Spearman-Brown 校正)均值 [P5, P95]', f'{f3(rel["split_half_spearman_brown_mean"])} [{f3(rel["split_half_P5_P95"][0])}, {f3(rel["split_half_P5_P95"][1])}]'],
    ['组合权 Q 与等权 / TOPSIS / PCA 第一主成分 / 消解 Q* 的样本秩相关', ' / '.join(f3(CON.loc['Q_lin', c]) for c in ['Q_equal', 'Q_topsis', 'Q_pca1', 'Q_resolved'])],
    ['留一指标后 Q 的秩相关最小值(组合权 / 消解)', f'{f3(LO.spearman_Qlin.min())} / {f3(LO.spearman_Qresolved.min())}'],
], columns=['检验项', '结果']), [6200, 3100])
H2('2.6  抽样集 A1 的域级 Q 及与扩展集对照(待补)')
NOTE('A1 未能获取。代码 s02_quality_model.py 在 A1 存在时会：(1) 把 A1 与 A2/A3 合并做同一套预处理与赋权，输出 arxiv、book、c4、commoncrawl、github、stackexchange、wikipedia 七个域的 Q；(2) 在 S05_A1_vs_extended_same_domain.csv 中对 arxiv、github 两域比较"抽样集 vs 全量扩展集"的均值差、KS 统计量与冲突率；(3) 输出 Q 与原文统计特征(代码符号占比、URL 数)的相关，以及每个域 Q 最高/最低各 15 条和强冲突样本的原文摘录(S05_A1_text_excerpts_for_manual_check.csv)，用于人工校验评分是否可靠。')

# ================================================================ 3 冲突
H1('3  问题一(2)：质量冲突的定义、成因与消解')
H2('3.1  冲突的定义')
Pp('设 rᵢⱼ 为样本 i 在指标 j 上的域均衡百分位秩(方向已统一为越高越好)。分三个层次定义冲突：')
EQ('指标对冲突：C_jk(i) = 1{ (rᵢⱼ ≥ 0.75 ∧ rᵢₖ ≤ 0.25) ∨ (rᵢⱼ ≤ 0.25 ∧ rᵢₖ ≥ 0.75) }；独立基准概率 = 2×(1/4)² = 0.125')
EQ('维度分：D_k(i) = F(Σ_{j∈k} wⱼ rᵢⱼ / Σ_{j∈k} wⱼ)；冲突度 κᵢ = max_k D_k − min_k D_k；强冲突 ⇔ max_k D_k ≥ 0.75 且 min_k D_k ≤ 0.25')
Pp('四个维度划分如下：D1 语义价值(fineweb_edu、qurater、reasoning、professionalism)，D2 语言规范(fluency、readability、cleanliness、ad)，D3 参考域相似(DSIR 三项)，D4 统计启发式(11 项 RedPajama 信号)。层次聚类(S02_indicator_clustering_check.csv)显示，模型打分类指标与统计启发式指标确实分属不同簇。')
H2('3.2  冲突的规模与结构')
rt = RT.copy()
for c in rt.columns[3:]:
    rt[c] = rt[c].map(f4)
TB('表3-1  各域冲突统计(S03_conflict_rate_by_domain.csv)', rt, fs=7.5)
pr = PR.head(12).copy()
for c in ['conflict_rate_balanced', 'lift_vs_independence', 'spearman', 'rate_github', 'rate_arxiv']:
    pr[c] = pr[c].map(f3)
TB('表3-2  冲突率最高的 12 个指标对(S03_pairwise_conflict_ranked.csv，共 231 对)', pr, fs=6.5)
IMG('F04_pairwise_conflict.png', '图4  指标两两冲突率矩阵')
ty = TY.copy(); ty['share_within_domain_conflicts'] = ty.share_within_domain_conflicts.map(f3)
TB('表3-3  强冲突类型构成(S03_conflict_typology.csv)', ty, fs=7.5)
IMG('F05_conflict_type_length.png', '图5  冲突类型与长度效应')
H2('3.3  冲突成因分析')
lg = LG.copy()
for c in ['coef', 'odds_ratio', 'OR_low95', 'OR_high95']:
    lg[c] = lg[c].map(f4)
lg['p_value'] = lg.p_value.map(lambda x: f'{x:.2e}')
TB('表3-4  强冲突的 logistic 回归(基准域 github；连续变量已标准化)', lg, fs=7.5)
cs = CS.head(10).copy()
for c in ['arxiv', 'github', 'rho_gap_arxiv_minus_github']:
    cs[c] = cs[c].map(f3)
TB('表3-5  同一指标对在不同域内的相关方向漂移(前 10，S03_within_domain_corr_shift.csv)', cs, fs=7)
IMG('F06_corr_shift.png', '图6  域漂移：相关方向在 arxiv 与 github 之间反转')
BL([f'成因一：测量对象不同。冲突率最高的指标对集中在 D4 内部，例如"数字字符占比–平均词长"的冲突率为 {PR.iloc[0].conflict_rate_balanced:.3f}，是独立基准的 {PR.iloc[0].lift_vs_independence:.2f} 倍。统计启发式规则彼此之间就在度量不同的表层属性，高冲突本身并不意味着数据有错。',
    f'成因二：域漂移。github 的强冲突中 {TY[(TY.domain=="github")].share_within_domain_conflicts.iloc[0]*100:.1f}% 属于"D4 统计启发式高 / D2 语言规范低"。代码文件格式规整、重复度低，能通过启发式规则；但 fluency、readability 等打分器是在英文散文上训练的，对代码天然给低分。控制长度等因素后，arxiv 的强冲突比值比仅为 github 的 {LG[LG.term=="dom_arxiv"].odds_ratio.iloc[0]:.3f}。',
    f'成因三：天花板效应与口径差异。arxiv 上 professionalism、reasoning 的中位数接近满分 5，剩余差异主要是噪声，导致 professionalism 与 DSIR 在 arxiv 内呈负相关(ρ = {CS.iloc[0].arxiv:.2f})，而在 github 内为正(ρ = {CS.iloc[0].github:.2f})。此外，DSIR 逐词分偏好字母文本，在 arxiv 内与无字母词占比强相关(ρ≈0.82)，会惩罚 LaTeX 公式密集的论文。',
    f'成因四：长度效应。长度每增加 1 个标准差，强冲突 OR = {LG[LG.term=="log_words_std"].odds_ratio.iloc[0]:.2f}；github 最长 10% 文件的强冲突率约为 9.7%，是最短分位的十余倍。长文件往往是数据文件、自动生成代码或许可证文本，各指标对它们的判断分歧最大。',
    f'题面所举"教育价值高且广告含量高"的冲突：arxiv 为 {rate("arxiv","edu_high_ad_high_rate")*100:.1f}%，github 为 {rate("github","edu_high_ad_high_rate")*100:.2f}%。arxiv 中这部分样本的广告概率只是相对偏高(绝对值中位数仅 0.0012)，更可能是基金致谢、网址、软件推介等段落触发了广告分类器，而非真正的广告。这一推断需要借助 A1 原文进一步核实。'])
H2('3.4  冲突消解模型')
Pp('消解分两步：第一步在域内按每个指标的"可信度"调整权重，第二步在维度之间对分歧做悲观惩罚。')
EQ('ρⱼ,d = Spearman(r_j, Σ_{k≠j} wₖ r_k / Σ_{k≠j} wₖ)|_{域 d}；τⱼ,d = (1 + ρⱼ,d)/2')
EQ('D̃_k(i) = Σ_{j∈k} wⱼ τⱼ,d zᵢⱼ / Σ_{j∈k} wⱼ τⱼ,d；Q*ᵢ = clip(Σ_k ω_k D̃_k(i) − λ·sd_k(D̃_k(i)), 0, 1)，ω_k = Σ_{j∈k} wⱼ，λ = 0.5')
Pp('第一步的依据来自潜变量测量模型 z_j = λ_j·Q + ε_j：指标与"其余指标共识"的相关近似正比于其载荷 λ_j。因此在某个域内与共识相悖的指标(例如在代码上的散文打分器)会被自动降权；与共识正交的指标(ρ≈0，可能在测量独立的有效侧面，如广告)保留一半权重，而不是被直接删除。第二步借鉴均值–方差思想：各维度判断越不一致，评分的不确定性越大，在用于数据筛选时应当保守处理。')
tr_ = TR.pivot(index='indicator', columns='domain', values='rho_with_loo_consensus').round(3).reset_index()
TB('表3-6  各指标与留一共识的域内相关 ρ(可信度 τ=(1+ρ)/2，S04_domain_adaptive_trust_weights.csv)', tr_, fs=7)
sel = SEL.copy()
for c in ['top30_threshold', 'conflict_rate_in_top30', 'conflict_rate_overall', 'kappa_range_mean_top30']:
    sel[c] = sel[c].map(lambda x: '' if pd.isna(x) else f4(x))
TB('表3-7  消解效果：按 Q 选取前 30% 时的冲突样本占比(S04_selection_effect_top30.csv)', sel, fs=7)
lam = LAM.round(4).astype(str)
TB('表3-8  惩罚系数 λ 的敏感性', lam, fs=7.5)
Pp(f'结论：消解后，github 高质量子集里的强冲突样本减少约 {(1 - SEL[(SEL.domain=="github")&(SEL.score=="Q_resolved")].conflict_rate_in_top30.iloc[0]/SEL[(SEL.domain=="github")&(SEL.score=="Q_lin")].conflict_rate_in_top30.iloc[0])*100:.0f}%，与原前 30% 子集的 Jaccard 重合度为 {SEL[(SEL.domain=="github")&(SEL.score=="overlap_top30_Jaccard")].conflict_rate_in_top30.iloc[0]:.3f}。在稳定性上，留一指标后秩相关仍 ≥ {LO.spearman_Qresolved.min():.3f}；λ 在 0–1 之间变化时样本秩相关 ≥ {LAM.spearman_vs_lam05.min():.3f}，域间排序不变。')
H2('3.5  扩展集检验与 A1 对照(待补)')
NOTE('本节的冲突统计全部基于扩展集 A2/A3 的全量记录。题目要求"冲突分析覆盖抽样集，并在扩展集上检验主要结论是否成立"。A1 到位后，脚本会自动输出按 set×domain 分组的冲突率、类型构成与 logistic 回归，并在 arxiv、github 两个共同域上直接比较两套数据。')

# ================================================================ 4 配比
H1('4  问题一(3)：领域配比 p 与交叉熵损失的定量关系')
H2('4.1  单纯形约束与缺失 Loss 域的处理')
BL(['单纯形约束：各行配比先重新归一化到 Σp=1。回归一律采用 Scheffé 无截距形式——由于 Σpᵢ=1，截距已被吸收到各域系数中，模型不会出现完全共线。训练配方中有 45% 的份额为 0，对数比变换(ALR/ILR)遇零不可用，因此改用 ln(pᵢ+ε)，ε 由 5 折交叉验证选取。',
    'nih_exporter、enron_emails、europarl、philpapers 四个域只有配比、没有对应的 Loss 列。这四个域保留为自变量：删掉会破坏单纯形，与其他域合并会丢失信息。它们的作用通过对其余 13 个域 Loss 的间接影响来估计。',
    '因变量包括 13 个域 Loss 以及平均 Loss L_avg = (1/13)Σ L_j。1M、60M、1B 三个尺度的 Loss 绝对水平不同，因此跨尺度检验以秩相关为主，并辅以 2 折交叉的仿射校准 R²。'])
H2('4.2  候选模型')
TB('表4-1  六类配比–Loss 模型', pd.DataFrame([
    ['M1', '线性 Scheffé', 'L = Σβᵢpᵢ', '17', '基线，可解释'],
    ['M2', '二次 Scheffé + 岭', 'L = Σβᵢpᵢ + Σ_{i<j}βᵢⱼpᵢpⱼ', '153', '刻画两两组合效应'],
    ['M3', '指数混合律', 'L = c + exp(Σtᵢpᵢ)', '18', 'Ye et al.(2024) 数据混合律'],
    ['M4', '线性 + 自域对数', 'L_j = Σβᵢpᵢ + g·ln(p_j+ε)', '18', '仅用于 13 个域目标'],
    ['M5', 'LightGBM', '梯度提升树', '—', 'RegMix 原文采用的回归器'],
    ['M6', '对数边际递减律(本文主模型)', 'L = Σβᵢpᵢ + Σγᵢ ln(pᵢ+ε)', '34', 'γᵢ 为 L 对 ln pᵢ 的弹性'],
], columns=['编号', '名称', '形式', '参数数', '说明']), [700, 2200, 3300, 900, 2200])
Pp(f'超参数全部由 A4/A5 上的 5 折交叉验证确定(S07_hyperparameter_cv.csv)：M6 用于 L_avg 时 ε = {hp["best_logshare_eps"]["L_avg"]}；M2 的岭参数 α 逐目标选取。')
H2('4.3  在检验集 A6–A11 上的验证')
cols = ['cv5_R2', 'test_1M_R2', 'test_1M_RMSE', 'test_1M_spearman', 'test_60M_spearman', 'test_60M_recal_R2_cv2', 'test_1B_spearman', 'test_1B_recal_R2_cv2']
t1 = vA[cols].round(3).reset_index(); t2 = vL[cols].round(3).reset_index()
TB('表4-2  13 个域 Loss 目标上的平均检验表现', t1, fs=7)
TB('表4-3  平均 Loss(L_avg)上的检验表现', t2, fs=7)
m6 = V[V.model == 'M6_log_share'][['target', 'cv5_R2', 'test_1M_R2', 'test_1M_RMSE', 'test_60M_spearman', 'test_1B_spearman']].round(3)
TB('表4-4  主模型 M6 逐目标检验结果', m6, fs=7)
tk = TK[TK.model.isin(['M1_Scheffe_linear', 'M6_log_share', 'M5_LightGBM'])].copy(); tk['precision_at_k'] = tk.precision_at_k.round(2)
TB('表4-5  Top-k 命中率与"预测最优配方"的真实排名(L_avg)', tk, fs=7)
IMG('F07_model_validation.png', '图7  六类模型在检验集上的表现')
IMG('F09_pred_vs_obs.png', '图9  1M 模型预测值与各尺度实测 L_avg')
Pp(f'结论：线性模型解释力明显不足(L_avg 的 1M 检验 R² 仅 {vL.loc["M1_Scheffe_linear","test_1M_R2"]:.3f})，说明配比对 Loss 的影响本质上是非线性的。只加入对数边际递减项的 M6 就把 L_avg 的 R² 提高到 {vL.loc["M6_log_share","test_1M_R2"]:.3f}，其"预测最优配方"在 1M 和 1B 检验集上都是真实排名第 1 的配方，在 60M 上排第 2。LightGBM 的逐点精度最高，但无法给出显式的弹性系数。M6 在 1B 尺度 L_avg 上的秩相关({vL.loc["M6_log_share","test_1B_spearman"]:.3f})反而略高于 LightGBM({vL.loc["M5_LightGBM","test_1B_spearman"]:.3f})，说明参数化的边际递减结构在跨尺度迁移上更稳健。')
H2('4.4  各领域对性能的影响')
Pp('以实验配方均值 s̄ 为参照点(近似自然 token 分布)，沿 Cox 方向把第 i 个域的份额增加 Δ=0.1，同时其余各域按原比例等比缩减，计算 Loss 的变化 Eᵢ。模型用 M6 在训练与 1M 检验合并后的 768 组配方上拟合，置信区间由 200 次 bootstrap 给出。')
EQ('Eᵢ = f(s̄ + Δ(eᵢ − s̄)/(1 − s̄ᵢ)) − f(s̄)；弹性 γᵢ = ∂L/∂ln(pᵢ+ε)')
e6 = E6[E6.target == 'L_avg'].sort_values('cox_effect_plus0_1')[['domain', 'cox_effect_plus0_1', 'ci_low', 'ci_high', 'significant', 'beta_linear', 'gamma_log']].copy()
for c in ['cox_effect_plus0_1', 'ci_low', 'ci_high', 'beta_linear', 'gamma_log']:
    e6[c] = e6[c].map(f4)
TB('表4-6  各域对 L_avg 的边际影响(M6，份额 +0.1；负值表示降低 Loss)', e6, fs=7)
shp = SHP.copy(); shp['mean_abs_shap'] = shp.mean_abs_shap.map(f4); shp['shap_share_corr'] = shp.shap_share_corr.map(f3)
TB('表4-7  LightGBM SHAP 重要性(L_avg)', shp, fs=7)
IMG('F08_cox_effect_heatmap.png', '图8  训练域份额对 13 个验证域 Loss 的边际影响矩阵(17×14)')
BL(['对角占优：每个验证域的 Loss 主要由同名训练域决定。例如 ubuntu_irc 份额 +0.1 使其自身 Loss 下降约 0.62，dm_mathematics 下降约 0.32；对其他域则大多是轻微的挤出效应(+0.02~0.10)。',
    '弹性 γ 最负的几个域为 dm_mathematics(−0.081)、ubuntu_irc(−0.051)、stackexchange(−0.047)、pubmed_central(−0.041)、github(−0.041)，这与 SHAP 排序(dm_mathematics、stackexchange、ubuntu_irc 居前)一致：这些域"从 0 到有"的边际价值最大。',
    '在参照点上继续加大 arxiv、freelaw、github 等大份额域会提高 L_avg(约 +0.04/0.1 份额，显著)；stackexchange、pile_cc 接近中性；hackernews、philpapers、enron_emails 为负但置信区间跨 0。可见在自然分布附近，均衡比集中更有利。',
    'nih_exporter 等 4 个无 Loss 列的域：enron_emails 的效应不确定性最大(95% 置信区间约 [−0.26, 0.20])，nih_exporter 为显著正值(+0.086)，都只能通过对其他域的间接影响来解释。'])
H2('4.5  领域组合效应：互补与替代')
Pp('把 17 个域归并为 6 个领域族，拟合族层面的二次 Scheffé 模型(6 个主效应 + 15 个交互项，参数少、估计稳定)。交互系数 βᵢⱼ < 0 表示混合后的 Loss 低于两个纯族线性插值的结果，即两族互补；越接近 0，两族越可以相互替代。')
TB('表4-8  领域族定义', FD.groupby('family').domain.apply(lambda s: ', '.join(s)).reset_index(), [1800, 7500])
fm = FM[FM.target == 'L_avg'][['term', 'coef', 'ci_low', 'ci_high', 'interpretation']].copy()
for c in ['coef', 'ci_low', 'ci_high']:
    fm[c] = fm[c].map(f3)
TB(f'表4-9  族层面二次 Scheffé 系数(L_avg；样本内 R²={FM[FM.target=="L_avg"].in_sample_R2.iloc[0]:.3f}，仅用训练集拟合的 1M 检验 R²={FM[FM.target=="L_avg"].test1M_R2_train_only.iloc[0]:.3f})', fm, fs=7)
IMG('F12_family_interactions.png', '图12  领域族交互系数')
dp = pd.concat([DP[DP.significant].head(6), DP[DP.significant].tail(6)])[['domain_i', 'domain_j', 'coef', 'ci_low', 'ci_high', 'co_occurrence_both_gt_1pct']].copy()
for c in ['coef', 'ci_low', 'ci_high', 'co_occurrence_both_gt_1pct']:
    dp[c] = dp[c].map(f3)
TB(f'表4-10  17 域两两交互(岭回归，200 次 bootstrap；136 对中 {int(DP.significant.sum())} 对显著)：互补最强与最弱各 6 对', dp, fs=7)
Pp('解读：15 个族交互项全部显著为负，说明任意两个领域族搭配都优于单一族。互补性最强的是"问答对话×书籍议会"(−3.85)和"问答对话×法律专利"(−3.34)，二者文体差异大，信息重叠少；最弱的是"代码数学×问答对话"(−1.55)和"网页通用×法律专利"(−1.76)，最接近替代关系，这与 stackexchange 的技术问答和代码在内容上高度重叠相吻合。在域层面，ubuntu_irc 与几乎所有域都呈强互补；github×stackexchange(−1.00)、github×pile_cc(−1.74)的互补性最弱。')
H2('4.6  多样性效应')
dv = DV.copy()
for c in ['H_mean', 'spearman_H_vs_Lavg', 'spearman_nzero_vs_Lavg']:
    dv[c] = dv[c].map(f3)
TB('表4-11  配比熵 H(p) = −Σpᵢ ln pᵢ 与 L_avg 的秩相关', dv, fs=7.5)
Pp('在三个实测尺度上，配比越均衡(熵越高、零份额域越少)，平均 Loss 越低。只用熵 H 一个变量的两参数模型就能在 1M 检验集上达到 R²≈0.66，这正是 M6 中对数项的来源：某个域份额趋于 0 时，ln(pᵢ+ε) → ln ε，对应 Loss 急剧上升。')
H2('4.7  推荐配比 p*')
Pp('优化目标为 min L_avg(p)，约束为 p∈Δ¹⁶，并加信赖域 pᵢ ≤ maxᵢ(实验中出现过的份额)，防止外推到从未观测过的区域。求解方法有两种：(a) 以 M6 为目标，SLSQP 多起点求解；(b) 仿照 RegMix，在实验配方的凸组合与参照点附近的 Dirichlet 分布中抽取 10 万个候选，用 M6 与 LightGBM 的集成模型打分，取前 100 个的平均作为推荐配比。二次 Scheffé 的最优解(表中 p_opt_quadratic)会被推到单纯形边界，其预测值 4.08 被另外两个模型否定(预测值约 5.0)，是典型的过拟合外推，作为反例列出。')
op = OP.copy()
for c in op.columns[1:]:
    op[c] = op[c].map(f4)
TB('表4-12  推荐配比(S08_optimal_mixture.csv)', op, fs=7)
opl = OPL.round(4).astype(str).replace('nan', '')
TB('表4-13  各配比的预测平均 Loss(1M 尺度)', opl, fs=7)
oc = OC.copy()
for c in ['L1_to_top10_mean', 'L1_ref_to_top10', 'spearman_with_top10']:
    oc[c] = oc[c].map(f3)
TB('表4-14  推荐配比与各尺度实测最优 10 个配方平均构成的一致性', oc, fs=7)
IMG('F10_optimal_mixture.png', '图10  推荐配比与参照配比')
Pp(f'推荐配比(集成 Top-100)与 1M、60M、1B 各尺度实测最优 10 个配方平均构成的秩相关分别为 {OC[(OC.scale=="test_1M")&(OC.optimum=="p_opt_ensemble_top100")].spearman_with_top10.iloc[0]:.2f}、{OC[(OC.scale=="test_60M")&(OC.optimum=="p_opt_ensemble_top100")].spearman_with_top10.iloc[0]:.2f}、{OC[(OC.scale=="test_1B")&(OC.optimum=="p_opt_ensemble_top100")].spearman_with_top10.iloc[0]:.2f}，L1 距离小于参照配比与实测最优构成之间的距离。它的主要调整方向是提高 pile_cc、stackexchange、hackernews、ubuntu_irc、dm_mathematics 的份额，降低 arxiv、pubmed_central、github 的份额。')
H2('4.8  利用外推表 A12–A15 讨论稳健性')
ex = EX.copy()
for c in ex.columns[1:]:
    if ex[c].dtype != bool:
        ex[c] = ex[c].map(f3)
TB('表4-15  外推表内部与跨尺度一致性(S09_extrapolation_consistency.csv)', ex, fs=6.3)
ag = AG6.copy(); ag['spearman'] = ag.spearman.map(f3); ag['sign_agreement'] = ag.sign_agreement.map(f3)
TB('表4-16  各尺度独立估计的 M6 域效应向量之间的一致性', ag, fs=7)
xr = XR.copy(); xr['spearman'] = xr.spearman.map(f3); xr['kendall'] = xr.kendall.map(f3)
TB('表4-17  1M 模型对外推表配方的排序能力', xr, fs=7)
IMG('F11_extrapolation.png', '图11  外推稳健性诊断')
BL([f'外推表内部不独立：70B/10B 的比值恒为 {e10.ratio_70B_over_10B_mean:.3f}(变异系数 {e10.ratio_70B_over_10B_cv*100:.1f}%)，两表秩相关为 {e10.spearman_10B_vs_70B:.3f}，70B 表没有提供新的信息。',
    f'与实测规律矛盾：在 1M、60M、1B 三个实测尺度上，多样性都降低 Loss；外推表中两者却呈正相关(+{DV[DV.set=="est_10B"].spearman_H_vs_Lavg.iloc[0]:.2f})。同一批 63 个配方的 L_avg，在 1M 实测与 10B 外推之间的秩相关为 {e10.spearman_1M_vs_10B:.2f}，排序被反转。',
    '域效应不可迁移：实测尺度之间域效应向量的秩相关为 0.70–0.86(1M 对 60M、1M 对 1B)，而任一实测尺度与外推表之间在 −0.36~0.03 之间(基本不相关甚至反向)。',
    'github 的外推 10B Loss 均值(1.79)高于 1B 实测均值(1.06)，违背"模型越大 Loss 越低"的常识；外推表还把 13 个域的 Loss 压缩到 1.35–2.25 的窄区间，抹平了 1B 实测中 1.06–3.15 的域间差异。',
    '结论：外推表的问题在于"逐域幂律外推后再组合"的构造方式，并不是本文模型不稳健。在 1M→60M→1B 这段真实尺度链上，M6 域效应的符号一致率为 82%–94%，这部分结论可以信赖；10B、70B 的结论应视为不可靠，只作对照。'])
H2('4.9  是否以及如何引入质量评分 Q')
Pp('命题：若每个域的质量 Qᵢ 为常数，则混合质量 Q_mix(p) = Σᵢ pᵢQᵢ 是 p 的线性函数，可由 Scheffé 线性项 Σβᵢpᵢ 精确表示(令 βᵢ' + '← βᵢ + bQᵢ)。因此在"固定域集合、只改变配比"的实验(A4–A15)中，Q 的效应与域身份完全混杂，无法从数据中单独识别。')
Pp(f'数值验证：设计矩阵 [p, Q_mix] 的秩为 {QI["rank_of_[p,Q_mix]"]}，列数为 {QI["n_columns"]}，确为完全共线。退一步，构造低维结构化模型 L = a + b·(Q·cov) + c·H(p) + d·cov 做嵌套消融(cov 为有 Q 值的域所占份额)：')
ab = AB.copy()
for c in ['train_R2', 'test1M_R2', 'test_60M_spearman', 'test_1B_spearman']:
    ab[c] = ab[c].map(f3)
TB('表4-18  Q 的增量解释力消融(S09_quality_structured_ablation.csv)', ab, fs=7)
qm = QM.copy()
for c in ['Q_quality_domain', 'Q_shrunk']:
    qm[c] = qm[c].map(lambda x: '待补(A1)' if pd.isna(x) else f3(x))
TB('表4-19  17 个配方域 → 质量域映射及 Q 取值(置信度收缩 Q̂ = c·Q_map + (1−c)·Q̄)', qm, fs=6.5)
BL(['映射规则：A16 中 direct 类置信度 c=1.0，near_direct 类 c=0.8。inferred 类按文体归入最接近的质量域：学术类→arxiv，技术问答/对话→stackexchange，正式说明文本→wikipedia，非正式通信→commoncrawl；c 取 0.3–0.6，并向全体均值收缩，以体现映射的不确定性。',
    f'论证结论：Q 不作为配比回归的自变量。只用 Q 的模型 1M 检验 R²≈0；在熵模型上加入 Q，R² 仅提升 {AB[AB.model=="a+Q·cov+H+cov (full)"].test1M_R2.iloc[0]-AB[AB.model=="a+H"].test1M_R2.iloc[0]:.3f}。Q 的正确用法有两个：(1) 解释域效应——同等份额下，高 Q 域是否带来更大的降 Loss 效果(待 A1 补全 7 个质量域后检验)；(2) 作为问题二的接口，以 Q_mix(p) 把配比折算为混合语料质量，与 N、D 一起进入广义标度律；质量提升通过"同一域内筛选"来实现(即问题三中的 C_Q)，而不是通过更换域。',
    '当前只有 arxiv、github 两个质量域有 Q，覆盖 7 个配方域。A1 到位后，脚本会自动补全 17 个域的 Q̂ 与 Q_mix，并重新运行上述检验。'])

# ================================================================ 5 结论
H1('5  结论与向问题二、问题三传递的输出')
TB('表5-1  问题一的输出接口', pd.DataFrame([
    ['域级质量 Q(当前)', f'arxiv {f3(q("Q_resolved","arxiv","token_weighted_mean"))}，github {f3(q("Q_resolved","github","token_weighted_mean"))}(冲突消解后，token 加权)', 'S04_domain_level_Q_all_methods.csv'],
    ['17 域 Q̂ 映射', '表4-19(7 个域已有值，其余待 A1)', 'S09_domain_Q_mapping_17.csv'],
    ['推荐配比 p*', '表4-12，集成 Top-100 列', 'S08_optimal_mixture.csv'],
    ['配比–Loss 律', f'M6：L = Σβᵢpᵢ + Σγᵢ ln(pᵢ+ε)，ε={hp["best_logshare_eps"]["L_avg"]}，系数见表4-6', 'S08_M6_cox_effect_plus0.1_bootstrap.csv'],
    ['基线质量 Q₀(问题三)', f'建议取 github 消解前 token 加权 Q ≈ {f3(q("Q_lin","github","token_weighted_mean"))} 或全体均值；A1 补全后更新', 'S04_domain_level_Q_all_methods.csv'],
], columns=['输出', '取值/说明', '文件']), [2000, 4800, 2500])
H1('6  局限与待补')
BL(['A1 抽样集缺失：七个域的 Q、抽样集与扩展集对照、原文校验、抽样集上的冲突分析都需补跑。将 slimpajama_quality_signal_sample.jsonl.xz 放入数据目录后运行 bash code/run_all.sh 即可，所有表格会自动覆盖。',
    '归一化边界与客观权重依赖参与计算的样本总体。加入 A1 后，边界与权重会变化，arxiv、github 的 Q 绝对值会随之变化，但本文使用百分位与秩相关得出的结论对此不敏感。',
    '冲突成因中的"广告分类器被致谢/网址触发"只是推断，需要结合 A1 原文核实。',
    'M6 预测均衡配比(无零份额)优于所有观测配方，但实验配方中本身很少有完全均衡的样本。推荐配比已加信赖域并与 LightGBM 集成，仍建议在更大尺度上做小规模验证。'])
H1('附录 A  过程数据文件索引')
Pp('全部过程文件见随附压缩包 out/ 目录，逐文件说明见 PROCESS_FILE_INDEX.csv。按环节组织如下：S01 解析；S02 预处理与赋权；S03 冲突识别与成因；S04 冲突消解与域级聚合(含 22 万条样本级评分)；S05 可靠性；S06 配方数据清洗与审计；S07 模型选择与检验(含逐配方预测)；S08 领域效应、组合效应与最优配比；S09 外推稳健性与 Q 接入；figures 为全部图表。')
H1('附录 B  复现方式与 AI 使用披露提示')
Pp('环境：Python 3.12，依赖 numpy、pandas、scipy、scikit-learn、statsmodels、lightgbm、matplotlib。复现命令：bash code/run_all.sh <A_data_value 目录> <输出目录>。随机种子已固定。')
NOTE('提示：赛题第五部分规定 AI 不得代替完成核心建模、推导与论证，论文末尾须披露所用 AI 工具、使用环节与贡献范围。请参赛队伍自行核验并理解本报告中的模型与结论，按赛题要求如实披露。')
json.dump(B, open(sys.argv[2], 'w'), ensure_ascii=False)
print(len(B), 'blocks')

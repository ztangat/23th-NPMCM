"""问题二报告内容(JSON 块), 数字全部读取 out/ 过程文件"""
import json, os, sys
import numpy as np, pandas as pd
R = lambda f, **k: pd.read_csv(os.path.join('out', f), **k); J = lambda f: json.load(open(os.path.join('out', f)))
B = []
H1 = lambda t: B.append({'t': 'h1', 'x': t}); H2 = lambda t: B.append({'t': 'h2', 'x': t}); Pp = lambda t: B.append({'t': 'p', 'x': t})
EQ = lambda t: B.append({'t': 'eq', 'x': t}); NOTE = lambda t: B.append({'t': 'note', 'x': t}); BL = lambda x: B.append({'t': 'bullets', 'x': x})
IMG = lambda f, c: B.append({'t': 'img', 'f': os.path.join('out/figures', f), 'cap': c})
def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ('nan', '<NA>', 'None'):
        return ''
    return str(v)
def TB(cap, df, w=None, fs=7.5, nd=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: '' if pd.isna(x) else (f'{x:.{nd}g}' if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0) else f'{x:.{nd}f}'))
    B.append({'t': 'table', 'cap': cap, 'head': [str(c) for c in d.columns], 'rows': [[clean(v) for v in r] for r in d.astype(object).values.tolist()], 'w': w, 'fs': fs})
f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'
main = J('T02/T02_main_params.json'); G = J('T06/T06_generalized_law_params.json'); QL = J('T04/T04b_final_quality_law.json'); BR = J('T05/T05_bridge_constants.json')
kc = J('T01/T01_key_checks.json'); chin = J('T03/T03_B1_vs_Chinchilla_published_params.json'); b10 = J('T03/T03_B10_generator_identification.json')
VS = R('T03/T03_validation_summary.csv').set_index('dataset'); LV = R('T04/T04b_anchor_levels_linear_Q.csv'); CM = R('T04/T04_candidate_comparison.csv')
SL = R('T05/T05_mixture_effect_slopes_by_scale.csv').set_index('set'); HQ = R('T05/T05_cross_domain_quality_channel_test.csv'); MX = R('T05/T05_mixture_Psi_and_Q.csv')
EL = R('T06/T06_elasticity_grid.csv'); SUB = R('T06/T06_quality_vs_scale_equivalence.csv'); CO = R('T06/T06_marginal_cost_effectiveness.csv'); CG = R('T06/T06_critical_quality_cost_gprime.csv')
GR = R('T07/T07_extrapolation_uncertainty_grid.csv'); B9 = J('T07/T07_B9_summary.json'); B2B = J('T07/T07_B2_family_B_transfer_all.json'); B8 = J('T04/T04_B8_diagnostics.json')
CI = R('T04/T04b_bootstrap_CI.csv'); cic = CI[CI.kind == 'cell_cluster']
cN_lo, cN_hi = cic.c_N.iloc[0], cic.c_N.iloc[2]; cD_lo, cD_hi = cic.c_D.iloc[0], cic.c_D.iloc[2]
anc = LV.iloc[0]; lomo = R('T02/T02_LOMO_cv.csv')
v = lambda k, c='MAPE_pct': VS.loc[k, c]
g1 = GR[(GR.tokens_per_param == 20)].set_index('N_B')

# ============ 摘要
H1('摘要')
Pp('本报告完成问题二：在经典标度律基础上建立同时包含参数量 N、数据量 D、数据质量 Q、领域配比 p 的广义标度律，完成参数估计、检验与验证，并据此分析各因素的边际效用、弹性，以及质量与规模之间的替代条件。经典律以 B1 为主拟合数据；Q 与 p 的信息来自问题一的输出(A1 补全版)。全部数字由随附代码计算，可逐项追溯到 out/T01–T07 下的过程文件。')
BL([f'经典律(B1，D≥20B 共 1,072 点)：E={main["E"]:.4f}，A={main["A"]:.2f}，α={main["alpha"]:.4f}，B={main["B"]:.2f}，β={main["beta"]:.4f}，拟合 MAPE 为 {main["in_sample"]["MAPE_pct"]:.4f}%。留一模型交叉验证与"D≤150B 拟合、D>150B 外推"两项检验的误差都在 10⁻⁴ 量级。这组参数与 Hoffmann 等(2022)的 Chinchilla 参数几乎完全一致：直接用 Chinchilla 原参数计算，B1 全部 1,176 点的相对误差中位数只有 {chin["median_abs_rel_err_pct"]:.4f}%。由此判断，B1 的 Loss 是按 Chinchilla 型标度律校准过的数据，其参数的统计不确定性极小，模型能否推广主要靠 B2、B4、B5 检验。',
    f'经典律的外部验证：B3 插值轨迹 MAPE {v("B3 插值轨迹(4000)"):.3f}%。B4 跨族数据直接预测 MAPE {v("B4 跨族(57) 直接预测"):.2f}%，加入族偏移后降到 {v("B4 跨族(57) + 族偏移(留一)"):.2f}%；B5 文献数据分别为 {v("B5 文献(44) 直接预测"):.2f}% 和 {v("B5 文献(44) + 族偏移(留一)"):.2f}%；两者的族内秩相关都接近 1。B2 Cerebras 半合成数据若直接外推，MAPE 高达 {v("B2 Cerebras 直接外推(Pythia 律)"):.1f}%；但保持 E、A、α、β 不变、只改数据系数 B，MAPE 就降到 {B2B["all"]["MAPE_pct"]:.2f}%。这说明族与族之间的差别主要在"数据效率"上，指数本身可以迁移。',
    f'质量通道：在 B6/B7 上比较了 8 种候选形式，要求它们在 Q=1 时都退化为经典律。结果选定"分项线性乘子"形式 L = E + A[1+c_N(1−Q)]N^−α + B[1+c_D(1−Q)]D^−β，其中 c_N={QL["c_N"]:.3f}(95%CI {cN_lo:.3f}–{cN_hi:.3f})，c_D={QL["c_D"]:.3f}(95%CI {cD_lo:.3f}–{cD_hi:.3f})，残差标准差 {QL["resid_sd"]:.3f}，与数据本身的噪声相当。c_N 显著大于 c_D，即低质量数据对"参数效率"的伤害大于对"数据效率"的伤害。"有效数据量 D·Q^κ"这一常见设想被数据明确否定(经典部分锚定时，其留一 N 交叉验证 RMSE 是选定形式的约 2 倍)。B8 与 B6/B7 的 Q 方向相反，且有 21% 的点被截断在 0.5，因此只作诊断，不参与拟合。',
    f'配比通道与跨源统一：问题一的配比律 M6 在 1M、60M、1B 三个尺度上的效应斜率依次为 {SL.loc["test_1M","slope_b"]:.3f}、{SL.loc["test_60M","slope_b"]:.3f}、{SL.loc["test_1B","slope_b"]:.3f}，说明配比的影响随规模收缩。用 b = κ(R/R_1M)^η 刻画(R 为可约损失)，得 η={G["eta"]:.3f}。跨域的混合质量 Q_mix(p) 不能解释 M6 的残差(1M、60M 上 |ρ|≤0.03)。因此把 Q 定义为"同一配比下，域内筛选带来的质量提升"，p 则通过 M6 项单独进入，避免重复计入。B 口径与 A 口径的 Q 由锚点 Q_ref={BR["Q_ref_A"]:.4f} 换算(即 Pile 自然配比的混合质量)。这一锚点有数据支撑：B7 在 Q=1 处与 B1 同 (N,D) 点的平均绝对差只有 {kc["B7_Q1_vs_B1_anchor_mean_abs_diff"]:.3f}。',
    f'广义标度律(A 口径，Q∈(0,1]，Q=1 表示完美数据)：L = E + A₁[1+h_N(1−Q)]N^−α + B₁[1+h_D(1−Q)]D^−β + κ[M6(p)−M6(p_ref)]·((L_ref−E)/R_1M)^η。参数为 A₁={G["A1"]:.2f}，B₁={G["B1"]:.2f}，h_N={G["hN"]:.4f}，h_D={G["hD"]:.4f}。当 Q=1 且 p=p_ref 时，公式严格退化为经典形式(完美数据版本)；当 Q=Q_ref 且 p=p_ref 时，逐点复现 B1。',
    f'替代与弹性：以 12B 参数、D=20N 的模型为例，把 Q 从 Q_ref 提高 0.1，等价于把参数量扩大到 {SUB[(SUB.N_B==12)&(SUB.tokens_per_param==20)&(SUB.dQ==0.1)&(SUB.Q.round(3)==round(BR["Q_ref_A"],3))].N_equiv_multiplier.iloc[0]:.2f} 倍；这个倍数随规模增大，到 1000B 时约 {SUB[(SUB.N_B==1000)&(SUB.tokens_per_param==20)&(SUB.dQ==0.1)&(SUB.Q.round(3)==round(BR["Q_ref_A"],3))].N_equiv_multiplier.iloc[0]:.2f} 倍。本文解析推出了"质量提升可由扩大 N 替代"的充要条件：ΔQ·[h_N A₁N^−α + h_D B₁D^−β] < a_Q·A₁N^−α。在数据稀缺(D/N 小)且质量缺口大时这一条件会失效，例如 Q 从 0.5 提到 1.0、D/N=5 时，有 37.5% 的规模点无法用扩大 N 来替代。按附录 B 的质量成本函数算账：在 10¹⁹ FLOPs 的低预算下，投参数与投质量各有优劣；在 10²² 和 10²⁴ 预算下，每 FLOP 投在质量上带来的 Loss 降幅是投在参数上的 10–600 倍。',
    f'百亿参数以上外推：B10 的估算 Loss 与本文经典律相差 RMSE {B9["B10_vs_B1law"]["RMSE"]:.4f}，说明 B10 同样由 Chinchilla 型律生成，不能作为独立验证。对 B9 中 115 个大模型，ΔQ=0.1 等价的参数倍数中位数为 {B9["N_equiv_mult_median"]:.2f}；在 1000B 时，推荐配比 p* 带来的 Loss 收益只有约 {abs(g1.loc[1000,"mixture_gain_pstar"]):.4f}，而质量从 Q_ref 提到 0.8 的收益约 {g1.loc[1000,"gain_Qref_to_0p8"]:.4f}，两者量级相当，但都在跨族结构不确定性(±{1.96*B9["struct_sd_cross_family"]:.2f})之内。'])

# ============ 1 数据
H1('1  数据读取、审计与可信度分层')
AU = R('T01/T01_data_audit.csv'); TB('表1-1  附件 B 数据审计(T01_data_audit.csv)', AU[['code', 'file', 'rows', 'cols', 'n_missing_cells', 'note']], [600, 2600, 600, 500, 700, 4300], 7)
BL([f'B1：8 个 Pythia 规模 × 147 个检查点，无缺失。C/6ND 的中位数为 1.0000(有 13 行偏差超过 2%)，D = 步数 × 2.1M tokens，与 B12 检查点索引中的 147/154 步一一对应(T01_B12_checkpoint_coverage.csv)。B11 的权重字节数除以名义参数量约为 4 字节/参数，佐证了 N 的数值(T01_B11_weight_bytes_check.csv)。',
    'B3：8×500 个点全部标记为插值点，与 B1 线性插值的平均差约 0.008，因此只能用作"插值轨迹"验证，不是独立数据。',
    'B4：其中的 Pythia 8 点在 (N,D) 上与 B1 近邻，但 Loss 最多相差 1.06(评测口径不同)，所以单独报告"剔除 Pythia 后"的验证结果。',
    f'B6/B7：B7 是 B6 的超集——360 个共同点的数值完全相同，另外新增 Q=0.5、0.7 两个水平共 90 点。这 90 点正好作为"只用 B6 拟合"的样本外检验。在 B7 的 Q=1 处与 B1 同 (N,D) 点对比，{kc["B7_Q1_vs_B1_anchor_n"]} 个锚点的平均绝对差为 {kc["B7_Q1_vs_B1_anchor_mean_abs_diff"]:.3f}，说明 B 口径的 Q=1 对应 Pythia(Pile)数据。',
    f'B8：在每个 (N,D) 单元内，Loss 与 Q 的秩相关中位数为 +{kc["Q_direction_median_spearman"]["B8"]:.3f}，而 B6/B7 为 {kc["Q_direction_median_spearman"]["B7"]:.3f}，方向相反。B8 有 {kc["B8_floor_share"]*100:.1f}% 的点恰好等于 0.5，低于任何物理上可能的不可约损失，属于截断值。与 B7 同坐标点的相关系数为 {kc["B7_vs_B8_same_point_corr"]:.3f}。因此 B8 不用于质量建模，只保留诊断(见 3.3 节)。',
    'B9/B10：B10 的 128 行与 B9 按名称一一对应，(N,D) 完全一致。B9 中另有 17 行 D<50B 的非 LLM 条目，已剔除；外推分析实际使用 115 行。'])
TB('表1-2  可信度分层与各数据在本问中的角色', pd.DataFrame([
    ['B1', '真实(但与 Chinchilla 律吻合到 10⁻⁴)', '经典律主拟合'], ['B2', '半合成', '族外验证(数据系数迁移)'], ['B3', '插值', '插值轨迹验证'],
    ['B4', '真实(评测口径不一)', '跨族验证(含族偏移)'], ['B5', '真实(文献)', '文献验证'], ['B6/B7', '半合成(噪声 sd≈0.05)', '质量通道主拟合 / 样本外检验'],
    ['B8', '半合成, 方向反转且截断', '仅诊断(Q=1 切片用于大 N 核对)'], ['B9', '真实元数据', '百亿以上外推对象'], ['B10', '估算(由标度律生成)', '一致性核对, 非独立验证'],
    ['B11/B12', '真实元数据', 'N、D 口径核对']], columns=['数据', '性质/可信度', '角色']), [1200, 4200, 3900], 8)

# ============ 2 经典律
H1('2  经典标度律：以 B1 为主拟合与多源验证')
H2('2.1  估计方法')
EQ('L(N,D) = E + A·N^−α + B·D^−β ；  ln L̂ = LSE(a − α ln N, b − β ln D, e)，A=eᵃ, B=eᵇ, E=eᵉ')
EQ('θ̂ = argmin Σᵢ Huber_δ( ln L̂ᵢ − ln Lᵢ ),  δ = 10⁻³ ；L-BFGS-B 多起点(400 个 + Chinchilla 初值) + Nelder-Mead 精修')
Pp('这是 Hoffmann 等(2022) Approach 3 的做法：在对数尺度上做 Huber 回归，对异常点稳健，并用 log-sum-exp 参数化保证正性。N、D 均以"个"为单位，因此系数可以直接与文献比较。多起点必不可少——早期实验中，只用 120 个随机起点时，优化曾陷入 (A≈152, B≈22156, β≈0.46) 的局部极小。这个局部解在 D≥10B 子集上的 Huber 目标值是全局解的 20 倍，RMSE 是 7 倍。')
H2('2.2  主拟合与子集敏感性')
SE = R('T02/T02_subset_sensitivity.csv'); TB('表2-1  不同 D 下限子集的拟合结果(T02_subset_sensitivity.csv)', SE[['subset', 'n', 'E', 'A', 'alpha', 'B', 'beta', 'in_RMSE', 'in_MAPE_pct']], fs=7.5)
Pp('主拟合取 D≥20B 子集，目的是剔除训练早期的预热和高学习率阶段(Pythia 在前 1% 的步数里做学习率预热)。从 D≥0 到 D≥50B 的五个子集，参数几乎不变。只用 8 个终点(每个规模一个)则无法识别 5 个参数：E 与 B 的估计严重漂移(E=0.96，B=22056)。这说明拟合必须利用训练轨迹中的中间检查点。')
TB('表2-2  留一模型交叉验证(T02_LOMO_cv.csv)', lomo[['held_out_N_B', 'extrapolation', 'RMSE', 'MAPE_pct', 'alpha', 'beta', 'E']], fs=7.5)
TX = J('T02/T02_time_extrapolation.json')
Pp(f'时间外推：只用 20≤D≤150B 的数据拟合，去预测 D>150B 的 576 个点，RMSE={TX["RMSE"]:.5f}，MAPE={TX["MAPE_pct"]:.4f}%。自助法 95% 置信区间(按模型簇抽 200 次、按 D 块抽 200 次)非常窄，例如按模型簇抽样时 α 落在 0.3399–0.3401。')
CP = R('T02/T02_compare_chinchilla.csv'); TB('表2-3  与 Chinchilla 参数对照', CP, fs=7.5)
NOTE(f'重要发现：B1 与 Chinchilla 律吻合到 10⁻⁴。直接用 Hoffmann 等(2022)的原参数(E=1.69, A=406.4, α=0.34, B=410.7, β=0.28)计算 B1 的 1,176 个点，相对误差中位数为 {chin["median_abs_rel_err_pct"]:.4f}%，最大值 {chin["max_abs_rel_err_pct"]:.3f}%。真实训练日志不可能这样与文献参数逐点吻合，因此 B1 应当是按 Chinchilla 律校准或生成的数据。这带来两点影响：(1) 本文估计出的经典参数 ≈ Chinchilla 参数，统计误差可以忽略，但它们对真实 Pythia 的代表性需要借助 B4、B5 评估；(2) 由此推出的计算最优分配与 Chinchilla Approach 3 相同，即 N*∝C^0.452、D*∝C^0.548(表2-4)。')
CA = R('T02/T02_compute_optimal_allocation.csv'); TB('表2-4  计算最优分配 C=6ND(T02_compute_optimal_allocation.csv)', CA, fs=7.5)
IMG('F01_B1_fit.png', '图1  B1 轨迹与经典律拟合、逐点残差')
H2('2.3  族外、插值、跨族与文献验证')
vs = VS.reset_index()[['dataset', 'n', 'RMSE', 'MAE', 'MAPE_pct', 'bias', 'R2', 'spearman']]
vs = pd.concat([vs, pd.DataFrame([dict(dataset='B2 Cerebras 共享 E,A,α,β, 仅族数据系数 B_f(1 参)', n=B2B['all']['n'], RMSE=B2B['all']['RMSE'], MAE=B2B['all']['MAE'], MAPE_pct=B2B['all']['MAPE_pct'],
                                             bias=B2B['all']['bias'], R2=B2B['all']['R2'], spearman=B2B['all']['spearman'])])])
TB('表2-5  经典律验证汇总(T03_validation_summary.csv 等)', vs, [3800, 500, 800, 800, 900, 800, 800, 800], 7)
BL([f'B2(族外，半合成)：直接外推误差很大(MAPE {v("B2 Cerebras 直接外推(Pythia 律)"):.1f}%)。只平移 E 仍不够(R²={VS.loc[[k for k in VS.index if k.startswith("B2 Cerebras + 族常数")][0],"R2"]:.3f})。保持 E、A、α、β 不变、只估计族数据系数 B_f，就能达到 MAPE {B2B["all"]["MAPE_pct"]:.2f}%；B_f/B = {B2B["ratio"]:.2f}，留一模型交叉验证误差与此相同(T07_B2_family_B_transfer_LOMO.csv)。族内重拟合得到的 α=0.339、β=0.279，与 B1 一致。结论：指数 α、β 可以跨族迁移，族差异集中体现在数据系数(数据效率)上。这与第 3 章质量通道"主要改变系数、不改变指数"的结构相呼应。',
    f'B4/B5(跨族与文献)：直接预测的 MAPE 约为 8%，且系统性偏高或偏低，同一族内的偏差方向一致。加入族常数偏移(留一估计)后，MAPE 降到 {v("B4 跨族(57) + 族偏移(留一)"):.2f}% 和 {v("B5 文献(44) + 族偏移(留一)"):.2f}%；族内秩相关绝大多数为 1.00(Qwen2 为 0.96，Gemma 仅 3 点为 0.5)。可见不同族之间的 Loss 数值受验证集和分词器影响，不能直接比较，但"随 N、D 单调下降"的规律是普适的。单独用 B4 或 B5 重拟合时参数不可识别(例如 B5 的 E→0)，所以这两份数据只能作验证，不能作拟合。',
    f'B3(插值轨迹)：MAPE {v("B3 插值轨迹(4000)"):.3f}%，符合预期(B3 是 B1 的插值)。'])
f4 = R('T03/T03_B4_by_family.csv'); TB('表2-6  B4 各族残差(T03_B4_by_family.csv)', f4, fs=7.5)
IMG('F02_validation.png', '图2  族外 / 跨族 / 文献验证')

# ============ 3 广义律
H1('3  广义标度律：把质量 Q 与配比 p 纳入')
H2('3.1  设计原则与可检验假设')
TB('表3-1  建模假设与检验结果', pd.DataFrame([
    ['H1 退化性', 'Q=1(且 p=p_ref)时退化为经典律', '所有候选形式按构造满足；数值核对见表3-8', '满足'],
    ['H2 质量作用于系数、不改变指数', '质量乘在 A、B 上，α、β 跨质量不变', f'B6/B7 选型：乘子形式 BIC 最优；B2 族外迁移也表现为只改 B', '支持'],
    ['H_U 跨源统一(经典部分取 B1)', 'B6/B7 的经典部分与 B1 相同', f'F 检验拒绝严格锚定(p<10⁻⁵)；但 c_N、c_D 在 4 个锚定层级下稳定(0.52–0.57 与 0.27–0.40)', '弱形式成立'],
    ['H_bridge 口径桥接', 'Q_B = Q_A / Q_ref', f'B7 在 Q=1 处与 B1 平均绝对差 {kc["B7_Q1_vs_B1_anchor_mean_abs_diff"]:.3f}(与噪声同级)', '支持(假设)'],
    ['H_Qp 跨域 Q_mix 通道', '配比改变 Q_mix 会按质量律改变 Loss', f'Q_mix 与 M6 残差：|ρ|≤0.03(1M、60M)，1B 上 ρ=0.19(p=0.14)', '否定 → Q 取域内提升'],
    ['H_mult 配比效应的尺度依赖', '配比效应 ∝ 可约损失 R^η', f'η 由 60M 反推为 {G["eta_from_60M"]:.2f}、由 1B 反推为 {G["eta_from_1B"]:.2f}；合并估计 η={G["eta"]:.2f}；严格 η=1 与"尺度不变"均被否定', '部分支持(η<1)'],
], columns=['假设', '内容', '证据', '结论']), [1700, 2500, 4000, 1100], 7.5)
H2('3.2  质量通道：8 种候选形式的比较(B6/B7)')
Pp('从数据形态出发(图3左)：以经典律为基准，超额 Loss 随 Q 近似线性下降；它随 N 增大而明显减小(Q=0.1 时，0.07B 模型约 0.49，12B 模型约 0.24)，随 D 只略有减小(10B 到 600B 从 0.37 降到 0.29)。这与"有效数据量 D·Q^κ"的预言相反——该形式要求超额 Loss 随 D 快速衰减、且与 N 无关。')
cm = CM[['model', 'classic_params', 'k', 'q_params', 'B7_RMSE', 'BIC', 'dBIC_vs_best', 'B6fit_to_B7new_RMSE', 'LONO_CV_RMSE', 'LOQO_CV_RMSE']].sort_values(['classic_params', 'BIC'])
TB('表3-2  候选形式比较(B7 拟合；B6 拟合→B7 新增 Q 水平样本外；LONO=留一 N，LOQO=留一 Q 水平)', cm, fs=6.3)
Pp('在两组设定下(经典部分锚定 B1 / 全部自由)，结论一致："分项乘子"(Q6)BIC 最优，其次是统一乘子(Q5)。有效数据(Q1)、有效参数(Q2)和幂乘子(Q4)都明显更差。Q6 中曲率参数 γ 的 95% 置信区间包含 1，因此取 γ=1 的线性乘子作为最终形式，参数更少，解释也更直接：')
EQ('L(N,D,Q) = E + A·[1 + c_N(1−Q)]·N^−α + B·[1 + c_D(1−Q)]·D^−β   (B 口径，Q=1 ⇔ B1/Pile 数据)')
lv = LV[['level', 'k', 'E', 'A', 'alpha', 'B', 'beta', 'c_N', 'c_D', 'RMSE', 'BIC', 'B6fit_B7new_RMSE', 'LONO_RMSE', 'F_vs_H_U0', 'p_vs_H_U0']]
TB('表3-3  跨源锚定层级检验(T04b_anchor_levels_linear_Q.csv)', lv, fs=6.5)
Pp(f'依题意，经典部分以 B1 为主(H_U0 全锚定)，质量参数为 c_N={QL["c_N"]:.4f}、c_D={QL["c_D"]:.4f}。放开经典部分虽然能显著减小误差(F 检验 p<10⁻⁵，说明 B6 的生成律与 B1 并不完全相同)，但质量参数只在 0.52–0.57 和 0.27–0.40 之间变动。所以"质量乘子与经典部分可分离"这一较弱的统一假设成立：乘子可以跨数据源移植。按单元簇自助 300 次，c_N−c_D 的 95% 置信区间为 [{QL["cN_minus_cD_CI95"][0]:.3f}, {QL["cN_minus_cD_CI95"][1]:.3f}]，不含 0，说明质量对参数项的放大作用确实强于对数据项。')
TB('表3-4  质量参数自助置信区间(T04b_bootstrap_CI.csv)', CI.rename(columns={'Unnamed: 1': 'quantile'}), fs=7.5)
IMG('F03_quality_law.png', '图3  质量通道：数据形态与候选形式比较')
H2('3.3  B8 的诊断与处理')
Pp(f'B8 在 Q=1 的切片上与经典律一致(138 点，MAPE {B8["Q1_vs_classic"]["MAPE_pct"]:.2f}%；校准段与外推段分别见 T07_B8_Q1_slice_vs_classic.csv)。但在 Q<1 时，Loss 随 Q 下降而下降。在每个 (N,D) 单元内，未截断点可以很好地表示为 L₈ ≈ Q·L_classic + (1−Q)·h(N,D)(RMSE 中位数 {B8["linear_repr_median_rmse"]:.3f})，其中 h 在 {B8["h_range"][0]:.2f} 到 {B8["h_range"][1]:.2f} 之间，有时为负。也就是说，B8 相当于把 Loss 向一个远低于不可约损失的值做线性插值，物理上不成立。因此 B8 不参与质量建模，只用它的 Q=1 切片核对大 N 的外推(第 5 章)。')
IMG('F04_B8_diagnostic.png', '图4  B8 的 Q 方向反转与截断')
H2('3.4  与问题一的口径桥接')
Pp(f'问题一给出了 7 个质量域与 17 个配方域的 Q(A 口径，范围 0.55–0.80)。取 Pile 自然配比(RegMix 实验配方均值 p_ref)的混合质量作为锚点，Q_ref = Σ p_ref,i·Q̂_i = {BR["Q_ref_A"]:.4f}。B 口径以 Pythia(Pile)数据为 1，于是 Q_B = Q_A/Q_ref。把 B 口径的定律改写到 A 口径(令 Q=1 表示完美数据)，形式不变：')
EQ('L = E + A₁[1+h_N(1−Q)]N^−α + B₁[1+h_D(1−Q)]D^−β ,  A₁ = A·m_N, B₁ = B·m_D, m_X = 1 + c_X(1 − 1/Q_ref), h_X = (c_X/Q_ref)/m_X')
Pp(f'代入数值：m_N={G["mN1"]:.4f}，m_D={G["mD1"]:.4f}，A₁={G["A1"]:.2f}，B₁={G["B1"]:.2f}，h_N={G["hN"]:.4f}，h_D={G["hD"]:.4f}。当 Q=Q_ref 时，A₁[1+h_N(1−Q_ref)]=A，即逐点复现 B1；当 Q=1 时，得到"完美数据"的经典律 E + A₁N^−α + B₁D^−β。这满足题目"Q=1 时退化为经典形式"的要求，且数值上与 B1 自洽。')
HQ_ = HQ[['set', 'spearman_Qmix_vs_M6resid', 'p', 'spearman_Qmix_vs_L', 'Qmix_min', 'Qmix_max', 'QB_min', 'QB_max']]
TB('表3-5  跨域质量通道检验：Q_mix(p) 与配比律残差(T05_cross_domain_quality_channel_test.csv)', HQ_, fs=7.5)
Pp('RegMix 配方的 Q_mix 换成 B 口径后在 0.82–1.18 之间。如果跨域质量差异也按 B6 的质量律起作用，1M 尺度上会产生约 0.4 的 Loss 差异，但实测中 Q_mix 与 L 及 M6 残差都几乎不相关。结论：问题一的域级 Q 差异反映的是文体差异，而不是"同一分布内的质量退化"，因此不应当通过 Q_mix 进入质量通道。本文把广义律中的 Q 定义为"固定配比下的域内质量提升指数"：')
EQ('Q(p, 筛选) = Q_ref · Σᵢ pᵢ·(Qᵢ^筛选后 / Qᵢ^原始)   —— 不筛选时 Q=Q_ref，与 p 无关；配比本身的影响由 M6 项承担')
H2('3.5  配比通道：配比效应随规模收缩')
sl = SL.reset_index()[['set', 'N_nominal', 'n', 'L_at_ref_intercept', 'slope_b', 'slope_CI_low', 'slope_CI_high', 'spearman', 'E_implied_by_H_mult']]
TB('表3-6  以问题一 M6(仅 A4/A5 训练)的预测为自变量，各尺度实测 L_avg 的回归斜率', sl, fs=7)
MV = R('T05/T05_H_mult_validation.csv'); TB('表3-7  60M/1B 上三种尺度规则的预测对比(T05_H_mult_validation.csv)', MV, fs=7)
Pp(f'配比效应的斜率从 1M 的 {SL.loc["test_1M","slope_b"]:.3f} 降到 1B 的 {SL.loc["test_1B","slope_b"]:.3f}，"尺度不变"的加性假设被否定(1B 上 MAPE 12.6%)。若严格假设配比效应与可约损失成正比(η=1)，由 60M、1B 反推出的不可约损失 E 相互矛盾(−0.74 与 1.20)。本文采用 b(R) = κ·(R/R_1M)^η，拟合得 η={G["eta"]:.3f}(两个尺度单独反推的范围为 {G["eta_from_60M"]:.2f}–{G["eta_from_1B"]:.2f}，第 5 章以此区间做不确定性传播)。η<1 意味着：配比效应的绝对量随规模缩小，但占可约损失的比例上升(1M 为 0.32，60M 为 0.44，1B 为 0.58)。换句话说，规模越大，配比越是"剩余可改进空间"中的重要部分。')
IMG('F05_mixture_scaling.png', '图5  配比效应的尺度收缩与 η 拟合')
mx = MX[['mixture', 'M6_L_1M', 'Psi', 'Q_mix_A', 'Q_B', 'entropy']]; TB('表3-8  典型配比的 M6 值、相对惩罚与混合质量(T05_mixture_Psi_and_Q.csv)', mx, fs=7.5)
H2('3.6  最终广义标度律与参数表')
EQ('L(N,D,Q,p) = E + A₁[1+h_N(1−Q)]N^−α + B₁[1+h_D(1−Q)]D^−β + κ·[M6(p) − M6(p_ref)]·((L_ref − E)/R_1M)^η')
EQ('M6(p) = Σᵢ βᵢpᵢ + Σᵢ γᵢ ln(pᵢ+ε)(问题一，ε=0.001)；L_ref = 前三项之和(p=p_ref)')
TB('表3-9  广义标度律参数(T06_generalized_law_params.json)', pd.DataFrame([
    ['E', f'{G['E']:.4f}', '不可约损失', 'B1'], ['A₁ / α', f'{G["A1"]:.3f} / {G["alpha"]:.4f}', '完美数据下的参数项', 'B1 + B6/B7 + 桥接'], ['B₁ / β', f'{G["B1"]:.3f} / {G["beta"]:.4f}', '完美数据下的数据项', 'B1 + B6/B7 + 桥接'],
    ['h_N', f'{G['hN']:.4f}', '质量对参数项的放大系数(A 口径)', 'B6/B7'], ['h_D', f'{G['hD']:.4f}', '质量对数据项的放大系数(A 口径)', 'B6/B7'],
    ['c_N / c_D (B 口径)', f'{G["c_N_Bscale"]:.4f} / {G["c_D_Bscale"]:.4f}', 'B 口径原始估计', 'B6/B7'], ['Q_ref', f'{G['Qref']:.4f}', 'Pile 自然配比的混合质量(A 口径)', '问题一'],
    ['κ', f'{G['kappa']:.4f}', '1M 检验集上的配比效应斜率', 'A6/A7 + 问题一 M6'], ['η', f'{G['eta']:.4f}', '配比效应的尺度指数', 'A8–A11'], ['R_1M', f'{G['R1M']:.4f}', '1M 尺度参照配比的可约损失', 'A6/A7'],
    ['β_i, γ_i, ε', '见 S08_M6_*.csv', '配比律 M6', '问题一']], columns=['参数', '取值', '含义', '来源']), [1800, 2200, 3500, 1800], 7.5)
TB('表3-10  退化性核对(T06_degeneracy_checks.csv)', R('T06/T06_degeneracy_checks.csv'), fs=7.5)

# ============ 4 边际效用、弹性与替代
H1('4  边际效用、弹性与替代关系')
H2('4.1  弹性')
EQ('ε_X = ∂ln L/∂ln X ；  ∂L/∂Q = −[h_N A₁N^−α + h_D B₁D^−β]  (与 Q 无关：Loss 对 Q 线性)')
el = EL[(EL.tokens_per_param == 20) & (EL.N_B.isin([0.1, 1, 10, 100, 1000]))][['N_B', 'Q', 'L', 'elas_N', 'elas_D', 'elas_Q', 'dL_dQ', 'share_reducible_N', 'share_reducible_D']]
TB('表4-1  各因素弹性(D=20N；全表见 T06_elasticity_grid.csv)', el, fs=6.8)
BL([f'三个弹性都随规模下降。以 D=20N 为例：|ε_N| 从 0.1B 时的 0.076 降到 1000B 时的 0.006，|ε_Q| 从 0.23 降到 0.026(Q=Q_ref)。|ε_Q| 始终是 |ε_N| 的 3–4 倍，说明在同样的相对变动下，Loss 对质量最敏感。',
    'Q 越高，|ε_Q| 越大(ε_Q = Q·∂L/∂Q/L，而 ∂L/∂Q 与 Q 无关)。所以在边际效用意义上，质量提升不会自动出现"越改越没用"的递减；真正的递减来自附录 B 中质量成本函数的凸性。',
    'D 项占可约损失的比例从 57% 上升到 70%(Q=Q_ref)：规模越大，剩余的可降 Loss 越集中在数据侧，这为"用质量提升数据效率"留出了空间。'])
IMG('F06_elasticity_substitution.png', '图6  弹性与"ΔQ=0.1 等价参数倍数"')
H2('4.2  质量提升与扩大规模的替代条件(解析推导)')
Pp('设 D 固定，把质量从 Q 提升 ΔQ 带来的 Loss 降幅为 G = ΔQ·(h_N A₁N^−α + h_D B₁D^−β)。若改为保持质量不变、把参数从 N 增加到 N′ 达到同样效果，需要满足 a_Q A₁(N^−α − N′^−α) = G，其中 a_Q = 1 + h_N(1−Q)。解得：')
EQ("N′/N = [1 − G/(a_Q A₁ N^−α)]^(−1/α) ；  可替代 ⇔ G < a_Q A₁N^−α ⇔ ΔQ·h_D·B₁D^−β < (a_Q − ΔQ·h_N)·A₁N^−α")
Pp('同理，用数据替代时 D′/D = [1 − G/(b_Q B₁D^−β)]^(−1/β)。这个条件有明确含义：质量提升既降低参数项也降低数据项，而扩大 N 只能压缩参数项。当"数据项相对参数项过大"(数据稀缺、D/N 小)且质量缺口 ΔQ 大时，就算 N→∞ 也追不上质量提升的效果，此时质量不可替代。临界的 tokens/param 见 T06_substitution_critical_D.csv。')
sb = SUB[(SUB.dQ == 0.1) & (SUB.Q.round(3) == round(G['Qref'], 3)) & (SUB.tokens_per_param.isin([5, 20, 100]))][['N_B', 'tokens_per_param', 'dL_gain', 'gain_over_Nterm', 'N_equiv_multiplier', 'extra_params_B', 'D_equiv_multiplier']]
TB('表4-2  "Q 从 Q_ref 提升 0.1"等价于参数/数据扩大多少倍(T06_quality_vs_scale_equivalence.csv)', sb, fs=7)
sb2 = SUB[SUB.dQ > 0.1].groupby(['Q', 'dQ', 'tokens_per_param']).agg(可替代比例=('substitutable_by_N', 'mean'), N倍数中位数=('N_equiv_multiplier', 'median')).reset_index()
TB('表4-3  大幅质量缺口下的可替代性', sb2, fs=7.5)
Pp(f'结论：在常规配比(D/N=20)、Q∈[0.5,0.9] 时，ΔQ=0.1 相当于参数扩大 1.5–2.3 倍，且模型越大这个倍数越大——大模型更值得投资质量。但当 Q=0.5 需要一次提升 0.5、且 D/N=5 时，37.5% 的规模点找不到任何有限的 N 来替代(表4-3)，此时质量提升不可替代。')
H2('4.3  "同样多花一块钱"：成本账')
Pp('按照附录 B 的成本口径 C_Q = D[g(Q) − g(Q₀)]₊，投在质量上的每 FLOP 边际收益为 −(∂L/∂Q)/(D·g′(Q))；投在参数上(D 固定，C = 6ND)为 −(∂L/∂N)/(6D)。N、D 取经典律下的计算最优分配。两者相等时对应一个临界 g′(Q)：')
EQ("g′*(Q) = 6·(∂L/∂Q)/(∂L/∂N) ；  g′(Q) < g′* 时投质量更划算")
co = CO[['C', 'N_opt_B', 'D_opt_B', 'Q', 'cost_form', 'g_prime', 'ratio_Q_over_N', 'better']]; TB('表4-4  每 FLOP 降 Loss：质量/参数之比(T06_marginal_cost_effectiveness.csv)', co, fs=6.8)
TB('表4-5  质量与参数等效时的临界 g′(Q)(FLOPs/token)', CG, fs=7.5)
Pp('低预算(10¹⁹)时，g′(Q) 与临界值 g′*≈6×10⁹ 处于同一量级：幂函数型成本下投参数更划算；指数型成本在 Q 较低时投质量更划算，Q 较高时则相反；对数渐进型成本下始终投质量更划算。到中高预算(10²²、10²⁴)，g′* 随 N 近似线性增长(1.3×10¹¹ 和 1.1×10¹²)，远高于三种成本函数的 g′(Q)，每 FLOP 投在质量上的收益是投在参数上的 10–600 倍。这正是问题三要考察的"结构性转移"：预算越高，质量越先于规模。')
IMG('F07_cost_effectiveness.png', '图7  成本账：质量 vs 参数')
H2('4.4  不同领域数据的替代与互补')
Pp('M6 是可分对数律，本身不含交叉项。领域之间的替代或互补体现为"在单纯形上以一个域的份额换另一个域"时的曲率：κᵢⱼ = −γᵢ/(pᵢ+ε)² − γⱼ/(pⱼ+ε)²。κ>0 表示互换会受到凸性惩罚，两个域互补、宜同时保留；κ≤0 表示可以互相替代。')
SG = R('T06/T06_domain_pair_curvature_substitution.csv'); sg = SG[SG.point == 'p*']
TB('表4-6  p* 处曲率最大的 8 对(最互补)与最小的 6 对(可替代)', pd.concat([sg.sort_values('curvature_1M', ascending=False).head(8), sg.sort_values('curvature_1M').head(6)])[
    ['domain_i', 'domain_j', 'p_i', 'p_j', 'curvature_1M', 'relation', 'swap_tolerance_share_at_1B', 'swap_tolerance_share_at_70B', 'swap_tolerance_share_at_1000B']], fs=6.5)
fs_ = R('T06/T06_family_interactions_scaled.csv')[['term', 'coef', 'coef_at_1B', 'coef_at_70B', 'coef_at_1000B']]
TB('表4-7  问题一领域族交互系数在不同规模下的绝对量级(×(R/R_1M)^η)', fs_, fs=7.5)
BL([f'在 {int((sg.curvature_1M>0).sum())}/{len(sg)} 个域对上 κ>0，互补是主流。其中 nih_exporter、dm_mathematics、ubuntu_irc、hackernews 这类份额小而弹性 γ 显著为负的域，与其他域最互补，"必须少量保留"。',
    'philpapers、enron_emails、europarl 三个域的 γ≥0(问题一中不显著)，与它们相关的域对 κ≤0，可以被其他域替代。',
    '规模效应：配比效应以 (R/R_1M)^η 的速度衰减。把 Loss 上升容忍度固定为 0.005，对最互补的域对，1B 时允许互换约 1% 的份额，1000B 时允许约 2%，也就是说大模型对配比的细节更不敏感。问题一中各领域族的交互系数(全部为负，即互补)到 1000B 时缩小到原来的约 1/8。'])
IMG('F09_domain_marginal_by_scale.png', '图9  各域份额的边际 Loss 效应随规模变化')

# ============ 5 外推
H1('5  百亿参数以上的外推(B9/B10)与不确定性')
Pp(f'B10 与本文经典律的 RMSE 为 {B9["B10_vs_B1law"]["RMSE"]:.5f}；用 B10 反拟合，得到 E={b10["B10_refit"]["E"]:.3f}，A={b10["B10_refit"]["A"]:.1f}，α={b10["B10_refit"]["alpha"]:.4f}，B={b10["B10_refit"]["B"]:.1f}，β={b10["B10_refit"]["beta"]:.4f}，仍然是 Chinchilla 参数。因此 B10 只能作为一致性核对，不构成独立验证。B8 的 Q=1 切片在外推段(N 为 20–700B)与经典律相比 MAPE 约 1.24%，也支持经典部分可以外推。')
gr = GR[['N_B', 'tokens_per_param', 'L_at_Qref', 'L_band_struct_low', 'L_band_struct_high', 'gain_Qref_to_0p8', 'gain_CI_low', 'gain_CI_high', 'mixture_gain_pstar', 'N_equiv_mult_dQ0p1', 'N_equiv_CI_low', 'N_equiv_CI_high']]
TB('表5-1  规模梯度上的外推与 95% 不确定性(T07_extrapolation_uncertainty_grid.csv)', gr, fs=6.2)
Pp(f'不确定性来源有四个：c_N、c_D 的单元簇自助分布；η 在 [{G["eta_from_60M"]:.2f}, {G["eta_from_1B"]:.2f}] 内均匀取值；Q_ref 取 ±3% 的扰动；跨族结构不确定性(B5 族偏移留一 RMSE {B9["struct_sd_cross_family"]:.3f}，给出 ±1.96σ 带)。前三项决定质量和配比收益的置信区间；最后一项决定 Loss 绝对水平的误差带(约 ±0.25)，它远大于质量和配比在大模型上的收益。所以在百亿参数以上，本模型对"方向和相对大小"的判断是可靠的，但对 Loss 绝对值的精度有限。')
j = R('T07/T07_B9_large_models_extrapolation.csv')
TB('表5-2  B9 大模型示例(前 15 行；全表 115 行见 T07_B9_large_models_extrapolation.csv)', j[['model_name', 'N_params_B', 'D_tokens_B', 'tokens_per_param', 'val_loss', 'L_Qref_pref', 'L_Q0p8_pref', 'L_Qref_pstar', 'N_equiv_mult_dQ0p1', 'overtrain_ratio_tokens_per_param_vs_opt', 'is_open']].head(15), fs=6.2)
BL([f'B9 中位 tokens/param 为 {B9["tokens_per_param_median"]:.1f}。按本文(Chinchilla Approach 3)的计算最优比例，只有 {B9["share_over_trained_gt1"]*100:.1f}% 的大模型"训练充分"；多数模型相对最优分配而言参数偏多、数据偏少。这类数据稀缺区恰恰是 4.2 节中质量最难被规模替代的区域。',
    f'在 B9 模型上，ΔQ=0.1 等价的参数倍数中位数为 {B9["N_equiv_mult_median"]:.2f}；Q 从 Q_ref 提到 0.8 的 Loss 收益中位数为 {B9["gain_from_Q0p8_median"]:.4f}；配比从 p_ref 调整到 p* 的收益中位数为 {B9["gain_from_pstar_median"]:.4f}。',
    '外推的主要风险：(1) 经典部分来自 Chinchilla 型校准数据，而真实大模型族之间存在约 0.1 的系统偏移；(2) 质量乘子是在 N≤12B、D≤600B 的半合成数据上估计的；(3) η 只由三个尺度确定。'])
IMG('F08_large_extrapolation.png', '图8  百亿参数以上外推')

# ============ 6 结论
H1('6  结论与向问题三传递的输出')
TB('表6-1  问题二输出接口', pd.DataFrame([
    ['广义标度律', 'L = E + A₁[1+h_N(1−Q)]N^−α + B₁[1+h_D(1−Q)]D^−β + κ[M6(p)−M6(p_ref)]((L_ref−E)/R_1M)^η', 'T06_generalized_law_params.json'],
    ['基线质量 Q₀', f'Q_ref = {G["Qref"]:.4f}(Pile 自然配比，A 口径；未做筛选时 Q₀=Q_ref)', 'T05_bridge_constants.json'],
    ['推荐配比 p*', '问题一集成 Top-100 配比；由于配比效应随规模收缩，可视作与预算无关的固定输入', 'S08_optimal_mixture.csv(问题一)'],
    ['质量–参数等效', f'ΔQ=0.1 ≈ N×1.5–2.3(D/N=20, Q∈[0.5,0.9])；可替代条件见 4.2 节', 'T06_quality_vs_scale_equivalence.csv'],
    ['成本临界', 'g′(Q) < g′* = 6(∂L/∂Q)/(∂L/∂N) 时投质量更划算；10¹⁹、10²²、10²⁴ 三档预算下的 g′* 见表4-5', 'T06_critical_quality_cost_gprime.csv'],
], columns=['输出', '内容', '文件']), [1700, 5200, 2400], 7.5)
H1('7  局限')
BL(['B1 与 B10 实际上都由 Chinchilla 型律生成，经典部分的"拟合优度"不代表真实训练的误差水平。真实误差应以 B4、B5 跨族验证中约 5% 的 MAPE(加入族偏移后)为准。',
    'B6/B7 是半合成数据，F 检验表明其经典部分与 B1 并不完全一致。本文只移植了无量纲的质量乘子；如果真实数据中质量也会改变指数 α、β，本模型会低估大规模下的质量收益。',
    '口径桥接 Q_B = Q_A/Q_ref 是一个假设。它得到 B7(Q=1)与 B1 吻合的支持，但无法直接检验。',
    '配比效应的尺度指数 η 只由 60M 和 1B 两个尺度确定，两者单独反推的值相差 0.2，外推到百亿参数以上的不确定性较大。',
    '"跨域 Q_mix 不进入质量通道"是由 RegMix 数据得出的结论；它依赖于问题一中"域→质量域"映射的准确性(inferred 映射的置信度只有 0.3–0.6)。'])
H1('附录 A  过程数据文件索引')
Pp('全部过程文件都在 out/ 目录下，按环节组织：T01 审计；T02 经典律拟合；T03 验证；T04 质量通道选型、锚定检验与 B8 诊断；T05 配比通道与口径桥接；T06 广义律、弹性、替代、成本与领域关系；T07 外推与不确定性；figures 为图表。逐文件说明见 PROCESS_FILE_INDEX.csv。')
H1('附录 B  复现与 AI 使用披露提示')
Pp('复现命令：bash code/run_all.sh(依赖 numpy、pandas、scipy、matplotlib；随机种子已固定)。数据目录中需放入 real_attachments/B_scaling_laws 和问题一的 out/ 目录。')
NOTE('提示：赛题第五部分规定，AI 不得代替完成核心建模、推导与论证，论文末尾须披露所用 AI 工具、使用环节与贡献范围。请参赛队伍自行核验并理解本报告中的模型与推导，按赛题要求如实披露。')
json.dump(B, open(sys.argv[1], 'w'), ensure_ascii=False); print(len(B))

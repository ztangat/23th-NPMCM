"""T01 附件 B 全部文件的读取、清洗与一致性审计(过程文件 → out/T01)"""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats
B = 'data/B'; O = 'out/T01'; os.makedirs(O, exist_ok=True)
P = lambda f: os.path.join(O, f)
aud = []
def rec(code, f, df, note=''):
    aud.append(dict(code=code, file=f, rows=len(df), cols=df.shape[1], n_missing_cells=int(df.isna().sum().sum()), note=note))

# B1 Pythia
p = pd.read_csv(f'{B}/pythia_training_log_existing.csv')
p['C_over_6ND'] = p.C_FLOPs_1e21 * 1e21 / (6 * p.N_params_B * 1e9 * p.D_tokens_B * 1e9)
p['D_over_steps_x_batch'] = p.D_tokens_B * 1e3 / (p.steps * p.batch_tokens_M)
p['val_minus_train'] = p.val_loss - p.train_loss
g = p.groupby('N_params_B').agg(n=('val_loss', 'size'), D_min=('D_tokens_B', 'min'), D_max=('D_tokens_B', 'max'), L_first=('val_loss', 'first'),
                                 L_last=('val_loss', 'last'), train_last=('train_loss', 'last'), gen_gap_last=('val_minus_train', 'last'),
                                 C6ND_median=('C_over_6ND', 'median'), C6ND_min=('C_over_6ND', 'min'), C6ND_max=('C_over_6ND', 'max'),
                                 n_val_increase=('val_loss', lambda v: int((v.diff() > 0).sum())), lr=('lr', 'median'))
g.to_csv(P('T01_B1_pythia_per_model_summary.csv'))
p[(p.C_over_6ND - 1).abs() > 0.02][['run_id', 'N_params_B', 'D_tokens_B', 'C_FLOPs_1e21', 'C_over_6ND']].to_csv(P('T01_B1_C_not_6ND_rows.csv'), index=False)
rec('B1', 'pythia_training_log_existing.csv', p, f'8 模型×147 检查点; C/6ND 中位数 {p.C_over_6ND.median():.4f}; |C/6ND-1|>2% 行 {int(((p.C_over_6ND-1).abs()>0.02).sum())}; 同模型 val_loss 单调不增')
# B12 检查点索引一致性
ck = pd.read_csv(f'{B}/pythia_checkpoint_index.csv')
size_map = {'70m': 0.070542, '160m': 0.162405, '410m': 0.409009, '1b': 1.040867, '1.4b': 1.416184, '2.8b': 2.782831, '6.9b': 6.86104, '12b': 11.965825}
cov = []
for s, n in size_map.items():
    st = set(ck[ck.model_size == s].step); lg = set(p[np.isclose(p.N_params_B, n)].steps)
    cov.append(dict(model_size=s, N_params_B=n, index_steps=len(st), log_steps=len(lg), log_steps_in_index=len(lg & st), index_only=len(st - lg)))
pd.DataFrame(cov).to_csv(P('T01_B12_checkpoint_coverage.csv'), index=False)
rec('B12', 'pythia_checkpoint_index.csv', ck, '9 规模×154 检查点(含 14m); 用于核对 B1 步数覆盖')
# B11 权重字节 → 参数量一致性
om = pd.read_csv(f'{B}/open_model_family_metadata.csv')
nom = {'14m': .014, '70m': .070542, '160m': .162405, '410m': .409009, '1b': 1.040867, '1.4b': 1.416184, '2.8b': 2.782831, '6.9b': 6.86104, '12b': 11.965825,
       '111M': .111, '256M': .256, '590M': .590, '1.3B': 1.3, '2.7B': 2.7, '6.7B': 6.7, '13B': 13.0, '1B': 1.18, '7B': 6.89}
om['size_tag'] = om.model_repo.str.split('-').str[-1]
om['N_B_nominal'] = om.size_tag.map(nom)
om['bytes_per_param_per_file'] = om.weight_total_bytes / (om.N_B_nominal * 1e9) / om.weight_file_count.clip(lower=1)
om['bytes_per_param_total'] = om.weight_total_bytes / (om.N_B_nominal * 1e9)
om.to_csv(P('T01_B11_weight_bytes_check.csv'), index=False)
rec('B11', 'open_model_family_metadata.csv', om, '权重字节/名义参数量, 辅助核对 N')
# B3 插值轨迹
tr = []
for f in sorted(glob.glob(f'{B}/training_trajectories/*.csv')):
    t = pd.read_csv(f); n = t.N_params_B.iloc[0]
    q = p[np.isclose(p.N_params_B, n)].sort_values('D_tokens_B')
    li = np.interp(t.D_tokens_B, q.D_tokens_B, q.val_loss)
    tr.append(dict(file=os.path.basename(f), N_params_B=n, rows=len(t), D_min=t.D_tokens_B.min(), D_max=t.D_tokens_B.max(), share_interpolated=t.interpolated.mean(),
                   max_abs_diff_vs_linear_interp_B1=np.abs(t.val_loss - li).max(), mean_abs_diff=np.abs(t.val_loss - li).mean()))
pd.DataFrame(tr).to_csv(P('T01_B3_trajectory_vs_B1.csv'), index=False)
rec('B3', 'training_trajectories/*.csv', pd.concat([pd.read_csv(f) for f in glob.glob(f'{B}/training_trajectories/*.csv')]), '8×500 插值点; 与 B1 线性插值差异见 T01_B3_trajectory_vs_B1.csv')
# B2 Cerebras
c = pd.read_csv(f'{B}/cerebras_training_log.csv')
c['C_over_6ND'] = c.C_FLOPs_1e21 * 1e21 / (6 * c.N_params_B * c.D_tokens_B * 1e18)
cg = c.groupby('N_params_B').agg(n=('val_loss', 'size'), D_min=('D_tokens_B', 'min'), D_max=('D_tokens_B', 'max'), L_first=('val_loss', 'first'),
                                  L_last=('val_loss', 'last'), n_val_increase=('val_loss', lambda v: int((v.diff() > 0).sum())), C6ND_median=('C_over_6ND', 'median'))
cg.to_csv(P('T01_B2_cerebras_per_model_summary.csv'))
rec('B2', 'cerebras_training_log.csv', c, '半合成; 7 模型×147 点; gpu_days/step_time/grad_norm 全缺失; 约 1/3 相邻点 Loss 回升(叠加噪声)')
# B4 / B5
bl = pd.read_csv(f'{B}/scaling_baseline.csv'); pu = pd.read_csv(f'{B}/published_scaling_data.csv')
near = []
for _, r in bl[bl.family == 'Pythia'].iterrows():
    q = p.iloc[((p.N_params_B - r.N_params_B).abs() / r.N_params_B + (p.D_tokens_B - r.D_tokens_B).abs() / r.D_tokens_B).argmin()]
    near.append(dict(N_B4=r.N_params_B, D_B4=r.D_tokens_B, L_B4=r.val_loss, N_B1=q.N_params_B, D_B1=q.D_tokens_B, L_B1=q.val_loss, abs_diff=abs(r.val_loss - q.val_loss)))
pd.DataFrame(near).to_csv(P('T01_B4_pythia_rows_nearest_B1.csv'), index=False)
bl.groupby('family').agg(n=('val_loss', 'size'), N_min=('N_params_B', 'min'), N_max=('N_params_B', 'max'), D_min=('D_tokens_B', 'min'), D_max=('D_tokens_B', 'max'),
                         L_min=('val_loss', 'min'), L_max=('val_loss', 'max')).to_csv(P('T01_B4_family_summary.csv'))
pu.groupby(['family', 'source']).agg(n=('val_loss', 'size'), N_min=('N_params_B', 'min'), N_max=('N_params_B', 'max'), D_min=('D_tokens_B', 'min'),
                                     D_max=('D_tokens_B', 'max'), L_min=('val_loss', 'min'), L_max=('val_loss', 'max')).to_csv(P('T01_B5_family_summary.csv'))
rec('B4', 'scaling_baseline.csv', bl, f'12 族 57 点; 其中 Pythia 8 点在 (N,D) 上与 B1 近邻, 但 Loss 与 B1 相差最大 {pd.DataFrame(near).abs_diff.max():.3f}(评测口径不同), 作为半独立校验并单独报告')
rec('B5', 'published_scaling_data.csv', pu, '11 族 44 点, 文献汇总')
# B6/B7/B8
b6 = pd.read_csv(f'{B}/supplementary_NQ_experiment.csv'); b7 = pd.read_csv(f'{B}/supplementary_NQ_experiment_expanded.csv'); b8 = pd.read_csv(f'{B}/supplementary_NQ_experiment_large.csv')
m67 = b6.merge(b7, on='experiment_id', suffixes=('_6', '_7'))
rec('B6', 'supplementary_NQ_experiment.csv', b6, '9N×5D×8Q 全网格 360 点, 半合成')
rec('B7', 'supplementary_NQ_experiment_expanded.csv', b7, f'B6 的超集(360 点完全相同 {int((abs(m67.val_loss_6-m67.val_loss_7)<1e-12).sum())}/360) + Q=0.5,0.7 共 90 点')
fl = (b8.val_loss <= 0.5 + 1e-9)
rec('B8', 'supplementary_NQ_experiment_large.csv', b8, f'15N×10D×12Q 1704 点; {int(fl.sum())} 点({fl.mean()*100:.1f}%) 恰为 0.5 下限截断; Loss 随 Q 上升(与 B6/B7 反向); 720 点为 extrapolated')
m78 = b7.merge(b8, on=['N_params_B', 'D_tokens_B', 'Q_score'], suffixes=('_B7', '_B8'))
m78.to_csv(P('T01_B7_vs_B8_same_grid_points.csv'), index=False)
dirs = []
for nm, d in [('B6', b6), ('B7', b7), ('B8', b8)]:
    for (n, dd), gg in d.groupby(['N_params_B', 'D_tokens_B']):
        dirs.append(dict(set=nm, N=n, D=dd, spearman_Q_vs_loss=stats.spearmanr(gg.Q_score, gg.val_loss).statistic if gg.Q_score.nunique() > 2 else np.nan,
                         loss_Qmin=gg.loc[gg.Q_score.idxmin(), 'val_loss'], loss_Qmax=gg.loc[gg.Q_score.idxmax(), 'val_loss']))
DR = pd.DataFrame(dirs); DR.to_csv(P('T01_B678_Q_direction_by_cell.csv'), index=False)
DR.groupby('set').spearman_Q_vs_loss.describe().to_csv(P('T01_B678_Q_direction_summary.csv'))
# B6 Q=1 与 B1 的对应(锚定检验)
anc = []
for _, r in b7[b7.Q_score == 1.0].iterrows():
    q = p[np.isclose(p.N_params_B, r.N_params_B, rtol=0.02)]
    if len(q) and r.D_tokens_B <= q.D_tokens_B.max() + 1:
        li = np.interp(r.D_tokens_B, q.D_tokens_B, q.val_loss); anc.append(dict(N=r.N_params_B, D=r.D_tokens_B, L_B7_Q1=r.val_loss, L_B1_interp=li, diff=r.val_loss - li))
AN = pd.DataFrame(anc); AN.to_csv(P('T01_B7_Q1_vs_B1_anchor.csv'), index=False)
# B9/B10
lm = pd.read_csv(f'{B}/supplementary_large_models.csv'); lb = pd.read_csv(f'{B}/supplementary_large_baseline.csv')
lm['model_name'] = lm.model_name.str.strip(); lb['family'] = lb.family.str.strip()
j = lm.merge(lb, left_on='model_name', right_on='family', how='left', suffixes=('', '_B10'))
j['FLOPs_over_6ND'] = j.FLOPs / (6 * j.N_params_B * j.D_tokens_B * 1e18)
j['tokens_per_param'] = j.D_tokens_B / j.N_params_B
j['is_open'] = j.accessibility.fillna('').str.startswith('Open weights')
j['usable_LLM_row'] = (j.D_tokens_B >= 50) & j.val_loss.notna()
j.to_csv(P('T01_B9_B10_joined.csv'), index=False)
rec('B9', 'supplementary_large_models.csv', lm, f'132 行; D=0 或极小的非 LLM 行 {int((lm.D_tokens_B<50).sum())}; FLOPs 缺失 {int(lm.FLOPs.isna().sum())}')
rec('B10', 'supplementary_large_baseline.csv', lb, '128 行估算 Loss(非观测), 与 B9 按名称一一对应(B9 另有 4 行无估算)')
AUD = pd.DataFrame(aud); AUD.to_csv(P('T01_data_audit.csv'), index=False)
json.dump({'B7_Q1_vs_B1_anchor_mean_abs_diff': float(AN['diff'].abs().mean()), 'B7_Q1_vs_B1_anchor_max_abs_diff': float(AN['diff'].abs().max()),
           'B7_Q1_vs_B1_anchor_n': int(len(AN)), 'B8_floor_share': float(fl.mean()),
           'B7_vs_B8_same_point_corr': float(np.corrcoef(m78.val_loss_B7, m78.val_loss_B8)[0, 1]), 'B7_vs_B8_n': int(len(m78)),
           'Q_direction_median_spearman': DR.groupby('set').spearman_Q_vs_loss.median().to_dict()}, open(P('T01_key_checks.json'), 'w'), indent=1, ensure_ascii=False)
print(AUD.to_string()); print(open(P('T01_key_checks.json')).read()); print(AN.round(4).to_string())

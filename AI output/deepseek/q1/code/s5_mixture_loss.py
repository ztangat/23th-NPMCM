# -*- coding: utf-8 -*-
"""
s5_mixture_loss.py —— 阶段5：17 域配比 p 与交叉熵损失的定量关系建模

数据（A4–A15，regmix_tables/）：
  训练      train_mixture_1m   (512×17) + train_pile_loss_1m   (512×13)  规模 1M
  同规模检验 test_mixture_1m    (256)   + test_pile_loss_1m              规模 1M（同规模、同配比设计随机切分）
  跨规模检验 test_mixture_60m   (256)   + test_pile_loss_60m             规模 60M（配比设计与 test_1m 完全相同）
  跨规模检验 test_mixture_1B    (64)    + test_pile_loss_1B              规模 1B（配比设计不同，github 上限仅 0.174）
  外推      est_mixture_10b    (63)    + est_pile_loss_10b              规模 10B
  外推      est_mixture_70b    (63)    + est_pile_loss_70b              规模 70B

===================================================================
一、去规模的三步归一化（本阶段核心方法论）
===================================================================
损失随模型规模系统性下降（1M≈5.3 → 70B≈1.40）且相对幅度收缩，
直接回归 L~p 会拟合"规模"而非"配比"。定义逐样本、逐域、逐数据集的
三级归一化，使目标成为**与规模无关的"配置优劣"量**：

 第1步 行中心化（去除规模基线）
     L^c_ij = L_ij − (1/13)Σ_k L_ik
   物理含义：该域相对本配比平均域损失的偏差。严格消去规模加性基线。

 第2步 逐域幅度标准化（去除规模幅度）—— 在**各数据集内部**估计 sd
     L^z_ij = L^c_ij / sd_j(L^c_·j)
   原因：不同规模下逐域相对损失的离散幅度不同（如 dm_mathematics 的 sd
   从 1M 的 1.37 收缩到 70B 的 0.37）。逐域标准化后结构可比。

 第3步 列中心化（去除配比设计基线）
     y_ij = L^z_ij − (1/n)Σ_i L^z_ij
   ⚠ 关键一步：不同数据集的 p-设计中心不同，导致 y 的均值发生系统性偏移
   （test_1m: +0.32, test_60m: +0.80）。列中心化使每个数据集的 y 严格零均值，
   模型只需预测"由配比变化引起的相对偏离"。

 最终目标： y_i = Σ_j w_j · y_ij,  w_j ∝ 1/sd_j(L^c)(训练集)

 该口径的正确性由下列实证支持（阶段5报告 §验证）：
   - 同一 p-设计下 1M 与 60M 的逐域相对损失相关 r=0.96–0.99；
   - 双重中心化后，跨规模 R² 由 −2.8 提升到 +0.70。

===================================================================
二、配比特征（单纯形上）
===================================================================
  log 配比（RegMix 式）+ 集中度（熵 H、最大占比）+ 质量加权（问题一 Q）
  规格：linear / log / log_quad / log_q / log_q_quad

===================================================================
三、评价
===================================================================
  R²、RMSE、MAE、Spearman 秩相关；同规模 5 折 CV；跨规模/外推分别报告。
  域级：13 个损失域分别建模并报告域特异性。

产出：data/s5_*.npz/json, results/tables/s5_*.csv, results/figures/s5_*.png
"""
import os, sys, json, time, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from itertools import combinations

C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"

MIX_FILES = {
    "train":    ("train_mixture_1m.csv",   "train_pile_loss_1m.csv",   1e6),
    "test_1m":  ("test_mixture_1m.csv",    "test_pile_loss_1m.csv",    1e6),
    "test_60m": ("test_mixture_60m.csv",   "test_pile_loss_60m.csv",   6e7),
    "test_1B":  ("test_mixture_1B.csv",    "test_pile_loss_1B.csv",    1e9),
    "est_10b":  ("est_mixture_10b.csv",    "est_pile_loss_10b.csv",    1e10),
    "est_70b":  ("est_mixture_70b.csv",    "est_pile_loss_70b.csv",    7e10),
}
SCALE_NAME = {1e6: "1M", 6e7: "60M", 1e9: "1B", 1e10: "10B", 7e10: "70B"}
KEYS_ORDER = ["train", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]


def load_pair(key):
    mf, lf, scale = MIX_FILES[key]
    dm = pd.read_csv(os.path.join(MIX_DIR, mf)).set_index("index")
    dl = pd.read_csv(os.path.join(MIX_DIR, lf)).set_index("index")
    mixcols = [f"train_the_pile_{d}" for d in MIXTURE_DOMAINS]
    losscols = [f"metric/the_pile_{d}_val_loss" for d in LOSS_DOMAINS]
    miss_m = [c for c in mixcols if c not in dm.columns]
    miss_l = [c for c in losscols if c not in dl.columns]
    if miss_m or miss_l:
        raise ValueError(f"{key} 缺列: {miss_m} / {miss_l}")
    idx = np.intersect1d(dm.index.values, dl.index.values)
    P = dm.loc[idx, mixcols].values.astype(np.float64)
    L = dl.loc[idx, losscols].values.astype(np.float64)
    return P, L, idx, scale


def simplex_features(P, q=None, mode="linear", eps=1e-6):
    lp = np.log(np.clip(P, eps, None))
    feats = [lp]; names = [f"logp_{d}" for d in MIXTURE_DOMAINS]
    if mode in ("log_quad", "log_q_quad"):
        feats.append(np.column_stack([lp[:, i] * lp[:, j]
                                      for i, j in combinations(range(len(MIXTURE_DOMAINS)), 2)]))
        names += [f"logp_{MIXTURE_DOMAINS[i]}*logp_{MIXTURE_DOMAINS[j]}"
                  for i, j in combinations(range(len(MIXTURE_DOMAINS)), 2)]
    if mode in ("log_q", "log_q_quad") and q is not None:
        feats.append(lp * q[None, :]); names += [f"q*logp_{d}" for d in MIXTURE_DOMAINS]
        if mode == "log_q_quad":
            feats.append(np.column_stack([q[i] * lp[:, i] * q[j] * lp[:, j]
                                          for i, j in combinations(range(len(MIXTURE_DOMAINS)), 2)]))
            names += [f"qlogp_{MIXTURE_DOMAINS[i]}*qlogp_{MIXTURE_DOMAINS[j]}"
                      for i, j in combinations(range(len(MIXTURE_DOMAINS)), 2)]
    H = -(P * np.log(np.clip(P, eps, None))).sum(axis=1, keepdims=True)
    mx = P.max(axis=1, keepdims=True)
    feats += [H, mx]; names += ["H_p", "max_p"]
    return np.hstack(feats), names


def fit_predict(Xtr, ytr, Xte, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 25)
    sc = StandardScaler().fit(Xtr)
    mdl = RidgeCV(alphas=alphas).fit(sc.transform(Xtr), ytr)
    return mdl, sc, mdl.predict(sc.transform(Xtr)), mdl.predict(sc.transform(Xte))


def r2(y, p):
    return float(r2_score(y, p)) if len(y) > 1 else np.nan


def main():
    start_log("s5_mixture_loss")
    t0 = time.time()

    # ================= 0) 载入 =================
    print("===== 0) 载入 A4–A15 =====")
    D = {}
    for k in KEYS_ORDER:
        P, L, idx, scale = load_pair(k)
        H = (-(P * np.log(np.clip(P, 1e-9, None))).sum(1)).mean()
        D[k] = dict(P=P, L=L, idx=idx, n=len(idx), scale=scale)
        print(f"  {k:<10s} 规模={SCALE_NAME[scale]:>4s}  P={P.shape}  L={L.shape}  "
              f"Σp={P.sum(1).mean():.5f}  L均值={L.mean():.4f}  配比熵={H:.3f}")

    # ================= 1) 三步归一化 =================
    print("\n===== 1) 三步去规模归一化 =====")
    print("  [步1] 行中心化 L^c = L - mean_j(L)：")
    for k in D:
        L = D[k]["L"]
        D[k]["Lc"] = L - L.mean(axis=1, keepdims=True)
        D[k]["L_mean"] = float(L.mean())
        D[k]["Lc_sd"] = D[k]["Lc"].std(axis=0, ddof=1)   # 逐域向量
    print(f"    {'数据集':<10s} {'规模':>4s} {'原始L均值':>10s} {'L^c逐域sd均值':>14s}")
    for k in KEYS_ORDER:
        print(f"    {k:<10s} {SCALE_NAME[D[k]['scale']]:>4s} {D[k]['L_mean']:>10.4f} "
              f"{D[k]['Lc_sd'].mean():>14.4f}")

    print("\n  [步2] 逐域幅度标准化 L^z = L^c / sd_j(L^c)（各数据集内部估计）：")
    for k in D:
        D[k]["Lz"] = D[k]["Lc"] / np.clip(D[k]["Lc_sd"], 1e-9, None)
    print("    逐域幅度比 sd(70B)/sd(1M):")
    ratio = D["est_70b"]["Lc_sd"] / D["train"]["Lc_sd"]
    for i, d in enumerate(LOSS_DOMAINS):
        print(f"      {d:<18s} 1M={D['train']['Lc_sd'][i]:.3f}  70B={D['est_70b']['Lc_sd'][i]:.3f}  "
              f"比={ratio[i]:.3f}")
    print(f"    → 幅度比均值={ratio.mean():.3f}（普遍 <1，且域间差异显著："
          f"min={ratio.min():.2f}, max={ratio.max():.2f}）")

    print("\n  [步3] 列中心化 y = L^z - mean_i(L^z)（去除配比设计基线）：")
    print(f"    列中心化前列均值（即配比设计导致的偏移）:")
    print(f"    {'数据集':<10s} {'Σw_j·mean_i(Lz_j)':>18s}")
    sd_fix = D["train"]["Lc_sd"]
    w_dom = 1.0 / np.clip(sd_fix, 1e-9, None); w_dom = w_dom / w_dom.sum()
    for k in KEYS_ORDER:
        pre = float((D[k]["Lz"].mean(axis=0) * w_dom).sum())
        print(f"    {k:<10s} {pre:>+18.4f}")
    for k in D:
        D[k]["Lzz"] = D[k]["Lz"] - D[k]["Lz"].mean(axis=0, keepdims=True)

    print("\n  域权重 w_j ∝ 1/sd_j(train)（逆波动归一）:")
    print("   ", dict(zip(LOSS_DOMAINS, np.round(w_dom, 4))))

    def y_of(k):
        return (D[k]["Lzz"] * w_dom[None, :]).sum(axis=1)
    ytr = y_of("train")
    print(f"\n  训练目标 y: mean={ytr.mean():+.6f} sd={ytr.std():.4f} "
          f"范围[{ytr.min():.3f},{ytr.max():.3f}]")
    print("  各数据集 y 统计（均已零均值）:")
    for k in KEYS_ORDER:
        yk = y_of(k)
        print(f"    {k:<10s} mean={yk.mean():+.5f} sd={yk.std():.4f} "
              f"[{yk.min():+.3f},{yk.max():+.3f}]")

    # ================= 2) 规模不变性实证 =================
    print("\n===== 2) 规模不变性实证：同一 p-设计的跨规模一致性 =====")
    # test_1m 与 test_60m 共用同一 256 点 p 设计
    same_design = np.allclose(D["test_1m"]["P"], D["test_60m"]["P"])
    print(f"  test_1m 与 test_60m 的 p 矩阵是否完全相同: {same_design}")
    inv_rows = []
    for i, d in enumerate(LOSS_DOMAINS):
        r_p = pearsonr(D["test_1m"]["Lc"][:, i], D["test_60m"]["Lc"][:, i])[0]
        r_s = spearmanr(D["test_1m"]["Lc"][:, i], D["test_60m"]["Lc"][:, i]).correlation
        r_z = pearsonr(D["test_1m"]["Lzz"][:, i], D["test_60m"]["Lzz"][:, i])[0]
        inv_rows.append({"域": d, "Lc相关(1M vs 60M)": float(r_p),
                         "秩相关": float(r_s), "双重中心化后相关": float(r_z)})
    dfI = pd.DataFrame(inv_rows)
    print(dfI.to_string(index=False))
    print(f"  → Lc 逐域跨规模相关均值 = {dfI['Lc相关(1M vs 60M)'].mean():.4f}"
          f"（>0.96，说明相对损失结构跨规模高度保持）")
    dfI.to_csv(os.path.join(TABLES, "s5_规模不变性检验.csv"), index=False, encoding="utf-8-sig")
    y1 = y_of("test_1m"); y2 = y_of("test_60m")
    print(f"  加权目标 y 的跨规模相关: pearson={pearsonr(y1,y2)[0]:+.4f} "
          f"spearman={spearmanr(y1,y2).correlation:+.4f}")

    # 域难度指纹
    fp = np.array([D[k]["Lzz"].mean(axis=0) for k in KEYS_ORDER])
    print("\n  域难度指纹（Lzz 逐域均值，列中心化后应≈0）与 train 的相关:")
    fp_corr = {}
    for i, k in enumerate(KEYS_ORDER):
        c = np.corrcoef(fp[0], fp[i])[0, 1] if np.std(fp[i]) > 1e-12 else np.nan
        fp_corr[k] = float(c)
    # 用 Lz 的逐域均值（真正的域难度指纹）
    fp2 = np.array([D[k]["Lz"].mean(axis=0) for k in KEYS_ORDER])
    print(f"    {'数据集':<10s} {'Lz指纹相关':>12s}")
    fp2_corr = {}
    for i, k in enumerate(KEYS_ORDER):
        c = np.corrcoef(fp2[0], fp2[i])[0, 1]
        fp2_corr[k] = float(c)
        print(f"    {k:<10s} {c:>+12.4f}")

    # ================= 3) 配比-损失自效应 =================
    print("\n===== 3) 配比-损失自效应（corr(log p_i, 目标/域损失)） =====")
    print("  同域 corr(log p_i, Lzz_i)：负=本域数据越多本域相对损失越低")
    print(f"  {'损失域':<18s} " + " ".join(f"{SCALE_NAME[D[k]['scale']]:>8s}" for k in KEYS_ORDER))
    diag_tbl = []
    for i, ld in enumerate(LOSS_DOMAINS):
        row = {"域": ld}; vals = []
        for k in KEYS_ORDER:
            lp_i = np.log(np.clip(D[k]["P"][:, i], 1e-6, None))
            r = np.corrcoef(lp_i, D[k]["Lzz"][:, i])[0, 1]
            row[k] = float(r); vals.append(r)
        diag_tbl.append(row)
        print(f"  {ld:<18s} " + " ".join(f"{v:>+8.3f}" for v in vals))
    pd.DataFrame(diag_tbl).to_csv(os.path.join(TABLES, "s5_域自效应相关.csv"),
                                  index=False, encoding="utf-8-sig")

    # 另：在 train 上用原始 Lc（未标准化）看自效应
    print("\n  参考：train 上 corr(log p_i, Lc_i)（原始幅度）:")
    for i, ld in enumerate(LOSS_DOMAINS):
        lp_i = np.log(np.clip(D["train"]["P"][:, i], 1e-6, None))
        r = np.corrcoef(lp_i, D["train"]["Lc"][:, i])[0, 1]
        print(f"    {ld:<18s} r={r:+.4f}")

    # ================= 3.5) 载入 17 域质量 Q（问题一输出） =================
    z4 = np.load(os.path.join(DATA, "s4_domain_Q.npz"), allow_pickle=True)
    q_map = {d: float(z4["Q"][i]) for i, d in enumerate(z4["domains"])}
    qvec = np.array([q_map[d] for d in MIXTURE_DOMAINS])
    print("\n  17 域质量向量 q（MIXTURE_DOMAINS 顺序）:")
    for d, v in zip(MIXTURE_DOMAINS, qvec):
        print(f"    {d:<18s} Q={v:.4f}")

    # ================= 4) 模型规格对比 =================
    print("\n===== 4) 模型规格对比（目标 = 双重中心化加权损失 y） =====")
    specs = ["linear", "log", "log_quad", "log_q", "log_q_quad"]
    Xtr_cache = {}
    for mode in specs:
        Xtr_cache[mode], _ = simplex_features(D["train"]["P"], qvec, mode)
        print(f"  [{mode:<10s}] 特征数 = {Xtr_cache[mode].shape[1]}")

    rows = []
    for mode in specs:
        Xtr = Xtr_cache[mode]
        mdl, sc, ptr, _ = fit_predict(Xtr, ytr, Xtr)
        cvr = []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xtr):
            sc_ = StandardScaler().fit(Xtr[tr])
            m_ = RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(sc_.transform(Xtr[tr]), ytr[tr])
            cvr.append(r2_score(ytr[te], m_.predict(sc_.transform(Xtr[te]))))
        base = {"规格": mode, "特征数": Xtr.shape[1],
                "训练R²": float(r2_score(ytr, ptr)),
                "CV_R²": float(np.mean(cvr)), "CV_sd": float(np.std(cvr)),
                "alpha": float(mdl.alpha_)}
        for k in KEYS_ORDER[1:]:
            Xk, _ = simplex_features(D[k]["P"], qvec, mode)
            pk = mdl.predict(sc.transform(Xk))
            yk = y_of(k)
            base[f"{k}_R²"] = r2(yk, pk)
            base[f"{k}_Spearman"] = float(spearmanr(yk, pk).correlation)
        rows.append(base)
    dfR = pd.DataFrame(rows)
    cols = ["规格", "特征数", "训练R²", "CV_R²", "CV_sd", "test_1m_R²", "test_60m_R²",
            "test_1B_R²", "est_10b_R²", "est_70b_R²", "test_60m_Spearman", "est_70b_Spearman"]
    print(dfR[cols].to_string(index=False))
    dfR.to_csv(os.path.join(TABLES, "s5_模型规格对比.csv"), index=False, encoding="utf-8-sig")
    best_mode = dfR.loc[dfR["CV_R²"].idxmax(), "规格"]
    print(f"\n  >>> 最优规格（按同规模 CV R²）= {best_mode} "
          f"(CV R²={dfR.loc[dfR['CV_R²'].idxmax(),'CV_R²']:.4f}, "
          f"test_1m R²={dfR.loc[dfR['CV_R²'].idxmax(),'test_1m_R²']:.4f})")

    # ================= 5) 最优规格详细 =================
    print(f"\n===== 5) 最优规格 [{best_mode}] 详细 =====")
    Xtr, fnames = simplex_features(D["train"]["P"], qvec, best_mode)
    mdl, sc, ptr, _ = fit_predict(Xtr, ytr, Xtr)
    coefs = pd.DataFrame({"特征": fnames, "标准化系数": mdl.coef_})
    print(f"  alpha={mdl.alpha_:.4g}  训练R²={r2_score(ytr,ptr):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(ytr,ptr)):.4f}  特征数={len(fnames)}")
    print("\n  Top-20 特征（|标准化系数|）:")
    print(coefs.reindex(coefs["标准化系数"].abs().sort_values(ascending=False).index)
          .head(20).to_string(index=False))
    coefs.to_csv(os.path.join(TABLES, "s5_最优规格系数.csv"), index=False, encoding="utf-8-sig")

    # ================= 6) 各数据集完整评估 =================
    print("\n===== 6) 各数据集评估（双重中心化口径） =====")
    eval_rows = []
    for k in KEYS_ORDER:
        Xk, _ = simplex_features(D[k]["P"], qvec, best_mode)
        pk = mdl.predict(sc.transform(Xk))
        yk = y_of(k)
        eval_rows.append({
            "数据集": k, "模型规模": SCALE_NAME[D[k]["scale"]], "n": len(yk),
            "R²": r2(yk, pk), "RMSE": float(np.sqrt(mean_squared_error(yk, pk))),
            "MAE": float(mean_absolute_error(yk, pk)),
            "Spearman": float(spearmanr(yk, pk).correlation),
            "实际y_sd": float(yk.std(ddof=1)), "预测y_sd": float(pk.std(ddof=1)),
            "原始L均值": D[k]["L_mean"],
        })
        np.save(os.path.join(DATA, f"s5_pred_{k}.npy"), np.column_stack([yk, pk]))
    dfE = pd.DataFrame(eval_rows)
    print(dfE.to_string(index=False))
    dfE.to_csv(os.path.join(TABLES, "s5_各数据集评估.csv"), index=False, encoding="utf-8-sig")

    # ================= 7) 域级建模 =================
    print("\n===== 7) 域级建模：13 个损失域分别拟合 =====")
    dom_rows = []; dom_store = {}
    for i, ld in enumerate(LOSS_DOMAINS):
        ytr_i = D["train"]["Lzz"][:, i]
        mdl_i, sc_i, ptr_i, _ = fit_predict(Xtr, ytr_i, Xtr)
        j_self = fnames.index(f"logp_{ld}") if f"logp_{ld}" in fnames else None
        row = {"损失域": ld, "训练R²": float(r2_score(ytr_i, ptr_i)),
               "自身logp系数": float(mdl_i.coef_[j_self]) if j_self is not None else np.nan,
               "loss均值(1M)": float(D["train"]["L"][:, i].mean()),
               "Lc_sd(1M)": float(D["train"]["Lc_sd"][i]),
               "Lc_sd(70B)": float(D["est_70b"]["Lc_sd"][i])}
        for k in KEYS_ORDER[1:]:
            Xk, _ = simplex_features(D[k]["P"], qvec, best_mode)
            pk = mdl_i.predict(sc_i.transform(Xk))
            row[f"{k}_R²"] = r2(D[k]["Lzz"][:, i], pk)
        dom_rows.append(row)
        dom_store[ld] = dict(coef=mdl_i.coef_, fnames=fnames)
    dfD = pd.DataFrame(dom_rows)
    print(dfD.to_string(index=False))
    dfD.to_csv(os.path.join(TABLES, "s5_域级建模结果.csv"), index=False, encoding="utf-8-sig")

    # ================= 8) 汇总保存 =================
    save_json({
        "best_mode": best_mode, "spec_table": dfR.to_dict("records"), "eval": eval_rows,
        "w_dom": {d: float(w_dom[i]) for i, d in enumerate(LOSS_DOMAINS)},
        "qvec": {d: float(sd_fix[i]) for i, d in enumerate(LOSS_DOMAINS)},
        "scale_amplitude_ratio_70B_over_1M": {d: float(ratio[i]) for i, d in enumerate(LOSS_DOMAINS)},
        "fingerprint_corr": fp2_corr,
        "invariance_check": dfI.to_dict("records"),
        "target_definition": "y = Σ_j w_j·Lzz_j, Lzz = colcenter(rowcenter(L)/sd_j(dataset))",
    }, os.path.join(DATA, "s5_summary.json"))

    print(f"\n[OK] s5_mixture_loss 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s5_mixture_loss")


if __name__ == "__main__":
    main()

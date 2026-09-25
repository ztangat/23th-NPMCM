# -*- coding: utf-8 -*-
"""
s2_quality.py —— 阶段2：综合评价模型与 Q 评分

步骤：
  S2.1  相关性/冗余诊断（Pearson + Spearman + 聚类）→ 指标筛选
  S2.2  四种赋权方案：
        (a) 等权 EW
        (b) 熵权法 EWM
        (c) CRITIC 法
        (d) 因子分析（PCA）第一主成分 / 方差贡献加权
  S2.3  一致性检验（方案间 Spearman 秩相关、Kendall W 协同系数）
  S2.4  组合赋权（CRITIC+EWM 几何平均，并与 PCA 秩融合）
  S2.5  样本级 Q 计算（含 bootstrap 不确定性）
  S2.6  语料级/领域级聚合：加权均值、加权中位数、Q 的下尾分位（风险口径）
  S2.7  抽样集 vs 扩展集（arxiv/github）对照

产出：data/s2_*.npz/json, results/tables/s2_*.csv, results/figures/s2_*.png
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# 统一配色（浅色主题）
C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"

def load(tag):
    z = np.load(os.path.join(DATA, f"s1_feat_{tag}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}

def entropy_weight(X, eps=1e-12):
    """熵权法：X 为已归一化到 [0,1] 的矩阵 (n,k)"""
    n, k = X.shape
    P = X / (X.sum(axis=0, keepdims=True) + eps)
    E = -(P * np.log(P + eps)).sum(axis=0) / math.log(n)
    d = 1.0 - E
    d = np.clip(d, 0, None)
    if d.sum() <= eps:
        return np.ones(k) / k
    return d / d.sum()

def critic_weight(X, eps=1e-12):
    """CRITIC 法：对比强度(标准差) × 冲突性(1-相关系数)"""
    n, k = X.shape
    sd = X.std(axis=0, ddof=1)
    try:
        R = np.corrcoef(X, rowvar=False)
    except Exception:
        R = np.eye(k)
    R = np.nan_to_num(R, nan=0.0)
    conflict = (1.0 - R).sum(axis=1)
    C = sd * conflict
    C = np.clip(C, 0, None)
    if C.sum() <= eps:
        return np.ones(k) / k
    return C / C.sum()

def pca_weight(X):
    """PCA：按各主成分的方差贡献率加权组合，取累计贡献 >= 0.85 的成分。
    使用 SVD 而非 eigh，避免近奇异协方差矩阵导致的特征值不收敛问题。"""
    Xs = X - X.mean(axis=0)
    sd = Xs.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    Xz = Xs / sd
    n = Xz.shape[0]
    # SVD: Xz = U S Vt, 主成分方向为 Vt 的行；方差 = S^2/(n-1)
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    w = (S ** 2) / max(n - 1, 1)
    V = Vt.T                      # (k, k) 列为载荷向量
    w = np.clip(w, 0, None)
    evr = w / w.sum()
    cum = np.cumsum(evr)
    m = int(np.searchsorted(cum, 0.85) + 1)
    m = max(1, min(m, X.shape[1]))
    score = Xz @ V[:, :m]
    # 方向对齐：确保与等权得分正相关
    ew = Xz.mean(axis=1)
    for j in range(m):
        if np.corrcoef(score[:, j], ew)[0, 1] < 0:
            score[:, j] = -score[:, j]
    return score, V[:, :m], w[:m], evr[:m], m

def topc_score(X):
    """PCA 第一主成分作为综合得分（标准化后映射到 [0,1]）"""
    S, V, w, evr, m = pca_weight(X)
    s = S[:, 0]
    z = (s - s.mean()) / (s.std(ddof=1) + 1e-12)
    return z, V, w, evr, m

def main():
    start_log("s2_quality")
    t0 = time.time()
    d = load("sample"); dA = load("arxiv"); dG = load("github")
    T = d["T"].astype(np.float64); DOM = d["DOM"]; ID = d["ID"]
    TA = dA["T"].astype(np.float64); DOMA = dA["DOM"]
    TG = dG["T"].astype(np.float64); DOMG = dG["DOM"]
    print(f"[数据] sample={T.shape} arxiv={TA.shape} github={TG.shape}")
    print(f"[域分布 sample] {dict(zip(*np.unique(DOM, return_counts=True)))}")
    print(f"[域分布 arxiv ] {dict(zip(*np.unique(DOMA, return_counts=True)))}")
    print(f"[域分布 github] {dict(zip(*np.unique(DOMG, return_counts=True)))}")

    K = len(ALL_METRICS)

    # ---------- S2.1 相关性诊断 ----------
    print("\n===== S2.1 相关性/冗余诊断 =====")
    Rp = np.corrcoef(T, rowvar=False)
    # Spearman
    from scipy.stats import spearmanr, kendalltau
    Rs, _ = spearmanr(T)
    Rs = np.nan_to_num(Rs, nan=0.0)
    np.save(os.path.join(DATA, "s2_corr_pearson.npy"), Rp)
    np.save(os.path.join(DATA, "s2_corr_spearman.npy"), Rs)
    # 高相关对（|r|>0.8）
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((ALL_METRICS[i], ALL_METRICS[j], Rp[i, j], Rs[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    print("  Top-15 高相关指标对（Pearson |r| 排序）：")
    for a, b, r, sr in pairs[:15]:
        print(f"    {a:<44s} ~ {b:<44s} r={r:+.3f} rho={sr:+.3f}")
    pd.DataFrame(pairs, columns=["指标A", "指标B", "Pearson_r", "Spearman_rho"]).to_csv(
        os.path.join(TABLES, "s2_指标相关性对.csv"), index=False, encoding="utf-8-sig")

    # ---------- S2.2 四种赋权 ----------
    print("\n===== S2.2 赋权方案 =====")
    k = K
    w_ew = np.ones(k) / k
    w_ewm = entropy_weight(T)
    w_critic = critic_weight(T)
    z_pc, V, ev, evr, mpc = topc_score(T)
    print(f"  PCA: 前{mpc}个主成分累计方差贡献={evr[:mpc].sum():.4f}; PC1={evr[0]:.4f}")
    # PCA 隐含权重：|载荷| × 方差贡献，归一化
    load1 = np.abs(V[:, 0]) * math.sqrt(max(ev[0], 1e-12))
    w_pca = load1 / load1.sum()
    # 组合赋权：CRITIC 与 EWM 几何平均，再与 PCA 权重权重融合
    w_combo = np.sqrt(np.clip(w_critic, 1e-12, None) * np.clip(w_ewm, 1e-12, None))
    w_combo = w_combo / w_combo.sum()
    # 秩融合更稳健：先算各方案得分，再取秩均值
    print("\n  各方案权重（前 12 项）：")
    for i in np.argsort(-w_combo)[:12]:
        print(f"    {ALL_METRICS[i]:<44s} EW={w_ew[i]:.4f} EWM={w_ewm[i]:.4f} "
              f"CRITIC={w_critic[i]:.4f} PCA={w_pca[i]:.4f} COMBO={w_combo[i]:.4f}")

    Wdf = pd.DataFrame({
        "指标": ALL_METRICS, "指标中文": [METRIC_CN[x] for x in ALL_METRICS],
        "指标族": [METRIC_FAMILY[x] for x in ALL_METRICS],
        "方向": [METRIC_DIRECTION[x] for x in ALL_METRICS],
        "等权EW": w_ew, "熵权EWM": w_ewm, "CRITIC": w_critic,
        "PCA隐含": w_pca, "组合权重": w_combo,
    }).sort_values("组合权重", ascending=False)
    Wdf.to_csv(os.path.join(TABLES, "s2_赋权方案对比.csv"), index=False, encoding="utf-8-sig")

    # ---------- S2.3 一致性检验 ----------
    print("\n===== S2.3 赋权方案一致性 =====")
    scores = {
        "EW": T @ w_ew,
        "EWM": T @ w_ewm,
        "CRITIC": T @ w_critic,
        "PCA": z_pc,
        "COMBO": T @ w_combo,
    }
    names = list(scores)
    cons = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            cons[i, j] = spearmanr(scores[a], scores[b]).correlation
    print("  Spearman 秩相关矩阵：")
    print("     " + "  ".join(f"{n:>8s}" for n in names))
    for i, a in enumerate(names):
        print(f"  {a:>6s} " + "  ".join(f"{cons[i,j]:>8.4f}" for j in range(len(names))))
    pd.DataFrame(cons, index=names, columns=names).to_csv(
        os.path.join(TABLES, "s2_赋权一致性矩阵.csv"), encoding="utf-8-sig")
    # Kendall W（协同系数）：k 个评价方案对 n 个样本的排序一致性
    # W = 12*S / ( k^2 * (n^3 - n) ),  S = Σ(R_j - mean R)^2（R_j 为样本 j 的秩和）
    M = np.vstack([scores[n] for n in names]).T     # (n_samples, k_schemes)
    k_scheme = M.shape[1]
    n_samp = M.shape[0]
    ranks = np.apply_along_axis(lambda r: np.argsort(np.argsort(r)), 0, M).astype(float)
    Rj = ranks.sum(axis=1)
    S_ = ((Rj - Rj.mean()) ** 2).sum()
    W_kendall = 12 * S_ / (k_scheme ** 2 * (n_samp ** 3 - n_samp))
    chi2 = k_scheme * (n_samp - 1) * W_kendall
    from scipy.stats import chi2 as chi2dist
    pval = 1 - chi2dist.cdf(chi2, df=n_samp - 1)
    print(f"  Kendall 协同系数 W = {W_kendall:.6f}  (k={k_scheme} 方案, n={n_samp} 样本)")
    print(f"  χ² = {chi2:.3f}, df = {n_samp-1}, p = {pval:.3e} "
          f"({'方案间高度一致' if pval<0.05 else '方案间一致性不显著'})")
    save_json({"kendall_W": float(W_kendall), "chi2": float(chi2), "df": int(n_samp - 1),
               "p_value": float(pval), "k_schemes": int(k_scheme), "n_samples": int(n_samp),
               "spearman_matrix": {a: {b: float(cons[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
               "方案": names}, os.path.join(DATA, "s2_consistency.json"))

    # ---------- S2.4 最终 Q：主方案 + 秩融合稳健方案 ----------
    print("\n===== S2.4 最终 Q 评分 =====")
    Q_combo = scores["COMBO"]
    # 秩融合：把 5 个方案得分转百分位秩后平均
    from scipy.stats import rankdata
    R = np.vstack([rankdata(scores[n]) / len(Q_combo) for n in names])
    Q_rank = R.mean(axis=0)
    # 主方案 Q：组合权重得分再映射到 [0,1]（用 1%/99% 分位稳健拉伸）
    lo, hi = np.percentile(Q_combo, 1), np.percentile(Q_combo, 99)
    Q = np.clip((Q_combo - lo) / (hi - lo), 0, 1)
    print(f"  Q(组合) 统计: mean={Q.mean():.4f} std={Q.std():.4f} "
          f"min={Q.min():.4f} p5={np.percentile(Q,5):.4f} p50={np.percentile(Q,50):.4f} "
          f"p95={np.percentile(Q,95):.4f} max={Q.max():.4f}")
    print(f"  Q(秩融合) 与 Q(组合) Spearman = {spearmanr(Q, Q_rank).correlation:.4f}")

    # ---------- S2.5 bootstrap 不确定性 ----------
    # 目的：量化"赋权方案"不确定性对 Q 的影响。
    # 做法：bootstrap 重采样估计权重，但把权重作用于【全量样本】的 T 上计算 Q，
    #       从而 SE 只反映权重扰动，而非样本量的缩放效应。
    rng = np.random.default_rng(20260924)
    B = 200
    qs = np.zeros((B, len(Q)))
    wboot = np.zeros((B, K))
    for b in range(B):
        idx = rng.integers(0, len(Q), len(Q))
        Tb = T[idx]
        wb = np.sqrt(np.clip(critic_weight(Tb), 1e-12, None) *
                     np.clip(entropy_weight(Tb), 1e-12, None))
        wb = wb / wb.sum()
        wboot[b] = wb
        sb = T @ wb                                   # 全量样本上应用扰动权重
        lo_b, hi_b = np.percentile(sb, 1), np.percentile(sb, 99)
        qs[b] = np.clip((sb - lo_b) / (hi_b - lo_b + 1e-12), 0, 1)
    Q_se = qs.std(axis=0, ddof=1)
    Q_lo = np.percentile(qs, 5, axis=0)
    Q_hi = np.percentile(qs, 95, axis=0)
    print(f"  Bootstrap Q 标准误: mean={Q_se.mean():.5f} median={np.median(Q_se):.5f} max={Q_se.max():.5f}")
    print(f"  Q 的 90% bootstrap 区间宽度: mean={(Q_hi-Q_lo).mean():.5f}")
    print(f"  权重变异系数(各指标): {dict(zip(ALL_METRICS[:5], np.round(100*wboot.std(axis=0)/wboot.mean(axis=0),2)))} ...")

    np.savez_compressed(os.path.join(DATA, "s2_scores_sample.npz"),
                        Q=Q, Q_rank=Q_rank, Q_se=Q_se, Q_lo=Q_lo, Q_hi=Q_hi, T=T, DOM=DOM, ID=ID,
                        scores_EW=scores["EW"], scores_EWM=scores["EWM"],
                        scores_CRITIC=scores["CRITIC"], scores_PCA=scores["PCA"])
    np.savez_compressed(os.path.join(DATA, "s2_weights.npz"),
                        w_ew=w_ew, w_ewm=w_ewm, w_critic=w_critic, w_pca=w_pca, w_combo=w_combo,
                        wboot=wboot, evr=evr, loadings=V[:, :mpc],
                        names=np.array(ALL_METRICS))

    # ---------- S2.6 领域级聚合 ----------
    print("\n===== S2.6 语料级/领域级聚合 =====")
    def aggregate(Q_, DOM_, chars_=None, prefix=""):
        rows = []
        for dom in sorted(np.unique(DOM_)):
            msk = DOM_ == dom
            q = Q_[msk]
            if chars_ is not None and msk.sum() > 0:
                ch = chars_[msk].astype(np.float64)
                wch = np.clip(ch, 0, None) + 1.0
            else:
                wch = None
            rec = {
                "域": dom, "样本数": int(msk.sum()),
                "Q_均值": float(q.mean()),
                "Q_标准差": float(q.std(ddof=1)) if msk.sum() > 1 else 0.0,
                "Q_中位数": float(np.median(q)),
                "Q_p10": float(np.percentile(q, 10)),
                "Q_p25": float(np.percentile(q, 25)),
                "Q_p75": float(np.percentile(q, 75)),
                "Q_p90": float(np.percentile(q, 90)),
            }
            if wch is not None:
                rec["Q_字符加权均值"] = float((q * wch).sum() / wch.sum())
                rec["平均字符数"] = float(ch.mean())
            rows.append(rec)
        df = pd.DataFrame(rows)
        df.insert(0, "数据集", prefix)
        return df

    aggS = aggregate(Q, DOM, d["CHARS"], "sample")
    print(aggS.to_string(index=False))

    # 扩展集：用抽样集的归一化参数与权重（同一套预处理流程）
    def apply_to(TX):
        s = TX @ w_combo
        return np.clip((s - lo) / (hi - lo), 0, 1)
    QA = apply_to(TA); QG = apply_to(TG)
    print(f"  arxiv 扩展集 Q: mean={QA.mean():.4f} p50={np.median(QA):.4f}")
    print(f"  github扩展集 Q: mean={QG.mean():.4f} p50={np.median(QG):.4f}")
    aggA = aggregate(QA, DOMA, dA["CHARS"], "arxiv_ext")
    aggG = aggregate(QG, DOMG, dG["CHARS"], "github_ext")
    agg = pd.concat([aggS, aggA, aggG], ignore_index=True)
    agg.to_csv(os.path.join(TABLES, "s2_领域级Q汇总.csv"), index=False, encoding="utf-8-sig")

    # 领域排序（抽样集）
    print("\n  sample 域级 Q 排序：")
    for _, r in aggS.sort_values("Q_均值", ascending=False).iterrows():
        print(f"    {r['域']:<16s} n={int(r['样本数']):>6d} Q_mean={r['Q_均值']:.4f} "
              f"p10={r['Q_p10']:.4f} p90={r['Q_p90']:.4f}")

    # 域×指标 均值矩阵（供后续映射）
    dom_ind = {}
    for dom in np.unique(DOM):
        dom_ind[str(dom)] = T[DOM == dom].mean(axis=0)
    np.savez_compressed(os.path.join(DATA, "s2_domain_indicator_mean.npz"),
                        **{k: v for k, v in dom_ind.items()},
                        names=np.array(ALL_METRICS))
    # 扩展集域×指标
    dom_ind_A = {str(d): TA[DOMA == d].mean(axis=0) for d in np.unique(DOMA)}
    dom_ind_G = {str(d): TG[DOMG == d].mean(axis=0) for d in np.unique(DOMG)}
    np.savez_compressed(os.path.join(DATA, "s2_domain_indicator_mean_ext.npz"),
                        **{("arxiv_" + k): v for k, v in dom_ind_A.items()},
                        **{("github_" + k): v for k, v in dom_ind_G.items()},
                        names=np.array(ALL_METRICS))

    save_json({
        "n_sample": int(len(Q)), "n_arxiv": int(len(QA)), "n_github": int(len(QG)),
        "Q_mean_sample": float(Q.mean()), "Q_std_sample": float(Q.std()),
        "Q_mean_arxiv": float(QA.mean()), "Q_mean_github": float(QG.mean()),
        "weights_combo": {ALL_METRICS[i]: float(w_combo[i]) for i in range(K)},
        "pca_evr": evr.tolist(), "pca_ncomp": int(mpc),
        "kendall_W": float(W_kendall),
    }, os.path.join(DATA, "s2_summary.json"))

    print(f"\n[OK] s2_quality 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s2_quality")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
s3_conflict.py —— 阶段3：质量冲突定义、成因分析与消解

【冲突的数学定义】
  设文本 i 在 K 个"越高越好"指标上的得分为 t_ik ∈ [0,1]（k=1..K）。
  定义 指标族 g 的族内一致得分 s_ig = 家族内指标的加权均值。
  定义 冲突强度：
      C_i = 2 * std_g( { s_ig } ) / (range of [0,1] = 1)   # 族间极差/离散度
  更规范地，定义"显著不一致"：
      存在两个指标族 g1, g2 使 |s_i,g1 - s_i,g2| > δ（δ 由分位数法确定）
  并定义冲突度 D_i = 族间得分标准差，冲突标记 = 1{ D_i > δ_conf }

  另一种定义（指标层）：D_i^k = |t_ik - Q_i^{-k}|，即以"去掉该指标后的综合分"
  为参照，衡量该指标与其余指标的不一致。

【成因分析】
  方向维度：正负向指标的语义张力（如 ad_en 高 ↔ fluency 高；
           word_count 长 ↔ uppercase 高）
  数据源维度：域先验差异（arxiv 学术 vs github 代码）
  指标族维度：分类器打分 vs 文本统计
  指标冗余维度：dsir_* 三兄弟高度共线

【消解规则】
  规则1 冲突感知加权：对样本级降低"离群指标"的权重（OWA / 惩罚项）
  规则2 族平衡：先族内合成，再族间等权，避免某一族（如文本统计 10 个指标）
        因数量多而主导
  规则3 可信度门控：对列表型分类器指标，用 confidence 作为可信度，
        低置信度样本降低其分类器得分的影响力
  规则4 冲突标记作为质量风险的显式暴露：输出 Q 的同时输出冲突度，
        供问题二/三使用（可选是否惩罚）

产出：data/s3_conflict_*.npz/json, results/tables/s3_*.csv, 图
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
from scipy.stats import spearmanr, pearsonr

C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"

# 指标族定义（用于族间冲突）
FAMILY_DEF = {
    "F1_分类器语义": ["fineweb_edu", "fluency_en", "modernbert_cleanliness",
                     "modernbert_readability", "modernbert_reasoning",
                     "modernbert_professionalism", "qurater", "ad_en"],
    "F2_DSIR重要性": ["dsir_books", "dsir_wiki", "dsir_math"],
    "F3_文本统计": ["rps_doc_word_count", "rps_doc_num_sentences",
                    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
                    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
                    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
                    "rps_lines_ending_with_terminal_punctution_mark",
                    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"],
}

def load(tag):
    z = np.load(os.path.join(DATA, f"s1_feat_{tag}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}

def family_scores(T, names):
    """按族计算族内等权平均得分"""
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for fam, mems in FAMILY_DEF.items():
        cols = [idx[m] for m in mems]
        out[fam] = T[:, cols].mean(axis=1)
    return out

def analyze(T, DOM, names, tag, MISS=None):
    K = len(names)
    fam = family_scores(T, names)
    fam_names = list(fam)
    F = np.vstack([fam[f] for f in fam_names]).T          # (n, 3)
    # 冲突度：族间标准差
    Dfam = F.std(axis=1, ddof=1)
    # 冲突度：指标层（相对其余指标均值的偏差）
    Dind = np.zeros(len(T))
    for k in range(K):
        others = np.delete(T, k, axis=1).mean(axis=1)
        Dind += np.abs(T[:, k] - others)
    Dind /= K
    print(f"\n[{tag}] 冲突度统计（族间标准差 Dfam）")
    print(f"  mean={Dfam.mean():.4f} std={Dfam.std():.4f} "
          f"p50={np.median(Dfam):.4f} p90={np.percentile(Dfam,90):.4f} p99={np.percentile(Dfam,99):.4f} max={Dfam.max():.4f}")
    print(f"[{tag}] 冲突度统计（指标层平均偏差 Dind）")
    print(f"  mean={Dind.mean():.4f} p50={np.median(Dind):.4f} p90={np.percentile(Dind,90):.4f} max={Dind.max():.4f}")
    # 阈值：δ = p90 分位
    delta_f = float(np.percentile(Dfam, 90))
    delta_i = float(np.percentile(Dind, 90))
    flag_f = (Dfam > delta_f).astype(np.float32)
    flag_i = (Dind > delta_i).astype(np.float32)
    print(f"[{tag}] 阈值 δ_fam=p90={delta_f:.4f} → 冲突样本占比 {flag_f.mean()*100:.1f}%")
    print(f"[{tag}] 阈值 δ_ind=p90={delta_i:.4f} → 冲突样本占比 {flag_i.mean()*100:.1f}%")

    # 域分布对比：冲突样本的域构成
    rows = []
    for dom in sorted(np.unique(DOM)):
        msk = DOM == dom
        rows.append({"数据集": tag, "域": dom, "样本数": int(msk.sum()),
                     "冲突率(Dfam>p90)": float(flag_f[msk].mean()),
                     "冲突率(Dind>p90)": float(flag_i[msk].mean()),
                     "Dfam均值": float(Dfam[msk].mean()),
                     "Dind均值": float(Dind[msk].mean())})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return dict(fam=fam, F=F, Dfam=Dfam, Dind=Dind, delta_f=delta_f, delta_i=delta_i,
                flag_f=flag_f, flag_i=flag_i, fam_df=df)

def main():
    start_log("s3_conflict")
    t0 = time.time()
    dS = load("sample"); dA = load("arxiv"); dG = load("github")
    T = dS["T"].astype(np.float64); DOM = dS["DOM"]; ID = dS["ID"]
    TA = dA["T"].astype(np.float64); DOMA = dA["DOM"]
    TG = dG["T"].astype(np.float64); DOMG = dG["DOM"]
    names = ALL_METRICS
    print(f"[数据] sample={T.shape} arxiv={TA.shape} github={TG.shape}")

    RS = analyze(T, DOM, names, "sample", dS.get("MISS"))
    RA = analyze(TA, DOMA, names, "arxiv_ext", dA.get("MISS"))
    RG = analyze(TG, DOMG, names, "github_ext", dG.get("MISS"))

    # ---------- 冲突成因 A：方向张力（负向指标 vs 正向指标族） ----------
    print("\n===== 冲突成因 A：方向张力分析 =====")
    idx = {n: i for i, n in enumerate(names)}
    # 典型张力对
    tension_pairs = [
        ("rps_doc_word_count", "rps_lines_uppercase_letter_fraction", "文档越长↔大写行越多(格式噪声)"),
        ("fluency_en", "ad_en", "流畅度↔广告含量"),
        ("rps_doc_frac_unique_words", "rps_doc_unigram_entropy", "唯一词占比↔词熵"),
        ("dsir_books", "rps_doc_unigram_entropy", "DSIR图书相似度↔词熵"),
        ("rps_lines_ending_with_terminal_punctution_mark", "rps_doc_frac_no_alph_words", "句末标点↔非字母词"),
        ("fineweb_edu", "rps_doc_mean_word_length", "教育价值↔平均词长"),
    ]
    rows = []
    for a, b, desc in tension_pairs:
        r = pearsonr(T[:, idx[a]], T[:, idx[b]])
        rows.append({"指标A": a, "指标B": b, "冲突含义": desc,
                     "Pearson_r": float(r[0]), "p": float(r[1]),
                     "冲突方向": "负向张力" if r[0] < 0 else "正向协同",
                     "强度": abs(float(r[0]))})
    dfT = pd.DataFrame(rows).sort_values("强度", ascending=False)
    print(dfT.to_string(index=False))
    dfT.to_csv(os.path.join(TABLES, "s3_冲突方向张力.csv"), index=False, encoding="utf-8-sig")

    # ---------- 冲突成因 B：族间相关矩阵 ----------
    print("\n===== 冲突成因 B：三族得分相关矩阵 =====")
    fam_names = list(RS["fam"])
    FS = np.vstack([RS["fam"][f] for f in fam_names]).T
    FA = np.vstack([RA["fam"][f] for f in fam_names]).T
    FG = np.vstack([RG["fam"][f] for f in fam_names]).T
    for nm, FF in [("sample", FS), ("arxiv_ext", FA), ("github_ext", FG)]:
        R = np.corrcoef(FF, rowvar=False)
        print(f"  [{nm}]")
        print("       " + "  ".join(f"{f:>12s}" for f in fam_names))
        for i, f in enumerate(fam_names):
            print(f"  {f:>12s} " + "  ".join(f"{R[i,j]:>12.4f}" for j in range(len(fam_names))))
        pd.DataFrame(R, index=fam_names, columns=fam_names).to_csv(
            os.path.join(TABLES, f"s3_族间相关矩阵_{nm}.csv"), encoding="utf-8-sig")

    # ---------- 冲突成因 C：域先验 ----------
    print("\n===== 冲突成因 C：域先验差异 =====")
    dom_fam = {}
    for dom in sorted(np.unique(DOM)):
        msk = DOM == dom
        dom_fam[str(dom)] = {f: float(RS["fam"][f][msk].mean()) for f in fam_names}
    dfD = pd.DataFrame(dom_fam).T
    dfD.index.name = "域"
    print(dfD.to_string())
    dfD.to_csv(os.path.join(TABLES, "s3_域×族得分.csv"), encoding="utf-8-sig")

    # ---------- 冲突样本示例 ----------
    print("\n===== 冲突样本 Top 示例（sample，Dfam 最大） =====")
    ordD = np.argsort(-RS["Dfam"])[:8]
    rows = []
    for i in ordD:
        fams = {f: float(RS["fam"][f][i]) for f in fam_names}
        rows.append({"id": str(ID[i]), "域": str(DOM[i]), "Dfam": float(RS["Dfam"][i]),
                     **{f: v for f, v in fams.items()},
                     "冲突主导族": max(fams, key=fams.get),
                     "冲突弱势族": min(fams, key=fams.get)})
    dfC = pd.DataFrame(rows)
    print(dfC.to_string(index=False))
    dfC.to_csv(os.path.join(TABLES, "s3_冲突样本示例.csv"), index=False, encoding="utf-8-sig")

    # ---------- 消解规则 ----------
    print("\n===== 冲突消解规则构建 =====")
    # 规则2 族平衡得分：先族内平均，再族间等权（3 族等权）
    QB_fam = FS.mean(axis=1)
    # 规则1 冲突感知加权：以"离群惩罚"降低冲突指标权重
    def conflict_aware(T_, D_):
        # 对每个样本，指标权重 ∝ 1 - |t_ik - others_mean|，再做归一化
        W = np.zeros_like(T_)
        for k in range(T_.shape[1]):
            others = np.delete(T_, k, axis=1).mean(axis=1)
            W[:, k] = np.clip(1.0 - np.abs(T_[:, k] - others), 1e-3, 1.0)
        W = W / W.sum(axis=1, keepdims=True)
        return (T_ * W).sum(axis=1), W
    QB_aw, Waw = conflict_aware(T, RS["Dfam"])
    # 规则3 可信度门控：分类器族的权重按 confidence 调整
    CONF = dS["CONF"].astype(np.float64)      # (n, 8) 对应 LIST_METRICS
    idxL = [idx[m] for m in LIST_METRICS]
    conf_mean = np.nanmean(CONF, axis=1)
    conf_mean = np.nan_to_num(conf_mean, nan=0.5)
    # 门控：低置信度时，把该样本分类器族向族内中位数收缩
    fam_cls = RS["fam"]["F1_分类器语义"]
    shrink = np.clip(conf_mean, 0.0, 1.0)
    fam_cls_gated = shrink * fam_cls + (1 - shrink) * np.median(fam_cls)
    QB_gate = np.vstack([fam_cls_gated, RS["fam"]["F2_DSIR重要性"], RS["fam"]["F3_文本统计"]]).T.mean(axis=1)

    print(f"  规则2 族平衡得分 QB_fam: mean={QB_fam.mean():.4f} std={QB_fam.std():.4f}")
    print(f"  规则1 冲突感知得分 QB_aw : mean={QB_aw.mean():.4f} std={QB_aw.std():.4f}")
    print(f"  规则3 可信度门控 QB_gate: mean={QB_gate.mean():.4f} std={QB_gate.std():.4f}")
    Z = np.load(os.path.join(DATA, "s2_scores_sample.npz"), allow_pickle=True)
    Q_main = Z["Q"]
    for nm, q in [("原组合Q", Q_main), ("族平衡", QB_fam), ("冲突感知", QB_aw), ("可信度门控", QB_gate)]:
        print(f"    vs 原组合Q 的 Spearman: {nm:<10s} {spearmanr(Q_main, q).correlation:.4f}")

    # ---------- 冲突度作为风险暴露：Q 与冲突度的关系 ----------
    print("\n===== 冲突度与 Q 的关系 =====")
    print(f"  corr(Q, Dfam) Pearson={pearsonr(Q_main, RS['Dfam'])[0]:.4f} "
          f"Spearman={spearmanr(Q_main, RS['Dfam']).correlation:.4f}")
    print(f"  corr(Q, Dind) Pearson={pearsonr(Q_main, RS['Dind'])[0]:.4f} "
          f"Spearman={spearmanr(Q_main, RS['Dind']).correlation:.4f}")
    # Q 与 bootstrap SE
    print(f"  corr(Q, Q_se) Spearman={spearmanr(Q_main, Z['Q_se']).correlation:.4f}")

    # 扩展集检验主要结论
    print("\n===== 扩展集结论一致性检验 =====")
    check = []
    for nm, RR, TT, DD in [("arxiv_ext", RA, TA, DOMA), ("github_ext", RG, TG, DOMG)]:
        c1 = pearsonr(RR["fam"]["F1_分类器语义"], RR["fam"]["F3_文本统计"])[0]
        c2 = pearsonr(RS["fam"]["F1_分类器语义"], RS["fam"]["F3_文本统计"])[0]
        check.append({"数据集": nm, "corr(分类器族,文本统计族)": float(c1)})
        check.append({"数据集": "sample", "corr(分类器族,文本统计族)": float(c2)})
        # ns
    # 强制：arxiv 与 github 上族间相关符号是否一致
    print("  族间相关符号一致性（分类器族 ↔ 文本统计族）：")
    for nm, RR in [("sample", RS), ("arxiv_ext", RA), ("github_ext", RG)]:
        Fx = np.vstack([RR["fam"][f] for f in fam_names]).T
        r = np.corrcoef(Fx, rowvar=False)[0, 2]
        print(f"    {nm:<12s} r(F1,F3) = {r:+.4f}")

    np.savez_compressed(os.path.join(DATA, "s3_conflict_sample.npz"),
                        Dfam=RS["Dfam"], Dind=RS["Dind"], flag_f=RS["flag_f"],
                        flag_i=RS["flag_i"], F=RS["F"], QB_fam=QB_fam,
                        QB_aw=QB_aw, QB_gate=QB_gate, conf_mean=conf_mean,
                        fam_names=np.array(fam_names))
    np.savez_compressed(os.path.join(DATA, "s3_conflict_ext.npz"),
                        Dfam_A=RA["Dfam"], Dind_A=RA["Dind"], flag_f_A=RA["flag_f"],
                        F_A=RA["F"], Dfam_G=RG["Dfam"], Dind_G=RG["Dind"],
                        flag_f_G=RG["flag_f"], F_G=RG["F"], fam_names=np.array(fam_names))
    pd.concat([RS["fam_df"], RA["fam_df"], RG["fam_df"]], ignore_index=True).to_csv(
        os.path.join(TABLES, "s3_域×冲突率.csv"), index=False, encoding="utf-8-sig")

    save_json({
        "fam_names": fam_names,
        "delta_f_sample": RS["delta_f"], "delta_i_sample": RS["delta_i"],
        "conflict_rate_sample": float(RS["flag_f"].mean()),
        "conflict_rate_arxiv": float(RA["flag_f"].mean()),
        "conflict_rate_github": float(RG["flag_f"].mean()),
        "Dfam_mean_sample": float(RS["Dfam"].mean()),
        "Dfam_mean_arxiv": float(RA["Dfam"].mean()),
        "Dfam_mean_github": float(RG["Dfam"].mean()),
        "corr_Q_Dfam_sample": float(pearsonr(Q_main, RS["Dfam"])[0]),
        "spearman_vs_mainQ": {
            "族平衡": float(spearmanr(Q_main, QB_fam).correlation),
            "冲突感知": float(spearmanr(Q_main, QB_aw).correlation),
            "可信度门控": float(spearmanr(Q_main, QB_gate).correlation),
        },
    }, os.path.join(DATA, "s3_summary.json"))
    print(f"\n[OK] s3_conflict 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s3_conflict")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
问题一(上·精化)：冲突定义修正 + 主评分口径确定 + 领域Q汇总重写
依据首轮发现的两个问题:
 (1) 熵权退化(长度类指标主导) -> 主评分改用 Q_equal(组间平衡的等权集成), Q_robust为冲突消解版
 (2) 极差型冲突指数无区分度 -> 改用 语义组/启发组 分歧度 + 分位数阈值
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = r"C:/Users/dkyyt/Desktop/F题"
OUT = os.path.join(ROOT, "outputs", "q1_quality")

a1q = pd.read_csv(os.path.join(OUT, "sampleQ_A1.csv"))
a2q = pd.read_csv(os.path.join(OUT, "sampleQ_A2_arxiv.csv"))
a3q = pd.read_csv(os.path.join(OUT, "sampleQ_A3_github.csv"))
a1s = pd.read_csv(os.path.join(OUT, "a1_scalar_metrics.csv"))

# ---------- 1. 冲突定义(精化) ----------
# 分歧度 gap = |semantic_mean - heuristic_mean|; 强冲突 = gap >= A1分布的90分位数
TH_GAP = float(a1q["group_gap"].quantile(0.90))
print(f"强冲突阈值(gap的P90) = {TH_GAP:.4f}")
for df in (a1q, a2q, a3q):
    df["conflict_v2"] = (df["group_gap"] >= TH_GAP).astype(int)

# 典型成对冲突: 语义高(教育价值P80以上) 但 广告原始分高 的样本
# 用定向后的列无法直接取广告原始分, 从a1_scalar_metrics取ad_en原始均值并分位判定
a1s_small = a1s[["id","domain","ad_en","fineweb_edu","content_chars","content_latex_marks",
                 "content_digit_frac","content_alpha_frac"]].copy()
m = a1q.merge(a1s_small, on=["id","domain"], how="left", suffixes=("","_raw"))
ad_hi = m["ad_en"] >= m["ad_en"].quantile(0.80)
edu_hi = m["fineweb_edu"] >= m["fineweb_edu"].quantile(0.80)
m["pair_conflict_edu_ad"] = (ad_hi & edu_hi).astype(int)
print("成对冲突(高教育价值&高广告)占比: {:.4f}".format(m["pair_conflict_edu_ad"].mean()))

# ---------- 2. 冲突率与成因(分域) ----------
rows = []
for dom, g in m.groupby("domain"):
    conf = g[g["conflict_v2"] == 1]
    nonc = g[g["conflict_v2"] == 0]
    rows.append({
        "domain": dom, "n": len(g),
        "conflict_rate_v2": g["conflict_v2"].mean(),
        "pair_conflict_rate": g["pair_conflict_edu_ad"].mean(),
        "latex_marks_conf": conf["content_latex_marks"].median(),
        "latex_marks_nonconf": nonc["content_latex_marks"].median(),
        "digit_frac_conf": conf["content_digit_frac"].median(),
        "digit_frac_nonconf": nonc["content_digit_frac"].median(),
        "chars_conf": conf["content_chars"].median(),
        "chars_nonconf": nonc["content_chars"].median(),
    })
cb = pd.DataFrame(rows)
ext_rows = [
    {"domain": "arxiv_A2_full", "n": len(a2q), "conflict_rate_v2": a2q["conflict_v2"].mean()},
    {"domain": "github_A3_full", "n": len(a3q), "conflict_rate_v2": a3q["conflict_v2"].mean()},
]
cb = pd.concat([cb, pd.DataFrame(ext_rows)], ignore_index=True)
cb.to_csv(os.path.join(OUT, "conflict_by_domain_v2.csv"), index=False)
print(cb.round(4).to_string())

# ---------- 3. 冲突消解: 截尾鲁棒聚合 ----------
# Q_robust 已在首轮计算(去头尾各4/22); 对比冲突样本上 Q_equal 与 Q_robust 的位移
m["dQ"] = m["Q_robust"] - m["Q_equal"]
res = {
    "全体样本 |dQ| 均值": float(m["dQ"].abs().mean()),
    "冲突样本 |dQ| 均值": float(m.loc[m["conflict_v2"]==1, "dQ"].abs().mean()),
    "非冲突样本 |dQ| 均值": float(m.loc[m["conflict_v2"]==0, "dQ"].abs().mean()),
    "Q_equal与Q_robust全样本Spearman": float(spearmanr(m["Q_equal"], m["Q_robust"])[0]),
}
pd.DataFrame([res]).to_csv(os.path.join(OUT, "conflict_resolution_effect.csv"), index=False)
print(res)

# 哪些指标最常"偏离群体": 用定向矩阵, 每样本找偏离样本中位数最远的指标
METRICS = ["fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
           "modernbert_reasoning","modernbert_professionalism","dsir_books","dsir_wiki",
           "dsir_math","qurater","ad_en","rps_doc_word_count","rps_doc_num_sentences",
           "rps_doc_unigram_entropy","rps_doc_frac_unique_words","rps_doc_frac_no_alph_words",
           "rps_doc_frac_chars_top_2gram","rps_doc_frac_chars_top_3gram",
           "rps_lines_uppercase_letter_fraction","rps_lines_ending_with_terminal_punctution_mark",
           "rps_lines_numerical_chars_fraction","rps_doc_mean_word_length"]
DIRECTION = {"ad_en": -1, "rps_doc_frac_no_alph_words": -1, "rps_doc_frac_chars_top_2gram": -1,
             "rps_doc_frac_chars_top_3gram": -1, "rps_lines_uppercase_letter_fraction": -1,
             "rps_lines_numerical_chars_fraction": -1}
np_ = pd.read_csv(os.path.join(OUT, "normalization_params.csv"), index_col=0)
def normalize(df):
    X = np.full((len(df), len(METRICS)), np.nan)
    for j, mt in enumerate(METRICS):
        x = df[mt].astype(float).values.copy()
        p1, p99 = np_.loc[mt, "p1"], np_.loc[mt, "p99"]
        x = np.clip(x, p1, p99)
        xn = (x - p1) / max(p99 - p1, 1e-12)
        if DIRECTION.get(mt, 1) == -1:
            xn = 1.0 - xn
        X[:, j] = xn
    return X
X1 = normalize(a1s)
Xf = np.nan_to_num(X1, nan=0.5)
med = np.median(Xf, axis=1, keepdims=True)
out_idx = np.argmax(np.abs(Xf - med), axis=1)
outlier_cnt = pd.Series(out_idx).value_counts()
out_tbl = pd.DataFrame({"metric": [METRICS[i] for i in outlier_cnt.index],
                        "outlier_count": outlier_cnt.values,
                        "outlier_rate": outlier_cnt.values / len(Xf)})
out_tbl.to_csv(os.path.join(OUT, "metric_outlier_contribution.csv"), index=False)
print(out_tbl.round(4).to_string())

# ---------- 4. 领域级Q汇总重写(主口径=Q_equal; 另列Q_robust/Q_entropy) ----------
def agg(df, tag):
    g = df.groupby("domain").agg(n=("Q_equal", "size"), Q_equal=("Q_equal", "mean"),
        Q_robust=("Q_robust", "mean"), Q_entropy=("Q_entropy", "mean"),
        Q_median=("Q_equal", "median"), Q_std=("Q_equal", "std")).reset_index()
    g["source"] = tag
    return g
d1 = agg(a1q, "A1抽样")
d2 = agg(a2q, "A2全量")
d3 = agg(a3q, "A3全量")
dom = pd.concat([d1, d2, d3], ignore_index=True)
dom["Q_primary"] = dom["Q_equal"]  # 主口径: 等权(组间平衡)集成
dom.to_csv(os.path.join(OUT, "domain_Q_summary.csv"), index=False)
print(dom.round(4).to_string())

# 域排序稳健性
sub = dom[dom["source"] == "A1抽样"]
r_eq, _ = spearmanr(sub["Q_equal"], sub["Q_robust"])
r_eq2, _ = spearmanr(sub["Q_equal"], sub["Q_entropy"])
print(f"域排序Spearman: equal vs robust = {r_eq:.4f}, equal vs entropy = {r_eq2:.4f}")

# 抽样 vs 全量对照(arxiv/github)
cmp = []
for domn in ["arxiv", "github"]:
    s = dom[(dom["domain"] == domn) & (dom["source"] == "A1抽样")].iloc[0]
    f = dom[(dom["domain"] == domn) & (dom["source"].str.contains("全量"))].iloc[0]
    cmp.append({"domain": domn, "Q_sample": s["Q_primary"], "n_sample": s["n"],
                "Q_full": f["Q_primary"], "n_full": f["n"],
                "abs_diff": abs(s["Q_primary"] - f["Q_primary"])})
cmp = pd.DataFrame(cmp)
cmp.to_csv(os.path.join(OUT, "sample_vs_full_check.csv"), index=False)
print(cmp.round(4).to_string())

# ---------- 5. 更新图表 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
dd = sub.sort_values("Q_primary", ascending=False)
axes[0].bar(dd["domain"], dd["Q_primary"], color="#28B463")
axes[0].set_title("领域级质量评分 Q(主口径: 定向等权集成)")
axes[0].set_ylabel("Q")
plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
cb7 = cb.dropna(subset=["conflict_rate_v2"]).head(7).sort_values("conflict_rate_v2", ascending=False)
axes[1].bar(cb7["domain"], cb7["conflict_rate_v2"], color="#CA6F1E")
axes[1].set_title(f"各域强冲突率(gap≥P90={TH_GAP:.2f})")
axes[1].set_ylabel("强冲突样本占比")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_domain_Q_v2.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
sel = m.sample(8000, random_state=0)
sc = ax.scatter(sel["semantic_mean"], sel["heuristic_mean"], s=4, alpha=0.3,
                c=sel["conflict_v2"], cmap="coolwarm")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("语义模型组均值(定向)"); ax.set_ylabel("启发统计组均值(定向)")
ax.set_title("组间分歧与冲突判定(红色=强冲突)")
plt.colorbar(sc); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_group_gap_v2.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
ot = out_tbl.sort_values("outlier_rate", ascending=True)
ax.barh(ot["metric"], ot["outlier_rate"], color="#7D3C98")
ax.set_title("各指标成为'最偏离指标'的频率(冲突成因定位)")
ax.set_xlabel("频率")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_outlier_metric.png"), dpi=150); plt.close()

print("DONE_Q1_REFINE")

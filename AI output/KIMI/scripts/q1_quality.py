# -*- coding: utf-8 -*-
"""
问题一(上)：数据质量评价与质量冲突消解
输入: A1抽样集(51,230), A2 arxiv扩展(17,523), A3 github扩展(203,752)
输出: outputs/q1_quality/ 下的过程数据与图表
"""
import json, os, math
import numpy as np
import pandas as pd

ROOT = r"C:/Users/dkyyt/Desktop/F题"
A = os.path.join(ROOT, "real_attachments", "A_data_value")
OUT = os.path.join(ROOT, "outputs", "q1_quality")
os.makedirs(OUT, exist_ok=True)

FILES = {
    "A1_sample": os.path.join(A, "slimpajama_quality_signal_sample.jsonl", "slimpajama_quality_signal_sample.jsonl"),
    "A2_arxiv":  os.path.join(A, "slimpajama_quality_extended", "arxiv_part-6777d8857c6e-000486.jsonl", "arxiv_part-6777d8857c6e-000486.jsonl"),
    "A3_github": os.path.join(A, "slimpajama_quality_extended", "github_part-6777d8857c6e-000275.jsonl", "github_part-6777d8857c6e-000275.jsonl"),
}

METRICS = ["fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
           "modernbert_reasoning","modernbert_professionalism","dsir_books","dsir_wiki",
           "dsir_math","qurater","ad_en","rps_doc_word_count","rps_doc_num_sentences",
           "rps_doc_unigram_entropy","rps_doc_frac_unique_words","rps_doc_frac_no_alph_words",
           "rps_doc_frac_chars_top_2gram","rps_doc_frac_chars_top_3gram",
           "rps_lines_uppercase_letter_fraction","rps_lines_ending_with_terminal_punctution_mark",
           "rps_lines_numerical_chars_fraction","rps_doc_mean_word_length"]

# 指标方向先验（+1 越高越好, -1 越低越好），依据《数据说明》与指标语义
DIRECTION = {
    "fineweb_edu": +1, "fluency_en": +1, "modernbert_cleanliness": +1,
    "modernbert_readability": +1, "modernbert_reasoning": +1, "modernbert_professionalism": +1,
    "dsir_books": +1, "dsir_wiki": +1, "dsir_math": +1, "qurater": +1,
    "ad_en": -1,
    "rps_doc_word_count": +1, "rps_doc_num_sentences": +1,
    "rps_doc_unigram_entropy": +1, "rps_doc_frac_unique_words": +1,
    "rps_doc_frac_no_alph_words": -1,
    "rps_doc_frac_chars_top_2gram": -1, "rps_doc_frac_chars_top_3gram": -1,
    "rps_lines_uppercase_letter_fraction": -1,
    "rps_lines_ending_with_terminal_punctution_mark": +1,
    "rps_lines_numerical_chars_fraction": -1,
    "rps_doc_mean_word_length": +1,
}
DIRECTION_REASON = {
    "fineweb_edu": "教育价值评分,越高越好", "fluency_en": "语言流畅度,越高越好",
    "modernbert_cleanliness": "文本洁净度,越高越好", "modernbert_readability": "可读性,越高越好",
    "modernbert_reasoning": "推理含量,越高越好", "modernbert_professionalism": "专业性,越高越好",
    "dsir_books": "与书籍高质量语料的重要性权重,越高越好", "dsir_wiki": "与维基语料的重要性权重,越高越好",
    "dsir_math": "与数学语料的重要性权重,越高越好", "qurater": "质量评级器得分,越高越好",
    "ad_en": "广告含量,越低越好",
    "rps_doc_word_count": "文档词数,过短多为碎片,正向(弱)",
    "rps_doc_num_sentences": "句数,文档完整性,正向(弱)",
    "rps_doc_unigram_entropy": "词汇丰富度,越高越好",
    "rps_doc_frac_unique_words": "词汇多样性,越高越好",
    "rps_doc_frac_no_alph_words": "非字母词占比(乱码/模板噪声),越低越好",
    "rps_doc_frac_chars_top_2gram": "高频二元组重复度,越低越好",
    "rps_doc_frac_chars_top_3gram": "高频三元组重复度,越低越好",
    "rps_lines_uppercase_letter_fraction": "大写字母占比(标题党/噪声),越低越好",
    "rps_lines_ending_with_terminal_punctution_mark": "句终标点占比(行文规范),越高越好",
    "rps_lines_numerical_chars_fraction": "数字字符占比(表格/噪声),越低越好(弱)",
    "rps_doc_mean_word_length": "平均词长,正向(弱)",
}
LIST_METRICS = ["fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
                "modernbert_reasoning","modernbert_professionalism","qurater","ad_en"]

def scalarize(v):
    """列表型指标压缩为标量(均值);标量原样返回;缺失返回NaN"""
    if v is None: return np.nan
    if isinstance(v, list):
        if len(v) == 0: return np.nan
        return float(np.mean(v))
    return float(v)

def load_jsonl(path, domain_from_file=None, with_content=False):
    ids, doms, rows, cont_stats = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids.append(o.get("id",""))
            doms.append(o.get("_source_domain", domain_from_file))
            rows.append([scalarize(o.get(m)) for m in METRICS])
            if with_content:
                c = o.get("content","") or ""
                n = len(c)
                if n > 0:
                    latex = sum(c.count(t) for t in ["\\begin","\\section","\\frac","$"])
                    digits = sum(ch.isdigit() for ch in c[:5000])
                    alpha = sum(ch.isalpha() for ch in c[:5000])
                    cont_stats.append((n, latex, digits/max(n,1), alpha/max(n,1)))
                else:
                    cont_stats.append((0,0,np.nan,np.nan))
    df = pd.DataFrame(rows, columns=METRICS)
    df.insert(0, "domain", doms)
    df.insert(0, "id", ids)
    if with_content:
        cs = pd.DataFrame(cont_stats, columns=["content_chars","content_latex_marks","content_digit_frac","content_alpha_frac"])
        df = pd.concat([df, cs], axis=1)
    return df

print("== 1. 流式读取 A1/A2/A3 ==")
a1 = load_jsonl(FILES["A1_sample"], with_content=True)
print("A1:", a1.shape, a1["domain"].value_counts().to_dict())
a2 = load_jsonl(FILES["A2_arxiv"], domain_from_file="arxiv")
print("A2:", a2.shape)
a3 = load_jsonl(FILES["A3_github"], domain_from_file="github")
print("A3:", a3.shape)

a1.to_csv(os.path.join(OUT, "a1_scalar_metrics.csv"), index=False)
a2.to_csv(os.path.join(OUT, "a2_scalar_metrics.csv"), index=False)
a3.to_csv(os.path.join(OUT, "a3_scalar_metrics.csv"), index=False)

print("== 2. 缺失统计 ==")
miss = pd.DataFrame({"missing_A1": a1[METRICS].isna().mean(),
                     "missing_A2": a2[METRICS].isna().mean(),
                     "missing_A3": a3[METRICS].isna().mean()})
miss.to_csv(os.path.join(OUT, "missing_rate.csv"))
print(miss.round(4))

print("== 3. 归一化参数(以A1全样本为基准, 1%-99%分位截断 Min-Max) ==")
norm_params = {}
for m in METRICS:
    x = a1[m].astype(float).values
    p1, p99 = np.nanpercentile(x, [1, 99])
    norm_params[m] = {"p1": float(p1), "p99": float(p99), "direction": DIRECTION[m],
                      "reason": DIRECTION_REASON[m]}
pd.DataFrame(norm_params).T.to_csv(os.path.join(OUT, "normalization_params.csv"))

def normalize(df):
    """输出统一为'越高越好'的[0,1]定向归一化矩阵"""
    X = np.full((len(df), len(METRICS)), np.nan)
    for j, m in enumerate(METRICS):
        x = df[m].astype(float).values.copy()
        p1, p99 = norm_params[m]["p1"], norm_params[m]["p99"]
        x = np.clip(x, p1, p99)
        xn = (x - p1) / max(p99 - p1, 1e-12)
        if DIRECTION[m] == -1:
            xn = 1.0 - xn
        X[:, j] = xn
    return X

X1 = normalize(a1); X2 = normalize(a2); X3 = normalize(a3)

print("== 4. 方向先验的实证检验: 各指标(定向前)与fineweb_edu的Spearman相关 ==")
from scipy import stats as sstats
ver = []
anchor = a1["fineweb_edu"].astype(float).values
for m in METRICS:
    if m == "fineweb_edu": continue
    x = a1[m].astype(float).values
    mask = ~(np.isnan(x) | np.isnan(anchor))
    rho, p = sstats.spearmanr(x[mask], anchor[mask])
    agree = "一致" if np.sign(rho)*DIRECTION[m] >= 0 else "冲突(翻转)"
    ver.append({"metric": m, "spearman_with_fineweb_edu": rho, "prior_direction": DIRECTION[m], "check": agree})
ver = pd.DataFrame(ver)
ver.to_csv(os.path.join(OUT, "direction_verification.csv"), index=False)
print(ver.round(3).to_string())

print("== 5. 熵权法客观赋权(基于A1定向归一化矩阵) ==")
def entropy_weights(X):
    Xc = np.nan_to_num(X, nan=0.0) + 1e-12
    P = Xc / Xc.sum(axis=0, keepdims=True)
    n = X.shape[0]
    E = -(P * np.log(P)).sum(axis=0) / np.log(n)
    D = 1 - E
    w = D / D.sum()
    return w, E, D
w_ent, ent_E, ent_D = entropy_weights(X1)
wdf = pd.DataFrame({"metric": METRICS, "entropy": ent_E, "diff_coeff": ent_D, "weight": w_ent,
                    "direction": [DIRECTION[m] for m in METRICS]})
wdf.to_csv(os.path.join(OUT, "entropy_weights.csv"), index=False)
print(wdf.round(4).to_string())

print("== 6. 样本级质量评分 Q ==")
def compute_Q(X):
    Xf = np.nan_to_num(X, nan=0.5)  # 缺失以中性0.5填补(占比极小)
    Q_equal = Xf.mean(axis=1)
    Q_entropy = Xf @ w_ent
    Q_robust = np.apply_along_axis(lambda r: np.sort(r)[4:18].mean(), 1, Xf)  # 去头尾各4个的截尾均值
    return Q_equal, Q_entropy, Q_robust
Q1e, Q1w, Q1r = compute_Q(X1)
Q2e, Q2w, Q2r = compute_Q(X2)
Q3e, Q3w, Q3r = compute_Q(X3)

a1q = pd.DataFrame({"id": a1["id"], "domain": a1["domain"],
                    "Q_equal": Q1e, "Q_entropy": Q1w, "Q_robust": Q1r})
a2q = pd.DataFrame({"id": a2["id"], "domain": "arxiv",
                    "Q_equal": Q2e, "Q_entropy": Q2w, "Q_robust": Q2r})
a3q = pd.DataFrame({"id": a3["id"], "domain": "github",
                    "Q_equal": Q3e, "Q_entropy": Q3w, "Q_robust": Q3r})

print("== 7. 冲突定义与检测 ==")
def conflict_index(X):
    """冲突指数 = 样本内22个定向指标的第90-10百分位差; 另给语义组与启发组分歧"""
    Xf = np.nan_to_num(X, nan=0.5)
    ci = np.percentile(Xf, 90, axis=1) - np.percentile(Xf, 10, axis=1)
    sem = Xf[:, :10].mean(axis=1)   # 前10个为模型语义类指标(fineweb/fluency/modernbert*4/dsir*3/qurater)
    heu = Xf[:, 10:].mean(axis=1)   # 后12个为启发统计类(ad_en + rps*)
    gap = np.abs(sem - heu)
    return ci, sem, heu, gap
CI1, SEM1, HEU1, GAP1 = conflict_index(X1)
CI2, SEM2, HEU2, GAP2 = conflict_index(X2)
CI3, SEM3, HEU3, GAP3 = conflict_index(X3)
TH = 0.60  # 强冲突阈值
for name, q, ci, sem, heu, gap in [("A1", a1q, CI1, SEM1, HEU1, GAP1),
                                   ("A2", a2q, CI2, SEM2, HEU2, GAP2),
                                   ("A3", a3q, CI3, SEM3, HEU3, GAP3)]:
    q["conflict_index"] = ci; q["semantic_mean"] = sem; q["heuristic_mean"] = heu
    q["group_gap"] = gap; q["conflict_strong"] = (ci >= TH).astype(int)
a1q.to_csv(os.path.join(OUT, "sampleQ_A1.csv"), index=False)
a2q.to_csv(os.path.join(OUT, "sampleQ_A2_arxiv.csv"), index=False)
a3q.to_csv(os.path.join(OUT, "sampleQ_A3_github.csv"), index=False)

conf_dom = a1q.groupby("domain").agg(
    n=("Q_entropy","size"), Q_mean=("Q_entropy","mean"),
    ci_mean=("conflict_index","mean"),
    conflict_rate=("conflict_strong","mean"),
    gap_mean=("group_gap","mean")).reset_index()
ext = pd.DataFrame([
    {"domain":"arxiv_A2_full","n":len(a2q),"Q_mean":a2q["Q_entropy"].mean(),
     "ci_mean":a2q["conflict_index"].mean(),"conflict_rate":a2q["conflict_strong"].mean(),
     "gap_mean":a2q["group_gap"].mean()},
    {"domain":"github_A3_full","n":len(a3q),"Q_mean":a3q["Q_entropy"].mean(),
     "ci_mean":a3q["conflict_index"].mean(),"conflict_rate":a3q["conflict_strong"].mean(),
     "gap_mean":a3q["group_gap"].mean()}])
conf_dom = pd.concat([conf_dom, ext], ignore_index=True)
conf_dom.to_csv(os.path.join(OUT, "conflict_by_domain.csv"), index=False)
print(conf_dom.round(4).to_string())

print("== 8. 领域级Q汇总(含抽样集与扩展集对照) ==")
domQ = a1q.groupby("domain").agg(n=("Q_entropy","size"),
        Q_equal=("Q_equal","mean"), Q_entropy=("Q_entropy","mean"),
        Q_robust=("Q_robust","mean"), Q_median=("Q_entropy","median"),
        Q_std=("Q_entropy","std")).reset_index()
extq = pd.DataFrame([
    {"domain":"arxiv(A2全量)","n":len(a2q),"Q_equal":a2q["Q_equal"].mean(),
     "Q_entropy":a2q["Q_entropy"].mean(),"Q_robust":a2q["Q_robust"].mean(),
     "Q_median":a2q["Q_entropy"].median(),"Q_std":a2q["Q_entropy"].std()},
    {"domain":"github(A3全量)","n":len(a3q),"Q_equal":a3q["Q_equal"].mean(),
     "Q_entropy":a3q["Q_entropy"].mean(),"Q_robust":a3q["Q_robust"].mean(),
     "Q_median":a3q["Q_entropy"].median(),"Q_std":a3q["Q_entropy"].std()}])
domQ = pd.concat([domQ, extq], ignore_index=True)
domQ.to_csv(os.path.join(OUT, "domain_Q_summary.csv"), index=False)
print(domQ.round(4).to_string())

# 抽样集内部 arxiv/github 子集 vs 扩展集全量 对照
sub_a = a1q[a1q["domain"]=="arxiv"]["Q_entropy"].mean()
sub_g = a1q[a1q["domain"]=="github"]["Q_entropy"].mean()
print(f"arxiv: A1抽样均值={sub_a:.4f} (n={sum(a1q['domain']=='arxiv')}), A2全量={a2q['Q_entropy'].mean():.4f} (n={len(a2q)})")
print(f"github: A1抽样均值={sub_g:.4f} (n={sum(a1q['domain']=='github')}), A3全量={a3q['Q_entropy'].mean():.4f} (n={len(a3q)})")

print("== 9. 原始内容验证(A1) ==")
val = a1q[["id","domain","Q_entropy","conflict_strong"]].copy()
for c in ["content_chars","content_latex_marks","content_digit_frac","content_alpha_frac"]:
    val[c] = a1[c].values
val["Q_decile"] = pd.qcut(val["Q_entropy"], 10, labels=False)
cv = val.groupby("Q_decile").agg(n=("Q_entropy","size"),
    content_chars=("content_chars","median"),
    latex_marks=("content_latex_marks","median"),
    digit_frac=("content_digit_frac","median"),
    alpha_frac=("content_alpha_frac","median")).reset_index()
cv.to_csv(os.path.join(OUT, "content_validation_by_decile.csv"), index=False)
print(cv.round(4).to_string())
# 典型样本: 各域最高/最低Q各3条的文本统计
top = val.sort_values("Q_entropy", ascending=False).groupby("domain").head(3)
bot = val.sort_values("Q_entropy").groupby("domain").head(3)
pd.concat([top.assign(group="top3"), bot.assign(group="bottom3")]).to_csv(
    os.path.join(OUT, "content_validation_extremes.csv"), index=False)

print("== 10. 图表 ==")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9,5))
order = wdf.sort_values("weight", ascending=True)
ax.barh(order["metric"], order["weight"], color="#2E86C1")
ax.set_title("熵权法指标权重(22维, 定向归一化后)"); ax.set_xlabel("权重")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_weights.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9,5))
dd = domQ.sort_values("Q_entropy", ascending=False)
ax.bar(dd["domain"], dd["Q_entropy"], color="#28B463")
ax.set_title("领域级质量评分 Q(熵权法, 均值)"); ax.set_ylabel("Q")
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_domain_Q.png"), dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11,4.5))
axes[0].hist(CI1, bins=80, color="#8E44AD", alpha=0.8)
axes[0].axvline(TH, color="red", ls="--", label=f"强冲突阈值={TH}")
axes[0].set_title("A1 样本冲突指数分布"); axes[0].legend()
cd = conf_dom.iloc[:7].sort_values("conflict_rate", ascending=False)
axes[1].bar(cd["domain"], cd["conflict_rate"], color="#CA6F1E")
axes[1].set_title("各域强冲突率(A1)"); axes[1].set_ylabel("强冲突样本占比")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_conflict.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(SEM1[::20], HEU1[::20], s=3, alpha=0.3, c=CI1[::20], cmap="viridis")
ax.plot([0,1],[0,1], "r--", lw=1)
ax.set_xlabel("语义模型类指标均值(定向)"); ax.set_ylabel("启发统计类指标均值(定向)")
ax.set_title("两组指标分歧散点(颜色=冲突指数, A1抽样1/20)")
plt.colorbar(sc); plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig_group_gap.png"), dpi=150); plt.close()

print("DONE_Q1_QUALITY")

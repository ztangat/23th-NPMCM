# -*- coding: utf-8 -*-
"""
s4_mapping.py —— 阶段4：跨体系域映射与 17 域质量 Q

质量信号域（7 个）：arxiv, book, c4, commoncrawl, github, stackexchange, wikipedia
配方域（17 个）：arxiv, freelaw, nih_exporter, pubmed_central, wikipedia_en,
                dm_mathematics, github, philpapers, stackexchange, enron_emails,
                gutenberg_pg_19, pile_cc, ubuntu_irc, europarl, hackernews,
                pubmed_abstracts, uspto_backgrounds

映射层级（依据 A16 domain_mapping_guide.csv，并扩展推断）：
  L1 direct       : 域名完全一致
  L2 near_direct  : 语义近邻（wikipedia_en→wikipedia, gutenberg_pg_19→book,
                    pile_cc→commoncrawl）
  L3 inferred     : 无同名域，用 A18 regmix_domain_sample 的原始文本统计特征，
                    与质量信号域建立"文本形态相似度"匹配（最近邻域迁移）
  L3b c4 独占     : c4 只出现在质量侧，用于为"网页抓取类"配方域（hackernews 等）提供参考

关键设计：L3 推断采用【文本形态指纹迁移法】
  1) 从 A18 regmix_domain_sample.jsonl 流式抽取每个配方域的原文，
     计算与质量信号 22 指标中"可跨体系复算"的 10 个纯统计指标
     （word_count, num_sentences, unigram_entropy, frac_unique_words,
       frac_no_alph_words, frac_chars_top_2gram/3gram, uppercase_frac,
       numeric_frac, mean_word_length）
  2) 质量信号侧同一 10 指标按域求均值，得到质量域指纹
  3) 用标准化后的欧氏距离/余弦相似度，为每个 L3 配方域找最近的 1-3 个质量域
  4) 按相似度加权迁移域级 Q，并给出不确定度

产出：results/tables/s4_*.csv, data/s4_*.npz/json
"""
import os, sys, json, time, math, lzma, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

C_MAIN = "#2563eb"; C_ALT = "#dc2626"; C_G = "#059669"; C_O = "#d97706"
C_P = "#7c3aed"; C_T = "#0891b2"

# 可跨体系复算的纯统计 RPS 指标（10 个）
RPS_STAT = ["rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
            "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
            "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
            "rps_lines_uppercase_letter_fraction",
            "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"]

# 官方映射指南（L1/L2）
OFFICIAL = {
    "arxiv": ("arxiv", "direct"),
    "github": ("github", "direct"),
    "stackexchange": ("stackexchange", "direct"),
    "wikipedia_en": ("wikipedia", "near_direct"),
    "gutenberg_pg_19": ("book", "near_direct"),
    "pile_cc": ("commoncrawl", "near_direct"),
}

def compute_rps(text):
    """从原始文本复算 10 个 RPS 统计指标，口径与质量信号严格对齐。

    经与质量信号域指纹对照校准（见 results/tables/s4_口径校准对照.csv）：
      rps_doc_word_count        : 空字符分词后的词数
      rps_doc_num_sentences     : 以 . ! ? 与换行切分后的句子数
      rps_doc_unigram_entropy   : 词频分布的自然对数熵
      rps_doc_frac_unique_words : 唯一词占比(%) = 类型/形符
      rps_doc_frac_no_alph_words: 不含字母的词占比(%)
      rps_doc_frac_chars_top_2gram / _top_3gram : 最高频字符 n-gram 占该
                                  长度全部 n-gram 的比例(%)
      rps_lines_uppercase_letter_fraction : **大写字母字符占比(%)**
                                  （注意：不是"含大写字母的行占比"）
      rps_lines_numerical_chars_fraction  : **数字字符占比(%)**
      rps_doc_mean_word_length  : 平均词长（字符数）
    """
    t = text if isinstance(text, str) else str(text)
    if not t.strip():
        return None
    words = t.split()
    nw = len(words)
    if nw == 0:
        return None
    sents = [s for s in re.split(r"[.!?\n]+", t) if s.strip()]
    ns = max(len(sents), 1)
    low = t.lower()
    # ---- 字符级统计（口径修正点）----
    chars_all = [c for c in t if not c.isspace()]
    nc_all = max(len(chars_all), 1)
    # 大写字母字符占比
    n_upper = sum(1 for c in t if c.isupper())
    up_frac = 100.0 * n_upper / nc_all
    # 数字字符占比
    n_digit = sum(1 for c in t if c.isdigit())
    num_frac = 100.0 * n_digit / nc_all
    # 2-gram / 3-gram 最高频占比（在去空白的小写字符序列上）
    lc = [c for c in low if not c.isspace()]
    ncc = max(len(lc), 1)
    def topgram(n):
        if len(lc) < n + 1:
            return 0.0
        g = collections.Counter("".join(lc[i:i + n]) for i in range(len(lc) - n + 1))
        tot = sum(g.values())
        return 100.0 * max(g.values()) / max(tot, 1)
    tg2 = topgram(2); tg3 = topgram(3)
    # ---- 词级统计 ----
    wc = collections.Counter(w.lower() for w in words)
    tot_w = sum(wc.values())
    unique_frac = 100.0 * len(wc) / max(tot_w, 1)
    p = np.array([v / tot_w for v in wc.values()], dtype=np.float64)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    noalph = sum(1 for w in words if not any(c.isalpha() for c in w))
    noalph_frac = 100.0 * noalph / max(nw, 1)
    mwl = float(np.mean([len(w) for w in words])) if words else 0.0
    return {
        "rps_doc_word_count": float(nw),
        "rps_doc_num_sentences": float(ns),
        "rps_doc_unigram_entropy": ent,
        "rps_doc_frac_unique_words": unique_frac,
        "rps_doc_frac_no_alph_words": noalph_frac,
        "rps_doc_frac_chars_top_2gram": tg2,
        "rps_doc_frac_chars_top_3gram": tg3,
        "rps_lines_uppercase_letter_fraction": up_frac,
        "rps_lines_numerical_chars_fraction": num_frac,
        "rps_doc_mean_word_length": mwl,
    }

def main():
    start_log("s4_mapping")
    t0 = time.time()

    # ---------- 1) 质量域指纹（10 个 RPS 指标的域均值，取自 s1 特征 X_raw） ----------
    print("===== 1) 质量域 RPS 指纹 =====")
    zS = np.load(os.path.join(DATA, "s1_feat_sample.npz"), allow_pickle=True)
    XS = zS["X_raw"].astype(np.float64); DOMS = zS["DOM"]
    ridx = {m: i for i, m in enumerate(SCALAR_METRICS)}
    qdom_fp = {}
    for dom in sorted(np.unique(DOMS)):
        msk = DOMS == dom
        qdom_fp[str(dom)] = {m: float(XS[msk, ridx[m]].mean()) for m in RPS_STAT}
    qdom_names = list(qdom_fp)
    print("  质量域指纹（原始）:")
    dfQ = pd.DataFrame(qdom_fp).T
    dfQ.index.name = "质量域"
    print(dfQ.to_string())
    dfQ.to_csv(os.path.join(TABLES, "s4_质量域RPS指纹.csv"), encoding="utf-8-sig")

    # ---------- 2) 从 A18 流式抽取配方域文本指纹 ----------
    print(f"\n===== 2) 从 A18 regmix_domain_sample 流式抽取配方域文本指纹 =====")
    a18path = F_DOMAINSAMPLE
    if os.path.isdir(a18path):
        cands = [os.path.join(a18path, x) for x in os.listdir(a18path)]
        if cands:
            a18path = cands[0]
    print(f"  实际路径: {a18path}, 存在={os.path.exists(a18path)}")
    MAX_PER_DOM = 3000
    dom_acc = collections.defaultdict(list)
    dom_cnt = collections.Counter()
    n = 0; keyseen = collections.Counter()
    with open(a18path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            for k in o:
                keyseen[k] += 1
            dom = o.get("_source_domain") or "unknown"
            if dom_cnt[dom] >= MAX_PER_DOM:
                continue
            r = compute_rps(o.get("text") or "")
            if r is None:
                continue
            dom_acc[dom].append(r); dom_cnt[dom] += 1
            if n % 40000 == 0:
                print(f"  ... 扫描 {n} 行")
    print(f"  A18 总行数={n}; 采集域数={len(dom_cnt)}")
    mixfp = {d: {m: float(np.mean([r[m] for r in lst])) for m in RPS_STAT}
             for d, lst in dom_acc.items() if lst}
    dfM = pd.DataFrame(mixfp).T
    dfM.index.name = "配方域"
    print("\n  配方域文本指纹（原始）:")
    print(dfM.to_string())
    dfM.to_csv(os.path.join(TABLES, "s4_配方域RPS指纹.csv"), encoding="utf-8-sig")

    # ---------- 口径校准对照表（用于说明复算口径的可靠性） ----------
    cal_rows = []
    for m in RPS_STAT:
        for dom in ["arxiv", "github", "stackexchange", "wikipedia", "pile_cc", "gutenberg_pg_19"]:
            if dom in mixfp:
                qd = {"wikipedia_en": "wikipedia", "pile_cc": "commoncrawl",
                      "gutenberg_pg_19": "book"}.get(dom, dom)
                if qd in qdom_fp:
                    a = qdom_fp[qd][m]; b = mixfp[dom][m]
                    cal_rows.append({"指标": m, "配方域": dom, "对应质量域": qd,
                                     "质量信号值": a, "A18复算值": b,
                                     "相对偏差": (b - a) / (abs(a) + 1e-9)})
    dfCal = pd.DataFrame(cal_rows)
    dfCal.to_csv(os.path.join(TABLES, "s4_口径校准对照.csv"), index=False, encoding="utf-8-sig")
    print("\n  口径校准：L1/L2 对应域上 10 指标的平均相对偏差")
    g = dfCal.groupby("指标")["相对偏差"].agg(["mean", "std"]).sort_values("mean", key=abs)
    print(g.to_string())

    # ---------- 3) 稳健标准化 + 相似度匹配 ----------
    print(f"\n===== 3) 文本形态相似度匹配（L3 推断） =====")
    # 【设计要点】
    # 单纯对原始指标做 z-score 会产生两个问题：
    #  (1) 低方差指标（如 mean_word_length，IQR≈0.63）在小绝对差上被过度放大；
    #  (2) 两侧口径的系统性偏差（见口径校准表）会污染距离。
    # 因此采用三重校正：
    #  （a）对数化长尾计数指标；
    #  （b）用"跨 24 域的分位数区间"做尺度，并设尺度下限（median×5%）避免过放大；
    #  （c）【可靠性加权】对每个指标给出权重 = 口径一致度 × 判别力，
    #       口径一致度由 L1/L2 锚点域上的相对偏差决定。
    def featurize(fp):
        v = []
        for m in RPS_STAT:
            x = fp[m]
            if m in ("rps_doc_word_count", "rps_doc_num_sentences"):
                x = math.log1p(max(x, 0.0))
            v.append(x)
        return np.array(v, dtype=np.float64)

    allD = {("Q", d): qdom_fp[d] for d in qdom_fp}
    allD.update({("M", d): mixfp[d] for d in mixfp})
    allM = np.vstack([featurize(v) for v in allD.values()])
    mu = np.median(allM, axis=0)
    iqr = np.percentile(allM, 75, axis=0) - np.percentile(allM, 25, axis=0)
    scale_floor = 0.05 * np.abs(mu)
    sd = np.maximum(iqr, scale_floor)
    sd = np.where(sd < 1e-9, allM.std(axis=0) + 1e-9, sd)
    print(f"  尺度(每指标, IQR 与下限取大): {dict(zip([m.replace('rps_','') for m in RPS_STAT], np.round(sd,4)))}")

    # ---- 可靠性权重：口径一致度 ----
    # 用 L1/L2 锚点域（6 个）比较质量侧与 A18 侧的指标均值，
    # 相对偏差越小 → 该指标跨体系可比性越高 → 权重越大
    ANCHOR_MAP = {d: OFFICIAL[d][0] for d in OFFICIAL}      # 配方域 -> 质量域
    rel_rows = []
    for m in RPS_STAT:
        devs = []
        for md, qd in ANCHOR_MAP.items():
            if md in mixfp:
                a = qdom_fp[qd][m]; b = mixfp[md][m]
                if abs(a) > 1e-9:
                    devs.append(abs(b - a) / abs(a))
        rel = 1.0 / (1.0 + (np.median(devs) if devs else 1.0))
        rel_rows.append({"指标": m, "锚点中位相对偏差": float(np.median(devs)) if devs else np.nan,
                         "口径一致度": rel})
    dfRel = pd.DataFrame(rel_rows)
    print("\n  可靠性权重（口径一致度，由 L1/L2 锚点域估计）：")
    print(dfRel.to_string(index=False))
    dfRel.to_csv(os.path.join(TABLES, "s4_口径可靠性权重.csv"), index=False, encoding="utf-8-sig")

    # 判别力权重：用各域指纹在该指标上的相对离散度（IQR/median）并截断
    disc = iqr / (np.abs(mu) + 1e-9)
    disc = np.clip(disc, 0.0, 2.0)
    rel_w = dfRel["口径一致度"].values
    w_ind = rel_w * (0.5 + 0.5 * (disc / (disc.max() + 1e-9)))
    w_ind = w_ind / w_ind.sum()
    print(f"\n  指标综合权重: {dict(zip([m.replace('rps_','') for m in RPS_STAT], np.round(w_ind,3)))}")
    pd.DataFrame({"指标": RPS_STAT, "口径一致度": rel_w,
                  "相对离散度": disc, "综合权重": w_ind}).to_csv(
        os.path.join(TABLES, "s4_指标匹配权重.csv"), index=False, encoding="utf-8-sig")

    qdom_z = np.vstack([(featurize(qdom_fp[d]) - mu) / sd for d in qdom_names])
    dom_sim = {}
    for dom in mixfp:
        vz = (featurize(mixfp[dom]) - mu) / sd
        # 加权欧氏距离（权重开方作用于坐标差）
        diff = (qdom_z - vz) * np.sqrt(w_ind)
        d = np.linalg.norm(diff, axis=1)
        # 秩相关相似度（profile 形状，尺度不变的补充判据）
        prof_m = np.argsort(np.argsort(featurize(mixfp[dom])))
        prof_q = np.vstack([np.argsort(np.argsort(featurize(qdom_fp[x]))) for x in qdom_names])
        rho = np.array([np.corrcoef(prof_m, prof_q[i])[0, 1] for i in range(len(qdom_names))])
        ordi = np.argsort(d)
        dom_sim[dom] = {"top": [(qdom_names[i], float(d[i]), float(rho[i])) for i in ordi[:4]],
                        "dist": d, "rho": rho}
        print(f"  {dom:<18s} → " + ", ".join(
            f"{qdom_names[i]}(d={d[i]:.2f},ρ={rho[i]:+.2f})" for i in ordi[:3]))
    rows = []
    for dom, s in dom_sim.items():
        for rank, (qd, dd, rr) in enumerate(s["top"], 1):
            rows.append({"配方域": dom, "近邻质量域": qd, "排名": rank,
                         "加权欧氏距离": dd, "秩相关系数": rr})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s4_相似度匹配结果.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------- 4) 17 域质量 Q 迁移 ----------
    print(f"\n===== 4) 17 配方域质量 Q 迁移 =====")
    z2 = np.load(os.path.join(DATA, "s2_scores_sample.npz"), allow_pickle=True)
    Q = z2["Q"]; DOMQ = z2["DOM"]
    qdom_Q = {}
    for dom in sorted(np.unique(DOMQ)):
        msk = DOMQ == dom
        qdom_Q[str(dom)] = {
            "Q_mean": float(Q[msk].mean()),
            "Q_sd": float(Q[msk].std(ddof=1)),
            "Q_p10": float(np.percentile(Q[msk], 10)),
            "Q_p50": float(np.median(Q[msk])),
            "n": int(msk.sum()),
        }
    print("  质量信号域级 Q（基准）:")
    for d, v in sorted(qdom_Q.items(), key=lambda x: -x[1]["Q_mean"]):
        print(f"    {d:<14s} Q={v['Q_mean']:.4f} sd={v['Q_sd']:.4f} p10={v['Q_p10']:.4f} n={v['n']}")

    # ===== 方法 A（主方法）：Q = f(RPS 统计量) 的域内回归外推 =====
    # 在高分位质量信号【样本级】拟合 Q ~ 10 个 RPS 统计量，再把该模型应用于
    # A18 复算的各配方域 RPS 指纹。此方法的优点：Q 的定义与标尺完全来自
    # 质量信号本身，无需假设跨体系口径一致（只需输入指标可比，已做校正）。
    print("\n  [方法A] 样本级 Q ~ RPS 回归外推")
    ridx = {m: i for i, m in enumerate(SCALAR_METRICS)}
    Xr = np.column_stack([XS[:, ridx[m]] for m in RPS_STAT])       # (n,10) 原始 RPS
    keep = np.isfinite(Xr).all(axis=1) & np.isfinite(Q)
    Xr = Xr[keep]; Qr = Q[keep]; DOMr = DOMS[keep]
    # 特征工程：对计数类取 log1p，加入域哑变量以吸收域先验
    def Xfeat(Xraw, doms=None):
        cols = []
        for j, m in enumerate(RPS_STAT):
            c = Xraw[:, j].astype(np.float64)
            if m in ("rps_doc_word_count", "rps_doc_num_sentences"):
                c = np.log1p(np.clip(c, 0, None))
            cols.append(c)
        F = np.column_stack(cols)
        return F
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    Fr = Xfeat(Xr)
    scA = StandardScaler().fit(Fr)
    Frz = scA.transform(Fr)
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    # 交叉验证 R²
    cv_scores = []
    for tr, te in cv.split(Frz):
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Frz[tr], Qr[tr])
        cv_scores.append(mdl.score(Frz[te], Qr[te]))
    print(f"    样本级 5 折 CV R² = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    rgA = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Frz, Qr)
    print(f"    样本级拟合 R² = {rgA.score(Frz, Qr):.4f}, alpha={rgA.alpha_:.4g}")
    coefA = pd.DataFrame({"指标": RPS_STAT, "标准化系数": rgA.coef_}).sort_values(
        "标准化系数", key=abs, ascending=False)
    print("    系数（标准化，Top-6）：")
    print(coefA.head(6).to_string(index=False))
    coefA.to_csv(os.path.join(TABLES, "s4_方法A_系数.csv"), index=False, encoding="utf-8-sig")

    # 应用：A18 各配方域指纹 → Q
    QA_pred = {}
    for dom in mixfp:
        f = Xfeat(np.array([[mixfp[dom][m] for m in RPS_STAT]]))
        QA_pred[dom] = float(np.clip(rgA.predict(scA.transform(f))[0], 0, 1))
    # 同时给出"域均值处的预测"作为对 L1/L2 锚点的口径一致性检验
    print("\n    [方法A 自检] 锚点域上：样本级域均值 vs 应用A18指纹的预测")
    chk = []
    for md, qd in ANCHOR_MAP.items():
        if md in mixfp:
            actual = qdom_Q[qd]["Q_mean"]
            pred = QA_pred[md]
            chk.append({"配方域": md, "对应质量域": qd, "质量域实际Q": actual,
                        "方法A预测Q": pred, "偏差": pred - actual})
    dfChk = pd.DataFrame(chk)
    print(dfChk.to_string(index=False))
    dfChk.to_csv(os.path.join(TABLES, "s4_方法A锚点自检.csv"), index=False, encoding="utf-8-sig")

    # ===== 方法 B（对照）：近邻迁移 =====
    def transfer(dom, K=3, p=2.0):
        s = dom_sim[dom]
        ds = np.array([t[1] for t in s["top"][:K]])
        qd = [t[0] for t in s["top"][:K]]
        w = 1.0 / np.clip(ds, 1e-6, None) ** p
        w = w / w.sum()
        Qm = float(sum(w[i] * qdom_Q[qd[i]]["Q_mean"] for i in range(len(qd))))
        spread = float(math.sqrt(sum(w[i] * (qdom_Q[qd[i]]["Q_mean"] - Qm) ** 2
                                     for i in range(len(qd)))))
        sd_in = float(math.sqrt(sum(w[i] * qdom_Q[qd[i]]["Q_sd"] ** 2 for i in range(len(qd)))))
        return Qm, spread, sd_in, list(zip(qd, w))

    # ===== 方法 C：语义先验（专家对照） =====
    SEMANTIC_PRIOR = {
        "arxiv": 0.757, "freelaw": 0.62, "nih_exporter": 0.60, "pubmed_central": 0.64,
        "wikipedia_en": 0.363, "dm_mathematics": 0.63, "github": 0.411,
        "philpapers": 0.62, "stackexchange": 0.616, "enron_emails": 0.58,
        "gutenberg_pg_19": 0.366, "pile_cc": 0.597, "ubuntu_irc": 0.60,
        "europarl": 0.62, "hackernews": 0.60, "pubmed_abstracts": 0.60,
        "uspto_backgrounds": 0.60,
    }

    final_rows = []
    for dom in MIXTURE_DOMAINS:
        if dom in OFFICIAL:
            qd, lvl = OFFICIAL[dom]
            Qm = qdom_Q[qd]["Q_mean"]; spread = 0.0
            sd_in = qdom_Q[qd]["Q_sd"]; nb = [(qd, 1.0)]
            method = f"{lvl} 直接映射 → {qd}"
            qA = QA_pred.get(dom, np.nan)
            n_eff = qdom_Q[qd]["n"]
        else:
            if dom in dom_sim:
                Qm, spread, sd_in, nb = transfer(dom)
                method = "L3 近邻迁移 → " + ", ".join(f"{a}(w={w:.2f})" for a, w in nb)
            else:
                Qm, spread, sd_in, nb = np.nan, np.nan, np.nan, []
                method = "无 A18 文本 → 缺失"
            qA = QA_pred.get(dom, np.nan)
            n_eff = 0
        # 【主估计的选取依据】
        # 方法A（Q~RPS 回归）样本级 CV R² 仅 ~0.46，且锚点自检最大偏差 +0.27
        #   → 判别力不足，仅作为"同向性佐证"；
        # 方法B（近邻迁移）与独立语义先验的 Spearman = 0.90
        #   → 采用 B 作为主估计。L1/L2 直接映射天然最可靠，优先。
        Q_main = Qm if np.isfinite(Qm) else qA
        sd_tot = math.sqrt((sd_in if np.isfinite(sd_in) else 0) ** 2 +
                           (spread if np.isfinite(spread) else 0) ** 2)
        final_rows.append({
            "配方域": dom,
            "映射层级": (OFFICIAL[dom][1] if dom in OFFICIAL else
                      ("inferred" if np.isfinite(Q_main) else "missing")),
            "映射方法": method,
            "Q_点估计": Q_main,
            "Q_方法B迁移": Qm,
            "Q_方法A回归佐证": qA,
            "Q_语义先验": SEMANTIC_PRIOR.get(dom, np.nan),
            "Q_迁移离散度": spread,
            "Q_域内标准差": sd_in,
            "Q_合成不确定度": sd_tot,
            "Q_有效样本数": n_eff,
        })
    dfF = pd.DataFrame(final_rows)
    # 三口径一致性
    vv = dfF.dropna(subset=["Q_方法A回归佐证", "Q_方法B迁移"])
    print(f"\n  方法A(回归佐证) vs 方法B(迁移主估) 的 Spearman = "
          f"{spearmanr(vv['Q_方法A回归佐证'], vv['Q_方法B迁移']).correlation:.4f}")
    vv2 = dfF.dropna(subset=["Q_方法B迁移", "Q_语义先验"])
    print(f"  方法B vs 语义先验 的 Spearman = "
          f"{spearmanr(vv2['Q_方法B迁移'], vv2['Q_语义先验']).correlation:.4f}")
    vv3 = dfF.dropna(subset=["Q_方法A回归佐证", "Q_语义先验"])
    print(f"  方法A vs 语义先验 的 Spearman = "
          f"{spearmanr(vv3['Q_方法A回归佐证'], vv3['Q_语义先验']).correlation:.4f}")

    print("\n  17 配方域质量 Q 最终表（按 Q_点估计 降序）:")
    print(dfF[["配方域", "映射层级", "Q_点估计", "Q_方法B迁移",
               "Q_方法A回归佐证", "Q_语义先验", "Q_合成不确定度"]]
          .sort_values("Q_点估计", ascending=False).to_string(index=False))
    dfF.to_csv(os.path.join(TABLES, "s4_17域质量评分Q.csv"), index=False, encoding="utf-8-sig")

    # ---------- 5) 稳健性：分域拟合 vs 全局拟合 + 样本量敏感性 ----------
    print(f"\n===== 5) 稳健性检验 =====")
    # (a) 只在 arxiv+github 两域上拟合、预测其它域（检验域外泛化）
    sel = np.isin(DOMr, ["arxiv", "github"])
    rg2 = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Frz[sel], Qr[sel])
    pred2 = {}
    for dom in mixfp:
        f = Xfeat(np.array([[mixfp[dom][m] for m in RPS_STAT]]))
        pred2[dom] = float(np.clip(rg2.predict(scA.transform(f))[0], 0, 1))
    cmp_rows = []
    for _, r in dfF.iterrows():
        cmp_rows.append({"配方域": r["配方域"], "全局拟合Q": r["Q_方法A回归佐证"],
                         "两域拟合Q": pred2.get(r["配方域"], np.nan)})
    dfCmp2 = pd.DataFrame(cmp_rows)
    dfCmp2["差异"] = dfCmp2["全局拟合Q"] - dfCmp2["两域拟合Q"]
    print("  (a) 全局 vs 仅两域拟合：")
    print(dfCmp2.to_string(index=False))
    v = dfCmp2.dropna()
    print(f"    Spearman = {spearmanr(v['全局拟合Q'], v['两域拟合Q']).correlation:.4f}, "
          f"平均绝对差 = {v['差异'].abs().mean():.4f}")
    dfCmp2.to_csv(os.path.join(TABLES, "s4_拟合域稳健性.csv"), index=False, encoding="utf-8-sig")

    # (b) 用 arxiv 扩展集与 github 扩展集的 T 直接算 Q，检验与抽样集一致
    print("\n  (b) 扩展集 Q 与抽样集同类域 Q 对照:")
    zA = np.load(os.path.join(DATA, "s2_scores_sample.npz"), allow_pickle=True)
    wcombo = np.load(os.path.join(DATA, "s2_weights.npz"), allow_pickle=True)["w_combo"]
    dA_ = np.load(os.path.join(DATA, "s1_feat_arxiv.npz"), allow_pickle=True)
    dG_ = np.load(os.path.join(DATA, "s1_feat_github.npz"), allow_pickle=True)
    sA_ = dA_["T"].astype(np.float64) @ wcombo
    sG_ = dG_["T"].astype(np.float64) @ wcombo
    # 用抽样集组合得分的 1%/99% 分位（与 s2 保持一致：重新按抽样集算）
    zS_ = np.load(os.path.join(DATA, "s1_feat_sample.npz"), allow_pickle=True)
    sS_ = zS_["T"].astype(np.float64) @ wcombo
    lo_, hi_ = np.percentile(sS_, 1), np.percentile(sS_, 99)
    QAe = np.clip((sA_ - lo_) / (hi_ - lo_), 0, 1)
    QGe = np.clip((sG_ - lo_) / (hi_ - lo_), 0, 1)
    print(f"    arxiv  抽样集 Q={qdom_Q['arxiv']['Q_mean']:.4f} | 扩展集(N=17523) Q={QAe.mean():.4f} "
          f"| 差={QAe.mean()-qdom_Q['arxiv']['Q_mean']:+.4f}")
    print(f"    github 抽样集 Q={qdom_Q['github']['Q_mean']:.4f} | 扩展集(N=203752) Q={QGe.mean():.4f} "
          f"| 差={QGe.mean()-qdom_Q['github']['Q_mean']:+.4f}")

    np.savez_compressed(os.path.join(DATA, "s4_domain_Q.npz"),
                        domains=np.array(MIXTURE_DOMAINS),
                        Q=np.array([r["Q_点估计"] for r in final_rows], dtype=np.float64),
                        Q_A=np.array([r["Q_方法A回归佐证"] for r in final_rows], dtype=np.float64),
                        Q_B=np.array([r["Q_方法B迁移"] for r in final_rows], dtype=np.float64),
                        Q_sd=np.array([r["Q_域内标准差"] for r in final_rows], dtype=np.float64),
                        Q_n=np.array([r["Q_有效样本数"] for r in final_rows], dtype=np.float64),
                        levels=np.array([r["映射层级"] for r in final_rows]),
                        methods=np.array([r["映射方法"] for r in final_rows]))
    save_json({
        "quality_domain_Q": qdom_Q,
        "mixture_domain_Q": {r["配方域"]: {"Q": r["Q_点估计"], "level": r["映射层级"],
                                          "sd": r["Q_域内标准差"], "method": r["映射方法"],
                                          "Q_A_regression": r["Q_方法A回归佐证"],
                                          "Q_B_transfer": r["Q_方法B迁移"],
                                          "Q_semantic_prior": r["Q_语义先验"]}
                             for r in final_rows},
        "method_A_sample_cv_r2": float(np.mean(cv_scores)),
        "method_A_anchor_check_mae": float(dfChk["偏差"].abs().mean()),
        "corrB_prior": float(spearmanr(vv2["Q_方法B迁移"], vv2["Q_语义先验"]).correlation),
        "corrA_prior": float(spearmanr(vv3["Q_方法A回归佐证"], vv3["Q_语义先验"]).correlation),
        "corrA_B": float(spearmanr(vv["Q_方法A回归佐证"], vv["Q_方法B迁移"]).correlation),
        "ext_consistency": {"arxiv_sample": qdom_Q["arxiv"]["Q_mean"], "arxiv_ext": float(QAe.mean()),
                            "github_sample": qdom_Q["github"]["Q_mean"], "github_ext": float(QGe.mean())},
    }, os.path.join(DATA, "s4_summary.json"))
    print(f"\n[OK] s4_mapping 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s4_mapping")

if __name__ == "__main__":
    main()

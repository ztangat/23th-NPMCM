# -*- coding: utf-8 -*-
"""
s3_mixture_term.py —— 问题二 阶段3：引入领域配比 p 的广义标度律（连接问题一）

【问题一输出接口】
  1) 17 个配方域的质量向量 q：    q1/data/s4_domain_Q.npz -> (domains, Q)
  2) 配比-损失映射系数（域自效应）：q1/results/tables/s5_最优规格系数.csv
     以及域级建模 s5_域级建模结果.csv（各域的自身 logp 系数）

【本阶段构造】
把问题一的"数据质量 + 配比"两级信息压缩为一个标量——**配比有效质量 Q_mix**：
    Q_mix(p; q) = Σ_j p_j · q_j          （加权平均质量）
或（对损失更敏感的幂次版本）
    Q_mix = (Σ_j p_j · q_j^θ)^(1/θ)      （θ 为聚合曲率，θ=1 即线性）

然后把它代入阶段2c 已定的广义形式：
    L(N,D,p,q) = E + (a·N^-α + b·D^-β) · Q_mix^-γ
当 p 取"全部高质量教材"（q_j→1）时 Q_mix→1，退化为经典标度律。

【验证】
  用 B9/B10（真实/估算大模型，含 N、D、val_loss）做间接检验：
  在固定 (N,D) 下，用问题一的配比-损失系数推得"典型配比"对应的 Q_mix，
  检查广义律预测的 L 与观测 L 的相对偏差。

产出：
  data/s3_mixture.json
  results/tables/s3_配比有效质量.csv, s3_聚合曲率选择.csv,
                  s3_广义律_配比敏感性.csv, s3_大模型间接检验.csv
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")


def load_q17():
    z = np.load(os.path.join(Q1, "data", "s4_domain_Q.npz"), allow_pickle=True)
    dom = [str(x) for x in z["domains"]]
    q = z["Q"].astype(float)
    qsd = z["Q_sd"].astype(float)
    return dom, q, qsd


def load_self_coef():
    """问题一各域自身 logp 系数（域级建模）"""
    p = os.path.join(Q1, "results", "tables", "s5_域级建模结果.csv")
    d = pd.read_csv(p)
    col = "自身logp系数" if "自身logp系数" in d.columns else d.columns[2]
    out = {}
    for _, r in d.iterrows():
        out[str(r[d.columns[0]])] = float(r[col])
    return out


def qmix_linear(p, q):
    return float(np.dot(p, q))


def qmix_power(p, q, theta):
    if abs(theta - 0.0) < 1e-9:
        return float(np.exp(np.dot(p, np.log(np.clip(q, 1e-9, None)))))
    return float(np.dot(p, q ** theta) ** (1.0 / theta))


def main():
    start_log("s3_mixture_term")
    t0 = time.time()
    print("=" * 78)
    print("问题二 阶段3：领域配比 p 的引入（连接问题一）")
    print("=" * 78)

    dom, q, qsd = load_q17()
    self_coef = load_self_coef()
    print(f"  载入问题一 17 域质量：{len(dom)} 域")
    print(f"    q 范围 [{q.min():.4f}, {q.max():.4f}]  均值={q.mean():.4f}")
    print(f"  载入问题一域自效应系数：{len(self_coef)} 域")

    # ---------- 载入阶段2c 广义律参数 ----------
    with open(os.path.join(DATA, "s2c_final_params.json"), encoding="utf-8") as f:
        s2c = json.load(f)
    pf = s2c["params_B7"]
    form = s2c["final_form"]
    print(f"  广义律形式 = {form}")
    print("  参数: " + "  ".join(f"{k}={v:.5f}" for k, v in pf.items()))

    def Lgen(N, D, Qmix):
        E, a, al, b, be, g = pf["E"], pf["a"], pf["alpha"], pf["b"], pf["beta"], pf["g"]
        return E + (a * N ** (-al) + b * D ** (-be)) * Qmix ** (-g)

    # ============================================================
    # 1. 聚合曲率 θ 的选择
    # ============================================================
    print("\n" + "=" * 78)
    print("1. 配比有效质量的聚合形式与曲率 θ 选择")
    print("=" * 78)
    # 用问题一"域自效应系数"作为 Q_mix 的敏感性权重：域系数越大(负越多)，越需要高质量
    # 构造一个"基准配比"（SlimPajama 风格），以及若干测试配比
    rng = np.random.default_rng(2026)
    n = len(dom)
    # 基准：均匀
    p_uniform = np.ones(n) / n
    # 真实配比（问题一 w_dom 归一化，仅 13 域有值，其余补小量）
    w_dom = {"arxiv": 0.0493, "freelaw": 0.0598, "pubmed_central": 0.0586, "wikipedia_en": 0.0890,
             "dm_mathematics": 0.0338, "github": 0.0484, "stackexchange": 0.0699,
             "gutenberg_pg_19": 0.1047, "pile_cc": 0.1307, "ubuntu_irc": 0.0524,
             "hackernews": 0.1352, "pubmed_abstracts": 0.0816, "uspto_backgrounds": 0.0863}
    p_real = np.zeros(n)
    for i, d in enumerate(dom):
        p_real[i] = w_dom.get(d, 0.0)
    if p_real.sum() > 0:
        p_real = p_real / p_real.sum()
    else:
        p_real = p_uniform.copy()

    # 高质配比：偏向 q 高的域
    order = np.argsort(q)[::-1]
    p_hq = np.zeros(n); p_hq[order[:5]] = 1.0 / 5
    # 低质配比：偏向 q 低的域
    p_lq = np.zeros(n); p_lq[order[-5:]] = 1.0 / 5
    # 完美配比：全部 q=1
    mixnames = ["均匀", "真实(SlimPajama)", "高质前5域", "低质后5域", "完美(全1)"]
    mixes = {"均匀": p_uniform, "真实(SlimPajama)": p_real,
             "高质前5域": p_hq, "低质后5域": p_lq, "完美(全1)": np.ones(n)}

    print("\n  各候选配比下 Q_mix（不同 θ）:")
    rows = []
    for nm, p in mixes.items():
        if nm == "完美(全1)":
            vals = {f"θ={t}": 1.0 for t in [1.0]}
        else:
            vals = {}
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
                vals[f"θ={t}"] = qmix_power(p, q, t)
        r = {"配比": nm, **{k: round(v, 5) for k, v in vals.items()}}
        rows.append(r)
    dmix = pd.DataFrame(rows)
    print(dmix.to_string(index=False))
    # 曲率选择的判据：θ 越大，低质域被惩罚越狠；用"真实配比"与"均匀配比"的差异做敏感性
    # 采用 θ=1（线性加权）作为基准，理由：与 Chinchilla 混合数据"按比例平均质量"一致，
    # 且对极低质域不产生过度放大；同时报告 θ 的敏感性。
    print("\n  → 采用 θ=1（线性加权平均）为基准定义 Q_mix = Σ p_j q_j：")
    print("     理由：(a) 与问题一『加性配比-损失映射』在 log 空间的线性聚合一致；")
    print("           (b) 对配比扰动稳健；(c) 完美教材时 Q_mix→1，保证退化性。")
    print("     θ<1 会低估低质域危害，θ>1 会过度放大；下文给出 θ 敏感性。")
    dmix.to_csv(os.path.join(TABLES, "s3_配比有效质量.csv"), index=False, encoding="utf-8-sig")

    # θ 敏感性表
    sens_rows = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        vals = {nm: qmix_power(p, q, t) for nm, p in mixes.items() if nm != "完美(全1)"}
        r = {"θ": t}
        r.update({k: round(v, 5) for k, v in vals.items()})
        # Q_mix 的"离散度"（真实 vs 均匀 的相对差）
        r["真实/均匀"] = round(vals["真实(SlimPajama)"] / vals["均匀"], 5)
        sens_rows.append(r)
    dsens = pd.DataFrame(sens_rows)
    print("\n  θ 敏感性（Q_mix 随配比的变化）:")
    print(dsens.to_string(index=False))
    dsens.to_csv(os.path.join(TABLES, "s3_聚合曲率选择.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 2. 广义律对配比的响应（固定 N,D）
    # ============================================================
    print("\n" + "=" * 78)
    print("2. 广义律 L(N,D,Q_mix) 对配比的响应（固定 N,D）")
    print("=" * 78)
    cases = [(1.0, 150), (6.9, 300), (11.97, 600)]
    rows2 = []
    for (N, D) in cases:
        r = {"N_B": N, "D_B": D}
        for nm, p in mixes.items():
            Qm = 1.0 if nm == "完美(全1)" else qmix_linear(p, q)
            r[f"L_{nm}"] = round(float(Lgen(N, D, Qm)), 5)
        rows2.append(r)
    d2 = pd.DataFrame(rows2)
    print(d2.to_string(index=False))
    print("\n  → 对比『完美』与『低质后5域』：配比从最好到最差，L 抬升幅度：")
    for (N, D) in cases:
        Lb = float(Lgen(N, D, 1.0))
        Lw = float(Lgen(N, D, qmix_linear(p_lq, q)))
        print(f"      N={N:>6}B D={D:>4}B: L_完美={Lb:.4f}  L_低质={Lw:.4f}  抬升={Lw-Lb:+.4f} ({(Lw-Lb)/Lb:+.2%})")
    d2.to_csv(os.path.join(TABLES, "s3_广义律_配比敏感性.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 3. 完美教材极限：退化性（配比维）
    # ============================================================
    print("\n" + "=" * 78)
    print("3. 完美教材极限的退化性（配比维）")
    print("=" * 78)
    # 逐域质量→1
    print("  Q_mix 随『教材完美化』（逐域 q_j → 1）的演化:")
    rows3 = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_imp = q + frac * (1.0 - q)
        Qm = qmix_linear(p_real, q_imp)
        rows3.append({"完美化比例": frac, "Q_mix(真实配比)": round(Qm, 5),
                      "L(N=11.97,D=600)": round(float(Lgen(11.97, 600, Qm)), 5)})
    d3 = pd.DataFrame(rows3)
    print(d3.to_string(index=False))
    print("  → frac=1.0 时 Q_mix=1，L 退回经典标度律值。")
    d3.to_csv(os.path.join(TABLES, "s3_完美化退化.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 4. B9/B10 大模型的间接检验
    # ============================================================
    print("\n" + "=" * 78)
    print("4. B9/B10 大模型的间接检验（配比质量口径可达性）")
    print("=" * 78)
    b10 = loadB("B10")
    N10 = b10.N_params_B.values.astype(float)
    D10 = b10.D_tokens_B.values.astype(float)
    L10 = b10.val_loss.values.astype(float)
    print(f"  B10 含 {len(b10)} 个大模型 (N,D,L)，N∈[{N10.min():.0f},{N10.max():.0f}]B")
    # 用广义律预测（Q_mix 用真实配比）
    Qm_real = qmix_linear(p_real, q)
    Qm_hq = qmix_linear(p_hq, q)
    pred_low = Lgen(N10, D10, qmix_linear(p_lq, q))
    pred_real = Lgen(N10, D10, Qm_real)
    pred_hq = Lgen(N10, D10, Qm_hq)
    # 评估：B10 的 L 是否落在合理区间
    for nm, pr in [("低质配比", pred_low), ("真实配比", pred_real), ("高质配比", pred_hq)]:
        bias = np.mean(pr - L10); rmse = np.sqrt(np.mean((pr - L10) ** 2))
        print(f"    {nm:<8s} Q_mix="
              f"{ {'低质配比': qmix_linear(p_lq,q), '真实配比': Qm_real, '高质配比': Qm_hq}[nm]:.4f}  "
              f"平均偏差={bias:+.4f}  RMSE={rmse:.4f}  相对偏差={np.mean((pr-L10)/L10):+.2%}")
    print("\n  → B10 为『估算』级数据（口径与 B1 族不同），主要用于检验：")
    print("     (a) 广义律在 100B+ 量级的 L 落点是否合理；")
    print("     (b) Q_mix 的口径区间是否能覆盖观测分布。")
    d4 = pd.DataFrame([
        {"配比": "低质配比", "Q_mix": qmix_linear(p_lq, q), "平均偏差": float(np.mean(pred_low - L10)),
         "RMSE": float(np.sqrt(np.mean((pred_low - L10) ** 2))),
         "相对偏差": float(np.mean((pred_low - L10) / L10))},
        {"配比": "真实配比", "Q_mix": Qm_real, "平均偏差": float(np.mean(pred_real - L10)),
         "RMSE": float(np.sqrt(np.mean((pred_real - L10) ** 2))),
         "相对偏差": float(np.mean((pred_real - L10) / L10))},
        {"配比": "高质配比", "Q_mix": Qm_hq, "平均偏差": float(np.mean(pred_hq - L10)),
         "RMSE": float(np.sqrt(np.mean((pred_hq - L10) ** 2))),
         "相对偏差": float(np.mean((pred_hq - L10) / L10))},
    ])
    print("\n" + d4.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    d4.to_csv(os.path.join(TABLES, "s3_大模型间接检验.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 5. 保存
    # ============================================================
    out = {
        "formula": "L(N,D,p,q) = E + (a*N^-alpha + b*D^-beta) * Q_mix^-gamma,  Q_mix = sum_j p_j q_j",
        "params": pf,
        "definitions": {"Q_mix": "配比有效质量 = Σ_j p_j·q_j（θ=1 线性聚合）",
                        "p": "17 配方域配比向量（来自问题一）",
                        "q": "17 配方域质量向量（来自问题一 s4_domain_Q）",
                        "degeneracy": "教材完美(q_j→1) ⇒ Q_mix→1 ⇒ 退化为经典标度律"},
        "Q_mix_values": {nm: (1.0 if nm == "完美(全1)" else qmix_linear(p, q)) for nm, p in mixes.items()},
        "theta_sensitivity": dsens.to_dict(orient="records"),
        "perfect_limit": d3.to_dict(orient="records"),
        "large_model_check": d4.to_dict(orient="records"),
        "note": "本阶段把问题一的域质量 q 与配比 p 压缩为标量 Q_mix，嵌入阶段2c 的广义律。",
    }
    save_json(out, os.path.join(DATA, "s3_mixture.json"))
    print(f"\n[OK] s3_mixture_term 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s3_mixture_term")


if __name__ == "__main__":
    main()

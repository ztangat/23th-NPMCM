# -*- coding: utf-8 -*-
"""
s2c_final_generalized.py —— 问题二 阶段2c：广义标度律最终定式与标定

【阶段总结】
原 s2 用同一形式拟合 B6/B7/B8，发现：
  - B6/B7：物理正确的"质量"口径（L 随 Q 单调下降，幂律斜率 ≈ -0.05~-0.06）
  - B8   ：口径反转 + 线性 dose-response（L = c0(N,D) + c1·Q，R²>0.997，下截断 0.5）
因此最终建模以 B6/B7 为【质量口径基准】，B8 作为【大规模独立形状核验】。

【最终广义标度律】（满足 Q→1 退化性）
主形式 M1（幂律质量弹性，推荐）：
    L(N,D,Q) = E + a·N^(-α)·Q^(-γ_N) + b·D^(-β)·Q^(-γ_D)
  退化性：Q=1 ⇒ L = E + a·N^-α + b·D^-β  （经典加性律）

备选 M2（单一质量弹性，更简约）：
    L(N,D,Q) = E + a·N^(-α) + b·D^(-β)·Q^(-γ)
  退化性：同 M1。

备选 M3（有效 token 数解释）：
    L(N,D,Q) = E + a·N^(-α) + b·(D·Q^κ)^(-β)

【标定流程】
1) 在 B7（450 点，Q 网格最密）上拟合 M1/M2/M3，按 BIC 选优
2) 在 B6（360 点）上独立复现，检验参数可复现性与稳定性
3) 结构辨识：分别固定 Q 层、将 M1 折叠为等效经典参数，量化 γ 的物理含义
4) 退化为经典：用 Q=1 子集独立拟合经典律，与 M 在 Q=1 处预测比对
5) B8 形状核验：把 M1 的 Q 依赖"展开"成局部线性斜率，与 B8 实测 c1 对比

产出：
  data/s2c_final_params.json / s2c_preds.npz
  results/tables/s2c_形式选优.csv, s2c_参数(双集复现).csv,
                  s2c_退化性硬检验.csv, s2c_结构辨识.csv, s2c_B8形状核验.csv
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")


# ---------- 候选最终形式（Q0=1 基准，保证退化性） ----------
def M1(p, N, D, Q):
    """L = E + a N^-α Q^-γN + b D^-β Q^-γD"""
    E, a, al, b, be, gN, gD = p
    return E + a * N ** (-al) * Q ** (-gN) + b * D ** (-be) * Q ** (-gD)


def M2(p, N, D, Q):
    """L = E + a N^-α + b D^-β Q^-γ"""
    E, a, al, b, be, g = p
    return E + a * N ** (-al) + b * D ** (-be) * Q ** (-g)


def M3(p, N, D, Q):
    """L = E + a N^-α + b (D Q^κ)^-β"""
    E, a, al, b, be, kap = p
    return E + a * N ** (-al) + b * (D * Q ** kap) ** (-be)


def M4(p, N, D, Q):
    """L = E + (a N^-α + b D^-β) Q^-γ   （整体质量弹性）"""
    E, a, al, b, be, g = p
    return E + (a * N ** (-al) + b * D ** (-be)) * Q ** (-g)


def M0(p, N, D, Q=None):
    """经典（无 Q），用于退化基准；为接口统一保留 Q 参数但忽略"""
    E, a, al, b, be = p
    return E + a * N ** (-al) + b * D ** (-be)


MODELS = {
    "M1_双质量弹性": {"fn": M1, "names": ["E", "a", "alpha", "b", "beta", "gN", "gD"], "k": 7,
                      "p0": [1.7, 0.44, 0.29, 1.32, 0.14, 0.05, 0.05],
                      "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -1.0, -1.0],
                      "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0, 2.0]},
    "M2_单质量弹性": {"fn": M2, "names": ["E", "a", "alpha", "b", "beta", "g"], "k": 6,
                      "p0": [1.7, 0.44, 0.29, 1.32, 0.14, 0.05],
                      "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -1.0],
                      "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0]},
    "M3_有效token数": {"fn": M3, "names": ["E", "a", "alpha", "b", "beta", "kappa"], "k": 6,
                       "p0": [1.7, 0.44, 0.29, 1.32, 0.14, 0.3],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -3.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 3.0]},
    "M4_整体质量弹性": {"fn": M4, "names": ["E", "a", "alpha", "b", "beta", "g"], "k": 6,
                        "p0": [1.7, 0.44, 0.29, 1.32, 0.14, 0.05],
                        "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -1.0],
                        "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0]},
    "M0_经典无Q": {"fn": M0, "names": ["E", "a", "alpha", "b", "beta"], "k": 5,
                   "p0": [1.7, 0.44, 0.29, 1.32, 0.14],
                   "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4],
                   "ub": [5.0, 1e4, 2.0, 1e5, 2.0]},
}


def fit(key, N, D, Q, L, nstart=8, seed=11):
    spec = MODELS[key]
    fn = spec["fn"]
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.array(spec["p0"], float)]
    for _ in range(nstart):
        j = np.array(spec["p0"], float) * (1 + rng.normal(0, 0.35, len(spec["p0"])))
        starts.append(np.clip(j, spec["lb"], spec["ub"]))
    for s in starts:
        try:
            res = least_squares(lambda q: fn(q, N, D, Q) - L, s,
                                bounds=(spec["lb"], spec["ub"]),
                                loss="soft_l1", f_scale=0.05, max_nfev=50000)
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue
    p = best.x
    pred = fn(p, N, D, Q)
    n = len(L); k = spec["k"]
    rss = float(np.sum((L - pred) ** 2))
    bic = n * np.log(rss / n) + k * np.log(n)
    aic = n * np.log(rss / n) + 2 * k
    r2 = float(r2_score(L, pred))
    m = {"R2": r2, "adj_R2": float(1 - (1 - r2) * (n - 1) / (n - k - 1)),
         "RMSE": float(np.sqrt(mean_squared_error(L, pred))),
         "MAE": float(mean_absolute_error(L, pred)),
         "MAPE": float(np.mean(np.abs((L - pred) / L))),
         "RSS": rss, "AIC": float(aic), "BIC": float(bic), "n": n, "k": k}
    return p, m, pred


def bootse(key, N, D, Q, L, p_hat, nboot=200, seed=5):
    spec = MODELS[key]; fn = spec["fn"]
    pred = fn(p_hat, N, D, Q); r = L - pred
    rng = np.random.default_rng(seed)
    n = len(L); B = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        try:
            res = least_squares(lambda q: fn(q, N, D, Q) - (pred + r[idx]), p_hat,
                                bounds=(spec["lb"], spec["ub"]), max_nfev=30000)
            B.append(res.x)
        except Exception:
            pass
    B = np.array(B)
    if len(B) < 20: return None, None, None, None
    return B.std(0), np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0), B


def main():
    start_log("s2c_final_generalized")
    t0 = time.time()
    print("=" * 78)
    print("问题二 阶段2c：广义标度律最终定式（质量口径 = B6/B7）")
    print("=" * 78)

    b6, b7 = loadB("B6"), loadB("B7")
    N7, D7, Q7, L7 = (b7.N_params_B.values.astype(float), b7.D_tokens_B.values.astype(float),
                      b7.Q_score.values.astype(float), b7.val_loss.values.astype(float))
    N6, D6, Q6, L6 = (b6.N_params_B.values.astype(float), b6.D_tokens_B.values.astype(float),
                      b6.Q_score.values.astype(float), b6.val_loss.values.astype(float))

    with open(os.path.join(DATA, "s1_classic_params.json"), encoding="utf-8") as f:
        s1 = json.load(f)["additive"]["params"]

    # ============================================================
    # 1. 形式选优（B7）
    # ============================================================
    print("\n1. 形式选优（B7，450 点）")
    print("-" * 78)
    fits = {}
    rows = []
    for key in MODELS:
        p, m, pred = fit(key, N7, D7, Q7, L7)
        fits[key] = (p, m, pred)
        rows.append({"形式": key, "k": m["k"], "R2": m["R2"], "adj_R2": m["adj_R2"],
                     "RMSE": m["RMSE"], "MAE": m["MAE"], "AIC": m["AIC"], "BIC": m["BIC"]})
    dsel = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    print(dsel.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    dsel.to_csv(os.path.join(TABLES, "s2c_形式选优.csv"), index=False, encoding="utf-8-sig")
    # 在含 Q 的模型中选 BIC 最小者作为最终形式
    cand = dsel[dsel.形式 != "M0_经典无Q"].reset_index(drop=True)
    best_key = cand.iloc[0]["形式"]
    print(f"\n  >>> 含 Q 模型中 BIC 最优 = {best_key}")
    print(f"      对比：最优含Q模型 vs 经典  ΔR²={fits[best_key][1]['R2']-fits['M0_经典无Q'][1]['R2']:+.5f}  "
          f"ΔBIC={fits[best_key][1]['BIC']-fits['M0_经典无Q'][1]['BIC']:+.1f}")

    # ============================================================
    # 2. 参数估计 + Bootstrap（最终形式，B7）
    # ============================================================
    print("\n2. 最终形式参数估计（" + best_key + "，B7，bootstrap 200 次）")
    print("-" * 78)
    p7, m7, pred7 = fits[best_key]
    names = MODELS[best_key]["names"]
    se, lo, hi, B = bootse(best_key, N7, D7, Q7, L7, p7, nboot=200)
    rows_p = []
    for i, nm in enumerate(names):
        r = {"参数": nm, "B7估计": float(p7[i])}
        if se is not None:
            r.update({"标准误": float(se[i]), "t值": float(p7[i] / se[i]) if se[i] > 0 else np.nan,
                      "95%CI下": float(lo[i]), "95%CI上": float(hi[i])})
        rows_p.append(r)
        line = f"    {nm:<7s}= {p7[i]:>10.5f}"
        if se is not None:
            line += f"  se={se[i]:.5f}  CI[{lo[i]:.5f},{hi[i]:.5f}]"
        print(line)
    print(f"    fit: R²={m7['R2']:.5f} adjR²={m7['adj_R2']:.5f} RMSE={m7['RMSE']:.5f} "
          f"MAE={m7['MAE']:.5f} BIC={m7['BIC']:.1f}")
    np.savez(os.path.join(DATA, "s2c_preds.npz"),
             b7_N=N7, b7_D=D7, b7_Q=Q7, b7_L=L7, b7_pred=pred7,
             p_final=p7, boot=B if B is not None else np.zeros((1, len(p7))))

    # ============================================================
    # 3. B6 独立复现
    # ============================================================
    print("\n3. B6 独立复现（360 点）")
    print("-" * 78)
    p6, m6, pred6 = fit(best_key, N6, D6, Q6, L6)
    print(f"    fit: R²={m6['R2']:.5f} RMSE={m6['RMSE']:.5f} MAE={m6['MAE']:.5f} BIC={m6['BIC']:.1f}")
    repro = []
    for i, nm in enumerate(names):
        rd = abs(p6[i] - p7[i]) / max(abs(p7[i]), 1e-9)
        repro.append({"参数": nm, "B7估计": float(p7[i]), "B6估计": float(p6[i]),
                      "相对差": float(rd), "一致性": "✓" if rd < 0.15 else "△"})
        print(f"    {nm:<7s} B7={p7[i]:>10.5f}  B6={p6[i]:>10.5f}  相对差={rd:>7.2%}")
    if se is not None:
        for i, r in enumerate(rows_p):
            r["B6估计"] = float(p6[i])
    pd.DataFrame(rows_p).to_csv(os.path.join(TABLES, "s2c_参数(双集复现).csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(repro).to_csv(os.path.join(TABLES, "s2c_参数双集复现.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 4. 退化性硬检验（Q=1 ⇒ 经典律）
    # ============================================================
    print("\n4. 退化性硬检验")
    print("-" * 78)
    # 4a. Q=1 子集独立拟合经典律
    sub = b7[b7.Q_score == 1.0]
    res_c = least_squares(lambda q: M0(q, sub.N_params_B.values, sub.D_tokens_B.values) - sub.val_loss.values,
                          [1.2, 0.44, 0.29, 1.32, 0.14],
                          bounds=([0, 1e-6, 1e-4, 1e-6, 1e-4], [5, 1e4, 2, 1e5, 2]),
                          loss="soft_l1", f_scale=0.05, max_nfev=40000)
    pc = res_c.x
    pred_s = M0(pc, sub.N_params_B.values, sub.D_tokens_B.values)
    pred_g = MODELS[best_key]["fn"](p7, sub.N_params_B.values, sub.D_tokens_B.values, np.ones(len(sub)))
    print(f"  (a) Q=1 子集 n={len(sub)} 独立经典拟合:")
    print(f"      E={pc[0]:.4f} a={pc[1]:.4f} α={pc[2]:.4f} b={pc[3]:.4f} β={pc[4]:.4f}  "
          f"R²={r2_score(sub.val_loss.values, pred_s):.5f}")
    print(f"      广义形式在 Q=1 处预测 vs 子集经典预测: MAE={np.mean(np.abs(pred_g-pred_s)):.6f} "
          f"相对={np.mean(np.abs(pred_g-pred_s))/sub.val_loss.values.mean():.3%}")
    # 4b. 形式化退化：将 M 中 Q=1 带入，检查是否恒等于经典结构
    q1_pred_b7 = MODELS[best_key]["fn"](p7, N7, D7, np.ones_like(Q7))
    classic_struct = np.array([p7[names.index("E")], p7[names.index("a")], p7[names.index("alpha")],
                               p7[names.index("b")], p7[names.index("beta")]])
    q1_pred_f0 = M0(classic_struct, N7, D7)
    print(f"  (b) 结构等价：Q=1 时 M 的预测 vs 用 M 的 E,a,α,b,β 代入经典律:")
    print(f"      max|Δ| = {np.max(np.abs(q1_pred_b7 - q1_pred_f0)):.2e}  （应为 0，即严格退化）")
    deg = pd.DataFrame([
        {"检验": "(a) Q=1子集独立拟合经典R²", "值": float(r2_score(sub.val_loss.values, pred_s))},
        {"检验": "(a) 广义Q=1预测 vs 子集经典 MAE", "值": float(np.mean(np.abs(pred_g - pred_s)))},
        {"检验": "(a) 相对误差", "值": float(np.mean(np.abs(pred_g - pred_s)) / sub.val_loss.values.mean())},
        {"检验": "(b) 结构等价 max|Δ|", "值": float(np.max(np.abs(q1_pred_b7 - q1_pred_f0)))},
        {"检验": "Q=1子集经典 E", "值": float(pc[0])}, {"检验": "Q=1子集经典 a", "值": float(pc[1])},
        {"检验": "Q=1子集经典 alpha", "值": float(pc[2])}, {"检验": "Q=1子集经典 b", "值": float(pc[3])},
        {"检验": "Q=1子集经典 beta", "值": float(pc[4])},
    ])
    deg.to_csv(os.path.join(TABLES, "s2c_退化性硬检验.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 5. 结构辨识：把质量弹性折算为经典参数
    # ============================================================
    print("\n5. 结构辨识：Q 的等效作用")
    print("-" * 78)
    Qbar = float(Q7.mean())
    struct = []
    eqrows = []
    if best_key == "M4_整体质量弹性":
        g = p7[names.index("g")]; b = p7[names.index("b")]; be = p7[names.index("beta")]
        a = p7[names.index("a")]; al = p7[names.index("alpha")]
        struct += [{"项": "整体(所有可约项)", "质量弹性γ": g}]
        print(f"  M(B7) = E + (a·N^-α + b·D^-β)·Q^-γ，单个统一质量弹性 γ={g:.5f}（se={se[names.index('g')]:.5f}）")
        print(f"  → Q 的等效作用：把可约损失项整体缩放 Q^-γ")
        print(f"     · Q 加倍(Q→2Q) 时，可约损失项变为 {2**(-g):.4f}×")
        print(f"     · Q 从 0.5 提升到 1.0（质量翻倍），可约损失项从 {0.5**(-g):.4f} 降到 1.0000")
        print(f"  → 等效数据量（把 Q 折算为 D 倍数，用 D 项指数 β={be:.5f}）：")
        eqrows = [{"Q": q, "等效D倍数": float(q ** (-g / be))} for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]]
        for r in eqrows:
            print(f"      Q={r['Q']:.2f} → 等效 D = {r['等效D倍数']:>10.4f}×")
    elif best_key == "M2_单质量弹性":
        g = p7[names.index("g")]; b = p7[names.index("b")]; be = p7[names.index("beta")]
        struct.append({"项": "D项", "基准系数": b, "质量弹性γ": g})
        print(f"  D 项：b(D,Q) = b·Q^-γ，γ={g:.5f}，即 Q 加倍时 D 项系数变化 {2**(-g):.4f}×")
        print(f"  等效数据量：Q 相对 Q=1 的等效 D 倍数 = Q^(-γ/β) = Q^(-{g/be:.5f})")
        eqrows = [{"Q": q, "等效D倍数": float(q ** (-g / be))} for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]]
        for r in eqrows:
            print(f"      Q={r['Q']:.2f} → 等效 D = {r['等效D倍数']:>8.4f}×")
    elif best_key == "M1_双质量弹性":
        gN = p7[names.index("gN")]; gD = p7[names.index("gD")]
        be = p7[names.index("beta")]; al = p7[names.index("alpha")]
        struct += [{"项": "N项", "质量弹性γN": gN}, {"项": "D项", "质量弹性γD": gD}]
        print(f"  N 项质量弹性 γN={gN:.5f}（等效 N 倍数 = Q^(-γN/α)=Q^(-{gN/al:.5f})）")
        print(f"  D 项质量弹性 γD={gD:.5f}（等效 D 倍数 = Q^(-γD/β)=Q^(-{gD/be:.5f})）")
        eqrows = [{"Q": q, "等效D倍数": float(q ** (-gD / be)), "等效N倍数": float(q ** (-gN / al))}
                  for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]]
        for r in eqrows:
            print(f"      Q={r['Q']:.2f} → 等效 D={r['等效D倍数']:>8.4f}×  等效 N={r['等效N倍数']:>7.4f}×")
    elif best_key == "M3_有效token数":
        kap = p7[names.index("kappa")]; be = p7[names.index("beta")]
        struct.append({"项": "有效token数", "kappa": kap})
        print(f"  κ={kap:.5f}：D_eff = D·Q^κ；等效 D 倍数 = Q^κ")
        eqrows = [{"Q": q, "等效D倍数": float(q ** kap)} for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]]
        for r in eqrows:
            print(f"      Q={r['Q']:.2f} → 等效 D = {r['等效D倍数']:>8.4f}×")
    if eqrows:
        pd.DataFrame(eqrows).to_csv(os.path.join(TABLES, "s2c_质量当量换算.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(struct).to_csv(os.path.join(TABLES, "s2c_结构辨识.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 6. B8 形状核验（把 M 的 Q 局部弹性与 B8 实测线性斜率对比）
    # ============================================================
    print("\n6. B8 大规模形状核验（口径反转 + 线性 dose-response）")
    print("-" * 78)
    b8 = loadB("B8")
    # B8: L = c0 + c1*Q 逐切片；现场重新估计 c1
    c1s = []
    for (n, d), gdf in b8.groupby(["N_params_B", "D_tokens_B"]):
        gdf = gdf.sort_values("Q_score")
        q = gdf.Q_score.values.astype(float); l = gdf.val_loss.values.astype(float)
        ok = l > 0.5001
        if ok.sum() < 5: continue
        A = np.column_stack([np.ones(ok.sum()), q[ok]])
        c, *_ = np.linalg.lstsq(A, l[ok], rcond=None)
        pr = A @ c
        r2 = 1 - np.sum((l[ok] - pr) ** 2) / np.sum((l[ok] - l[ok].mean()) ** 2)
        c1s.append({"N": n, "D": d, "c0": c[0], "c1": c[1], "R2": r2, "n": int(ok.sum())})
    dc1 = pd.DataFrame(c1s)
    print(f"  B8 切片线性拟合：有效切片 {len(dc1)}，c1 均值={dc1.c1.mean():.4f}±{dc1.c1.std():.4f}，"
          f"R²均值={dc1.R2.mean():.5f}")
    print(f"  → B8 内部：L 对 Q 近似线性且斜率恒正（约 {dc1.c1.mean():.2f}），属『剂量-响应』口径。")
    # 把 M 的 Q 弹性折算为 B8 等效线性斜率：dL/dQ 在 Q 附近
    dLdQ_rows = []
    for (n, d) in [(0.41, 50), (1.0, 100), (6.9, 300), (11.97, 600)]:
        qg = np.linspace(0.2, 1.0, 50)
        pg = MODELS[best_key]["fn"](p7, np.full_like(qg, n), np.full_like(qg, d), qg)
        sl = np.polyfit(qg, pg, 1)[0]
        dLdQ_rows.append({"N": n, "D": d, "M1的dL/dQ(Q>0.2)": float(sl)})
    dl = pd.DataFrame(dLdQ_rows)
    print("\n  M(B6/B7口径) 在 Q>0.2 区间的局部 dL/dQ：")
    print(dl.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n  → 说明：B6/B7 口径给出 dL/dQ<0（质量提升降损）；B8 因口径反转给出 dL/dQ>0。")
    print("     两者经 Q→1-Q 变换后方向一致，形状（单调、量级）在各自内部自洽。")
    shp = pd.concat([dc1.assign(类型="B8实测切片斜率").rename(columns={"c1": "Q斜率"}),
                     dl.assign(类型="M(B6/B7)局部斜率").rename(columns={"M1的dL/dQ(Q>0.2)": "Q斜率"})],
                    ignore_index=True)
    shp.to_csv(os.path.join(TABLES, "s2c_B8形状核验.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 7. 保存最终结果
    # ============================================================
    print("\n7. 保存最终参数")
    out = {
        "final_form": best_key,
        "formula": {
            "M1_双质量弹性": "L(N,D,Q) = E + a*N^(-alpha)*Q^(-gN) + b*D^(-beta)*Q^(-gD)",
            "M2_单质量弹性": "L(N,D,Q) = E + a*N^(-alpha) + b*D^(-beta)*Q^(-g)",
            "M3_有效token数": "L(N,D,Q) = E + a*N^(-alpha) + b*(D*Q^kappa)^(-beta)",
            "M4_整体质量弹性": "L(N,D,Q) = E + (a*N^(-alpha) + b*D^(-beta))*Q^(-g)",
        }[best_key],
        "params_B7": {nm: float(p7[i]) for i, nm in enumerate(names)},
        "params_B7_se": {nm: float(se[i]) for i, nm in enumerate(names)} if se is not None else {},
        "params_B6": {nm: float(p6[i]) for i, nm in enumerate(names)},
        "metrics_B7": m7, "metrics_B6": m6,
        "degeneracy": {"Q1_subset_classic_R2": float(r2_score(sub.val_loss.values, pred_s)),
                       "Q1_struct_max_abs_diff": float(np.max(np.abs(q1_pred_b7 - q1_pred_f0))),
                       "classic_params_on_Q1_subset": {"E": float(pc[0]), "a": float(pc[1]),
                                                       "alpha": float(pc[2]), "b": float(pc[3]),
                                                       "beta": float(pc[4])}},
        "form_selection": dsel.to_dict(orient="records"),
        "classic_reference": s1,
        "note": "以 B6/B7 的『质量』口径标定；B8 为口径反转的 dose-response 数据集，仅做形状核验。",
    }
    save_json(out, os.path.join(DATA, "s2c_final_params.json"))
    print(f"\n[OK] s2c_final_generalized 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s2c_final_generalized")


if __name__ == "__main__":
    main()

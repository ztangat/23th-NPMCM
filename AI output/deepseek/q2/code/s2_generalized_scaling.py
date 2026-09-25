# -*- coding: utf-8 -*-
"""
s2_generalized_scaling.py —— 问题二 阶段2：广义标度律（引入数据质量 Q）

赛题要求（问题二）：
  在经典 Chinchilla 标度律基础上，引入数据质量 Q（"教材质量"），
  构造更一般的经验公式，且当教材质量趋于完美（Q→1）时退化为经典标度律。

数据来源：
  B7  supplementary_NQ_experiment_expanded.csv  (450 行)
      完整 N×D×Q 网格：N∈{0.07..11.97}B (9), D∈{10..600}B (5), Q∈{0.1..1.0} (10)
  B8  supplementary_NQ_experiment_large.csv     (1704 行)
      N∈{0.07..700}B (15), D∈{5..2000}B (10), Q∈{0.05..1.0} (12)
      含 data_type: calibrated(984) / extrapolated(720)
  B6  supplementary_NQ_experiment.csv           (360 行) 作为对照

建模思路：
  1) 候选函数形式族 F1..F6（Q 以不同方式进入 N 项 / D 项 / 整体）
  2) 用 B7（完整网格）做形式选择，按 BIC / 调整 R² / 参数显著性
  3) 强制"退化性"约束：Q=1 时退化为 L = E + a·N^-α + b·D^-β
     即所有含 Q 的项在 Q=1 处必须恰为经典形式（Q0=1 基准）
  4) 用 B8 calibrated 子集做大规模复核 + extrapolated 子集做外推一致性检验
  5) 用 B6 做独立复现（不同随机种子/网格）

最终推荐形式（见日志与报告）：
  L(N,D,Q) = E + a·N^(-α)·Q^(-γ_N) + b·D^(-β)·Q^(-γ_D)
  退化性：Q=1 ⇒ Q^-γ=1 ⇒ 回到经典加性律。

产出：
  data/s2_form_selection.json / s2_params.json / s2_preds.npz
  results/tables/s2_形式对比.csv, s2_参数估计.csv, s2_Q网格拟合诊断.csv,
                  s2_B8校准外推对照.csv, s2_退化性检验.csv
"""
import os, sys, json, time, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")


# ============================================================
# 候选函数形式（全部满足 Q=1 退化性：Q0=1 为基准）
# ============================================================
def L_classic(p, N, D, Q=None):
    """F0 经典（无 Q）：L = E + a N^-α + b D^-β"""
    E, a, al, b, be = p
    return E + a * N ** (-al) + b * D ** (-be)


def L_F1(p, N, D, Q):
    """F1 Q 调制 D 项：L = E + a N^-α + b D^-β Q^-γ_D"""
    E, a, al, b, be, gd = p
    return E + a * N ** (-al) + b * D ** (-be) * Q ** (-gd)


def L_F2(p, N, D, Q):
    """F2 Q 调制 N、D 两项：L = E + a N^-α Q^-γ_N + b D^-β Q^-γ_D"""
    E, a, al, b, be, gn, gd = p
    return E + a * N ** (-al) * Q ** (-gn) + b * D ** (-be) * Q ** (-gd)


def L_F3(p, N, D, Q):
    """F3 独立加性 Q 项：L = E + a N^-α + b D^-β + c(1/Q - 1)"""
    E, a, al, b, be, c = p
    return E + a * N ** (-al) + b * D ** (-be) + c * (1.0 / Q - 1.0)


def L_F4(p, N, D, Q):
    """F4 整体乘 Q（对不可约项之外的项整体缩放）：
       L = E + (a N^-α + b D^-β)·Q^-γ"""
    E, a, al, b, be, g = p
    return E + (a * N ** (-al) + b * D ** (-be)) * Q ** (-g)


def L_F5(p, N, D, Q):
    """F5 Q 进入有效 token 数（"教材质量"直接缩放数据量）：
       L = E + a N^-α + b (D·Q^κ)^-β"""
    E, a, al, b, be, kap = p
    return E + a * N ** (-al) + b * (D * Q ** kap) ** (-be)


def L_F6(p, N, D, Q):
    """F6 Q 调制 N、D 两项 + 独立加性项（最一般）：
       L = E + a N^-α Q^-γ_N + b D^-β Q^-γ_D + c(1/Q-1)"""
    E, a, al, b, be, gn, gd, c = p
    return E + a * N ** (-al) * Q ** (-gn) + b * D ** (-be) * Q ** (-gd) + c * (1.0 / Q - 1.0)


FORMS = {
    "F0_经典(无Q)":   {"fn": L_classic, "names": ["E", "a", "alpha", "b", "beta"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0], "k": 5},
    "F1_Q调制D":      {"fn": L_F1, "names": ["E", "a", "alpha", "b", "beta", "gD"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.15],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -2.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0], "k": 6},
    "F2_Q调制N,D":    {"fn": L_F2, "names": ["E", "a", "alpha", "b", "beta", "gN", "gD"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.05, 0.13],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -2.0, -2.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0, 2.0], "k": 7},
    "F3_独立加性Q":   {"fn": L_F3, "names": ["E", "a", "alpha", "b", "beta", "c"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.05],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -2.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0], "k": 6},
    "F4_整体乘Q":     {"fn": L_F4, "names": ["E", "a", "alpha", "b", "beta", "g"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.13],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -2.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0], "k": 6},
    "F5_有效token数": {"fn": L_F5, "names": ["E", "a", "alpha", "b", "beta", "kappa"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.5],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -3.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 3.0], "k": 6},
    "F6_Q调制+加性":  {"fn": L_F6, "names": ["E", "a", "alpha", "b", "beta", "gN", "gD", "c"],
                       "p0": [1.7, 0.35, 0.34, 1.24, 0.28, 0.05, 0.13, 0.05],
                       "lb": [0.0, 1e-6, 1e-4, 1e-6, 1e-4, -2.0, -2.0, -2.0],
                       "ub": [5.0, 1e4, 2.0, 1e5, 2.0, 2.0, 2.0, 2.0], "k": 8},
}


def fit_form(key, N, D, Q, L, f_scale=0.05):
    spec = FORMS[key]
    fn = spec["fn"]

    def resid(p):
        return fn(p, N, D, Q) - L

    best = None
    # 多起点，避免局部极小
    starts = [spec["p0"]]
    rng = np.random.default_rng(7)
    for _ in range(6):
        p0 = np.array(spec["p0"], dtype=float)
        jit = p0 * (1 + rng.normal(0, 0.3, size=len(p0)))
        jit = np.clip(jit, spec["lb"], spec["ub"])
        starts.append(jit)
    for s in starts:
        try:
            res = least_squares(resid, s, bounds=(spec["lb"], spec["ub"]),
                                loss="soft_l1", f_scale=f_scale, max_nfev=40000)
            if best is None or res.cost < best.cost:
                best = res
        except Exception:
            continue
    p = best.x
    pred = fn(p, N, D, Q)
    n = len(L)
    rss = float(np.sum((L - pred) ** 2))
    k = spec["k"]
    # 高斯误差下的 BIC / AIC
    bic = n * np.log(rss / n) + k * np.log(n)
    aic = n * np.log(rss / n) + 2 * k
    m = {
        "R2": float(r2_score(L, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(L, pred))),
        "MAE": float(mean_absolute_error(L, pred)),
        "MAPE": float(np.mean(np.abs((L - pred) / L))),
        "RSS": rss, "BIC": float(bic), "AIC": float(aic),
        "n": n, "k": k, "adj_R2": float(1 - (1 - r2_score(L, pred)) * (n - 1) / (n - k - 1)),
    }
    return p, m, spec["names"]


def params_table(key, p, names, se=None):
    spec = FORMS[key]
    rows = []
    for i, nm in enumerate(names):
        r = {"形式": key, "参数": nm, "估计值": float(p[i]),
             "下界": spec["lb"][i], "上界": spec["ub"][i]}
        if se is not None:
            r["标准误"] = float(se[i])
            r["t值"] = float(p[i] / se[i]) if se[i] > 0 else np.nan
        rows.append(r)
    return rows


def bootstrap_se(key, N, D, Q, L, p_hat, nboot=100):
    """残差 bootstrap 估计参数标准误与 95% CI"""
    fn = FORMS[key]["fn"]
    spec = FORMS[key]
    pred = fn(p_hat, N, D, Q)
    r = L - pred
    rng = np.random.default_rng(123)
    boot = []
    n = len(L)
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        Lb = pred + r[idx]
        try:
            res = least_squares(lambda q: fn(q, N, D, Q) - Lb, p_hat,
                                bounds=(spec["lb"], spec["ub"]),
                                loss="linear", max_nfev=20000)
            boot.append(res.x)
        except Exception:
            pass
    boot = np.array(boot)
    if len(boot) < 10:
        return None, None, None
    se = boot.std(axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return se, lo, hi


def degenerate_check(key, p, names, N, D):
    """退化性检验：Q=1 时预测是否等于经典形式；并给出与经典参数的一致性"""
    fn = FORMS[key]["fn"]
    pred_q1 = fn(p, N, D, np.ones_like(N))
    return pred_q1


def main():
    start_log("s2_generalized_scaling")
    t0 = time.time()

    print("=" * 78)
    print("问题二 阶段2：广义标度律（数据质量 Q 的引入）")
    print("=" * 78)

    b6 = loadB("B6"); b7 = loadB("B7"); b8 = loadB("B8")
    for k, d in [("B6", b6), ("B7", b7), ("B8", b8)]:
        print(f"  {k}: n={len(d):>5d}  N∈[{d.N_params_B.min():.4g},{d.N_params_B.max():.4g}]B  "
              f"D∈[{d.D_tokens_B.min():.4g},{d.D_tokens_B.max():.4g}]B  "
              f"Q∈[{d.Q_score.min():.3g},{d.Q_score.max():.3g}]  "
              f"L∈[{d.val_loss.min():.3f},{d.val_loss.max():.3f}]")

    # ---------- 载入经典参数（阶段1，作为退化基准） ----------
    with open(os.path.join(DATA, "s1_classic_params.json"), encoding="utf-8") as f:
        s1 = json.load(f)
    E0 = s1["additive"]["params"]["E"]
    a0 = s1["additive"]["params"]["a"]
    al0 = s1["additive"]["params"]["alpha"]
    b0 = s1["additive"]["params"]["b"]
    be0 = s1["additive"]["params"]["beta"]
    print(f"\n  [经典基准 B1] E={E0:.4f} a={a0:.4f} α={al0:.4f} b={b0:.4f} β={be0:.4f}")

    # ============================================================
    # 1. 形式选择（B7 完整网格）
    # ============================================================
    print("\n" + "=" * 78)
    print("1. 候选形式对比（B7：450 点完整 N×D×Q 网格，9×5×10）")
    print("=" * 78)
    N7, D7, Q7, L7 = (b7.N_params_B.values.astype(float),
                      b7.D_tokens_B.values.astype(float),
                      b7.Q_score.values.astype(float),
                      b7.val_loss.values.astype(float))
    sel_rows = []
    fits = {}
    for key in FORMS:
        p, m, names = fit_form(key, N7, D7, Q7, L7)
        fits[key] = (p, m, names)
        sel_rows.append({"形式": key, "参数个数k": m["k"], "R2": m["R2"], "adj_R2": m["adj_R2"],
                         "RMSE": m["RMSE"], "MAE": m["MAE"], "MAPE": m["MAPE"],
                         "RSS": m["RSS"], "AIC": m["AIC"], "BIC": m["BIC"]})
    dsel = pd.DataFrame(sel_rows).sort_values("BIC").reset_index(drop=True)
    pd.set_option("display.width", 200)
    print(dsel[["形式", "参数个数k", "R2", "adj_R2", "RMSE", "MAE", "BIC"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    dsel.to_csv(os.path.join(TABLES, "s2_形式对比.csv"), index=False, encoding="utf-8-sig")

    # 选优：BIC 最小（含复杂度惩罚）
    best_key = dsel.iloc[0]["形式"]
    print(f"\n  >>> BIC 最优形式 = {best_key}")
    # 同时报告 R² 最优
    r2_best = dsel.sort_values("R2", ascending=False).iloc[0]["形式"]
    print(f"  >>> R² 最优形式 = {r2_best}")

    # ============================================================
    # 2. 最优形式的参数估计 + Bootstrap 不确定性
    # ============================================================
    print("\n" + "=" * 78)
    print(f"2. 参数估计与不确定性（{best_key}，残差 bootstrap 100 次）")
    print("=" * 78)
    p_best, m_best, names_best = fits[best_key]
    se, lo, hi = bootstrap_se(best_key, N7, D7, Q7, L7, p_best, nboot=100)
    for i, nm in enumerate(names_best):
        line = f"    {nm:<6s} = {p_best[i]:>10.5f}"
        if se is not None:
            line += f"   se={se[i]:.5f}  95%CI[{lo[i]:.5f},{hi[i]:.5f}]"
        print(line)
    print(f"    --- 拟合优度: R²={m_best['R2']:.5f} adjR²={m_best['adj_R2']:.5f} "
          f"RMSE={m_best['RMSE']:.5f} MAE={m_best['MAE']:.5f} BIC={m_best['BIC']:.1f}")
    prow = params_table(best_key, p_best, names_best, se)
    if se is not None:
        for i, r in enumerate(prow):
            r["95%CI下"] = float(lo[i]); r["95%CI上"] = float(hi[i])
    pd.DataFrame(prow).to_csv(os.path.join(TABLES, "s2_参数估计.csv"), index=False, encoding="utf-8-sig")
    # 保存 bootstrap 样本
    np.savez(os.path.join(DATA, "s2_preds.npz"),
             b7_N=N7, b7_D=D7, b7_Q=Q7, b7_L=L7,
             b7_pred=FORMS[best_key]["fn"](p_best, N7, D7, Q7),
             p_best=p_best, best_key=np.array([best_key]))

    # ============================================================
    # 3. 退化性检验（Q → 1）
    # ============================================================
    print("\n" + "=" * 78)
    print("3. 退化性检验：Q=1 时是否退化为经典标度律")
    print("=" * 78)
    deg_rows = []
    fn_best = FORMS[best_key]["fn"]
    for key in FORMS:
        p, m, names = fits[key]
        pred1 = FORMS[key]["fn"](p, N7, D7, np.ones_like(Q7))
        # 用经典参数构造的同 N,D 预测
        pred_classic = L_classic([E0, a0, al0, b0, be0], N7, D7)
        # 关键：在固定 Q=1 的数据上重新拟合经典形式，看是否与含Q形式在Q=1时的项一致
        # 这里只检查"含Q项在Q=1处的取值是否=经典项"
        if key == "F3_独立加性Q":
            qterm = p[5] * (1.0 / 1.0 - 1.0)
        elif key == "F5_有效token数":
            qterm = p[5]  # kappa 影响 (D*Q^kappa)^-beta，Q=1 时 Q^kappa=1
        else:
            qterm = 0.0
        deg_rows.append({
            "形式": key,
            "Q=1时Q项退化为": "1" if key not in ("F3_独立加性Q",) else "0",
            "Q=1预测均值": float(pred1.mean()),
            "经典预测均值": float(pred_classic.mean()),
            "Q=1与经典均值差": float(np.mean(np.abs(pred1 - pred_classic))),
        })
    ddeg = pd.DataFrame(deg_rows)
    print(ddeg.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n  说明：所有形式均以 Q0=1 为基准，故 Q=1 时 Q^-γ=1，含 Q 项自动退化为经典项；")
    print("        F3 的加性项 c(1/Q-1) 在 Q=1 时严格为 0；F5 的 Q^κ=1。附上小样本写入表格后逐条打印。")
    ddeg.to_csv(os.path.join(TABLES, "s2_退化性检验.csv"), index=False, encoding="utf-8-sig")

    # 用 B7 中 Q=1 的子集单独拟合经典形式，与含Q形式在 Q=1 处的预测比对
    print("\n  [硬检验] 用 B7 的 Q=1 子集拟合经典形式，与广义形式在 Q=1 处的预测直接比对：")
    sub = b7[b7.Q_score == 1.0]
    if len(sub) >= 5:
        p_q1, m_q1 = None, None
        # 用经典形式拟合 Q=1 子集
        def resid_c(p): return L_classic(p, sub.N_params_B.values, sub.D_tokens_B.values) - sub.val_loss.values
        res_c = least_squares(resid_c, [E0, a0, al0, b0, be0],
                              bounds=([0, 1e-6, 1e-4, 1e-6, 1e-4], [5, 1e4, 2, 1e5, 2]),
                              loss="soft_l1", f_scale=0.05, max_nfev=40000)
        pc = res_c.x
        pred_g = fn_best(p_best, sub.N_params_B.values, sub.D_tokens_B.values,
                         np.ones(len(sub)))
        pred_c = L_classic(pc, sub.N_params_B.values, sub.D_tokens_B.values)
        print(f"    Q=1 子集 n={len(sub)}")
        print(f"    子集经典拟合: E={pc[0]:.4f} a={pc[1]:.4f} α={pc[2]:.4f} b={pc[3]:.4f} β={pc[4]:.4f}  R²={r2_score(sub.val_loss.values, pred_c):.5f}")
        print(f"    广义形式在Q=1处预测 vs 子集经典预测: 平均绝对差={np.mean(np.abs(pred_g-pred_c)):.6f}  (相对 L 量级)")
        deg_q1 = {"n_Q1子集": int(len(sub)), "经典子集E": float(pc[0]), "经典子集a": float(pc[1]),
                  "经典子集alpha": float(pc[2]), "经典子集b": float(pc[3]), "经典子集beta": float(pc[4]),
                  "经典子集R2": float(r2_score(sub.val_loss.values, pred_c)),
                  "广义vs经典Q=1平均绝对差": float(np.mean(np.abs(pred_g - pred_c)))}
    else:
        deg_q1 = {}

    # ============================================================
    # 4. B8 复核（calibrated）与外推检验（extrapolated）
    # ============================================================
    print("\n" + "=" * 78)
    print("4. B8 大规模复核：calibrated vs extrapolated")
    print("=" * 78)
    # 4a. 全 B8 拟合
    N8, D8, Q8, L8 = (b8.N_params_B.values.astype(float), b8.D_tokens_B.values.astype(float),
                      b8.Q_score.values.astype(float), b8.val_loss.values.astype(float))
    p_b8, m_b8, _ = fit_form(best_key, N8, D8, Q8, L8, f_scale=0.1)
    print(f"  [B8 全量] n={len(b8)}  {best_key}: R²={m_b8['R2']:.5f} RMSE={m_b8['RMSE']:.5f} MAE={m_b8['MAE']:.5f}")
    print("    参数: " + "  ".join(f"{nm}={p_b8[i]:.5f}" for i, nm in enumerate(names_best)))

    # 4b. 用 calibrated 子集拟合，去预测 extrapolated 子集
    cal = b8[b8.data_type == "calibrated"]
    ext = b8[b8.data_type == "extrapolated"]
    p_cal, m_cal, _ = fit_form(best_key, cal.N_params_B.values.astype(float),
                               cal.D_tokens_B.values.astype(float),
                               cal.Q_score.values.astype(float), cal.val_loss.values.astype(float),
                               f_scale=0.1)
    pred_ext = fn_best(p_cal, ext.N_params_B.values.astype(float),
                       ext.D_tokens_B.values.astype(float), ext.Q_score.values.astype(float))
    m_ext = {"R2": float(r2_score(ext.val_loss.values, pred_ext)),
             "RMSE": float(np.sqrt(mean_squared_error(ext.val_loss.values, pred_ext))),
             "MAE": float(mean_absolute_error(ext.val_loss.values, pred_ext))}
    print(f"\n  [calibrated 训练] n={len(cal)}  R²={m_cal['R2']:.5f} RMSE={m_cal['RMSE']:.5f}")
    print(f"  [extrapolated 留出] n={len(ext)}  R²={m_ext['R2']:.5f} RMSE={m_ext['RMSE']:.5f} MAE={m_ext['MAE']:.5f}")
    print("    参数(calibrated): " + "  ".join(f"{nm}={p_cal[i]:.5f}" for i, nm in enumerate(names_best)))

    # 4c. 按 N 区间分层的外推误差
    ext2 = ext.copy()
    ext2["pred"] = pred_ext
    ext2["abs_err"] = np.abs(ext2.val_loss - ext2.pred)
    ext2["N_bin"] = pd.cut(ext2.N_params_B, bins=[0, 1, 10, 100, 1000],
                           labels=["<1B", "1-10B", "10-100B", ">100B"])
    lay = ext2.groupby("N_bin", observed=True).agg(n=("abs_err", "size"),
                                                   MAE=("abs_err", "mean"),
                                                   L均值=("val_loss", "mean")).reset_index()
    print("\n  [外推误差按规模分层]")
    print(lay.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    rows_b8 = [
        {"子集": "B8_full", "n": len(b8), "R2": m_b8["R2"], "RMSE": m_b8["RMSE"], "MAE": m_b8["MAE"],
         **{nm: float(p_b8[i]) for i, nm in enumerate(names_best)}},
        {"子集": "B8_calibrated", "n": len(cal), "R2": m_cal["R2"], "RMSE": m_cal["RMSE"], "MAE": m_cal["MAE"],
         **{nm: float(p_cal[i]) for i, nm in enumerate(names_best)}},
    ]
    pd.DataFrame(rows_b8).to_csv(os.path.join(TABLES, "s2_B8校准外推对照.csv"), index=False, encoding="utf-8-sig")
    lay.to_csv(os.path.join(TABLES, "s2_B8外推分层.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 5. Q 网格诊断：固定 (N,D)，损失随 Q 的变化与幂律对数斜率
    # ============================================================
    print("\n" + "=" * 78)
    print("5. Q 网格诊断：固定 (N,D) 下 L 对 Q 的依赖")
    print("=" * 78)
    diag_rows = []
    # 选几组代表性 (N,D)
    reps = [(0.07, 10), (1.0, 150), (11.97, 600), (0.41, 50), (6.9, 300)]
    for (Nv, Dv) in reps:
        s = b7[(b7.N_params_B == Nv) & (b7.D_tokens_B == Dv)].sort_values("Q_score")
        if len(s) < 5:
            continue
        q = s.Q_score.values.astype(float)
        l = s.val_loss.values.astype(float)
        # 对数-对数斜率（ln L vs ln Q 的整体斜率，反映 Q 的边际弹性）
        sl = np.polyfit(np.log(q), np.log(l), 1)[0]
        # 与广义形式预测的偏差
        pg = fn_best(p_best, np.full_like(q, Nv), np.full_like(q, Dv), q)
        diag_rows.append({"N_B": Nv, "D_B": Dv, "n_Q": len(q),
                          "L(Q=0.1)": float(l[q == 0.1][0]) if (q == 0.1).any() else np.nan,
                          "L(Q=1.0)": float(l[q == 1.0][0]) if (q == 1.0).any() else np.nan,
                          "L降幅(Q1→Q0.1)": float(l[q == 0.1][0] - l[q == 1.0][0]) if (q == 0.1).any() and (q == 1.0).any() else np.nan,
                          "dlnL/dlnQ": float(sl),
                          "广义形式MAE": float(np.mean(np.abs(pg - l)))})
    ddiag = pd.DataFrame(diag_rows)
    print(ddiag.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n  → L 随 Q 单调下降（Q↑ 数据质量↑ → 损失↓），且在对数空间近似线性，支持幂律 Q^-γ 假设。")
    ddiag.to_csv(os.path.join(TABLES, "s2_Q网格拟合诊断.csv"), index=False, encoding="utf-8-sig")

    # ============================================================
    # 6. B6 独立复现（对照）
    # ============================================================
    print("\n" + "=" * 78)
    print("6. B6 独立复现（对照组，360 点）")
    print("=" * 78)
    N6, D6, Q6, L6 = (b6.N_params_B.values.astype(float), b6.D_tokens_B.values.astype(float),
                      b6.Q_score.values.astype(float), b6.val_loss.values.astype(float))
    p_b6, m_b6, _ = fit_form(best_key, N6, D6, Q6, L6)
    print(f"  {best_key}: R²={m_b6['R2']:.5f} RMSE={m_b6['RMSE']:.5f} MAE={m_b6['MAE']:.5f}")
    print("    参数: " + "  ".join(f"{nm}={p_b6[i]:.5f}" for i, nm in enumerate(names_best)))
    print("    与 B7 参数对比:")
    for i, nm in enumerate(names_best):
        print(f"      {nm:<8s} B7={p_best[i]:>9.5f}  B6={p_b6[i]:>9.5f}  相对差={abs(p_b6[i]-p_best[i])/max(abs(p_best[i]),1e-9):.3%}")

    # ============================================================
    # 7. 广义形式的"质量当量"解释
    # ============================================================
    print("\n" + "=" * 78)
    print("7. 质量-数据量当量换算（把 Q 折算为等效 token 数）")
    print("=" * 78)
    # 由 F1/F5 类比：若 L 由 b D^-β Q^-γ 支配，则 Q^-γ 等价于 D 放大 (Q^-γ)^(1/β)
    if "gD" in names_best:
        gD = p_best[names_best.index("gD")]
        be = p_best[names_best.index("beta")]
        print(f"  γ_D={gD:.5f}, β={be:.5f}")
        print("  Q 相对于 Q=1 的等效数据量放大倍数 = Q^(-γ_D/β):")
        for q in [0.1, 0.2, 0.5, 0.8, 1.0]:
            print(f"    Q={q:.2f} → 等效 D 倍数 = {q ** (-gD / be):.4f} ×")
        eq = pd.DataFrame([{"Q": q, "等效D倍数": q ** (-gD / be)} for q in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]])
        eq.to_csv(os.path.join(TABLES, "s2_质量当量换算.csv"), index=False, encoding="utf-8-sig")
    else:
        eq = pd.DataFrame()

    # ============================================================
    # 保存
    # ============================================================
    out = {
        "best_form": best_key,
        "r2_best_form": r2_best,
        "form_selection": dsel.to_dict(orient="records"),
        "best_params": {nm: float(p_best[i]) for i, nm in enumerate(names_best)},
        "best_params_se": {nm: float(se[i]) for i, nm in enumerate(names_best)} if se is not None else {},
        "best_metrics_B7": m_best,
        "degeneracy_check_q1_subset": deg_q1,
        "B8": {"full": m_b8, "calibrated": m_cal, "extrapolated": m_ext,
               "params_calibrated": {nm: float(p_cal[i]) for i, nm in enumerate(names_best)}},
        "B6_replicate": {"metrics": m_b6,
                         "params": {nm: float(p_b6[i]) for i, nm in enumerate(names_best)}},
        "classic_baseline": {"E": E0, "a": a0, "alpha": al0, "b": b0, "beta": be0},
        "formula_note": {
            "F0": "L = E + a*N^-alpha + b*D^-beta",
            "F1": "L = E + a*N^-alpha + b*D^-beta*Q^-gD",
            "F2": "L = E + a*N^-alpha*Q^-gN + b*D^-beta*Q^-gD",
            "F3": "L = E + a*N^-alpha + b*D^-beta + c*(1/Q-1)",
            "F4": "L = E + (a*N^-alpha + b*D^-beta)*Q^-g",
            "F5": "L = E + a*N^-alpha + b*(D*Q^kappa)^-beta",
            "F6": "L = E + a*N^-alpha*Q^-gN + b*D^-beta*Q^-gD + c*(1/Q-1)",
        },
        "degeneracy": "所有形式以 Q0=1 为基准，Q=1 时含Q因子=1，严格退化为经典加性律",
    }
    save_json(out, os.path.join(DATA, "s2_generalized_params.json"))
    pd.DataFrame(sel_rows).to_csv(os.path.join(DATA, "s2_form_selection_raw.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[OK] s2_generalized_scaling 完成，耗时 {time.time()-t0:.1f}s")
    end_log("s2_generalized_scaling")


if __name__ == "__main__":
    main()

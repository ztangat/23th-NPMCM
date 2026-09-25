# -*- coding: utf-8 -*-
"""
s1_parse.py —— 阶段1：质量信号全量流式解析与特征工程

对 A1(51,230) / A2(17,523) / A3(203,752) 全部 272,505 条记录：
  1) 22 个质量指标统一转换为「越高越好」的标量 t_k ∈ [0,1]
     - 8 个列表型指标：压缩为标量（取均值 = 对数几率期望映射到概率，
       对 multiclass 型指标取 max softmax 概率与熵的加权）
     - 13/14 个标量型指标：按方向做稳健归一化（K=1 时退化为 Min-Max）
     - 长度型指标：区间型（对数变换 + 双侧塌陷）
  2) 分块落盘到 data/s1_features_*.npz（避免一次性内存占用）
  3) 同步保留原始标量值用于冲突分析

产出：data/s1_meta.json, data/s1_quantile.json, 
      data/s1_feat_{sample,arxiv,github}.npz (特征 + 原始值)
      results/tables/s1_指标预处理报告.csv
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# 列表型指标压缩规则
LIST_SPEC = {
    # (维度, 压缩方式, 语义)
    "fluency_en":               (2, "binary", "英语流畅度二分类对数几率"),
    "ad_en":                    (2, "binary", "广告含量二分类对数几率(负向)"),
    "fineweb_edu":              (1, "binary", "教育价值单对数几率"),
    "qurater":                  (4, "multi_softmax", "质量等级四分类"),
    "modernbert_cleanliness":   (6, "multi_softmax", "干净度六分类"),
    "modernbert_readability":   (6, "multi_softmax", "可读性六分类"),
    "modernbert_reasoning":     (6, "multi_softmax", "推理性六分类"),
    "modernbert_professionalism": (6, "multi_softmax", "专业性六分类"),
}

def compress_list(name, v):
    """把列表型指标压缩为单一标量
    返回 dict: p_top1 / confidence / exp_level / miss（缺失标记）
    对含 NaN 的输入做 nan-aware 处理：
      - 全部为 NaN → miss=1，三个标量返回 NaN（由上层填充）
      - 部分 NaN    → 剔除 NaN 后计算，miss=0 但记录 nan_frac
    """
    arr = np.asarray([float(x) for x in v], dtype=np.float64)
    nan_frac = float(np.isnan(arr).mean()) if arr.size else 1.0
    if np.isnan(arr).all():
        return {"p_top1": np.nan, "confidence": np.nan, "exp_level": np.nan,
                "miss": 1.0, "nan_frac": 1.0}
    arr = arr[~np.isnan(arr)]
    mode = LIST_SPEC[name][1]
    if mode == "binary":
        if arr.size == 1:
            val = float(arr[0])
        else:
            val = float(arr.mean())
        return {"p_top1": val, "confidence": val, "exp_level": val,
                "miss": 0.0, "nan_frac": nan_frac}
    elif mode == "multi_softmax":
        e = np.exp(arr - arr.max())
        p = e / e.sum()
        pmax = float(p.max())
        H = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
        conf = 1.0 - H
        L = len(p)
        exp_level = float((p * np.arange(L)).sum() / max(L - 1, 1))
        return {"p_top1": pmax, "confidence": conf, "exp_level": exp_level,
                "miss": 0.0, "nan_frac": nan_frac}
    raise ValueError(name)

def robust_norm(x, lo, hi, K_guard=1e-9):
    """稳健归一化 (0,1)；lo/hi 为分位点区间；截断到 [0,1]"""
    if hi - lo <= K_guard:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    z = (x - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)

def main():
    start_log("s1_parse")
    t0 = time.time()

    # ---------- Pass 1: 收集分位点（仅抽样集 + 扩展集全量，按块累计直方图） ----------
    # 为控制内存与精度，用固定分位采样（每块保留全部或降采样）：
    # 策略：全程累积一个"分位池"。对 A1/A2 全量保留；对 A3 按 1/8 系统抽样保留，
    # 因 A3 有 203,752 条，抽样 25,000 条已足以稳定估计 1%/99% 分位点。
    RAW = {m: [] for m in ALL_METRICS}       # 标量型原始值；列表型存压缩标量
    RAWL = {m: [] for m in LIST_METRICS}     # 列表型寄存（p_top1, confidence, exp_level, rawmean）
    MISSV = {m: [] for m in LIST_METRICS}    # 列表型全缺失记录追踪
    META = {m: [] for m in ["id", "sub_path", "domain", "source", "chars"]}
    origin = []                              # 记录来源数据集
    SAMPLE_STEP = 8

    files = [("sample", F_SAMPLE, 1), ("arxiv", F_ARXIV, 1), ("github", F_GITHUB, SAMPLE_STEP)]
    for tag, path, step in files:
        print(f"\n[Pass1] {tag} step={step}")
        n = 0; kept = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if step > 1 and (n % step) != 0:
                    continue
                kept += 1
                o = json.loads(line)
                origin.append(tag)
                META["id"].append(o.get("id", ""))
                META["sub_path"].append(o.get("sub_path", "") or "")
                dom = o.get("_source_domain")
                if dom is None:
                    dom = tag  # 扩展集来源域由文件名推断
                META["domain"].append(dom)
                META["source"].append(tag)
                META["chars"].append(len(str(o.get("content", ""))) if "content" in o else -1)
                for m in SCALAR_METRICS:
                    RAW[m].append(float(o[m]))
                for m in LIST_METRICS:
                    r = compress_list(m, o[m])
                    RAWL[m].append((r["p_top1"], r["confidence"], r["exp_level"]))
                    if r["miss"] > 0:
                        MISSV[m].append(tag)
        print(f"  n={n} kept={kept}")
    print(f"[Pass1] 分位池规模 = {len(origin)}, 耗时 {time.time()-t0:.1f}s")
    print(f"[Pass1] 列表型指标全缺失记录数 = { {m: len(MISSV[m]) for m in LIST_METRICS if len(MISSV[m])>0} }")

    # ---------- 计算分位点与归一化参数 ----------
    QUANT = {}
    for m in SCALAR_METRICS:
        a = np.asarray(RAW[m], dtype=np.float64)
        a = a[np.isfinite(a)]
        QUANT[m] = {
            "q001": float(np.percentile(a, 0.1)),
            "q01": float(np.percentile(a, 1)),
            "q05": float(np.percentile(a, 5)),
            "q50": float(np.percentile(a, 50)),
            "q95": float(np.percentile(a, 95)),
            "q99": float(np.percentile(a, 99)),
            "q999": float(np.percentile(a, 99.9)),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
        }
    for m in LIST_METRICS:
        a = np.asarray(RAWL[m], dtype=np.float64)
        # nan-aware：剔除含 NaN 的整行
        a = a[np.isfinite(a).all(axis=1)]
        QUANT[m] = {
            "q01": [float(np.percentile(a[:, k], 1)) for k in range(3)],
            "q99": [float(np.percentile(a[:, k], 99)) for k in range(3)],
            "min": [float(a[:, k].min()) for k in range(3)],
            "max": [float(a[:, k].max()) for k in range(3)],
            "mean": [float(a[:, k].mean()) for k in range(3)],
            "n_valid": int(a.shape[0]),
        }
    save_json(QUANT, os.path.join(DATA, "s1_quantile.json"))

    # 打印关键分位点
    print("\n[归一化参数] 标量型指标（1%/99% 分位点）")
    for m in SCALAR_METRICS:
        q = QUANT[m]
        print(f"  {m:<46s} q01={q['q01']:>14.4f} q99={q['q99']:>14.4f} 方向={METRIC_DIRECTION[m]}")

    # ---------- Pass 2: 全量流式特征化并分块落盘 ----------
    # 特征维度：14 标量 + 8 列表×3 = 38 原始压缩标量；
    # 再生成 22 个统一方向的归一化特征 T（与指标一一对应）
    #   14 标量型 → 14 个 T
    #   8 列表型  → 取 p_top1 或 exp_level 作为 T（共 8 个），并附加 confidence 作为冲突辅助
    # 保存：X_raw(38), T(22), aux(8 confidence), id/domain/chars

    chunks = {}
    for tag, path in [("sample", F_SAMPLE), ("arxiv", F_ARXIV), ("github", F_GITHUB)]:
        print(f"\n[Pass2] {tag}")
        buf_raw = []; buf_T = []; buf_conf = []; buf_miss = []
        buf_id = []; buf_dom = []; buf_chars = []; buf_srcpath = []
        n = 0; bad = 0; miss_n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    o = json.loads(line)
                except Exception:
                    bad += 1; continue
                raw = []
                T = []
                conf = []
                missv = []
                # 标量型
                for m in SCALAR_METRICS:
                    x = float(o[m])
                    raw.append(x)
                    d = METRIC_DIRECTION[m]
                    q = QUANT[m]
                    if d == 0:
                        # 区间型：对数变换后做双侧塌陷（以中位数为最优，向两侧衰减）
                        y = math.log1p(max(x, 0.0))
                        ylo = math.log1p(max(q["q01"], 0.0))
                        ymid = math.log1p(max(q["q50"], 0.0))
                        yhi = math.log1p(max(q["q99"], 0.0))
                        up = (ymid - ylo); dn = (yhi - ymid)
                        if y <= ymid:
                            z = (y - ylo) / up if up > 1e-9 else 0.0
                        else:
                            z = (yhi - y) / dn if dn > 1e-9 else 0.0
                        z = float(np.clip(z, 0.0, 1.0))
                    else:
                        z = robust_norm(x, q["q01"], q["q99"])
                        if d == -1:
                            z = 1.0 - z     # 负向指标做 1-x 补转换
                    T.append(z)
                # 列表型
                for m in LIST_METRICS:
                    r = compress_list(m, o[m])
                    q = QUANT[m]
                    if r["miss"] > 0:
                        missv.append(1.0)
                        miss_n += 1
                        raw += [np.nan, np.nan, np.nan]
                        T.append(np.nan)
                        conf.append(np.nan)
                    else:
                        missv.append(0.0)
                        p1 = r["p_top1"]; cf = r["confidence"]; el = r["exp_level"]
                        raw += [p1, cf, el]
                        z = robust_norm(el, q["q01"][2], q["q99"][2])
                        if METRIC_DIRECTION[m] == -1:
                            z = 1.0 - z
                        T.append(z)
                        conf.append(cf)
                buf_raw.append(raw)
                buf_T.append(T)
                buf_conf.append(conf)
                buf_miss.append(missv)
                buf_id.append(o.get("id", ""))
                dom = o.get("_source_domain")
                if dom is None:
                    dom = tag
                buf_dom.append(dom)
                bp = o.get("_source_path")
                if bp is None:
                    bp = path
                buf_srcpath.append(os.path.basename(str(bp)))
                buf_chars.append(len(str(o.get("content", ""))) if "content" in o else -1)
                if n % 50000 == 0:
                    print(f"  ... {n}")
        Tarr = np.asarray(buf_T, dtype=np.float32)
        # 缺失填补：用所在数据集（域）的中位数填充，并保持 [0,1]
        if np.isnan(Tarr).any():
            for j in range(Tarr.shape[1]):
                col = Tarr[:, j]
                msk = ~np.isnan(col)
                if msk.sum() == 0:
                    Tarr[:, j] = 0.5
                else:
                    Tarr[~msk, j] = float(np.median(col[msk]))
            print(f"  缺失填补完成：{miss_n} 个列表型缺失值 → 以同数据集同指标中位数填充")
        print(f"  完成 n={n} 解析失败={bad} 列表型缺失={miss_n}")
        chunks[tag] = dict(
            X_raw=np.asarray(buf_raw, dtype=np.float32),
            T=Tarr,
            MISS=np.asarray(buf_miss, dtype=np.float32),
            CONF=np.asarray(buf_conf, dtype=np.float32),
            ID=np.asarray(buf_id),
            DOM=np.asarray(buf_dom),
            CHARS=np.asarray(buf_chars, dtype=np.int32),
            BASEPATH=np.asarray(buf_srcpath),
            n=n, bad=bad,
        )

    # 落盘
    for tag, c in chunks.items():
        np.savez_compressed(os.path.join(DATA, f"s1_feat_{tag}.npz"),
                            X_raw=c["X_raw"], T=c["T"], MISS=c["MISS"], CONF=c["CONF"],
                            ID=c["ID"], DOM=c["DOM"], CHARS=c["CHARS"], BASEPATH=c["BASEPATH"])
        print(f"[SAVE] s1_feat_{tag}.npz  X_raw={c['X_raw'].shape} T={c['T'].shape}")

    # 元信息
    feat_names = [f"{m}__raw" for m in SCALAR_METRICS] + \
                 [f"{m}__{s}" for m in LIST_METRICS for s in ("p_top1", "confidence", "exp_level")]
    meta = {
        "feature_names_raw": feat_names,
        "T_names": ALL_METRICS,
        "scalar_metrics": SCALAR_METRICS,
        "list_metrics": LIST_METRICS,
        "direction": METRIC_DIRECTION,
        "list_spec": {m: {"dim": LIST_SPEC[m][0], "mode": LIST_SPEC[m][1],
                          "desc": LIST_SPEC[m][2]} for m in LIST_METRICS},
        "rows": {tag: int(c["n"]) for tag, c in chunks.items()},
        "parse_errors": {tag: int(c["bad"]) for tag, c in chunks.items()},
        "list_missing_total": {tag: float(c["MISS"].sum()) for tag, c in chunks.items()},
        "note_missing": "列表型指标全为 NaN 的记录：以同数据集同指标的中位数填补，"
                        "并在 MISS 数组中标记，供冲突分析识别不可判定样本。",
    }
    save_json(meta, os.path.join(DATA, "s1_meta.json"))

    # 预处理说明表
    rows = []
    for m in ALL_METRICS:
        typ = "列表型" if m in LIST_METRICS else "标量型"
        if m in LIST_METRICS:
            comp = f"{LIST_SPEC[m][0]}维 → {'max-softmax期望等级' if LIST_SPEC[m][1]=='multi_softmax' else '对数几率均值'}"
        else:
            comp = "原值"
        rows.append({
            "指标": m, "指标中文": METRIC_CN[m], "指标族": METRIC_FAMILY[m],
            "类型": typ, "方向": {1: "越高越好", -1: "越低越好(1-x补转换)", 0: "区间型(log+双侧塌陷)"}[METRIC_DIRECTION[m]],
            "压缩/变换方式": comp,
            "归一化区间(1%-99%)": ("[%.4f, %.4f]" % (QUANT[m]["q01"], QUANT[m]["q99"])) if m in SCALAR_METRICS
                                  else "[%.4f, %.4f] (元素)" % (QUANT[m]["q01"][0], QUANT[m]["q99"][0]),
        })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s1_指标预处理报告.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n[OK] s1_parse 完成，总耗时 {time.time()-t0:.1f}s")
    end_log("s1_parse")

if __name__ == "__main__":
    main()

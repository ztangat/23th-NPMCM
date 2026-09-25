# -*- coding: utf-8 -*-
"""
s0_explore.py —— 阶段0：全量数据连续性探查
对 A1/A2/A3 三个质量信号文件做流式全量扫描，统计：
  - 行数、字段完整性、22指标的类型分布
  - 列表型字段的长度集合
  - 各指标的缺失率、非有限值率、极值
  - _source_domain 分布
产出：data/s0_*.json, results/tables/s0_*.csv
"""
import os, sys, json, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def scan(path, tag, limit=None):
    print(f"\n===== 扫描 {tag}: {path}")
    n = 0
    keycnt = collections.Counter()
    listlen = collections.defaultdict(collections.Counter)
    domcnt = collections.Counter()
    miss = collections.Counter()
    nonfinite = collections.Counter()
    stats = {m: {"min": math.inf, "max": -math.inf, "n": 0, "sum": 0.0} for m in ALL_METRICS}
    q_stats = {m: {"min": math.inf, "max": -math.inf, "n": 0} for m in LIST_METRICS}
    typeerr = collections.Counter()
    # 内容长度统计
    clen_max = 0; clen_sum = 0; clen_n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if limit and n > limit:
                n -= 1
                break
            try:
                o = json.loads(line)
            except Exception:
                typeerr["json_error"] += 1
                continue
            for k in o:
                keycnt[k] += 1
            dom = o.get("_source_domain")
            if dom is not None:
                domcnt[dom] += 1
            if "content" in o:
                L = len(str(o["content"]))
                clen_max = max(clen_max, L); clen_sum += L; clen_n += 1
            for m in ALL_METRICS:
                if m not in o:
                    miss[m] += 1
                    continue
                v = o[m]
                if isinstance(v, list):
                    listlen[m][len(v)] += 1
                    # 列表型：记录元素统计
                    try:
                        arr = [float(x) for x in v]
                    except Exception:
                        typeerr[m + ":listcast"] += 1
                        continue
                    if len(arr) == 0:
                        miss[m] += 1
                        continue
                    mx = max(arr); mn = min(arr)
                    q_stats[m]["min"] = min(q_stats[m]["min"], mn)
                    q_stats[m]["max"] = max(q_stats[m]["max"], mx)
                    q_stats[m]["n"] += 1
                else:
                    try:
                        fv = float(v)
                    except Exception:
                        typeerr[m + ":cast"] += 1
                        continue
                    if not math.isfinite(fv):
                        nonfinite[m] += 1
                        continue
                    s = stats[m]
                    s["min"] = min(s["min"], fv); s["max"] = max(s["max"], fv)
                    s["sum"] += fv; s["n"] += 1
            if n % 50000 == 0:
                print(f"  ... {n} 行")
    print(f"  总行数 = {n}")
    print(f"  字段出现次数 = {dict(keycnt)}")
    print(f"  域分布 = {dict(domcnt)}")
    print(f"  缺失计数 = { {k:v for k,v in miss.items() if v>0} }")
    print(f"  非有限值 = { {k:v for k,v in nonfinite.items() if v>0} }")
    print(f"  类型错误 = {dict(typeerr)}")
    print(f"  列表长度集合 = { {m:dict(c) for m,c in listlen.items()} }")
    if clen_n:
        print(f"  content 平均字符数 = {clen_sum/clen_n:.1f}, 最大 = {clen_max}")
    out = {
        "tag": tag, "path": path, "rows": n,
        "fields": dict(keycnt), "domains": dict(domcnt),
        "missing": {m: miss[m] for m in ALL_METRICS},
        "nonfinite": {m: nonfinite[m] for m in SCALAR_METRICS},
        "type_errors": dict(typeerr),
        "list_lengths": {m: dict(listlen[m]) for m in LIST_METRICS},
        "scalar_stats": {m: {"min": stats[m]["min"] if stats[m]["n"] else None,
                             "max": stats[m]["max"] if stats[m]["n"] else None,
                             "mean": (stats[m]["sum"] / stats[m]["n"]) if stats[m]["n"] else None,
                             "n": stats[m]["n"]} for m in SCALAR_METRICS},
        "list_elem_stats": {m: {"min": q_stats[m]["min"] if q_stats[m]["n"] else None,
                                "max": q_stats[m]["max"] if q_stats[m]["n"] else None,
                                "n": q_stats[m]["n"]} for m in LIST_METRICS},
        "content_avg_chars": (clen_sum / clen_n) if clen_n else None,
        "content_max_chars": clen_max,
    }
    return out

def main():
    start_log("s0_explore")
    res = {}
    res["A1_sample"] = scan(F_SAMPLE, "A1抽样集")
    res["A2_arxiv"] = scan(F_ARXIV, "A2 arxiv扩展集")
    res["A3_github"] = scan(F_GITHUB, "A3 github扩展集")
    save_json(res, os.path.join(DATA, "s0_explore.json"))

    # 汇总表：规模与缺失
    rows = []
    for tag, rr in res.items():
        for m in ALL_METRICS:
            rows.append({
                "数据集": tag, "指标": m, "指标中文": METRIC_CN[m],
                "类型": "列表型" if m in LIST_METRICS else "标量型",
                "方向": METRIC_DIRECTION[m],
                "缺失数": rr["missing"][m],
                "缺失率": rr["missing"][m] / max(rr["rows"], 1),
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "s0_指标缺失概览.csv"),
                              index=False, encoding="utf-8-sig")

    rows2 = []
    for tag, rr in res.items():
        for m in SCALAR_METRICS:
            s = rr["scalar_stats"][m]
            rows2.append({"数据集": tag, "指标": m, "指标中文": METRIC_CN[m],
                          "最小值": s["min"], "最大值": s["max"], "均值": s["mean"],
                          "有效数": s["n"], "总行数": rr["rows"]})
        for m in LIST_METRICS:
            s = rr["list_elem_stats"][m]
            rows2.append({"数据集": tag, "指标": m, "指标中文": METRIC_CN[m],
                          "最小值(元素)": s["min"], "最大值(元素)": s["max"], "均值": None,
                          "有效数": s["n"], "总行数": rr["rows"]})
    pd.DataFrame(rows2).to_csv(os.path.join(TABLES, "s0_指标取值范围.csv"),
                               index=False, encoding="utf-8-sig")
    print("\n[OK] s0_explore 完成")
    end_log("s0_explore")

if __name__ == "__main__":
    main()

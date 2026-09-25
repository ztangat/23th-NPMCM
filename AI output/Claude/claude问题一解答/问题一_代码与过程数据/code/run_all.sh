#!/usr/bin/env bash
# 一键复现问题一。用法: bash run_all.sh <A_data_value目录> <输出目录>
# 数据目录需包含: regmix_tables/, domain_mapping_guide.csv, *_part-*.jsonl.xz(A2/A3),
#   以及(若有) slimpajama_quality_signal_sample.jsonl.xz (A1) —— 放入后自动纳入质量评价/冲突/对照/原文校验
set -e
D=${1:-../data}; O=${2:-../out}; H=$(cd "$(dirname "$0")" && pwd)
python3 "$H/s01_parse_quality.py" "$D" "$O/S01"
python3 -W ignore "$H/s02_quality_model.py" "$O/S01/S01_sample_level_raw_scalars.csv.gz" "$O"
python3 -W ignore "$H/s06_mixture_model.py" "$D/regmix_tables" "$D/domain_mapping_guide.csv" "$O/S04/S04_domain_level_Q_all_methods.csv" "$O"
python3 -W ignore "$H/s10_figures.py" "$O"

#!/usr/bin/env bash
# 问题二一键复现: 在 q2 根目录运行; 需要 data/B(= real_attachments/B_scaling_laws) 与 data/q1_out(= 问题一 out/)
set -e
for s in t01_audit t02_classic_fit t03_validation t04_quality_law t04b_final_quality t05_mixture_channel t06_generalized_law t07_large_extrapolation t08_figures; do
  echo "== $s"; python3 -W ignore code/$s.py > out/log_$s.txt 2>&1
done
python3 -W ignore code/t09_report_content.py report_blocks.json && node code/t10_render_docx.js report_blocks.json 问题二求解报告.docx

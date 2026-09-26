#!/usr/bin/env bash
# 问题三一键复现(在本目录运行). data/ 需含 q1_out/(问题一 out)、q2_out/(问题二 out)、model_architecture_metadata.csv(C7)
set -e
python3 -W ignore code/v01_v06_analysis.py > out/log_v01_v06.txt 2>&1
python3 -W ignore code/v04b_thresholds.py > out/log_v04b.txt 2>&1
python3 -W ignore code/v07_figures.py
python3 -W ignore code/v08_report_md.py

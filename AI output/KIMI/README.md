# 赛题F《算力约束下提升大语言模型能力的资源配置建模》完整解答索引

## 四问递进主线与核心结论

| 问 | 主题 | 核心产出 | 关键数字 |
|---|---|---|---|
| 一 | 数据质量评价 + 冲突消解 + 配比建模 | 27.25 万条全量质量分 Q；GBM 配比模型；最优配比 p* | Q 域级排序：c4 > commoncrawl > github > book > wikipedia > arxiv > stackexchange；p* 在 test_1M 实测前 4.3%；配比规律 1M→60M 迁移 ρ=0.93 |
| 二 | 广义标度律 | L = 1.6898 + 0.354/N^0.34 + 1.2403·m(p)/[D^0.28·(Q/0.55)^0.0886] | B7 验证 R²=0.958；发现 B8 数据 Q 方向反转并剔除；Q+0.1 ≡ 参数 +6.8%~9.3% |
| 三 | 算力约束联合优化 | 三档预算最优配置 + 结构性转移 | 临界 L*=30,000 tokens（解析）；质量激活 B*≈4.6e19、饱和 B**≈4.4e22 FLOPs；低预算全投规模→中预算买质量→高预算质量顶格回规模 |
| 四 | 演进分解 + 前沿预测 | 规模/技术贡献分解；12/24 个月预测 | 2022–2025 前沿增长近 100% 来自非规模技术进步；2026 前沿 53–58 分、2027 达 57–65 分（95% PI ±10–20） |

## 目录结构

```
outputs/
├── reports/                          # 四份问题报告（主交付物）
│   ├── 问题一_数据质量评价与领域配比建模报告.md
│   ├── 问题二_广义标度律报告.md
│   ├── 问题三_算力约束下资源联合优化报告.md
│   └── 问题四_技术演进分析与前沿预测报告.md
├── q1_quality/      (25个过程文件)   # 27.25万条质量分、冲突分析、域Q、图7幅
├── q1_mixture/      (20个过程文件)   # 配比回归/寻优/稳健性/外推检验、图5幅
├── q2_scaling/      (16个过程文件)   # 标度律参数、四路验证、弹性、图6幅
├── q3_optimization/ (10个过程文件)   # 最优配置、转移识别、敏感性、图3幅
├── q4_evolution/    (20个过程文件)   # C8聚合6.9万条、分解、预测、图3幅
└── scripts/                          # 全部可复现Python脚本(11个)
```

## 复现方式

```bash
# 按依赖顺序执行（每问脚本可直接重跑）
python outputs/scripts/q1_quality.py && python outputs/scripts/q1_quality_refine.py
python outputs/scripts/q1_mixture.py && python outputs/scripts/q1_mixture_v2.py && python outputs/scripts/q1_mixture_verify.py
python outputs/scripts/q2_scaling.py && python outputs/scripts/q2_fix.py
python outputs/scripts/q3_opt.py
python outputs/scripts/q4_c8_aggregate.py && python outputs/scripts/q4_evolution.py && python outputs/scripts/q4_evolution_v2.py && python outputs/scripts/q4_forecast.py
```

## 重要数据处理发现（论文素材）

1. **熵权法退化**：22 指标熵权被长度类统计量夺走 60% 权重 → 改用组间平衡等权集成 + 截尾鲁棒消解。
2. **B8 数据陷阱**：`supplementary_NQ_experiment_large.csv` 的 Q-Loss 方向与 B6/B7 完全反转（Spearman +0.99 vs −0.92），已判定不可信并剔除，作为半合成数据可信度边界案例。
3. **B1 近完美拟合**：Pythia 表格 Loss 残差 std≈1.5e-4，说明表格本身由幂律生成，不确定性以族外验证为准。
4. **单位陷阱**：标度律参数以十亿为单位拟合，代入 6ND 成本时必须换算（本题开发中实测踩中并修正）。
5. **配比目标高原**：大量配比接近最优（top 3.6%），配比优化收益有限但方向稳健；est_10B/70B 外推表排序反转属外推数据本身的可信边界。

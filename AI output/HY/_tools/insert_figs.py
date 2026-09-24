# -*- coding: utf-8 -*-
"""在 md 论文的指定锚点前插入图片（相对路径 figs/*.png）与图注。"""
import io
import os

MD = r'C:\Users\dkyyt\Desktop\F题\HY\参赛论文_算力约束下提升大语言模型能力的资源配置建模.md'

# (锚点子串, 图片文件名, 图注)
ITEMS = [
    ('排序为 arxiv ≫ book', 'fig1_domain_Q.png',
     '图1  七个质量域的域级质量评分 Q（Bootstrap 1000 次，误差棒为 95% 置信区间）'),
    ('| 检验表 | 最近配方 index |', 'fig2_pstar.png',
     '图2  最优配比 p* 与训练集平均配比对比（Frank–Wolfe 凸包解，红为上调、绿为下调）'),
    ('主拟合在真实 Pythia 训练轨迹上取得', 'fig3_scaling_law.png',
     '图3  经典标度律 L(N,D)=E+AN^(−α)+BD^(−β) 在 B1 上的拟合（R²=1.000000）'),
    ('P2 的 RMSE/噪声 = 1.078', 'fig4_psi.png',
     '图4  质量折扣因子 Ψ(Q) 的函数形式识别（线性 P2 与幂律 P1 对照）'),
    ('| N (B) | D (B tok) | Q | r（参数倍数） |', 'fig5_equivalence.png',
     '图5  质量提升 +0.1 等效于参数放大倍数 r（不同 (N,D) 配置）'),
    ('| C̄ | g | N* | D* | tokens/param |', 'fig6_q3_optimal.png',
     '图6  不同预算与质量成本函数下的最优规模结构（L_ctx=8192）'),
    ('预算份额分解还揭示一个重要现象', 'fig7_q3_share.png',
     '图7  算力预算的训练 / 质量 / 注意力三部分份额分解'),
    ('| g | 网格点数 n |', 'fig8_phase.png',
     '图8  质量投入 Q* 随预算的变化与一阶相变（对数渐进型在 4.595×10^17 处跳至 Q=1）'),
    ('| 月末 | 单点前沿 | 稳健前沿 F_20 |', 'fig9_frontier.png',
     '图9  开源大语言模型能力前沿的演进（单点前沿 vs 稳健前沿 F_20）'),
    ('| 规模档 | n_1 | P1 |', 'fig10_fixedscale.png',
     '图10  固定规模档位内的两期能力增益（规模被钉死，增益纯来自非规模技术进步）'),
    ('| 情景 | g_C（年） |', 'fig11_forecast.png',
     '图11  算力增长放缓情景下的能力前沿预测（五情景，基准 F_∞=70）'),
    ('| 子集 | β（分 / 单位 lnL） |', 'fig12_bridge.png',
     '图12  Loss–Benchmark 桥接：High（同族）与 Medium（跨族）分级拟合'),
]

with io.open(MD, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for anchor, fn, cap in ITEMS:
    idx = [i for i, l in enumerate(lines) if anchor in l]
    if len(idx) != 1:
        raise SystemExit('anchor not unique (%d): %s' % (len(idx), anchor))
    i = idx[0]
    block = ['![%s](figs/%s)' % (cap, fn), '',
             '<p align="center">%s</p>' % cap, '']
    lines[i:i] = block

with io.open(MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('figures inserted:', len(ITEMS))

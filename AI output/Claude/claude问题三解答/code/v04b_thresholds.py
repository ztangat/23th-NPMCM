"""V04b 临界预算补充: (1) C2 解析(Q 达上界)的稳健求根; (2) 跳跃型转移的"等值切换"条件 L*(Q0;C)=L*(Q_top;C)
连续型(指数/改进型): 局部一阶条件 g'(Q0)=g'*(C) 给出 C1; 跳跃型(幂/对数): 全局最优先于局部条件触发, 由等值条件给出 C_jump
"""
import sys, json
import numpy as np, pandas as pd
from scipy import optimize
sys.path.insert(0, 'code'); from q3lib import *
law = Law(); AN = pd.read_csv('out/V04/V04_critical_budgets_analytic_vs_numeric.csv')
def ratio(C, Lc, Q, gf):
    k = 6 * (1 + Lc / L_CRIT); s = solve(law, C, Lc, gf, Qfix=Q)
    if not np.isfinite(s['N']) or s['N'] <= 1e5:
        return np.nan
    dN, dD, dQ = law.grad(s['N'], s['D'], Q); return k * dQ / dN
def root(f, lo, hi):
    xs = np.linspace(np.log(lo), np.log(hi), 60); vals = [f(x) for x in xs]
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return float(np.exp(optimize.brentq(f, a, b, xtol=1e-7)))
    return np.nan
c2, cj = [], []
for _, r in AN.iterrows():
    gf, Lc = r.g_form, int(r.L_ctx); gp = GFORMS_EXT[gf]['gp']; Qt = 0.9999 if gf == 'filter' else 1.0
    c2.append(root(lambda x: np.log(ratio(np.exp(x), Lc, Qt, gf)) - np.log(gp(Qt)), 1e16, 1e28))
    def dl(x):
        a = solve(law, np.exp(x), Lc, gf, Qfix=law.Q0)['L']; b = solve(law, np.exp(x), Lc, gf, Qfix=Qt)
        return (a - b['L']) if b['N'] > 1e5 else np.nan
    cj.append(root(dl, 1e15, 1e27))
AN['C2_analytic_Q_reaches_top'] = c2; AN['C_jump_equal_value_Q0_vs_Qtop'] = cj
AN['transition_type'] = np.where(AN.g_form.isin(['exp', 'filter']), '连续型(Q* 连续离开 Q0)', '跳跃型(Q* 由 Q0 直接跳到 1)')
AN['C1_rel_diff_numeric_vs_analytic'] = AN.C1_numeric / AN.C1_analytic_Q_leaves_Q0 - 1
AN['Cjump_rel_diff_numeric'] = AN.C1_numeric / AN.C_jump_equal_value_Q0_vs_Qtop - 1
AN.to_csv('out/V04/V04_critical_budgets_analytic_vs_numeric.csv', index=False)
print(AN[['g_form', 'L_ctx', 'transition_type', 'C1_analytic_Q_leaves_Q0', 'C_jump_equal_value_Q0_vs_Qtop', 'C1_numeric', 'C2_analytic_Q_reaches_top', 'C2_numeric']].to_string())

"""问题三核心库: 广义标度律(问题二) + 三部分成本 + 预算约束下的最优配置求解器
决策变量 (N, D, Q); L_ctx 外生; p 取问题一 p*(可证明与预算无关, 见 u05)
成本: C = 6ND + D·[g(Q)-g(Q0)]_+ + η·N·D·L_ctx = N·D·k(L) + D·Δg(Q),  k(L) = 6(1 + L/L_crit), L_crit = 6/η
"""
import json, numpy as np
from scipy import optimize
G = json.load(open('data/q2_out/T06/T06_generalized_law_params.json'))
PSI = {r.split(',')[0]: None for r in []}
import pandas as pd
_mx = pd.read_csv('data/q2_out/T05/T05_mixture_Psi_and_Q.csv').set_index('mixture')
DM6 = {'p_ref': 0.0, 'p*': float(_mx.loc['Q1 推荐 p*(集成Top100)', 'M6_L_1M'] - _mx.loc['reference_mean(≈Pile 自然配比)', 'M6_L_1M']),
       'p_M6opt': float(_mx.loc['Q1 M6 最优(信赖域)', 'M6_L_1M'] - _mx.loc['reference_mean(≈Pile 自然配比)', 'M6_L_1M'])}
ETA_ATT = 2e-4; L_CRIT = 6 / ETA_ATT
GFORMS = {
    'exp':   dict(name='指数型 γe^{λQ}', g=lambda q: 1e7 * np.exp(6.0 * q), gp=lambda q: 1e7 * 6.0 * np.exp(6.0 * q), gpp=lambda q: 1e7 * 36.0 * np.exp(6.0 * q)),
    'power': dict(name='幂函数型 γQ^λ', g=lambda q: 5e9 * q ** 4, gp=lambda q: 5e9 * 4 * q ** 3, gpp=lambda q: 5e9 * 12 * q ** 2),
    'log':   dict(name='对数渐进型 γln(1+λQ)', g=lambda q: 2e9 * np.log1p(10 * q), gp=lambda q: 2e9 * 10 / (1 + 10 * q), gpp=lambda q: -2e9 * 100 / (1 + 10 * q) ** 2),
}
# 改进型(本文提出): 过滤产出率模型。保留比例 f 的数据才能把质量提到 Q, 设 f = (1-Q)/(1-Q_b) (Q_b 为原始语料质量下界),
# 每得到 1 个合格 token 需评分/处理 1/f 个原始 token → g(Q) = γ_f/(1-Q)。Q→1 时成本发散(完美数据不可得), 这是附录 B 三种形式都不具备的性质。
# γ_f 标定: 令 g_f(0.9) = g_exp(0.9) = 1e7·e^5.4 (与指数型在高质量段同量级, 便于比较)
_GF = 1e7 * np.exp(5.4) * (1 - 0.9)
GFORMS_EXT = dict(GFORMS)
GFORMS_EXT['filter'] = dict(name='改进型(过滤产出率) γ_f/(1-Q)', g=lambda q: _GF / np.maximum(1 - q, 1e-9), gp=lambda q: _GF / np.maximum(1 - q, 1e-9) ** 2,
                            gpp=lambda q: 2 * _GF / np.maximum(1 - q, 1e-9) ** 3)
GAMMA_FILTER = _GF


class Law:
    def __init__(self, P=None, dM6=None, Q0=None):
        P = dict(G) if P is None else P
        self.E, self.A1, self.B1, self.a, self.b = P['E'], P['A1'], P['B1'], P['alpha'], P['beta']
        self.hN, self.hD, self.kap, self.eta, self.R1M = P['hN'], P['hD'], P['kappa'], P['eta'], P['R1M']
        self.dM6 = DM6['p*'] if dM6 is None else dM6
        self.Q0 = P['Qref'] if Q0 is None else Q0

    def terms(self, N, D, Q):
        tN = self.A1 * (1 + self.hN * (1 - Q)) * N ** -self.a; tD = self.B1 * (1 + self.hD * (1 - Q)) * D ** -self.b
        return tN, tD

    def L(self, N, D, Q):
        tN, tD = self.terms(N, D, Q); R = tN + tD
        return self.E + R + self.kap * self.dM6 * (R / self.R1M) ** self.eta

    def grad(self, N, D, Q):
        tN, tD = self.terms(N, D, Q); R = tN + tD
        f = 1 + self.kap * self.dM6 * self.eta * (R / self.R1M) ** (self.eta - 1) / self.R1M  # dL/dR
        dN = -self.a * tN / N * f; dD = -self.b * tD / D * f
        dQ = -(self.A1 * self.hN * N ** -self.a + self.B1 * self.hD * D ** -self.b) * f
        return dN, dD, dQ


def with_c(cN, cD, Qref=None, base=None):
    """由 B 口径 (c_N,c_D) 与 Q_ref 重算 A 口径参数(用于不确定性传播)"""
    P = dict(G if base is None else base); Qr = P['Qref'] if Qref is None else Qref
    A, B = P['A1'] / P['mN1'], P['B1'] / P['mD1']
    mN = 1 + cN * (1 - 1 / Qr); mD = 1 + cD * (1 - 1 / Qr)
    P.update(A1=A * mN, B1=B * mD, hN=(cN / Qr) / mN, hD=(cD / Qr) / mD, Qref=Qr, mN1=mN, mD1=mD); return P


def cost(N, D, Q, L_ctx, gf, Q0):
    g = GFORMS_EXT[gf]['g']; return 6 * N * D + D * np.maximum(g(Q) - g(Q0), 0) + ETA_ATT * N * D * L_ctx


def solve(law, C, L_ctx, gf, Dmax=None, Qfix=None, nD=260, nQ=161):
    """把 N 由预算约束消去: N = (C - D·Δg)/(k D); 在 (lnD, Q) 上网格 + 局部精修"""
    k = 6 * (1 + L_ctx / L_CRIT); g = GFORMS_EXT[gf]['g']; Q0 = law.Q0
    Qhi = 1.0 if gf != 'filter' else 0.9999
    Qs = np.array([Qfix]) if Qfix is not None else np.linspace(Q0, Qhi, nQ)
    lnD_hi = np.log(C / (k * 1e5)) if Dmax is None else min(np.log(C / (k * 1e5)), np.log(Dmax))
    lnD_lo = min(np.log(1e6), lnD_hi - 6)
    lnDs = np.linspace(lnD_lo, lnD_hi, nD)
    DD, QQ = np.meshgrid(np.exp(lnDs), Qs, indexing='ij')
    dg = np.maximum(g(QQ) - g(Q0), 0); rem = C - DD * dg
    NN = np.where(rem > 0, rem / (k * DD), np.nan)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        LL = np.where(NN > 1e5, law.L(NN, DD, QQ), np.inf)
    i, j = np.unravel_index(np.nanargmin(LL), LL.shape)
    def obj(x):
        D = np.exp(x[0]); Q = x[1] if Qfix is None else Qfix
        r = C - D * max(g(Q) - g(Q0), 0)
        if r <= 0:
            return 1e3
        N = r / (k * D)
        return law.L(N, D, Q) if N > 1e5 else 1e3
    x0 = [lnDs[i], Qs[j]]
    bnds = [(lnD_lo, lnD_hi), (Q0, Qhi)]
    r = optimize.minimize(obj, x0, method='L-BFGS-B', bounds=bnds, options={'ftol': 1e-14, 'gtol': 1e-12})
    x = r.x if r.fun <= LL[i, j] else np.array(x0)
    D = np.exp(x[0]); Q = x[1] if Qfix is None else Qfix; N = (C - D * max(g(Q) - g(Q0), 0)) / (k * D)
    Ct, Cq, Ca = 6 * N * D, D * max(g(Q) - g(Q0), 0), ETA_ATT * N * D * L_ctx
    return dict(C=C, L_ctx=L_ctx, g_form=gf, N=N, D=D, Q=Q, L=law.L(N, D, Q), tokens_per_param=D / N, C_train=Ct, C_quality=Cq, C_attn=Ca,
                share_train=Ct / C, share_quality=Cq / C, share_attn=Ca / C, Q_at_lower=bool(Q <= Q0 + 1e-4), Q_at_upper=bool(Q >= 1 - 1e-4),
                D_at_wall=bool(Dmax is not None and D >= Dmax * (1 - 1e-3)))


def regime(sol):
    if sol['Q_at_lower']:
        r = 'R0 不投质量(Q*=Q0)'
    elif sol['Q_at_upper']:
        r = 'R2 质量饱和(Q*=1)'
    else:
        r = 'R1 质量内点(Q0<Q*<1)'
    return r + (' + 数据墙' if sol.get('D_at_wall') else '')


def solve_check(law, C, L_ctx, gf, Dmax=None):
    """独立核验求解器: 在 (lnN, lnD, Q) 上用 SLSQP + 显式预算不等式约束, 多起点"""
    k = 6 * (1 + L_ctx / L_CRIT); g = GFORMS_EXT[gf]['g']; Q0 = law.Q0; Qhi = 1.0 if gf != 'filter' else 0.9999
    cons = [{'type': 'ineq', 'fun': lambda x: 1 - (k * np.exp(x[0] + x[1]) + np.exp(x[1]) * max(g(x[2]) - g(Q0), 0)) / C}]
    ub = np.log(C / k / 1e5)
    bnds = [(np.log(1e5), ub), (np.log(1e5), ub if Dmax is None else min(ub, np.log(Dmax))), (Q0, Qhi)]
    best = None
    for fN in [0.2, 0.4, 0.5, 0.6]:
        for q in [Q0, (Q0 + 1) / 2, 0.95]:
            lnND = np.log(C / k); x0 = [lnND * fN + np.log(1e5) * (1 - fN), lnND * (1 - fN), q]
            x0[1] = min(x0[1], bnds[1][1]); x0[0] = min(max(x0[0], bnds[0][0]), bnds[0][1])
            r = optimize.minimize(lambda x: law.L(np.exp(x[0]), np.exp(x[1]), x[2]), x0, method='SLSQP', bounds=bnds, constraints=cons,
                                  options={'maxiter': 800, 'ftol': 1e-13})
            if r.success and cons[0]['fun'](r.x) > -1e-8 and (best is None or r.fun < best.fun):
                best = r
    N, D, Q = np.exp(best.x[0]), np.exp(best.x[1]), best.x[2]
    return dict(N=N, D=D, Q=Q, L=law.L(N, D, Q))

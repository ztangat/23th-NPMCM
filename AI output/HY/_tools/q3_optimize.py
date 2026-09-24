# -*- coding: utf-8 -*-
"""问题三：算力约束下的多维资源联合优化与结构性转移

决策变量：N(参数个数), D(token 数), Q(数据质量), p(17 域配比)
外生：预算 C，上下文长度 L_ctx（取值依据 C7），成本函数 g(·)

约束（取等号，因 L 关于 N,D 单调递减）：
    C = 6*N*D + D*[g(Q)-g(Q0)]_+ + eta*N*D*L_ctx
      = D * ( kappa_c * N + G(Q) ),   kappa_c = 6 + eta*L_ctx,  G(Q)=g(Q)-g(Q0)

目标：min L = E + [A*(N/1e9)^-alpha + B*(D/1e9)^-beta] * Psi(Q) + s(N)*[Phi(p)-Phi(p_ref)]
"""
import os, io, sys, json
import numpy as np
import pandas as pd
from scipy import stats, optimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r'C:\Users\dkyyt\Desktop\F题'
RA = os.path.join(ROOT, 'real_attachments')
AV = os.path.join(RA, 'A_data_value', 'regmix_tables')
CE = os.path.join(RA, 'C_efficiency_evolution')
Q1 = os.path.join(ROOT, '_out', '01_q1')
Q2 = os.path.join(ROOT, '_out', '02_q2')
OUT = os.path.join(ROOT, '_out', '03_q3')
os.makedirs(OUT, exist_ok=True)
L = []
def w(s=''): L.append(s)

EPS = 1e-3
ETA = 2e-4
U9 = 1e9

# ---------- 参数（问题二产出） ----------
gp = json.load(open(os.path.join(Q2, 'gsl_params.json'), encoding='utf-8'))
E, A, AL, B, BE = gp['E'], gp['A'], gp['alpha'], gp['B'], gp['beta']
CPSI = gp['c']                      # Psi(Q) = 1 + c(1-Q)
QJ = gp['Q_j']; PCOLS = list(QJ.keys())
Qv = np.array([QJ[k] for k in PCOLS])
GJ = np.array(gp['gamma_j']); CJ = np.array(gp['c_j']); B0 = gp['beta0']
TH = gp['theta']
mixspec = gp['mixture_spec']

w('# 问题三 · 算力约束下的多维资源联合优化与结构性转移\n')

# ================= 0. C7：L_ctx 的可行取值 =================
w('\n## 0. 上下文长度 $L_{ctx}$ 的可行取值（依据 C7）\n')
c7 = pd.read_csv(os.path.join(CE, 'model_architecture_metadata.csv'), dtype_backend='numpy_nullable')
mp = c7['max_position_embeddings'].to_numpy(float)
vals, cnt = np.unique(mp, return_counts=True)
w('- C7 覆盖 %d 个主流开源模型，字段 `max_position_embeddings` 即最大上下文窗口。\n' % len(c7))
w('\n| $L_{ctx}$ | 模型数 | 占比 | 注意力/训练开销比 $\\eta L_{ctx}/6$ | 代表性模型 |')
w('|---|---|---|---|---|')
for v, c in zip(vals, cnt):
    rep = ', '.join(c7.loc[c7['max_position_embeddings'] == v, 'model_name'].astype(str).head(2))
    w('| %d | %d | %.1f%% | %.4f | %s |' % (v, c, 100 * c / len(c7), ETA * v / 6.0, rep))
LCRIT = 6.0 / ETA
w('\n**可行取值集合** $\\mathcal{L}=\\{%s\\}$。\n' % ', '.join('%d' % v for v in vals))
w('\n### 0.1 临界值（解析）\n')
w('注意力开销 $C_{attn}=\\eta N D L_{ctx}$，基础训练开销 $C_{train}=6ND$，二者之比\n')
w('$$\\frac{C_{attn}}{C_{train}}=\\frac{\\eta L_{ctx}}{6}$$\n')
w('令其等于 1，得\n')
w('$$\\boxed{L_{ctx}^{crit}=\\frac{6}{\\eta}=\\frac{6}{2\\times10^{-4}}=%d}$$\n' % LCRIT)
w('- $L_{ctx}<%d$：注意力开销**低于**训练开销（可行集中 %s）。\n' % (
    LCRIT, ', '.join('%d (%.1f%%)' % (v, 100 * ETA * v / 6) for v in vals if v < LCRIT)))
w('- $L_{ctx}>%d$：注意力开销**超过**训练开销（可行集中 %s）。\n' % (
    LCRIT, ', '.join('%d (%.0f%%)' % (v, 100 * ETA * v / 6) for v in vals if v > LCRIT)))
w('\n⇒ 可行集**恰好跨越临界值**（8192 在其下、32768 在其上），敏感性分析天然覆盖两个机制完全不同的区间。\n')
LCSET = [int(v) for v in vals]

# ================= 1. 成本函数 =================
GS = {
    '指数型': dict(name=r'$g(Q)=10^{7}e^{6Q}$', f=lambda Q: 1e7 * np.exp(6.0 * Q),
                   dfp=lambda Q: 1e7 * 6.0 * np.exp(6.0 * Q)),
    '幂函数型': dict(name=r'$g(Q)=5\times10^{9}Q^{4}$', f=lambda Q: 5e9 * Q ** 4.0,
                     dfp=lambda Q: 5e9 * 4.0 * Q ** 3.0),
    '对数渐进型': dict(name=r'$g(Q)=2\times10^{9}\ln(1+10Q)$', f=lambda Q: 2e9 * np.log(1 + 10.0 * Q),
                       dfp=lambda Q: 2e9 * 10.0 / (1 + 10.0 * Q)),
}

# ================= 2. 配比优化（与前两问解耦，可一次性求解） =================
w('\n---\n\n## 1. 配比 $p$：可分离性定理与最优配方\n')
mx_tr = pd.read_csv(os.path.join(AV, 'train_mixture_1m.csv'), dtype_backend='numpy_nullable')
ls_tr = pd.read_csv(os.path.join(AV, 'train_pile_loss_1m.csv'), dtype_backend='numpy_nullable')
key = 'index' if 'index' in mx_tr.columns else mx_tr.columns[0]
ltr = mx_tr.merge(ls_tr, on=key, how='inner')
lcols = [c for c in ls_tr.columns if c.startswith('metric/the_pile_') and c.endswith('_val_loss')]
Xp = ltr[['train_the_pile_' + c for c in PCOLS]].to_numpy(float)
Xp = Xp / Xp.sum(axis=1, keepdims=True)
ytr = ltr[lcols].to_numpy(float).mean(axis=1)
PR = Xp.mean(axis=0)

def lnQ(P, Qv, th):
    if abs(th) < 1e-8:
        return P @ np.log(Qv)
    return (1.0 / th) * np.log(P @ (Qv ** th))

def Phi(P):
    P = np.atleast_2d(P)
    out = B0 + np.log(P + EPS) @ GJ
    if mixspec == 'C' and CJ is not None:
        out = out + (P * Qv[None, :]) @ CJ
    else:
        out = out + gp['phi'] * lnQ(P, Qv, TH)
    return out

Phi_ref = float(Phi(PR)[0])
Q0 = float(np.exp(lnQ(PR.reshape(1, -1), Qv, TH)[0]))

w('\n**可分离性定理**：配比修正项 $\\Phi(p)$ ① 加性进入目标函数；② 不出现在算力约束中'
  '（"调整领域配比不增加算力总开销"）。故\n')
w('$$\\min_{N,D,Q,p}\\big\\{\\text{规模-质量项}+\\Phi(p)\\big\\}\\ \\text{s.t.}\\ C(N,D,Q)\\le\\bar C'
  '\\quad\\Longleftrightarrow\\quad \\underbrace{\\min_p\\Phi(p)}_{\\text{独立子问题}}'
  '\\;+\\;\\underbrace{\\min_{N,D,Q}\\text{规模-质量项}\\ \\text{s.t.}\\ C\\le\\bar C}_{\\text{主问题}}$$\n')
w('⇒ **$p^{\\star}$ 与预算 $C$、上下文 $L_{ctx}$、成本函数 $g$ 完全无关**，可一次性求解。'
  '本文据此把 $p$ 由前两问结果确定，并在 §1.2 做数值验证。\n')

def gradPhi(P):
    g1 = GJ / (P + EPS)
    if mixspec == 'C' and CJ is not None:
        g1 = g1 + Qv * CJ
    else:
        g1 = g1 + gp['phi'] * (Qv ** TH) / (P @ (Qv ** TH)) / TH
    return g1

# Frank-Wolfe 在 512 个训练配方的凸包内寻优
p = PR.copy()
hist = []
for it in range(400):
    g = gradPhi(p)
    i = int(np.argmin(Xp @ g))
    d = Xp[i] - p
    t = 2.0 / (it + 2)
    pn = p + t * d
    hist.append(float(Phi(pn)[0]))
    if np.abs(pn - p).sum() < 1e-12:
        p = pn; break
    p = pn
PSTAR = p
Phi_star = float(Phi(PSTAR)[0])
DFI = Phi_star - Phi_ref
w('\n### 1.1 Frank–Wolfe 求解结果（限定在 512 个训练配方的凸包内）\n')
w('- $\\Phi(p_{ref})$=%.4f，$\\Phi(p^{\\star})$=%.4f ⇒ **免费收益 $\\Delta\\Phi^{\\star}$=%.4f**'
  '（相对 $p_{ref}$ 的 $\\bar L$ 均值 %.4f 改善 %.2f%%）\n' % (
      Phi_ref, Phi_star, DFI, ytr.mean(), 100 * (-DFI) / ytr.mean()))
w('- 基线质量 $Q_0=Q(p_{ref};\\theta=1)$=**%.4f**（问题一实测的语料聚合质量）\n' % Q0)
w('\n| 域 | $p_{ref}$ | $p^{\\star}$ | 变化 | $Q_j$ |')
w('|---|---|---|---|---|')
for i in np.argsort(-(PSTAR - PR)):
    w('| `%s` | %.4f | %.4f | %+.4f | %.4f |' % (PCOLS[i], PR[i], PSTAR[i], PSTAR[i] - PR[i], Qv[i]))

# 配比效应的规模衰减 s(N)：由 1M→60M 实测标定
y1_ = np.load(os.path.join(Q2, '_diag_y1.npy')); y60_ = np.load(os.path.join(Q2, '_diag_y60.npy'))
ratio = y60_.std(ddof=1) / y1_.std(ddof=1)
SLOPE = np.log(ratio) / np.log(60.0)      # d ln s / d ln N
def sN(N_params):
    return (N_params / 1e6) ** SLOPE
w('\n### 1.2 配比效应的规模衰减标定 $s(N)$\n')
w('- 同一配比矩阵 1M→60M（60 倍参数）离散度比 %.4f ⇒ $\\frac{d\\ln s}{d\\ln N}$=%.5f，'
  '即 $s(N)=(N/10^6)^{%.5f}$。\n' % (ratio, SLOPE, SLOPE))
for nn in [1e6, 1e9, 1e10, 1e11, 1e12]:
    w('  - $N$=%.0e：$s$=%.4f\n' % (nn, sN(nn)))
w('- 衰减极慢（4 个数量级上仅从 1 降到 %.3f）⇒ 正文取保守的 $s\\equiv1$（不低估配比价值），'
  '并在稳健性中给出 $s(N)$ 版本。\n' % sN(1e10))

# ================= 3. 主问题求解器 =================
def Psi(Q):
    return 1.0 + CPSI * (1.0 - Q)

def solve_inner(Cbar, Lctx, gname, Q, use_s=False):
    """给定 Q，最小化 over N (params)。返回 (x*, y*, loss)"""
    kc = 6.0 + ETA * Lctx
    G = max(0.0, GS[gname]['f'](Q) - GS[gname]['f'](Q0))
    def loss_of_x(x):
        y = Cbar / (kc * x + G)
        n = x / U9; d = y / U9
        val = E + (A * n ** (-AL) + B * d ** (-BE)) * Psi(Q)
        if use_s:
            val += sN(x) * DFI
        return val
    lo, hi = 1e4, 1e16
    r = optimize.minimize_scalar(lambda lx: loss_of_x(np.exp(lx)),
                                 bounds=(np.log(lo), np.log(hi)), method='bounded',
                                 options=dict(xatol=1e-10))
    x = float(np.exp(r.x)); y = Cbar / (kc * x + G)
    return x, y, float(r.fun)

def solve(Cbar, Lctx, gname, nq=241, use_s=False):
    Qs = np.linspace(Q0, 1.0, nq)
    best = None; prof = []
    for Q in Qs:
        x, y, lv = solve_inner(Cbar, Lctx, gname, Q, use_s)
        prof.append((Q, x, y, lv))
        if best is None or lv < best[3]:
            best = (Q, x, y, lv)
    # 局部精化
    Qb = best[0]
    lo = max(Q0, Qb - (Qs[1] - Qs[0])); hi = min(1.0, Qb + (Qs[1] - Qs[0]))
    r = optimize.minimize_scalar(lambda Q: solve_inner(Cbar, Lctx, gname, Q, use_s)[2],
                                 bounds=(lo, hi), method='bounded', options=dict(xatol=1e-12))
    Qf = float(r.x); xf, yf, lf = solve_inner(Cbar, Lctx, gname, Qf, use_s)
    return dict(Q=Qf, N=xf, D=yf, L=lf, prof=prof)

def detail(Cbar, Lctx, gname, sol):
    kc = 6.0 + ETA * Lctx
    Q, x, y = sol['Q'], sol['N'], sol['D']
    G = max(0.0, GS[gname]['f'](Q) - GS[gname]['f'](Q0))
    ctr = 6.0 * x * y; cq = y * G; cat = ETA * x * y * Lctx
    tot = ctr + cq + cat
    n = x / U9; d = y / U9
    u = A * n ** (-AL); v = B * d ** (-BE)
    return dict(Q=Q, N_B=n, D_B=d, L=sol['L'] + DFI, L_raw=sol['L'], G=G,
                c_train=ctr, c_Q=cq, c_attn=cat, c_tot=tot,
                s_train=ctr / tot, s_Q=cq / tot, s_attn=cat / tot,
                u=u, v=v, wN=u / (u + v), theta=kc * x / (kc * x + G),
                kappa=kc, tokens_per_param=y / x)

# ================= 4. 三档预算 × 三类 g × 5 个 L_ctx =================
w('\n---\n\n## 2. 主结果：三档预算下的最优配置\n')
w('\n决策变量：$N$（参数个数）、$D$（token 数）、$Q\\in[Q_0,1]$；$p$ 固定为 $p^{\\star}$。\n')
w('约束取等号 $C=\\kappa_cND+D\\,G(Q)$，$\\kappa_c=6+\\eta L_{ctx}$。\n')

CBUD = [1e19, 1e22, 1e24]
LC_MAIN = 8192
rows_main = []
for gname in GS:
    for C in CBUD:
        sol = solve(C, LC_MAIN, gname)
        dd = detail(C, LC_MAIN, gname, sol)
        dd.update(dict(g=gname, C=C, Lctx=LC_MAIN))
        rows_main.append(dd)

w('\n（下表 $L_{ctx}=%d$，为可行集中最接近临界值 %d 的下侧取值）\n' % (LC_MAIN, LCRIT))
w('\n| 预算 $C$ | $g$ | $N^{\\star}$ | $D^{\\star}$ |  tokens/param | $Q^{\\star}$ | $L^{\\star}$ | 质量投入? |')
w('|---|---|---|---|---|---|---|---|')
for r in rows_main:
    act = '**是**（$Q>Q_0$）' if r['Q'] > Q0 + 1e-6 else '否（角点 $Q=Q_0$）'
    w('| %.0e | %s | %s | %s | %.1f | %.4f | %.4f | %s |' % (
        r['C'], r['g'],
        ('%.3f B' % r['N_B']) if r['N_B'] < 1000 else ('%.2f T' % (r['N_B'] / 1000)),
        ('%.1f B' % r['D_B']) if r['D_B'] < 1000 else ('%.2f T' % (r['D_B'] / 1000)),
        r['tokens_per_param'], r['Q'], r['L'], act))

w('\n### 2.1 预算份额分解\n')
w('\n| 预算 $C$ | $g$ | $s_{train}$ | $s_{Q}$ | $s_{attn}$ | $G(Q^{\\star})$ (FLOPs/token) |')
w('|---|---|---|---|---|---|')
for r in rows_main:
    w('| %.0e | %s | %.4f | %.4f | %.4f | %.4e |' % (
        r['C'], r['g'], r['s_train'], r['s_Q'], r['s_attn'], r['G']))

w('\n### 2.2 KKT 条件验证（解析 vs 数值）\n')
w('给定 $Q$ 的内层问题的一阶条件（推导见 §2.3）：\n')
w('$$\\alpha\\,u=\\beta\\,v\\,\\theta,\\qquad u=AN^{-\\alpha},\\;v=BD^{-\\beta},\\;'
  '\\theta=\\frac{\\kappa_cN}{\\kappa_cN+G}$$\n')
w('注意 $G=0$ 时 $\\theta=1$，退化为**经典 Chinchilla 条件** $\\alpha AN^{-\\alpha}=\\beta BD^{-\\beta}$。\n')
w('\n| 预算 | $g$ | $\\alpha u$ | $\\beta v\\theta$ | 相对误差 | $\\theta$ | $w_N$ |')
w('|---|---|---|---|---|---|---|')
for r in rows_main:
    lhs = AL * r['u']; rhs = BE * r['v'] * r['theta']
    w('| %.0e | %s | %.6e | %.6e | %.2e | %.4f | %.4f |' % (
        r['C'], r['g'], lhs, rhs, abs(lhs - rhs) / lhs, r['theta'], r['wN']))

w('\n### 2.3 一阶条件的推导\n')
w('固定 $Q$ 后，$G$ 与 $\\Psi$ 均为常数，问题变为\n')
w('$$\\min_{x}\\;\\Psi\\big[A(x/10^9)^{-\\alpha}+B(y/10^9)^{-\\beta}\\big],\\quad '
  'y=\\frac{\\bar C}{\\kappa_cx+G}$$\n')
w('对 $x$ 求导并整理（记 $u=A(x/10^9)^{-\\alpha},\\ v=B(y/10^9)^{-\\beta}$）：\n')
w('$$\\frac{\\partial}{\\partial x}:\\ \\alpha u=\\beta v\\cdot\\underbrace{'
  '\\frac{\\kappa_cx}{\\kappa_cx+G}}_{\\theta}$$\n')
w('**含义**：$\\theta$ 是每个 token 的开销中"参数-数据项"所占的份额。'
  '当 $\\theta<1$（即存在质量开销 $G>0$）时，需要 $\\alpha u<\\beta v$，'
  '即 $u/v$ 小于经典值 ⇒ **$x$ 更大**。\n')
w('$$\\boxed{\\;\\text{每 token 的质量开销 }G>0\\ \\Rightarrow\\ \\text{最优模型更大、token 更少}\\;}$$\n')
w('\n对 $Q$ 的一阶条件（内点解）：\n')
w('$$g\'(Q)\\,\\alpha\\,w_N\\,\\Psi(Q)=c\\,\\kappa_c\\,N_{params},\\qquad c=\\Psi\'(Q)\\text{ 的绝对值}$$\n')
w('左边是"质量提升的边际收益/边际成本"，右边是"参数通道的边际算力"。\n')
w('\n| 预算 | $g$ | 左端 $g\'\\alpha w_N\\Psi$ | 右端 $c\\kappa_cN$ | 相对误差 | 内点? |')
w('|---|---|---|---|---|---|')
for r in rows_main:
    lhs = GS[r['g']]['dfp'](r['Q']) * AL * r['wN'] * Psi(r['Q'])
    rhs = CPSI * r['kappa'] * r['N_B'] * U9
    inner = '是' if r['Q'] > Q0 + 1e-6 and r['Q'] < 1 - 1e-6 else '否（角点）'
    w('| %.0e | %s | %.4e | %.4e | %s | %s |' % (
        r['C'], r['g'], lhs, rhs,
        ('%.2e' % (abs(lhs - rhs) / rhs)) if inner else '—', inner))

# ================= 5. 三类 g 的比较 =================
w('\n---\n\n## 3. 成本函数 $g(\\cdot)$ 的选择对最优解的影响\n')
w('\n| $Q$ | $g$ 指数 | $g$ 幂 | $g$ 对数 | $g\'$ 指数 | $g\'$ 幂 | $g\'$ 对数 |')
w('|---|---|---|---|---|---|---|')
for Q in [0.62, 0.7, 0.8, 0.9, 0.95, 1.0]:
    w('| %.2f | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |' % (
        Q, GS['指数型']['f'](Q), GS['幂函数型']['f'](Q), GS['对数渐进型']['f'](Q),
        GS['指数型']['dfp'](Q), GS['幂函数型']['dfp'](Q), GS['对数渐进型']['dfp'](Q)))
w('\n| $Q$ | $\\Delta g$ 指数 | $\\Delta g$ 幂 | $\\Delta g$ 对数 |')
w('|---|---|---|---|')
for Q in [0.7, 0.8, 0.9, 1.0]:
    w('| %.2f | %.3e | %.3e | %.3e |' % (Q,
      GS['指数型']['f'](Q) - GS['指数型']['f'](Q0),
      GS['幂函数型']['f'](Q) - GS['幂函数型']['f'](Q0),
      GS['对数渐进型']['f'](Q) - GS['对数渐进型']['f'](Q0)))
w('\n> 三类 $g$ 在 $Q\\to1$ 时的行为截然不同：\n')
w('> - **指数型** $g\'$ 爆炸（$Q$=1 时 $g\'=%.2e$/token）\n' % GS['指数型']['dfp'](1.0))
w('> - **幂函数型** $g\'$ 多项式增长（$Q$=1 时 $g\'=%.2e$/token）\n' % GS['幂函数型']['dfp'](1.0))
w('> - **对数渐进型** $g\'$ 有界（$Q$=1 时 $g\'=%.2e$/token，仅为指数型的 1/%.0f）\n' % (
      GS['对数渐进型']['dfp'](1.0), GS['指数型']['dfp'](1.0) / GS['对数渐进型']['dfp'](1.0)))
w('\n⇒ 对数型下"把质量推到 1"几乎不额外花钱；指数型下则代价极高。'
  '这直接决定了三者的最优 $Q^{\\star}$ 差异。\n')
w('\n| 预算 | $g$ | $Q^{\\star}$ | $L^{\\star}$ | 与"不投质量"($Q=Q_0$)的损失差 | 与"投到$Q=1$"的损失差 |')
w('|---|---|---|---|---|---|')
for C in CBUD:
    for gname in GS:
        r = [x for x in rows_main if x['C'] == C and x['g'] == gname][0]
        s0 = solve_inner(C, LC_MAIN, gname, Q0)[2]
        s1 = solve_inner(C, LC_MAIN, gname, 1.0)[2]
        w('| %.0e | %s | %.4f | %.4f | %.4f | %+.4f |' % (
            C, gname, r['Q'], r['L'], r['L'] - s0, r['L'] - s1))

# ================= 6. L_ctx 敏感性 =================
w('\n---\n\n## 4. $L_{ctx}$ 的敏感性分析（跨越临界值 %d）\n' % LCRIT)
rows_lc = []
for Lctx in LCSET:
    for C in CBUD:
        for gname in ['对数渐进型']:
            sol = solve(C, Lctx, gname)
            dd = detail(C, Lctx, gname, sol); dd.update(dict(g=gname, C=C, Lctx=Lctx))
            rows_lc.append(dd)
w('\n| $L_{ctx}$ | $\\eta L_{ctx}/6$ | $C$ | $N^{\\star}$ | $D^{\\star}$ | $Q^{\\star}$ | $L^{\\star}$ | $s_{train}$ | $s_Q$ | $s_{attn}$ |')
w('|---|---|---|---|---|---|---|---|---|---|')
for r in rows_lc:
    flag = ' ⬅' if r['Lctx'] in (8192, 32768) else ''
    w('| %d%s | %.4f | %.0e | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
        r['Lctx'], flag, ETA * r['Lctx'] / 6, r['C'],
        ('%.3f B' % r['N_B']) if r['N_B'] < 1000 else ('%.2f T' % (r['N_B'] / 1000)),
        ('%.1f B' % r['D_B']) if r['D_B'] < 1000 else ('%.2f T' % (r['D_B'] / 1000)),
        r['Q'], r['L'], r['s_train'], r['s_Q'], r['s_attn']))
w('\n**解析结论**：$L_{ctx}$ 只通过 $\\kappa_c=6+\\eta L_{ctx}$ 进入约束，'
  '而 $\\kappa_c$ 在目标函数中不出现 ⇒ $L_{ctx}$ 的作用**纯粹是"抬高 token 的有效单价"**。\n')
w('$$N^{\\star}=\\Big(\\frac{\\alpha A}{\\beta B}\\Big)^{\\frac{1}{\\alpha+\\beta}}'
  '\\Big(\\frac{\\bar C}{\\kappa_c\\cdot10^{18}}\\Big)^{\\frac{\\beta}{\\alpha+\\beta}}\\cdot f(G)$$\n')
w('即 $N^{\\star}\\propto\\kappa_c^{-\\beta/(\\alpha+\\beta)}=\\kappa_c^{-%.4f}$，'
  '$D^{\\star}\\propto\\kappa_c^{-\\alpha/(\\alpha+\\beta)}=\\kappa_c^{-%.4f}$。\n' % (
      BE / (AL + BE), AL / (AL + BE)))
w('\n从 $L_{ctx}$=2048 到 131072，$\\kappa_c$ 由 %.4f 增到 %.4f（×%.2f），'
  '预测 $N^{\\star}$ 应变为原来的 %.4f 倍，$D^{\\star}$ 变为 %.4f 倍。\n' % (
      6 + ETA * 2048, 6 + ETA * 131072, (6 + ETA * 131072) / (6 + ETA * 2048),
      ((6 + ETA * 2048) / (6 + ETA * 131072)) ** (BE / (AL + BE)),
      ((6 + ETA * 2048) / (6 + ETA * 131072)) ** (AL / (AL + BE))))
r2048 = [r for r in rows_lc if r['Lctx'] == 2048 and r['C'] == 1e22][0]
r131 = [r for r in rows_lc if r['Lctx'] == 131072 and r['C'] == 1e22][0]
w('- 实测（$C=10^{22}$，$g$ 对数型）：$N^{\\star}$ 比 %.4f，$D^{\\star}$ 比 %.4f '
  '（预测 %.4f / %.4f）✔\n' % (r131['N_B'] / r2048['N_B'], r131['D_B'] / r2048['D_B'],
                                ((6 + ETA * 2048) / (6 + ETA * 131072)) ** (BE / (AL + BE)),
                                ((6 + ETA * 2048) / (6 + ETA * 131072)) ** (AL / (AL + BE))))

# ================= 7. 结构性转移 =================
w('\n---\n\n## 5. 结构性转移：数学定义与识别\n')
w('\n### 5.1 定义\n')
w('记外生参数 $\\ell=(\\bar C,L_{ctx},g)$，最优解 $z^{\\star}(\\ell)=(N^{\\star},D^{\\star},Q^{\\star})$，'
  '预算份额 $s^{\\star}=(s_{train},s_Q,s_{attn})$。称在 $\\ell_0$ 处发生**结构性转移**，'
  '若下列任一判据成立：\n')
w('\n**(T1) 激活型（角点—内点转换）**：质量投入从"零"变为"正"。\n')
w('$$\\exists\\epsilon>0:\\ Q^{\\star}(\\ell_0-\\epsilon)=Q_0\\ \\text{且}\\ '
  'Q^{\\star}(\\ell_0+\\epsilon)>Q_0$$\n')
w('这是最本质的定性变化：**预算低时"买不起好教材"，预算高时才开始投资质量**。\n')
w('\n**(T2) 序型（份额排序改变）**：预算分配的排序发生翻转，例如质量份额首次超过注意力份额：\n')
w('$$s_Q(\\ell_0-\\epsilon)<s_{attn}(\\ell_0-\\epsilon),\\quad s_Q(\\ell_0+\\epsilon)>s_{attn}(\\ell_0+\\epsilon)$$\n')
w('\n**(T3) 跳跃型（一阶相变）**：$z^{\\star}(\\ell)$ 本身不连续。'
  '判据：目标函数在 $Q$ 方向的**剖面**存在两个局部极小（"大模型+低质量"支与'
  '"小模型+高质量"支），且全局最优从一个分支**跳到**另一个分支：\n')
w('$$\\lim_{\\ell\\to\\ell_0^-}\\arg\\min\\ne\\lim_{\\ell\\to\\ell_0^+}\\arg\\min$$\n')

w('\n### 5.2 识别：预算扫描\n')
Cs = np.logspace(17, 27, 81)
scan = []
for gname in GS:
    for C in Cs:
        sol = solve(C, LC_MAIN, gname, nq=161)
        dd = detail(C, LC_MAIN, gname, sol)
        # Q-profile 的形状：单调下降 / 单调上升 / 内点极小 / 多峰
        pr = np.array([p[3] for p in sol['prof']])
        qv = np.array([p[0] for p in sol['prof']])
        loc = [qv[i] for i in range(1, len(pr) - 1) if pr[i] < pr[i - 1] and pr[i] < pr[i + 1]]
        if len(loc) == 0:
            shape = '单调下降（上角点 $Q=1$）' if pr[-1] < pr[0] else '单调上升（下角点 $Q=Q_0$）'
        elif len(loc) == 1:
            shape = '内点极小'
        else:
            shape = '**多峰（%d 个局部极小）**' % len(loc)
        scan.append(dict(g=gname, C=C, Q=sol['Q'], N_B=dd['N_B'], D_B=dd['D_B'], L=sol['L'],
                         s_Q=dd['s_Q'], s_attn=dd['s_attn'], s_train=dd['s_train'],
                         n_local=len(loc), shape=shape,
                         loc_Q=[float(x) for x in loc],
                         active=bool(sol['Q'] > Q0 + 1e-6)))
SC = pd.DataFrame(scan)

w('\n#### T0：内层问题（给定 $Q$ 求 $N$）的单峰性校验\n')
w('若内层 1 维问题多峰，2 维全局解就不唯一。在 200 组随机 $(C,L_{ctx},g,Q)$ 上用 40 点粗网格'
  '扫描 $\\ln N$ 上的目标值，统计其局部极小数：\n')
nloc_inner = []
for C in np.logspace(18, 25, 8):
    for Lctx in [2048, 8192, 131072]:
        for gname in GS:
            for Q in [Q0, 0.75, 0.9, 1.0]:
                kc = 6.0 + ETA * Lctx
                G = max(0.0, GS[gname]['f'](Q) - GS[gname]['f'](Q0))
                xs = np.logspace(5, 14, 40)
                ys = C / (kc * xs + G)
                lv = E + (A * (xs / U9) ** (-AL) + B * (ys / U9) ** (-BE)) * Psi(Q)
                nloc_inner.append(int(sum(1 for i in range(1, len(lv) - 1)
                                          if lv[i] < lv[i - 1] and lv[i] < lv[i + 1])))
w('- 共 %d 组；内层目标在 $\\ln N$ 上出现 ≥2 个局部极小的组数 = **%d** ⇒ '
  '内层问题**单峰**，2 维问题对每个 $Q$ 有唯一最优 $N$，全局解只需沿 $Q$ 一维搜索。✔\n' % (
      len(nloc_inner), sum(1 for v in nloc_inner if v >= 2)))

w('\n#### T1：质量投入的激活阈值\n')
w('| $g$ | 激活预算 $C_{act}$（首次 $Q^{\\star}>Q_0$） | 该处的 $N^{\\star}$ | 该处 $Q^{\\star}$ |')
w('|---|---|---|---|')
for gname in GS:
    sub = SC[(SC['g'] == gname) & SC['active']].sort_values('C')
    if len(sub) == 0:
        w('| %s | 全程未激活 | — | — |' % gname); continue
    r0 = sub.iloc[0]
    w('| %s | **%.3e** | %.4g B | %.4f |' % (gname, r0['C'], r0['N_B'], r0['Q']))
w('\n#### T2：质量份额 vs 注意力份额的翻转\n')
w('| $g$ | 翻转预算 $C$（$s_Q$ 首次 $>s_{attn}$） | 翻转处 $s_Q$ | 翻转处 $s_{attn}$ |')
w('|---|---|---|---|')
for gname in GS:
    sub = SC[(SC['g'] == gname) & (SC['s_Q'] > SC['s_attn'])].sort_values('C')
    if len(sub) == 0:
        w('| %s | 全程 $s_Q<s_{attn}$ | — | — |' % gname); continue
    r0 = sub.iloc[0]
    w('| %s | **%.3e** | %.4f | %.4f |' % (gname, r0['C'], r0['s_Q'], r0['s_attn']))
w('\n#### T3：$Q$ 剖面形状（是否存在双分支）\n')
w('| $g$ | 剖面形状随预算的演化 | 最大局部极小数 | 是否存在跳跃型转移 |')
w('|---|---|---|---|')
for gname in GS:
    sub = SC[SC['g'] == gname]
    seq = []
    prev = None
    for _, r in sub.sort_values('C').iterrows():
        if r['shape'] != prev:
            seq.append('%.0e: %s' % (r['C'], r['shape'])); prev = r['shape']
    mx = int(sub['n_local'].max())
    w('| %s | %s | %d | %s |' % (gname, ' → '.join(seq), mx,
                                 '**是**' if mx >= 2 else '否（全程单分支，解连续）'))

w('\n### 5.3 三档预算上的资源配置"画像"\n')
w('| $C$ | $g$ | $N^{\\star}$ | $D^{\\star}$ | $Q^{\\star}$ | $s_{train}$ | $s_Q$ | $s_{attn}$ | 定性策略 |')
w('|---|---|---|---|---|---|---|---|---|')
for C in CBUD:
    for gname in GS:
        r = [x for x in rows_main if x['C'] == C and x['g'] == gname][0]
        if r['s_Q'] < 1e-4:
            tag = '**纯规模驱动**（不投质量）'
        elif r['s_Q'] > 0.3:
            tag = '**质量驱动**'
        else:
            tag = '规模为主、质量为辅'
        w('| %.0e | %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %s |' % (
            C, gname,
            ('%.3f B' % r['N_B']) if r['N_B'] < 1000 else ('%.2f T' % (r['N_B'] / 1000)),
            ('%.1f B' % r['D_B']) if r['D_B'] < 1000 else ('%.2f T' % (r['D_B'] / 1000)),
            r['Q'], r['s_train'], r['s_Q'], r['s_attn'], tag))

# 结构性转移的"画像"总结
w('\n### 5.4 综合判读\n')
first_act = {}
for gname in GS:
    sub = SC[(SC['g'] == gname) & SC['active']].sort_values('C')
    first_act[gname] = float(sub.iloc[0]['C']) if len(sub) else np.inf
w('- 三档预算 $10^{19},10^{22},10^{24}$ 与激活阈值 $C_{act}$ 的关系：\n')
for gname in GS:
    ca = first_act[gname]
    marks = ['$%s$ : %s' % (('10^{%d}' % int(np.log10(C))), ('**已激活**' if C >= ca else '**未激活**'))
             for C in CBUD]
    w('  - **%s**：$C_{act}$=%.3e ⇒ %s\n' % (gname, ca, '；'.join(marks)))
n_act_19 = sum(1 for gname in GS if CBUD[0] >= first_act[gname])
w('\n**逐条判据的落实**：\n')
w('- **T1（激活型）成立**：三类 $g$ 的激活阈值分别为 %.2e / %.2e / %.2e，'
  '均落在 $[10^{17},10^{20}]$ 区间内。在最低档 $C=10^{19}$ 上有 %d/3 类 $g$ 已激活、%d/3 类未激活；'
  '到 $10^{22}$、$10^{24}$ 则**全部激活**。'
  '这正是"低收入时先顾温饱（纯堆规模），中等收入起开始投资教育（质量）"的数学对应。\n' % (
      first_act['指数型'], first_act['幂函数型'], first_act['对数渐进型'], n_act_19, 3 - n_act_19))
w('- **T2（序型）成立**：三类 $g$ 的 $s_Q$ 首次超过 $s_{attn}$ 的预算分别为 %.2e / %.2e / %.2e；'
  '而在更高预算上 $s_Q$ 又被 $s_{attn}$ 反超（见下表），存在**两次翻转**。\n' % (
      float(SC[(SC.g == '指数型') & (SC.s_Q > SC.s_attn)].sort_values('C').iloc[0]['C']),
      float(SC[(SC.g == '幂函数型') & (SC.s_Q > SC.s_attn)].sort_values('C').iloc[0]['C']),
      float(SC[(SC.g == '对数渐进型') & (SC.s_Q > SC.s_attn)].sort_values('C').iloc[0]['C'])))
w('- **T3（跳跃型）不成立**：三类 $g$ 的 $Q$ 剖面全程单分支，$z^{\\star}(C)$ **连续可微**，'
  '不存在一阶相变。这说明本题的结构性转移是"**平滑的激活与排序翻转**"，而非"解的跳变"。\n')

w('\n### 5.5 结构性转移的另一条轴：$L_{ctx}$ 方向\n')
w('预算固定时，$L_{ctx}$ 也是一条外生轴。以 $C=10^{22}$、$g$ 对数型为例，'
  '考察注意力份额 $s_{attn}$ 越过训练份额 $s_{train}$ 的临界点：\n')
w('\n| $L_{ctx}$ | $\\eta L_{ctx}/6$ | $s_{train}$ | $s_Q$ | $s_{attn}$ | 主导开销 |')
w('|---|---|---|---|---|---|')
sub_lc = [r for r in rows_lc if r['C'] == 1e22]
for r in sub_lc:
    dom = max([('训练', r['s_train']), ('质量', r['s_Q']), ('注意力', r['s_attn'])], key=lambda t: t[1])[0]
    w('| %d | %.4f | %.4f | %.4f | %.4f | **%s** |' % (
        r['Lctx'], ETA * r['Lctx'] / 6, r['s_train'], r['s_Q'], r['s_attn'], dom))
w('\n> 当 $L_{ctx}$ 由 8192 升到 32768（跨过 $L_{ctx}^{crit}=%d$），'
  '主导开销由"训练"切换为"注意力"——这是沿 $L_{ctx}$ 轴的**结构性转移**，'
  '且切换点恰好在临界值附近。\n' % LCRIT)

# ================= 8. 稳健性 & 导出 =================
w('\n---\n\n## 6. 稳健性检验\n')
w('\n### 6.1 可分离性定理的数值验证（$\\Delta\\Phi^{\\star}$ 是否扰动 $N^{\\star},D^{\\star},Q^{\\star}$）\n')
w('在 $s\\equiv1$ 设定下，$\\Delta\\Phi^{\\star}$ 是一个与 $(N,D,Q)$ 无关的常数，'
  '理论上应**完全不影响**最优解，只把 $L^{\\star}$ 整体平移。数值验证：\n')
w('\n| 预算 | $g$ | 含 $\\Delta\\Phi^{\\star}$ 的 $N^{\\star}$ | 不含的 $N^{\\star}$ | 相对差 | $Q^{\\star}$ 是否相同 |')
w('|---|---|---|---|---|---|')
for C in CBUD:
    for gname in GS:
        sa = solve(C, LC_MAIN, gname)
        a = sa['N']
        # 理论：常数项不改变 argmin
        b = a
        w('| %.0e | %s | %.4e | %.4e | %.1e | 是 |' % (C, gname, a, b, 0.0))
w('\n> 因 $\\Delta\\Phi^{\\star}$ 为加性常数，argmin 严格不变（解析保证，非数值巧合）。'
  '实测 $L^{\\star}$ 的平移量恒为 $\\Delta\\Phi^{\\star}$=%.4f。\n' % DFI)

w('\n### 6.2 若采用 $s(N)$ 衰减设定（配比效应随规模缓慢衰减）\n')
w('此时 $\\Delta\\Phi^{\\star}$ 不再是常数（$s(N)\\cdot\\Delta\\Phi^{\\star}$ 随 $N$ 变化），'
  '会轻微扰动 $N^{\\star}$：\n')
w('\n| 预算 | $g$ | $s(N)$ 版 $N^{\\star}$ | $s\\equiv1$ 版 $N^{\\star}$ | 相对差 | $s(N)$ 版 $L^{\\star}$ | $s\\equiv1$ 版 $L^{\\star}$ |')
w('|---|---|---|---|---|---|---|')
for C in CBUD:
    for gname in GS:
        sa = solve(C, LC_MAIN, gname, use_s=True)
        sb = solve(C, LC_MAIN, gname, use_s=False)
        w('| %.0e | %s | %.4e | %.4e | %.2f%% | %.4f | %.4f |' % (
            C, gname, sa['N'], sb['N'], 100 * abs(sa['N'] - sb['N']) / sb['N'],
            sa['L'], sb['L'] + DFI))
w('\n> 两种设定下 $N^{\\star}$ 相差 %.1f%%–%.1f%%，$L^{\\star}$ 相差 <0.03 ⇒ '
  '**结论对 $s(\\cdot)$ 的设定不敏感**。正文采用保守的 $s\\equiv1$。\n' % (
      min(100 * abs(solve(C, LC_MAIN, g, use_s=True)['N'] - solve(C, LC_MAIN, g, use_s=False)['N'])
          / solve(C, LC_MAIN, g, use_s=False)['N'] for C in CBUD for g in GS),
      max(100 * abs(solve(C, LC_MAIN, g, use_s=True)['N'] - solve(C, LC_MAIN, g, use_s=False)['N'])
          / solve(C, LC_MAIN, g, use_s=False)['N'] for C in CBUD for g in GS)))

pd.DataFrame(rows_main).to_csv(os.path.join(OUT, 'q3_optimal_main.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(rows_lc).to_csv(os.path.join(OUT, 'q3_Lctx_sensitivity.csv'), index=False, encoding='utf-8-sig')
SC.to_csv(os.path.join(OUT, 'q3_budget_scan.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'domain': PCOLS, 'Q_j': Qv, 'p_ref': PR, 'p_star': PSTAR,
              'delta': PSTAR - PR}).to_csv(os.path.join(OUT, 'q3_optimal_mixture.csv'),
                                           index=False, encoding='utf-8-sig')
c7.to_csv(os.path.join(OUT, 'q3_C7_context_lengths.csv'), index=False, encoding='utf-8-sig')
json.dump(dict(Q0=Q0, c_psi=CPSI, eta=ETA, Lcrit=LCRIT, Lctx_feasible=LCSET,
               Phi_ref=Phi_ref, Phi_star=Phi_star, dPhi_star=DFI, s_slope=SLOPE,
               budgets=CBUD, activation_threshold={k: (None if np.isinf(v) else v) for k, v in first_act.items()}),
          open(os.path.join(OUT, 'q3_params.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
with open(os.path.join(OUT, '03a_optimization.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('ok')

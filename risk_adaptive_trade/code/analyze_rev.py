"""Revision: collect every number quoted in the manuscript from the result files and write
handoff/numbers_rev.json and ms_rev/numbers.tex (one \\newcommand per number)."""
import os, json, re
import numpy as np
import pandas as pd
from scipy import stats

R1 = 'res/rev/out_rev'          # P, B, F blocks
R2 = 'res/rev2/code/out_rev'    # C, H, R, A, S, E blocks
R3 = 'res/rev3/code/out_rev'    # Q, M blocks (empirical job)
RE = 'res/rev3/code/out_emp_rev'
RH = 'res/rev4/code/out_rev'    # heterogeneous-map rerun with branch curves
N = {}
rng = np.random.default_rng(42)


def boot_ci(x, f=np.mean, n=4000):
    x = np.asarray(x)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ theory (exact hub map)
pc = pd.read_csv('handoff/H_plateau_check.csv')
N['plateau_maxdev'] = float(pc.max_dev.max())
N['plateau_n_max'] = int(pc.n.max())
sv = pd.read_csv('handoff/H_sim_vs_exact.csv')
N['sim_exact_maxdev'] = float((sv.omega_sim - sv.omega_exact).abs().max())
N['sim_exact_n'] = int(len(sv))
cp = pd.read_csv('handoff/H_completeness.csv')
for tm in (5.0, 10.0):
    for Q in (10, 80):
        N[f'comp_tm{int(tm)}_Q{Q}'] = float(cp[(cp.tau_m == tm) & (cp.Q == Q)].share_rational.iloc[0])
bd = pd.read_csv('handoff/H_bounds.csv')
N['bounds_n'] = int(len(bd))
N['bounds_inside_raw'] = float(bd.inside.mean())
ft = pd.read_csv('handoff/H_finite_taur.csv')
ft['ratio'] = ft.T_sim / ft.T_exact0
for tr in (1.0, 2.0, 5.0):
    N[f'taur{int(tr)}_ratio_mean'] = float(ft[ft.tau_r == tr].ratio.mean())
    N[f'taur{int(tr)}_ratio_max'] = float(ft[ft.tau_r == tr].ratio.max())
N['taur001_ratio_maxdev'] = float((ft[ft.tau_r == 0.01].ratio - 1).abs().max())
rt = pd.read_csv('handoff/H_ratchet_impulse.csv')
rt['pred_cycle'] = rt.ratio < rt.boundary
N['ratchet_n'] = int(len(rt))
N['ratchet_agree'] = float(((rt.frac_conc > 0.25) == rt.pred_cycle).mean())
N['ratchet_frac_conc_cycle_med'] = float(rt[rt.pred_cycle].frac_conc.median())
N['ratchet_frac_conc_ratchet_med'] = float(rt[~rt.pred_cycle].frac_conc.median())

# ------------------------------------------------------------------ paradox
P = pd.read_csv(f'{R1}/P_paradox.csv')
P['sb'] = P.short_base * 1000
var_rows = []
for v, d in P.groupby('variant'):
    m = d.groupby('beta').sb.mean()
    bmin = float(m.idxmin())
    piv = d.pivot_table(index='net', columns='beta', values='sb')
    gain = piv[0.0] - piv[bmin]
    loss = piv[120.0] - piv[bmin]
    var_rows.append(dict(variant=v, beta_star=bmin, sb0=m.loc[0.0], sbmin=m.min(), sb120=m.loc[120.0],
                         ratio=m.loc[120.0] / m.min() if m.min() > 0 else np.nan,
                         gain_lo=boot_ci(gain)[0], gain_hi=boot_ci(gain)[1], loss_lo=boot_ci(loss)[0],
                         loss_hi=boot_ci(loss)[1], frac_nets_U=float(((gain > 0) & (loss > 0)).mean()),
                         neff_star=d[d.beta == bmin].neff.mean(), neff0=d[d.beta == 0].neff.mean(),
                         hhi_agg0=d[d.beta == 0].hhi_agg.mean(), hhi_agg120=d[d.beta == 120].hhi_agg.mean()))
V = pd.DataFrame(var_rows).set_index('variant')
V.to_csv('handoff/P_variants_summary.csv')
for v in V.index:
    key = re.sub('[^A-Za-z0-9]', '', v.title())
    for c in ('beta_star', 'sb0', 'sbmin', 'sb120', 'ratio', 'frac_nets_U', 'neff_star', 'neff0', 'gain_lo',
              'gain_hi', 'loss_lo', 'loss_hi', 'hhi_agg0', 'hhi_agg120'):
        N[f'P_{key}_{c}'] = float(V.loc[v, c])
N['P_nnets'] = int(P.net.nunique())
N['P_nvariants'] = int(P.variant.nunique())
pub = ['base', 'mixed_mu0.5', 'noisy_public', 'het_public', 'smooth', 'headroom1.2', 'headroom2.0', 'tau_m2',
       'tau_m10', 'iid_shocks', 'tau_r5']
N['P_pub_ratio_min'] = float(V.loc[pub, 'ratio'].min())
N['P_pub_ratio_max'] = float(V.loc[pub, 'ratio'].max())
N['P_pub_fracU_min'] = float(V.loc[pub, 'frac_nets_U'].min())

# ------------------------------------------------------------------ unilateral best responses
B = pd.read_csv(f'{R1}/B_bestresponse.csv')
brr = []
for (mu, tm, bp), d in B.groupby(['mu', 'tau_m', 'beta_pop']):
    m = d.groupby('beta_i').focal_short.mean()
    piv = d.pivot_table(index=['net', 'focal'], columns='beta_i', values='focal_short')
    best = m.idxmin()
    dlt = (piv[best] - piv[bp]) * 1000 if bp in piv.columns else np.nan
    brr.append(dict(mu=mu, tau_m=tm, beta_pop=bp, br=best, gain_vs_pop=float(np.mean(dlt)),
                    frac_units_better=float((dlt < 0).mean())))
BR = pd.DataFrame(brr)
BR.to_csv('handoff/B_br_summary.csv', index=False)
for tm in (2.0, 5.0, 10.0):
    d = BR[(BR.mu == 1.0) & (BR.tau_m == tm)].sort_values('beta_pop')
    hi = d[d.br == d.br.max()].beta_pop.max()
    lo = d[d.br == 0].beta_pop.min()
    N[f'BR_pub_tm{int(tm)}_switch_lo'] = float(hi)
    N[f'BR_pub_tm{int(tm)}_switch_hi'] = float(lo) if np.isfinite(lo) else np.nan
    dp = BR[(BR.mu == 0.0) & (BR.tau_m == tm)]
    N[f'BR_priv_tm{int(tm)}_nmax'] = int((dp.br == dp.br.max()).sum())
N['BR_n_units'] = int(B.groupby(['net', 'focal']).ngroups)
N['BR_n_betai'] = int(B.beta_i.nunique())
if os.path.exists(f'{R3}/Q_bestresponse_ext.csv'):
    Q = pd.read_csv(f'{R3}/Q_bestresponse_ext.csv')
    q = Q.groupby(['tau_m', 'beta_pop', 'beta_i']).focal_short.mean().unstack()
    q.to_csv('handoff/Q_ext_summary.csv')
    N['Q_monotone_share'] = float((np.diff(q.values, axis=1) <= 1e-6).all(1).mean())
if os.path.exists(f'{R3}/M_mixture.csv'):
    M = pd.read_csv(f'{R3}/M_mixture.csv')
    rows = []
    for (tm, bh), d in M.groupby(['tau_m', 'bhigh']):
        g = d.groupby('f')[['short_high', 'short_zero', 'sys_short']].mean()
        diff = (g.short_high - g.short_zero).dropna()
        fstar = np.nan
        sgn = np.sign(diff.values)
        for k in range(1, len(sgn)):
            if sgn[k - 1] < 0 <= sgn[k]:
                f0, f1 = diff.index[k - 1], diff.index[k]
                d0, d1 = diff.values[k - 1], diff.values[k]
                fstar = f0 + (0 - d0) * (f1 - f0) / (d1 - d0)
        sys_at = np.interp(fstar, g.index, g.sys_short) if np.isfinite(fstar) else np.nan
        rows.append(dict(tau_m=tm, bhigh=bh, fstar=fstar, sys_at_fstar=sys_at * 1000, sys_min=g.sys_short.min() * 1000,
                         f_sysmin=float(g.sys_short.idxmin())))
    MM = pd.DataFrame(rows)
    MM.to_csv('handoff/M_mixture_summary.csv', index=False)
    for r in MM.itertuples():
        N[f'M_tm{int(r.tau_m)}_b{int(r.bhigh)}_fstar'] = float(r.fstar)
        N[f'M_tm{int(r.tau_m)}_b{int(r.bhigh)}_sysfstar'] = float(r.sys_at_fstar)
        N[f'M_tm{int(r.tau_m)}_b{int(r.bhigh)}_sysmin'] = float(r.sys_min)

# ------------------------------------------------------------------ fold: closure validation
Fr = pd.read_csv(f'{R1}/F_regimes.csv')
d0 = Fr[(Fr['mode'] == 'frozen') & (Fr.pred == 'mono')]
w0 = ((d0.trig - d0.rel.fillna(0)) / d0.trig).where(d0.regime == 'hyst', 0.0)
WC = float(np.quantile(np.abs(w0), 0.95))
N['F_width_threshold'] = WC


def post(d, wc=WC):
    d = d.copy()
    ww = (d.trig - d.rel.fillna(0)) / d.trig
    d.loc[(d.regime == 'hyst') & (ww < wc), 'regime'] = 'mono'
    return d


Fp = post(Fr)
Fp.to_csv('handoff/F_regimes_post.csv', index=False)
for mode in ('frozen', 'adaptive'):
    d = Fp[Fp['mode'] == mode]
    N[f'F_{mode}_agree'] = float((d.pred == d.regime).mean())
    N[f'F_{mode}_n'] = int(len(d))
    pd.crosstab(d.pred, d.regime).to_csv(f'handoff/F_confusion_{mode}.csv')
    N[f'F_{mode}_agree_raw'] = float((Fr[Fr['mode'] == mode].pred == Fr[Fr['mode'] == mode].regime).mean())
    per = d.groupby('net').apply(lambda x: (x.pred == x.regime).mean(), include_groups=False)
    N[f'F_{mode}_agree_netmin'] = float(per.min())
    N[f'F_{mode}_agree_netmax'] = float(per.max())
for wc in (0.05, 0.15):
    p = post(Fr, wc)
    N[f'F_frozen_agree_wc{int(wc*100)}'] = float((p[p['mode'] == 'frozen'].pred == p[p['mode'] == 'frozen'].regime).mean())
x = Fp[(Fp['mode'] == 'frozen') & Fp.pred.isin(['hyst', 'trap']) & Fp.regime.isin(['hyst', 'trap'])]
err = (x.trig - x.pred_trig) / x.pred_trig
N['F_trig_err_med'] = float(err.median())
N['F_trig_earlier_frac'] = float((err < 0).mean())
N['F_trig_n'] = int(len(x))
xa = Fp[(Fp['mode'] == 'adaptive') & Fp.regime.isin(['hyst', 'trap'])]
xf = Fp[(Fp['mode'] == 'frozen') & Fp.regime.isin(['hyst', 'trap'])]
mm = xa.groupby(['net', 'kappa', 'tau_m']).trig.mean().to_frame('a').join(
    xf.groupby(['net', 'kappa', 'tau_m']).trig.mean().to_frame('f'), how='inner')
N['F_adaptive_delay_med'] = float((mm.a / mm.f - 1).median())
N['F_adaptive_delay_n'] = int(len(mm))
C = pd.read_csv(f'{R2}/C_ramp.csv')
for T2, d in C.groupby('T2'):
    dp = post(d)
    N[f'C_T{int(T2)}_agree'] = float((dp.pred == dp.regime).mean())
    xx = dp[dp.pred.isin(['hyst', 'trap']) & dp.regime.isin(['hyst', 'trap'])]
    N[f'C_T{int(T2)}_trigerr'] = float(((xx.trig - xx.pred_trig) / xx.pred_trig).median())
Cm = pd.read_csv(f'{R2}/C_mf.csv')
ref = Cm[(Cm.ndraw == 600) & (Cm.nus == 171)].set_index(['net', 'kappa', 'tau_m']).pred
st = [(d.set_index(['net', 'kappa', 'tau_m']).pred == ref).mean() for (nd, nu), d in Cm.groupby(['ndraw', 'nus'])
      if not (nd == 600 and nu == 171)]
N['C_mf_stab_min'] = float(min(st))
N['C_mf_stab_max'] = float(max(st))
Rr = pd.read_csv(f'{R2}/R_residence.csv')
tr = Rr[Rr.pred.isin(['trap', 'prefrag'])]
N['R_trap_runs'] = int(len(tr))
N['R_trap_noescape'] = int(np.isinf(tr.residence).sum())
N['R_censor'] = int(Rr.censor.iloc[0])
hy = Rr[Rr.pred == 'hyst']
N['R_hyst_runs'] = int(len(hy))
N['R_hyst_med'] = float(hy.residence.median())
if os.path.exists(f'{RH}/H_hetmap.json'):
    H = json.load(open(f'{RH}/H_hetmap.json'))
    rows = []
    for h in H:
        g, up, dn = np.array(h['grid']), np.array(h['F_up']), np.array(h['F_dn'])
        lo_lvl = up[0]
        frag = lambda F: F > 0.25
        if frag(up[0]):
            reg = 'prefrag'
        elif not frag(up).any():
            reg = 'mono'
        else:
            gap = np.max(np.abs(up - dn))
            if frag(dn[0]):
                reg = 'trap'
            elif gap > 0.05:
                reg = 'hyst'
            else:
                reg = 'mono'
        tg = g[np.argmax(frag(up))] if frag(up).any() and not frag(up[0]) else np.nan
        rows.append(dict(net=h['net'], kappa=h['kappa'], tau_m=h['tau_m'], pred=h['pred'], pred_trig=h['pred_trig'],
                         het=reg, het_trig=tg))
    HH = pd.DataFrame(rows)
    HH.to_csv('handoff/H_het_summary.csv', index=False)
    N['H_n'] = int(len(HH))
    N['H_agree'] = float((HH.pred == HH.het).mean())
    xx = HH[np.isfinite(HH.het_trig) & np.isfinite(HH.pred_trig) & (HH.pred_trig > 0)]
    N['H_trig_ratio_med'] = float((xx.het_trig / xx.pred_trig).median()) if len(xx) else np.nan
    pd.crosstab(HH.pred, HH.het).to_csv('handoff/H_confusion.csv')
A = post(pd.read_csv(f'{R2}/A_closures.csv'))
A.to_csv('handoff/A_closures_post.csv', index=False)
for clo, d in A.groupby('closure'):
    key = re.sub('[^A-Za-z]', '', clo.title()) + ('' if not any(ch.isdigit() for ch in clo) else
                                                  re.sub('[^0-9]', '', clo))
    N[f'A_{key}_bistable'] = float(d.regime.isin(['hyst', 'trap']).mean())
    N[f'A_{key}_anyfold'] = float(d.regime.isin(['hyst', 'trap', 'prefrag']).mean())
    N[f'A_{key}_sbmax'] = float(d.sb_max.max())
    if (d.pred != 'na').all():
        N[f'A_{key}_agree'] = float((d.pred == d.regime).mean())
N['A_sb_max_all'] = float(A.sb_max.max())
S = pd.read_csv(f'{R2}/S_subst.csv')
base = S[S.kind == 'none'][['trig0', 'trig1', 'trig2', 'trig3']].mean()
N['S_max_trig_shift'] = float(max((S[S.kind == k][['trig0', 'trig1', 'trig2', 'trig3']].mean() - base).abs().max()
                                   for k in S.kind.unique()))
N['S_sb'] = float(S.sb0.mean())
N['S_so'] = float(S.so0.mean())
E = np.load(f'{R2}/E_ews.npy', allow_pickle=True)
sv_ = np.array([e['sal_var'] for e in E])
sa_ = np.array([e['sal_ac1'] for e in E])
iv_ = np.array([e['innov_var'] for e in E])
N['E_reps'] = int(len(E))
N['E_var_ratio'] = float(np.median(sv_[:, -1] / sv_[:, 0]))
N['E_ac1_start'] = float(sa_[:, 0].mean())
N['E_ac1_end'] = float(sa_[:, -1].mean())
N['E_innov_ratio'] = float(np.median(iv_[:, -1] / iv_[:, 0]))
a0, a1 = N['E_ac1_start'], N['E_ac1_end']
N['E_linear_ratio'] = float((1 - a0 ** 2) / (1 - a1 ** 2))
N['E_combined_ratio'] = N['E_linear_ratio'] * N['E_innov_ratio']
meta = json.load(open(f'{R2}/meta.json'))
N['E_rstop_frac'] = float(meta['E_rstop'] / meta['E_ptrig'])
rc = pd.read_csv('handoff/H_bounds_recheck.csv')
N['bounds_recheck_n'] = int(len(rc))
N['bounds_inside_final'] = float((bd.inside.sum() + rc.inside.sum()) / len(bd))
mm2 = Fr[(Fr['mode'] == 'frozen')]
N['F_nnets'] = int(mm2.net.nunique())
N['F_ncells'] = int(mm2.groupby(['kappa', 'tau_m']).ngroups)

json.dump(N, open('handoff/numbers_rev.json', 'w'), indent=1, default=float)
print(len(N))

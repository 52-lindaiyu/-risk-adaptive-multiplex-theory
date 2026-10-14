"""Revision: empirical numbers (stress scans, decomposition, sweeps, loops, held-out closure tests, event study
and model benchmark).  Merges into handoff/numbers_rev.json and writes summary CSVs into handoff/."""
import os, json, re
import numpy as np
import pandas as pd
from scipy import stats

RE = 'res/rev3/code/out_emp_rev'
N = json.load(open('handoff/numbers_rev.json'))
LN = ['wheat', 'maize', 'rice', 'soybean']
rng = np.random.default_rng(7)
US = np.round(np.linspace(1.0, 4.0, 61), 3)


def sen_slope(y, x):
    ok = np.isfinite(y)
    return stats.theilslopes(y[ok], x[ok])[0] if ok.sum() > 5 else np.nan


def block_boot_slope(y, x, L=5, n=2000):
    """Moving-block bootstrap of residuals around the Sen line (preserves short-range dependence)."""
    ok = np.isfinite(y)
    y, x = y[ok], x[ok]
    s, icpt = stats.theilslopes(y, x)[:2]
    res = y - (icpt + s * x)
    T = len(y)
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(st, st + L) for st in rng.integers(0, T - L + 1, int(np.ceil(T / L)))])[:T]
        yb = icpt + s * x + res[idx]
        out.append(stats.theilslopes(yb, x)[0])
    return np.percentile(out, [2.5, 97.5])


S = pd.read_json(f'{RE}/S_static.json')
S['hm1'] = S.h50 - 1
DEF = (S.spec == 'mirror_raw_lowess30') & (S.eps == 0.2) & (S.b == 0.5) & (S.q == 0.8) & (S.smooth == 1)
rows = []
for mg in ('uniform', 'revealed'):
    d = S[DEF & (S.margin == mg)]
    for l in range(4):
        x = d[d.layer == l].sort_values('year')
        yrs, h = x.year.values.astype(float), x.hm1.values
        sl = sen_slope(h, yrs)
        lo, hi = block_boot_slope(h, yrs)
        kt = stats.kendalltau(yrs, h)
        rows.append(dict(margin=mg, layer=LN[l], start=np.nanmean(h[:3]), end=np.nanmean(h[-3:]), sen=sl * 10,
                         sen_lo=lo * 10, sen_hi=hi * 10, tau=kt.statistic, p_naive=kt.pvalue,
                         n_cens=int(np.isnan(h).sum()), sb1_start=x.sb1_mean.values[:3].mean(),
                         sb1_end=x.sb1_mean.values[-3:].mean(), sb12_start=x.sb12_mean.values[:3].mean(),
                         sb12_end=x.sb12_mean.values[-3:].mean(), cvar_start=x.sb1_cvar95.values[:3].mean(),
                         cvar_end=x.sb1_cvar95.values[-3:].mean()))
TR = pd.DataFrame(rows)
TR.to_csv('handoff/EMP_h50_trends.csv', index=False)
for r in TR.itertuples():
    k = f'EMP_{r.margin[:3]}_{r.layer}'
    for c in ('start', 'end', 'sen', 'sen_lo', 'sen_hi', 'tau', 'n_cens', 'sb1_start', 'sb1_end', 'sb12_start',
              'sb12_end', 'cvar_start', 'cvar_end'):
        N[f'{k}_{c}'] = float(getattr(r, c))
# robustness grid: sign and block-bootstrap significance of the Sen slope across eps x (b,q) x smoothing
G = S[S.spec == 'mirror_raw_lowess30']
grow = []
for (mg, e, b, q, sm, l), d in G.groupby(['margin', 'eps', 'b', 'q', 'smooth', 'layer']):
    d = d.sort_values('year')
    sl = sen_slope(d.hm1.values, d.year.values.astype(float))
    lo, hi = block_boot_slope(d.hm1.values, d.year.values.astype(float), n=400)
    grow.append(dict(margin=mg, eps=e, b=b, q=q, smooth=sm, layer=LN[l], sen=sl, neg_sig=hi < 0, pos_sig=lo > 0,
                     n_cens=int(d.hm1.isna().sum())))
GR = pd.DataFrame(grow)
GR.to_csv('handoff/EMP_h50_grid.csv', index=False)
for (mg, l), d in GR.groupby(['margin', 'layer']):
    N[f'EMPG_{mg[:3]}_{l}_negsig'] = float(d.neg_sig.mean())
    N[f'EMPG_{mg[:3]}_{l}_possig'] = float(d.pos_sig.mean())
    N[f'EMPG_{mg[:3]}_{l}_neg'] = float((d.sen < 0).mean())
N['EMPG_ncombos'] = int(GR.groupby(['margin', 'layer']).size().iloc[0])
N['EMPG_cens_total'] = int(GR.n_cens.sum())
# panel-construction variants (default parameters)
V = S[(S.eps == 0.2) & (S.b == 0.5) & (S.q == 0.8) & (S.smooth == 1) & (S.margin == 'uniform')]
vrows = []
for (sp, l), d in V.groupby(['spec', 'layer']):
    d = d.sort_values('year')
    vrows.append(dict(spec=sp, layer=LN[l], start=d.hm1.values[:3].mean(), end=d.hm1.values[-3:].mean(),
                      sen=sen_slope(d.hm1.values, d.year.values.astype(float)) * 10))
VV = pd.DataFrame(vrows)
VV.to_csv('handoff/EMP_h50_specs.csv', index=False)
for l in LN:
    d = VV[VV.layer == l]
    N[f'EMPV_{l}_neg_specs'] = int((d.sen < 0).sum())
    N[f'EMPV_{l}_nspecs'] = int(len(d))
# post-1992 window (after the USSR break-up)
d = S[DEF & (S.margin == 'uniform') & (S.year >= 1993)]
for l in range(4):
    x = d[d.layer == l].sort_values('year')
    N[f'EMP93_{LN[l]}_sen'] = float(sen_slope(x.hm1.values, x.year.values.astype(float)) * 10)

# scalar-closure collapse points per layer-year (frozen shares), kappa x tau_m grid
D0 = S[DEF & (S.margin == 'uniform')].sort_values(['layer', 'year'])
frows = []
for r in D0.itertuples():
    F = np.array(r.F_curve)
    dF = np.gradient(F, US)
    for kap in (0.05, 0.1, 0.2):
        for tm in (1.0, 2.0, 3.0, 5.0):
            lam = np.exp(-1 / tm)
            gain = kap * dF / (1 - lam)
            rext = (US - 1) / kap - F / (1 - lam)
            if (gain > 1).any():
                k0 = int(np.argmax(gain > 1))
                trig = rext[:k0].max() if k0 > 0 else -np.inf
            else:
                trig = np.nan          # no fold on u in [1, 4]
            frows.append(dict(layer=LN[r.layer], year=r.year, kappa=kap, tau_m=tm, trig=trig))
FD = pd.DataFrame(frows)
FD.to_csv('handoff/EMP_scalar_folds.csv', index=False)
for (kap, tm), d in FD.groupby(['kappa', 'tau_m']):
    N[f'EMPF_k{int(kap*100)}_t{int(tm)}_nofold'] = float(d.trig.isna().mean())
    N[f'EMPF_k{int(kap*100)}_t{int(tm)}_prefrag'] = float((d.trig < 0).mean())
d = FD[(FD.kappa == 0.1) & (FD.tau_m == 2.0)]
for l in LN:
    x = d[d.layer == l].sort_values('year')
    N[f'EMPF_k10_t2_{l}_start'] = float(x.trig.replace(-np.inf, np.nan).values[:3].mean())
    N[f'EMPF_k10_t2_{l}_end'] = float(x.trig.replace(-np.inf, np.nan).values[-3:].mean())

# factorial (Shapley) decomposition of the change since 1986-88
X = pd.read_csv(f'{RE}/X_factorial.csv')
xrows = []
for r in X.itertuples():
    for met in ('h50', 'sb12'):
        v = {k: getattr(r, f'{k}_{met}') for k in ('s0p0m0', 's0p0m1', 's0p1m0', 's0p1m1', 's1p0m0', 's1p0m1',
                                                   's1p1m0', 's1p1m1')}
        f = lambda s, p, m: v[f's{s}p{p}m{m}']
        import itertools
        contrib = {}
        for fac in 'spm':
            tot = 0.0
            others = [o for o in 'spm' if o != fac]
            for combo in itertools.product((0, 1), repeat=2):
                st = dict(zip(others, combo))
                k = sum(combo)
                wgt = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}[k]
                a = dict(st, **{fac: 1})
                b0 = dict(st, **{fac: 0})
                tot += wgt * (f(a['s'], a['p'], a['m']) - f(b0['s'], b0['p'], b0['m']))
            contrib[fac] = tot
        xrows.append(dict(layer=LN[r.layer], year=r.year, margin=r.margin, metric=met,
                          total=f(1, 1, 1) - f(0, 0, 0), structure=contrib['s'], exposure=contrib['p'],
                          margins=contrib['m']))
XD = pd.DataFrame(xrows)
XD.to_csv('handoff/EMP_factorial.csv', index=False)
xs = []
for (l, mg, met), d in XD.groupby(['layer', 'margin', 'metric']):
    d = d.sort_values('year')
    row = dict(layer=l, margin=mg, metric=met)
    for c in ('total', 'structure', 'exposure', 'margins'):
        row[c] = sen_slope(d[c].values, d.year.values.astype(float)) * 10
    xs.append(row)
XS = pd.DataFrame(xs)
XS.to_csv('handoff/EMP_factorial_slopes.csv', index=False)
for r in XS[XS.metric == 'h50'].itertuples():
    k = f'EMPXS_{r.margin[:3]}_{r.layer}'
    for c in ('total', 'structure', 'exposure', 'margins'):
        N[f'{k}_{c}'] = float(getattr(r, c))
for r in XD[(XD.year == 2023) & (XD.metric == 'h50')].itertuples():
    k = f'EMPX_{r.margin[:3]}_{r.layer}'
    N[k + '_total'] = float(r.total)
    N[k + '_structure'] = float(r.structure)
    N[k + '_exposure'] = float(r.exposure)
    N[k + '_margins'] = float(r.margins)

# risk-aversion sweeps on empirical topologies
Dd = pd.read_json(f'{RE}/D_beta.json')
drows = []
for (y, ly, tm, mu), d in Dd.groupby(['year', 'layers', 'tau_m', 'mu']):
    m = d.groupby('beta').sb_mean.mean()
    bmin = m.idxmin()
    drows.append(dict(year=y, layers=ly, tau_m=tm, mu=mu, beta_star=bmin, sb0=m.loc[0], sbmin=m.min(),
                      sbmax=m.iloc[-1], ratio=m.iloc[-1] / m.min(), gain=(m.loc[0] - m.min()) / m.loc[0],
                      Mt_min=m.min() * d.total_demand.iloc[0] / 1e6, Mt_0=m.loc[0] * d.total_demand.iloc[0] / 1e6,
                      Mt_max=m.iloc[-1] * d.total_demand.iloc[0] / 1e6))
DS = pd.DataFrame(drows)
DS.to_csv('handoff/EMP_beta_sweeps.csv', index=False)
pubU = DS[(DS.mu == 1.0)]
N['EMPD_pub_U_share'] = float(((pubU.ratio > 1.5) & (pubU.gain > 0.05)).mean())
N['EMPD_priv_U_share'] = float(((DS[DS.mu == 0].ratio > 1.5) & (DS[DS.mu == 0].gain > 0.05)).mean())
N['EMPD_n'] = int(len(pubU))
for r in DS[(DS.year == 2006) & (DS.tau_m == 2.0)].itertuples():
    k = f'EMPD_{"pub" if r.mu == 1 else "priv"}_{r.layers.replace("-", "")}'
    N[k + '_ratio'] = float(r.ratio)
    N[k + '_betastar'] = float(r.beta_star)
    N[k + '_Mtmin'] = float(r.Mt_min)
    N[k + '_Mt0'] = float(r.Mt_0)
    N[k + '_Mtmax'] = float(r.Mt_max)

# loops on the empirical multiplex
Lp = pd.read_json(f'{RE}/L_loops.json')
lrows = []
for r in Lp.itertuples():
    for l in range(4):
        lrows.append(dict(year=r.year, kappa=r.kappa, chi=r.chi, mode=r.mode, seed=r.seed, layer=LN[l],
                          regime=r.regimes[l], trig=r.trig[l], rel=r.rel[l]))
LL = pd.DataFrame(lrows)
LL.to_csv('handoff/EMP_loops.csv', index=False)
pv = LL.pivot_table(index=['year', 'kappa', 'mode', 'seed', 'layer'], columns='chi', values='trig')
N['EMPL_chi_trig_shift_med'] = float((pv.iloc[:, 1] - pv.iloc[:, 0]).abs().median())
N['EMPL_chi_trig_shift_mean'] = float((pv.iloc[:, 1] - pv.iloc[:, 0]).mean())
N['EMPL_regime_change_share'] = float((LL.pivot_table(index=['year', 'kappa', 'mode', 'seed', 'layer'], columns='chi',
                                       values='regime', aggfunc='first').nunique(axis=1) > 1).mean())
N['EMPL_chi_trig_shift_max'] = float(LL.pivot_table(index=['year', 'kappa', 'mode', 'seed', 'layer'], columns='chi',
                                                     values='trig').diff(axis=1).abs().max().max())
# held-out closure validation on empirical layers
Hh = pd.read_csv(f'{RE}/H_heldout.csv')
ww = (Hh.trig - Hh.rel.fillna(0)) / Hh.trig
Hh.loc[(Hh.regime == 'hyst') & (ww < N['F_width_threshold']), 'regime'] = 'mono'
N['EMPH_n'] = int(len(Hh))
N['EMPH_agree'] = float((Hh.pred == Hh.regime).mean())
pd.crosstab(Hh.pred, Hh.regime).to_csv('handoff/EMP_heldout_confusion.csv')
xx = Hh[Hh.pred.isin(['hyst', 'trap']) & Hh.regime.isin(['hyst', 'trap'])]
N['EMPH_trig_err_med'] = float(((xx.trig - xx.pred_trig) / xx.pred_trig).median()) if len(xx) else np.nan

# event study: empirical, placebo-calibrated, and model replay benchmark
EV = pd.read_csv('res/emp_rev/EV_main.csv')
PL = pd.read_csv('res/emp_rev/EV_placebo.csv')
for oc in ('k', 'a'):
    e = EV[(EV.group == 'all') & (EV.outcome == oc)].set_index('k')
    p = PL[PL.outcome == oc]
    N[f'EV_{oc}_nevents'] = int(e.n_events.iloc[0])
    N[f'EV_{oc}_npairs'] = int(e.n_pairs.iloc[0])
    N[f'EV_{oc}_pretrend_p'] = float(e.pretrend_p.iloc[0])
    for k in (-3, -2, 0, 1, 2, 3):
        N[f'EV_{oc}_k{k}'.replace('-', 'm')] = float(e.loc[k, 'est'])
        N[f'EV_{oc}_k{k}_lo'.replace('-', 'm')] = float(e.loc[k, 'lo'])
        N[f'EV_{oc}_k{k}_hi'.replace('-', 'm')] = float(e.loc[k, 'hi'])
    for k in (0, 1, 2):
        pk = p[f'est_k{k}'].values
        N[f'EV_{oc}_k{k}_plac_mean'] = float(pk.mean())
        N[f'EV_{oc}_k{k}_plac_p'] = float(2 * min((pk >= e.loc[k, 'est']).mean(), (pk <= e.loc[k, 'est']).mean()))
        N[f'EV_{oc}_k{k}_naive_size'] = float((p[f'p_k{k}'] < 0.05).mean())
    N[f'EV_{oc}_plac_pretrend_size'] = float((p.pretrend_p < 0.05).mean())
VR = pd.read_csv(f'{RE}/V_replay_eventstudy.csv')
vr = VR[VR.group == 'all'].groupby(['outcome', 'beta', 'tau_m', 'k']).agg(est=('est', 'mean'), sd=('est', 'std'),
                                                                         sig_neg=('hi', lambda s: (s < 0).mean()))
vr = vr.reset_index()
vr.to_csv('handoff/EV_model_replay.csv', index=False)
for r in vr[(vr.outcome == 'k') & vr.k.isin([0, 1, 2])].itertuples():
    N[f'EVM_b{int(r.beta)}_t{str(r.tau_m).replace(".", "p")}_k{r.k}'] = float(r.est)
    N[f'EVM_b{int(r.beta)}_t{str(r.tau_m).replace(".", "p")}_k{r.k}_signeg'] = float(r.sig_neg)
emp = EV[(EV.group == 'all') & (EV.outcome == 'k')].set_index('k')
mods = vr[(vr.outcome == 'k')]
# share of model configurations whose mean replay coefficient lies outside the empirical 95% CI at k=0..2
out = []
for (b, tm), d in mods.groupby(['beta', 'tau_m']):
    d = d.set_index('k')
    out.append(dict(beta=b, tau_m=tm, outside=all((d.loc[k, 'est'] < emp.loc[k, 'lo']) or
                                                   (d.loc[k, 'est'] > emp.loc[k, 'hi']) for k in (0, 1, 2))))
OU = pd.DataFrame(out)
for k in (0, 1, 2):
    dk = mods[mods.k == k]
    N[f'EVM_outside_k{k}'] = int(((dk.est < emp.loc[k, 'lo']) | (dk.est > emp.loc[k, 'hi'])).sum())
    N[f'EVM_range_k{k}_lo'] = float(dk.est.min())
    N[f'EVM_range_k{k}_hi'] = float(dk.est.max())
mem = mods[(mods.beta > 0) & (mods.tau_m >= 1)]
for k in (1, 2):
    N[f'EVM_mem_range_k{k}_lo'] = float(mem[mem.k == k].est.min())
    N[f'EVM_mem_range_k{k}_hi'] = float(mem[mem.k == k].est.max())
N['EVM_nconfigs'] = int(len(OU))
N['EVM_outside_all'] = int(OU.outside.sum())
json.dump(N, open('handoff/numbers_rev.json', 'w'), indent=1, default=float)
print(len(N))

"""Stacked event-study estimator shared by the empirical panel and the model-simulated panels (revision).

Design.  Event: a severe yield shock of exporter j in year t0 (detrended log-yield z <= -1.28) with NO other
severe shock of j in the symmetric clean window [t0-3, t0+3].  Controls: importer-exporter pairs whose exporter
has no severe shock anywhere in [t0-3, t0+3].  Pairs enter only if importer i bought >= 5% of its imports from j
in the base year t0-1.  Balanced cohort: only events whose full window k = -3..+3 lies inside the panel.
Outcome 1: change in j's share of i's imports relative to t0-1.  Outcome 2 (supply-adjusted): outcome 1 minus
the change in j's share of world exports of that crop, i.e. reallocation beyond j's market-wide supply loss.
Treated-minus-control differences are formed within layer x year x base-share quintile; inference by
cluster bootstrap over shock events (exporter x layer x year).
"""
import numpy as np
import pandas as pd

ZSH = -1.28
KS = list(range(-3, 6))
BAL = (-3, 3)


def build_events(layer, S, z, mshare, years, base_min=0.05, clean=3):
    """S: (Ni, Mj, T) shares of j in i's imports (nan = importer absent); z: (Mj, T) shock z-scores (nan = no
    yield data); mshare: (Mj, T) share of j in world exports.  Returns a DataFrame of candidate events."""
    Ni, Mj, T = S.shape
    sev = np.nan_to_num(z, nan=0.0) <= ZSH
    rows = []
    for t0 in range(-BAL[0], T - BAL[1]):
        for j in range(Mj):
            if np.isnan(z[j, t0]):
                continue
            lo, hi = t0 - clean, min(T, t0 + clean + 1)
            others = sev[j, lo:hi].copy()
            others[t0 - lo] = False
            treated = sev[j, t0] and not others.any()
            control = not sev[j, lo:hi].any()
            if not (treated or control):
                continue
            base = S[:, j, t0 - 1]
            ii = np.where(base >= base_min)[0]
            for i in ii:
                traj, traj2 = [], []
                for k in KS:
                    tk = t0 + k
                    if 0 <= tk < T and not np.isnan(S[i, j, tk]):
                        d = S[i, j, tk] - base[i]
                        traj.append(d)
                        traj2.append(d - (mshare[j, tk] - mshare[j, t0 - 1]))
                    else:
                        traj.append(np.nan)
                        traj2.append(np.nan)
                rows.append([layer, i, j, int(years[t0]), base[i], int(treated)] + traj + traj2)
    cols = (['layer', 'imp', 'exp', 'year', 'base', 'treated'] + ['k%d' % k for k in KS] +
            ['a%d' % k for k in KS])
    return pd.DataFrame(rows, columns=cols)


def estimate(E, outcome='k', nboot=2000, seed=1, by_layer=True):
    rng = np.random.default_rng(seed)
    ks = [outcome + str(k) for k in KS]
    E = E.copy()
    bal = [outcome + str(k) for k in range(BAL[0], BAL[1] + 1)]
    E = E.dropna(subset=bal)
    if E.treated.sum() == 0:
        return None, None
    E['bbin'] = pd.qcut(E.base.rank(method='first'), 5, labels=False)
    ctrl = E[E.treated == 0].groupby(['layer', 'year', 'bbin'])[ks].mean()
    Tr = E[E.treated == 1].join(ctrl, on=['layer', 'year', 'bbin'], rsuffix='_c')
    for k in ks:
        Tr['d_' + k] = Tr[k] - Tr[k + '_c']
    Tr = Tr.dropna(subset=['d_' + outcome + '0'])
    Tr['cl'] = Tr.layer.astype(str) + '_' + Tr.exp.astype(str) + '_' + Tr.year.astype(str)
    dcols = ['d_' + k for k in ks]

    def summarise(T, tag):
        cl = T.cl.unique()
        g = {c: df[dcols].values for c, df in T.groupby('cl')}
        est = np.nanmean(T[dcols].values, 0)
        boots = np.array([np.nanmean(np.vstack([g[c] for c in rng.choice(cl, len(cl), replace=True)]), 0)
                          for _ in range(nboot)])
        # pre-trend joint test (k=-3,-2): bootstrap Wald statistic, centred
        pre = [KS.index(-3), KS.index(-2)]
        V = np.cov(boots[:, pre].T)
        e = est[pre]
        try:
            W = float(e @ np.linalg.solve(V, e))
        except np.linalg.LinAlgError:
            W = np.nan
        from scipy.stats import chi2
        out = pd.DataFrame({'k': KS, 'est': est, 'se': boots.std(0), 'lo': np.nanpercentile(boots, 2.5, 0),
                            'hi': np.nanpercentile(boots, 97.5, 0),
                            'n_obs': np.sum(np.isfinite(T[dcols].values), 0)})
        out['p_two'] = [2 * min((boots[:, m] <= 0).mean(), (boots[:, m] >= 0).mean()) for m in range(len(KS))]
        out['group'] = tag
        out['n_events'] = len(cl)
        out['n_pairs'] = len(T)
        out['pretrend_wald'] = W
        out['pretrend_p'] = 1 - chi2.cdf(W, 2) if np.isfinite(W) else np.nan
        return out

    res = [summarise(Tr, 'all')]
    if by_layer:
        for lay, T in Tr.groupby('layer'):
            if T.cl.nunique() >= 5:
                res.append(summarise(T, str(lay)))
    return pd.concat(res, ignore_index=True), Tr


def shares_from_flows(FLl, min_imp=1e5):
    """FLl: (T, K, K) importer x exporter flows.  Returns S (Ni, K, T) for importers with mean imports >= min_imp,
    importer indices, and world export shares (K, T)."""
    tot = FLl.sum(2)                          # (T, K)
    imps = np.where(tot.mean(0) >= min_imp)[0]
    S = np.full((len(imps), FLl.shape[1], FLl.shape[0]), np.nan)
    for a, i in enumerate(imps):
        for t in range(FLl.shape[0]):
            if tot[t, i] > 0:
                S[a, :, t] = FLl[t, i] / tot[t, i]
    ex = FLl.sum(1)                           # (T, K) exports of j
    msh = (ex / np.maximum(ex.sum(1, keepdims=True), 1e-12)).T
    return S, imps, msh

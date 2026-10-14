"""Raw FAOSTAT bulk files -> analysis panels.  Fully executable pipeline (revision).

Inputs (public FAOSTAT bulk downloads, https://bulks-faostat.fao.org/production/):
  data_raw/tm.zip   Trade_DetailedTradeMatrix_E_All_Data_(Normalized).csv
  data_raw/qcl.zip  Production_Crops_Livestock_E_All_Data_(Normalized).csv
Outputs:
  data_raw/trade_cereals.parquet, data_raw/prod_cereals.parquet           (extracts)
  data_raw/flows_<recon>.parquet       reconciled bilateral flows, recon in {mirror, imponly, exponly}
  data_raw/yield_anoms_<det>.parquet   detrended yield anomalies, det in {lowess30, lowess20, lowess50, quad}
  handoff/emp_<recon>_<ent>_<det>.npz  dense panels (layers x years x countries [x countries])
  handoff/panel_coverage.csv           mirror-fill share, countries and volume per layer-year
Usage: python build_panel.py [extract]
"""
import os, sys, zipfile, time
import numpy as np
import pandas as pd

LAYERS = ['wheat', 'maize', 'rice', 'soybean']
# item code -> (layer, grain-equivalent factor). Wheat flour 1.37 t grain / t flour; rice paddy 0.67 and
# husked 0.84 milled equivalent; milled, broken and other rice counted 1:1.
ITEMS = {15: ('wheat', 1.0), 16: ('wheat', 1.37), 56: ('maize', 1.0), 27: ('rice', 0.67), 28: ('rice', 0.84),
         29: ('rice', 1.0), 31: ('rice', 1.0), 32: ('rice', 1.0), 236: ('soybean', 1.0)}
PROD_ITEMS = {15: 'wheat', 56: 'maize', 27: 'rice', 236: 'soybean'}
YEARS = np.arange(1986, 2025)
# entity harmonisation: successor states -> predecessor aggregate (FAO area codes)
SUCC = {228: [1, 52, 57, 63, 73, 108, 113, 119, 126, 146, 185, 208, 213, 230, 235],   # USSR
        248: [80, 98, 154, 198, 186, 272, 273],                                       # Yugoslav SFR
        51: [167, 199],                                                               # Czechoslovakia
        15: [255, 256],                                                               # Belgium-Luxembourg
        62: [238, 178],                                                               # Ethiopia PDR
        206: [276, 277]}                                                              # Sudan (former)
HMAP = {c: p for p, cs in SUCC.items() for c in cs}


def extract():
    t0 = time.time()
    z = zipfile.ZipFile('data_raw/tm.zip')
    name = [n for n in z.namelist() if n.endswith('.csv') and 'All_Data' in n][0]
    cols = ['Reporter Country Code', 'Reporter Countries', 'Partner Country Code', 'Partner Countries',
            'Item Code', 'Element Code', 'Year', 'Value']
    parts = []
    for ch in pd.read_csv(z.open(name), usecols=cols, chunksize=4_000_000, encoding='latin1', low_memory=False):
        ch = ch[ch['Item Code'].isin(ITEMS.keys()) & ch['Element Code'].isin([5610, 5910]) & (ch['Year'] >= 1986)]
        parts.append(ch)
    tm = pd.concat(parts)
    tm.columns = ['rep', 'rep_name', 'par', 'par_name', 'item', 'elem', 'year', 'value']
    tm['layer'] = tm['item'].map(lambda c: ITEMS[c][0])
    tm['qty'] = tm['value'] * tm['item'].map(lambda c: ITEMS[c][1])
    tm.to_parquet('data_raw/trade_cereals.parquet')
    print('trade extract', tm.shape, round(time.time() - t0), flush=True)
    zq = zipfile.ZipFile('data_raw/qcl.zip')
    name = [n for n in zq.namelist() if n.endswith('.csv') and 'All_Data' in n][0]
    q = pd.read_csv(zq.open(name), encoding='latin1', usecols=['Area Code', 'Area', 'Item Code', 'Element Code',
                                                               'Year', 'Value'], low_memory=False)
    q = q[q['Item Code'].isin(PROD_ITEMS.keys()) & q['Element Code'].isin([5312, 5412, 5510]) & (q['Year'] >= 1961)]
    q.columns = ['area', 'area_name', 'item', 'elem', 'year', 'value']
    q.to_parquet('data_raw/prod_cereals.parquet')
    print('prod extract', q.shape, round(time.time() - t0), flush=True)


def reconcile(tm):
    """Directed importer<-exporter flows.  mirror: importer-reported quantity, exporter-reported mirror where
    the importer did not report; imponly / exponly: a single reporting side."""
    names = pd.concat([tm[['rep', 'rep_name']].rename(columns={'rep': 'code', 'rep_name': 'name'}),
                       tm[['par', 'par_name']].rename(columns={'par': 'code', 'par_name': 'name'})]
                      ).drop_duplicates('code').set_index('code').name
    tm = tm[(tm.rep < 5000) & (tm.par < 5000) & (tm.rep != 351) & (tm.par != 351) & (tm.rep != tm.par)]
    # reconcile at the FAO item level (reporting differs across items), then aggregate items to layers
    imp = tm[tm.elem == 5610].groupby(['layer', 'item', 'year', 'rep', 'par']).qty.sum().rename('q_imp')
    exp = tm[tm.elem == 5910].groupby(['layer', 'item', 'year', 'par', 'rep']).qty.sum().rename('q_exp')
    exp.index.names = ['layer', 'item', 'year', 'rep', 'par']
    fl = pd.concat([imp, exp], axis=1).reset_index().rename(columns={'rep': 'imp', 'par': 'exp'})
    out = {}
    m = fl.copy()
    m['q'] = m.q_imp.where(m.q_imp.notna(), m.q_exp)
    m['src'] = np.where(m.q_imp.notna(), 'importer', 'mirror')
    out['mirror'] = m
    out['imponly'] = fl.assign(q=fl.q_imp, src='importer')
    out['exponly'] = fl.assign(q=fl.q_exp, src='exporter')
    for k in out:
        d = out[k]
        d = d[d.q > 0]
        d = d.assign(q_mirror=np.where(d.src == 'mirror', d.q, 0.0))
        d = d.groupby(['layer', 'year', 'imp', 'exp'])[['q', 'q_mirror']].sum().reset_index()
        d['src'] = np.where(d.q_mirror >= 0.5 * d.q, 'mirror', 'importer')
        out[k] = d
    return out, names


def harmonise(F):
    F = F.copy()
    F['imp'] = F.imp.map(lambda c: HMAP.get(c, c))
    F['exp'] = F.exp.map(lambda c: HMAP.get(c, c))
    F = F[F.imp != F.exp]
    return F.groupby(['layer', 'year', 'imp', 'exp']).q.sum().reset_index()


def anomalies(q, method='lowess30'):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    q = q[(q.area < 5000) & (q.area != 351)].copy()
    q['layer'] = q.item.map(PROD_ITEMS)
    P = q[q.elem == 5510].pivot_table(index=['layer', 'area'], columns='year', values='value')
    Y = q[q.elem == 5412].pivot_table(index=['layer', 'area'], columns='year', values='value')
    P.loc['rice'] = P.loc['rice'].values * 0.67  # paddy -> milled equivalent
    rows = []
    for key in Y.index:
        y = Y.loc[key].dropna()
        y = y[y > 0]
        if len(y) < 20:
            continue
        ly = np.log(y.values)
        yrs = y.index.values.astype(float)
        if method.startswith('lowess'):
            tr = lowess(ly, yrs, frac=int(method[6:]) / 100, return_sorted=False)
        else:
            tr = np.polyval(np.polyfit(yrs - yrs.mean(), ly, 2), yrs - yrs.mean())
        res = ly - tr
        sd = res.std(ddof=1)
        for yr, rr in zip(yrs, res):
            rows.append((key[0], key[1], int(yr), np.exp(rr) - 1, rr / sd))
    A = pd.DataFrame(rows, columns=['layer', 'area', 'year', 'anom', 'z'])
    Pl = P.stack().rename('prod').reset_index()
    A = A.merge(Pl, on=['layer', 'area', 'year'], how='left')
    return A[A.year >= 1980].reset_index(drop=True)


def pack(F, A, path, harmonised=False):
    F = F[F.year >= 1986]
    if harmonised:
        A = A.copy()
        A['area'] = A.area.map(lambda c: HMAP.get(c, c))
        # aggregate production; production-weighted anomaly
        A['pa'] = A['prod'].fillna(0) * A.anom
        A['pz'] = A['prod'].fillna(0) * A.z
        g = A.groupby(['layer', 'area', 'year'])
        A = g[['prod', 'pa', 'pz']].sum().reset_index()
        A['anom'] = A.pa / A['prod'].replace(0, np.nan)
        A['z'] = A.pz / A['prod'].replace(0, np.nan)
    ctry = np.array(sorted(set(F.imp) | set(F.exp)))
    K = len(ctry)
    cidx = {c: k for k, c in enumerate(ctry)}
    FL = np.zeros((4, len(YEARS), K, K), np.float32)
    for l, layer in enumerate(LAYERS):
        d = F[F.layer == layer]
        np.add.at(FL[l], ((d.year - 1986).values, d.imp.map(cidx).values, d.exp.map(cidx).values),
                  d.q.values.astype(np.float32))
    PR = np.zeros((4, len(YEARS), K), np.float32)
    AN = np.zeros((4, len(YEARS), K), np.float32)
    ZZ = np.full((4, len(YEARS), K), np.nan, np.float32)
    Ay = A[(A.year >= 1986) & (A.year <= YEARS[-1]) & (A.area.isin(cidx))]
    for l, layer in enumerate(LAYERS):
        d = Ay[Ay.layer == layer]
        kk = d.area.map(cidx).values
        tt = (d.year - 1986).values
        PR[l, tt, kk] = np.nan_to_num(d['prod'].values)
        AN[l, tt, kk] = np.clip(np.nan_to_num(d.anom.values), -0.8, 0.8)
        ZZ[l, tt, kk] = d.z.values
    np.savez_compressed(path, ctry=ctry, years=YEARS, flows=FL, prod=PR, anom=AN, z=ZZ, layers=np.array(LAYERS))
    return K


if __name__ == '__main__':
    if 'extract' in sys.argv or not os.path.exists('data_raw/trade_cereals.parquet'):
        extract()
    os.makedirs('handoff', exist_ok=True)
    tm = pd.read_parquet('data_raw/trade_cereals.parquet')
    q = pd.read_parquet('data_raw/prod_cereals.parquet')
    flows, names = reconcile(tm)
    names.to_csv('data_raw/country_names.csv')
    for k, F in flows.items():
        F.to_parquet(f'data_raw/flows_{k}.parquet')
    m = flows['mirror']
    cov = m.groupby(['layer', 'year']).apply(lambda d: pd.Series(dict(
        mirror_share_volume=d.q_mirror.sum() / d.q.sum(), n_links=len(d),
        n_countries=len(set(d.imp) | set(d.exp)), volume_Mt=d.q.sum() / 1e6))).reset_index()
    for k in ('imponly', 'exponly'):
        v = flows[k].groupby(['layer', 'year']).q.sum().rename(f'volume_{k}_Mt') / 1e6
        cov = cov.merge(v.reset_index(), on=['layer', 'year'], how='left')
    cov.to_csv('handoff/panel_coverage.csv', index=False)
    dets = {}
    for det in ('lowess30', 'lowess20', 'lowess50', 'quad'):
        dets[det] = anomalies(q, det)
        dets[det].to_parquet(f'data_raw/yield_anoms_{det}.parquet')
    specs = [('mirror', 'raw', 'lowess30'), ('imponly', 'raw', 'lowess30'), ('exponly', 'raw', 'lowess30'),
             ('mirror', 'harm', 'lowess30'), ('mirror', 'raw', 'lowess20'), ('mirror', 'raw', 'lowess50'),
             ('mirror', 'raw', 'quad')]
    for rc, ent, det in specs:
        F = flows[rc][['layer', 'year', 'imp', 'exp', 'q']]
        if ent == 'harm':
            F = harmonise(F)
        K = pack(F, dets[det], f'handoff/emp_{rc}_{ent}_{det}.npz', harmonised=(ent == 'harm'))
        print(rc, ent, det, 'K =', K, flush=True)

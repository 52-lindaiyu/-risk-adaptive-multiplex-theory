"""
Risk-adaptive multiplex trade model (RAMT): simulation core.

State per layer l (commodity): import shares w[l,i,j] (importer i, exporter j) and
public perceived-risk salience r[l,j] of each exporter.  One period = one marketing year.

Per period and layer:
  1. exportable supply  X_j = max(0, Xbar_j + Psc_j xi_j) (1 - z 1[large shock_j]),
     xi ~ N(0, Sigma) (spatially correlated) or a bootstrapped historical yield-anomaly vector;
     restriction threshold Th_j = X_j + slack_j (slack = domestic use that can be drawn down)
  2. demand  D_i = Dbar_i (1 + kappa (rho_i + rho_ext(t))) + spill_i,  rho_i = sum_j w_ij r_j
  3. orders  o_ij = D_i w_ij
  4. restriction cascade: exporter j restricts if O_j = sum_i o_ij > Th_j; it withholds a
     fraction b of the volume ordered from it; importer i re-routes a fraction q of withheld
     volume to its open suppliers in proportion to w_ik.  Iterate to a fixed point.
  5. deliveries capped by Th_j (open) or X_j (restricted); failure rate f_j = 1 - delivered_j / requested_j
  6. salience  r_j <- lam r_j + f_j,   lam = exp(-1/tau_m)
  7. target shares w*_ij ∝ A_ij exp(-gam c_ij - beta r_j);  w <- w + eta (w* - w),
     eta = 1 - exp(-1/tau_r)
  8. multiplex: a share chi of importer i's unmet demand in layer l is added to its demand in
     the other layers in the next period (substitution in consumption / feed rations).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def cascade_layer(o, w, X, Th, b, q, open_out, deliv_out):
    """Restriction cascade for one layer and period (o modified in place).
    Returns n_restricted, requested_j, delivered_j, rerouting-unmet_i."""
    N, M = o.shape
    restricted = np.zeros(M, dtype=np.bool_)
    req = np.zeros(M)
    O = np.zeros(M)
    for j in range(M):
        s = 0.0
        for i in range(N):
            s += o[i, j]
        req[j] = s
        O[j] = s
    unmet = np.zeros(N)
    nres = 0
    newly = np.zeros(M, dtype=np.bool_)
    for it in range(M + 1):
        anynew = False
        for j in range(M):
            newly[j] = False
            if (not restricted[j]) and O[j] > 0.0 and O[j] > Th[j]:
                newly[j] = True
                restricted[j] = True
                anynew = True
                nres += 1
        if not anynew:
            break
        for i in range(N):
            withheld = 0.0
            for j in range(M):
                if newly[j]:
                    dv = b * o[i, j]
                    withheld += dv
                    o[i, j] -= dv
                    O[j] -= dv
            if withheld <= 0.0:
                continue
            wo = 0.0
            for k in range(M):
                if not restricted[k]:
                    wo += w[i, k]
            if wo > 1e-12:
                for k in range(M):
                    if not restricted[k]:
                        add = q * withheld * w[i, k] / wo
                        o[i, k] += add
                        O[k] += add
                        req[k] += add
                unmet[i] += (1.0 - q) * withheld
            else:
                unmet[i] += withheld
    deliv_j = np.zeros(M)
    for j in range(M):
        cap = 1.0
        lim = Th[j] if not restricted[j] else X[j]
        if O[j] > lim and O[j] > 0.0:
            cap = lim / O[j]
        for i in range(N):
            d = o[i, j] * cap
            deliv_out[i, j] = d
            deliv_j[j] += d
        open_out[j] = not restricted[j]
    return nres, req, deliv_j, unmet


@njit(cache=True)
def simulate(A, Cst, Dbar, Xbar, slack, Psc, Lch, anom, w0, r0,
             beta, gam, tau_m, tau_r, b, q, kappa, chi,
             p_shock, z_shock, shock_mask, rho_ext, T, burn, seed, rec_w):
    """Run the multiplex model for T periods; record T-burn periods.
    Arrays: A, Cst, w0 (Lyr,N,M); Dbar (Lyr,N); Xbar, slack, Psc, r0, shock_mask (Lyr,M);
    exportable supply X_j = max(0, Xbar_j + Psc_j xi_j), restriction threshold Th_j = X_j + slack_j;
    xi either Gaussian with Cholesky factor Lch (Lyr,M,M) [anom.shape[1]==1] or a bootstrap draw of
    one historical year from anom (Lyr,Y,M) of relative yield anomalies; rho_ext (T,).
    beta: (N,) risk-aversion of each importer.  Extra outputs: s_corr = demand-weighted mean
    portfolio supply-shock correlation sum_jk w_ij w_ik Sigma_jk; s_shi = per-importer shortfall (layer 0).
    rec_w: if >=0, record the full share trajectory of importer rec_w in layer 0.
    Returns dict-like tuple of time series (TT,Lyr): shortfall, phi (restricted export-capacity share),
    n_restrictions, aggregate HHI of loads, demand-weighted importer HHI, load-weighted salience,
    plus importer-0 shares (TT,M), salience of layer 0 (TT,M) and final w, r."""
    np.random.seed(seed)
    Lyr, N, M = A.shape
    lam = np.exp(-1.0 / tau_m)
    eta = 1.0 - np.exp(-1.0 / tau_r)
    w = w0.copy()
    r = r0.copy()
    spill = np.zeros((Lyr, N))
    newspill = np.zeros((Lyr, N))
    TT = T - burn
    s_short = np.zeros((TT, Lyr))
    s_phi = np.zeros((TT, Lyr))
    s_nres = np.zeros((TT, Lyr))
    s_hagg = np.zeros((TT, Lyr))
    s_himp = np.zeros((TT, Lyr))
    s_rbar = np.zeros((TT, Lyr))
    s_corr = np.zeros((TT, Lyr))
    s_shi = np.zeros((TT, N))
    Corr = np.zeros((Lyr, M, M))
    for l in range(Lyr):
        Corr[l] = Lch[l] @ Lch[l].T
    s_w = np.zeros((TT, M))
    s_r = np.zeros((TT, M))
    o = np.zeros((N, M))
    deliv = np.zeros((N, M))
    openm = np.zeros(M, dtype=np.bool_)
    X = np.zeros(M)
    Th = np.zeros(M)
    xi = np.zeros(M)
    tgt = np.zeros(M)
    loads = np.zeros(M)
    for t in range(T):
        # one historical year is drawn per period and shared by all layers (keeps cross-crop correlation)
        yy = np.random.randint(anom.shape[1]) if anom.shape[1] > 1 else 0
        for l in range(Lyr):
            for i in range(N):
                newspill[l, i] = 0.0
        for l in range(Lyr):
            if anom.shape[1] > 1:
                for j in range(M):
                    xi[j] = anom[l, yy, j]
            else:
                g = np.random.standard_normal(M)
                xi[:] = Lch[l] @ g
            for j in range(M):
                X[j] = Xbar[l, j] + Psc[l, j] * xi[j]
                if X[j] < 0.0:
                    X[j] = 0.0
                if shock_mask[l, j] > 0.0 and np.random.random() < p_shock:
                    X[j] *= (1.0 - z_shock)
                Th[j] = X[j] + slack[l, j]
            Dtot = 0.0
            for i in range(N):
                rho = 0.0
                for j in range(M):
                    rho += w[l, i, j] * r[l, j]
                D = Dbar[l, i] * (1.0 + kappa * (rho + rho_ext[t])) + spill[l, i]
                Dtot += D
                for j in range(M):
                    o[i, j] = D * w[l, i, j]
            # orders placed before cascade, per importer
            Dord = np.zeros(N)
            for i in range(N):
                s = 0.0
                for j in range(M):
                    s += o[i, j]
                Dord[i] = s
            nres, req, deliv_j, unmet = cascade_layer(o, w[l], X, Th, b, q, openm, deliv)
            deltot = 0.0
            for i in range(N):
                di = 0.0
                for j in range(M):
                    di += deliv[i, j]
                deltot += di
                if Lyr > 1 and chi > 0.0:
                    gap = Dord[i] - di
                    if gap > 0.0:
                        for l2 in range(Lyr):
                            if l2 != l:
                                newspill[l2, i] += chi * gap / (Lyr - 1)
            if t >= burn:
                tt = t - burn
                s_short[tt, l] = 1.0 - deltot / Dtot if Dtot > 0.0 else 0.0
                vr = 0.0
                vv = 0.0
                for j in range(M):
                    vv += Xbar[l, j]
                    if not openm[j]:
                        vr += Xbar[l, j]
                s_phi[tt, l] = vr / vv
                s_nres[tt, l] = nres
                tot = 0.0
                for j in range(M):
                    s = 0.0
                    for i in range(N):
                        s += Dbar[l, i] * w[l, i, j]
                    loads[j] = s
                    tot += s
                h = 0.0
                rb = 0.0
                for j in range(M):
                    h += (loads[j] / tot) ** 2
                    rb += loads[j] / tot * r[l, j]
                s_hagg[tt, l] = h
                s_rbar[tt, l] = rb
                hi = 0.0
                ds = 0.0
                for i in range(N):
                    hh = 0.0
                    for j in range(M):
                        hh += w[l, i, j] ** 2
                    hi += Dbar[l, i] * hh
                    ds += Dbar[l, i]
                s_himp[tt, l] = hi / ds
                cc = 0.0
                for i in range(N):
                    pc = 0.0
                    for j in range(M):
                        if w[l, i, j] > 1e-6:
                            for k in range(M):
                                pc += w[l, i, j] * w[l, i, k] * Corr[l, j, k]
                    cc += Dbar[l, i] * pc
                s_corr[tt, l] = cc / ds
                if l == 0:
                    for i in range(N):
                        di = 0.0
                        for j in range(M):
                            di += deliv[i, j]
                        s_shi[tt, i] = 1.0 - di / Dord[i] if Dord[i] > 0 else 0.0
                if l == 0 and rec_w >= 0:
                    for j in range(M):
                        s_w[tt, j] = w[0, rec_w, j]
                        s_r[tt, j] = r[0, j]
            for j in range(M):
                f = 0.0
                if req[j] > 1e-12:
                    f = 1.0 - deliv_j[j] / req[j]
                    if f < 0.0:
                        f = 0.0
                r[l, j] = lam * r[l, j] + f
            for i in range(N):
                mx = -1e300
                for j in range(M):
                    if A[l, i, j] > 0.0:
                        u = -gam * Cst[l, i, j] - beta[i] * r[l, j]
                        if u > mx:
                            mx = u
                s = 0.0
                for j in range(M):
                    if A[l, i, j] > 0.0:
                        tgt[j] = np.exp(-gam * Cst[l, i, j] - beta[i] * r[l, j] - mx)
                        s += tgt[j]
                    else:
                        tgt[j] = 0.0
                if s > 0.0:
                    for j in range(M):
                        w[l, i, j] += eta * (tgt[j] / s - w[l, i, j])
        for l in range(Lyr):
            for i in range(N):
                spill[l, i] = newspill[l, i]
    return s_short, s_phi, s_nres, s_hagg, s_himp, s_rbar, s_w, s_r, w, r, s_corr, s_shi


@njit(cache=True)
def static_cascade(o0, w, X, Th, b, q):
    """Cascade on fixed orders; returns (restricted export-capacity indicator, shortfall)."""
    N, M = o0.shape
    o = o0.copy()
    openm = np.zeros(M, dtype=np.bool_)
    deliv = np.zeros((N, M))
    nres, req, dj, unmet = cascade_layer(o, w, X, Th, b, q, openm, deliv)
    D = 0.0
    for i in range(N):
        for j in range(M):
            D += o0[i, j]
    return openm, 1.0 - dj.sum() / D, nres


@njit(cache=True)
def static_scan(D, w, Xbar, Psc, slack, anom_rows, us, ndraw, seed, b, q):
    """Noise-averaged static cascade response on a fixed network.
    For each of ndraw bootstrap draws of a historical supply-anomaly row and each demand scale u:
    returns Sh (ndraw,nu) shortfall fraction, Ph (ndraw,nu) restricted export-capacity share,
    Fl (ndraw,nu) load-weighted failure rate, NR (ndraw,nu) number of restricting exporters."""
    np.random.seed(seed)
    N, M = w.shape
    nu = us.shape[0]
    Sh = np.zeros((ndraw, nu))
    Ph = np.zeros((ndraw, nu))
    Fl = np.zeros((ndraw, nu))
    NR = np.zeros((ndraw, nu))
    loads = np.zeros(M)
    tot = 0.0
    for j in range(M):
        s = 0.0
        for i in range(N):
            s += D[i] * w[i, j]
        loads[j] = s
        tot += s
    vol = 0.0
    for j in range(M):
        vol += Xbar[j]
    X = np.zeros(M)
    Th = np.zeros(M)
    o = np.zeros((N, M))
    deliv = np.zeros((N, M))
    openm = np.zeros(M, dtype=np.bool_)
    for d in range(ndraw):
        yy = np.random.randint(anom_rows.shape[0])
        for j in range(M):
            X[j] = Xbar[j] + Psc[j] * anom_rows[yy, j]
            if X[j] < 0.0:
                X[j] = 0.0
            Th[j] = X[j] + slack[j]
        for k in range(nu):
            Dt = 0.0
            for i in range(N):
                for j in range(M):
                    o[i, j] = us[k] * D[i] * w[i, j]
                    Dt += o[i, j]
            nres, req, dj, unmet = cascade_layer(o, w, X, Th, b, q, openm, deliv)
            dsum = 0.0
            f = 0.0
            pr = 0.0
            for j in range(M):
                dsum += dj[j]
                if req[j] > 1e-12:
                    fj = 1.0 - dj[j] / req[j]
                    if fj < 0.0:
                        fj = 0.0
                    f += loads[j] / tot * fj
                if not openm[j]:
                    pr += Xbar[j]
            Sh[d, k] = 1.0 - dsum / Dt if Dt > 0 else 0.0
            Ph[d, k] = pr / vol if vol > 0 else 0.0
            Fl[d, k] = f
            NR[d, k] = nres
    return Sh, Ph, Fl, NR

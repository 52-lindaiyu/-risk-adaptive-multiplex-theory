"""
Risk-adaptive multiplex trade model, generalised core (revision).

Compared with sim_core.py this version adds, all switchable:
  * importer-specific risk aversion beta_i, memory lambda_i = exp(-1/tau_m,i) and friction eta_i
  * public salience r_j (memory lam_pub) and private salience r_ij learned only from own deliveries;
    perceived salience  r~_ij = mu * r_j + (1-mu) * r_ij + noise_sd * |eps|
  * three restriction closures: 0 = hard threshold (withhold b when O_j > Theta_j),
    1 = smooth (withholding share b * logistic((O_j/Theta_j - 1)/sw)), 2 = rationing only (no withholding;
    deliveries capped at Theta_j, a share q of unmet volume re-routed to suppliers with spare capacity)
  * bounded precautionary demand  D_i = Dbar_i [1 + kmax tanh(kappa (rho_i + rho_ext)/kmax)]  (kmax <= 0: linear)
  * directed, capped cross-crop substitution of UNSERVED BASELINE demand:
      spill_{i,l2} = min(cap * Dbar_{i,l2}, sum_l chi[l,l2] * max(0, Dbar_{i,l} - delivered_{i,l}))
  * exogenous shocks either as capacity losses (shock_mode 0) or as direct salience impulses of size z (mode 1)
  * outputs both shortfall of orders and shortfall of baseline demand (unserved baseline consumption)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def cascade_mode(o, w, X, Th, b, q, mode, sw, open_out, deliv_out, bint_out):
    """One-period allocation for one layer.  o: (N,M) orders, modified in place to effective requests.
    Returns nres (count of exporters with withholding share > b/2 or at capacity), req_j, deliv_j."""
    N, M = o.shape
    req = np.zeros(M)
    O = np.zeros(M)
    for j in range(M):
        s = 0.0
        for i in range(N):
            s += o[i, j]
        req[j] = s
        O[j] = s
    nres = 0
    if mode == 0:
        restricted = np.zeros(M, dtype=np.bool_)
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
        for j in range(M):
            bint_out[j] = b if restricted[j] else 0.0
            open_out[j] = not restricted[j]
        for j in range(M):
            lim = Th[j] if not restricted[j] else X[j]
            cap = 1.0
            if O[j] > lim and O[j] > 0.0:
                cap = lim / O[j]
            for i in range(N):
                deliv_out[i, j] = o[i, j] * cap
    elif mode == 1:
        base = o.copy()
        bj = np.zeros(M)
        extra = np.zeros((N, M))
        for it in range(60):
            # total requests = base + rerouted extra
            for j in range(M):
                s = 0.0
                for i in range(N):
                    s += base[i, j] + extra[i, j]
                O[j] = s
            maxd = 0.0
            for j in range(M):
                if Th[j] > 0.0:
                    z = (O[j] / Th[j] - 1.0) / sw
                else:
                    z = 50.0 if O[j] > 0.0 else -50.0
                if z > 50.0:
                    z = 50.0
                if z < -50.0:
                    z = -50.0
                nb = b / (1.0 + np.exp(-z))
                d = abs(nb - bj[j])
                if d > maxd:
                    maxd = d
                bj[j] = 0.5 * bj[j] + 0.5 * nb
            # re-routing of withheld volume
            for i in range(N):
                wh = 0.0
                for j in range(M):
                    wh += bj[j] * (base[i, j] + extra[i, j])
                den = 0.0
                for k in range(M):
                    den += w[i, k] * (1.0 - bj[k] / b if b > 0 else 1.0)
                for k in range(M):
                    tgt = 0.0
                    if den > 1e-12:
                        tgt = q * wh * w[i, k] * (1.0 - bj[k] / b if b > 0 else 1.0) / den
                    extra[i, k] = 0.5 * extra[i, k] + 0.5 * tgt
            if maxd < 1e-6 and it > 5:
                break
        for j in range(M):
            s = 0.0
            for i in range(N):
                o[i, j] = base[i, j] + extra[i, j]
                s += o[i, j]
            req[j] = s
            bint_out[j] = bj[j]
            open_out[j] = bj[j] < 0.5 * b
            if bj[j] >= 0.5 * b:
                nres += 1
        for j in range(M):
            kept = (1.0 - bj[j]) * req[j]
            lim = Th[j]
            cap = 1.0
            if kept > lim and kept > 0.0:
                cap = lim / kept
            for i in range(N):
                deliv_out[i, j] = (1.0 - bj[j]) * o[i, j] * cap
    else:
        # rationing only: capacity caps, one or two rounds of re-routing to spare capacity
        for rnd in range(3):
            for j in range(M):
                s = 0.0
                for i in range(N):
                    s += o[i, j]
                O[j] = s
            spare = np.zeros(M)
            stot = 0.0
            for j in range(M):
                if Th[j] > O[j]:
                    spare[j] = Th[j] - O[j]
                    stot += spare[j]
            if stot <= 1e-12:
                break
            moved = False
            for i in range(N):
                un = 0.0
                for j in range(M):
                    if O[j] > Th[j] and O[j] > 0.0:
                        un += o[i, j] * (1.0 - Th[j] / O[j])
                if un <= 1e-12:
                    continue
                den = 0.0
                for k in range(M):
                    if spare[k] > 0.0:
                        den += w[i, k]
                if den <= 1e-12:
                    continue
                for k in range(M):
                    if spare[k] > 0.0:
                        add = q * un * w[i, k] / den
                        if add > spare[k]:
                            add = spare[k]
                        o[i, k] += add
                        req[k] += add
                        moved = True
            if not moved:
                break
        for j in range(M):
            s = 0.0
            for i in range(N):
                s += o[i, j]
            O[j] = s
            cap = 1.0
            if O[j] > Th[j] and O[j] > 0.0:
                cap = Th[j] / O[j]
                nres += 1
            open_out[j] = cap >= 1.0
            bint_out[j] = 1.0 - cap
            for i in range(N):
                deliv_out[i, j] = o[i, j] * cap
    dj = np.zeros(M)
    for j in range(M):
        s = 0.0
        for i in range(N):
            s += deliv_out[i, j]
        dj[j] = s
    return nres, req, dj


@njit(cache=True)
def simulate2(A, Cst, Dbar, Xbar, slack, Psc, Lch, anom, w0,
              beta_i, lam_i, eta_i, lam_pub, mu, noise_sd,
              gam, b, q, mode, sw, kappa, kmax, chi, cap,
              p_shock, z_shock, shock_mode, shock_mask, rho_ext, T, burn, seed, rec_i):
    """Returns (TT,L) arrays: short_orders, short_base, phi, nres, hhi_agg, hhi_imp, rbar(perceived, load-weighted),
    pc(portfolio corr), plus (TT,N) per-importer baseline shortfall of layer 0, (TT,M) shares of importer rec_i
    in layer 0, (TT,M) public salience of layer 0, final w, and (if rec_i == -2) (TT,N,M) layer-0 deliveries."""
    np.random.seed(seed)
    L, N, M = A.shape
    w = w0.copy()
    rpub = np.zeros((L, M))
    rpriv = np.zeros((L, N, M))
    spill = np.zeros((L, N))
    newspill = np.zeros((L, N))
    TT = T - burn
    s_so = np.zeros((TT, L))
    s_sb = np.zeros((TT, L))
    s_phi = np.zeros((TT, L))
    s_nres = np.zeros((TT, L))
    s_hagg = np.zeros((TT, L))
    s_himp = np.zeros((TT, L))
    s_rbar = np.zeros((TT, L))
    s_pc = np.zeros((TT, L))
    s_shi = np.zeros((TT, N))
    s_w = np.zeros((TT, M))
    s_r = np.zeros((TT, M))
    if rec_i == -2:
        s_del = np.zeros((TT, N, M))
    else:
        s_del = np.zeros((1, 1, 1))
    Corr = np.zeros((L, M, M))
    for l in range(L):
        Corr[l] = Lch[l] @ Lch[l].T
    o = np.zeros((N, M))
    deliv = np.zeros((N, M))
    openm = np.zeros(M, dtype=np.bool_)
    bint = np.zeros(M)
    X = np.zeros(M)
    Th = np.zeros(M)
    xi = np.zeros(M)
    tgt = np.zeros(M)
    rt = np.zeros(M)
    loads = np.zeros(M)
    Dord = np.zeros(N)
    dgap = np.zeros((L, N))
    for t in range(T):
        if shock_mode == 2:
            yy = t % anom.shape[1]          # replay anomaly rows in order (historical sequence)
        else:
            yy = np.random.randint(anom.shape[1]) if anom.shape[1] > 1 else 0
        for l in range(L):
            if anom.shape[1] > 1:
                for j in range(M):
                    xi[j] = anom[l, yy, j]
            else:
                g = np.random.standard_normal(M)
                xi[:] = Lch[l] @ g
            imp = np.zeros(M)
            for j in range(M):
                X[j] = Xbar[l, j] + Psc[l, j] * xi[j]
                if X[j] < 0.0:
                    X[j] = 0.0
                if shock_mask[l, j] > 0.0 and np.random.random() < p_shock:
                    if shock_mode == 0:
                        X[j] *= (1.0 - z_shock)
                    else:
                        imp[j] = z_shock
                Th[j] = X[j] + slack[l, j]
            Dtot = 0.0
            Dbt = 0.0
            for i in range(N):
                rho = 0.0
                for j in range(M):
                    rp = mu * rpub[l, j] + (1.0 - mu) * rpriv[l, i, j]
                    rho += w[l, i, j] * rp
                x_ = kappa * (rho + rho_ext[t])
                if kmax > 0.0:
                    x_ = kmax * np.tanh(x_ / kmax)
                D = Dbar[l, i] * (1.0 + x_) + spill[l, i]
                Dtot += D
                Dbt += Dbar[l, i]
                Dord[i] = D
                for j in range(M):
                    o[i, j] = D * w[l, i, j]
            orig = o.copy()
            nres, req, dj = cascade_mode(o, w[l], X, Th, b, q, mode, sw, openm, deliv, bint)
            deltot = 0.0
            sbase = 0.0
            for i in range(N):
                di = 0.0
                for j in range(M):
                    di += deliv[i, j]
                deltot += di
                gb = Dbar[l, i] - di
                if gb < 0.0:
                    gb = 0.0
                dgap[l, i] = gb
                sbase += gb
            if t >= burn:
                tt = t - burn
                s_so[tt, l] = 1.0 - deltot / Dtot if Dtot > 0.0 else 0.0
                s_sb[tt, l] = sbase / Dbt if Dbt > 0.0 else 0.0
                vr = 0.0
                vv = 0.0
                for j in range(M):
                    vv += Xbar[l, j]
                    vr += Xbar[l, j] * (bint[j] / b if (mode != 2 and b > 0) else bint[j])
                s_phi[tt, l] = vr / vv if vv > 0 else 0.0
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
                s_hagg[tt, l] = h
                hi = 0.0
                ds = 0.0
                cc = 0.0
                for i in range(N):
                    hh = 0.0
                    pc = 0.0
                    for j in range(M):
                        hh += w[l, i, j] ** 2
                        rb += Dbar[l, i] * w[l, i, j] * (mu * rpub[l, j] + (1.0 - mu) * rpriv[l, i, j])
                        if w[l, i, j] > 1e-6:
                            for k in range(M):
                                pc += w[l, i, j] * w[l, i, k] * Corr[l, j, k]
                    hi += Dbar[l, i] * hh
                    cc += Dbar[l, i] * pc
                    ds += Dbar[l, i]
                s_himp[tt, l] = hi / ds
                s_pc[tt, l] = cc / ds
                s_rbar[tt, l] = rb / ds
                if l == 0:
                    for i in range(N):
                        s_shi[tt, i] = dgap[0, i] / Dbar[0, i] if Dbar[0, i] > 0 else 0.0
                    if rec_i >= 0:
                        for j in range(M):
                            s_w[tt, j] = w[0, rec_i, j]
                            s_r[tt, j] = rpub[0, j]
                    if rec_i == -2:
                        for i in range(N):
                            for j in range(M):
                                s_del[tt, i, j] = deliv[i, j]
            # salience updates
            for j in range(M):
                f = 0.0
                if req[j] > 1e-12:
                    f = 1.0 - dj[j] / req[j]
                    if f < 0.0:
                        f = 0.0
                rpub[l, j] = lam_pub * rpub[l, j] + f + imp[j]
            for i in range(N):
                for j in range(M):
                    fij = 0.0
                    if o[i, j] > 1e-12:
                        fij = 1.0 - deliv[i, j] / o[i, j]
                        if fij < 0.0:
                            fij = 0.0
                    rpriv[l, i, j] = lam_i[i] * rpriv[l, i, j] + fij + imp[j]
            # rewiring
            for i in range(N):
                mx = -1e300
                for j in range(M):
                    rt[j] = mu * rpub[l, j] + (1.0 - mu) * rpriv[l, i, j]
                    if noise_sd > 0.0:
                        rt[j] += noise_sd * abs(np.random.standard_normal())
                    if A[l, i, j] > 0.0:
                        u = -gam * Cst[l, i, j] - beta_i[i] * rt[j]
                        if u > mx:
                            mx = u
                s = 0.0
                for j in range(M):
                    if A[l, i, j] > 0.0:
                        tgt[j] = np.exp(-gam * Cst[l, i, j] - beta_i[i] * rt[j] - mx)
                        s += tgt[j]
                    else:
                        tgt[j] = 0.0
                if s > 0.0:
                    for j in range(M):
                        w[l, i, j] += eta_i[i] * (tgt[j] / s - w[l, i, j])
        # substitution of unserved baseline demand (directed, capped), applied next period
        for l2 in range(L):
            for i in range(N):
                s = 0.0
                for l in range(L):
                    if l != l2:
                        s += chi[l, l2] * dgap[l, i]
                lim = cap * Dbar[l2, i]
                if s > lim:
                    s = lim
                newspill[l2, i] = s
        for l in range(L):
            for i in range(N):
                spill[l, i] = newspill[l, i]
    return s_so, s_sb, s_phi, s_nres, s_hagg, s_himp, s_rbar, s_pc, s_shi, s_w, s_r, w, s_del


@njit(cache=True)
def static_scan2(D, w, Xbar, Psc, slack, anom_rows, us, ndraw, seed, b, q, mode, sw):
    """Noise-averaged static response with frozen shares.  Returns per draw x u:
    Sh (order shortfall), Sb (baseline shortfall, D at u=1 as baseline), Ph (restricted capacity share),
    Fl (load-weighted failure rate)."""
    np.random.seed(seed)
    N, M = w.shape
    nu = us.shape[0]
    Sh = np.zeros((ndraw, nu))
    Sb = np.zeros((ndraw, nu))
    Ph = np.zeros((ndraw, nu))
    Fl = np.zeros((ndraw, nu))
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
    bint = np.zeros(M)
    Db = 0.0
    for i in range(N):
        Db += D[i]
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
            nres, req, dj = cascade_mode(o, w, X, Th, b, q, mode, sw, openm, deliv, bint)
            dsum = 0.0
            f = 0.0
            pr = 0.0
            sb = 0.0
            for i in range(N):
                di = 0.0
                for j in range(M):
                    di += deliv[i, j]
                g = D[i] - di
                if g > 0.0:
                    sb += g
            for j in range(M):
                dsum += dj[j]
                if req[j] > 1e-12:
                    fj = 1.0 - dj[j] / req[j]
                    if fj < 0.0:
                        fj = 0.0
                    f += loads[j] / tot * fj
                pr += Xbar[j] * (bint[j] / b if (mode != 2 and b > 0) else bint[j])
            Sh[d, k] = 1.0 - dsum / Dt if Dt > 0 else 0.0
            Sb[d, k] = sb / Db if Db > 0 else 0.0
            Ph[d, k] = pr / vol if vol > 0 else 0.0
            Fl[d, k] = f
    return Sh, Sb, Ph, Fl


@njit(cache=True)
def expected_failure(D, w, Xbar, Psc, slack, anom_rows, draw_idx, r, kappa, rho_ext, b, q, mode, sw):
    """Deterministic heterogeneous map ingredient: E_draws[f_j] when importer demand is
    D_i (1 + kappa (sum_j w_ij r_j + rho_ext)), shares frozen, over a FIXED list of anomaly rows (CRN)."""
    N, M = w.shape
    X = np.zeros(M)
    Th = np.zeros(M)
    o = np.zeros((N, M))
    deliv = np.zeros((N, M))
    openm = np.zeros(M, dtype=np.bool_)
    bint = np.zeros(M)
    Ef = np.zeros(M)
    mult = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(M):
            s += w[i, j] * r[j]
        mult[i] = 1.0 + kappa * (s + rho_ext)
    nd = draw_idx.shape[0]
    for d in range(nd):
        yy = draw_idx[d]
        for j in range(M):
            X[j] = Xbar[j] + Psc[j] * anom_rows[yy, j]
            if X[j] < 0.0:
                X[j] = 0.0
            Th[j] = X[j] + slack[j]
        for i in range(N):
            for j in range(M):
                o[i, j] = mult[i] * D[i] * w[i, j]
        nres, req, dj = cascade_mode(o, w, X, Th, b, q, mode, sw, openm, deliv, bint)
        for j in range(M):
            if req[j] > 1e-12:
                fj = 1.0 - dj[j] / req[j]
                if fj < 0.0:
                    fj = 0.0
                Ef[j] += fj
    for j in range(M):
        Ef[j] /= nd
    return Ef

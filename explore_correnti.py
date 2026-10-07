#!/usr/bin/env python3
"""Validazione una tantum degli orari di inversione delle correnti nello Stretto di Messina.
1) livello del mare (zos) del modello Copernicus vs mareografi ISPRA (Catania, Messina, Reggio, Ginostra, Strombolicchio): correlazione e sfasamento;
2) corrente lungo lo stretto del modello vs dislivello misurato tra imbocco sud (Catania) e nord (Stromboli): correlazione, sfasamento e orari di inversione.
Scrive correnti_explore.json."""
import datetime as dt, json, math, os, sys
import numpy as np, requests
import copernicusmarine as cm

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
KN = 1.943844
DS_SSH = "cmems_mod_med_phy-ssh_anfc_4.2km-2D_PT1H-m"
DS_CUR = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
STAZ = {"CT03": ("Catania", 37.498, 15.093), "ME13": ("Messina", 38.196, 15.563), "RC09": ("Reggio Calabria", 38.121, 15.648),
        "GI20": ("Ginostra", 38.784, 15.193), "ST44": ("Strombolicchio", 38.817, 15.251)}
now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
t0, t1 = now - dt.timedelta(hours=71), now
res = {"generated": dt.datetime.utcnow().isoformat(), "finestra_utc": [t0.isoformat(), t1.isoformat()]}
hours = [t0 + dt.timedelta(hours=i) for i in range(72)]
fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S")

def hourly_obs(code):
    r = requests.get("https://www.ioc-sealevelmonitoring.org/service.php", timeout=90, params={
        "query": "data", "code": code, "format": "json", "timestart": fmt(t0 - dt.timedelta(hours=1)), "timestop": fmt(t1 + dt.timedelta(hours=1))})
    r.raise_for_status()
    rows = [(dt.datetime.strptime(x["stime"][:19], "%Y-%m-%d %H:%M:%S"), float(x["slevel"])) for x in r.json() if x.get("slevel") is not None]
    ref = np.median([v for _, v in rows])
    rows = [(t, v) for t, v in rows if abs(v - ref) < 1.5]
    out = []
    for h in hours:
        v = [x for t, x in rows if h - dt.timedelta(minutes=30) <= t < h + dt.timedelta(minutes=30)]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)

def corr_lag(a, b, maxlag=4):
    """correlazione tra a(t) e b(t+lag); ritorna (lag_migliore_ore, corr). lag>0: b in ritardo su a."""
    best = (0, -2)
    for lag in range(-maxlag, maxlag + 1):
        x, y = (a[max(0, -lag): len(a) - max(0, lag)], b[max(0, lag): len(b) - max(0, -lag)])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 24: continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        if c > best[1]: best = (lag, float(c))
    return best

# ---------- 1) livello del mare: modello vs mareografi ----------
try:
    ds = cm.open_dataset(dataset_id=DS_SSH, variables=["zos"], minimum_longitude=14.8, maximum_longitude=16.0,
                         minimum_latitude=37.4, maximum_latitude=38.9, start_datetime=fmt(t0), end_datetime=fmt(t1))
    z = ds["zos"].load()
    lon, lat = z["longitude"].values, z["latitude"].values
    tt = [str(t)[:13] for t in z["time"].values]
    V1 = {}
    obs = {}
    for code, (nome, la, lo) in STAZ.items():
        try:
            o = hourly_obs(code); obs[code] = o
            # cella di mare piu' vicina alla stazione
            d = (lat[:, None] - la) ** 2 + (lon[None, :] - lo) ** 2
            d = np.where(np.isfinite(z.values[0]), d, np.inf)
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            m = z.values[:, a, b].astype(float); m = m - np.nanmean(m)
            ob = o - np.nanmean(o)
            n = min(len(m), len(ob))
            lag, c = corr_lag(ob[:n], m[:n])
            V1[code] = {"stazione": nome, "cella": [round(float(lat[a]), 3), round(float(lon[b]), 3)], "corr": round(c, 2), "lag_modello_ore": lag,
                        "escursione_oss_cm": round(float((np.nanmax(ob) - np.nanmin(ob)) * 100)), "escursione_mod_cm": round(float((np.nanmax(m) - np.nanmin(m)) * 100))}
        except Exception as e:
            V1[code] = {"errore": repr(e)[:200]}
    res["livello_modello_vs_mareografi"] = V1
except Exception as e:
    res["errore_livello"] = repr(e)[:400]; print("LIVELLO FALLITO", repr(e)[:400])

# ---------- 2) corrente lungo lo stretto vs dislivello misurato ----------
try:
    ds = cm.open_dataset(dataset_id=DS_CUR, variables=["uo", "vo"], minimum_longitude=15.45, maximum_longitude=15.80,
                         minimum_latitude=37.95, maximum_latitude=38.45, start_datetime=fmt(t0), end_datetime=fmt(t1))
    u = ds["uo"].load(); v = ds["vo"].load()
    if "depth" in u.dims: u = u.isel(depth=0); v = v.isel(depth=0)
    ax = (math.sin(math.radians(20)), math.cos(math.radians(20)))
    core = []
    for i in range(u.shape[0]):
        uu, vv = u.values[i], v.values[i]; ok = np.isfinite(uu) & np.isfinite(vv)
        if not ok.any(): core.append(np.nan); continue
        spd = np.hypot(uu[ok], vv[ok]); k = int(np.argmax(spd))
        core.append(float(((uu * ax[0] + vv * ax[1])[ok] * KN)[k]))
    core = np.array(core)
    res["corrente_nucleo_kn"] = [round(float(x), 2) for x in core]
    sud = obs.get("CT03")
    res["dislivello"] = {}
    def zero_cross(x, soglia):
        out, last = [], 0
        for i, val in enumerate(x):
            sg = 1 if val > soglia else -1 if val < -soglia else 0
            if sg and last and sg != last: out.append([hours[i].strftime("%m-%d %H:00"), "montante(N)" if sg > 0 else "scendente(S)"])
            if sg: last = sg
        return out
    res["inversioni_modello"] = zero_cross(core, 0.2)
    for cod in ("ST44", "GI20"):
        nord = obs.get(cod)
        if sud is None or nord is None or np.nanstd(nord) < 0.01 or np.isfinite(nord).sum() < 48:
            res["dislivello"][cod] = {"scartato": "sensore fermo o dati insufficienti"}; continue
        d = (sud - np.nanmean(sud)) - (nord - np.nanmean(nord))     # >0: Ionio piu' alto del Tirreno -> corrente verso nord (montante)
        n = min(len(core), len(d))
        lag, c = corr_lag(core[:n], d[:n], maxlag=6)
        sm = np.convolve(np.nan_to_num(d), np.ones(3) / 3, mode="same")
        res["dislivello"][cod] = {"corr": round(c, 2), "lag_dislivello_ore": lag, "inversioni": zero_cross(sm, 0.02),
                                  "dislivello_cm": [round(float(x) * 100, 1) if np.isfinite(x) else None for x in d]}
except Exception as e:
    res["errore_corrente"] = repr(e)[:400]; print("CORRENTE FALLITA", repr(e)[:400])

json.dump(res, open(os.path.join(OUT, "correnti_explore.json"), "w"), ensure_ascii=False)
print("scritto correnti_explore.json:", [k for k in res])

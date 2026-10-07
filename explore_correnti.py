#!/usr/bin/env python3
"""Esplorazione una tantum delle correnti Copernicus (modello fisico Mediterraneo, 4 km):
 A) Stretto di Sicilia: dove si concentra il getto, profilo lungo un meridiano;
 B) Stretto di Messina e Bocche di Bonifacio: serie oraria della componente lungo lo stretto e orari di inversione.
Scrive correnti_explore.json."""
import datetime as dt, json, math, os, sys
import numpy as np
import copernicusmarine as cm

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
KN = 1.943844
DS_D = "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m"
DS_H = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
res = {"generated": dt.datetime.utcnow().isoformat()}
today = dt.datetime.utcnow().strftime("%Y-%m-%dT00:00:00")

# ---------- A) Stretto di Sicilia, media giornaliera ----------
try:
    ds = cm.open_dataset(dataset_id=DS_D, variables=["uo", "vo"], minimum_depth=0, maximum_depth=2,
                         minimum_longitude=9.5, maximum_longitude=16.5, minimum_latitude=34.8, maximum_latitude=38.8,
                         start_datetime=today, end_datetime=(dt.datetime.utcnow() + dt.timedelta(days=1)).strftime("%Y-%m-%dT23:59:59"))
    u = ds["uo"].isel(depth=0).load(); v = ds["vo"].isel(depth=0).load()
    lon, lat = u["longitude"].values, u["latitude"].values
    days = [str(t)[:10] for t in u["time"].values]
    A = {"giorni": days}
    for i, day in enumerate(days):
        uu, vv = u.values[i], v.values[i]
        spd = np.hypot(uu, vv) * KN
        ok = np.isfinite(spd)
        flat = np.where(ok, spd, -1).ravel()
        top = np.argsort(flat)[::-1][:12]
        pts = []
        for k in top:
            a, b = divmod(int(k), len(lon))
            pts.append({"lat": round(float(lat[a]), 2), "lon": round(float(lon[b]), 2), "kn": round(float(spd[a, b]), 2),
                        "verso_deg": round(float(math.degrees(math.atan2(uu[a, b], vv[a, b])) % 360))})
        # profilo lungo meridiani 11.5, 13.0, 14.5: componente verso est (kn) per latitudine
        prof = {}
        for m in (11.5, 13.0, 14.5):
            j = int(np.argmin(abs(lon - m)))
            prof[str(m)] = [[round(float(lat[a]), 2), None if not np.isfinite(uu[a, j]) else round(float(uu[a, j] * KN), 2)]
                            for a in range(0, len(lat), 6)]
        # statistiche
        s = spd[ok]
        A[day] = {"velocita_media_kn": round(float(s.mean()), 3), "p95_kn": round(float(np.percentile(s, 95)), 3),
                  "max_kn": round(float(s.max()), 3), "celle_>0.8kn": int((s > 0.8).sum()), "celle_totali": int(ok.sum()),
                  "top12": pts, "profilo_u_est": prof}
    res["stretto_di_sicilia"] = A
except Exception as e:
    res["errore_sicilia"] = repr(e)[:500]; print("A FALLITO", repr(e)[:500])


# ---------- B) Stretti a marea, serie oraria ----------
def strait(name, bbox, axis_deg, hours=72):
    ds = cm.open_dataset(dataset_id=DS_H, variables=["uo", "vo"], minimum_longitude=bbox[0], maximum_longitude=bbox[1],
                         minimum_latitude=bbox[2], maximum_latitude=bbox[3],
                         start_datetime=today, end_datetime=(dt.datetime.utcnow() + dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S"))
    u = ds["uo"].load(); v = ds["vo"].load()
    if "depth" in u.dims:
        u = u.isel(depth=0); v = v.isel(depth=0)
    lon, lat = u["longitude"].values, u["latitude"].values
    times = [str(t)[:16] for t in u["time"].values]
    ax = (math.sin(math.radians(axis_deg)), math.cos(math.radians(axis_deg)))      # versore asse (verso 'axis_deg')
    series, along_mean = [], []
    for i, t in enumerate(times):
        uu, vv = u.values[i], v.values[i]
        ok = np.isfinite(uu) & np.isfinite(vv)
        if ok.sum() == 0:
            continue
        along = (uu * ax[0] + vv * ax[1])[ok] * KN
        spd = np.hypot(uu[ok], vv[ok]) * KN
        # cella con la corrente piu' forte (nucleo)
        k = int(np.argmax(spd))
        series.append({"t": t, "lungo_asse_medio_kn": round(float(along.mean()), 2), "lungo_asse_nucleo_kn": round(float(along[k]), 2),
                       "vel_max_kn": round(float(spd.max()), 2)})
        along_mean.append(float(along[k]))
    # inversioni: cambi di segno del nucleo con soglia 0.2 nodi
    rev = []
    last = 0
    for i, a in enumerate(along_mean):
        sg = 1 if a > 0.2 else -1 if a < -0.2 else 0
        if sg != 0 and last != 0 and sg != last:
            rev.append({"t": series[i]["t"], "verso": "asse+" if sg > 0 else "asse-"})
        if sg != 0:
            last = sg
    return {"asse_deg_verso": axis_deg, "celle_mare": int(np.isfinite(u.values[0]).sum()), "ore": len(series),
            "max_nucleo_kn": round(max(abs(x) for x in along_mean), 2) if along_mean else None, "inversioni": rev, "serie": series[:72]}


try:
    res["messina"] = strait("messina", (15.45, 15.80, 37.95, 38.45), 20)       # asse N-NE (20 gradi)
except Exception as e:
    res["errore_messina"] = repr(e)[:500]; print("MESSINA FALLITO", repr(e)[:500])
try:
    res["bonifacio"] = strait("bonifacio", (8.95, 9.55, 41.18, 41.50), 90)    # asse E-W (verso est)
except Exception as e:
    res["errore_bonifacio"] = repr(e)[:500]; print("BONIFACIO FALLITO", repr(e)[:500])

json.dump(res, open(os.path.join(OUT, "correnti_explore.json"), "w"), ensure_ascii=False)
print("scritto correnti_explore.json:", [k for k in res])

#!/usr/bin/env python3
"""Dati "ambiente" per il pannello Meteo ambiente: pH del mare e livello del mare per le 15 aree.

Fonti:
  - Copernicus Marine (CMEMS) MEDSEA_ANALYSISFORECAST_BGC_006_014: pH superficiale, media giornaliera, previsione 10 giorni
  - Copernicus Marine (CMEMS) MEDSEA_ANALYSISFORECAST_PHY_006_013: altezza del mare (zos) oraria, maree comprese
  - ISPRA / IOC Sea Level Monitoring: livello del mare osservato dai mareografi della Rete Mareografica Nazionale

Credenziali CMEMS nelle variabili d'ambiente COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD (segreti GitHub).
Uso: python fetch_ambiente.py [cartella_output]
"""
import datetime as dt
import json
import math
import os
import statistics
import sys
from zoneinfo import ZoneInfo

import numpy as np
import requests

# le aree sono le stesse di fetch_meteo.py (tenute in sincronia a mano)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_meteo import AREAS, in_polygon   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
BBOX = dict(minimum_longitude=5.5, maximum_longitude=20.5, minimum_latitude=34.8, maximum_latitude=46.3)
ROME = ZoneInfo("Europe/Rome")

DS_PH = "cmems_mod_med_bgc-car_anfc_4.2km_P1D-m"
DS_SSH = "cmems_mod_med_phy-ssh_anfc_4.2km-2D_PT1H-m"
DS_CURH = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"   # correnti orarie (comprendono la marea): per lo Stretto di Messina
DS_CUR = "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m"      # correnti (3D, media giornaliera): si usa lo strato superficiale
DS_WAV = "cmems_mod_med_wav_anfc_4.2km_PT1H-i"           # onde (Hm0, direzione, periodo), dati orari, previsione 10 giorni
DS_SST = "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"      # temperatura (3D, media giornaliera): si usa lo strato superficiale

# mareografo ISPRA piu' rappresentativo per ciascuna area (codice IOC, nome). Alcune aree non ne hanno.
STATIONS = {
    "mar-ligure": ("GE25", "Genova"),
    "tirreno-settentr": ("LI11", "Livorno"),
    "tirreno-centr-est": ("NA23", "Napoli"),
    "tirreno-merid-est": ("PA07", "Palermo"),
    "mar-di-sardegna": ("PT17", "Porto Torres"),
    "canale-di-sardegna": ("CA02", "Cagliari"),
    "stretto-di-sicilia": ("LA23", "Lampedusa"),
    "ionio-settentr": ("TA18", "Taranto"),
    "ionio-merid": ("CT03", "Catania"),
    "adriatico-settentr": ("VE19", "Venezia"),
    "adriatico-centr": ("AN15", "Ancona"),
    "adriatico-merid": ("BA05", "Bari"),
}


def area_mask(lon, lat, poly):
    LON, LAT = np.meshgrid(lon, lat)
    return in_polygon(LON, LAT, poly)


def local_day(ts):
    """datetime64 (UTC) -> data locale Europe/Rome in formato ISO."""
    t = dt.datetime.utcfromtimestamp(int(ts) / 1e9).replace(tzinfo=dt.timezone.utc)
    return t.astimezone(ROME).strftime("%Y-%m-%d")


def load_ph_clim():
    try:
        return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ph_clim.json"), encoding="utf-8"))["areas"]
    except Exception as e:
        print("climatologia pH non disponibile:", e)
        return {}


def cmems_ph(cm, result):
    """pH superficiale: media d'area per giorno (previsione fino a 10 giorni)."""
    start = dt.datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
    end = (dt.datetime.utcnow() + dt.timedelta(days=10)).strftime("%Y-%m-%dT00:00:00")
    ds = cm.open_dataset(dataset_id=DS_PH, variables=["ph"], minimum_depth=0, maximum_depth=2,
                         start_datetime=start, end_datetime=end, **BBOX)
    da = ds["ph"].isel(depth=0).load()
    lon, lat = da["longitude"].values, da["latitude"].values
    times = da["time"].values
    print("pH: forma", da.shape, "giorni", len(times), "da", str(times[0])[:10], "a", str(times[-1])[:10])
    clim = load_ph_clim()
    for key, name, poly in AREAS:
        m = area_mask(lon, lat, poly)
        vals, zs, pcts = {}, {}, {}
        c = clim.get(key)
        for i, t in enumerate(times):
            v = da.values[i][m]
            v = v[np.isfinite(v)]
            if len(v) >= 5:
                day = str(t)[:10]
                vals[day] = round(float(v.mean()), 3)
                if c:   # scarto in deviazioni standard dalla media mensile 1999-2019, corretto dello scarto tra i due sistemi
                    mth = int(day[5:7]) - 1
                    z = (float(v.mean()) - c["bias"] - c["avg"][mth]) / max(c["std"][mth], 0.004)
                    zs[day] = round(z, 1)
                    # acidita' = concentrazione di ioni idrogeno = 10^(-pH): variazione percentuale rispetto alla media del mese
                    pcts[day] = round((10 ** (c["avg"][mth] - (float(v.mean()) - c["bias"])) - 1) * 100)
        if vals:
            result[key]["ph"] = vals
        if zs:
            result[key]["ph_z"] = zs
        if pcts:
            result[key]["ph_pct"] = pcts


def cmems_messina(cm, result):
    """Stretto di Messina: orari in cui la corrente di marea cambia verso (stima dal modello orario Copernicus).
    Si segue il nucleo (cella piu' veloce) della corrente lungo l'asse dello stretto (verso 20 gradi, NNE): positivo = montante (verso N),
    negativo = scendente (verso S). Un'inversione e' un cambio di segno oltre la soglia di 0,2 nodi.
    Validato (7/10/2026): il livello del modello coincide coi mareografi ISPRA (corr 0.76-0.97, scarto <=1 h) e la corrente e' in sintonia col
    dislivello misurato tra gli imbocchi (corr 0.93), ma gli orari di inversione possono differire di circa 2 ore: vanno presentati come stime."""
    base = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(hours=3)
    ds = cm.open_dataset(dataset_id=DS_CURH, variables=["uo", "vo"], minimum_longitude=15.45, maximum_longitude=15.80,
                         minimum_latitude=37.95, maximum_latitude=38.45,
                         start_datetime=base.strftime("%Y-%m-%dT%H:%M:%S"), end_datetime=(base + dt.timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%S"))
    u = ds["uo"].load(); v = ds["vo"].load()
    if "depth" in u.dims:
        u = u.isel(depth=0); v = v.isel(depth=0)
    ax = (math.sin(math.radians(20)), math.cos(math.radians(20)))
    times, core = [], []
    for i, t in enumerate(u["time"].values):
        uu, vv = u.values[i], v.values[i]
        ok = np.isfinite(uu) & np.isfinite(vv)
        if not ok.any():
            continue
        spd = np.hypot(uu[ok], vv[ok])
        k = int(np.argmax(spd))
        along = float(((uu * ax[0] + vv * ax[1])[ok] * 1.943844)[k])
        times.append(dt.datetime.utcfromtimestamp(int(t) / 1e9).replace(tzinfo=dt.timezone.utc).astimezone(ROME))
        core.append(along)
    print("Messina: ore", len(core), "max nucleo", round(max(abs(x) for x in core), 2) if core else None)
    # fasi = tratti consecutivi con lo stesso segno (oltre 0,2 nodi); si scartano quelle brevi (< 2 ore) o deboli (picco < 0,6 nodi),
    # che sono code della marea e non vere inversioni utili alla navigazione, poi si fondono le fasi adiacenti dello stesso verso
    seg = []
    for t, a in zip(times, core):
        sg = 1 if a > 0.2 else -1 if a < -0.2 else 0
        if not sg:
            continue
        if seg and seg[-1]["sg"] == sg and (t - seg[-1]["end"]) <= dt.timedelta(hours=2):
            seg[-1]["end"] = t; seg[-1]["peak"] = max(seg[-1]["peak"], abs(a))
        else:
            seg.append({"sg": sg, "start": t, "end": t, "peak": abs(a)})
    seg = [x for x in seg if (x["end"] - x["start"]) >= dt.timedelta(hours=1) and x["peak"] >= 0.6]
    fused = []
    for x in seg:
        if fused and fused[-1]["sg"] == x["sg"]:
            fused[-1]["end"] = x["end"]; fused[-1]["peak"] = max(fused[-1]["peak"], x["peak"])
        else:
            fused.append(dict(x))
    out = {}
    for t, a in zip(times, core):
        e = out.setdefault(t.strftime("%Y-%m-%d"), {"inv": [], "max": 0.0})
        e["max"] = max(e["max"], abs(a))
    for prev, cur in zip(fused, fused[1:]):
        day = cur["start"].strftime("%Y-%m-%d")
        out.setdefault(day, {"inv": [], "max": 0.0})["inv"].append([cur["start"].strftime("%H:%M"), "N" if cur["sg"] > 0 else "S"])
    for day in out:
        out[day]["max"] = round(out[day]["max"], 1)
    for key in ("tirreno-merid-est", "ionio-settentr"):
        result[key]["messina"] = out


def cmems_cur(cm, result):
    """Corrente superficiale (media giornaliera) per area: velocita' media e 95 percentile in nodi, verso di provenienza del flusso medio.
    Il verso e' quello verso cui va l'acqua (non da dove viene, a differenza del vento); se il flusso medio e' debole
    rispetto alla velocita' tipica la corrente e' indicata come variabile."""
    start = dt.datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
    end = (dt.datetime.utcnow() + dt.timedelta(days=10)).strftime("%Y-%m-%dT23:59:59")
    ds = cm.open_dataset(dataset_id=DS_CUR, variables=["uo", "vo"], minimum_depth=0, maximum_depth=2,
                         start_datetime=start, end_datetime=end, **BBOX)
    u = ds["uo"].isel(depth=0).load()
    v = ds["vo"].isel(depth=0).load()
    lon, lat = u["longitude"].values, u["latitude"].values
    days = [str(t)[:10] for t in u["time"].values]
    print("correnti: forma", u.shape, "giorni", days[0], "->", days[-1])
    KN = 1.943844
    for key, name, poly in AREAS:
        m = area_mask(lon, lat, poly)
        out = {}
        for i, day in enumerate(days):
            uu, vv = u.values[i][m], v.values[i][m]
            ok = np.isfinite(uu) & np.isfinite(vv)
            if ok.sum() < 5:
                continue
            uu, vv = uu[ok], vv[ok]
            spd = np.hypot(uu, vv) * KN
            mu, mv = float(uu.mean()), float(vv.mean())
            mean_vec = math.hypot(mu, mv) * KN
            d = (math.degrees(math.atan2(mu, mv)) % 360) if mean_vec >= 0.25 * float(spd.mean()) and mean_vec >= 0.05 else None
            out[day] = {"v": round(float(spd.mean()), 2), "m": round(float(np.percentile(spd, 95)), 2), "d": None if d is None else round(d)}
        if out:
            result[key]["cur"] = out


def cmems_sst(cm, result):
    """Temperatura del mare prevista, ancorata al dato osservato NOAA.
    temperatura(giorno) = osservato(ultimo giorno NOAA) + [modello(giorno) - modello(ultimo giorno NOAA)]
    L'anomalia e' rispetto alla media giornaliera 1991-2020 NOAA dello stesso giorno dell'anno."""
    from fetch_meteo import sea_surface_temperature, _oisst_grid
    s = requests.Session()
    obs = sea_surface_temperature(s)                      # {area: {temp, clim, anom, date}} sull'ultimo giorno NOAA
    if not obs:
        raise RuntimeError("nessun dato NOAA")
    obs_day = next(iter(obs.values()))["date"]
    d0 = dt.date.fromisoformat(obs_day)
    end = dt.date.today() + dt.timedelta(days=10)
    ds = cm.open_dataset(dataset_id=DS_SST, variables=["thetao"], minimum_depth=0, maximum_depth=2,
                         start_datetime=d0.strftime("%Y-%m-%dT00:00:00"), end_datetime=end.strftime("%Y-%m-%dT23:59:59"), **BBOX)
    da = ds["thetao"].isel(depth=0).load()
    lon, lat = da["longitude"].values, da["latitude"].values
    days = [str(t)[:10] for t in da["time"].values]
    print("SST modello: forma", da.shape, "giorni", days[0], "->", days[-1], "| dato NOAA del", obs_day)
    # media NOAA 1991-2020 per ciascun giorno richiesto, sulla stessa finestra e griglia 0.25 gradi usate per le osservazioni
    i0, i1 = int((35.0 + 89.875) / 0.25), int((46.2 + 89.875) / 0.25) + 1
    j0, j1 = int((5.9 - 0.125) / 0.25), int((20.1 - 0.125) / 0.25) + 1
    LON, LAT = np.meshgrid(0.125 + 0.25 * np.arange(j0, j1 + 1), -89.875 + 0.25 * np.arange(i0, i1 + 1))
    clim_day = {}
    for day in days:
        if day < obs_day:
            continue
        dd = dt.date.fromisoformat(day)
        doy = min((dt.date(2001, dd.month, dd.day) - dt.date(2001, 1, 1)).days, 364)
        g = _oisst_grid(s, "sst.day.mean.ltm.1991-2020.nc", doy, i0, i1, j0, j1)
        clim_day[day] = {key: float(np.nanmean(g[in_polygon(LON, LAT, poly) & np.isfinite(g)])) for key, _, poly in AREAS
                         if (in_polygon(LON, LAT, poly) & np.isfinite(g)).sum() >= 5}
    for key, name, poly in AREAS:
        if key not in obs:
            continue
        m = area_mask(lon, lat, poly)
        series = {}
        for i, day in enumerate(days):
            v = da.values[i][m]
            v = v[np.isfinite(v)]
            if len(v) >= 5:
                series[day] = float(v.mean())
        if obs_day not in series:
            print("SST: giorno di ancoraggio assente per", key); continue
        base = series[obs_day]
        out = {}
        for day, v in series.items():
            if day < obs_day or day not in clim_day or key not in clim_day[day]:
                continue
            t = obs[key]["temp"] + (v - base)
            out[day] = {"t": round(t, 1), "a": round(t - clim_day[day][key], 1)}
        if out:
            result[key]["sst_fc"] = out
            result[key]["sst_base"] = obs_day


def cmems_grid(cm, result=None):
    """Campi su griglia (sottocampionata) per la mappa della pagina di dettaglio: correnti, temperatura del modello, pH.
    Scrive dettaglio_mare.json. Correnti in 0.01 nodi, temperatura in 0.1 C, pH come (pH-7.5)*1000, tutti interi."""
    NDAYS = 5
    t0 = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start, end = t0.strftime("%Y-%m-%dT00:00:00"), (t0 + dt.timedelta(days=NDAYS - 1)).strftime("%Y-%m-%dT23:59:59")
    KN = 1.943844
    out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "source": "Copernicus Marine (CMEMS) MED", "layers": {}}
    spec = (("cur", DS_CUR, ["uo", "vo"], 3), ("sst", DS_SST, ["thetao"], 4), ("ph", DS_PH, ["ph"], 4))
    for name, dsid, vars_, stride in spec:
        ds = cm.open_dataset(dataset_id=dsid, variables=vars_, minimum_depth=0, maximum_depth=2,
                             start_datetime=start, end_datetime=end, **BBOX)
        arrs = [ds[v].isel(depth=0).load() for v in vars_]
        lon, lat = arrs[0]["longitude"].values, arrs[0]["latitude"].values
        days = [str(t)[:10] for t in arrs[0]["time"].values][:NDAYS]
        vals = [a.values[:NDAYS, ::stride, ::stride] for a in arrs]
        lo, la = lon[::stride], lat[::stride]
        ok = np.isfinite(vals[0][0])
        for v in vals[1:]:
            ok &= np.isfinite(v[0])
        ii, jj = np.where(ok)
        layer = {"days": days, "pts": [[round(float(lo[j]), 2), round(float(la[i]), 2)] for i, j in zip(ii, jj)]}
        for k, v in enumerate(vals):
            rows = []
            for d in range(v.shape[0]):
                x = v[d][ii, jj]
                if name == "cur":
                    x = x * KN * 100
                elif name == "sst":
                    x = x * 10
                else:
                    x = (x - 7.5) * 1000
                rows.append([int(round(float(a))) if np.isfinite(a) else None for a in x])
            layer[("u", "v")[k] if name == "cur" else "z"] = rows
        layer["unit"] = {"cur": "0.01 kn", "sst": "0.1 C", "ph": "(pH-7.5)*1000"}[name]
        out["layers"][name] = layer
        print("griglia", name, len(layer["pts"]), "punti,", len(days), "giorni")
    path = os.path.join(OUT, "dettaglio_mare.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("Scritto", path, os.path.getsize(path), "byte")


def cmems_waves(cm, result=None):
    """Onde orarie (modello Copernicus MedWAM, ~4 km): per area altezza significativa media e massima, direzione media di provenienza e periodo di picco.
    Scrive dettaglio_onde.json: una riga all'ora a partire da t0, per 7 giorni."""
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - dt.timedelta(hours=3)
    end = now + dt.timedelta(hours=168)
    ds = cm.open_dataset(dataset_id=DS_WAV, variables=["VHM0", "VMDR", "VTPK"],
                         start_datetime=start.strftime("%Y-%m-%dT%H:00:00"), end_datetime=end.strftime("%Y-%m-%dT%H:00:00"), **BBOX)
    h = ds["VHM0"].load(); d = ds["VMDR"].load(); t = ds["VTPK"].load()
    lon, lat = h["longitude"].values, h["latitude"].values
    times = [str(x)[:16] + "Z" for x in h["time"].values]
    print("onde: forma", h.shape, times[0], "->", times[-1])
    out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "source": "Copernicus Marine (CMEMS) MED Waves (MedWAM)",
           "t0": times[0], "step_h": 1, "cols": ["hm0_mean_m", "hm0_max_m", "dir_from_deg", "tp_s"], "areas": {}}
    for key, name, poly in AREAS:
        m = area_mask(lon, lat, poly)
        hv = h.values[:, m]; dv = d.values[:, m]; tv = t.values[:, m]
        rows = []
        for i in range(hv.shape[0]):
            x = hv[i]; ok = np.isfinite(x)
            if ok.sum() < 5:
                rows.append(None); continue
            ang = np.radians(dv[i][ok]); w = x[ok]
            dr = (math.degrees(math.atan2(float((w * np.sin(ang)).sum()), float((w * np.cos(ang)).sum()))) % 360)
            tp = tv[i][ok]
            rows.append([round(float(x[ok].mean()), 2), round(float(x[ok].max()), 2), round(dr), round(float(np.nanmean(tp)), 1) if np.isfinite(tp).any() else None])
        out["areas"][key] = rows
    path = os.path.join(OUT, "dettaglio_onde.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("Scritto", path, os.path.getsize(path), "byte")


def cmems_level(cm, result):
    """Altezza del mare prevista (zos): per giorno locale, minimo e massimo dell'area rispetto alla media del periodo (cm)."""
    now = dt.datetime.utcnow()
    start = (now - dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:00:00")
    end = (now + dt.timedelta(days=9)).strftime("%Y-%m-%dT00:00:00")
    ds = cm.open_dataset(dataset_id=DS_SSH, variables=["zos"], start_datetime=start, end_datetime=end, **BBOX)
    da = ds["zos"].load()
    lon, lat = da["longitude"].values, da["latitude"].values
    times = da["time"].values
    print("zos: forma", da.shape, "ore", len(times))
    days = [local_day(t) for t in times]
    for key, name, poly in AREAS:
        m = area_mask(lon, lat, poly)
        series = np.array([np.nanmean(da.values[i][m]) if np.isfinite(da.values[i][m]).any() else np.nan
                           for i in range(len(times))])
        if not np.isfinite(series).sum() > 24:
            continue
        ref = np.nanmean(series)
        out = {}
        for d in sorted(set(days)):
            s = series[[i for i, x in enumerate(days) if x == d]]
            s = s[np.isfinite(s)]
            if len(s) >= 12:
                out[d] = {"min": round(float((s.min() - ref) * 100)), "max": round(float((s.max() - ref) * 100))}
        if out:
            result[key]["level_fc"] = out


def ioc_level(result):
    """Livello osservato: mediana degli ultimi 20 minuti rispetto alla mediana delle ultime 72 ore (cm).
    Il dato dei mareografi e' grezzo: si scartano le stazioni bloccate (variazione < 1 cm in 3 giorni, come un sensore fermo)
    o con valori incoerenti, e le singole letture anomale."""
    now = dt.datetime.utcnow()
    s = requests.Session()
    for key, (code, name) in STATIONS.items():
        try:
            r = s.get("https://www.ioc-sealevelmonitoring.org/service.php", timeout=90, params={
                "query": "data", "code": code, "format": "json",
                "timestart": (now - dt.timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S"),
                "timestop": now.strftime("%Y-%m-%dT%H:%M:%S")})
            r.raise_for_status()
            rows = sorted((dt.datetime.strptime(x["stime"][:19], "%Y-%m-%d %H:%M:%S"), float(x["slevel"])) for x in r.json()
                          if x.get("slevel") is not None)
            if len(rows) < 500:
                print("mareografo", code, "pochi dati:", len(rows)); continue
            ref = statistics.median(v for _, v in rows)
            rows = [(t, v) for t, v in rows if abs(v - ref) < 1.0]               # letture anomale isolate
            vals = [v for _, v in rows]
            spread = np.percentile(vals, 95) - np.percentile(vals, 5)
            if spread < 0.03:                                                      # sensore bloccato: nessuna marea visibile
                print("mareografo", code, "scartato: sensore apparentemente fermo, escursione", round(spread * 100, 1), "cm"); continue
            recent = [v for t, v in rows if t >= rows[-1][0] - dt.timedelta(minutes=20)]
            last_t = rows[-1][0]
            if (now - last_t) > dt.timedelta(hours=3) or len(recent) < 3:
                print("mareografo", code, "dato non recente"); continue
            cm = round((statistics.median(recent) - statistics.median(vals)) * 100)
            if abs(cm) > 80:                                                       # oltre ogni sovralzo plausibile: probabile guasto
                print("mareografo", code, "scartato: valore implausibile", cm, "cm"); continue
            result[key]["level_obs"] = {"station": name, "cm": cm, "time": last_t.strftime("%Y-%m-%dT%H:%MZ")}
        except Exception as e:
            print("mareografo", code, "non disponibile:", e)


def main():
    result = {key: {"id": key} for key, _, _ in AREAS}
    try:
        import copernicusmarine as cm
        for fn in (cmems_ph, cmems_sst, cmems_cur, cmems_messina, cmems_grid, cmems_waves):                 # livello del mare (cmems_level, ioc_level) disattivato: non mostrato nel pannello
            try:
                fn(cm, result)
            except Exception as e:
                print(fn.__name__, "FALLITO:", repr(e)[:400])
    except Exception as e:
        print("Copernicus Marine non disponibile:", repr(e)[:400])
    doc = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
           "source": "Copernicus Marine (CMEMS) MED; ISPRA/IOC mareografi", "areas": list(result.values())}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "ambiente.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    n = {k: sum(1 for a in result.values() if k in a) for k in ("ph", "ph_z", "ph_pct", "sst_fc", "cur")}
    print("Scritto", path, os.path.getsize(path), "byte; aree con dati:", n)
    if not any(n.values()):
        raise SystemExit("Nessun dato ambientale disponibile")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scarica le previsioni ECMWF Open Data (CC-BY-4.0) e produce meteo.json
con vento, raffiche, precipitazioni e stato del mare per le 15 aree dei bollettini.

Uso: python fetch_meteo.py [cartella_output]
"""
import json, math, os, sys, tempfile, datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import eccodes

BASE = "https://data.ecmwf.int/forecasts"
STEPS = list(range(0, 49, 3))          # 0..48 h, ogni 3 h
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# Aree (nome, lon_min, lon_max, lat_min, lat_max). Solo punti di mare.
# Confini approssimati dalla cartina: da affinare con le coordinate ufficiali.
AREAS = [
    ("mar-ligure",          "Mar Ligure",                   6.0, 10.5, 43.0, 44.5),
    ("mar-di-corsica",      "Mar di Corsica",               6.0,  9.7, 41.3, 43.0),
    ("tirreno-settentr",    "Tirreno Settentrionale",       9.7, 12.5, 41.9, 43.0),
    ("tirreno-centr-ovest", "Tirreno Centrale – Ovest",     9.7, 11.6, 40.0, 41.9),
    ("tirreno-centr-est",   "Tirreno Centrale – Est",      11.6, 15.0, 40.0, 41.9),
    ("tirreno-merid-ovest", "Tirreno Meridionale – Ovest",  9.7, 12.5, 37.9, 40.0),
    ("tirreno-merid-est",   "Tirreno Meridionale – Est",   12.5, 16.1, 37.9, 40.0),
    ("mar-di-sardegna",     "Mar di Sardegna",              6.0,  9.7, 39.0, 41.3),
    ("canale-di-sardegna",  "Canale di Sardegna",           6.0, 10.2, 35.0, 39.0),
    ("stretto-di-sicilia",  "Stretto di Sicilia",          10.2, 15.0, 35.0, 37.9),
    ("ionio-settentr",      "Ionio Settentrionale",        15.5, 20.0, 37.8, 39.8),
    ("ionio-merid",         "Ionio Meridionale",           15.0, 20.0, 35.0, 37.8),
    ("adriatico-settentr",  "Adriatico Settentrionale",    12.0, 16.0, 43.9, 46.0),
    ("adriatico-centr",     "Adriatico Centrale",          13.0, 19.0, 41.8, 43.9),
    ("adriatico-merid",     "Adriatico Meridionale",       15.5, 20.0, 39.8, 41.8),
]

DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_IT = ["Tramontana", "Grecale", "Levante", "Scirocco", "Ostro", "Libeccio", "Ponente", "Maestrale"]
DOUGLAS = [(0.1, 0, "Calmo"), (0.5, 1, "Quasi calmo"), (1.25, 2, "Poco mosso"), (2.5, 3, "Mosso"),
           (4.0, 4, "Molto mosso"), (6.0, 5, "Agitato"), (9.0, 6, "Molto agitato"),
           (14.0, 7, "Grosso"), (1e9, 8, "Molto grosso")]


def beaufort(kn):
    lim = [1, 4, 7, 11, 17, 22, 28, 34, 41, 48, 56, 64]
    return sum(kn >= x for x in lim)


def compass(deg):
    return DIRS[int((deg + 22.5) // 45) % 8]


def douglas(h):
    for lim, g, name in DOUGLAS:
        if h < lim:
            return g, name


def find_run(session):
    now = dt.datetime.utcnow()
    for back in range(0, 48, 6):
        t = (now - dt.timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        t = t.replace(hour=t.hour - t.hour % 6)
        d, h = t.strftime("%Y%m%d"), t.strftime("%H")
        url = f"{BASE}/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-{STEPS[-1]}h-oper-fc.index"
        if session.get(url, timeout=30).status_code == 200:
            return t
    raise SystemExit("Nessun run ECMWF disponibile")


def fetch_fields(session, run, step, stream, params):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    kind = "oper" if stream == "oper" else "wave"
    root = f"{BASE}/{d}/{h}z/ifs/0p25/{kind}/{d}{h}0000-{step}h-{kind}-fc"
    idx = session.get(root + ".index", timeout=60).text.splitlines()
    out = {}
    for line in idx:
        j = json.loads(line)
        if j["param"] in params and j.get("levtype") == "sfc":
            r = session.get(root + ".grib2", timeout=120,
                            headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
                f.write(r.content); f.flush()
                with open(f.name, "rb") as fh:
                    gid = eccodes.codes_grib_new_from_file(fh)
                    eccodes.codes_set(gid, "missingValue", 1e20)
                    out[j["param"]] = (
                        eccodes.codes_get_array(gid, "values"),
                        eccodes.codes_get_array(gid, "latitudes"),
                        eccodes.codes_get_array(gid, "longitudes"),
                    )
                    eccodes.codes_release(gid)
    return out


def main():
    s = requests.Session()
    run = find_run(s)
    print("Run ECMWF:", run)

    def job(step):
        a = fetch_fields(s, run, step, "oper", {"10u", "10v", "10fg", "tp"})
        b = fetch_fields(s, run, step, "wave", {"swh", "mwd", "mwp"})
        a.update(b)
        return step, a

    with ThreadPoolExecutor(6) as ex:
        data = dict(ex.map(job, STEPS))

    _, lats, lons = data[0]["10u"]
    lons = np.where(lons > 180, lons - 360, lons)
    sea = data[STEPS[0]]["swh"][0] < 1e10  # punti mare dove le onde sono definite
    # il campo onde può avere una griglia diversa dall'atmosfera: lo usiamo come riferimento
    wvals, wlats, wlons = data[0]["swh"]
    wlons = np.where(wlons > 180, wlons - 360, wlons)
    sea_w = wvals < 1e10

    areas_idx = {}
    for key, name, lo0, lo1, la0, la1 in AREAS:
        m = (lons >= lo0) & (lons <= lo1) & (lats >= la0) & (lats <= la1)
        # maschera mare: cerca per ogni punto atmosferico il punto onde più vicino (stessa griglia 0.25°)
        if len(wvals) == len(lons):
            m &= sea_w
        areas_idx[key] = np.where(m)[0]

    local = dt.timezone(dt.timedelta(hours=2))  # CEST fino a fine ottobre; l'app web usa ore locali
    times = [run + dt.timedelta(hours=st) for st in STEPS]

    result = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
              "run": run.strftime("%Y-%m-%dT%H:%MZ"),
              "source": "ECMWF Open Data (CC-BY-4.0)", "areas": []}

    for key, name, *_ in AREAS:
        ix = areas_idx[key]
        series = []
        for st, t in zip(STEPS, times):
            f = data[st]
            u = f["10u"][0][ix]; v = f["10v"][0][ix]
            spd = np.hypot(u, v) * 1.943844  # m/s -> nodi
            gust = f["10fg"][0][ix] * 1.943844 if "10fg" in f else spd
            swh = f["swh"][0][ix]; swh = np.where(swh < 1e10, swh, np.nan)
            mwd = f["mwd"][0][ix]
            tp = f["tp"][0][ix] * 1000.0  # mm cumulati dall'inizio del run
            wdir = (270 - np.degrees(np.arctan2(v, u))) % 360  # direzione di provenienza
            series.append({
                "t": t.strftime("%Y-%m-%dT%H:%MZ"),
                "wind_kn": float(np.mean(spd)),
                "wind_u": float(np.mean(u)), "wind_v": float(np.mean(v)),
                "gust_kn": float(np.nanmax(gust)),
                "wave_m": float(np.nanmean(swh)) if np.isfinite(swh).any() else None,
                "wave_max_m": float(np.nanmax(swh)) if np.isfinite(swh).any() else None,
                "wave_dir": float(np.nanmean(mwd)) if len(mwd) else None,
                "tp_cum_mm": float(np.mean(tp)),
                "tp_cum_max_mm": float(np.max(tp)),
                "n": int(len(ix)),
            })
        result["areas"].append({"id": key, "name": name, "series": series})

    # sintesi per giorno (giorno locale Europe/Rome) sull'orizzonte disponibile
    for a in result["areas"]:
        days = {}
        ser = a["series"]
        for i, p in enumerate(ser):
            lt = dt.datetime.strptime(p["t"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
            days.setdefault(lt.astimezone(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d"), []).append(i)
        a["days"] = []
        for day, idxs in days.items():
            if len(idxs) < 3:
                continue
            pts = [ser[i] for i in idxs]
            u = np.mean([p["wind_u"] for p in pts]); v = np.mean([p["wind_v"] for p in pts])
            mean_kn = float(np.mean([p["wind_kn"] for p in pts]))
            wdir = float((270 - math.degrees(math.atan2(v, u))) % 360)
            gust = max(p["gust_kn"] for p in pts)
            wave = max(p["wave_max_m"] or 0 for p in pts)
            wave_mean = float(np.mean([p["wave_m"] or 0 for p in pts]))
            first, last = idxs[0], idxs[-1]
            before = ser[first - 1]["tp_cum_mm"] if first > 0 else 0.0
            before_max = ser[first - 1]["tp_cum_max_mm"] if first > 0 else 0.0
            rain = max(0.0, ser[last]["tp_cum_mm"] - before)
            rain_max = max(0.0, ser[last]["tp_cum_max_mm"] - before_max)
            dg, dname = douglas(wave_mean)
            alerts = []
            if gust >= 41 or mean_kn >= 28:
                alerts.append({"level": "rosso", "text": "Burrasca forte: raffiche da Beaufort 9"})
            elif gust >= 34 or mean_kn >= 22:
                alerts.append({"level": "arancio", "text": "Burrasca: raffiche da Beaufort 8"})
            elif gust >= 27 or mean_kn >= 17:
                alerts.append({"level": "giallo", "text": "Vento sostenuto, raffiche oltre 27 nodi"})
            if wave >= 4.0:
                alerts.append({"level": "rosso", "text": f"Mare molto agitato, onde fino a {wave:.1f} m"})
            elif wave >= 2.5:
                alerts.append({"level": "arancio", "text": f"Mare agitato, onde fino a {wave:.1f} m"})
            if rain_max >= 40:
                alerts.append({"level": "arancio", "text": "Precipitazioni intense possibili"})
            elif rain_max >= 20:
                alerts.append({"level": "giallo", "text": "Rovesci localmente forti"})
            a["days"].append({
                "date": day,
                "wind_mean_kn": round(mean_kn), "wind_gust_kn": round(gust),
                "wind_bft": beaufort(mean_kn), "wind_dir_deg": round(wdir),
                "wind_dir": compass(wdir), "wind_name": DIR_IT[int((wdir + 22.5) // 45) % 8],
                "wave_mean_m": round(wave_mean, 1), "wave_max_m": round(wave, 1),
                "douglas": dg, "sea": dname,
                "rain_mm": round(rain, 1), "rain_max_mm": round(rain_max, 1),
                "alerts": alerts,
            })
        # le serie dettagliate non servono al sito
        a["series_points"] = len(ser)
        del a["series"]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "meteo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print("Scritto", path, os.path.getsize(path), "byte")


if __name__ == "__main__":
    main()

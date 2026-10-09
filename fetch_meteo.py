#!/usr/bin/env python3
"""Scarica le previsioni ECMWF Open Data (CC-BY-4.0) e produce meteo.json
con vento, raffiche, precipitazioni e stato del mare per le 15 aree dei bollettini.

Uso: python fetch_meteo.py [cartella_output]
"""
import json, math, os, re, sys, tempfile, time, datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import eccodes

BASE = "https://data.ecmwf.int/forecasts"
STEPS = list(range(0, 73, 3)) + list(range(78, 169, 6))   # 0-72 h ogni 3 h, poi ogni 6 h fino a 7 giorni
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# Aree dei bollettini Meteomar come poligoni (lon, lat). I limiti sono stati ricavati dalla cartina
# "Limiti dei mari nei bollettini meteo" georeferenziata sui gradi stampati in cornice (errore ~0,1-0,2 gradi).
# Il mare si seleziona dai punti in cui il modello onde e' definito, quindi i poligoni possono attraversare la terra.
AREAS = [
    ("mar-ligure",          "Mar Ligure",                      [(6.0, 43.0), (10.7, 43.0), (10.7, 46.0), (6.0, 46.0)]),
    ("mar-di-corsica",      "Mar di Corsica",                  [(6.0, 41.3), (9.6, 41.3), (9.6, 43.0), (6.0, 43.0)]),
    ("tirreno-settentr",    "Tirreno Settentrionale",          [(9.6, 42.1), (12.2, 42.1), (12.2, 43.0), (9.6, 43.0)]),
    ("tirreno-centr-ovest", "Tirreno Centrale – Settore Ovest", [(9.6, 40.0), (11.8, 40.0), (10.8, 42.1), (9.6, 42.1)]),
    ("tirreno-centr-est",   "Tirreno Centrale – Settore Est",  [(11.8, 40.0), (10.8, 42.1), (12.5, 42.1), (16.0, 40.0)]),
    ("tirreno-merid-ovest", "Tirreno Meridionale – Settore Ovest", [(9.6, 40.0), (12.3, 40.0), (12.3, 38.1), (10.0, 38.0), (10.0, 39.0), (9.6, 39.0)]),
    ("tirreno-merid-est",   "Tirreno Meridionale – Settore Est",   [(12.3, 40.0), (16.0, 40.0), (16.0, 38.1), (12.3, 38.1)]),
    ("mar-di-sardegna",     "Mar di Sardegna",                 [(6.0, 39.0), (9.6, 39.0), (9.6, 41.3), (6.0, 41.3)]),
    ("canale-di-sardegna",  "Canale di Sardegna",              [(6.0, 35.0), (6.0, 39.0), (10.0, 39.0), (10.0, 38.0), (10.6, 37.2), (10.6, 35.0)]),
    ("stretto-di-sicilia",  "Stretto di Sicilia",              [(10.0, 38.0), (15.0, 38.25), (15.0, 35.0), (10.6, 35.0), (10.6, 37.2)]),
    ("ionio-settentr",      "Ionio Settentrionale",            [(15.7, 38.0), (20.0, 38.0), (20.0, 39.8), (18.35, 39.8), (18.35, 40.6), (16.0, 40.6), (15.7, 40.0)]),
    ("ionio-merid",         "Ionio Meridionale",               [(15.0, 35.0), (20.0, 35.0), (20.0, 38.0), (15.0, 38.0)]),
    ("adriatico-settentr",  "Adriatico Settentrionale",        [(12.0, 44.0), (16.0, 44.0), (16.0, 46.0), (12.0, 46.0)]),
    ("adriatico-centr",     "Adriatico Centrale",              [(12.0, 42.0), (20.0, 42.0), (20.0, 44.0), (12.0, 44.0)]),
    ("adriatico-merid",     "Adriatico Meridionale",           [(15.0, 42.0), (20.0, 42.0), (20.0, 39.8), (18.35, 39.8), (16.9, 41.2)]),
]


def in_polygon(lon, lat, poly):
    """Ray casting vettorizzato: True per i punti dentro il poligono."""
    inside = np.zeros(lon.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        cross = ((y1 > lat) != (y2 > lat)) & (lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cross
    return inside


DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_IT = ["Tramontana", "Grecale", "Levante", "Scirocco", "Ostro", "Libeccio", "Ponente", "Maestrale"]
# Scala Douglas: (limite superiore altezza onda m, grado, descrizione)
DOUGLAS = [(0.05, 0, "Calmo"), (0.1, 1, "Quasi calmo"), (0.5, 2, "Poco mosso"), (1.25, 3, "Mosso"),
           (2.5, 4, "Molto mosso"), (4.0, 5, "Agitato"), (6.0, 6, "Molto agitato"),
           (9.0, 7, "Grosso"), (14.0, 8, "Molto grosso"), (1e9, 9, "Tempestoso")]


def beaufort(kn):
    kn = math.floor(kn + 0.5)      # la scala si applica ai nodi interi mostrati (es. 10,6 -> 11 -> forza 4)
    lim = [1, 4, 7, 11, 17, 22, 28, 34, 41, 48, 56, 64]
    return sum(kn >= x for x in lim)


def compass(deg):
    return DIRS[int((deg + 22.5) // 45) % 8]


def douglas(h):
    for lim, g, name in DOUGLAS:
        if h < lim:
            return g, name


def get(session, url, tries=8, **kw):
    """GET con attese crescenti se ECMWF risponde 429 (troppe richieste) o 5xx."""
    for n in range(tries):
        r = session.get(url, **kw)
        if r.status_code not in (429, 500, 502, 503, 504):
            return r
        wait = float(r.headers.get("Retry-After", 0) or 0) or min(2 ** n, 30)
        time.sleep(wait)
    r.raise_for_status()
    return r


def candidate_runs(session):
    """Run ECMWF disponibili, dal piu' recente: se l'ultimo e' ancora incompleto si ripiega sul precedente."""
    now = dt.datetime.utcnow()
    found = []
    for back in range(0, 48, 6):
        t = (now - dt.timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        t = t.replace(hour=t.hour - t.hour % 6)
        d, h = t.strftime("%Y%m%d"), t.strftime("%H")
        url = f"{BASE}/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-{STEPS[-1]}h-oper-fc.index"
        if get(session, url, timeout=30).status_code == 200:
            found.append(t)
        if len(found) == 3:
            break
    if not found:
        raise SystemExit("Nessun run ECMWF disponibile")
    return found


def fetch_fields(session, run, step, stream, params):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    kind = "oper" if stream == "oper" else "wave"
    root = f"{BASE}/{d}/{h}z/ifs/0p25/{kind}/{d}{h}0000-{step}h-{kind}-fc"
    idx = get(session, root + ".index", timeout=60).text.splitlines()
    out = {}
    for line in idx:
        j = json.loads(line)
        if j["param"] in params and j.get("levtype") == "sfc":
            r = get(session, root + ".grib2", timeout=120,
                    headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
                f.write(r.content); f.flush()
                with open(f.name, "rb") as fh:
                    gid = eccodes.codes_grib_new_from_file(fh)
                    eccodes.codes_set(gid, "missingValue", 1e20)
                    vals = eccodes.codes_get_array(gid, "values")
                    la = eccodes.codes_get_array(gid, "latitudes")
                    lo = eccodes.codes_get_array(gid, "longitudes")
                    keep = (la >= 34.5) & (la <= 46.5) & (lo >= 5.5) & (lo <= 20.5)   # solo Mediterraneo: meno memoria
                    out[j["param"]] = (vals[keep], la[keep], lo[keep])
                    eccodes.codes_release(gid)
    return out


OISST = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"


def _oisst_grid(session, nc, t, i0, i1, j0, j1):
    """Scarica una finestra della griglia OISST (0.25 gradi) via OPeNDAP in formato ascii."""
    r = session.get(f"{OISST}/{nc}.ascii?sst[{t}:{t}][{i0}:{i1}][{j0}:{j1}]", timeout=90)
    r.raise_for_status()
    rows = []
    for line in r.text.splitlines():
        m = re.match(r"\[0\]\[(\d+)\],\s*(.*)", line)
        if m:
            rows.append([float(x) for x in m.group(2).split(",")])
    a = np.array(rows)
    a[(a < -100) | (a > 100)] = np.nan
    return a


def sea_surface_temperature(session):
    """Temperatura superficiale del mare per area: ultimo dato osservato NOAA OISST (ritardo 1-3 giorni)
    e anomalia rispetto alla media giornaliera 1991-2020 dello stesso dataset."""
    i0, i1 = int((35.0 + 89.875) / 0.25), int((46.2 + 89.875) / 0.25) + 1
    j0, j1 = int((5.9 - 0.125) / 0.25), int((20.1 - 0.125) / 0.25) + 1
    lats = -89.875 + 0.25 * np.arange(i0, i1 + 1)
    lons = 0.125 + 0.25 * np.arange(j0, j1 + 1)
    LON, LAT = np.meshgrid(lons, lats)
    year = dt.datetime.utcnow().year
    nc = f"sst.day.mean.{year}.nc"
    n = int(re.search(r"time = (\d+)\]", session.get(f"{OISST}/{nc}.dds", timeout=60).text).group(1))
    obs, day = None, None
    for back in range(0, 6):                       # l'ultimo giorno puo' essere ancora vuoto
        o = _oisst_grid(session, nc, n - 1 - back, i0, i1, j0, j1)
        if np.isfinite(o).sum() > 500:
            obs, day = o, dt.date(year, 1, 1) + dt.timedelta(days=n - 1 - back)
            break
    if obs is None:
        raise RuntimeError("OISST senza dati recenti")
    doy = min((dt.date(2001, day.month, day.day) - dt.date(2001, 1, 1)).days, 364)
    clim = _oisst_grid(session, "sst.day.mean.ltm.1991-2020.nc", doy, i0, i1, j0, j1)
    out = {}
    for key, name, poly in AREAS:
        m = in_polygon(LON, LAT, poly) & np.isfinite(obs) & np.isfinite(clim)
        if m.sum() >= 5:
            out[key] = {"temp": round(float(obs[m].mean()), 1), "clim": round(float(clim[m].mean()), 1),
                        "anom": round(float((obs[m] - clim[m]).mean()), 1), "date": day.isoformat()}
    return out


def main():
    s = requests.Session()
    data = None
    for run in candidate_runs(s):
        print("Run ECMWF:", run)

        def job(step, run=run):
            a = fetch_fields(s, run, step, "oper", {"10u", "10v", "10fg", "tp", "msl"})
            b = fetch_fields(s, run, step, "wave", {"swh", "mwd", "mwp"})
            a.update(b)
            return step, a

        try:
            with ThreadPoolExecutor(3) as ex:
                data = dict(ex.map(job, STEPS))
            break
        except Exception as e:      # run ancora in pubblicazione: si prova quello precedente
            print("Run incompleto, riprovo con il precedente:", e)
    if data is None:
        raise SystemExit("Nessun run ECMWF completo")

    _, lats, lons = data[0]["10u"]
    lons = np.where(lons > 180, lons - 360, lons)
    sea = data[STEPS[0]]["swh"][0] < 1e10  # punti mare dove le onde sono definite
    # il campo onde può avere una griglia diversa dall'atmosfera: lo usiamo come riferimento
    wvals, wlats, wlons = data[0]["swh"]
    wlons = np.where(wlons > 180, wlons - 360, wlons)
    sea_w = wvals < 1e10

    areas_idx = {}
    for key, name, poly in AREAS:
        m = in_polygon(lons, lats, poly)
        if len(wvals) == len(lons):
            m &= sea_w
        areas_idx[key] = np.where(m)[0]

    local = dt.timezone(dt.timedelta(hours=2))  # CEST fino a fine ottobre; l'app web usa ore locali
    times = [run + dt.timedelta(hours=st) for st in STEPS]

    result = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
              "run": run.strftime("%Y-%m-%dT%H:%MZ"),
              "source": "ECMWF Open Data (CC-BY-4.0)", "areas": []}

    try:
        sst = sea_surface_temperature(s)
    except Exception as e:              # il resto del pannello funziona anche senza temperatura del mare
        print("SST non disponibile:", e)
        sst = {}

    for key, name, _poly in AREAS:
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
                "gust_p90_kn": float(np.nanpercentile(gust, 90)),
                "p_hpa": float(np.mean(f["msl"][0][ix]) / 100.0),
                "wave_m": float(np.nanmean(swh)) if np.isfinite(swh).any() else None,
                "wave_max_m": float(np.nanmax(swh)) if np.isfinite(swh).any() else None,
                "wave_dir": float(np.nanmean(mwd)) if len(mwd) else None,
                "tp_cum_mm": float(np.mean(tp)),
                "tp_cum_max_mm": float(np.max(tp)),
                "n": int(len(ix)),
            })
        entry = {"id": key, "name": name, "series": series, "pressure_now_hpa": round(series[0]["p_hpa"], 1)}
        if key in sst:
            entry["sst"] = sst[key]
        result["areas"].append(entry)

    # dettaglio per la pagina "approfondisci": serie per area (colonne in "cols") e campo di vento sui punti di mare
    detail = {"generated": result["generated"], "run": result["run"], "source": result["source"],
              "cols": ["t", "wind_kn", "wind_dir", "gust_kn", "wave_m", "wave_max_m", "p_hpa", "rain_mm", "wave_dir"], "areas": []}
    gm = sea_w.copy() if len(wvals) == len(lons) else np.ones(len(lons), bool)
    gm &= (lats >= 34.5) & (lats <= 46.5) & (lons >= 5.5) & (lons <= 20.5)
    gix = np.where(gm)[0]
    wsteps = [st for st in STEPS if st <= 72]
    detail["wind"] = {"pts": [[round(float(lons[i]), 2), round(float(lats[i]), 2)] for i in gix],
                      "t": [(run + dt.timedelta(hours=st)).strftime("%Y-%m-%dT%H:%MZ") for st in wsteps],
                      "u": [[int(round(float(data[st]["10u"][0][i]) * 19.43844)) for i in gix] for st in wsteps],
                      "v": [[int(round(float(data[st]["10v"][0][i]) * 19.43844)) for i in gix] for st in wsteps],
                      "unit": "0.1 kn"}

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
            peak_kn = max(p["wind_kn"] for p in pts)          # vento sostenuto massimo (media d'area su 3 h)
            gust = max(p["gust_p90_kn"] for p in pts)         # raffica rappresentativa: 90 percentile dei punti dell'area
            wave = max(p["wave_max_m"] or 0 for p in pts)
            wave_mean = float(np.mean([p["wave_m"] or 0 for p in pts]))
            first, last = idxs[0], idxs[-1]
            before = ser[first - 1]["tp_cum_mm"] if first > 0 else 0.0
            before_max = ser[first - 1]["tp_cum_max_mm"] if first > 0 else 0.0
            rain = max(0.0, ser[last]["tp_cum_mm"] - before)
            rain_max = max(0.0, ser[last]["tp_cum_max_mm"] - before_max)
            p_vals = [p["p_hpa"] for p in pts]
            p_start = ser[first - 1]["p_hpa"] if first > 0 else ser[first]["p_hpa"]
            p_delta = ser[last]["p_hpa"] - p_start
            dg, dname = douglas(wave_mean)
            alerts = []
            # Le allerte di vento si basano sul vento SOSTENUTO (Beaufort), non su raffiche isolate di un punto
            if peak_kn >= 34:
                alerts.append({"level": "rosso", "text": "Burrasca forte: vento da forza 8"})
            elif peak_kn >= 28:
                alerts.append({"level": "arancio", "text": "Burrasca: vento da forza 7"})
            elif peak_kn >= 22 or gust >= 34:
                alerts.append({"level": "giallo", "text": "Vento forte, raffiche fino a %d nodi" % round(gust)})
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
                "pressure_mean_hpa": round(float(np.mean(p_vals))), "pressure_max_hpa": round(max(p_vals)),
                "pressure_min_hpa": round(min(p_vals)), "pressure_delta_hpa": round(p_delta, 1),
                "wind_mean_kn": round(mean_kn), "wind_peak_kn": round(peak_kn), "wind_gust_kn": round(gust),
                "wind_bft": beaufort(peak_kn), "wind_dir_deg": round(wdir),
                "wind_dir": compass(wdir), "wind_name": DIR_IT[int((wdir + 22.5) // 45) % 8],
                "wave_mean_m": round(wave_mean, 1), "wave_max_m": round(wave, 1),
                "douglas": dg, "sea": dname,
                "rain_mm": round(rain, 1), "rain_max_mm": round(rain_max, 1),
                "alerts": alerts,
            })
        # serie compatta per la pagina di dettaglio (dettaglio_meteo.json)
        det_rows = []
        for i, p in enumerate(ser):
            pw = ser[i - 1]["tp_cum_mm"] if i > 0 else 0.0
            wdr = (270 - math.degrees(math.atan2(p["wind_v"], p["wind_u"]))) % 360
            det_rows.append([p["t"], round(p["wind_kn"], 1), round(wdr), round(p["gust_p90_kn"]),
                             None if p["wave_m"] is None else round(p["wave_m"], 1),
                             None if p["wave_max_m"] is None else round(p["wave_max_m"], 1),
                             round(p["p_hpa"], 1), round(max(0.0, p["tp_cum_mm"] - pw), 1),
                             None if p["wave_dir"] is None else round(p["wave_dir"])])
        detail["areas"].append({"id": a["id"], "name": a["name"], "rows": det_rows})
        a["series_points"] = len(ser)
        del a["series"]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "meteo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print("Scritto", path, os.path.getsize(path), "byte")
    dpath = os.path.join(OUT, "dettaglio_meteo.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, separators=(",", ":"))
    print("Scritto", dpath, os.path.getsize(dpath), "byte")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controllo del vento: modello ECMWF (IFS 0,25 gradi) contro ICON-EU (DWD, 6,5 km), con le misure reali dei METAR di aeroporti costieri.

Ogni esecuzione (due volte al giorno, dopo i cicli delle 00 e delle 12 UTC):
  1. scarica i METAR delle ultime ore e li accumula in verifica/osservazioni.jsonl;
  2. scarica dai due modelli, per lo stesso ciclo, il vento a 10 m alle scadenze +6, +12, +24, +36, +48, +72 ore e salva il valore al punto
     piu' vicino a ciascun aeroporto in verifica/previsioni.jsonl;
  3. confronta ogni previsione il cui istante e' ormai passato con la misura piu' vicina (entro 30 minuti) e scrive verifica/riepilogo.json:
     errore medio assoluto, scarto medio (bias), scarto quadratico medio e errore di direzione, per modello e per scadenza,
     solo sulle coppie in cui sono disponibili entrambi i modelli (confronto alla pari).
Nessuna modifica al sito: serve solo a scegliere il modello piu' attendibile."""
import bz2, datetime as dt, json, math, os, sys, tempfile, traceback
import numpy as np
import requests
import eccodes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_meteo import get, candidate_runs, fetch_fields    # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
VDIR = os.path.join(OUT, "verifica")
os.makedirs(VDIR, exist_ok=True)
LOG = []
IDS = "LIMJ,LIRP,LIRF,LIRN,LIEO,LIEE,LICJ,LICD,LICC,LIBD,LIBP,LIPY,LIPQ,LIEA,LICT,LIBR,LIPZ,LICG,LICR"
LEADS = [6, 12, 24, 36, 48, 72]
KN = 1.943844
ICON = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


def read_jsonl(name):
    p = os.path.join(VDIR, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def append_jsonl(name, rows):
    with open(os.path.join(VDIR, name), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def wind_dir(u, v):
    return (270 - math.degrees(math.atan2(v, u))) % 360


# ---------------------------------------------------------------- 1) osservazioni METAR
def fetch_metar(session):
    r = session.get("https://aviationweather.gov/api/data/metar", params={"ids": IDS, "format": "json", "hours": 18}, timeout=90)
    r.raise_for_status()
    seen = {(o["s"], o["t"]) for o in read_jsonl("osservazioni.jsonl")}
    new, stations = [], {}
    for m in r.json():
        if m.get("wspd") is None or m.get("obsTime") is None:
            continue
        stations[m["icaoId"]] = (m["lat"], m["lon"])
        key = (m["icaoId"], int(m["obsTime"]))
        if key in seen:
            continue
        seen.add(key)
        d = m.get("wdir")
        new.append({"s": m["icaoId"], "t": int(m["obsTime"]), "d": d if isinstance(d, (int, float)) else None, "v": m["wspd"]})
    append_jsonl("osservazioni.jsonl", new)
    log("METAR: %d nuove osservazioni, %d stazioni" % (len(new), len(stations)))
    return stations


# ---------------------------------------------------------------- 2) previsioni dei due modelli
def nearest(vals, la, lo, lat, lon):
    d2 = (la - lat) ** 2 + ((lo - lon) * math.cos(math.radians(lat))) ** 2
    k = int(np.argmin(d2))
    return float(vals[k]), math.sqrt(float(d2[k]))


def ecmwf_forecasts(session, run, stations):
    out = []
    for lead in LEADS:
        f = fetch_fields(session, run, lead, "oper", {"10u", "10v"})
        if "10u" not in f or "10v" not in f:
            log("ECMWF: campi mancanti a +%d h" % lead); continue
        (uu, la, lo), (vv, _, _) = f["10u"], f["10v"]
        lo = np.where(lo > 180, lo - 360, lo)
        for s, (lat, lon) in stations.items():
            u, dist = nearest(uu, la, lo, lat, lon)
            if dist > 0.3:
                continue
            v, _ = nearest(vv, la, lo, lat, lon)
            out.append({"run": run.strftime("%Y-%m-%dT%H:%MZ"), "model": "ecmwf", "lead": lead, "target": int((run + dt.timedelta(hours=lead)).replace(tzinfo=dt.timezone.utc).timestamp()),
                        "s": s, "v": round(math.hypot(u, v) * KN, 1), "d": round(wind_dir(u, v))})
    log("ECMWF: %d previsioni" % len(out))
    return out


def icon_nearest(path, stations):
    with open(path, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
        res = {}
        for s, (lat, lon) in stations.items():
            n = eccodes.codes_grib_find_nearest(gid, lat, lon)[0]
            res[s] = (float(n["value"]), float(n["distance"]))
        eccodes.codes_release(gid)
    return res


def icon_fetch(session, run, lead, var):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    url = f"{ICON}/{h}/{var.lower()}/icon-eu_europe_regular-lat-lon_single-level_{d}{h}_{lead:03d}_{var}.grib2.bz2"
    r = get(session, url, timeout=120)
    if r.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
        f.write(bz2.decompress(r.content)); f.flush()
        return f.name, icon_nearest(f.name, STATIONS_FOR_ICON)


STATIONS_FOR_ICON = {}


def icon_forecasts(session, run, stations):
    global STATIONS_FOR_ICON
    STATIONS_FOR_ICON = stations
    out = []
    for lead in LEADS:
        try:
            cu = icon_fetch(session, run, lead, "U_10M"); cv = icon_fetch(session, run, lead, "V_10M")
        except Exception as e:
            log("ICON-EU +%d h: errore %s" % (lead, repr(e)[:150])); continue
        if cu is None or cv is None:
            log("ICON-EU +%d h: file non disponibili" % lead); continue
        for s in stations:
            (u, du), (v, dv) = cu[1][s], cv[1][s]
            if du > 30:                       # chilometri: punto troppo lontano
                continue
            out.append({"run": run.strftime("%Y-%m-%dT%H:%MZ"), "model": "icon", "lead": lead, "target": int((run + dt.timedelta(hours=lead)).replace(tzinfo=dt.timezone.utc).timestamp()),
                        "s": s, "v": round(math.hypot(u, v) * KN, 1), "d": round(wind_dir(u, v))})
    log("ICON-EU: %d previsioni" % len(out))
    return out


# ---------------------------------------------------------------- 3) confronto
def stats(errs, derrs):
    if not errs:
        return None
    a = np.array(errs)
    r = {"n": len(errs), "mae": round(float(np.mean(np.abs(a))), 2), "bias": round(float(np.mean(a)), 2), "rmse": round(float(np.sqrt(np.mean(a ** 2))), 2)}
    r["dir_mae"] = round(float(np.mean(np.abs(derrs))), 1) if derrs else None
    r["n_dir"] = len(derrs)
    return r


def summary():
    fc = read_jsonl("previsioni.jsonl")
    ob = {}
    for o in read_jsonl("osservazioni.jsonl"):
        ob.setdefault(o["s"], []).append(o)
    for s in ob:
        ob[s].sort(key=lambda x: x["t"])
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp()
    groups = {}
    for f in fc:
        groups.setdefault((f["run"], f["lead"], f["s"]), {})[f["model"]] = f
    pairs = []
    for key, g in groups.items():
        if "ecmwf" not in g or "icon" not in g:
            continue
        t = g["ecmwf"]["target"]
        if t > now - 1800 or key[2] not in ob:
            continue
        best = min(ob[key[2]], key=lambda o: abs(o["t"] - t))
        if abs(best["t"] - t) > 1800:
            continue
        pairs.append((key, g, best))

    def collect(sel):
        res = {}
        for m in ("ecmwf", "icon"):
            e, de = [], []
            for key, g, o in sel:
                e.append(g[m]["v"] - o["v"])
                if o["d"] is not None and o["v"] >= 5 and g[m]["v"] >= 5:
                    x = abs(g[m]["d"] - o["d"]) % 360
                    de.append(min(x, 360 - x))
            res[m] = stats(e, de)
        return res

    out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "n_previsioni": len(fc), "n_coppie": len(pairs),
           "cicli": sorted({k[0] for k, _, _ in pairs}), "stazioni": sorted({k[2] for k, _, _ in pairs}),
           "totale": collect(pairs), "per_scadenza": {}}
    for lead in LEADS:
        sel = [p for p in pairs if p[0][1] == lead]
        if sel:
            out["per_scadenza"][str(lead)] = collect(sel)
    json.dump(out, open(os.path.join(VDIR, "riepilogo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("RIEPILOGO: %d coppie su %d previsioni, %d cicli" % (len(pairs), len(fc), len(out["cicli"])))
    for lead, r in out["per_scadenza"].items():
        e, i = r["ecmwf"], r["icon"]
        if e and i:
            log("  +%2s h  n=%3d  ECMWF mae %.2f bias %+.2f | ICON mae %.2f bias %+.2f" % (lead, e["n"], e["mae"], e["bias"], i["mae"], i["bias"]))
    return out


def main():
    s = requests.Session()
    stations = fetch_metar(s)
    run = candidate_runs(s)[0]
    log("Ciclo ECMWF:", run)
    done = {(f["run"], f["model"], f["lead"], f["s"]) for f in read_jsonl("previsioni.jsonl")}
    new = []
    for fn, name in ((ecmwf_forecasts, "ecmwf"), (icon_forecasts, "icon")):
        try:
            rows = fn(s, run, stations)
        except Exception as e:
            log("ERRORE", name, repr(e)[:300]); traceback.print_exc(); continue
        new += [r for r in rows if (r["run"], r["model"], r["lead"], r["s"]) not in done]
    append_jsonl("previsioni.jsonl", new)
    log("Previsioni nuove salvate:", len(new))
    summary()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERRORE FATALE", repr(e)[:400]); traceback.print_exc()
    finally:
        with open(os.path.join(VDIR, "ultimo_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")

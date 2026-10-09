#!/usr/bin/env python3
"""Esplorazione una tantum del modello onde Copernicus (MEDSEA_ANALYSISFORECAST_WAV_006_017):
1) trova il dataset orario e le variabili disponibili;
2) calcola per le 15 aree altezza media/massima oraria nelle prossime 72 ore;
3) le confronta con le onde ECMWF gia' pubblicate in dettaglio_meteo.json.
Scrive onde_explore.json."""
import datetime as dt, json, os, sys
import numpy as np, requests
import copernicusmarine as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_meteo import AREAS, in_polygon   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BBOX = dict(minimum_longitude=5.5, maximum_longitude=20.5, minimum_latitude=34.8, maximum_latitude=46.3)
PRODUCT = "MEDSEA_ANALYSISFORECAST_WAV_006_017"
CANDIDATES = ["cmems_mod_med_wav_anfc_4.2km_PT1H-i", "cmems_mod_med_wav_anfc_4.2km_PT1H-i_202411"]
res = {"generated": dt.datetime.utcnow().isoformat()}

# 1) elenco dei dataset del prodotto
try:
    cat = cm.describe(product_id=PRODUCT)
    txt = cat.model_dump_json() if hasattr(cat, "model_dump_json") else str(cat)
    ids = sorted(set(__import__("re").findall(r"cmems_mod_med_wav[A-Za-z0-9_.\-]*", txt)))
    res["dataset_trovati"] = ids
    print("dataset onde:", ids)
    CANDIDATES = [i for i in ids if "PT1H" in i] + CANDIDATES
except Exception as e:
    res["errore_describe"] = repr(e)[:300]
    print("describe fallito:", repr(e)[:300])

ds = None
for did in CANDIDATES:
    try:
        ds = cm.open_dataset(dataset_id=did, **BBOX,
                             start_datetime=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:00:00"),
                             end_datetime=(dt.datetime.utcnow() + dt.timedelta(hours=72)).strftime("%Y-%m-%dT%H:00:00"))
        res["dataset"] = did
        break
    except Exception as e:
        print("non apribile", did, repr(e)[:200])
if ds is None:
    res["errore"] = "nessun dataset onde apribile"
    json.dump(res, open(os.path.join(OUT, "onde_explore.json"), "w"), ensure_ascii=False)
    raise SystemExit("nessun dataset")

res["variabili"] = {v: str(ds[v].attrs.get("long_name", "")) + " [" + str(ds[v].attrs.get("units", "")) + "]" for v in ds.data_vars}
res["dimensioni"] = {k: int(v) for k, v in ds.sizes.items()}
print("variabili:", res["variabili"])
vh = ds["VHM0"].load()
lon, lat = vh["longitude"].values, vh["latitude"].values
times = [str(t)[:16] for t in vh["time"].values]
res["tempi"] = [times[0], times[-1], len(times)]
LON, LAT = np.meshgrid(lon, lat)

# 3) confronto con ECMWF
try:
    det = requests.get("https://bolinasailmagazine-wq.github.io/bolina-meteo/dettaglio_meteo.json", timeout=60).json()
    ecm = {a["name"]: a["rows"] for a in det["areas"]}
except Exception as e:
    ecm = {}
    res["errore_ecmwf"] = repr(e)[:200]
res["aree"] = {}
tidx = {t: i for i, t in enumerate(times)}
for key, name, poly in AREAS:
    m = in_polygon(LON, LAT, poly)
    arr = vh.values[:, m]
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mean = np.nanmean(arr, axis=1)
    mx = np.nanmax(arr, axis=1)
    out = {"celle": int(m.sum()), "cmems_media_m": [round(float(x), 2) for x in mean[::6]], "cmems_max_m": [round(float(x), 2) for x in mx[::6]]}
    rows = ecm.get(name)
    if rows:
        pairs = []
        for r in rows[:25]:
            t = r[0][:13].replace("T", "T") + ":00"
            t = t[:16]
            if t in tidx and r[4] is not None:
                pairs.append((r[4], r[5], float(mean[tidx[t]]), float(mx[tidx[t]])))
        if len(pairs) >= 6:
            p = np.array(pairs)
            out["n_confronti"] = len(p)
            out["ecmwf_media_vs_cmems_media"] = {"mae": round(float(np.mean(np.abs(p[:, 0] - p[:, 2]))), 2), "bias_cmems_meno_ecmwf": round(float(np.mean(p[:, 2] - p[:, 0])), 2),
                                                  "corr": round(float(np.corrcoef(p[:, 0], p[:, 2])[0, 1]), 2) if p[:, 0].std() > 0 and p[:, 2].std() > 0 else None}
            out["ecmwf_max_vs_cmems_max"] = {"mae": round(float(np.mean(np.abs(p[:, 1] - p[:, 3]))), 2), "bias_cmems_meno_ecmwf": round(float(np.mean(p[:, 3] - p[:, 1])), 2)}
    res["aree"][name] = out
    print(name, out.get("celle"), out.get("ecmwf_media_vs_cmems_media"))

json.dump(res, open(os.path.join(OUT, "onde_explore.json"), "w"), ensure_ascii=False)
print("scritto onde_explore.json")

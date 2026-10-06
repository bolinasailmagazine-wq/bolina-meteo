#!/usr/bin/env python3
"""Esplorazione una tantum: climatologia del pH Copernicus (Mediterraneo) e confronto tra rianalisi e sistema di previsione.
Scrive ph_explore.json con: variabili/attributi della climatologia, valori medi per area e mese, e lo scarto
(previsione - rianalisi) nel periodo in cui i due prodotti coincidono."""
import datetime as dt, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_meteo import AREAS, in_polygon  # noqa: E402
import copernicusmarine as cm  # noqa: E402

BBOX = dict(minimum_longitude=5.5, maximum_longitude=20.5, minimum_latitude=34.8, maximum_latitude=46.3)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
CLIM = "cmems_mod_med_bgc_my_4.2km-climatology_P1M-m"
MY_M = "cmems_mod_med_bgc-car_my_4.2km_P1M-m"
AF_M = "cmems_mod_med_bgc-car_anfc_4.2km_P1M-m"
res = {"generated": dt.datetime.utcnow().isoformat()}


def surf(da):
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da


def area_means(da):
    lon, lat = da["longitude"].values, da["latitude"].values
    LON, LAT = np.meshgrid(lon, lat)
    out = {}
    for key, name, poly in AREAS:
        m = in_polygon(LON, LAT, poly)
        out[key] = [None if not np.isfinite(da.values[i][m]).any() else round(float(np.nanmean(da.values[i][m])), 4)
                    for i in range(da.shape[0])]
    return out


try:
    ds = cm.open_dataset(dataset_id=CLIM, minimum_depth=0, maximum_depth=2, **BBOX)
    res["clim_vars"] = {v: {k: str(a)[:200] for k, a in ds[v].attrs.items() if k in ("long_name", "units", "standard_name")}
                        for v in ds.data_vars}
    res["clim_dims"] = {k: int(v) for k, v in ds.sizes.items()}
    res["clim_attrs"] = {k: str(v)[:300] for k, v in ds.attrs.items() if k in ("title", "summary", "history", "comment", "time_coverage_start", "time_coverage_end")}
    res["clim_times"] = [str(t)[:10] for t in ds["time"].values]
    for v in ("ph_avg", "ph_std"):
        if v in ds:
            da = surf(ds[v]).load()
            res["clim_" + v] = area_means(da)
except Exception as e:
    res["clim_error"] = repr(e)[:500]
    print("CLIM FALLITO", repr(e)[:500])

try:
    my = cm.open_dataset(dataset_id=MY_M, variables=["ph"], minimum_depth=0, maximum_depth=2, **BBOX)
    af = cm.open_dataset(dataset_id=AF_M, variables=["ph"], minimum_depth=0, maximum_depth=2, **BBOX)
    res["my_time"] = [str(my["time"].values[0])[:10], str(my["time"].values[-1])[:10]]
    res["af_time"] = [str(af["time"].values[0])[:10], str(af["time"].values[-1])[:10]]
    t0 = max(my["time"].values[0], af["time"].values[0])
    t1 = min(my["time"].values[-1], af["time"].values[-1])
    res["overlap"] = [str(t0)[:10], str(t1)[:10]]
    if t0 <= t1:
        a = surf(my["ph"].sel(time=slice(t0, t1))).load()
        b = surf(af["ph"].sel(time=slice(t0, t1))).load()
        res["my_overlap"] = area_means(a)
        res["af_overlap"] = area_means(b)
        res["overlap_months"] = [str(t)[:7] for t in a["time"].values]
except Exception as e:
    res["overlap_error"] = repr(e)[:500]
    print("OVERLAP FALLITO", repr(e)[:500])

json.dump(res, open(os.path.join(OUT, "ph_explore.json"), "w"), ensure_ascii=False)
print("scritto ph_explore.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in res.items()})

# %%
# extract_rs41_geopot_from_mwx.py
import csv
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
try:
    from pygeodesy.geoids import GeoidKarney
    from pygeodesy.ellipsoidalVincenty import LatLon
except Exception:
    from pygeodesy import GeoidKarney
    from pygeodesy.ellipsoidalVincenty import LatLon

# %%
L0_DIRS = [
    Path("../level0/BCO"),
    Path("../level0/Meteor"),
]
GEOID_DIR = Path("./geoid")
GEOID_PGM = GEOID_DIR / "EGM96-15.pgm"
OUT_DIR = Path("./Vaisala_Geopotential_Height")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_ONE: Optional[str] = None  # e.g. "BCO_20240909_164839.mwx"

class EarthConstants:
    MajorAxis = 6378137.0
    Flattening = 1.0 / 298.257222101
    MinorAxis = MajorAxis * (1.0 - Flattening)
    E1Square = Flattening * (2.0 - Flattening)
    E2Square = ((MajorAxis ** 2) - (MinorAxis ** 2)) / (MinorAxis ** 2)
    EquatorGravity = 9.7803253359
    GravityConst = 0.00193185265241
    GravityAcc = 9.80665

def convert_to_geopotential(lat_deg: float, geometric_height_m: float) -> float:
    lat_rad = np.deg2rad(lat_deg)
    s2 = np.sin(lat_rad) ** 2
    temp = np.sqrt(1 - EarthConstants.E1Square * s2)
    g = EarthConstants.EquatorGravity * (1 + EarthConstants.GravityConst * s2) / temp
    R = EarthConstants.MajorAxis / temp
    return float(g / EarthConstants.GravityAcc * (R * geometric_height_m) / (R + geometric_height_m))

def require_geoid(pgm_path: Path) -> GeoidKarney:
    if not pgm_path.is_file():
        raise FileNotFoundError(f"Geoid file not found: {pgm_path}")
    return GeoidKarney(str(pgm_path))

def find_sync_xml(zf: zipfile.ZipFile) -> Optional[str]:
    for name in zf.namelist():
        if name.lower().endswith(".xml") and "synchronizedsoundingdata" in name.lower():
            return name
    for name in zf.namelist():
        if name.lower().endswith(".xml") and "sounding" in name.lower():
            return name
    return None

# --- add near the top (after imports) ---
MISSING_SENTINELS = {-32768.0, -32768, -9999.0, -9999, 9999.0, 9999, 999999.0, 999999}

def _f(val: Optional[str]) -> Optional[float]:
    """Parse float; return None for empty or known sentinels."""
    if val is None or val == "":
        return None
    try:
        x = float(val)
    except ValueError:
        return None
    if x in MISSING_SENTINELS:
        return None
    return x

def _valid_lat(x: float) -> bool:
    return x is not None and -90.0 <= x <= 90.0

def _valid_lon(x: float) -> bool:
    return x is not None and -180.0 <= x <= 180.0

def _valid_alt(x: float) -> bool:
    # allow slightly negative near sea level; cap absurd values
    return x is not None and -2000.0 <= x <= 100000.0


def parse_rows(xml_root: ET.Element) -> Dict[str, List]:
    lats, lons, alt_wgs = [], [], []
    h_ptu, h_geom_xml, times, dropping = [], [], [], []
    pres, temp, rh, wspd = [], [], [], []

    skipped = 0
    for row in xml_root.iter("Row"):
        lat = _f(row.get("Latitude"))
        lon = _f(row.get("Longitude"))
        alt = _f(row.get("Altitude"))  # WGS84 ellipsoidal
        if not (_valid_lat(lat) and _valid_lon(lon) and _valid_alt(alt)):
            skipped += 1
            continue

        lats.append(lat); lons.append(lon); alt_wgs.append(alt)
        h_ptu.append(_f(row.get("Height")))
        h_geom_xml.append(_f(row.get("GeometricHeight")))
        times.append(row.get("DataSrvTime") or row.get("UtcTime") or row.get("RadioRxTimePk") or "")
        dropping.append(int(_f(row.get("Dropping")) or 0))  # 0 ascent, 1 descent
        pres.append(_f(row.get("Pressure")))
        temp.append(_f(row.get("Temperature")))
        rh.append(_f(row.get("Humidity")))
        wspd.append(_f(row.get("WindSpeed")))

    if not lats:
        raise ValueError("No (Latitude, Longitude, Altitude) rows found after filtering invalid values.")
    if skipped:
        print(f"  parse_rows: skipped {skipped} row(s) with invalid/missing lat/lon/alt")

    return {
        "lat": lats, "lon": lons, "alt_wgs84": alt_wgs,
        "height_ptu": h_ptu, "height_geom_xml": h_geom_xml,
        "time": times, "dropping": dropping,
        "pressure": pres, "temperature": temp, "rel_humidity": rh, "wind_speed": wspd,
    }


def process_one_mwx(mwx_path: Path, geoid: GeoidKarney) -> Dict[str, Path]:
    with zipfile.ZipFile(mwx_path, "r") as zf:
        xml_name = find_sync_xml(zf)
        if xml_name is None:
            raise FileNotFoundError(f"No SynchronizedSoundingData XML in {mwx_path.name}")
        with zf.open(xml_name) as f:
            root = ET.parse(f).getroot()

    data = parse_rows(root)

    geoid_m, geom_m, geopot_m = [], [], []
    for lat, lon, alt_wgs in zip(data["lat"], data["lon"], data["alt_wgs84"]):
        try:
            und = geoid(LatLon(lat, lon))                 # geoid height [m]
            h_geom = alt_wgs - und                        # orthometric height
            h_phi  = convert_to_geopotential(lat, h_geom) # geopotential height
        except Exception:
            # If pygeodesy still complains, just mark this sample NaN
            und, h_geom, h_phi = (np.nan, np.nan, np.nan)

        geoid_m.append(float(und) if und == und else np.nan)
        geom_m.append(h_geom)
        geopot_m.append(h_phi)

    header = [
        "index","source_file","time","dropping",
        "latitude","longitude",
        "alt_wgs84_m","geoid_m",
        "geom_height_calc_m","geopotential_height_m",
        "height_geom_xml_m","height_ptu_m",
        "pressure_hPa","temperature_K","rel_humidity_pct","wind_speed_ms"
    ]

    rows = []
    for i in range(len(data["lat"])):
        rows.append([
            i, mwx_path.name, data["time"][i], data["dropping"][i],
            data["lat"][i], data["lon"][i],
            data["alt_wgs84"][i], geoid_m[i],
            geom_m[i], geopot_m[i],
            data["height_geom_xml"][i], data["height_ptu"][i],
            data["pressure"][i], data["temperature"][i], data["rel_humidity"][i], data["wind_speed"][i]
        ])

    out_all = OUT_DIR / f"{mwx_path.stem}_gps_geopot.csv"
    with out_all.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    asc_rows = [r for r in rows if r[3] == 0]
    des_rows = [r for r in rows if r[3] == 1]

    out_asc = OUT_DIR / f"{mwx_path.stem}_gps_geopot_ascent.csv"
    out_des = OUT_DIR / f"{mwx_path.stem}_gps_geopot_descent.csv"
    for out_path, part in [(out_asc, asc_rows), (out_des, des_rows)]:
        with out_path.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(part)

    return {"all": out_all, "ascent": out_asc, "descent": out_des}

def main():
    geoid = require_geoid(GEOID_PGM)

    mwx_files: List[Path] = []
    for base in L0_DIRS:
        if base.is_dir():
            mwx_files += list(base.glob("*.mwx"))
            mwx_files += list(base.rglob("*.mwx"))
    mwx_files = sorted(set(mwx_files))

    if TEST_ONE:
        mwx_files = [p for p in mwx_files if p.name == TEST_ONE]

    if not mwx_files:
        print("No .mwx files found.")
        return

    combined_csv = OUT_DIR / "ALL_soundings_gps_geopot.csv"
    wrote_header = False

    print(f"Found {len(mwx_files)} .mwx file(s). Processing…")
    with combined_csv.open("w", newline="") as fout:
        wcombo = csv.writer(fout)
        for i, mwx in enumerate(mwx_files, 1):
            try:
                outpaths = process_one_mwx(mwx, geoid)
                print(f"[{i}/{len(mwx_files)}] OK: {mwx.name} -> "
                      f"{outpaths['all'].name}, {outpaths['ascent'].name}, {outpaths['descent'].name}")

                # append only the “all” file to the combined CSV
                with outpaths["all"].open("r", newline="") as fin:
                    r = csv.reader(fin)
                    header = next(r)
                    if not wrote_header:
                        wcombo.writerow(["source_dir"] + header)
                        wrote_header = True
                    for row in r:
                        wcombo.writerow([str(mwx.parent)] + row)

            except Exception as e:
                print(f"[{i}/{len(mwx_files)}] ERROR: {mwx.name}: {e}")

    print(f"\nDone.\nPer-sounding CSVs: {OUT_DIR}\nCombined CSV: {combined_csv}")


# %%
if __name__ == "__main__":
    main()

# %%

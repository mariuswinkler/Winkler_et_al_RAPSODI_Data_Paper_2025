# %%
import pandas as pd
from pathlib import Path
import numpy as np

# %%
all_csv = Path("Vaisala_Geopotential_Height/ALL_soundings_gps_geopot.csv")

df = pd.read_csv(all_csv)

asc_all = df[df["dropping"] == 0].copy()
des_all = df[df["dropping"] == 1].copy()

asc_path = all_csv.with_name("ALL_gps_geopot_ascent.csv")
des_path = all_csv.with_name("ALL_gps_geopot_descent.csv")

asc_all.to_csv(asc_path, index=False)
des_all.to_csv(des_path, index=False)

print(f"Wrote:\n  {asc_path}\n  {des_path}")

# %%
# Bins ascent & descent profiles to a fixed 10 m geopotential-height grid (0..31000 m).
ASC_PATH = Path("Vaisala_Geopotential_Height/ALL_gps_geopot_ascent.csv")
DES_PATH = Path("Vaisala_Geopotential_Height/ALL_gps_geopot_descent.csv")

Z_MIN, Z_MAX, DZ = 0.0, 31000.0, 10.0
BIN_EDGES = np.arange(Z_MIN, Z_MAX + DZ, DZ)               
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) * 0.5      

VARS = {
    "pressure_hPa": "pressure_hPa",
    "temperature_K": "temperature_K",
    "rel_humidity_pct": "rel_humidity_pct",
    "wind_speed_ms": "wind_speed_ms",
}
HEIGHT_COL = "geopotential_height_m"

def bin_profile(df: pd.DataFrame, height_col: str, vars_to_bin: list[str]) -> pd.DataFrame:
    """Bin variables to fixed-height grid using mean within each bin. Empty bins -> NaN."""
    df = df.copy()
    df = df[np.isfinite(df[height_col])]

    idx = np.digitize(df[height_col].to_numpy(), BIN_EDGES) - 1
    mask = (idx >= 0) & (idx < len(BIN_CENTERS))
    df = df.loc[mask].copy()
    idx = idx[mask]

    out = pd.DataFrame({
        "z_m": BIN_CENTERS,       
        "bin_lower_m": BIN_EDGES[:-1],
        "bin_upper_m": BIN_EDGES[1:],
        "count": np.zeros(len(BIN_CENTERS), dtype=int),
    })

    sums = {v: np.zeros(len(BIN_CENTERS), dtype=float) for v in vars_to_bin}
    counts = {v: np.zeros(len(BIN_CENTERS), dtype=int) for v in vars_to_bin}

    for v in vars_to_bin:
        vals = df[v].to_numpy()
        valid = np.isfinite(vals)
        idx_v = idx[valid]
        vals_v = vals[valid]
        np.add.at(sums[v], idx_v, vals_v)
        np.add.at(counts[v], idx_v, 1)

    np.add.at(out["count"].values, idx, 1)

    for v in vars_to_bin:
        mean = np.full(len(BIN_CENTERS), np.nan, dtype=float)
        c = counts[v]
        m = c > 0
        mean[m] = sums[v][m] / c[m]
        out[v] = mean

    return out

def main():
    asc = pd.read_csv(ASC_PATH)
    des = pd.read_csv(DES_PATH)

    vars_to_bin = list(VARS.values())

    asc_b = bin_profile(asc, HEIGHT_COL, vars_to_bin)
    des_b = bin_profile(des, HEIGHT_COL, vars_to_bin)

    # write out next to inputs
    asc_out = ASC_PATH.with_name(ASC_PATH.stem.replace("_ascent", "_binned10m_ascent") + ".csv")
    des_out = DES_PATH.with_name(DES_PATH.stem.replace("_descent", "_binned10m_descent") + ".csv")

    asc_b.to_csv(asc_out, index=False)
    des_b.to_csv(des_out, index=False)

    print(f"Wrote:\n  {asc_out}\n  {des_out}")

# %%
if __name__ == "__main__":
    main()

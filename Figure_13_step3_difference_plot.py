# %%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
RAPSODI_LEVEL1 = "ipfs://bafybeia34auwyvbh2rq7cn7aguzz7pulq2krdddieesm6kpysirsui7c4m"
VAI_ASC = "Vaisala_Geopotential_Height/ALL_gps_geopot_binned10m_ascent.csv"
VAI_DES = "Vaisala_Geopotential_Height/ALL_gps_geopot_binned10m_descent.csv"

# %%
bottom_cutoff   = 200.0     
tick_mean_top_m = 15000.0   
z_max_m         = 25000.0   
DZ              = 10.0      
SIZE            = 15

plt.rcParams.update({
    "axes.labelsize": SIZE,
    "legend.fontsize": SIZE,
    "xtick.labelsize": SIZE,
    "ytick.labelsize": SIZE,
    "font.size": SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6
})

vars_to_plot   = ["p", "ta", "rh", "wspd"]
xlabels        = ["Pressure / Pa", "Air Temperature / K", "Relative Humidity / %", "Wind Speed / ms$^{-1}$"]
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

meteomodem_platforms = ["INMG"]                
vaisala_platforms    = ["BCO", "RV_Meteor"]  

Z_MIN, Z_MAX = 0.0, 31000.0
BIN_EDGES   = np.arange(Z_MIN, Z_MAX + DZ, DZ)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
N_BINS      = len(BIN_CENTERS)

# %% 
def _phase_masks_from_sonde_id(ds: xr.Dataset):
    """Return ascent/descent boolean masks along the profile dimension (not 'level')."""
    if "sonde_id" not in ds:
        raise ValueError("Expected 'sonde_id' coord to split ascent/descent.")
    sid = ds["sonde_id"].astype(str).str.lower()
    asc_mask = sid.str.contains("ascent")
    des_mask = sid.str.contains("descent") | sid.str.contains("decent")
    return asc_mask, des_mask

def _pooled_binned_mean(ds: xr.Dataset, var: str, zcoord: str, profile_mask: xr.DataArray) -> np.ndarray:
    """
    Pool ALL samples from selected profiles, then bin to the fixed BIN_EDGES using mean.
    - ds[var] and ds[zcoord] must be (profile_dim, level).
    - profile_mask is 1-D over profile_dim (True for profiles to include).
    Returns array length N_BINS with NaNs for empty bins.
    """
    profile_dim = profile_mask.dims[0]
    v = ds[var].sel({profile_dim: ds[profile_dim][profile_mask]})
    z = ds[zcoord].sel({profile_dim: ds[profile_dim][profile_mask]})

    v_flat = v.values.ravel()
    z_flat = z.values.ravel()

    good = np.isfinite(v_flat) & np.isfinite(z_flat)
    if good.sum() == 0:
        return np.full(N_BINS, np.nan)

    v_flat = v_flat[good]
    z_flat = z_flat[good]

    idx = np.digitize(z_flat, BIN_EDGES) - 1  
    inside = (idx >= 0) & (idx < N_BINS)
    if inside.sum() == 0:
        return np.full(N_BINS, np.nan)

    idx = idx[inside]
    vals = v_flat[inside]

    sums   = np.bincount(idx, weights=vals, minlength=N_BINS).astype(float)
    counts = np.bincount(idx, minlength=N_BINS).astype(float)

    out = np.full(N_BINS, np.nan)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out

def _binned_mean_diff(ds_group: xr.Dataset, var: str, zcoord: str) -> np.ndarray:
    """Compute binned mean(ASC) − mean(DES) using vertical 'zcoord'."""
    asc_mask, des_mask = _phase_masks_from_sonde_id(ds_group)
    asc_mean = _pooled_binned_mean(ds_group, var, zcoord, asc_mask)
    des_mean = _pooled_binned_mean(ds_group, var, zcoord, des_mask)
    return asc_mean - des_mean

# %%
ds = xr.open_dataset(RAPSODI_LEVEL1, engine="zarr")
ds_mm  = ds.where(ds.platform.isin(meteomodem_platforms), drop=True)

def load_vaisala_diff(asc_csv, des_csv):
    asc = pd.read_csv(asc_csv)
    des = pd.read_csv(des_csv)
    cols_keep = ["z_m", "pressure_hPa", "temperature_K", "rel_humidity_pct", "wind_speed_ms"]
    asc = asc[cols_keep].rename(columns={
        "pressure_hPa": "p_asc",
        "temperature_K": "ta_asc",
        "rel_humidity_pct": "rh_asc",
        "wind_speed_ms": "wspd_asc",
    })
    des = des[cols_keep].rename(columns={
        "pressure_hPa": "p_des",
        "temperature_K": "ta_des",
        "rel_humidity_pct": "rh_des",
        "wind_speed_ms": "wspd_des",
    })
    m = pd.merge(asc, des, on="z_m", how="inner")
    grid = pd.DataFrame({"z_m": BIN_CENTERS})
    m = grid.merge(m, on="z_m", how="left")

    diff = {
        "p":   (m["p_asc"]   - m["p_des"]) * 100.0,   # hPa -> Pa
        "ta":  (m["ta_asc"]  - m["ta_des"]),
        "rh":  (m["rh_asc"]  - m["rh_des"]),
        "wspd":(m["wspd_asc"]- m["wspd_des"]),
        "z":   m["z_m"].to_numpy()
    }
    return diff

vai = load_vaisala_diff(VAI_ASC, VAI_DES)

diff_mm = {v: _binned_mean_diff(ds_mm, v, zcoord="alt") for v in vars_to_plot}

if "rh" in diff_mm:
    diff_mm["rh"] = diff_mm["rh"] * 100.0


# %%

ds

# %% 
fig, axs = plt.subplots(1, 4, figsize=(15, 6), sharey=True,
                        gridspec_kw={"wspace": 0.4, "hspace": 0.6})

mask_plot = (BIN_CENTERS >= bottom_cutoff) & (BIN_CENTERS <= z_max_m)
mask_tick = (BIN_CENTERS >= bottom_cutoff) & (BIN_CENTERS <= tick_mean_top_m)

for i, var in enumerate(vars_to_plot):
    ax = axs[i]

    y_v = np.asarray(vai[var], dtype=float)
    ax.plot(
        y_v[mask_plot],
        (BIN_CENTERS[mask_plot] / 1000.0),
        color="black", lw=1, label=("Vaisala" if i == 0 else "_nolegend_"),
        zorder=2
    )

    y_m = np.asarray(diff_mm[var], dtype=float)
    ax.plot(
        y_m[mask_plot],
        (BIN_CENTERS[mask_plot] / 1000.0),
        color="royalblue", lw=1.4, alpha=0.85,
        label=("Meteomodem" if i == 0 else "_nolegend_"),
        zorder=3
    )

    ax.axvline(0, color="black", linewidth=1, ls="dotted", zorder=1)
    ax.set_ylim(0, z_max_m / 1000.0)
    ax.set_xlabel(xlabels[i])
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.2, 1.12, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, va="top")

    if i == 0:
        ax.set_ylabel("Altitude / km")

    mean_v = float(np.nanmean(y_v[mask_tick]))
    ax.vlines(mean_v, ymin=-1.5, ymax=-0.8, color="black", linewidth=3, zorder=10, clip_on=False)

    mean_m = float(np.nanmean(y_m[mask_tick]))
    ax.vlines(mean_m, ymin=-1.5, ymax=-0.8, color="royalblue", linewidth=3, zorder=11, clip_on=False)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13),
           ncol=2, frameon=True, fontsize=SIZE+1)

plt.tight_layout()

import os; os.makedirs("Figures", exist_ok=True)
plt.savefig("./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_Combined.svg",
            format="svg", facecolor="white", bbox_inches="tight", dpi=300)
plt.savefig("./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_Combined.png",
            format="png", facecolor="white", bbox_inches="tight", dpi=150)
plt.show()

# %%
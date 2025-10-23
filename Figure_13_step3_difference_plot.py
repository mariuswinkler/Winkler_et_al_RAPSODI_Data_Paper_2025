# %%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
ds = xr.open_dataset("ipfs://bafybeid7cnw62zmzfgxcvc6q6fa267a7ivk2wcchbmkoyk4kdi5z2yj2w4", engine="zarr")

VAI_ASC = "Vaisala_Geopotential_Height/ALL_gps_geopot_binned10m_ascent.csv"
VAI_DES = "Vaisala_Geopotential_Height/ALL_gps_geopot_binned10m_descent.csv"

bottom_cutoff = 200.0   # m
tick_mean_top_m = 15000.0  # m range for mean tick
z_max_m = 25000.0       # m
SIZE = 15
# ---------------------------

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

meteomodem_platforms = ['INMG']
vaisala_platforms = ['BCO', 'RV_Meteor']

def get_diff_from_ds(ds_in, var):
    """Mean ascent − descent profile from xarray dataset."""
    ascent  = ds_in[var].where(ds_in['ascent_flag'] == 0, drop=True).mean(dim='launch_time')
    descent = ds_in[var].where(ds_in['ascent_flag'] == 1, drop=True).mean(dim='launch_time')
    return ascent - descent  # DataArray over alt

# Meteomodem differences
ds_m = ds.where(ds.platform.isin(meteomodem_platforms), drop=True)
diff_m = {var: get_diff_from_ds(ds_m, var) for var in ['p', 'ta', 'rh', 'wspd']}
z_m = ds_m.height

# ---- Load Vaisala (CSV) ----
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
    return {
        'p': (m["p_asc"] - m["p_des"]) * 100.0,        # hPa → Pa
        'ta': m["ta_asc"] - m["ta_des"],
        'rh': (m["rh_asc"] - m["rh_des"]) / 100.0,     # % → fraction
        'wspd': m["wspd_asc"] - m["wspd_des"],
        'z': m["z_m"]
    }

vai = load_vaisala_diff(VAI_ASC, VAI_DES)

# ---- Plot ----
vars_to_plot = ['p', 'ta', 'rh', 'wspd']
xlabels = ["Pressure / Pa", "Air Temperature / K", "Relative Humidity / 1", "Wind Speed / ms$^{-1}$"]
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axs = plt.subplots(1, 4, figsize=(15, 6), sharey=True,
                        gridspec_kw={"wspace": 0.4, "hspace": 0.6})

for i, var in enumerate(vars_to_plot):
    ax = axs[i]

    # Masks for altitude range
    mask_m = (z_m >= bottom_cutoff) & (z_m <= z_max_m)
    mask_v = (vai['z'] >= bottom_cutoff) & (vai['z'] <= z_max_m)

    # ---- Vaisala first (black), then Meteomodem on top (blue, alpha) ----
    ln_v, = ax.plot(vai[var][mask_v],
                    (vai['z'][mask_v] / 1000.0),
                    color='black', lw=1, label=('Vaisala' if i == 0 else '_nolegend_'),
                    zorder=2)

    ln_m, = ax.plot(diff_m[var].sel(height=slice(bottom_cutoff, z_max_m)),
                    (z_m.sel(height=slice(bottom_cutoff, z_max_m)) / 1000.0),
                    color='royalblue', lw=1.4, alpha=0.85,
                    label=('Meteomodem' if i == 0 else '_nolegend_'),
                    zorder=3)

    # Zero line & cosmetics
    ax.axvline(0, color='black', linewidth=1, ls='dotted', zorder=1)
    ax.set_ylim(0, z_max_m / 1000.0)
    ax.set_xlabel(xlabels[i])     # <- moved former titles to x-labels
    ax.spines[["left", "bottom"]].set_position(("outward", 20))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.2, 1.12, subplot_labels[i], transform=ax.transAxes,
            fontsize=SIZE + 4, va="top")

    if i == 0:
        ax.set_ylabel("Altitude / km")

    # ---- Short “tick” showing mean diff (200–15 km) for each curve ----
    # Vaisala mean:
    mv_mask = (vai['z'] >= bottom_cutoff) & (vai['z'] <= tick_mean_top_m)
    mean_v = float(np.nanmean(vai[var][mv_mask]))
    ax.vlines(mean_v, ymin=-1.5, ymax=-0.8, color='black', linewidth=3, zorder=10, clip_on=False)
    #print(f"for {variables_to_plot} the mean diff of Meteomodem is: {mean_v}")

    # Meteomodem mean:
    mean_m = float(diff_m[var].sel(height=slice(bottom_cutoff, tick_mean_top_m)).mean().values)
    ax.vlines(mean_m, ymin=-1.5, ymax=-0.8, color='royalblue', linewidth=3, zorder=11, clip_on=False)
    #print(f"for {variables_to_plot} the mean diff of Meteomodem is: {mean_m}")

# One legend total, with two entries
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
           ncol=2, frameon=True, fontsize=SIZE)

plt.tight_layout()
plt.savefig("./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_Combined.svg",
            format='svg', facecolor='white', bbox_inches="tight", dpi=300)
plt.savefig("./Figures/Fig13_Appendix_Histogram_Diff_Ascent_Descent_Combined.png",
            format='png', facecolor='white', bbox_inches="tight", dpi=150)
plt.show()
# %%

ds

# %%

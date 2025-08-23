# %%
import os
os.chdir("/Users/marius/ownCloud/PhD/12_Orcestra_Campaign/00_ORCESTRA_Radiosondes_Winkler/Winkler_et_al_RAPSODI_Data_paper_2025")

# %%
import xarray as xr
from RAPSODI_functions import add_IWV

# Load the datasets
DS_VERSION = 'v4.0.4'
DS_directory = '../level2/merged_dataset/'
DS = xr.open_dataset(DS_directory+f"RS_ORCESTRA_level2_{DS_VERSION}_raw.nc")
DS


# %%
def prep_for_ipfs(ds):
    """
    Prepare a (sounding, alt) radiosonde dataset for publishing:
      - swap to launch_time, sort
      - keep along-track lat/lon and add launch_lat/lon (1-D per profile)
      - rename flight_time->bin_average_time (seconds since launch)
      - ensure CF-ish attributes on coords and data variables
      - drop unused bounds/aux dims
    """
    # ---------- 1) Dimensions / indexing
    if "launch_time" in ds.coords and "sounding" in ds.dims and "launch_time" not in ds.dims:
        ds = ds.swap_dims({"sounding": "launch_time"})
    if "launch_time" in ds.coords:
        ds = ds.sortby("launch_time")

    # Ensure profile id as coord with cf_role
    if "sounding" in ds.coords:
        ds["sounding"] = ds["sounding"].assign_attrs(
            dict(
                cf_role="profile_id",
                long_name="sounding identifier",
                description="unique string describing the sounding's origin (PLATFORM_SND-DIRECTION_LAT_LON_TIME)"
            )
        )

    # ---------- 2) Time variables
    if "flight_time" in ds and "bin_average_time" not in ds:
        ds = ds.rename({"flight_time": "bin_average_time"})
    if "bin_average_time" in ds:
        # keep as coordinate if it isn't already
        if "bin_average_time" not in ds.coords:
            ds = ds.set_coords(["bin_average_time"])
        ds["bin_average_time"].attrs.update(dict(
            standard_name="time",
            long_name="bin-average time",
            units="s",
            comment="seconds since launch (per-profile relative time)"
        ))

    # ---------- 3) Location coords + launch_* (1-D metadata)
    if {"lat", "lon"}.issubset(ds.coords):
        if "alt" in ds.dims:
            first_valid_lvl = ds["lat"].isnull().argmin("alt")
            launch_lat = ds["lat"].isel(alt=first_valid_lvl).rename(None)
            launch_lon = ds["lon"].isel(alt=first_valid_lvl).rename(None)
            ds = ds.assign(launch_lat=launch_lat, launch_lon=launch_lon)
            ds["launch_lat"].attrs.update(dict(
                standard_name="latitude", long_name="launch latitude", units="degrees_north", axis="Y"
            ))
            ds["launch_lon"].attrs.update(dict(
                standard_name="longitude", long_name="launch longitude", units="degrees_east", axis="X"
            ))

    # ---------- 4) Vertical coordinate housekeeping
    if "alt" in ds.coords:
        # If your 'alt' is WGS84 ellipsoidal height, use the CF standard_name below.
        ds["alt"].attrs.setdefault("standard_name", "height_above_reference_ellipsoid")
        ds["alt"].attrs.setdefault("long_name", "altitude height above reference ellipsoid (WGS84)")
        ds["alt"].attrs.setdefault("units", "m")
        ds["alt"].attrs.setdefault("axis", "Z")
        ds["alt"].attrs.setdefault("positive", "up")

    # Drop any leftover bounds/aux dims
    for v in ("alt_bnds",):
        if v in ds.variables: ds = ds.drop_vars(v)
    if "nv" in ds.dims:
        ds = ds.drop_dims("nv")

    # ---------- 5) CF-ish attribute backfill per variable/coord
    attr_map = {
        # Coords
        "launch_time": dict(standard_name="time", long_name="time at which the radiosonde was launched"),
        "lat":         dict(standard_name="latitude", long_name="latitude", axis="Y",
                            description="Latitudinal position during flight", units="degree_north"),
        "lon":         dict(standard_name="longitude", long_name="longitude", axis="X",
                            description="Longitudinal position during flight", units="degree_east"),
        "p":           dict(standard_name="air_pressure", long_name="air pressure",
                            description="Air Pressure during flight", units="Pa"),

        # PTU/GPS meta & counts/flags
        "N_ptu": dict(standard_name="number_of_observations", long_name="number of observations (PTU)", units="1"),
        "N_gps": dict(standard_name="number_of_observations", long_name="number of observations (GPS)", units="1"),
        "m_ptu": dict(standard_name="status_flag", long_name="bin method (PTU)", units="1"),
        "m_gps": dict(standard_name="status_flag", long_name="bin method (GPS)", units="1"),

        # Core thermo & wind
        "ta":    dict(standard_name="air_temperature", long_name="air temperature", units="K"),
        "dp":    dict(standard_name="dew_point_temperature", long_name="dew point temperature", units="K"),
        "theta": dict(standard_name="potential_temperature", long_name="potential temperature", units="K"),
        "q":     dict(standard_name="specific_humidity", long_name="specific humidity", units="1"),
        "mr":    dict(standard_name="mixing_ratio", long_name="water vapor mixing ratio", units="1"),
        "rh":    dict(standard_name="relative_humidity", long_name="relative humidity", units="1"),
        "u":     dict(standard_name="eastward_wind", long_name="u-component of the wind", units="m s-1"),
        "v":     dict(standard_name="northward_wind", long_name="v-component of the wind", units="m s-1"),
        "wspd":  dict(standard_name="wind_speed", long_name="wind speed", units="m s-1"),
        "wdir":  dict(standard_name="wind_direction", long_name="wind from direction", units="degree"),

        # Heights
        "height_ptu": dict(standard_name="geopotential_height", long_name="geopotential height from PTU (barometric)", units="m"),
        # Platform motion estimate from height changes (NOT air motion)
        "dz": dict(standard_name="ascent_descent_rate", long_name="ascent/descent rate of measuring device", description="ascent rate is positive/ decent rate is negative", units="m s-1"),

        # Metadata / strings
        "platform": dict(long_name="launching platform"),

        # Products
        "iwv": dict(standard_name="atmosphere_mass_content_of_water_vapor", long_name="Integrated water vapor", units="kg m-2"),
    }


    # Apply defaults without clobbering existing values
    for name, add in attr_map.items():
        if name in ds.variables or name in ds.coords:
            tgt = ds[name]
            for k, v in add.items():
                tgt.attrs.setdefault(k, v)

    # Re-attach helpful comments where relevant
    if "dz" in ds:
        ds["dz"].attrs.setdefault("comment",
            "Derived from vertical changes of geopotential height; describes platform motion, not air vertical velocity.")
    if "wspd" in ds:
        ds["wspd"].attrs.setdefault("cell_methods", "alt: point (derived from averaged u, v)")
    if "ta" in ds:
        ds["ta"].attrs.setdefault("cell_methods", "alt: point (derived from averaged theta)")

    return ds



# %%
# ============================================================
# Prepare the dataset for IPFS
# ============================================================

DS = prep_for_ipfs(DS)
DS



# %%
# ============================================================
# Add Integrated Water Vapor (IWV)
# ============================================================

DS = add_IWV(DS)
DS



# %%
DS = DS.sortby(DS.launch_time)
DS.to_netcdf(DS_directory+f"for_IPFS/RS_ORCESTRA_level2_{DS_VERSION}_for_IPFS.nc")


# %%

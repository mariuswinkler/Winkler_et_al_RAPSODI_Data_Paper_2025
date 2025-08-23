# %%
import os
import numcodecs
import subprocess
import xarray as xr

DATAPATH_TO_STORE = "../00_data_for_IPFS/"
CARPATH_TO_STORE  = os.path.join(DATAPATH_TO_STORE, "00_CAR_files/")

# ============================================================
# Helper: Add to IPFS + export CAR
# ============================================================

def add_to_ipfs_and_export(zarr_dirs, basepath=""):
    """
    Add directories/files to IPFS and export CAR files for each CID.
    zarr_dirs: list of directories or files to add
    basepath: prepend path (e.g., "../00_data_for_IPFS/")
    """
    os.makedirs(CARPATH_TO_STORE, exist_ok=True)

    cids = {}
    for zarr_dir in zarr_dirs:
        full_path = os.path.join(basepath, zarr_dir) if basepath else zarr_dir
        print(f"Adding {full_path} to IPFS...")

        # Run the IPFS add command and get the CID
        result = subprocess.run(
            ["ipfs", "add", "-r", "--hidden", "-Q", "--raw-leaves", "--chunker=size-1048576", full_path],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            cid = result.stdout.strip()
            cids[zarr_dir] = cid
            print(f"✅ Added {zarr_dir} with CID: {cid}")

            # Display the public IPFS link
            print(f"🌍 Your file {zarr_dir} is now accessible at:")
            print(f"https://ipfs.io/ipfs/{cid}")

            # Export to CAR file (always in CARPATH_TO_STORE)
            car_path = os.path.join(CARPATH_TO_STORE, f"{cid}.car")
            with open(car_path, "wb") as car_file:
                export_result = subprocess.run(
                    ["ipfs", "dag", "export", cid],
                    stdout=car_file,
                    stderr=subprocess.PIPE,
                    text=False
                )
            if export_result.returncode == 0:
                print(f"📦 Exported {cid} to {car_path}\n")
            else:
                print(f"❌ Error exporting {cid}: {export_result.stderr.decode()}")

        else:
            print(f"❌ Error adding {zarr_dir}: {result.stderr}")

    print("✅ All files processed.")
    return cids


# ============================================================
# Level - 0: Upload Files to IPFS
# ============================================================

INMG_RAW_files    = "../level0/INMG"
BCO_RAW_files     = "../level0/BCO"
MET_RAW_files     = "../level0/Meteor"
MET_OSC_RAW_files = "../level0/Meteor_Oscillating"

zarr_dirs = [INMG_RAW_files, BCO_RAW_files, MET_RAW_files, MET_OSC_RAW_files]

cids_level0 = add_to_ipfs_and_export(zarr_dirs)


# ============================================================
# Level - 1: Upload Files to IPFS
# ============================================================

INMG_RAW_files    = "../level1/INMG"
BCO_RAW_files     = "../level1/BCO"
MET_RAW_files     = "../level1/Meteor"

zarr_dirs = [INMG_RAW_files, BCO_RAW_files, MET_RAW_files]

cids_level1 = add_to_ipfs_and_export(zarr_dirs)


# ============================================================
# Level - 2: Upload Files to IPFS
# ============================================================

DS_VERSION = 'v4.0.4'
DS_directory = '../level2/merged_dataset/for_IPFS/'
DS = xr.open_dataset(DS_directory + f"RS_ORCESTRA_level2_{DS_VERSION}_for_IPFS.nc")

# Prepare for IPFS
# Convert to zarr file format

def get_chunks(dimensions):
    rules = {
        "sounding": 256,
        "launch_time": 256,
        "alt": 1550,
        "sample": 1550,
        "level": -1,
        "nv": -1,
    }
    return tuple(rules.get(d, -1) for d in dimensions)

def get_encoding(dataset):
    numcodecs.blosc.set_nthreads(1)  # IMPORTANT FOR DETERMINISTIC CIDs
    codec = numcodecs.Blosc("zstd", clevel=6)
    return {
        var: {"chunks": get_chunks(dataset[var].dims), "compressor": codec}
        for var in dataset.variables if var not in dataset.dims
    }


# RAPSODI Radiosondes
DS_zarr_file = "RAPSODI_RS_ORCESTRA_level2.zarr"
DS.to_zarr(DATAPATH_TO_STORE + DS_zarr_file, encoding=get_encoding(DS), mode="w")

# RAPSODI Oscillating Radiosondes
OSC_DS = xr.open_dataset("../level1/Meteor_Oscillating/RS_ORCESTRA_Meteor_Oscillating_level1_for_IPFS.nc")
OSC_DS_zarr_file = "RAPSODI_RS_Oscillating_ORCESTRA_level1.zarr"
OSC_DS.to_zarr(DATAPATH_TO_STORE + OSC_DS_zarr_file, encoding=get_encoding(OSC_DS), mode="w")

# Upload both to IPFS
zarr_dirs = [DS_zarr_file, OSC_DS_zarr_file]

cids_level2 = add_to_ipfs_and_export(zarr_dirs, basepath=DATAPATH_TO_STORE)



# %%
# Now dial: ```localhost:5001/webui``` in browser
# %%

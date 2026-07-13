"""Rebuild basins_mouginotGrid_ocx.nc + node_basin field with the CRS bug fixed:
the OCX mesh's own topology (mesh2D_node_x/y) is stored in lon/lat degrees, not
projected metres -- must reproject to EPSG:3031 before regridding, exactly like
the original notebook's pattern for elmer_outputs/ismip6_init."""
import sys, time
sys.path.insert(0, "/home/jagereli/Postdoc/Data/postpro/elmerugrid")
import numpy as np
import xarray as xr
import xugrid as xu
from pathlib import Path
from ElmerUgrid import ugrid  # noqa

JUP = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/postpro/jupyter")
OCX_STATES = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/SSA_POC/AA_SSA_ISMIP7_OCX_MAHTI/ssp126_c005_states.nc")
OBS_FILE   = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/postpro/DATA/AntarcticaObsISMIP7-v1.2.nc")
BASINS_OUT = JUP / "basins_mouginotGrid_ocx.nc"
OBS_OUT_RAW = JUP / "obs_on_elmer_mesh_ocx.nc"
OBS_OUT_PV  = JUP / "obs_on_elmer_mesh_pv_ocx.nc"

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("Opening OCX mesh and reprojecting EPSG:4326 -> EPSG:3031 ...")
uds_ocx = xu.open_dataset(str(OCX_STATES), decode_times=False)
uds_ocx.ugrid.set_crs("EPSG:4326")
uds_ocx_proj = uds_ocx.ugrid.to_crs("EPSG:3031")
elmer_grid = uds_ocx_proj.ugrid.grid
log(f"reprojected bounds: {elmer_grid.bounds} (expect ~ +-3e6 m)")

log("Loading ISMIPobs.mouginot_basins ...")
ISMIPobs = xr.open_dataset(str(OBS_FILE), decode_times=False)
step = 4
data_median = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").median()
data_min    = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").min()
data_max    = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").max()
data = data_median.where(data_median - data_min == 0).fillna(data_max)
data_ugrid = xu.UgridDataArray.from_structured2d(data)

log("Regridding (OverlapRegridder, mode) onto OCX mesh faces (now correctly projected) ...")
regridder = xu.OverlapRegridder(source=data_ugrid, target=elmer_grid, method="mode")
basin = regridder.regrid(data_ugrid)
basin.name = "basins_mouginot"
ids = np.unique(basin.values[~np.isnan(basin.values)])
log(f"basin IDs found: {ids}")
assert len(ids) >= 15, f"still broken -- only {len(ids)} basin IDs"

out = basin.to_dataset()
out.ugrid.to_netcdf_forpv(str(BASINS_OUT))
log(f"Wrote {BASINS_OUT}")
ISMIPobs.close()

# ---- Rebuild node_basin in the existing obs-on-mesh file (velocity/dhdt/bedmachine
# interpolation was already correct -- only node_basin depended on the broken basins) ----
log("Rebuilding node_basin via majority vote from the fixed face basins ...")
fnc = elmer_grid.face_node_connectivity
n_node = elmer_grid.n_node
basins_face = basin.values
votes = [[] for _ in range(n_node)]
for f_idx in range(len(basins_face)):
    bid = basins_face[f_idx]
    if np.isnan(bid):
        continue
    for n_idx in fnc[f_idx]:
        if n_idx >= 0:
            votes[n_idx].append(bid)
node_basins = np.full(n_node, np.nan, dtype=np.float32)
for n_idx, v in enumerate(votes):
    if v:
        vals, counts = np.unique(v, return_counts=True)
        node_basins[n_idx] = vals[np.argmax(counts)]
ids_node = np.unique(node_basins[~np.isnan(node_basins)])
log(f"node basin IDs: {ids_node}")
assert len(ids_node) >= 15, f"node vote still broken -- only {len(ids_node)} IDs"

log("Patching node_basin into the existing obs-on-mesh files ...")
xr_raw = xr.open_dataset(str(OBS_OUT_RAW), decode_times=False)
xr_raw = xr_raw.load()   # pull into memory before we overwrite the source file
xr_raw.close()
xr_raw["node_basin"].values[:] = node_basins
xr_raw.to_netcdf(str(OBS_OUT_RAW) + ".tmp")
import os
os.replace(str(OBS_OUT_RAW) + ".tmp", str(OBS_OUT_RAW))

uds_obs = xu.UgridDataset(xr_raw, grids=[elmer_grid])
uds_obs.attrs = {"title": "ISMIP6 observations on OCX Elmer/Ice mesh nodes (CRS-fixed)"}
uds_obs.ugrid.to_netcdf_forpv(str(OBS_OUT_PV))
log(f"Wrote {OBS_OUT_PV}")

log("ALL DONE")

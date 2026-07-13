#!/usr/bin/env python3
"""Regrid the Mouginot basins + ISMIP7 observations onto an Elmer/Ice mesh.

The analysis notebooks compare a model run against observations node-by-node on
the model's OWN mesh, so the observations and the basin definitions have to be
interpolated onto that mesh first. Every mesh has a different node/face count,
so this must be rerun once per mesh. It writes three files (all mesh-specific):

    basins_mouginotGrid_<tag>.nc    Mouginot catchments, face-located
    obs_on_elmer_mesh_<tag>.nc      obs on mesh nodes (plain netCDF)
    obs_on_elmer_mesh_pv_<tag>.nc   same, UGRID/ParaView-readable -- the notebooks read this

Point it at your own XIOS output with --states; nothing needs editing in here:

    python build_mesh_products.py \
        --states ../DATA/my_run_states.nc \
        --obs    ../DATA/AntarcticaObsISMIP7-v1.2.nc \
        --elmerugrid /path/to/elmerugrid \
        --tag    myrun

CRS NOTE (this bit was a real bug): raw XIOS UGRID files store the mesh TOPOLOGY
(mesh2D_node_x/y) in lon/lat degrees, while the separate `x`/`y` data variables
are already projected metres. The basin regrid runs against the topology, so the
mesh MUST be reprojected to EPSG:3031 first -- otherwise OverlapRegridder
silently matches degrees against metres and returns near-empty/garbage basins.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import xugrid as xu
import netCDF4 as nc4
from scipy.interpolate import RegularGridInterpolator


def parse_args():
    p = argparse.ArgumentParser(
        description="Regrid Mouginot basins + ISMIP7 obs onto an Elmer/Ice mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--states", required=True,
                   help="XIOS states file defining the target mesh (e.g. <run>_states.nc)")
    p.add_argument("--obs", default="../DATA/AntarcticaObsISMIP7-v1.2.nc",
                   help="Raw ISMIP7 observations (velocity/dhdt/BedMachine/basins)")
    p.add_argument("--outdir", default=".", help="Where to write the three products")
    p.add_argument("--tag", default="ocx",
                   help="Suffix for the output filenames (identifies the mesh)")
    p.add_argument("--elmerugrid", default=os.environ.get("ELMERUGRID_DIR", ""),
                   help="Path to the ElmerUgrid checkout (registers .ugrid.to_netcdf_forpv). "
                        "Defaults to $ELMERUGRID_DIR.")
    p.add_argument("--coarsen", type=int, default=4,
                   help="Coarsening factor for the basin raster before the overlap regrid")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if the output files already exist")
    return p.parse_args()


args = parse_args()

if not args.elmerugrid:
    sys.exit("error: --elmerugrid (or $ELMERUGRID_DIR) is required -- see README.")
sys.path.insert(0, os.path.abspath(args.elmerugrid))
from ElmerUgrid import ugrid  # noqa: F401,E402  registers .ugrid.to_netcdf_forpv

OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)
BASINS_OUT = OUTDIR / f"basins_mouginotGrid_{args.tag}.nc"
OBS_OUT_RAW = OUTDIR / f"obs_on_elmer_mesh_{args.tag}.nc"
OBS_OUT_PV = OUTDIR / f"obs_on_elmer_mesh_pv_{args.tag}.nc"

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# ============================================================================
# 0. Target mesh -- reprojected to EPSG:3031 (see CRS NOTE in the docstring)
# ============================================================================
log("Opening target mesh and reprojecting EPSG:4326 -> EPSG:3031 ...")
uds_mesh = xu.open_dataset(str(args.states), decode_times=False)
uds_mesh.ugrid.set_crs("EPSG:4326")
uds_proj = uds_mesh.ugrid.to_crs("EPSG:3031")
elmer_grid = uds_proj.ugrid.grid
n_node = elmer_grid.n_node
log(f"mesh: {n_node} nodes, {elmer_grid.n_face} faces")
log(f"reprojected bounds: {elmer_grid.bounds} (expect ~ +-3e6 m, NOT +-180)")

# Node coordinates in metres: prefer the projected x/y data variables if the run
# wrote them (XIOS usually does); otherwise fall back to the reprojected topology.
if "x" in uds_mesh and "y" in uds_mesh:
    node_x = np.asarray(uds_mesh["x"].values, dtype=np.float32)
    node_y = np.asarray(uds_mesh["y"].values, dtype=np.float32)
    if node_x.ndim > 1:            # (time, n_nodes) -> static
        node_x, node_y = node_x[0], node_y[0]
else:
    node_x = elmer_grid.node_x.astype(np.float32)
    node_y = elmer_grid.node_y.astype(np.float32)
node_xy = np.column_stack([node_x, node_y])

# ============================================================================
# 1. Mouginot basins -> mesh faces (categorical: coarsen + mode)
# ============================================================================
if BASINS_OUT.exists() and not args.force:
    log(f"{BASINS_OUT.name} exists, skipping basin regrid (--force to rebuild)")
    basin = xu.open_dataset(str(BASINS_OUT))["basins_mouginot"]
else:
    log("Loading mouginot_basins ...")
    ISMIPobs = xr.open_dataset(str(args.obs), decode_times=False)
    step = args.coarsen
    data_median = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").median()
    data_min = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").min()
    data_max = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").max()
    data = data_median.where(data_median - data_min == 0).fillna(data_max)

    log("Building source ugrid from the coarsened raster ...")
    from_structured = getattr(xu.UgridDataArray, "from_structured2d",
                              xu.UgridDataArray.from_structured)
    data_ugrid = from_structured(data)

    log("Regridding (OverlapRegridder, mode) onto mesh faces ...")
    regridder = xu.OverlapRegridder(source=data_ugrid, target=elmer_grid, method="mode")
    basin = regridder.regrid(data_ugrid)
    basin.name = "basins_mouginot"

    ids = np.unique(basin.values[~np.isnan(basin.values)])
    log(f"basin IDs found: {ids}")
    # Antarctica has 18 IMBIE catchments; anything far below that means the regrid
    # silently failed (the classic symptom of the unprojected-mesh CRS bug).
    assert len(ids) >= 15, f"basin regrid looks broken -- only {len(ids)} basin IDs"

    basin.to_dataset().ugrid.to_netcdf_forpv(str(BASINS_OUT))
    log(f"Wrote {BASINS_OUT}")
    ISMIPobs.close()

# ============================================================================
# 2. ISMIP obs (velocity, dhdt, BedMachine surface) -> mesh nodes
# ============================================================================
if OBS_OUT_PV.exists() and not args.force:
    log(f"{OBS_OUT_PV.name} exists, skipping obs regrid (--force to rebuild)")
    log("ALL DONE")
    sys.exit(0)

log("Loading obs (velocity/dhdt/bedmachine) ...")
# decode_times=True is REQUIRED: vel_time/cpom_dhdt_time use "days since 1900-1-1".
# Casting the *raw* undecoded integers straight to datetime64[D] reinterprets them
# against numpy's 1970 epoch instead -> every timestamp lands ~70 years late.
ISMIPobs = xr.open_dataset(str(args.obs), decode_times=True)

x_min, x_max = node_x.min(), node_x.max()
y_min, y_max = node_y.min(), node_y.max()
pad = 50_000


def clip_obs_grid(obs_x, obs_y):
    xi = np.where((obs_x >= x_min - pad) & (obs_x <= x_max + pad))[0]
    yi = np.where((obs_y >= y_min - pad) & (obs_y <= y_max + pad))[0]
    return slice(xi[0], xi[-1] + 1), slice(yi[0], yi[-1] + 1)


obs_x_vel = ISMIPobs.vx_timeseries.x.values.astype(float)
obs_y_vel = ISMIPobs.vx_timeseries.y.values.astype(float)
xi_vel, yi_vel = clip_obs_grid(obs_x_vel, obs_y_vel)
x_vel_clip, y_vel_clip = obs_x_vel[xi_vel], obs_y_vel[yi_vel]

obs_x_dhdt = ISMIPobs.dhdt_cpom.x1km.values.astype(float)
obs_y_dhdt = ISMIPobs.dhdt_cpom.y1km.values.astype(float)
xi_dhdt, yi_dhdt = clip_obs_grid(obs_x_dhdt, obs_y_dhdt)
x_dhdt_clip, y_dhdt_clip = obs_x_dhdt[xi_dhdt], obs_y_dhdt[yi_dhdt]

obs_x_bm = ISMIPobs.surface_bedmachine.x.values.astype(float)
obs_y_bm = ISMIPobs.surface_bedmachine.y.values.astype(float)
xi_bm, yi_bm = clip_obs_grid(obs_x_bm, obs_y_bm)

BATCH = 200_000


def make_rgi(data_2d, y_coords, x_coords):
    """Bilinear interpolator + a companion NaN-fraction interpolator, so that
    nodes that fall mostly on missing observations come back NaN rather than
    being silently filled with a biased value."""
    y = y_coords.copy(); d = data_2d.copy().astype(np.float64)
    if y[0] > y[-1]:
        y = y[::-1]; d = d[::-1, :]
    d_filled = np.where(np.isfinite(d), d, 0.0)
    nan_mask = ~np.isfinite(d)
    rgi = RegularGridInterpolator((y, x_coords), d_filled, method="linear",
                                  bounds_error=False, fill_value=np.nan)
    rgi_nan = RegularGridInterpolator((y, x_coords), nan_mask.astype(float), method="linear",
                                      bounds_error=False, fill_value=1.0)
    return rgi, rgi_nan


def interp_to_nodes(rgi, rgi_nan, threshold=0.5):
    out = np.full(n_node, np.nan, dtype=np.float32)
    for start in range(0, n_node, BATCH):
        end = min(start + BATCH, n_node)
        pts = np.column_stack([node_y[start:end], node_x[start:end]])
        vals = rgi(pts).astype(np.float32)
        nan_frac = rgi_nan(pts)
        vals[nan_frac > threshold] = np.nan
        out[start:end] = vals
    return out


vel_years = ((ISMIPobs.vx_timeseries.vel_time.values.astype("datetime64[D]") -
              np.datetime64("1950-01-01")).astype(float) / 365.25 + 1950.0).astype(np.float32)
dhdt_years = ((ISMIPobs.dhdt_cpom.cpom_dhdt_time.values.astype("datetime64[D]") -
               np.datetime64("1950-01-01")).astype(float) / 365.25 + 1950.0).astype(np.float32)
n_vel, n_dhdt = len(vel_years), len(dhdt_years)
log(f"n_vel={n_vel} ({vel_years[0]:.1f}-{vel_years[-1]:.1f}), "
    f"n_dhdt={n_dhdt} ({dhdt_years[0]:.1f}-{dhdt_years[-1]:.1f}), n_nodes={n_node}")

node_dim = [d for d in uds_mesh.dims if "node" in d][0]
log(f"node dim: {node_dim}")

ds_nc = nc4.Dataset(str(OBS_OUT_RAW), "w")
ds_nc.createDimension(node_dim, n_node)
ds_nc.createDimension("vel_time", n_vel)
ds_nc.createDimension("dhdt_time", n_dhdt)
v = ds_nc.createVariable("vel_time", "f4", ("vel_time",)); v[:] = vel_years
v = ds_nc.createVariable("dhdt_time", "f4", ("dhdt_time",)); v[:] = dhdt_years
ds_nc.createVariable("velocity", "f4", ("vel_time", node_dim), fill_value=np.nan, zlib=True, complevel=4)
ds_nc["velocity"].units = "m yr-1"; ds_nc["velocity"].location = "node"
ds_nc.createVariable("dhdt", "f4", ("dhdt_time", node_dim), fill_value=np.nan, zlib=True, complevel=4)
ds_nc["dhdt"].units = "m yr-1"; ds_nc["dhdt"].location = "node"
ds_nc.createVariable("surface_bedmachine", "f4", (node_dim,), fill_value=np.nan, zlib=True, complevel=4)
ds_nc.createVariable("node_basin", "f4", (node_dim,), fill_value=np.nan, zlib=True, complevel=4)

log("Regridding BedMachine surface ...")
bm_clip = ISMIPobs.surface_bedmachine.values[yi_bm, :][:, xi_bm]
rgi_bm, rgi_bm_nan = make_rgi(bm_clip, obs_y_bm[yi_bm], obs_x_bm[xi_bm])
ds_nc["surface_bedmachine"][:] = interp_to_nodes(rgi_bm, rgi_bm_nan)
del bm_clip, rgi_bm, rgi_bm_nan

log("Regridding velocity ...")
for t in range(n_vel):
    vx_t = ISMIPobs.vx_timeseries.isel(vel_time=t).values[yi_vel, :][:, xi_vel]
    vy_t = ISMIPobs.vy_timeseries.isel(vel_time=t).values[yi_vel, :][:, xi_vel]
    spd = np.sqrt(vx_t**2 + vy_t**2).astype(np.float32)
    rgi, rgi_nan = make_rgi(spd, y_vel_clip, x_vel_clip)
    ds_nc["velocity"][t, :] = interp_to_nodes(rgi, rgi_nan)
    if (t + 1) % 4 == 0:
        ds_nc.sync(); log(f"  velocity {t+1}/{n_vel}")
ds_nc.sync(); log(f"velocity done ({n_vel})")

log("Regridding dhdt ...")
for t in range(n_dhdt):
    dhdt_t = ISMIPobs.dhdt_cpom.isel(cpom_dhdt_time=t).values[yi_dhdt, :][:, xi_dhdt]
    rgi, rgi_nan = make_rgi(dhdt_t, y_dhdt_clip, x_dhdt_clip)
    ds_nc["dhdt"][t, :] = interp_to_nodes(rgi, rgi_nan)
    if (t + 1) % 4 == 0:
        ds_nc.sync(); log(f"  dhdt {t+1}/{n_dhdt}")
ds_nc.sync(); log(f"dhdt done ({n_dhdt})")

# node_basin: majority vote over the faces touching each node
log("Node basins by majority vote from the face basins ...")
basins_face = basin.values
fnc = elmer_grid.face_node_connectivity
votes = [[] for _ in range(n_node)]
for f_idx in range(len(basins_face)):
    bid = basins_face[f_idx]
    if np.isnan(bid):
        continue
    for n_idx in fnc[f_idx]:
        if n_idx >= 0:
            votes[n_idx].append(bid)
node_basins = np.full(n_node, np.nan, dtype=np.float32)
for n_idx, vv in enumerate(votes):
    if vv:
        vals, counts = np.unique(vv, return_counts=True)
        node_basins[n_idx] = vals[np.argmax(counts)]
ids_node = np.unique(node_basins[~np.isnan(node_basins)])
log(f"node basin IDs: {ids_node}")
assert len(ids_node) >= 15, f"node vote looks broken -- only {len(ids_node)} IDs"

ds_nc["node_basin"][:] = node_basins
ds_nc.sync(); ds_nc.close()
log(f"Wrote {OBS_OUT_RAW}")

log("Reopening as xugrid, writing the ParaView/UGRID version ...")
xr_raw = xr.open_dataset(str(OBS_OUT_RAW), decode_times=False)
uds_obs = xu.UgridDataset(xr_raw, grids=[elmer_grid])
uds_obs.attrs = {"title": f"ISMIP7 observations on Elmer/Ice mesh nodes ({args.tag})"}
uds_obs.ugrid.to_netcdf_forpv(str(OBS_OUT_PV))
log(f"Wrote {OBS_OUT_PV}")
ISMIPobs.close()

log("ALL DONE")

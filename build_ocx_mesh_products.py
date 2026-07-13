"""Regrid Mouginot basins + ISMIP obs (velocity/dhdt/bedmachine) onto the OCX
mesh (different node/face count than SmallEnsTrans). Mirrors the exact recipe
already proven in CompareModVsObs.ipynb cells 6-16 (basins) and COMP1a-e
(obs-on-mesh), just retargeted at the OCX states.nc mesh."""
import sys, time, functools
sys.path.insert(0, "/home/jagereli/Postdoc/Data/postpro/elmerugrid")
import numpy as np
import xarray as xr
import xugrid as xu
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc4
from ElmerUgrid import ugrid  # noqa: registers .ugrid.to_netcdf_forpv

JUP = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/postpro/jupyter")
OCX_STATES = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/SSA_POC/AA_SSA_ISMIP7_OCX_MAHTI/ssp126_c005_states.nc")
OBS_FILE = Path("/media/jagereli/Expansion1/TLC_ISMIP7_ANT/postpro/DATA/AntarcticaObsISMIP7-v1.2.nc")
BASINS_OUT = JUP / "basins_mouginotGrid_ocx.nc"
OBS_OUT_RAW = JUP / "obs_on_elmer_mesh_ocx.nc"
OBS_OUT_PV = JUP / "obs_on_elmer_mesh_pv_ocx.nc"

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("Opening OCX mesh (states.nc) ...")
uds_ocx = xu.open_dataset(str(OCX_STATES), decode_times=False)
elmer_grid = uds_ocx.ugrid.grid
n_node = elmer_grid.n_node
log(f"OCX mesh: {n_node} nodes, {elmer_grid.n_face} faces")

node_x = uds_ocx["x"].values.astype(np.float32)   # already projected EPSG:3031, time-independent
node_y = uds_ocx["y"].values.astype(np.float32)
node_xy = np.column_stack([node_x, node_y])

# ============================================================================
# 1. Mouginot basins -> OCX mesh faces (categorical: coarsen+mode, per old notebook)
# ============================================================================
if BASINS_OUT.exists():
    log(f"{BASINS_OUT.name} already exists, skipping basin regrid")
else:
    log("Loading ISMIPobs.mouginot_basins ...")
    ISMIPobs = xr.open_dataset(str(OBS_FILE), decode_times=False)
    step = 4
    data_median = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").median()
    data_min    = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").min()
    data_max    = ISMIPobs.mouginot_basins.coarsen(x=step, y=step, boundary="trim").max()
    data = data_median.where(data_median - data_min == 0).fillna(data_max)
    log("Building source ugrid from coarsened raster ...")
    data_ugrid = xu.UgridDataArray.from_structured(data)
    log("Regridding (OverlapRegridder, mode) onto OCX mesh faces ...")
    regridder = xu.OverlapRegridder(source=data_ugrid, target=elmer_grid, method="mode")
    basin = regridder.regrid(data_ugrid)
    basin.name = "basins_mouginot"
    out = basin.to_dataset()   # already a UgridDataset in this xugrid version
    out.ugrid.to_netcdf_forpv(str(BASINS_OUT))
    log(f"Wrote {BASINS_OUT}")
    ISMIPobs.close()

# ============================================================================
# 2. ISMIP obs (velocity, dhdt, BedMachine surface) -> OCX mesh nodes
# ============================================================================
if OBS_OUT_PV.exists():
    log(f"{OBS_OUT_PV.name} already exists, skipping obs regrid")
else:
    log("Loading ISMIPobs (velocity/dhdt/bedmachine) ...")
    # decode_times=True here: vel_time/cpom_dhdt_time use "days since 1900-1-1",
    # and the vel_years/dhdt_years conversion below assumes real datetime64 input
    # (casting *raw* undecoded ints straight to datetime64[D] silently reinterprets
    # them against numpy's 1970 epoch -> years came out ~70 yr too late; caught by
    # comparing obs_on_elmer_mesh_pv_ocx.nc against the already-correct
    # obs_on_elmer_mesh_pv.nc, which was built while ISMIPobs was decoded).
    ISMIPobs = xr.open_dataset(str(OBS_FILE), decode_times=True)

    x_min, x_max = node_xy[:, 0].min(), node_xy[:, 0].max()
    y_min, y_max = node_xy[:, 1].min(), node_xy[:, 1].max()
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
        n = len(node_x); out = np.full(n, np.nan, dtype=np.float32)
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
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
    log(f"n_vel={n_vel}, n_dhdt={n_dhdt}, n_nodes={n_node}")

    node_dim = [d for d in uds_ocx.dims if "node" in d][0]
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

    # node basin (majority-vote from the just-built face-located basin regrid)
    log("Loading OCX face basins for node majority-vote ...")
    basins_face_ds = xu.open_dataset(str(BASINS_OUT))
    basins_face = basins_face_ds["basins_mouginot"].values
    fnc = elmer_grid.face_node_connectivity
    votes = [[] for _ in range(n_node)]
    for f_idx in range(len(basins_face)):
        bid = basins_face[f_idx]
        if np.isnan(bid): continue
        for n_idx in fnc[f_idx]:
            if n_idx >= 0: votes[n_idx].append(bid)
    node_basins = np.full(n_node, np.nan, dtype=np.float32)
    for n_idx, v in enumerate(votes):
        if v:
            vals, counts = np.unique(v, return_counts=True)
            node_basins[n_idx] = vals[np.argmax(counts)]
    ds_nc["node_basin"][:] = node_basins
    ds_nc.sync(); ds_nc.close()
    log(f"Wrote {OBS_OUT_RAW}")

    log("Reopening as xugrid, writing ParaView version ...")
    xr_raw = xr.open_dataset(str(OBS_OUT_RAW), decode_times=False)
    uds_obs = xu.UgridDataset(xr_raw, grids=[elmer_grid])
    uds_obs.attrs = {"title": "ISMIP6 observations on OCX Elmer/Ice mesh nodes"}
    uds_obs.ugrid.to_netcdf_forpv(str(OBS_OUT_PV))
    log(f"Wrote {OBS_OUT_PV}")
    ISMIPobs.close()

log("ALL DONE")

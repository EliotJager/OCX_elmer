"""Shared setup for the ISMIP7 OCX analysis notebooks (the old notebook's "§0").

Both notebooks -- `AnalyseMemberVsObs.ipynb` (single member vs observations) and
`AnalyseEnsemble.ipynb` (ensemble + Sobol) -- need the same foundation: the Elmer
mesh geometry, the node->face helpers, the flux integrators, and the per-member
mass-budget diagnostics. That code lives here so there is exactly one copy of it.

Usage from a notebook (its config cell stays in the notebook, so each notebook
still says out loud which run it is pointing at):

    import elmer_analysis as ea

    CONFIG = ea.Config(
        postpro_dir      = "/path/to/elmerugrid",
        ocx_states_file  = "../DATA/my_run_states.nc",
        ...
    )
    globals().update(ea.init(CONFIG))     # -> mesh, obs, basins, times, helpers

`init()` populates this module's globals (mesh arrays, basins, times, obs) AND
returns them as a dict, so `globals().update(...)` pulls the same names into the
notebook. The flux functions below deliberately read those module globals rather
than taking 15 arguments each -- that keeps their bodies identical to the
original notebook cells, which were validated against real runs.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import matplotlib.pyplot as plt

# ── Physical constants (match COLD.lua) ──────────────────────────────────────
RHO_ICE = 917.0     # kg m-3
RHO_SW = 1028.0     # kg m-3
GT = 1e-12          # kg -> Gt
V_FILL_THRESHOLD = 30000.0   # |velocity| above this = fill/garbage node -> mask


@dataclass
class Config:
    """Everything that changes between runs / machines. Set this in the notebook."""

    # Where the ElmerUgrid tools live (registers `.ugrid.to_netcdf_forpv`).
    postpro_dir: str

    # Mesh-specific products, built by preprocessing/build_mesh_products.py.
    basins_file: str            # Mouginot catchments regridded onto THIS mesh (faces)
    obs_mesh_file: str          # ISMIP obs regridded onto THIS mesh (nodes)

    # Raw observations.
    obs_ismip_file: str = "../DATA/AntarcticaObsISMIP7-v1.2.nc"
    obs_discharge_file: str = "../DATA/AIS_discharge_BMHF14.nc"

    # Which model output to analyse: "ocx" (a single XIOS run) or
    # "smallenstrans" (one member of the SmallEnsTrans ensemble).
    member_kind: str = "ocx"

    # --- ocx: point these at YOUR XIOS output -------------------------------
    ocx_states_file: Optional[str] = None    # geometry, velocity, groundedmask, haf, cell_area
    ocx_forcing_file: Optional[str] = None   # smb_total_flux (face), bmb (node), ligroundf

    # --- smallenstrans ------------------------------------------------------
    member_dir: Optional[str] = None
    member_glob: str = "*/*t_aa_trans*.nc"
    member_glmelt: str = "fmp"               # BMB treatment at the GL: 'fmp' or 'nmp'
    ens_agg_file: Optional[str] = None       # per-member per-catchment budget

    figure_dir: str = "figures"

    def __post_init__(self):
        if self.member_kind not in ("ocx", "smallenstrans"):
            raise ValueError(f"member_kind must be 'ocx' or 'smallenstrans', got {self.member_kind!r}")
        if self.member_kind == "ocx" and not self.ocx_states_file:
            raise ValueError("member_kind='ocx' requires ocx_states_file")
        if self.member_kind == "smallenstrans" and not self.member_dir:
            raise ValueError("member_kind='smallenstrans' requires member_dir")


# ── Module state, populated by init() ────────────────────────────────────────
_cfg: Optional[Config] = None
FIGURE_DIR: Optional[Path] = None


def init(cfg: Config) -> dict:
    """Load mesh, basins and observations; return the shared namespace."""
    global _cfg, FIGURE_DIR
    global ISMIPobs, basin, elmer_grid, grid, uds_obs
    global node_xy, node_x, node_y, fnc, enc, efc, face_xy, face_areas
    global basins, basin_ids, n_basins, times, n_time, years
    global n0_idx, n1_idx, edge_len, normal_left, normal_right, edge_mid
    global member_files, MESH_REF_FILE

    _cfg = cfg

    # ElmerUgrid registers .ugrid.to_netcdf_forpv and the interpolation helpers.
    sys.path.insert(0, os.path.abspath(cfg.postpro_dir))
    from ElmerUgrid import ugrid  # noqa: F401

    FIGURE_DIR = Path(cfg.figure_dir)
    FIGURE_DIR.mkdir(exist_ok=True)

    # ISMIP7 gridded observations (velocity time series, dhdt, BedMachine surface)
    ISMIPobs = xr.open_dataset(cfg.obs_ismip_file, decode_times=False)

    # Mouginot catchments already regridded onto the ACTIVE mesh (face-located).
    # decode_times=False: the SmallEnsTrans-tagged file carries a placeholder time
    # coord in "years since 0000-01-01" units that cftime cannot parse at all
    # (not just "cftime missing" -- the unit itself is unsupported for this calendar).
    _basin_ds = xu.open_dataset(cfg.basins_file, decode_times=False)
    basin = _basin_ds["basins_mouginot"]          # UgridDataArray, (n_faces,)
    elmer_grid = basin.ugrid.grid
    print(f"Mesh ({cfg.member_kind}):", elmer_grid.n_node, "nodes,", elmer_grid.n_face, "faces")

    if cfg.member_kind == "ocx":
        member_files = []
        MESH_REF_FILE = cfg.ocx_states_file
    else:
        member_files = sorted(Path(cfg.member_dir).glob(cfg.member_glob))
        print(f"Found {len(member_files)} SmallEnsTrans members")
        MESH_REF_FILE = member_files[0]
    print("Mesh reference file:", MESH_REF_FILE)

    # ── Reference mesh: node coordinates, connectivity, face areas ───────────
    uds_ref = xu.open_dataset(str(MESH_REF_FILE), mask_and_scale=True, decode_times=False)
    grid = uds_ref.ugrid.grid

    # x/y are (n_nodes,) in OCX, (time, n_nodes) in SmallEnsTrans (written every step
    # though time-invariant for a fixed SSA mesh) -> squeeze out any time dim.
    def _static_node_field(da):
        return da.values[0] if "time" in da.dims else da.values

    node_xy = np.column_stack([_static_node_field(uds_ref["x"]),
                               _static_node_field(uds_ref["y"])])

    fnc = grid.face_node_connectivity              # (n_faces, max_nodes)  fill=-1/0
    enc = grid.edge_node_connectivity              # (n_edges, 2)
    efc = grid.edge_face_connectivity              # (n_edges, 2)  -1 = boundary

    valid_fnc = fnc >= 0
    safe_fnc = np.where(valid_fnc, fnc, 0)
    face_xy = np.column_stack([
        np.nanmean(np.where(valid_fnc, node_xy[safe_fnc, 0], np.nan), axis=1),
        np.nanmean(np.where(valid_fnc, node_xy[safe_fnc, 1], np.nan), axis=1),
    ])

    basins = basin.values
    basin_ids = np.unique(basins[~np.isnan(basins)]).astype(int)
    n_basins = len(basin_ids)

    if cfg.member_kind == "ocx":
        # states.nc time = days since 1990-01-01, 360-day calendar, annual writes.
        times = 1990.0 + uds_ref["time"].values / 360.0
    else:
        # Confirmed via restart_time_*.nc's `elmer_time` field: 25 annual snapshots,
        # simulation years 1990-2014 (the raw `time` coord in this file is NOT the
        # absolute clock -- elmer_time is authoritative).
        times = np.linspace(1990, 2014, 25)
    n_time = len(times)
    years = times
    print(f"n_time={n_time}, years {times[0]:.1f}-{times[-1]:.1f}")

    if cfg.member_kind == "ocx" and "cell_area" in uds_ref:
        face_areas = uds_ref["cell_area"].values.astype(np.float64)
        print(f"cell_area (real, from XIOS): total {face_areas.sum():.3e} m2")
    else:
        # SmallEnsTrans: true_cell_area is all-fill -> compute from geometry.
        face_areas = compute_face_areas(node_xy, fnc)
    uds_ref.close()
    print(f"face_areas: min={face_areas.min():.3e}, max={face_areas.max():.3e} m2, "
          f"total={face_areas.sum():.3e} m2 (expect ~1.4e13)")

    # ── Edge geometry (in metres) ────────────────────────────────────────────
    n0_idx = enc[:, 0]
    n1_idx = enc[:, 1]

    edge_vec = node_xy[n1_idx] - node_xy[n0_idx]
    edge_len = np.linalg.norm(edge_vec, axis=1)       # metres, expect 100-50000 m
    edge_unit = edge_vec / np.where(edge_len > 0, edge_len, 1)[:, None]

    normal_left = np.column_stack([-edge_unit[:, 1], edge_unit[:, 0]])
    normal_right = np.column_stack([edge_unit[:, 1], -edge_unit[:, 0]])
    edge_mid = (node_xy[n0_idx] + node_xy[n1_idx]) / 2.0
    print(f"edge_len: min={edge_len.min():.1f}, max={edge_len.max():.1f} m")

    # ── Observations already interpolated onto this mesh's nodes ─────────────
    uds_obs = xu.open_dataset(cfg.obs_mesh_file)
    node_x = node_xy[:, 0].astype(np.float32)
    node_y = node_xy[:, 1].astype(np.float32)

    return dict(
        cfg=cfg, ISMIPobs=ISMIPobs, basin=basin, elmer_grid=elmer_grid, grid=grid,
        uds_obs=uds_obs, node_xy=node_xy, node_x=node_x, node_y=node_y,
        fnc=fnc, enc=enc, efc=efc, face_xy=face_xy, face_areas=face_areas,
        basins=basins, basin_ids=basin_ids, n_basins=n_basins,
        times=times, n_time=n_time, years=years,
        member_files=member_files, MESH_REF_FILE=MESH_REF_FILE,
        FIGURE_DIR=FIGURE_DIR,
        RHO_ICE=RHO_ICE, RHO_SW=RHO_SW, GT=GT, V_FILL_THRESHOLD=V_FILL_THRESHOLD,
        # helpers, so the notebook can call them unqualified
        node_to_face=node_to_face, node_to_face_minmax=node_to_face_minmax,
        get_gl_straddling_faces=get_gl_straddling_faces,
        gl_discharge_one_step=gl_discharge_one_step,
        calving_edge_flux=calving_edge_flux,
        load_member_fields=load_member_fields,
        compute_member_diagnostics=compute_member_diagnostics,
        member_diag_dataset=member_diag_dataset,
        load_obs_discharge=load_obs_discharge,
        get_catchment_mask=get_catchment_mask,
        get_catchment_bounds=get_catchment_bounds,
        get_obs_velocity_at_year=get_obs_velocity_at_year,
        get_obs_dhdt_at_year=get_obs_dhdt_at_year,
        load_model_snapshot_ugrid=load_model_snapshot_ugrid,
        plot_catchment_field=plot_catchment_field,
        savefig=savefig, slug=slug,
    )


# ── Figures ──────────────────────────────────────────────────────────────────
def slug(s: str) -> str:
    """Filesystem-safe slug for figure filenames (label -> no spaces/parens)."""
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_")


def savefig(fig, name: str):
    """Save a figure to FIGURE_DIR at publication-ish resolution and say so."""
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    return path


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def compute_face_areas(node_xy: np.ndarray, face_node_conn: np.ndarray) -> np.ndarray:
    """Shoelace-formula face areas (m2) for arbitrary (triangle/quad) polygons."""
    n_faces, max_nodes = face_node_conn.shape
    areas = np.zeros(n_faces)
    for k in range(max_nodes):
        k_next = (k + 1) % max_nodes
        idx_k, idx_k_next = face_node_conn[:, k], face_node_conn[:, k_next]
        valid = (idx_k >= 0) & (idx_k_next >= 0)
        xk      = np.where(valid, node_xy[np.where(idx_k      >= 0, idx_k,      0), 0], 0.0)
        yk      = np.where(valid, node_xy[np.where(idx_k      >= 0, idx_k,      0), 1], 0.0)
        xk_next = np.where(valid, node_xy[np.where(idx_k_next >= 0, idx_k_next, 0), 0], 0.0)
        yk_next = np.where(valid, node_xy[np.where(idx_k_next >= 0, idx_k_next, 0), 1], 0.0)
        areas += xk * yk_next - xk_next * yk
    return np.abs(areas) / 2.0


def node_to_face(node_values: np.ndarray, face_node_conn: np.ndarray) -> np.ndarray:
    """
    Average node-located values to face centres.
    node_values   : (..., n_nodes)
    face_node_conn: (n_faces, max_nodes)  fill=-1
    returns       : (..., n_faces)
    """
    valid    = face_node_conn >= 0                      # (n_faces, max_nodes)
    safe_idx = np.where(valid, face_node_conn, 0)
    gathered = node_values[..., safe_idx]               # (..., n_faces, max_nodes)
    # np.where broadcasts `valid` over all leading dims — no dimension collapse
    gathered = np.where(valid, gathered, np.nan)
    return np.nanmean(gathered, axis=-1)                # (..., n_faces)


def node_to_face_minmax(node_values: np.ndarray, face_node_conn: np.ndarray):
    """
    Compute per-face min and max of node values.
    node_values   : (..., n_nodes)
    face_node_conn: (n_faces, max_nodes)  fill=-1
    Returns: face_min, face_max  each (..., n_faces)
    """
    valid    = face_node_conn >= 0
    safe_idx = np.where(valid, face_node_conn, 0)
    gathered = node_values[..., safe_idx]          # (..., n_faces, max_nodes)
    gathered = np.where(valid, gathered, np.nan)

    return np.nanmin(gathered, axis=-1), np.nanmax(gathered, axis=-1)


def get_gl_straddling_faces(gm_node: np.ndarray, face_node_conn: np.ndarray) -> np.ndarray:
    """
    Identify faces that straddle the grounding line:
    at least one node grounded (gm >= 0) AND at least one floating (gm < 0).

    Parameters
    ----------
    gm_node : (..., n_nodes)  raw groundedmask node values (-1, 0, 1)

    Returns
    -------
    gl_straddle : (..., n_faces)  bool
    """
    gm_min, gm_max = node_to_face_minmax(gm_node, face_node_conn)
    # Has at least one floating node (min < 0) AND one grounded/GL node (max >= 0)
    return (gm_min < 0) & (gm_max >= 0)


# ── Flux integrators ─────────────────────────────────────────────────────────
def gl_discharge_one_step(
    gm_face_t: np.ndarray,     # (n_faces,)  grounded mask averaged to faces
    u_node_t:  np.ndarray,     # (n_nodes,)  x-velocity
    v_node_t:  np.ndarray,     # (n_nodes,)  y-velocity
    h_node_t:  np.ndarray,     # (n_nodes,)  ice thickness
    basins:    np.ndarray,     # (n_faces,)  basin IDs
    basin_ids: np.ndarray,     # (n_basins,)
) -> np.ndarray:
    """
    Compute grounding-line discharge per basin for a single time step.

    Grounding-line edges are edges where one adjacent face is grounded (gm > 0.5)
    and the other is floating/ocean (gm ≤ 0.5, or boundary = -1).

    Flux through each GL edge:
        Q = (u·n̂_out + v·n̂_out) × h_edge × L_edge    [m³ yr⁻¹ ice equiv.]

    n̂_out points from the grounded face toward the floating side.
    """
    # ── Identify grounding-line edges ────────────────────────────────────────
    # efc has shape (n_edges, 2): face index on each side, -1 = domain boundary.
    f0 = efc[:, 0]
    f1 = efc[:, 1]

    # Grounded flag for each adjacent face (-1 face → treated as floating)
    def face_grounded(fidx):
        grounded = np.zeros(len(fidx), dtype=bool)
        valid = fidx >= 0
        grounded[valid] = gm_face_t[fidx[valid]] >= 0.0
        return grounded

    gm_f0 = face_grounded(f0)
    gm_f1 = face_grounded(f1)

    # GL edge: exactly one side grounded
    is_gl = gm_f0 ^ gm_f1                             # XOR: (n_edges,)
    gl_idx = np.where(is_gl)[0]                        # indices of GL edges

    if len(gl_idx) == 0:
        return np.zeros(len(basin_ids))

    # ── Determine outward normal (from grounded toward floating) ─────────────
    # For each GL edge, identify which face is grounded
    grounded_face = np.where(gm_f0[gl_idx], f0[gl_idx], f1[gl_idx])   # (n_gl,)

    # Vector from edge midpoint → grounded face centroid
    to_grounded = face_xy[grounded_face] - edge_mid[gl_idx]            # (n_gl, 2)

    # Outward normal = the candidate normal with negative dot product toward grounded centroid
    # (i.e., the one pointing AWAY from the grounded face)
    dot_left = np.sum(normal_left[gl_idx] * to_grounded, axis=1)       # (n_gl,)
    # If dot_left < 0 → left normal already points away from grounded → use it
    n_out = np.where(
        dot_left[:, None] < 0,
        normal_left [gl_idx],
        normal_right[gl_idx],
    )                                                                    # (n_gl, 2)

    # ── Edge-averaged velocity and thickness ─────────────────────────────────
    ni = n0_idx[gl_idx]                                 # (n_gl,) node indices
    nj = n1_idx[gl_idx]

    u_edge = 0.5 * (u_node_t[ni] + u_node_t[nj])       # (n_gl,)
    v_edge = 0.5 * (v_node_t[ni] + v_node_t[nj])
    h_edge = 0.5 * (h_node_t[ni] + h_node_t[nj])

    # Normal velocity (positive = outward from grounded domain)
    v_norm = u_edge * n_out[:, 0] + v_edge * n_out[:, 1]               # (n_gl,)

    # Volumetric flux [m² yr⁻¹ × m = m³ yr⁻¹]
    flux = np.maximum(0.0, v_norm) * h_edge * edge_len[gl_idx]         # (n_gl,)

    # ── Assign each GL edge to a basin ───────────────────────────────────────
    # Use the basin of the grounded adjacent face
    # (boundary face → grounded_face already resolved to a valid face index)
    edge_basin = basins[grounded_face]                                  # (n_gl,)

    # ── Sum per basin [m³ yr⁻¹] → [Gt yr⁻¹] ─────────────────────────────────
    gl_per_basin = np.zeros(len(basin_ids))
    for i, bid in enumerate(basin_ids):
        mask = edge_basin == bid
        gl_per_basin[i] = np.nansum(flux[mask]) * RHO_ICE * GT

    return gl_per_basin


def calving_edge_flux(gm_face_t, u_node_t, v_node_t, h_node_t):
    """
    Calving flux via boundary edges adjacent to floating faces.
    Analogous to gl_discharge_one_step but for the ice front.
    Boundary edges: efc[:,0] or efc[:,1] == -1  (domain boundary)
    """
    # Boundary edges: one side has no face (-1)
    f0, f1   = efc[:, 0], efc[:, 1]
    is_bdy   = (f0 < 0) | (f1 < 0)

    # The valid adjacent face index
    valid_f  = np.where(f0 >= 0, f0, f1)             # (n_edges,)

    # Floating mask for that face
    is_float = np.zeros(len(efc), dtype=bool)
    is_float[is_bdy] = (gm_face_t[valid_f[is_bdy]] < 0.0)

    # Calving front edges: boundary AND floating
    is_cf    = is_bdy & is_float
    cf_idx   = np.where(is_cf)[0]

    if len(cf_idx) == 0:
        return 0.0, np.zeros(len(basin_ids))

    # Outward normal: points away from the ice (away from the valid face)
    to_face  = face_xy[valid_f[cf_idx]] - edge_mid[cf_idx]   # (n_cf, 2)
    dot_left = np.sum(normal_left[cf_idx] * to_face, axis=1)
    n_out    = np.where(dot_left[:, None] < 0,
                        normal_left[cf_idx], normal_right[cf_idx])

    # Edge-averaged velocity and thickness
    ni = n0_idx[cf_idx]; nj = n1_idx[cf_idx]
    u_cf = 0.5 * (u_node_t[ni] + u_node_t[nj])
    v_cf = 0.5 * (v_node_t[ni] + v_node_t[nj])
    h_cf = 0.5 * (h_node_t[ni] + h_node_t[nj])

    v_norm   = u_cf * n_out[:, 0] + v_cf * n_out[:, 1]
    flux     = np.maximum(0.0, v_norm) * h_cf * edge_len[cf_idx]  # m³/yr per edge

    # Total calving [Gt/yr]
    total_calving = flux.sum() * RHO_ICE * GT

    # Per basin (using the adjacent floating face's basin ID)
    edge_basin    = basins[valid_f[cf_idx]]
    calv_per_basin = np.zeros(len(basin_ids))
    for i, bid in enumerate(basin_ids):
        mask = edge_basin == bid
        calv_per_basin[i] = np.nansum(flux[mask]) * RHO_ICE * GT

    return total_calving, calv_per_basin


# ── Member loading & diagnostics ─────────────────────────────────────────────
def load_member_fields(fpath=None, kind=None) -> dict:
    """Open one member (any kind) -> dict of normalized (time, n_nodes/faces)
    float32 arrays (halves peak memory vs. the netCDF's native float64 -- the
    OCX case briefly holds two large files open at once)."""
    kind = kind or _cfg.member_kind
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    if kind == "ocx":
        uds_s = xu.open_dataset(str(_cfg.ocx_states_file),  mask_and_scale=True, decode_times=False)
        uds_f = xu.open_dataset(str(_cfg.ocx_forcing_file), mask_and_scale=True, decode_times=False)
        u_raw, v_raw = f32(uds_s["ssavelocity_x"].values), f32(uds_s["ssavelocity_y"].values)
        fields = dict(
            h_node    = f32(uds_s["h"].values),
            gm_node   = f32(uds_s["groundedmask"].values),
            bed_node  = f32(uds_s["bedrock"].values),
            u_node    = np.where(np.abs(u_raw) < V_FILL_THRESHOLD, u_raw, 0.0).astype(np.float32),
            v_node    = np.where(np.abs(v_raw) < V_FILL_THRESHOLD, v_raw, 0.0).astype(np.float32),
            calv_face = f32(np.nan_to_num(uds_s["calving_front_flux"].values, nan=0.0)),
            haf_node  = f32(uds_s["haf"].values) if "haf" in uds_s else None,
            bmb_node  = f32(uds_f["bmb"].values),
            smb_face  = f32(uds_f["smb_total_flux"].values),   # already face-located
            smb_node  = None,
        )
        uds_s.close(); uds_f.close()
        return fields
    else:
        fpath = fpath or member_files[0]
        uds = xu.open_dataset(str(fpath), mask_and_scale=True)
        u_raw, v_raw = f32(uds["ssavelocity 1"].values), f32(uds["ssavelocity 2"].values)
        fields = dict(
            h_node    = f32(uds["h"].values),
            gm_node   = f32(uds["groundedmask"].values),
            bed_node  = f32(uds["bedrock"].values),
            u_node    = np.where(np.abs(u_raw) < V_FILL_THRESHOLD, u_raw, 0.0).astype(np.float32),
            v_node    = np.where(np.abs(v_raw) < V_FILL_THRESHOLD, v_raw, 0.0).astype(np.float32),
            calv_face = f32(np.nan_to_num(uds["calving_front_flux"].values, nan=0.0)),
            haf_node  = None,
            bmb_node  = f32(uds["bmb"].values),
            smb_face  = None,
            smb_node  = f32(uds["smb"].values),
        )
        uds.close()
        return fields


def compute_member_diagnostics(fields: dict, gl_melt: str = "fmp") -> dict:
    """
    Per-basin mass budget diagnostics for one member, from normalized `fields`
    (see load_member_fields). Math identical regardless of source format.
    Loops one YEAR at a time (not vectorized over all years) to keep peak
    memory to ~O(n_faces) rather than O(n_time * n_faces) -- the OCX case
    already holds two large files' worth of (35, ~1M) arrays in `fields`.

    BMB conventions (confirmed from Fortran SSAEffectiveBMB):
        - Fully grounded faces (all nodes grounded, gm_min >= 0): BMB = 0
        - GL-straddling faces (mixed nodes): fmp -> BMB as stored, nmp -> BMB = 0
        - Fully floating faces: BMB applied as stored
    """
    h_node, gm_node, bed_node = fields["h_node"], fields["gm_node"], fields["bed_node"]
    u_node, v_node, calv_face = fields["u_node"], fields["v_node"], fields["calv_face"]
    bmb_node = fields["bmb_node"]
    smb_face_all, smb_node = fields.get("smb_face"), fields.get("smb_node")
    haf_node = fields.get("haf_node")
    a = face_areas                                        # (n_faces,) static

    out = {k: np.zeros((n_time, n_basins)) for k in [
        "iaf_mass", "grounded_mass", "shelf_mass", "grounded_area", "floating_area",
        "smb_grounded", "smb_floating", "bmb_grounded", "bmb_floating",
        "calving_face", "calving_edge", "gl_discharge",
    ]}

    for t in range(n_time):
        h_face_t   = node_to_face(h_node[t],   fnc)
        gm_face_t  = node_to_face(gm_node[t],  fnc)
        bmb_face_t = node_to_face(bmb_node[t], fnc)
        bed_face_t = node_to_face(bed_node[t], fnc)
        smb_face_t = smb_face_all[t] if smb_face_all is not None else node_to_face(smb_node[t], fnc)

        gm_min_t, gm_max_t = node_to_face_minmax(gm_node[t], fnc)
        fully_grounded_t = gm_min_t >= 0                    # BMB = 0 here (confirmed)
        gl_straddle_t    = (gm_min_t < 0) & (gm_max_t >= 0)  # mixed nodes

        grounded_t = gm_face_t >= 0.0
        floating_t = ~grounded_t

        bmb_face_t = np.where(fully_grounded_t, 0.0, bmb_face_t)
        if gl_melt == "nmp":
            bmb_face_t = np.where(gl_straddle_t, 0.0, bmb_face_t)

        if haf_node is not None:                            # OCX: use Elmer's own haf
            haf_face_t = node_to_face(haf_node[t], fnc)
            h_af_t = np.where(grounded_t, np.maximum(0.0, haf_face_t), 0.0)
        else:                                                # SmallEnsTrans: compute it
            h_fl_t = np.maximum(0.0, -bed_face_t * (RHO_SW / RHO_ICE))
            h_af_t = np.where(grounded_t, np.maximum(0.0, h_face_t - h_fl_t), 0.0)

        iaf_per_face_t       = h_af_t                             * a * RHO_ICE * GT
        grounded_mass_face_t = np.where(grounded_t, h_face_t, 0.) * a * RHO_ICE * GT
        shelf_mass_face_t    = np.where(floating_t, h_face_t, 0.) * a * RHO_ICE * GT

        gl_t = gl_discharge_one_step(gm_face_t, u_node[t], v_node[t], h_node[t],
                                     basins, basin_ids)
        _, calv_edge_t = calving_edge_flux(gm_face_t, u_node[t], v_node[t], h_node[t])

        for i, bid in enumerate(basin_ids):
            m  = basins == bid
            mg = m & grounded_t
            mf = m & floating_t
            out["iaf_mass"]      [t, i] = np.nansum(iaf_per_face_t      [m])
            out["grounded_mass"] [t, i] = np.nansum(grounded_mass_face_t[m])
            out["shelf_mass"]    [t, i] = np.nansum(shelf_mass_face_t   [m])
            out["grounded_area"] [t, i] = np.nansum(a[mg]) * 1e-6
            out["floating_area"] [t, i] = np.nansum(a[mf]) * 1e-6
            out["smb_grounded"]  [t, i] = np.nansum(smb_face_t[mg] * a[mg]) * RHO_ICE * GT
            out["smb_floating"]  [t, i] = np.nansum(smb_face_t[mf] * a[mf]) * RHO_ICE * GT
            out["bmb_grounded"]  [t, i] = np.nansum(bmb_face_t[mg] * a[mg]) * RHO_ICE * GT
            out["bmb_floating"]  [t, i] = np.nansum(bmb_face_t[mf] * a[mf]) * RHO_ICE * GT
            out["calving_face"]  [t, i] = np.nansum(calv_face[t, mf] * a[mf]) * RHO_ICE * GT
            out["calving_edge"]  [t, i] = calv_edge_t[i]
            out["gl_discharge"]  [t, i] = gl_t[i]

    return out


def member_diag_dataset(fpath=None, kind=None, gl_melt=None, label=None) -> xr.Dataset:
    """Load one member (any kind) -> (time, catchment) diagnostic dataset."""
    kind = kind or _cfg.member_kind
    gl_melt = gl_melt or _cfg.member_glmelt
    fields = load_member_fields(fpath, kind=kind)
    diag   = compute_member_diagnostics(fields, gl_melt=gl_melt)
    if label is None:
        label = "OCX C011" if kind == "ocx" else Path(fpath or member_files[0]).parent.name
    ds1 = xr.Dataset(
        {k: (["time", "catchment"], v) for k, v in diag.items()},
        coords={"time": years, "catchment": basin_ids},
        attrs={"label": label},
    )
    ds1["smb"]          = ds1["smb_grounded"] + ds1["smb_floating"]
    ds1["mass_balance"] = ds1["smb_grounded"] - ds1["gl_discharge"]   # grounded MB
    return ds1


# ── Observed discharge (BMHF14 / IMBIE) ──────────────────────────────────────
def load_obs_discharge(path=None):
    """(obs_years, d[time,18], derr[time,18]) mapped to catchments 1..18, or None
    if the file is absent."""
    path = path or _cfg.obs_discharge_file
    if not Path(path).exists():
        print(f"[obs] {path} not on disk -> discharge obs skipped.")
        return None
    dso = xr.open_dataset(path, decode_times=False)   # raw days-since-1950, avoids cftime
    btype = dso["basin_type"].values
    imbie = btype == "imbie"
    oyears = dso["time"].values / 365.25 + 1950.0      # BMHF14 covers ~1996-2025
    sel = (oyears >= years.min() - 1) & (oyears <= years.max() + 1)
    d  = dso["discharge_mean"].values[sel][:, imbie]      # (nt, 18)
    de = dso["discharge_error"].values[sel][:, imbie]
    dso.close()
    return oyears[sel], d, de


# ── 2D map helpers (obs-on-mesh, catchment masking, model snapshots) ─────────
def get_catchment_mask(catchment: int) -> np.ndarray:
    """Boolean node mask. (n_nodes,)"""
    return uds_obs["node_basin"].values == catchment


def get_catchment_bounds(catchment: int, pad_frac: float = 0.05) -> tuple:
    """(xlim, ylim) from node coordinates."""
    mask = get_catchment_mask(catchment)
    x_c, y_c = node_x[mask], node_y[mask]
    if len(x_c) == 0:
        raise ValueError(f"No nodes for catchment {catchment}")
    pad_x = (x_c.max() - x_c.min()) * pad_frac
    pad_y = (y_c.max() - y_c.min()) * pad_frac
    return ((x_c.min()-pad_x, x_c.max()+pad_x), (y_c.min()-pad_y, y_c.max()+pad_y))


def get_obs_velocity_at_year(year: float):
    t = int(np.argmin(np.abs(uds_obs.coords["vel_time"].values - year)))
    return uds_obs["velocity"].isel(vel_time=t), f"{uds_obs.coords['vel_time'].values[t]:.1f}"


def get_obs_dhdt_at_year(year: float):
    t = int(np.argmin(np.abs(uds_obs.coords["dhdt_time"].values - year)))
    return uds_obs["dhdt"].isel(dhdt_time=t), f"{uds_obs.coords['dhdt_time'].values[t]:.1f}"


_reprojected_cache = {}   # (fpath, kind) -> reprojected xu.Dataset, built at most once


def _reproject_mesh(fpath, kind):
    """Raw XIOS UGRID files store mesh2D_node_x/y as lon/lat degrees (not the
    separate projected x/y data vars) -- .ugrid.plot() renders straight from the
    mesh topology, so it must be reprojected or plots come out blank/misaligned.
    Reprojecting a ~1M-node mesh is expensive -- cache per file so functions that
    loop over years/members only pay this cost once per file, not once per call."""
    key = (str(fpath), kind)
    if key not in _reprojected_cache:
        decode_times = False if kind == "ocx" else True
        uds_m = xu.open_dataset(str(fpath), mask_and_scale=True, decode_times=decode_times)
        uds_m.ugrid.set_crs("EPSG:4326")
        _reprojected_cache[key] = uds_m.ugrid.to_crs("EPSG:3031")
    return _reprojected_cache[key]


def load_model_snapshot_ugrid(fpath, variable: str, t_idx: int, kind: str = None):
    """One time-step node-located snapshot, any format, no node->face conversion."""
    kind = kind or _cfg.member_kind
    uds_m = _reproject_mesh(fpath, kind)
    if kind == "ocx":
        if variable == "velocity":
            u = uds_m["ssavelocity_x"].isel(time=t_idx).values
            v = uds_m["ssavelocity_y"].isel(time=t_idx).values
            u = np.where(np.abs(u) < V_FILL_THRESHOLD, u, np.nan)
            v = np.where(np.abs(v) < V_FILL_THRESHOLD, v, np.nan)
            vals = np.sqrt(u**2 + v**2).astype(np.float32)
            da = uds_m["ssavelocity_x"].isel(time=t_idx).copy(data=vals)
        elif variable == "dhdt":
            da = uds_m["dhdt"].isel(time=t_idx)         # written directly by XIOS
        elif variable == "zs":
            da = uds_m["zs"].isel(time=t_idx)
    else:
        if variable == "velocity":
            u = uds_m["ssavelocity 1"].isel(time=t_idx).values
            v = uds_m["ssavelocity 2"].isel(time=t_idx).values
            u = np.where(np.abs(u) < V_FILL_THRESHOLD, u, np.nan)
            v = np.where(np.abs(v) < V_FILL_THRESHOLD, v, np.nan)
            vals = np.sqrt(u**2 + v**2).astype(np.float32)
            da = uds_m["ssavelocity 1"].isel(time=t_idx).copy(data=vals)
        elif variable == "dhdt":
            da = uds_m["h velocity"].isel(time=t_idx)   # differentiated by XIOS
        elif variable == "zs":
            da = uds_m["zs"].isel(time=t_idx)

    return da                          # (n_nodes,) UgridDataArray


def plot_catchment_field(ax, da, catchment: int, xlim, ylim, cmap, vmin, vmax):
    """Mask to catchment, clip to its bounding box, and plot with xugrid's own
    plotting.

    NOTE: `.ugrid.plot()` raises `TypeError: tripcolor() takes 4 positional
    arguments but 5 were given` for node-located fields when the axes come from
    proplot 0.9.7 (its axes wrapper mis-forwards xugrid's Gouraud-shaded
    tripcolor call). It works fine on a *plain* matplotlib Axes -- so map panels
    must use `plt.subplots()`, NOT `pplt.subplots()`.

    Calling `.ugrid.plot()` on the full ~1-2M-face continental mesh (even
    NaN-masked outside the catchment) rebuilds the whole triangulation every
    time: ~10s/call, no caching. `.ugrid.clip_box()` first cuts that to ~1s/call
    -- it builds a spatial index once per underlying grid (~10-15s on the very
    first call) and reuses it for every later clip against that same grid.
    """
    catch_mask = (uds_obs["node_basin"] == catchment).values
    da_masked = da.copy(data=np.where(catch_mask, da.values, np.nan).astype(np.float32))
    da_clip = da_masked.ugrid.clip_box(xlim[0], ylim[0], xlim[1], ylim[1])
    p = da_clip.ugrid.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("")
    return p

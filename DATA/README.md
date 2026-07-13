# DATA

Put the data files here. Nothing in this folder is committed (`*.nc` is
gitignored) — the files are far too large for git, and the observations are
public and better downloaded from source.

## Download these

| File | What | Size | Where from |
|---|---|---|---|
| `AntarcticaObsISMIP7-v1.2.nc` | Satellite velocity time series, dh/dt, BedMachine surface, Mouginot basins | ~9 GB | [ISMIP7 observations focus group](https://www.ismip.org/participants/focus-groups/observations) — "no restrictions on access or use" |
| `AIS_discharge_BMHF14.nc` | Per-catchment grounding-line discharge (IMBIE basins) | ~67 MB | Davison et al., [doi:10.5281/zenodo.10051893](https://doi.org/10.5281/zenodo.10051893) (v7.0) |

`AIS_discharge_BMHF14.nc` is optional: without it the notebooks simply skip the
observed-discharge overlays instead of failing.

## Your own model output

Put your XIOS output here too (or point `CONFIG` at wherever it lives):

- `<run>_states.nc` — geometry, velocity, `groundedmask`, `haf`, `cell_area`
- `<run>_forcing.nc` — `smb_total_flux` (face), `bmb` (node), `ligroundf`

## Generated here by preprocessing

Built by `preprocessing/build_mesh_products.py`, one set per mesh — do not
download these, generate them:

- `basins_mouginotGrid_<tag>.nc` — Mouginot catchments on the mesh faces
- `obs_on_elmer_mesh_<tag>.nc` — observations on the mesh nodes (plain netCDF)
- `obs_on_elmer_mesh_pv_<tag>.nc` — same, UGRID/ParaView-readable; **this is the one the notebooks read**

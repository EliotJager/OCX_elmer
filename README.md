# OCX_elmer

Post-processing / model-vs-observation analysis for the ISMIP7 Antarctic Ice
Sheet OCX experiment (Elmer/Ice, SSA), run on CSC LUMI/Mahti.

## What's here

- **`AnalyseMemberVsObs.ipynb`** -- the analysis notebook: compares a model
  member's simulated ice-sheet state (velocity, dh/dt, mass balance by
  catchment, grounding-line discharge, ...) against satellite observations
  regridded onto the same Elmer/Ice mesh. Supports two mesh targets via a
  config toggle at the top (`MEMBER_KIND = "ocx"` or `"smallenstrans"`).
  §1a plots can save PNGs to `figures/` (git-ignored, not included in this
  repo). Edited and committed directly -- there is no build step.
  The first code cell (`# ── EDIT ME ──`) is the single place to point the
  notebook at your own data: in particular `OCX_STATES_FILE`/`OCX_FORCING_FILE`
  should be set to your own run's XIOS output.
- **`build_ocx_mesh_products.py`** -- regrids the raw satellite observations
  (`AntarcticaObsISMIP7-v1.2.nc`) and Mouginot basin definitions onto the OCX
  Elmer mesh, producing the `obs_on_elmer_mesh*_ocx.nc` /
  `basins_mouginotGrid_ocx.nc` files the notebook reads. Only needs rerunning
  if your Elmer mesh differs from the one these files were built for (different
  node/face count). Paths are hardcoded near the top of the file -- edit them
  before rerunning.
- **`fix_ocx_basins.py`**, **`fix_smallenstrans_obs_velocity.py`** -- one-off
  fix scripts, kept for provenance/reproducibility of two bugs found and
  patched in the regridded obs products (a basin-regrid CRS bug, and a frozen
  `velocity` field caused by a stale interpolation run). See each script's
  docstring for details.
- **`environment.yml`** -- conda environment (`postpro_ocx`) used to run the
  notebook and scripts. Recreate with:
  ```bash
  conda env create -f environment.yml
  conda activate postpro_ocx
  ```

## Dependencies not in this repo

- **ElmerUgrid** -- registers the `.ugrid.to_netcdf_forpv` accessor used by
  `build_ocx_mesh_products.py`. Base tool by Fabien Gillet-Chaulet:
  https://gricad-gitlab.univ-grenoble-alpes.fr/gilletcf/elmerugrid.git
  Eliot's working copy has local, not-yet-upstreamed modifications (and isn't
  currently in a clean, publishable state) -- if you hit reproducibility
  issues, ask Eliot Jager for his exact copy rather than assuming the public
  repo alone matches.

## Data (not included -- download separately)

Data files (`*.nc`: raw observations, regridded obs/basin products, ensemble
aggregates) and generated figures are excluded via `.gitignore` -- they're
large (tens of MB to several GB) and live only on the working disk, not in
this repo.

- **`AntarcticaObsISMIP7-v1.2.nc`** (velocity, dh/dt, BedMachine surface;
  ~9 GB) -- ISMIP7 observations dataset for Antarctica, Mathieu Morlighem on
  behalf of the ISMIP7 observations group. "No restrictions on access or
  use." https://www.ismip.org/participants/focus-groups/observations
  Place at `../DATA/AntarcticaObsISMIP7-v1.2.nc` relative to this directory
  (i.e. `postpro/DATA/`).
- **`AIS_discharge_BMHF14.nc`** (per-catchment grounding-line discharge;
  ~67 MB) -- Benjamin Davison (University of Leeds / University of
  Sheffield), product v7.0. https://doi.org/10.5281/zenodo.10051893
  Place at `../DATA/AIS_discharge_BMHF14.nc`. If missing, the notebook skips
  the discharge-vs-obs plots gracefully rather than failing.
- Model output (`OCX_STATES_FILE`/`OCX_FORCING_FILE` in the notebook's config
  cell), and the regridded basin/obs products (`basins_mouginotGrid_ocx.nc`,
  `obs_on_elmer_mesh*_ocx.nc`) -- ask Eliot Jager, or regenerate the latter
  with `build_ocx_mesh_products.py` against your own mesh.

## Context

This is part of the ISMIP7 Antarctic Ice Sheet projections effort using
Elmer/Ice. See the parent project for the forward-model setup; this repo
covers Phase 2 (model-vs-observation validation) of that work.

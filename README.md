# OCX_elmer

Model-vs-observation analysis for ISMIP7 Antarctic Ice Sheet runs with
**Elmer/Ice** (SSA), as run for the OCX experiment on CSC LUMI/Mahti.

Takes a run's XIOS output and compares it against satellite observations on the
model's own unstructured mesh: grounding-line discharge and mass balance per
Mouginot catchment, and 2D velocity / dH/dt maps. A second notebook does
parameter-effect and Sobol sensitivity analysis across an ensemble.

## Layout

```
├── environment.yml            conda environment (postpro_ocx)
├── DATA/                      all data lives here — nothing committed (see DATA/README.md)
├── preprocessing/
│   └── build_mesh_products.py regrid observations + basins onto YOUR mesh
└── jupyter/
    ├── elmer_analysis.py      shared setup: mesh geometry, flux integrators, diagnostics
    ├── AnalyseMemberVsObs.ipynb   one run vs observations
    └── AnalyseEnsemble.ipynb      ensemble: parameter effects & Sobol indices
```

## Getting started

**1. Environment**

```bash
conda env create -f environment.yml
conda activate postpro_ocx
```

**2. ElmerUgrid** (external dependency, not on PyPI). Base tool by Fabien
Gillet-Chaulet:
<https://gricad-gitlab.univ-grenoble-alpes.fr/gilletcf/elmerugrid.git>. It
registers the `.ugrid.to_netcdf_forpv` accessor used to write UGRID output.

> Note: Eliot's working copy carries local modifications that are not yet
> upstreamed, so the public repo alone may not reproduce these results exactly.
> Ask Eliot if you hit trouble.

**3. Data** — download the observations into `DATA/`, and put your XIOS output
there too. See [`DATA/README.md`](DATA/README.md) for links and file
descriptions.

**4. Preprocessing** — the observations and basin definitions have to be
interpolated onto your mesh once (every mesh has a different node/face count):

```bash
python preprocessing/build_mesh_products.py \
    --states     DATA/<your_run>_states.nc \
    --obs        DATA/AntarcticaObsISMIP7-v1.2.nc \
    --elmerugrid /path/to/elmerugrid \
    --outdir     DATA \
    --tag        myrun
```

This writes `basins_mouginotGrid_myrun.nc` and `obs_on_elmer_mesh_pv_myrun.nc`
into `DATA/`. It takes a while — the raw observation file is ~9 GB.

**5. Analyse** — open `jupyter/AnalyseMemberVsObs.ipynb` and edit the single
`CONFIG` cell at the top to point at your run and the files you just built:

```python
CONFIG = ea.Config(
    postpro_dir      = "/path/to/elmerugrid",
    member_kind      = "ocx",
    ocx_states_file  = "../DATA/<your_run>_states.nc",
    ocx_forcing_file = "../DATA/<your_run>_forcing.nc",
    basins_file      = "../DATA/basins_mouginotGrid_myrun.nc",
    obs_mesh_file    = "../DATA/obs_on_elmer_mesh_pv_myrun.nc",
)
```

Nothing else in the notebook needs touching. Figures are written to
`jupyter/figures/` (created automatically, not committed).

## Notes

- `elmer_analysis.py` holds everything the two notebooks share — mesh geometry,
  node→face helpers, the grounding-line / calving flux integrators, and the
  per-catchment mass-budget diagnostics. `ea.init(CONFIG)` loads the mesh, basins
  and observations, and returns those names into the notebook's namespace.
- `member_kind="smallenstrans"` switches to the SmallEnsTrans ensemble output
  format (different variable names; cell areas computed rather than read).
- Map panels use plain `plt.subplots()`, **not** proplot: xugrid's `.ugrid.plot()`
  breaks on a proplot axes (`tripcolor() takes 4 positional arguments but 5 were
  given`).

## Context

Part of the ISMIP7 Antarctic Ice Sheet projections effort with Elmer/Ice
(Phase 2: model-vs-observation validation).

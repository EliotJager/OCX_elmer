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

**2. ElmerUgrid** — external dependency (not on PyPI), by Fabien Gillet-Chaulet.
It registers the `.ugrid.to_netcdf_forpv` accessor used to write UGRID output.

```bash
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/gilletcf/elmerugrid.git
```

Pass its path to `--elmerugrid` (preprocessing) and `elmerugrid_dir` (notebooks).
The current `main` branch is what this repo is tested against.

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
`CONFIG` cell at the top. Declare one entry in `RUNS` per run you want to
compare; every per-catchment plot then draws one line per run:

```python
RUNS = {
    "baseline":    ("../DATA/states_baseline.nc",    "../DATA/forcing_baseline.nc"),
    "fric0_shelf": ("../DATA/states_fric0_shelf.nc", "../DATA/forcing_fric0_shelf.nc"),
}

CONFIG = ea.Config(
    elmerugrid_dir = "/path/to/elmerugrid",   # only used to import ElmerUgrid
    member_kind    = "ocx",
    runs           = RUNS,
    basins_file    = "../DATA/basins_mouginotGrid_myrun.nc",
    obs_mesh_file  = "../DATA/obs_on_elmer_mesh_pv_myrun.nc",
)
```

Nothing else in the notebook needs touching. Figures are written to
`jupyter/figures/` (created automatically, not committed).

The per-catchment mass budget takes minutes to compute (it reads the whole
states+forcing files) but is only a few hundred kB, so it is **cached** to
`jupyter/diag_cache/`. The cache records the exact source files it was built
from, so it is rebuilt automatically whenever an input changes *or* a run label
is repointed at a different file — you cannot silently plot stale numbers.
`load_runs(force=True)` rebuilds regardless.

## Interpolation: why it is not conservative

`build_mesh_products.py` uses two different methods, deliberately:

| What | Method | Located on |
|---|---|---|
| Mouginot basins | `xugrid.OverlapRegridder`, `method="mode"` (categorical) | faces |
| velocity, dh/dt, BedMachine surface | bilinear point-sampling (`scipy.RegularGridInterpolator`) | nodes |

**The observation fields are NOT conservatively interpolated, and that is the
right choice here.** Conservative (area-weighted) remapping preserves *integrals*
— you want it when a quantity must not be created or destroyed, e.g. remapping
SMB, thickness or any mass/flux field between meshes. These observations are used
for a *pointwise* comparison instead: "what does the satellite say the velocity is
**at this node**, versus what the model says there". For that, bilinear sampling of
the value at the node location is what you actually want; conservative averaging
would smear the observation over a cell and is not what is being asked.

It is also far cheaper. ElmerUgrid's interpolators (`node_2_node_interpolation`,
`face_2_face_interpolation`, ...) are **mesh-to-mesh**: they take a
`xu.UgridDataArray` source and do polygon-polygon clipping via a celltree index.
The observations are a structured raster of **12161 × 12161 = 148 million cells**
at 500 m, so feeding them to those functions would first require converting the
raster into a UGRID mesh of 148 M quad faces and then intersecting it against the
Elmer mesh — intractable. (This is exactly why the categorical basin regrid,
which genuinely does need an overlap method, coarsens the raster ×4 first, and is
still the slowest step in the script.) `RegularGridInterpolator` never builds a
mesh: it exploits the regular grid, so each of the ~929k node lookups is O(1).

Gap handling is deliberate too: the data are interpolated with missing values
filled as zero, a NaN-*fraction* field is interpolated alongside, and any node
where more than 50% of the surrounding cells were missing is masked back to NaN.
This tolerates small holes in the satellite coverage instead of letting one
missing pixel poison a node.

**Use a conservative method (ElmerUgrid / `OverlapRegridder`) instead whenever
the quantity is a mass or a flux** — e.g. interpolating SMB forcing or ice
thickness onto the mesh.

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

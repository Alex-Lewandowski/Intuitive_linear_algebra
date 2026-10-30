# Custom NISAR GUNW from local RSLCs

[`notebooks/nisar_gunw_from_rslcs.ipynb`](notebooks/nisar_gunw_from_rslcs.ipynb) makes a NISAR
L2 **GUNW** from two RSLC files on your disk with **ISCE3's NISAR InSAR workflow**
(`nisar.workflows.insar`, the code NISAR runs in production). It uses the **standard NISAR GUNW
processing parameters** from an official GUNW's runconfig:
[`config/nisar_gunw_solid_earth_cpu_20_05.yaml`](config/nisar_gunw_solid_earth_cpu_20_05.yaml).

**PLAnT-ISCE3** is used to inspect the input RSLCs (`plant_isce3_info`). It has no InSAR
processing and can't read GUNWs, so ISCE3 does all the processing.

## Quick start

```bash
conda env create -f environment.yml
conda activate nisar-gunw
jupyter lab notebooks/nisar_gunw_from_rslcs.ipynb
```

In section 1 of the notebook, set `REF_RSLC` and `SEC_RSLC`, then run all cells. With
`DEM_FILE = None`, it downloads the DEM for the scene with [sardem](https://github.com/scottstanie/sardem):

* `DEM_SOURCE = "NISAR"` (default): the NISAR DEM v1.2. Needs an Earthdata Login; the
  notebook asks once and saves it to `~/.netrc`.
* `DEM_SOURCE = "COP"`: Copernicus GLO-30, no login needed.

**ISCE3 0.25 or newer is required.** Older versions (e.g. 0.24) reject the standard
parameters, and the notebook stops with a clear message if it finds one.

## What the notebook does

1. **Inspects both RSLCs** with PLAnT-ISCE3. It checks that they pair: same track, pass
   direction and look direction; the polarization exists in both; the footprints overlap. It
   also checks whether frequency B is present, which the ionosphere step needs.
2. **Gets the DEM** with sardem, covering both footprints plus 5 km (NISAR DEM v1.2 or
   Copernicus GLO-30), or uses your own.
3. **Builds the runconfig** from the standard parameters. It fills in:
   * the inputs and output paths
   * the NISAR projection for the scene: UTM zone, or polar stereographic beyond ±75°
   * whether ionosphere and troposphere can run with the data you have

   It then **lists every processing setting that differs from the standard set**.
4. **Validates** the runconfig against ISCE3's schema and checks before the long run.
5. **Runs** `python -m nisar.workflows.insar` and prints each step as it completes. With
   `RESTART = False`, it continues an interrupted run from the last completed step.
6. **Plots** the GUNW: wrapped phase, coherence, unwrapped phase, connected components and
   ionosphere screen. It converts the unwrapped phase to LOS displacement and exports the main
   layers to GeoTIFF.

## Standard parameters

The official runconfig used to make the parameter file came from the NISAR parameter set
`LSAR_PARAMETER_SET/05020/INSAR/NISAR_ParameterSet_INSAR_20000101_solid_earth_cpu_20_05_v0.1.yaml`
(PGE r05.02.3, CRID P05023). The parameter file keeps its processing settings:

| Step | Setting |
|---|---|
| Input | frequency A HH |
| Dense offsets | 64 × 96 windows, ±32 search, 40 × 75 skip |
| Rubbersheet | polynomial, degree 2 |
| Fine resampling | on |
| Crossmul | 3 × 6 looks |
| Unwrapping | 7 × 16 looks, subswath-mask preprocessing, bridging |
| Ionosphere | `main_diff_ms_band` with the official dispersive filter |
| Troposphere | RAiDER with ECMWF HRES |
| Geocoding | 80 m / 20 m (wrapped, bilinear) |

It also keeps the worker and QA settings. Paths, ancillary files and the frame-specific
geocoding box are filled in per pair. One schema fix: `mask_type` is written as a list.

What a custom pair can't always match:

* **Geocoding box**: official GUNWs use a fixed box per track/frame. The notebook geocodes the
  whole reference scene instead, in the same projection and at the same postings.
* **Ionosphere**: needs frequency B (20+5 MHz modes) in both RSLCs. Otherwise it is switched off.
* **Troposphere**: needs ECMWF HRES weather-model files and RAiDER. Otherwise it is switched off.
* **Water mask and TEC file**: official runs use them, and you can add them. They aren't needed
  for the GUNW layers with these parameters.

## Notes on the environment

The conda-forge `isce3` package doesn't pull in three modules that the InSAR workflow imports:

* `pyaps3`: imported even when troposphere is off
* `snaphu`: the default unwrapper
* `whirlwind`: imported by the unwrap step

`environment.yml` installs all three. `whirlwind` isn't on conda-forge, so it is built from
GitHub, which needs the C/C++ compilers listed in the file.

## Testing

The notebook was run end to end with ISCE3 0.25.22 and PLAnT-ISCE3 on ISCE3's own test RSLC
pair (`SanAnd_129.h5` / `SanAnd_138.h5`, with `SanAnd_dem.tif`). Because that scene is only
150 × 200 samples, the run used smaller dense-offset windows, unwrapping looks and ionosphere
filter kernels via `OVERRIDES`. All other settings were standard.

* Every ISCE3 step completed: dense offsets, rubbersheet, fine resampling, crossmul,
  unwrapping, frequency-B side-band ionosphere, geocoding, solid earth tides and baseline.
* The GUNW and GeoTIFF exports were checked.

Not tested here (no network access to ASF from the build environment):

* the live sardem download (its NISAR path was run against a local copy of the VRT layout)
* a full-size NISAR pair

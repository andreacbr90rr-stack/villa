# Reproduce the Villa zero-fill SIMD test

The patch targets the CPU eight-lane AVX2 interpolation path for uint8 volumes. When all eight corners of a lane's trilinear cell fall inside one confirmed absent chunk with declared positive-zero fill, the lane can return zero without a scalar call. Temporarily unavailable chunks, other fill values, chunk boundaries, invalid coordinates and the level-selection policy keep their existing behavior.

## Build and run

Use two builds from the same Villa revision, compiler, CPU feature settings and optimization flags. The reference revision is `659e2fadb836d44063ed831d85517154ec852f11`. Follow the [Villa vc_reader build instructions](https://github.com/ScrollPrize/villa/tree/659e2fadb836d44063ed831d85517154ec852f11/vc_reader) for system dependencies and install each build into its own Python environment. Both environments need NumPy.

Build the stock revision above and this branch in separate checkouts. The candidate change is already applied to `vc_reader/src/cpp/grid/grid.cpp`.

After building and installing both variants, run these commands from this directory. Use the same machine and CPU affinity. Add `--public-cache ./ct_fixture` to **both** capture commands for the optional public-data check.

```sh
/path/to/stock-env/bin/python reproduce.py capture --output stock.npz
/path/to/candidate-env/bin/python reproduce.py capture --output candidate.npz
/path/to/stock-env/bin/python reproduce.py compare stock.npz candidate.npz
```

The default test uses deterministic synthetic local Zarr data. It compares raw float32 bits and level codes for present and absent chunks, fill 0 and 17, boundary cells, invalid coordinates, pixel masks, cold progressive reads, and a two-level fallback. It also times a fixed sparse workload after a warm-up render. Repeat with alternating build order to investigate timing noise. On a machine without AVX2, the correctness check can pass through the stock scalar path.

## Public CT control

The optional fixture uses one 128 × 128 historical surface window of PHerc1447, segment `20250502182456`, at Y=3968 and X=2688. It uses the 8.64 µm volume `20250521151220`, official TIFXYZ coordinates, the original 31-plane historical tile and two raw CT chunks. The repository contains only public object paths and checksums. The script downloads missing files from public S3, checks SHA-256 and renders locally. Expect 14 historically all-zero columns with recovered CT signal. The fixture tests bitwise parity and coordinate handling on a small real raster. Its footprint contains two present CT chunks, so the synthetic sparse workload exercises the missing-fill optimization directly.

Treat the timing output as a development measurement. Results depend on CPU, build flags, geometry and the prevalence of absent chunks. The public fixture establishes sample parity on its specified window. It provides no ink interpretation or whole-surface performance measurement.

Data attribution and terms are provided in [Vesuvius Challenge Data Formats](https://scrollprize.org/data#licenses). The upstream SIMD implementation is documented in [Villa PR #2536](https://github.com/ScrollPrize/villa/pull/2536).

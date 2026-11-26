#!/usr/bin/env python3
"""Compare an unmodified and a patched Villa installation offline."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from vc_reader import _reader, grid, tifxyz
from vc_reader.reader import ZarrReader

HERE = Path(__file__).resolve().parent


def read_volume(path):
    return ZarrReader.open(path, cache_dir=[], cache_writes=False, memory_cache=32 << 20, threads=2)


def make_volume(root, name, *, fill, shape=(48, 32, 32), side=16, keys=((0, 0, 0), (1, 0, 1), (0, 1, 0))):
    destination = root / name
    destination.mkdir()
    metadata = {"zarr_format": 2, "shape": shape, "chunks": [side] * 3, "dtype": "|u1",
                "compressor": None, "filters": None, "fill_value": fill, "order": "C", "dimension_separator": "/"}
    (destination / ".zarray").write_text(json.dumps(metadata) + "\n")
    rng = np.random.default_rng(1126)
    for coordinate in keys:
        p = destination.joinpath(*(str(i) for i in coordinate))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rng.integers(0, 255, size=(side, side, side), dtype=np.uint8).tobytes())
    return destination


def save_grid(data, name, result):
    data[name + "_float32_bits"] = result.values.view(np.uint32)
    data[name + "_level"] = result.level


def semantic_controls(root, output):
    x = np.concatenate((np.linspace(.25, 14.7, 8), [15.5], np.linspace(16.2, 30.25, 12),
                        [31.5, 32.25, -.5, -2.0, np.nan])).astype(np.float32)
    xyz = np.tile(np.array([0, 4.125, 5.25], dtype=np.float32), (16, x.size, 1))
    xyz[..., 0] = x
    xyz[1, :, 1] = 15.5
    xyz[2, :, 2] = 15.5
    xyz[3, :, 2] = 16.15
    xyz[4, :, 0] = 0.0
    xyz[5, :, 0] = 16.0
    xyz[6, :, 0] = 25.125
    xyz[7, :, 1] = -1.6
    xyz[8, :, 2] = np.nan
    normals = np.zeros_like(xyz)
    normals[..., 2] = 1
    pixels = np.ones(xyz.shape[:2], dtype=np.uint8)
    pixels[9, 1::3] = 0
    plane = ((0.5, 4.125, 5.25), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 28, 16)
    offsets = [-.25, 0, .5]

    for fill in (0, 17):
        source = make_volume(root, f"fill_{fill}", fill=fill)
        with read_volume(source) as reader:
            level = reader.grid_level()
            pending = grid.Pending()
            cold = grid.sample_grid_cached([level], plane=plane, order=1, offsets=offsets, pending=pending)
            assert np.any(cold.level == grid.LEVEL_MISSING) and not pending.empty
            output[f"fill_{fill}_pending"] = np.asarray(pending.reads, dtype=np.int64)
            grid.load(pending)
            warmed = grid.sample_grid_cached([level], plane=plane, order=1, offsets=offsets)
            regular = grid.sample_grid([level], plane=plane, order=1, offsets=offsets)
            assert np.array_equal(warmed.values.view(np.uint32), regular.values.view(np.uint32))
            assert np.array_equal(warmed.level, regular.level)
            sample = grid.sample_grid([level], xyz=xyz, normals=normals, order=1, offsets=offsets, pixels=pixels)
            assert (sample.level[8] == grid.LEVEL_NONE).all()
            assert (sample.level[9, 1::3] == grid.LEVEL_NONE).all()
            assert (sample.level[6] == 0).all()
            assert np.allclose(sample.values[:, 6], fill, atol=5e-6, rtol=0)
            for label, item in [("cold", cold), ("warm", warmed), ("regular", regular), ("points", sample)]:
                save_grid(output, f"fill_{fill}_{label}", item)

    with read_volume(root / "fill_0") as upper, read_volume(root / "fill_17") as lower:
        top, bottom = upper.grid_level(), lower.grid_level()
        grid.sample_grid([bottom], plane=plane, order=1, offsets=offsets)
        pending = grid.Pending()
        fallback = grid.sample_grid_cached([top, bottom], plane=plane, order=1, offsets=offsets, pending=pending)
        assert (fallback.level == 1).any() and not pending.empty
        grid.load(pending)
        restored = grid.sample_grid_cached([top, bottom], plane=plane, order=1, offsets=offsets)
        assert (restored.level == 0).any()
        save_grid(output, "two_levels_fallback", fallback)
        save_grid(output, "two_levels_restored", restored)


def sparse_benchmark(root, output, repeats):
    keys = [(0, y, x) for y in range(10) for x in range(12) if (y * 11 + x) % 7 == 0]
    volume = make_volume(root, "sparse_performance", fill=0, shape=(64, 320, 384), side=32, keys=keys)
    plane = ((.25, .375, 12.125), (.90, 0, 0), (0, 1.03, 0), (0, 0, 1), 384, 256)
    offsets = np.linspace(-9.0, 9.0, 31)
    with read_volume(volume) as reader:
        level = reader.grid_level()
        initial = grid.sample_grid([level], plane=plane, order=1, offsets=offsets)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = grid.sample_grid([level], plane=plane, order=1, offsets=offsets)
            times.append(time.perf_counter() - start)
        assert np.array_equal(result.values.view(np.uint32), initial.values.view(np.uint32))
        assert np.array_equal(result.level, initial.level)
        save_grid(output, "synthetic_sparse", result)
    return {"times_s":times, "median_s":statistics.median(times),"materialized_chunks":len(keys),
            "sample_count":int(result.values.size),"positive_sample_count":int(np.count_nonzero(result.values > 0))}


def public_fixture(cache):
    spec = json.loads((HERE / "public_fixture.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    for name, obj in spec["objects"].items():
        local = cache / name
        if "sha256" not in obj:
            assert not local.exists()
            try:
                with urlopen(Request(spec["data_origin"] + obj["key"]), timeout=45) as reply:
                    raise RuntimeError(f"An object declared absent returned {reply.status}: {obj['key']}")
            except HTTPError as exc:
                if exc.code != 404:
                    raise
            continue
        raw = local.read_bytes() if local.is_file() else None
        if raw is None:
            with urlopen(Request(spec["data_origin"] + obj["key"], headers={"Accept-Encoding": "identity"}), timeout=45) as reply:
                raw = reply.read()
        if sha256(raw).hexdigest() != obj["sha256"]:
            raise ValueError(f"Public fixture SHA-256 mismatch for {name}")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("xb") as stream:
                stream.write(raw)
    return spec


def render_public(cache, output):
    spec = public_fixture(cache)
    mesh = tifxyz.read(cache / "mesh")
    with read_volume(cache / "volume") as reader:
        out = grid.sample_grid([reader.grid_level()], surface=mesh, raster=spec["raster_uv"],
                               interpolation="bicubic", order=1, offsets=spec["depth_offsets"])
    raw = np.ascontiguousarray(out.values, dtype=np.float32)
    display = np.empty_like(raw, dtype=np.uint8)
    _reader.display_to_uint8(raw.ravel(), display.ravel(), scale=1, rounding="nearest_even", saturate_infinity=True)
    historical = np.fromfile(cache / "historical.u8", dtype=np.uint8).reshape(display.shape)
    repaired = int((~historical.any(axis=0) & display.any(axis=0)).sum())
    display_sha = sha256(display.tobytes()).hexdigest()
    assert repaired == spec["expected_restored_historical_columns"]
    assert display_sha == spec["expected_dense_u8_sha256"]
    save_grid(output, "pherc1447", out)
    output["pherc1447_display_u8"] = display
    return {"recovered_columns":repaired,"dense_display_sha256":display_sha,
            "staged_ct_chunks":sum(name.startswith("volume/") and "sha256" in obj and name != "volume/.zarray"
                                   for name,obj in spec["objects"].items())}


def capture(args):
    fields = {}
    with tempfile.TemporaryDirectory(prefix="villa-zero-fill-") as working:
        root = Path(working)
        semantic_controls(root, fields)
        timing = sparse_benchmark(root, fields, args.repeats)
    real = render_public(args.public_cache, fields) if args.public_cache is not None else None
    result = {"binary":Path(_reader.__file__).name,"native_binary_sha256":sha256(Path(_reader.__file__).read_bytes()).hexdigest(),
              "sampled_fields":list(fields),"synthetic_sparse":timing,"public_fixture":real}
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **fields)
    receipt = args.output.with_suffix(".json")
    with receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def compare(args):
    with np.load(args.stock, allow_pickle=False) as a, np.load(args.candidate, allow_pickle=False) as b:
        assert a.files == b.files, "Capture both installs with identical options"
        unequal = [key for key in a.files if not np.array_equal(a[key], b[key])]
    if unequal:
        raise SystemExit(f"FAIL: different fields: {', '.join(unequal)}")
    old,new = (json.loads(x.with_suffix(".json").read_text()) for x in (args.stock,args.candidate))
    stock = old["synthetic_sparse"]["median_s"]
    candidate = new["synthetic_sparse"]["median_s"]
    print(json.dumps({"bitwise_identical_fields":len(old["sampled_fields"]),"stock_synthetic_median_s":stock,
                      "candidate_synthetic_median_s":candidate,"synthetic_saved_fraction":1-candidate/stock,
                      "public_fixture":new["public_fixture"],"source_binaries_distinct":
                      old["native_binary_sha256"] != new["native_binary_sha256"]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--output",type=Path,required=True)
    c.add_argument("--public-cache",type=Path,help="Download and hash the optional public CT fixture if needed.")
    c.add_argument("--repeats",type=int,default=5)
    a = sub.add_parser("compare")
    a.add_argument("stock",type=Path)
    a.add_argument("candidate",type=Path)
    args = p.parse_args()
    if args.command == "capture":
        if args.repeats < 1:
            p.error("--repeats must be positive")
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

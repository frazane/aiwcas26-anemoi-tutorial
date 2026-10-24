#!/usr/bin/env python
"""Instructor prep for hour-2 option B1: cut a small Italy window out of the CERRA test sample.

The public CERRA sample (`scripts/download_data.sh`) stores every date as ONE zarr chunk of all
17 variables x 1,142,761 points (78 MB uncompressed). Every training sample needs 3 dates, so
the data loader would allocate, decompress and then throw away ~230 MB per sample, even after
`area`/`thinning`. This script writes the same dataset restricted to a rectangle of the 2-D
Lambert grid (rows x columns, *not* a lat/lon mask), so:

  * it is still an anemoi dataset with a 2-D `field_shape`: `open_dataset(..., area=..., thinning=N)`
    work on it exactly as on the full CERRA grid (thinning needs a 2-D field shape);
  * the native 5.5 km resolution is kept: students choose `thinning` and `area` themselves;
  * a date is ~6 MB instead of 78 MB.

Statistics (mean/stdev/min/max and the 6h/12h/1d tendency statistics) are recomputed over the
window for the statistics period of the original dataset (2017-01-01T06..01-08T18). The other
metadata is copied, with `field_shape`, `uuid`, `description` and `history` updated.

    python scripts/crop_cerra.py                                  # default window and paths
    python scripts/crop_cerra.py --box 49 4 34 21 --out /path/to/out.zarr

Run it on a login node (one date at a time, ~1-3 min). Standard anemoi env (zarr 2, numpy).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import zarr

DATA = Path(os.environ.get("WORK", "/leonardo_work/tra26_aiwcas")) / "a08trd25" / "data" / "anemoi"
SRC = DATA / "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr"
DST = DATA / "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing-italy.zarr"


def stats(sum_, sq, n, mn, mx):
    mean = sum_ / n
    var = np.maximum(sq / n - mean**2, 0.0)
    return {"mean": mean, "stdev": np.sqrt(var), "minimum": mn, "maximum": mx, "sums": sum_, "squares": sq,
            "count": np.full(sum_.shape, n, dtype="i8"), "has_nans": np.zeros(sum_.shape, dtype=bool)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DST))
    ap.add_argument("--box", nargs=4, type=float, default=[49.0, 4.0, 34.0, 21.0], metavar=("N", "W", "S", "E"),
                    help="every grid point in this lat/lon box is inside the window (default: Italy + margin)")
    args = ap.parse_args()

    src = zarr.open(args.src, mode="r")
    attrs = dict(src.attrs)
    ny, nx = attrs["field_shape"]
    lat = src["latitudes"][:].reshape(ny, nx)
    lon = src["longitudes"][:].reshape(ny, nx)
    n, w, s, e = args.box
    lon_e = np.where(lon > 180, lon - 360, lon)
    inside = (lat <= n) & (lat >= s) & (lon_e >= w) & (lon_e <= e)
    rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    shape2d = (r1 - r0, c1 - c0)
    npts = shape2d[0] * shape2d[1]
    print(f"window rows {r0}:{r1}, cols {c0}:{c1} -> field_shape {shape2d}, {npts} points "
          f"(lat {lat[r0:r1, c0:c1].min():.1f}..{lat[r0:r1, c0:c1].max():.1f}, "
          f"lon {lon_e[r0:r1, c0:c1].min():.1f}..{lon_e[r0:r1, c0:c1].max():.1f})")

    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    dst = zarr.open_group(str(tmp), mode="w")

    data = src["data"]
    nt, nv, ne, _ = data.shape
    comp = data.compressor
    d_out = dst.create_dataset("data", shape=(nt, nv, ne, npts), chunks=(1, nv, ne, npts), dtype=data.dtype,
                               compressor=comp, fill_value=data.fill_value)
    d_out.attrs.update(dict(data.attrs))
    for name in ("latitudes", "longitudes"):
        a = src[name][:].reshape(ny, nx)[r0:r1, c0:c1].ravel()
        dst.create_dataset(name, data=a, chunks=a.shape, compressor=comp).attrs.update(dict(src[name].attrs))
    dst.create_dataset("dates", data=src["dates"][:], compressor=comp).attrs.update(dict(src["dates"].attrs))

    dates = src["dates"][:]
    s0 = np.datetime64(attrs["statistics_start_date"])
    s1 = np.datetime64(attrs["statistics_end_date"])
    in_stats = (dates >= s0) & (dates <= s1)
    step_h = int(attrs["frequency"].rstrip("h"))
    lags = {"6h": 6 // step_h, "12h": 12 // step_h, "1d": 24 // step_h}

    # All big buffers are allocated ONCE and reused: on the Leonardo login nodes first-touch page
    # faults are so slow that allocating a fresh 78 MB array per date costs seconds of *system*
    # CPU time (a naive version hit the 600 s CPU limit after 2 minutes).
    acc = {k: [np.zeros(nv), np.zeros(nv), 0, np.full(nv, np.inf), np.full(nv, -np.inf)] for k in ["", *lags]}
    nlag = max(lags.values())
    raw = np.empty((nv, ne, ny, nx), dtype=data.dtype)            # one decoded source chunk
    win = np.empty((nv, ne, shape2d[0], shape2d[1]), dtype=data.dtype)
    ring = np.empty((nlag + 1, nv, ne * npts), dtype="f8")        # last dates of the window, float64
    work = np.empty((nv, ne * npts), dtype="f8")
    diff_buf = np.empty((nv, ne * npts), dtype="f8")
    chunk_dir = Path(args.src) / "data"
    sep = data._dimension_separator or "."

    def update(key, x):
        a = acc[key]
        a[0] += x.sum(axis=1)
        np.multiply(x, x, out=work)
        a[1] += work.sum(axis=1)
        a[2] += x.shape[1]
        a[3] = np.minimum(a[3], x.min(axis=1))
        a[4] = np.maximum(a[4], x.max(axis=1))

    for t in range(nt):
        blob = (chunk_dir / sep.join([str(t), "0", "0", "0"])).read_bytes()
        comp.decode(blob, out=raw)
        np.copyto(win, raw[:, :, r0:r1, c0:c1])
        flat = win.reshape(nv, ne, npts)
        assert not np.isnan(flat).any(), f"NaN at date {t}"
        d_out[t] = flat
        cur = ring[t % (nlag + 1)]
        np.copyto(cur, flat.reshape(nv, ne * npts))
        if in_stats[t]:
            update("", cur)
            for key, lag in lags.items():
                if t - lag >= 0 and in_stats[t - lag]:
                    np.subtract(cur, ring[(t - lag) % (nlag + 1)], out=diff_buf)
                    update(key, diff_buf)
        print(f"  date {t + 1}/{nt}", end="\r", flush=True)
    print()

    for key, (sum_, sq, cnt, mn, mx) in acc.items():
        prefix = "" if key == "" else f"statistics_tendencies_{key}_"
        for name, arr in stats(sum_, sq, cnt, mn, mx).items():
            dst.create_dataset(prefix + name, data=arr, compressor=comp)

    now = dt.datetime.now().isoformat()
    attrs.update({
        "field_shape": list(shape2d),
        "uuid": str(uuid.uuid4()),
        "latest_write_timestamp": now,
        "description": attrs.get("description", "") + (
            f"\nAIWCAS 2026 tutorial: window rows {r0}:{r1}, cols {c0}:{c1} of the 1069x1069 CERRA grid "
            f"(covers {n}N {w}E {s}N {e}E), made by scripts/crop_cerra.py; statistics recomputed on the window."),
    })
    hist = attrs.get("history", [])
    attrs["history"] = [*hist, {"action": "aiwcas26 crop_cerra.py", "timestamp": now,
                                "window": [int(r0), int(r1), int(c0), int(c1)], "source": str(args.src)}]
    dst.attrs.update(json.loads(json.dumps(attrs, default=str)))
    if (Path(args.src) / ".zmetadata").exists():
        zarr.consolidate_metadata(str(tmp))

    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    subprocess.run(["chmod", "-R", "u+rwX,go+rX", str(out)], check=True)   # group-readable, like the rest
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

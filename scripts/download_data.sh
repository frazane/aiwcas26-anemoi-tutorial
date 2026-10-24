#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Instructor prep for notebooks/01_datasets.ipynb and hour-2 option B1
# (AIWCAS 2026 Anemoi tutorial)
#
# Downloads small, public anemoi-format sample datasets from the ECMWF object
# store (the same bucket anemoi uses for its own tests and demos), and the
# Natural Earth coastlines cartopy needs, so the notebook runs fully offline
# (e.g. on compute nodes without internet).
#
# Run on a LOGIN node (needs internet). Total: ~560 MB, plus the CERRA sample
# for option B1 (987 MB download, 1.1 GB unpacked). Skip CERRA with CERRA=0.
# Re-running skips everything that is already complete.
#
#   module load python/3.11.7
#   source $SCRATCH/envs/anemoi/bin/activate
#   bash scripts/download_data.sh            # default target: $WORK/a08trd25/data
#   DATA_DIR=/some/where bash scripts/download_data.sh
# -----------------------------------------------------------------------------
set -euo pipefail

DATA_DIR="${DATA_DIR:-${WORK:?WORK not set}/a08trd25/data}"
mkdir -p "$DATA_DIR/anemoi" "$DATA_DIR/cartopy"

BUCKET="https://object-store.os-api.cci1.ecmwf.int/ml-tests"

DATASETS=(
  # Main dataset: ERA5, O96 octahedral reduced Gaussian grid (~1 deg),
  # 101 variables, 6-hourly, 2017-01-01T06 .. 2017-01-10T18 (39 dates). ~380 MB
  "test-data/samples/demo/datasets/aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing.zarr"
  # ERA5, O48 reduced Gaussian grid (~2 deg), same 101 variables, 1979-01-01..10, 6-hourly. ~110 MB
  # (used as the coarse "global" dataset in the cutout demo)
  "test-data/samples/demo/datasets/aifs-ea-an-oper-0001-mars-o48-1979-19-6h-v6-testset.zarr"
  # ERA5 from WeatherBench2 on a regular 1 deg lat/lon grid (field_shape 360x181), long CF-style
  # variable names, 3-hourly, 4 dates on 1979-01-01. ~65 MB (thinning + rename + cutout demos)
  "test-data/samples/demo/datasets/keisler-era5-gs-weatherbench2-1p0d-1979-1979-3h-v0.zarr"
  # Tiny 20-degree test datasets (< 1 MB each) used for join / concat / ensemble demos
  "test-data/samples/anemoi-datasets/aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-sfc.zarr"
  "test-data/samples/anemoi-datasets/aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-pl.zarr"
  "test-data/samples/anemoi-datasets/aifs-ea-an-oper-0001-mars-20p0-2016-2016-6h-v1.zarr"
  "test-data/samples/anemoi-datasets/aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1.zarr"
  "test-data/samples/anemoi-datasets/aifs-ea-an-enda-0001-mars-20p0-2017-2017-6h-v1.zarr"
)

for ds in "${DATASETS[@]}"; do
  name="$(basename "$ds")"
  if [[ -f "$DATA_DIR/anemoi/$name/.download-complete" ]]; then
    echo "[skip] $name (already downloaded)"
    continue
  fi
  echo "[get ] $name"
  # List every object under the prefix (S3 ListObjectsV2, paginated) and fetch it.
  python3 -I - "$BUCKET" "$ds/" "$DATA_DIR/anemoi/$name" <<'EOF'
import os, re, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

bucket, prefix, target = sys.argv[1:4]
keys, token = [], None
while True:
    q = {"list-type": "2", "prefix": prefix}
    if token:
        q["continuation-token"] = token
    xml = urllib.request.urlopen(f"{bucket}/?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
    keys += [k for k in re.findall(r"<Key>([^<]*)</Key>", xml) if not k.endswith("/")]
    m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
    if not m:
        break
    token = m.group(1)

def fetch(key):
    rel = key[len(prefix):]
    dst = os.path.join(target, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(f"{bucket}/{urllib.parse.quote(key)}", timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dst)

with ThreadPoolExecutor(16) as pool:
    list(pool.map(fetch, keys))
print(f"       {len(keys)} objects")
EOF
  touch "$DATA_DIR/anemoi/$name/.download-complete"
done

# Archives (.tgz of a whole zarr): fetched with curl (resumable), unpacked into a
# fresh staging directory, then moved into place. The tgz is deleted afterwards.
ARCHIVES=()
if [[ "${CERRA:-1}" != 0 ]]; then
  ARCHIVES+=(
    # CERRA regional reanalysis, 5.5 km Lambert-conformal grid over Europe (1069 x 1069 =
    # 1,142,761 points), 6-hourly, the SAME 39 dates as the O96 sample (2017-01-01T06..01-10T18),
    # 17 variables (msl, sp, t/r at 50 and 100 hPa, tp, sf + forcings). CC-BY-4.0. 987 MB tgz.
    # Used by hour-2 option B1 (Italy LAM, cut out into the O96 sample).
    "test-data/samples/anemoi-integration-tests/training/datasets/cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr"
  )
fi

for ds in ${ARCHIVES[@]+"${ARCHIVES[@]}"}; do
  name="$(basename "$ds")"
  dest="$DATA_DIR/anemoi/$name"
  if [[ -f "$dest/.download-complete" ]]; then
    echo "[skip] $name (already downloaded)"
    continue
  fi
  echo "[get ] $name.tgz"
  tgz="$DATA_DIR/anemoi/.$name.tgz"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$tgz" "$BUCKET/$ds.tgz"
  stage="$(mktemp -d "$DATA_DIR/anemoi/.unpack-XXXXXX")"
  tar -xzf "$tgz" -C "$stage" --no-same-owner --no-same-permissions
  src="$(find "$stage" -maxdepth 2 -type d -name "$name" | head -1)"
  [[ -n "$src" ]] || { echo "error: $name not found inside $tgz" >&2; exit 1; }
  rm -rf "$dest"
  mv "$src" "$dest"
  rm -rf "$stage" "$tgz"
  chmod -R u+rwX,go+rX "$dest"   # readable by the group (and others), like the rest of the data
  touch "$dest/.download-complete"
done

# Natural Earth coastlines/borders for cartopy (so plotting works offline).
echo "[get ] cartopy Natural Earth features"
python3 - "$DATA_DIR/cartopy" <<'EOF'
import sys
import cartopy
cartopy.config["data_dir"] = sys.argv[1]
import cartopy.feature as cfeature
for res in ("110m", "50m"):
    for feat in (cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAND):
        list(feat.with_scale(res).geometries())
print("       ok")
EOF

echo
du -sh --apparent-size "$DATA_DIR"/anemoi/* "$DATA_DIR/cartopy"
echo "Done. Data in $DATA_DIR"

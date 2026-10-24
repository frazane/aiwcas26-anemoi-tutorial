#!/bin/bash
# Build the shared student environment of the AIWCAS 2026 Anemoi tutorial (instructor only).
#
#   scripts/build_shared_env.sh [--name anemoi-vN] [--root DIR] [--lock] [--no-sqfs] [--activate]
#
# Builds, from the hashed lock env/requirements.txt (non-editable, released anemoi wheels):
#   $ROOT/<name>/        a normal, relocatable venv (login node, Jupyter kernel, VS Code, validate)
#   $ROOT/<name>.sqfs    a squashfs image of the same directory (mounted by the Slurm jobs, see
#                        scripts/anemoi_env.sh), so imports read one big file instead of ~57k
#                        small Lustre files
# and, with --activate, points the stable names $ROOT/anemoi and $ROOT/anemoi.sqfs at them.
# Everything is group $GROUP, readable, and not writable by the group (setgid dirs).
#
# Options:
#   --name NAME   env directory name (default anemoi-v1). Envs are immutable: a rebuild uses a NEW
#                 name, then --activate flips the symlinks. Running jobs keep their mounted image.
#   --root DIR    default /leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs ($FAST, SSD pool)
#   --lock        re-resolve env/requirements.in into env/requirements.txt first (keeps the
#                 existing pins unless you delete them or pass --upgrade through UV_LOCK_ARGS)
#   --no-sqfs     skip the squashfs image
#   --activate    point $ROOT/anemoi(.sqfs) at this build (after a successful build)
#
# Run it on a login node (needs internet). Takes ~10 min; uv cache in $SCRATCH/.cache/uv.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs
NAME=anemoi-v1
GROUP=tra26_aiwcas
DO_LOCK=0 DO_SQFS=1 DO_ACTIVATE=0
UV_VERSION=0.13.5
PYTHON_MODULE=python/3.11.7

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --lock) DO_LOCK=1; shift ;;
        --no-sqfs) DO_SQFS=0; shift ;;
        --activate) DO_ACTIVATE=1; shift ;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

DEST="$ROOT/$NAME"
IMG="$ROOT/$NAME.sqfs"
REQ_IN="$REPO_DIR/env/requirements.in"
LOCK="$REPO_DIR/env/requirements.txt"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export UV_LINK_MODE=copy             # cache and env are on different filesystems
export UV_PYTHON_DOWNLOADS=never     # always use the module python
umask 0022

log() { echo "== $(date +%H:%M:%S) $*"; }

module load "$PYTHON_MODULE" >/dev/null 2>&1
PYTHON="$(command -v python3.11)"
log "python: $PYTHON"

# uv: a pinned copy in $SCRATCH (not inside the shared env)
UV="${UV:-}"
if [[ -z "$UV" ]]; then
    UV_TOOL="$SCRATCH/.cache/aiwcas26-uv-$UV_VERSION"
    if [[ ! -x "$UV_TOOL/bin/uv" ]]; then
        log "bootstrapping uv $UV_VERSION in $UV_TOOL"
        "$PYTHON" -m venv "$UV_TOOL"
        "$UV_TOOL/bin/pip" install -q "uv==$UV_VERSION"
    fi
    UV="$UV_TOOL/bin/uv"
fi

UV_RESOLVE_ARGS=(--torch-backend cu126 --python-version 3.11.7 --python-platform x86_64-manylinux_2_28)

if [[ $DO_LOCK == 1 ]]; then
    log "resolving $REQ_IN -> $LOCK"
    "$UV" pip compile "$REQ_IN" "${UV_RESOLVE_ARGS[@]}" --generate-hashes ${UV_LOCK_ARGS:-} \
        --custom-compile-command "scripts/build_shared_env.sh --lock" -o "$LOCK"
fi

if [[ -e "$DEST" || -e "$IMG" ]]; then
    echo "$DEST or $IMG already exists. Envs are immutable: pick a new --name (e.g. anemoi-v2)." >&2
    exit 1
fi

# Shared parent: setgid, group-readable, only the owner writes
mkdir -p "$ROOT"
for d in "$(dirname "$ROOT")" "$ROOT"; do
    if [[ -O "$d" ]]; then chgrp "$GROUP" "$d"; chmod 2755 "$d"; fi
done

log "creating venv $DEST"
# --relocatable: activate and entry-point scripts find the env relative to themselves, so the same
# tree works at its Lustre path and wherever a job mounts the squashfs image.
"$UV" venv --relocatable --python "$PYTHON" "$DEST"

log "installing $LOCK"
"$UV" pip sync --python "$DEST/bin/python" --require-hashes "${UV_RESOLVE_ARGS[@]:0:2}" "$LOCK"

log "pre-compiling bytecode (unchecked-hash: the env is read-only, no source stat/compare at import)"
"$DEST/bin/python" -m compileall -q -j 8 --invalidation-mode unchecked-hash \
    "$DEST/lib/python3.11/site-packages" >/dev/null || log "(compileall reported errors in some files; harmless)"

log "Jupyter kernel spec inside the env"
"$DEST/bin/python" -m ipykernel install --prefix "$DEST" --name aiwcas26-anemoi \
    --display-name "Python (aiwcas26 anemoi)" >/dev/null

cat > "$DEST/AIWCAS26_ENV.txt" <<EOF
AIWCAS 2026 Anemoi tutorial, shared environment $NAME
built     : $(date -Is) by $USER on $(hostname)
repo      : $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
lock      : env/requirements.txt sha256 $(sha256sum "$LOCK" | cut -d' ' -f1)
python    : $PYTHON
EOF
"$UV" pip freeze --python "$DEST/bin/python" > "$DEST/AIWCAS26_FREEZE.txt"

log "smoke import"
"$DEST/bin/python" -c "import anemoi.datasets, anemoi.graphs, anemoi.models, anemoi.training, torch, trimesh, datashader; print('torch', torch.__version__, '| anemoi-training', anemoi.training.__version__)"

log "permissions: group $GROUP, read-only for the group"
chgrp -R "$GROUP" "$DEST"
chmod -R go-w,a+rX "$DEST"
find "$DEST" -type d -exec chmod g+s {} +

if [[ $DO_SQFS == 1 ]]; then
    log "squashfs image $IMG"
    # Stripe the image over 8 SSD OSTs so ~30 nodes reading it at once do not all hit one OST.
    lfs setstripe -c 8 -S 4M "$IMG" 2>/dev/null || true
    # gzip: the only compressor both the kernel (zlib/lzo/xz) and squashfuse_ll (zlib) support.
    # Level 6 keeps mksquashfs well under the login-node CPU limit at almost the same size.
    mksquashfs "$DEST" "$IMG" -noappend -no-progress -processors 8 \
        -comp gzip -Xcompression-level 6 -all-root -no-xattrs >/dev/null
    chgrp "$GROUP" "$IMG"; chmod 0444 "$IMG"
    ls -l "$IMG"
fi

if [[ $DO_ACTIVATE == 1 ]]; then
    log "activating: $ROOT/anemoi -> $NAME"
    ln -sfn "$NAME" "$ROOT/anemoi"
    [[ -e "$IMG" ]] && ln -sfn "$NAME.sqfs" "$ROOT/anemoi.sqfs"
fi

log "done: $DEST"
du -sh "$DEST"; [[ -e "$IMG" ]] && ls -lh "$IMG"
exit 0

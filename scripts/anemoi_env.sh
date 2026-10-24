# Activate the anemoi environment inside a Slurm job (AIWCAS 2026 tutorial). Source it, do not run it:
#
#   source "$REPO_DIR/scripts/anemoi_env.sh"
#
# Environment variables (all optional):
#   ANEMOI_VENV      the venv directory (default: the shared env
#                    /leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi)
#   ANEMOI_ENV_MODE  auto (default): mount the squashfs image next to the venv if there is one, else
#                                    fall back to the plain venv (with a warning if the mount fails)
#                    sqfs: the image or fail;   venv: always the plain venv on Lustre
#   ANEMOI_SQFS      image to mount (default: <realpath of ANEMOI_VENV>.sqfs)
#
# Why: importing torch + anemoi from a Lustre venv means stat/open of thousands of small files and
# takes minutes on a busy system. The squashfs image is one big file; squashfuse_ll (shipped with
# Singularity, /usr/libexec/singularity/bin) mounts it with FUSE in a per-job directory under /tmp.
# The shared venv is relocatable (uv venv --relocatable), so the mounted copy works at that path.
# The mount is removed when the job script exits (EXIT trap).

ANEMOI_VENV="${ANEMOI_VENV:-/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi}"
ANEMOI_ENV_MODE="${ANEMOI_ENV_MODE:-auto}"

_anemoi_env_venv() {
    if [[ ! -f "$ANEMOI_VENV/bin/activate" ]]; then
        echo "== ERROR: no venv at ANEMOI_VENV=$ANEMOI_VENV" >&2
        return 1
    fi
    # The module python runs without `module load` (rpath); loading the module costs ~8 s per job.
    # shellcheck disable=SC1091
    source "$ANEMOI_VENV/bin/activate"
    ANEMOI_ENV_ACTIVE="$ANEMOI_VENV (venv on Lustre)"
}

_anemoi_env_unmount() {
    [[ -n "${_ANEMOI_MNT:-}" ]] || return 0
    cd / 2>/dev/null || true
    fusermount3 -u "$_ANEMOI_MNT" 2>/dev/null || fusermount3 -uz "$_ANEMOI_MNT" 2>/dev/null || true
    rmdir "$_ANEMOI_MNT" "$(dirname "$_ANEMOI_MNT")" 2>/dev/null || true
}

_anemoi_env_sqfs() {
    local real img fuse t0
    real="$(realpath -e "$ANEMOI_VENV" 2>/dev/null)" || { echo "== no venv at $ANEMOI_VENV" >&2; return 1; }
    img="${ANEMOI_SQFS:-$real.sqfs}"
    [[ -r "$img" ]] || { echo "== no squashfs image $img" >&2; return 1; }
    fuse="$(command -v squashfuse_ll || echo /usr/libexec/singularity/bin/squashfuse_ll)"
    [[ -x "$fuse" ]] || { echo "== squashfuse_ll not found" >&2; return 1; }
    _ANEMOI_MNT="${TMPDIR:-/tmp}/aiwcas26-$USER-${SLURM_JOB_ID:-$$}/$(basename "$real")"
    mkdir -p "$_ANEMOI_MNT" || return 1
    t0=$(date +%s%3N)
    if ! "$fuse" "$img" "$_ANEMOI_MNT"; then
        echo "== squashfuse_ll failed to mount $img" >&2
        rmdir "$_ANEMOI_MNT" 2>/dev/null; _ANEMOI_MNT=""
        return 1
    fi
    trap _anemoi_env_unmount EXIT     # replaces any EXIT trap set before sourcing this file
    # shellcheck disable=SC1091
    source "$_ANEMOI_MNT/bin/activate"
    ANEMOI_ENV_ACTIVE="$img mounted at $_ANEMOI_MNT (mount took $(( $(date +%s%3N) - t0 )) ms)"
}

case "$ANEMOI_ENV_MODE" in
    venv) _anemoi_env_venv ;;
    sqfs) _anemoi_env_sqfs || { echo "== ERROR: ANEMOI_ENV_MODE=sqfs but the image could not be mounted" >&2; exit 1; } ;;
    auto) if [[ -e "${ANEMOI_SQFS:-$(realpath "$ANEMOI_VENV" 2>/dev/null).sqfs}" ]]; then
              _anemoi_env_sqfs || { echo "== WARNING: squashfs mount failed, using the plain venv" >&2; _anemoi_env_venv; }
          else
              _anemoi_env_venv
          fi ;;
    *) echo "== ERROR: ANEMOI_ENV_MODE must be auto, sqfs or venv (got '$ANEMOI_ENV_MODE')" >&2; exit 1 ;;
esac || exit 1
export ANEMOI_VENV ANEMOI_ENV_MODE ANEMOI_ENV_ACTIVE
echo "== env: $ANEMOI_ENV_ACTIVE"

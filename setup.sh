# AIWCAS 2026 Anemoi tutorial: set up this terminal. From the repository root:
#
#   source setup.sh          # in every new terminal; safe to run again
#   source setup.sh -q       # quiet: print problems only (e.g. from ~/.bashrc)
#
# It activates the shared Python environment, puts the `aiwcas` command on your PATH, registers the
# Jupyter kernel "Python (aiwcas26 anemoi)", creates your working directories in $SCRATCH/aiwcas26
# and checks that you can read everything the tutorial needs. It never changes files outside
# $SCRATCH/aiwcas26 and your Jupyter kernel list.
#
# All the paths of the tutorial are set HERE (instructors: change them here only). Configs can read
# them as ${oc.env:AIWCAS_DATA}, notebooks as os.environ["AIWCAS_DATA"], jobs inherit them.

# Executed (`bash setup.sh`, `./setup.sh`) instead of sourced: nothing would stick, so say so and stop.
# (`exit` is safe here: it ends only the separate process that runs this file.)
if [ -z "${BASH_VERSION:-}" ]; then
    echo "setup.sh needs bash. Type 'bash' first, then: source setup.sh"
    return 1 2>/dev/null || exit 1
fi
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please SOURCE this file instead of running it, so it can set up your current terminal:"
    echo "    source $0"
    exit 1
fi

_aiwcas_setup() {
    local quiet=0 errors=0 repo
    [[ "${1:-}" == -q || "${1:-}" == --quiet ]] && quiet=1
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

    # ------------------------------------------------------------------ the one place for paths
    export AIWCAS_REPO="$repo"
    export AIWCAS_ENV="${AIWCAS_ENV:-${ANEMOI_VENV:-/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi}}"
    export AIWCAS_DATA="${AIWCAS_DATA:-/leonardo_work/tra26_aiwcas/a08trd25/data}"   # anemoi/ (zarr), cartopy/
    export AIWCAS_WORKDIR="${AIWCAS_WORKDIR:-${SCRATCH:-$HOME}/aiwcas26}"            # your outputs
    export AIWCAS_LOGS="$AIWCAS_WORKDIR/logs"                                         # Slurm logs, loss plots
    export AIWCAS_PROJECT=tra26_aiwcas
    export ANEMOI_VENV="$AIWCAS_ENV"            # read by scripts/anemoi_env.sh inside jobs
    export MLFLOW_ALLOW_FILE_STORE=true         # offline MLflow (see AGENTS.md)
    export MLFLOW_DISABLE_AGENT_HINT=1
    export JUPYTER_PREFER_ENV_PATH=0            # our kernel (with the paths above) over the env's own copy

    local ok="  ok " bad="  !! "
    _say()  { [[ $quiet == 1 ]] || echo "$@"; }
    _fail() { echo "$bad$1"; shift; local l; for l in "$@"; do echo "       $l"; done; errors=$((errors + 1)); }

    _say "AIWCAS 2026 Anemoi tutorial: setting up this terminal"

    # ------------------------------------------------------------------ checks
    case "$(hostname)" in
        login*) _say "$ok on a login node ($(hostname -s))" ;;
        *) _fail "you are not on a login node ($(hostname -s))." \
                 "Run this (and aiwcas) on a login node: ssh <username>@login.leonardo.cineca.it" ;;
    esac

    if [[ -z "${SCRATCH:-}" ]]; then
        _fail "\$SCRATCH is not set, so your outputs would go to \$HOME." \
              "Log in again with a normal login shell (ssh), or tell the instructor."
    fi

    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx "$AIWCAS_PROJECT"; then
        _say "$ok member of the project group $AIWCAS_PROJECT"
    else
        _fail "your account ($USER) is not in the project group $AIWCAS_PROJECT." \
              "Ask the instructor to add you to the project, then log out and in again."
    fi

    local py=""
    if [[ -r "$AIWCAS_ENV/bin/activate" && -x "$AIWCAS_ENV/bin/python" ]]; then
        py="$(realpath "$AIWCAS_ENV/bin/python" 2>/dev/null)"
        if [[ -n "$py" && -x "$py" ]] && "$AIWCAS_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
            _say "$ok shared Python environment ($AIWCAS_ENV)"
        else
            _fail "the shared environment cannot start its Python 3.11.7 (module python/3.11.7): ${py:-$AIWCAS_ENV/bin/python}" \
                  "This is a cluster-side problem: tell the instructor."
            py=""
        fi
    else
        _fail "cannot read the shared Python environment at $AIWCAS_ENV" \
              "If 'id' does not list $AIWCAS_PROJECT, see the group message above; otherwise tell the instructor."
    fi

    if [[ -r "$AIWCAS_DATA/anemoi/aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing.zarr/.zattrs" ]]; then
        _say "$ok tutorial data ($AIWCAS_DATA)"
    else
        _fail "cannot read the tutorial data in $AIWCAS_DATA/anemoi" \
              "If the group check above failed, fix that first; otherwise tell the instructor."
    fi

    # The anemoi sources (git submodules) are read by some notebooks: fetch them if the clone lacks them
    if [[ -d "$repo/.git" && ! -e "$repo/src/anemoi-core/training/src" ]]; then
        echo "  .. fetching the anemoi sources (git submodules, once, ~1 min)"
        if git -C "$repo" submodule update --init --quiet; then
            _say "$ok anemoi sources in src/"
        else
            _fail "could not fetch the git submodules." "Run: git -C $repo submodule update --init"
        fi
    fi

    # ------------------------------------------------------------------ environment + PATH
    if [[ -n "$py" ]]; then
        if [[ "${VIRTUAL_ENV:-}" != "$(realpath "$AIWCAS_ENV")" ]]; then
            # shellcheck disable=SC1091
            source "$AIWCAS_ENV/bin/activate"
        fi
        _say "$ok python is now $(command -v python)"
    fi
    case ":$PATH:" in
        *":$repo/bin:"*) ;;
        *) export PATH="$repo/bin:$PATH" ;;
    esac

    # ------------------------------------------------------------------ working directories
    if mkdir -p "$AIWCAS_LOGS" "$AIWCAS_WORKDIR/hour2/graphs" "$AIWCAS_WORKDIR/training" "$AIWCAS_WORKDIR/graphs" 2>/dev/null; then
        _say "$ok your working directory: $AIWCAS_WORKDIR"
    else
        _fail "cannot create $AIWCAS_WORKDIR" "Check that \$SCRATCH is writable (df -h \$SCRATCH), or tell the instructor."
    fi

    # ------------------------------------------------------------------ Jupyter kernel (VS Code, JupyterLab)
    if [[ -n "$py" ]]; then
        local kdir="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/aiwcas26-anemoi" kjson
        kjson=$(cat <<EOF
{
 "argv": ["$AIWCAS_ENV/bin/python", "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
 "display_name": "Python (aiwcas26 anemoi)",
 "language": "python",
 "env": {
  "AIWCAS_DATA": "$AIWCAS_DATA",
  "AIWCAS_WORKDIR": "$AIWCAS_WORKDIR",
  "MLFLOW_ALLOW_FILE_STORE": "true",
  "MLFLOW_DISABLE_AGENT_HINT": "1"
 },
 "metadata": {"debugger": true, "supported_encryption": ["curve"]},
 "kernel_protocol_version": "5.5"
}
EOF
)
        if [[ "$(cat "$kdir/kernel.json" 2>/dev/null)" == "$kjson" ]]; then
            _say "$ok Jupyter kernel \"Python (aiwcas26 anemoi)\""
        elif mkdir -p "$kdir" && printf '%s\n' "$kjson" > "$kdir/kernel.json"; then
            _say "$ok Jupyter kernel \"Python (aiwcas26 anemoi)\" registered ($kdir)"
        else
            _fail "could not write the Jupyter kernel to $kdir" "Check that your \$HOME is not full (quota)."
        fi
    fi

    # ------------------------------------------------------------------ summary
    unset -f _say _fail
    if [[ $errors -gt 0 ]]; then
        echo "  -> $errors problem(s) above. Fix them, then run 'source setup.sh' again."
        return 1
    fi
    if [[ $quiet == 0 ]]; then
        echo ""
        echo "You're ready. Next:"
        echo "  * notebooks: open notebooks/01_datasets.ipynb in VS Code (kernel: Python (aiwcas26 anemoi))"
        echo "  * hour 2   : exercises/README.md"
        echo "  * commands : aiwcas help"
        if ! grep -qs "$repo/setup.sh" "$HOME/.bashrc"; then
            echo "Optional, to set up every new terminal automatically:"
            echo "  echo 'source $repo/setup.sh -q' >> ~/.bashrc"
        fi
    fi
    return 0
}

_aiwcas_setup "$@" || { unset -f _aiwcas_setup; return 1; }
unset -f _aiwcas_setup

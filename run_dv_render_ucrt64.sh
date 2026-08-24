#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXE="$ROOT_DIR/build/tools/dv_render.exe"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash run_dv_render_ucrt64.sh INPUT PTS MODE [CR_STRENGTH] [OUTPUT]

Example:
  bash run_dv_render_ucrt64.sh \
    /g/28.Years.Later.The.Bone.Temple.mkv \
    72.5 contrast-recovery 0.4 output.raw

OUTPUT defaults to /dev/null because dv_render writes raw frame data to stdout.
EOF
    exit 2
}

INPUT=""
PTS=""
MODE=""
CR_STRENGTH=""
FIRE_POP=""
OUTPUT="/dev/null"

if [ "${1:-}" = "--input" ]; then
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --input) INPUT="${2:?missing value for --input}"; shift 2 ;;
            --pts) PTS="${2:?missing value for --pts}"; shift 2 ;;
            --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
            --cr-strength) CR_STRENGTH="${2:?missing value for --cr-strength}"; shift 2 ;;
            --fire-pop-strength) FIRE_POP="${2:?missing value for --fire-pop-strength}"; shift 2 ;;
            --output) OUTPUT="${2:?missing value for --output}"; shift 2 ;;
            *) echo "ERROR: unknown argument: $1" >&2; usage ;;
        esac
    done
else
    [ "$#" -ge 3 ] || usage
    INPUT="$1"
    PTS="$2"
    MODE="$3"
    CR_STRENGTH="${4:-}"
    OUTPUT="${5:-/dev/null}"
fi

[ -n "$INPUT" ] && [ -n "$PTS" ] && [ -n "$MODE" ] || usage

# Accept Windows drive paths as well as MSYS2 paths.
if command -v cygpath >/dev/null 2>&1; then
    case "$INPUT" in
        [A-Za-z]:/*) INPUT="$(cygpath -u -- "$INPUT")" ;;
    esac
    case "$OUTPUT" in
        [A-Za-z]:/*) OUTPUT="$(cygpath -u -- "$OUTPUT")" ;;
    esac
fi

if [ -x /ucrt64/bin/python3.exe ]; then
    UCRT_BIN=/ucrt64/bin
elif [ -d /c/msys64/ucrt64/bin ]; then
    UCRT_BIN=/c/msys64/ucrt64/bin
else
    echo "ERROR: MSYS2 UCRT64 runtime was not found." >&2
    echo "Expected /ucrt64/bin or /c/msys64/ucrt64/bin." >&2
    exit 1
fi

export PATH="$UCRT_BIN:/usr/bin:$PATH"

if [ ! -f "$EXE" ]; then
    echo "ERROR: dv_render was not built: $EXE" >&2
    echo "Run: bash build_daemon_ucrt64.sh" >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input file not found: $INPUT" >&2
    exit 1
fi

# Windows DLL lookup searches the executable directory first.
if [ ! -f "$ROOT_DIR/build/tools/libplacebo-360.dll" ] &&
   [ -f "$ROOT_DIR/build/src/libplacebo-360.dll" ]; then
    cp "$ROOT_DIR/build/src/libplacebo-360.dll" "$ROOT_DIR/build/tools/"
fi

if ! ldd "$EXE" 2>/dev/null | grep -q "not found"; then
    :
else
    echo "ERROR: one or more runtime DLLs are missing." >&2
    ldd "$EXE" >&2 || true
    exit 1
fi

ARGS=(--input "$INPUT" --pts "$PTS" --mode "$MODE")
if [ -n "$CR_STRENGTH" ]; then
    ARGS+=(--cr-strength "$CR_STRENGTH")
fi
if [ -n "$FIRE_POP" ]; then
    ARGS+=(--fire-pop-strength "$FIRE_POP")
fi

printf 'Running dv_render\n  input: %s\n  pts: %s\n  mode: %s\n  output: %s\n' "$INPUT" "$PTS" "$MODE" "$OUTPUT" >&2

"$EXE" "${ARGS[@]}" > "$OUTPUT"

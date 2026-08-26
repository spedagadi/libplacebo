#!/bin/bash
# deploy_to_mpv.sh — Build custom libplacebo+mpv and stage dist/ for PyQt app
# Run from UCRT64 terminal: bash /c/Code/libplacebo/deploy_to_mpv.sh
#
# Output: /c/Code/mpv/dist/ contains all DLLs needed by python-mpv in PyQt

set -e
set -o pipefail

LIBPLACEBO_BUILD=/c/Code/libplacebo/build_persistent
LIBPLACEBO_SRC=/c/Code/libplacebo/src
SYSTEM_LIB=/c/msys64/ucrt64/lib
SYSTEM_BIN=/c/msys64/ucrt64/bin
SYSTEM_INC=/c/msys64/ucrt64/include/libplacebo
MPV_BUILD=/c/Code/mpv
MPV_BUILD_DIR="${MPV_BUILD_DIR:-build-local-placebo}"   # override with env var if needed
DIST_DIR=/c/Code/mpv/dist

echo "=== [1/3] Deploying custom libplacebo to MSYS2 system ==="

cp "$LIBPLACEBO_BUILD/src/libplacebo-360.dll" "$SYSTEM_BIN/libplacebo-360.dll"
cp "$LIBPLACEBO_BUILD/src/libplacebo.dll.a"   "$SYSTEM_LIB/libplacebo.dll.a"
echo "  DLL + import lib copied"

for h in ml_render.h ml_model.h ml_features.h ml_radiance.h; do
    if [ -f "$LIBPLACEBO_SRC/include/libplacebo/$h" ]; then
        cp "$LIBPLACEBO_SRC/include/libplacebo/$h" "$SYSTEM_INC/$h"
        # Also update the meson build-dir shadow copy so mpv's build picks up changes
        SHADOW_INC="$LIBPLACEBO_BUILD/../build_persistent/src/include/libplacebo"
        if [ -f "$SHADOW_INC/$h" ]; then
            cp "$LIBPLACEBO_SRC/include/libplacebo/$h" "$SHADOW_INC/$h"
        fi
        echo "  header: $h"
    fi
done

echo ""
echo "=== [2/3] Building mpv ==="
cd "$MPV_BUILD"

echo "  Using mpv build dir: $MPV_BUILD_DIR"

if [ ! -f "$MPV_BUILD_DIR/build.ninja" ]; then
    echo "  No build dir — running meson.exe setup..."
    meson.exe setup "$MPV_BUILD_DIR" 2>&1 | tee /c/tmp/mpv_meson.log
fi

meson.exe compile -C "$MPV_BUILD_DIR" 2>&1 | tee /c/tmp/mpv_build.log
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ] || [ ! -f "$MPV_BUILD/$MPV_BUILD_DIR/mpv.exe" ]; then
    echo ""
    echo "=== BUILD FAILED ==="
    tail -40 /c/tmp/mpv_build.log
    exit 1
fi

echo ""
echo "=== [3/3] Staging dist/ for PyQt + python-mpv ==="
mkdir -p "$DIST_DIR"

# Core: libmpv (the one python-mpv loads via ctypes)
cp "$MPV_BUILD/$MPV_BUILD_DIR/libmpv-2.dll"    "$DIST_DIR/libmpv-2.dll"
cp "$MPV_BUILD/$MPV_BUILD_DIR/libmpv.dll.a"    "$DIST_DIR/libmpv.dll.a"  2>/dev/null || true
echo "  libmpv-2.dll"

# Our custom libplacebo (with ML extensions)
cp "$LIBPLACEBO_BUILD/src/libplacebo-360.dll" "$DIST_DIR/libplacebo-360.dll"
echo "  libplacebo-360.dll (custom ML build)"

# mpv.exe itself (useful for testing)
cp "$MPV_BUILD/$MPV_BUILD_DIR/mpv.exe"         "$DIST_DIR/mpv.exe"
echo "  mpv.exe"

# Copy runtime dependencies libmpv needs (from MSYS2)
echo "  Runtime deps from UCRT64..."
DEPS=(
    libavcodec-61.dll libavformat-61.dll libavutil-59.dll
    libswresample-5.dll libswscale-8.dll libavfilter-10.dll
    libass-9.dll libbluray-2.dll
    libgcc_s_seh-1.dll libwinpthread-1.dll libstdc++-6.dll
    zlib1.dll libiconv-2.dll libbz2-1.dll liblzma-5.dll
    libzimg-2.dll libvulkan-1.dll
    libshaderc_shared.dll libspirv-cross-c-shared.dll
)
for dep in "${DEPS[@]}"; do
    if [ -f "$SYSTEM_BIN/$dep" ]; then
        cp "$SYSTEM_BIN/$dep" "$DIST_DIR/$dep"
        echo "    $dep"
    fi
done

echo ""
echo "=== DONE ==="
ls -lh "$DIST_DIR/"
echo ""
echo "PyQt app usage:"
echo "  import ctypes, os"
echo "  os.add_dll_directory(r'C:\\Code\\mpv\\dist')"
echo "  import mpv  # python-mpv finds libmpv-2.dll from dist/"
echo ""
echo "Verify our libplacebo is loaded:"
objdump -p "$DIST_DIR/libmpv-2.dll" 2>/dev/null | grep -i "libplacebo" || true

# Build `dv_render` with MSYS2 UCRT64

These instructions build the libplacebo extraction tools and the `dv_render` application with the MSYS2 UCRT64 toolchain.

## Important: Do This, Not That

Follow this table exactly. It prevents the common terminal, Python, DLL, and
argument mistakes that can make the build or render appear to be stuck.

| Do | Do not |
| --- | --- |
| Run `bash build_daemon_ucrt64.sh` to build the applications. | Do not run `dv_render.exe` before building it. |
| Run `bash run_dv_render_ucrt64.sh` to launch `dv_render`. | Do not launch `build/tools/dv_render.exe` directly. |
| Use `/g/28.Years.Later.The.Bone.Temple.mkv` for a file on the Windows `G:` drive. | Do not use `/c/28.Years.Later.The.Bone.Temple.mkv` unless the file is actually on `C:`. |
| Let the launcher configure the UCRT64 DLL path. | Do not copy only `libplacebo-360.dll` to `/ucrt64/bin`. |
| Use `/c/msys64/ucrt64/bin` when running from Git Bash. | Do not use `/ucrt64/bin` from Git Bash; that mount exists only in a native MSYS2 UCRT64 shell. |
| Use the launcher’s `--input`, `--pts`, `--mode`, and `--cr-strength` options. | Do not pass those options to the old positional-argument form or mix invocation styles. |
| Allow the launcher to redirect raw frame output to `/dev/null` or a file. | Do not send raw `dv_render` output directly to the terminal. |
| Use the UCRT64 Python selected by the build script. | Do not use pyenv Python or `/mingw64/bin/python3`. |
| Run one build at a time in the `build` directory. | Do not run multiple Meson/Ninja processes against the same build directory. |

### Canonical build command

```bash
cd /c/Code/libplacebo
bash build_daemon_ucrt64.sh
```

### Canonical render command

```bash
cd /c/Code/libplacebo
bash run_dv_render_ucrt64.sh \
  --input /g/28.Years.Later.The.Bone.Temple.mkv \
  --pts 72.5 \
  --mode contrast-recovery \
  --cr-strength 0.4
```

The render command defaults to `/dev/null` for stdout because `dv_render`
writes raw frame data there. To keep the raw output, add:

```bash
  --output output.raw
```

Do not add `> output.raw` after the command when using `--output`; choose one
output method.

## 1. Close conflicting processes

Before starting:

- Close Git Bash terminals.
- Close regular VS Code Bash terminals.
- Stop other Meson, Ninja, Python, or MSYS2 build processes.
- Do not run two builds against the same `build` directory.

This avoids `.ninja_lock` permission errors.

## 2. Open a supported terminal

The preferred terminal is:

```text
MSYS2 UCRT64
```

The repository script also supports Git Bash. This is useful when another
Copilot instance can only access the VS Code or Git Bash terminal. PowerShell,
MSYS2 MSYS, and MSYS2 MinGW64 are not supported build environments.

When using Git Bash, do not try to make `/ucrt64/bin` appear as a mounted path.
The build script automatically uses the installed Windows path:

```text
/c/msys64/ucrt64/bin/python3.14.exe
```

Run the script directly:

```bash
cd /c/Code/libplacebo
bash build_daemon_ucrt64.sh
```

The script explicitly selects UCRT64 Python, so a pyenv `python3` does not
interfere with the build.

## 3. Verify the environment

Run:

```bash
echo "$MSYSTEM"
which gcc
which g++
which python3
which meson
which ninja
```

Expected results:

```text
UCRT64
/ucrt64/bin/gcc
/ucrt64/bin/g++
/ucrt64/bin/python3
/ucrt64/bin/meson
/ucrt64/bin/ninja
```

The Python path is especially important. If it points to a path such as:

```text
/c/Users/Sateesh/.pyenv/pyenv-win/shims/python3
```

then pyenv is active and the wrong Python is being used.

Temporarily force the correct paths with:

```bash
export PATH="/ucrt64/bin:/usr/bin:$PATH"
```

Then verify again:

```bash
which python3
python3 --version
```

The path must be `/ucrt64/bin/python3` or equivalent.

## 4. Install or update required packages

Run these commands from the MSYS2 UCRT64 terminal:

```bash
pacman -Syu
```

If `msys2-runtime` is upgraded, close the terminal and open a new MSYS2 UCRT64 terminal before continuing.

Install the build tools:

```bash
pacman -S \
  mingw-w64-ucrt-x86_64-meson \
  mingw-w64-ucrt-x86_64-ninja \
  mingw-w64-ucrt-x86_64-python
```

If pacman reports that Meson files already exist in the filesystem, use:

```bash
pacman -S --overwrite \
  '/ucrt64/lib/python3.14/site-packages/meson*' \
  mingw-w64-ucrt-x86_64-meson \
  mingw-w64-ucrt-x86_64-ninja \
  mingw-w64-ucrt-x86_64-python
```

## 5. Enter the repository

Use the MSYS2 path format:

```bash
cd /c/Code/libplacebo
pwd
```

Expected output:

```text
/c/Code/libplacebo
```

## 6. Important Meson launcher workaround

Do not use the plain `meson` command for this project if it produces an error like:

```text
C:/code/libplacebo/-c --internal exe --unpickle ...
```

That happens when the MSYS2 Meson shell wrapper records its `-c` argument as Meson's own executable.

Use Meson through the UCRT64 Python module instead:

```bash
/ucrt64/bin/python3.exe -m mesonbuild.mesonmain
```

## 7. Clean configure

Remove the old build directory:

```bash
rm -rf build
```

Configure with the UCRT64 native file:

```bash
/ucrt64/bin/python3.exe -m mesonbuild.mesonmain \
  setup build \
  --native-file msys2-ucrt64.ini
```

The native file selects the UCRT64 GCC, G++, linker, archiver, strip tool, and pkg-config paths.

## 8. Compile the targets

Build all three required executables:

```bash
/ucrt64/bin/python3.exe -m mesonbuild.mesonmain \
  compile -C build \
  pl_extract_features \
  pl_extract_features_daemon \
  dv_render
```

A successful build ends with linking messages similar to:

```text
Linking target tools/pl_extract_features.exe
Linking target tools/pl_extract_features_daemon.exe
Linking target tools/dv_render.exe
```

## 9. Verify the output files

Run:

```bash
ls -lh build/tools/pl_extract_features.exe
ls -lh build/tools/pl_extract_features_daemon.exe
ls -lh build/tools/dv_render.exe
```

Expected files:

```text
build/tools/pl_extract_features.exe
build/tools/pl_extract_features_daemon.exe
build/tools/dv_render.exe
```

## 10. Use the convenience script

The repository already includes a script with the correct configuration and target list:

```bash
cd /c/Code/libplacebo
bash build_daemon_ucrt64.sh
```

The script performs the following actions:

1. Removes the old `build` directory.
2. Configures Meson with `msys2-ucrt64.ini`.
3. Invokes Meson through Python.
4. Builds `pl_extract_features`.
5. Builds `pl_extract_features_daemon`.
6. Builds `dv_render`.
7. Lists the generated executables.
8. Performs a daemon smoke test.

## 11. Test `dv_render`

Run the executable without arguments:

```bash
build/tools/dv_render.exe
```

It should display usage information and report missing required arguments. That is expected.

## 12. Run `dv_render` with the UCRT64 DLLs

When launching from Git Bash, add the UCRT64 runtime directory to `PATH` first:

```bash
export PATH="/c/msys64/ucrt64/bin:/usr/bin:$PATH"
```

Do not copy only `libplacebo-360.dll` into `/ucrt64/bin`. The executable is
built with its libplacebo DLL beside it in `build/tools`, while FFmpeg,
shaderc, Vulkan, and the other runtime DLLs are supplied by the UCRT64 bin
directory through `PATH`.

Check the executable dependencies before running:

```bash
cd /c/Code/libplacebo
ldd build/tools/dv_render.exe | grep "not found" || echo "All DLLs resolved"
```

If the libplacebo DLL is not beside the executable, copy it there:

```bash
test -f build/tools/libplacebo-360.dll || \
  cp build/src/libplacebo-360.dll build/tools/
```

Confirm the input file exists before starting a potentially long render:

```bash
test -f /c/28.Years.Later.The.Bone.Temple.mkv && \
  echo "Input found" || \
  echo "ERROR: input file not found"
```

Then run:

```bash
build/tools/dv_render.exe \
  --input /g/28.Years.Later.The.Bone.Temple.mkv \
  --pts 72.5 \
  --mode contrast-recovery \
  --cr-strength 0.4 > output.raw
```

`dv_render` writes the rendered frame to standard output. Redirect stdout to
a file or `/dev/null`; otherwise the terminal displays binary frame data and
can appear to hang. The `/g/` path maps to the Windows `G:` drive. Change it
if the file is stored elsewhere. A missing input file is not a DLL problem.

For a render-only smoke test without retaining the raw frame:

```bash
build/tools/dv_render.exe \
  --input /g/28.Years.Later.The.Bone.Temple.mkv \
  --pts 72.5 \
  --mode contrast-recovery \
  --cr-strength 0.4 > /dev/null
```

A normal invocation requires at least:

```bash
build/tools/dv_render.exe \
  --input <video-file> \
  --pts <seconds> \
  --mode <mode>
```

Supported modes include:

```text
gold
spline
st2094-10
st2094-40
bt2390
ml
contrast-recovery
```

Example:

```bash
build/tools/dv_render.exe \
  --input /c/path/to/input.hevc \
  --pts 0 \
  --mode contrast-recovery
```

## Troubleshooting

### `No module named mesonbuild`

The wrong Python is active, usually pyenv Python. Check:

```bash
which python3
```

Force UCRT64 paths:

```bash
export PATH="/ucrt64/bin:/usr/bin:$PATH"
```

Or use the absolute interpreter:

```bash
/ucrt64/bin/python3.exe -m mesonbuild.mesonmain
```

### `/mingw64/bin/python3` not found

`mingw64` is a different MSYS2 environment. This project requires UCRT64:

```bash
/ucrt64/bin/python3.exe
```

### `C:/code/libplacebo/-c`

The plain Meson shell wrapper was used. Use:

```bash
/ucrt64/bin/python3.exe -m mesonbuild.mesonmain
```

### `.ninja_lock: Permission denied`

Another process is using the same build directory. Close other builds, then run:

```bash
rm -rf build
bash build_daemon_ucrt64.sh
```

### Missing dependencies

If Meson reports missing dependencies such as `shaderc`, `vulkan`, `dovi`, or FFmpeg libraries, install the matching `mingw-w64-ucrt-x86_64-*` development packages from the MSYS2 UCRT64 terminal.

## Shortest reliable build

```bash
export PATH="/ucrt64/bin:/usr/bin:$PATH"
cd /c/Code/libplacebo
bash build_daemon_ucrt64.sh
```

This build procedure was verified successfully and produced all three executables.

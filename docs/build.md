# Building

The recipe is `onnxruntime-full/PKGBUILD`; its header comment is the reference for the build inputs and this page explains them. Build from the `onnxruntime-full/` directory with `makepkg`.

## Build inputs

Build inputs are environment variables. `prepare()` resolves and checks them, prints each value with where it came from, and records them for `build()` and `package()`, so `makepkg --printsrcinfo` and the package metadata never depend on them. After a failed build, `makepkg -e -f` continues where it stopped with the inputs `prepare()` recorded.

| Variable | Default | Meaning |
|---|---|---|
| `ORTF_EPS` | `cuda tensorrt openvino` | The execution providers to build, space- or comma-separated, from `cuda`, `tensorrt`, `openvino`; or `cpu` alone for a CPU-only build. `tensorrt` needs `cuda`. The CPU provider is always built |
| `ORTF_CUDA_ARCHS` | `120` | CUDA architectures (`CMAKE_CUDA_ARCHITECTURES`), such as `120`, `sm_120` or `"89 120"`. Used only with `cuda` |
| `ORTF_NVCC_THREADS` | `1` | `nvcc --threads` per CUDA compile job |
| `ORTF_TENSORRT_HOME` | see below | The TensorRT root: `include/NvInfer.h` and `lib/libnvinfer.so` |
| `ORTF_PYTHON` | `1` | `1` also builds the Python module for `python-onnxruntime-full`; `0` skips it, and `makepkg` then stops at `python-onnxruntime-full` after writing `onnxruntime-full`'s package |
| `ORTF_CCACHE` | `auto` | `auto` uses `ccache` as compiler launcher when it is on `PATH`; `1` requires it; `0` disables it |
| `MAKEFLAGS` | none | `-jN` sets the number of parallel compile jobs. Without it ninja runs `nproc+2` jobs |

Examples:

```bash
MAKEFLAGS=-j8 makepkg -f                                  # all engines, RTX 50 series
ORTF_EPS="cuda tensorrt" ORTF_CUDA_ARCHS="89 120" MAKEFLAGS=-j8 makepkg -f
ORTF_EPS=openvino MAKEFLAGS=-j8 makepkg -f                # OpenVINO only
ORTF_TENSORRT_HOME=/usr MAKEFLAGS=-j8 makepkg -f          # the AUR's tensorrt
```

### CUDA architectures

`ORTF_CUDA_ARCHS` takes numbers, optionally with an `sm_` prefix, separated by spaces, commas or semicolons. ONNX Runtime turns a bare number into machine code for that GPU only: `120` becomes `120a-real`. That is the smallest and fastest build, with no PTX to JIT-compile on another GPU. The default `120` is `sm_120`, the RTX 50 series; for another GPU give its compute capability without the dot (for example `89` for an RTX 40 series card).

- **No `a` or `f` suffix.** `120a` and `120f` are refused. ONNX Runtime adds the `a` itself, and it turns its SM120 kernels on only for a bare `120`, so a suffix would silently drop them.
- **PTX.** `-virtual` entries embed PTX as well, for example `120-virtual`. ONNX Runtime keeps one `-virtual` entry, so at most one is accepted.
- **More architectures** make the build longer and larger; give only the GPUs you have.

### TensorRT location

`ORTF_TENSORRT_HOME` is resolved in this order, and the result must hold `include/NvInfer.h` and `libnvinfer.so*` under `lib` or `lib64`:

1. the environment variable, an absolute path, when set and not empty;
2. `/opt/nvidia/tensorrt`, when it holds `include/NvInfer.h` (the omarchy-nvidia-sdks package);
3. `/usr` (the AUR's `tensorrt`).

When TensorRT is not under `/usr/lib` or `/usr/lib64`, `package()` adds its library directory to the TensorRT provider's RUNPATH with `patchelf`; see [engines.md](engines.md#tensorrt).

### Python

The module is built with the first `python` on `PATH`, which for a package must be `/usr/bin/python`. Any other python is for recipe checks only, and must be the system's Python version: the module is installed into `/usr/bin/python`'s site-packages whichever python builds it, and `prepare()` stops when the two disagree on that directory. The build python needs `numpy`, `build`, `installer`, `setuptools` and `wheel`.

## Dependencies are mirrored, not fetched

ONNX Runtime fetches its C++ dependencies with CMake FetchContent at the versions pinned in its `cmake/deps.txt`. The recipe downloads those archives as `makepkg` sources instead (`ortf-dep-*.zip`, checksummed, cached in `SRCDEST`) and passes them to the build as a local mirror (`onnxruntime_CMAKE_DEPS_MIRROR_DIR`). Consequences:

- `build()` needs no network, and ONNX Runtime still applies its own patches to the dependencies.
- The system's protobuf and abseil are not used (`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER`). Arch's versions differ from the pinned ones, and mixing them is what makes Arch's own recipe carry several patches. The pinned versions are the combination upstream tests; they are linked statically and hidden by the libraries' version scripts.
- `prepare()` checks each mirrored archive's URL and SHA-1 against `cmake/deps.txt`, and the source tag against its commit (`_ortf_commit`). A version bump that misses one fails there, not in the middle of the build.
- Not mirrored, because this build does not use them: the unit-test, benchmark, XNNPACK, CoreML, WebGPU, DirectML, training, telemetry and Arm dependencies, and `onnx_tensorrt` (the TensorRT provider uses TensorRT's own ONNX parser).

The sources total about 3.2 GiB, of which the upstream git clone is 1.6 GiB.

## ccache

With `ccache` on `PATH` the recipe uses it as the C, C++ and CUDA compiler launcher (`ORTF_CCACHE=auto`). It does not help a first build, but it makes a rebuild after a failure, a version bump or changed inputs much cheaper. `ORTF_CCACHE=1` fails when `ccache` is missing, and `0` turns it off.

## Parallelism, memory and disk

- **Time.** A build with CUDA takes hours. CUDA kernels (CUTLASS, flash attention) dominate.
- **Memory.** A CUDA compile job takes several GiB of RAM, so the job count (`MAKEFLAGS=-jN`), not the thread count, decides whether the build fits in memory. The PKGBUILD's guidance for 24 threads and 30 GiB of RAM is `MAKEFLAGS=-j8 ORTF_NVCC_THREADS=1`; lower it to `-j6` if the build is killed for lack of memory. Each job's nvcc threads multiply on top of the job count, so raise `ORTF_NVCC_THREADS` only with memory to spare.
- **Disk.** Tens of GiB under the build directory. `BUILDDIR=/path/with/space makepkg ...` moves the build tree, and `SRCDEST=/path/with/space` moves the downloaded sources, including the 1.6 GiB git clone. Set both in `makepkg.conf` to make this permanent.
- **No LTO, no debug package.** The recipe sets `!lto` (link-time optimisation of the CUDA provider costs hours and many GiB of RAM; upstream's releases are not LTO builds) and `!debug` (the CUDA provider's debug information alone runs to GiB).

Measured so far: a CPU-only build (`ORTF_EPS=cpu`) took about 7 to 8.5 minutes at `-j8` on a 24-thread Core Ultra 9 275HX, without ccache. A full build with CUDA, TensorRT and OpenVINO has not been timed yet; the CUDA kernels dominate it.

## Checking the recipe without the vendor runtimes

A CPU-only build checks the recipe on a machine without CUDA, TensorRT or OpenVINO:

```bash
cd onnxruntime-full
ORTF_EPS=cpu makepkg -d -f
```

`-d` skips the dependency check, which would demand `cuda`, `cudnn` and `openvino`. With `ORTF_EPS=cpu` the build needs only the tools: `cmake` and `ninja` can come from a Python virtual environment on `PATH` instead of from pacman. The `python` first on `PATH` then must agree with `/usr/bin/python` on the site-packages directory, because the Python split package installs into the system's; `prepare()` checks this and stops with a message naming both directories. The result is a recipe check, not a package for general use: install the packages only on a machine meant to run them.

`scripts/test.sh` does not build; it checks the PKGBUILD's syntax, metadata and build-input handling ([README](../README.md#testing)).

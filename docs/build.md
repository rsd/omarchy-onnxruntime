# Building

The recipe is `onnxruntime-split/PKGBUILD`; its header comment is the reference for the build inputs and this page explains them. Build from the `onnxruntime-split/` directory with `makepkg`. The recipe is one build, pkgbase `onnxruntime-split`, that writes one package per engine. A second recipe, `onnxruntime-split-cuda12/PKGBUILD`, builds the CUDA 12 legacy provider for older NVIDIA GPUs; it is generated from this one ([below](#the-cuda-12-legacy-line)).

This page is the reference for the build inputs. The step-by-step build and install is in [build-from-source.md](build-from-source.md). Terms are in the [glossary](glossary.md).

## Build inputs

Build inputs are environment variables.

- `prepare()` resolves and checks them, prints each value with where it came from, and records them for `build()` and `package()`. It records the TensorRT version in its headers and the installed versions of OpenVINO, Python and (for MIGraphX) `migraphx` and `hip-runtime-amd`; the packages' dependency bounds are taken from these ([dependencies.md](dependencies.md#the-version-bounds)).
- Only a few fields of the package metadata depend on the inputs: the package names (see [Package names](#package-names)); `makedepends`, which gains `tensorrt` when the TensorRT provider is built (the AUR's `tensorrt` and `nvidia-tensorrt-sdk-local` both provide it) and `migraphx` and `hip-runtime-amd` when `migraphx` is selected; and the optional dependencies of `onnxruntime-core` and `python-onnxruntime-core`, which name `onnxruntime-provider-migraphx` when it is built. Every other field of `makepkg --printsrcinfo` is the same for any inputs.
- After a failed build, `makepkg -e -f` continues where it stopped with the inputs `prepare()` recorded.

| Variable | Default | Meaning |
|---|---|---|
| `ORTF_EPS` | `cuda tensorrt openvino` | The execution providers to build, space- or comma-separated, from `cuda`, `tensorrt`, `openvino`, `migraphx`; or `cpu` alone for a CPU-only build. `tensorrt` needs `cuda`. `migraphx` (AMD GPUs, **untested on AMD hardware**) is not in the default; it stands alone or joins the others (untested together). The CPU provider is always built |
| `ORTF_CUDA_ARCHS` | `rtx` | CUDA architectures (`CMAKE_CUDA_ARCHITECTURES`): a preset (`rtx`, `all`, and `legacy` for CUDA 12) or a list such as `120`, `sm_120` or `"89 120"`. Used only with `cuda`. It also sets the CUDA package's name |
| `ORTF_NVCC_THREADS` | `1` | `nvcc --threads` per CUDA compile job |
| `ORTF_TENSORRT_HOME` | see below | The TensorRT root: `include/NvInfer.h` and `lib/libnvinfer.so`. Its `include/NvInferVersion.h` gives the major.minor that the TensorRT provider's `tensorrt` dependency is bounded to |
| `ORTF_PYTHON` | `1` | `1` also builds the Python module and `python-onnxruntime-core`; `0` skips both |
| `ORTF_CCACHE` | `auto` | `auto` uses `ccache` as compiler launcher when it is on `PATH`; `1` requires it; `0` disables it |
| `MAKEFLAGS` | none | `-jN` sets the number of parallel compile jobs. Without it ninja runs `nproc+2` jobs |
| `CFLAGS`, `CXXFLAGS` | `makepkg.conf`'s | Arch's `-march=x86-64 -mtune=generic` is the portable baseline; see [Portable baseline](#portable-baseline) |

The defaults of `ORTF_EPS`, `ORTF_CUDA_ARCHS` and `ORTF_PYTHON` are the `_ortf_default_eps`, `_ortf_default_cuda_archs` and `_ortf_default_python` lines of the PKGBUILD (`cuda tensorrt openvino`, `rtx`, `1`); a set, non-empty `ORTF_*` environment variable wins. The lines let a copy of the PKGBUILD build other inputs where the environment does not reach makepkg, as in a clean chroot (`extra-x86_64-build`): [release.md](release.md) builds the `all` preset that way, with `sed -i 's/^_ortf_default_cuda_archs=.*/_ortf_default_cuda_archs=all/' PKGBUILD` on a copy.

Examples:

```bash
MAKEFLAGS=-j8 makepkg -f                                  # all engines, RTX 20 to 50 series: package -cuda13-rtx
ORTF_CUDA_ARCHS=120 MAKEFLAGS=-j8 makepkg -f              # RTX 50 series only: -cuda13-sm120
ORTF_CUDA_ARCHS=all MAKEFLAGS=-j8 makepkg -f              # every x86 GPU CUDA 13 supports: -cuda13-all
ORTF_EPS="cuda tensorrt" ORTF_CUDA_ARCHS="89 120" MAKEFLAGS=-j8 makepkg -f
ORTF_EPS=openvino MAKEFLAGS=-j8 makepkg -f                # OpenVINO only
ORTF_EPS=migraphx MAKEFLAGS=-j8 makepkg -d -f             # AMD GPUs only (untested): see below
ORTF_EPS='cuda tensorrt openvino migraphx' MAKEFLAGS=-j8 makepkg -s -f   # every provider (untested together)
ORTF_TENSORRT_HOME=/usr MAKEFLAGS=-j8 makepkg -f          # the AUR's tensorrt
```

### MIGraphX (AMD GPUs)

`migraphx` in `ORTF_EPS` builds the MIGraphX provider, with `-Donnxruntime_USE_MIGRAPHX=ON -Donnxruntime_MIGRAPHX_HOME=/opt/rocm` and no other ROCm flags. **It is untested on AMD hardware**, and this recipe has not been built with it yet. No GPU architecture is chosen at build time: MIGraphX compiles each model for the GPU it finds at run time. The prerequisites (`migraphx` and `hip-runtime-amd` installed first), the `-d` flag for an AMD-only build and the open questions are in [amd.md](amd.md#building-it).

### CUDA architectures

`ORTF_CUDA_ARCHS` is a preset, alone, or a list of numbers, optionally with an `sm_` prefix, separated by spaces, commas or semicolons. A number is the GPU's compute capability without the dot (the architecture version of an NVIDIA GPU: `89` for an RTX 40 series card, `120` for RTX 50; [glossary](glossary.md#nvidia-gpus)). The default is the preset `rtx`.

ONNX Runtime turns a bare number into machine code ("SASS") for that GPU only: `120` becomes `120a-real`. That is the smallest and fastest build, with no PTX (NVIDIA's intermediate code) to compile on another GPU at run time. A bare `120` is `sm_120`, the RTX 50 series alone.

| Preset or list | Architectures | CUDA package | Installed size | CUDA kernel compile time |
|---|---|---|---|---|
| `rtx` (default) | `75 86 89 120`: the GeForce and RTX lines, Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx` | 154.7 MiB (measured) | about 4 times the `sm_120` build (estimate) |
| `all` | `75 80 86 89 90 100 103 120`: every x86 GPU CUDA 13 supports, including A100, H100, B200 and B300 | `onnxruntime-provider-cuda13-all` | 294.9 MiB (measured) | about 8 times (estimate) |
| `legacy` | `60 61 70 75 80 86 89 120` and the PTX of `60` (`60-virtual`): Pascal and Volta, which CUDA 13 dropped, with Turing to Blackwell beside them. Needs CUDA 12 | `onnxruntime-provider-cuda12-legacy` | 557.5 MiB (measured) | not measured |
| `120` | `sm_120`: RTX 50 series | `onnxruntime-provider-cuda13-sm120` | 89.4 MiB (the library, measured on an earlier build) | the baseline (one target) |
| a list, such as `89 120` | the listed GPUs | `onnxruntime-provider-cuda13-sm89-sm120` | not measured | about one `sm_120` build per target (estimate) |

Each preset has a fixed package name whatever its list, so a build of it can be told from another. Maxwell (`50`, `52`) is not in `legacy`: ONNX Runtime 1.29's fp16 kernels need `sm_53` or later. The ARM and Jetson targets (`87`, `88`, `107`, `110`, `121`, which are Jetson, Thor and DGX Spark) are left out of `all`.

- **Sizes.** The installed sizes are those of the 1.29.0-6 package files (`rtx`, `all`) and the 1.29.0-1 file (`legacy`); the CUDA library is nearly all of each. The targets do not cost the same: the library is 89.4 MiB for `sm_120` alone (55.7 MiB of host code and about 33.7 MiB of GPU code), so the extra targets of `rtx` (`75`, `86`, `89`) average about 22 MiB each and the extra ones of `all` over `rtx` (`80`, `90`, `100`, `103`) about 35 MiB each (derived from those measurements). Download sizes are in [packages.md](packages.md#sizes).
- **Time.** The CUDA kernels' compile time grows about linearly with the number of targets. The total build time is longer by less, since the CPU, TensorRT, OpenVINO and Python parts do not change. Only the `sm_120` full build was timed: 27 minutes ([below](#parallelism-memory-and-disk)).
- **Rule for PTX.** ONNX Runtime compiles `90`, `100`, `103` and `120` as accelerated (`a`) machine code, for exactly that GPU, and the others as plain machine code. `rtx` and `all` embed no PTX; `legacy` embeds that of `60`. A preset's PTX is for the oldest architecture the build compiles, the one PTX that every newer GPU can JIT-compile (compile at run time).
- **The 18 `sm_90a` kernels.** Whatever `ORTF_CUDA_ARCHS` says, ONNX Runtime also compiles 18 kernels for `sm_90a`, so a built provider holds that machine code as well (the CUDA 13 `rtx` package has the same 18).
- **Presets and the CUDA major.** A preset is tied to the CUDA major of the toolkit that builds it, and `prepare()` refuses the other combinations: `rtx` and `all` need CUDA 13 or later, `legacy` needs CUDA 12. With a CUDA 12 toolkit the main recipe therefore takes an explicit list (or `legacy`), and never the default `rtx`.
- **No `a` or `f` suffix.** `120a` and `120f` are refused. ONNX Runtime adds the `a` itself, and it turns its SM120 kernels on only for a bare `120`, so a suffix would silently drop them.
- **PTX in a list.** `-virtual` entries embed PTX as well, for example `60-virtual` or `120-virtual`. ONNX Runtime keeps one `-virtual` entry, so at most one is accepted. A preset cannot be combined with other words; give the list instead.
- **More architectures** make the build longer and larger; give only the GPUs you have.

### Package names

makepkg must know every package name before `prepare()` runs, so the names are read from the environment when it sources the PKGBUILD:

| Package | Exists when | Name |
|---|---|---|
| core | always | `onnxruntime-core` |
| CUDA | `ORTF_EPS` has `cuda` | `onnxruntime-provider-cuda<M>-<sfx>` |
| TensorRT | `ORTF_EPS` has `tensorrt` | `onnxruntime-provider-tensorrt-cuda<M>` |
| OpenVINO | `ORTF_EPS` has `openvino` | `onnxruntime-provider-openvino` |
| MIGraphX | `ORTF_EPS` has `migraphx` | `onnxruntime-provider-migraphx` |
| Python | `ORTF_PYTHON` is not `0` | `python-onnxruntime-core` |

- **`<M>`** is the major version of the CUDA toolkit: `CUDA_VERSION` in `/opt/cuda/include/cuda.h` (Arch's `cuda`), which `prepare()` checks against `nvcc`. It is never written into the PKGBUILD. CUDA 13.x gives `-cuda13`. Where `cuda.h` does not exist yet when makepkg reads the PKGBUILD (an AUR helper reads it before it installs the makedepends), the name uses major 13: the name never has a placeholder in place of the number. A CUDA 12 toolkit gives `-cuda12-` names. The main recipe with CUDA 12 then needs `ORTF_CUDA_ARCHS` set to an explicit list or `legacy` (the default `rtx` needs CUDA 13), and no `tensorrt` in `ORTF_EPS` (TensorRT 11 needs CUDA 13), or a TensorRT 10. That is not the supported route to a CUDA 12 package: the one that is released, `onnxruntime-provider-cuda12-legacy`, comes from its own recipe, not from this one with other inputs ([below](#the-cuda-12-legacy-line)).
- **`<sfx>`** is the preset's name (`rtx`, `all`), or, for a list, the real architectures as `smN` in ascending order without repeats, joined by `-`, then the `-virtual` one as `computeN`: `120` gives `sm120`, `"89 120"` gives `sm89-sm120`, `"89 120 120-virtual"` gives `sm89-sm120-compute120`.
- **The TensorRT package has no architecture suffix**: its library holds no GPU code, because TensorRT builds its kernels at run time. It depends on `onnxruntime-provider-cuda<M>=<pkgver>`, a name every CUDA package provides, because it links that CUDA's runtime, and on `tensorrt`, bounded to the major.minor of the headers the build used ([install.md](install.md#2-tensorrt)). It provides and conflicts with `onnxruntime-provider-tensorrt`.
- **CUDA packages of different builds conflict**: each provides and conflicts with `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`, because all install the same `/usr/lib/libonnxruntime_providers_cuda.so`. One is installed at a time.
- **The MIGraphX package has no architecture suffix and no ROCm version**: its library holds no GPU code, and ROCm 7.2 and 10.1 keep the sonames it links (`libmigraphx_c.so.3`, `libamdhip64.so.7`), so one name serves both. It depends on `migraphx` and `hip-runtime-amd`, bounded to the major version of the installed packages ([dependencies.md](dependencies.md#the-version-bounds)). A ROCm major upgrade therefore needs a rebuild.
- **`ORTF_EPS=cpu`** gives `onnxruntime-core` (and the Python package) only.
- **Dependencies.** The providers and the Python package depend on `onnxruntime-core=<pkgver>-<pkgrel>`, and every package on the packages of the libraries it links, with version bounds for the vendor runtimes: [dependencies.md](dependencies.md).

Each package function checks that the inputs recorded by `prepare()` in `src/ortf.conf` give the same names. They would not under `makepkg -e` or `-R` run with other inputs, or after a CUDA upgrade, and the message says what to pass.

### The CUDA 12 legacy line

`onnxruntime-split-cuda12/PKGBUILD` builds one package, `onnxruntime-provider-cuda12-legacy`: the CUDA provider compiled with CUDA 12.9 for the preset `legacy`, to load into the main line's `onnxruntime-core`. It is generated from the main recipe by `scripts/gen-cuda12.sh` and never edited by hand, and it builds only in a clean chroot with the AUR's `gcc14`, `cuda-12.9` and `cudnn9.10-cuda12.9`. How it is generated, what it refuses, its dependencies and its compiler are in [cuda12-legacy.md](cuda12-legacy.md#building-from-source); the build is in [release-cuda12.md](release-cuda12.md#3-build-in-a-clean-chroot).

### Portable baseline

The build uses the `CFLAGS` and `CXXFLAGS` of `makepkg.conf`; Arch's `-march=x86-64 -mtune=generic` makes a package for any x86-64 CPU, and the recipe passes `onnxruntime_BUILD_FOR_NATIVE_MACHINE=OFF`. `-march=native` and `-mtune=native` are refused, as they make a package for the building CPU only. Flags such as `-march=x86-64-v3` are the builder's choice, and the package then needs such a CPU. ONNX Runtime's MLAS library chooses its AVX2 or AVX-512 kernels at run time, so the baseline does not remove them. The presets' fixed package names keep one build distinguishable from another. The official prebuilt (`-bin`) packages (AUR `onnxruntime-split-bin`) follow the convention in the PKGBUILD header (`<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>`); [install.md](install.md#prebuilt-packages) installs them and [release.md](release.md) says how they are made.

### TensorRT location

`ORTF_TENSORRT_HOME` is resolved in this order, and the result must hold `include/NvInfer.h` and `libnvinfer.so*` under `lib` or `lib64`:

1. the environment variable, an absolute path, when set and not empty;
2. `/opt/nvidia/tensorrt`, when it holds `include/NvInfer.h` (the omarchy-nvidia-sdks package);
3. `/usr` (the AUR's `tensorrt`).

`package()` sets the TensorRT provider's RUNPATH with `patchelf` in every build with TensorRT: `/opt/nvidia/tensorrt/lib`, preceded by the build's TensorRT library directory when that is neither `/usr/lib` nor `/usr/lib64`. The RUNPATH is independent of where the build found TensorRT, because the package depends on the name `tensorrt`, which either source provides; see [engines.md](engines.md#tensorrt).

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

- **Time.** CUDA kernels (CUTLASS, flash attention) dominate a build with CUDA, and the time grows about linearly with the number of architectures in `ORTF_CUDA_ARCHS` ([presets](#cuda-architectures): `rtx` has four targets, `all` eight). A full build with one architecture (`ORTF_CUDA_ARCHS=120`) took under half an hour on the machine below; no build of the default `rtx` or of `all` has been timed. See the measurements at the end of this section.
- **Memory.** A CUDA compile job takes several GiB of RAM, so the job count (`MAKEFLAGS=-jN`), not the thread count, decides whether the build fits in memory. The PKGBUILD's guidance for 24 threads and 30 GiB of RAM is `MAKEFLAGS=-j8 ORTF_NVCC_THREADS=1`; lower it to `-j6` if the build is killed for lack of memory. Each job's nvcc threads multiply on top of the job count, so raise `ORTF_NVCC_THREADS` only with memory to spare.
- **Disk.** Tens of GiB under the build directory. `BUILDDIR=/path/with/space makepkg ...` moves the build tree, and `SRCDEST=/path/with/space` moves the downloaded sources, including the 1.6 GiB git clone. Set both in `makepkg.conf` to make this permanent.
- **No LTO, no debug package.** The recipe sets `!lto` (link-time optimisation of the CUDA provider costs hours and many GiB of RAM; upstream's releases are not LTO builds) and `!debug` (the CUDA provider's debug information alone runs to GiB).

Measured so far, on a 24-thread Intel Core Ultra 9 275HX with 30 GiB of RAM:

| Build | Time | Conditions |
|---|---|---|
| CPU-only (`ORTF_EPS=cpu`) | about 7 to 8.5 minutes | `-j8`, without ccache |
| Full (`ORTF_EPS` default, `ORTF_CUDA_ARCHS=120`) | 27 minutes, including the download of the 3.2 GiB of sources | `MAKEFLAGS=-j8 ORTF_NVCC_THREADS=1`, under `nice -n 19` and `ionice -c 3`, cold ccache, with other builds running on the machine |

The full build succeeded without any change to the recipe against CUDA 13.4.3 (nvcc 13.4, host gcc 16.2.1), cuDNN 9.27.0.42, TensorRT 11.4.0.106 (`nvidia-tensorrt-sdk-local`, under `/opt/nvidia/tensorrt`) and OpenVINO 2026.4.1 with its GPU and NPU plugins (`intel-compute-runtime`, `level-zero-loader`, `intel-npu-driver` 1.38.0). That build made the single `onnxruntime-full` 1.29.0-1, which held every provider: 52 MiB (`.pkg.tar.zst`; 121 MiB installed), and `python-onnxruntime-full` at 13 MiB. Since 1.29.0-2 the same files are split into the packages of [Package names](#package-names); their sizes are in [packages.md](packages.md#sizes). The CUDA provider holds `sm_120a` machine code, plus a few `sm_90a` kernels that upstream compiles unconditionally.

More architectures, fewer jobs or a slower machine lengthen a build. The results of the build's checks are in [results.md](results.md).

## Checking the recipe without the vendor runtimes

A CPU-only build checks the recipe on a machine without CUDA, TensorRT or OpenVINO:

```bash
cd onnxruntime-split
ORTF_EPS=cpu makepkg -d -f
```

`-d` skips the dependency check, which would demand `cuda`, `cudnn` and `openvino` (the same holds for `ORTF_EPS=migraphx`, which needs `migraphx` and `hip-runtime-amd` installed instead). The build writes `onnxruntime-core` and `python-onnxruntime-core` only. With `ORTF_EPS=cpu` the build needs only the tools: `cmake` and `ninja` can come from a Python virtual environment on `PATH` instead of from pacman. The `python` first on `PATH` then must agree with `/usr/bin/python` on the site-packages directory, because the Python split package installs into the system's; `prepare()` checks this and stops with a message naming both directories. The result is a recipe check, not a package for general use: install the packages only on a machine meant to run them.

`scripts/test.sh` does not build; it checks the PKGBUILD's syntax, metadata and build-input handling ([testing.md](testing.md)).

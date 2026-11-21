# Building

The recipe is `onnxruntime-split/PKGBUILD`. Its header comment is the reference for the build inputs, and this page explains them. Build from the `onnxruntime-split/` directory with `makepkg`. The recipe is one build, pkgbase `onnxruntime-split`, that writes one package per engine. Two more recipes are generated from this one:

- `onnxruntime-split-cuda12/PKGBUILD` builds the CUDA 12 legacy provider for older NVIDIA GPUs ([below](#the-cuda-12-legacy-line)).
- `onnxruntime-split-migraphx/PKGBUILD` builds the MIGraphX provider for AMD GPUs ([below](#migraphx-amd-gpus)).

This page is the reference for the build inputs. The step-by-step build and install is in [build-from-source.md](build-from-source.md). Terms are in the [glossary](glossary.md).

## Build inputs

Build inputs are environment variables.

- `prepare()` resolves and checks them, prints each value with where it came from, and records them for `build()` and `package()`. It records the TensorRT version in its headers and the installed versions of OpenVINO, Python and (for MIGraphX) `migraphx` and `hip-runtime-amd`. The packages' dependency bounds are taken from these ([dependencies.md](dependencies.md#the-version-bounds)).
- Only a few fields of the package metadata depend on the inputs:
  - the package names (see [Package names](#package-names));
  - the source (`ORTF_SOURCE`, see [The ONNX Runtime source](#the-onnx-runtime-source-mirror-or-upstream));
  - `makedepends`, which gains `tensorrt` when the TensorRT provider is built (the AUR's `tensorrt` and `nvidia-tensorrt-sdk-local` both provide it), `migraphx` and `hip-runtime-amd` when `migraphx` is selected, and `nccl` (or `rccl` in the MIGraphX line);
  - the optional dependencies of `onnxruntime-core` and `python-onnxruntime-core`, which name `onnxruntime-provider-dnnl` when it is built.

  With `ORTF_NCCL=0` the names gain the suffix `-nccloff`, and the dependencies on NCCL and on the core change ([below](#nccl-and-the-nccloff-packages)). Every other field of `makepkg --printsrcinfo` is the same for any inputs.
- After a failed build, `makepkg -e -f` continues where it stopped with the inputs `prepare()` recorded.

| Variable | Default | Meaning |
|---|---|---|
| `ORTF_EPS` | `cuda tensorrt openvino dnnl` | The execution providers to build, space- or comma-separated, from `cuda`, `tensorrt`, `openvino`, `dnnl`, `migraphx`. Use `cpu` alone for a CPU-only build. `tensorrt` needs `cuda`. `dnnl` (oneDNN, CPU) needs no vendor install. `migraphx` (AMD GPUs, **untested on AMD hardware**) is not in the default. It stands alone or joins the others (untested together), for a local build: the published package is built from its own recipe. The CPU provider is always built. A build without `cuda` needs `ORTF_NCCL=0` |
| `ORTF_CUDA_ARCHS` | `rtx` | CUDA architectures (`CMAKE_CUDA_ARCHITECTURES`): a preset (`rtx`, `all`, and `legacy` for CUDA 12) or a list such as `120`, `sm_120` or `"89 120"`. Used only with `cuda`. It also sets the CUDA package's name |
| `ORTF_NVCC_THREADS` | `1` | `nvcc --threads` per CUDA compile job |
| `ORTF_TENSORRT_HOME` | see below | The TensorRT root: `include/NvInfer.h` and `lib/libnvinfer.so`. Its `include/NvInferVersion.h` gives the major.minor that the TensorRT provider's `tensorrt` dependency is bounded to |
| `ORTF_PYTHON` | `1` | `1` also builds the Python module and `python-onnxruntime-core`. `0` skips both |
| `ORTF_NCCL` | `1` | `1` builds with NCCL. `0` builds without it and names every package with the suffix `-nccloff` ([below](#nccl-and-the-nccloff-packages)). A build without `cuda` in `ORTF_EPS` needs `0` |
| `ORTF_SOURCE` | `mirror` | Where the ONNX Runtime source comes from: `mirror`, this project's archive of the `v1.29.0` tree, or `upstream`, the git tag of `microsoft/onnxruntime`, which has answered 404 since 2026-11-17 ([below](#the-onnx-runtime-source-mirror-or-upstream)) |
| `ORTF_CCACHE` | `auto` | `auto` uses `ccache` as compiler launcher when it is on `PATH`. `1` requires it. `0` disables it |
| `MAKEFLAGS` | none | `-jN` sets the number of parallel compile jobs. Without it ninja runs `nproc+2` jobs |
| `CFLAGS`, `CXXFLAGS` | `makepkg.conf`'s | Arch's `-march=x86-64 -mtune=generic` is the portable baseline; see [Portable baseline](#portable-baseline) |

The defaults of `ORTF_EPS`, `ORTF_CUDA_ARCHS`, `ORTF_PYTHON`, `ORTF_NCCL` and `ORTF_SOURCE` are the `_ortf_default_eps`, `_ortf_default_cuda_archs`, `_ortf_default_python`, `_ortf_default_nccl` and `_ortf_default_source` lines of the PKGBUILD (`cuda tensorrt openvino dnnl`, `rtx`, `1`, `1`, `mirror`). A set, non-empty `ORTF_*` environment variable wins. The lines let a copy of the PKGBUILD build other inputs where the environment does not reach makepkg, as in a clean chroot (`extra-x86_64-build`). [release.md](release.md) builds the `all` preset that way, with `sed -i 's/^_ortf_default_cuda_archs=.*/_ortf_default_cuda_archs=all/' PKGBUILD` on a copy.

Examples:

```bash
MAKEFLAGS=-j8 makepkg -f                                  # all engines, RTX 20 to 50 series: package -cuda13-rtx
ORTF_CUDA_ARCHS=120 MAKEFLAGS=-j8 makepkg -f              # RTX 50 series only: -cuda13-sm120
ORTF_CUDA_ARCHS=all MAKEFLAGS=-j8 makepkg -f              # every x86 GPU CUDA 13 supports: -cuda13-all
ORTF_EPS="cuda tensorrt" ORTF_CUDA_ARCHS="89 120" MAKEFLAGS=-j8 makepkg -f
ORTF_EPS=openvino ORTF_NCCL=0 MAKEFLAGS=-j8 makepkg -d -f # OpenVINO only: no CUDA, so no NCCL; the packages are named -nccloff
ORTF_EPS=dnnl ORTF_NCCL=0 MAKEFLAGS=-j8 makepkg -d -f     # the oneDNN provider alone
ORTF_EPS=migraphx ORTF_NCCL=0 MAKEFLAGS=-j8 makepkg -d -f # AMD GPUs only, a local build without NCCL (untested): see below
ORTF_EPS='cuda tensorrt openvino dnnl migraphx' MAKEFLAGS=-j8 makepkg -s -f   # every provider (untested together)
ORTF_TENSORRT_HOME=/usr MAKEFLAGS=-j8 makepkg -f          # the AUR's tensorrt
ORTF_SOURCE=upstream MAKEFLAGS=-j8 makepkg -f             # the source from the git tag, not the mirror's archive
```

### NCCL and the nccloff packages

NCCL (NVIDIA's collective communication library) is on in every line by default (`ORTF_NCCL=1`, the CMake option `onnxruntime_USE_NCCL=ON`). The training operators that train across several GPUs need it, and Arch's CUDA builds have it. Like the training operators, it changes the interface between `onnxruntime-core` and a provider library. Nothing checks at run time that both sides agree, so the core and every provider are built with it, or every one without it ([packages.md](packages.md#the-core-and-the-other-lines-providers-release-together)).

- **What it costs.** `onnxruntime-core` and `python-onnxruntime-core` depend on Arch's `nccl`, and the CUDA package on it too (the CUDA 12 line on the AUR's `nccl-cuda12.9`). `nccl` is 364.18 MiB installed on a machine that may never train on several GPUs.
- **`ORTF_NCCL=0`** builds without it, with any `ORTF_EPS`. Every package of that build gets the suffix `-nccloff`: `onnxruntime-core-nccloff`, `onnxruntime-provider-cuda13-rtx-nccloff`, `python-onnxruntime-core-nccloff`, and so on. Each provides and conflicts with its base name, and the providers depend on `onnxruntime-core-nccloff`. The NCCL packages conflict with `onnxruntime-core-nccloff`, so a machine has NCCL packages or `-nccloff` ones, never a mixture.
- **No prebuilt `-nccloff`.** `scripts/gen-bin.sh` refuses them. They are for local builds. A prebuilt-only user cannot avoid `nccl`.
- **A build without `cuda` requires it.** In the main recipe, `ORTF_EPS=cpu`, `openvino`, `dnnl`, `migraphx` or any mixture without `cuda` is refused unless `ORTF_NCCL=0`. A core built with NCCL and without the CUDA provider's NCCL interface would not fit the NCCL CUDA providers that its pins accept. `prepare()` says so and what to pass.
- **Which NCCL.** Arch's `nccl` (a CUDA 13 build, kernels for `sm_75` and newer) in the main line. The AUR's `nccl-cuda12.9` in the CUDA 12 line, whose GPUs (Pascal, Volta) Arch's does not run on. AMD's `rccl` in the MIGraphX line, as a build input only (the MIGraphX provider links none).

### MIGraphX (AMD GPUs)

The published MIGraphX package comes from its own recipe, `onnxruntime-split-migraphx/PKGBUILD`. `scripts/gen-migraphx.sh` generates it from the main one, and you never edit it by hand. It builds `onnxruntime-provider-migraphx` and its add-on `python-onnxruntime-provider-migraphx` (the link into the system Python's `capi/`, [amd.md](amd.md#the-package)). It builds with `-Donnxruntime_USE_MIGRAPHX=ON -Donnxruntime_MIGRAPHX_HOME=/opt/rocm` and `rccl` as the NCCL library, with no Python module and no other provider. It loads into the main line's core ([amd.md](amd.md#building-it)).

`migraphx` in `ORTF_EPS` of the main recipe still builds the provider, for a local build, alone or beside the others (untested together). **It is untested on AMD hardware.** No GPU architecture is chosen at build time: MIGraphX compiles each model for the GPU it finds at run time. The prerequisites (`migraphx` and `hip-runtime-amd` installed first), the `-d` flag for an AMD-only build and the open questions are in [amd.md](amd.md#building-it).

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

Each preset has a fixed package name whatever its list, so a build of it can be told from another. Maxwell (`50`, `52`) is not in `legacy`, because ONNX Runtime 1.29's fp16 kernels need `sm_53` or later. The ARM and Jetson targets (`87`, `88`, `107`, `110`, `121`, which are Jetson, Thor and DGX Spark) are left out of `all`.

- **Sizes.** The installed sizes are those of the 1.29.0-6 package files (`rtx`, `all`) and the 1.29.0-1 file (`legacy`). The CUDA library is nearly all of each. The targets do not cost the same. The library is 89.4 MiB for `sm_120` alone (55.7 MiB of host code and about 33.7 MiB of GPU code). The extra targets of `rtx` (`75`, `86`, `89`) therefore average about 22 MiB each, and the extra ones of `all` over `rtx` (`80`, `90`, `100`, `103`) about 35 MiB each (derived from those measurements). Download sizes are in [packages.md](packages.md#sizes).
- **Time.** The CUDA kernels' compile time grows about linearly with the number of targets. The total build time grows by less, since the CPU, TensorRT, OpenVINO and Python parts do not change. The `sm_120` full build took 27 minutes ([below](#parallelism-memory-and-disk)). The default four-architecture build took about 2 hours 10 minutes (CUDA 13 line, from the prepare log to the end of the build log), and the CUDA 12 line's about 1 hour 25 minutes. Both ran on the same machine with other work running. These times are those of 1.29.0-6 and 1.29.0-1. The training operators, NCCL and oneDNN of 1.29.0-7 add build time that has not been measured yet.
- **Rule for PTX.** ONNX Runtime compiles `90`, `100`, `103` and `120` as accelerated (`a`) machine code, for exactly that GPU, and the others as plain machine code. `rtx` and `all` embed no PTX. `legacy` embeds that of `60`. A preset's PTX is for the oldest architecture the build compiles, the one PTX that every newer GPU can JIT-compile (compile at run time).
- **The 18 `sm_90a` kernels.** Whatever `ORTF_CUDA_ARCHS` says, ONNX Runtime also compiles 18 kernels for `sm_90a`, so a built provider holds that machine code as well (the CUDA 13 `rtx` package has the same 18).
- **Presets and the CUDA major.** A preset is tied to the CUDA major of the toolkit that builds it, and `prepare()` refuses the other combinations. `rtx` and `all` need CUDA 13 or later. `legacy` needs CUDA 12. With a CUDA 12 toolkit the main recipe therefore takes an explicit list (or `legacy`), and never the default `rtx`.
- **No `a` or `f` suffix.** `120a` and `120f` are refused. ONNX Runtime adds the `a` itself, and it turns its SM120 kernels on only for a bare `120`, so a suffix would silently drop them.
- **PTX in a list.** `-virtual` entries embed PTX as well, for example `60-virtual` or `120-virtual`. ONNX Runtime keeps one `-virtual` entry, so at most one is accepted. A preset cannot be combined with other words. Give the list instead.
- **More architectures** make the build longer and larger. Give only the GPUs you have.

### Package names

makepkg must know every package name before `prepare()` runs, so it reads the names from the environment when it sources the PKGBUILD:

| Package | Exists when | Name |
|---|---|---|
| core | always | `onnxruntime-core` |
| CUDA | `ORTF_EPS` has `cuda` | `onnxruntime-provider-cuda<M>-<sfx>` |
| TensorRT | `ORTF_EPS` has `tensorrt` | `onnxruntime-provider-tensorrt-cuda<M>` |
| OpenVINO | `ORTF_EPS` has `openvino` | `onnxruntime-provider-openvino` |
| oneDNN | `ORTF_EPS` has `dnnl` | `onnxruntime-provider-dnnl` |
| MIGraphX | `ORTF_EPS` has `migraphx` (the MIGraphX line's recipe also makes the Python add-on `python-onnxruntime-provider-migraphx`) | `onnxruntime-provider-migraphx` |
| Python | `ORTF_PYTHON` is not `0` | `python-onnxruntime-core` |

- **`<M>`** is the major version of the CUDA toolkit: `CUDA_VERSION` in `/opt/cuda/include/cuda.h` (Arch's `cuda`), which `prepare()` checks against `nvcc`. It is never written into the PKGBUILD. CUDA 13.x gives `-cuda13`. Where `cuda.h` does not exist yet when makepkg reads the PKGBUILD (an AUR helper reads it before it installs the makedepends), the name uses major 13. The name never has a placeholder in place of the number.
  - A CUDA 12 toolkit gives `-cuda12-` names. The main recipe with CUDA 12 then needs `ORTF_CUDA_ARCHS` set to an explicit list or `legacy` (the default `rtx` needs CUDA 13). It also needs no `tensorrt` in `ORTF_EPS` (TensorRT 11 needs CUDA 13), or a TensorRT 10.
  - That is not the supported route to a CUDA 12 package. The one that is released, `onnxruntime-provider-cuda12-legacy`, comes from its own recipe, not from this one with other inputs ([below](#the-cuda-12-legacy-line)).
- **`<sfx>`** is the preset's name (`rtx`, `all`). For a list, it is the real architectures as `smN` in ascending order without repeats, joined by `-`, then the `-virtual` one as `computeN`: `120` gives `sm120`, `"89 120"` gives `sm89-sm120`, `"89 120 120-virtual"` gives `sm89-sm120-compute120`.
- **The TensorRT package has no architecture suffix.** Its library holds no GPU code, because TensorRT builds its kernels at run time. It depends on `onnxruntime-provider-cuda<M>=<pkgver>`, a name every CUDA package provides, because it links that CUDA's runtime. It also depends on `tensorrt`, bounded to the major.minor of the headers the build used ([INSTALL.md](../INSTALL.md#install-tensorrt)). It provides and conflicts with `onnxruntime-provider-tensorrt`.
- **CUDA packages of different builds conflict.** Each provides and conflicts with `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`, because all install the same `/usr/lib/libonnxruntime_providers_cuda.so`. Install one at a time.
- **The oneDNN package has no suffix.** It depends on `onnxruntime-core` and `libgomp` only, because oneDNN is linked into it statically ([dependencies.md](dependencies.md#onednn-is-bundled-statically)).
- **The MIGraphX provider has no architecture suffix and no ROCm version.** Its library holds no GPU code, and ROCm 7.2 and 10.1 keep the sonames it links (`libmigraphx_c.so.3`, `libamdhip64.so.7`), so one name serves both. It depends on `migraphx` and `hip-runtime-amd`, bounded to the major version of the installed packages ([dependencies.md](dependencies.md#the-version-bounds)). A ROCm major upgrade therefore needs a rebuild. The add-on `python-onnxruntime-provider-migraphx` has no suffix either. It holds a link only, and depends on the provider, `python-onnxruntime-core` and `python` ([amd.md](amd.md#the-package)).
- **`ORTF_NCCL=0`** adds the suffix `-nccloff` to every name in the table: `onnxruntime-core-nccloff`, `onnxruntime-provider-cuda13-rtx-nccloff` and so on ([above](#nccl-and-the-nccloff-packages)).
- **`ORTF_EPS=cpu`** gives `onnxruntime-core` (and the Python package) only. A build without `cuda` needs `ORTF_NCCL=0`, so the packages are `onnxruntime-core-nccloff` and `python-onnxruntime-core-nccloff`.
- **Dependencies.** The providers and the Python package depend on `onnxruntime-core=<pkgver>-<pkgrel>` (the CUDA 12 and MIGraphX packages on a range, [dependencies.md](dependencies.md#the-core-pin-of-the-cuda-12-and-migraphx-providers)). Every package depends on the packages of the libraries it links, with version bounds for the vendor runtimes: [dependencies.md](dependencies.md).

Each package function checks that the inputs recorded by `prepare()` in `src/ortf.conf` give the same names. They would not under `makepkg -e` or `-R` run with other inputs, or after a CUDA upgrade, and the message says what to pass.

### The CUDA 12 legacy line

`onnxruntime-split-cuda12/PKGBUILD` builds one package, `onnxruntime-provider-cuda12-legacy`: the CUDA provider compiled with CUDA 12.9 for the preset `legacy`, to load into the main line's `onnxruntime-core`. `scripts/gen-cuda12.sh` generates it from the main recipe, and you never edit it by hand. It builds only in a clean chroot with the AUR's `gcc14`, `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9`. How it is generated, what it refuses, its dependencies and its compiler are in [cuda12-legacy.md](cuda12-legacy.md#building-from-source). The build is in [release-cuda12.md](release-cuda12.md#3-build-in-a-clean-chroot).

### Portable baseline

The build uses the `CFLAGS` and `CXXFLAGS` of `makepkg.conf`. Arch's `-march=x86-64 -mtune=generic` makes a package for any x86-64 CPU, and the recipe passes `onnxruntime_BUILD_FOR_NATIVE_MACHINE=OFF`. `-march=native` and `-mtune=native` are refused, as they make a package for the building CPU only. Flags such as `-march=x86-64-v3` are the builder's choice, and the package then needs such a CPU. ONNX Runtime's MLAS library chooses its AVX2 or AVX-512 kernels at run time, so the baseline does not remove them.

The presets' fixed package names keep one build distinguishable from another. The official prebuilt (`-bin`) packages (AUR `onnxruntime-split-bin`) follow the convention in the PKGBUILD header (`<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>`). [INSTALL.md](../INSTALL.md#prebuilt-packages) installs them, and [release.md](release.md) says how they are made.

### TensorRT location

`ORTF_TENSORRT_HOME` is resolved in this order. The result must hold `include/NvInfer.h` and `libnvinfer.so*` under `lib` or `lib64`:

1. the environment variable, an absolute path, when set and not empty;
2. `/opt/nvidia/tensorrt`, when it holds `include/NvInfer.h` (the omarchy-nvidia-sdks package);
3. `/usr` (the AUR's `tensorrt`).

`package()` sets the TensorRT provider's RUNPATH with `patchelf` in every build with TensorRT: `/opt/nvidia/tensorrt/lib`, preceded by the build's TensorRT library directory when that is neither `/usr/lib` nor `/usr/lib64`. The RUNPATH is independent of where the build found TensorRT, because the package depends on the name `tensorrt`, which either source provides. See [engines.md](engines.md#tensorrt).

### Python

The module is built with the first `python` on `PATH`, which for a package must be `/usr/bin/python`. Any other python is for recipe checks only. It must be the system's Python version: the module is installed into `/usr/bin/python`'s site-packages whichever python builds it, and `prepare()` stops when the two disagree on that directory. The build python needs `numpy`, `build`, `installer`, `setuptools` and `wheel`.

## Dependencies are mirrored, not fetched

ONNX Runtime fetches its C++ dependencies with CMake FetchContent at the versions pinned in its `cmake/deps.txt`. The recipe downloads those archives as `makepkg` sources instead (`ortf-dep-*.zip`, checksummed, cached in `SRCDEST`) and passes them to the build as a local mirror (`onnxruntime_CMAKE_DEPS_MIRROR_DIR`). Consequences:

- `build()` needs no network, and ONNX Runtime still applies its own patches to the dependencies.
- The system's protobuf and abseil are not used (`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER`). Arch's versions differ from the pinned ones, and mixing them is what makes Arch's own recipe carry several patches. The pinned versions are the combination upstream tests. They are linked statically and hidden by the libraries' version scripts.
- `prepare()` checks each mirrored archive's URL and SHA-1 against `cmake/deps.txt`, and the source's commit against `_ortf_commit` (the commit embedded in the mirror's archive, or the git tag's). A version bump that misses one fails there, not in the middle of the build.
- Not mirrored, because this build does not use them: the unit-test, benchmark, XNNPACK, CoreML, WebGPU, DirectML, telemetry and Arm dependencies, and `onnx_tensorrt` (the TensorRT provider uses TensorRT's own ONNX parser). The training operators add two that are mirrored, `cxxopts` and `tensorboard`.
- oneDNN is not in `cmake/deps.txt`: upstream clones oneDNN 3.0.1 from git at build time, which would need the network. The recipe downloads a pinned, checksummed archive of oneDNN 3.11.3 (`ortf-dep-onednn.zip`, a `makepkg` source like the others) and patches `cmake/external/dnnl.cmake` to build from it, as a static library ([engines.md](engines.md#onednn)).

The sources totalled about 3.2 GiB before the oneDNN and training archives were added (not re-measured). The upstream git clone is 1.6 GiB of that. With the default mirror the clone is replaced by a 264 MiB archive, so the total is smaller by about that difference (not re-measured).

## The ONNX Runtime source: mirror or upstream

`github.com/microsoft/onnxruntime` has returned 404 since 2026-11-17, with no explanation from Microsoft or GitHub, and related repositories are gone too. A build that clones the `v1.29.0` tag from it fails for anyone without a copy already in `SRCDEST`. The recipes therefore take the source from this project's copy by default. The `-bin` packages are not affected: they download this project's release assets.

`ORTF_SOURCE` chooses, in all three recipes (`onnxruntime-split`, `onnxruntime-split-cuda12` and `onnxruntime-split-migraphx`):

- **`mirror`** (the default, `_ortf_default_source`) downloads `onnxruntime-1.29.0.tar.gz`, an asset of the release `source-v1.29.0` of `rsd/omarchy-onnxruntime` (<https://github.com/rsd/omarchy-onnxruntime/releases/download/source-v1.29.0/onnxruntime-1.29.0.tar.gz>, 276402461 bytes). It is the output of `git archive --format=tar --prefix=onnxruntime-1.29.0/ 2e2543fbe9fae542f921d47a72d21d5a4ef0b710 | gzip -n -9`: the tree of the tag `v1.29.0` and nothing else, the same bytes on every run. The PKGBUILD pins it by b2sum.
- **`upstream`** clones the git tag, as the recipes did before: `ORTF_SOURCE=upstream makepkg ...`. In a clean chroot, where the environment does not reach `makepkg`, set it in a copy of the PKGBUILD with `sed -i 's/^_ortf_default_source=.*/_ortf_default_source=upstream/' PKGBUILD` (as for the other inputs, [release.md](release.md#2-build-the-all-cuda-package)). It works only while the repository answers.

Switching needs nothing else, not even a clean `src/`. `prepare()` checks the source's commit against `_ortf_commit` in both. For the mirror it reads the commit that `git archive` wrote into the archive (`git get-tar-commit-id`), and it unpacks the archive afresh on every run. The two sources unpack into different directories under `src/`, `onnxruntime-1.29.0` for the mirror and `onnxruntime` for the git tag, so they never meet. A patch fixes the build information, so `libonnxruntime.so` reports `git-branch=makepkg, git-commit-id=2e2543fbe9f` from either source. The packages are the same as the ones released as 1.29.0-7: the change of source location changed no `pkgrel`. Any value other than `mirror` or `upstream` is refused.

You can check the mirror. Its release notes are `PROVENANCE.md`. They give:

- how the archive was made, its sha256 (`3aec9fa0c8f4064a8b043af1afbc3d5c2046443bb466243d5b843ca9b4a1fa94`) and its b2sum;
- the steps that compare it with the upstream commit from any clone that still has it. The uncompressed tar stream equals `git archive` of the commit, and `git get-tar-commit-id` prints the commit. The tree without the prefix has the b2sum that the recipe's git source and Arch's `onnxruntime` PKGBUILD pin for this tag.

The maintainer's side, which makes and publishes the archive and watches for upstream's return, is in [release.md](release.md#the-source-mirror).

## ccache

With `ccache` on `PATH` the recipe uses it as the C, C++ and CUDA compiler launcher (`ORTF_CCACHE=auto`). It does not help a first build, but it makes a rebuild after a failure, a version bump or changed inputs much cheaper. `ORTF_CCACHE=1` fails when `ccache` is missing, and `0` turns it off.

## Parallelism, memory and disk

- **Time.** CUDA kernels (CUTLASS, flash attention) dominate a build with CUDA. The time grows about linearly with the number of architectures in `ORTF_CUDA_ARCHS` ([presets](#cuda-architectures): `rtx` has four targets, `all` eight). A full build with one architecture (`ORTF_CUDA_ARCHS=120`) took under half an hour on the machine below. No build of the default `rtx` or of `all` has been timed. See the measurements at the end of this section.
- **Memory.** A CUDA compile job takes several GiB of RAM, so the job count (`MAKEFLAGS=-jN`), not the thread count, decides whether the build fits in memory. The PKGBUILD's guidance for 24 threads and 30 GiB of RAM is `MAKEFLAGS=-j8 ORTF_NVCC_THREADS=1`. Lower it to `-j6` if the build is killed for lack of memory. Each job's nvcc threads multiply on top of the job count, so raise `ORTF_NVCC_THREADS` only with memory to spare.
- **Disk.** A build needs tens of GiB under the build directory. `BUILDDIR=/path/with/space makepkg ...` moves the build tree. `SRCDEST=/path/with/space` moves the downloaded sources, including the 264 MiB source archive, or with `ORTF_SOURCE=upstream` the 1.6 GiB git clone. Set both in `makepkg.conf` to make this permanent.
- **No LTO, no debug package.** The recipe sets `!lto` (link-time optimisation of the CUDA provider costs hours and many GiB of RAM, and upstream's releases are not LTO builds) and `!debug` (the CUDA provider's debug information alone runs to GiB).

Measured so far, on a 24-thread Intel Core Ultra 9 275HX with 30 GiB of RAM:

| Build | Time | Conditions |
|---|---|---|
| CPU-only (`ORTF_EPS=cpu`) | about 7 to 8.5 minutes | `-j8`, without ccache |
| Full (`ORTF_EPS` default, `ORTF_CUDA_ARCHS=120`) | 27 minutes, including the download of the 3.2 GiB of sources (with the git clone: measured before the mirror existed) | `MAKEFLAGS=-j8 ORTF_NVCC_THREADS=1`, under `nice -n 19` and `ionice -c 3`, cold ccache, with other builds running on the machine |

The full build succeeded without any change to the recipe against CUDA 13.4.3 (nvcc 13.4, host gcc 16.2.1), cuDNN 9.27.0.42, TensorRT 11.4.0.106 (`nvidia-tensorrt-sdk-local`, under `/opt/nvidia/tensorrt`) and OpenVINO 2026.4.1 with its GPU and NPU plugins (`intel-compute-runtime`, `level-zero-loader`, `intel-npu-driver` 1.38.0). That build made the single `onnxruntime-full` 1.29.0-1, which held every provider: 52 MiB (`.pkg.tar.zst`; 121 MiB installed), and `python-onnxruntime-full` at 13 MiB. Since 1.29.0-2 the same files are split into the packages of [Package names](#package-names). Their sizes are in [packages.md](packages.md#sizes). The CUDA provider holds `sm_120a` machine code, plus a few `sm_90a` kernels that upstream compiles unconditionally.

More architectures, fewer jobs or a slower machine lengthen a build. The results of the build's checks are in [results.md](results.md).

## Checking the recipe without the vendor runtimes

A CPU-only build checks the recipe on a machine without CUDA, TensorRT or OpenVINO. It needs `ORTF_NCCL=0`, because NCCL is for the CUDA provider and the recipe refuses it without `cuda` ([above](#nccl-and-the-nccloff-packages)):

```bash
cd onnxruntime-split
ORTF_EPS=cpu ORTF_NCCL=0 makepkg -d -f
```

`-d` skips the dependency check, which would demand `cuda`, `cudnn` and `openvino`. The same holds for `ORTF_EPS=migraphx`, which needs `migraphx` and `hip-runtime-amd` installed instead. The build writes `onnxruntime-core-nccloff` and `python-onnxruntime-core-nccloff` only.

With `ORTF_EPS=cpu` the build needs only the tools. `cmake` and `ninja` can come from a Python virtual environment on `PATH` instead of from pacman. The `python` first on `PATH` then must agree with `/usr/bin/python` on the site-packages directory, because the Python split package installs into the system's. `prepare()` checks this and stops with a message naming both directories. The result is a recipe check, not a package for general use. Install the packages only on a machine meant to run them.

`scripts/test.sh` does not build. It checks the PKGBUILD's syntax, metadata and build-input handling ([testing.md](testing.md)).

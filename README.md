# omarchy-onnxruntime

A local Arch Linux package recipe for [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.29.0 with every execution provider a typical NVIDIA and Intel workstation can use, in one installation: CPU, CUDA, TensorRT and OpenVINO (CPU, Intel GPU and Intel NPU devices). Arch's own ONNX Runtime packages each carry one GPU vendor and have no TensorRT or OpenVINO provider. On request (`ORTF_EPS=migraphx`) the recipe also builds the MIGraphX provider for AMD GPUs. **That package is untested on AMD hardware** (the maintainer has no AMD GPU) and has not been built yet either; see [AMD GPUs](#amd-gpus).

The recipe `onnxruntime-split/PKGBUILD` is one build of the upstream source tag `v1.29.0` (pkgbase `onnxruntime-split`), split into one package per engine:

| Package | Contents | Depends on |
|---|---|---|
| `onnxruntime-core` | `libonnxruntime.so` with the CPU provider, `libonnxruntime_providers_shared.so` (in a build with any other provider), the C and C++ headers (`/usr/include/onnxruntime`), `libonnxruntime.pc` and the CMake files. It works on its own, on the CPU. Provides `onnxruntime=1.29.0` and `libonnxruntime.so` | `glibc`, `libgcc`, `libstdc++`, `zlib` |
| `onnxruntime-provider-cuda<M>-<sfx>` | The CUDA provider, `libonnxruntime_providers_cuda.so`. `<M>` is the major version of the CUDA toolkit the build used (`13` today); `<sfx>` names the GPU architectures it was compiled for ([below](#cuda-architecture-presets)). Provides `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>` | `onnxruntime-core`, `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-tensorrt-cuda<M>` | The TensorRT provider. It has no architecture suffix because it holds no GPU code: TensorRT builds its kernels at run time. Provides `onnxruntime-provider-tensorrt` | `onnxruntime-core`, `onnxruntime-provider-cuda<M>`, `cuda`, `cudnn`, `nvidia-utils`, `tensorrt` (bounded to the major.minor of the TensorRT headers the build used) |
| `onnxruntime-provider-openvino` | The OpenVINO provider (Intel CPU, GPU and NPU devices) | `onnxruntime-core`, `openvino` |
| `onnxruntime-provider-migraphx` | The MIGraphX provider (AMD GPUs), built only with `migraphx` in `ORTF_EPS`. **Untested on AMD hardware.** It holds no GPU code: MIGraphX compiles the model for the GPU it finds at run time. The name carries no ROCm version, because ROCm 7.2 and 10.1 keep the same sonames | `onnxruntime-core`, `migraphx`, `hip-runtime-amd` (bounded to the major version of the installed packages) |
| `python-onnxruntime-core` | The Python module. Its links to the provider libraries dangle until that provider's package is installed. Provides `python-onnxruntime` | `onnxruntime-core`, `python` and its packages |

With the default build and Arch's CUDA 13, the files are `onnxruntime-core`, `onnxruntime-provider-cuda13-sm120`, `onnxruntime-provider-tensorrt-cuda13`, `onnxruntime-provider-openvino` and `python-onnxruntime-core`. The names depend on the build inputs `ORTF_EPS`, `ORTF_PYTHON` and `ORTF_CUDA_ARCHS` and on the CUDA toolkit: `ORTF_EPS=cpu` builds `onnxruntime-core` (and the Python package) only, and `migraphx` in `ORTF_EPS` adds `onnxruntime-provider-migraphx`. A CUDA 12 toolkit would give `-cuda12` names, but a CUDA 12 variant is not provided at present.

`onnxruntime-core` and `python-onnxruntime-core` replace and conflict with `onnxruntime-full` and `python-onnxruntime-full`, the single packages of 1.29.0-1 ([upgrading](docs/install.md#upgrading-from-the-single-package-1290-1)).

### Dependencies and version bounds

Every package depends on the package of each library its own libraries link, so none can be installed without what it needs to run: the build fails when a library maps to no package in `depends`. The providers and the Python module depend on `onnxruntime-core=<pkgver>-<pkgrel>`. Where the ABI demands a rebuild on an upgrade, the dependency carries a bound taken when the package was built: `cuda` by major version from the sonames, `cudnn` by major version from the cuDNN installed at build time (the CUDA provider `dlopen()`s cuDNN, so no soname gives it), `tensorrt` by major.minor from the TensorRT headers the build used, `openvino` by year.minor, `python` by minor version, and `migraphx` and `hip-runtime-amd` by the major version of the installed package (ROCm keeps its sonames across majors, so a soname would give no bound). pacman then refuses a system update that would move one of them past its bound, instead of a provider failing to load; when Arch moves one past a bound, rebuild with a `pkgrel` bump. `scripts/check-deps.sh <package files>` sees only the `DT_NEEDED` entries, so libraries that are `dlopen()`ed at run time are covered by the recipe's recorded bounds instead; it lists every linked library, its owning package and whether the installed versions are still inside the bounds, so it shows when a rebuild is due ([docs/install.md](docs/install.md#dependencies-and-version-bounds)).

Version 1.29.0 is the release Arch ships, so programs built against Arch's `onnxruntime` see the same API.

## Why not Arch's onnxruntime packages

Arch's `extra` repository ships ONNX Runtime 1.29.0 (`1.29.0-3` when this was written) as separate builds: `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` (AVX2 build), `onnxruntime-rocm` and `onnxruntime-opt-rocm`, each with a `python-` counterpart. Each build provides `onnxruntime` and conflicts with it, so one is installed at a time. This recipe exists for the combination none of them offers:

| | Arch `extra` builds | This recipe (`onnxruntime-core` and the provider packages) |
|---|---|---|
| CPU, CUDA | CUDA builds include the CPU provider | yes |
| TensorRT, OpenVINO (CPU, Intel GPU, Intel NPU) | not enabled in the recipe | yes, in one library |
| NVIDIA and Intel engines in one install | no: CUDA and ROCm are separate packages, and the CUDA builds add oneDNN (`dnnl`) as the other provider | yes |
| AMD GPUs (MIGraphX) | `onnxruntime-rocm` and `onnxruntime-opt-rocm` carry the MIGraphX provider | `onnxruntime-provider-migraphx`, on request; **untested on AMD hardware** |
| CUDA architectures | a list from `75` to `121`, including `120` (Blackwell) | `120` by default; the presets `rtx` and `all` or a list, set with `ORTF_CUDA_ARCHS` ([presets](#cuda-architecture-presets)) |

Support for `sm_120` is therefore not the difference: the Arch CUDA builds include it. The difference is the TensorRT and OpenVINO providers, which the Arch recipe does not build. The version is the same, so programs built against Arch's package see the same API. Source: the Arch recipe at <https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime> and `pacman -Si`.

### Why one package per engine, not one per build

Arch splits per build: each variant is a separate compilation of the whole library, they conflict, and one is installed at a time. Here the providers are separate shared objects (`libonnxruntime_providers_<provider>.so`) that one `libonnxruntime.so` loads at run time, so a single build serves every engine. The split therefore follows the engines: each package carries only its provider library and the vendor runtimes that library needs as dependencies, and the core package runs without any of them. A machine installs the engines it has and nothing else, for example the core and the OpenVINO package on a machine without an NVIDIA GPU.

## Engines

Each provider is a separate library, in a separate package, that ONNX Runtime loads only when a session asks for it, and the core library links no vendor runtime. Install the packages of the engines you use. Details are in [docs/engines.md](docs/engines.md). Which GPUs, CPUs and NPUs each engine covers, and what is not packaged (NVIDIA cards older than Turing, the Ryzen AI NPU, older Intel graphics; the AMD GPU provider is packaged but untested), is in [Hardware coverage and what is not packaged](docs/engines.md#hardware-coverage-and-what-is-not-packaged).

| Engine (execution provider) | Hardware | This recipe's package | Vendor runtime packages | Provided by |
|---|---|---|---|---|
| CPU | Any x86_64 CPU | `onnxruntime-core` | none | built in |
| CUDA | NVIDIA GPU | `onnxruntime-provider-cuda<M>-<sfx>` | `nvidia-utils`, `cuda`, `cudnn` | Arch `extra` |
| TensorRT | NVIDIA GPU | `onnxruntime-provider-tensorrt-cuda<M>` | TensorRT 11 (and the CUDA package's runtimes) | `nvidia-tensorrt-sdk-local` from the omarchy-nvidia-sdks repository (under `/opt/nvidia/tensorrt`, pkgrel 2 or later), or the AUR's `tensorrt` (under `/usr`) |
| OpenVINO, device `CPU` | Intel CPU | `onnxruntime-provider-openvino` | `openvino` | Arch `extra` |
| OpenVINO, device `GPU` | Intel graphics | `onnxruntime-provider-openvino` | `openvino-intel-gpu-plugin`, `intel-compute-runtime`, `level-zero-loader` | Arch `extra` |
| OpenVINO, device `NPU` | Intel NPU | `onnxruntime-provider-openvino` | `openvino-intel-npu-plugin`, `intel-npu-driver`, `level-zero-loader` | Arch `extra` |
| MIGraphX (untested on AMD hardware) | AMD GPU supported by ROCm | `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd` | Arch `extra` |

`cuda`, `cudnn` and `openvino` (and TensorRT's headers and libraries, if you build that provider) must be installed when you build; so must `migraphx` and `hip-runtime-amd` for the MIGraphX provider. Installing a provider package needs its runtime (a hard dependency); the OpenVINO device plugins and drivers are optional dependencies, installed for the devices you use.

### CUDA architecture presets

`ORTF_CUDA_ARCHS` takes a preset or a list of compute capabilities ([docs/build.md](docs/build.md#cuda-architectures)). The CUDA provider's size and the build time grow with the number of targets, so the presets give a choice and a fixed package name:

| `ORTF_CUDA_ARCHS` | Targets | CUDA package name | Provider size | CUDA kernel compile time |
|---|---|---|---|---|
| `120` (default) | `sm_120`: RTX 50 series | `onnxruntime-provider-cuda13-sm120` | 89.4 MiB (measured) | the baseline (one target) |
| `rtx` | `75 86 89 120`: Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx` | about 190 MiB (estimate) | about 4 times the `sm_120` kernel compile |
| `all` | `75 80 86 89 90 100 103 120`: every x86 GPU CUDA 13 supports, including A100, H100, B200 and B300 | `onnxruntime-provider-cuda13-all` | about 325 MiB (estimate) | about 8 times the `sm_120` kernel compile |
| a list, such as `89 120` | the listed GPUs | `onnxruntime-provider-cuda13-sm89-sm120` | 55.7 MiB + 33.7 MiB per target (estimate) | about one `sm_120` kernel compile per target |

What was measured: the CUDA provider is 55.7 MiB of host code plus about 33.7 MiB of GPU code per target, and `sm_120` alone makes 89.4 MiB. The `rtx` (4 targets: 55.7 + 4 x 33.7, about 190 MiB) and `all` (8 targets, about 325 MiB) sizes are estimates computed from the per-target figure, not measurements of those builds. The CUDA kernels' compile time grows about linearly with the number of targets; the total build time is longer by less, since the CPU, TensorRT, OpenVINO and Python parts do not change. The `sm_120` build measured 27 minutes in total ([build.md](docs/build.md#parallelism-memory-and-disk)). The ARM and Jetson targets (`87`, `88`, `107`, `110`, `121`) are left out of `all`.

Each preset has a fixed package name whatever its list, so a build of it can be told from another. Prebuilt packages (`-bin`), should there ever be any, follow one convention, stated in the PKGBUILD header: `<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>`, so the dependencies between the packages hold for source-built and prebuilt packages alike. This is a convention only; no prebuilt package is promised. CUDA packages of different builds provide and conflict with each other (they install the same `libonnxruntime_providers_cuda.so`), so one is installed at a time.

### Portable baseline

The build targets generic x86-64: `makepkg`'s `-march=x86-64 -mtune=generic` and `onnxruntime_BUILD_FOR_NATIVE_MACHINE=OFF`. The recipe refuses `-march=native` and `-mtune=native` in `CFLAGS` and `CXXFLAGS`, which would make a package for the building CPU only. ONNX Runtime's MLAS library chooses its AVX2 or AVX-512 kernels at run time, so those kernels still use what the CPU has; the rest of the code is generic x86-64. Setting `-march=x86-64-v3` or similar is the builder's choice, and the package then needs such a CPU.

### AMD GPUs

ONNX Runtime 1.29 has no ROCm execution provider (it was removed in 1.23); the only AMD GPU provider is MIGraphX, which this recipe builds when `migraphx` is in `ORTF_EPS`: `ORTF_EPS=migraphx makepkg -d -f` for AMD only, or `ORTF_EPS='cuda tensorrt openvino migraphx' makepkg -s -f` for every provider ([docs/build.md](docs/build.md#build-inputs)). **It is untested on AMD hardware**, because the maintainer has no AMD GPU, and the package recipe has not been built yet. The combination with the other providers is untested too. The provider holds no GPU code: MIGraphX compiles each model for the detected GPU at run time (the first session is slow), so GPU coverage is what Arch's ROCm packages were built for ([docs/engines.md](docs/engines.md#amd-gpus-migraphx-packaged-untested)).

The package depends on `migraphx` and `hip-runtime-amd`, never on `rocm-hip-sdk`. `onnxruntime-core` lists it as an optional dependency; `python-onnxruntime-core` lists it too. The Python module can load the provider: when the build also makes `python-onnxruntime-core`, the MIGraphX package installs a symlink `<site-packages>/onnxruntime/capi/libonnxruntime_providers_migraphx.so` to `/usr/lib/libonnxruntime_providers_migraphx.so`, the way the CUDA, TensorRT and OpenVINO libraries reach the Python module (those links belong to `python-onnxruntime-core`; this one belongs to the provider package, because upstream's `setup.py` leaves the MIGraphX library out of the wheel unless built with `--use_migraphx`). This is untested on AMD hardware, and the package has not been built yet. Alternatively use Arch's `onnxruntime-rocm` or `onnxruntime-opt-rocm`. `onnxruntime-core` conflicts with them (and with `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and `onnxruntime-git`), so only one is installed at a time; pacman offers to remove the other.

## Licence and personal use

- **The recipe** (this repository's content) is MIT OR Apache-2.0, at your option: see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
- **The built ONNX Runtime** is MIT-licensed, plus the open-source dependencies compiled into its libraries. The PKGBUILD's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0`. The notices are installed with each package, in `/usr/share/licenses/<package>/` (`LICENSE`, `ThirdPartyNotices.txt`).
- **Each engine's runtime is under its vendor's licence.** The CUDA, cuDNN and TensorRT runtimes are under NVIDIA's licences; OpenVINO is Apache-2.0, and the Intel graphics and NPU drivers have their own licences. Read the licence of each runtime you install.
- **Nothing from NVIDIA or Intel is redistributed by this repository.** It holds only the recipe, scripts and documentation. The package is built locally, against the runtimes you install yourself or download from the vendor yourself, and contains none of them.
- **Keep the built package on your own machine.** It is built against vendor runtimes under licences that you accepted, so do not upload it to a repository, a release page or the AUR as a binary.
- This is not legal advice.

The guard `scripts/check-no-vendor.sh` refuses vendor and binary files (libraries, archives, wheels, packages, models, engine caches, vendor headers and licence documents, makepkg work directories) and file content that carries an NVIDIA or Intel marker. Enable the hooks in your clone with:

```bash
git config core.hooksPath .githooks
```

The hooks are `pre-commit`, `pre-merge-commit` and `pre-push`. `git commit --no-verify` bypasses them, so the guard is a safety net, not an access control. `scripts/check-no-vendor.sh --all` checks the working tree, not the history.

## Quick start

The full procedure, with the prerequisites, is in [docs/install.md](docs/install.md). In short:

```bash
# build tools and the runtimes of the engines you want (docs/install.md has the lists)
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools cuda cudnn openvino

# TensorRT first: nvidia-tensorrt-sdk-local (pkgrel 2 or later) from omarchy-nvidia-sdks, or the AUR's tensorrt; the provider package depends on tensorrt

cd onnxruntime-split
MAKEFLAGS=-j8 makepkg -f
# one transaction, naming the exact files of this build (a glob can catch packages of another build)
sudo pacman -U onnxruntime-core-1.29.0-4-x86_64.pkg.tar.zst onnxruntime-provider-cuda13-sm120-1.29.0-4-x86_64.pkg.tar.zst onnxruntime-provider-tensorrt-cuda13-1.29.0-4-x86_64.pkg.tar.zst onnxruntime-provider-openvino-1.29.0-4-x86_64.pkg.tar.zst python-onnxruntime-core-1.29.0-4-x86_64.pkg.tar.zst
cd ..

# optional: the packages' libraries, their owners and the version bounds, against this system
scripts/check-deps.sh onnxruntime-split/*-1.29.0-4-x86_64.pkg.tar.zst

scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU
```

A subset is fine: the core alone, the core and `onnxruntime-provider-openvino`, and so on. The commands above are for the default engines; for an AMD GPU see [AMD GPUs](#amd-gpus). Upgrading from the earlier single `onnxruntime-full` 1.29.0-1, which held every provider library, needs all the new files in the one transaction, and pacman offers to remove the old packages in it: [docs/install.md](docs/install.md#upgrading-from-the-single-package-1290-1).

A full build with one CUDA architecture took 27 minutes on a 24-thread machine ([measured](docs/build.md#parallelism-memory-and-disk)), with more architectures taking longer ([presets](#cuda-architecture-presets)), and it needs tens of GiB of disk. Build inputs (engines, GPU architecture, TensorRT location, parallelism) are in [docs/build.md](docs/build.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/install.md](docs/install.md) | Prerequisites, runtimes, TensorRT, building and installing the packages, the dependency policy and `scripts/check-deps.sh`, upgrading from 1.29.0-1, verifying |
| [docs/build.md](docs/build.md) | Build inputs, dependency mirroring, ccache, parallelism and resources, a CPU-only recipe check |
| [docs/engines.md](docs/engines.md) | Per-engine notes, TensorRT plugins, hardware coverage and what is not packaged (NVIDIA, Intel, AMD), using the library from C, Rust and Python |
| [docs/results.md](docs/results.md) | Measured results: smoke test and model check of a full build, with the conditions |
| [tools/ort-smoke/README.md](tools/ort-smoke/README.md) | The smoke test: targets, options, how to read its output |
| [tools/ort-model-check/README.md](tools/ort-model-check/README.md) | The model check: runs a real ONNX model on each engine and compares the outputs and timings |

## Repository layout

```
README.md
LICENSE-MIT, LICENSE-APACHE
.githooks/                        pre-commit, pre-merge-commit, pre-push: run the licence guard
onnxruntime-split/PKGBUILD         the recipe: one build (pkgbase onnxruntime-split), split into onnxruntime-core, one package per engine, and python-onnxruntime-core
scripts/check-no-vendor.sh        licence guard: refuses vendor and binary files
scripts/check-deps.sh              checks built packages' library dependencies and version bounds against this system
scripts/smoke-test.sh             builds and runs tools/ort-smoke against an installed ONNX Runtime
scripts/test.sh                   runs every check of this repository
scripts/tests/                    tests of the guard, of the smoke-test wrapper, of check-deps.sh and of the PKGBUILD's build inputs
tools/ort-smoke/                  the smoke test program (C) and its model generator
tools/ort-model-check/            runs a real ONNX model on each engine, compares outputs and timings (Python)
docs/                             install, build, engine notes and measured results
```

## Testing

```bash
scripts/test.sh
```

This runs the checks that need no GPU, no network and no vendor install: shellcheck over every script and the PKGBUILD, the PKGBUILD's syntax and metadata (`makepkg --printsrcinfo`, with the names, provides and conflicts that consumers rely on), `namcap` when it is installed, every `scripts/tests/*-test.sh`, and the licence guard over the whole tree. It needs shellcheck: from `PATH`, from `SHELLCHECK=/path/to/shellcheck`, or, when `uvx` is installed, run as `uvx --from shellcheck-py shellcheck`. Checks of the packages on real hardware are done with [scripts/smoke-test.sh](tools/ort-smoke/README.md).

## Licence of this repository's content

MIT OR Apache-2.0, at your option: see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). This covers only the content of this repository. It does not apply to ONNX Runtime, to any vendor runtime, or to a package built from them.

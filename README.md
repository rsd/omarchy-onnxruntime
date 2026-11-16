# omarchy-onnxruntime

Arch Linux packages for [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.29.0 that put every execution provider an NVIDIA and Intel workstation can use into one installation: CPU, CUDA, TensorRT and OpenVINO (Intel CPU, GPU and NPU). Arch's own ONNX Runtime packages each serve one GPU vendor and have no TensorRT or OpenVINO provider. On request, the recipe also builds the MIGraphX provider for AMD GPUs; **that package is untested on AMD hardware** and has not been built yet ([AMD GPUs](#amd-gpus)).

An *execution provider* (EP) is the part of ONNX Runtime that runs a model on one kind of hardware. Each provider here is its own package, so a machine installs only the engines it has. Terms such as `pkgrel`, `-bin` and SM are explained in the [glossary](docs/glossary.md).

This repository holds the package recipe (`onnxruntime-split/PKGBUILD`), the scripts that check and release it, and the documentation. There are two ways to get the packages:

- **Install prebuilt packages** from the AUR package `onnxruntime-split-bin` ([Quick start](#quick-start)).
- **Build them from the recipe** with `makepkg`.

## Who should read what

- **You want to install the packages.** Read [docs/install.md](docs/install.md). Pick your hardware in [docs/hardware.md](docs/hardware.md) and the engine details in [docs/engines.md](docs/engines.md). For NVIDIA Pascal and Volta cards, see [docs/cuda12-legacy.md](docs/cuda12-legacy.md); for AMD GPUs, [docs/amd.md](docs/amd.md).
- **You want to build them.** Read [docs/build-from-source.md](docs/build-from-source.md), then [docs/build.md](docs/build.md) for the build inputs (engines, GPU architectures, TensorRT location, parallelism). [docs/dependencies.md](docs/dependencies.md) explains the version bounds and `scripts/check-deps.sh`.
- **You maintain a release.** Read [docs/release.md](docs/release.md) (and [docs/release-cuda12.md](docs/release-cuda12.md) for the CUDA 12 line) and [docs/testing.md](docs/testing.md).

## Packages

The recipe is one build of the upstream tag `v1.29.0`, split into one package per engine. With the default build and Arch's CUDA 13, the packages are:

| Package | Holds |
|---|---|
| `onnxruntime-core` | `libonnxruntime.so` with the CPU provider, the C and C++ headers, the pkg-config and CMake files. Works on its own, on the CPU. Every other package needs it |
| `onnxruntime-provider-cuda13-rtx` | The CUDA provider, compiled for NVIDIA GPUs from Turing (RTX 20) to Blackwell (RTX 50) |
| `onnxruntime-provider-tensorrt-cuda13` | The TensorRT provider (NVIDIA GPUs) |
| `onnxruntime-provider-openvino` | The OpenVINO provider (Intel CPU, GPU and NPU) |
| `python-onnxruntime-core` | The Python module |

Other builds name the CUDA package differently (`-cuda13-all`, `-cuda13-sm120`, `-cuda12-...`), and the MIGraphX provider is `onnxruntime-provider-migraphx`. [docs/packages.md](docs/packages.md) has the full table with dependencies and sizes, and the rules for the names.

`onnxruntime-core` and `python-onnxruntime-core` replace and conflict with `onnxruntime-full` and `python-onnxruntime-full`, the single packages of version 1.29.0-1 ([upgrading](docs/install.md#upgrading-from-the-single-package-1290-1)). Version 1.29.0 is the release Arch ships, so programs built against Arch's `onnxruntime` see the same API.

### Dependencies and version bounds

Every package depends on the package of each library it links. The vendor runtimes (`cuda`, `cudnn`, `tensorrt`, `openvino`, `python`, and for AMD `migraphx` and `hip-runtime-amd`) carry version bounds, so pacman refuses a system update that would break a provider instead of leaving one that fails to load. When Arch moves a runtime past a bound, the packages are rebuilt with a `pkgrel` bump. Details, and `scripts/check-deps.sh`, which shows when a rebuild is due: [docs/dependencies.md](docs/dependencies.md).

## Why not Arch's onnxruntime packages

Arch's `extra` repository ships 1.29.0 as separate builds (`onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm`, each with a `python-` counterpart). Each provides `onnxruntime` and conflicts with the others, so one is installed at a time, and none has the TensorRT or OpenVINO provider. Support for `sm_120` (Blackwell) is not the difference: the Arch CUDA builds have it. The difference is TensorRT and OpenVINO, and having the NVIDIA and Intel engines in one installation. The comparison table and the sources are in [docs/packages.md](docs/packages.md#why-not-archs-onnxruntime-packages).

### Why one package per engine, not one per build

The providers are shared libraries that one `libonnxruntime.so` loads when a session asks for them, so a single build serves every engine. Each package carries one provider and depends on that vendor's runtime. See [docs/packages.md](docs/packages.md#why-one-package-per-engine-not-one-per-build).

## Engines

| Engine (execution provider) | Hardware | Package | Vendor runtime packages |
|---|---|---|---|
| CPU | any x86_64 CPU | `onnxruntime-core` | none |
| CUDA | NVIDIA GPU | `onnxruntime-provider-cuda<M>-<sfx>` | `nvidia-utils`, `cuda`, `cudnn` |
| TensorRT | NVIDIA GPU | `onnxruntime-provider-tensorrt-cuda<M>` | TensorRT 11, and the CUDA runtimes |
| OpenVINO | Intel CPU, GPU, NPU | `onnxruntime-provider-openvino` | `openvino`, plus a plugin and driver per device |
| MIGraphX (untested on AMD hardware) | AMD GPU supported by ROCm | `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd` |

`<M>` is the major version of the CUDA toolkit that built the package (`13`) and `<sfx>` names the GPU architectures it holds code for. [docs/engines.md](docs/engines.md) lists the runtime packages and where each comes from, and how to use the library from C, Rust and Python. [docs/hardware.md](docs/hardware.md) says which GPUs, CPUs and NPUs each engine covers and what is not packaged (NVIDIA Maxwell cards, the Ryzen AI NPU, older Intel graphics, AMD Vega and Polaris).

### CUDA architecture presets

`ORTF_CUDA_ARCHS` chooses the GPU architectures the CUDA provider is compiled for. The default preset `rtx` covers Turing to Blackwell (`75 86 89 120`, package `-cuda13-rtx`, 154.7 MiB installed). The preset `all` adds A100, H100 and B200-class GPUs (`-cuda13-all`, 294.9 MiB). A list such as `120` builds for those GPUs only. The table with sizes and build times is in [docs/build.md](docs/build.md#cuda-architectures).

### Portable baseline

Packages target generic x86-64 and run on any x86-64 CPU; ONNX Runtime picks its AVX2 or AVX-512 kernels at run time. See [docs/build.md](docs/build.md#portable-baseline).

### AMD GPUs

ONNX Runtime 1.29 has no ROCm provider; the only AMD GPU provider is MIGraphX, built when `migraphx` is in `ORTF_EPS`. It is **untested on AMD hardware** (the maintainer has none) and no build of it has been made yet. It is not shipped prebuilt. See [docs/amd.md](docs/amd.md).

## Licence and redistribution

- **The recipe** (the content of this repository) is MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
- **The built ONNX Runtime** is MIT-licensed, plus the open-source dependencies compiled into its libraries. The PKGBUILD's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0`. Each package installs the notices in `/usr/share/licenses/<package>/`: ONNX Runtime's `LICENSE` and `ThirdPartyNotices.txt`, plus the licences of cpuinfo (`cpuinfo-LICENSE`, BSD-2-Clause) and protobuf (`protobuf-LICENSE`, BSD-3-Clause), taken from their source archives, so that each licence in the expression that Arch does not ship under `/usr/share/licenses/spdx/` has its text in the package (the check `namcap` makes). The CUDA package also carries cudnn-frontend's (`cudnn-frontend-LICENSE.txt`).
- **Each engine's runtime is under its vendor's licence.** The CUDA, cuDNN and TensorRT runtimes are under NVIDIA's licences; OpenVINO is Apache-2.0, and the Intel graphics and NPU drivers have their own licences. Read the licence of each runtime you install.
- **Nothing from NVIDIA, Intel or AMD is contained in any package, built from source or prebuilt, nor in this repository.** The repository holds only the recipe, scripts and documentation. The packages contain only ONNX Runtime and its bundled open-source dependencies, with their licences (the CUDA package also carries the licence of cudnn-frontend, an MIT-licensed header-only library compiled into it). The vendor runtimes are dependencies, which you install from their own packages: Arch's `cuda`, `cudnn`, `openvino`, `migraphx` and `hip-runtime-amd`, and, for TensorRT, the AUR's `tensorrt` or `nvidia-tensorrt-sdk-local`.
- **You may redistribute builds of this recipe as they are**, as the official prebuilt packages are. Never bundle vendor libraries in or alongside them. A build you change so that it contains a vendor library is outside this description.
- This is not legal advice.

The script `scripts/check-no-vendor.sh` refuses vendor and binary files, and content with an NVIDIA or Intel marker, in this repository. Enable its git hooks in your clone with `git config core.hooksPath .githooks`; [docs/testing.md](docs/testing.md#the-licence-guard-and-the-git-hooks) has the details.

## Quick start

### Prebuilt packages

With an AUR helper (`yay` here; any works), install the `-bin` packages of the engines you use. The core is always needed. The vendor runtimes are installed as dependencies from their own packages, except TensorRT, which is not in Arch's repositories: install the AUR's `tensorrt` or `nvidia-tensorrt-sdk-local` first ([TensorRT](docs/install.md#2-tensorrt)). Choose one CUDA package, `-rtx` or `-all`; they conflict.

```bash
yay -S onnxruntime-core-bin onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-tensorrt-cuda13-bin onnxruntime-provider-openvino-bin python-onnxruntime-core-bin
```

Without a helper, or for the exact files, see [Prebuilt packages](docs/install.md#prebuilt-packages) in the install guide. The prebuilt packages are built from this recipe in a clean chroot and published at <https://github.com/rsd/omarchy-onnxruntime/releases>, with a `SHA256SUMS` file. They hold no NVIDIA, Intel or AMD library ([licence](#licence-and-redistribution)).

### Building from source

```bash
# 1. build tools, and the runtimes of the engines you want (docs/install.md and docs/build-from-source.md list them)
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools cuda cudnn openvino

# 2. TensorRT, for the TensorRT provider: nvidia-tensorrt-sdk-local (pkgrel 2 or later) or the AUR's tensorrt (docs/install.md#2-tensorrt)

# 3. build
cd onnxruntime-split
MAKEFLAGS=-j8 makepkg -f
cd ..
```

Then install the packages the build wrote, naming the exact files in one `pacman -U` (a glob can catch packages of an earlier build), and run the smoke test. The commands, the checks, the build inputs and the time and disk a build needs are in [docs/build-from-source.md](docs/build-from-source.md). A full build with one CUDA architecture (`ORTF_CUDA_ARCHS=120`) took 27 minutes on a 24-thread machine and needs tens of GiB of disk; the default `rtx` has four architectures and takes longer.

A subset of engines is fine: the core alone, the core and `onnxruntime-provider-openvino`, and so on.

## Documentation

| Page | Contents |
|---|---|
| [docs/install.md](docs/install.md) | Installing: prebuilt packages, runtimes, TensorRT, verifying, upgrading from 1.29.0-1, uninstalling |
| [docs/build-from-source.md](docs/build-from-source.md) | Building and installing from the recipe, step by step |
| [docs/build.md](docs/build.md) | Build inputs, CUDA architectures, package names, dependency mirroring, ccache, time, memory and disk |
| [docs/packages.md](docs/packages.md) | The packages, their dependencies and sizes, and why they are split this way |
| [docs/dependencies.md](docs/dependencies.md) | Version bounds on vendor runtimes, and `scripts/check-deps.sh` |
| [docs/engines.md](docs/engines.md) | Per-engine notes (CUDA, TensorRT, OpenVINO), using the library from C, Rust and Python |
| [docs/hardware.md](docs/hardware.md) | Which GPUs, CPUs and NPUs each engine covers; what is not packaged |
| [docs/amd.md](docs/amd.md) | AMD GPUs: the MIGraphX provider (untested) |
| [docs/cuda12-legacy.md](docs/cuda12-legacy.md) | NVIDIA Pascal and Volta: the CUDA 12 provider, install and build |
| [docs/release.md](docs/release.md) | The maintainer's release procedure |
| [docs/release-cuda12.md](docs/release-cuda12.md) | The release procedure of the CUDA 12 line |
| [docs/testing.md](docs/testing.md) | `scripts/test.sh`, the licence guard and the git hooks |
| [docs/results.md](docs/results.md) | Measured results of the smoke test and model check |
| [docs/glossary.md](docs/glossary.md) | The terms these pages use |
| [tools/ort-smoke/README.md](tools/ort-smoke/README.md) | The smoke test: targets, options, how to read its output |
| [tools/ort-model-check/README.md](tools/ort-model-check/README.md) | The model check: runs a real ONNX model on each engine and compares outputs and timings |

## Repository layout

```
README.md
LICENSE-MIT, LICENSE-APACHE
.githooks/                          pre-commit, pre-merge-commit, pre-push: run the licence guard
onnxruntime-split/PKGBUILD          the recipe: one build (pkgbase onnxruntime-split), split into onnxruntime-core, one package per engine, and python-onnxruntime-core
onnxruntime-split/.SRCINFO          the recipe's metadata for the AUR, at its default inputs
onnxruntime-split-cuda12/PKGBUILD   the CUDA 12 recipe (pkgbase onnxruntime-split-cuda12), generated by scripts/gen-cuda12.sh; do not edit
onnxruntime-split-cuda12/.SRCINFO   its metadata for the AUR
scripts/check-no-vendor.sh          licence guard: refuses vendor and binary files
scripts/check-deps.sh               checks built packages' library dependencies and version bounds against this system
scripts/check-cuda-archs.sh         checks which GPU architectures a built CUDA provider holds
scripts/srcinfo.sh                  writes, checks or copies to an AUR clone a recipe's .SRCINFO
scripts/gen-bin.sh                  generates the PKGBUILD, .SRCINFO and SHA256SUMS of a prebuilt (-bin) AUR package from a release's packages
scripts/gen-cuda12.sh               generates onnxruntime-split-cuda12/PKGBUILD from onnxruntime-split/PKGBUILD
scripts/smoke-test.sh               builds and runs tools/ort-smoke against an installed ONNX Runtime
scripts/test.sh                     runs every check of this repository
scripts/tests/                      tests of the scripts above and of the PKGBUILD's build inputs
tools/ort-smoke/                    the smoke test program (C) and its model generator
tools/ort-model-check/              runs a real ONNX model on each engine, compares outputs and timings (Python)
docs/                               the documentation, indexed above
aur/                                (not tracked) clones of the AUR repositories, see docs/release.md
```

## Testing

`scripts/test.sh` runs every check that needs no GPU, no network and no vendor install. Checks of the packages on real hardware use `scripts/smoke-test.sh` and `tools/ort-model-check`. See [docs/testing.md](docs/testing.md).

## Licence of this repository's content

MIT OR Apache-2.0, at your option: see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). This covers only the content of this repository. It does not apply to ONNX Runtime, to any vendor runtime, or to a package built from them.

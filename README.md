# omarchy-onnxruntime

Arch Linux packages for [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.29.0 that aim at the broadest hardware and feature support from one package set: CPU, oneDNN, CUDA, TensorRT and OpenVINO (Intel CPU, GPU and NPU) in one installation, a CUDA 12 package for NVIDIA Pascal and Volta, a MIGraphX package for AMD GPUs, and the training operators. It is an alternative to Arch's own `onnxruntime-*` packages, never installed beside them: it has every engine and feature they have, plus TensorRT, OpenVINO and the CUDA 12 line ([choosing.md](docs/choosing.md)). The CUDA 12 and MIGraphX packages are **untested on their hardware** (Pascal and Volta, AMD GPUs).

An *execution provider* (EP) is the part of ONNX Runtime that runs a model on one kind of hardware. Each provider here is its own package, so a machine installs only the engines it has. Terms such as `pkgrel`, `-bin` and SM are explained in the [glossary](docs/glossary.md).

This repository holds the package recipe (`onnxruntime-split/PKGBUILD`), the scripts that check and release it, and the documentation. There are two ways to get the packages:

- **Install prebuilt packages** from the AUR package `onnxruntime-split-bin` ([Quick start](#quick-start)).
- **Build them from the recipe** with `makepkg`.

## Who should read what

- **You are choosing between this set and Arch's packages.** Read [docs/choosing.md](docs/choosing.md): what each offers, a comparison table and which package for which hardware.
- **You want to install the packages.** Read [docs/install.md](docs/install.md). Pick your hardware in [docs/hardware.md](docs/hardware.md) and the engine details in [docs/engines.md](docs/engines.md). For NVIDIA Pascal and Volta cards, see [docs/cuda12-legacy.md](docs/cuda12-legacy.md); for AMD GPUs, [docs/amd.md](docs/amd.md).
- **You want to build them.** Read [docs/build-from-source.md](docs/build-from-source.md), then [docs/build.md](docs/build.md) for the build inputs (engines, GPU architectures, TensorRT location, parallelism). [docs/dependencies.md](docs/dependencies.md) explains the version bounds and `scripts/check-deps.sh`.
- **You maintain a release.** Read [docs/release.md](docs/release.md) (and [docs/release-cuda12.md](docs/release-cuda12.md) for the CUDA 12 line, [docs/release-migraphx.md](docs/release-migraphx.md) for the MIGraphX line; the main and CUDA 12 lines release together) and [docs/testing.md](docs/testing.md).

## Packages

The recipe is one build of the upstream tag `v1.29.0`, split into one package per engine. With the default build and Arch's CUDA 13, the packages are:

| Package | Holds |
|---|---|
| `onnxruntime-core` | `libonnxruntime.so` with the CPU provider and the training operators, the C and C++ headers, the pkg-config and CMake files. Works on its own, on the CPU. Depends on `nccl`. Every other package needs it |
| `onnxruntime-provider-cuda13-rtx` | The CUDA provider, compiled for NVIDIA GPUs from Turing (RTX 20) to Blackwell (RTX 50) |
| `onnxruntime-provider-tensorrt-cuda13` | The TensorRT provider (NVIDIA GPUs) |
| `onnxruntime-provider-openvino` | The OpenVINO provider (Intel CPU, GPU and NPU) |
| `onnxruntime-provider-dnnl` | The oneDNN provider (CPU), with oneDNN built in; no dependency on Arch's `onednn` |
| `python-onnxruntime-core` | The Python module, with `onnxruntime.training` (which needs `python-pytorch`, optional) |

Other builds name the CUDA package differently (`-cuda13-all`, `-cuda13-sm120`). Two packages come from their own recipes: `onnxruntime-provider-cuda12-legacy` (Pascal and Volta, [docs/cuda12-legacy.md](docs/cuda12-legacy.md)) and `onnxruntime-provider-migraphx`, with its Python add-on `python-onnxruntime-provider-migraphx` (AMD GPUs, AUR source only, [docs/amd.md](docs/amd.md)). A build with `ORTF_NCCL=0` names every package `-nccloff` and leaves out the `nccl` dependency (local builds only, [docs/build.md](docs/build.md#nccl-and-the-nccloff-packages)). [docs/packages.md](docs/packages.md) has the full table with dependencies and sizes, and the rules for the names.

`onnxruntime-core` and `python-onnxruntime-core` replace and conflict with `onnxruntime-full` and `python-onnxruntime-full`, the single packages of version 1.29.0-1 ([upgrading](docs/install.md#upgrading-from-the-single-package-1290-1)). Version 1.29.0 is the release Arch ships, so programs built against Arch's `onnxruntime` see the same API.

### Dependencies and version bounds

Every package depends on the package of each library it links. The vendor runtimes (`cuda`, `cudnn`, `tensorrt`, `openvino`, `python`, and for AMD `migraphx` and `hip-runtime-amd`) carry version bounds, so pacman refuses a system update that would break a provider instead of leaving one that fails to load. When Arch moves a runtime past a bound, the packages are rebuilt with a `pkgrel` bump. Details, and `scripts/check-deps.sh`, which shows when a rebuild is due: [docs/dependencies.md](docs/dependencies.md).

## Choosing between this set and Arch's

Arch's `extra` repository ships 1.29.0 as five builds (`onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm`, each with a `python-` twin). Each provides `onnxruntime` and conflicts with the others, and this set's `onnxruntime-core` conflicts with all of them, because they install the same `libonnxruntime.so`: one set per machine. This set has every engine and feature Arch's has (the training operators, oneDNN, NCCL and the MIGraphX provider) and adds TensorRT, OpenVINO and the CUDA 12 line for Pascal and Volta. Arch's packages are the better choice if you want only the official repository, or the smallest CPU-only install: by default this set depends on `nccl` even on a machine without an NVIDIA GPU. The comparison table, with the sources, and which package fits which hardware are in [docs/choosing.md](docs/choosing.md).

### Why one package per engine, not one per build

The providers are shared libraries that one `libonnxruntime.so` loads when a session asks for them, so a single build serves every engine. Each package carries one provider and depends on that vendor's runtime. See [docs/packages.md](docs/packages.md#why-one-package-per-engine-not-one-per-build).

## Engines

| Engine (execution provider) | Hardware | Package | Vendor runtime packages |
|---|---|---|---|
| CPU | any x86_64 CPU | `onnxruntime-core` | none |
| CUDA | NVIDIA GPU | `onnxruntime-provider-cuda<M>-<sfx>` | `nvidia-utils`, `cuda`, `cudnn` |
| TensorRT | NVIDIA GPU | `onnxruntime-provider-tensorrt-cuda<M>` | TensorRT 11, and the CUDA runtimes |
| OpenVINO | Intel CPU, GPU, NPU | `onnxruntime-provider-openvino` | `openvino`, plus a plugin and driver per device |
| oneDNN | any x86_64 CPU | `onnxruntime-provider-dnnl` | none (oneDNN is built in) |
| MIGraphX (untested on AMD hardware) | AMD GPU in Arch's ROCm build list (Vega and newer) | `onnxruntime-provider-migraphx` (AUR source package; `python-onnxruntime-provider-migraphx` for Python) | `migraphx`, `hip-runtime-amd` |

`<M>` is the major version of the CUDA toolkit that built the package (`13`) and `<sfx>` names the GPU architectures it holds code for. [docs/engines.md](docs/engines.md) lists the runtime packages and where each comes from, and how to use the library from C, Rust and Python. [docs/hardware.md](docs/hardware.md) says which GPUs, CPUs and NPUs each engine covers and what is not packaged (NVIDIA Maxwell cards, the Ryzen AI NPU, older Intel graphics, AMD Vega and Polaris).

### CUDA architecture presets

`ORTF_CUDA_ARCHS` chooses the GPU architectures the CUDA provider is compiled for. The default preset `rtx` covers Turing to Blackwell (`75 86 89 120`, package `-cuda13-rtx`, 154.7 MiB installed). The preset `all` adds A100, H100 and B200-class GPUs (`-cuda13-all`, 294.9 MiB). A list such as `120` builds for those GPUs only. The table with sizes and build times is in [docs/build.md](docs/build.md#cuda-architectures).

### Portable baseline

Packages target generic x86-64 and run on any x86-64 CPU; ONNX Runtime picks its AVX2 or AVX-512 kernels at run time. See [docs/build.md](docs/build.md#portable-baseline).

### AMD GPUs

ONNX Runtime 1.29 has no ROCm provider; the only AMD GPU provider is MIGraphX. It is shipped as `onnxruntime-provider-migraphx` and its Python add-on `python-onnxruntime-provider-migraphx` from their own AUR package base, `onnxruntime-split-migraphx`, as a source build only (no prebuilt package), and it is **untested on AMD hardware**: the maintainer has none. If you have an AMD GPU, a report of a result is welcome ([how to report](docs/amd.md#reporting-a-result)). See [docs/amd.md](docs/amd.md).

## Licence and redistribution

- **The recipe** (the content of this repository) is MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
- **The built ONNX Runtime** is MIT-licensed, plus the open-source dependencies compiled into its libraries. The PKGBUILD's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0`. Each package installs the notices in `/usr/share/licenses/<package>/`: ONNX Runtime's `LICENSE` and `ThirdPartyNotices.txt`, plus the licences of cpuinfo (`cpuinfo-LICENSE`, BSD-2-Clause) and protobuf (`protobuf-LICENSE`, BSD-3-Clause), taken from their source archives, so that each licence in the expression that Arch does not ship under `/usr/share/licenses/spdx/` has its text in the package (the check `namcap` makes). The CUDA package also carries cudnn-frontend's (`cudnn-frontend-LICENSE.txt`), and the oneDNN package oneDNN's licence (Apache-2.0) and third-party notices (`onednn-LICENSE`, `onednn-THIRD-PARTY-PROGRAMS`).
- **Each engine's runtime is under its vendor's licence.** The CUDA, cuDNN and TensorRT runtimes are under NVIDIA's licences; NCCL is NVIDIA's, RCCL and the ROCm runtimes are AMD's; OpenVINO is Apache-2.0, and the Intel graphics and NPU drivers have their own licences. Read the licence of each runtime you install.
- **Nothing from NVIDIA, Intel or AMD is contained in any package, built from source or prebuilt, nor in this repository.** The repository holds only the recipe, scripts and documentation. The packages contain only ONNX Runtime and its bundled open-source dependencies, with their licences (the CUDA package also carries the licence of cudnn-frontend, an MIT-licensed header-only library compiled into it; the oneDNN package contains oneDNN, which is Apache-2.0, built into it). The vendor runtimes are dependencies, which you install from their own packages: Arch's `cuda`, `cudnn`, `nccl`, `openvino`, `migraphx` and `hip-runtime-amd`, the AUR's `nccl-cuda12.9` for the CUDA 12 line, and, for TensorRT, the AUR's `tensorrt` or `nvidia-tensorrt-sdk-local`.
- **You may redistribute builds of this recipe as they are**, as the official prebuilt packages are. Never bundle vendor libraries in or alongside them. A build you change so that it contains a vendor library is outside this description.
- This is not legal advice.

The script `scripts/check-no-vendor.sh` refuses vendor and binary files, and content with an NVIDIA or Intel marker, in this repository. Enable its git hooks in your clone with `git config core.hooksPath .githooks`; [docs/testing.md](docs/testing.md#the-licence-guard-and-the-git-hooks) has the details.

## Quick start

### Prebuilt packages

With an AUR helper (`yay` here; any works), install the `-bin` packages of the engines you use. The core is always needed. The vendor runtimes are installed as dependencies from their own packages, except TensorRT, which is not in Arch's repositories: install the AUR's `tensorrt` or `nvidia-tensorrt-sdk-local` first ([TensorRT](docs/install.md#2-tensorrt)). Choose one CUDA package, `-rtx` or `-all`; they conflict.

```bash
yay -S onnxruntime-core-bin onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-tensorrt-cuda13-bin onnxruntime-provider-openvino-bin onnxruntime-provider-dnnl-bin python-onnxruntime-core-bin
```

Without a helper, or for the exact files, see [Prebuilt packages](docs/install.md#prebuilt-packages) in the install guide. The prebuilt packages are built from this recipe in a clean chroot and published at <https://github.com/rsd/omarchy-onnxruntime/releases>, with a `SHA256SUMS` file. They hold no NVIDIA, Intel or AMD library ([licence](#licence-and-redistribution)).

### Building from source

```bash
# 1. build tools, and the runtimes of the engines you want (docs/install.md and docs/build-from-source.md list them)
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools python-cerberus python-h5py python-onnx python-sympy cuda cudnn nccl openvino

# 2. TensorRT, for the TensorRT provider: nvidia-tensorrt-sdk-local (pkgrel 2 or later) or the AUR's tensorrt (docs/install.md#2-tensorrt)

# 3. build
cd onnxruntime-split
MAKEFLAGS=-j8 makepkg -f
cd ..
```

Then install the packages the build wrote, naming the exact files in one `pacman -U` (a glob can catch packages of an earlier build), and run the smoke test. The commands, the checks, the build inputs and the time and disk a build needs are in [docs/build-from-source.md](docs/build-from-source.md). A full build with one CUDA architecture (`ORTF_CUDA_ARCHS=120`) took 27 minutes on a 24-thread machine and needs tens of GiB of disk; the default `rtx` has four architectures and takes longer.

`github.com/microsoft/onnxruntime` has returned 404 since 2026-11-17, so the recipes fetch the ONNX Runtime source from this project's release `source-v1.29.0` by default; `ORTF_SOURCE=upstream` selects the git tag again ([docs/build.md](docs/build.md#the-onnx-runtime-source-mirror-or-upstream)). The prebuilt packages are not affected.

A subset of engines is fine: the core alone, the core and `onnxruntime-provider-openvino`, and so on. A build without CUDA needs `ORTF_NCCL=0` ([docs/build.md](docs/build.md#nccl-and-the-nccloff-packages)).

## Documentation

| Page | Contents |
|---|---|
| [docs/choosing.md](docs/choosing.md) | This set versus Arch's `onnxruntime-*` packages: what each offers, a comparison table, which package for which hardware |
| [docs/install.md](docs/install.md) | Installing: prebuilt packages, runtimes, TensorRT, verifying, upgrading from 1.29.0-1, uninstalling |
| [docs/build-from-source.md](docs/build-from-source.md) | Building and installing from the recipe, step by step |
| [docs/build.md](docs/build.md) | Build inputs, CUDA architectures, package names, dependency mirroring, ccache, time, memory and disk |
| [docs/packages.md](docs/packages.md) | The packages, their dependencies and sizes, and why they are split this way |
| [docs/dependencies.md](docs/dependencies.md) | Version bounds on vendor runtimes, and `scripts/check-deps.sh` |
| [docs/engines.md](docs/engines.md) | Per-engine notes (CUDA, TensorRT, OpenVINO, oneDNN), using the library from C, Rust and Python |
| [docs/hardware.md](docs/hardware.md) | Which GPUs, CPUs and NPUs each engine covers; what is not packaged |
| [docs/amd.md](docs/amd.md) | AMD GPUs: the MIGraphX provider (untested), how to install it and how to report a result |
| [docs/cuda12-legacy.md](docs/cuda12-legacy.md) | NVIDIA Pascal and Volta: the CUDA 12 provider, install and build |
| [docs/release.md](docs/release.md) | The maintainer's release procedure |
| [docs/release-cuda12.md](docs/release-cuda12.md) | The release procedure of the CUDA 12 line |
| [docs/release-migraphx.md](docs/release-migraphx.md) | The release procedure of the MIGraphX line (AUR source only) |
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
onnxruntime-split-migraphx/PKGBUILD the MIGraphX recipe (pkgbase onnxruntime-split-migraphx), generated by scripts/gen-migraphx.sh; do not edit
onnxruntime-split-migraphx/.SRCINFO its metadata for the AUR
scripts/check-no-vendor.sh          licence guard: refuses vendor and binary files
scripts/check-deps.sh               checks built packages' library dependencies and version bounds against this system
scripts/check-cuda-archs.sh         checks which GPU architectures a built CUDA provider holds
scripts/make-source-archive.sh      makes the ONNX Runtime source archive that the mirror release hosts, reproducibly
scripts/check-upstream.sh           says whether github.com/microsoft/onnxruntime is back and still has our commit
scripts/srcinfo.sh                  writes, checks or copies to an AUR clone a recipe's .SRCINFO
scripts/gen-bin.sh                  generates the PKGBUILD, .SRCINFO and SHA256SUMS of a prebuilt (-bin) AUR package from a release's packages
scripts/gen-cuda12.sh               generates onnxruntime-split-cuda12/PKGBUILD from onnxruntime-split/PKGBUILD
scripts/gen-migraphx.sh             generates onnxruntime-split-migraphx/PKGBUILD from onnxruntime-split/PKGBUILD
scripts/gen-line-lib.sh             the generation rules shared by the two generators above
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

# omarchy-onnxruntime

Run ONNX models on the GPU or NPU of your machine, installed with pacman. These are pacman packages of ONNX Runtime 1.29.0 for Arch Linux and Omarchy (x86_64), split into one package per hardware engine: the CPU, NVIDIA CUDA and TensorRT, and Intel OpenVINO. They are for programs and people that need ONNX Runtime beyond the CPU.

ONNX (Open Neural Network Exchange) is a standard file format for trained neural networks. ONNX Runtime is the library that runs such a file. An *engine* (an execution provider, or EP) is the part of ONNX Runtime that runs a model on one kind of hardware. An NPU (neural processing unit) is an AI accelerator chip, such as the one in recent Intel laptops. Here each engine is its own package, so a machine installs only the engines it has. Terms such as `pkgrel`, `-bin` and SM (an NVIDIA GPU's architecture version) are in the [glossary](docs/glossary.md).

## What you get

- **Every engine in one installation.** The CPU, oneDNN (Intel's CPU math library), CUDA (NVIDIA's GPU platform), TensorRT (NVIDIA's inference optimizer) and OpenVINO (Intel's toolkit for Intel CPUs, GPUs and NPUs) work side by side.
- **A CUDA 12 package** for NVIDIA Pascal and Volta, which CUDA 13 dropped. It is untested on that hardware.
- **A MIGraphX package** for AMD GPUs. It is a source build only and untested on AMD hardware.
- **The training operators** and the Python module `onnxruntime`, with `onnxruntime.training`.
- **The same API as Arch's package.** Version 1.29.0 is the release Arch ships, so programs built against Arch's `onnxruntime` see the same API.

## Why use these packages

- **More engines than Arch's packages.** Arch's `extra` repository ships 1.29.0 as five builds (`onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm`, each with a `python-` twin). This set has every engine and feature they have: the training operators, oneDNN, NCCL and the MIGraphX provider. It adds TensorRT, OpenVINO and the CUDA 12 line for Pascal and Volta.
- **One package per engine.** The providers are shared libraries that one `libonnxruntime.so` loads when a session asks for them, so a single build serves every engine. Each package carries one provider and depends on that vendor's runtime. See [docs/packages.md](docs/packages.md#why-one-package-per-engine-not-one-per-build).
- **Dependencies that cannot break silently.** Every package depends on the package of each library it links. The vendor runtimes carry version bounds: `cuda`, `cudnn`, `tensorrt`, `openvino`, `python`, and for AMD `migraphx` and `hip-runtime-amd`. Pacman refuses a system update that would break a provider instead of leaving one that fails to load.
- **Rebuilds when a bound moves.** When Arch moves a runtime past a bound, the packages are rebuilt with a `pkgrel` bump. `scripts/check-deps.sh` shows when a rebuild is due: [docs/dependencies.md](docs/dependencies.md).
- **Portable binaries.** Packages target generic x86-64 and run on any x86-64 CPU. ONNX Runtime picks its AVX2 or AVX-512 kernels at run time ([docs/build.md](docs/build.md#portable-baseline)).
- **Prebuilt, from a clean chroot.** The `-bin` packages are built from this recipe in a clean chroot and published at <https://github.com/rsd/omarchy-onnxruntime/releases> with a `SHA256SUMS` file. They hold no NVIDIA, Intel or AMD library ([licence](#licence-and-redistribution)).

Arch's packages are the better choice in some cases:

- You want only the official repository, or the smallest CPU-only install. By default this set depends on `nccl` (NVIDIA's library for communication between GPUs), even on a machine without an NVIDIA GPU.
- Each of Arch's packages provides `onnxruntime` and conflicts with the others. This set's `onnxruntime-core` conflicts with all of them, because they install the same `libonnxruntime.so`. Install one set per machine.

The comparison table, with its sources, and the package for each kind of hardware are in [docs/choosing.md](docs/choosing.md).

## Install

On Arch Linux or Omarchy, install from the AUR (the Arch User Repository) with an AUR helper such as `yay`. Install the core, which runs on the CPU, together with the engines your machine has, in one call. The core replaces Arch's own `onnxruntime-*` packages if you have them. This example adds NVIDIA CUDA, Intel OpenVINO and the Python module:

```bash
yay -S onnxruntime-core-bin onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-openvino-bin python-onnxruntime-core-bin
```

For the CPU only, install `onnxruntime-core-bin` alone.

Check that it works. This lists the installed packages, the engine libraries beside `libonnxruntime.so`, and the engines Python sees:

```bash
pacman -Q | grep onnxruntime
ls /usr/lib/libonnxruntime*.so
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

[INSTALL.md](INSTALL.md) has the rest: the package for each kind of hardware, the vendor runtimes, TensorRT (it is not in Arch's repositories), installing without an AUR helper, building from the recipe, troubleshooting and uninstalling.

These are pacman packages. On another Linux distribution, use ONNX Runtime from [onnxruntime.ai](https://onnxruntime.ai), or the source in this project's [`source-v1.29.0` release](https://github.com/rsd/omarchy-onnxruntime/releases/tag/source-v1.29.0).

[omarchy-broadcast](https://github.com/rsd/omarchy-broadcast) is a program that can use these packages. Its [ONNX setup guide](https://github.com/rsd/omarchy-broadcast/blob/main/docs/setup/onnx.md) shows which engines it uses and how to set them up.

## Limits

- **Arch Linux and Omarchy on x86_64 only.** The packages are pacman packages.
- **Untested hardware:** the CUDA 12 package on NVIDIA Pascal and Volta, and the MIGraphX package on AMD GPUs ([docs/hardware.md](docs/hardware.md), [docs/amd.md](docs/amd.md)). The rest is tested on one machine, an NVIDIA GeForce RTX 5080 Laptop GPU and an Intel Core Ultra 9 275HX ([docs/choosing.md](docs/choosing.md)).
- **Not packaged:** NVIDIA Maxwell, the AMD Ryzen AI NPU, older Intel graphics, and AMD Polaris and Vega APUs ([docs/hardware.md](docs/hardware.md)).
- **NCCL on every machine.** The prebuilt core depends on `nccl`, 364.18 MiB installed, even without an NVIDIA GPU. Only a source build with `ORTF_NCCL=0` leaves it out ([docs/choosing.md](docs/choosing.md#where-arch-is-the-better-choice)).

## Documentation

| Page | Contents |
|---|---|
| [INSTALL.md](INSTALL.md) | Installing: choosing packages, prebuilt or source, runtimes, TensorRT, upgrades, checks, troubleshooting, uninstalling |
| [docs/choosing.md](docs/choosing.md) | This set versus Arch's `onnxruntime-*` packages: what each offers, a comparison table, which package for which hardware |
| [docs/packages.md](docs/packages.md) | The packages, their dependencies and sizes, and why they are split this way |
| [docs/engines.md](docs/engines.md) | Per-engine notes (CUDA, TensorRT, OpenVINO, oneDNN), the vendor runtimes, using the library from C, Rust and Python |
| [docs/hardware.md](docs/hardware.md) | Which GPUs, CPUs and NPUs each engine covers; what is not packaged |
| [docs/amd.md](docs/amd.md) | AMD GPUs: the MIGraphX provider (untested), how to install it and how to report a result |
| [docs/cuda12-legacy.md](docs/cuda12-legacy.md) | NVIDIA Pascal and Volta: the CUDA 12 provider, install and build |
| [docs/build-from-source.md](docs/build-from-source.md) | Building and installing from the recipe, step by step |
| [docs/build.md](docs/build.md) | Build inputs, CUDA architectures, package names, dependency mirroring, the source mirror, ccache, time, memory and disk |
| [docs/dependencies.md](docs/dependencies.md) | Version bounds on vendor runtimes, and `scripts/check-deps.sh` |
| [docs/results.md](docs/results.md) | Measured results of the smoke test and model check |
| [docs/glossary.md](docs/glossary.md) | The terms these pages use |
| [tools/ort-smoke/README.md](tools/ort-smoke/README.md) | The smoke test: targets, options, how to read its output |
| [tools/ort-model-check/README.md](tools/ort-model-check/README.md) | The model check: runs a real ONNX model on each engine and compares outputs and timings |

## Maintainers

The repository layout, the tests and the release procedures are in [docs/maintainers.md](docs/maintainers.md).

## Licence and redistribution

- **The recipe** (the content of this repository) is MIT OR Apache-2.0, at your option: [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). This covers only the content of this repository. It does not apply to ONNX Runtime, to any vendor runtime, or to a package built from them.
- **The built ONNX Runtime** is MIT-licensed, plus the open-source dependencies compiled into its libraries. The PKGBUILD's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0`. Each package installs its notices in `/usr/share/licenses/<package>/`:
  - ONNX Runtime's `LICENSE` and `ThirdPartyNotices.txt`.
  - The licences of cpuinfo (`cpuinfo-LICENSE`, BSD-2-Clause) and protobuf (`protobuf-LICENSE`, BSD-3-Clause), taken from their source archives. Each licence in the expression that Arch does not ship under `/usr/share/licenses/spdx/` therefore has its text in the package, which is the check `namcap` makes.
  - In the CUDA package, also cudnn-frontend's licence (`cudnn-frontend-LICENSE.txt`).
  - In the oneDNN package, also oneDNN's licence (Apache-2.0) and third-party notices (`onednn-LICENSE`, `onednn-THIRD-PARTY-PROGRAMS`).
- **Each engine's runtime is under its vendor's licence.** The CUDA, cuDNN and TensorRT runtimes are under NVIDIA's licences. NCCL is NVIDIA's, and RCCL and the ROCm runtimes are AMD's. OpenVINO is Apache-2.0, and the Intel graphics and NPU drivers have their own licences. Read the licence of each runtime you install.
- **Nothing from NVIDIA, Intel or AMD is contained in any package, built from source or prebuilt, or in this repository.** The repository holds only the recipe, scripts and documentation. The packages contain only ONNX Runtime and its bundled open-source dependencies, with their licences. The CUDA package also carries the licence of cudnn-frontend, an MIT-licensed header-only library compiled into it. The oneDNN package contains oneDNN, which is Apache-2.0, built into it. You install the vendor runtimes from their own packages: Arch's `cuda`, `cudnn`, `nccl`, `openvino`, `migraphx` and `hip-runtime-amd`, the AUR's `nccl-cuda12.9` for the CUDA 12 line, and, for TensorRT, the AUR's `tensorrt` or `nvidia-tensorrt-sdk-local`.
- **You may redistribute builds of this recipe as they are**, as the official prebuilt packages are. Never bundle vendor libraries in or alongside them. A build you change so that it contains a vendor library is outside this description.
- This is not legal advice.

The script `scripts/check-no-vendor.sh` refuses vendor and binary files, and content with an NVIDIA or Intel marker, in this repository. Enable its git hooks in your clone with this command:

```bash
git config core.hooksPath .githooks
```

[docs/testing.md](docs/testing.md#the-licence-guard-and-the-git-hooks) has the details.

# Choosing between this repository's packages and Arch's

Arch's `extra` repository has ONNX Runtime packages of its own. This repository builds another set from the same upstream release, 1.29.0, and the two cannot be installed together. This page says what each offers so that you can pick one set for a machine. Terms such as pkgrel, `-bin` and SM are in the [glossary](glossary.md).

In short: this set has every engine and feature that Arch's has (the training operators, oneDNN, NCCL and the MIGraphX provider for AMD GPUs), and adds TensorRT, OpenVINO (Intel CPU, GPU and NPU) and a CUDA 12 line for NVIDIA Pascal and Volta. The costs are that it comes from the AUR and not from a repository, that two of its packages (MIGraphX and CUDA 12) have not run on their hardware, and that its default packages pull in NCCL on every machine ([below](#where-arch-is-the-better-choice)).

## What Arch ships

`pacman -Ss onnxruntime` lists five builds of 1.29.0 (pkgrel 3 on 2026-11-17), each with a `python-` twin: ten packages. Each build is a separate compilation of the whole library. Every one provides `onnxruntime` and conflicts with it, so one is installed at a time. Sizes and dependencies below are from `pacman -Si`; the engines are from the build options in Arch's [PKGBUILD](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime).

| Package | Engines | CPU code | Depends on (besides `glibc`, `libgcc`, `libstdc++`) | Download / installed |
|---|---|---|---|---|
| `onnxruntime-cpu` | CPU | generic x86-64 | `cpuinfo`, `protobuf` | 6.92 / 25.68 MiB |
| `onnxruntime-cuda` | CPU, CUDA, oneDNN | generic x86-64 | `cpuinfo`, `protobuf`, `cuda`, `cudnn`, `cudnn-frontend`, `nccl`, `onednn` | 406.37 / 618.75 MiB |
| `onnxruntime-opt-cuda` | CPU, CUDA, oneDNN | built with `-march=haswell` (AVX2) | the same | 406.86 / 619.35 MiB |
| `onnxruntime-rocm` | CPU, MIGraphX, oneDNN | generic x86-64 | `cpuinfo`, `protobuf`, `rocm-hip-sdk`, `roctracer`, `rccl`, `migraphx` | 7.25 / 26.67 MiB |
| `onnxruntime-opt-rocm` | CPU, MIGraphX, oneDNN | built with `-march=haswell` | the same | 7.37 / 27.17 MiB |

The `python-onnxruntime-*` twins have the same dependencies plus `python-numpy`, `python-psutil`, `python-py-cpuinfo`, `python-sympy`, `python-scipy`, `python-pillow`, `python-flatbuffers`, `python-protobuf` and `python-packaging` (and no `cudnn-frontend`). Their installed sizes are 64.41 MiB (`cpu`), 657.69 MiB (`cuda`), 658.79 MiB (`opt-cuda`), 65.44 MiB (`rocm`) and 66.45 MiB (`opt-rocm`).

Arch's recipe builds the CUDA variants for the architectures `75` to `121` (a list that includes the Arm-only Jetson and Thor targets), so Blackwell (`sm_120`) is supported. Every variant is configured with the training operators. The ROCm variants are configured with ONNX Runtime's MIGraphX provider: ONNX Runtime 1.29 has no ROCm provider, and the HIP architecture list in that recipe has no effect ([amd.md](amd.md#what-exists-for-amd-gpus)). The recipe enables neither TensorRT nor OpenVINO.

## Why one set per machine

Arch's packages and this repository's `onnxruntime-core` both install ONNX Runtime's core library, `/usr/lib/libonnxruntime.so` and its versioned names. Two packages cannot own one file, so the choice is exclusive, and `onnxruntime-core` declares that it conflicts with every Arch variant by name (`onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm`, `onnxruntime-git`). `python-onnxruntime-core` does the same for the `python-` twins. When you install one set on a machine that has the other, pacman offers to remove the other.

Programs built against Arch's `onnxruntime` keep working: `onnxruntime-core` provides `onnxruntime=1.29.0` and `libonnxruntime.so`, and the version, and so the API, is the same. Building this repository's engines on top of Arch's library was not chosen: it would tie every release here to Arch's pkgrel, flags and patches.

## Comparison

"Arch" is the five builds above. "This repository" is the default prebuilt set (the AUR's `onnxruntime-split-bin`, 1.29.0-7) plus the two separate lines, CUDA 12 and MIGraphX.

| | Arch `extra` | This repository |
|---|---|---|
| Version | 1.29.0 (pkgrel 3) | 1.29.0 (pkgrel 7; CUDA 12 line 2, MIGraphX line 1) |
| Engines | CPU; CUDA, MIGraphX and oneDNN depending on the build | CPU, CUDA, TensorRT, OpenVINO, oneDNN; MIGraphX in its own package |
| Combining engines | one build per machine: CUDA and MIGraphX never together | one package per engine, so CUDA, TensorRT, OpenVINO and oneDNN together |
| NVIDIA Turing (RTX 20) and newer | yes (CUDA 13, `75` to `121`) | yes (CUDA 13; presets `rtx`: `75 86 89 120` and `all`: `75 80 86 89 90 100 103 120`) |
| NVIDIA Pascal and Volta | no (CUDA 13 builds start at `sm_75`) | CUDA 12 line, `onnxruntime-provider-cuda12-legacy`; **untested on that hardware** ([cuda12-legacy.md](cuda12-legacy.md)) |
| NVIDIA Maxwell | no | no |
| TensorRT | not enabled | yes |
| Intel CPU, GPU and NPU (OpenVINO) | not enabled | yes ([hardware.md](hardware.md#intel-openvino-provider)) |
| oneDNN (CPU) | in the CUDA and ROCm builds, using Arch's shared `onednn` package | `onnxruntime-provider-dnnl` (optional, for Intel server CPUs): oneDNN 3.11.3 built here and linked statically; no dependency on Arch's `onednn` ([engines.md](engines.md#onednn)) |
| AMD GPUs | `onnxruntime-rocm`, `onnxruntime-opt-rocm` (MIGraphX) | `onnxruntime-provider-migraphx` and its Python add-on `python-onnxruntime-provider-migraphx`, AUR source build only; **untested on AMD hardware** ([amd.md](amd.md)) |
| Training operators | yes, in every build | yes; the Python module carries `onnxruntime.training` (needs `python-pytorch`, optional) |
| NCCL | `nccl` in the CUDA builds, `rccl` in the ROCm builds | NCCL on by default, so `onnxruntime-core` depends on `nccl`; `ORTF_NCCL=0` source builds drop it ([build.md](build.md#nccl-and-the-nccloff-packages)) |
| CPU baseline | generic x86-64, except the `opt` builds, compiled with `-march=haswell` (AVX2 CPUs only) | generic x86-64 for every package, for any x86-64 CPU |
| Python | one `python-` twin per build | one `python-onnxruntime-core` for every engine |
| Installed size | table above | see [packages.md](packages.md#sizes) (1.29.0-6 measured; 1.29.0-7 not yet) |
| Vendor dependency versions | bare names (`cuda`, `cudnn`, `migraphx`) | bounded: major for `cuda`, `cudnn`, `migraphx` and `hip-runtime-amd`, major.minor for TensorRT and OpenVINO ([dependencies.md](dependencies.md#the-version-bounds)) |
| How it is installed | `pacman -S` from `extra` | prebuilt: AUR `-bin` packages, whose files are GitHub release assets; or built from the AUR source packages; MIGraphX is source only |
| Maintenance | the Arch packagers, with the repository | one maintainer; a vendor upgrade past a bound is followed by a rebuild and a pkgrel bump |

This repository's hardware has been tested on one machine only: an NVIDIA GeForce RTX 5080 Laptop GPU and an Intel Core Ultra 9 275HX with its integrated GPU and NPU ([results.md](results.md)). The CUDA 12 and MIGraphX packages have not run on Pascal, Volta or AMD hardware. This page has no statement about the hardware Arch's packages were tested on.

## Which of this repository's packages for which hardware

Install the core and then the packages of the engines you have ([install.md](install.md)). The core needs no vendor runtime and runs on any x86-64 CPU.

| You have | Install | Notes |
|---|---|---|
| Any x86-64 CPU | `onnxruntime-core` | the CPU provider is in the core |
| Intel CPU (Core Ultra, Core) | the core alone, or add `onnxruntime-provider-openvino` | OpenVINO's `CPU` device uses oneDNN internally and also reaches the Intel GPU and NPU |
| Intel Xeon with AVX-512 or AMX | the core, and optionally `onnxruntime-provider-dnnl` | oneDNN is meant for int8 and bf16 models that it can take in one piece; measure against the CPU provider ([engines.md](engines.md#onednn)) |
| AMD CPU | `onnxruntime-core` | the CPU provider; the oneDNN provider is not meant for it |
| Intel GPU (Xe, Arc) or NPU | `onnxruntime-provider-openvino` and the device's plugin and driver | [engines.md](engines.md#openvino) |
| NVIDIA Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx`, and `onnxruntime-provider-tensorrt-cuda13` for TensorRT | `onnxruntime-provider-cuda13-all` for A100, H100 and B200-class GPUs |
| NVIDIA Pascal or Volta | `onnxruntime-provider-cuda12-legacy` | replaces the CUDA 13 setup on that machine; untested: [cuda12-legacy.md](cuda12-legacy.md) |
| AMD GPU in Arch's ROCm build list (Vega, RDNA 1 to 4, Instinct) | `onnxruntime-provider-migraphx` | AUR source build, with the Python add-on; untested: [amd.md](amd.md#which-amd-gpus) |
| Model training in Python | `python-onnxruntime-core`, and `python-pytorch` | the training modules need PyTorch even to import |
| NVIDIA Maxwell, AMD Polaris and Vega APUs, older Intel graphics, the Ryzen AI NPU | no package | [hardware.md](hardware.md) |

The engines can be combined: a laptop with an Intel CPU, an NVIDIA GPU and an NPU takes the core, the CUDA, TensorRT and OpenVINO packages, and, optionally, the oneDNN one (meant for Intel server CPUs; on a laptop it was 7 to 19 times slower than the CPU provider in the [measurements](results.md#release-1290-7)).

## Where Arch is the better choice

- **You want the official repository only**, with no AUR package. Arch's builds are in `extra`; this set is not.
- **A small CPU-only install.** `onnxruntime-cpu` is 25.68 MiB installed. This repository's `onnxruntime-core` is 29.4 MiB (1.29.0-6), but by default it depends on `nccl`, which is 364.18 MiB installed (`pacman -Si nccl`), even on a machine with no NVIDIA GPU. A source build with `ORTF_NCCL=0` omits it, but no prebuilt `-nccloff` package exists ([build.md](build.md#nccl-and-the-nccloff-packages)).
- **Vendor packages without version bounds**, if you would rather not have pacman hold back a CUDA or ROCm upgrade until a rebuild ([dependencies.md](dependencies.md#the-version-bounds)).
- **A build compiled for AVX2** (`onnxruntime-opt-cuda`, `onnxruntime-opt-rocm`) is Arch's. Here every package targets generic x86-64; ONNX Runtime's own CPU math library (MLAS) still picks its AVX2 or AVX-512 kernels at run time, as it does in Arch's generic builds.

Sources: `pacman -Ss onnxruntime` and `pacman -Si` for each package (2026-11-17); Arch's [PKGBUILD](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime) for the engines, the architectures and the training and NCCL options.

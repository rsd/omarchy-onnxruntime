# Choosing between this repository's packages and Arch's

Pick one ONNX Runtime package set per machine. Arch's `extra` repository and this repository both build 1.29.0, and the two sets cannot be installed together. This set has every engine and feature that Arch's has (the training operators, oneDNN, NCCL and the MIGraphX provider for AMD GPUs). It adds TensorRT, OpenVINO (Intel CPU, GPU and NPU), WebGPU (any GPU with a Vulkan driver) and a CUDA 12 line for NVIDIA Pascal and Volta.

The costs:

- It comes from the AUR, not from a repository.
- Two of its packages (MIGraphX and CUDA 12) are not tested on their hardware.
- Its default packages pull in NCCL on every machine ([below](#where-arch-is-the-better-choice)).

Terms such as pkgrel, `-bin` and SM are in the [glossary](glossary.md).

## What Arch ships

This command lists Arch's builds:

```bash
pacman -Ss onnxruntime
```

It shows five builds of 1.29.0 (pkgrel 3 on 2026-11-17), each with a `python-` twin: ten packages. Each build is a separate compilation of the whole library. Every one provides `onnxruntime` and conflicts with it, so you install one at a time. Sizes and dependencies below are from `pacman -Si`. The engines are from the build options in Arch's [PKGBUILD](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime).

| Package | Engines | CPU code | Depends on (besides `glibc`, `libgcc`, `libstdc++`) | Download / installed |
|---|---|---|---|---|
| `onnxruntime-cpu` | CPU | generic x86-64 | `cpuinfo`, `protobuf` | 6.92 / 25.68 MiB |
| `onnxruntime-cuda` | CPU, CUDA, oneDNN | generic x86-64 | `cpuinfo`, `protobuf`, `cuda`, `cudnn`, `cudnn-frontend`, `nccl`, `onednn` | 406.37 / 618.75 MiB |
| `onnxruntime-opt-cuda` | CPU, CUDA, oneDNN | built with `-march=haswell` (AVX2) | the same | 406.86 / 619.35 MiB |
| `onnxruntime-rocm` | CPU, MIGraphX, oneDNN | generic x86-64 | `cpuinfo`, `protobuf`, `rocm-hip-sdk`, `roctracer`, `rccl`, `migraphx` | 7.25 / 26.67 MiB |
| `onnxruntime-opt-rocm` | CPU, MIGraphX, oneDNN | built with `-march=haswell` | the same | 7.37 / 27.17 MiB |

The `python-onnxruntime-*` twins have the same dependencies plus `python-numpy`, `python-psutil`, `python-py-cpuinfo`, `python-sympy`, `python-scipy`, `python-pillow`, `python-flatbuffers`, `python-protobuf` and `python-packaging` (and no `cudnn-frontend`). Their installed sizes are 64.41 MiB (`cpu`), 657.69 MiB (`cuda`), 658.79 MiB (`opt-cuda`), 65.44 MiB (`rocm`) and 66.45 MiB (`opt-rocm`).

Arch's recipe builds the CUDA variants for the architectures `75` to `121`, a list that includes the Arm-only Jetson and Thor targets. Blackwell (`sm_120`) is in that list. Every variant is configured with the training operators. The ROCm variants are configured with ONNX Runtime's MIGraphX provider: ONNX Runtime 1.29 has no ROCm provider, and the HIP architecture list in that recipe has no effect ([amd.md](amd.md#what-exists-for-amd-gpus)). The recipe enables neither TensorRT nor OpenVINO.

## Why one set per machine

Arch's packages and this repository's `onnxruntime-core` both install ONNX Runtime's core library, `/usr/lib/libonnxruntime.so`, and its versioned names. Two packages cannot own one file, so the choice is exclusive. `onnxruntime-core` declares a conflict with every Arch variant by name (`onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm`, `onnxruntime-git`). `python-onnxruntime-core` does the same for the `python-` twins. When you install one set on a machine that has the other, pacman offers to remove the other.

Programs built against Arch's `onnxruntime` keep working. `onnxruntime-core` provides `onnxruntime=1.29.0` and `libonnxruntime.so`, and the version, and so the API, is the same. This repository's engines are not built on top of Arch's library, because that would tie every release here to Arch's pkgrel, flags and patches.

## Comparison

"Arch" is the five builds above. "This repository" is the default prebuilt set (the AUR's `onnxruntime-split-bin`, which includes the WebGPU package starting with 1.29.0-8) plus the two separate lines, CUDA 12 and MIGraphX.

| | Arch `extra` | This repository |
|---|---|---|
| Version | 1.29.0 (pkgrel 3) | 1.29.0 (pkgrel 8; CUDA 12 line 2, MIGraphX line 1) |
| Engines | CPU; CUDA, MIGraphX and oneDNN depending on the build | CPU, CUDA, TensorRT, OpenVINO, oneDNN, WebGPU; MIGraphX in its own package |
| Combining engines | one build per machine: CUDA and MIGraphX never together | one package per engine, so CUDA, TensorRT, OpenVINO, oneDNN and WebGPU together |
| NVIDIA Turing (RTX 20) and newer | yes (CUDA 13, `75` to `121`) | supported (CUDA 13; presets `rtx`: `75 86 89 120` and `all`: `75 80 86 89 90 100 103 120`); `sm_120` tested, the other architectures not tested ([hardware.md](hardware.md#nvidia)) |
| NVIDIA Pascal and Volta | no (CUDA 13 builds start at `sm_75`) | CUDA 12 line, `onnxruntime-provider-cuda12-legacy`; supported, **not tested** on that hardware ([cuda12-legacy.md](cuda12-legacy.md)) |
| NVIDIA Maxwell | no | CUDA: not supported, ONNX Runtime 1.29's fp16 kernels need `sm_53` or later ([cuda12-legacy.md](cuda12-legacy.md#who-needs-it)). WebGPU: supported, **not tested** ([below](#webgpu-or-a-vendor-provider)) |
| TensorRT | not enabled | yes |
| Intel CPU, GPU and NPU (OpenVINO) | not enabled | yes ([hardware.md](hardware.md#intel-openvino-provider)) |
| WebGPU (GPUs with a Vulkan driver) | not among its engines | `onnxruntime-provider-webgpu`: supported; **tested** on the RTX 5080 Laptop GPU and the Arrow Lake integrated GPU, **not tested** elsewhere. A plugin provider: the application loads it with the plugin API, `providers=["WebGpuExecutionProvider"]` does not reach it ([engines.md](engines.md#webgpu)) |
| oneDNN (CPU) | in the CUDA and ROCm builds, using Arch's shared `onednn` package | `onnxruntime-provider-dnnl` (optional, for Intel server CPUs): oneDNN 3.11.3 built here and linked statically; no dependency on Arch's `onednn` ([engines.md](engines.md#onednn)) |
| AMD GPUs | `onnxruntime-rocm`, `onnxruntime-opt-rocm` (MIGraphX) | `onnxruntime-provider-migraphx` and its Python add-on `python-onnxruntime-provider-migraphx`, AUR source build only; supported for Arch's ROCm build list, **not tested** on AMD hardware ([amd.md](amd.md)) |
| Training operators | yes, in every build | yes; the Python module carries `onnxruntime.training` (needs `python-pytorch`, optional) |
| NCCL | `nccl` in the CUDA builds, `rccl` in the ROCm builds | NCCL on by default, so `onnxruntime-core` depends on `nccl`; `ORTF_NCCL=0` source builds drop it ([build.md](build.md#nccl-and-the-nccloff-packages)) |
| CPU baseline | generic x86-64, except the `opt` builds, compiled with `-march=haswell` (AVX2 CPUs only) | generic x86-64 for every package, for any x86-64 CPU |
| Python | one `python-` twin per build | one `python-onnxruntime-core` for every engine |
| Installed size | table above | see [packages.md](packages.md#sizes) (1.29.0-6 measured; 1.29.0-7 and 1.29.0-8 not yet, except the WebGPU package) |
| Vendor dependency versions | bare names (`cuda`, `cudnn`, `migraphx`) | bounded: major for `cuda`, `cudnn`, `migraphx` and `hip-runtime-amd`, major.minor for TensorRT and OpenVINO ([dependencies.md](dependencies.md#the-version-bounds)) |
| How you install it | `pacman -S` from `extra` | prebuilt: AUR `-bin` packages, whose files are GitHub release assets; or built from the AUR source packages; MIGraphX is source only |
| Maintenance | the Arch packagers, with the repository | one maintainer; a vendor upgrade past a bound is followed by a rebuild and a pkgrel bump |

This repository's hardware is **tested** on one machine only: an NVIDIA GeForce RTX 5080 Laptop GPU and an Intel Core Ultra 9 275HX with its integrated GPU and NPU ([results.md](results.md)). WebGPU ran on that machine's two GPUs. Everything else the packages are built for is supported and **not tested**, including the CUDA 12 and MIGraphX packages on Pascal, Volta and AMD hardware ([hardware.md](hardware.md)). This page makes no statement about the hardware Arch's packages were tested on.

## Which of this repository's packages for which hardware

Install the core, then the packages of the engines you have ([INSTALL.md](../INSTALL.md)). The core needs no vendor runtime and runs on any x86-64 CPU.

| You have | Install | Notes |
|---|---|---|
| Any x86-64 CPU | `onnxruntime-core` | the CPU provider is in the core |
| Intel CPU (Core Ultra, Core) | the core alone, or add `onnxruntime-provider-openvino` | OpenVINO's `CPU` device uses oneDNN internally and also reaches the Intel GPU and NPU |
| Intel Xeon with AVX-512 or AMX | the core, and optionally `onnxruntime-provider-dnnl` | oneDNN is meant for int8 and bf16 models that it can take in one piece; measure against the CPU provider ([engines.md](engines.md#onednn)) |
| AMD CPU | `onnxruntime-core` | the CPU provider; the oneDNN provider is not meant for it |
| Intel GPU (Xe, Arc) or NPU | `onnxruntime-provider-openvino` and the device's plugin and driver | [engines.md](engines.md#openvino) |
| Intel Gen8, Gen9 or Gen11 integrated GPU | `onnxruntime-provider-openvino`, with the AUR's `intel-compute-runtime-legacy` in place of `intel-compute-runtime` | supported with that legacy driver, which you install yourself; not tested: [hardware.md](hardware.md#intel-openvino-provider). A second route is `onnxruntime-provider-webgpu` with `vulkan-intel` ([below](#webgpu-or-a-vendor-provider)) |
| NVIDIA Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx`, and `onnxruntime-provider-tensorrt-cuda13` for TensorRT | `onnxruntime-provider-cuda13-all` for A100, H100 and B200-class GPUs |
| NVIDIA Pascal or Volta | `onnxruntime-provider-cuda12-legacy` | replaces the CUDA 13 setup on that machine; supported, not tested: [cuda12-legacy.md](cuda12-legacy.md) |
| AMD GPU in Arch's ROCm build list (Vega, RDNA 1 to 4, Instinct) | `onnxruntime-provider-migraphx` | AUR source build, with the Python add-on; supported, not tested: [amd.md](amd.md#which-amd-gpus) |
| Model training in Python | `python-onnxruntime-core`, and `python-pytorch` | the training modules need PyTorch even to import |
| NVIDIA Maxwell, AMD Polaris, AMD Vega APUs | `onnxruntime-provider-webgpu` and the Vulkan driver of the GPU (`nvidia-utils`, `vulkan-radeon`); not a vendor provider | supported, not tested; applications must load it with the plugin API: [below](#webgpu-or-a-vendor-provider) |
| The Ryzen AI NPU | no package: not supported | the reasons are in [hardware.md](hardware.md) |

Engines can be combined. A laptop with an Intel CPU, an NVIDIA GPU and an NPU takes the core, the CUDA, TensorRT and OpenVINO packages, and, optionally, the oneDNN one. The oneDNN provider is meant for Intel server CPUs. On a laptop it was 7 to 19 times slower than the CPU provider in the [measurements](results.md#release-1290-7).

## WebGPU or a vendor provider

Pick `onnxruntime-provider-webgpu` only when no vendor provider covers your GPU. Where CUDA, TensorRT, OpenVINO or MIGraphX applies, use that one: it is the provider its vendor supports, and applications select it in the usual way.

The GPUs WebGPU is for:

- NVIDIA Maxwell, which the CUDA provider cannot be built for.
- Intel graphics from Gen8 on, through Mesa's `vulkan-intel`, as a second route beside OpenVINO's `GPU` device (Gen8, Gen9 and Gen11 need a legacy driver you install yourself there).
- AMD Polaris and Vega APUs, which are not in Arch's ROCm build list, and AMD discrete cards through `vulkan-radeon`.

It needs `vulkan-icd-loader` and one Vulkan driver for the GPU. It is not shown to be faster than the CPU. On the test machine's CPU and three small models the Intel GPU's median was 2.1 to 5.8 times the CPU's, and the RTX was faster on two of the three ([results.md](results.md#webgpu-provider-1290-8)). A speed-up on older machines is not tested. An application that selects providers by name with `providers=[...]` cannot use it without a code change, and the MODNet model does not run on it ([engines.md](engines.md#webgpu)).

## Where Arch is the better choice

- **You want the official repository only**, with no AUR package. Arch's builds are in `extra`. This set is not.
- **You want a small CPU-only install.** `onnxruntime-cpu` is 25.68 MiB installed. This repository's `onnxruntime-core` is 29.4 MiB (1.29.0-6), but by default it depends on `nccl`, which is 364.18 MiB installed, even on a machine with no NVIDIA GPU. A source build with `ORTF_NCCL=0` omits it, but no prebuilt `-nccloff` package exists ([build.md](build.md#nccl-and-the-nccloff-packages)).

  This command reads the size of `nccl`:

  ```bash
  pacman -Si nccl
  ```

- **You want vendor packages without version bounds**, so that pacman never holds back a CUDA or ROCm upgrade until a rebuild ([dependencies.md](dependencies.md#the-version-bounds)).
- **You want a build compiled for AVX2** (`onnxruntime-opt-cuda`, `onnxruntime-opt-rocm`). Those are Arch's. Here every package targets generic x86-64. ONNX Runtime's own CPU math library (MLAS) still picks its AVX2 or AVX-512 kernels at run time, as it does in Arch's generic builds.

Sources: `pacman -Ss onnxruntime` and `pacman -Si` for each package (2026-11-17). Arch's [PKGBUILD](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime) for the engines, the architectures and the training and NCCL options.

Ran it on hardware marked not tested? Report the result, working or not, in an [issue](https://github.com/rsd/omarchy-onnxruntime/issues). A report moves the flag.

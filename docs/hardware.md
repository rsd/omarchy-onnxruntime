# Hardware coverage and what is not supported

This page says which hardware each engine covers, what is left out, and why. Use it to choose the packages for your machine. Terms are in the [glossary](glossary.md).

Every claim on this page uses the same words:

- **supported**: the recipe builds for it. A supported entry is flagged **tested** or **not tested**.
  - **tested**: it ran on this project's test machine, and the result is in [results.md](results.md).
  - **not tested**: it is built for, but it has not run on that hardware.
- **planned**: intended, not built yet.
- **not supported**: not built or not buildable. The reason is always given.

"Source only" is a packaging note, not a status: it means there is no prebuilt (`-bin`) package, only an AUR source package. It does not say whether the hardware is supported.

The test machine is a GeForce RTX 5080 Laptop GPU (`sm_120`) and an Intel Core Ultra 9 275HX with its CPU, integrated GPU (Arrow Lake) and NPU. Tested is per engine and per device: an engine that ran on one device of a kind is not tested on the others.

A claim covers what the build compiles for, not what a vendor's support matrix lists. The sources are the CUDA architecture presets ([build.md](build.md#cuda-architectures)), Arch's ROCm build list ([amd.md](amd.md#which-amd-gpus)), the engine list of the recipe (`_ortf_default_eps` in `onnxruntime-split/PKGBUILD`) and, for WebGPU, the Vulkan backend of Dawn that the package carries. Hardware lists reflect the vendors' documentation in November 2026 and change with each release. Check the linked sources for your card or CPU.

| Hardware | Status | Source of the claim |
|---|---|---|
| Any x86-64 CPU (CPU provider in `onnxruntime-core`; oneDNN provider `onnxruntime-provider-dnnl`) | supported; tested on an Intel Core Ultra 9 275HX, not tested on other CPUs | generic x86-64 build ([choosing.md](choosing.md#comparison)) |
| NVIDIA Blackwell, RTX 50 (`sm_120`), with CUDA 13 and TensorRT | supported; tested with `onnxruntime-provider-cuda13-rtx` on the RTX 5080 Laptop GPU; `onnxruntime-provider-cuda13-all` not tested | preset `rtx` (`75 86 89 120`) and `all` ([presets](build.md#cuda-architectures)) |
| NVIDIA Turing (RTX 20, GTX 16, `sm_75`), Ampere GeForce (`sm_86`) and Ada (`sm_89`), with CUDA 13 | supported; not tested | the same presets |
| NVIDIA Ampere data centre (A100, `sm_80`), Hopper (`sm_90`) and Blackwell data centre (`sm_100`, `sm_103`), with CUDA 13 | supported in `onnxruntime-provider-cuda13-all` only, not in the default `rtx` package; not tested | preset `all` (`75 80 86 89 90 100 103 120`) |
| NVIDIA Pascal (`sm_60`, `sm_61`) and Volta (`sm_70`) | supported (CUDA 12 legacy package, `onnxruntime-provider-cuda12-legacy`); not tested | preset `legacy` (`60 61 70 75 80 86 89 120`, PTX of `60`) |
| NVIDIA Maxwell, CUDA provider | not supported: ONNX Runtime 1.29's fp16 kernels need `sm_53` or later, so they cannot be built for it | [cuda12-legacy.md](cuda12-legacy.md#who-needs-it) |
| NVIDIA Maxwell, WebGPU provider (`onnxruntime-provider-webgpu`) | supported, as a second route for these cards, with the NVIDIA Vulkan driver the card still has (the 580xx series); not tested | [WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus) |
| NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell) and Intel Arrow Lake integrated GPU, WebGPU provider | supported; **tested** on both, with NVIDIA's driver 615.78.08 and Mesa's `vulkan-intel` 26.2.5 | [WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus), [results.md](results.md#webgpu-provider-1290-8) |
| Intel CPUs, Xe and Arc GPUs, NPUs (OpenVINO provider) | supported; tested on the Core Ultra 9 275HX CPU, integrated GPU and NPU, not tested on other Intel hardware ([the devices](#intel-openvino-provider)) | OpenVINO has no per-device build |
| Intel Gen8, Gen9 and Gen11 integrated GPUs | supported with a legacy driver you install yourself (the AUR's `intel-compute-runtime-legacy`, in place of `intel-compute-runtime`), because Intel's current compute runtime dropped them; not tested | [below](#intel-openvino-provider) |
| Intel Gen8 and later integrated GPUs and Arc, WebGPU provider, through Mesa's `vulkan-intel` | supported, as a second route (it needs no compute runtime); tested on the Arrow Lake integrated GPU of the 275HX, not tested on other Intel graphics | [WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus) |
| AMD GPUs in Arch's ROCm build list: Vega 10 and 20 (gfx900, gfx906), RDNA 1 to 4 (gfx1010, gfx1012, gfx1030, gfx1031, gfx1035, gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1153, gfx1200, gfx1201) and Instinct MI100 to MI355X | supported; not tested on any AMD GPU. Source only: `onnxruntime-provider-migraphx` and its Python add-on, AUR pkgbase `onnxruntime-split-migraphx`. AMD itself does not list Vega and RDNA 1 | Arch's build list ([the list](amd.md#which-amd-gpus)) |
| AMD Polaris (gfx803), Vega APUs (gfx90c), RX 6600 and 6500 series (gfx1032, gfx1034), MIGraphX provider | not supported: not in Arch's ROCm build list | [the list](amd.md#which-amd-gpus) |
| AMD Polaris and Vega APUs, and AMD discrete cards, WebGPU provider, through Mesa's `vulkan-radeon` | supported, as a second route for the Polaris and Vega APUs the MIGraphX provider leaves out; not tested | [WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus) |
| Any other GPU with a Vulkan driver that Dawn accepts (WebGPU provider) | supported; not tested | [WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus) |
| AMD Ryzen AI NPU (Vitis AI provider) | not supported: the working provider is a proprietary AMD library that cannot be redistributed | [below](#amd-ryzen-ai-npu-vitis-ai-provider-not-supported) |

## NVIDIA

The CUDA provider is built with CUDA 13, which Arch's `cuda` ships. CUDA 13 dropped GPUs older than Turing, so the architecture presets start at `sm_75`:

- `rtx` (`75 86 89 120`, the default): `onnxruntime-provider-cuda13-rtx`.
- `all` (`75 80 86 89 90 100 103 120`): `onnxruntime-provider-cuda13-all`. Only this one has code for `sm_80`, `sm_90`, `sm_100` and `sm_103`.
- A list of your own, such as `120` for the RTX 50 series alone ([presets and sizes](build.md#cuda-architectures)).

A GPU outside the list of its package has no machine code in it and is not supported by that package. Of these architectures only `sm_120` is **tested**: it ran with CUDA (the `rtx` package) and with TensorRT on the RTX 5080 Laptop GPU ([results.md](results.md)). The `all` package was checked for its build, files and dependencies only, so it is **not tested**. The others are **not tested**. TensorRT's own requirements apply to the TensorRT provider as well.

The CUDA 12 legacy package serves cards older than Turing, described next.

## NVIDIA Pascal and Volta (CUDA 12 legacy; supported, not tested)

`onnxruntime-provider-cuda12-legacy` is the CUDA provider built with CUDA 12.9 for Pascal and Volta. CUDA 13, cuDNN 9.11 and the 590 driver series dropped those cards. The package needs the CUDA 12 userland and the 580xx driver from the AUR, and it replaces the CUDA 13 setup on that machine.

The package is supported for `sm_60`, `sm_61` and `sm_70`, and it is **not tested** on Pascal or Volta hardware. Its `legacy` preset also holds code for `sm_75`, `sm_80`, `sm_86`, `sm_89` and `sm_120` (not for `sm_90`, `sm_100` or `sm_103`). Of those only `sm_120` is **tested**, on the RTX 5080 Laptop GPU with the CUDA 12 libraries ([results.md](results.md#cuda-12-legacy-provider)).

The CUDA provider for Maxwell is **not supported**: its kernels do not build with ONNX Runtime 1.29. The WebGPU provider is a second route for those cards, **supported** and **not tested** ([WebGPU](#webgpu-dawn-over-vulkan-supported-tested-on-two-gpus)).

Everything about this package is in [cuda12-legacy.md](cuda12-legacy.md). Sources: [CUDA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus), [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Intel (OpenVINO provider)

The OpenVINO provider has no per-device build. One package serves every Intel device, and OpenVINO's device plugins decide what runs where. OpenVINO and the driver for each device set what a device can run. The table is from OpenVINO 2026.4's [system requirements](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/docs/articles_en/about-openvino/release-notes-openvino/system-requirements.rst):

| Device (`device_type`) | Hardware in OpenVINO's list | Status here | Packages for the device |
|---|---|---|---|
| `CPU` | Core Ultra series 1, 2 and 3; 6th to 14th generation Core; Xeon 6; 1st to 5th generation Xeon Scalable; Atom; Pentium N series | supported; tested on the Core Ultra 9 275HX CPU, not tested on the others | `openvino` |
| `GPU` | Integrated graphics from Tiger Lake (Iris Xe) through Rocket Lake, Alder Lake, Raptor Lake, Meteor Lake, Arrow Lake, Lunar Lake and Panther Lake; Arc Alchemist and Battlemage; Data Center GPU Flex and Max | supported; tested on the Arrow Lake integrated graphics of the 275HX, not tested on the others | `openvino-intel-gpu-plugin`, `intel-compute-runtime`, `level-zero-loader` |
| `NPU` | Core Ultra with an NPU: Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake and Wildcat Lake | supported; tested on the Arrow Lake NPU of the 275HX, not tested on the others | `openvino-intel-npu-plugin`, `intel-npu-driver`, `level-zero-loader`, and the kernel's `intel_vpu` driver |

The OpenVINO provider has no per-device build, so what it supports is what OpenVINO and the driver of each device support. The middle column is OpenVINO's list. The NPU generations are those of Intel's [Linux NPU driver](https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md). The GPU generations are those of Intel's [compute runtime](https://github.com/intel/compute-runtime). OpenVINO lists Ubuntu and RHEL as tested systems, not Arch. The Arch packages are what this recipe was checked with ([results](results.md)).

**Older integrated GPUs (Gen8, Gen9, Gen11).** Intel's current compute runtime, which is Arch's `intel-compute-runtime`, no longer supports Broadwell, Skylake to Coffee Lake, Ice Lake and the Apollo, Gemini and Elkhart Lake parts. Intel moved them to a separate legacy branch, which Arch does not package. The AUR has `intel-compute-runtime-legacy` for it. Install it yourself in place of `intel-compute-runtime`. With that legacy driver those GPUs are supported, and **not tested**: this recipe has not run them, so whether OpenVINO 2026 still runs on them is not known. The CPU device is supported on those machines whatever the GPU.

## WebGPU (Dawn over Vulkan; supported, tested on two GPUs)

`onnxruntime-provider-webgpu` runs a model on a GPU through Dawn, a WebGPU implementation, using Dawn's Vulkan backend. It covers a GPU by its Vulkan driver, not by an architecture list: the claim is any GPU with a Vulkan driver that Dawn accepts. It is built for no particular GPU, so unlike the CUDA presets it has no per-architecture build list to check a card against. The driver packages are optional dependencies, one per vendor. They are not installed by the provider, so install the one for your GPU ([engines.md](engines.md#webgpu)):

| GPU | Vulkan driver package | Status |
|---|---|---|
| NVIDIA GeForce RTX 5080 Laptop GPU (`10de:2c19`) | `nvidia-utils` (driver 615.78.08) | supported, **tested** |
| Intel Arrow Lake integrated GPU (`8086:7d67`, Core Ultra 9 275HX) | `vulkan-intel` (Mesa's ANV, 26.2.5) | supported, **tested** |
| Other NVIDIA GPUs, Maxwell and later | `nvidia-utils`; for Maxwell the 580xx series, which is the last that supports it ([cuda12-legacy.md](cuda12-legacy.md#what-the-machine-needs)) | supported, **not tested** |
| Other Intel graphics, Gen8 and later | `vulkan-intel` | supported, **not tested** |
| AMD graphics, Polaris and Vega APUs and discrete cards included | `vulkan-radeon` (Mesa's RADV) | supported, **not tested** |

The use of the provider:

- **It is a second route, not a replacement.** Where a vendor provider covers the GPU, that provider is the one to use. WebGPU is the route for NVIDIA Maxwell (no CUDA provider), for Intel graphics through Mesa without Intel's compute runtime, and for the AMD Polaris and Vega APUs that Arch's ROCm build list leaves out.
- **It is not shown to be faster than the CPU.** On the test machine, with a fast CPU and small models, the Intel GPU was slower than the CPU, and the RTX was faster on two of three models ([results.md](results.md#webgpu-provider-1290-8)). A speed-up on older machines (NVIDIA Maxwell, Intel Gen8 and later graphics through Mesa, AMD Polaris and Vega APUs) is **not tested**. The value is a working GPU path where no vendor provider exists.
- **One known limitation.** The MODNet model runs on it only with CPU fallback allowed, because one int64 `Gather` has no WebGPU kernel in ONNX Runtime 1.29.0. Its output is correct on both tested GPUs, because this recipe fixes a WebGPU convolution bug of 1.29.0 from 1.29.0-8 on, pending upstream ([engines.md](engines.md#webgpu-limitations)).
- **A different way to load it.** An application has to register the provider and pick a device, and `providers=["WebGpuExecutionProvider"]` does not reach it ([engines.md](engines.md#using-it-a-plugin-provider)).

## oneDNN (CPU; supported, tested)

ONNX Runtime has a oneDNN execution provider, which runs on the CPU. Its documentation lists a CPU backend only. A GPU path needs a separate OpenCL build, so oneDNN on a GPU is not supported: this recipe does not build it. `onnxruntime-provider-dnnl` carries the provider, with oneDNN built into the package, and it needs no vendor install ([engines.md](engines.md#onednn)). It is supported on any x86-64 CPU. It is **tested** on the Core Ultra 9 275HX only.

It is optional and meant for:

- Intel server CPUs (Xeon with AVX-512 or AMX).
- int8 or bf16 models.
- Models it can take in one piece.

On desktops and laptops use the CPU provider, or on Intel, OpenVINO. OpenVINO uses oneDNN internally and also reaches the integrated GPU and the NPU. On AMD CPUs use the CPU provider. On a Core Ultra 9 275HX the oneDNN provider was 7 to 19 times slower than the CPU provider on the four models measured ([results.md](results.md#the-onednn-provider-against-the-cpu)). Source: [oneDNN execution provider](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html).

## AMD GPUs (MIGraphX; supported, not tested)

ONNX Runtime 1.29 has no ROCm provider. The only AMD GPU provider is MIGraphX. This repository ships it as `onnxruntime-provider-migraphx`, with its Python add-on `python-onnxruntime-provider-migraphx`. It comes from its own AUR pkgbase as a source build, with no prebuilt package (a packaging note, not a status). The GPUs below are **supported** and **not tested**: no AMD GPU has run it.

[amd.md](amd.md#which-amd-gpus) lists the GPUs it covers (Arch's ROCm build list, which includes Vega), the ones it does not (Polaris, the Vega APUs, gfx1032 and gfx1034) and how to build it.


## AMD Ryzen AI NPU (Vitis AI provider; not supported)

ONNX Runtime's Vitis AI provider is **not supported**: this recipe does not build it, for the reasons below. The in-tree provider is a shim. The working implementation is a proprietary AMD library loaded at run time. ONNX Runtime documents the Linux build of the provider only for AMD's embedded Adaptable SoCs ([build documentation](https://onnxruntime.ai/docs/build/eps.html), [provider documentation](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)).

Using a Ryzen AI NPU on Linux needs AMD's own [Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/linux.html). It is a download that requires accepting AMD's licence, delivered as a tarball for Ubuntu 24.04, with its own ONNX Runtime build, for Strix Point and newer. It cannot be redistributed, and this repository does not redistribute vendor files. Supporting it would take a separate recipe repository that builds from your own download, as [omarchy-nvidia-sdks](https://github.com/rsd/omarchy-nvidia-sdks) does for NVIDIA's SDKs.

On the Arch side, the kernel driver `amdxdna` is in the mainline kernel, and `xrt` and `xrt-plugin-amdxdna` are in Arch's `extra` repository.

Ran it on hardware marked not tested? Report the result, working or not, in an [issue](https://github.com/rsd/omarchy-onnxruntime/issues). A report moves the flag.

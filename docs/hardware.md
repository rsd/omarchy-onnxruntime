# Hardware coverage and what is not packaged

Which hardware each engine covers, what is deliberately left out, and why. Use this page to choose the packages for your machine. Terms are in the [glossary](glossary.md).

Status words are used strictly:

- **packaged**: the recipe builds it.
- **planned**: intended, not built yet.
- **untested**: the recipe builds it, but it has never run on that hardware.
- **not packaged**: there is no plan.

Hardware lists reflect the vendors' documentation in November 2026 and change with each release, so check the linked sources for your card or CPU.

| Hardware | Status |
|---|---|
| NVIDIA Turing (RTX 20, GTX 16) and newer | packaged (CUDA 13, [presets](build.md#cuda-architectures)) |
| NVIDIA Pascal and Volta | packaged (CUDA 12 legacy package, `onnxruntime-provider-cuda12-legacy`), untested on Pascal and Volta hardware |
| NVIDIA Maxwell | not supported: ONNX Runtime 1.29's fp16 kernels need `sm_53` or later; a WebGPU provider is planned for these cards |
| Intel CPUs, Xe and Arc GPUs, NPUs | packaged (OpenVINO provider) |
| Intel Gen8, Gen9 and Gen11 integrated GPUs | possible with a legacy driver you install yourself; unverified |
| Intel oneDNN provider | not packaged |
| AMD GPUs supported by ROCm (gfx1030 and newer, Instinct) | packaged on request (`ORTF_EPS=migraphx`, MIGraphX provider), untested on AMD hardware |
| AMD Vega and Polaris GPUs | not supported |
| AMD Ryzen AI NPU (Vitis AI provider) | not packaged |

## NVIDIA

The CUDA provider is built with CUDA 13, which is what Arch's `cuda` ships. CUDA 13 dropped GPUs older than Turing, so the architecture presets start at `sm_75`: `rtx` (`75 86 89 120`, the default) and `all` (`75 80 86 89 90 100 103 120`), or a list of your own, such as `120` for the RTX 50 series alone ([presets and sizes](build.md#cuda-architectures)).

Cards older than Turing are served by the CUDA 12 legacy package, described next.

## NVIDIA Pascal and Volta (CUDA 12 legacy; packaged, untested)

`onnxruntime-provider-cuda12-legacy` is the CUDA provider built with CUDA 12.9 for Pascal and Volta, which CUDA 13, cuDNN 9.11 and the 590 driver series dropped. It needs the CUDA 12 userland and the 580xx driver from the AUR, and it replaces the CUDA 13 setup on that machine. Maxwell is not offered: its kernels do not build with ONNX Runtime 1.29, and a WebGPU provider is planned for those cards. Everything about this package is in [cuda12-legacy.md](cuda12-legacy.md). Sources: [CUDA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus), [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Intel (OpenVINO provider)

The OpenVINO provider has no per-device build: the one package serves every Intel device, and OpenVINO's device plugins decide what runs where. What a device can run is set by OpenVINO and by the driver for that device. From OpenVINO 2026.4's [system requirements](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/docs/articles_en/about-openvino/release-notes-openvino/system-requirements.rst):

| Device (`device_type`) | Supported hardware | Packages for the device |
|---|---|---|
| `CPU` | Core Ultra series 1, 2 and 3; 6th to 14th generation Core; Xeon 6; 1st to 5th generation Xeon Scalable; Atom; Pentium N series | `openvino` |
| `GPU` | Integrated graphics from Tiger Lake (Iris Xe) through Rocket Lake, Alder Lake, Raptor Lake, Meteor Lake, Arrow Lake, Lunar Lake and Panther Lake; Arc Alchemist and Battlemage; Data Center GPU Flex and Max | `openvino-intel-gpu-plugin`, `intel-compute-runtime`, `level-zero-loader` |
| `NPU` | Core Ultra with an NPU: Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake and Wildcat Lake | `openvino-intel-npu-plugin`, `intel-npu-driver`, `level-zero-loader`, and the kernel's `intel_vpu` driver |

The NPU generations are those of Intel's [Linux NPU driver](https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md); the GPU generations are those of Intel's [compute runtime](https://github.com/intel/compute-runtime). OpenVINO lists Ubuntu and RHEL as tested systems, not Arch; the Arch packages are what this recipe was checked with ([results](results.md)).

**Older integrated GPUs (Gen8, Gen9, Gen11).** Intel's current compute runtime, which is Arch's `intel-compute-runtime`, no longer supports Broadwell, Skylake to Coffee Lake, Ice Lake and the Apollo, Gemini and Elkhart Lake parts. Intel moved them to a separate legacy branch, which Arch does not package. The AUR has `intel-compute-runtime-legacy` for it; install it yourself in place of `intel-compute-runtime`. Whether OpenVINO 2026 still runs on those GPUs with that driver is **unverified**, and this recipe does not test it. The CPU device works on those machines whatever the GPU.

**oneDNN provider: not packaged.** ONNX Runtime also has a oneDNN execution provider. Its documentation lists a CPU backend only (a GPU path needs a separate OpenCL build), and it adds little over ONNX Runtime's own CPU provider and OpenVINO's CPU device, so the recipe leaves it out. Source: [oneDNN execution provider](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html).

## AMD GPUs (MIGraphX; packaged, untested)

ONNX Runtime 1.29 has no ROCm provider. The only AMD GPU provider is MIGraphX, which this recipe builds on request. It is **untested on AMD hardware**. Supported GPUs, the old GPUs that are not supported, and the build are in [amd.md](amd.md).

## AMD Ryzen AI NPU (Vitis AI provider; not packaged)

ONNX Runtime's Vitis AI provider is not packaged. The in-tree provider is a shim: the working implementation is a proprietary AMD library loaded at run time, and ONNX Runtime documents the Linux build of the provider only for AMD's embedded Adaptable SoCs ([build documentation](https://onnxruntime.ai/docs/build/eps.html), [provider documentation](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)).

Using a Ryzen AI NPU on Linux needs AMD's own [Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/linux.html): a download that requires accepting AMD's licence, delivered as a tarball for Ubuntu 24.04, with its own ONNX Runtime build, for Strix Point and newer. It cannot be redistributed, and this repository does not redistribute vendor files. Supporting it would take a separate recipe repository that builds from the user's own download, as [omarchy-nvidia-sdks](https://github.com/rsd/omarchy-nvidia-sdks) does for NVIDIA's SDKs.

On the Arch side, the kernel driver `amdxdna` is in the mainline kernel, and `xrt` and `xrt-plugin-amdxdna` are in Arch's `extra` repository.

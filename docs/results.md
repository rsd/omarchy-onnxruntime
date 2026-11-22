# Measured results

This page reports what the recipe's checks measured for three builds, and the conditions of each run. The first build is the single-package 1.29.0-1 (with the CUDA 12 legacy provider 1.29.0-1). The second is the [1.29.0-7 packages](#release-1290-7), which add ONNX Runtime's training support, the oneDNN provider and the MIGraphX packages. The third is the [WebGPU provider of 1.29.0-8](#webgpu-provider-1290-8), run on the RTX 5080 Laptop GPU and on the Intel integrated GPU.

The figures are for the raw model only, without any pre- or post-processing. They are indicative: the desktop was in use during the runs. The [smoke test](../tools/ort-smoke/README.md) and the [model check](../tools/ort-model-check/README.md) produced them. Terms such as TF32 and fp16 are in the [glossary](glossary.md).

## What is tested

The flags **tested** and **not tested** in [hardware.md](hardware.md) come from this page. Everything that ran on the test machine (below) is tested. Everything else the packages are built for is supported and not tested.

| Engine | Tested on | Not tested |
|---|---|---|
| CPU provider, oneDNN | Intel Core Ultra 9 275HX (the oneDNN provider needs the CPU fallback on the four models, [below](#the-onednn-provider-against-the-cpu)) | other CPUs |
| CUDA 13 (preset `rtx`, and a `120` build) | GeForce RTX 5080 Laptop GPU (`sm_120`) | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_103` |
| TensorRT | GeForce RTX 5080 Laptop GPU (`sm_120`) | other NVIDIA GPUs |
| OpenVINO `CPU`, `GPU` and `NPU` devices | the CPU, integrated GPU and NPU of the Core Ultra 9 275HX | other Intel CPUs, GPUs (Arc included) and NPUs |
| CUDA 12 legacy | `sm_120` of the RTX 5080 Laptop GPU, with the CUDA 12 libraries | `sm_60`, `sm_61`, `sm_70` and `sm_75` to `sm_89` on real cards |
| WebGPU (`onnxruntime-provider-webgpu`) | the GeForce RTX 5080 Laptop GPU (NVIDIA's Vulkan driver) and the Arrow Lake integrated GPU of the Core Ultra 9 275HX (Mesa's `vulkan-intel`), [below](#webgpu-provider-1290-8) | NVIDIA Maxwell and other NVIDIA GPUs, other Intel graphics, every AMD GPU (Polaris and Vega APUs included) |
| MIGraphX | none (a load check only, on a machine with no AMD GPU) | every AMD GPU |

## Conditions

| | |
|---|---|
| ONNX Runtime | 1.29.0, built by this recipe with `ORTF_EPS` at its default and `ORTF_CUDA_ARCHS=120` ([build.md](build.md#parallelism-memory-and-disk) has the build and the vendor runtime versions). It is the build of the single package `onnxruntime-full` 1.29.0-1; since 1.29.0-2 the same libraries are split into `onnxruntime-core` and one package per engine ([build.md](build.md#package-names)), and the split packages were not measured separately in this first run; the [1.29.0-7 packages](#release-1290-7) were |
| NVIDIA GPU | GeForce RTX 5080 Laptop GPU |
| CPU | Intel Core Ultra 9 275HX, 24 threads, 30 GiB of RAM |
| Intel GPU and NPU | the integrated graphics and the NPU of that CPU |
| Runtimes | CUDA 13.4.3, cuDNN 9.27.0.42, TensorRT 11.4.0.106, OpenVINO 2026.4.1 (intel-npu-driver 1.38.0) |
| Where it ran | the build tree, before installing ([build-from-source.md](build-from-source.md#before-installing-the-build-tree)) |
| Model check | `tools/ort-model-check`: 1 first run, 3 warm-up runs and 50 timed runs per target, CPU EP fallback disabled, uniform-random input in [0, 1), `--precision default` unless stated; the [README](../tools/ort-model-check/README.md) explains the columns and the error bounds |
| TensorRT | engine cache off, fp16 off |

## C smoke test

`tools/ort-smoke` (a tiny synthetic model) reported `ok` with an exact result (max error 0) on all six targets: `cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU` and `openvino:NPU` ([tools/ort-smoke/README.md](../tools/ort-smoke/README.md)).

## Model check

Four segmentation models ran on all six targets, each `ok`. `create` is the session creation, `first` the first run, `median` and `p95` the timed runs, all in milliseconds. The errors are the largest (`max`) and mean (`mean`) absolute difference from the CPU outputs. The CPU is the reference, so its errors are 0.

### selfie_segmentation_landscape, input 1x144x256x3

| Target | create ms | first ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 14.8 | 14.94 | 1.83 | 3.50 | 0 | 0 |
| cuda | 768.8 | 629.40 | 0.54 | 0.57 | 6.0e-8 | 9.7e-12 |
| tensorrt | 9612.5 | 4.31 | 0.23 | 0.28 | 6.1e-8 | 1.0e-11 |
| openvino:CPU | 259.5 | 4.86 | 1.64 | 3.77 | 6.1e-8 | 9.8e-12 |
| openvino:GPU | 1547.8 | 2.29 | 1.01 | 1.68 | 7.2e-7 | 4.2e-11 |
| openvino:NPU | 564.5 | 73.24 | 1.65 | 2.18 | 7.2e-7 | 4.2e-11 |

### pp_humansegv2_mobile_192x192, input 1x3x192x192

| Target | create ms | first ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 53.9 | 8.46 | 3.56 | 9.44 | 0 | 0 |
| cuda | 309.0 | 281.35 | 1.27 | 3.27 | 3.6e-5 | 4.5e-6 |
| tensorrt | 9531.8 | 4.22 | 0.74 | 0.76 | 2.7e-5 | 3.5e-6 |
| openvino:CPU | 145.0 | 8.24 | 3.51 | 10.93 | 1.2e-7 | 1.7e-9 |
| openvino:GPU | 2140.5 | 13.88 | 2.04 | 13.39 | 5.9e-4 | 1.3e-4 |
| openvino:NPU | 512.4 | 9.21 | 2.46 | 3.31 | 1.5e-3 | 1.3e-4 |

### pp_humansegv2_lite_192x192, input 1x3x192x192

| Target | create ms | first ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 32.2 | 6.92 | 4.11 | 7.19 | 0 | 0 |
| cuda | 266.4 | 398.64 | 0.76 | 0.80 | 8.0e-4 | 6.1e-5 |
| tensorrt | 12839.6 | 4.07 | 0.42 | 0.42 | 4.7e-4 | 6.1e-5 |
| openvino:CPU | 168.7 | 4.65 | 3.58 | 4.24 | 4.8e-7 | 6.6e-8 |
| openvino:GPU | 2869.7 | 2.49 | 1.78 | 1.86 | 2.0e-3 | 2.5e-4 |
| openvino:NPU | 561.9 | 9.75 | 2.69 | 3.83 | 2.2e-3 | 2.6e-4 |

### modnet_photographic, input 1x3x512x512

The model's height and width are dynamic and were set to 512 with `--size 512x512`, so TensorRT builds its engine and OpenVINO compiles the model at the first run, not at session creation: the cost is in the `first ms` column.

| Target | create ms | first ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 38.4 | 71.69 | 56.21 | 71.39 | 0 | 0 |
| cuda | 321.5 | 393.87 | 4.96 | 5.15 | 6.3e-2 | 8.7e-4 |
| tensorrt | 226.5 | 9402.83 | 3.42 | 3.48 | 9.0e-2 | 1.2e-3 |
| openvino:CPU | 172.1 | 170.88 | 50.99 | 62.98 | 5.3e-5 | 7.2e-7 |
| openvino:GPU | 3055.6 | 7111.74 | 35.51 | 37.22 | 3.4e-1 | 5.4e-3 |
| openvino:NPU | 49.5 | 1265.92 | 54.00 | 56.73 | 1.35e-1 | 2.0e-3 |

MODNet with `--precision fp32`, which asks each provider for fp32 arithmetic where it can (errors only):

| Target | max err | mean err |
|---|---:|---:|
| cuda | 2.78e-4 | 1.0e-6 |
| tensorrt | 5.08e-5 | 6.5e-7 |
| openvino:CPU | 5.32e-5 | 7.2e-7 |
| openvino:GPU | 5.30e-5 | 6.1e-7 |
| openvino:NPU (fp16 only) | 1.35e-1 | 2.0e-3 |

## Reading the figures

- **Precision.** By default CUDA and TensorRT compute in TF32, and the OpenVINO GPU and NPU in fp16. That is why MODNet's default-mode errors are larger than the other models': a deep network fed noise amplifies the rounding of a 10-bit mantissa. With fp32 arithmetic the same model is within 2.8e-4 of the CPU on every target that has fp32. The errors come from the arithmetic, not from a faulty provider. The NPU has no fp32 inference, so its fp32 row repeats the default. The error bounds and the reasoning are in the model-check README's [Precision modes](../tools/ort-model-check/README.md#precision-modes). Switch TensorRT to fp32 with `NVIDIA_TF32_OVERRIDE=0` ([engines.md](engines.md#tensorrt)). The OpenVINO GPU needs `INFERENCE_PRECISION_HINT` ([engines.md](engines.md#openvino)).
- **TensorRT's `create` is the engine build**, about 10 s per model with static shapes here, with no engine cache. With dynamic shapes it moves to the first run, as for MODNet. `--trt-cache` removes it from later invocations.
- **OpenVINO GPU** pays seconds of kernel compilation at session creation.
- **`first ms`** includes lazy initialisation: CUDA's first run takes hundreds of milliseconds for that reason.
- **Small models** run in about a millisecond or less on the accelerators, so their medians are close to the cost of dispatching the work, and the p95 is sensitive to what else the machine was doing.
- **The selfie segmentation and PP-HumanSeg v2 models give outputs that depend little on the input noise** ([model-check README](../tools/ort-model-check/README.md#running-it)), so for them agreement shows that the engine runs the graph, not that it is sensitive to the input.

## CUDA 12 legacy provider

The CUDA 12 legacy provider, `onnxruntime-provider-cuda12-legacy` 1.29.0-1 ([cuda12-legacy.md](cuda12-legacy.md)), built in a clean chroot. It was run on a Blackwell GPU, not on the Pascal or Volta cards it is for, so the figures show that the library loads and computes with the CUDA 12 runtimes, not that its `sm_60`, `sm_61` and `sm_70` code runs.

| | |
|---|---|
| Package | `onnxruntime-provider-cuda12-legacy` 1.29.0-1: download 172.6 MiB (180943965 B), installed 557.5 MiB (584557139 B). Extracted, not installed |
| Core | the installed `onnxruntime-core` 1.29.0-6 (its `libonnxruntime.so.1.29.0`, copied next to the provider) and `python-onnxruntime-core` 1.29.0-6 |
| Runtimes | the libraries of `cuda-12.9` 12.9.2-1 (`libcudart.so.12.9.79`, `libcublas.so.12.9.2.10`, `libcublasLt.so.12.9.2.10`) and `cudnn9.10-cuda12.9` 9.10.2.21-1, extracted from the packages, not installed, on `LD_LIBRARY_PATH`; the host's CUDA 13.4.3 was untouched. `LD_DEBUG` showed the provider and every CUDA and cuDNN library loading from the CUDA 12 copies, and no `.so.13` |
| Host | GeForce RTX 5080 Laptop GPU (compute capability 12.0), driver 615.78.08, Intel Core Ultra 9 275HX |
| Model check | as above: `tools/ort-model-check --targets cpu,cuda`, `--precision default` (TF32) unless stated |

The provider's `DT_NEEDED` entries are `libcublasLt.so.12`, `libcublas.so.12`, `libcudart.so.12` and the system libraries (`libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6`, `ld-linux-x86-64.so.2`): no `.so.13`. This measurement is of 1.29.0-1, which did not link cuDNN. The 1.29.0-2 provider does, so its `DT_NEEDED` entries also have `libcudnn.so.9`; it was checked in the [1.29.0-7 round](#release-1290-7). Its only undefined symbol beyond the system's is `Provider_GetHost`. `cuobjdump` (of `cuda-12.9`) lists machine code for `sm_60`, `sm_61` and `sm_70` (181 kernels each), `sm_75` (222), `sm_80`, `sm_86` and `sm_89` (270 each), `sm_90a` (18), `sm_120a` (271), and the PTX of `compute_60` only. The 18 `sm_90a` kernels are compiled by ONNX Runtime whatever the architecture list is; the CUDA 13 `rtx` package has the same 18. `scripts/check-deps.sh` on the package found no gap and nothing missing; its three unresolved libraries (`libcudart`, `libcublas`, `libcublasLt` `.so.12`) and two bound mismatches (`cuda<13`, `cudnn<9.11`) are what a CUDA 13 host shows.

### Smoke test

`scripts/smoke-test.sh --only cpu,cuda --require cuda` (CPU EP fallback disabled): 2 ok, 0 failed. The `cpu` target reported max error 0; `cuda` reported `ok` with max error 0 (create 557.4 ms, first run 3647.18 ms, warm run 0.03 ms).

### Model check

All four models ran `ok` on `cuda` in the default precision. Columns as above; the CPU median is for comparison.

| Model | create ms | first ms | median ms | p95 ms | max err | mean err | cpu median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| selfie_segmentation_landscape, 1x144x256x3 | 394.2 | 2605.16 | 0.48 | 0.54 | 5.96e-8 | 9.70e-12 | 1.85 |
| pp_humansegv2_mobile_192x192 | 298.5 | 1611.07 | 1.19 | 1.33 | 1.84e-5 | 1.70e-6 | 2.87 |
| pp_humansegv2_lite_192x192 | 289.0 | 638.80 | 0.70 | 0.73 | 3.88e-4 | 2.72e-5 | 3.61 |
| modnet_photographic, `--size 512x512` | 389.7 | 2319.69 | 4.77 | 4.84 | 1.29e-1 | 1.80e-3 | 47.00 |

With `--precision fp32`:

| Model | max err | mean err |
|---|---:|---:|
| selfie_segmentation_landscape | 5.96e-8 | not recorded |
| pp_humansegv2_mobile_192x192 | 1.19e-7 | not recorded |
| pp_humansegv2_lite_192x192 | 5.96e-7 | not recorded |
| modnet_photographic | 1.23e-3 | 3.73e-6 |

In fp32 mode MODNet's CUDA maximum error is 1.23e-3, above the 1e-3 bound of the model check, although its mean error is 3.7e-6; it reproduced three times. The CUDA 13 provider gives 2.78e-4 on the same model and run (the table above). The default (TF32) mode passes on all four models.

**This is a known difference, not a failure of the package.** The bound is kept as it is. The older cuDNN (9.10) uses a different summation order from the cuDNN 9.27 of the CUDA 13 run. The mean error is 3.7e-6, and a difference of this size is invisible in a 0 to 1 mask.

### Not tested

- The legacy architectures on real Pascal or Volta hardware, and the `sm_75` to `sm_89` machine code on real cards of those generations.
- CUDA 12.9 with cuDNN 9.10 on the 580xx driver series. This run used driver 615.78.08 on a Blackwell GPU.

The 1.29.0-2 provider was run again, on the same GPU, in the [1.29.0-7 round](#release-1290-7); that round did not change either point.

## Release 1.29.0-7

This is the second set of measurements. It covers these packages ([packages.md](packages.md)):

- `onnxruntime-core`, `onnxruntime-provider-cuda13-rtx`, `onnxruntime-provider-tensorrt-cuda13`, `onnxruntime-provider-openvino`, `onnxruntime-provider-dnnl` and `python-onnxruntime-core`, all 1.29.0-7;
- `onnxruntime-provider-cuda13-all` 1.29.0-7;
- `onnxruntime-provider-cuda12-legacy` 1.29.0-2;
- `onnxruntime-provider-migraphx` and `python-onnxruntime-provider-migraphx` 1.29.0-1.

They are built with ONNX Runtime's training support and NCCL, so they are larger than 1.29.0-6 (see [sizes](#package-sizes)). The first build of 1.29.0-7 had two faults that only running it showed. They are described [below](#faults-found-and-fixed). The figures here are of the rebuilt packages.

### Conditions

| | |
|---|---|
| Machine | GeForce RTX 5080 Laptop GPU (driver 615.78.08); Intel Core Ultra 9 275HX (Arrow Lake, AVX2, no AVX-512) with its integrated GPU and NPU. The same machine as above |
| Where it ran | The packages were extracted into a scratch directory, not installed, and the installed 1.29.0-6 packages were left alone. Each run directory held the extracted core library next to the extracted provider, which is where the core looks for providers. For Python, the extracted `python-onnxruntime-core` tree was put on `PYTHONPATH` and run with `python -s -P` (`python -I` ignores `PYTHONPATH` and would have loaded the installed package) with its `capi` links pointed at the extracted libraries. No `LD_PRELOAD` was used anywhere |
| Vendor libraries | The host's CUDA 13, cuDNN, TensorRT and OpenVINO, versions not recorded for this run. NCCL 2.32.3 from Arch's `nccl` package, extracted and put on `LD_LIBRARY_PATH` (the main packages depend on `nccl`; it was not installed on this machine). For the CUDA 12 legacy provider, the libraries of `cuda-12.9`, `cudnn9.10-cuda12.9` (cuDNN 9.10) and `nccl-cuda12.9` (NCCL 2.30.7), extracted in the same way. For MIGraphX, Arch's ROCm libraries, extracted, where stated |
| Tools | `scripts/smoke-test.sh --runs 50` and `tools/ort-model-check/model-check.py --runs 50`, as in the conditions above; CPU EP fallback disabled unless stated |

### Results at a glance

| Check | Result |
|---|---|
| C smoke test, main packages | pass: 7 targets `ok` with max error 0 |
| C smoke test, `cuda` of the CUDA 12 legacy provider on the RTX 5080 with CUDA 12.9 libraries | pass: `ok`, max error 0; no `.so.13` file opened |
| MIGraphX provider, no AMD device | with ROCm libraries: `ok (no device)`, as designed; without them: `unavailable`; both exit 0 |
| Python `get_available_providers()` | pass; lists the oneDNN provider, and MIGraphX when the add-on is present |
| oneDNN provider from Python | pass: no crash and no out-of-memory in 34 separate processes |
| oneDNN provider on the four models without CPU fallback | fail: every model has nodes it cannot run (see [below](#the-onednn-provider-against-the-cpu)) |
| MODNet fp32, CUDA 13 `cuda` and `tensorrt` | pass (2.78e-4 and 5.07e-5, bound 1e-3) |
| MODNet fp32, CUDA 12 legacy `cuda` | known difference: 1.23e-3 against the 1e-3 bound, as in the first run (older cuDNN 9.10, different summation order; mean error 3.7e-6); default (TF32) mode passes |
| Package checks (dependencies, search paths, vendor files, metadata) | pass |

### C smoke test

`scripts/smoke-test.sh --runs 50` on the main packages: 7 ok, 0 failed. `create` is the session creation, `first` the first run and `warm` a later run, in milliseconds.

| Target | create ms | first ms | warm ms | max err |
|---|---:|---:|---:|---:|
| cpu | 3.2 | 0.08 | 0.02 | 0 |
| cuda | 581.3 | 373.34 | 0.02 | 0 |
| tensorrt | 3219.6 | 2.16 | 0.02 | 0 |
| openvino:CPU | 123.8 | 4.83 | 0.02 | 0 |
| openvino:GPU | 257.3 | 0.76 | 0.60 | 0 |
| openvino:NPU | 24.2 | 81.92 | 0.36 | 0 |
| dnnl | 4.8 | 5.11 | 2.36 | 0 |

CUDA 12 legacy 1.29.0-2 on the RTX 5080 with the CUDA 12.9 libraries: `cuda` ok, max error 0 (create 718.7 ms, first run 476.01 ms, warm run 0.02 ms). `LD_DEBUG` showed no `.so.13` file opened; cudart, cuBLAS and cuBLASLt 12, cuDNN 9.10.2 and NCCL 2.30.7 all came from the CUDA 12 copies.

MIGraphX: the machine has no AMD GPU. With Arch's ROCm libraries on the library path, the smoke test reports `ok (no device)`: the provider loads against the core and HIP reports no device (failure 100 from `hipGetDeviceCount`). Without them it reports `unavailable` (`libmigraphx_c.so.3` not found). Both exit 0. Nothing was run on an AMD GPU.

### Python

`get_available_providers()` from the extracted `python-onnxruntime-core`:

| | Providers listed |
|---|---|
| Without the MIGraphX link | `Tensorrt`, `CUDA`, `OpenVINO`, `Dnnl`, `CPU` |
| With the link that `python-onnxruntime-provider-migraphx` installs | `Tensorrt`, `CUDA`, `MIGraphX`, `OpenVINO`, `Dnnl`, `CPU` |

### The oneDNN provider against the CPU

Median milliseconds over 50 runs, with `--targets cpu,openvino:CPU,dnnl`. With CPU fallback disabled the oneDNN provider fails on all four models, because some nodes are assigned to the default CPU provider; `ok*` marks a result with `--allow-cpu-fallback`. The `openvino:CPU` column was measured with fallback allowed; the median of its fallback-disabled run is in brackets. The last column is the oneDNN median divided by the CPU median.

| Model | cpu | openvino:CPU | dnnl (ok*) | dnnl max err | dnnl / cpu |
|---|---:|---:|---:|---:|---:|
| selfie_segmentation_landscape | 1.81 | 1.86 (1.17) | 18.95 | 6.06e-8 | 10.5 |
| pp_humansegv2_mobile_192x192 | 3.14 | 3.46 (2.85) | 58.65 | 1.19e-7 | 18.7 |
| pp_humansegv2_lite_192x192 | 4.42 | 4.12 (4.26) | 29.86 | 3.58e-7 | 6.8 |
| modnet_photographic, `--size 512x512` | 50.05 | 45.86 (46.10) | 349.86 | 4.75e-5 | 7.0 |

The oneDNN outputs are correct, but it runs these models 7 to 19 times slower than the CPU provider, and it cannot run them without the CPU provider's help. The reasons, from the profiles of the two larger models (oneDNN 3.11.3 on a CPU with AVX2):

1. **The graph is split into many pieces.** `pp_humansegv2_lite` becomes 55 oneDNN subgraphs and 38 CPU nodes, MODNet 61 subgraphs and 94 CPU nodes. For MODNet the oneDNN kernels take 343 ms per run against 17 ms for the CPU nodes, and the 532 data-layout conversions at the subgraph boundaries over 5 runs add about 59 ms per run.
2. **Two thread pools compete.** ONNX Runtime's own pool (24 threads that spin while waiting) runs next to the 24 OpenMP threads of `libgomp`. With `session.intra_op.allow_spinning=0`, `pp_humansegv2_lite` goes from 33.8 to 9.9 ms and MODNet from 362.6 to 151.1 ms (with one intra-op thread: 9.8 and 225.7 ms). `OMP_NUM_THREADS=8` did not help. Even with spinning off the provider stays slower than the CPU provider on these models.
3. **The primitives are created for training.** The 1.29.0-7 build has training support, and oneDNN primitives are created with `forward_training`, not `forward_inference`.

So on these four models the oneDNN provider is not usable without the CPU fallback, and the CPU provider or OpenVINO's `CPU` device is the better choice ([engines.md](engines.md#onednn)). The figures are for this CPU and these models only.

### MODNet in fp32

`--size 512x512 --precision fp32`, bound 1e-3 on the maximum error, as in the first run.

| Provider | max err | mean err | median ms | Result |
|---|---:|---:|---:|---|
| CUDA 13, `cuda` | 2.78e-4 | 1.01e-6 | 11.18 | ok |
| CUDA 13, `tensorrt` | 5.07e-5 | 6.48e-7 | 3.43 | ok |
| CUDA 12 legacy, `cuda` | 1.23e-3 | 3.73e-6 | 7.83, 8.04, 7.94 | known difference: over the bound in 3 of 3 runs, see below |

The CUDA 12 figure is the same as in the first run, just above the bound while the mean error is small. It is a **known difference, not a failure of the package**: the bound stays at 1e-3, and the older cuDNN (9.10) uses a different summation order. The mean error is 3.7e-6 and the difference is invisible in a 0 to 1 mask. The default (TF32) mode of the CUDA 12 provider passes: max error 1.29e-1, mean 1.80e-3, median 4.82 ms, the same errors as in the [CUDA 12 section](#model-check-1) above.

### Package sizes

Installed size is the `size` field of the package, in MiB; 1.29.0-6 was built without training support.

| Package | 1.29.0-6 | 1.29.0-7 | Growth | Download of -7 |
|---|---:|---:|---:|---:|
| `onnxruntime-core` | 29.44 | 33.51 | +4.07 | 8.98 |
| `onnxruntime-provider-cuda13-rtx` | 154.74 | 162.98 | +8.24 | 103.63 |
| `onnxruntime-provider-cuda13-all` | 294.88 | 307.87 | +12.99 | 235.57 |
| `python-onnxruntime-core` | 43.15 | 52.39 | +9.24 | 12.12 |
| `onnxruntime-provider-cuda12-legacy` (1.29.0-1 to -2) | 557.48 | 587.86 | +30.39 | 186.09 |
| `onnxruntime-provider-dnnl` | none | 64.26 | new | 10.15 |
| `onnxruntime-provider-tensorrt-cuda13` | 1.15 | 1.15 | 0 | 0.31 |
| `onnxruntime-provider-openvino` | 1.70 | 1.70 | 0 | 0.51 |
| `onnxruntime-provider-migraphx` | none | 0.73 | new | 0.17 |
| `python-onnxruntime-provider-migraphx` | none | 0 (two symbolic links) | new | 3.4 KiB |

The growth is the training code. The oneDNN provider carries oneDNN statically and links only `libgomp`, `libstdc++`, `libm`, `libgcc_s` and `libc`.

### Build times

Clean-chroot builds on this machine, each a forced rebuild, wall-clock time:

| Build | Time |
|---|---:|
| Main packages, `rtx` preset (`ORTF_CUDA_ARCHS=120`) | 1 h 04 min 27 s |
| Main packages, `all` preset (all CUDA architectures) | 2 h 08 min 53 s |
| CUDA 12 legacy line 1.29.0-2 | 1 h 28 min 56 s |
| MIGraphX line 1.29.0-1 (ROCm chroot) | 8 min 51 s |
| `nccl-cuda12.9` 2.30.7-2 (AUR package the CUDA 12 line depends on) | 11 min 24 s |

### Package checks

On the extracted packages, by hand and with the recipe's own guards:

- Every CUDA provider (`rtx`, `all`, CUDA 12 legacy) lists `libcudnn.so.9` and `libnccl.so.2` as needed libraries. The core and the Python module need neither `libnccl` nor the CUDA libraries.
- No package has a search path other than `$ORIGIN` (the TensorRT provider also has `/opt/nvidia/tensorrt/lib`, on purpose), and no file mentions the build directory except the build metadata that `makepkg` writes.
- No package contains a vendor library (CUDA, cuDNN, NCCL, TensorRT, OpenVINO, ROCm, MIGraphX, oneDNN or `libgomp`).
- `python-onnxruntime-provider-migraphx` contains two symbolic links and no file with content.
- The recipe's search-path and undefined-symbol checks pass on all 11 packages; the only undefined symbols are the two the core provides, `Provider_GetHost` and `Provider_SetHost`.
- Each package's metadata agrees with the recipe's `.SRCINFO`, apart from the version bounds that are added when the package is built.

### Faults found and fixed

An earlier build of 1.29.0-7 was run the same way and had two faults, which the rebuilt packages above no longer have:

- **The CUDA provider could not load.** Training adds CUDA code that calls cuDNN directly, but the provider did not link cuDNN, so it failed with `undefined symbol: cudnnGetConvolutionBackwardDataWorkspaceSize`, and the TensorRT provider then reported that the CUDA provider was not available. The recipe now links cuDNN into the provider and, when packaging, checks every provider for undefined symbols.
- **The oneDNN provider crashed from Python.** Sometimes it segfaulted, sometimes `libgomp` reported an out-of-memory error of hundreds of gigabytes. ONNX Runtime 1.29.0's Python binding gives the provider's factory a pointer to a local variable holding the thread count, and the provider reads it after the function has returned, so it passed a garbage number (in one run 1459638976) to `omp_set_num_threads`. The C API leaves the pointer null, which is why the C smoke test passed. The same code is in Arch's package, which should have the same fault; that was not run here. The recipe now patches the provider's factory to keep its own copy of the value. With the patch `omp_set_num_threads` is called once, with 24, the number of CPU threads.

Two smaller faults were fixed too: the Python module carried the build directory in its search path, and the MIGraphX Python add-on carried copies of licence files, which are now a link.

### Installed packages

The packages were then installed with `pacman` and run from `/usr`, with no environment overrides: no `PYTHONPATH`, `LD_LIBRARY_PATH` or `LD_PRELOAD` was set. The set is `onnxruntime-core`, `onnxruntime-provider-cuda13-rtx`, `onnxruntime-provider-tensorrt-cuda13`, `onnxruntime-provider-openvino`, `onnxruntime-provider-dnnl` and `python-onnxruntime-core`, all 1.29.0-7, with `nccl` 2.32.3-1 from the repository. They were installed as an upgrade over 1.29.0-6 (`pacman -U`; the pacman log shows the five 1.29.0-6 packages upgraded and the oneDNN provider newly installed). The machine is the GeForce RTX 5080 Laptop GPU and Intel Core Ultra 9 275HX of the runs above, with CUDA 13.4.3, cuDNN 9.27.0.42, TensorRT 11.4.0.106, OpenVINO 2026.4.1 and Python 3.14.8.

- **C smoke test.** `scripts/smoke-test.sh --runs 50` reported 7 of 7 targets `ok`, each with max error 0. The providers available were TensorRT, CUDA, OpenVINO, DNNL and CPU. `nccl` came from the package, so no library path was needed. `create`, `first` and `warm` are in milliseconds:

| Target | create ms | first ms | warm ms | max err |
|---|---:|---:|---:|---:|
| cpu | 8.5 | 0.44 | 0.02 | 0 |
| cuda | 713.5 | 361.37 | 0.02 | 0 |
| tensorrt | 3205.8 | 2.12 | 0.02 | 0 |
| openvino:CPU | 141.9 | 0.54 | 0.01 | 0 |
| openvino:GPU | 251.7 | 0.78 | 0.55 | 0 |
| openvino:NPU | 38.4 | 7.28 | 0.54 | 0 |
| dnnl | 15.3 | 24.93 | 2.16 | 0 |

  `--only cpu,migraphx-load` gives `cpu` `ok` and `migraphx-load` `unavailable`, because the MIGraphX provider library is not installed; the exit status is 0.

- **Python.** `onnxruntime` 1.29.0 loads from `/usr/lib/python3.14/site-packages` and lists `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `OpenVINOExecutionProvider`, `DnnlExecutionProvider` and `CPUExecutionProvider`. `get_device()` reports `GPU-DNNL-MIGRAPHX` although no MIGraphX package is installed. That is a string fixed when the library is built, so it does not show what is loaded; `get_available_providers()` does.
- **Training import.** `import onnxruntime.training` needs PyTorch, as the [packages](packages.md) page says. `python-pytorch` is an optional dependency and was not installed, so the import stops with `ModuleNotFoundError: No module named 'torch'`, and `from onnxruntime.training import artifacts` fails the same way. The other training dependencies import: onnx 1.22.0, sympy 1.14.0, cerberus 1.3.8, h5py 3.16.0 and numpy 2.5.4.
- **Library links.** The six entries in the Python `capi` directory (the core library and the five `providers_*` links) all point to `/usr/lib` and resolve. `onnxruntime_pybind11_state.so` has `RUNPATH` `$ORIGIN`. `pacman -Qkk` reports no altered file: 52 in `onnxruntime-core`, 11 in the CUDA 13 provider, 10 in the TensorRT provider, 10 in OpenVINO, 12 in DNNL and 1433 in `python-onnxruntime-core`.
- **oneDNN provider from Python.** The repro ran 19 times without a crash or an out-of-memory error: 10 runs of PP-HumanSeg v2 mobile, 3 of selfie segmentation, 3 of PP-HumanSeg v2 lite and 3 of MODNet at 512x512, each with the CPU provider registered as fallback. The crash and the out-of-memory error described [above](#faults-found-and-fixed) did not recur.
- **MODNet in fp32.** `--size 512x512 --precision fp32` on `cpu`, `cuda` and `tensorrt`, with exit 0 and every error within the 1e-3 bound. TensorRT ran with TF32 off (`NVIDIA_TF32_OVERRIDE=0`, as the tool notes).

| Target | create ms | first ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 37.8 | 60.75 | 48.99 | 56.41 | 0 | 0 |
| cuda | 361.7 | 520.27 | 10.69 | 11.38 | 2.78e-4 | 1.01e-6 |
| tensorrt | 229.5 | 8867.10 | 3.46 | 3.90 | 5.14e-5 | 6.55e-7 |

  The CUDA error matches the extracted run above (2.78e-4); the TensorRT error is 5.14e-5 here against 5.07e-5 there, under the same bound.
- **A consumer application on the system library.** A Rust program using the `ort` crate 2.0.0-rc.15 with `load-dynamic` loads `libonnxruntime.so.1` from the system path. Its `auto` setting selects NVIDIA TensorRT and the engine loads. The test suite passes with `ORT_DYLIB_PATH` set to that library: 59 unit tests (one benchmark ignored), 1 exit test (which ran the OpenVINO CPU, GPU and NPU engines), 6 module tests and 1 runtime test. The module tests also pass with the variable unset, resolving the library from the system path. The ignored hardware and Intel GPU tests pass too; on CUDA frames `auto` (TensorRT) took 3.14 to 3.24 ms per frame, `cuda` 3.45 ms and `cpu` 3.86 ms.
- With TensorRT 11, the provider logs that the BF16, FP16 and INT8 precision flags have been removed and that strongly-typed networks are used. An engine that asks for FP16 therefore runs the model's own types. The message is a notice, not a failure.

### Not tested

- The MIGraphX provider on an AMD GPU. Only loading, and the no-device result, were run.
- The CUDA 12 legacy provider on Pascal or Volta hardware; see [above](#not-tested).
- The installed CUDA 12 line. The `onnxruntime-provider-cuda12-legacy` package was not installed, because three of its dependencies are AUR builds.
- Training with PyTorch. `python-pytorch` was not installed, so `onnxruntime.training` was only imported without it ([above](#installed-packages)), and training itself was not run.

## WebGPU provider 1.29.0-8

`onnxruntime-provider-webgpu` ([engines.md](engines.md#webgpu)) was run on two GPUs: the NVIDIA GPU and the Intel integrated GPU of the test machine. The measurements are of this one package. The WebGPU provider is **not shown to be faster than the CPU** here: see [the reading](#webgpu-reading-the-figures).

### Conditions

| | |
|---|---|
| Package | The WebGPU provider of 1.29.0-8, built alone and offline (`ORTF_EPS=webgpu ORTF_NCCL=0 ORTF_PYTHON=0`, so `-nccloff` names), with Dawn `v20260714.215939` built in. 5.9 MiB to download, 16.2 MiB installed; the library is 16,606,312 bytes stripped and needs only `libstdc++`, `libm`, `libgcc_s` and `libc` ([build.md](build.md#parallelism-memory-and-disk)). The packages were extracted, not installed |
| NVIDIA GPU | GeForce RTX 5080 Laptop GPU (`10de:2c19`, PCI `0000:01:00.0`), driver 615.78.08 (`nvidia-utils`) |
| Intel GPU | the Arrow Lake integrated graphics of the Core Ultra 9 275HX (`8086:7d67`, PCI `0000:00:02.0`), Mesa's `vulkan-intel` 1:26.2.5-1 |
| CPU and kernel | Intel Core Ultra 9 275HX, 24 threads; Linux 7.2.15-1-omarchy |
| Where it ran | The smoke test against the extracted core and provider of 1.29.0-8, with `--prefix`. The model check used the machine's installed Python module (1.29.0-7), with `--webgpu-lib` naming the extracted provider library of 1.29.0-8; the plugin loaded into that core |
| Smoke test | `scripts/smoke-test.sh --require webgpu:intel --only cpu,webgpu:intel`, and `--require webgpu:nvidia --only webgpu:nvidia`, each target alone, 5 warm runs, CPU EP fallback disabled |
| Model check | `tools/ort-model-check/model-check.py --targets cpu,webgpu:intel,webgpu:nvidia --webgpu-lib <library>`, 50 timed runs after 3 warm-up runs, CPU EP fallback disabled, the defaults of [the conditions above](#conditions). The desktop was in use. The Intel GPU did not appear in Dawn's device list on the first runs of that day and did after a reboot; the cause was not investigated, and the figures below are from after it |

### Smoke test

Each WebGPU target ran alone, so that its result does not depend on a Dawn device left over from an earlier target. Milliseconds; max error 0 on every target.

| Target | status | create ms | first ms | warm ms | max err |
|---|---|---:|---:|---:|---:|
| cpu | ok | 12.4 | 0.42 | 0.01 | 0 |
| webgpu:intel | ok | 89.1 | 55.38 | 1.11 | 0 |
| webgpu:nvidia, with the Intel GPU active | ok | 149.9 | 9.88 | 0.32 | 0 |

The `device:` lines of the smoke test say which GPU did the work. For the Intel GPU: `Intel GPU 8086:7d67 at 0000:00:02.0, power preference low-power: the kernel counted the work on this GPU`. For the NVIDIA GPU: `NVIDIA GPU 10de:2c19 at 0000:01:00.0, power preference high-performance: Dawn's adapter is inferred from the power preference (no kernel usage counters for this GPU)`. The kernel keeps usage counters for the Intel GPU, so that verdict is measured. The NVIDIA driver keeps none, so that one is inferred from the power preference, not measured. The NVIDIA target also passed when the Intel GPU was the active one.

Before the reboot, the NVIDIA target had also passed on the extracted package (create 161.3 ms, first 8.90 ms, warm 0.33 ms, max error 0).

### Model check

The same models as above. Medians are in milliseconds, and the errors are against the CPU outputs.

#### selfie_segmentation_landscape, input 1x144x256x3

| Target | create ms | first ms | mean ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 15.4 | 9.53 | 1.89 | 1.85 | 2.35 | 0 | 0 |
| webgpu:nvidia | 179.1 | 31.64 | 1.07 | 1.05 | 1.15 | 6.06e-8 | 9.80e-12 |
| webgpu:intel | 23.1 | 145.07 | 5.13 | 4.57 | 9.09 | 6.06e-8 | 9.80e-12 |

#### pp_humansegv2_mobile_192x192, input 1x3x192x192

| Target | create ms | first ms | mean ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 43.7 | 6.64 | 2.94 | 2.92 | 3.11 | 0 | 0 |
| webgpu:nvidia | 201.0 | 44.85 | 4.10 | 3.86 | 5.71 | 1.79e-7 | 5.88e-9 |
| webgpu:intel | 79.4 | 364.80 | 16.97 | 16.99 | 17.09 | 1.79e-7 | 5.82e-9 |

#### pp_humansegv2_lite_192x192, input 1x3x192x192

| Target | create ms | first ms | mean ms | median ms | p95 ms | max err | mean err |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 24.8 | 11.87 | 4.57 | 3.69 | 7.54 | 0 | 0 |
| webgpu:nvidia | 198.2 | 38.72 | 1.91 | 1.70 | 2.54 | 5.96e-7 | 9.26e-8 |
| webgpu:intel | 77.3 | 190.08 | 7.87 | 7.60 | 9.62 | 5.96e-7 | 8.42e-8 |

An earlier run on the RTX alone gave medians of 1.38, 3.72 and 1.70 ms for the three models, with the same maximum and mean errors, so the timings move a little from run to run.

#### modnet_photographic, input 1x3x512x512: does not run

`--size 512x512`, the model of the earlier tables. With CPU fallback disabled, session creation fails on the Intel GPU (this run) and on the RTX (an earlier run, same tool and library) with `This session contains graph nodes that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled`. The CPU target ran (median 48.35 ms here, 51.55 ms in the earlier run). The cause is one int64 `Gather`, fed by a `Shape` node, which has no WebGPU kernel in ONNX Runtime 1.29.0.

With CPU fallback allowed, MODNet runs on the RTX but the output is wrong: maximum error 1.00 and mean error 0.14 against the CPU's output. The first wrong node is a convolution (`Conv_304`). A sweep of single 3x3 convolutions found them wrong exactly when the input channel count is a multiple of 3, is not a multiple of 4, and is above 3 (6, 9, 15, 18 and so on). This was seen on the NVIDIA GPU through Vulkan; whether the Intel GPU does the same was not checked, because it stops earlier at the `Gather`. Both look like limitations of ONNX Runtime's own WebGPU provider, which this recipe does not change. It is a **known limitation**, and not fixed here.

### Python

On the machine's installed 1.29.0-7 Python module, `InferenceSession(model, providers=["WebGpuExecutionProvider"])` does not reach the plugin provider, whether or not `register_execution_provider_library` was called first: ONNX Runtime warns, drops the name and runs on the CPU. With `SessionOptions.add_provider_for_devices` and a device from `get_ep_devices()`, the session lists `WebGpuExecutionProvider` and `CPUExecutionProvider` and runs the model on the GPU (the model check works this way). The 1.29.0-8 Python module, which carries a patch for the plugin provider, was not run.

### WebGPU: reading the figures

- **No speed-up is shown.** Medians over 50 runs, CPU against the GPU: the RTX was faster on selfie segmentation (1.05 against 1.85 ms) and PP-HumanSeg lite (1.70 against 3.69 ms) and slower on PP-HumanSeg mobile (3.86 against 2.92 ms). The Intel GPU was slower on all three: 4.57 against 1.85 ms, 16.99 against 2.92 ms and 7.60 against 3.69 ms, which is 2.1 to 5.8 times the CPU's median. The CPU here is a fast desktop-class laptop CPU, and the models are small, so each run takes a few milliseconds. A speed-up on older machines (NVIDIA Maxwell, Intel Gen8 and later graphics through Mesa, AMD Polaris and Vega APUs) is **not tested**. The value of the provider is a working GPU path where no vendor provider exists.
- **Accuracy.** The outputs agree with the CPU's to a maximum error of at most 6e-7, in fp32, on both GPUs.
- **Start-up.** Session creation on a GPU took 23 to 201 ms and the first run 9.9 to 365 ms, against 0.4 to 12 ms for the CPU's first run. In the model check the Intel GPU's first run was the slowest, 145 to 365 ms.
- **What was not run.** The WebGPU package of the default (NCCL) build, installed with `pacman` and run from `/usr`; the Python module of 1.29.0-8; any GPU other than the two above.

Ran it on hardware marked not tested? Report the result, working or not, in an [issue](https://github.com/rsd/omarchy-onnxruntime/issues). A report moves the flag.

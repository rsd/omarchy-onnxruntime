# Measured results

What the recipe's checks reported for one full build, and the conditions they were run under. The figures are ONNX Runtime figures for the raw model only, without any pre- or post-processing, and are indicative: the desktop was in use during the runs. The tools that produced them are the [smoke test](../tools/ort-smoke/README.md) and the [model check](../tools/ort-model-check/README.md); terms such as TF32 and fp16 are in the [glossary](glossary.md).

## Conditions

| | |
|---|---|
| ONNX Runtime | 1.29.0, built by this recipe with `ORTF_EPS` at its default and `ORTF_CUDA_ARCHS=120` ([build.md](build.md#parallelism-memory-and-disk) has the build and the vendor runtime versions). It is the build of the single package `onnxruntime-full` 1.29.0-1; since 1.29.0-2 the same libraries are split into `onnxruntime-core` and one package per engine ([build.md](build.md#package-names)), and the split packages were not measured separately |
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

- **Precision.** By default CUDA and TensorRT compute in TF32 and the OpenVINO GPU and NPU in fp16. That is why MODNet's default-mode errors are larger than the other models': a deep network fed noise amplifies the rounding of a 10-bit mantissa. With fp32 arithmetic the same model is within 2.8e-4 of the CPU on every target that has fp32, so the errors come from the arithmetic and not from a faulty provider. The NPU has no fp32 inference, so its fp32 row repeats the default. The error bounds and the reasoning are in the model-check README's [Precision modes](../tools/ort-model-check/README.md#precision-modes). TensorRT can be switched to fp32 with `NVIDIA_TF32_OVERRIDE=0` ([engines.md](engines.md#tensorrt)); the OpenVINO GPU needs `INFERENCE_PRECISION_HINT` ([engines.md](engines.md#openvino)).
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

The provider's `DT_NEEDED` entries are `libcublasLt.so.12`, `libcublas.so.12`, `libcudart.so.12` and the system libraries (`libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6`, `ld-linux-x86-64.so.2`): no `.so.13`. Its only undefined symbol beyond the system's is `Provider_GetHost`. `cuobjdump` (of `cuda-12.9`) lists machine code for `sm_60`, `sm_61` and `sm_70` (181 kernels each), `sm_75` (222), `sm_80`, `sm_86` and `sm_89` (270 each), `sm_90a` (18), `sm_120a` (271), and the PTX of `compute_60` only. The 18 `sm_90a` kernels are compiled by ONNX Runtime whatever the architecture list is; the CUDA 13 `rtx` package has the same 18. `scripts/check-deps.sh` on the package found no gap and nothing missing; its three unresolved libraries (`libcudart`, `libcublas`, `libcublasLt` `.so.12`) and two bound mismatches (`cuda<13`, `cudnn<9.11`) are what a CUDA 13 host shows.

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

### Not tested

- The legacy architectures on real Pascal or Volta hardware, and the `sm_75` to `sm_89` machine code on real cards of those generations.
- CUDA 12.9 with cuDNN 9.10 on the 580xx driver series. This run used driver 615.78.08 on a Blackwell GPU.

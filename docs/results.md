# Measured results

What the recipe's checks reported for one full build, and the conditions they were run under. The figures are ONNX Runtime figures for the raw model only, without any pre- or post-processing, and are indicative: the desktop was in use during the runs.

## Conditions

| | |
|---|---|
| ONNX Runtime | 1.29.0, built by this recipe with `ORTF_EPS` at its default and `ORTF_CUDA_ARCHS=120` ([build.md](build.md#parallelism-memory-and-disk) has the build and the vendor runtime versions). It is the build of the single `onnxruntime-full` 1.29.0-1; 1.29.0-2 splits the same libraries into `onnxruntime-core` and one package per engine ([build.md](build.md#package-names)) and was not measured separately |
| NVIDIA GPU | GeForce RTX 5080 Laptop GPU |
| CPU | Intel Core Ultra 9 275HX, 24 threads, 30 GiB of RAM |
| Intel GPU and NPU | the integrated graphics and the NPU of that CPU |
| Runtimes | CUDA 13.4.3, cuDNN 9.27.0.42, TensorRT 11.4.0.106, OpenVINO 2026.4.1 (intel-npu-driver 1.38.0) |
| Where it ran | the build tree, before installing ([install.md](install.md#before-installing-the-build-tree)) |
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

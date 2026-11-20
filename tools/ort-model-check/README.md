# ort-model-check

Runs a real ONNX model on every execution provider of an ONNX Runtime and
compares the results. It complements [ort-smoke](../ort-smoke/README.md), which
runs a tiny synthetic model: a real model exercises the operator coverage,
precision and speed of each engine on a graph of realistic size.

For each target the script creates a session that holds only that execution
provider, feeds every target the same deterministic input, and compares every
output with the CPU target's. It reports the session creation time, the first run, and the mean, median and
95th percentile of the later runs. With dynamic input shapes TensorRT builds
its engine and OpenVINO compiles the model at the first run, not at session
creation, so that cost lands in `first ms` (a model with static shapes pays it
at creation); a TensorRT engine cache (`--trt-cache`) removes it from later
invocations.

Requirements: Python 3, `numpy` and the `onnxruntime` Python module. Nothing
else outside the standard library.

## Targets

The names are those of ort-smoke.

| Target | Execution provider | `default` mode | `fp32` mode |
|---|---|---|---|
| `cpu` | CPU (the reference) | fp32: max 0 | fp32: max 0 |
| `cuda` | CUDA, device 0 | TF32: max 0.25, mean 5e-3 | fp32 (`use_tf32=0`): max 1e-3 |
| `tensorrt` | TensorRT, fp16 off, engine cache off unless `--trt-cache` | TF32: max 0.25, mean 5e-3 | fp32 (`NVIDIA_TF32_OVERRIDE=0`): max 1e-3 |
| `openvino:CPU` | OpenVINO, Intel CPU plugin | fp32: max 1e-3 | fp32: max 1e-3 |
| `openvino:GPU` | OpenVINO, Intel GPU plugin | fp16: max 1.0, mean 1e-2 | fp32 (`INFERENCE_PRECISION_HINT=f32`): max 1e-3 |
| `openvino:NPU` | OpenVINO, Intel NPU plugin | fp16: max 1.0, mean 1e-2 | fp16 (no fp32 inference): max 1.0, mean 1e-2 |
| `dnnl` | oneDNN, on the CPU (default options; never measured) | fp32: max 1e-3 | fp32: max 1e-3 |

Each cell gives the arithmetic the target runs in and the bounds on the maximum
and mean absolute error against the CPU outputs; a target passes when every
bound holds. `openvino` selects the three OpenVINO devices. The bounds are for
outputs of magnitude around 1; use `--tol` for a model whose outputs are on
another scale.

## Precision modes

`--precision default` (the default) runs every provider with its own defaults,
which is what an application gets: TF32 convolutions and matrix products on
CUDA and TensorRT, fp16 inference on the OpenVINO GPU and NPU. `--precision
fp32` asks each provider for fp32 arithmetic where it has a way to do so:

- CUDA: the provider option `use_tf32=0`.
- TensorRT: ONNX Runtime 1.29 has no TensorRT option for TF32, so the script
  sets `NVIDIA_TF32_OVERRIDE=0` in its own environment before loading ONNX
  Runtime. cuBLAS, cuDNN and TensorRT honour it, so it applies to the `cuda`
  target as well. The row's note says so.
- OpenVINO GPU: `load_config` with `{"GPU": {"INFERENCE_PRECISION_HINT": "f32"}}`.
  The OpenVINO EP's own `precision=FP32` option is accepted but, with this
  build, left the GPU at fp16 (its results were identical to the default's),
  so it is not used.
- OpenVINO NPU: the NPU has no fp32 inference (an f32 precision hint fails to
  compile), so it runs at its fp16 default with the fp16 bounds, and the row's
  note says so.
- CPU and OpenVINO CPU already run in fp32.

Why the reduced-precision bounds are loose on the maximum and tight on the
mean: TF32 and fp16 keep a 10-bit mantissa, and a deep network fed uniform
noise amplifies that rounding. On MODNet at 512x512, whose output is a matte in
[0, 1], seeds 0 to 3 gave maximum errors of up to 9.7e-2 (CUDA), 1.1e-1
(TensorRT), 4.6e-1 (OpenVINO GPU) and 2.9e-1 (NPU), while the mean errors
stayed at or below 1.4e-3 under TF32 and 5.4e-3 under fp16. With TF32 or fp16
turned off, the same model is within 2.8e-4 of the CPU, so the large maximum
errors come from the arithmetic, not from a faulty provider. The mean is therefore the meaningful
test for those targets: its bounds (5e-3 for TF32, 1e-2 for fp16) sit about
3.5 and 1.9 times above the worst measured value. A wrong result lies above
them: MODNet's output for a different noise input differs from the reference by
a mean of 7.1e-2 at 512x512 and 1.9e-2 at 256x256, and an all-zero output by
2.7e-1. The maximum keeps a loose bound, about twice the worst measured value,
against isolated gross outliers. For a strict comparison of the engines' results,
use `--precision fp32`, whose bounds are tight (1e-3 on the maximum; the worst
measured on the reference models was 2.8e-4).

For every target except `cpu`, the session option
`session.disable_cpu_ep_fallback=1` is set, so a graph the provider cannot take
fails instead of quietly running on the CPU (see `--allow-cpu-fallback` below
for when that is too strict). The session's provider list must also start with
the requested provider. The CPU target always runs first as
the reference, even when it is not in `--targets`; it is reported only when
requested.

## Running it

Against an installed package (`python-onnxruntime-core`):

    tools/ort-model-check/model-check.py model.onnx
    tools/ort-model-check/model-check.py model.onnx --targets cpu,cuda --require cpu,cuda

Against a build tree before installing, point `PYTHONPATH` at the build
directory that holds the `onnxruntime/` Python package built by CMake:

    PYTHONPATH=<build dir> tools/ort-model-check/model-check.py model.onnx

With TensorRT installed under `/opt`, load the TensorRT environment first, so
the provider finds the TensorRT libraries:

    set -a; . /usr/share/nvidia-tensorrt/nvidia-tensorrt.env; set +a

Options:

| Option | Meaning |
|---|---|
| `--targets LIST` | targets to run and report, comma-separated, default all |
| `--runs N` | timed runs per target, default 50 |
| `--warmup N` | untimed runs after the first one, default 3 |
| `--size HxW` | height and width for dynamic spatial dimensions |
| `--require LIST` | exit 1 unless these targets are `ok` |
| `--json FILE` | also write the results as JSON |
| `--trt-cache DIR` | enable the TensorRT engine cache, stored in `DIR` |
| `--allow-cpu-fallback` | do not disable the CPU EP fallback for the non-cpu targets; their `ok` rows read `ok*` |
| `--precision MODE` | `default` (each provider's own arithmetic) or `fp32`; see [Precision modes](#precision-modes) |
| `--tol TARGET=MAX[:MEAN]` | override the bounds on the max and mean abs error for a target; repeatable |
| `--seed N` | seed of the input data, default 0 |

Why `--allow-cpu-fallback` exists: for models with dynamic input shapes the CUDA
EP deliberately keeps small shape-computation nodes (`Shape`, `Gather` on shape
tensors and the like) on the CPU, and the TensorRT EP alone leaves the nodes it
rejects to another provider. With the fallback disabled those targets fail
although the engine runs the heavy part of the graph. With the flag, such a
session is created, and a target that passes is marked `ok*` with the note
"CPU fallback allowed: some nodes may run on the CPU". An `ok*` target counts as
ok for `--require`, but the timings include whatever ran on the CPU, so do not
read it as a pure-engine result. The session's first provider must still be the
requested one.

`--tol` examples: `--tol cuda=1e-2` sets the maximum bound of `cuda` and keeps
its mean bound; `--tol openvino:GPU=0.5:5e-3` sets both; `--tol cuda=:1e-3`
sets only the mean bound; `inf` as the mean removes that bound. `TARGET` may be
`openvino` for all three devices, and several overrides can be given in one
comma-separated argument. The overrides apply on top of the selected mode's
bounds.

Input data: each model input gets uniform random values in [0, 1) from
`numpy.random.default_rng(seed)`, as float32 (cast to float16 for float16
inputs); other input types are rejected. A dynamic batch dimension becomes 1.
Dynamic spatial dimensions come from `--size`; the layout is read from where the
channel dimension (1, 3 or 4) sits, so both NCHW and NHWC inputs work. When a
dimension is dynamic and `--size` is missing, the script exits with status 2
and names the input and its shape.

Noise input limits what the comparison can detect. Some models give an output
on noise that barely depends on the input: the MediaPipe selfie segmentation
model outputs values near 0 everywhere (an all-zero output is within 1e-10 of
it), and PP-HumanSeg v2 mobile gives nearly the same output for any noise
input. For such models the check confirms that the engine runs the graph and
agrees numerically on that output, but a provider that produced a plausible
output unrelated to the input could go unnoticed.

## Reading the output

    ONNX Runtime 1.29.0; model model.onnx (0123456789ab); inputs input 1x3x192x192 float32; available providers ...
    precision default: each provider's own arithmetic, reduced-precision bounds for TF32 and fp16
    3 warm-up and 50 timed run(s) per target; CPU EP fallback disabled for non-cpu targets

    target         status       precision  create ms   first ms    mean ms  median ms     p95 ms    max err   mean err
    cpu            ok           fp32            35.2      18.40      12.10      12.00      13.20   0.00e+00   0.00e+00
    cuda           ok           TF32           424.3     278.65       1.36       1.37       1.52   3.62e-05   4.53e-06
    openvino:NPU   unavailable  fp16               -          -          -          -          -          -          -
        note: openvino:NPU: OpenVINO device NPU not present (ORT said: ...)

The second header line names the precision mode. When `NVIDIA_TF32_OVERRIDE` is
already set in the environment in `default` mode, a third line says so, since it
may turn TF32 off. The `precision` column is the arithmetic the target was asked
to run in, which selects its bounds. Times are in milliseconds. `max err` and
`mean err` are the largest and the
mean absolute difference from the CPU outputs over all floating-point outputs;
the bounds apply to those only. Integer, bool and string outputs must
match the CPU's exactly, and outputs that are not tensors (sequences, maps) are
not compared, which the notes say. The header gives
the first 12 hex digits of the model's SHA-256.

- **ok** - the session was created, ran, and the errors are within the
  target's bounds.
- **ok\*** - as ok, with `--allow-cpu-fallback`: some nodes may have run on the
  CPU.
- **FAIL** - the provider is available, but session creation, a run or the
  comparison failed: an error, a non-CPU session that does not list the
  requested provider first, a NaN in any output, an output of another shape or
  type, an integer output that differs, or a maximum or mean error above its
  bound. The first line of the error follows in
  the notes.
- **unavailable** - the engine cannot be used on this machine; not a failure
  unless the target is in `--require`. The same causes as ort-smoke: the
  provider is not in `onnxruntime.get_available_providers()`; loading the
  provider library fails with a "cannot open shared object file" error (the
  vendor runtime is not installed; the note names the library); the provider
  library does not exist in the package's `capi/` directory ("provider not
  built"); or OpenVINO reports `Device <name> is not available`. Any other
  error is a `FAIL`.

  The script loads `capi/libonnxruntime_providers_<cuda|tensorrt|openvino|dnnl>.so`
  itself before creating the session. ONNX Runtime only logs a failure to load
  the provider and then fails the session with nodes assigned to the CPU EP,
  which hides the cause; and when session creation fails, `InferenceSession`
  prints an "EP Error ... Falling back" banner (the script captures it) and
  retries on the CPU, whose error is raised from the original one, so the script
  classifies the whole exception chain.

The JSON file (`--json`) records the mode as `precision` at the top level, and
each target's `precision`, `max_tolerance` and `mean_tolerance` (`null` when
there is no mean bound) next to its status, timings and errors.

Exit status: `0` when no available target failed and every `--require`d target
is `ok`; `1` otherwise (also when the CPU provider cannot load the model); `2`
for a usage error or an input the script cannot build data for; `3` when
`onnxruntime` or `numpy` cannot be imported. `--help` works without either.

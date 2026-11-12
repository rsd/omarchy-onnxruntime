# ort-smoke

A small smoke test for an installed ONNX Runtime. It lists the execution
providers the library offers, then runs a tiny model on every engine and device
it can find and checks the numbers.

## What it checks

For each target, the program creates a session with only that execution
provider, runs the model once (the "first" run) and then a few more times
("warm"), and compares every output element against a reference computed in
plain C.

| Target | Execution provider | Device |
|---|---|---|
| `cpu` | CPU | host CPU |
| `cuda` | CUDA | NVIDIA GPU 0 |
| `tensorrt` | TensorRT (engine cache off) | NVIDIA GPU 0 |
| `openvino:CPU` | OpenVINO | Intel CPU plugin |
| `openvino:GPU` | OpenVINO | Intel GPU plugin |
| `openvino:NPU` | OpenVINO | Intel NPU plugin |

For every target except `cpu`, the session option
`session.disable_cpu_ep_fallback=1` is set. A provider that registers but cannot
take the graph therefore fails loudly instead of quietly running on the CPU and
reporting success.

The model (`make-model.py`) is `Y = Add(Relu(Conv3x3(X, W, B)), A)` with
`X` of shape `[1, 3, 16, 16]` and `Y` of shape `[1, 4, 16, 16]`. All shapes are
static, which some accelerators (NPUs) require. Weights and the input are
multiples of 1/8, so every intermediate value is exact in fp16 as well as fp32;
a correct engine reproduces the reference exactly, and the per-target tolerance
only leaves room for reduced-precision paths (see the comment above the target
table in `smoke.c`).

The model is generated with the Python standard library only (the `onnx`
package is not needed), and the bytes are deterministic.

## Running it

Through the wrapper, which builds the program and generates the model in a
temporary directory:

    scripts/smoke-test.sh                       # installed package, all targets
    scripts/smoke-test.sh --require cpu,cuda    # exit 1 unless these are ok
    scripts/smoke-test.sh --only openvino       # the three OpenVINO devices
    scripts/smoke-test.sh --prefix /opt/ort     # a different install prefix

Against a build tree instead of an installed package:

    scripts/smoke-test.sh \
        --include <source>/include/onnxruntime \
        --libdir  <source>/build/Linux/Release

Options:

| Option | Meaning |
|---|---|
| `--prefix DIR` | install prefix, default `/usr` (headers in `DIR/include/onnxruntime`, libraries in `DIR/lib`) |
| `--include DIR` | directory with `onnxruntime_c_api.h`; a source tree's `include/onnxruntime` is accepted |
| `--libdir DIR` | directory with `libonnxruntime.so` |
| `--only LIST` | test only these targets (comma-separated) |
| `--require LIST` | exit 1 unless these targets are `ok` |
| `--runs N` | warm runs per target, default 5 |
| `--keep` | keep the temporary directory and print its path |

Needs a C compiler (`CC`, default `cc`) and Python 3 (`PYTHON`, default
`python3`).

To build by hand:

    python3 make-model.py model.onnx
    cc -std=c11 smoke.c -I/usr/include/onnxruntime -lonnxruntime -lm -o smoke
    ./smoke model.onnx --require cpu

## Reading the output

    ONNX Runtime 1.31.1 (API 31, built against headers for API 31)
    available providers: TensorrtExecutionProvider CUDAExecutionProvider ... CPUExecutionProvider

    target         status        create ms   first ms    warm ms  max error
    cpu            ok                  4.1       0.30       0.05   0.00e+00
    cuda           ok                180.2       2.10       0.09   0.00e+00
    openvino:NPU   unavailable           -          -          -          -
        note: ...

- **ok** - the session was created, ran, and every output element is within
  tolerance. `create ms` includes provider initialisation and, for TensorRT,
  engine building (seconds are normal); `first ms` includes lazy initialisation;
  `warm ms` is the mean of the later runs; `max error` is the largest absolute
  difference from the reference.
- **FAIL** - the provider is available and loaded, but session creation, the run
  or the numbers failed. The ORT error message follows on the next line.
- **unavailable** - the engine cannot be used on this machine. This is not a
  failure, unless the target was named in `--require`. Three causes:
  - the provider is not in the available-providers list;
  - the provider's vendor runtime is not installed. The package compiles every
    provider in, so `available providers` lists CUDA, TensorRT and OpenVINO even
    when their runtimes (optional dependencies) are missing. ORT then fails the
    provider's setup with a dlopen error, and the note reads
    `vendor runtime not installed: libcudnn.so.9`. Only a "cannot open shared
    object file" error from loading a provider counts; other load errors (an
    undefined symbol, say) stay `FAIL`;
  - OpenVINO says `Device GPU is not available` (or `CPU`/`NPU`). The match is
    that exact message from the OpenVINO EP; any other OpenVINO error is `FAIL`.

  A failure after the provider loaded fine (creating the session, running, or
  wrong numbers) is always `FAIL`.

Exit status of the program: `0` when no available target failed and every
`--require`d target is `ok`; `1` otherwise; `2` for a usage error; `3` when the
ONNX Runtime API cannot be initialised. The wrapper adds `3` for a missing
ONNX Runtime installation, compiler or Python.

### TensorRT and its plugin library

Whether TensorRT's default plugin library loaded is not observable through the C
API, so the report cannot show it. With TensorRT installed under
`/opt/nvidia/tensorrt`, ORT loads `libnvinfer_plugin.so.11` with `dlopen` from
inside `libonnxruntime.so`, which has no RUNPATH, so the built-in TensorRT
plugins are found only if the TensorRT environment file
`/usr/share/nvidia-tensorrt/nvidia-tensorrt.env` (from the `nvidia-tensorrt`
package) is sourced or `LD_LIBRARY_PATH` includes the TensorRT `lib` directory.
The tiny model in this test needs no plugin, so a `tensorrt` `ok` says nothing
about them. Unverified on hardware.

Only the C API is used. The program builds against headers from ONNX Runtime
1.17 or newer, and at run time it uses the newest API version both the header
and the library support.

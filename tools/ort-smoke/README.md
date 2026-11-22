# ort-smoke

`ort-smoke` is a small smoke test for an installed ONNX Runtime. It lists the execution providers the library offers, then runs a tiny model on every engine and device it can find and checks the numbers. Run it after installing the packages to confirm that each engine works. Install the packages first: [INSTALL.md](../../INSTALL.md).

## What it checks

For each target, the program creates a session with only that execution provider. It runs the model once (the "first" run) and then a few more times ("warm"). It compares every output element against a reference computed in plain C.

| Target | Execution provider | Device |
|---|---|---|
| `cpu` | CPU | host CPU |
| `cuda` | CUDA | NVIDIA GPU 0 |
| `tensorrt` | TensorRT (engine cache off) | NVIDIA GPU 0 |
| `openvino:CPU` | OpenVINO | Intel CPU plugin |
| `openvino:GPU` | OpenVINO | Intel GPU plugin |
| `openvino:NPU` | OpenVINO | Intel NPU plugin |
| `dnnl` | oneDNN (default options) | host CPU |
| `migraphx-load` | MIGraphX, device 0 (not in the default set, see below) | AMD GPU |

For every target except `cpu`, the program sets the session option `session.disable_cpu_ep_fallback=1`. A provider that registers but cannot take the graph therefore fails loudly. It does not run quietly on the CPU and report success.

The model (`make-model.py`) is `Y = Add(Relu(Conv3x3(X, W, B)), A)` with `X` of shape `[1, 3, 16, 16]` and `Y` of shape `[1, 4, 16, 16]`. All shapes are static, which some accelerators (NPUs) require. Weights and the input are multiples of 1/8, so every intermediate value is exact in fp16 as well as fp32. A correct engine reproduces the reference exactly. The per-target tolerance only leaves room for reduced-precision paths (see the comment above the target table in `smoke.c`).

The model is generated with the Python standard library only (the `onnx` package is not needed), and the bytes are deterministic.

## Running it

Run it through the wrapper, which builds the program and generates the model in a temporary directory. It needs a C compiler (`CC`, default `cc`) and Python 3 (`PYTHON`, default `python3`).

```bash
scripts/smoke-test.sh                       # installed package, all targets
scripts/smoke-test.sh --require cpu,cuda    # exit 1 unless these are ok
scripts/smoke-test.sh --only openvino       # the three OpenVINO devices
scripts/smoke-test.sh --only cpu,dnnl       # the oneDNN provider
scripts/smoke-test.sh --only migraphx-load  # the MIGraphX provider loads
scripts/smoke-test.sh --prefix /opt/ort     # a different install prefix
```

To test a build tree instead of an installed package:

```bash
scripts/smoke-test.sh \
    --include <source>/include/onnxruntime \
    --libdir  <source>/build/Linux/Release
```

Options:

| Option | Meaning |
|---|---|
| `--prefix DIR` | install prefix, default `/usr` (headers in `DIR/include/onnxruntime`, libraries in `DIR/lib`) |
| `--include DIR` | directory with `onnxruntime_c_api.h`; a source tree's `include/onnxruntime` is accepted |
| `--libdir DIR` | directory with `libonnxruntime.so` |
| `--only LIST` | test only these targets (comma-separated); default: all but `migraphx-load` |
| `--require LIST` | exit 1 unless these targets are `ok` (`ok (no device)` counts) |
| `--runs N` | warm runs per target, default 5 |
| `--keep` | keep the temporary directory and print its path |

To build by hand:

```bash
python3 make-model.py model.onnx
cc -std=c11 smoke.c -I/usr/include/onnxruntime -lonnxruntime -lm -o smoke
./smoke model.onnx --require cpu
```

## Reading the output

```
ONNX Runtime 1.31.1 (API 31, built against headers for API 31)
available providers: TensorrtExecutionProvider CUDAExecutionProvider ... CPUExecutionProvider

target         status           create ms   first ms    warm ms  max error
cpu            ok                     4.1       0.30       0.05   0.00e+00
cuda           ok                   180.2       2.10       0.09   0.00e+00
openvino:NPU   unavailable              -          -          -          -
    note: ...
migraphx-load  ok (no device)           -          -          -          -
    note: no device: ...
```

- **ok**: the session was created, ran, and every output element is within tolerance.
  - `create ms` includes provider initialisation and, for TensorRT, engine building (seconds are normal).
  - `first ms` includes lazy initialisation.
  - `warm ms` is the mean of the later runs.
  - `max error` is the largest absolute difference from the reference.
- **FAIL**: the provider is available and loaded, but session creation, the run or the numbers failed. The ONNX Runtime error message follows on the next line.
- **unavailable**: the engine cannot be used on this machine. This is not a failure, unless you named the target in `--require`. There are three causes:
  - The provider is not in the available-providers list. This does not apply to `migraphx-load`: the core never lists MIGraphX, so the provider library on disk decides (see below).
  - The provider's library or its vendor runtime is not installed. The list is fixed when ONNX Runtime is built, so `available providers` lists CUDA, TensorRT and OpenVINO even when their provider package is not installed (`onnxruntime-provider-cuda<M>-<sfx>`, `onnxruntime-provider-tensorrt-cuda<M>`, `onnxruntime-provider-openvino`), or when the runtime is missing. A missing runtime is possible with a build tree or `--nodeps`, because an installed provider package depends on its runtimes. ONNX Runtime then fails the provider's setup with a dlopen error, and the note reads `vendor runtime not installed: libcudnn.so.9`. Only a "cannot open shared object file" error from loading a provider counts. Other load errors (an undefined symbol, say) stay `FAIL`.
  - OpenVINO says `Device GPU is not available` (or `CPU` or `NPU`). The match is that exact message from the OpenVINO EP. Any other OpenVINO error is `FAIL`.

  For `dnnl` and `migraphx-load`, a third form counts too: the provider library itself (`libonnxruntime_providers_dnnl.so`, `libonnxruntime_providers_migraphx.so`; separate packages) is absent. The output shows `provider library not installed`. For the other providers that is a packaging fault, a `FAIL`.

  A failure after the provider loaded fine (creating the session, running, or wrong numbers) is always `FAIL`.

Exit status of the program:

- `0` when no available target failed and every `--require`d target is `ok`.
- `1` otherwise.
- `2` for a usage error.
- `3` when the ONNX Runtime API cannot be initialised.

The wrapper adds `3` for a missing ONNX Runtime installation, compiler or Python. The wrapper also exits `1`, with a message naming the signal, when the program is killed by a signal (a crash) instead of exiting on its own.

### The two newest targets

`dnnl` runs the model on the oneDNN provider like any other engine. The program appends the provider with default options, turns the CPU fallback off, and checks the numbers (tolerance 1e-3). The target is `unavailable` when the provider is not in the list, its library is not installed, or `libdnnl` is missing.

`migraphx-load` exists because the MIGraphX provider needs an AMD GPU, which the maintainer's machines do not have. It appends the provider by name (`SessionOptionsAppendExecutionProvider(..., "MIGraphX", ...)`, device 0) and creates the session. For the name to be known, the core must be built with ONNX Runtime's MIGraphX provider-interface option. The core does not list MIGraphX in `GetAvailableProviders`, so the program does not consult that list for this target. Its results:

- `ok (no device)`: the provider's device-ID check failed the way a machine without an AMD GPU makes it fail. The message is printed on the next line (`note: no device: ...`). That is the pass: the provider library loaded and failed cleanly, with no crash and no signal. Only these signatures count (from `migraphx_execution_provider_info.cc`, `hipGetDeviceCount`):
  - `HIP failure 100:` (hipErrorNoDevice) together with `hipGetDeviceCount` (the failing call named in the message);
  - `hipErrorNoDevice`;
  - `no ROCm-capable device`;
  - `Invalid device ID: 0, ... and 0 (exclusive)`.

  A `HIP failure` with another code (for example 35, `hipErrorInsufficientDriver`, a broken driver) or of another call (`hipMalloc`, `hipSetDevice`, ...) is a `FAIL`.
- `FAIL`: any other error, with the message shown. Examples are `not supported in this build` (a core without MIGraphX), `Unknown provider name` (a library too old for this call), `undefined symbol` (soname or ABI drift between the provider and the core) and a session error. On a machine that does have an AMD GPU, `ok (no device)` means something is wrong: read the message.
- `ok`: a device works. The model ran and the numbers are within 1e-2.
- `unavailable`: `libonnxruntime_providers_migraphx.so` is not installed next to `libonnxruntime.so`, or `libmigraphx_c`, `libamdhip64` or `libhsa` is missing.
- A crash: the program dies from a signal. The results printed before that stay visible (the output is flushed after every target), and the wrapper exits 1 and names the signal.

`migraphx-load` is not in the default run. A HIP runtime without a usable device may abort the process, and that must not turn a default run into a failure. It runs only when you name it in `--only` or `--require` (`--require migraphx-load` alone also runs it, together with the default set). With `--only`, the default set does not apply at all.

### Process exit with OpenVINO

In this test, releasing the ONNX Runtime environment (`ReleaseEnv`) after the OpenVINO execution provider had been used crashed the process at exit (`SIGSEGV`) with ONNX Runtime 1.29.0 and OpenVINO 2026.4. Every target had reported `ok`. Releasing the environment makes ONNX Runtime unload the OpenVINO provider library and, with it, OpenVINO's CPU plugin. The crash follows when OpenVINO or TBB threads, or thread-local destructors, run in code that is no longer loaded. Preloading a `dlclose` shim that skipped only those libraries made the exit clean. The cause inside OpenVINO was not investigated.

C API users should keep the environment for the life of the process and not release it at exit. `smoke.c` does this. It still releases its sessions, options, values and memory info. The Python API is not affected, since it does not release its environment.

### TensorRT and its plugin library

The C API cannot show whether TensorRT's default plugin library loaded, so the report cannot show it. With TensorRT installed under `/opt/nvidia/tensorrt`, ONNX Runtime loads `libnvinfer_plugin.so.11` with `dlopen` from inside `libonnxruntime.so`, whose RUNPATH is only `$ORIGIN` (`/usr/lib`). With `nvidia-tensorrt-sdk-local` pkgrel 4 or later, its loader entry makes them findable. Otherwise the built-in TensorRT plugins are found only if you source the TensorRT environment file `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env` (from the `nvidia-tensorrt` package) or `LD_LIBRARY_PATH` includes the TensorRT `lib` directory. The tiny model in this test needs no plugin, so a `tensorrt` `ok` says nothing about them. This is unverified on hardware.

Only the C API is used. The program builds against headers from ONNX Runtime 1.17 or newer. At run time it uses the newest API version that both the header and the library support.

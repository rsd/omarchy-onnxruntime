# Glossary

The terms the other pages use, each in one or two sentences.

## ONNX Runtime

| Term | Meaning |
|---|---|
| **ONNX Runtime** | Microsoft's open-source engine that runs models in the ONNX format. |
| **Execution provider (EP, "engine")** | The part of ONNX Runtime that runs a model on one kind of hardware: CPU, CUDA, TensorRT, OpenVINO, oneDNN or MIGraphX. A session asks for the providers it wants. |
| **Core library** | `libonnxruntime.so`. It holds the CPU provider and loads the other providers when a session asks for them. |
| **Provider library** | A shared library `libonnxruntime_providers_<provider>.so` that holds one non-CPU provider. Here each one is in its own package. |
| **Vendor runtime** | The libraries a provider needs from the hardware vendor: CUDA, cuDNN and TensorRT from NVIDIA, OpenVINO from Intel, MIGraphX and the HIP runtime from AMD. They are installed from their own packages and are never part of the packages of this repository. |
| **oneDNN (`dnnl`)** | Intel's open-source deep-learning library for CPUs. ONNX Runtime has an execution provider for it, `DnnlExecutionProvider`, which `onnxruntime-provider-dnnl` carries with oneDNN linked in statically. |
| **Training operators** | The operators and the Python modules (`onnxruntime.training`) that ONNX Runtime builds with `onnxruntime_ENABLE_TRAINING`. They are in the core of every line and change the core's interface to the provider libraries (see *provider interface*). The training Python modules need PyTorch. |
| **NCCL, RCCL** | NVIDIA's and AMD's libraries for communication between several GPUs, used by the training operators. `onnxruntime-core` depends on Arch's `nccl` by default. A build with `ORTF_NCCL=0` omits it and has `-nccloff` package names. |
| **Provider interface (the plug)** | The set of functions through which the core library and a provider library call each other. Training and NCCL add members to it; nothing checks at run time that both sides have the same members, so a mismatch crashes. It is why the core and the CUDA 12 and MIGraphX providers are released together. |
| **Session** | One loaded model in ONNX Runtime, bound to a list of providers. |
| **Static shape** | A model whose input sizes are fixed. Some accelerators (NPUs among them) need this. |

## NVIDIA GPUs

| Term | Meaning |
|---|---|
| **Compute capability, SM** | The architecture version of an NVIDIA GPU, written without the dot: `120` (or `sm_120`) is compute capability 12.0, the RTX 50 series. `75` is Turing (RTX 20, GTX 16), `86` and `89` are Ampere and Ada (RTX 30 and 40), `60` and `61` are Pascal, `70` is Volta. "SM" stands for streaming multiprocessor, the unit that the architecture version describes. |
| **SASS** | Machine code for one GPU architecture. It runs only on GPUs of that architecture. |
| **PTX** | NVIDIA's intermediate code. The driver compiles it ("JIT") for a GPU that has no SASS in the library. A build with a `-virtual` entry embeds it; the presets `rtx` and `all` do not. |
| **JIT** | Just-in-time compilation, here by the GPU driver the first time code runs. |
| **Preset** | A named list of CUDA architectures for `ORTF_CUDA_ARCHS`: `rtx`, `all`, `legacy`. |
| **TF32, fp16, fp32** | Arithmetic precisions. fp32 is 32-bit floating point, fp16 is 16-bit, and TF32 is NVIDIA's reduced-precision mode for matrix and convolution maths (a 10-bit mantissa, fp32 range). CUDA and TensorRT use TF32 by default on GPUs that have it. |
| **CUDA 13 line, CUDA 12 line** | The packages built with CUDA 13 (the main recipe) and the one package built with CUDA 12.9 for older cards (`onnxruntime-provider-cuda12-legacy`). |
| **MIGraphX line** | The provider `onnxruntime-provider-migraphx` for AMD GPUs and its Python add-on `python-onnxruntime-provider-migraphx`, built from their own recipe `onnxruntime-split-migraphx`, source only and untested on AMD hardware. |

## Arch packaging

| Term | Meaning |
|---|---|
| **PKGBUILD, recipe** | The file `makepkg` reads to build packages. "The recipe" here is `onnxruntime-split/PKGBUILD`. |
| **pkgbase, pkgname** | A PKGBUILD can build several packages. The pkgbase names the PKGBUILD (`onnxruntime-split`); the pkgnames are its packages (`onnxruntime-core`, ...). |
| **pkgver, pkgrel** | The upstream version (`1.29.0`) and the packaging release number (`7`). The package version is `pkgver-pkgrel`, for example `1.29.0-7`. A `pkgrel` bump rebuilds the same upstream version with a packaging change. |
| **`-nccloff` package** | A package of a build with `ORTF_NCCL=0`: every name has the suffix, it provides and conflicts with its base name, and it never installs beside the NCCL packages. Local builds only; none is prebuilt. |
| **ABI pkgrel** | The first `pkgrel` of a `pkgver` whose core has the provider interface that the CUDA 12 and MIGraphX providers are built for (7 for 1.29.0). Those providers depend on `onnxruntime-core>=<pkgver>-<that>` and `onnxruntime-core<` the next patch release. It resets to 1 with a new `pkgver`. |
| **`-bin` package** | A prebuilt package on the AUR. Its PKGBUILD downloads the already-built package files of a GitHub release instead of compiling. Here `<name>-bin` provides and conflicts with `<name>`. |
| **AUR, AUR helper** | The Arch User Repository, and a tool such as `yay` that fetches and builds AUR packages and installs their dependencies. |
| **depends, makedepends, optdepends** | Packages needed to run, to build, and optionally for extra features. |
| **provides, conflicts, replaces** | A package can offer another package's name (`provides`), refuse to be installed beside it (`conflicts`), and take over its upgrade (`replaces`, acted on only by a sync upgrade). |
| **`.SRCINFO`** | The metadata of a PKGBUILD in a form the AUR reads. `scripts/srcinfo.sh` generates it. |
| **Clean chroot** | A throw-away minimal Arch system (made with the `devtools` package) in which a package is built, so that only the declared `makedepends` are present and the result does not depend on the builder's installed packages. |
| **`SRCDEST`, `BUILDDIR`, `PKGDEST`** | `makepkg` settings: where downloaded sources go, where the build tree goes, and where finished packages are written. |

## Libraries and the dynamic loader

| Term | Meaning |
|---|---|
| **soname** | The name a shared library carries inside itself, such as `libcudart.so.13`. A program records the sonames it needs. |
| **`DT_NEEDED`** | The entries in an ELF binary that list the sonames it links. `scripts/check-deps.sh` and the recipe's `package()` read them. |
| **`dlopen()`** | Loading a library by name while the program runs. Libraries loaded this way do not appear in `DT_NEEDED`. Earlier CUDA provider builds did this for cuDNN; since 2026-11-19 the provider links `libcudnn.so.9` directly. |
| **RUNPATH** | A list of directories stored in a binary where the loader looks for the libraries that binary needs. `$ORIGIN` means the directory of the binary itself. |
| **ABI** | The binary interface between a library and its users. Where it can change with a vendor's minor version (TensorRT), the dependency is bounded to that version. |
| **MLAS** | ONNX Runtime's CPU math library. It picks its AVX2 or AVX-512 kernels at run time. |

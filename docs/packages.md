# Packages

What the recipe builds, what each package holds and depends on, how the names are formed, and why the packages are split by engine. Terms are in the [glossary](glossary.md).

## The packages

The recipe `onnxruntime-split/PKGBUILD` is one build of the upstream source tag `v1.29.0` (pkgbase `onnxruntime-split`), split into one package per engine:

| Package | Contents | Depends on |
|---|---|---|
| `onnxruntime-core` | `libonnxruntime.so` with the CPU provider, `libonnxruntime_providers_shared.so` (in a build with any other provider), the C and C++ headers (`/usr/include/onnxruntime`), `libonnxruntime.pc` and the CMake files. It works on its own, on the CPU. Provides `onnxruntime=1.29.0` and `libonnxruntime.so` | `glibc`, `libgcc`, `libstdc++` |
| `onnxruntime-provider-cuda<M>-<sfx>` | The CUDA provider, `libonnxruntime_providers_cuda.so`. `<M>` is the major version of the CUDA toolkit the build used (`13` today); `<sfx>` names the GPU architectures it was compiled for ([CUDA architectures](build.md#cuda-architectures)). Provides `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>` | `onnxruntime-core`, `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-tensorrt-cuda<M>` | The TensorRT provider. It has no architecture suffix because it holds no GPU code: TensorRT builds its kernels at run time. Provides `onnxruntime-provider-tensorrt` | `onnxruntime-core`, `onnxruntime-provider-cuda<M>`, `cuda`, `cudnn`, `nvidia-utils`, `tensorrt` (bounded to the major.minor of the TensorRT headers the build used) |
| `onnxruntime-provider-openvino` | The OpenVINO provider (Intel CPU, GPU and NPU devices) | `onnxruntime-core`, `openvino` |
| `onnxruntime-provider-migraphx` | The MIGraphX provider (AMD GPUs), built only with `migraphx` in `ORTF_EPS`. **Untested on AMD hardware.** It holds no GPU code: MIGraphX compiles the model for the GPU it finds at run time. The name carries no ROCm version, because ROCm 7.2 and 10.1 keep the same sonames. See [amd.md](amd.md) | `onnxruntime-core`, `migraphx`, `hip-runtime-amd` (bounded to the major version of the installed packages) |
| `python-onnxruntime-core` | The Python module. Its links to the provider libraries dangle until that provider's package is installed. Provides `python-onnxruntime` | `onnxruntime-core`, `python` and its packages |

All dependencies also include the system libraries the package links (`glibc`, `libgcc`, `libstdc++`). The providers and the Python module depend on `onnxruntime-core` at the same `pkgver-pkgrel`. The full dependency list with version bounds is in [dependencies.md](dependencies.md).

The CUDA 12 line adds one more package, `onnxruntime-provider-cuda12-legacy`, from a separate recipe: [cuda12-legacy.md](cuda12-legacy.md).

`onnxruntime-core` and `python-onnxruntime-core` replace and conflict with `onnxruntime-full` and `python-onnxruntime-full`, the single packages of 1.29.0-1 ([upgrading](install.md#upgrading-from-the-single-package-1290-1)). `onnxruntime-core` also conflicts with Arch's `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm` and `onnxruntime-git`, and provides `onnxruntime` and `libonnxruntime.so`, so programs built against Arch's package keep working: pacman offers to remove the conflicting package.

## Which packages a build makes

The names depend on the build inputs `ORTF_EPS`, `ORTF_PYTHON` and `ORTF_CUDA_ARCHS` and on the CUDA toolkit installed when `makepkg` reads the PKGBUILD:

- **The default build with Arch's CUDA 13** makes `onnxruntime-core`, `onnxruntime-provider-cuda13-rtx`, `onnxruntime-provider-tensorrt-cuda13`, `onnxruntime-provider-openvino` and `python-onnxruntime-core`.
- `ORTF_EPS=cpu` makes `onnxruntime-core` (and the Python package) only. `migraphx` in `ORTF_EPS` adds `onnxruntime-provider-migraphx`.
- The CUDA package's suffix is the preset name (`rtx`, `all`) or, for a list, the architectures (`sm120`, `sm89-sm120`).
- Where the CUDA toolkit's `cuda.h` is not installed yet when `makepkg` reads the PKGBUILD (an AUR helper reads it before it installs the build dependencies), the names use CUDA major 13. A CUDA 12 toolkit gives `-cuda12` names. The CUDA 12 package that is released comes from its own recipe, not from this one with other inputs.

The rules, with the table of which input decides which name, are in [build.md](build.md#package-names).

## Packages of different builds

CUDA packages of different builds provide and conflict with each other (`onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`), because they all install the same `/usr/lib/libonnxruntime_providers_cuda.so`. One is installed at a time. The TensorRT package depends on the name `onnxruntime-provider-cuda<M>`, which every CUDA package of that major provides.

## Sizes

Measured on the files of the 1.29.0-6 build (download: the `.pkg.tar.zst` file; installed: the `size` field of the package's `.PKGINFO`):

| Package | Download | Installed |
|---|---|---|
| `onnxruntime-core` | 7.9 MiB | 29.4 MiB |
| `onnxruntime-provider-cuda13-rtx` | 99.2 MiB | 154.7 MiB |
| `onnxruntime-provider-cuda13-all` | 226.9 MiB | 294.9 MiB |
| `onnxruntime-provider-tensorrt-cuda13` | 0.3 MiB | 1.2 MiB |
| `onnxruntime-provider-openvino` | 0.5 MiB | 1.7 MiB |
| `python-onnxruntime-core` | 10.0 MiB | 43.2 MiB |
| `onnxruntime-provider-cuda12-legacy` (1.29.0-1) | 172.6 MiB | 557.5 MiB |

The CUDA provider grows with the number of GPU architectures it holds; see [build.md](build.md#cuda-architectures).

## Why not Arch's onnxruntime packages

Arch's `extra` repository ships ONNX Runtime 1.29.0 (`pacman -Si onnxruntime-cpu` shows the current package release) as separate builds: `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` (AVX2 build), `onnxruntime-rocm` and `onnxruntime-opt-rocm`, each with a `python-` counterpart. Each build provides `onnxruntime` and conflicts with it, so one is installed at a time. This recipe exists for the combination none of them offers:

| | Arch `extra` builds | This recipe (`onnxruntime-core` and the provider packages) |
|---|---|---|
| CPU, CUDA | CUDA builds include the CPU provider | yes |
| TensorRT, OpenVINO (CPU, Intel GPU, Intel NPU) | not enabled in the recipe | yes, in one library |
| NVIDIA and Intel engines in one install | no: CUDA and ROCm are separate packages, and the CUDA builds add oneDNN (`dnnl`) as the other provider | yes |
| AMD GPUs (MIGraphX) | `onnxruntime-rocm` and `onnxruntime-opt-rocm` carry the MIGraphX provider | `onnxruntime-provider-migraphx`, on request; **untested on AMD hardware** |
| CUDA architectures | a list from `75` to `121`, including `120` (Blackwell) | the preset `rtx` (`75 86 89 120`) by default; the preset `all` or a list, set with `ORTF_CUDA_ARCHS` ([presets](build.md#cuda-architectures)) |

Support for `sm_120` is therefore not the difference: the Arch CUDA builds include it. The difference is the TensorRT and OpenVINO providers, which the Arch recipe does not build, and having the NVIDIA and Intel engines in one installation. The version is the same, so programs built against Arch's package see the same API. Source: the Arch recipe at <https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime> and `pacman -Si`.

## Why one package per engine, not one per build

Arch splits per build: each variant is a separate compilation of the whole library, the variants conflict, and one is installed at a time. Here the providers are separate shared objects (`libonnxruntime_providers_<provider>.so`) that one `libonnxruntime.so` loads at run time, so a single build serves every engine. The split therefore follows the engines: each package carries only its provider library, the vendor runtimes that library needs are its dependencies, and the core package runs without any of them. A machine installs the engines it has and nothing else, for example the core and the OpenVINO package on a machine without an NVIDIA GPU. How the loading works is in [engines.md](engines.md#how-the-providers-load).

# Packages

This page lists what the recipe builds, what each package holds and depends on, how the names are formed, and why the packages are split by engine. Terms are in the [glossary](glossary.md). [choosing.md](choosing.md) compares this set with Arch's `onnxruntime-*` packages.

## The packages

The recipe `onnxruntime-split/PKGBUILD` is one build of the upstream source tag `v1.29.0` (pkgbase `onnxruntime-split`), split into one package per engine. By default the source is this project's mirror of that tag ([build.md](build.md#the-onnx-runtime-source-mirror-or-upstream)).

| Package | Contents | Depends on |
|---|---|---|
| `onnxruntime-core` | `libonnxruntime.so` with the CPU provider and the training operators, `libonnxruntime_providers_shared.so` (in a build with any other provider), the C and C++ headers (`/usr/include/onnxruntime`), `libonnxruntime.pc` and the CMake files. It works on its own, on the CPU. Provides `onnxruntime=1.29.0` and `libonnxruntime.so` | `glibc`, `libgcc`, `libstdc++`, `nccl` |
| `onnxruntime-provider-cuda<M>-<sfx>` | The CUDA provider, `libonnxruntime_providers_cuda.so`. `<M>` is the major version of the CUDA toolkit the build used (`13` today). `<sfx>` names the GPU architectures it was compiled for ([CUDA architectures](build.md#cuda-architectures)). Provides `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>` | `onnxruntime-core`, `cuda`, `cudnn`, `nvidia-utils`, `nccl` |
| `onnxruntime-provider-tensorrt-cuda<M>` | The TensorRT provider. It has no architecture suffix because it holds no GPU code: TensorRT builds its kernels at run time. Provides `onnxruntime-provider-tensorrt` | `onnxruntime-core`, `onnxruntime-provider-cuda<M>`, `cuda`, `cudnn`, `nvidia-utils`, `tensorrt` (bounded to the major.minor of the TensorRT headers the build used) |
| `onnxruntime-provider-openvino` | The OpenVINO provider (Intel CPU, GPU and NPU devices) | `onnxruntime-core`, `openvino` |
| `onnxruntime-provider-dnnl` | The oneDNN provider (CPU). It is optional and meant for Intel server CPUs, not for most desktops and laptops (use the CPU provider or OpenVINO there). oneDNN 3.11.3 is built from source and linked into the provider statically, so the package depends on no oneDNN package and installs no `libdnnl.so`. It carries oneDNN's licence and third-party notices (`onednn-LICENSE`, `onednn-THIRD-PARTY-PROGRAMS`). See [engines.md](engines.md#onednn) | `onnxruntime-core`, `libgomp` |
| `python-onnxruntime-core` | The Python module, with `onnxruntime.training`. Its links to the provider libraries dangle until that provider's package is installed. Provides `python-onnxruntime` | `onnxruntime-core`, `nccl`, `python` and its packages, and for the training modules `python-cerberus`, `python-h5py`, `python-onnx`, `python-sympy` and `python-setuptools` |
| `python-onnxruntime-provider-migraphx` | The Python add-on of the MIGraphX line. It comes from the pkgbase `onnxruntime-split-migraphx`, not from this recipe. It holds one symbolic link, `onnxruntime/capi/libonnxruntime_providers_migraphx.so` in the system Python's site-packages, to the provider's library, so that the Python module lists `MIGraphXExecutionProvider`. Source only, untested on AMD hardware ([amd.md](amd.md#the-package)) | `onnxruntime-provider-migraphx` at the same version, `python-onnxruntime-core`, `python` (bounded to its minor) |

Every package also depends on the system libraries it links (`glibc`, `libgcc`, `libstdc++`). The providers and the Python module depend on `onnxruntime-core` at the same `pkgver-pkgrel`. [dependencies.md](dependencies.md) has the full dependency list with version bounds.

`onnxruntime-core` and `python-onnxruntime-core` list the engine packages as optional dependencies. They always list `onnxruntime-provider-migraphx`, which comes from another pkgbase. The Python package also lists `python-pytorch`, which the training modules need. Even `import onnxruntime.training` imports it. `import onnxruntime` needs none of it.

Two more packages come from separate recipes, each with its own `pkgrel`:

- **`onnxruntime-provider-cuda12-legacy`**, the CUDA provider built with CUDA 12.9 for Pascal and Volta (pkgbase `onnxruntime-split-cuda12`): [cuda12-legacy.md](cuda12-legacy.md).
- **`onnxruntime-provider-migraphx`**, the MIGraphX provider for AMD GPUs (pkgbase `onnxruntime-split-migraphx`). It is source only and **untested on AMD hardware**. Its Python add-on `python-onnxruntime-provider-migraphx` is in the table above: [amd.md](amd.md).

Both load into the core of this recipe and depend on `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`. [The joint release rule](#the-core-and-the-other-lines-providers-release-together) explains the pair. The MIGraphX add-on depends on its provider at the same version, not on the core.

`onnxruntime-core` and `python-onnxruntime-core` replace and conflict with `onnxruntime-full` and `python-onnxruntime-full`, the single packages of 1.29.0-1 ([upgrading](../INSTALL.md#upgrade-from-the-single-package-1290-1)). `onnxruntime-core` also conflicts with Arch's `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda`, `onnxruntime-rocm`, `onnxruntime-opt-rocm` and `onnxruntime-git`. It provides `onnxruntime` and `libonnxruntime.so`, so programs built against Arch's package keep working. Pacman offers to remove the conflicting package ([choosing.md](choosing.md#why-one-set-per-machine)). `onnxruntime-core` conflicts with `onnxruntime-provider-cuda12-legacy` and `onnxruntime-provider-cuda12-legacy-bin` below `1.29.0-2`.

## Which packages a build makes

The names depend on the build inputs `ORTF_EPS`, `ORTF_PYTHON`, `ORTF_CUDA_ARCHS` and `ORTF_NCCL`, and on the CUDA toolkit installed when `makepkg` reads the PKGBUILD:

- **The default build with Arch's CUDA 13** makes `onnxruntime-core`, `onnxruntime-provider-cuda13-rtx`, `onnxruntime-provider-tensorrt-cuda13`, `onnxruntime-provider-openvino`, `onnxruntime-provider-dnnl` and `python-onnxruntime-core`.
- **A build without `cuda` in `ORTF_EPS`** needs `ORTF_NCCL=0` and makes `-nccloff` packages ([below](#packages-without-nccl)). `ORTF_EPS=cpu ORTF_NCCL=0` makes `onnxruntime-core-nccloff` (and the Python package) only. `dnnl` in `ORTF_EPS` adds `onnxruntime-provider-dnnl`. `migraphx` adds `onnxruntime-provider-migraphx`, for a local build. The published MIGraphX packages come from their own pkgbase.
- **The CUDA package's suffix** is the preset name (`rtx`, `all`) or, for a list, the architectures (`sm120`, `sm89-sm120`).
- **Without `cuda.h`.** When the CUDA toolkit's `cuda.h` is not installed yet when `makepkg` reads the PKGBUILD (an AUR helper reads it before it installs the build dependencies), the names use CUDA major 13. A CUDA 12 toolkit gives `-cuda12` names. The CUDA 12 package that is released comes from its own recipe, not from this one with other inputs.

The rules, with the table of which input decides which name, are in [build.md](build.md#package-names).

## Packages of different builds

CUDA packages of different builds provide and conflict with each other (`onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`), because they all install the same `/usr/lib/libonnxruntime_providers_cuda.so`. Install one at a time. The TensorRT package depends on the name `onnxruntime-provider-cuda<M>`, which every CUDA package of that major provides.

## The core and the other lines' providers release together

Training and NCCL change the plug between `onnxruntime-core` and a provider library. The provider interface the core exposes has members that exist only in a build with them. Nothing checks at run time that both sides agree, so a provider built without them crashes in a core built with them.

Hence the rule:

- The main line (1.29.0-7) and the CUDA 12 line (1.29.0-2) are built and released together.
- The MIGraphX line (1.29.0-1) is built against the same core.
- The CUDA 12 and MIGraphX providers depend on `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`. The lower bound is the first pkgrel with the new plug (the fixed "ABI pkgrel", reset to 1 with a new `pkgver`). Any later packaging-only pkgrel of 1.29.0 still satisfies the pair.
- `onnxruntime-core` in turn conflicts with the CUDA 12 provider below 1.29.0-2.

See [release.md](release.md#joint-release).

## Packages without NCCL

NCCL is on by default in every line. `onnxruntime-core` and `python-onnxruntime-core` depend on Arch's `nccl` (364.18 MiB installed), which the training operators for several GPUs need. A source build with `ORTF_NCCL=0` leaves it out and gives every package of that build the suffix `-nccloff`: `onnxruntime-core-nccloff`, `onnxruntime-provider-cuda13-rtx-nccloff`, `python-onnxruntime-core-nccloff`, and so on.

- Each `-nccloff` package provides and conflicts with its base name, so it replaces the NCCL package of that name.
- The providers and the Python package of such a build depend on `onnxruntime-core-nccloff`, which no NCCL core provides. Every NCCL provider and the NCCL Python package conflict with `onnxruntime-core-nccloff`. The two sets never mix, because NCCL changes the plug between the core and the providers, as training does.
- No `-nccloff` package is prebuilt: `scripts/gen-bin.sh` refuses them. They are for local builds. The shipped CUDA 12 and MIGraphX providers are built with NCCL and so need an NCCL core.

How to build them: [build.md](build.md#nccl-and-the-nccloff-packages).

## Sizes

These figures are measured on the files of the 1.29.0-6 build. The download size is the `.pkg.tar.zst` file. The installed size is the `size` field of the package's `.PKGINFO`. **The 1.29.0-7 packages, which add the training operators, NCCL and oneDNN, have not been measured for this table.** The figures below are those of the build before them. [results.md](results.md#package-sizes) has measured sizes of 1.29.0-7.

| Package | Download | Installed |
|---|---|---|
| `onnxruntime-core` | 7.9 MiB | 29.4 MiB |
| `onnxruntime-provider-cuda13-rtx` | 99.2 MiB | 154.7 MiB |
| `onnxruntime-provider-cuda13-all` | 226.9 MiB | 294.9 MiB |
| `onnxruntime-provider-tensorrt-cuda13` | 0.3 MiB | 1.2 MiB |
| `onnxruntime-provider-openvino` | 0.5 MiB | 1.7 MiB |
| `python-onnxruntime-core` | 10.0 MiB | 43.2 MiB |
| `onnxruntime-provider-cuda12-legacy` (1.29.0-1) | 172.6 MiB | 557.5 MiB |

The packages these depend on add to this. Arch's `nccl` is 292.29 MiB to download and 364.18 MiB installed (`pacman -Si nccl`, 2.32.3-1). The CUDA provider grows with the number of GPU architectures it holds: see [build.md](build.md#cuda-architectures).

## Why one package per engine, not one per build

Arch splits per build. Each variant is a separate compilation of the whole library, the variants conflict, and you install one at a time ([choosing.md](choosing.md#what-arch-ships)).

Here the providers are separate shared objects (`libonnxruntime_providers_<provider>.so`) that one `libonnxruntime.so` loads at run time, so a single build serves every engine. The split follows the engines:

- Each package carries only its provider library.
- The vendor runtimes that library needs are its dependencies.
- The core package runs without any of them.

A machine installs the engines it has and nothing else, for example the core and the OpenVINO package on a machine without an NVIDIA GPU. How the loading works is in [engines.md](engines.md#how-the-providers-load).

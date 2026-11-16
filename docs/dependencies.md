# Dependencies and version bounds

What each package depends on, why the vendor runtimes carry version bounds, and how to see whether a rebuild is due. Terms such as `DT_NEEDED`, soname and `dlopen()` are in the [glossary](glossary.md).

## The policy

Every package's `depends` cover what its libraries need. During `package()` the recipe reads the `DT_NEEDED` entries of each library it installs and fails the build for a library that is neither a system library nor mapped to a package in that package's `depends`. A package therefore cannot be built that needs a library it does not depend on, and a provider package cannot be installed without its vendor runtime.

The providers and the Python module depend on `onnxruntime-core` at the same `pkgver-pkgrel`. `onnxruntime-core` itself links no vendor library: the build fails if the core library or the Python module needs a CUDA, cuDNN, TensorRT, OpenVINO, MIGraphX or HIP library.

## What each package depends on

| Package | Depends on |
|---|---|
| `onnxruntime-core` | `glibc`, `libgcc`, `libstdc++` |
| `onnxruntime-provider-cuda<M>-<sfx>` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-cuda12-legacy` ([the CUDA 12 line](cuda12-legacy.md)) | `onnxruntime-core=<pkgver>` (any pkgrel), `glibc`, `libgcc`, `libstdc++`, `nvidia-utils`, `cuda>=12`, `cuda<13`, `cudnn>=9.10`, `cudnn<9.11` (no bare `cuda` or `cudnn`) |
| `onnxruntime-provider-tensorrt-cuda<M>` | everything the CUDA package depends on, plus `onnxruntime-provider-cuda<M>=<pkgver>` and `tensorrt` (bounded as below) |
| `onnxruntime-provider-openvino` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `openvino` |
| `onnxruntime-provider-migraphx` (only with `migraphx` in `ORTF_EPS`; untested on AMD hardware) | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `migraphx`, `hip-runtime-amd` (bounded as below; never `rocm-hip-sdk`) |
| `python-onnxruntime-core` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `python`, `python-numpy`, `python-flatbuffers`, `python-packaging`, `python-protobuf` |

The OpenVINO device plugins and drivers are optional dependencies of `onnxruntime-provider-openvino`, because each device needs only its own ([engines.md](engines.md#openvino)).

## The version bounds

The vendor runtimes and Python carry version bounds where the ABI demands a rebuild after an upgrade. A built package lists such a dependency by its bounds alone (`cuda>=13` and `cuda<14`, not also a bare `cuda`). The bare name is in `.SRCINFO`, where the bounds are not known yet. The CUDA 12 recipe's `.SRCINFO` carries its bounds, because its CUDA and cuDNN versions are fixed.

| Dependency | Bound | Taken from |
|---|---|---|
| `cuda` | major version | the sonames the libraries link (`libcudart.so.13` gives `cuda>=13` and `cuda<14`) |
| `cudnn` | major version (major.minor in the CUDA 12 legacy package: `cudnn>=9.10` and `cudnn<9.11`, because cuDNN 9.11 dropped Pascal and Volta) | `CUDNN_MAJOR` (and `CUDNN_MINOR`) in `cudnn_version.h` of the cuDNN installed when `prepare()` ran. The CUDA provider `dlopen()`s cuDNN and has no `libcudnn` in its `DT_NEEDED` entries, so no soname gives it |
| `tensorrt` | major.minor (the ABI is sensitive to the minor version) | `include/NvInferVersion.h` of the TensorRT the build used. For the build, `tensorrt` is in `makedepends` when the TensorRT provider is built: a TensorRT installed from a package (the AUR's `tensorrt`, or `nvidia-tensorrt-sdk-local` pkgrel 2 or later) satisfies it. A TensorRT unpacked by hand needs `makepkg -d`, which skips the dependency check. `prepare()` warns when none provides `tensorrt` |
| `openvino` | year.minor | the version installed when `prepare()` ran |
| `python` | minor version | the version of the python that built the module |
| `migraphx`, `hip-runtime-amd` | major version of the package (`migraphx` 7.2.3 gives `>=7` and `<8`) | the versions installed when `prepare()` ran, not the sonames: ROCm 7.2 and 10.1 keep `libmigraphx_c.so.3` and `libamdhip64.so.7`, so a soname bound would admit a ROCm the provider was never built against. The sonames are checked against the ones the build linked. Upgrading ROCm across a major needs a rebuild, and pacman refuses the upgrade until then |

`prepare()` records those versions in `src/ortf.conf`. The bounds mean that pacman refuses a system update that would move one of these packages past what the provider was built for, instead of leaving a provider that fails to load. When Arch moves one of them past its bound, rebuild the packages (a `pkgrel` bump) against the new version. `onnxruntime-core` has no vendor bound and keeps working CPU-only whatever the vendor packages do.

Libraries that a provider loads with `dlopen()` do not appear in `DT_NEEDED`. `libcuda`, `libnvrtc` and `libcufft` are covered by `nvidia-utils` and the `cuda` bound, and cuDNN by its recorded bound, instead of by what the library links.

### TensorRT: one dependency name for two sources

The TensorRT provider package depends on the name `tensorrt`, bounded to the major.minor of the TensorRT the build used (`tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4). Both TensorRT sources provide that name: the AUR's `tensorrt` (under `/usr`) and `nvidia-tensorrt-sdk-local` (under `/opt/nvidia/tensorrt`, pkgrel 2 or later; see [install.md](install.md#2-tensorrt)). The dependency names the shared `tensorrt`, not the package that owns the build's library, so that a provider built against one source installs with the other.

## Prebuilt packages

The packages depend on each other only by names that a prebuilt package provides as well. The convention, from the PKGBUILD header, is that `<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>` (and provides and conflicts with the plain virtual names `<name>` provides, such as `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`, at the same versions). The chain therefore holds whether a package was built from source or prebuilt, and `scripts/gen-bin.sh` refuses to generate a `-bin` package whose dependencies break it. The prebuilt packages have the same bounds as a source build.

## Checking built packages: `scripts/check-deps.sh`

```bash
scripts/check-deps.sh onnxruntime-split/*-1.29.0-6-x86_64.pkg.tar.zst
```

The script takes package files and is read-only: it queries `pacman` and `ldconfig` and writes only under a temporary directory. It needs `bsdtar`, `readelf`, `pacman` and `vercmp`. For each package it unpacks, and for each ELF shared library in it, it prints:

- every needed soname with the path it resolves to (the library's RUNPATH, then a library of the package files given, then `ldconfig`'s cache);
- the package that owns that path, with its version;
- whether a dependency of the package covers that owner.

It then prints every dependency of the package, the installed package (or package file given) that meets it, that package's version, and whether the version is inside the dependency's bounds (compared as `vercmp` does). The rows carry a status; these flag a problem:

| Status | Meaning |
|---|---|
| `GAP` | a needed library whose owner no dependency covers, or that no package owns |
| `UNRESOLVED` | a needed soname that resolves to no file on this system |
| `BOUND` | an installed version outside a dependency's bounds: the vendor package (`cuda`, `cudnn`, TensorRT, `openvino`, `python`, `migraphx`, `hip-runtime-amd`) was upgraded past what the package was built for, and a rebuild is due |
| `MISSING` | a dependency that no installed package and no package file given meets |

The exit status is 0 when nothing is flagged, 1 when anything is (the last line then counts each flag), and 2 on a usage error or an unreadable package file. `scripts/check-deps.sh --help` prints the script's header.

Use it twice. Run it on freshly built package files before installing them, to see every library and its owner. Run it again after an Arch update, on the package files you installed from, to see whether a rebuild is due: a `BOUND` row names the dependency that moved.

Because the script sees only `DT_NEEDED` entries, libraries loaded with `dlopen()` are covered by the recorded bounds instead (above).

While the installed `nvidia-tensorrt-sdk-local` is pkgrel 1, or no TensorRT is installed, a run on the TensorRT provider package reports `tensorrt>=11.4` and `tensorrt<11.5` as `MISSING`, and the two TensorRT libraries it links (`libnvinfer` and `libnvonnxparser`) as `GAP`. Nothing installed provides `tensorrt` (pkgrel 1 provides only `nvidia-tensorrt-sdk`). The script counts provided names, so these rows clear once the pkgrel 2 package, or the AUR's `tensorrt`, is installed.

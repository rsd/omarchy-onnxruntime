# Dependencies and version bounds

This page says what each package depends on, why the vendor runtimes carry version bounds, and how to see whether a rebuild is due. Terms such as `DT_NEEDED`, soname and `dlopen()` are in the [glossary](glossary.md).

## The policy

Every package's `depends` cover what its libraries need. During `package()` the recipe reads the `DT_NEEDED` entries of each library it installs. It fails the build for a library that is neither a system library nor mapped to a package in that package's `depends`. A package therefore cannot be built that needs a library it does not depend on, and a provider package cannot be installed without its vendor runtime. The same package-time pass checks every library for undefined symbols (`ldd -r`) and for RUNPATH entries other than `$ORIGIN` and the TensorRT provider's `/opt/nvidia/tensorrt/lib`.

The providers and the Python module depend on `onnxruntime-core` at the same `pkgver-pkgrel`. The CUDA 12 and MIGraphX providers depend on a pair of bounds, below.

`onnxruntime-core` itself links no vendor library. The build fails if the core library or the Python module needs a CUDA, cuDNN, TensorRT, OpenVINO, MIGraphX or HIP library.

NCCL is the one exception to "depends on what it links". NCCL is a library of the CUDA provider, and CMake also links it into the core and the Python module. The linker is asked to drop libraries nothing calls, so they likely have no `DT_NEEDED` entry for it. `onnxruntime-core` and `python-onnxruntime-core` depend on the build's NCCL package all the same, so that the packages stay consistent with a build that does keep the entry (not yet seen in a build).

## What each package depends on

| Package | Depends on |
|---|---|
| `onnxruntime-core` | `glibc`, `libgcc`, `libstdc++`, `nccl` |
| `onnxruntime-provider-cuda<M>-<sfx>` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `cuda`, `cudnn`, `nvidia-utils`, `nccl` |
| `onnxruntime-provider-cuda12-legacy` ([the CUDA 12 line](cuda12-legacy.md)) | `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`, `glibc`, `libgcc`, `libstdc++`, `nvidia-utils`, `cuda>=12`, `cuda<13`, `cudnn>=9.10`, `cudnn<9.11` (no bare `cuda` or `cudnn`), `nccl-cuda12.9` (by name, not the bare `nccl`) |
| `onnxruntime-provider-tensorrt-cuda<M>` | everything the CUDA package depends on except `nccl`, plus `onnxruntime-provider-cuda<M>=<pkgver>` and `tensorrt` (bounded as below) |
| `onnxruntime-provider-openvino` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `openvino` |
| `onnxruntime-provider-dnnl` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `libgomp` (oneDNN's OpenMP runtime). No `onednn`: oneDNN is inside the provider ([below](#onednn-is-bundled-statically)) |
| `onnxruntime-provider-migraphx` ([the MIGraphX line](amd.md); untested on AMD hardware) | `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`, `glibc`, `libgcc`, `libstdc++`, `migraphx`, `hip-runtime-amd` (bounded as below; never `rocm-hip-sdk`, and no `rccl`) |
| `python-onnxruntime-provider-migraphx` (the Python add-on, [the MIGraphX line](amd.md#the-package)) | `onnxruntime-provider-migraphx` at the same version, `python-onnxruntime-core`, `python` (bounded to the minor of the site-packages it links into, as below) |
| `python-onnxruntime-core` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `nccl`, `python`, `python-numpy`, `python-flatbuffers`, `python-packaging`, `python-protobuf`, and for `onnxruntime.training` `python-cerberus`, `python-h5py`, `python-onnx`, `python-sympy` and `python-setuptools` |

The OpenVINO device plugins and drivers are optional dependencies of `onnxruntime-provider-openvino`, because each device needs only its own ([engines.md](engines.md#openvino)). `python-pytorch` is an optional dependency of `python-onnxruntime-core`. The training modules import it, and `import onnxruntime.training` fails without it. `import onnxruntime` does not need it.

A build with `ORTF_NCCL=0` names every package with the suffix `-nccloff`, drops `nccl` and depends on `onnxruntime-core-nccloff` instead of `onnxruntime-core` ([packages.md](packages.md#packages-without-nccl)).

### The core pin of the CUDA 12 and MIGraphX providers

These two providers come from their own recipes and load into the main line's core. They depend on a range of cores, not on one build's: `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`.

- Version 1.29.0-7 is the first pkgrel of the core built with the training operators and NCCL. They change the interface between the core and the providers, and nothing checks at run time that both agree.
- The lower bound refuses a core below 1.29.0-7. A later packaging-only pkgrel of 1.29.0 meets the pair.
- The bound resets to 1 with a new `pkgver`.
- `scripts/gen-bin.sh` accepts only the whole pair, never one half and never an inclusive upper bound ([release-cuda12.md](release-cuda12.md#the-core-pin)).

## The version bounds

The vendor runtimes and Python carry version bounds where the ABI demands a rebuild after an upgrade. A built package lists such a dependency by its bounds alone (`cuda>=13` and `cuda<14`, not also a bare `cuda`). The bare name is in `.SRCINFO`, where the bounds are not known yet. The CUDA 12 recipe's `.SRCINFO` carries its bounds, because its CUDA and cuDNN versions are fixed.

| Dependency | Bound | Taken from |
|---|---|---|
| `cuda` | major version | the sonames the libraries link (`libcudart.so.13` gives `cuda>=13` and `cuda<14`) |
| `cudnn` | major version (major.minor in the CUDA 12 legacy package: `cudnn>=9.10` and `cudnn<9.11`, because cuDNN 9.11 dropped Pascal and Volta) | `CUDNN_MAJOR` (and `CUDNN_MINOR`) in `cudnn_version.h` of the cuDNN installed when `prepare()` ran. The CUDA provider links cuDNN directly (with the training operators on), so its `DT_NEEDED` entries have `libcudnn.so.9`. Earlier builds `dlopen()`ed it. That soname is the same for every cuDNN 9, so it cannot give the minor bound, and the header can |
| `tensorrt` | major.minor (the ABI is sensitive to the minor version) | `include/NvInferVersion.h` of the TensorRT the build used. For the build, `tensorrt` is in `makedepends` when the TensorRT provider is built. A TensorRT installed from a package (the AUR's `tensorrt`, or `nvidia-tensorrt-sdk-local` pkgrel 2 or later) satisfies it. A TensorRT unpacked by hand needs `makepkg -d`, which skips the dependency check. `prepare()` warns when none provides `tensorrt` |
| `openvino` | year.minor | the version installed when `prepare()` ran |
| `python` | minor version | the version of the python that built the module. For the MIGraphX add-on, the minor of the system site-packages it links into (`/usr/lib/python<minor>/site-packages`) |
| `nccl` | none | the library's soname, `libnccl.so.2`, has been NCCL's since 2.0, and the AUR's `nccl-cuda12.9` provides `nccl` without a version, which a versioned dependency would not accept |
| `migraphx`, `hip-runtime-amd` | major version of the package (`migraphx` 7.2.3 gives `>=7` and `<8`) | the versions installed when `prepare()` ran, not the sonames. ROCm 7.2 and 10.1 keep `libmigraphx_c.so.3` and `libamdhip64.so.7`, so a soname bound would admit a ROCm the provider was never built against. The sonames are checked against the ones the build linked. Upgrading ROCm across a major needs a rebuild, and pacman refuses the upgrade until then |

`prepare()` records those versions in `src/ortf.conf`. The bounds mean that pacman refuses a system update that would move one of these packages past what the provider was built for, instead of leaving a provider that fails to load. When Arch moves one of them past its bound, rebuild the packages (a `pkgrel` bump) against the new version. `onnxruntime-core` has no vendor bound and keeps working CPU-only whatever the vendor packages do.

Libraries that a provider loads with `dlopen()` do not appear in `DT_NEEDED`. `libcuda`, `libnvrtc` and `libcufft` are covered by `nvidia-utils` and the `cuda` bound instead of by what the library links. cuDNN was in that group in earlier builds, covered by its recorded bound alone. The CUDA provider now links it, so `libcudnn.so.9` is in `DT_NEEDED`.

### NCCL: Arch's `nccl`, and the AUR's `nccl-cuda12.9` for the CUDA 12 line

The CUDA 13 packages and the core use Arch's `nccl` (2.32.3, built for `sm_75` and newer). Pascal and Volta, which the CUDA 12 line exists for, are older than that. So that line's provider depends on the AUR's `nccl-cuda12.9` by name. It provides and conflicts with `nccl` and builds its kernels for `sm_52` to `sm_72`. A bare `nccl` dependency would let pacman pick Arch's.

The core's own dependency stays the bare `nccl`, which `nccl-cuda12.9` provides. Installing the CUDA 12 line therefore swaps one NCCL for the other ([cuda12-legacy.md](cuda12-legacy.md#what-the-machine-needs)). `rccl` is a build input of the MIGraphX line only. No shipped package depends on it.

### oneDNN is bundled statically

`onnxruntime-provider-dnnl` carries oneDNN 3.11.3, built from a checksummed source archive during the package build and linked into the provider as a static library. It therefore depends on no oneDNN package and installs no `libdnnl.so.3`, which is a file of Arch's `onednn`. The two never conflict, and a system with Arch's `onednn` installed uses it for other programs. The only runtime library it adds is `libgomp` (`gcc-libs`), oneDNN's OpenMP runtime.

`package()` fails the build if the provider has a `libdnnl` among its `DT_NEEDED` entries, and so does `scripts/check-deps.sh` (status `DNNL`, below). Version 3.11.3 is the one Arch's `onednn` ships today. The provider does not follow Arch's updates. Moving to Arch's package later would trade a smaller package and Arch's security updates for a tie to Arch's oneDNN releases.

### TensorRT: one dependency name for two sources

The TensorRT provider package depends on the name `tensorrt`, bounded to the major.minor of the TensorRT the build used (`tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4). Both TensorRT sources provide that name: the AUR's `tensorrt` (under `/usr`) and `nvidia-tensorrt-sdk-local` (under `/opt/nvidia/tensorrt`, pkgrel 2 or later; see [INSTALL.md](../INSTALL.md#install-tensorrt)). The dependency names the shared `tensorrt`, not the package that owns the build's library, so that a provider built against one source installs with the other.

## Prebuilt packages

The packages depend on each other only by names that a prebuilt package provides as well. The convention, from the PKGBUILD header, is that `<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>`. It also provides and conflicts with the plain virtual names `<name>` provides, such as `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`, at the same versions.

The chain therefore holds whether a package was built from source or prebuilt, and `scripts/gen-bin.sh` refuses to generate a `-bin` package whose dependencies break it. The prebuilt packages have the same bounds as a source build. No `-nccloff` package is prebuilt, and there is no `-bin` for the MIGraphX line.

## Checking built packages: `scripts/check-deps.sh`

```bash
scripts/check-deps.sh onnxruntime-split/*-1.29.0-7-x86_64.pkg.tar.zst
```

The script takes package files and is read-only. It queries `pacman` and `ldconfig` and writes only under a temporary directory. It needs `bsdtar`, `readelf`, `pacman` and `vercmp`. For each package it unpacks, and for each ELF shared library in it, it prints:

- every needed soname with the path it resolves to (the library's RUNPATH, then a library of the package files given, then `ldconfig`'s cache);
- the package that owns that path, with its version;
- whether a dependency of the package covers that owner.

It then prints every dependency of the package, the installed package (or package file given) that meets it, that package's version, and whether the version is inside the dependency's bounds (compared as `vercmp` does). The rows carry a status. These flag a problem:

| Status | Meaning |
|---|---|
| `GAP` | a needed library whose owner no dependency covers, or that no package owns |
| `UNRESOLVED` | a needed soname that resolves to no file on this system |
| `BOUND` | an installed version outside a dependency's bounds: the vendor package (`cuda`, `cudnn`, TensorRT, `openvino`, `python`, `migraphx`, `hip-runtime-amd`) was upgraded past what the package was built for, and a rebuild is due |
| `MISSING` | a dependency that no installed package and no package file given meets |
| `DNNL` | a needed `libdnnl`, whoever owns it: the oneDNN provider links oneDNN statically, and no package of this recipe may need Arch's `onednn` |

The exit status is 0 when nothing is flagged, 1 when anything is (the last line then counts each flag), and 2 on a usage error or an unreadable package file. `scripts/check-deps.sh --help` prints the script's header.

Use it twice:

- Run it on freshly built package files before installing them, to see every library and its owner.
- Run it again after an Arch update, on the package files you installed from, to see whether a rebuild is due. A `BOUND` row names the dependency that moved.

The script sees only `DT_NEEDED` entries, so the recorded bounds cover libraries loaded with `dlopen()` instead (above). In builds that link cuDNN, `libcudnn.so.9` is resolved and its owner checked like any other library.

While the installed `nvidia-tensorrt-sdk-local` is pkgrel 1, or no TensorRT is installed, a run on the TensorRT provider package reports `tensorrt>=11.4` and `tensorrt<11.5` as `MISSING`. It reports the two TensorRT libraries the package links (`libnvinfer` and `libnvonnxparser`) as `GAP`. Nothing installed provides `tensorrt` (pkgrel 1 provides only `nvidia-tensorrt-sdk`). The script counts provided names, so these rows clear once the pkgrel 2 package, or the AUR's `tensorrt`, is installed.

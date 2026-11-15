# Installing

The order matters: the runtimes of the engines you want come first, then this package is built against them. Every command here is run by you; nothing in this repository runs `sudo` or installs anything.

```mermaid
flowchart LR
    A[1. build tools and runtimes] --> B[2. TensorRT]
    B --> C[3. makepkg and pacman -U]
    C --> D[4. smoke test]
```

Skip the runtimes of engines you do not want, and build without them with `ORTF_EPS` ([build.md](build.md#build-inputs)). A CPU-only package needs no vendor runtime and no step 2.

## 1. Prerequisites

Build tools. This is the PKGBUILD's `makedepends` (`git`, `cmake`, `ninja`, `patchelf`, the Python build modules and NumPy) plus `base-devel` for `makepkg` itself:

```bash
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools
```

`ccache` is optional and makes a rebuild much cheaper ([build.md](build.md#ccache)):

```bash
sudo pacman -S --needed ccache
```

The PKGBUILD's `makedepends` also lists `cuda`, `cudnn` and `openvino`, because the default build has the CUDA, TensorRT and OpenVINO providers. TensorRT is not listed; see step 2. With `migraphx` in `ORTF_EPS` it also lists `migraphx` and `hip-runtime-amd`.

Runtimes. Install the ones for the engines you build and use (vendor licences apply: see the [README](../README.md#licence-and-personal-use)):

```bash
# CUDA and TensorRT providers
sudo pacman -S --needed nvidia-utils cuda cudnn

# OpenVINO provider: CPU, Intel GPU and Intel NPU devices
sudo pacman -S --needed openvino openvino-intel-gpu-plugin openvino-intel-npu-plugin intel-compute-runtime level-zero-loader intel-npu-driver

# MIGraphX provider (AMD GPUs; untested on AMD hardware, built only with ORTF_EPS=migraphx)
sudo pacman -S --needed migraphx hip-runtime-amd
```

`nvidia-utils` is the NVIDIA driver's user-space part; you need it at run time, not to build. The provider packages depend on these runtimes: `onnxruntime-provider-cuda<M>-<sfx>` on `cuda`, `cudnn` and `nvidia-utils`, `onnxruntime-provider-tensorrt-cuda<M>` on the same plus `tensorrt` (step 2), `onnxruntime-provider-openvino` on `openvino`, and `onnxruntime-provider-migraphx` on `migraphx` and `hip-runtime-amd`. A provider package cannot be installed without its runtimes, so install them before step 3. Their dependencies carry version bounds, so that pacman holds back a vendor upgrade that would break the provider until the packages are rebuilt ([Dependencies and version bounds](#dependencies-and-version-bounds)). The OpenVINO device plugins and drivers (`openvino-intel-gpu-plugin`, `openvino-intel-npu-plugin`, `intel-compute-runtime`, `level-zero-loader`, `intel-npu-driver`) are optional dependencies of `onnxruntime-provider-openvino`, because each device needs only its own.

For the Intel NPU, check that the kernel's `intel_vpu` driver is loaded and that your user can open `/dev/accel/accel0` (look at the device's group with `ls -l /dev/accel/accel0`; on many systems it is `render`, so your user would need to be in that group). Check the Intel GPU device's render node, `/dev/dri/renderD*`, in the same way.

## 2. TensorRT

Only for the TensorRT provider. Either:

- **`nvidia-tensorrt-sdk-local`** from the omarchy-nvidia-sdks repository (directory `nvidia-tensorrt`). It packages your own download of NVIDIA TensorRT 11.4 under `/opt/nvidia/tensorrt`, with an environment file `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env`. You download the archive from NVIDIA yourself, after accepting its licence; that repository's `docs/tensorrt.md` has the download, the options and the checks. The build and install commands are, from that repository:

  ```bash
  cd nvidia-tensorrt
  ./configure.sh --src ~/Downloads --write
  PKGEXT=.pkg.tar makepkg -f
  sudo pacman -U nvidia-tensorrt-sdk-local-11.4.0.106-2-x86_64.pkg.tar
  ```

  Use pkgrel 2 or later. Pkgrel 2 provides `tensorrt=11.4.0.106` as well as `nvidia-tensorrt-sdk`. Pkgrel 1 provides only `nvidia-tensorrt-sdk`, so it does not satisfy the provider package's `tensorrt` dependency; install pkgrel 2 over it.

- **The AUR's `tensorrt`**, which installs into `/usr` and provides `tensorrt`. Build and install it as you do any AUR package. Its major.minor must be that of the headers the build uses (see below).

This recipe's build finds TensorRT in `/opt/nvidia/tensorrt` when that holds `include/NvInfer.h`, and otherwise in `/usr`; `ORTF_TENSORRT_HOME` overrides this ([build.md](build.md#build-inputs)). The PKGBUILD requires TensorRT 11: its provider has the code paths for it.

The TensorRT provider package depends on `tensorrt`, bounded to the major.minor of the TensorRT the build used: `tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4. The bound is read from `include/NvInferVersion.h` under the TensorRT root, not from an installed package, so a build needs no TensorRT package installed. Either source above provides the name, so the provider package does not care which one supplies it, only that the version is inside the bound. `prepare()` warns when no installed package provides `tensorrt`, or when the installed one has another major.minor.

The name is shared so that the dependency is the same whichever TensorRT the build used. The earlier dependency named the package that owned the build's TensorRT (`nvidia-tensorrt-sdk` or `tensorrt`, or the owner of `libnvinfer`), so a provider built against one source could not be installed from the other.

Install order: update `nvidia-tensorrt-sdk-local` to pkgrel 2 (or install the AUR's `tensorrt`) before step 3, then install the packages of step 3 in one `pacman -U`. pacman refuses the TensorRT provider package while no installed package provides `tensorrt` within its bound.

## 3. This package

```bash
cd onnxruntime-split
MAKEFLAGS=-j8 makepkg -f
sudo pacman -U onnxruntime-core-1.29.0-4-x86_64.pkg.tar.zst \
  onnxruntime-provider-cuda13-sm120-1.29.0-4-x86_64.pkg.tar.zst \
  onnxruntime-provider-tensorrt-cuda13-1.29.0-4-x86_64.pkg.tar.zst \
  onnxruntime-provider-openvino-1.29.0-4-x86_64.pkg.tar.zst \
  python-onnxruntime-core-1.29.0-4-x86_64.pkg.tar.zst
```

The install in the `pacman -U` above needs step 2 done first: the TensorRT provider package is refused until an installed package provides `tensorrt` within its bound. Packages of 1.29.0-2 or 1.29.0-3 are replaced by the same names at 1.29.0-4 in that one transaction.

`makepkg` downloads the upstream source and the pinned dependencies as its sources, then builds the library once and writes one package per engine. A full build with one CUDA architecture took 27 minutes on a 24-thread machine ([build.md](build.md#parallelism-memory-and-disk)); it needs tens of GiB of disk, so choose `-jN` for your memory and read that section before the first run. If you installed `ccache` it is used automatically.

Install the packages of one build in a single `pacman -U` that names the exact files. Do not use a glob such as `*onnxruntime-*-1.29.0-4-*`: it also matches packages that an earlier build with other inputs left in the directory (for example `onnxruntime-provider-cuda13-rtx` beside `-cuda13-sm120`), and two CUDA packages conflict, so the transaction fails. `makepkg` names a package `<name>-1.29.0-4-x86_64<PKGEXT>`, with `PKGEXT` from `/etc/makepkg.conf` (`.pkg.tar.zst` on Arch), and `PKGDEST=/an/empty/directory makepkg ...` keeps the packages of one build apart.

The names above are those of the default build. A subset is fine: the core alone (`onnxruntime-core`, CPU only), the core and `onnxruntime-provider-openvino`, and so on; the core is always needed, because the others depend on it at the same version (`onnxruntime-core=<pkgver>-<pkgrel>`). The CUDA package's suffix depends on `ORTF_CUDA_ARCHS` (`-cuda13-rtx` for the preset `rtx`, `-cuda13-all` for `all`, `-cuda13-sm89-sm120` for `"89 120"`), and `<M>` in `-cuda<M>` on the CUDA toolkit that built it; [build.md](build.md#cuda-architectures) has the rules. A build with `ORTF_EPS=cpu` writes `onnxruntime-core` and the Python package only. With `migraphx` in `ORTF_EPS` there is also `onnxruntime-provider-migraphx-1.29.0-4-x86_64.pkg.tar.zst` to name in the `pacman -U`; for an AMD-only build use `ORTF_EPS=migraphx makepkg -d -f` after installing `migraphx` and `hip-runtime-amd` ([build.md](build.md#migraphx-amd-gpus)). This package is **untested on AMD hardware**, and the smoke test below has no MIGraphX target. `onnxruntime-core` and `python-onnxruntime-core` list it as an optional dependency, and the package installs a symlink `<site-packages>/onnxruntime/capi/libonnxruntime_providers_migraphx.so` to `/usr/lib/libonnxruntime_providers_migraphx.so`, so the Python module can load it too (only when the build also makes `python-onnxruntime-core`). That is untested on AMD hardware as well.

If an Arch ONNX Runtime variant (`onnxruntime`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and so on, and their `python-` packages) is installed, pacman reports the conflict and offers to remove it; programs built against it keep working, because `onnxruntime-core` provides `onnxruntime` and `libonnxruntime.so`, and `python-onnxruntime-core` provides `python-onnxruntime`, at the same version.

For a CPU-only package, or any other combination, set the build inputs first: [build.md](build.md#build-inputs).

### Upgrading from the single package 1.29.0-1

Version 1.29.0-1 was one `onnxruntime-full` that owned every provider library, with `python-onnxruntime-full` for the Python module. `onnxruntime-core` has `replaces=(onnxruntime-full)` and `python-onnxruntime-core` has `replaces=(python-onnxruntime-full)`, and each conflicts with the name it replaces. `replaces=` acts only on a sync upgrade (`pacman -Syu` from a repository). With `pacman -U`, the conflicts make pacman offer to remove the old packages in the same transaction, and a file that moves from a package removed in that transaction to a new one is not a file conflict.

Install every new file in the one `pacman -U` shown above, not in several: pacman moves a file from one package to another only within one transaction, and the answer to its offer to remove `onnxruntime-full` and `python-onnxruntime-full` is yes: that prompt defaults to N, so `--noconfirm` aborts the upgrade; answer y. The old `python-onnxruntime-full` depends on `onnxruntime-full=1.29.0-1`, so `python-onnxruntime-core` must be in the same transaction. With `ORTF_PYTHON=0` there is no new Python package to replace it with: remove it first with `sudo pacman -R python-onnxruntime-full`, or the transaction fails on that dependency.

### Dependencies and version bounds

Every package's `depends` cover what its libraries need. `package()` reads the `DT_NEEDED` entries of each library it installs and fails the build for one that is neither a system library nor mapped to a package in that package's `depends`, so a package cannot be built that needs a library it does not depend on. The result:

| Package | Depends on |
|---|---|
| `onnxruntime-core` | `glibc`, `libgcc`, `libstdc++` |
| `onnxruntime-provider-cuda<M>-<sfx>` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-tensorrt-cuda<M>` | the same, plus `onnxruntime-provider-cuda<M>=<pkgver>` and `tensorrt` (bounded as below) |
| `onnxruntime-provider-openvino` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `openvino` |
| `onnxruntime-provider-migraphx` (only with `migraphx` in `ORTF_EPS`; untested on AMD hardware) | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `migraphx`, `hip-runtime-amd` (bounded as below; never `rocm-hip-sdk`) |
| `python-onnxruntime-core` | `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `python`, `python-numpy`, `python-flatbuffers`, `python-packaging`, `python-protobuf` |

The vendor runtimes and Python carry version bounds where the ABI demands a rebuild after an upgrade:

| Dependency | Bound | Taken from |
|---|---|---|
| `cuda` | major version | the sonames the libraries link (`libcudart.so.13` gives `cuda>=13` and `cuda<14`) |
| `cudnn` | major version | `CUDNN_MAJOR` of the cuDNN installed when `prepare()` ran: the CUDA provider `dlopen()`s cuDNN, so no soname gives it (`scripts/check-deps.sh` sees only `DT_NEEDED` entries, so this and the other `dlopen()`ed libraries are covered by the recorded bounds) |
| `tensorrt` | major.minor (the ABI is sensitive to the minor version) | `include/NvInferVersion.h` of the TensorRT the build used, not an installed package: no TensorRT package is needed to build, and `prepare()` warns when none provides `tensorrt` |
| `openvino` | year.minor | the version installed when `prepare()` ran |
| `python` | minor version | the version of the python that built the module |
| `migraphx`, `hip-runtime-amd` | major version of the package (`migraphx` 7.2.3 gives `>=7` and `<8`) | the versions installed when `prepare()` ran, not the sonames: ROCm 7.2 and 10.1 keep `libmigraphx_c.so.3` and `libamdhip64.so.7`, so a soname bound would admit a ROCm the provider was never built against. The sonames are checked against the ones the build linked. Upgrading ROCm across a major needs a rebuild, and pacman refuses the upgrade until then |

`prepare()` records those versions in `src/ortf.conf`. The bounds mean that pacman refuses a system update that would move one of these packages past what the provider was built for, instead of leaving a provider that fails to load. When Arch moves one of them past its bound, rebuild the packages (a `pkgrel` bump) against the new version. `onnxruntime-core` has no vendor bound and keeps working CPU-only whatever the vendor packages do.

The packages depend on each other only by names that a prebuilt package would provide as well. The convention, from the PKGBUILD header, is that `<name>-bin` provides `<name>=<pkgver>-<pkgrel>` and conflicts with `<name>` (and provides and conflicts with the plain virtual names `<name>` provides, such as `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda<M>`, at the same versions), so the chain holds whether a package was built from source or prebuilt. It is a convention only: no `-bin` package is promised.

#### Checking built packages: `scripts/check-deps.sh`

```bash
scripts/check-deps.sh onnxruntime-split/*-1.29.0-4-x86_64.pkg.tar.zst
```

The script takes package files and is read-only: it queries `pacman` and `ldconfig` and writes only under a temporary directory. For each package it unpacks, and for each ELF shared library in it, it prints every needed soname with the path it resolves to (the library's RUNPATH, a library of the package files given, then `ldconfig`'s cache), the package that owns that path with its version, and whether a dependency of the package covers that owner. It then prints every dependency of the package, the installed package (or package file given) that meets it, that package's version, and whether the version is inside the dependency's bounds (compared as `vercmp` does). The rows carry a status; these flag a problem:

| Status | Meaning |
|---|---|
| `GAP` | a needed library whose owner no dependency covers, or that no package owns |
| `UNRESOLVED` | a needed soname that resolves to no file on this system |
| `BOUND` | an installed version outside a dependency's bounds: the vendor package (`cuda`, `cudnn`, TensorRT, `openvino`, `python`, `migraphx`, `hip-runtime-amd`) was upgraded past what the package was built for, and a rebuild is due |
| `MISSING` | a dependency that no installed package and no package file given meets |

The exit status is 0 when nothing is flagged, 1 when anything is (the last line then counts each flag), and 2 on a usage error or an unreadable package file. Run it on freshly built package files before installing them, to see every library and its owner, and again after an Arch update, on the package files you installed from, to see whether a rebuild is due: a `BOUND` row names the dependency that moved. `scripts/check-deps.sh --help` prints the header. It needs `bsdtar`, `readelf`, `pacman` and `vercmp`.

While the installed `nvidia-tensorrt-sdk-local` is pkgrel 1, or no TensorRT is installed, a run on the TensorRT provider package reports `tensorrt>=11.4` and `tensorrt<11.5` as `MISSING`, and the two TensorRT libraries it links (`libnvinfer` and `libnvonnxparser`) as `GAP`, because nothing installed provides `tensorrt` (pkgrel 1 provides only `nvidia-tensorrt-sdk`). The script counts provided names, so these rows should clear once the pkgrel 2 package, or the AUR's `tensorrt`, is installed.

## 4. Verify

### Before installing: the build tree

`makepkg` leaves the build tree in `onnxruntime-split/src/build`. It holds `libonnxruntime.so`, the provider libraries and the Python package, so both tools can check the build before `pacman -U` replaces anything. From the repository root, with `<src>` the upstream source directory under `onnxruntime-split/src/`:

```bash
# the C smoke test against the build tree
scripts/smoke-test.sh --include <src>/include/onnxruntime --libdir onnxruntime-split/src/build \
    --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU

# a real model on every engine; with TensorRT under /opt, the TensorRT environment file puts its lib/ on LD_LIBRARY_PATH (the provider's RUNPATH already finds its libraries)
set -a; . /usr/share/nvidia-tensorrt/nvidia-tensorrt.env; set +a
PYTHONPATH=onnxruntime-split/src/build tools/ort-model-check/model-check.py model.onnx
```

Skip the environment file when TensorRT is under `/usr`: the AUR's `tensorrt` does not install it. The option meanings are in the [smoke test](../tools/ort-smoke/README.md#running-it) and [model check](../tools/ort-model-check/README.md#running-it) READMEs; results of both on a full build are in [results.md](results.md).

### After installing

```bash
scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU
```

The test lists the providers and runs a small model on every engine and device it finds, comparing the numbers with a reference. `--require` takes a comma-separated list of targets (`cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU`) and exits 1 unless each is `ok`. Without `--require`, an engine whose runtime or device is missing is reported `unavailable`, not as a failure. List only the engines your machine has, for example `--require cpu,cuda` on a machine without Intel hardware.

The wrapper needs a C compiler and Python 3. `--only` and the other options, and how to read the output (`ok`, `FAIL`, `unavailable`), are in [tools/ort-smoke/README.md](../tools/ort-smoke/README.md).

The model check also runs against the installed package, without `PYTHONPATH`.

A TensorRT `ok` does not show that TensorRT's built-in plugins load; see [engines.md](engines.md#tensorrt).

## Uninstalling

```bash
sudo pacman -R python-onnxruntime-core onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-cuda13-sm120 onnxruntime-provider-openvino onnxruntime-core
```

Add `onnxruntime-provider-migraphx` if you built it. Name the packages you installed: the provider packages depend on `onnxruntime-core`, so it cannot be removed while one remains. `pacman -Qq | grep onnxruntime` lists them.

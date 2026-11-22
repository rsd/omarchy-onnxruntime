# Building from source

This guide builds the packages from the recipe with `makepkg` and installs them. For prebuilt packages see [INSTALL.md](../INSTALL.md#prebuilt-packages). You run every command. Nothing here runs `sudo` or installs for you.

```mermaid
flowchart LR
    A[1. build tools and runtimes] --> B[2. TensorRT]
    B --> C[3. makepkg and pacman -U]
    C --> D[4. check]
```

## Before you start

- **Vendor runtimes and TensorRT.** Steps 1 and 2 are shared with the prebuilt path: [Install the vendor runtimes](../INSTALL.md#2-install-the-vendor-runtimes) and [Install TensorRT](../INSTALL.md#3-install-tensorrt). Install the runtimes of the engines you build.
- **Fewer engines.** Skip the runtimes of the engines you do not want, and build without them with `ORTF_EPS` ([build.md](build.md#build-inputs)). A CPU-only build needs no vendor runtime and no TensorRT. Since it has no `cuda`, it needs `ORTF_NCCL=0` ([build.md](build.md#nccl-and-the-nccloff-packages)).
- **Disk and time.** A build needs tens of GiB of disk. Read [build.md](build.md#parallelism-memory-and-disk) before the first run to choose `-jN` for your memory.

## Build tools

Install the PKGBUILD's `makedepends` for the build itself, plus `base-devel` for `makepkg`. They are `git`, `cmake`, `ninja`, `patchelf`, the Python build modules, NumPy, and the training modules' Python packages `python-cerberus`, `python-h5py`, `python-onnx` and `python-sympy`.

```bash
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools python-cerberus python-h5py python-onnx python-sympy
```

`ccache` is optional and makes a rebuild much cheaper ([build.md](build.md#ccache)):

```bash
sudo pacman -S --needed ccache
```

The PKGBUILD's `makedepends` also lists `cuda`, `cudnn`, `nccl` and `openvino`, because the default build has the CUDA, TensorRT and OpenVINO providers and NCCL. The oneDNN provider needs nothing installed: the recipe builds oneDNN from a source archive it downloads. The recipe adds more `makedepends` for some engines:

- `tensorrt` when it builds the TensorRT provider. The AUR's `tensorrt` and `nvidia-tensorrt-sdk-local`, pkgrel 2 or later, both provide it ([Install TensorRT](../INSTALL.md#3-install-tensorrt)).
- `migraphx` and `hip-runtime-amd` when `migraphx` is in `ORTF_EPS`. The MIGraphX package has its own recipe: [amd.md](amd.md#building-it).

## TensorRT for a source build

The recipe finds TensorRT in `/opt/nvidia/tensorrt` when that holds `include/NvInfer.h`, and otherwise in `/usr`. `ORTF_TENSORRT_HOME` overrides this ([build.md](build.md#tensorrt-location)). The recipe reads the bound for the provider's `tensorrt` dependency from `include/NvInferVersion.h` under the TensorRT root.

- **`makedepends`.** The recipe adds `tensorrt` to `makedepends` when it builds the TensorRT provider. A TensorRT installed from one of the packages in [Install TensorRT](../INSTALL.md#3-install-tensorrt) satisfies it.
- **A TensorRT unpacked by hand**, with `ORTF_TENSORRT_HOME` pointing at it, needs `makepkg -d`, which skips the dependency check:

  ```bash
  cd onnxruntime-split
  ORTF_TENSORRT_HOME=/path/to/tensorrt makepkg -d -f
  ```

- **Warnings.** `prepare()` warns when no installed package provides `tensorrt`, or when the installed one has another major.minor.

## 3. Build and install

1. Build in the recipe directory:

   ```bash
   cd onnxruntime-split
   MAKEFLAGS=-j8 makepkg -f
   ```

2. Install TensorRT first if you build the TensorRT provider. Pacman refuses the TensorRT provider package until an installed package provides `tensorrt` within its bound ([Install TensorRT](../INSTALL.md#3-install-tensorrt)).

3. Install the packages in one `pacman -U` that names the exact files:

   ```bash
   sudo pacman -U onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-cuda13-rtx-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-tensorrt-cuda13-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-openvino-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-dnnl-1.29.0-7-x86_64.pkg.tar.zst \
     python-onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst
   ```

`1.29.0-7` is the version in the recipe now (`pkgver` 1.29.0, `pkgrel` 7). Use the version in the PKGBUILD you build. The core and the Python package depend on `nccl`, which pacman installs from `extra` with them.

What `makepkg` does:

- Downloads the ONNX Runtime source and the pinned dependencies as its sources. The source comes from this project's mirror, because `github.com/microsoft/onnxruntime` has been gone since 2026-11-17. `ORTF_SOURCE=upstream` selects the git tag ([build.md](build.md#the-onnx-runtime-source-mirror-or-upstream)).
- Builds the library once and writes one package per engine.
- Uses `ccache` automatically when it is installed.

A full build with one CUDA architecture took 27 minutes on a 24-thread machine ([build.md](build.md#parallelism-memory-and-disk)).

### Naming the files

Install the packages of one build in a single `pacman -U` that names the exact files. Do not use a glob such as `*onnxruntime-*-1.29.0-7-*`. It also matches packages that an earlier build with other inputs left in the directory, for example `onnxruntime-provider-cuda13-sm120` beside `-cuda13-rtx`. Two CUDA packages conflict, so the transaction fails.

`makepkg` names a package `<name>-1.29.0-7-x86_64<PKGEXT>`, with `PKGEXT` from `/etc/makepkg.conf` (`.pkg.tar.zst` on Arch). `PKGDEST=/an/empty/directory makepkg ...` keeps the packages of one build apart.

If you installed a local build made before 1.29.0-5, its CUDA package was named `onnxruntime-provider-cuda13-sm120`. The default then was one architecture, `120`. `-cuda13-rtx` conflicts with it, so pacman offers to remove it. Answer `y`.

### Other combinations

The names above are those of the default build (the CUDA preset `rtx`). A subset is fine: the core alone (CPU only), the core and `onnxruntime-provider-openvino`, and so on. The core is always needed, because the other packages depend on it at the same version.

- **CUDA package name.** The suffix depends on `ORTF_CUDA_ARCHS` (`-cuda13-rtx` for `rtx`, `-cuda13-all` for `all`, `-cuda13-sm89-sm120` for `"89 120"`). The `13` in `-cuda13` is the CUDA toolkit's major version. [build.md](build.md#cuda-architectures) has the rules.
- **No CUDA.** A build without `cuda` in `ORTF_EPS`, such as `ORTF_EPS=cpu`, needs `ORTF_NCCL=0` and writes `-nccloff` packages. This command gives `onnxruntime-core-nccloff-1.29.0-7-x86_64.pkg.tar.zst` and the Python package's:

  ```bash
  ORTF_EPS=cpu ORTF_NCCL=0 makepkg -d -f
  ```

  - An `ORTF_NCCL=0` build also gives every package of a build with CUDA a `-nccloff` name. Its packages never install beside the NCCL ones ([build.md](build.md#nccl-and-the-nccloff-packages)).
  - On a machine with no NVIDIA GPU, this is how to avoid the `nccl` dependency (364.18 MiB installed). No `-nccloff` package is prebuilt.
- **oneDNN.** `dnnl` is in the default `ORTF_EPS`, so `onnxruntime-provider-dnnl-1.29.0-7-x86_64.pkg.tar.zst` is among the files. It is optional and not needed on most machines ([engines.md](engines.md#onednn)). Leave `dnnl` out of `ORTF_EPS` to skip it.
- **MIGraphX.** The provider is built from its own recipe, `onnxruntime-split-migraphx` ([amd.md](amd.md#building-it)). It is **untested on AMD hardware**. With `migraphx` in `ORTF_EPS` of this recipe, there is also `onnxruntime-provider-migraphx-nccloff-1.29.0-7-x86_64.pkg.tar.zst` to name. This command is an AMD-only local build:

  ```bash
  ORTF_EPS=migraphx ORTF_NCCL=0 makepkg -d -f
  ```

- **Other combinations.** Set the build inputs first: [build.md](build.md#build-inputs).
- **Pascal and Volta GPUs.** The CUDA 12 provider has its own recipe: [cuda12-legacy.md](cuda12-legacy.md#building-from-source).

### Conflicts with Arch's packages

If an Arch ONNX Runtime variant is installed (`onnxruntime`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and so on, and their `python-` packages), pacman reports the conflict and offers to remove it. Programs built against it keep working. `onnxruntime-core` provides `onnxruntime` and `libonnxruntime.so`, and `python-onnxruntime-core` provides `python-onnxruntime`, at the same version.

### Upgrading from the single package 1.29.0-1

Put all new files in the one `pacman -U`: [Upgrade from the single package 1.29.0-1](../INSTALL.md#upgrade-from-the-single-package-1290-1).

## 4. Check that it works

### Before installing: the build tree

`makepkg` leaves the build tree in `onnxruntime-split/src/build`. It leaves the ONNX Runtime source in `onnxruntime-split/src/onnxruntime-1.29.0` (with `ORTF_SOURCE=upstream`: `onnxruntime-split/src/onnxruntime`). The build tree holds `libonnxruntime.so`, the provider libraries and the Python package. Both tools can check the build before `pacman -U` replaces anything. From the repository root:

```bash
# the C smoke test against the build tree
scripts/smoke-test.sh --include onnxruntime-split/src/onnxruntime-1.29.0/include/onnxruntime \
    --libdir onnxruntime-split/src/build \
    --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU

# a real model on every engine
set -a; . /usr/share/nvidia-tensorrt/nvidia-tensorrt.env; set +a
PYTHONPATH=onnxruntime-split/src/build tools/ort-model-check/model-check.py model.onnx
```

- `--include` points at the source tree and `--libdir` at the build tree.
- `model.onnx` is a model of your own. The repository ships none.
- The environment file belongs to `nvidia-tensorrt-sdk-local` and puts TensorRT's `lib/` on `LD_LIBRARY_PATH`. Skip that line when TensorRT is under `/usr`, because the AUR's `tensorrt` does not install the file.
- Pass only the targets your machine has to `--require`.

The meaning of the options is in the [smoke test](../tools/ort-smoke/README.md#running-it) and [model check](../tools/ort-model-check/README.md#running-it) READMEs. Results of both on a full build are in [results.md](results.md).

### After installing

See [Check that it works](../INSTALL.md#check-that-it-works) in the install guide.

## Check the recipe without the vendor runtimes

A CPU-only build checks the recipe on a machine without CUDA, TensorRT or OpenVINO: [build.md](build.md#checking-the-recipe-without-the-vendor-runtimes).

# Building from source

How to build the packages from the recipe and install them. For prebuilt packages see [install.md](install.md#prebuilt-packages). Every command here is run by you; nothing in this repository runs `sudo` or installs anything.

The order matters: the runtimes of the engines you want come first, then the recipe is built against them.

```mermaid
flowchart LR
    A[1. build tools and runtimes] --> B[2. TensorRT]
    B --> C[3. makepkg and pacman -U]
    C --> D[4. smoke test]
```

Steps 1 and 2 are shared with the prebuilt path and are described in [install.md](install.md#1-prerequisites). This page adds the build tools and steps 3 and 4. Skip the runtimes of the engines you do not want, and build without them with `ORTF_EPS` ([build.md](build.md#build-inputs)): a CPU-only build needs no vendor runtime and no TensorRT.

## Build tools

These are the PKGBUILD's `makedepends` for the build itself (`git`, `cmake`, `ninja`, `patchelf`, the Python build modules and NumPy), plus `base-devel` for `makepkg`:

```bash
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools
```

`ccache` is optional and makes a rebuild much cheaper ([build.md](build.md#ccache)):

```bash
sudo pacman -S --needed ccache
```

The PKGBUILD's `makedepends` also lists `cuda`, `cudnn` and `openvino`, because the default build has the CUDA, TensorRT and OpenVINO providers. It adds `tensorrt` when the TensorRT provider is built (the AUR's `tensorrt` and `nvidia-tensorrt-sdk-local`, pkgrel 2 or later, both provide it; see [TensorRT](install.md#2-tensorrt)), and `migraphx` and `hip-runtime-amd` when `migraphx` is in `ORTF_EPS`. Install the runtimes of the engines you build: [install.md](install.md#1-prerequisites).

## 3. Build and install

```bash
cd onnxruntime-split
MAKEFLAGS=-j8 makepkg -f
sudo pacman -U onnxruntime-core-1.29.0-6-x86_64.pkg.tar.zst \
  onnxruntime-provider-cuda13-rtx-1.29.0-6-x86_64.pkg.tar.zst \
  onnxruntime-provider-tensorrt-cuda13-1.29.0-6-x86_64.pkg.tar.zst \
  onnxruntime-provider-openvino-1.29.0-6-x86_64.pkg.tar.zst \
  python-onnxruntime-core-1.29.0-6-x86_64.pkg.tar.zst
```

`1.29.0-6` is the version in the recipe now (`pkgver` 1.29.0, `pkgrel` 6); use the one in the PKGBUILD you build.

The install needs [TensorRT](install.md#2-tensorrt) done first: pacman refuses the TensorRT provider package until an installed package provides `tensorrt` within its bound.

`makepkg` downloads the upstream source and the pinned dependencies as its sources, then builds the library once and writes one package per engine. A full build with one CUDA architecture took 27 minutes on a 24-thread machine ([build.md](build.md#parallelism-memory-and-disk)); it needs tens of GiB of disk, so choose `-jN` for your memory and read that section before the first run. If `ccache` is installed it is used automatically.

### Naming the files

Install the packages of one build in a single `pacman -U` that names the exact files. Do not use a glob such as `*onnxruntime-*-1.29.0-6-*`: it also matches packages that an earlier build with other inputs left in the directory (for example `onnxruntime-provider-cuda13-sm120` beside `-cuda13-rtx`), and two CUDA packages conflict, so the transaction fails. `makepkg` names a package `<name>-1.29.0-6-x86_64<PKGEXT>`, with `PKGEXT` from `/etc/makepkg.conf` (`.pkg.tar.zst` on Arch). `PKGDEST=/an/empty/directory makepkg ...` keeps the packages of one build apart.

If you installed a local build made before 1.29.0-5, its CUDA package was named `onnxruntime-provider-cuda13-sm120` (the default then was one architecture, `120`). `-cuda13-rtx` conflicts with it, so pacman offers to remove it: answer y.

### Other combinations

The names above are those of the default build (the CUDA preset `rtx`). A subset is fine: the core alone (CPU only), the core and `onnxruntime-provider-openvino`, and so on. The core is always needed, because the other packages depend on it at the same version.

- The CUDA package's suffix depends on `ORTF_CUDA_ARCHS` (`-cuda13-rtx` for `rtx`, `-cuda13-all` for `all`, `-cuda13-sm89-sm120` for `"89 120"`), and the `13` in `-cuda13` on the CUDA toolkit that built it. [build.md](build.md#cuda-architectures) has the rules.
- A build with `ORTF_EPS=cpu` writes `onnxruntime-core` and the Python package only.
- With `migraphx` in `ORTF_EPS` there is also `onnxruntime-provider-migraphx-1.29.0-6-x86_64.pkg.tar.zst` to name. This package is **untested on AMD hardware**: see [amd.md](amd.md), and use `ORTF_EPS=migraphx makepkg -d -f` for an AMD-only build.
- For any other combination, set the build inputs first: [build.md](build.md#build-inputs).
- For Pascal and Volta GPUs, the CUDA 12 provider has its own recipe: [cuda12-legacy.md](cuda12-legacy.md#building-from-source).

### Conflicts with Arch's packages

If an Arch ONNX Runtime variant (`onnxruntime`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and so on, and their `python-` packages) is installed, pacman reports the conflict and offers to remove it. Programs built against it keep working, because `onnxruntime-core` provides `onnxruntime` and `libonnxruntime.so`, and `python-onnxruntime-core` provides `python-onnxruntime`, at the same version.

### Upgrading from the single package 1.29.0-1

See [install.md](install.md#upgrading-from-the-single-package-1290-1): all new files must be in the one `pacman -U`.

## 4. Verify

### Before installing: the build tree

`makepkg` leaves the build tree in `onnxruntime-split/src/build`, and the upstream source in `onnxruntime-split/src/onnxruntime`. The build tree holds `libonnxruntime.so`, the provider libraries and the Python package, so both tools can check the build before `pacman -U` replaces anything. From the repository root:

```bash
# the C smoke test against the build tree
scripts/smoke-test.sh --include onnxruntime-split/src/onnxruntime/include/onnxruntime \
    --libdir onnxruntime-split/src/build \
    --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU

# a real model on every engine
set -a; . /usr/share/nvidia-tensorrt/nvidia-tensorrt.env; set +a
PYTHONPATH=onnxruntime-split/src/build tools/ort-model-check/model-check.py model.onnx
```

`--include` points at the upstream source tree and `--libdir` at the build tree. `model.onnx` is a model of your own: the repository ships none. The environment file belongs to `nvidia-tensorrt-sdk-local` and puts TensorRT's `lib/` on `LD_LIBRARY_PATH`; skip that line when TensorRT is under `/usr`, because the AUR's `tensorrt` does not install the file. Pass only the targets your machine has to `--require`. Their meaning is in the [smoke test](../tools/ort-smoke/README.md#running-it) and [model check](../tools/ort-model-check/README.md#running-it) READMEs; results of both on a full build are in [results.md](results.md).

### After installing

See [Verify](install.md#4-verify) in the install guide.

## Checking the build without the vendor runtimes

A CPU-only build checks the recipe on a machine without CUDA, TensorRT or OpenVINO: [build.md](build.md#checking-the-recipe-without-the-vendor-runtimes).

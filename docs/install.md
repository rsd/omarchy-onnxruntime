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

The PKGBUILD's `makedepends` also lists `cuda`, `cudnn` and `openvino`, because the default build has the CUDA, TensorRT and OpenVINO providers. TensorRT is not listed; see step 2.

Runtimes. Install the ones for the engines you build and use (vendor licences apply: see the [README](../README.md#licence-and-personal-use)):

```bash
# CUDA and TensorRT providers
sudo pacman -S --needed nvidia-utils cuda cudnn

# OpenVINO provider: CPU, Intel GPU and Intel NPU devices
sudo pacman -S --needed openvino openvino-intel-gpu-plugin openvino-intel-npu-plugin intel-compute-runtime level-zero-loader intel-npu-driver
```

`nvidia-utils` is the NVIDIA driver's user-space part; you need it at run time, not to build. These names are the PKGBUILD's optional dependencies. They are optional for the installed package because each provider is loaded only when a session asks for it.

For the Intel NPU, check that the kernel's `intel_vpu` driver is loaded and that your user can open `/dev/accel/accel0` (look at the device's group with `ls -l /dev/accel/accel0`; on many systems it is `render`, so your user would need to be in that group). Check the Intel GPU device's render node, `/dev/dri/renderD*`, in the same way.

## 2. TensorRT

Only for the TensorRT provider. Either:

- **`nvidia-tensorrt-sdk-local`** from the omarchy-nvidia-sdks repository (directory `nvidia-tensorrt`). It packages your own download of NVIDIA TensorRT 11.4 under `/opt/nvidia/tensorrt`, with an environment file `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env`. You download the archive from NVIDIA yourself, after accepting its licence; that repository's `docs/tensorrt.md` has the download, the options and the checks. The build and install commands are, from that repository:

  ```bash
  cd nvidia-tensorrt
  ./configure.sh --src ~/Downloads --write
  PKGEXT=.pkg.tar makepkg -f
  sudo pacman -U nvidia-tensorrt-sdk-local-11.4.0.106-1-x86_64.pkg.tar
  ```

- **The AUR's `tensorrt`**, which installs into `/usr`. Build and install it as you do any AUR package.

This recipe's build finds TensorRT in `/opt/nvidia/tensorrt` when that holds `include/NvInfer.h`, and otherwise in `/usr`; `ORTF_TENSORRT_HOME` overrides this ([build.md](build.md#build-inputs)). The PKGBUILD requires TensorRT 11: its provider has the code paths for it.

## 3. This package

```bash
cd onnxruntime-full
MAKEFLAGS=-j8 makepkg -f
sudo pacman -U onnxruntime-full-1.29.0-1-x86_64.pkg.tar.zst python-onnxruntime-full-1.29.0-1-x86_64.pkg.tar.zst
```

`makepkg` downloads the upstream source and the pinned dependencies as its sources, then builds both packages. A build with CUDA takes hours and needs tens of GiB of disk; choose `-jN` for your memory, and see [build.md](build.md#parallelism-memory-and-disk) before the first run. If you installed `ccache` it is used automatically.

Install both packages in one `pacman -U`. If an Arch ONNX Runtime variant (`onnxruntime`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and so on, and their `python-` packages) is installed, pacman reports the conflict and offers to remove it; programs built against it keep working, because this package provides `onnxruntime`, `python-onnxruntime` and `libonnxruntime.so` at the same version.

For a CPU-only package, or any other combination, set the build inputs first: [build.md](build.md#build-inputs).

## 4. Verify

```bash
scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU
```

The test lists the providers and runs a small model on every engine and device it finds, comparing the numbers with a reference. `--require` takes a comma-separated list of targets (`cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU`) and exits 1 unless each is `ok`. Without `--require`, an engine whose runtime or device is missing is reported `unavailable`, not as a failure. List only the engines your machine has, for example `--require cpu,cuda` on a machine without Intel hardware.

The wrapper needs a C compiler and Python 3. `--only` and the other options, and how to read the output (`ok`, `FAIL`, `unavailable`), are in [tools/ort-smoke/README.md](../tools/ort-smoke/README.md).

A TensorRT `ok` does not show that TensorRT's built-in plugins load; see [engines.md](engines.md#tensorrt).

## Uninstalling

```bash
sudo pacman -R python-onnxruntime-full onnxruntime-full
```

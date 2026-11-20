# Installing

There are two ways to get the packages: install the prebuilt ones from the AUR (or from a release), or build them from the recipe ([build-from-source.md](build-from-source.md)). Both follow the same order, because the TensorRT provider and every provider package need their vendor runtime first:

```mermaid
flowchart LR
    A[1. vendor runtimes] --> B[2. TensorRT]
    B --> C[3. the packages: prebuilt or built]
    C --> D[4. verify]
```

Skip the runtimes of the engines you do not want. A CPU-only installation (`onnxruntime-core`) needs no vendor runtime and no step 2. Every command here is run by you; nothing in this repository runs `sudo` or installs anything. Terms are in the [glossary](glossary.md).

## 1. Prerequisites

Install the runtimes of the engines you use (vendor licences apply: see the [README](../README.md#licence-and-redistribution)):

```bash
# CUDA and TensorRT providers
sudo pacman -S --needed nvidia-utils cuda cudnn

# OpenVINO provider: CPU, Intel GPU and Intel NPU devices
sudo pacman -S --needed openvino openvino-intel-gpu-plugin openvino-intel-npu-plugin intel-compute-runtime level-zero-loader intel-npu-driver

# MIGraphX provider runtimes (AMD GPUs; untested on AMD hardware). The provider and its Python add-on are AUR source packages: amd.md#installing
sudo pacman -S --needed migraphx hip-runtime-amd
```

The oneDNN provider (`onnxruntime-provider-dnnl`) is optional and needs no runtime installed: oneDNN is inside the package. It is meant for Intel server CPUs (Xeon with AVX-512 or AMX) and int8 or bf16 models; on desktops and laptops use the CPU provider, or OpenVINO on Intel, and on AMD CPUs the CPU provider ([engines.md](engines.md#onednn)). `onnxruntime-core` and the CUDA package depend on `nccl`, which pacman installs from `extra` with them, and which needs no step of its own (364.18 MiB installed, on every machine, NVIDIA or not; a source build with `ORTF_NCCL=0` avoids it: [build.md](build.md#nccl-and-the-nccloff-packages)).

`nvidia-utils` is the NVIDIA driver's user-space part; you need it at run time, not to build. Each provider package depends on its runtimes, so it cannot be installed without them:

| Provider package | Depends on |
|---|---|
| `onnxruntime-provider-cuda<M>-<sfx>` | `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-tensorrt-cuda<M>` | the same, plus `tensorrt` (step 2) |
| `onnxruntime-provider-openvino` | `openvino` |
| `onnxruntime-provider-dnnl` (optional) | nothing beyond `libgomp` (`gcc-libs`) |
| `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd` |
| `python-onnxruntime-provider-migraphx` (the Python add-on) | `onnxruntime-provider-migraphx` at the same version, `python-onnxruntime-core`, `python`; no vendor runtime of its own |

Install the runtimes before step 3. Their dependencies carry version bounds, so pacman holds back a vendor upgrade that would break the provider until the packages are rebuilt ([dependencies.md](dependencies.md)). The OpenVINO device plugins and drivers (`openvino-intel-gpu-plugin`, `openvino-intel-npu-plugin`, `intel-compute-runtime`, `level-zero-loader`, `intel-npu-driver`) are optional dependencies of `onnxruntime-provider-openvino`, because each device needs only its own.

For the Intel NPU, check that the kernel's `intel_vpu` driver is loaded and that your user can open `/dev/accel/accel0` (look at the device's group with `ls -l /dev/accel/accel0`; on many systems it is `render`, so your user would need to be in that group). Check the Intel GPU's render node, `/dev/dri/renderD*`, in the same way.

The build tools (`base-devel`, `cmake`, `ninja` and so on) are only for building from source: [build-from-source.md](build-from-source.md#build-tools).

## 2. TensorRT

Only for the TensorRT provider. TensorRT is not in Arch's repositories. Install one of:

- **`nvidia-tensorrt-sdk-local`** from the [omarchy-nvidia-sdks](https://github.com/rsd/omarchy-nvidia-sdks) repository (directory `nvidia-tensorrt`). It packages your own download of NVIDIA TensorRT 11.4 under `/opt/nvidia/tensorrt`, with an environment file `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env`. You download the archive from NVIDIA yourself, after accepting its licence; that repository's `docs/tensorrt.md` has the download, the options and the checks. The build and install commands, from that repository:

  ```bash
  cd nvidia-tensorrt
  ./configure.sh --src ~/Downloads --write
  PKGEXT=.pkg.tar makepkg -f
  sudo pacman -U nvidia-tensorrt-sdk-local-11.4.0.106-2-x86_64.pkg.tar
  ```

  Use pkgrel 2 or later. Pkgrel 2 provides `tensorrt=11.4.0.106` as well as `nvidia-tensorrt-sdk`. Pkgrel 1 provides only `nvidia-tensorrt-sdk`, so it does not satisfy the provider package's `tensorrt` dependency; install pkgrel 2 over it.

- **The AUR's `tensorrt`**, which installs into `/usr` and provides `tensorrt`. Build and install it as you do any AUR package. Its major.minor must be that of the TensorRT the provider package was built against (below).

The recipe requires TensorRT 11: its provider has the code paths for it. The TensorRT provider package depends on `tensorrt`, bounded to the major.minor of the TensorRT the build used (`tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4). Either source provides the name, so the package does not care which one supplies it, only that the version is inside the bound ([dependencies.md](dependencies.md#tensorrt-one-dependency-name-for-two-sources)).

When building from source, the recipe finds TensorRT in `/opt/nvidia/tensorrt` when that holds `include/NvInfer.h`, and otherwise in `/usr`; `ORTF_TENSORRT_HOME` overrides this ([build.md](build.md#tensorrt-location)). The bound is read from `include/NvInferVersion.h` under the TensorRT root. When the TensorRT provider is built, the recipe's `makedepends` adds `tensorrt`, which a TensorRT installed from one of the packages above satisfies. A TensorRT unpacked by hand, with `ORTF_TENSORRT_HOME` pointing at it, needs `makepkg -d`, which skips the dependency check. `prepare()` warns when no installed package provides `tensorrt`, or when the installed one has another major.minor.

Install order: install `nvidia-tensorrt-sdk-local` pkgrel 2 (or later), or the AUR's `tensorrt`, before step 3. pacman refuses the TensorRT provider package while no installed package provides `tensorrt` within its bound.

## 3. This package

Step 3 installs the packages of this repository, prebuilt or built by you. Install the packages of one build or release in a single `pacman -U` (or one AUR helper call): the providers depend on `onnxruntime-core` at the same version.

### Prebuilt packages

The AUR package base `onnxruntime-split-bin` provides prebuilt versions of the packages. A `-bin` package installs a package that was built in advance, from this recipe in a clean chroot with its default inputs: the CUDA package is the preset `rtx` (`75 86 89 120`, Turing to Blackwell), and the CUDA package of the `all` preset (`75 80 86 89 90 100 103 120`, with A100, H100 and B200) is offered too. Each is named `<name>-bin`:

| Package | Replaces the source-built |
|---|---|
| `onnxruntime-core-bin` | `onnxruntime-core` |
| `onnxruntime-provider-cuda13-rtx-bin` | `onnxruntime-provider-cuda13-rtx` (`75 86 89 120`) |
| `onnxruntime-provider-cuda13-all-bin` | `onnxruntime-provider-cuda13-all` (`75 80 86 89 90 100 103 120`); conflicts with the `rtx` one |
| `onnxruntime-provider-tensorrt-cuda13-bin` | `onnxruntime-provider-tensorrt-cuda13` |
| `onnxruntime-provider-openvino-bin` | `onnxruntime-provider-openvino` |
| `onnxruntime-provider-dnnl-bin` (optional) | `onnxruntime-provider-dnnl` |
| `python-onnxruntime-core-bin` | `python-onnxruntime-core` |

They hold only ONNX Runtime and its bundled open-source dependencies. The vendor runtimes are dependencies, installed from their own packages (steps 1 and 2): install those first, or let the AUR helper pull the ones that are in Arch's repositories. TensorRT is not: install it first (step 2), or the TensorRT provider package is refused.

**With an AUR helper**, name the packages of the engines you use (the core is always needed; `onnxruntime-provider-dnnl-bin` is optional and can be left out):

```bash
yay -S onnxruntime-core-bin onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-tensorrt-cuda13-bin onnxruntime-provider-openvino-bin onnxruntime-provider-dnnl-bin python-onnxruntime-core-bin
```

For the `all` preset, name `onnxruntime-provider-cuda13-all-bin` instead of the `rtx` one: they conflict, and one CUDA package is installed at a time.

**Without a helper**, install the release's files directly. Download the `.pkg.tar.zst` files and `SHA256SUMS` from <https://github.com/rsd/omarchy-onnxruntime/releases> (tag `v1.29.0-7`; the tag is `v<pkgver>-<pkgrel>`), check them, and install the ones you want in one `pacman -U`:

```bash
sha256sum -c --ignore-missing SHA256SUMS
sudo pacman -U onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst \
  onnxruntime-provider-cuda13-rtx-1.29.0-7-x86_64.pkg.tar.zst \
  onnxruntime-provider-tensorrt-cuda13-1.29.0-7-x86_64.pkg.tar.zst \
  onnxruntime-provider-openvino-1.29.0-7-x86_64.pkg.tar.zst \
  onnxruntime-provider-dnnl-1.29.0-7-x86_64.pkg.tar.zst \
  python-onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst
```

The oneDNN file (`onnxruntime-provider-dnnl`) is optional: leave it out unless you run an Intel server CPU, as above.

Alternatively clone the AUR package (`git clone https://aur.archlinux.org/onnxruntime-split-bin.git`) and run `makepkg` in it. That downloads the release files and writes the `-bin` packages; install the ones you want with `sudo pacman -U`. Do not use `makepkg -i`, which would try to install all of them, including both conflicting CUDA packages.

The release's files are named without `-bin`, and they carry the same dependencies, so they work with each other. They do not satisfy a `-bin` package's name, so do not mix `pacman -U` of a release file with the AUR's `-bin` package of the same name. If you have source-built packages of the same names, pacman offers to replace them.

A prebuilt package is not a source build with other inputs: for another CUDA architecture list, or another engine set, build from the recipe ([build-from-source.md](build-from-source.md)). The MIGraphX provider and its Python add-on are not prebuilt, because the provider has never run on AMD hardware: they are source packages on the AUR ([amd.md](amd.md#installing)). For an AMD GPU, `yay -S onnxruntime-split-migraphx` builds both from that pkgbase, after the main line's core is installed. There are no prebuilt packages without NCCL (`-nccloff`). The CUDA 12 provider for Pascal and Volta cards is a separate AUR package base, `onnxruntime-split-cuda12-bin`, and needs a different CUDA on the machine: see [Older NVIDIA GPUs](#older-nvidia-gpus-pascal-and-volta-cuda-12). The prebuilt packages have the same dependency bounds as a source build ([dependencies.md](dependencies.md)): pacman refuses a vendor upgrade that goes past one, and a rebuilt release follows. Then continue at [step 4](#4-verify).

### Building from source

The prerequisites of this page, then [build-from-source.md](build-from-source.md): build tools, `makepkg`, the exact `pacman -U` command, and the checks of the build tree.

### Upgrading from the single package 1.29.0-1

Version 1.29.0-1 was one `onnxruntime-full` that owned every provider library, with `python-onnxruntime-full` for the Python module. `onnxruntime-core` has `replaces=(onnxruntime-full)` and `python-onnxruntime-core` has `replaces=(python-onnxruntime-full)`, and each conflicts with the name it replaces. `replaces=` acts only on a sync upgrade (`pacman -Syu` from a repository). With `pacman -U`, the conflicts make pacman offer to remove the old packages in the same transaction, and a file that moves from a package removed in that transaction to a new one is not a file conflict.

Install every new file in one `pacman -U` (as in the commands of [Prebuilt packages](#prebuilt-packages) and [build-from-source.md](build-from-source.md#3-build-and-install)), not in several: pacman moves a file from one package to another only within one transaction. Answer y to its offer to remove `onnxruntime-full` and `python-onnxruntime-full`: that prompt defaults to N, so `--noconfirm` aborts the upgrade. The old `python-onnxruntime-full` depends on `onnxruntime-full=1.29.0-1`, so `python-onnxruntime-core` must be in the same transaction. With `ORTF_PYTHON=0` there is no new Python package to replace it with: remove it first with `sudo pacman -R python-onnxruntime-full`, or the transaction fails on that dependency.

### Upgrading from 1.29.0-6

Version 1.29.0-7 builds the core with the training operators and NCCL. That changes the interface between the core and the providers, and a provider built without them crashes in a core built with them, so install every new file of the release in one `pacman -U` (or one AUR helper call), as above. The new core also depends on `nccl`, which pacman installs from `extra`. If you use the CUDA 12 provider for Pascal and Volta, `onnxruntime-core` 1.29.0-7 conflicts with its 1.29.0-1 package: upgrade it to 1.29.0-2 or later in the same transaction ([cuda12-legacy.md](cuda12-legacy.md#upgrading)).

### Dependencies and version bounds

The vendor runtimes carry version bounds, so that pacman refuses an update that would break a provider until the packages are rebuilt. The policy, the bounds and `scripts/check-deps.sh` are in [dependencies.md](dependencies.md).

#### Checking built packages: `scripts/check-deps.sh`

`scripts/check-deps.sh <package files>` lists every library of a package, its owner, and whether the installed versions are inside the bounds: [dependencies.md](dependencies.md#checking-built-packages-scriptscheck-depssh).

## 4. Verify

### Before installing: the build tree

For a build from source, the build tree can be checked before `pacman -U` replaces anything: [build-from-source.md](build-from-source.md#before-installing-the-build-tree).

### After installing

```bash
scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU,dnnl
```

The test lists the providers and runs a small model on every engine and device it finds, comparing the numbers with a reference. `--require` takes a comma-separated list of targets (`cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU`, `dnnl`) and exits 1 unless each is `ok`. The target `migraphx-load` is for the MIGraphX package and is not in the default run: `scripts/smoke-test.sh --only cpu,migraphx-load` ([amd.md](amd.md#using-it)). Without `--require`, an engine whose runtime or device is missing is reported `unavailable`, not as a failure. List only the engines your machine has, for example `--require cpu,cuda` on a machine without Intel hardware.

The script is in this repository (`scripts/smoke-test.sh`); a prebuilt-only installation needs a clone to run it. The wrapper needs a C compiler and Python 3. `--only` and the other options, and how to read the output (`ok`, `FAIL`, `unavailable`), are in [tools/ort-smoke/README.md](../tools/ort-smoke/README.md).

The model check (`tools/ort-model-check/model-check.py model.onnx`, with a model of your own) also runs against the installed package, with `python-onnxruntime-core` and NumPy installed and without `PYTHONPATH`. Results of both tools on a full build are in [results.md](results.md).

A TensorRT `ok` does not show that TensorRT's built-in plugins load; see [engines.md](engines.md#tensorrt).

## Older NVIDIA GPUs (Pascal and Volta): CUDA 12

CUDA 13, cuDNN 9.11 and the 590 driver series dropped Pascal and Volta, so the packages above cannot use those cards. The CUDA 12 provider `onnxruntime-provider-cuda12-legacy` serves them. It replaces the CUDA 13 setup on that machine, swaps Arch's `nccl` for the AUR's `nccl-cuda12.9`, and is untested on Pascal and Volta hardware. The install steps, the swap from CUDA 13 and the way back are in [cuda12-legacy.md](cuda12-legacy.md#installing).

## Uninstalling

```bash
sudo pacman -R python-onnxruntime-core onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-cuda13-rtx onnxruntime-provider-openvino onnxruntime-provider-dnnl onnxruntime-core
```

Add `python-onnxruntime-provider-migraphx` and `onnxruntime-provider-migraphx` if you built them. The CUDA 12 legacy package, if installed, replaces the CUDA 13 one in that list (`onnxruntime-provider-cuda12-legacy`), and has no TensorRT package. For the prebuilt packages, name the `-bin` packages instead, and the `all` CUDA package (`onnxruntime-provider-cuda13-all`) if that is the one you installed. Name the packages you installed: the provider packages depend on `onnxruntime-core`, so it cannot be removed while one remains. `pacman -Qq | grep onnxruntime` lists them. `nccl`, which the core pulled in, stays installed until you remove it; `pacman -Qdtq` lists such leftovers.

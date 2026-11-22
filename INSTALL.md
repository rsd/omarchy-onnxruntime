# Install

This guide installs the ONNX Runtime 1.29.0 packages from this repository on Arch Linux or Omarchy: the core library plus one package for each engine your machine has. Prebuilt packages from the AUR (the Arch User Repository) are the fast path. Building from the recipe is the alternative. This set replaces Arch's own `onnxruntime-*` packages.

Terms such as pkgrel, `-bin`, SM and execution provider (EP) are in the [glossary](docs/glossary.md). You run every command. Nothing here runs `sudo` or installs for you.

## Before you start

- **Arch Linux or Omarchy, on x86_64.** The packages are pacman packages.
- **An AUR helper** such as `yay`, and `sudo`. Without a helper, you can install the release files with `pacman -U` ([step 4](#4-install-the-packages)).
- **The vendor runtimes come from their own packages.** CUDA, cuDNN, NCCL, OpenVINO, MIGraphX and the HIP runtime are in Arch's repositories. TensorRT is not ([step 3](#3-install-tensorrt)). [Step 2](#2-install-the-vendor-runtimes) installs them.
- **Free disk.** The sizes are in [docs/packages.md](docs/packages.md#sizes). `nccl` is 364.18 MiB installed, on every machine. The vendor runtimes add more.
- **No other ONNX Runtime set.** Arch's `onnxruntime`, `onnxruntime-cuda` and the other Arch variants conflict with this set, because they install the same `libonnxruntime.so`. Pacman offers to remove them. Programs built against Arch's package keep working ([docs/choosing.md](docs/choosing.md#why-one-set-per-machine)).

## 1. Choose your packages

The core, `onnxruntime-core`, is always needed. It runs on any x86-64 CPU and needs no vendor runtime. Add one provider package for each engine you want.

| You have | Install | Vendor runtime you need first |
|---|---|---|
| Any x86-64 CPU | `onnxruntime-core` | none |
| NVIDIA Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx` | `nvidia-utils`, `cuda`, `cudnn` |
| NVIDIA A100, H100 or B200-class GPU | `onnxruntime-provider-cuda13-all` instead of `-rtx` | the same |
| NVIDIA GPU, with TensorRT | `onnxruntime-provider-tensorrt-cuda13`, beside a CUDA 13 package | the same, plus TensorRT ([step 3](#3-install-tensorrt)) |
| Intel CPU, GPU or NPU | `onnxruntime-provider-openvino` | `openvino`, plus a plugin and driver for each Intel GPU or NPU |
| Intel Xeon with AVX-512 or AMX (optional) | `onnxruntime-provider-dnnl` | none (oneDNN is built in) |
| Python | `python-onnxruntime-core` | none |
| NVIDIA Pascal or Volta | `onnxruntime-provider-cuda12-legacy` | CUDA 12 userland: [Older NVIDIA GPUs](#older-nvidia-gpus-pascal-and-volta-cuda-12) |
| AMD GPU | `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd`: [amd.md](docs/amd.md#installing) |

Not sure which engine fits your hardware? See [docs/hardware.md](docs/hardware.md) for coverage and [docs/engines.md](docs/engines.md) for what each engine needs. [docs/choosing.md](docs/choosing.md) compares this set with Arch's own `onnxruntime-*` packages.

Combine engines freely. A laptop with an Intel CPU, an NVIDIA GPU and an NPU takes the core and the CUDA, TensorRT and OpenVINO packages.

### Prebuilt or built from source

| | Prebuilt (`-bin`) | Built from source |
|---|---|---|
| AUR package base | `onnxruntime-split-bin` | `onnxruntime-split` |
| Package names | `<name>-bin`, such as `onnxruntime-core-bin` | `<name>`, such as `onnxruntime-core` |
| How you get it | Download of the files built in a clean chroot and published as [GitHub releases](https://github.com/rsd/omarchy-onnxruntime/releases) | `makepkg` compiles the recipe on your machine |
| CUDA architectures | `rtx` (`75 86 89 120`) or `all` (`75 80 86 89 90 100 103 120`) | any list you set |
| Build cost | none | a full build takes an hour or more and needs tens of GiB of disk ([build.md](docs/build.md#parallelism-memory-and-disk)) |
| Without NCCL (`-nccloff`) | not offered | yes ([docs/build.md](docs/build.md#nccl-and-the-nccloff-packages)) |

Choose prebuilt unless you need another CUDA architecture list, another engine set or a build without NCCL. For a source build, follow [docs/build-from-source.md](docs/build-from-source.md) after steps 2 and 3.

`-bin` packages and source-built packages have the same dependencies and version bounds. The packages hold no NVIDIA, Intel or AMD library: you install the vendor runtimes from their own packages ([licence](README.md#licence-and-redistribution)).

## 2. Install the vendor runtimes

Install the runtimes of the engines you chose. Vendor licences apply. Skip this step for a CPU-only installation.

```bash
# CUDA and TensorRT providers
sudo pacman -S --needed nvidia-utils cuda cudnn

# OpenVINO provider: CPU, Intel GPU and Intel NPU devices
sudo pacman -S --needed openvino openvino-intel-gpu-plugin openvino-intel-npu-plugin intel-compute-runtime level-zero-loader intel-npu-driver

# MIGraphX provider (AMD GPUs; untested on AMD hardware)
sudo pacman -S --needed migraphx hip-runtime-amd
```

`nvidia-utils` is the NVIDIA driver's user-space part. You need it at run time, not to build.

Each provider package depends on its runtimes, so pacman refuses to install it without them:

| Provider package | Depends on |
|---|---|
| `onnxruntime-provider-cuda<M>-<sfx>` | `cuda`, `cudnn`, `nvidia-utils` |
| `onnxruntime-provider-tensorrt-cuda<M>` | the same, plus `tensorrt` |
| `onnxruntime-provider-openvino` | `openvino` |
| `onnxruntime-provider-dnnl` (optional) | nothing beyond `libgomp` (`gcc-libs`) |
| `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd` |
| `python-onnxruntime-provider-migraphx` (the Python add-on) | `onnxruntime-provider-migraphx` at the same version, `python-onnxruntime-core`, `python`; no vendor runtime of its own |

`<M>` is the CUDA major version (`13`). `<sfx>` names the GPU architectures (`rtx`, `all`).

Notes:

- **OpenVINO devices.** The device plugins and drivers are optional dependencies of `onnxruntime-provider-openvino`, because each device needs only its own.
- **Intel NPU.** Check that the kernel's `intel_vpu` driver is loaded and that your user can open `/dev/accel/accel0`. Read the device's group with this command. On many systems it is `render`, so your user must be in that group:

  ```bash
  ls -l /dev/accel/accel0
  ```

- **Intel GPU.** Check the render node, `/dev/dri/renderD*`, the same way.
- **oneDNN.** The oneDNN provider is optional and needs no installed runtime. It is meant for Intel server CPUs (Xeon with AVX-512 or AMX) and int8 or bf16 models. On desktops and laptops use the CPU provider, or OpenVINO on Intel. On AMD CPUs use the CPU provider ([engines.md](docs/engines.md#onednn)).
- **NCCL.** `onnxruntime-core` and the CUDA package depend on `nccl`. Pacman installs it from `extra` with them. It is 364.18 MiB installed, on every machine, NVIDIA or not. A source build with `ORTF_NCCL=0` avoids it ([build.md](docs/build.md#nccl-and-the-nccloff-packages)).
- **Version bounds.** The vendor dependencies carry version bounds. Pacman holds back a vendor upgrade that would break a provider until the packages are rebuilt ([dependencies.md](docs/dependencies.md)).
- **Build tools.** `base-devel`, `cmake`, `ninja` and the rest are only for building from source: [build-from-source.md](docs/build-from-source.md#build-tools).

## 3. Install TensorRT

Do this step only for the TensorRT provider. TensorRT is not in Arch's repositories. Install one of these:

- **`nvidia-tensorrt-sdk-local`** from the [omarchy-nvidia-sdks](https://github.com/rsd/omarchy-nvidia-sdks) repository (directory `nvidia-tensorrt`). It packages your own download of NVIDIA TensorRT 11.4 under `/opt/nvidia/tensorrt`. From pkgrel 4 it adds a loader entry, so every program finds TensorRT without an environment file. It still ships `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env`, for compatibility. You download the archive from NVIDIA yourself, after accepting its licence. That repository's `docs/tensorrt.md` has the download, the options and the checks. The build and install commands, from that repository, where `<version>` is the version `makepkg` prints in the file name it writes:

  ```bash
  cd nvidia-tensorrt
  ./configure.sh --src ~/Downloads --write
  PKGEXT=.pkg.tar makepkg -f
  sudo pacman -U nvidia-tensorrt-sdk-local-<version>-x86_64.pkg.tar
  ```

  Use pkgrel 2 or later. Pkgrel 2 provides `tensorrt=11.4.0.106` and `nvidia-tensorrt-sdk`. Pkgrel 1 provides only `nvidia-tensorrt-sdk`, so it does not satisfy the provider package's `tensorrt` dependency. Pkgrel 4 or later also needs no environment file. Install a newer pkgrel over an older one.

- **The AUR's `tensorrt`**, which installs into `/usr` and provides `tensorrt`. Build and install it as you do any AUR package. Its major.minor must match the TensorRT the provider package was built against.

The recipe requires TensorRT 11: the provider has the code paths for it. The TensorRT provider package depends on `tensorrt`, bounded to the major.minor of the build (`tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4). Either source provides that name, so the package accepts either as long as the version is inside the bound ([dependencies.md](docs/dependencies.md#tensorrt-one-dependency-name-for-two-sources)).

Install TensorRT before the packages. Pacman refuses the TensorRT provider while no installed package provides `tensorrt` within its bound. To build the provider from source, read [how the recipe finds TensorRT](docs/build-from-source.md#tensorrt-for-a-source-build).

## 4. Install the packages

Install all packages of one build or release in a single `pacman -U` or a single AUR helper call. The providers depend on `onnxruntime-core` at the same version.

### Prebuilt packages

The AUR package base `onnxruntime-split-bin` provides the prebuilt packages. Each is named `<name>-bin`:

| Package | Replaces the source-built |
|---|---|
| `onnxruntime-core-bin` | `onnxruntime-core` |
| `onnxruntime-provider-cuda13-rtx-bin` | `onnxruntime-provider-cuda13-rtx` (`75 86 89 120`) |
| `onnxruntime-provider-cuda13-all-bin` | `onnxruntime-provider-cuda13-all` (`75 80 86 89 90 100 103 120`); conflicts with the `rtx` one |
| `onnxruntime-provider-tensorrt-cuda13-bin` | `onnxruntime-provider-tensorrt-cuda13` |
| `onnxruntime-provider-openvino-bin` | `onnxruntime-provider-openvino` |
| `onnxruntime-provider-dnnl-bin` (optional) | `onnxruntime-provider-dnnl` |
| `python-onnxruntime-core-bin` | `python-onnxruntime-core` |

Each `-bin` package installs a package built in advance from this recipe in a clean chroot with its default inputs.

**With an AUR helper** (`yay` here; any works), name the packages of the engines you use. The core is always needed. The oneDNN package is optional.

```bash
yay -S onnxruntime-core-bin onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-tensorrt-cuda13-bin onnxruntime-provider-openvino-bin onnxruntime-provider-dnnl-bin python-onnxruntime-core-bin
```

The helper installs the vendor runtimes that are in Arch's repositories. TensorRT is not, so install it first ([step 3](#3-install-tensorrt)). For the `all` preset, name `onnxruntime-provider-cuda13-all-bin` instead of the `rtx` one. They conflict, so install one CUDA package at a time.

**Without a helper**, install the release files directly:

1. Open <https://github.com/rsd/omarchy-onnxruntime/releases> and take the newest release of the main line. Its tag is `v<pkgver>-<pkgrel>`. The release notes and the file names carry the same version.
2. Download the `.pkg.tar.zst` files you want and `SHA256SUMS`.
3. Check the files:

   ```bash
   sha256sum -c --ignore-missing SHA256SUMS
   ```

4. Install the ones you want in one `pacman -U`. Replace `<version>` with the `<pkgver>-<pkgrel>` of the release. Leave out the oneDNN file unless you run an Intel server CPU:

   ```bash
   sudo pacman -U onnxruntime-core-<version>-x86_64.pkg.tar.zst \
     onnxruntime-provider-cuda13-rtx-<version>-x86_64.pkg.tar.zst \
     onnxruntime-provider-tensorrt-cuda13-<version>-x86_64.pkg.tar.zst \
     onnxruntime-provider-openvino-<version>-x86_64.pkg.tar.zst \
     onnxruntime-provider-dnnl-<version>-x86_64.pkg.tar.zst \
     python-onnxruntime-core-<version>-x86_64.pkg.tar.zst
   ```

You can also build the `-bin` packages from the AUR package. This downloads the release files and writes the `-bin` packages:

1. Clone the AUR package.

   ```bash
   git clone https://aur.archlinux.org/onnxruntime-split-bin.git
   cd onnxruntime-split-bin
   ```

2. Build the packages.

   ```bash
   makepkg
   ```

3. Install the ones you want.

   ```bash
   sudo pacman -U <the files you want>
   ```

Do not use `makepkg -i`: it tries to install all of them, including both conflicting CUDA packages.

The release files are named without `-bin` and carry the same dependencies, so they work with each other. They do not satisfy a `-bin` package's name. Do not mix `pacman -U` of a release file with the AUR's `-bin` package of the same name. If you have source-built packages of the same names, pacman offers to replace them.

What the prebuilt set leaves out:

- **MIGraphX.** The provider and its Python add-on are not prebuilt, because the provider has never run on AMD hardware. They are source packages on the AUR. For an AMD GPU, this command builds both from that package base, after the core is installed ([amd.md](docs/amd.md#installing)):

  ```bash
  yay -S onnxruntime-split-migraphx
  ```

- **No NCCL.** There are no prebuilt `-nccloff` packages.
- **CUDA 12.** The provider for Pascal and Volta is a separate AUR package base, `onnxruntime-split-cuda12-bin`, and needs a different CUDA on the machine: see [below](#older-nvidia-gpus-pascal-and-volta-cuda-12).
- **Other builds.** For another CUDA architecture list or engine set, build from the recipe.

The prebuilt packages have the same dependency bounds as a source build ([dependencies.md](docs/dependencies.md)). Pacman refuses a vendor upgrade that goes past one, and a rebuilt release follows.

### Build from source

Finish steps 2 and 3. Then follow [docs/build-from-source.md](docs/build-from-source.md): build tools, `makepkg`, the exact `pacman -U` command, and the checks of the build tree.

The recipes fetch the ONNX Runtime source from this project's `source-v1.29.0` release, because `github.com/microsoft/onnxruntime` has returned 404 since 2026-11-17. `ORTF_SOURCE=upstream` selects the git tag again ([build.md](docs/build.md#the-onnx-runtime-source-mirror-or-upstream)). The prebuilt packages are not affected.

## Check that it works

Run these checks after you install. The first three need no clone of this repository.

1. List the installed packages and their versions:

   ```bash
   pacman -Q | grep onnxruntime
   ```

   You see `onnxruntime-core` (or `onnxruntime-core-bin`) and one line for each provider package you installed. The main-line packages share one version; the CUDA 12 and MIGraphX providers come from their own recipes and carry their own release number:

   ```text
   onnxruntime-core 1.29.0-7
   onnxruntime-provider-cuda13-rtx 1.29.0-7
   onnxruntime-provider-openvino 1.29.0-7
   python-onnxruntime-core 1.29.0-7
   ```

2. List the engine libraries beside the core:

   ```bash
   ls /usr/lib/libonnxruntime*.so
   ```

   You see `libonnxruntime.so` and one `libonnxruntime_providers_<engine>.so` for each provider package you installed (`cuda`, `tensorrt`, `openvino`, `dnnl`). `libonnxruntime_providers_shared.so` is part of the core.

3. Ask the Python module for its engines:

   ```bash
   python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
   ```

   You see the version `1.29.0` and one provider name for each engine package you installed. On a machine with every engine:

   ```text
   1.29.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DnnlExecutionProvider', 'CPUExecutionProvider']
   ```

4. Optional: run the smoke test from a clone of this repository. It runs a small model on every engine and device it finds and compares the numbers with a reference. The wrapper needs a C compiler and Python 3.

   ```bash
   scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU,dnnl
   ```

   You see one row per target with the status `ok`, `FAIL` or `unavailable`. On a machine with every engine:

   ```text
   target         status           create ms   first ms    warm ms  max error
   cpu            ok                     9.9       0.46       0.01   0.00e+00
   cuda           ok                   561.9     306.53       0.04   0.00e+00
   tensorrt       ok                  3290.9       2.25       0.03   0.00e+00
   openvino:CPU   ok                   103.0       0.46       0.03   0.00e+00
   openvino:GPU   ok                   283.9       0.25       0.24   0.00e+00
   openvino:NPU   ok                    24.8      76.56       0.47   0.00e+00
   dnnl           ok                    17.5      29.91       3.55   0.00e+00

   7 ok, 0 failed, 0 unavailable
   ```

   The times differ on your machine. `--require` takes a comma-separated list of targets and exits 1 unless each is `ok`. List only the engines your machine has, for example `--require cpu,cuda` on a machine without Intel hardware. Without `--require`, an engine whose runtime or device is missing shows `unavailable`, not a failure.

Valid smoke-test targets are `cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU` and `dnnl`. The target `migraphx-load` is for the MIGraphX package and is not in the default run ([amd.md](docs/amd.md#using-it)):

```bash
scripts/smoke-test.sh --only cpu,migraphx-load
```

The options, and how to read `ok`, `FAIL` and `unavailable`, are in [tools/ort-smoke/README.md](tools/ort-smoke/README.md). A TensorRT `ok` does not show that TensorRT's built-in plugins load. See [engines.md](docs/engines.md#tensorrt).

To run a real model on each engine, use the model check with a model of your own, with `python-onnxruntime-core` and NumPy installed and without `PYTHONPATH`:

```bash
tools/ort-model-check/model-check.py model.onnx
```

Results of both tools on a full build are in [docs/results.md](docs/results.md).

### Before installing a source build

For a build from source, check the build tree before `pacman -U` replaces anything: [build-from-source.md](docs/build-from-source.md#before-installing-the-build-tree).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Pacman refuses the TensorRT provider | No installed package provides `tensorrt` within the provider's bound | Install `nvidia-tensorrt-sdk-local` pkgrel 2 or later, or the AUR's `tensorrt`, first ([step 3](#3-install-tensorrt)) |
| Pacman reports a conflict with `onnxruntime`, `onnxruntime-cuda` or another Arch variant | Arch's packages install the same `libonnxruntime.so` | Answer `y` to remove them. Programs built against Arch's package keep working ([docs/choosing.md](docs/choosing.md#why-one-set-per-machine)) |
| Upgrade aborts with `--noconfirm` | The question about removing `onnxruntime-full` defaults to N | Run `pacman -U` without `--noconfirm` and answer `y` ([upgrades](#upgrade-from-the-single-package-1290-1)) |
| Crash after a partial upgrade | A provider built without training and NCCL loaded into a core built with them | Install every file of the release in one `pacman -U` ([upgrades](#upgrade-from-1290-6)) |
| Two CUDA packages conflict | The `rtx` and `all` packages install the same library | Install one CUDA package at a time |
| Pacman holds back a CUDA, cuDNN, TensorRT or OpenVINO upgrade | The provider's version bound | Wait for the rebuilt packages. `scripts/check-deps.sh` shows which bound moved ([dependencies.md](docs/dependencies.md#the-version-bounds)) |
| Smoke test says `unavailable` for an engine | The provider package or its vendor runtime is missing. ONNX Runtime fails with "cannot open shared object file" | Install the provider package and its runtime |
| `Device GPU is not available` (or `NPU`) | The OpenVINO plugin or driver is missing, or your user cannot open the device | Install the plugin and driver, and check `/dev/dri/renderD*` or `/dev/accel/accel0` ([step 2](#2-install-the-vendor-runtimes)) |
| `import onnxruntime.training` fails with `No module named 'torch'` | The training modules import PyTorch | Install `python-pytorch`. `import onnxruntime` needs no PyTorch |
| TensorRT logs that the default plugin library was not found | TensorRT under `/opt` is not on `LD_LIBRARY_PATH` | Load the TensorRT environment file for that program only ([engines.md](docs/engines.md#tensorrt)) |
| A program crashes at exit after using OpenVINO | It released the ONNX Runtime environment | Keep the environment for the life of the process ([engines.md](docs/engines.md#openvino)) |
| `makepkg -i` tries to install both CUDA packages | `-i` installs every package it builds | Install the files you want with `pacman -U` |
| Source build conflicts on `-cuda13-sm120` | An older local build used that name | Answer `y` to remove it ([build-from-source.md](docs/build-from-source.md#naming-the-files)) |

## Uninstall

Name the packages you installed. The provider packages depend on `onnxruntime-core`, so it cannot be removed while one remains. This command lists them:

```bash
pacman -Qq | grep onnxruntime
```

Remove them:

```bash
sudo pacman -R python-onnxruntime-core onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-cuda13-rtx onnxruntime-provider-openvino onnxruntime-provider-dnnl onnxruntime-core
```

Adjust the list:

- Add `python-onnxruntime-provider-migraphx` and `onnxruntime-provider-migraphx` if you built them.
- Name the `-bin` packages if you installed the prebuilt ones.
- Use `onnxruntime-provider-cuda13-all` if that is the CUDA package you installed.
- Replace the CUDA 13 package with `onnxruntime-provider-cuda12-legacy` if you use the legacy provider. It has no TensorRT package.

`nccl`, which the core pulled in, stays installed until you remove it. This command lists such leftovers:

```bash
pacman -Qdtq
```

## Details

### Upgrade from the single package 1.29.0-1

Version 1.29.0-1 was one `onnxruntime-full` package that owned every provider library, with `python-onnxruntime-full` for the Python module. `onnxruntime-core` has `replaces=(onnxruntime-full)` and `python-onnxruntime-core` has `replaces=(python-onnxruntime-full)`. Each also conflicts with the name it replaces.

- `replaces=` acts only on a sync upgrade (`pacman -Syu` from a repository).
- With `pacman -U`, the conflicts make pacman offer to remove the old packages in the same transaction. A file that moves from a package removed in that transaction to a new one is not a file conflict.
- Install every new file in one `pacman -U`, not in several. Pacman moves a file from one package to another only within one transaction.
- Answer `y` to the offer to remove `onnxruntime-full` and `python-onnxruntime-full`. That question defaults to N, so `--noconfirm` aborts the upgrade.
- The old `python-onnxruntime-full` depends on `onnxruntime-full=1.29.0-1`, so `python-onnxruntime-core` must be in the same transaction.
- With `ORTF_PYTHON=0` there is no new Python package to replace it. Remove it first, or the transaction fails on that dependency:

  ```bash
  sudo pacman -R python-onnxruntime-full
  ```

### Upgrade from 1.29.0-6

Version 1.29.0-7 builds the core with the training operators and NCCL. That changes the interface between the core and the providers. A provider built without them crashes in a core built with them. Install every new file of the release in one `pacman -U` (or one AUR helper call), as in [step 4](#4-install-the-packages).

The new core also depends on `nccl`, which pacman installs from `extra`. If you use the CUDA 12 provider for Pascal and Volta, `onnxruntime-core` 1.29.0-7 conflicts with its 1.29.0-1 package. Upgrade it to 1.29.0-2 or later in the same transaction ([cuda12-legacy.md](docs/cuda12-legacy.md#upgrading)).

### Check built packages against your system

This command lists every library of a package, its owner, and whether the installed versions are inside the bounds:

```bash
scripts/check-deps.sh <package files>
```

See [dependencies.md](docs/dependencies.md#checking-built-packages-scriptscheck-depssh).

### Older NVIDIA GPUs (Pascal and Volta): CUDA 12

CUDA 13, cuDNN 9.11 and the 590 driver series dropped Pascal and Volta, so the CUDA 13 packages cannot use those cards. The CUDA 12 provider `onnxruntime-provider-cuda12-legacy` serves them. It replaces the CUDA 13 setup on that machine and swaps Arch's `nccl` for the AUR's `nccl-cuda12.9`. It is untested on Pascal and Volta hardware. The install steps, the swap from CUDA 13 and the way back are in [cuda12-legacy.md](docs/cuda12-legacy.md#installing).

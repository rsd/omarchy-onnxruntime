# Install

This guide installs the ONNX Runtime 1.29.0 packages from this repository on Arch Linux: the core library plus one package for each engine your machine has. Prebuilt packages from the AUR are the fast path. Building from the recipe is the alternative.

Terms such as pkgrel, `-bin`, SM and execution provider (EP) are in the [glossary](docs/glossary.md). Every command here is run by you. Nothing in this repository runs `sudo` or installs anything.

## Choose your packages

The core, `onnxruntime-core`, is always needed. It runs on any x86-64 CPU and needs no vendor runtime. Add one provider package for each engine you want.

| You have | Install | Vendor runtime you need first |
|---|---|---|
| Any x86-64 CPU | `onnxruntime-core` | none |
| NVIDIA Turing (RTX 20, GTX 16) to Blackwell (RTX 50) | `onnxruntime-provider-cuda13-rtx` | `nvidia-utils`, `cuda`, `cudnn` |
| NVIDIA A100, H100 or B200-class GPU | `onnxruntime-provider-cuda13-all` instead of `-rtx` | the same |
| NVIDIA GPU, with TensorRT | `onnxruntime-provider-tensorrt-cuda13`, beside a CUDA 13 package | the same, plus TensorRT ([Install TensorRT](#install-tensorrt)) |
| Intel CPU, GPU or NPU | `onnxruntime-provider-openvino` | `openvino`, plus a plugin and driver for each Intel GPU or NPU |
| Intel Xeon with AVX-512 or AMX (optional) | `onnxruntime-provider-dnnl` | none (oneDNN is built in) |
| Python | `python-onnxruntime-core` | none |
| NVIDIA Pascal or Volta | `onnxruntime-provider-cuda12-legacy` | CUDA 12 userland: [Older NVIDIA GPUs](#older-nvidia-gpus-pascal-and-volta-cuda-12) |
| AMD GPU | `onnxruntime-provider-migraphx` | `migraphx`, `hip-runtime-amd`: [amd.md](docs/amd.md#installing) |

Not sure which engine fits your hardware? See [docs/hardware.md](docs/hardware.md) for coverage and [docs/engines.md](docs/engines.md) for what each engine needs. [docs/choosing.md](docs/choosing.md) compares this set with Arch's own `onnxruntime-*` packages. The two sets conflict, so install one per machine.

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

Choose prebuilt unless you need another CUDA architecture list, another engine set or a build without NCCL. For a source build, follow [docs/build-from-source.md](docs/build-from-source.md) after steps 1 and 2 below.

`-bin` packages and source-built packages have the same dependencies and version bounds. The packages hold no NVIDIA, Intel or AMD library: you install the vendor runtimes from their own packages ([licence](README.md#licence-and-redistribution)).

## Install the vendor runtimes

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
- **Intel NPU.** Check that the kernel's `intel_vpu` driver is loaded and that your user can open `/dev/accel/accel0`. Read the device's group with `ls -l /dev/accel/accel0`. On many systems it is `render`, so your user must be in that group.
- **Intel GPU.** Check the render node, `/dev/dri/renderD*`, the same way.
- **oneDNN.** The oneDNN provider is optional and needs no installed runtime. It is meant for Intel server CPUs (Xeon with AVX-512 or AMX) and int8 or bf16 models. On desktops and laptops use the CPU provider, or OpenVINO on Intel. On AMD CPUs use the CPU provider ([engines.md](docs/engines.md#onednn)).
- **NCCL.** `onnxruntime-core` and the CUDA package depend on `nccl`. Pacman installs it from `extra` with them. It is 364.18 MiB installed, on every machine, NVIDIA or not. A source build with `ORTF_NCCL=0` avoids it ([build.md](docs/build.md#nccl-and-the-nccloff-packages)).
- **Version bounds.** The vendor dependencies carry version bounds. Pacman holds back a vendor upgrade that would break a provider until the packages are rebuilt ([dependencies.md](docs/dependencies.md)).
- **Build tools.** `base-devel`, `cmake`, `ninja` and the rest are only for building from source: [build-from-source.md](docs/build-from-source.md#build-tools).

## Install TensorRT

Do this step only for the TensorRT provider. TensorRT is not in Arch's repositories. Install one of these:

- **`nvidia-tensorrt-sdk-local`** from the [omarchy-nvidia-sdks](https://github.com/rsd/omarchy-nvidia-sdks) repository (directory `nvidia-tensorrt`). It packages your own download of NVIDIA TensorRT 11.4 under `/opt/nvidia/tensorrt`. It adds an environment file, `/usr/share/nvidia-tensorrt/nvidia-tensorrt.env`. You download the archive from NVIDIA yourself, after accepting its licence. That repository's `docs/tensorrt.md` has the download, the options and the checks. The build and install commands, from that repository:

  ```bash
  cd nvidia-tensorrt
  ./configure.sh --src ~/Downloads --write
  PKGEXT=.pkg.tar makepkg -f
  sudo pacman -U nvidia-tensorrt-sdk-local-11.4.0.106-2-x86_64.pkg.tar
  ```

  Use pkgrel 2 or later. Pkgrel 2 provides `tensorrt=11.4.0.106` and `nvidia-tensorrt-sdk`. Pkgrel 1 provides only `nvidia-tensorrt-sdk`, so it does not satisfy the provider package's `tensorrt` dependency. Install pkgrel 2 over it.

- **The AUR's `tensorrt`**, which installs into `/usr` and provides `tensorrt`. Build and install it as you do any AUR package. Its major.minor must match the TensorRT the provider package was built against.

The recipe requires TensorRT 11: the provider has the code paths for it. The TensorRT provider package depends on `tensorrt`, bounded to the major.minor of the build (`tensorrt>=11.4` and `tensorrt<11.5` for TensorRT 11.4). Either source provides that name, so the package accepts either as long as the version is inside the bound ([dependencies.md](docs/dependencies.md#tensorrt-one-dependency-name-for-two-sources)).

Install TensorRT before the packages. Pacman refuses the TensorRT provider while no installed package provides `tensorrt` within its bound.

For a source build, the recipe finds TensorRT in `/opt/nvidia/tensorrt` when that holds `include/NvInfer.h`, and otherwise in `/usr`. `ORTF_TENSORRT_HOME` overrides this ([build.md](docs/build.md#tensorrt-location)). The bound is read from `include/NvInferVersion.h` under the TensorRT root. The recipe's `makedepends` adds `tensorrt` when it builds the TensorRT provider. A TensorRT installed from one of the packages above satisfies it. A TensorRT unpacked by hand, with `ORTF_TENSORRT_HOME` pointing at it, needs `makepkg -d`, which skips the dependency check. `prepare()` warns when no installed package provides `tensorrt`, or when the installed one has another major.minor.

## Install the packages

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

The helper installs the vendor runtimes that are in Arch's repositories. TensorRT is not, so install it first ([Install TensorRT](#install-tensorrt)). For the `all` preset, name `onnxruntime-provider-cuda13-all-bin` instead of the `rtx` one. They conflict, so install one CUDA package at a time.

**Without a helper**, install the release files directly:

1. Download the `.pkg.tar.zst` files and `SHA256SUMS` from <https://github.com/rsd/omarchy-onnxruntime/releases>. The tag is `v<pkgver>-<pkgrel>`, here `v1.29.0-7`.
2. Check the files, then install the ones you want in one `pacman -U`:

   ```bash
   sha256sum -c --ignore-missing SHA256SUMS
   sudo pacman -U onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-cuda13-rtx-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-tensorrt-cuda13-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-openvino-1.29.0-7-x86_64.pkg.tar.zst \
     onnxruntime-provider-dnnl-1.29.0-7-x86_64.pkg.tar.zst \
     python-onnxruntime-core-1.29.0-7-x86_64.pkg.tar.zst
   ```

Leave out the oneDNN file unless you run an Intel server CPU.

You can also clone the AUR package (`git clone https://aur.archlinux.org/onnxruntime-split-bin.git`) and run `makepkg` in it. That downloads the release files and writes the `-bin` packages. Install the ones you want with `sudo pacman -U`. Do not use `makepkg -i`: it tries to install all of them, including both conflicting CUDA packages.

The release files are named without `-bin` and carry the same dependencies, so they work with each other. They do not satisfy a `-bin` package's name. Do not mix `pacman -U` of a release file with the AUR's `-bin` package of the same name. If you have source-built packages of the same names, pacman offers to replace them.

What the prebuilt set leaves out:

- **MIGraphX.** The provider and its Python add-on are not prebuilt, because the provider has never run on AMD hardware. They are source packages on the AUR. For an AMD GPU, `yay -S onnxruntime-split-migraphx` builds both from that package base, after the core is installed ([amd.md](docs/amd.md#installing)).
- **No NCCL.** There are no prebuilt `-nccloff` packages.
- **CUDA 12.** The provider for Pascal and Volta is a separate AUR package base, `onnxruntime-split-cuda12-bin`, and needs a different CUDA on the machine: see [below](#older-nvidia-gpus-pascal-and-volta-cuda-12).
- **Other builds.** For another CUDA architecture list or engine set, build from the recipe.

The prebuilt packages have the same dependency bounds as a source build ([dependencies.md](docs/dependencies.md)). Pacman refuses a vendor upgrade that goes past one, and a rebuilt release follows.

### Build from source

Finish the two steps above. Then follow [docs/build-from-source.md](docs/build-from-source.md): build tools, `makepkg`, the exact `pacman -U` command, and the checks of the build tree.

### Upgrade from the single package 1.29.0-1

Version 1.29.0-1 was one `onnxruntime-full` package that owned every provider library, with `python-onnxruntime-full` for the Python module. `onnxruntime-core` has `replaces=(onnxruntime-full)` and `python-onnxruntime-core` has `replaces=(python-onnxruntime-full)`. Each also conflicts with the name it replaces.

- `replaces=` acts only on a sync upgrade (`pacman -Syu` from a repository).
- With `pacman -U`, the conflicts make pacman offer to remove the old packages in the same transaction. A file that moves from a package removed in that transaction to a new one is not a file conflict.
- Install every new file in one `pacman -U`, not in several. Pacman moves a file from one package to another only within one transaction.
- Answer `y` to the offer to remove `onnxruntime-full` and `python-onnxruntime-full`. That question defaults to N, so `--noconfirm` aborts the upgrade.
- The old `python-onnxruntime-full` depends on `onnxruntime-full=1.29.0-1`, so `python-onnxruntime-core` must be in the same transaction.
- With `ORTF_PYTHON=0` there is no new Python package to replace it. Remove it first with `sudo pacman -R python-onnxruntime-full`, or the transaction fails on that dependency.

### Upgrade from 1.29.0-6

Version 1.29.0-7 builds the core with the training operators and NCCL. That changes the interface between the core and the providers. A provider built without them crashes in a core built with them. Install every new file of the release in one `pacman -U` (or one AUR helper call), as above.

The new core also depends on `nccl`, which pacman installs from `extra`. If you use the CUDA 12 provider for Pascal and Volta, `onnxruntime-core` 1.29.0-7 conflicts with its 1.29.0-1 package. Upgrade it to 1.29.0-2 or later in the same transaction ([cuda12-legacy.md](docs/cuda12-legacy.md#upgrading)).

### Check built packages against your system

`scripts/check-deps.sh <package files>` lists every library of a package, its owner, and whether the installed versions are inside the bounds. See [dependencies.md](docs/dependencies.md#checking-built-packages-scriptscheck-depssh).

## Older NVIDIA GPUs (Pascal and Volta): CUDA 12

CUDA 13, cuDNN 9.11 and the 590 driver series dropped Pascal and Volta, so the CUDA 13 packages cannot use those cards. The CUDA 12 provider `onnxruntime-provider-cuda12-legacy` serves them. It replaces the CUDA 13 setup on that machine and swaps Arch's `nccl` for the AUR's `nccl-cuda12.9`. It is untested on Pascal and Volta hardware. The install steps, the swap from CUDA 13 and the way back are in [cuda12-legacy.md](docs/cuda12-legacy.md#installing).

## Check that it works

### After installing

Run the smoke test from a clone of this repository:

```bash
scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU,dnnl
```

The test lists the providers and runs a small model on every engine and device it finds. It compares the numbers with a reference. You see one row per target with the status `ok`, `FAIL` or `unavailable`. `--require` takes a comma-separated list of targets and exits 1 unless each is `ok`. List only the engines your machine has, for example `--require cpu,cuda` on a machine without Intel hardware. Without `--require`, an engine whose runtime or device is missing shows `unavailable`, not a failure.

Valid targets are `cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU` and `dnnl`. The target `migraphx-load` is for the MIGraphX package and is not in the default run: `scripts/smoke-test.sh --only cpu,migraphx-load` ([amd.md](docs/amd.md#using-it)).

The script lives in the repository, so a prebuilt-only installation needs a clone to run it. The wrapper needs a C compiler and Python 3. The options, and how to read `ok`, `FAIL` and `unavailable`, are in [tools/ort-smoke/README.md](tools/ort-smoke/README.md).

A TensorRT `ok` does not show that TensorRT's built-in plugins load. See [engines.md](docs/engines.md#tensorrt).

Check the Python module:

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

You see the version `1.29.0` and one provider name for each engine package you installed.

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
| Pacman refuses the TensorRT provider | No installed package provides `tensorrt` within the provider's bound | Install `nvidia-tensorrt-sdk-local` pkgrel 2 or later, or the AUR's `tensorrt`, first ([Install TensorRT](#install-tensorrt)) |
| Pacman reports a conflict with `onnxruntime`, `onnxruntime-cuda` or another Arch variant | Arch's packages install the same `libonnxruntime.so` | Answer `y` to remove them. Programs built against Arch's package keep working ([docs/choosing.md](docs/choosing.md#why-one-set-per-machine)) |
| Upgrade aborts with `--noconfirm` | The question about removing `onnxruntime-full` defaults to N | Run `pacman -U` without `--noconfirm` and answer `y` ([above](#upgrade-from-the-single-package-1290-1)) |
| Crash after a partial upgrade | A provider built without training and NCCL loaded into a core built with them | Install every file of the release in one `pacman -U` ([above](#upgrade-from-1290-6)) |
| Two CUDA packages conflict | The `rtx` and `all` packages install the same library | Install one CUDA package at a time |
| Pacman holds back a CUDA, cuDNN, TensorRT or OpenVINO upgrade | The provider's version bound | Wait for the rebuilt packages. `scripts/check-deps.sh` shows which bound moved ([dependencies.md](docs/dependencies.md#the-version-bounds)) |
| Smoke test says `unavailable` for an engine | The provider package or its vendor runtime is missing. ONNX Runtime fails with "cannot open shared object file" | Install the provider package and its runtime |
| `Device GPU is not available` (or `NPU`) | The OpenVINO plugin or driver is missing, or your user cannot open the device | Install the plugin and driver, and check `/dev/dri/renderD*` or `/dev/accel/accel0` ([above](#install-the-vendor-runtimes)) |
| `import onnxruntime.training` fails with `No module named 'torch'` | The training modules import PyTorch | Install `python-pytorch`. `import onnxruntime` needs no PyTorch |
| TensorRT logs that the default plugin library was not found | TensorRT under `/opt` is not on `LD_LIBRARY_PATH` | Load the TensorRT environment file for that program only ([engines.md](docs/engines.md#tensorrt)) |
| A program crashes at exit after using OpenVINO | It released the ONNX Runtime environment | Keep the environment for the life of the process ([engines.md](docs/engines.md#openvino)) |
| `makepkg -i` tries to install both CUDA packages | `-i` installs every package it builds | Install the files you want with `pacman -U` |
| Source build conflicts on `-cuda13-sm120` | An older local build used that name | Answer `y` to remove it ([build-from-source.md](docs/build-from-source.md#naming-the-files)) |

## Uninstall

Name the packages you installed. The provider packages depend on `onnxruntime-core`, so it cannot be removed while one remains. `pacman -Qq | grep onnxruntime` lists them.

```bash
sudo pacman -R python-onnxruntime-core onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-cuda13-rtx onnxruntime-provider-openvino onnxruntime-provider-dnnl onnxruntime-core
```

Adjust the list:

- Add `python-onnxruntime-provider-migraphx` and `onnxruntime-provider-migraphx` if you built them.
- Name the `-bin` packages if you installed the prebuilt ones.
- Use `onnxruntime-provider-cuda13-all` if that is the CUDA package you installed.
- Replace the CUDA 13 package with `onnxruntime-provider-cuda12-legacy` if you use the legacy provider. It has no TensorRT package.

`nccl`, which the core pulled in, stays installed until you remove it. `pacman -Qdtq` lists such leftovers.

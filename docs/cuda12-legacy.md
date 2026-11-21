# NVIDIA Pascal and Volta: the CUDA 12 line

`onnxruntime-provider-cuda12-legacy` is the CUDA provider of ONNX Runtime 1.29.0 built with CUDA 12.9, for the NVIDIA GPUs that CUDA 13 dropped. You need it only if you have a Pascal or Volta card. It is a separate line from the packages of the main recipe, which use CUDA 13.

**It is untested on Pascal and Volta hardware**, and CUDA 12.9 with cuDNN 9.10 on the 580xx driver is untested too. This is version 1.29.0-2, built with the training operators and NCCL (below). It is released together with the main line's 1.29.0-7. Terms such as compute capability, SASS and PTX are in the [glossary](glossary.md).

## Who needs it

| GPU | Compute capability | Package |
|---|---|---|
| Turing (RTX 20, GTX 16) and newer | `75` and up | the CUDA 13 packages ([INSTALL.md](../INSTALL.md)); this line also covers them |
| Pascal (GTX 10, P100) and Volta (V100, Titan V) | `60`, `61`, `70` | this line |
| Maxwell | `50`, `52` (`sm_50`, `sm_52`) | not supported |

CUDA 13 dropped Maxwell, Pascal and Volta. cuDNN 9.11 removed them, and so did the driver series from 590 on, so Arch's current driver packages do not support them. CUDA 12 still supports Pascal and Volta.

**This package does not support Maxwell**, although CUDA 12 does. ONNX Runtime 1.29's fp16 kernels need `sm_53` or later (`ptxas` refuses their f16 arithmetic for `compute_50`), so they cannot be built for it. A WebGPU provider is planned as the route for those cards.

## What the machine needs

The packages below that are not in Arch's repositories (`nvidia-580xx-dkms`, `cuda-12.9`, `cudnn9.10-cuda12.9`, `nccl-cuda12.9`, `gcc14`) are third-party AUR packages, not part of this repository. Read their PKGBUILDs.

- **The CUDA 12 userland replaces CUDA 13.** The AUR's `cuda-12.9` provides `cuda=12.9` and conflicts with `cuda`. The AUR's `cudnn9.10-cuda12.9` provides `cudnn=9.10` and conflicts with `cudnn`. So CUDA 13 and cuDNN come off that machine, and so do the packages that need them. Only these two AUR packages provide versions inside the provider's bounds. Arch's `cuda` is CUDA 13 and its `cudnn` is 9.11 or later. With the CUDA 12 userland already installed, pacman or the AUR helper finds the bounds met and never reaches for Arch's CUDA 13.
- **Pascal and Volta need the 580xx driver.** The 590 series dropped them, so Arch's `nvidia-open*` packages and `nvidia-utils` no longer serve them. The AUR's `nvidia-580xx-dkms` is the 580 series for them. It is a DKMS package, so it needs the headers of your kernel. The AUR's `nvidia-580xx-utils` provides `nvidia-utils`, so it meets the `nvidia-utils` dependency of `cuda-12.9` and of this package. No 580xx package name appears in any dependency of this repository's packages.
- **GCC 14.** `cuda-12.9` depends on the AUR's `gcc14`, a long source build that the AUR helper runs on the way. `gcc14` installs beside Arch's `gcc`. Only a builder needs it, in the clean chroot. The prebuilt package does not.
- **The main line's core.** `onnxruntime-core` or `onnxruntime-core-bin`, 1.29.0-7 or later, installed already (the provider's pin, below).

## Installing

Install the driver, `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9` before the provider, never the provider first. The core depends on `nccl`, and `nccl-cuda12.9` provides it, so Arch's `nccl` is replaced. Pacman offers to remove it. Answer `y`.

1. **Remove the CUDA 13 side.** Name what you have installed (`pacman -Qq | grep -E '^(nvidia|cuda|cudnn|onnxruntime-provider)'` lists it). Remove the CUDA 13 provider packages first, since they depend on `cuda` and `cudnn`, then those, then the current driver. Pacman names anything else that needs `cuda` or `cudnn`. Remove that too, or the transaction is refused.

   ```bash
   sudo pacman -R onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-cuda13-rtx
   sudo pacman -R cudnn cuda
   sudo pacman -R nvidia-open-dkms        # whichever nvidia-open* packages you have
   ```

   These are the names of source-built packages. If you installed the prebuilt ones, name them with `-bin` (`onnxruntime-provider-tensorrt-cuda13-bin`, `onnxruntime-provider-cuda13-rtx-bin`).

2. **Install the 580xx driver and the CUDA 12 userland, then the provider in a second command.** `gcc14` comes with `cuda-12.9`.

   ```bash
   yay -S nvidia-580xx-dkms cuda-12.9 cudnn9.10-cuda12.9 nccl-cuda12.9
   yay -S onnxruntime-provider-cuda12-legacy-bin
   ```

   Reboot to load the new driver.

   Without a helper, build and install the AUR packages with `makepkg -si` in their clones: `gcc14` first, then `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9`, all before the provider. Install the provider from its release. The tag is `cuda12-v1.29.0-2` (the scheme is `cuda12-v<pkgver>-<pkgrel>`, a separate release from the main line's `v<pkgver>-<pkgrel>`). Download the `.pkg.tar.zst` and `SHA256SUMS` from <https://github.com/rsd/omarchy-onnxruntime/releases>, check them, and install:

   ```bash
   sha256sum -c --ignore-missing SHA256SUMS
   sudo pacman -U onnxruntime-provider-cuda12-legacy-1.29.0-2-x86_64.pkg.tar.zst
   ```

   To build the provider from source instead, see [Building from source](#building-from-source). The package is then `onnxruntime-provider-cuda12-legacy`, without `-bin`.

3. **Check that it works**, as in [Check that it works](../INSTALL.md#check-that-it-works), with the targets you have:

   ```bash
   scripts/smoke-test.sh --require cpu,cuda
   ```

   You see `ok` for `cuda`. [results.md](results.md#cuda-12-legacy-provider) has the figures of a run on a Blackwell GPU.

### Upgrading

From the CUDA 12 provider 1.29.0-1: version 1.29.0-2 is built with training and NCCL, so it needs the main line's core 1.29.0-7 or later, which conflicts with 1.29.0-1. Install `nccl-cuda12.9` first (answer `y` to replacing `nccl`). Then install the new core and the new provider in one `pacman -U` (or one helper call). The core's conflict with the old provider resolves in that one transaction.

### Going back to CUDA 13

This works only where the card supports CUDA 13. A machine with only Pascal or Volta cards has no CUDA 13 setup to go back to. Remove the CUDA 12 side (the same order reversed) and install what step 1 removed:

```bash
sudo pacman -R onnxruntime-provider-cuda12-legacy-bin cudnn9.10-cuda12.9 cuda-12.9 nvidia-580xx-dkms   # drop -bin if you built the provider from source; add the other 580xx packages you have
sudo pacman -S nvidia-open-dkms cuda cudnn nccl onnxruntime-provider-cuda13-rtx-bin   # your driver, and the CUDA 13 packages you removed; nccl replaces nccl-cuda12.9: answer y
```

`pacman -Qdtq` then lists `gcc14` and other packages the CUDA 12 userland brought in, if nothing else needs them.

## What it is

- **One package** from its own pkgbase `onnxruntime-split-cuda12`, with its own `pkgrel`. Its recipe is generated from `onnxruntime-split/PKGBUILD`. Never edit it by hand ([building](#building-from-source)).
  - It holds `libonnxruntime_providers_cuda.so`, compiled for the preset `legacy`: `60 61 70 75 80 86 89 120`, plus the PTX of `compute_60` (`60-virtual`). A preset's PTX is always that of its oldest architecture, the one every newer GPU can JIT-compile.
  - It also holds 18 kernels for `sm_90a` that ONNX Runtime compiles whatever the architecture list says. The CUDA 13 `rtx` package has the same 18.
  - The Turing to Blackwell targets are there so that a machine pinned to CUDA 12 with a newer card beside the old one is served too.
- **It loads into the main line's core.** It depends on `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`. Any packaging-only pkgrel of the core from 1.29.0-7 on satisfies it, because the core holds no CUDA code and links no CUDA library.
  - The lower bound is the first core built with the training operators and NCCL. Those change the interface between the core and a provider, nothing checks that both agree, and a mismatch crashes.
  - That is also why this line is released together with the main line, and why the core 1.29.0-7 conflicts with this package below 1.29.0-2 ([the core pin](release-cuda12.md#the-core-pin)).
  - `python-onnxruntime-core` works with it too, because its module links no CUDA library either. There is no TensorRT provider and no Python package in this line.
- **Training and NCCL.** Like the main line, it is built with the training operators and with NCCL. Its NCCL is the AUR's `nccl-cuda12.9`, depended on by name. It provides and conflicts with `nccl` and is built for `sm_52` to `sm_72`, the GPUs this line exists for. Arch's `nccl` (kernels for `sm_75` and newer) does not run on them. A bare `nccl` dependency would let pacman pick Arch's. The main core's own dependency on `nccl` is met by `nccl-cuda12.9`'s provide.
- **Dependencies.** `cuda>=12` and `cuda<13` (from `libcudart.so.12`), `cudnn>=9.10` and `cudnn<9.11`, `nccl-cuda12.9`, `nvidia-utils`. The cuDNN bound is to the minor version, not the major as for the CUDA 13 packages, because cuDNN 9.11 dropped these GPUs. A bound on 9 alone would let pacman install a cuDNN that cannot run on them. The build refuses any cuDNN other than 9.10 for the same reason.
- **One CUDA provider at a time.** Like every CUDA package it provides and conflicts with `onnxruntime-provider-cuda` and `onnxruntime-provider-cuda12`. You cannot install it beside any `-cuda13-*` package, the CUDA 13 TensorRT provider included (that one depends on the CUDA 13 provider).

## Building from source

`onnxruntime-split-cuda12/PKGBUILD` builds the one package `onnxruntime-provider-cuda12-legacy`. It is a pkgbase of its own, because the two toolchains both own `/opt/cuda` and cannot share a build, and because its AUR `makedepends` would be wrong for one of the two.

- **Generated, never edited.** `scripts/gen-cuda12.sh` writes it from `onnxruntime-split/PKGBUILD` and changes only a few lines:
  - `pkgbase`;
  - the defaults `_ortf_default_eps='cuda'`, `_ortf_default_cuda_archs=legacy` and `_ortf_default_python=0`;
  - `_ortf_line=cuda12` and `_ortf_cuda_major_default=12`;
  - `pkgrel`, which is the line's own (`line_pkgrel` in the script; 2 now);
  - a header comment.

  The same `pkgver`, sources, patches and build follow. `scripts/gen-cuda12.sh --check` fails, with a diff, when the committed file is stale. The script stops without writing when a source line it replaces is missing or repeated. Its `.SRCINFO` comes from `scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --write` (`--check` and `--aur DIR` take the same `--recipe`).
- **What the line refuses.** `_ortf_line=cuda12` builds the CUDA provider alone. `prepare()` stops for an `ORTF_EPS` other than `cuda`, for `ORTF_PYTHON=1`, for a cuDNN other than 9.10 (cuDNN 9.11 dropped Maxwell, Pascal and Volta; the main line is not bound by this), and for a toolkit that is not CUDA 12. The names carry `12` whatever `/opt/cuda` holds when makepkg reads the file, so the published `.SRCINFO` is the same on a CUDA 13 host. `package()` fails the build on a CUDA library with a `.so.13` soname, and on a provider that does not link `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12`.
- **Dependencies.** `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1` (this line makes no core), `nccl-cuda12.9`, `cuda>=12`, `cuda<13`, `cudnn` bounded to the minor of the cuDNN installed at build time (`cudnn>=9.10`, `cudnn<9.11` with cuDNN 9.10), and `nvidia-utils`.
  - `makedepends` are the build tools and the AUR packages by name: `gcc14`, `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9`. They are never bare `cuda`, `cudnn` and `nccl`, which an AUR helper would resolve to Arch's CUDA 13, cuDNN 9.11 and NCCL.
  - The `.SRCINFO` lists the bounds too, not the bare names. The bounds resolve against the provides of the AUR's `cuda-12.9` (`cuda=12.9`) and `cudnn9.10-cuda12.9` (`cudnn=9.10`).
- **Compiler: GCC 14.** `cuda-12.9`'s `/etc/profile.d/cuda.sh` sets `NVCC_CCBIN=/usr/bin/g++-14`. The recipe uses nvcc's host compiler for its C++ as well, so that host code compiled by nvcc and by the C++ compiler agrees. Nothing is set in the recipe for it. The build needs the AUR's `gcc14` (`gcc-14` and `g++-14`).
- **Build.** Only in a clean chroot, with the four AUR packages passed in with `-I`: [release-cuda12.md](release-cuda12.md#3-build-in-a-clean-chroot).
- **Check the result.** `scripts/check-cuda-archs.sh [--cuobjdump PATH] FILE ARCH...` takes a package file or the library. It fails unless `cuobjdump` lists exactly those SASS and PTX architectures.

`scripts/test.sh` runs the generator's `--check`, the `--check` of both `.SRCINFO` files, and the names, provides and conflicts of the generated one.

Building the main recipe against a CUDA 12 toolkit is a different thing, and it is not what is released. It needs an explicit `ORTF_CUDA_ARCHS` list (or `legacy`), because the presets `rtx` and `all` need CUDA 13. It also needs no `tensorrt` in `ORTF_EPS` (TensorRT 11 needs CUDA 13), or a TensorRT 10 ([build.md](build.md#package-names)).

## Releasing

The release of this line has its own tag, prebuilt pkgbase and AUR packages: [release-cuda12.md](release-cuda12.md).

## Known difference: MODNet in fp32

With `--precision fp32` the model check reports a maximum error of 1.23e-3 for MODNet on this provider. That is above its 1e-3 bound. The CUDA 13 provider gives 2.78e-4. The bound is kept, and this is a **known difference, not a failure of the package**: the older cuDNN (9.10) uses a different summation order from the cuDNN of the CUDA 13 packages. The mean error is 3.7e-6, and the difference is invisible in a 0 to 1 mask. With the default TF32 math the check passes. It reproduced in three of three runs ([results.md](results.md#modnet-in-fp32)).

## Test status

The package has not run on real Pascal or Volta hardware, so its `sm_60`, `sm_61` and `sm_70` code is unconfirmed on the cards it is for. CUDA 12.9 with cuDNN 9.10 on the 580xx driver is untested as well.

The only measurement is a [run on a Blackwell GPU with the CUDA 12 libraries](results.md#cuda-12-legacy-provider). It shows that the library loads and computes with the CUDA 12 runtimes. Until someone confirms the package on an older card, treat it as the MIGraphX provider is treated: built from a recipe, never run on that hardware. `scripts/check-cuda-archs.sh` checks the other half, what a built library contains.

Sources: [CUDA GPU compute capabilities](https://developer.nvidia.com/cuda-gpus), [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

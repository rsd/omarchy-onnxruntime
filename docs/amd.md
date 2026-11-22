# AMD GPUs (MIGraphX)

This guide installs the MIGraphX provider, the only ONNX Runtime provider for AMD GPUs. **It is untested on AMD hardware.** The maintainer has no AMD GPU, so no model has run on one. It ships as source only, from its own AUR package base `onnxruntime-split-migraphx`. There is no prebuilt (`-bin`) package, because that would publish a binary nobody has run.

If you have an AMD GPU, [reporting a result](#reporting-a-result) is the most useful thing you can do. Terms are in the [glossary](glossary.md).

## Before you start

- **An AMD GPU in Arch's ROCm build list** ([Which AMD GPUs](#which-amd-gpus)).
- **The core installed**: `onnxruntime-core` or `onnxruntime-core-bin`, 1.29.0-7 or later. It depends on `nccl`.
- **An AUR helper**, or the AUR package cloned by hand.

## Installing

1. Install the main line's core, then build the provider and its add-on from their AUR package base:

   ```bash
   yay -S onnxruntime-core-bin       # or the source-built onnxruntime-core
   yay -S onnxruntime-split-migraphx
   ```

   The second command builds both packages on your machine. It installs `migraphx`, `hip-runtime-amd` and `rccl` as build dependencies.

2. Without a helper, clone the AUR package and build it:

   ```bash
   git clone https://aur.archlinux.org/onnxruntime-split-migraphx.git
   cd onnxruntime-split-migraphx
   makepkg -s -f
   ```

   Then run `pacman -U` on the files it writes, in one transaction, as [INSTALL.md](../INSTALL.md#4-install-the-packages) asks for one build's files:
   - `onnxruntime-provider-migraphx-1.29.0-1-x86_64.pkg.tar.zst`
   - `python-onnxruntime-provider-migraphx-1.29.0-1-x86_64.pkg.tar.zst`, if you use Python.

The build downloads ONNX Runtime (from this project's mirror of the source, [build.md](build.md#the-onnx-runtime-source-mirror-or-upstream)) and the pinned dependencies as `makepkg` sources. [build.md](build.md#dependencies-are-mirrored-not-fetched) has the sizes. The release is built in a clean chroot. A build on your own system uses the ROCm packages you have installed, and `prepare()` records their versions for the dependency bounds.

A core built with `ORTF_NCCL=0` (`onnxruntime-core-nccloff`, see [build.md](build.md#nccl-and-the-nccloff-packages)) does not fit this package. This package is built with NCCL, and it conflicts with that core. An AMD-only machine that wants no NCCL builds the provider locally from the main recipe instead, after installing `migraphx` and `hip-runtime-amd`:

```bash
cd onnxruntime-split
ORTF_EPS=migraphx ORTF_NCCL=0 MAKEFLAGS=-j8 makepkg -d -f
```

That makes `-nccloff` packages for local use, untested like the rest.

## Using it

- **C API.** On the main line's core the provider loads by name: `SessionOptionsAppendExecutionProvider(options, "MIGraphX", ...)`, or the legacy `SessionOptionsAppendExecutionProvider_MIGraphX`. That core is built without MIGraphX and with ONNX Runtime's MIGraphX provider interface, so it opens `libonnxruntime_providers_migraphx.so` when asked.
- **`GetAvailableProviders` does not list MIGraphX** on this core. The list names only the providers built into the library. An application must try the provider and handle the failure, not look for the name in that list.
- **Python.** Python's `get_available_providers()` is not the C list. `python-onnxruntime-core` adds `MIGraphXExecutionProvider` to it when the provider's library is installed (the link the add-on puts in the module's `capi/` directory). With `onnxruntime-provider-migraphx`, `python-onnxruntime-provider-migraphx` and `python-onnxruntime-core` all installed, this command shows it, and `providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"]` selects it. Without the add-on the name is absent. This has not been run on AMD hardware.

  ```bash
  python -c 'import onnxruntime; print(onnxruntime.get_available_providers())'
  ```

- **First session.** It is slow, because MIGraphX compiles the model for the GPU. `ORT_MIGRAPHX_MODEL_CACHE_PATH` enables a compiled-model cache.
- **Check without a GPU.** This command appends the provider by name and creates a session:

  ```bash
  scripts/smoke-test.sh --only cpu,migraphx-load
  ```

  On a machine without an AMD GPU the pass is `ok (no device)`: the provider library loaded and failed cleanly with a no-device message, with no crash. The target is not in the default run, because a HIP runtime without a usable device may abort the process. [tools/ort-smoke/README.md](../tools/ort-smoke/README.md#the-two-newest-targets) lists the exact outcomes.

## Reporting a result

Because no one has run it, a report from an AMD GPU is what moves it out of "untested". Open an issue at <https://github.com/rsd/omarchy-onnxruntime/issues> with:

- the GPU model (and `gfx` code, from `rocminfo` if you have it) and the Arch kernel;
- the ROCm version, from this command:

  ```bash
  pacman -Q migraphx hip-runtime-amd
  ```

- the output of this command, from a clone of this repository (the smoke test in [INSTALL.md](../INSTALL.md#check-that-it-works)):

  ```bash
  scripts/smoke-test.sh --only cpu,migraphx-load
  ```

- if `python-onnxruntime-core` is installed, the output of the Python command above, which should list `MIGraphXExecutionProvider` once the provider and its add-on `python-onnxruntime-provider-migraphx` are installed;
- what you ran with the provider, and whether it worked, in a sentence.

On a machine that has an AMD GPU, `ok (no device)` means something is wrong: the driver, your user's access to `/dev/kfd`, or a GPU outside ROCm's support. `ok` means the model ran. A report of a failure is as useful as one of a success.

## Which AMD GPUs

**The list is Arch's, not AMD's.** Arch's `migraphx` (and its `rocblas` and `miopen-hip`) are built with `GPU_TARGETS="$(rocm-supported-gfx)"`. That script is in Arch's `rocm-toolchain` package and follows Fedora's ROCm target list. The provider here loads Arch's `migraphx`, so it inherits that list. The package version checked is `rocm-toolchain` 0.4.0-2 (2026-11-18). Its script prints these 23 targets (`gfx906`, `gfx908` and `gfx90a` with an `xnack` mode):

`gfx900`, `gfx906:xnack-`, `gfx908:xnack-`, `gfx90a:xnack+`, `gfx90a:xnack-`, `gfx942`, `gfx950`, `gfx1010`, `gfx1012`, `gfx1030`, `gfx1031`, `gfx1035`, `gfx1036`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153`, `gfx1200`, `gfx1201`

A GPU with code in that list **should work through Arch's ROCm and MIGraphX build, untested on AMD hardware** like the whole line. Arch is not one of AMD's supported distributions. The "AMD's list" columns below say only what AMD itself documents for the ROCm versions Arch has ([system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html), [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)). "Community" means AMD does not list the GPU but Arch builds code for it. Check your own gfx code with `rocminfo`. Check the list a newer package prints with `rocm-supported-gfx` (from `rocm-toolchain`).

| Target (generation) | Cards | AMD's list, ROCm 7.2 (Arch `extra`) | AMD's list, ROCm 10.1 (`extra-testing`) |
|---|---|---|---|
| gfx900 (Vega 10) | RX Vega 56 and 64, Radeon Pro WX 9100, Instinct MI25 | not listed | not listed |
| gfx906 (Vega 20) | Radeon VII, Radeon Pro VII, Instinct MI50 and MI60 | not listed | not listed |
| gfx908 (CDNA 1) | Instinct MI100 | supported | supported |
| gfx90a (CDNA 2) | Instinct MI210, MI250, MI250X | supported | supported |
| gfx942 (CDNA 3) | Instinct MI300A, MI300X, MI325X | supported | supported |
| gfx950 (CDNA 4) | Instinct MI350X, MI355X | supported | supported |
| gfx1010 (RDNA 1) | RX 5700 XT, 5700, 5600 XT | not listed | not listed |
| gfx1012 (RDNA 1) | RX 5500 XT, 5500 | not listed | not listed |
| gfx1030 (RDNA 2) | RX 6800, 6800 XT, 6900 XT, 6950 XT, PRO W6800 | supported for PRO cards; consumer cards community | supported as PRO W6000 |
| gfx1031 (RDNA 2) | RX 6700 XT, 6750 XT | community | community |
| gfx1035 (RDNA 2 APU) | Radeon 680M (Ryzen 6000, Rembrandt) | not checked | not checked |
| gfx1036 (RDNA 2 APU) | Radeon Graphics of Ryzen 7000 desktop (Raphael) | not checked | not checked |
| gfx1100 (RDNA 3) | RX 7900 XTX, 7900 XT, 7900 GRE, PRO W7900 and W7800 | supported | supported |
| gfx1101 (RDNA 3) | RX 7800 XT, 7700 XT | supported | supported |
| gfx1102 (RDNA 3) | RX 7600, 7600 XT | community | supported |
| gfx1103 (RDNA 3 APU) | Radeon 780M (Ryzen 7040 and 8040) | community | supported (Ryzen APU, Ryzen AI) |
| gfx1150 to gfx1153 (RDNA 3.5 APUs) | Strix Point (1150), Strix Halo (1151), Krackan Point (1152); gfx1153 is in the list, its product is not named here | community | supported (Ryzen APU, Ryzen AI) |
| gfx1200, gfx1201 (RDNA 4) | RX 9060 and 9060 XT (1200); RX 9070, 9070 XT and AI PRO R9700 (1201) | supported | supported |

The product names beside each gfx code are for orientation. The codes are authoritative. "Not listed" is from AMD's current ROCm documents (Vega and RDNA 1 are not on them). "Not checked" means this page did not look the target up in AMD's documents. The AMD columns do not decide support here: Arch's list does.

**Not in the list, so not supported:**

- Polaris (gfx803: RX 400 and 500 series).
- The Vega APUs (gfx90c, gfx902: Ryzen 2000 to 5000 graphics).
- RDNA 1's gfx1011 (Radeon Pro 5600M).
- The RDNA 2 and 3 chips gfx1032 (RX 6600 and 6650 XT), gfx1033 and gfx1034 (RX 6500 and 6400).

Arch built no code for them, so MIGraphX has nothing to compile for. Running Polaris would need an old ONNX Runtime (1.22 or earlier, the last with the ROCm provider) against an old ROCm (5.x or 6.x) that Arch no longer ships. This recipe does not provide that and documents these GPUs as unsupported.

## What exists for AMD GPUs

**There is no ROCm execution provider in ONNX Runtime 1.29.** It was removed in 1.23, whose release notes point AMD users to the MIGraphX and Vitis AI providers ([1.23 release notes](https://newreleases.io/project/pypi/onnxruntime/release/1.23.0), [MIGraphX provider documentation](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)). The source tree at `v1.29.0` has no ROCm build option. Arch's `onnxruntime-rocm` and `onnxruntime-opt-rocm` still pass ROCm-related flags (a HIP architecture list, HIP and NCCL settings), but they are inert in 1.29. The built package holds only the MIGraphX, oneDNN and shared provider libraries and no HIP code ([Arch recipe](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime/-/blob/main/PKGBUILD)). Those packages are therefore, in effect, the MIGraphX provider plus the oneDNN provider.

MIGraphX is the only AMD GPU provider. It is a thin layer, and it **contains no GPU code compiled for a specific card**. It hands the model to the MIGraphX library, which compiles it for the detected GPU when the session is created. The first session is slow, and `ORT_MIGRAPHX_MODEL_CACHE_PATH` enables a compiled-model cache. The per-GPU code is in Arch's `migraphx`, `rocblas` and `miopen-hip` packages, built for the GPU list of Arch's ROCm toolchain. So GPU coverage is whatever Arch's ROCm packages were built for, not anything in this recipe. An architecture preset like CUDA's would produce identical packages.

An AMD GPU can also use Arch's `onnxruntime-rocm` or `onnxruntime-opt-rocm`, which carry the MIGraphX provider. `onnxruntime-core` conflicts with both (and with `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and `onnxruntime-git`), so you install one set at a time. Pacman offers to remove the other ([choosing.md](choosing.md)).

## The package

The pkgbase `onnxruntime-split-migraphx` builds two packages, both at version 1.29.0-1: the provider, `onnxruntime-provider-migraphx`, and `python-onnxruntime-provider-migraphx`, a small add-on that makes the provider usable from Python. `scripts/gen-migraphx.sh` generates their recipe from `onnxruntime-split/PKGBUILD`, as it does for the CUDA 12 line, and you never edit it by hand ([release-migraphx.md](release-migraphx.md)). The provider loads into the main line's `onnxruntime-core`. The add-on is for Python only.

- **Name and suffix.** No architecture suffix (it holds no GPU code), and no ROCm version in the name. ROCm 7.2 and 10.1 keep the same sonames (`libmigraphx_c.so.3`, `libamdhip64.so.7`; not yet seen in a build's `DT_NEEDED` entries). A suffix would be added only if those sonames change.
- **Dependencies.** `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`, `glibc`, `libgcc`, `libstdc++`, `migraphx` and `hip-runtime-amd`. Never `rocm-hip-sdk`, and no `python`: the add-on depends on that instead.
  - `migraphx` and `hip-runtime-amd` are bounded at package time to the major version of the installed package (`migraphx` 7.2.3 gives `>=7` and `<8`). The sonames are checked against the ones the build linked but give no bound, because they outlast ROCm's majors. Upgrading ROCm across a major (7.2 to 10.1) therefore needs a rebuild, and pacman refuses the upgrade until then ([dependencies.md](dependencies.md#the-version-bounds)).
  - The core pin is the pair that the CUDA 12 line uses, because the core's interface to the providers changed with the training operators and NCCL ([dependencies.md](dependencies.md#the-core-pin-of-the-cuda-12-and-migraphx-providers)).
  - The provider depends on no `rccl`, because it links none. `rccl` is a build input only.
- **Build dependencies.** `migraphx`, `hip-runtime-amd` and `rccl`, all from Arch's `extra`, so the clean chroot needs no AUR package. They are large: `migraphx` is 1153.48 MiB installed (7.2.3-7), `rccl` 446.36 MiB (7.2.4-1), and the chroot with ROCm is about 8 GiB (`pacman -Si`; the chroot figure is an estimate).
- **Optional dependency.** `onnxruntime-core` and `python-onnxruntime-core` always list `onnxruntime-provider-migraphx` as an optional dependency, with a note that it comes from the AUR pkgbase `onnxruntime-split-migraphx` and is untested on AMD hardware.
- **Python.** The add-on `python-onnxruntime-provider-migraphx` holds only one symbolic link, in the system Python's site-packages under `onnxruntime/capi/`, to the provider's library. The main line's `python-onnxruntime-core` provides that `capi/` directory, and this line builds no Python module of its own. The add-on depends on `onnxruntime-provider-migraphx` at the same version, on `python-onnxruntime-core`, and on `python`, bounded to the minor version of that site-packages. Pacman therefore refuses a Python upgrade to another minor until the line is rebuilt. The provider lists the add-on as an optional dependency. With the provider, the add-on and `python-onnxruntime-core` installed, `onnxruntime.get_available_providers()` lists `MIGraphXExecutionProvider`. See [Using it](#using-it).

## Building it

```bash
cd onnxruntime-split-migraphx
extra-x86_64-build        # a clean chroot: its makedepends are all in Arch's extra
```

- **Prerequisites.** None beyond the chroot tools ([release.md](release.md#one-time-setup)). The chroot installs `migraphx`, `hip-runtime-amd` and `rccl`. A build outside a chroot needs them installed ([INSTALL.md](../INSTALL.md#2-install-the-vendor-runtimes)); `makepkg -s -f` installs them. `prepare()` checks the ROCm install under `/opt/rocm` and records both packages' versions.
- **What the recipe refuses.** The line builds the MIGraphX provider and its add-on, and no Python module. `prepare()` stops for an `ORTF_EPS` other than `migraphx` and for `ORTF_PYTHON=1`.
- **Flags.** The build passes `-Donnxruntime_USE_MIGRAPHX=ON -Donnxruntime_MIGRAPHX_HOME=/opt/rocm`, the training and NCCL options of every line, and `rccl` as the NCCL library (a build input; the provider links no RCCL). Whether configuration needs more ROCm packages (`rocm-llvm`, `rocm-device-libs`, `rocm-cmake`) is **unverified** until a chroot build with them settles it.
- **Every provider in one local build** from the main recipe, `ORTF_EPS='cuda tensorrt openvino dnnl migraphx'`, is possible and untested. The published package is this line's.

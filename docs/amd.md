# AMD GPUs (MIGraphX)

**The MIGraphX provider is packaged and untested on AMD hardware.** The maintainer has no AMD GPU, and the recipe has not been built with `migraphx` yet, so even the build has had no trial run. It is not shipped prebuilt. Terms are in the [glossary](glossary.md).

## What exists for AMD GPUs

**There is no ROCm execution provider in ONNX Runtime 1.29.** It was removed in 1.23, whose release notes point AMD users to the MIGraphX and Vitis AI providers ([1.23 release notes](https://newreleases.io/project/pypi/onnxruntime/release/1.23.0), [MIGraphX provider documentation](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)). The source tree at `v1.29.0` has no ROCm build option. Arch's `onnxruntime-rocm` and `onnxruntime-opt-rocm` still pass ROCm-related flags (a HIP architecture list, HIP and NCCL settings), but they are inert in 1.29: the built package holds only the MIGraphX, oneDNN and shared provider libraries and no HIP code ([Arch recipe](https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime/-/blob/main/PKGBUILD)). Those packages are therefore, in effect, the MIGraphX provider plus the oneDNN provider.

MIGraphX is the only AMD GPU provider. It is a thin layer, and it **contains no GPU code compiled for a specific card**: it hands the model to the MIGraphX library, which compiles it for the detected GPU when the session is created. The first session is slow, and `ORT_MIGRAPHX_MODEL_CACHE_PATH` enables a compiled-model cache. The per-GPU code is in Arch's `migraphx`, `rocblas` and `miopen-hip` packages, built for the GPU list of Arch's ROCm toolchain. So GPU coverage is whatever Arch's ROCm packages were built for, not anything in this recipe, and an architecture preset like CUDA's would produce identical packages.

## The package

The package is `onnxruntime-provider-migraphx`, built when `migraphx` is in `ORTF_EPS`.

- **Name and suffix.** No architecture suffix (it holds no GPU code), and no ROCm version in the name: ROCm 7.2 and 10.1 keep the same sonames (`libmigraphx_c.so.3`, `libamdhip64.so.7`). A suffix would be added only if those sonames change.
- **Dependencies.** `onnxruntime-core=<pkgver>-<pkgrel>`, `glibc`, `libgcc`, `libstdc++`, `migraphx` and `hip-runtime-amd`, never `rocm-hip-sdk`. `migraphx` and `hip-runtime-amd` are bounded at package time to the major version of the installed package (`migraphx` 7.2.3 gives `>=7` and `<8`). The sonames are checked against the ones the build linked but give no bound, because they outlast ROCm's majors. Upgrading ROCm across a major (7.2 to 10.1) therefore needs a rebuild, and pacman refuses the upgrade until then ([dependencies.md](dependencies.md#the-version-bounds)).
- **Optional dependency.** In a build that makes it, `onnxruntime-core` and `python-onnxruntime-core` list the package as an optional dependency. Other builds (the default one and the prebuilt packages) do not name it, as nothing else provides it.
- **Python.** The Python module reaches the CUDA, TensorRT and OpenVINO libraries through symlinks from `<site-packages>/onnxruntime/capi/` to `/usr/lib/`, which `python-onnxruntime-core` installs. The MIGraphX link is installed by this package instead, only when the build also makes `python-onnxruntime-core`: `<site-packages>/onnxruntime/capi/libonnxruntime_providers_migraphx.so -> /usr/lib/libonnxruntime_providers_migraphx.so`. It belongs here because upstream's `setup.py` puts the MIGraphX library into the wheel only with `--use_migraphx`, so the Python package does not carry it. Python use of the provider is untested on AMD hardware, like the rest of the package.
- **Smoke test.** The smoke test has no MIGraphX target, so the package has no check in this repository beyond the recipe's own.
- **Alternative.** An AMD GPU can use Arch's `onnxruntime-rocm` or `onnxruntime-opt-rocm`, which carry the MIGraphX provider. `onnxruntime-core` conflicts with both (and with `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and `onnxruntime-git`), so only one is installed at a time; pacman offers to remove the other.

## Building it

```bash
ORTF_EPS=migraphx MAKEFLAGS=-j8 makepkg -d -f   # AMD GPUs only: the core, this package and the Python package
ORTF_EPS='cuda tensorrt openvino migraphx' MAKEFLAGS=-j8 makepkg -s -f   # every provider (untested together)
```

- **Prerequisites.** Install `migraphx` and `hip-runtime-amd` from Arch's `extra` before building ([install.md](install.md#1-prerequisites)). They are added to `makedepends` only when `migraphx` is selected, because their install runs to GiB: `migraphx`, `rocblas` and `miopen-hip` together are several GiB installed. `prepare()` checks the ROCm install under `/opt/rocm` and records both packages' versions.
- **`-d` for AMD only.** `cuda`, `cudnn` and `openvino` are unconditional `makedepends`, so `makepkg -s` would install them too; with `migraphx` alone use `-d` (skip the dependency check). With every provider selected, `-s` is right. The combination of all providers is untested: nothing in the 1.29 build files makes them exclusive, but it has not been tried.
- **Flags.** The build passes `-Donnxruntime_USE_MIGRAPHX=ON -Donnxruntime_MIGRAPHX_HOME=/opt/rocm` and nothing else. Whether configuration needs more (`rocm-llvm`, `rocm-device-libs`, `rocm-cmake`) is **unverified** until a trial build.
- **Install.** Name `onnxruntime-provider-migraphx-1.29.0-6-x86_64.pkg.tar.zst` in the `pacman -U` with the others ([build-from-source.md](build-from-source.md#other-combinations)).

## Which AMD GPUs

Officially supported GPUs are AMD's to define, and Arch is not one of AMD's supported distributions, so on Arch the practical test is "works if Arch's package was built for it". AMD's lists, for the ROCm versions Arch has ([system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html), [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)):

| GPU generation (gfx) | ROCm 7.2 (Arch `extra`) | ROCm 10.1 (Arch `extra-testing`) |
|---|---|---|
| RDNA 4: RX 9060 and 9070 series, AI PRO R9700 (gfx1200, gfx1201) | supported | supported |
| RDNA 3: RX 7900, 7800 and 7700 series, PRO W7000 (gfx1100, gfx1101) | supported | supported |
| RDNA 3: RX 7600 (gfx1102) | community | supported |
| RDNA 2: RX 6800 and 6900, PRO W6800 (gfx1030) | supported for PRO cards; consumer cards community | supported as PRO W6000 |
| RDNA 2: other RX 6000 (gfx1031, gfx1032) | community | community |
| APUs: Radeon 780M (gfx1103), Strix Point, Strix Halo, Krackan (gfx1150 to gfx1153) | community | supported (Ryzen APU, Ryzen AI) |
| Instinct MI100, MI200, MI300, MI350 (gfx908, gfx90a, gfx942, gfx950) | supported | supported |

"Community" means AMD does not list the GPU but Arch builds code for it, so it often works without AMD's support. The product names beside each gfx code are given for orientation; the codes are authoritative.

**Old AMD GPUs are not supported.** Vega (gfx900), Radeon VII and MI50 (gfx906) and Polaris (gfx803) are outside current ROCm, and ROCm decides support. Running them would need an old ONNX Runtime (1.22 or earlier, the last with the ROCm provider) against an old ROCm (5.x or 6.x) that Arch no longer ships. This recipe does not provide that and documents these GPUs as unsupported.

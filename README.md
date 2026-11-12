# omarchy-onnxruntime

A local Arch Linux package recipe for [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.29.0 with every execution provider a typical NVIDIA and Intel workstation can use, in one installation: CPU, CUDA, TensorRT and OpenVINO (CPU, Intel GPU and Intel NPU devices). Arch's own ONNX Runtime packages each carry one GPU vendor and have no TensorRT or OpenVINO provider.

The recipe `onnxruntime-full/PKGBUILD` builds two packages from the upstream source tag `v1.29.0`:

| Package | Contents |
|---|---|
| `onnxruntime-full` | `libonnxruntime.so`, one `libonnxruntime_providers_<provider>.so` per provider built, the C and C++ headers (`/usr/include/onnxruntime`) and `libonnxruntime.pc` |
| `python-onnxruntime-full` | The Python module, linked to the libraries of `onnxruntime-full` |

Version 1.29.0 is the release Arch ships, so programs built against Arch's `onnxruntime` see the same API. The packages provide `onnxruntime` and `python-onnxruntime`, and `libonnxruntime.so`.

## Engines

Each provider is a separate library that ONNX Runtime loads only when a session asks for it, and the core library links no vendor runtime. A missing vendor runtime disables that provider only. Details are in [docs/engines.md](docs/engines.md).

| Engine (execution provider) | Hardware | Vendor runtime packages | Provided by |
|---|---|---|---|
| CPU | Any x86_64 CPU | none | built in |
| CUDA | NVIDIA GPU | `nvidia-utils`, `cuda`, `cudnn` | Arch `extra` |
| TensorRT | NVIDIA GPU | TensorRT 11 | `nvidia-tensorrt-sdk-local` from the omarchy-nvidia-sdks repository (under `/opt/nvidia/tensorrt`), or the AUR's `tensorrt` (under `/usr`) |
| OpenVINO, device `CPU` | Intel CPU | `openvino` | Arch `extra` |
| OpenVINO, device `GPU` | Intel graphics | `openvino-intel-gpu-plugin`, `intel-compute-runtime`, `level-zero-loader` | Arch `extra` |
| OpenVINO, device `NPU` | Intel NPU | `openvino-intel-npu-plugin`, `intel-npu-driver`, `level-zero-loader` | Arch `extra` |

The TensorRT provider needs the CUDA provider's runtimes as well. `cuda`, `cudnn` and `openvino` (and a TensorRT, if you build that provider) must be installed when you build; at run time you need only the engines you use.

**AMD GPUs.** CUDA and ROCm cannot be combined in one ONNX Runtime build, so this package has no ROCm provider. On an AMD GPU use Arch's `onnxruntime-rocm` or `onnxruntime-opt-rocm`. `onnxruntime-full` conflicts with them (and with `onnxruntime`, `onnxruntime-cpu`, `onnxruntime-cuda`, `onnxruntime-opt-cuda` and `onnxruntime-git`), so only one is installed at a time; pacman offers to remove the other.

## Licence and personal use

- **The recipe** (this repository's content) is MIT OR Apache-2.0, at your option: see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
- **The built ONNX Runtime** is MIT-licensed, plus the open-source dependencies compiled into its libraries. The PKGBUILD's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0`. The notices are installed in `/usr/share/licenses/onnxruntime-full/` (`LICENSE`, `ThirdPartyNotices.txt`).
- **Each engine's runtime is under its vendor's licence.** The CUDA, cuDNN and TensorRT runtimes are under NVIDIA's licences; OpenVINO is Apache-2.0, and the Intel graphics and NPU drivers have their own licences. Read the licence of each runtime you install.
- **Nothing from NVIDIA or Intel is redistributed by this repository.** It holds only the recipe, scripts and documentation. The package is built locally, against the runtimes you install yourself or download from the vendor yourself, and contains none of them.
- **Keep the built package on your own machine.** It is built against vendor runtimes under licences that you accepted, so do not upload it to a repository, a release page or the AUR as a binary.
- This is not legal advice.

The guard `scripts/check-no-vendor.sh` refuses vendor and binary files (libraries, archives, wheels, packages, models, engine caches, vendor headers and licence documents, makepkg work directories) and file content that carries an NVIDIA or Intel marker. Enable the hooks in your clone with:

```bash
git config core.hooksPath .githooks
```

The hooks are `pre-commit`, `pre-merge-commit` and `pre-push`. `git commit --no-verify` bypasses them, so the guard is a safety net, not an access control. `scripts/check-no-vendor.sh --all` checks the working tree, not the history.

## Quick start

The full procedure, with the prerequisites, is in [docs/install.md](docs/install.md). In short:

```bash
# build tools and the runtimes of the engines you want (docs/install.md has the lists)
sudo pacman -S --needed base-devel git cmake ninja patchelf python python-numpy python-build python-installer python-wheel python-setuptools cuda cudnn openvino

# TensorRT: nvidia-tensorrt-sdk-local from omarchy-nvidia-sdks, or the AUR's tensorrt

cd onnxruntime-full
MAKEFLAGS=-j8 makepkg -f
sudo pacman -U onnxruntime-full-1.29.0-1-x86_64.pkg.tar.zst python-onnxruntime-full-1.29.0-1-x86_64.pkg.tar.zst
cd ..

scripts/smoke-test.sh --require cpu,cuda,tensorrt,openvino:CPU,openvino:GPU,openvino:NPU
```

The build is long with CUDA and needs tens of GiB of disk. Build inputs (engines, GPU architecture, TensorRT location, parallelism) are in [docs/build.md](docs/build.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/install.md](docs/install.md) | Prerequisites, runtimes, TensorRT, building and installing the packages, verifying |
| [docs/build.md](docs/build.md) | Build inputs, dependency mirroring, ccache, parallelism and resources, a CPU-only recipe check |
| [docs/engines.md](docs/engines.md) | Per-engine notes, TensorRT plugins, ROCm, using the library from C, Rust and Python |
| [tools/ort-smoke/README.md](tools/ort-smoke/README.md) | The smoke test: targets, options, how to read its output |

## Repository layout

```
README.md
LICENSE-MIT, LICENSE-APACHE
.githooks/                        pre-commit, pre-merge-commit, pre-push: run the licence guard
onnxruntime-full/PKGBUILD         the recipe: onnxruntime-full and python-onnxruntime-full
scripts/check-no-vendor.sh        licence guard: refuses vendor and binary files
scripts/smoke-test.sh             builds and runs tools/ort-smoke against an installed ONNX Runtime
scripts/test.sh                   runs every check of this repository
scripts/tests/                    tests of the guard, of the smoke-test wrapper and of the PKGBUILD's build inputs
tools/ort-smoke/                  the smoke test program (C) and its model generator
docs/                             install, build and engine notes
```

## Testing

```bash
scripts/test.sh
```

This runs the checks that need no GPU, no network and no vendor install: shellcheck over every script and the PKGBUILD, the PKGBUILD's syntax and metadata (`makepkg --printsrcinfo`, with the names, provides and conflicts that consumers rely on), `namcap` when it is installed, every `scripts/tests/*-test.sh`, and the licence guard over the whole tree. It needs shellcheck: from `PATH`, from `SHELLCHECK=/path/to/shellcheck`, or, when `uvx` is installed, run as `uvx --from shellcheck-py shellcheck`. Checks of the packages on real hardware are done with [scripts/smoke-test.sh](tools/ort-smoke/README.md).

## Licence of this repository's content

MIT OR Apache-2.0, at your option: see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). This covers only the content of this repository. It does not apply to ONNX Runtime, to any vendor runtime, or to a package built from them.

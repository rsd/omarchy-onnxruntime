# Releasing the CUDA 12 line

The maintainer's procedure for the package `onnxruntime-provider-cuda12-legacy` (the CUDA provider built with CUDA 12.9 for Pascal and Volta; see [cuda12-legacy.md](cuda12-legacy.md)). It has its own pkgbase, `onnxruntime-split-cuda12`, its own prebuilt pkgbase, `onnxruntime-split-cuda12-bin`, its own `pkgrel` and its own GitHub release. Its PKGBUILD is generated from `onnxruntime-split/PKGBUILD`; never edit it by hand. It is released with the same `pkgver` as the main line. The one-time setup and the main line's procedure are in [release.md](release.md); read them first. The commands below are for the release `cuda12-v1.29.0-1`.

```mermaid
flowchart LR
    A[gen-cuda12.sh, srcinfo.sh --recipe] --> B[build gcc14, cuda-12.9, cudnn9.10-cuda12.9]
    B --> C[chroot build with -I]
    C --> D[check-deps.sh, check-cuda-archs.sh]
    D --> E[clone the AUR repos, add LICENSE]
    E --> F[gen-bin.sh, gh release create]
    F --> G[makepkg --verifysource, both]
    G --> H[AUR: onnxruntime-split-cuda12, then -bin]
```

## 1. Regenerate

Regenerate after any change to `onnxruntime-split/PKGBUILD` (the generated recipe differs from it only in a few input lines, a header comment and the `pkgrel`), and after a rebuild of this line alone, which is a new `cuda12_pkgrel` in `scripts/gen-cuda12.sh`:

```bash
scripts/gen-cuda12.sh                                          # writes onnxruntime-split-cuda12/PKGBUILD
scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --write   # its .SRCINFO
```

`scripts/gen-cuda12.sh --check` and `scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --check` fail when either is stale; `scripts/test.sh` runs both.

## 2. Build the CUDA 12 userland and `gcc14`

Build them from the AUR with `makepkg`, outside the chroot. The recipe's build uses GCC 14 end to end, because the `cuda-12.9` package sets nvcc's host compiler to `g++-14` and the recipe compiles its C++ with that compiler; Arch's `gcc` stays as it is. The three packages are `gcc14`, `cuda-12.9` and `cudnn9.10-cuda12.9`. `gcc14` is a long build. They are third-party AUR packages: read their PKGBUILDs. `cuda-12.9` conflicts with `cuda` and depends on `gcc14`, so build `gcc14` first, and keep the three package files: the chroot installs them from those files.

## 3. Build in a clean chroot

Pass the three packages with `-I`, as the TensorRT package is passed in [the main line's build](release.md#1-build-the-rtx-packages). A chroot cannot fetch AUR packages, and the recipe's `makedepends` name these three explicitly (`gcc14`, `cuda-12.9` and `cudnn9.10-cuda12.9`, not a bare `cuda` and `cudnn`, which an AUR helper would resolve to Arch's CUDA 13 and cuDNN 9.11 or later), so the `-I` files meet them:

```bash
cd onnxruntime-split-cuda12
extra-x86_64-build -- -I /path/to/gcc14-<ver>-x86_64.pkg.tar.zst \
  -I /path/to/cuda-12.9-<ver>-x86_64.pkg.tar.zst \
  -I /path/to/cudnn9.10-cuda12.9-<ver>-x86_64.pkg.tar.zst
```

The recipe builds the CUDA provider alone (no TensorRT, OpenVINO or Python package). `prepare()` refuses a toolkit that is not CUDA 12, and `package()` fails the build on a CUDA library with a `.so.13` soname or a CUDA provider that does not link `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12`.

## 4. Check the package file

```bash
scripts/check-deps.sh onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-1-x86_64.pkg.tar.zst \
  /path/to/cuda-12.9-<ver>-x86_64.pkg.tar.zst /path/to/cudnn9.10-cuda12.9-<ver>-x86_64.pkg.tar.zst
scripts/check-cuda-archs.sh [--cuobjdump /path/to/cuda-12.9/bin/cuobjdump] \
  onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-1-x86_64.pkg.tar.zst \
  60 61 70 75 80 86 89 90 120 60-virtual
```

The list has `90` because ONNX Runtime compiles 18 kernels for `sm_90a` whatever `ORTF_CUDA_ARCHS` says (the CUDA 13 packages have them too).

`check-deps.sh` compares the package with what is installed ([what it reports](dependencies.md#checking-built-packages-scriptscheck-depssh)). On a machine whose `cuda` and `cudnn` are CUDA 13, nothing installed meets `cuda>=12`, `cuda<13` or `cudnn<9.11`; the two package files given after the provider count as meeting them by their provides. The library rows resolve through the host's `ldconfig` cache, so a host without the CUDA 12 userland installed can show the CUDA 12 libraries as `UNRESOLVED` or `GAP`. Read the flags with that in mind, or run it on a machine that has `cuda-12.9` and `cudnn9.10-cuda12.9` installed.

`check-cuda-archs.sh` fails unless the library holds exactly the SASS of those architectures and one PTX (`compute_60`, the oldest architecture built: a preset's PTX is always that of its oldest). `--cuobjdump` points it at the `cuobjdump` of a CUDA 12.9 toolkit, where the host's is another version.

`prepare()` of this line refuses a cuDNN other than 9.10 ("the CUDA 12 legacy line needs cuDNN 9.10 (found 9.11)"), because cuDNN 9.11 dropped Pascal and Volta; the main line is not bound by it.

The package conflicts with every CUDA 13 package, so run the smoke test on a machine set up as in [cuda12-legacy.md](cuda12-legacy.md#installing).

## 5. Clone the two AUR repositories and add the licence

Do this before `gen-bin.sh` writes into `aur/onnxruntime-split-cuda12-bin` (a clone into a directory that is not empty fails). The two pkgbases are new, so the clones are empty repositories, as in [One-time setup](release.md#one-time-setup). Each AUR repository carries a 0BSD `LICENSE` for the packaging files, as the main line's two do; copy it in from there, to be committed with the first push:

```bash
git clone ssh://aur@aur.archlinux.org/onnxruntime-split-cuda12.git aur/onnxruntime-split-cuda12
git clone ssh://aur@aur.archlinux.org/onnxruntime-split-cuda12-bin.git aur/onnxruntime-split-cuda12-bin
cp aur/onnxruntime-split/LICENSE aur/onnxruntime-split-cuda12/
cp aur/onnxruntime-split/LICENSE aur/onnxruntime-split-cuda12-bin/
```

## 6. Generate the `-bin` recipe and publish

The `cuda12-` tag selects the CUDA 12 line: every asset must come from pkgbase `onnxruntime-split-cuda12`, a set that mixes it with the main line's is refused, and the pin `onnxruntime-core=<pkgver>` is accepted, because the main line's core meets it.

Before running the release command, replace `<title>` with the release title and `<notes.md>` with the path to a Markdown file of release notes.

```bash
mkdir ~/release-cuda12 && cp onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-1-x86_64.pkg.tar.zst ~/release-cuda12/
scripts/gen-bin.sh ~/release-cuda12 cuda12-v1.29.0-1
gh release create cuda12-v1.29.0-1 ~/release-cuda12/*.pkg.tar.zst ~/release-cuda12/SHA256SUMS \
  --repo rsd/omarchy-onnxruntime --title "<title>" --notes-file "<notes.md>"
```

`gen-bin.sh` writes `~/release-cuda12/SHA256SUMS` and `aur/onnxruntime-split-cuda12-bin/{PKGBUILD,.SRCINFO}`. The tag scheme is `cuda12-v<pkgver>-<pkgrel>`, apart from the main line's `v<pkgver>-<pkgrel>`.

## 7. Verify the sources and push to the AUR

Push the source package first, as in [the main line's step 5](release.md#5-push-to-the-aur). `makepkg --verifysource` checks each recipe's sources against the sums it holds. The `-bin` one downloads the release's asset. The source recipe's sources include the 1.6 GiB ONNX Runtime clone. The one-time setup assumes `SRCDEST` in `~/.config/pacman/makepkg.conf`, which makes `makepkg` reuse the clone of your builds; without it, `makepkg --verifysource` in `aur/onnxruntime-split-cuda12` clones the whole tree again. If `makepkg.conf` does not set it, point `SRCDEST` at the directory that holds the existing clone and zips, here the recipe directory `onnxruntime-split-cuda12` (an absolute path):

```bash
(cd aur/onnxruntime-split-cuda12-bin && makepkg --verifysource)

scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --aur aur/onnxruntime-split-cuda12
(cd aur/onnxruntime-split-cuda12 && SRCDEST="$PWD/../../onnxruntime-split-cuda12" makepkg --verifysource)   # drop SRCDEST=... when makepkg.conf sets it
(cd aur/onnxruntime-split-cuda12 && git add LICENSE PKGBUILD .SRCINFO && git commit -m '1.29.0-1' && git push)

(cd aur/onnxruntime-split-cuda12-bin && git add LICENSE PKGBUILD .SRCINFO && git commit -m '1.29.0-1' && git push)
```

## The core pin

The provider depends on `onnxruntime-core=<pkgver>`, any pkgrel, and not on the core of its own build, because the core links no CUDA library and the provider has no version handshake with it. What the two share is ONNX Runtime's version and the layout-relevant build defines (`DISABLE_*`, `ENABLE_*`, the C++ standard, `_GLIBCXX_USE_CXX11_ABI`). A core that changes one of those would break the provider without any dependency noticing. The rule is therefore:

- **A main-line release that changes `cmake` flags or patches** also rebuilds and re-tests the CUDA 12 line, as a new `cuda12_pkgrel` released under a new tag.
- **A main-line release that only changes packaging** (dependencies, descriptions, licence files, a vendor bound) does not.

## When ONNX Runtime drops CUDA 12

Upstream has deprecated CUDA 12 since 1.27. When the main line's ONNX Runtime moves past the last version that builds with it, this line stays at that version and is not rebuilt, because a provider built for one ONNX Runtime version cannot follow a newer core. Continuing it past that point would need its own versioned core, which is a different design from this one, and the decision belongs before the main line leaves that version.

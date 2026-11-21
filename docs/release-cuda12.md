# Releasing the CUDA 12 line

The maintainer's procedure for the package `onnxruntime-provider-cuda12-legacy` (the CUDA provider built with CUDA 12.9 for Pascal and Volta; see [cuda12-legacy.md](cuda12-legacy.md)). It has its own pkgbase, `onnxruntime-split-cuda12`, its own prebuilt pkgbase, `onnxruntime-split-cuda12-bin`, its own `pkgrel` and its own GitHub release. Its PKGBUILD is generated from `onnxruntime-split/PKGBUILD`; never edit it by hand. It is released with the same `pkgver` as the main line, and, for 1.29.0, together with it: the main line's 1.29.0-7 and this line's 1.29.0-2 are built and released in one round ([the joint release rule](release.md#joint-release)). The one-time setup and the main line's procedure are in [release.md](release.md); read them first. The commands below are for the release `cuda12-v1.29.0-2`.

```mermaid
flowchart LR
    A[gen-cuda12.sh, srcinfo.sh --recipe] --> B[build gcc14, cuda-12.9, cudnn9.10-cuda12.9, nccl-cuda12.9]
    B --> C[chroot build with -I]
    C --> D[check-deps.sh, check-cuda-archs.sh]
    D --> E[clone the AUR repos, add LICENSE]
    E --> F[gen-bin.sh, gh release create]
    F --> G[makepkg --verifysource, both]
    G --> H[AUR: onnxruntime-split-cuda12, then -bin]
```

## 1. Regenerate

Regenerate after any change to `onnxruntime-split/PKGBUILD` (the generated recipe differs from it only in a few input lines, a header comment and the `pkgrel`), and after a rebuild of this line alone, which is a new `line_pkgrel` in `scripts/gen-cuda12.sh`:

```bash
scripts/gen-cuda12.sh                                          # writes onnxruntime-split-cuda12/PKGBUILD
scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --write   # its .SRCINFO
```

`scripts/gen-cuda12.sh --check` and `scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --check` fail when either is stale; `scripts/test.sh` runs both.

## 2. Build the CUDA 12 userland, `nccl-cuda12.9` and `gcc14`

Build them from the AUR with `makepkg`, outside the chroot. The recipe's build uses GCC 14 end to end, because the `cuda-12.9` package sets nvcc's host compiler to `g++-14` and the recipe compiles its C++ with that compiler; Arch's `gcc` stays as it is. The four packages are `gcc14`, `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9`. `gcc14` is a long build. They are third-party AUR packages: read their PKGBUILDs. `cuda-12.9` conflicts with `cuda` and depends on `gcc14`, so build `gcc14` first. `nccl-cuda12.9` provides and conflicts with `nccl` and makedepends `gcc14`, `git`, `cuda=12.9` and `python3`: build it in a clean chroot too, with `gcc14` and `cuda-12.9` passed with `-I` (`cuda-12.9` provides `cuda=12.9`, which meets that makedepend). Keep the four package files: the chroot installs them from those files. The line's NCCL is that package, by name, because Arch's `nccl` is built for `sm_75` and newer and does not run on the GPUs this line is for ([cuda12-legacy.md](cuda12-legacy.md#what-it-is)).

## 3. Build in a clean chroot

Pass the four packages with `-I`, as the TensorRT package is passed in [the main line's build](release.md#1-build-the-rtx-packages). A chroot cannot fetch AUR packages, and the recipe's `makedepends` name these four explicitly (`gcc14`, `cuda-12.9`, `cudnn9.10-cuda12.9` and `nccl-cuda12.9`, not a bare `cuda`, `cudnn` and `nccl`, which an AUR helper would resolve to Arch's CUDA 13, cuDNN 9.11 or later and NCCL), so the `-I` files meet them:

```bash
cd onnxruntime-split-cuda12
extra-x86_64-build -- -I /path/to/gcc14-<ver>-x86_64.pkg.tar.zst \
  -I /path/to/cuda-12.9-<ver>-x86_64.pkg.tar.zst \
  -I /path/to/cudnn9.10-cuda12.9-<ver>-x86_64.pkg.tar.zst \
  -I /path/to/nccl-cuda12.9-<ver>-x86_64.pkg.tar.zst
```

The recipe builds the CUDA provider alone (no TensorRT, OpenVINO, oneDNN or Python package), with the training operators and NCCL, as the main line does. `prepare()` refuses a toolkit that is not CUDA 12, and `package()` fails the build on a CUDA library with a `.so.13` soname or a CUDA provider that does not link `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12`.

## 4. Check the package file

```bash
scripts/check-deps.sh onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-2-x86_64.pkg.tar.zst \
  /path/to/cuda-12.9-<ver>-x86_64.pkg.tar.zst /path/to/cudnn9.10-cuda12.9-<ver>-x86_64.pkg.tar.zst \
  /path/to/nccl-cuda12.9-<ver>-x86_64.pkg.tar.zst
scripts/check-cuda-archs.sh [--cuobjdump /path/to/cuda-12.9/bin/cuobjdump] \
  onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-2-x86_64.pkg.tar.zst \
  60 61 70 75 80 86 89 90 120 60-virtual
```

The list has `90` because ONNX Runtime compiles 18 kernels for `sm_90a` whatever `ORTF_CUDA_ARCHS` says (the CUDA 13 packages have them too).

`check-deps.sh` compares the package with what is installed ([what it reports](dependencies.md#checking-built-packages-scriptscheck-depssh)). On a machine whose `cuda` and `cudnn` are CUDA 13, nothing installed meets `cuda>=12`, `cuda<13` or `cudnn<9.11`; the package files given after the provider count as meeting them by their provides (and `nccl-cuda12.9` meets the `nccl-cuda12.9` dependency). The library rows resolve through the host's `ldconfig` cache, so a host without the CUDA 12 userland installed can show the CUDA 12 libraries as `UNRESOLVED` or `GAP`. Read the flags with that in mind, or run it on a machine that has `cuda-12.9` and `cudnn9.10-cuda12.9` installed.

`check-cuda-archs.sh` fails unless the library holds exactly the SASS of those architectures and one PTX (`compute_60`, the oldest architecture built: a preset's PTX is always that of its oldest). `--cuobjdump` points it at the `cuobjdump` of a CUDA 12.9 toolkit, where the host's is another version.

`prepare()` of this line refuses a cuDNN other than 9.10 ("the CUDA 12 legacy line needs cuDNN 9.10 (found 9.11)"), because cuDNN 9.11 dropped Pascal and Volta; the main line is not bound by it.

The package conflicts with every CUDA 13 package, so run the smoke test on a machine set up as in [cuda12-legacy.md](cuda12-legacy.md#installing).

## 5. Clone the two AUR repositories and add the licence

Do this before `gen-bin.sh` writes into `aur/onnxruntime-split-cuda12-bin` (a clone into a directory that is not empty fails). Once the clones exist, as they do after the 1.29.0-1 release, skip this step and `git pull` in each. For a first release the two pkgbases are new, so the clones are empty repositories, as in [One-time setup](release.md#one-time-setup). Each AUR repository carries a 0BSD `LICENSE` for the packaging files, as the main line's two do; copy it in from there, to be committed with the first push:

```bash
git clone ssh://aur@aur.archlinux.org/onnxruntime-split-cuda12.git aur/onnxruntime-split-cuda12
git clone ssh://aur@aur.archlinux.org/onnxruntime-split-cuda12-bin.git aur/onnxruntime-split-cuda12-bin
cp aur/onnxruntime-split/LICENSE aur/onnxruntime-split-cuda12/
cp aur/onnxruntime-split/LICENSE aur/onnxruntime-split-cuda12-bin/
```

## 6. Generate the `-bin` recipe and publish

The `cuda12-` tag selects the CUDA 12 line: every asset must come from pkgbase `onnxruntime-split-cuda12`, a set that mixes it with the main line's is refused. The pin on the main line's core must be the pair `onnxruntime-core>=<pkgver>-<N>` and `onnxruntime-core<<next patch release>`, which the main line's core meets, with `N` at least `_ortf_core_abi_pkgrel` of `onnxruntime-split/PKGBUILD` (7 for 1.29.0): a lower `N` would admit cores of another interface and is refused. The bare `onnxruntime-core=<pkgver>`, which any core of the version meets, is accepted only while that pkgrel is 1, so `cuda12-v1.29.0-2` is refused, as are a half pair and an inclusive upper bound. A bare `nccl` dependency is refused too: this line's is `nccl-cuda12.9`, carried over as it is.

Before running the release command, replace `<title>` with the release title and `<notes.md>` with the path to a Markdown file of release notes.

```bash
mkdir ~/release-cuda12 && cp onnxruntime-split-cuda12/onnxruntime-provider-cuda12-legacy-1.29.0-2-x86_64.pkg.tar.zst ~/release-cuda12/
scripts/gen-bin.sh ~/release-cuda12 cuda12-v1.29.0-2
gh release create cuda12-v1.29.0-2 ~/release-cuda12/*.pkg.tar.zst ~/release-cuda12/SHA256SUMS \
  --repo rsd/omarchy-onnxruntime --title "<title>" --notes-file "<notes.md>"
```

`gen-bin.sh` writes `~/release-cuda12/SHA256SUMS` and `aur/onnxruntime-split-cuda12-bin/{PKGBUILD,.SRCINFO}`. The tag scheme is `cuda12-v<pkgver>-<pkgrel>`, apart from the main line's `v<pkgver>-<pkgrel>`.

## 7. Verify the sources and push to the AUR

Push the source package first, as in [the main line's step 5](release.md#5-push-to-the-aur). `makepkg --verifysource` checks each recipe's sources against the sums it holds. The `-bin` one downloads the release's asset. The source recipe's sources include the ONNX Runtime source: the mirror's 264 MiB archive, which is the same asset as the main line's ([the source mirror](release.md#the-source-mirror)), or with `ORTF_SOURCE=upstream` the 1.6 GiB clone. The one-time setup assumes `SRCDEST` in `~/.config/pacman/makepkg.conf`, which makes `makepkg` reuse the downloads of your builds; without it, `makepkg --verifysource` in `aur/onnxruntime-split-cuda12` downloads everything again. If `makepkg.conf` does not set it, point `SRCDEST` at the directory that holds the existing downloads, here the recipe directory `onnxruntime-split-cuda12` (an absolute path):

```bash
(cd aur/onnxruntime-split-cuda12-bin && makepkg --verifysource)

scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --aur aur/onnxruntime-split-cuda12
(cd aur/onnxruntime-split-cuda12 && SRCDEST="$PWD/../../onnxruntime-split-cuda12" makepkg --verifysource)   # drop SRCDEST=... when makepkg.conf sets it
(cd aur/onnxruntime-split-cuda12 && git add LICENSE PKGBUILD .SRCINFO && git commit -m '1.29.0-2' && git push)

(cd aur/onnxruntime-split-cuda12-bin && git add LICENSE PKGBUILD .SRCINFO && git commit -m '1.29.0-2' && git push)
```

## The core pin

The provider depends on `onnxruntime-core>=1.29.0-7` and `onnxruntime-core<1.29.1`, not on the core of its own build, because the core links no CUDA library and the provider has no version handshake with it. What the two share is ONNX Runtime's version and the layout-relevant build defines (`DISABLE_*`, `ENABLE_*`, the C++ standard, `_GLIBCXX_USE_CXX11_ABI`) and, since the training operators and NCCL, the interface the core exposes to the providers: `ENABLE_STRIDED_TENSORS` and the NCCL member each add virtual members to it. A core that changes one of those breaks the provider without any dependency noticing, and the result is a crash. The pair is the guard: `1.29.0-7` is `_ortf_core_abi_pkgrel`, the first core with the new interface, and the upper bound `1.29.1` excludes the next upstream patch release. The rule is therefore:

- **A main-line release that changes `cmake` flags or patches in a way that alters that interface** raises `_ortf_core_abi_pkgrel`, and so also rebuilds and re-tests the CUDA 12 line, as a new `line_pkgrel` released under a new tag. Do this in the same round.
- **A main-line release that only changes packaging** (dependencies, descriptions, licence files, a vendor bound) does not. The pair still holds for the new pkgrel, and `_ortf_core_abi_pkgrel` stays.
- **Why not leave the pkgrel unpinned**, with `onnxruntime-core=1.29.0` only (the 1.29.0-1 pin): pacman would keep the old provider on the new core. Why not an exact pin on the core's pkgrel: that would force this line to be rebuilt on every packaging change, which the pair avoids. The core conflicts with the 1.29.0-1 provider and its `-bin` as well, as a second guard.
- **A `pkgver` bump** moves the pair (`>=<new pkgver>-1`, `<` the next patch release) and resets `_ortf_core_abi_pkgrel` to 1.

## When ONNX Runtime drops CUDA 12

Upstream has deprecated CUDA 12 since 1.27. When the main line's ONNX Runtime moves past the last version that builds with it, this line stays at that version and is not rebuilt, because a provider built for one ONNX Runtime version cannot follow a newer core. Continuing it past that point would need its own versioned core, which is a different design from this one, and the decision belongs before the main line leaves that version.

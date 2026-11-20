# Testing

Two kinds of checks: the repository's own checks, which need no GPU, no network and no vendor install, and the checks of the packages on real hardware.

## `scripts/test.sh`: the repository's checks

```bash
scripts/test.sh
```

It runs, in order:

- shellcheck over every shell script (the hooks, `scripts/`, `scripts/tests/`, `tools/`) and the PKGBUILD;
- the PKGBUILD's syntax and metadata (`makepkg --printsrcinfo`, with the names, provides and conflicts that consumers rely on);
- the CUDA 12 pkgbase `onnxruntime-split-cuda12` and the MIGraphX pkgbase `onnxruntime-split-migraphx`: their generated PKGBUILDs and `.SRCINFO` files are up to date, and the names in their committed `.SRCINFO` files;
- `namcap`, when it is installed;
- every `scripts/tests/*-test.sh` (tests of the licence guard, the smoke-test wrapper, `check-deps.sh`, `check-cuda-archs.sh`, `srcinfo.sh`, `gen-bin.sh`, `gen-cuda12.sh`, `gen-migraphx.sh` and the PKGBUILD's build inputs, including the `-nccloff` names);
- the byte-compile and `--help` of `tools/ort-model-check/model-check.py`;
- the licence guard over the whole tree.

Scratch files go to a `mktemp` directory, never into the working tree. The script exits non-zero when any check fails. It does not build the packages.

It needs shellcheck: from `PATH`, from `SHELLCHECK=/path/to/shellcheck`, or, when `uvx` is installed, run as `uvx --from shellcheck-py shellcheck`. Because `SHELLCHECK` takes one program path, wrap a multi-word command in a script:

```bash
printf '#!/bin/sh\nexec uvx --from shellcheck-py shellcheck "$@"\n' > /path/to/shellcheck
chmod +x /path/to/shellcheck
SHELLCHECK=/path/to/shellcheck scripts/test.sh
```

## Checks of the packages on real hardware

- **Smoke test.** `scripts/smoke-test.sh` builds a small C program and runs a tiny model on every engine and device it finds, oneDNN (`dnnl`) included; the opt-in target `migraphx-load` checks that the MIGraphX provider loads and fails cleanly where there is no AMD GPU. Options and output: [tools/ort-smoke/README.md](../tools/ort-smoke/README.md). Use it after installing ([install.md](install.md#after-installing)) or against a build tree ([build-from-source.md](build-from-source.md#before-installing-the-build-tree)).
- **Model check.** `tools/ort-model-check/model-check.py` runs a real ONNX model on each engine and compares the outputs and timings with the CPU's: [tools/ort-model-check/README.md](../tools/ort-model-check/README.md).
- **Package checks.** `scripts/check-deps.sh` (libraries, owners and version bounds: [dependencies.md](dependencies.md#checking-built-packages-scriptscheck-depssh)) and `scripts/check-cuda-archs.sh` (the GPU architectures a built CUDA provider holds).
- **Results** of the smoke test and model check on a full build: [results.md](results.md).

## The licence guard and the git hooks

`scripts/check-no-vendor.sh` refuses vendor and binary files and content that carries an NVIDIA or Intel marker. This repository holds only recipes, scripts and documentation; nothing from NVIDIA, Intel or AMD ships in it, and no build output may enter it. It refuses libraries, archives, wheels, packages, models, engine caches, vendor headers and licence documents, and makepkg work directories.

```bash
scripts/check-no-vendor.sh --staged        # the git index (the default)
scripts/check-no-vendor.sh --all           # every tracked and untracked, not ignored file of the working tree
scripts/check-no-vendor.sh --range REV...  # every file added or modified by the commits `git rev-list REV...` lists
```

`--all` checks the working tree, not the history; `--range` covers commits. Enable the hooks in your clone with:

```bash
git config core.hooksPath .githooks
```

The hooks are `pre-commit` and `pre-merge-commit` (the staged files) and `pre-push` (every commit the push would send). `git commit --no-verify` bypasses them, so the guard is a safety net, not an access control.

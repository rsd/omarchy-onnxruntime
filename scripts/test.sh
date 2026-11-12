#!/usr/bin/env bash
# Runs every check of this repository that needs no GPU, no network and no
# vendor install: shellcheck over every shell script (the hooks, scripts/,
# scripts/tests/, tools/) and the PKGBUILD, the PKGBUILD's syntax and
# metadata (`makepkg --printsrcinfo` and the names, provides and conflicts
# consumers rely on), namcap when it is installed, every
# scripts/tests/*-test.sh, and the licence guard over the whole tree.
# Scratch files go to mktemp, never into the working tree.
#
#   scripts/test.sh                      # shellcheck from PATH, else uvx
#   SHELLCHECK=/path/to/shellcheck scripts/test.sh
#
# Without shellcheck on PATH and without SHELLCHECK, it runs
# `uvx --from shellcheck-py shellcheck` when uvx is installed.
# Exits non-zero when any check fails.
set -uo pipefail
shopt -s nullglob globstar

top=$(cd "$(dirname "$0")/.." && pwd)
cd "$top" || exit 2

if [[ -n ${SHELLCHECK:-} ]]; then
    shellcheck=("$SHELLCHECK")
elif command -v shellcheck >/dev/null 2>&1; then
    shellcheck=(shellcheck)
elif command -v uvx >/dev/null 2>&1; then
    shellcheck=(uvx --from shellcheck-py shellcheck)
else
    echo "shellcheck not found: install it, install uv (for uvx), or set SHELLCHECK=/path/to/shellcheck" >&2
    exit 2
fi
if ! "${shellcheck[@]}" --version >/dev/null 2>&1; then
    echo "shellcheck does not run: ${shellcheck[*]}" >&2
    exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

failed=()
run() { # run <name> <command...>
    local name=$1
    shift
    echo "== $name"
    if "$@"; then
        echo "-- ok: $name"
    else
        echo "-- FAIL: $name"
        failed+=("$name")
    fi
}

recipe=onnxruntime-full

# Every shell script, wherever it is (other contributors' files included as
# they appear).
scripts=(.githooks/pre-commit .githooks/pre-merge-commit .githooks/pre-push
    scripts/*.sh scripts/tests/*.sh tools/**/*.sh)
run "shellcheck (scripts)" "${shellcheck[@]}" -x "${scripts[@]}"
# makepkg defines the PKGBUILD's variables (SC2034: assigned but unused,
# SC2154: referenced but unassigned, e.g. $srcdir) and runs each function in
# a fresh directory (SC2164: cd without a fallback).
run "shellcheck ($recipe/PKGBUILD)" "${shellcheck[@]}" -x -s bash -e SC2034,SC2154,SC2164 "$recipe/PKGBUILD"
run "bash -n $recipe/PKGBUILD" bash -n "$recipe/PKGBUILD"

# srcinfo_has <file> <line...>: every line is in the .SRCINFO, exactly.
srcinfo_has() {
    local file=$1 line missing=0
    shift
    for line in "$@"; do
        if ! grep -qxF -- "$line" "$file"; then
            echo "missing from .SRCINFO: $line"
            missing=1
        fi
    done
    return "$missing"
}
# srcinfo_lacks <file> <extended regex>: no .SRCINFO line matches.
srcinfo_lacks() {
    if grep -nE -- "$2" "$1"; then
        echo "unexpected in .SRCINFO (above)"
        return 1
    fi
}
if command -v makepkg >/dev/null 2>&1; then
    srcinfo=$tmp/SRCINFO
    printsrcinfo() { (cd "$recipe" && makepkg --printsrcinfo >"$srcinfo"); }
    run "makepkg --printsrcinfo" printsrcinfo
    if [[ -s $srcinfo ]]; then
        T=$'\t'
        run ".SRCINFO: package names" srcinfo_has "$srcinfo" \
            'pkgbase = onnxruntime-full' 'pkgname = onnxruntime-full' 'pkgname = python-onnxruntime-full'
        # Split into the two packages' sections.
        sed -n '/^pkgname = onnxruntime-full$/,/^$/p' "$srcinfo" >"$tmp/main"
        sed -n '/^pkgname = python-onnxruntime-full$/,/^$/p' "$srcinfo" >"$tmp/python"
        run ".SRCINFO: onnxruntime-full provides and conflicts" srcinfo_has "$tmp/main" \
            "${T}provides = onnxruntime=1.29.0" "${T}provides = libonnxruntime.so" \
            "${T}conflicts = onnxruntime" "${T}conflicts = onnxruntime-cpu" \
            "${T}conflicts = onnxruntime-cuda" "${T}conflicts = onnxruntime-opt-cuda" \
            "${T}conflicts = onnxruntime-rocm" "${T}conflicts = onnxruntime-opt-rocm" \
            "${T}conflicts = onnxruntime-git"
        run ".SRCINFO: python-onnxruntime-full provides and conflicts" srcinfo_has "$tmp/python" \
            "${T}provides = python-onnxruntime=1.29.0" "${T}depends = onnxruntime-full=1.29.0-1" \
            "${T}conflicts = python-onnxruntime" "${T}conflicts = python-onnxruntime-cpu" \
            "${T}conflicts = python-onnxruntime-cuda" "${T}conflicts = python-onnxruntime-opt-cuda" \
            "${T}conflicts = python-onnxruntime-rocm" "${T}conflicts = python-onnxruntime-opt-rocm"
        # The vendor runtimes are optional: libonnxruntime.so links none.
        run ".SRCINFO: no vendor runtime in depends" srcinfo_lacks "$srcinfo" \
            "^${T}depends = (cuda|cudnn|nccl|tensorrt|openvino|nvidia-tensorrt-sdk)([<>=].*)?$"
        run ".SRCINFO: the build inputs" srcinfo_has "$srcinfo" \
            "${T}makedepends = cuda" "${T}makedepends = cudnn" "${T}makedepends = openvino" \
            "${T}optdepends = tensorrt: TensorRT under /usr, from the AUR (TensorRT provider)" \
            "${T}optdepends = nvidia-tensorrt-sdk: TensorRT under /opt/nvidia/tensorrt, e.g. nvidia-tensorrt-sdk-local from omarchy-nvidia-sdks (TensorRT provider)" \
            "${T}optdepends = openvino-intel-npu-plugin: the OpenVINO NPU device (Intel NPU)"
    fi
else
    echo "== makepkg --printsrcinfo: skipped (makepkg not found)"
fi
if command -v namcap >/dev/null 2>&1; then
    # namcap reports by severity; E: lines are errors.
    namcap_pkgbuild() {
        local out
        out=$(namcap "$recipe/PKGBUILD") || return 1
        [[ -z $out ]] || printf '%s\n' "$out"
        ! grep -q ' E: ' <<<"$out"
    }
    run "namcap ($recipe/PKGBUILD)" namcap_pkgbuild
else
    echo "== namcap: skipped (not installed; pacman -S namcap)"
fi

for t in scripts/tests/*-test.sh; do
    run "${t#scripts/tests/}" "$t"
done
run "licence guard (--all)" scripts/check-no-vendor.sh --all

if ((${#failed[@]} > 0)); then
    printf 'FAILED: %s\n' "${failed[@]}"
    exit 1
fi
echo "all checks passed"

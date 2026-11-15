#!/usr/bin/env bash
# Runs every check of this repository that needs no GPU, no network and no
# vendor install: shellcheck over every shell script (the hooks, scripts/,
# scripts/tests/, tools/) and the PKGBUILD, the PKGBUILD's syntax and
# metadata (`makepkg --printsrcinfo` and the names, provides and conflicts
# consumers rely on), namcap when it is installed, every
# scripts/tests/*-test.sh, the byte-compile and --help of
# tools/ort-model-check/model-check.py, and the licence guard over the whole
# tree.
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

recipe=onnxruntime-split

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
# The CUDA major in the CUDA-linked package names comes from the toolkit's
# cuda.h, as the PKGBUILD reads it (CUDA_VERSION / 1000); without one there,
# the PKGBUILD names them with its _ortf_cuda_major_default, 13.
cuda_h=/opt/cuda/include/cuda.h
cuda_major=13
if [[ -r $cuda_h ]] && cuda_version=$(sed -n 's/^#define CUDA_VERSION \([0-9][0-9]*\).*/\1/p' "$cuda_h" | head -n 1) &&
    [[ -n $cuda_version ]]; then
    cuda_major=$((cuda_version / 1000))
    echo "== CUDA $cuda_major ($cuda_h): the CUDA-linked names carry -cuda$cuda_major"
else
    echo "== no CUDA_VERSION in $cuda_h: the CUDA-linked names carry the default -cuda13"
fi
cu=cuda$cuda_major
if command -v makepkg >/dev/null 2>&1; then
    srcinfo=$tmp/SRCINFO
    printsrcinfo() { (cd "$recipe" && makepkg --printsrcinfo >"$srcinfo"); }
    run "makepkg --printsrcinfo" printsrcinfo
    if [[ -s $srcinfo ]]; then
        T=$'\t'
        # The default build inputs: every provider, the CUDA preset rtx.
        run ".SRCINFO: package names" srcinfo_has "$srcinfo" \
            'pkgbase = onnxruntime-split' 'pkgname = onnxruntime-core' \
            "pkgname = onnxruntime-provider-$cu-rtx" "pkgname = onnxruntime-provider-tensorrt-$cu" \
            'pkgname = onnxruntime-provider-openvino' 'pkgname = python-onnxruntime-core' \
            "${T}pkgrel = 5"
        # makepkg writes the packages in this order: onnxruntime-core first
        # (the provider packages copy from its install), the Python one last
        # (it checks its libraries against the others).
        package_order() {
            local got
            got=$(sed -n 's/^pkgname = //p' "$srcinfo" | tr '\n' ' ')
            [[ $got == "onnxruntime-core onnxruntime-provider-$cu-rtx onnxruntime-provider-tensorrt-$cu onnxruntime-provider-openvino python-onnxruntime-core " ]] ||
                { echo "package order: $got"; return 1; }
        }
        run ".SRCINFO: package order" package_order
        # Split into the packages' sections.
        section() { sed -n "/^pkgname = $1\$/,/^\$/p" "$srcinfo" >"$tmp/$2"; }
        section onnxruntime-core main
        section "onnxruntime-provider-$cu-rtx" cuda
        section "onnxruntime-provider-tensorrt-$cu" tensorrt
        section onnxruntime-provider-openvino openvino
        section python-onnxruntime-core python
        # Each package describes what it holds.
        run ".SRCINFO: package descriptions" srcinfo_has "$srcinfo" \
            "${T}pkgdesc = Cross-platform inference engine for ONNX models: the C and C++ library with the CPU execution provider" \
            "${T}pkgdesc = The CUDA execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)" \
            "${T}pkgdesc = The TensorRT execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)" \
            "${T}pkgdesc = The OpenVINO execution provider of ONNX Runtime, for onnxruntime-core (Intel CPU, GPU and NPU)" \
            "${T}pkgdesc = The Python module of ONNX Runtime, the cross-platform inference engine for ONNX models (onnxruntime-core)"
        run ".SRCINFO: no package claims every provider" srcinfo_lacks "$srcinfo" \
            "pkgdesc = .*CPU, CUDA, TensorRT and OpenVINO"
        run ".SRCINFO: onnxruntime-core" srcinfo_has "$tmp/main" \
            "${T}provides = onnxruntime=1.29.0" "${T}provides = libonnxruntime.so" \
            "${T}conflicts = onnxruntime" "${T}conflicts = onnxruntime-cpu" \
            "${T}conflicts = onnxruntime-cuda" "${T}conflicts = onnxruntime-opt-cuda" \
            "${T}conflicts = onnxruntime-rocm" "${T}conflicts = onnxruntime-opt-rocm" \
            "${T}conflicts = onnxruntime-git" \
            "${T}conflicts = onnxruntime-full" "${T}replaces = onnxruntime-full" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}optdepends = onnxruntime-provider-cuda: the CUDA execution provider" \
            "${T}optdepends = onnxruntime-provider-tensorrt: the TensorRT execution provider" \
            "${T}optdepends = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)" \
            "${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs, untested)"
        # The plain name and the one with the CUDA major, provided and
        # conflicted: one CUDA package at a time.
        run ".SRCINFO: onnxruntime-provider-$cu-rtx" srcinfo_has "$tmp/cuda" \
            "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = cuda" "${T}depends = cudnn" \
            "${T}depends = nvidia-utils" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-$cu=1.29.0" \
            "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-$cu"
        # No architecture suffix: it depends on any CUDA package of its CUDA
        # major by the name they provide, and provides the plain name the
        # optdepends use. Its TensorRT package is decided by prepare(), so
        # it is not here.
        run ".SRCINFO: onnxruntime-provider-tensorrt-$cu" srcinfo_has "$tmp/tensorrt" \
            "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = onnxruntime-provider-$cu=1.29.0" \
            "${T}depends = cuda" "${T}depends = cudnn" "${T}depends = nvidia-utils" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}provides = onnxruntime-provider-tensorrt=1.29.0" "${T}conflicts = onnxruntime-provider-tensorrt"
        run ".SRCINFO: onnxruntime-provider-tensorrt-$cu names no suffixed CUDA package" srcinfo_lacks "$tmp/tensorrt" \
            "^${T}(depends|provides|conflicts) = onnxruntime-provider-cuda[0-9x]*-|^${T}depends = onnxruntime-provider-cuda=|^${T}(provides|conflicts) = onnxruntime-provider-cuda"
        run ".SRCINFO: onnxruntime-provider-openvino" srcinfo_has "$tmp/openvino" \
            "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = openvino" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}optdepends = openvino-intel-gpu-plugin: the OpenVINO GPU device (Intel graphics)" \
            "${T}optdepends = openvino-intel-npu-plugin: the OpenVINO NPU device (Intel NPU)" \
            "${T}optdepends = intel-compute-runtime: Intel graphics compute runtime (OpenVINO GPU device)" \
            "${T}optdepends = level-zero-loader: oneAPI Level Zero loader (OpenVINO GPU and NPU devices)" \
            "${T}optdepends = intel-npu-driver: Intel NPU user-space driver (OpenVINO NPU device)"
        run ".SRCINFO: python-onnxruntime-core" srcinfo_has "$tmp/python" \
            "${T}provides = python-onnxruntime=1.29.0" "${T}depends = onnxruntime-core=1.29.0-5" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}depends = python" "${T}depends = python-numpy" "${T}depends = python-flatbuffers" \
            "${T}depends = python-packaging" "${T}depends = python-protobuf" \
            "${T}conflicts = python-onnxruntime-full" "${T}replaces = python-onnxruntime-full" \
            "${T}optdepends = python-onnx: the quantization, transformers optimisation and model tools" \
            "${T}optdepends = python-sympy: symbolic shape inference (onnxruntime.tools)" \
            "${T}conflicts = python-onnxruntime" "${T}conflicts = python-onnxruntime-cpu" \
            "${T}conflicts = python-onnxruntime-cuda" "${T}conflicts = python-onnxruntime-opt-cuda" \
            "${T}conflicts = python-onnxruntime-rocm" "${T}conflicts = python-onnxruntime-opt-rocm" \
            "${T}optdepends = onnxruntime-provider-cuda: the CUDA execution provider" \
            "${T}optdepends = onnxruntime-provider-tensorrt: the TensorRT execution provider" \
            "${T}optdepends = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)"
        # The MIGraphX package links its library into the Python module's
        # capi directory, so the Python module lists it too.
        run ".SRCINFO: python-onnxruntime-core lists the migraphx provider" srcinfo_has "$tmp/python" \
            "${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs, untested)"
        # The core and the Python module run without any vendor runtime; the
        # provider packages depend on theirs (the bounds are added at
        # package time). No vendor package is an optdepend any more, and the
        # Python package does not ask for the C library's provider packages'
        # vendors either.
        vendor="(cuda|cudnn|nccl|nvidia-utils|tensorrt|openvino[a-z-]*|intel-[a-z-]*|level-zero-loader|migraphx|hip-runtime-amd|rocm[a-z-]*)"
        run ".SRCINFO: no vendor runtime in onnxruntime-core" srcinfo_lacks "$tmp/main" \
            "^${T}(depends|optdepends) = $vendor([<>=:].*)?$"
        run ".SRCINFO: no vendor runtime in python-onnxruntime-core" srcinfo_lacks "$tmp/python" \
            "^${T}(depends|optdepends) = $vendor([<>=:].*)?$"
        run ".SRCINFO: no python-onnxruntime-core optdepend in onnxruntime-core" srcinfo_lacks "$tmp/main" \
            "^${T}optdepends = python-onnxruntime"
        # No name of the unsplit 1.29.0-1 is depended on or provided: only
        # the conflicts and replaces that upgrade from it name them. The
        # bounds on TensorRT, openvino and python are added at package
        # time, from the installed versions, so none is in .SRCINFO.
        run ".SRCINFO: the old names only in conflicts and replaces" srcinfo_lacks "$srcinfo" \
            "^${T}(depends|provides|optdepends) = (python-)?onnxruntime-full"
        run ".SRCINFO: no installed-version bounds" srcinfo_lacks "$srcinfo" \
            "^${T}depends = (tensorrt|openvino|python)[<>=]"
        # A list names the CUDA package by its architectures; the TensorRT
        # one keeps its name.
        srcinfo_120=$tmp/SRCINFO.120
        printsrcinfo_120() { (cd "$recipe" && ORTF_CUDA_ARCHS=120 makepkg --printsrcinfo >"$srcinfo_120"); }
        run "makepkg --printsrcinfo (ORTF_CUDA_ARCHS=120)" printsrcinfo_120
        run ".SRCINFO (ORTF_CUDA_ARCHS=120): package names" srcinfo_has "$srcinfo_120" \
            "pkgname = onnxruntime-provider-$cu-sm120" "pkgname = onnxruntime-provider-tensorrt-$cu"
        # tensorrt with the TensorRT package (either TensorRT package
        # provides it).
        run ".SRCINFO: the build inputs" srcinfo_has "$srcinfo" \
            "${T}makedepends = cuda" "${T}makedepends = cudnn" "${T}makedepends = openvino" \
            "${T}makedepends = tensorrt"
        # ROCm only with migraphx in ORTF_EPS: the default build neither
        # makes the MIGraphX package nor needs ROCm to build.
        run ".SRCINFO: no MIGraphX package or ROCm build input by default" srcinfo_lacks "$srcinfo" \
            "^pkgname = onnxruntime-provider-migraphx$|^${T}makedepends = (migraphx|hip-runtime-amd|rocm[a-z-]*)$"
        srcinfo_mgx=$tmp/SRCINFO.migraphx
        printsrcinfo_mgx() { (cd "$recipe" && ORTF_EPS=migraphx makepkg --printsrcinfo >"$srcinfo_mgx"); }
        run "makepkg --printsrcinfo (ORTF_EPS=migraphx)" printsrcinfo_mgx
        run ".SRCINFO (ORTF_EPS=migraphx): packages and build inputs" srcinfo_has "$srcinfo_mgx" \
            'pkgname = onnxruntime-core' 'pkgname = onnxruntime-provider-migraphx' 'pkgname = python-onnxruntime-core' \
            "${T}makedepends = migraphx" "${T}makedepends = hip-runtime-amd" \
            "${T}pkgdesc = The MIGraphX execution provider of ONNX Runtime, for onnxruntime-core (AMD GPUs; untested on AMD hardware)" \
            "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = migraphx" "${T}depends = hip-runtime-amd"
        run ".SRCINFO (ORTF_EPS=migraphx): no NVIDIA or OpenVINO package, no rocm-hip-sdk" srcinfo_lacks "$srcinfo_mgx" \
            "^pkgname = onnxruntime-provider-(cuda|tensorrt|openvino)|rocm-hip-sdk"
        # The migraphx and hip-runtime-amd bounds come from the installed
        # versions at package time.
        run ".SRCINFO (ORTF_EPS=migraphx): no installed-version bounds" srcinfo_lacks "$srcinfo_mgx" \
            "^${T}depends = (migraphx|hip-runtime-amd)[<>=]"
        srcinfo_every=$tmp/SRCINFO.every
        printsrcinfo_every() {
            (cd "$recipe" && ORTF_EPS='cuda tensorrt openvino migraphx' makepkg --printsrcinfo >"$srcinfo_every")
        }
        run "makepkg --printsrcinfo (ORTF_EPS='cuda tensorrt openvino migraphx')" printsrcinfo_every
        run ".SRCINFO (every provider): package names and build inputs" srcinfo_has "$srcinfo_every" \
            "pkgname = onnxruntime-provider-$cu-rtx" "pkgname = onnxruntime-provider-tensorrt-$cu" \
            'pkgname = onnxruntime-provider-openvino' 'pkgname = onnxruntime-provider-migraphx' \
            "${T}makedepends = cuda" "${T}makedepends = openvino" \
            "${T}makedepends = migraphx" "${T}makedepends = hip-runtime-amd"
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

# tools/ort-model-check/model-check.py: byte-compiles, and --help works with no
# onnxruntime installed (the module is imported only after the arguments are
# parsed).  The bytecode goes to a temporary directory, never into the tree.
model_check_compile() {
    PYTHONPYCACHEPREFIX=$tmp/pycache python3 -m py_compile tools/ort-model-check/model-check.py
}
model_check_help() {
    PYTHONPYCACHEPREFIX=$tmp/pycache python3 tools/ort-model-check/model-check.py --help >/dev/null
}
if command -v python3 >/dev/null 2>&1; then
    run "ort-model-check: py_compile" model_check_compile
    run "ort-model-check: --help" model_check_help
else
    echo "== ort-model-check: skipped (python3 not found)"
fi
run "licence guard (--all)" scripts/check-no-vendor.sh --all

if ((${#failed[@]} > 0)); then
    printf 'FAILED: %s\n' "${failed[@]}"
    exit 1
fi
echo "all checks passed"

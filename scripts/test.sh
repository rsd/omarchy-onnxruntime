#!/usr/bin/env bash
# Runs every check of this repository that needs no GPU, no network and no
# vendor install: shellcheck over every shell script (the hooks, scripts/,
# scripts/tests/, tools/) and the PKGBUILD, the PKGBUILD's syntax and
# metadata (`makepkg --printsrcinfo` and the names, provides and conflicts
# consumers rely on), the CUDA 12 legacy pkgbase onnxruntime-split-cuda12
# and the MIGraphX pkgbase onnxruntime-split-migraphx (their generated
# PKGBUILD and .SRCINFO up to date, and the names in their committed
# .SRCINFO), namcap when it is installed, every
# scripts/tests/*-test.sh, scripts/tests/pybind-providers-test.py (the
# Python module's patched get_available_providers, with a stub C++ module),
# the byte-compile and --help of tools/ort-model-check/model-check.py, and
# the licence guard over the whole tree.
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
            'pkgname = onnxruntime-provider-openvino' 'pkgname = onnxruntime-provider-dnnl' \
            'pkgname = onnxruntime-provider-webgpu' 'pkgname = python-onnxruntime-core' "${T}pkgrel = 8"
        # makepkg writes the packages in this order: onnxruntime-core first
        # (the provider packages copy from its install), the Python one last
        # (it checks its libraries against the others).
        package_order() {
            local got
            got=$(sed -n 's/^pkgname = //p' "$srcinfo" | tr '\n' ' ')
            [[ $got == "onnxruntime-core onnxruntime-provider-$cu-rtx onnxruntime-provider-tensorrt-$cu onnxruntime-provider-openvino onnxruntime-provider-dnnl onnxruntime-provider-webgpu python-onnxruntime-core " ]] ||
                { echo "package order: $got"; return 1; }
        }
        run ".SRCINFO: package order" package_order
        # Split into the packages' sections.
        section() { sed -n "/^pkgname = $1\$/,/^\$/p" "$srcinfo" >"$tmp/$2"; }
        section onnxruntime-core main
        section "onnxruntime-provider-$cu-rtx" cuda
        section "onnxruntime-provider-tensorrt-$cu" tensorrt
        section onnxruntime-provider-openvino openvino
        section onnxruntime-provider-dnnl dnnl
        section onnxruntime-provider-webgpu webgpu
        section python-onnxruntime-core python
        # Each package describes what it holds.
        run ".SRCINFO: package descriptions" srcinfo_has "$srcinfo" \
            "${T}pkgdesc = Cross-platform inference engine for ONNX models: the C and C++ library with the CPU execution provider" \
            "${T}pkgdesc = The CUDA execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)" \
            "${T}pkgdesc = The TensorRT execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)" \
            "${T}pkgdesc = The OpenVINO execution provider of ONNX Runtime, for onnxruntime-core (Intel CPU, GPU and NPU)" \
            "${T}pkgdesc = The oneDNN execution provider of ONNX Runtime, for onnxruntime-core (CPU)" \
            "${T}pkgdesc = The WebGPU execution provider of ONNX Runtime, for onnxruntime-core (GPUs with a Vulkan driver; a plugin provider)" \
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
            "${T}conflicts = onnxruntime-provider-cuda12-legacy<1.29.0-2" \
            "${T}conflicts = onnxruntime-provider-cuda12-legacy-bin<1.29.0-2" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" "${T}depends = nccl" \
            "${T}optdepends = onnxruntime-provider-cuda: the CUDA execution provider" \
            "${T}optdepends = onnxruntime-provider-tensorrt: the TensorRT execution provider" \
            "${T}optdepends = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)" \
            "${T}optdepends = onnxruntime-provider-dnnl: the oneDNN execution provider (CPU)" \
            "${T}optdepends = onnxruntime-provider-webgpu: the WebGPU execution provider (GPUs with a Vulkan driver; a plugin provider)" \
            "${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs; AUR pkgbase onnxruntime-split-migraphx, supported, not tested on AMD hardware)"
        # The plain name and the one with the CUDA major, provided and
        # conflicted: one CUDA package at a time.
        run ".SRCINFO: onnxruntime-provider-$cu-rtx" srcinfo_has "$tmp/cuda" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = cuda" "${T}depends = cudnn" \
            "${T}depends = nvidia-utils" "${T}depends = nccl" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-$cu=1.29.0" \
            "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-$cu" \
            "${T}conflicts = onnxruntime-core-nccloff"
        # No architecture suffix: it depends on any CUDA package of its CUDA
        # major by the name they provide, and provides the plain name the
        # optdepends use. Its TensorRT package is decided by prepare(), so
        # it is not here.
        run ".SRCINFO: onnxruntime-provider-tensorrt-$cu" srcinfo_has "$tmp/tensorrt" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = onnxruntime-provider-$cu=1.29.0" \
            "${T}depends = cuda" "${T}depends = cudnn" "${T}depends = nvidia-utils" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}provides = onnxruntime-provider-tensorrt=1.29.0" "${T}conflicts = onnxruntime-provider-tensorrt"
        run ".SRCINFO: onnxruntime-provider-tensorrt-$cu names no suffixed CUDA package" srcinfo_lacks "$tmp/tensorrt" \
            "^${T}(depends|provides|conflicts) = onnxruntime-provider-cuda[0-9x]*-|^${T}depends = onnxruntime-provider-cuda=|^${T}(provides|conflicts) = onnxruntime-provider-cuda"
        run ".SRCINFO: onnxruntime-provider-openvino" srcinfo_has "$tmp/openvino" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = openvino" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}optdepends = openvino-intel-gpu-plugin: the OpenVINO GPU device (Intel graphics)" \
            "${T}optdepends = openvino-intel-npu-plugin: the OpenVINO NPU device (Intel NPU)" \
            "${T}optdepends = intel-compute-runtime: Intel graphics compute runtime (OpenVINO GPU device)" \
            "${T}optdepends = level-zero-loader: oneAPI Level Zero loader (OpenVINO GPU and NPU devices)" \
            "${T}optdepends = intel-npu-driver: Intel NPU user-space driver (OpenVINO NPU device)"
        # oneDNN is linked in statically: no onednn, nothing it provides,
        # and libgomp, its OpenMP runtime.
        run ".SRCINFO: onnxruntime-provider-dnnl" srcinfo_has "$tmp/dnnl" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = libgomp" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++"
        run ".SRCINFO: onnxruntime-provider-dnnl depends on no oneDNN package and provides nothing" srcinfo_lacks "$tmp/dnnl" \
            "^${T}(depends|optdepends) = (onednn|onednn-[a-z-]*|intel-oneapi[a-z-]*|libdnnl[.a-z0-9]*)([<>=].*)?$|^${T}provides = "
        # Dawn is linked in statically: the Vulkan loader, which Dawn opens
        # at run time, and a Vulkan driver per vendor as optdepends; no
        # Dawn package, nothing provided, and Dawn's build inputs are
        # makedepends only.
        run ".SRCINFO: onnxruntime-provider-webgpu" srcinfo_has "$tmp/webgpu" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = vulkan-icd-loader" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}optdepends = vulkan-intel: the Vulkan driver for Intel graphics (Mesa ANV)" \
            "${T}optdepends = vulkan-radeon: the Vulkan driver for AMD graphics (Mesa RADV)" \
            "${T}optdepends = nvidia-utils: the Vulkan driver for NVIDIA graphics (the proprietary driver)"
        run ".SRCINFO: onnxruntime-provider-webgpu depends on no Dawn or SPIR-V package and provides nothing" srcinfo_lacks "$tmp/webgpu" \
            "^${T}depends = ([a-z-]*dawn[a-z-]*|spirv-[a-z]*|vulkan-headers|vulkan-utility-libraries|python-[a-z]*)([<>=].*)?$|^${T}provides = "
        run ".SRCINFO: python-onnxruntime-core" srcinfo_has "$tmp/python" \
            "${T}provides = python-onnxruntime=1.29.0" "${T}depends = onnxruntime-core=1.29.0-8" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}depends = python" "${T}depends = python-numpy" "${T}depends = python-flatbuffers" \
            "${T}depends = python-packaging" "${T}depends = python-protobuf" \
            "${T}depends = python-cerberus" "${T}depends = python-h5py" "${T}depends = python-onnx" \
            "${T}depends = python-sympy" "${T}depends = python-setuptools" \
            "${T}conflicts = python-onnxruntime-full" "${T}replaces = python-onnxruntime-full" \
            "${T}optdepends = python-pytorch: onnxruntime.training.ortmodule (import onnxruntime.training needs it)" \
            "${T}conflicts = python-onnxruntime" "${T}conflicts = python-onnxruntime-cpu" \
            "${T}conflicts = python-onnxruntime-cuda" "${T}conflicts = python-onnxruntime-opt-cuda" \
            "${T}conflicts = python-onnxruntime-rocm" "${T}conflicts = python-onnxruntime-opt-rocm" \
            "${T}optdepends = onnxruntime-provider-cuda: the CUDA execution provider" \
            "${T}optdepends = onnxruntime-provider-tensorrt: the TensorRT execution provider" \
            "${T}optdepends = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)" \
            "${T}optdepends = onnxruntime-provider-dnnl: the oneDNN execution provider (CPU)" \
            "${T}optdepends = onnxruntime-provider-webgpu: the WebGPU execution provider (GPUs with a Vulkan driver; a plugin provider)" \
            "${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs; AUR pkgbase onnxruntime-split-migraphx, supported, not tested on AMD hardware)"
        # The MIGraphX package is an optdepend of every build, of
        # onnxruntime-core and python-onnxruntime-core alone (the AUR pkgbase
        # onnxruntime-split-migraphx makes it; it is never prebuilt).
        mgx_optdepends_default() {
            local want="${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs; AUR pkgbase onnxruntime-split-migraphx, supported, not tested on AMD hardware)"
            [[ $(grep -c "^${T}optdepends = onnxruntime-provider-migraphx" "$srcinfo") == 2 &&
                $(grep -cxF -- "$want" "$srcinfo") == 2 ]] ||
                { grep -n migraphx "$srcinfo"; return 1; }
        }
        run ".SRCINFO: two MIGraphX optdepends by default, naming onnxruntime-split-migraphx, not tested" mgx_optdepends_default
        # The core and the Python module run without any vendor runtime but
        # the core's NCCL (nccl, checked above: CMake links it into
        # libonnxruntime.so and the Python module); the provider packages depend on theirs (the
        # bounds are added at package time). No vendor package is an
        # optdepend any more, and the Python package does not ask for the C
        # library's provider packages' vendors either.
        vendor="(cuda|cudnn|nccl|rccl|nvidia-utils|tensorrt|openvino[a-z-]*|intel-[a-z-]*|level-zero-loader|migraphx|hip-runtime-amd|rocm[a-z-]*|onednn|libgomp)"
        run ".SRCINFO: no vendor runtime in onnxruntime-core but nccl" srcinfo_lacks "$tmp/main" \
            "^${T}(depends|optdepends) = ${vendor/nccl|/}([<>=:].*)?$|^${T}(optdepends = nccl|depends = nccl[<>=])"
        # Every provider and the Python package conflict with
        # onnxruntime-core-nccloff (a build without NCCL, whose provide of
        # onnxruntime-core would meet their depends); the core does not.
        nccloff_conflicts() {
            local f
            for f in cuda tensorrt openvino dnnl webgpu python; do
                grep -qxF "${T}conflicts = onnxruntime-core-nccloff" "$tmp/$f" || { echo "$f: no conflict with onnxruntime-core-nccloff"; return 1; }
            done
            ! grep -qF 'nccloff' "$tmp/main" || { echo "onnxruntime-core names nccloff"; return 1; }
        }
        run ".SRCINFO: the providers and the Python package conflict with onnxruntime-core-nccloff" nccloff_conflicts
        run ".SRCINFO: python-onnxruntime-core depends on nccl" srcinfo_has "$tmp/python" "${T}depends = nccl"
        run ".SRCINFO: no vendor runtime in python-onnxruntime-core but nccl" srcinfo_lacks "$tmp/python" \
            "^${T}(depends|optdepends) = ${vendor/nccl|/}([<>=:].*)?$|^${T}(optdepends = nccl|depends = nccl[<>=])"
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
            "${T}makedepends = tensorrt" "${T}makedepends = nccl" \
            "${T}makedepends = python-jinja" "${T}makedepends = python-markupsafe" "${T}makedepends = spirv-headers" \
            "${T}makedepends = spirv-tools" "${T}makedepends = vulkan-headers" "${T}makedepends = vulkan-utility-libraries" \
            "${T}source = ortf-dep-dawn.zip::https://github.com/google/dawn/archive/refs/tags/v20260714.215939.zip"
        # ORTF_NCCL=0: every package named -nccloff, no NCCL build input.
        srcinfo_off=$tmp/SRCINFO.nccloff
        printsrcinfo_off() { (cd "$recipe" && ORTF_NCCL=0 makepkg --printsrcinfo >"$srcinfo_off"); }
        run "makepkg --printsrcinfo (ORTF_NCCL=0)" printsrcinfo_off
        nccloff_names() {
            local got
            got=$(sed -n 's/^pkgname = //p' "$srcinfo_off" | tr '\n' ' ')
            [[ $got == "onnxruntime-core-nccloff onnxruntime-provider-$cu-rtx-nccloff onnxruntime-provider-tensorrt-$cu-nccloff onnxruntime-provider-openvino-nccloff onnxruntime-provider-dnnl-nccloff onnxruntime-provider-webgpu-nccloff python-onnxruntime-core-nccloff " ]] ||
                { echo "pkgnames: $got"; return 1; }
            srcinfo_has "$srcinfo_off" "${T}provides = onnxruntime-core=1.29.0-8" "${T}conflicts = onnxruntime-core" \
                "${T}depends = onnxruntime-core-nccloff=1.29.0-8" || return 1
            srcinfo_lacks "$srcinfo_off" "^${T}(make)?depends = (nccl|rccl)|^${T}depends = onnxruntime-core=|conflicts = onnxruntime-core-nccloff"
        }
        run ".SRCINFO (ORTF_NCCL=0): every package -nccloff, pinned to onnxruntime-core-nccloff, no NCCL" nccloff_names
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
            "${T}pkgdesc = The MIGraphX execution provider of ONNX Runtime, for onnxruntime-core (AMD GPUs; supported, not tested on AMD hardware)" \
            "${T}depends = onnxruntime-core=1.29.0-8" "${T}depends = migraphx" "${T}depends = hip-runtime-amd"
        # The MIGraphX package links its library into the Python module's
        # capi directory, so the Python module lists it too.
        mgx_optdepends() {
            local p want="${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs; AUR pkgbase onnxruntime-split-migraphx, supported, not tested on AMD hardware)"
            for p in onnxruntime-core python-onnxruntime-core; do
                sed -n "/^pkgname = $p\$/,/^\$/p" "$srcinfo_mgx" | grep -qxF -- "$want" ||
                    { echo "no MIGraphX optdepend in $p"; return 1; }
            done
        }
        run ".SRCINFO (ORTF_EPS=migraphx): onnxruntime-core and python-onnxruntime-core list the MIGraphX provider" mgx_optdepends
        # The main line keeps the link in the MIGraphX package itself: no
        # Python add-on (the migraphx line's), no add-on optdepend.
        run ".SRCINFO (ORTF_EPS=migraphx): no python-onnxruntime-provider-migraphx in the main line" srcinfo_lacks "$srcinfo_mgx" \
            'python-onnxruntime-provider-migraphx'
        run ".SRCINFO (ORTF_EPS=migraphx): no NVIDIA or OpenVINO package, no rocm-hip-sdk" srcinfo_lacks "$srcinfo_mgx" \
            "^pkgname = onnxruntime-provider-(cuda|tensorrt|openvino|dnnl|webgpu)|rocm-hip-sdk"
        # The migraphx and hip-runtime-amd bounds come from the installed
        # versions at package time.
        run ".SRCINFO (ORTF_EPS=migraphx): no installed-version bounds" srcinfo_lacks "$srcinfo_mgx" \
            "^${T}depends = (migraphx|hip-runtime-amd)[<>=]"
        srcinfo_every=$tmp/SRCINFO.every
        printsrcinfo_every() {
            (cd "$recipe" && ORTF_EPS='cuda tensorrt openvino dnnl webgpu migraphx' makepkg --printsrcinfo >"$srcinfo_every")
        }
        run "makepkg --printsrcinfo (ORTF_EPS='cuda tensorrt openvino dnnl webgpu migraphx')" printsrcinfo_every
        run ".SRCINFO (every provider): package names and build inputs" srcinfo_has "$srcinfo_every" \
            "pkgname = onnxruntime-provider-$cu-rtx" "pkgname = onnxruntime-provider-tensorrt-$cu" \
            'pkgname = onnxruntime-provider-openvino' 'pkgname = onnxruntime-provider-dnnl' \
            'pkgname = onnxruntime-provider-webgpu' 'pkgname = onnxruntime-provider-migraphx' \
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

# The CUDA 12 legacy line: a second pkgbase, onnxruntime-split-cuda12, whose
# PKGBUILD is generated from onnxruntime-split's. Its recipe and .SRCINFO are
# checked for being up to date, and its COMMITTED .SRCINFO is read for the
# names, the pkgdesc and the dependencies: the cuda and cudnn bounds, which
# its package function lists itself (package time checks them against the
# DT_NEEDED entries and the recorded cuDNN version; pkgbuild-settings-test.sh
# checks that), never the bare names, and the AUR's CUDA 12.9 and cuDNN
# 9.10 as makedepends.
run "gen-cuda12.sh --check" scripts/gen-cuda12.sh --check
run "srcinfo.sh --recipe onnxruntime-split-cuda12 --check" scripts/srcinfo.sh --recipe onnxruntime-split-cuda12 --check
si12=onnxruntime-split-cuda12/.SRCINFO
if [[ -r $si12 ]]; then
    T=$'\t'
    srcinfo12_names() {
        local got
        got=$(sed -n 's/^pkgname = //p' "$si12" | tr '\n' ' ')
        [[ $got == 'onnxruntime-provider-cuda12-legacy ' ]] || { echo "pkgnames: $got"; return 1; }
        srcinfo_has "$si12" 'pkgbase = onnxruntime-split-cuda12' \
            "${T}pkgdesc = The CUDA execution provider of ONNX Runtime, CUDA 12 build for Pascal, Volta and newer NVIDIA GPUs, for onnxruntime-core" \
            'pkgname = onnxruntime-provider-cuda12-legacy' \
            "${T}depends = onnxruntime-core>=1.29.0-7" "${T}depends = onnxruntime-core<1.29.1" \
            "${T}depends = cuda>=12" "${T}depends = cuda<13" \
            "${T}depends = cudnn>=9.10" "${T}depends = cudnn<9.11" "${T}depends = nvidia-utils" \
            "${T}depends = nccl-cuda12.9" "${T}conflicts = onnxruntime-core-nccloff" \
            "${T}makedepends = cuda-12.9" "${T}makedepends = cudnn9.10-cuda12.9" "${T}makedepends = gcc14" \
            "${T}makedepends = nccl-cuda12.9" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-cuda12=1.29.0" \
            "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-cuda12" || return 1
        # No bare cuda or cudnn (an AUR helper would resolve them to Arch's
        # CUDA 13 and cuDNN 9.11 or later), no bare nccl to build with or
        # depend on (Arch's CUDA 13 build), and no CUDA 13 anywhere (a
        # .so.13 soname, a cuda>=13 bound, a -cuda13 name).
        srcinfo_lacks "$si12" "^${T}(make)?depends = (cuda|cudnn|nccl)\$" || return 1
        # No core pin a training-off core meets (1.29.0-1's pkgver-only one).
        srcinfo_lacks "$si12" "^${T}depends = onnxruntime-core(=.*)?\$" || return 1
        srcinfo_lacks "$si12" '\.so\.13|cuda>=13|cuda<14|cuda13'
    }
    run "$si12: names, pkgdesc, bounded depends, AUR makedepends, provides, conflicts, no CUDA 13" srcinfo12_names
else
    echo "== $si12: not found"
    failed+=("$si12: not found")
fi

# The MIGraphX line: a third pkgbase, onnxruntime-split-migraphx, generated
# from onnxruntime-split's like the cuda12 one, its committed .SRCINFO read
# for its one package, its pkgdesc (supported, not tested on AMD hardware), the pin pair
# on the main line's core, Arch's migraphx and hip-runtime-amd, no python
# (its optdepend is its Python add-on), and the add-on,
# python-onnxruntime-provider-migraphx (it links the provider's library
# into the main line's Python module): on that provider exactly, on
# python-onnxruntime-core by the pin pair and on python; no CUDA,
# OpenVINO, Python module or -bin anywhere.
run "gen-migraphx.sh --check" scripts/gen-migraphx.sh --check
run "srcinfo.sh --recipe onnxruntime-split-migraphx --check" scripts/srcinfo.sh --recipe onnxruntime-split-migraphx --check
simgx=onnxruntime-split-migraphx/.SRCINFO
if [[ -r $simgx ]]; then
    T=$'\t'
    srcinfo_mgx_names() {
        local got
        got=$(sed -n 's/^pkgname = //p' "$simgx" | tr '\n' ' ')
        [[ $got == 'onnxruntime-provider-migraphx python-onnxruntime-provider-migraphx ' ]] || { echo "pkgnames: $got"; return 1; }
        srcinfo_has "$simgx" 'pkgbase = onnxruntime-split-migraphx' "${T}pkgrel = 1" \
            "${T}pkgdesc = The MIGraphX execution provider of ONNX Runtime, for onnxruntime-core (AMD GPUs; supported, not tested on AMD hardware)" \
            "${T}depends = onnxruntime-core>=1.29.0-7" "${T}depends = onnxruntime-core<1.29.1" \
            "${T}depends = migraphx" "${T}depends = hip-runtime-amd" \
            "${T}depends = glibc" "${T}depends = libgcc" "${T}depends = libstdc++" \
            "${T}makedepends = migraphx" "${T}makedepends = hip-runtime-amd" "${T}makedepends = rccl" \
            "${T}conflicts = onnxruntime-core-nccloff" \
            "${T}optdepends = python-onnxruntime-provider-migraphx: MIGraphX from Python (MIGraphXExecutionProvider in python-onnxruntime-core)" ||
            return 1
        local prov addon
        prov=$(sed -n '/^pkgname = onnxruntime-provider-migraphx$/,/^$/p' "$simgx")
        addon=$(sed -n '/^pkgname = python-onnxruntime-provider-migraphx$/,/^$/p' "$simgx")
        # No python in the provider's depends: the add-on has it.
        ! grep -E "^${T}depends = python" <<<"$prov" || return 1
        [[ $(sed -n "s/^${T}depends = //p" <<<"$addon" | tr '\n' ' ') == \
            'onnxruntime-provider-migraphx=1.29.0-1 python-onnxruntime-core>=1.29.0-7 python-onnxruntime-core<1.29.1 python ' ]] ||
            { echo "add-on depends: $addon"; return 1; }
        grep -qxF "${T}conflicts = onnxruntime-core-nccloff" <<<"$addon" || { echo "add-on conflicts: $addon"; return 1; }
        # RCCL is a build input only: the MIGraphX provider links none.
        srcinfo_lacks "$simgx" "^${T}depends = rccl" || return 1
        # python itself builds ONNX's protobuf sources (every line) and
        # gives the site-packages of the add-on's link; no Python module is
        # built (no python- build input; the add-on's bounds on python are
        # added at package time).
        srcinfo_lacks "$simgx" "^${T}makedepends = python-|^${T}depends = python[<>=]" || return 1
        srcinfo_lacks "$simgx" "[Cc][Uu][Dd][Aa]|-bin\b|openvino|tensorrt|^${T}depends = onnxruntime-core(=.*)?\$|rocm-hip-sdk"
    }
    run "$simgx: the provider and its Python add-on, pkgdesc, pin pairs, ROCm depends, python in the add-on only, RCCL to build with, no CUDA, OpenVINO, Python module or -bin" srcinfo_mgx_names
else
    echo "== $simgx: not found"
    failed+=("$simgx: not found")
fi

for t in scripts/tests/*-test.sh; do
    run "${t#scripts/tests/}" "$t"
done
# The Python module's get_available_providers as the PKGBUILD patches it
# (_ortf_patch_pybind_providers); no ONNX Runtime needed. The bytecode of
# its temporary package trees stays in their temporary directory.
if command -v python3 >/dev/null 2>&1; then
    run "pybind-providers-test.py" env PYTHONPYCACHEPREFIX="$tmp/pycache" python3 scripts/tests/pybind-providers-test.py
else
    echo "== pybind-providers-test.py: skipped (python3 not found)"
fi

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

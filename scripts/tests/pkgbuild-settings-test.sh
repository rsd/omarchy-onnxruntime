#!/usr/bin/env bash
# Tests for the build-input logic of onnxruntime-full/PKGBUILD: how
# prepare() resolves and checks ORTF_EPS, ORTF_CUDA_ARCHS,
# ORTF_NVCC_THREADS, ORTF_TENSORRT_HOME, ORTF_PYTHON, ORTF_CCACHE,
# MAKEFLAGS and NVCC_CCBIN (_ortf_resolve), how it lays out the dependency
# mirror (_ortf_mirror), which site-packages the Python package goes to
# under /usr/bin/python and a venv (_ortf_py_site), and that
# `makepkg --printsrcinfo` does not depend on any of them. The PKGBUILD is
# sourced in a subshell per case, with its vendor locations pointed at
# fixtures under mktemp: no CUDA, TensorRT or OpenVINO is needed, nothing is
# downloaded or built.
# Usage: scripts/tests/pkgbuild-settings-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
pkgbuild=$(cd "$here/../../onnxruntime-full" && pwd)/PKGBUILD

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

# The vendor fixtures: a CUDA, a cuDNN, an OpenVINO CMake package, and two
# TensorRT trees (lib/ and lib64/). Empty files: only their names count.
fx=$tmp/fx
mkdir -p "$fx/cuda/bin" "$fx/cudnn/include" "$fx/cmake/openvino2026.4.1" \
    "$fx/trt/include" "$fx/trt/lib" "$fx/trt64/include" "$fx/trt64/lib64" "$fx/trt-nolib/include" \
    "$fx/empty" "$fx/bin"
printf '#!/bin/sh\n' >"$fx/cuda/bin/nvcc"
chmod +x "$fx/cuda/bin/nvcc"
: >"$fx/cudnn/include/cudnn.h"
: >"$fx/cmake/openvino2026.4.1/OpenVINOConfig.cmake"
: >"$fx/trt/include/NvInfer.h"
: >"$fx/trt/lib/libnvinfer.so.11"
: >"$fx/trt64/include/NvInfer.h"
: >"$fx/trt64/lib64/libnvinfer.so"
: >"$fx/trt-nolib/include/NvInfer.h"
printf "export NVCC_CCBIN='/usr/bin/g++-15'\n" >"$fx/cuda.sh"
# Stubs found first on PATH by some cases.
printf '#!/bin/sh\nexit 0\n' >"$fx/bin/python"
printf '#!/bin/sh\nexit 0\n' >"$fx/bin/ccache"
chmod +x "$fx/bin/python" "$fx/bin/ccache"

# resolve <env assignments...>: runs _ortf_resolve with the fixtures and the
# given environment (every ORTF_* variable, MAKEFLAGS and NVCC_CCBIN unset
# otherwise). Sets out (its output, stdout and stderr), status and conf
# (the file it wrote, empty when none).
resolve() {
    local dir
    dir=$(mktemp -d "$tmp/case.XXXXXX")
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_NVCC_THREADS -u ORTF_TENSORRT_HOME \
        -u ORTF_PYTHON -u ORTF_CCACHE -u MAKEFLAGS -u NVCC_CCBIN \
        ORTF_PYTHON=0 ORTF_CCACHE=0 "$@" bash -c '
            set -euo pipefail
            # shellcheck disable=SC1090
            . "$1"
            _ortf_cuda_home=$2/cuda _ortf_cudnn_home=$2/cudnn _ortf_openvino_cmake_dir=$2/cmake
            _ortf_tensorrt_default=$2/trt _ortf_cuda_profile=$2/no-cuda.sh
            [[ -z ${FX_PROFILE:-} ]] || _ortf_cuda_profile=$2/cuda.sh
            [[ -z ${FX_NO_TRT_DEFAULT:-} ]] || _ortf_tensorrt_default=$2/empty
            [[ -z ${FX_NO_VENDOR:-} ]] ||
                _ortf_cuda_home=$2/empty _ortf_cudnn_home=$2/empty _ortf_openvino_cmake_dir=$2/empty
            _ortf_resolve "$3/ortf.conf"
        ' _ "$pkgbuild" "$fx" "$dir" 2>&1) || status=$?
    conf=''
    [[ ! -f $dir/ortf.conf ]] || conf=$(<"$dir/ortf.conf")
}

# expect_ok <description> <KEY=value...> -- <env...>: resolve succeeds and
# ortf.conf has each KEY=value line.
expect_ok() {
    local desc=$1 line missing=''
    shift
    local -a want=()
    while (($#)) && [[ $1 != -- ]]; do want+=("$1"); shift; done
    shift
    resolve "$@"
    if ((status != 0)); then
        fail "$desc (exit $status): $out"
        return
    fi
    for line in "${want[@]}"; do
        grep -qxF -- "$line" <<<"$conf" || missing+=" [$line]"
    done
    if [[ -n $missing ]]; then fail "$desc: ortf.conf lacks$missing: $conf"; else ok "$desc"; fi
}

# expect_fail <description> <message fragment> <env...>: resolve fails,
# writes no ortf.conf and says <message fragment>.
expect_fail() {
    local desc=$1 msg=$2
    shift 2
    resolve "$@"
    if ((status == 0)); then
        fail "$desc: succeeded: $out"
    elif [[ -n $conf ]]; then
        fail "$desc: wrote ortf.conf anyway"
    elif ! grep -qF -- "$msg" <<<"$out"; then
        fail "$desc: message lacks '$msg': $out"
    else
        ok "$desc"
    fi
}

echo "== ORTF_EPS"
expect_ok "the default is every provider" 'ORTF_EPS=cuda tensorrt openvino' \
    "ORTF_TENSORRT_HOME=$fx/trt" "ORTF_TENSORRT_LIBDIR=$fx/trt/lib" --
if grep -qF 'ORTF_EPS=cuda tensorrt openvino (default)' <<<"$out"; then
    ok "prepare() prints the providers and their source"
else
    fail "printed: $out"
fi
expect_ok "cpu alone checks no vendor install" 'ORTF_EPS=cpu' 'ORTF_TENSORRT_HOME=' -- \
    ORTF_EPS=cpu FX_NO_VENDOR=1
if grep -qF 'ORTF_EPS=cpu (environment)' <<<"$out"; then ok "an environment value says so"; else
    fail "printed: $out"; fi
expect_ok "comma-separated, in any order, comes out canonical" 'ORTF_EPS=cuda openvino' -- \
    ORTF_EPS=openvino,cuda
expect_ok "cuda alone" 'ORTF_EPS=cuda' 'ORTF_TENSORRT_HOME=' -- ORTF_EPS=cuda
expect_fail "cpu with another provider fails" 'cpu stands alone' ORTF_EPS='cpu cuda'
expect_fail "tensorrt without cuda fails" 'tensorrt needs cuda' ORTF_EPS=tensorrt
expect_fail "an unknown provider fails" "unknown provider 'rocm'" ORTF_EPS='cuda rocm'
expect_fail "only separators fail" 'ORTF_EPS: empty' ORTF_EPS=' , '
expect_fail "cuda without nvcc fails" 'nvcc not found' ORTF_EPS=cuda FX_NO_VENDOR=1
expect_fail "openvino without its CMake package fails" 'OpenVINOConfig.cmake' ORTF_EPS=openvino \
    FX_NO_VENDOR=1

echo "== ORTF_TENSORRT_HOME"
expect_ok "an explicit home with lib64/" "ORTF_TENSORRT_HOME=$fx/trt64" \
    "ORTF_TENSORRT_LIBDIR=$fx/trt64/lib64" -- "ORTF_TENSORRT_HOME=$fx/trt64/"
expect_fail "a relative home fails" 'not an absolute path' ORTF_TENSORRT_HOME=opt/trt
expect_fail "a home without NvInfer.h fails" "$fx/empty/include/NvInfer.h not found" \
    "ORTF_TENSORRT_HOME=$fx/empty"
expect_fail "a home without libnvinfer fails" 'no libnvinfer.so' "ORTF_TENSORRT_HOME=$fx/trt-nolib"
if [[ -f /usr/include/NvInfer.h ]]; then
    expect_ok "without /opt/nvidia/tensorrt, /usr is used" 'ORTF_TENSORRT_HOME=/usr' -- FX_NO_TRT_DEFAULT=1
else
    expect_fail "without /opt/nvidia/tensorrt, /usr is used (and has no TensorRT here)" \
        '/usr/include/NvInfer.h not found' FX_NO_TRT_DEFAULT=1
fi

echo "== ORTF_CUDA_ARCHS and ORTF_NVCC_THREADS"
expect_ok "the default architecture is 120" 'ORTF_CUDA_ARCHS=120' 'ORTF_NVCC_THREADS=1' -- ORTF_EPS=cuda
expect_ok "sm_ names, spaces and commas become a CMake list" 'ORTF_CUDA_ARCHS=120;89;120-virtual' -- \
    ORTF_EPS=cuda ORTF_CUDA_ARCHS='sm_120 sm89,120-virtual'
expect_ok "real and one virtual architecture" 'ORTF_CUDA_ARCHS=89-real;120;120-virtual' -- \
    ORTF_EPS=cuda ORTF_CUDA_ARCHS='89-real 120 120-virtual'
for w in 120a 120f 120a-real sm_120a 120f-virtual; do
    expect_fail "an a/f suffix fails ($w)" 'without the a/f suffix' ORTF_EPS=cuda ORTF_CUDA_ARCHS="$w"
done
expect_fail "two virtual architectures fail" 'one -virtual architecture only' ORTF_EPS=cuda \
    ORTF_CUDA_ARCHS='89-virtual 120-virtual'
expect_fail "a non-architecture fails" "'native' is not a CUDA architecture" ORTF_CUDA_ARCHS=native
expect_fail "an empty list fails" 'ORTF_CUDA_ARCHS: empty' ORTF_CUDA_ARCHS=';'
expect_ok "nvcc threads" 'ORTF_NVCC_THREADS=2' -- ORTF_EPS=cuda ORTF_NVCC_THREADS=2
expect_fail "zero nvcc threads fail" 'not a positive number' ORTF_NVCC_THREADS=0

echo "== NVCC_CCBIN"
expect_ok "nvcc's host compiler defaults to g++" 'NVCC_CCBIN=/usr/bin/g++' -- ORTF_EPS=cuda
expect_ok "it is read from the cuda profile script" 'NVCC_CCBIN=/usr/bin/g++-15' -- ORTF_EPS=cuda FX_PROFILE=1
expect_ok "the environment wins" 'NVCC_CCBIN=/opt/gcc14/bin/g++' -- ORTF_EPS=cuda FX_PROFILE=1 \
    NVCC_CCBIN=/opt/gcc14/bin/g++
expect_ok "no host compiler without cuda" 'NVCC_CCBIN=' -- ORTF_EPS=openvino

echo "== ORTF_PYTHON and ORTF_CCACHE"
expect_ok "python off" 'ORTF_PYTHON=0' 'ORTF_PYTHON_EXE=' -- ORTF_EPS=cpu
expect_ok "python on uses the python on PATH" 'ORTF_PYTHON=1' "ORTF_PYTHON_EXE=$fx/bin/python" -- \
    ORTF_EPS=cpu ORTF_PYTHON=1 PATH="$fx/bin:$PATH"
if grep -qF 'is not /usr/bin/python' <<<"$out"; then ok "a python other than /usr/bin/python is flagged"; else
    fail "printed: $out"; fi
expect_fail "ORTF_PYTHON=2 fails" "'2' is neither 0 nor 1" ORTF_PYTHON=2
expect_ok "ccache off" 'ORTF_CCACHE=' -- ORTF_EPS=cpu
expect_ok "ccache auto finds it on PATH" "ORTF_CCACHE=$fx/bin/ccache" -- ORTF_EPS=cpu ORTF_CCACHE=auto \
    PATH="$fx/bin:$PATH"
expect_ok "ccache required, found" "ORTF_CCACHE=$fx/bin/ccache" -- ORTF_EPS=cpu ORTF_CCACHE=1 \
    PATH="$fx/bin:$PATH"
expect_fail "a bogus ORTF_CCACHE fails" "'yes' is not auto, 0 or 1" ORTF_CCACHE=yes

echo "== MAKEFLAGS"
for case in '-j8=8' '-j 6=6' '--jobs=4=4' '--jobs 3=3' '-l3 -j12=12' '-j2 -j5=5' '=' '-j=' '-k='; do
    flags=${case%=*} jobs=${case##*=}
    expect_ok "MAKEFLAGS='$flags' gives jobs '$jobs'" "ORTF_JOBS=$jobs" -- ORTF_EPS=cpu MAKEFLAGS="$flags"
done

echo "== the dependency mirror"
mirror_case() { # mirror_case <deps.txt editor (sed script)>: sets status, out
    local dir
    dir=$(mktemp -d "$tmp/mirror.XXXXXX")
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(bash -c '
        set -euo pipefail
        # shellcheck disable=SC1090
        . "$1"
        srcdir=$2
        for entry in "${source[@]}"; do
            [[ $entry == ortf-dep-*::* ]] || continue
            file=${entry%%::*} name=${file#ortf-dep-} name=${name%.zip}
            printf "fake %s\n" "$name" >"$srcdir/$file"
            printf "%s;%s;%s\n" "$name" "${entry#*::}" "$(sha1sum "$srcdir/$file" | cut -d" " -f1)"
        done | sed -e "$3" >"$srcdir/deps.txt"
        _ortf_mirror "$srcdir/deps.txt" "$srcdir/mirror"
        n=0
        for entry in "${source[@]}"; do
            [[ $entry == ortf-dep-*::* ]] || continue
            url=${entry#*::}
            [[ -L $srcdir/mirror/${url#https://} && -f $srcdir/mirror/${url#https://} ]] ||
                { echo "not mirrored: $url"; exit 1; }
            n=$((n + 1))
        done
        echo "mirrored $n of ${#_ortf_deps[@]}"
    ' _ "$pkgbuild" "$dir" "$1" 2>&1) || status=$?
}
mirror_case ''
if ((status == 0)) && [[ $out =~ ^mirrored\ ([0-9]+)\ of\ ([0-9]+)$ ]] &&
    [[ ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]]; then
    ok "every archive is linked at its URL path ($out)"
else
    fail "the mirror (exit $status): $out"
fi
mirror_case 's/^\(protobuf;[^;]*;\).*/\1da39a3ee5e6b4b0d3255bfef95601890afd80709/'
if ((status != 0)) && grep -qF 'protobuf: SHA-1' <<<"$out"; then ok "a wrong SHA-1 fails"; else
    fail "a wrong SHA-1 (exit $status): $out"; fi
mirror_case 's|^onnx;[^;]*;|onnx;https://github.com/onnx/onnx/archive/refs/tags/v9.9.9.zip;|'
if ((status != 0)) && grep -qF 'onnx: cmake/deps.txt pins' <<<"$out"; then ok "a version bump in deps.txt fails"; else
    fail "a deps.txt URL mismatch (exit $status): $out"; fi
mirror_case '/^cutlass;/d'
if ((status != 0)) && grep -qF 'cmake/deps.txt has no cutlass' <<<"$out"; then ok "a dependency missing from deps.txt fails"; else
    fail "a missing deps.txt entry (exit $status): $out"; fi

echo "== the Python install directory"
if [[ -x /usr/bin/python ]]; then
    # shellcheck disable=SC2016 # the script expands its own arguments
    site_of() { bash -c '. "$1"; _ortf_py_site "$2"' _ "$pkgbuild" "$1" 2>&1; }
    sys_site=$(site_of /usr/bin/python) || sys_site="FAILED: $sys_site"
    if [[ $sys_site =~ ^/usr/lib/python3\.[0-9]+t?/site-packages$ ]]; then
        ok "/usr/bin/python installs to $sys_site"
    else
        fail "/usr/bin/python: $sys_site"
    fi
    if /usr/bin/python -m venv --without-pip "$tmp/venv" >/dev/null 2>&1; then
        venv_site=$(site_of "$tmp/venv/bin/python") || venv_site="FAILED: $venv_site"
        if [[ $venv_site == "$sys_site" ]]; then ok "a venv python gives the same directory"; else
            fail "a venv python gives $venv_site, /usr/bin/python $sys_site"; fi
    else
        echo "skip the venv case (python -m venv failed)"
    fi
else
    echo "skip the Python install directory (no /usr/bin/python)"
fi

echo "== metadata"
if command -v makepkg >/dev/null 2>&1; then
    a=$(cd "${pkgbuild%/*}" && env -u ORTF_EPS -u ORTF_PYTHON makepkg --printsrcinfo 2>&1) || a="FAILED: $a"
    b=$(cd "${pkgbuild%/*}" && ORTF_EPS=cpu ORTF_PYTHON=0 ORTF_CUDA_ARCHS=89 ORTF_TENSORRT_HOME=/x \
        MAKEFLAGS=-j3 makepkg --printsrcinfo 2>&1) || b="FAILED: $b"
    if [[ $a == "$b" && $a != FAILED* ]]; then
        ok "makepkg --printsrcinfo does not depend on the build inputs"
    else
        fail "makepkg --printsrcinfo differs with the build inputs set: $(diff <(echo "$a") <(echo "$b") | head -5)"
    fi
else
    echo "skip makepkg --printsrcinfo (makepkg not found)"
fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"

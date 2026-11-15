#!/usr/bin/env bash
# Tests for the build-input logic of onnxruntime-split/PKGBUILD: how
# prepare() resolves and checks ORTF_EPS, ORTF_CUDA_ARCHS,
# its presets, ORTF_NVCC_THREADS, ORTF_TENSORRT_HOME (and the TensorRT
# version its headers give), ORTF_PYTHON, ORTF_CCACHE,
# MAKEFLAGS, NVCC_CCBIN, the CUDA major (cuda.h against nvcc) and
# CFLAGS/CXXFLAGS (_ortf_resolve), how it lays out the dependency mirror (_ortf_mirror),
# which site-packages the Python package goes to under /usr/bin/python and
# a venv (_ortf_py_site), the package names the inputs give
# (_ortf_cuda_header_major, _ortf_cuda_suffix, _ortf_pkgnames,
# _ortf_check_names), the TensorRT header version and the installed openvino
# and python versions prepare() records and the bounds they give (_ortf_installed_version,
# _ortf_minor_bounds), the depends a library's DT_NEEDED entries give
# (_ortf_lib_depends, on the real lists of a 1.29.0 build), the package
# functions of every package run as makepkg runs them (set -eE) on a stub
# install, that of `makepkg --printsrcinfo` only the package names depend
# on the inputs, and that every dependency between the packages is met by
# the packages' names and provides, and by those of their -bin
# counterparts (the convention at the top of the PKGBUILD).
# The PKGBUILD is copied with its CUDA root pointed at a fixture (the
# package names read its cuda.h when the file is sourced) and sourced in a
# subshell per case, with its other vendor locations pointed at fixtures
# under mktemp and pacman, python, cmake, readelf and patchelf replaced by
# stubs (pacman's vercmp, read-only, is the real one):
# no CUDA, TensorRT or OpenVINO is needed, nothing is downloaded or built.
# Usage: scripts/tests/pkgbuild-settings-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
recipe_pkgbuild=$(cd "$here/../../onnxruntime-split" && pwd)/PKGBUILD

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

# The vendor fixtures: a CUDA 13.4 (cuda.h, an nvcc that prints its
# release), a CUDA 12.8 header, a CUDA with nvcc but no cuda.h, a cuDNN, an
# OpenVINO CMake package, and TensorRT trees (lib/ and lib64/, and ones
# without or with a bad NvInferVersion.h). Empty files otherwise: only
# their names count.
fx=$tmp/fx
mkdir -p "$fx/cuda/bin" "$fx/cuda/include" "$fx/cuda12/include" "$fx/cuda-nohdr/bin" \
    "$fx/cuda-nodefine/include" "$fx/cudnn/include" "$fx/cudnn-nohdr/include" "$fx/cudnn-nodefine/include" "$fx/cmake/openvino2026.4.1" \
    "$fx/trt/include" "$fx/trt/lib" "$fx/trt64/include" "$fx/trt64/lib64" "$fx/trt-nolib/include" \
    "$fx/trt-old/include" "$fx/trt-old/lib" "$fx/trt-nover/include" "$fx/trt-nover/lib" \
    "$fx/trt-badver/include" "$fx/trt-badver/lib" "$fx/trt-relver/include" "$fx/trt-relver/lib" \
    "$fx/empty" "$fx/bin" "$fx/pacbin" "$fx/pkgbin"
# cuda.h as CUDA's: the version define some way down, after comments and
# other defines (line 272 in CUDA 13.4).
cuda_h() { # cuda_h <CUDA_VERSION>
    local i
    printf '/* A stand-in for cuda.h: the version define some way down. */\n#ifndef __cuda_cuda_h__\n'
    for ((i = 0; i < 250; i++)); do printf '#define CU_FILLER_%d %d\n' "$i" "$i"; done
    printf '/**\n * CUDA API version number\n */\n#define CUDA_VERSION %s\n\n#define CUDA_VERSION_LATER 99000\n' "$1"
}
cuda_h 13040 >"$fx/cuda/include/cuda.h"
cuda_h 12080 >"$fx/cuda12/include/cuda.h"
cuda_h 13040 | grep -v '^#define CUDA_VERSION ' >"$fx/cuda-nodefine/include/cuda.h"
# nvcc --version as CUDA's, with the release FX_NVCC_RELEASE (13.4), or
# none for FX_NVCC_RELEASE=none.
cat >"$fx/cuda/bin/nvcc" <<'EOF2'
#!/bin/sh
[ "$1" = --version ] || { echo "stub nvcc: unexpected $*" >&2; exit 2; }
echo 'nvcc: NVIDIA (R) Cuda compiler driver'
[ "${FX_NVCC_RELEASE:-}" = none ] && exit 0
r=${FX_NVCC_RELEASE:-13.4}
echo "Cuda compilation tools, release $r, V$r.92"
EOF2
chmod +x "$fx/cuda/bin/nvcc"
cp "$fx/cuda/bin/nvcc" "$fx/cuda-nohdr/bin/nvcc"
: >"$fx/cudnn/include/cudnn.h"
: >"$fx/cudnn-nohdr/include/cudnn.h"
: >"$fx/cudnn-nodefine/include/cudnn.h"
# cudnn_version.h as cuDNN's, in its own words (no vendor text): the version
# defines among others.
cudnn_version_h() { # cudnn_version_h <CUDNN_MAJOR>
    printf '/* A stand-in for cudnn_version.h. */\n#ifndef CUDNN_VERSION_H_\n#define CUDNN_VERSION_H_\n'
    printf '#define CUDNN_MAJOR %s\n#define CUDNN_MINOR 27\n#define CUDNN_PATCHLEVEL 0\n' "$1"
    printf '#define CUDNN_MAX_SM_MAJOR_NUMBER 12\n#endif\n'
}
cudnn_version_h 9 >"$fx/cudnn/include/cudnn_version.h"
cudnn_version_h 9 | grep -v '^#define CUDNN_MAJOR ' >"$fx/cudnn-nodefine/include/cudnn_version.h"
: >"$fx/cmake/openvino2026.4.1/OpenVINOConfig.cmake"
: >"$fx/trt/include/NvInfer.h"
: >"$fx/trt/lib/libnvinfer.so.11"
: >"$fx/trt64/include/NvInfer.h"
: >"$fx/trt64/lib64/libnvinfer.so"
: >"$fx/trt-nolib/include/NvInfer.h"
# NvInferVersion.h in both styles: TensorRT 10 and 11 (NV_TENSORRT_MAJOR is
# TRT_MAJOR_ENTERPRISE, ...), older ones (numbers).
cat >"$fx/trt/include/NvInferVersion.h" <<'EOF2'
#define TRT_MAJOR_ENTERPRISE 11
#define TRT_MINOR_ENTERPRISE 4
#define TRT_PATCH_ENTERPRISE 0
#define TRT_BUILD_ENTERPRISE 106
#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< TensorRT major version.
#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE //!< TensorRT minor version.
#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE //!< TensorRT patch version.
#define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE //!< TensorRT build number.
#define NV_TENSORRT_LWS_MAJOR 0 //!< TensorRT LWS major version.
EOF2
cp "$fx/trt/include/NvInferVersion.h" "$fx/trt64/include/NvInferVersion.h"
printf '#define NV_TENSORRT_MAJOR 8\n#define NV_TENSORRT_MINOR 6\n#define NV_TENSORRT_PATCH 1\n#define NV_TENSORRT_BUILD 6\n' \
    >"$fx/trt-old/include/NvInferVersion.h"
printf '#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE\n#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE\n' \
    >"$fx/trt-badver/include/NvInferVersion.h"
printf '#define NV_TENSORRT_MAJOR 10\n#define NV_TENSORRT_MINOR 13\n' >"$fx/trt-relver/include/NvInferVersion.h"
for d in trt-old trt-nover trt-badver trt-relver; do
    : >"$fx/$d/include/NvInfer.h"
    : >"$fx/$d/lib/libnvinfer.so"
done
printf "export NVCC_CCBIN='/usr/bin/g++-15'\n" >"$fx/cuda.sh"
# Stubs found first on PATH by some cases. The python prints its version,
# FX_PYTHON_VERSION (default 3.14.8), for -I -c.
cat >"$fx/bin/python" <<'EOF2'
#!/bin/sh
[ "$1" = -I ] && [ "$2" = -c ] && printf '%s\n' "${FX_PYTHON_VERSION-3.14.8}"
exit 0
EOF2
printf '#!/bin/sh\nexit 0\n' >"$fx/bin/ccache"
chmod +x "$fx/bin/python" "$fx/bin/ccache"
# pacman, read-only queries only:
#   -T -- <dep>...: prints each dependency named in FX_MISSING (as not
#     satisfied) and exits 127 if there is one, else prints nothing;
#   -Qi -- <name>: the Name, Version and Provides lines of the installed
#     package of FX_INSTALLED that is or provides <name> (entries
#     name|version|provides, space-separated; provides comma-separated),
#     or fails as for a package not installed. The default: Arch's
#     openvino and cudnn, a local TensorRT package that provides tensorrt.
cat >"$fx/pacbin/pacman" <<'EOF2'
#!/bin/sh
case $1 in
-Qi)
    [ "$2" = -- ] && shift
    for e in ${FX_INSTALLED-openvino|2026.4.1-1| cudnn|9.27.0.2-1| nvidia-tensorrt-sdk-local|11.4.0.106-2|tensorrt=11.4.0.106}; do
        n=${e%%|*} r=${e#*|} v=${r%%|*} p=${r#*|}
        hit=0
        [ "$n" = "$2" ] && hit=1
        for q in $(printf '%s' "$p" | tr ',' ' '); do
            [ "$q" = "$2" ] || [ "${q%%=*}" = "$2" ] && hit=1
        done
        if [ "$hit" = 1 ]; then
            printf 'Name            : %s\nVersion         : %s\n' "$n" "$v"
            printf 'Description     : a stub\nProvides        : %s\nDepends On      : None\n' "$(printf '%s' "${p:-None}" | sed 's/,/  /g')"
            exit 0
        fi
    done
    echo "error: package '$2' was not found" >&2
    exit 1
    ;;
-T)
    shift
    [ "$1" = -- ] && shift
    rc=0
    for d in "$@"; do
        case " ${FX_MISSING:-} " in *" $d "*) printf '%s\n' "$d"; rc=127 ;; esac
    done
    exit "$rc"
    ;;
esac
echo "stub pacman: unexpected $*" >&2
exit 2
EOF2
chmod +x "$fx/pacbin/pacman"

# The PKGBUILD as tested: the recipe's, its CUDA root FX_CUDA_HOME (default
# the CUDA 13.4 fixture), in a directory of its own for makepkg.
mkdir -p "$tmp/recipe"
pkgbuild=$tmp/recipe/PKGBUILD
sed "s|^_ortf_cuda_home=/opt/cuda\$|_ortf_cuda_home=\${FX_CUDA_HOME:-$fx/cuda}|" "$recipe_pkgbuild" >"$pkgbuild"
if grep -qxF "_ortf_cuda_home=\${FX_CUDA_HOME:-$fx/cuda}" "$pkgbuild"; then
    ok "the tested PKGBUILD reads the CUDA fixture"
else
    fail "the PKGBUILD has no line _ortf_cuda_home=/opt/cuda to point at the fixture"
fi

# resolve <env assignments...>: runs _ortf_resolve with the fixtures and the
# given environment (every ORTF_* variable, MAKEFLAGS, NVCC_CCBIN, CFLAGS
# and CXXFLAGS unset otherwise). Sets out (its output, stdout and stderr), status and conf
# (the file it wrote, empty when none).
resolve() {
    local dir
    dir=$(mktemp -d "$tmp/case.XXXXXX")
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_NVCC_THREADS -u ORTF_TENSORRT_HOME \
        -u ORTF_PYTHON -u ORTF_CCACHE -u MAKEFLAGS -u NVCC_CCBIN -u CFLAGS -u CXXFLAGS \
        ORTF_PYTHON=0 ORTF_CCACHE=0 "$@" bash -c '
            set -euo pipefail
            # shellcheck disable=SC1090
            . "$1"
            # pacman -T and -Qi for tensorrt.
            PATH=$2/pacbin:$PATH
            _ortf_cuda_home=$2/cuda _ortf_cudnn_home=$2/cudnn _ortf_openvino_cmake_dir=$2/cmake
            _ortf_tensorrt_default=$2/trt _ortf_cuda_profile=$2/no-cuda.sh
            [[ -z ${FX_CUDA_NOHDR:-} ]] || _ortf_cuda_home=$2/cuda-nohdr
            [[ -z ${FX_CUDNN_HOME:-} ]] || _ortf_cudnn_home=$2/$FX_CUDNN_HOME
            [[ -z ${FX_PROFILE:-} ]] || _ortf_cuda_profile=$2/cuda.sh
            [[ -z ${FX_NO_TRT_DEFAULT:-} ]] || _ortf_tensorrt_default=$2/empty
            [[ -z ${FX_USR:-} ]] || _ortf_tensorrt_usr=$2/trt64
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
    "ORTF_TENSORRT_HOME=$fx/trt" "ORTF_TENSORRT_LIBDIR=$fx/trt/lib" \
    'ORTF_TENSORRT_VERSION=11.4.0.106' 'ORTF_CUDA_MAJOR=13' --
if grep -qF 'ORTF_EPS=cuda tensorrt openvino (default)' <<<"$out"; then
    ok "prepare() prints the providers and their source"
else
    fail "printed: $out"
fi
expect_ok "cpu alone checks no vendor install" 'ORTF_EPS=cpu' 'ORTF_TENSORRT_HOME=' 'ORTF_TENSORRT_VERSION=' \
    'ORTF_CUDA_MAJOR=' -- \
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

echo "== the CUDA major (cuda.h and nvcc)"
expect_ok "cuda records the toolkit's major" 'ORTF_CUDA_MAJOR=13' -- ORTF_EPS=cuda
if grep -qF "CUDA 13.4 ($fx/cuda: nvcc and include/cuda.h; packages named -cuda13)" <<<"$out"; then
    ok "prepare() prints the CUDA version and the names it gives"
else
    fail "printed: $out"
fi
expect_ok "no CUDA major without cuda" 'ORTF_CUDA_MAJOR=' -- ORTF_EPS=openvino
expect_fail "an nvcc of another major than cuda.h fails" 'is CUDA 12.9,' ORTF_EPS=cuda FX_NVCC_RELEASE=12.9
expect_ok "another minor of the same major passes" 'ORTF_CUDA_MAJOR=13' -- ORTF_EPS=cuda FX_NVCC_RELEASE=13.0
expect_fail "an nvcc that prints no release fails" "prints no 'release N.M,' line" ORTF_EPS=cuda \
    FX_NVCC_RELEASE=none
expect_fail "cuda without cuda.h fails" "no '#define CUDA_VERSION' in $fx/cuda-nohdr/include/cuda.h" \
    ORTF_EPS=cuda FX_CUDA_NOHDR=1

echo "== ORTF_TENSORRT_HOME"
expect_ok "an explicit home with lib64/" "ORTF_TENSORRT_HOME=$fx/trt64" \
    "ORTF_TENSORRT_LIBDIR=$fx/trt64/lib64" -- "ORTF_TENSORRT_HOME=$fx/trt64/"
expect_fail "a relative home fails" 'not an absolute path' ORTF_TENSORRT_HOME=opt/trt
expect_fail "a home without NvInfer.h fails" "$fx/empty/include/NvInfer.h not found" \
    "ORTF_TENSORRT_HOME=$fx/empty"
expect_fail "a home without libnvinfer fails" 'no libnvinfer.so' "ORTF_TENSORRT_HOME=$fx/trt-nolib"
if [[ -f /usr/include/NvInfer.h ]]; then
    expect_ok "without /opt/nvidia/tensorrt, /usr is used" 'ORTF_TENSORRT_HOME=/usr' \
        'ORTF_TENSORRT_HOME=/usr' -- FX_NO_TRT_DEFAULT=1
else
    expect_fail "without /opt/nvidia/tensorrt, /usr is used (and has no TensorRT here)" \
        '/usr/include/NvInfer.h not found' FX_NO_TRT_DEFAULT=1
fi

echo "== the TensorRT version (NvInferVersion.h) and the tensorrt package"
expect_ok "TensorRT 11: the version from the TRT_*_ENTERPRISE defines" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- \
    ORTF_EPS='cuda tensorrt'
if grep -qF 'ORTF_TENSORRT_VERSION=11.4.0.106 (' <<<"$out" && grep -qF 'so tensorrt>=11.4 tensorrt<11.5)' <<<"$out"; then
    ok "prepare() prints the TensorRT version and bounds"
else
    fail "printed: $out"; fi
expect_ok "an older TensorRT: the numeric NV_TENSORRT_* defines" 'ORTF_TENSORRT_VERSION=8.6.1.6' -- \
    ORTF_EPS='cuda tensorrt' "ORTF_TENSORRT_HOME=$fx/trt-old"
expect_ok "a header with major and minor only: the version is those" 'ORTF_TENSORRT_VERSION=10.13' -- \
    ORTF_EPS='cuda tensorrt' "ORTF_TENSORRT_HOME=$fx/trt-relver"
expect_ok "TensorRT in the AUR's root (/usr)" "ORTF_TENSORRT_HOME=$fx/trt64" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- \
    ORTF_EPS='cuda tensorrt' FX_NO_TRT_DEFAULT=1 FX_USR=1
expect_fail "a home without NvInferVersion.h fails" "$fx/trt-nover/include/NvInferVersion.h not found" \
    ORTF_EPS='cuda tensorrt' "ORTF_TENSORRT_HOME=$fx/trt-nover"
expect_fail "a header without a numeric major.minor fails" 'gives no numeric NV_TENSORRT_MAJOR and NV_TENSORRT_MINOR' \
    ORTF_EPS='cuda tensorrt' "ORTF_TENSORRT_HOME=$fx/trt-badver"
expect_ok "no TensorRT version without tensorrt" 'ORTF_TENSORRT_VERSION=' -- ORTF_EPS='cuda openvino'
expect_ok "no package provides tensorrt: a warning, the build goes on" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- \
    ORTF_EPS='cuda tensorrt' FX_MISSING=tensorrt
if grep -qF 'warning: tensorrt: no installed package is or provides tensorrt' <<<"$out" &&
    grep -qF 'update nvidia-tensorrt-sdk-local or install the AUR tensorrt' <<<"$out"; then
    ok "the warning says what to install"
else
    fail "printed: $out"; fi
expect_ok "a provided tensorrt: no warning" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- ORTF_EPS='cuda tensorrt'
if ! grep -qF 'warning: tensorrt' <<<"$out"; then ok "a provider of tensorrt at the header's version warns of nothing"; else
    fail "printed: $out"; fi
expect_ok "an installed tensorrt of another minor: a warning" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- \
    ORTF_EPS='cuda tensorrt' FX_INSTALLED='tensorrt|11.5.0.1-1|'
if grep -qF 'the installed tensorrt is 11.5.0.1-1' <<<"$out"; then ok "the version mismatch is warned of"; else
    fail "printed: $out"; fi
expect_ok "an installed tensorrt whose minor only starts the same: a warning" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- \
    ORTF_EPS='cuda tensorrt' FX_INSTALLED='tensorrt|11.40.0.1-1|'
if grep -qF 'the installed tensorrt is 11.40.0.1-1' <<<"$out"; then ok "11.40 is not taken for 11.4"; else
    fail "printed: $out"; fi
expect_ok "no ORTF_TENSORRT_PKG is recorded" 'ORTF_TENSORRT_VERSION=11.4.0.106' -- ORTF_EPS='cuda tensorrt'
if ! grep -q 'ORTF_TENSORRT_PKG' <<<"$conf"; then ok "ortf.conf has no ORTF_TENSORRT_PKG"; else fail "ortf.conf: $conf"; fi

echo "== the installed versions the bounds come from"
expect_ok "the TensorRT version is the header's, openvino's the installed one" \
    'ORTF_TENSORRT_VERSION=11.4.0.106' 'ORTF_OPENVINO_VERSION=2026.4.1-1' -- \
    ORTF_EPS='cuda tensorrt openvino'
if grep -qF 'so tensorrt>=11.4 tensorrt<11.5)' <<<"$out" &&
    grep -qF 'openvino 2026.4.1-1 (installed, so openvino>=2026.4 openvino<2026.5)' <<<"$out"; then
    ok "prepare() prints the TensorRT and openvino versions and bounds"
else
    fail "printed: $out"
fi
expect_fail "no openvino version fails" 'no version recorded for openvino' ORTF_EPS=openvino FX_INSTALLED=
expect_ok "openvino provided by another package: the provided version" 'ORTF_OPENVINO_VERSION=2026.5.0' -- \
    ORTF_EPS=openvino FX_INSTALLED='openvino-git|2026.5.0.r1-1|openvino=2026.5.0'
expect_ok "an epoch is kept" 'ORTF_OPENVINO_VERSION=1:2026.4.1-1' -- ORTF_EPS=openvino \
    FX_INSTALLED='openvino|1:2026.4.1-1|'
if grep -qF 'openvino>=1:2026.4 openvino<1:2026.5' <<<"$out"; then ok "the bounds keep the epoch"; else
    fail "printed: $out"; fi
expect_fail "a version without a numeric major.minor fails" "version 'git.r12-1' has no numeric major.minor" \
    ORTF_EPS=openvino FX_INSTALLED='openvino|git.r12-1|'
expect_ok "the Python version is the build python's" 'ORTF_PYTHON_VERSION=3.14.8' -- \
    ORTF_EPS=cpu ORTF_PYTHON=1 PATH="$fx/bin:$PATH"
if grep -qF 'Python 3.14.8, so python>=3.14 python<3.15)' <<<"$out"; then ok "prepare() prints the Python bounds"; else
    fail "printed: $out"; fi
expect_fail "a python that prints no version fails" 'no version recorded for' \
    ORTF_EPS=cpu ORTF_PYTHON=1 PATH="$fx/bin:$PATH" FX_PYTHON_VERSION=
expect_ok "no version without the engine" 'ORTF_TENSORRT_VERSION=' 'ORTF_OPENVINO_VERSION=' 'ORTF_PYTHON_VERSION=' -- \
    ORTF_EPS=cpu FX_NO_VENDOR=1 FX_INSTALLED=
echo "== the cuDNN major (cudnn_version.h; the CUDA provider dlopen()s cuDNN)"
expect_ok "the cuDNN major is the header's" 'ORTF_CUDNN_MAJOR=9' -- ORTF_EPS=cuda
if grep -qF 'cuDNN 9 (' <<<"$out" && grep -qF 'cudnn>=9 cudnn<10)' <<<"$out"; then ok "prepare() prints the cuDNN major and bounds"; else
    fail "printed: $out"; fi
expect_ok "no cuDNN major without cuda" 'ORTF_CUDNN_MAJOR=' -- ORTF_EPS=openvino
expect_ok "without cudnn_version.h: the installed cudnn package's major" 'ORTF_CUDNN_MAJOR=9' -- \
    ORTF_EPS=cuda FX_CUDNN_HOME=cudnn-nohdr
expect_ok "a header without CUDNN_MAJOR: the installed cudnn package's major" 'ORTF_CUDNN_MAJOR=10' -- \
    ORTF_EPS=cuda FX_CUDNN_HOME=cudnn-nodefine FX_INSTALLED='cudnn|10.1.0.5-1|'
expect_ok "the installed cudnn's epoch is dropped" 'ORTF_CUDNN_MAJOR=9' -- \
    ORTF_EPS=cuda FX_CUDNN_HOME=cudnn-nohdr FX_INSTALLED='cudnn|1:9.27.0.2-1|'
expect_fail "no cuDNN header version and no installed cudnn fails" 'no CUDNN_MAJOR in' \
    ORTF_EPS=cuda FX_CUDNN_HOME=cudnn-nohdr FX_INSTALLED=
expect_fail "an installed cudnn of no numeric version fails" 'no CUDNN_MAJOR in' \
    ORTF_EPS=cuda FX_CUDNN_HOME=cudnn-nohdr FX_INSTALLED='cudnn|git-1|'
# minor_bounds <name> <version>: _ortf_minor_bounds, lines joined by spaces.
# shellcheck disable=SC2016 # the script expands its own arguments
minor_bounds() { bash -c 'set -euo pipefail; . "$1"; _ortf_minor_bounds "$2" "$3"' _ "$pkgbuild" "$1" "$2" 2>&1 | tr '\n' ' '; }
for case in 'tensorrt|11.4.0.106|tensorrt>=11.4 tensorrt<11.5 ' \
    'openvino|2026.4.1-1|openvino>=2026.4 openvino<2026.5 ' 'python|3.14.8|python>=3.14 python<3.15 ' \
    'tensorrt|10.13.3.9-1|tensorrt>=10.13 tensorrt<10.14 ' 'python|3.9.1|python>=3.9 python<3.10 ' \
    'openvino|2026.08.0-1|openvino>=2026.08 openvino<2026.9 ' 'openvino|2026.09.1-1|openvino>=2026.09 openvino<2026.10 ' \
    'tensorrt|11.09.0.1|tensorrt>=11.09 tensorrt<11.10 '; do
    IFS='|' read -r name version wanted <<<"$case"
    got=$(minor_bounds "$name" "$version")
    if [[ $got == "$wanted" ]]; then ok "$name $version is bounded by $wanted"; else
        fail "$name $version: '$got', not '$wanted'"; fi
done
# pacman's own comparison: the installed version is inside its bounds, and
# the next minor is not.
if command -v vercmp >/dev/null 2>&1; then
    for case in '11.4.0.106 11.4 11.5' '11.4.0.106-1 11.4 11.5' '2026.4.1-1 2026.4 2026.5' '3.14.8-1 3.14 3.15' \
        '1:2026.4.1-1 1:2026.4 1:2026.5'; do
        read -r v lo hi <<<"$case"
        if (($(vercmp "$v" "$lo") >= 0 && $(vercmp "$v" "$hi") < 0)); then ok "vercmp: $lo <= $v < $hi"; else
            fail "vercmp: $v is not in [$lo, $hi)"; fi
    done
    for case in '11.5.0.1 11.4 11.5' '2026.5.0-1 2026.4 2026.5' '3.15.0-1 3.14 3.15' '11.3.9 11.4 11.5'; do
        read -r v lo hi <<<"$case"
        if (($(vercmp "$v" "$lo") >= 0 && $(vercmp "$v" "$hi") < 0)); then fail "vercmp: $v is in [$lo, $hi)"; else
            ok "vercmp: $v is outside [$lo, $hi)"; fi
    done
else
    echo "skip the vercmp cases (no vercmp: pacman is not installed)"
fi

echo "== ORTF_CUDA_ARCHS and ORTF_NVCC_THREADS"
expect_ok "the default architecture is 120" 'ORTF_CUDA_ARCHS=120' 'ORTF_CUDA_PRESET=' 'ORTF_NVCC_THREADS=1' -- \
    ORTF_EPS=cuda
expect_ok "the rtx preset expands to its list" 'ORTF_CUDA_ARCHS=75;86;89;120' 'ORTF_CUDA_PRESET=rtx' -- \
    ORTF_EPS=cuda ORTF_CUDA_ARCHS=rtx
if grep -qF 'ORTF_CUDA_ARCHS=rtx: 75;86;89;120 (environment)' <<<"$out"; then
    ok "prepare() prints the preset and its expansion"
else
    fail "printed: $out"
fi
expect_ok "the all preset: every x86 GPU of CUDA 13" 'ORTF_CUDA_ARCHS=75;80;86;89;90;100;103;120' \
    'ORTF_CUDA_PRESET=all' -- ORTF_EPS=cuda ORTF_CUDA_ARCHS=' all '
for v in 'rtx 120' '89,all' 'rtx all' 'rtx rtx'; do
    expect_fail "a preset with other words fails ($v)" 'stands alone' ORTF_EPS=cuda ORTF_CUDA_ARCHS="$v"
done
expect_fail "an unknown word fails and names the presets" "'gtx' is not a CUDA architecture such as 120, sm_120 or 120-virtual, nor a preset (rtx, all)" \
    ORTF_EPS=cuda ORTF_CUDA_ARCHS=gtx
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

echo "== CFLAGS and CXXFLAGS"
expect_ok "Arch's baseline passes" 'ORTF_EPS=cpu' -- ORTF_EPS=cpu \
    CFLAGS='-march=x86-64 -mtune=generic -O2 -pipe' CXXFLAGS='-march=x86-64 -mtune=generic -O2 -pipe'
if grep -qF 'CPU baseline: -march=x86-64 (CFLAGS, CXXFLAGS)' <<<"$out"; then ok "prepare() prints the -march"; else
    fail "printed: $out"; fi
expect_ok "an x86-64-vN level is the builder's choice" 'ORTF_EPS=cpu' -- ORTF_EPS=cpu \
    CFLAGS='-march=x86-64-v3 -O2' CXXFLAGS='-march=x86-64-v3 -O2'
expect_fail "-march=native in CFLAGS fails" 'CFLAGS: -march=native builds for this machine' ORTF_EPS=cpu \
    CFLAGS='-march=native -O2' CXXFLAGS='-march=x86-64'
expect_fail "-mtune=native in CXXFLAGS fails" 'CXXFLAGS: -mtune=native builds for this machine' ORTF_EPS=cpu \
    CFLAGS='-march=x86-64' CXXFLAGS='-march=x86-64 -mtune=native'
expect_fail "the message names makepkg.conf" '/etc/makepkg.conf' ORTF_EPS=cpu CFLAGS='-mtune=native'

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

echo "== the package names"
# shellcheck disable=SC2016 # the script expands its own arguments
suffix_of() { bash -c 'set -euo pipefail; . "$1"; _ortf_cuda_suffix "$2"' _ "$pkgbuild" "$1" 2>&1; }
for case in '120=sm120' 'sm_120=sm120' '89 120=sm89-sm120' '120,89=sm89-sm120' '120;89=sm89-sm120' \
    '120 120-real=sm120' '89 120 120-virtual=sm89-sm120-compute120' '120-virtual=compute120' \
    '89-real sm_100,120=sm89-sm100-sm120' '89-virtual 120=sm120-compute89' \
    'rtx=rtx' 'all=all' ' rtx =rtx' 'rtx,=rtx' 'rtx 120=invalid' '120 all=invalid' \
    'rtx all=invalid' 'gtx=invalid' 'RTX=invalid' 'native=invalid' '120a=invalid' '89-virtual 120-virtual=invalid' ';=invalid' '=invalid'; do
    archs=${case%=*} wanted=${case##*=}
    got=$(suffix_of "$archs") || got="FAILED: $got"
    if [[ $got == "$wanted" ]]; then ok "ORTF_CUDA_ARCHS='$archs' gives the suffix $wanted"; else
        fail "ORTF_CUDA_ARCHS='$archs' gives '$got', not $wanted"; fi
done
# shellcheck disable=SC2016 # the script expands its own arguments
major_of() { bash -c 'set -euo pipefail; . "$1"; _ortf_cuda_header_major "$2"' _ "$pkgbuild" "$1" 2>&1; }
for case in "cuda=13" "cuda12=12"; do
    dir=${case%=*} wanted=${case##*=}
    got=$(major_of "$fx/$dir") || got="FAILED: $got"
    if [[ $got == "$wanted" ]]; then ok "cuda.h of $dir gives the CUDA major $wanted"; else
        fail "cuda.h of $dir gives '$got', not $wanted"; fi
done
for dir in empty cuda-nodefine; do
    if got=$(major_of "$fx/$dir"); then fail "$dir: a major without CUDA_VERSION: $got"; else
        ok "no CUDA major from $dir (no cuda.h, or no CUDA_VERSION in it)"; fi
done
# names_for <env...>: the pkgname the PKGBUILD sets with that environment.
names_for() {
    # shellcheck disable=SC2016 # the script expands its own arguments
    env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_PYTHON "$@" \
        bash -c 'set -euo pipefail; . "$1"; echo "${pkgname[*]}"' _ "$pkgbuild" 2>&1
}
expect_names() { # expect_names <names> <env...>
    local wanted=$1 got
    shift
    got=$(names_for "$@") || got="FAILED: $got"
    if [[ $got == "$wanted" ]]; then ok "names with [$*]: $wanted"; else fail "names with [$*]: '$got', not '$wanted'"; fi
}
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-sm120 onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-openvino python-onnxruntime-core'
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-sm120 onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-openvino python-onnxruntime-core' \
    ORTF_EPS= ORTF_CUDA_ARCHS= ORTF_PYTHON=
expect_names 'onnxruntime-core' ORTF_EPS=cpu ORTF_PYTHON=0
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-rtx onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-openvino python-onnxruntime-core' \
    ORTF_CUDA_ARCHS=rtx
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-all onnxruntime-provider-tensorrt-cuda13 python-onnxruntime-core' \
    ORTF_EPS='cuda tensorrt' ORTF_CUDA_ARCHS=all
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-invalid onnxruntime-provider-tensorrt-cuda13 onnxruntime-provider-openvino python-onnxruntime-core' \
    ORTF_CUDA_ARCHS='rtx 120'
expect_names 'onnxruntime-core python-onnxruntime-core' ORTF_EPS=cpu
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-sm89-sm120 onnxruntime-provider-openvino' \
    ORTF_EPS=openvino,cuda ORTF_CUDA_ARCHS='120 89' ORTF_PYTHON=0
expect_names 'onnxruntime-core onnxruntime-provider-cuda13-sm120-compute120 onnxruntime-provider-tensorrt-cuda13 python-onnxruntime-core' \
    ORTF_EPS='cuda tensorrt' ORTF_CUDA_ARCHS='120 120-virtual'
expect_names 'onnxruntime-core onnxruntime-provider-openvino python-onnxruntime-core' ORTF_EPS=openvino \
    ORTF_CUDA_ARCHS=native
# The CUDA major is the toolkit's, never assumed.
expect_names 'onnxruntime-core onnxruntime-provider-cuda12-sm89 onnxruntime-provider-openvino' \
    FX_CUDA_HOME="$fx/cuda12" ORTF_EPS='cuda openvino' ORTF_CUDA_ARCHS=89 ORTF_PYTHON=0
expect_names 'onnxruntime-core onnxruntime-provider-cudax-sm120 onnxruntime-provider-tensorrt-cudax onnxruntime-provider-openvino python-onnxruntime-core' \
    FX_CUDA_HOME="$fx/empty"
expect_names 'onnxruntime-core onnxruntime-provider-openvino' FX_CUDA_HOME="$fx/empty" ORTF_EPS=openvino ORTF_PYTHON=0

# check_names <ortf.conf lines> <env...>: _ortf_check_names with the
# recorded values and the environment's names. Sets status, out.
check_names() {
    local dir conf=$1
    shift
    dir=$(mktemp -d "$tmp/names.XXXXXX")
    printf '%s\n' "$conf" >"$dir/ortf.conf"
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_PYTHON "$@" bash -c '
        set -euo pipefail
        # shellcheck disable=SC1090
        . "$1"
        srcdir=$2
        # As in a package function: makepkg sets pkgname[0] to the package.
        pkgname=onnxruntime-core
        _ortf_check_names
    ' _ "$pkgbuild" "$dir" 2>&1) || status=$?
}
check_names $'ORTF_EPS=cuda tensorrt openvino\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=13\nORTF_PYTHON=1'
if ((status == 0)); then ok "recorded inputs that give makepkg's names pass"; else fail "check_names: $out"; fi
check_names $'ORTF_EPS=cuda tensorrt openvino\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=12\nORTF_PYTHON=1'
if ((status != 0)) && grep -qF "the build used CUDA 12, $fx/cuda/include/cuda.h is now CUDA 13" <<<"$out"; then
    ok "a build of another CUDA major than cuda.h now has fails and says so"
else
    fail "check_names (exit $status): $out"
fi
check_names $'ORTF_EPS=cuda\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_PYTHON=0' ORTF_EPS=cuda ORTF_PYTHON=0
if ((status != 0)) && grep -qF 'ortf.conf: no ORTF_CUDA_MAJOR' <<<"$out"; then
    ok "an ortf.conf without ORTF_CUDA_MAJOR fails"
else
    fail "check_names (exit $status): $out"
fi
check_names $'ORTF_EPS=openvino\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=\nORTF_PYTHON=0' \
    ORTF_EPS=openvino ORTF_PYTHON=0 FX_CUDA_HOME="$fx/empty"
if ((status == 0)); then ok "without cuda, no CUDA major is needed"; else fail "check_names: $out"; fi
check_names $'ORTF_EPS=cuda\nORTF_CUDA_ARCHS=120;89\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=13\nORTF_PYTHON=0' ORTF_EPS=cuda ORTF_CUDA_ARCHS='89,120' ORTF_PYTHON=0
if ((status == 0)); then ok "the recorded canonical form gives the same names"; else fail "check_names: $out"; fi
check_names $'ORTF_EPS=cuda\nORTF_CUDA_ARCHS=120;89\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=13\nORTF_PYTHON=0'
if ((status != 0)) && grep -qF "ORTF_EPS='cuda' ORTF_CUDA_ARCHS='120;89' ORTF_PYTHON=0" <<<"$out"; then
    ok "makepkg -e with another environment fails and says what to pass"
else
    fail "check_names (exit $status): $out"
fi
# A preset: its name, not its list, gives the CUDA package's name.
rtx_conf=$'ORTF_EPS=cuda tensorrt\nORTF_CUDA_ARCHS=75;86;89;120\nORTF_CUDA_PRESET=rtx\nORTF_CUDA_MAJOR=13\nORTF_PYTHON=0'
check_names "$rtx_conf" ORTF_EPS='cuda tensorrt' ORTF_CUDA_ARCHS=rtx ORTF_PYTHON=0
if ((status == 0)); then ok "a recorded preset gives the preset's names"; else fail "check_names: $out"; fi
check_names "$rtx_conf" ORTF_EPS='cuda tensorrt' ORTF_CUDA_ARCHS='75 86 89 120' ORTF_PYTHON=0
if ((status != 0)) && grep -qF "ORTF_CUDA_ARCHS='rtx'" <<<"$out"; then
    ok "the preset's list instead of its name fails and says to pass the preset"
else
    fail "check_names (exit $status): $out"
fi
check_names $'ORTF_EPS=cuda\nORTF_CUDA_ARCHS=120\nORTF_PYTHON=0' ORTF_EPS=cuda ORTF_PYTHON=0
if ((status != 0)) && grep -qF 'ortf.conf: no ORTF_CUDA_PRESET' <<<"$out"; then
    ok "an ortf.conf without ORTF_CUDA_PRESET fails"
else
    fail "check_names (exit $status): $out"
fi
check_names $'ORTF_EPS=cpu'
if ((status != 0)) && grep -qF 'ortf.conf: no ORTF_CUDA_ARCHS' <<<"$out"; then ok "an incomplete ortf.conf fails"; else
    fail "check_names (exit $status): $out"; fi

echo "== the depends a library's DT_NEEDED entries give"
# The DT_NEEDED lists (readelf -d) of onnxruntime-full 1.29.0-1, the
# unsplit build of this recipe (CUDA 13.4, cuDNN 9, TensorRT 11.4, OpenVINO
# 2026.4), and of its python-onnxruntime-full's module, as its package
# files hold them. The CUDA provider's has no libcudnn: it dlopen()s cuDNN
# (as libcuda, libnvrtc and libcufft), which a NEEDED check cannot see, so
# the CUDA package's cudnn bound is the one prepare() recorded.
needed_core='libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2'
needed_shared=''
needed_cuda='libcublasLt.so.13 libcublas.so.13 libcudart.so.13 libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2'
needed_tensorrt='libcublas.so.13 libnvinfer.so.11 libnvonnxparser.so.11 libcudart.so.13 libcudnn.so.9 libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2'
needed_openvino='libopenvino.so.2641 libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2'
needed_python='libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2'
# lib_depends <package> <depends> <trt> <CUDA major> <TensorRT version>
# <NEEDED...>: runs _ortf_lib_depends for a library of <package> with
# depends=(<depends>, space-separated) on those NEEDED entries, with the
# cuDNN major lib_cudnn (a global, empty when unset). Sets
# status, out (stdout and stderr).
lib_depends() {
    local self=$1 deps=$2 trt=$3 major=$4 trt_version=$5
    shift 5
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(bash -c '
        set -euo pipefail
        # shellcheck disable=SC1090
        . "$1"
        read -ra depends <<<"$3"
        self=$2 trt=$4 major=$5 trt_version=$6 cudnn=$7
        shift 7
        printf "%s\n" "$@" | _ortf_lib_depends "$self" "$self: lib" "$trt" "$major" "$trt_version" "$cudnn"
    ' _ "$pkgbuild" "$self" "$deps" "$trt" "$major" "$trt_version" "${lib_cudnn:-}" "$@" 2>&1) || status=$?
}
expect_libs() { # expect_libs <description> <want stdout, lines joined by spaces> <lib_depends arguments...>
    local desc=$1 wanted=$2
    shift 2
    lib_depends "$@"
    if ((status == 0)) && [[ ${out//$'\n'/ } == "$wanted" ]]; then ok "$desc"; else
        fail "$desc (exit $status): $out"; fi
}
expect_libs_fail() { # expect_libs_fail <description> <message fragment> <lib_depends arguments...>
    local desc=$1 msg=$2
    shift 2
    lib_depends "$@"
    if ((status != 0)) && grep -qF -- "$msg" <<<"$out"; then ok "$desc"; else fail "$desc (exit $status): $out"; fi
}
# The static depends of each package, as the PKGBUILD writes them, with
# what package() adds before the check (the TensorRT package).
core_deps='glibc libgcc libstdc++'
cuda_deps='onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ cuda cudnn nvidia-utils'
trt_deps='onnxruntime-core=1.29.0-3 onnxruntime-provider-cuda13=1.29.0 glibc libgcc libstdc++ cuda cudnn nvidia-utils tensorrt'
ov_deps='onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ openvino'
py_deps='onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ python python-numpy python-flatbuffers python-packaging python-protobuf'
cu_pkg=onnxruntime-provider-cuda13-sm120 trt_pkg=onnxruntime-provider-tensorrt-cuda13 ov_pkg=onnxruntime-provider-openvino
# shellcheck disable=SC2086 # the NEEDED lists are words
{
    expect_libs "the real CUDA provider: cuda bounds from libcudart" 'cuda>=13 cuda<14' "$cu_pkg" "$cuda_deps" '' 13 '' \
        $needed_cuda
    expect_libs "the real TensorRT provider: cuda and cudnn bounds (TensorRT's come from its version)" \
        'cuda>=13 cuda<14 cudnn>=9 cudnn<10' "$trt_pkg" "$trt_deps" tensorrt 13 11.4.0.106 $needed_tensorrt
    expect_libs "the real OpenVINO provider: no soname bounds" '' "$ov_pkg" "$ov_deps" '' '' '' $needed_openvino
    expect_libs "the real libonnxruntime.so: system libraries only" '' onnxruntime-core "$core_deps" '' '' '' $needed_core
    expect_libs "the real libonnxruntime_providers_shared.so: nothing" '' onnxruntime-core "$core_deps" '' '' '' $needed_shared
    expect_libs "the real Python module: system libraries only" '' python-onnxruntime-core "$py_deps" '' '' '' $needed_python
}
# Every depend a library's package is mapped to is needed: without it, the
# check fails (each package, each system or vendor package it links).
# shellcheck disable=SC2086 # the NEEDED lists and depends are words
for case in "onnxruntime-core|$core_deps|glibc libgcc libstdc++|$needed_core" \
    "$cu_pkg|$cuda_deps|glibc libgcc libstdc++ cuda|$needed_cuda" \
    "$trt_pkg|$trt_deps|glibc libgcc libstdc++ cuda cudnn tensorrt|$needed_tensorrt" \
    "$ov_pkg|$ov_deps|glibc libgcc libstdc++ openvino|$needed_openvino" \
    "python-onnxruntime-core|$py_deps|glibc libgcc libstdc++|$needed_python"; do
    IFS='|' read -r self deps drop needed <<<"$case"
    for d in $drop; do
        less=" $deps "
        less=${less// $d / }
        expect_libs_fail "$self without $d in depends fails" "belongs to $d, which is not in the depends of $self" \
            "$self" "$less" tensorrt 13 11.4.0.106 $needed
    done
done
# A hypothetical libcudnn NEEDED (the real CUDA provider has none; the real
# TensorRT provider does), to test the soname's own bound and its major.
expect_libs "a library with libcuda, libcudnn and libcufft in its NEEDED (hypothetical): all bounds" \
    'cuda>=13 cuda<14 cudnn>=9 cudnn<10' \
    "$cu_pkg" "$cuda_deps" '' 13 '' libcuda.so.1 libcudart.so.13 libcudnn.so.9 libcufft.so.12 libcurand.so.10 libc.so.6
lib_cudnn=9
# shellcheck disable=SC2086 # the NEEDED list is words
expect_libs "the real TensorRT provider with the recorded cuDNN major: the same bounds" \
    'cuda>=13 cuda<14 cudnn>=9 cudnn<10' "$trt_pkg" "$trt_deps" tensorrt 13 11.4.0.106 $needed_tensorrt
expect_libs_fail "a libcudnn of another major than the recorded cuDNN fails" \
    'libcudnn.so.10 is cuDNN 10, the build used cuDNN 9' "$trt_pkg" "$trt_deps" \
    tensorrt 13 11.4.0.106 libcudnn.so.10
lib_cudnn=
expect_libs "libtbb is onetbb's, as a provider's own depend" '' "$ov_pkg" "$ov_deps onetbb" '' '' '' libtbb.so.12 libopenvino.so.2641
expect_libs_fail "libtbb without onetbb in depends fails" 'libtbb.so.12 belongs to onetbb, which is not in the depends' \
    "$ov_pkg" "$ov_deps" '' '' '' libtbb.so.12
expect_libs "a versioned depends entry counts as the package" 'cuda>=13 cuda<14' pkg 'cuda>=13.0' '' '' '' libcudart.so.13
expect_libs "the other glibc libraries and ONNX Runtime's own pass in onnxruntime-core" '' onnxruntime-core \
    "$core_deps" '' '' '' libdl.so.2 libpthread.so.0 librt.so.1 libonnxruntime.so.1 libonnxruntime_providers_shared.so
expect_libs "a provider linking libonnxruntime: onnxruntime-core" '' "$ov_pkg" "$ov_deps" '' '' '' \
    libonnxruntime.so.1 libonnxruntime_providers_shared.so libopenvino.so.2641
expect_libs "the Python module linking libpython and libonnxruntime" '' python-onnxruntime-core "$py_deps" '' '' '' \
    libpython3.14.so.1.0 libonnxruntime.so.1 libc.so.6
expect_libs_fail "libpython without python in depends fails" 'libpython3.14.so.1.0 belongs to python' \
    "$cu_pkg" "$cuda_deps" '' 13 '' libpython3.14.so.1.0
expect_libs_fail "a libz in libonnxruntime.so's NEEDED (not in the real one) fails: nothing maps it" \
    'libz.so.1 is a library no package is known for' onnxruntime-core "$core_deps" '' '' '' libz.so.1
expect_libs_fail "a provider linking libonnxruntime without onnxruntime-core fails" 'belongs to onnxruntime-core' \
    "$ov_pkg" 'glibc openvino' '' '' '' libonnxruntime.so.1
expect_libs_fail "the CUDA provider needing libnvinfer fails" \
    'libnvinfer.so.11 belongs to a TensorRT package, which is not in the depends' "$cu_pkg" "$cuda_deps" '' 13 '' \
    libcudart.so.13 libnvinfer.so.11
expect_libs_fail "the OpenVINO provider needing libcudart fails" 'libcudart.so.13 belongs to cuda' \
    "$ov_pkg" "$ov_deps" '' '' '' libopenvino.so.2641 libcudart.so.13
expect_libs_fail "libonnxruntime.so needing a vendor library fails" 'libcudart.so.13 belongs to cuda, which is not in the depends of onnxruntime-core' \
    onnxruntime-core "$core_deps" '' '' '' libstdc++.so.6 libcudart.so.13
expect_libs_fail "the Python module needing a vendor library fails" 'libopenvino.so.2641 belongs to openvino' \
    python-onnxruntime-core "$py_deps" '' '' '' libopenvino.so.2641
expect_libs_fail "TensorRT of another package than the depends fails" 'belongs to nvidia-tensorrt-sdk' "$trt_pkg" "$trt_deps" \
    nvidia-tensorrt-sdk 13 11.4.0.106 libnvinfer.so.11
expect_libs_fail "a libnvinfer of another major than the installed TensorRT fails" \
    'libnvinfer.so.10 is TensorRT 10, the installed tensorrt is TensorRT 11' "$trt_pkg" "$trt_deps" \
    tensorrt 13 11.4.0.106 libnvinfer.so.10
for lib in libcutensor.so.2 libcudss.so.0 libnvjpeg.so.13 libnvshmem_host.so.3 libcusparse.so.12 \
    libOpenCL.so.1 libze_loader.so.1 libsomething.so libonnxruntime_providers_cuda.so; do
    expect_libs_fail "an unknown library fails ($lib)" "$lib is a library no package is known for" \
        "$cu_pkg" "$cuda_deps" '' 13 '' libcudart.so.13 "$lib"
done
expect_libs_fail "a libcudart of another CUDA major than the names fails" \
    'libcudart.so.12 is CUDA 12, the package names say CUDA 13' "$cu_pkg" "$cuda_deps" '' 13 '' libcudart.so.12
expect_libs "without a CUDA major, libcudart's is not checked" 'cuda>=12 cuda<13' "$cu_pkg" "$cuda_deps" '' '' '' \
    libcudart.so.12

echo "== the package functions"
# Stubs for a package function (makepkg runs them after build()):
#   cmake --install: installs a stub build under DESTDIR: libonnxruntime.so
#     with its links, the provider libraries of FX_STAGE (default: shared
#     cuda tensorrt openvino), headers, pkg-config and CMake files, and two
#     files of the bundled dependencies;
#   readelf -d -- <file>: the NEEDED entries in FX_NEEDED/<file's name>
#     (one per line; no fixture is an error) and a RUNPATH from
#     FX_RPATH/<name>, in readelf's format;
#   patchelf --add-rpath <dir> <file>: appends <dir> to FX_RPATH/<name>;
#   the build python ($fx/pystub/python, also the system python in these
#     cases): -I -c prints the site-packages of Python 3.14; -m installer
#     --destdir=<dir> ... installs a stub wheel there: the module, the
#     wheel's copies of the libraries of FX_WHEEL_LIBS (default: core shared
#     cuda tensorrt openvino) and a script.
mkdir -p "$fx/pystub"
cat >"$fx/pystub/python" <<'EOF2'
#!/bin/sh
set -e
site=/usr/lib/python3.14/site-packages
if [ "$1" = -I ] && [ "$2" = -c ]; then echo "$site"; exit 0; fi
[ "$1" = -m ] && [ "$2" = installer ] || { echo "stub python: unexpected $*" >&2; exit 2; }
d=${3#--destdir=}
c=$d$site/onnxruntime/capi
mkdir -p "$c" "$d/usr/bin"
echo pybind >"$c/onnxruntime_pybind11_state.so"
: >"$d$site/onnxruntime/__init__.py"
for l in ${FX_WHEEL_LIBS-core shared cuda tensorrt openvino}; do
    case $l in
    core) echo copy >"$c/libonnxruntime.so.1.29.0" ;;
    *) echo copy >"$c/libonnxruntime_providers_$l.so" ;;
    esac
done
printf '#!/build/venv/bin/python3\nprint(1)\n' >"$d/usr/bin/onnxruntime_test"
EOF2
chmod +x "$fx/pystub/python"
cat >"$fx/pkgbin/cmake" <<'EOF2'
#!/bin/sh
set -e
[ "$1" = --install ] || { echo "stub cmake: unexpected $*" >&2; exit 2; }
d=$DESTDIR/usr
mkdir -p "$d/lib/pkgconfig" "$d/lib/cmake/onnxruntime" "$d/include/onnxruntime" "$d/include/absl"
echo core >"$d/lib/libonnxruntime.so.1.29.0"
ln -s libonnxruntime.so.1.29.0 "$d/lib/libonnxruntime.so.1"
ln -s libonnxruntime.so.1 "$d/lib/libonnxruntime.so"
for ep in ${FX_STAGE-shared cuda tensorrt openvino}; do echo "$ep" >"$d/lib/libonnxruntime_providers_$ep.so"; done
: >"$d/include/onnxruntime/onnxruntime_c_api.h"
: >"$d/include/onnxruntime/onnxruntime_cxx_api.h"
: >"$d/lib/pkgconfig/libonnxruntime.pc"
: >"$d/lib/cmake/onnxruntime/onnxruntimeConfig.cmake"
: >"$d/include/absl/base.h"
: >"$d/lib/libprotobuf.a"
EOF2
cat >"$fx/pkgbin/readelf" <<'EOF2'
#!/bin/sh
for f; do :; done
n=${f##*/}
[ -f "$FX_NEEDED/$n" ] || { echo "stub readelf: no NEEDED fixture for $n" >&2; exit 1; }
while read -r lib; do
    [ -n "$lib" ] && printf ' 0x0000000000000001 (NEEDED)             Shared library: [%s]\n' "$lib"
done <"$FX_NEEDED/$n"
if [ -f "$FX_RPATH/$n" ]; then
    printf ' 0x000000000000001d (RUNPATH)            Library runpath: [%s]\n' "$(cat "$FX_RPATH/$n")"
fi
exit 0
EOF2
cat >"$fx/pkgbin/patchelf" <<'EOF2'
#!/bin/sh
[ "$1" = --set-rpath ] || { echo "stub patchelf: unexpected $*" >&2; exit 2; }
n=${3##*/}
printf '%s' "$2" >"$FX_RPATH/$n"
EOF2
chmod +x "$fx/pkgbin/cmake" "$fx/pkgbin/readelf" "$fx/pkgbin/patchelf"

# package_run <ortf.conf lines> <library=NEEDED list...> -- <env...>: runs
# the package function of every package the environment names (ORTF_PYTHON
# as the ortf.conf lines record it unless given, else 0; the Python package
# installs the stub wheel), in makepkg's order, each
# in a subshell with set -eE and an ERR trap as makepkg does, pkgname set
# to the package and pkgdir to its own directory. The NEEDED lists default
# to the real ones above. Sets status (of the first package that failed,
# 0 if none), pkgroot (the packages' parent directory), and out: per
# package, its depends, provides, conflicts and replaces and its files, or
# its output when it failed.
package_run() {
    local dir conf=$1 kv python=0
    shift
    [[ $'\n'$conf$'\n' != *$'\nORTF_PYTHON=1\n'* ]] || python=1
    dir=$(mktemp -d "$tmp/pkgfn.XXXXXX")
    mkdir -p "$dir/src/onnxruntime" "$dir/src/build" "$dir/src/dist" "$dir/pkg" "$dir/needed" "$dir/rpath"
    : >"$dir/src/dist/onnxruntime-1.29.0-cp314-cp314-linux_x86_64.whl"
    echo 'MIT License' >"$dir/src/onnxruntime/LICENSE"
    echo 'notices' >"$dir/src/onnxruntime/ThirdPartyNotices.txt"
    printf '%s\n' "$conf" >"$dir/src/ortf.conf"
    tr ' ' '\n' <<<"$needed_core" >"$dir/needed/libonnxruntime.so.1.29.0"
    : >"$dir/needed/libonnxruntime_providers_shared.so"
    tr ' ' '\n' <<<"$needed_cuda" >"$dir/needed/libonnxruntime_providers_cuda.so"
    tr ' ' '\n' <<<"$needed_tensorrt" >"$dir/needed/libonnxruntime_providers_tensorrt.so"
    tr ' ' '\n' <<<"$needed_openvino" >"$dir/needed/libonnxruntime_providers_openvino.so"
    tr ' ' '\n' <<<"$needed_python" >"$dir/needed/onnxruntime_pybind11_state.so"
    while (($#)) && [[ $1 != -- ]]; do
        kv=$1
        tr ' ' '\n' <<<"${kv#*=}" >"$dir/needed/${kv%%=*}"
        shift
    done
    (($# == 0)) || shift
    pkgroot=$dir/pkg
    status=0
    # shellcheck disable=SC2016 # the script expands its own arguments
    out=$(env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_PYTHON ORTF_PYTHON="$python" "$@" \
        PATH="$fx/pkgbin:$PATH" FX_NEEDED="$dir/needed" FX_RPATH="$dir/rpath" bash -c '
        # shellcheck disable=SC1090
        . "$1"
        srcdir=$2/src
        _ortf_system_python=$3
        first=0
        for p in "${pkgname[@]}"; do
            (
                set -eE
                trap "echo \"ERR: \$BASH_COMMAND\" >&2; exit 1" ERR
                pkgname=$p
                pkgdir=$2/pkg/$p
                mkdir -p "$pkgdir"
                "package_$p" >"$2/$p.log" 2>&1
                echo "$p depends: ${depends[*]}"
                echo "$p provides: ${provides[*]}"
                echo "$p conflicts: ${conflicts[*]}"
                echo "$p replaces: ${replaces[*]}"
                echo "$p files: $(cd "$pkgdir" && find . ! -type d | LC_ALL=C sort | tr "\n" " ")"
            )
            rc=$?
            if ((rc != 0)); then
                echo "$p FAILED (exit $rc):"
                cat "$2/$p.log"
                ((first != 0)) || first=$rc
            fi
        done
        exit "$first"
    ' _ "$pkgbuild" "$dir" "$fx/pystub/python" 2>&1) || status=$?
}
# has_line <line>: out has that line, exactly.
has_line() { grep -qxF -- "$1" <<<"$out"; }
conf_all=$'ORTF_EPS=cuda tensorrt openvino\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=13\nORTF_CUDNN_MAJOR=9\nORTF_PYTHON=1\nORTF_TENSORRT_HOME=/opt/nvidia/tensorrt\nORTF_TENSORRT_LIBDIR=/opt/nvidia/tensorrt/lib\nORTF_TENSORRT_PKG=a-stale-package\nORTF_TENSORRT_VERSION=11.4.0.106\nORTF_OPENVINO_VERSION=2026.4.1-1'
conf_all+=$'\nORTF_PYTHON_EXE='"$fx/pystub/python"$'\nORTF_PYTHON_VERSION=3.14.8'
package_run "$conf_all" --
if ((status == 0)); then ok "every package function of the default build runs under set -eE"; else
    fail "the package functions (exit $status): $out"; fi
lic() { printf './usr/share/licenses/%s/LICENSE ./usr/share/licenses/%s/ThirdPartyNotices.txt' "$1" "$1"; }
cu=onnxruntime-provider-cuda13-sm120 trt=onnxruntime-provider-tensorrt-cuda13 ov=onnxruntime-provider-openvino
py=python-onnxruntime-core site=usr/lib/python3.14/site-packages
for line in \
    'onnxruntime-core depends: glibc libgcc libstdc++' \
    'onnxruntime-core provides: onnxruntime=1.29.0 libonnxruntime.so' \
    'onnxruntime-core conflicts: onnxruntime onnxruntime-cpu onnxruntime-cuda onnxruntime-opt-cuda onnxruntime-rocm onnxruntime-opt-rocm onnxruntime-git onnxruntime-full' \
    'onnxruntime-core replaces: onnxruntime-full' \
    "onnxruntime-core files: ./usr/include/onnxruntime/onnxruntime_c_api.h ./usr/include/onnxruntime/onnxruntime_cxx_api.h ./usr/lib/cmake/onnxruntime/onnxruntimeConfig.cmake ./usr/lib/libonnxruntime.so ./usr/lib/libonnxruntime.so.1 ./usr/lib/libonnxruntime.so.1.29.0 ./usr/lib/libonnxruntime_providers_shared.so ./usr/lib/pkgconfig/libonnxruntime.pc $(lic onnxruntime-core) " \
    "$cu depends: onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ cuda cudnn nvidia-utils cuda>=13 cuda<14 cudnn>=9 cudnn<10" \
    "$cu provides: onnxruntime-provider-cuda=1.29.0 onnxruntime-provider-cuda13=1.29.0" \
    "$cu conflicts: onnxruntime-provider-cuda onnxruntime-provider-cuda13" \
    "$cu files: ./usr/lib/libonnxruntime_providers_cuda.so $(lic "$cu") " \
    "$trt depends: onnxruntime-core=1.29.0-3 onnxruntime-provider-cuda13=1.29.0 glibc libgcc libstdc++ cuda cudnn nvidia-utils tensorrt>=11.4 tensorrt<11.5 cuda>=13 cuda<14 cudnn>=9 cudnn<10" \
    "$trt provides: onnxruntime-provider-tensorrt=1.29.0" \
    "$trt conflicts: onnxruntime-provider-tensorrt" \
    "$trt files: ./usr/lib/libonnxruntime_providers_tensorrt.so $(lic "$trt") " \
    "$ov depends: onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ openvino openvino>=2026.4 openvino<2026.5" \
    "$ov provides: " "$ov conflicts: " "$ov replaces: " \
    "$ov files: ./usr/lib/libonnxruntime_providers_openvino.so $(lic "$ov") " \
    "$py depends: onnxruntime-core=1.29.0-3 glibc libgcc libstdc++ python python-numpy python-flatbuffers python-packaging python-protobuf python>=3.14 python<3.15" \
    "$py provides: python-onnxruntime=1.29.0" \
    "$py conflicts: python-onnxruntime python-onnxruntime-cpu python-onnxruntime-cuda python-onnxruntime-opt-cuda python-onnxruntime-rocm python-onnxruntime-opt-rocm python-onnxruntime-full" \
    "$py replaces: python-onnxruntime-full" \
    "$py files: ./usr/bin/onnxruntime_test ./$site/onnxruntime/__init__.py ./$site/onnxruntime/capi/libonnxruntime.so.1.29.0 ./$site/onnxruntime/capi/libonnxruntime_providers_cuda.so ./$site/onnxruntime/capi/libonnxruntime_providers_openvino.so ./$site/onnxruntime/capi/libonnxruntime_providers_shared.so ./$site/onnxruntime/capi/libonnxruntime_providers_tensorrt.so ./$site/onnxruntime/capi/onnxruntime_pybind11_state.so $(lic "$py") "; do
    if has_line "$line"; then ok "package: $line"; else fail "package: no line [$line] in: $out"; fi
done
if [[ -f $pkgroot/$cu/usr/lib/libonnxruntime_providers_cuda.so && $(<"$pkgroot/$cu/usr/lib/libonnxruntime_providers_cuda.so") == cuda &&
    ! -L $pkgroot/onnxruntime-core/usr/lib/libonnxruntime.so.1.29.0 && -L $pkgroot/onnxruntime-core/usr/lib/libonnxruntime.so ]]; then
    ok "each package holds the staged library itself, the links as links"
else
    fail "the packaged files are not the staged ones"
fi
if [[ $(<"${pkgroot%/pkg}/rpath/libonnxruntime_providers_tensorrt.so") == /opt/nvidia/tensorrt/lib ]] &&
    grep -qF 'RUNPATH /opt/nvidia/tensorrt/lib' "${pkgroot%/pkg}/$trt.log" &&
    [[ ! -e ${pkgroot%/pkg}/rpath/libonnxruntime_providers_cuda.so ]]; then
    ok "the TensorRT provider alone gets TensorRT's lib/ as its RUNPATH"
else
    fail "the TensorRT RUNPATH: $(cat "${pkgroot%/pkg}/$trt.log")"
fi
# The RUNPATH of a build against /usr (the AUR's tensorrt) and of one
# against another root.
conf_usr=${conf_all//\/opt\/nvidia\/tensorrt\/lib/\/usr\/lib}
conf_other=${conf_all//\/opt\/nvidia\/tensorrt\/lib/\/srv\/trt\/lib64}
for case in "usr|/opt/nvidia/tensorrt/lib" "other|/srv/trt/lib64:/opt/nvidia/tensorrt/lib"; do
    IFS='|' read -r name want <<<"$case"
    conf=$conf_usr
    [[ $name != other ]] || conf=$conf_other
    package_run "$conf" --
    if ((status == 0)) && [[ $(<"${pkgroot%/pkg}/rpath/libonnxruntime_providers_tensorrt.so") == "$want" ]] &&
        grep -qF "RUNPATH $want" "${pkgroot%/pkg}/$trt.log"; then
        ok "a TensorRT build against $name: RUNPATH $want"
    else
        fail "the RUNPATH of a $name build (exit $status): $(cat "${pkgroot%/pkg}/$trt.log")"
    fi
done
package_run "$conf_all" --
if [[ $(readlink "$pkgroot/$py/$site/onnxruntime/capi/libonnxruntime_providers_cuda.so") == /usr/lib/libonnxruntime_providers_cuda.so &&
    $(head -n 1 "$pkgroot/$py/usr/bin/onnxruntime_test") == "#!$fx/pystub/python" ]] &&
    grep -qxF 'onnxruntime_pybind11_state.so needs: libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-x86-64.so.2' \
        "${pkgroot%/pkg}/$py.log"; then
    ok "the Python package links the libraries, rewrites the script's #! and checks the module's DT_NEEDED"
else
    fail "the Python package: $(cat "${pkgroot%/pkg}/$py.log")"
fi

package_run $'ORTF_EPS=cpu\nORTF_CUDA_ARCHS=120\nORTF_CUDA_PRESET=\nORTF_CUDA_MAJOR=\nORTF_PYTHON=0\nORTF_TENSORRT_HOME=\nORTF_TENSORRT_LIBDIR=\nORTF_TENSORRT_VERSION=' \
    -- ORTF_EPS=cpu FX_STAGE=
if ((status == 0)) && [[ $out == 'onnxruntime-core depends: '* ]] &&
    has_line "onnxruntime-core files: ./usr/include/onnxruntime/onnxruntime_c_api.h ./usr/include/onnxruntime/onnxruntime_cxx_api.h ./usr/lib/cmake/onnxruntime/onnxruntimeConfig.cmake ./usr/lib/libonnxruntime.so ./usr/lib/libonnxruntime.so.1 ./usr/lib/libonnxruntime.so.1.29.0 ./usr/lib/pkgconfig/libonnxruntime.pc $(lic onnxruntime-core) "; then
    ok "a CPU-only build: onnxruntime-core alone, without the shared provider library"
else
    fail "a CPU-only build (exit $status): $out"
fi
# expect_package_fail <description> <failing package> <message fragment> <package_run arguments...>
expect_package_fail() {
    local desc=$1 pkg=$2 msg=$3
    shift 3
    package_run "$@"
    if ((status != 0)) && has_line "$pkg FAILED (exit 1):" && grep -qF -- "$msg" <<<"$out"; then ok "$desc"; else
        fail "$desc (exit $status): $out"; fi
}
expect_package_fail "a CUDA provider linking another CUDA major fails" "$cu" \
    'libcudart.so.12 is CUDA 12, the package names say CUDA 13' "$conf_all" \
    'libonnxruntime_providers_cuda.so=libcublas.so.12 libcudart.so.12 libc.so.6' --
expect_package_fail "the CUDA package of a build recorded without a cuDNN major fails" "$cu" \
    'ORTF_CUDNN_MAJOR is '"''"', not a major version' "${conf_all/ORTF_CUDNN_MAJOR=9/ORTF_CUDNN_MAJOR=}" --
expect_package_fail "an ortf.conf without the cuDNN major fails" "$cu" 'ortf.conf: no ORTF_CUDNN_MAJOR' \
    "${conf_all/ORTF_CUDNN_MAJOR=9/}" --
expect_package_fail "a TensorRT provider linking another cuDNN major than the build's fails" "$trt" \
    'libcudnn.so.10 is cuDNN 10, the build used cuDNN 9' "$conf_all" \
    'libonnxruntime_providers_tensorrt.so=libnvinfer.so.11 libcudart.so.13 libcudnn.so.10 libc.so.6' --
expect_package_fail "a recorded cuDNN major that the TensorRT provider's libcudnn.so.9 does not match fails" "$trt" \
    'libcudnn.so.9 is cuDNN 9, the build used cuDNN 10' "${conf_all/ORTF_CUDNN_MAJOR=9/ORTF_CUDNN_MAJOR=10}" --
expect_package_fail "a provider linking an unknown library fails" "$ov" \
    'libze_loader.so.1 is a library no package is known for' "$conf_all" \
    'libonnxruntime_providers_openvino.so=libopenvino.so.2641 libze_loader.so.1 libc.so.6' --
expect_package_fail "libonnxruntime.so linking a vendor library fails" onnxruntime-core \
    'onnxruntime-core: libonnxruntime.so.1.29.0: libcudart.so.13 belongs to cuda, which is not in the depends of onnxruntime-core' "$conf_all" \
    'libonnxruntime.so.1.29.0=libstdc++.so.6 libcudart.so.13 libc.so.6' --
expect_package_fail "the shared provider library linking a vendor library fails" onnxruntime-core \
    'onnxruntime-core: libonnxruntime_providers_shared.so: libcuda.so.1 belongs to nvidia-utils' "$conf_all" \
    'libonnxruntime_providers_shared.so=libcuda.so.1' --
expect_package_fail "a build recorded with another CUDA major fails by name" onnxruntime-core \
    'the build used CUDA 12' "${conf_all/ORTF_CUDA_MAJOR=13/ORTF_CUDA_MAJOR=12}" --
expect_package_fail "the CUDA package of a build without cuda fails" "$cu" \
    'libonnxruntime_providers_cuda.so not found' "$conf_all" -- FX_STAGE='shared tensorrt openvino'
expect_package_fail "a TensorRT provider linking another TensorRT major fails" "$trt" \
    'libnvinfer.so.10 is TensorRT 10, the installed tensorrt is TensorRT 11' "$conf_all" \
    'libonnxruntime_providers_tensorrt.so=libnvinfer.so.10 libcudart.so.13 libc.so.6' --
expect_package_fail "an empty recorded openvino version fails" "$ov" \
    'ortf.conf: ORTF_OPENVINO_VERSION is empty (run makepkg without -e' \
    "${conf_all/ORTF_OPENVINO_VERSION=2026.4.1-1/ORTF_OPENVINO_VERSION=}" --
expect_package_fail "an ortf.conf without the TensorRT version fails" "$trt" 'ortf.conf: no ORTF_TENSORRT_VERSION' \
    "${conf_all/ORTF_TENSORRT_VERSION=11.4.0.106/}" --
expect_package_fail "an empty recorded Python version fails" "$py" 'ortf.conf: ORTF_PYTHON_VERSION is empty' \
    "${conf_all/ORTF_PYTHON_VERSION=3.14.8/ORTF_PYTHON_VERSION=}" --
expect_package_fail "the Python module linking a vendor library fails" "$py" \
    'python-onnxruntime-core: onnxruntime_pybind11_state.so: libcudart.so.13 belongs to cuda' "$conf_all" \
    'onnxruntime_pybind11_state.so=libcudart.so.13 libc.so.6' --
expect_package_fail "a library of the wheel that no other package holds fails" "$py" \
    'libonnxruntime_providers_rocm.so is in the wheel but in no other package' "$conf_all" -- \
    FX_WHEEL_LIBS='core shared rocm'

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
    # srcinfo <env...>: makepkg --printsrcinfo with that environment.
    srcinfo() {
        (cd "${pkgbuild%/*}" && env -u ORTF_EPS -u ORTF_CUDA_ARCHS -u ORTF_PYTHON -u ORTF_TENSORRT_HOME \
            -u MAKEFLAGS -u FX_CUDA_HOME "$@" makepkg --printsrcinfo 2>&1) || echo "FAILED"
    }
    # only_blocks <pkgname...>: the .SRCINFO on stdin, with the pkgbase
    # block and the blocks of those packages only.
    only_blocks() {
        awk -v keep=" $* " 'BEGIN { RS = ""; ORS = "\n\n" }
            /^pkgbase = / || index(keep, " " substr($1 " " $3, length($1) + 2) " ") { print }'
    }
    a=$(srcinfo)
    # Only the package names depend on the build inputs: with fewer
    # providers, the remaining blocks are the same; with other
    # architectures, the same but for the suffix.
    b=$(srcinfo ORTF_EPS=cpu ORTF_PYTHON=0 ORTF_CUDA_ARCHS=89 ORTF_TENSORRT_HOME=/x MAKEFLAGS=-j3)
    a_core=$(only_blocks onnxruntime-core <<<"$a")
    if [[ $a != *FAILED* && $(only_blocks onnxruntime-core <<<"$b") == "$a_core" &&
        $(grep -c '^pkgname = ' <<<"$b") == 1 ]]; then
        ok "a CPU-only build without Python: the pkgbase and onnxruntime-core blocks alone, unchanged"
    else
        fail "ORTF_EPS=cpu ORTF_PYTHON=0: $(diff <(echo "$a_core") <(echo "$b") | head -8)"
    fi
    c=$(srcinfo ORTF_CUDA_ARCHS='120,89' ORTF_TENSORRT_HOME=/x ORTF_NVCC_THREADS=4)
    if [[ $a != *FAILED* && ${c//-sm89-sm120/-sm120} == "$a" && $c == *onnxruntime-provider-cuda13-sm89-sm120* ]]; then
        ok "ORTF_CUDA_ARCHS changes the CUDA name only"
    else
        fail "ORTF_CUDA_ARCHS='120,89': $(diff <(echo "$a") <(echo "${c//-sm89-sm120/-sm120}") | head -8)"
    fi
    e=$(srcinfo ORTF_CUDA_ARCHS=rtx)
    if [[ $a != *FAILED* && ${e//-cuda13-rtx/-cuda13-sm120} == "$a" && $e == *'pkgname = onnxruntime-provider-cuda13-rtx'* ]]; then
        ok "ORTF_CUDA_ARCHS=rtx names the CUDA package onnxruntime-provider-cuda13-rtx, nothing else changes"
    else
        fail "ORTF_CUDA_ARCHS=rtx: $(diff <(echo "$a") <(echo "${e//-cuda13-rtx/-cuda13-sm120}") | head -8)"
    fi
    # The CUDA major is the toolkit's: another toolkit renames the CUDA and
    # TensorRT packages and their names in depends, provides and conflicts,
    # and changes nothing else.
    g=$(srcinfo FX_CUDA_HOME="$fx/cuda12")
    if [[ $a != *FAILED* && ${g//-cuda12/-cuda13} == "$a" && $g == *'pkgname = onnxruntime-provider-cuda12-sm120'* &&
        $g == *'pkgname = onnxruntime-provider-tensorrt-cuda12'* && $g == *$'\tdepends = onnxruntime-provider-cuda12=1.29.0\n'* ]]; then
        ok "a CUDA 12 toolkit gives -cuda12- names, nothing else changes"
    else
        fail "a CUDA 12 toolkit: $(diff <(echo "$a") <(echo "${g//-cuda12/-cuda13}") | head -8)"
    fi
    h=$(srcinfo FX_CUDA_HOME="$fx/empty")
    if [[ $a != *FAILED* && ${h//-cudax/-cuda13} == "$a" && $h == *'pkgname = onnxruntime-provider-cudax-sm120'* ]]; then
        ok "no CUDA toolkit: the fallback -cudax- names, nothing else changes"
    else
        fail "no CUDA toolkit: $(diff <(echo "$a") <(echo "${h//-cudax/-cuda13}") | head -8)"
    fi
    # The TensorRT package, without architectures, works with any CUDA
    # package of its CUDA major.
    trt_block=$(only_blocks onnxruntime-provider-tensorrt-cuda13 <<<"$a")
    if [[ $trt_block == *$'\tdepends = onnxruntime-provider-cuda13=1.29.0\n'* &&
        $trt_block != *'onnxruntime-provider-cuda13-'* && $trt_block != *'onnxruntime-provider-cuda='* &&
        $(grep -c $'^\t\(provides\|conflicts\) = ' <<<"$trt_block") == 2 &&
        $trt_block == *$'\tprovides = onnxruntime-provider-tensorrt=1.29.0\n'* ]] &&
        grep -qxF $'\tconflicts = onnxruntime-provider-tensorrt' <<<"$trt_block"; then
        ok "the TensorRT package depends on onnxruntime-provider-cuda13, provides and conflicts onnxruntime-provider-tensorrt only"
    else
        fail "the TensorRT block: $trt_block"
    fi
    d=$(srcinfo ORTF_EPS='openvino cuda')
    if [[ $(only_blocks onnxruntime-core onnxruntime-provider-cuda13-sm120 onnxruntime-provider-openvino python-onnxruntime-core <<<"$a") == \
        "$(only_blocks onnxruntime-core onnxruntime-provider-cuda13-sm120 onnxruntime-provider-openvino python-onnxruntime-core <<<"$d")" &&
        $d != *'pkgname = onnxruntime-provider-tensorrt'* && $d != *FAILED* ]]; then
        ok "ORTF_EPS without tensorrt drops the TensorRT package only"
    else
        fail "ORTF_EPS='openvino cuda': $d"
    fi

    # The dependencies between the packages (the -bin convention at the top
    # of the PKGBUILD). Each package of a build is installed either as
    # built here, under its own name and version (pkgver-pkgrel) with its
    # provides, or as a hypothetical <name>-bin, which provides
    # <name>=<pkgver>-<pkgrel> and the same provides. Every depend of a
    # package on a name of this recipe's (onnxruntime-*, python-onnxruntime*)
    # must be met, compared as pacman does (vercmp), in both views; a
    # conflict must hit no other package of the same build. The vendor
    # depends are pacman's to meet.
    if command -v vercmp >/dev/null 2>&1; then
        # satisfies <dependency> <provision>: whether a package or provision
        # <name>[=<version>] meets <name>[<op><version>]. A provision
        # without a version meets only a dependency without one.
        satisfies() {
            local dname=$1 op='' dver='' pname=${2%%=*} pver='' c
            if [[ $1 =~ ^([^\<\>=]+)(\<=|\>=|\<|\>|=)(.+)$ ]]; then
                dname=${BASH_REMATCH[1]} op=${BASH_REMATCH[2]} dver=${BASH_REMATCH[3]}
            fi
            [[ $2 != *=* ]] || pver=${2#*=}
            [[ $pname == "$dname" ]] || return 1
            [[ -n $op ]] || return 0
            [[ -n $pver ]] || return 1
            c=$(vercmp "$pver" "$dver")
            case $op in
                '=') ((c == 0)) ;;
                '>=') ((c >= 0)) ;;
                '<=') ((c <= 0)) ;;
                '>') ((c > 0)) ;;
                '<') ((c < 0)) ;;
            esac
        }
        # field <.SRCINFO> <pkgname> <key>: that package's values of <key>.
        field() {
            awk -v p="$2" -v k="$3" 'BEGIN { RS = ""; FS = "\n" }
                $1 == "pkgname = " p { for (i = 2; i <= NF; i++) if (index($i, "\t" k " = ") == 1) print substr($i, length(k) + 5) }' <<<"$1"
        }
        # chain <.SRCINFO> <source|bin> [<extra depends of the first package>]:
        # checks the depends and conflicts in that view; prints the
        # problems, fails if there is one.
        chain() {
            local info=$1 view=$2 ver p q d c have bad=0
            local -a pkgs provs
            ver=$(sed -n 's/^\tpkgver = //p' <<<"$info")-$(sed -n 's/^\tpkgrel = //p' <<<"$info")
            mapfile -t pkgs < <(sed -n 's/^pkgname = //p' <<<"$info")
            # provs[i]: what package i offers, one per line.
            for i in "${!pkgs[@]}"; do
                p=${pkgs[i]}
                provs[i]="$p=$ver"$'\n'"$(field "$info" "$p" provides)"
                [[ $view == source ]] || provs[i]+=$'\n'"$p-bin=$ver"
            done
            for i in "${!pkgs[@]}"; do
                p=${pkgs[i]}
                local -a deps
                mapfile -t deps < <(field "$info" "$p" depends)
                ((i != 0)) || [[ -z ${3:-} ]] || deps+=("$3")
                for d in "${deps[@]}"; do
                    [[ $d == onnxruntime-* || $d == python-onnxruntime* ]] || continue
                    have=''
                    for j in "${!pkgs[@]}"; do
                        ((j != i)) || continue
                        while IFS= read -r q; do
                            [[ -n $q ]] && satisfies "$d" "$q" && have=${pkgs[j]}
                        done <<<"${provs[j]}"
                    done
                    if [[ -z $have ]]; then
                        echo "$view: $p depends on $d, which no other package of the build meets"
                        bad=1
                    fi
                done
                while IFS= read -r c; do
                    [[ -n $c ]] || continue
                    for j in "${!pkgs[@]}"; do
                        ((j != i)) || continue
                        while IFS= read -r q; do
                            if [[ -n $q ]] && satisfies "$c" "$q"; then
                                echo "$view: $p conflicts with $c, which ${pkgs[j]} of the same build is or provides"
                                bad=1
                            fi
                        done <<<"${provs[j]}"
                    done
                done < <(field "$info" "$p" conflicts)
            done
            return "$bad"
        }
        for view in source bin; do
            for variant in a e g; do
                if [[ ${!variant} != *FAILED* ]] && msg=$(chain "${!variant}" "$view"); then
                    ok "every dependency between the packages is met ($view packages, .SRCINFO \$$variant)"
                else
                    fail "the dependencies ($view, \$$variant): $msg"
                fi
            done
            # A depend no package meets fails: a name only the old packages
            # had, a version none is at, and (built here) a -bin name.
            deps=('onnxruntime-full=1.29.0-2' 'onnxruntime-provider-cuda13>=1.30' 'onnxruntime-core=1.29.0-1')
            [[ $view != source ]] || deps+=('onnxruntime-provider-tensorrt-cuda13-bin')
            for dep in "${deps[@]}"; do
                if msg=$(chain "$a" "$view" "$dep"); then fail "$view: a depend on $dep passes"; else
                    ok "$view: a depend on $dep, which no package meets, fails"; fi
            done
        done
        # A package of the build that conflicts with another fails.
        bad_info=$(sed $'/^pkgname = onnxruntime-provider-openvino$/a\\\tconflicts = onnxruntime-provider-cuda' <<<"$a")
        if ! msg=$(chain "$bad_info" source) &&
            [[ $msg == *'onnxruntime-provider-openvino conflicts with onnxruntime-provider-cuda, which onnxruntime-provider-cuda13-sm120'* ]]; then
            ok "a conflict with another package of the build fails"
        else
            fail "a conflict within the build: $msg"
        fi
        # The tensorrt package's depend on the CUDA package's major name, and
        # pacman's comparison: the provision 1.29.0 meets =1.29.0 and
        # =1.29.0-2 (no pkgrel on one side: not compared).
        if [[ $(field "$a" onnxruntime-provider-tensorrt-cuda13 depends) == *'onnxruntime-provider-cuda13=1.29.0'* ]] &&
            satisfies 'onnxruntime-provider-cuda13=1.29.0' 'onnxruntime-provider-cuda13=1.29.0' &&
            ! satisfies 'onnxruntime-provider-cuda13>=1.29.1' 'onnxruntime-provider-cuda13=1.29.0' &&
            ! satisfies 'onnxruntime-core=1.29.0-2' 'onnxruntime-core' &&
            satisfies 'onnxruntime-core' 'onnxruntime-core'; then
            ok "satisfies: versioned provisions only meet versioned depends"
        else
            fail "satisfies does not compare as pacman does"
        fi
    else
        echo "skip the dependency chain (no vercmp: pacman is not installed)"
    fi
else
    echo "skip makepkg --printsrcinfo (makepkg not found)"
fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"

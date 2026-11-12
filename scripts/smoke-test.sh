#!/usr/bin/env bash
# Smoke test for an installed ONNX Runtime (package onnxruntime-full): lists the
# available execution providers and runs a tiny model on each engine/device.
#
# Usage: scripts/smoke-test.sh [--prefix DIR] [--include DIR] [--libdir DIR]
#                              [--require LIST] [--only LIST] [--runs N] [--keep]
#
#   --prefix DIR    installation prefix (default /usr): headers in
#                   DIR/include/onnxruntime, libraries in DIR/lib
#   --include DIR   directory holding onnxruntime_c_api.h, overriding --prefix.
#                   A source/build tree also works: DIR/include/onnxruntime
#                   (the script looks in core/session/ below it)
#   --libdir DIR    directory holding libonnxruntime.so, overriding --prefix
#                   (e.g. a build tree's build/Linux/Release)
#   --require LIST  targets that must be "ok", else exit 1 (default: none)
#   --only LIST     test only these targets (default: all)
#   --runs N        warm runs per target (default 5)
#   --keep          keep the temporary build directory and print its path
#
# Targets (LIST is comma-separated): cpu, cuda, tensorrt, openvino:CPU,
# openvino:GPU, openvino:NPU ("openvino" selects all three devices).
#
# Environment: CC selects the C compiler (default cc), PYTHON the interpreter
# that writes the model (default python3).
#
# Exit status: 0 success, 1 a test failed (or a required target is not ok),
# 2 usage error, 3 missing ONNX Runtime, compiler or Python.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../tools/ort-smoke
cc=${CC:-cc}
python=${PYTHON:-python3}

prefix=/usr include='' libdir='' require='' only='' runs='' keep=0

die() { # die <status> <message>
    echo "smoke-test.sh: $2" >&2
    exit "$1"
}
usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
need_value() { # need_value <option> <remaining arg count>
    (($2 >= 2)) || die 2 "$1 needs a value"
}

while (($# > 0)); do
    case $1 in
    -h | --help) usage; exit 0 ;;
    --prefix) need_value "$1" $#; prefix=$2; shift 2 ;;
    --include) need_value "$1" $#; include=$2; shift 2 ;;
    --libdir) need_value "$1" $#; libdir=$2; shift 2 ;;
    --require) need_value "$1" $#; require=$2; shift 2 ;;
    --only) need_value "$1" $#; only=$2; shift 2 ;;
    --runs) need_value "$1" $#; runs=$2; shift 2 ;;
    --keep) keep=1; shift ;;
    *) die 2 "unknown argument: $1 (see --help)" ;;
    esac
done
[[ -z $runs || $runs =~ ^[0-9]+$ ]] || die 2 "--runs needs a non-negative integer, got '$runs'"

include=${include:-$prefix/include/onnxruntime}
libdir=${libdir:-$prefix/lib}

# A source tree keeps the header under core/session/.
if [[ ! -f $include/onnxruntime_c_api.h && -f $include/core/session/onnxruntime_c_api.h ]]; then
    include=$include/core/session
fi
[[ -f $include/onnxruntime_c_api.h ]] ||
    die 3 "onnxruntime_c_api.h not found in $include: is onnxruntime-full installed? (see --prefix / --include)"
[[ -e $libdir/libonnxruntime.so ]] ||
    die 3 "libonnxruntime.so not found in $libdir: is onnxruntime-full installed? (see --prefix / --libdir)"
command -v "$cc" >/dev/null || die 3 "C compiler '$cc' not found (set CC)"
command -v "$python" >/dev/null || die 3 "'$python' not found (set PYTHON)"

tmp=$(mktemp -d)
if ((keep)); then
    trap 'echo "kept: $tmp" >&2' EXIT
else
    trap 'rm -rf "$tmp"' EXIT
fi

"$python" "$tool/make-model.py" "$tmp/model.onnx"
"$cc" -std=c11 -O1 -Wall -Wextra -o "$tmp/smoke" "$tool/smoke.c" \
    -I"$include" -L"$libdir" -lonnxruntime -lm

args=("$tmp/model.onnx")
[[ -z $only ]] || args+=(--only "$only")
[[ -z $require ]] || args+=(--require "$require")
[[ -z $runs ]] || args+=(--runs "$runs")

# The system library path already finds an installed ORT; anything else needs help.
case $libdir in
/usr/lib | /usr/lib64) ;;
*) export LD_LIBRARY_PATH=$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
esac

status=0
"$tmp/smoke" "${args[@]}" || status=$?
exit "$status"

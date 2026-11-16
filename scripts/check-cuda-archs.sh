#!/usr/bin/env bash
# Asserts the GPU code embedded in a CUDA provider library.
#
#   scripts/check-cuda-archs.sh [--cuobjdump PATH] FILE ARCH...
#   e.g. scripts/check-cuda-archs.sh onnxruntime-provider-cuda12-legacy-1.29.0-1-x86_64.pkg.tar.zst \
#            60 61 70 75 80 86 89 120 60-virtual
#
# FILE is a package file (*.pkg.tar.zst: usr/lib/libonnxruntime_providers_cuda.so*
# is unpacked with bsdtar into a temporary directory) or the shared library
# itself. ARCH... is the expected list in the word form of ORTF_CUDA_ARCHS:
# N or N-real is SASS (machine code) for sm_N, N-virtual is PTX for compute_N.
# ONNX Runtime compiles 90, 100, 103 and 120 as Na (sm_120a): the suffix
# letter is dropped, so sm_120a counts as 120.
#
# It runs `cuobjdump --list-elf` and `cuobjdump --list-ptx` and requires
#   - exactly the expected set of SASS archs (no missing one, no extra one),
#   - exactly the expected PTX archs; with no N-virtual in the list, no PTX
#     at all.
# Each line of the listings names the embedded file ("ELF file 1:
# libonnxruntime_providers_cuda.1.sm_120a.cubin", "PTX file 1: ....sm_120.ptx");
# the parser is tolerant: it takes the sm_<N>[letter] before .cubin / .ptx at
# the end of any line and ignores the rest. A listing with no such line is an
# empty set (cuobjdump says "does not contain" and may exit non-zero then).
# It prints what it found next to what is expected.
#
# --cuobjdump PATH defaults to cuobjdump on PATH, else /opt/cuda/bin/cuobjdump.
# Exit status: 0 when SASS and PTX match, 1 when they do not, 2 on a usage
# error, 3 when cuobjdump, bsdtar or the file (or the library in the package)
# is missing or cannot be read. Read-only: it writes only under mktemp.
set -euo pipefail

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; }
die() { # die <status> <message>
    echo "check-cuda-archs: $2" >&2
    exit "$1"
}

cuobjdump=''
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --cuobjdump) (($# >= 2)) || die 2 "--cuobjdump needs a path"; cuobjdump=$2; shift 2 ;;
        --cuobjdump=*) cuobjdump=${1#*=}; shift ;;
        --) shift; break ;;
        -*) die 2 "unknown option: $1 (see --help)" ;;
        *) break ;;
    esac
done
if (($# < 2)); then
    usage >&2
    exit 2
fi
file=$1
shift

# The expected sets, sorted and unique.
sass_list=() ptx_list=()
for a in "$@"; do
    if [[ $a =~ ^([0-9]+)(-real)?$ ]]; then
        sass_list+=("${BASH_REMATCH[1]}")
    elif [[ $a =~ ^([0-9]+)-virtual$ ]]; then
        ptx_list+=("${BASH_REMATCH[1]}")
    else
        die 2 "$a: not an arch (N, N-real or N-virtual)"
    fi
done
sorted() { (($# == 0)) || printf '%s\n' "$@" | sort -nu | tr '\n' ' ' | sed 's/ $//'; }
want_sass=$(sorted "${sass_list[@]}")
want_ptx=$(sorted "${ptx_list[@]}")

if [[ -z $cuobjdump ]]; then
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump=cuobjdump
    else
        cuobjdump=/opt/cuda/bin/cuobjdump
    fi
fi
if ! command -v "$cuobjdump" >/dev/null 2>&1; then
    die 3 "$cuobjdump: cuobjdump not found (use --cuobjdump PATH)"
fi
[[ -f $file ]] || die 3 "$file: no such file"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

lib=$file
if [[ $file == *.pkg.tar.* ]]; then
    command -v bsdtar >/dev/null 2>&1 || die 3 "bsdtar not found"
    if ! bsdtar -tf "$file" >/dev/null 2>"$tmp/err"; then
        die 3 "$file: cannot read it: $(head -n 1 "$tmp/err")"
    fi
    # bsdtar exits non-zero when the pattern matches nothing: checked below.
    bsdtar -xf "$file" -C "$tmp" --include 'usr/lib/libonnxruntime_providers_cuda.so*' 2>/dev/null || true
    libs=("$tmp"/usr/lib/libonnxruntime_providers_cuda.so*)
    [[ -e ${libs[0]} ]] || die 3 "$file: no usr/lib/libonnxruntime_providers_cuda.so* in it"
    lib=${libs[0]}
fi

# found <--list-elf|--list-ptx>: the sm numbers listed, sorted and unique.
found() {
    local out rc=0
    out=$("$cuobjdump" "$1" "$lib" 2>&1) || rc=$?
    if ((rc != 0)) && ! grep -qi 'does not contain' <<<"$out"; then
        die 3 "cuobjdump $1 $lib failed (exit $rc): $(head -n 1 <<<"$out")"
    fi
    local nums=()
    mapfile -t nums < <(sed -nE 's/.*sm_([0-9]+)[a-z]*\.(cubin|ptx)[[:space:]]*$/\1/p' <<<"$out")
    sorted "${nums[@]}"
}
got_sass=$(found --list-elf)
got_ptx=$(found --list-ptx)

status=0
show() { # show <label> <found> <expected>
    local verdict=ok
    if [[ $2 != "$3" ]]; then
        verdict=MISMATCH
        status=1
    fi
    printf '%-4s found: %s\n%-4s want:  %s\n%-4s %s\n' "$1" "${2:-none}" "$1" "${3:-none}" "$1" "$verdict"
}
echo "$file ($lib)"
show SASS "$got_sass" "$want_sass"
show PTX "$got_ptx" "$want_ptx"
exit "$status"

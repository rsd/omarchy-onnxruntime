#!/usr/bin/env bash
# The licence guard: fails when a file that must never be committed is about
# to be. This repository holds recipes only; nothing NVIDIA, Intel or AMD ships
# (libraries, headers, archives, wheels, packages, models, engine caches,
# licence documents) and no build output may enter it.
#
#   check-no-vendor.sh [--staged]        # the index (default; pre-commit and
#                                        # pre-merge-commit hooks)
#   check-no-vendor.sh --all             # every tracked and untracked, not
#                                        # ignored file of the work tree
#   check-no-vendor.sh --range REV...    # every file added or modified by the
#                                        # commits `git rev-list REV...` lists
#                                        # (pre-push hook), e.g.
#                                        #   --range origin/main..main
#                                        #   --range main --not --remotes=origin
#
# --all looks at the work tree only, not at the history; --range covers
# commits. Hooks can be bypassed with --no-verify: the guard is a safety net,
# not an access control.
#
# Every added, modified or type-changed entry is checked (a rename counts as
# an addition; only deletions are skipped). An entry fails when:
#   - its path matches a vendor or binary pattern: libraries (.so, .a, .o,
#     .dll), packages and archives (.pkg.tar*, .tar*, .tgz, .zip, .deb,
#     .rpm, .whl), models and engine caches (.onnx, .engine, .trt, .plan,
#     .trtpkg, .cubin, .fatbin, .blob, .bin, which includes OpenVINO IR
#     weights), PDFs, logs, TensorRT, cuDNN, CUDA, OpenVINO, MIGraphX, HIP
#     and oneDNN headers, a TensorRT-<version>/, OpenVINO toolkit or oneDNN
#     source directory (oneDNN-<version>/, oneapi/dnnl/), a CUDA targets/
#     tree, an opt/rocm/ tree, build output or makepkg's work directories
#     and sources;
#   - it is a submodule (gitlink), or, in --all, a nested repository;
#   - it is a symbolic link whose target is absolute or leaves the repository;
#   - its content is larger than 1 MiB, starts with a binary or archive magic
#     number (ELF, gzip, zip, PDF, xz, zstd, bzip2, ar, POSIX tar), has a NUL
#     byte in its first 8 KiB, or carries a marker of a vendor file: an
#     NVIDIA, Intel or AMD copyright line, a vendor licence agreement's title on
#     a line of its own, or the root element of an OpenVINO IR model (.xml).
# Staged and committed content is read from the object database with git
# cat-file (read-only), so what is checked is what would be committed or
# pushed. Exit status: 0 clean, 1 offending files, 2 usage or git error.
set -euo pipefail

prog=${0##*/}
max_size=$((1024 * 1024))

usage() {
    echo "usage: $prog [--staged | --all | --range REV...]" >&2
    exit 2
}

mode=staged range=()
if (($# > 0)); then
    case $1 in
        --staged) (($# == 1)) || usage; mode=staged ;;
        --all) (($# == 1)) || usage; mode=all ;;
        --range) (($# >= 2)) || usage; mode=range; shift; range=("$@") ;;
        *) usage ;;
    esac
fi

top=$(git rev-parse --show-toplevel 2>/dev/null) ||
    { echo "$prog: not inside a git work tree" >&2; exit 2; }
cd "$top"

tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

# Prints the reason when the path is forbidden, nothing otherwise. A path may
# hold any byte but NUL (spaces, newlines): it is matched with patterns and
# parameter expansion only, never split with read.
path_reason() {
    local path=$1 base lcbase rest component
    base=${path##*/}
    lcbase=${base,,}

    # Build output (also in .gitignore).
    case /$path/ in
        */build/* | */target/* | */dist/* | */__pycache__/*) echo "build output"; return ;;
    esac
    # makepkg's work directories and the sources it downloads next to the
    # recipe (SRCDEST defaults to it): the ONNX Runtime clone, the
    # dependency archives.
    case $path in
        onnxruntime-*/src/* | onnxruntime-*/pkg/*) echo "makepkg work directory"; return ;;
        onnxruntime/* | onnxruntime-*/onnxruntime/*) echo "ONNX Runtime source checkout"; return ;;
    esac

    # Vendor header names: TensorRT, cuDNN, CUDA runtime and libraries,
    # NCCL (nccl and nccl-cuda12.9, the build's NCCL).
    case $base in
        NvInfer*.h | NvOnnxParser*.h | NvOnnxConfig.h | NvUtils.h | NvCaffeParser.h | \
            cudnn*.h | cuda.h | cuda_runtime*.h | cublas*.h | cufft*.h | curand*.h | \
            nvrtc*.h | cupti*.h | nccl.h | nccl_net.h)
            echo "NVIDIA header"; return ;;
        openvino.hpp | ie_core.hpp | inference_engine.hpp)
            echo "OpenVINO header"; return ;;
        migraphx.h | migraphx.hpp)
            echo "MIGraphX header"; return ;;
        hip_runtime*.h | hip_version.h | hiprtc.h | amd_hip_*.h)
            echo "HIP header"; return ;;
        # hipblas*.h covers hipblaslt*.h; rccl.h is RCCL's (Arch's rccl,
        # the migraphx line's NCCL).
        rocblas*.h | rocm_version.h | hipblas*.h | miopen*.h | rccl.h)
            echo "ROCm library header"; return ;;
        # oneDNN (Intel; the oneDNN provider builds it from its source
        # archive, which stays out of the tree), and its old mkldnn names.
        dnnl.h | dnnl.hpp | dnnl_*.h | dnnl_*.hpp | mkldnn*.h | mkldnn*.hpp)
            echo "oneDNN header"; return ;;
    esac

    # Binary, package, model and archive extensions (case-insensitive).
    # Libraries are known by their suffix, whatever their name.
    case $lcbase in
        *.pkg.tar*) echo "built package"; return ;;
        *.so | *.so.* | *.a | *.o | *.dll | *.lib) echo "library or object file"; return ;;
        *.whl) echo "Python wheel"; return ;;
        *.onnx) echo "ONNX model"; return ;;
        *.engine | *.trt | *.plan | *.trtpkg | *.trtpkg.*) echo "TensorRT engine"; return ;;
        *.cubin | *.fatbin) echo "CUDA binary"; return ;;
        *.blob) echo "OpenVINO compiled model"; return ;;
        *.bin) echo "binary data (e.g. OpenVINO IR weights)"; return ;;
        *.tgz | *.tar | *.tar.* | *.zip | *.deb | *.rpm) echo "archive"; return ;;
        *.pdf) echo "PDF document"; return ;;
        *.log) echo "log file"; return ;;
    esac

    # Vendor trees: any directory component.
    case /${path,,}/ in
        */openvino/*.hpp/ | */openvino/*.h/) echo "OpenVINO header"; return ;;
        */migraphx/*.hpp/ | */migraphx/*.h/) echo "MIGraphX header"; return ;;
        */include/hip/*.h/ | */include/hip/*.hpp/) echo "HIP header"; return ;;
        */rocblas/*.h/ | */miopen/*.h/ | */hipblas/*.h/ | */hipblaslt/*.h/ | */rocm/*.h/ | */rccl/*.h/)
            echo "ROCm library header"; return ;;
        */opt/rocm/*) echo "ROCm tree"; return ;;
        */oneapi/dnnl/*) echo "oneDNN tree"; return ;;
        */targets/x86_64-linux/*) echo "CUDA toolkit tree"; return ;;
    esac
    rest=''
    [[ $path != */* ]] || rest=${path%/*}
    while [[ -n $rest ]]; do
        component=${rest%%/*}
        case $component in
            TensorRT-[0-9]*) echo "TensorRT tree ($component/)"; return ;;
            l_openvino_toolkit_* | openvino_toolkit_* | openvino_[0-9]*)
                echo "OpenVINO toolkit tree ($component/)"; return ;;
            [Oo]ne[Dd][Nn][Nn]-[0-9]* | mkl-dnn-[0-9]*) echo "oneDNN source tree ($component/)"; return ;;
            .ngc) echo "NGC configuration"; return ;;
        esac
        if [[ $rest == */* ]]; then rest=${rest#*/}; else rest=''; fi
    done
}

# Prints the reason when a symbolic link at <path> pointing to <target> is
# forbidden: an absolute target, or one that leaves the repository.
link_reason() {
    local path=$1 target=$2 rest component depth=0
    [[ -n $target ]] || return 0
    if [[ $target == /* ]]; then
        echo "symbolic link to an absolute path"
        return
    fi
    # The link's directory, then the target, resolved lexically.
    rest=''
    [[ $path != */* ]] || rest=${path%/*}/
    rest+=$target
    while [[ -n $rest ]]; do
        component=${rest%%/*}
        case $component in
            '' | .) ;;
            ..) depth=$((depth - 1)) ;;
            *) depth=$((depth + 1)) ;;
        esac
        if ((depth < 0)); then
            echo "symbolic link leaving the repository"
            return
        fi
        if [[ $rest == */* ]]; then rest=${rest#*/}; else rest=''; fi
    done
}

hex() { od -An -v -tx1 | tr -d ' \n'; }

# Text that marks a vendor file under any name (extended regular
# expressions, matched case-insensitively): copyright lines of NVIDIA,
# Intel and AMD headers, sources and licence files; a vendor licence agreement's
# title on a line of its own (prose that names the agreement is fine); the
# root element of an OpenVINO IR model. The bracketed spaces keep this
# script from matching its own patterns.
markers=(
    -e 'copyright.*nvidia[ ]corporation'
    -e 'nvidia[ ]corporation.*all[ ]rights[ ]reserved'
    -e 'copyright.*intel[ ]corporation'
    -e 'copyright.*advanced[ ]micro[ ]devices'
    -e '^[[:space:]]*nvidia[ ]software[ ]license[ ]agreement[[:space:]]*$'
    -e '^[[:space:]]*(nvidia[ ])?tensorrt[ ]license[ ]agreement[[:space:]]*$'
    -e '^[[:space:]]*(nvidia[ ])?(cudnn|cuda)[ ](supplement[ ]to[ ])?(software[ ])?license[ ]agreement[[:space:]]*$'
    -e '^[[:space:]]*intel[ ](simplified[ ]software|end[ ]user)[ ]license([ ]agreement)?[[:space:]]*$'
    -e '<net[ ]+name="[^"]*"[ ]+version="[0-9]+"'
)

# Prints the reason when the content of <file> (a regular file of <size>
# bytes) is forbidden, nothing otherwise.
content_reason() {
    local file=$1 size=$2 magic
    if ((size > max_size)); then
        echo "larger than 1 MiB ($size bytes)"
        return
    fi
    magic=$(head -c 8 -- "$file" | hex)
    case $magic in
        7f454c46*) echo "ELF binary"; return ;;
        1f8b*) echo "gzip data"; return ;;
        504b0304*) echo "zip data"; return ;;
        25504446*) echo "PDF document"; return ;;
        fd377a585a*) echo "xz data"; return ;;
        28b52ffd*) echo "zstd data"; return ;;
        425a68*) echo "bzip2 data"; return ;;
        213c617263683e0a*) echo "ar archive"; return ;;
    esac
    # POSIX tar: "ustar" at offset 257. (head may stop tail early.)
    if ((size >= 262)) && [[ $({ tail -c +258 -- "$file" || true; } | head -c 5 | hex) == 7573746172 ]]; then
        echo "tar archive"
        return
    fi
    if (($(head -c 8192 -- "$file" | tr -dc '\000' | wc -c) > 0)); then
        echo "binary data (a NUL byte in the first 8 KiB)"
        return
    fi
    if LC_ALL=C grep -qaiE "${markers[@]}" -- "$file"; then
        echo "vendor file content (copyright line, licence title or OpenVINO IR)"
    fi
}

failures=0
report() {
    local shown=$1
    [[ $shown != *[[:cntrl:]]* ]] || printf -v shown '%q' "$1"
    printf '%s: %s\n' "$shown" "$2"
    failures=$((failures + 1))
}

# check_object <path> <mode> <object id>: an index or tree entry.
check_object() {
    local path=$1 objmode=$2 oid=$3 reason size
    reason=$(path_reason "$path")
    if [[ -z $reason ]]; then
        case $objmode in
            160000) reason="submodule (gitlink)" ;;
            120000) reason=$(link_reason "$path" "$(git cat-file blob "$oid")") ;;
            100644 | 100755)
                size=$(git cat-file -s "$oid")
                if ((size > max_size)); then
                    reason="larger than 1 MiB ($size bytes)"
                else
                    git cat-file blob "$oid" >"$tmp/blob"
                    reason=$(content_reason "$tmp/blob" "$size")
                fi
                ;;
            *) reason="unexpected entry (mode $objmode)" ;;
        esac
    fi
    [[ -z $reason ]] || report "$path" "$reason"
}

# Reads `git diff --raw -z --no-abbrev --no-renames` records on stdin and
# checks each entry once.
check_raw() {
    local meta path dstmode dstoid
    local -A seen=()
    while IFS= read -r -d '' meta && IFS= read -r -d '' path; do
        read -r _ dstmode _ dstoid _ <<<"${meta#:}"
        # An unmerged entry has no object yet (git refuses to commit it).
        [[ $dstmode != 000000 ]] || continue
        [[ -z ${seen[$dstmode:$dstoid:$path]:-} ]] || continue
        seen[$dstmode:$dstoid:$path]=1
        check_object "$path" "$dstmode" "$dstoid"
    done
}

check_staged() {
    # Without a HEAD (no commit yet) git compares with the empty tree. In a
    # hook, GIT_INDEX_FILE names the index being committed (commit -a,
    # commit <path>).
    git diff --cached --raw -z --no-abbrev --no-renames --diff-filter=d \
        --ignore-submodules=none >"$tmp/raw" ||
        { echo "$prog: git diff --cached failed" >&2; exit 2; }
    check_raw <"$tmp/raw"
}

check_range() {
    git rev-list "${range[@]}" -- >"$tmp/commits" ||
        { echo "$prog: git rev-list ${range[*]} failed" >&2; exit 2; }
    # -m compares a merge with each of its parents, so what the merge itself
    # brings (a conflict resolution) is checked too.
    git diff-tree --stdin -r -m --root --no-commit-id --raw -z --no-abbrev \
        --no-renames --diff-filter=d --ignore-submodules=none <"$tmp/commits" >"$tmp/raw" ||
        { echo "$prog: git diff-tree failed" >&2; exit 2; }
    check_raw <"$tmp/raw"
}

check_all() {
    local path reason
    git ls-files -z -co --exclude-standard >"$tmp/files" ||
        { echo "$prog: git ls-files failed" >&2; exit 2; }
    while IFS= read -r -d '' path; do
        # An untracked nested repository is listed as dir/.
        path=${path%/}
        reason=$(path_reason "$path")
        if [[ -z $reason ]]; then
            if [[ -L $path ]]; then
                reason=$(link_reason "$path" "$(readlink -- "$path")")
            elif [[ -d $path ]]; then
                reason="submodule or nested repository"
            elif [[ -f $path ]]; then
                reason=$(content_reason "$path" "$(stat -c %s -- "$path")")
            fi
            # Anything else is a tracked file deleted from the work tree.
        fi
        [[ -z $reason ]] || report "$path" "$reason"
    done < <(sort -zu "$tmp/files")
}

case $mode in
    staged) check_staged ;;
    all) check_all ;;
    range) check_range ;;
esac

if ((failures > 0)); then
    echo "This repository holds recipes only; NVIDIA, Intel and AMD files must never be committed (README, licence rules)." >&2
    exit 1
fi
echo "ok: no vendor or binary files ($mode)"

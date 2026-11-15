#!/usr/bin/env bash
# Prints, writes or checks onnxruntime-split/.SRCINFO, the one the AUR
# shows: the recipe's default build inputs, the same on any host.
#
#   scripts/srcinfo.sh               # prints it
#   scripts/srcinfo.sh --write       # writes onnxruntime-split/.SRCINFO
#   scripts/srcinfo.sh --check       # fails, with a diff, when the
#                                    # committed one is stale
#   scripts/srcinfo.sh --aur DIR     # --check, then copies PKGBUILD and
#                                    # .SRCINFO into DIR, the top of a git
#                                    # clone of the AUR repository (e.g.
#                                    # ssh://aur@aur.archlinux.org/onnxruntime-split.git)
#
# The package names depend on the host when makepkg sources the PKGBUILD:
# on the ORTF_* variables of the environment and on the CUDA major of
# /opt/cuda/include/cuda.h. The .SRCINFO is therefore made from a copy of
# the PKGBUILD (under mktemp) whose CUDA root is a directory that does not
# exist, as on a machine without cuda installed (the names then carry the
# PKGBUILD's _ortf_cuda_major_default), with every ORTF_* variable unset
# (the _ortf_default_* values of the file), by `makepkg --printsrcinfo`.
# --aur commits nothing: DIR's git index is left to the maintainer.
# Exit status: 0 done (or up to date), 1 stale or not a git clone, 2 usage
# or makepkg error.
set -euo pipefail

prog=${0##*/}
top=$(cd "$(dirname "$0")/.." && pwd)
recipe=$top/onnxruntime-split

usage() {
    echo "usage: $prog [--write | --check | --aur DIR]" >&2
    exit 2
}

mode=print aur=''
case ${1:-} in
    '') (($# == 0)) || usage ;;
    --write) (($# == 1)) || usage; mode='write' ;;
    --check) (($# == 1)) || usage; mode=check ;;
    --aur) (($# == 2)) || usage; mode=aur; aur=$2 ;;
    *) usage ;;
esac

command -v makepkg >/dev/null 2>&1 || { echo "$prog: makepkg not found" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# generate <file>: writes the .SRCINFO to <file>.
generate() {
    local home_line='_ortf_cuda_home=/opt/cuda' nocuda=$tmp/no-cuda
    mkdir -p "$tmp/recipe"
    if ! grep -qxF -- "$home_line" "$recipe/PKGBUILD"; then
        echo "$prog: $recipe/PKGBUILD has no line $home_line to point at a missing CUDA" >&2
        return 2
    fi
    sed "s|^$home_line\$|_ortf_cuda_home=$nocuda|" "$recipe/PKGBUILD" >"$tmp/recipe/PKGBUILD"
    if ! (
        cd "$tmp/recipe"
        # Every ORTF_* variable, exported or not ("${!ORTF_@}" lists their
        # names; none, nothing).
        unset "${!ORTF_@}"
        LC_ALL=C makepkg --printsrcinfo >"$1"
    ); then
        echo "$prog: makepkg --printsrcinfo failed" >&2
        return 2
    fi
}

# check: whether the committed .SRCINFO is the generated one; prints a
# diff when it is not.
check() {
    local want=$tmp/SRCINFO
    generate "$want" || return
    if [[ ! -f $recipe/.SRCINFO ]]; then
        echo "$prog: $recipe/.SRCINFO does not exist: run $prog --write" >&2
        return 1
    fi
    if ! diff -u --label "committed onnxruntime-split/.SRCINFO" --label "generated" \
        "$recipe/.SRCINFO" "$want"; then
        echo "$prog: onnxruntime-split/.SRCINFO is stale: run $prog --write" >&2
        return 1
    fi
}

case $mode in
    print)
        generate "$tmp/SRCINFO"
        cat "$tmp/SRCINFO"
        ;;
    write)
        generate "$tmp/SRCINFO"
        cp -- "$tmp/SRCINFO" "$recipe/.SRCINFO.tmp"
        mv -f -- "$recipe/.SRCINFO.tmp" "$recipe/.SRCINFO"
        echo "wrote $recipe/.SRCINFO"
        ;;
    check)
        check
        echo "onnxruntime-split/.SRCINFO is up to date"
        ;;
    aur)
        # The top of a git work tree, not a directory inside one (such as
        # this repository's): an AUR repository has the two files at its
        # top.
        if ! dir=$(cd -- "$aur" 2>/dev/null && pwd -P) ||
            ! git_top=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) ||
            [[ $(cd -- "$git_top" && pwd -P) != "$dir" ]]; then
            echo "$prog: $aur is not the top of a git clone (git clone ssh://aur@aur.archlinux.org/onnxruntime-split.git)" >&2
            exit 1
        fi
        check
        cp -- "$recipe/PKGBUILD" "$recipe/.SRCINFO" "$dir/"
        echo "copied PKGBUILD and .SRCINFO into $dir; review, commit and push there"
        ;;
esac

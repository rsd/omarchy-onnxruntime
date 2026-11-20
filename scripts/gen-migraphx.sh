#!/usr/bin/env bash
# Generates, or checks, onnxruntime-split-migraphx/PKGBUILD, the recipe of
# the MIGraphX line (onnxruntime-provider-migraphx, AMD GPUs, untested on
# AMD hardware, and its Python add-on python-onnxruntime-provider-migraphx),
# from onnxruntime-split/PKGBUILD.
#
#   scripts/gen-migraphx.sh            # writes onnxruntime-split-migraphx/PKGBUILD
#   scripts/gen-migraphx.sh --check    # fails, with a diff, when the
#                                      # committed one is stale
#
# The generated file is the source PKGBUILD with these lines replaced, and
# nothing else changed (the same pkgver, sources, checksums and build):
#   pkgbase=...                    pkgbase=onnxruntime-split-migraphx
#   _ortf_default_eps=...          _ortf_default_eps='migraphx'
#   _ortf_default_python=...       _ortf_default_python=0
#   _ortf_default_nccl=...         _ortf_default_nccl=1 (NCCL, as the
#                                  main line: the provider bridge)
#   _ortf_line=...                 _ortf_line=migraphx
#   pkgrel=...                     pkgrel=$line_pkgrel (below), with a
#                                  comment that it is this line's own
# and a comment after the Maintainer line saying it is generated. The
# line builds the MIGraphX provider alone (prepare() refuses another
# ORTF_EPS, and ORTF_PYTHON=1), with no CUDA, OpenVINO or Python build
# input, against Arch's migraphx and hip-runtime-amd, with NCCL as AMD's
# RCCL (Arch's rccl: a build input only, as the provider links no RCCL);
# its onnxruntime-core pin is the source's (_ortf_core_dep). Its packages
# are the provider and the add-on, which links the provider's library into
# the main line's python-onnxruntime-core (_ortf_pkgnames). It has no -bin
# counterpart (scripts/gen-bin.sh refuses both packages and a migraphx-
# tag).
# The rules (each stamped line exactly once, nothing written otherwise, an
# up-to-date file left untouched) and the exit status are
# scripts/gen-line-lib.sh's, shared with scripts/gen-cuda12.sh.
# After a change, scripts/srcinfo.sh --recipe onnxruntime-split-migraphx
# --write regenerates its .SRCINFO.
# Exit status: 0 done (or up to date), 1 stale, 2 usage error or a source
# line missing or repeated.
set -euo pipefail

line_dir=onnxruntime-split-migraphx
# The migraphx line's own pkgrel: bumped for a rebuild of this line alone.
line_pkgrel=1

# The replacements, name -> the line(s) that replace the source's
# name=... line (pkgrel: gen-line-lib.sh's).
declare -A repl=(
    [pkgbase]="pkgbase=$line_dir"
    [_ortf_default_eps]="_ortf_default_eps='migraphx'"
    [_ortf_default_python]='_ortf_default_python=0'
    [_ortf_default_nccl]='_ortf_default_nccl=1'
    [_ortf_line]='_ortf_line=migraphx'
)

# shellcheck source-path=SCRIPTDIR source=gen-line-lib.sh
. "$(dirname "$0")/gen-line-lib.sh"
gen_line_main "$@"

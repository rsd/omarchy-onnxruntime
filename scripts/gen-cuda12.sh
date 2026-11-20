#!/usr/bin/env bash
# Generates, or checks, onnxruntime-split-cuda12/PKGBUILD, the recipe of
# the CUDA 12 legacy line (onnxruntime-provider-cuda12-legacy), from
# onnxruntime-split/PKGBUILD.
#
#   scripts/gen-cuda12.sh            # writes onnxruntime-split-cuda12/PKGBUILD
#   scripts/gen-cuda12.sh --check    # fails, with a diff, when the
#                                    # committed one is stale
#
# The generated file is the source PKGBUILD with these lines replaced, and
# nothing else changed (the same pkgver, sources, checksums and build):
#   pkgbase=...                    pkgbase=onnxruntime-split-cuda12
#   _ortf_default_eps=...          _ortf_default_eps='cuda'
#   _ortf_default_cuda_archs=...   _ortf_default_cuda_archs=legacy
#   _ortf_default_python=...       _ortf_default_python=0
#   _ortf_default_nccl=...         _ortf_default_nccl=1 (NCCL, as the
#                                  main line: the provider bridge)
#   _ortf_line=...                 _ortf_line=cuda12
#   _ortf_cuda_major_default=...   _ortf_cuda_major_default=12
#   pkgrel=...                     pkgrel=$line_pkgrel (below), with a
#                                  comment that it is this line's own
# and a comment after the Maintainer line saying it is generated. The
# provider's onnxruntime-core pin is the source's (_ortf_core_dep:
# onnxruntime-core>=<pkgver>-<_ortf_core_abi_pkgrel> and
# onnxruntime-core<<next patch release>), from its pkgver, not stamped;
# its NCCL is the AUR's nccl-cuda12.9, a makedepend and the CUDA
# package's depend by name (_ortf_nccl_pkg, from _ortf_line), not stamped.
# The rules (each stamped line exactly once, nothing written otherwise, an
# up-to-date file left untouched) and the exit status are
# scripts/gen-line-lib.sh's, shared with scripts/gen-migraphx.sh.
# After a change, scripts/srcinfo.sh --recipe onnxruntime-split-cuda12
# --write regenerates its .SRCINFO.
# Exit status: 0 done (or up to date), 1 stale, 2 usage error or a source
# line missing or repeated.
set -euo pipefail

line_dir=onnxruntime-split-cuda12
# The cuda12 line's own pkgrel: bumped for a rebuild of this line alone
# (the main line's pkgrel is not this one's). 2: built with the training
# ops, for the main line's onnxruntime-core 1.29.0-7 (the joint release
# rule, onnxruntime-split/PKGBUILD); its CUDA provider links cuDNN
# (_ortf_patch_cudnn_link), a change made before 2 was released, which
# therefore stays 2.
line_pkgrel=2

# The replacements, name -> the line(s) that replace the source's
# name=... line (pkgrel: gen-line-lib.sh's).
declare -A repl=(
    [pkgbase]="pkgbase=$line_dir"
    [_ortf_default_eps]="_ortf_default_eps='cuda'"
    [_ortf_default_cuda_archs]='_ortf_default_cuda_archs=legacy'
    [_ortf_default_python]='_ortf_default_python=0'
    [_ortf_default_nccl]='_ortf_default_nccl=1'
    [_ortf_line]='_ortf_line=cuda12'
    [_ortf_cuda_major_default]='_ortf_cuda_major_default=12'
)

# shellcheck source-path=SCRIPTDIR source=gen-line-lib.sh
. "$(dirname "$0")/gen-line-lib.sh"
gen_line_main "$@"

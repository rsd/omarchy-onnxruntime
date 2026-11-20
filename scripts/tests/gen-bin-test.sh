#!/usr/bin/env bash
# Tests for scripts/gen-bin.sh on fake release assets built under mktemp:
# a staging root per package (a .PKGINFO modelled on the real 1.29.0
# packages, a dummy .BUILDINFO and .MTREE, usr/lib/lib<name>.so.1 and
# usr/share/licenses/<name>/LICENSE), packed with bsdtar --zstd. The
# packages: onnxruntime-core, the CUDA package of the rtx and of the all
# build, the TensorRT, OpenVINO and oneDNN providers and the Python module.
# The cases: the PKGBUILD parses, passes shellcheck (as scripts/test.sh
# checks recipes; skipped without shellcheck) and gives a .SRCINFO with the
# expected names, provides, conflicts, depends, optdepends and checksums;
# the two CUDA -bin packages conflict with each other; SHA256SUMS and the
# printed gh command; then makepkg builds every -bin package from file://
# sources, each holding the asset's usr/ with its licence directory
# renamed, no .PKGINFO but its own, and the expected provides and
# conflicts. Refused: a version other than the tag's, a duplicate pkgname,
# the MIGraphX provider, an unmet onnxruntime-core=1.29.0-4 depend, an
# asset with an .INSTALL script; --pkgrel (bumps only the PKGBUILD's pkgrel, a
# non-integer is exit 2). The CUDA 12 line: a cuda12-v tag and a package of
# pkgbase onnxruntime-split-cuda12 (the legacy provider, with its core pins,
# met by the main line's core: the pin pair onnxruntime-core>=1.29.0-7 and
# onnxruntime-core<1.29.1 of 1.29.0-2) give pkgbase
# onnxruntime-split-cuda12-bin, the cuda12-v tag in the source URLs and the
# provides and conflicts of the -bin convention; refused: an exact core pin
# with a pkgrel, the bare pin onnxruntime-core=1.29.0 of the training-off
# 1.29.0-1 (accepted beside a recipe whose ABI pkgrel is 1), a lower bound
# or a bare pin of a pkgver other than the recipe's, a lower bound of
# another pkgver, without a pkgrel number or below the recipe's
# _ortf_core_abi_pkgrel (one above it passes), an upper bound not above
# pkgver, a set mixing the two pkgbases,
# either pkgbase under the other line's tag, and an unknown pkgbase.
# Needs bsdtar, makepkg (with fakeroot), vercmp and sha256sum; without them
# it says so and skips. makepkg refuses to run as root: then it fails.
# Usage: scripts/tests/gen-bin-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../gen-bin.sh

for t in bsdtar makepkg fakeroot vercmp sha256sum; do
    if ! command -v "$t" >/dev/null 2>&1; then
        echo "skip: $t not found"
        exit 0
    fi
done
if [[ -n ${SHELLCHECK:-} ]]; then
    shellcheck=("$SHELLCHECK")
elif command -v shellcheck >/dev/null 2>&1; then
    shellcheck=(shellcheck)
else
    shellcheck=()
fi
# A shellcheck that does not run (a version-manager shim, say) is none.
if ((${#shellcheck[@]})) && ! "${shellcheck[@]}" --version >/dev/null 2>&1; then
    shellcheck=()
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

srcbase=onnxruntime-split # the pkgbase pack() writes into .PKGINFO
lic='MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MPL-2.0'
# pack <dir> <name> <version> [<extra top-level file>]: builds
# <dir>/<name>-<version>-x86_64.pkg.tar.zst from a staging root holding the
# .PKGINFO read from stdin (after the name, pkgbase, version and arch
# lines), and the extra file (e.g. .INSTALL) when given.
pack() {
    local dir=$1 name=$2 version=$3 extra=${4:-} root
    root=$(mktemp -d "$tmp/root.XXXXXX")
    {
        printf 'pkgname = %s\npkgbase = %s\nxdata = pkgtype=split\n' "$name" "$srcbase"
        printf 'pkgver = %s\narch = x86_64\nlicense = %s\n' "$version" "$lic"
        cat
    } >"$root/.PKGINFO"
    echo 'format = 2' >"$root/.BUILDINFO"
    echo '#mtree' >"$root/.MTREE"
    mkdir -p "$root/usr/lib" "$root/usr/share/licenses/$name"
    echo "fake $name" >"$root/usr/lib/lib$name.so.1"
    echo "MIT licence of $name" >"$root/usr/share/licenses/$name/LICENSE"
    local entries=(.PKGINFO .BUILDINFO .MTREE usr)
    if [[ -n $extra ]]; then
        echo 'post_install() { :; }' >"$root/$extra"
        entries+=("$extra")
    fi
    mkdir -p "$dir"
    (cd "$root" && bsdtar --zstd -cf "$dir/$name-$version-x86_64.pkg.tar.zst" "${entries[@]}")
}

core() { # core <dir> <version>
    pack "$1" onnxruntime-core "$2" <<'EOF'
pkgdesc = Cross-platform inference engine for ONNX models: the C and C++ library with the CPU execution provider
replaces = onnxruntime-full
conflict = onnxruntime
conflict = onnxruntime-full
provides = onnxruntime=1.29.0
provides = libonnxruntime.so=1-64
depend = glibc
depend = libstdc++
optdepend = onnxruntime-provider-cuda: the CUDA execution provider
optdepend = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)
makedepend = cmake
EOF
}
cuda() { # cuda <dir> <name>
    pack "$1" "$2" 1.29.0-5 <<'EOF'
pkgdesc = The CUDA execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)
conflict = onnxruntime-provider-cuda
conflict = onnxruntime-provider-cuda13
provides = onnxruntime-provider-cuda=1.29.0
provides = onnxruntime-provider-cuda13=1.29.0
depend = onnxruntime-core=1.29.0-5
depend = cuda
depend = cuda>=13
depend = cuda<14
EOF
}
tensorrt() { # tensorrt <dir>
    pack "$1" onnxruntime-provider-tensorrt-cuda13 1.29.0-5 <<'EOF'
pkgdesc = The TensorRT execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)
conflict = onnxruntime-provider-tensorrt
provides = onnxruntime-provider-tensorrt=1.29.0
depend = onnxruntime-core=1.29.0-5
depend = onnxruntime-provider-cuda13=1.29.0
depend = tensorrt>=11.4
depend = tensorrt<11.5
EOF
}
openvino() { # openvino <dir> <core version it depends on>
    pack "$1" onnxruntime-provider-openvino 1.29.0-5 <<EOF
pkgdesc = The OpenVINO execution provider of ONNX Runtime, for onnxruntime-core (Intel CPU, GPU and NPU)
depend = onnxruntime-core=$2
depend = openvino>=2026.4
depend = openvino<2026.5
optdepend = intel-npu-driver: Intel NPU user-space driver (OpenVINO NPU device)
EOF
}
dnnl() { # dnnl <dir>: oneDNN linked in statically, so no onednn depend
    pack "$1" onnxruntime-provider-dnnl 1.29.0-5 <<'EOF'
pkgdesc = The oneDNN execution provider of ONNX Runtime, for onnxruntime-core (CPU)
depend = onnxruntime-core=1.29.0-5
depend = glibc
depend = libgomp
EOF
}
pymod() { # pymod <dir>
    pack "$1" python-onnxruntime-core 1.29.0-5 <<'EOF'
pkgdesc = The Python module of ONNX Runtime's inference engine for ONNX models (onnxruntime-core)
replaces = python-onnxruntime-full
conflict = python-onnxruntime
provides = python-onnxruntime=1.29.0
depend = onnxruntime-core=1.29.0-5
depend = python>=3.14
depend = python<3.15
optdepend = python-onnx: the quantization, transformers optimisation and model tools
EOF
}
release() { # release <dir>: the seven packages of a good release
    core "$1" 1.29.0-5
    cuda "$1" onnxruntime-provider-cuda13-rtx
    cuda "$1" onnxruntime-provider-cuda13-all
    tensorrt "$1"
    openvino "$1" 1.29.0-5
    dnnl "$1"
    pymod "$1"
}

# gen <args...>: runs the generator; sets status and out.
gen() {
    status=0
    out=$("$tool" "$@" 2>&1) || status=$?
}
has() { grep -qE -- "$1" <<<"$out"; }

good=$tmp/good
release "$good"
# A debug package is skipped.
cp "$good/onnxruntime-core-1.29.0-5-x86_64.pkg.tar.zst" "$good/onnxruntime-core-debug-1.29.0-5-x86_64.pkg.tar.zst"
aur=$tmp/aur
gen --base-url "file://$good/" --out "$aur" "$good" v1.29.0-5
if ((status == 0)) && [[ -s $aur/PKGBUILD && -s $aur/.SRCINFO ]]; then
    ok "a good release: exit 0, PKGBUILD and .SRCINFO written"
else
    fail "good release (exit $status): $out"
fi
if has 'skipped \(debug package\)'; then ok "a debug package is skipped"; else fail "debug package: $out"; fi
if bash -n "$aur/PKGBUILD" 2>"$tmp/err"; then ok "PKGBUILD: bash -n"; else fail "bash -n: $(cat "$tmp/err")"; fi
if ((${#shellcheck[@]} == 0)); then
    echo "skip: shellcheck of the PKGBUILD (no working shellcheck on PATH, SHELLCHECK unset)"
elif sc=$("${shellcheck[@]}" -s bash -e SC2034,SC2154,SC2164 "$aur/PKGBUILD" 2>&1); then
    ok "PKGBUILD: shellcheck"
else
    fail "shellcheck: $sc"
fi
if grep -qxF '# Maintainer: Raul Dias <raul@dias.com.br>' "$aur/PKGBUILD" &&
    grep -q 'do not edit it by hand' "$aur/PKGBUILD" && grep -q 'release v1.29.0-5' "$aur/PKGBUILD"; then
    ok "PKGBUILD: maintainer and generated-file notice"
else
    fail "PKGBUILD header: $(head -n 8 "$aur/PKGBUILD")"
fi

T=$'\t'
srcinfo=$aur/.SRCINFO
# srcinfo_has <file> <line...>: every line is in the file, exactly.
srcinfo_has() {
    local file=$1 line missing=0
    shift
    for line in "$@"; do
        if ! grep -qxF -- "$line" "$file"; then
            echo "  missing: $line"
            missing=1
        fi
    done
    return "$missing"
}
section() { sed -n "/^pkgname = $1\$/,/^\$/p" "$srcinfo" >"$tmp/$1"; }
check() { # check <description> <file> <line...>
    local what=$1
    shift
    if srcinfo_has "$@"; then ok "$what"; else fail "$what"; fi
}
file=onnxruntime-core-1.29.0-5-x86_64.pkg.tar.zst
sum=$(sha256sum "$good/$file" | cut -d' ' -f1)
check ".SRCINFO: pkgbase, version, options, the core asset" "$srcinfo" \
    'pkgbase = onnxruntime-split-bin' "${T}pkgver = 1.29.0" "${T}pkgrel = 5" "${T}arch = x86_64" \
    "${T}url = https://github.com/rsd/omarchy-onnxruntime" "${T}license = $lic" \
    "${T}options = !strip" "${T}options = !debug" "${T}options = !lto" \
    "${T}source = $file::file://$good/$file" "${T}noextract = $file" "${T}sha256sums = $sum"
names=(onnxruntime-core onnxruntime-provider-cuda13-all onnxruntime-provider-cuda13-rtx
    onnxruntime-provider-dnnl onnxruntime-provider-openvino onnxruntime-provider-tensorrt-cuda13 python-onnxruntime-core)
got=$(sed -n 's/^pkgname = //p' "$srcinfo" | tr '\n' ' ')
if [[ $got == "$(printf '%s-bin ' "${names[@]}")" ]]; then
    ok ".SRCINFO: one -bin package per asset, no debug one"
else
    fail ".SRCINFO package names: $got"
fi
if (($(grep -c "^${T}noextract = " "$srcinfo") == 7 && $(grep -c "^${T}sha256sums = " "$srcinfo") == 7)); then
    ok ".SRCINFO: every asset in noextract and sha256sums"
else
    fail ".SRCINFO noextract/sha256sums: $(grep -E 'noextract|sha256sums' "$srcinfo")"
fi
for n in "${names[@]}"; do section "$n-bin"; done
check ".SRCINFO: onnxruntime-core-bin" "$tmp/onnxruntime-core-bin" \
    "${T}pkgdesc = Cross-platform inference engine for ONNX models: the C and C++ library with the CPU execution provider (prebuilt)" \
    "${T}provides = onnxruntime-core=1.29.0-5" "${T}provides = onnxruntime=1.29.0" "${T}provides = libonnxruntime.so=1-64" \
    "${T}conflicts = onnxruntime-core" "${T}conflicts = onnxruntime" "${T}conflicts = onnxruntime-full" \
    "${T}replaces = onnxruntime-full" "${T}depends = glibc" "${T}depends = libstdc++" \
    "${T}optdepends = onnxruntime-provider-cuda: the CUDA execution provider" \
    "${T}optdepends = onnxruntime-provider-openvino: the OpenVINO execution provider (Intel CPU, GPU and NPU)"
for c in rtx all; do
    check ".SRCINFO: onnxruntime-provider-cuda13-$c-bin" "$tmp/onnxruntime-provider-cuda13-$c-bin" \
        "${T}provides = onnxruntime-provider-cuda13-$c=1.29.0-5" \
        "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-cuda13=1.29.0" \
        "${T}conflicts = onnxruntime-provider-cuda13-$c" \
        "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-cuda13" \
        "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = cuda" "${T}depends = cuda>=13" "${T}depends = cuda<14"
done
# conflicts_with <a> <b>: a conflicts with a name b provides (or is).
conflicts_with() {
    local c p
    while IFS= read -r c; do
        while IFS= read -r p; do
            [[ ${p%%=*} == "$c" ]] && return 0
        done < <(sed -n "s/^${T}provides = //p" "$tmp/$2"; echo "$2")
    done < <(sed -n "s/^${T}conflicts = //p" "$tmp/$1")
    return 1
}
if conflicts_with onnxruntime-provider-cuda13-rtx-bin onnxruntime-provider-cuda13-all-bin &&
    conflicts_with onnxruntime-provider-cuda13-all-bin onnxruntime-provider-cuda13-rtx-bin; then
    ok "the rtx and all CUDA -bin packages conflict with each other"
else
    fail "the two CUDA -bin packages do not conflict"
fi
check ".SRCINFO: onnxruntime-provider-tensorrt-cuda13-bin" "$tmp/onnxruntime-provider-tensorrt-cuda13-bin" \
    "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = onnxruntime-provider-cuda13=1.29.0" \
    "${T}depends = tensorrt>=11.4" "${T}depends = tensorrt<11.5" \
    "${T}provides = onnxruntime-provider-tensorrt-cuda13=1.29.0-5" "${T}provides = onnxruntime-provider-tensorrt=1.29.0" \
    "${T}conflicts = onnxruntime-provider-tensorrt-cuda13" "${T}conflicts = onnxruntime-provider-tensorrt"
check ".SRCINFO: onnxruntime-provider-openvino-bin" "$tmp/onnxruntime-provider-openvino-bin" \
    "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = openvino>=2026.4" "${T}depends = openvino<2026.5" \
    "${T}optdepends = intel-npu-driver: Intel NPU user-space driver (OpenVINO NPU device)" \
    "${T}provides = onnxruntime-provider-openvino=1.29.0-5" "${T}conflicts = onnxruntime-provider-openvino"
check ".SRCINFO: onnxruntime-provider-dnnl-bin" "$tmp/onnxruntime-provider-dnnl-bin" \
    "${T}pkgdesc = The oneDNN execution provider of ONNX Runtime, for onnxruntime-core (CPU) (prebuilt)" \
    "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = libgomp" \
    "${T}provides = onnxruntime-provider-dnnl=1.29.0-5" "${T}conflicts = onnxruntime-provider-dnnl"
if grep -nE "^${T}(depends|provides) = (onednn|libdnnl)" "$tmp/onnxruntime-provider-dnnl-bin"; then
    fail ".SRCINFO: onnxruntime-provider-dnnl-bin names oneDNN (above)"
else
    ok ".SRCINFO: onnxruntime-provider-dnnl-bin depends on no onednn"
fi
check ".SRCINFO: python-onnxruntime-core-bin (an apostrophe in pkgdesc)" "$tmp/python-onnxruntime-core-bin" \
    "${T}pkgdesc = The Python module of ONNX Runtime's inference engine for ONNX models (onnxruntime-core) (prebuilt)" \
    "${T}depends = onnxruntime-core=1.29.0-5" "${T}depends = python>=3.14" "${T}depends = python<3.15" \
    "${T}provides = python-onnxruntime-core=1.29.0-5" "${T}provides = python-onnxruntime=1.29.0" \
    "${T}conflicts = python-onnxruntime-core" "${T}conflicts = python-onnxruntime" \
    "${T}replaces = python-onnxruntime-full" \
    "${T}optdepends = python-onnx: the quantization, transformers optimisation and model tools"
if grep -q makedepends "$srcinfo"; then fail ".SRCINFO: makedepends copied"; else ok ".SRCINFO: no makedepends copied"; fi

expected=$(cd "$good" && for n in "${names[@]}"; do sha256sum "$n-1.29.0-5-x86_64.pkg.tar.zst"; done | LC_ALL=C sort -k 2)
if [[ -f $good/SHA256SUMS && $(cat "$good/SHA256SUMS") == "$expected" ]] && (cd "$good" && sha256sum -c --quiet SHA256SUMS); then
    ok "SHA256SUMS: every asset, sorted, verifies"
else
    fail "SHA256SUMS: $(cat "$good/SHA256SUMS" 2>&1)"
fi
if has "^  gh release create v1\.29\.0-5 .*$good/onnxruntime-core-1\.29\.0-5-x86_64\.pkg\.tar\.zst .*$good/SHA256SUMS\$" &&
    ! has 'gh release create .*debug'; then
    ok "prints the gh release create command"
else
    fail "gh command: $out"
fi

# End to end: makepkg builds every -bin package from the file:// sources.
mk=$tmp/makepkg
mkdir -p "$mk/pkg" "$mk/src" "$mk/build" "$mk/log"
if (cd "$aur" && PKGDEST=$mk/pkg SRCDEST=$mk/src BUILDDIR=$mk/build LOGDEST=$mk/log \
    makepkg -d -f --nosign --noconfirm >"$mk/out" 2>&1); then
    ok "makepkg builds the -bin packages"
else
    fail "makepkg -d -f: $(tail -n 20 "$mk/out")"
fi
for n in "${names[@]}"; do
    built=("$mk/pkg/$n-bin-1.29.0-5-x86_64.pkg.tar"*)
    if ((${#built[@]} != 1)); then
        fail "$n-bin: no built package in $mk/pkg ($(ls "$mk/pkg"))"
        continue
    fi
    list=$(bsdtar -tf "${built[0]}")
    info=$(bsdtar -xOf "${built[0]}" .PKGINFO)
    if grep -qx "usr/lib/lib$n.so.1" <<<"$list" && grep -qx "usr/share/licenses/$n-bin/LICENSE" <<<"$list" &&
        ! grep -q "^usr/share/licenses/$n/" <<<"$list" && (($(grep -c '\.PKGINFO$' <<<"$list") == 1)) &&
        ! grep -qE '^usr/(.*/)?\.(PKGINFO|BUILDINFO|MTREE)$' <<<"$list" &&
        grep -qx "pkgname = $n-bin" <<<"$info" && grep -qx "provides = $n=1.29.0-5" <<<"$info" &&
        grep -qx "conflict = $n" <<<"$info" &&
        [[ $(bsdtar -xOf "${built[0]}" "usr/lib/lib$n.so.1") == "fake $n" ]]; then
        ok "$n-bin: the asset's files, licence in $n-bin, its own .PKGINFO"
    else
        fail "$n-bin: $list"$'\n'"$info"
    fi
done
built=("$mk/pkg/onnxruntime-provider-cuda13-rtx-bin-1.29.0-5-x86_64.pkg.tar"*)
info=$(bsdtar -xOf "${built[0]}" .PKGINFO 2>/dev/null || true)
if grep -qx 'provides = onnxruntime-provider-cuda=1.29.0' <<<"$info" && grep -qx 'conflict = onnxruntime-provider-cuda' <<<"$info" &&
    grep -qx 'depend = cuda>=13' <<<"$info"; then
    ok "onnxruntime-provider-cuda13-rtx-bin: built .PKGINFO keeps provides, conflicts and bounds"
else
    fail "cuda13-rtx-bin .PKGINFO: $info"
fi

# Refusals.
refused() { # refused <description> <regex> <package dir>
    local dest
    dest=$(mktemp -d "$tmp/aur-bad.XXXXXX")
    gen --out "$dest" "$3" v1.29.0-5
    if ((status == 1)) && has "$2" && [[ ! -e $dest/PKGBUILD ]]; then
        ok "$1: exit 1"
    else
        fail "$1 (exit $status): $out"
    fi
}
bad=$tmp/version
release "$bad"
core "$bad" 1.29.0-4
rm "$bad/onnxruntime-core-1.29.0-5-x86_64.pkg.tar.zst"
refused "a version other than the tag's" 'version 1\.29\.0-4, but the tag v1\.29\.0-5 says 1\.29\.0-5' "$bad"
bad=$tmp/duplicate
release "$bad"
# The core package again, under a file name sorting after it.
cp "$bad/onnxruntime-core-1.29.0-5-x86_64.pkg.tar.zst" "$bad/zz-onnxruntime-core-1.29.0-5-x86_64.pkg.tar.zst"
refused "a duplicate pkgname" 'pkgname onnxruntime-core is also that of' "$bad"
bad=$tmp/migraphx
release "$bad"
pack "$bad" onnxruntime-provider-migraphx 1.29.0-5 <<<'depend = onnxruntime-core=1.29.0-5'
refused "the MIGraphX provider" 'onnxruntime-provider-migraphx is not shipped as -bin' "$bad"
# Its Python add-on (the migraphx line's python-onnxruntime-provider-migraphx),
# and its -nccloff copy, by name.
for name in python-onnxruntime-provider-migraphx python-onnxruntime-provider-migraphx-nccloff; do
    bad=$(mktemp -d "$tmp/mgxpy.XXXXXX")
    release "$bad"
    pack "$bad" "$name" 1.29.0-5 <<<'depend = onnxruntime-provider-migraphx=1.29.0-5'
    refused "the MIGraphX Python add-on ($name)" "$name is not shipped as -bin" "$bad"
done
# A build without NCCL (ORTF_NCCL=0): every package named -nccloff, none
# shipped prebuilt.
for name in onnxruntime-core-nccloff onnxruntime-provider-cuda13-rtx-nccloff python-onnxruntime-core-nccloff; do
    bad=$(mktemp -d "$tmp/nccloff.XXXXXX")
    release "$bad"
    pack "$bad" "$name" 1.29.0-5 <<<'depend = onnxruntime-core-nccloff=1.29.0-5'
    refused "a -nccloff package ($name)" "$name is of a build without NCCL \\(ORTF_NCCL=0\\), which is not shipped as -bin" "$bad"
done
bad=$tmp/unmet
release "$bad"
openvino "$bad" 1.29.0-4
refused "an unmet onnxruntime-core=1.29.0-4 depend" 'depend onnxruntime-core=1\.29\.0-4 is met by no -bin package' "$bad"
bad=$tmp/install
release "$bad"
pack "$bad" onnxruntime-provider-openvino 1.29.0-5 .INSTALL <<<'depend = onnxruntime-core=1.29.0-5'
refused "an asset with an .INSTALL script" 'has an \.INSTALL script' "$bad"
# --pkgrel bumps only the generated PKGBUILD's pkgrel.
aur6=$tmp/aur6
gen --pkgrel 6 --base-url "file://$good/" --out "$aur6" "$good" v1.29.0-5
if ((status == 0)) && grep -qxF "${T}pkgrel = 6" "$aur6/.SRCINFO" && ! grep -qxF "${T}pkgrel = 5" "$aur6/.SRCINFO" &&
    grep -qxF "${T}provides = onnxruntime-core=1.29.0-5" "$aur6/.SRCINFO" &&
    ! grep -q 'provides = .*=1\.29\.0-6' "$aur6/.SRCINFO"; then
    ok "--pkgrel 6: .SRCINFO pkgrel 6, provides still =1.29.0-5"
else
    fail "--pkgrel 6 (exit $status): $out"
fi
for bad_rel in x 0 1.5 -1 ''; do
    gen --pkgrel "$bad_rel" --out "$tmp/aur-rel" "$good" v1.29.0-5
    if ((status == 2)) && has 'not a positive integer'; then ok "--pkgrel '$bad_rel': usage error, exit 2"; else fail "--pkgrel '$bad_rel' (exit $status): $out"; fi
done
# The CUDA 12 line: pkgbase onnxruntime-split-cuda12, tag cuda12-v<pkgver>-<pkgrel>.
cuda12() { # cuda12 <dir> <version> <core depend...>: the legacy provider
    local saved=$srcbase dir=$1 version=$2 pins=''
    shift 2
    printf -v pins 'depend = %s\n' "$@"
    srcbase=onnxruntime-split-cuda12
    pack "$dir" onnxruntime-provider-cuda12-legacy "$version" <<EOF
pkgdesc = The CUDA execution provider of ONNX Runtime, for onnxruntime-core (NVIDIA GPUs)
conflict = onnxruntime-provider-cuda
conflict = onnxruntime-provider-cuda12
conflict = onnxruntime-core-nccloff
provides = onnxruntime-provider-cuda=1.29.0
provides = onnxruntime-provider-cuda12=1.29.0
${pins%$'\n'}
depend = cuda>=12
depend = cuda<13
depend = cudnn>=9.10
depend = cudnn<9.11
depend = nvidia-utils
depend = ${CUDA12_NCCL:-nccl-cuda12.9}
depend = glibc
depend = libgcc
depend = libstdc++
EOF
    srcbase=$saved
}
c12=$tmp/cuda12
# 1.29.0-2, built with the training ops: the pin pair on the main line's
# cores from 1.29.0-7 (onnxruntime-split/PKGBUILD's _ortf_core_dep).
cuda12 "$c12" 1.29.0-2 'onnxruntime-core>=1.29.0-7' 'onnxruntime-core<1.29.1'
aur12=$tmp/aur12
gen --base-url "file://$c12/" --out "$aur12" "$c12" cuda12-v1.29.0-2
if ((status == 0)) && [[ -s $aur12/PKGBUILD && -s $aur12/.SRCINFO ]]; then
    ok "cuda12: a good release, exit 0"
else
    fail "cuda12 release (exit $status): $out"
fi
file12=onnxruntime-provider-cuda12-legacy-1.29.0-2-x86_64.pkg.tar.zst
srcinfo=$aur12/.SRCINFO
section onnxruntime-provider-cuda12-legacy-bin
check "cuda12 .SRCINFO: pkgbase, version" "$srcinfo" \
    'pkgbase = onnxruntime-split-cuda12-bin' "${T}pkgver = 1.29.0" "${T}pkgrel = 2"
check "cuda12 .SRCINFO: the source is under the base URL given, the file named as the asset" "$srcinfo" \
    "${T}source = $file12::file://$c12/$file12" "${T}noextract = $file12"
check "cuda12 .SRCINFO: provides, conflicts, the core pins carried through" "$tmp/onnxruntime-provider-cuda12-legacy-bin" \
    "${T}provides = onnxruntime-provider-cuda12-legacy=1.29.0-2" \
    "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-cuda12=1.29.0" \
    "${T}conflicts = onnxruntime-provider-cuda12-legacy" \
    "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-cuda12" \
    "${T}depends = onnxruntime-core>=1.29.0-7" "${T}depends = onnxruntime-core<1.29.1" "${T}depends = cuda>=12" "${T}depends = cuda<13" \
    "${T}depends = cudnn>=9.10" "${T}depends = cudnn<9.11" "${T}depends = nvidia-utils" \
    "${T}depends = nccl-cuda12.9" "${T}conflicts = onnxruntime-core-nccloff"
# The asset's own pkgdesc (here the released 1.29.0-1's older text) gives way to
# the line's, onnxruntime-split-cuda12's .SRCINFO's, in the package and in
# the pkgbase (namcap's "Missing description in PKGBUILD").
desc12=$(sed -n '/^pkgbase = /,/^$/s/^\tpkgdesc = //p' "$here/../../onnxruntime-split-cuda12/.SRCINFO")
if [[ -n $desc12 && $(grep -c "^${T}pkgdesc = " "$srcinfo") == 2 ]] &&
    [[ $(sed -n '/^pkgbase = /,/^$/s/^\tpkgdesc = //p' "$srcinfo") == "$desc12 (prebuilt)" &&
    $(sed -n 's/^\tpkgdesc = //p' "$tmp/onnxruntime-provider-cuda12-legacy-bin") == "$desc12 (prebuilt)" ]] &&
    ! grep -qF 'for onnxruntime-core (NVIDIA GPUs)' "$aur12/PKGBUILD"; then
    ok "cuda12: the pkgbase and the package have onnxruntime-split-cuda12's pkgdesc (prebuilt), not the asset's"
else
    fail "cuda12 pkgdesc (want '$desc12 (prebuilt)'): $(grep -F pkgdesc "$srcinfo")"
fi
if [[ $(sed -n 's/^pkgname = //p' "$srcinfo") == onnxruntime-provider-cuda12-legacy-bin ]] &&
    grep -q 'built by onnxruntime-split-cuda12)' "$aur12/PKGBUILD" &&
    grep -q 'release cuda12-v1.29.0-2' "$aur12/PKGBUILD"; then
    ok "cuda12 PKGBUILD: the one -bin package, header names pkgbase and tag"
else
    fail "cuda12 PKGBUILD: $(head -n 8 "$aur12/PKGBUILD")"
fi
if has "^  gh release create cuda12-v1\.29\.0-2 .*$c12/$file12 "; then
    ok "cuda12: gh release create with the cuda12-v tag"
else
    fail "cuda12 gh command: $out"
fi
# A cuda12 asset on the bare nccl (Arch's CUDA 13 build) is refused.
for bare in nccl 'nccl>=2.30'; do
    bad=$(mktemp -d "$tmp/cuda12-nccl.XXXXXX")
    CUDA12_NCCL=$bare cuda12 "$bad" 1.29.0-2 'onnxruntime-core>=1.29.0-7' 'onnxruntime-core<1.29.1'
    gen --out "$tmp/aur12-nccl" "$bad" cuda12-v1.29.0-2
    if ((status == 1)) && has "depend $bare: the cuda12 line's NCCL is nccl-cuda12.9 by name"; then
        ok "cuda12: a depend on the bare $bare is refused, exit 1"
    else
        fail "cuda12 with $bare (exit $status): $out"
    fi
done
# Without --base-url the assets come from the cuda12-v tag, and without --out
# the pkgbase's own aur/ directory is written (not run here: it is the AUR
# clone), so read both from the generated PKGBUILD / the script.
gen --out "$tmp/aur12-url" "$c12" cuda12-v1.29.0-2
if grep -qF "::https://github.com/rsd/omarchy-onnxruntime/releases/download/cuda12-v1.29.0-2/$file12" "$tmp/aur12-url/PKGBUILD"; then
    ok "cuda12: the default source URL is under the cuda12-v1.29.0-2 tag"
else
    fail "cuda12 default URL: $(grep -F source "$tmp/aur12-url/.SRCINFO")"
fi
# shellcheck disable=SC2016 # the script's own text, not an expansion
if grep -qF 'out=${out:-$top/aur/$srcbase-bin}' "$tool"; then
    ok "cuda12: the default out directory is aur/<pkgbase>-bin (onnxruntime-split-cuda12-bin)"
else
    fail "the default out directory is not derived from the pkgbase"
fi
refused_tag() { # refused_tag <description> <regex> <package dir> <tag>
    local dest
    dest=$(mktemp -d "$tmp/aur-bad.XXXXXX")
    gen --out "$dest" "$3" "$4"
    if ((status == 1)) && has "$2" && [[ ! -e $dest/PKGBUILD ]]; then
        ok "$1: exit 1"
    else
        fail "$1 (exit $status): $out"
    fi
}
bad=$tmp/c12-pin
cuda12 "$bad" 1.29.0-1 onnxruntime-core=1.29.0-1
refused_tag "cuda12: a core pin with a pkgrel" 'depend onnxruntime-core=1\.29\.0-1 is met by no -bin package' "$bad" cuda12-v1.29.0-1
# The bare pin onnxruntime-core=<pkgver> (the released cuda12-v1.29.0-1,
# built without the training ops) is met by every core pkgrel of pkgver:
# refused while the recipe's ABI pkgrel is above 1, so that it is never
# taken for a fit of the cores from 1.29.0-7.
bad=$(mktemp -d "$tmp/c12-bare.XXXXXX")
cuda12 "$bad" 1.29.0-1 onnxruntime-core=1.29.0
refused_tag "cuda12: the bare pin onnxruntime-core=1.29.0 (cuda12-v1.29.0-1, training off)" \
    'depend onnxruntime-core=1\.29\.0: every pkgrel of 1\.29\.0 meets it, but the core.s ABI pkgrel is [0-9]+' "$bad" cuda12-v1.29.0-1
# Accepted only where the ABI pkgrel is 1 (every pkgrel of pkgver one
# bridge): gen-bin.sh beside a recipe that says so.
tree=$tmp/tree-abi1
mkdir -p "$tree/scripts" "$tree/onnxruntime-split"
cp "$tool" "$tree/scripts/gen-bin.sh"
printf 'pkgver=1.29.0\n_ortf_core_abi_pkgrel=1\n' >"$tree/onnxruntime-split/PKGBUILD"
saved_tool=$tool tool=$tree/scripts/gen-bin.sh
gen --base-url "file://$bad/" --out "$tmp/aur12-abi1" "$bad" cuda12-v1.29.0-1
tool=$saved_tool
if ((status == 0)) && grep -qxF "${T}depends = onnxruntime-core=1.29.0" "$tmp/aur12-abi1/.SRCINFO"; then
    ok "cuda12: the bare pin is accepted where the recipe's ABI pkgrel is 1"
else
    fail "cuda12: the bare pin with ABI pkgrel 1 (exit $status): $out"
fi
# A pin on a core of another pkgver than the recipe's: its ABI pkgrel is
# unknown, for the lower bound and the bare pin alike.
bad=$(mktemp -d "$tmp/c12-otherver.XXXXXX")
cuda12 "$bad" 1.28.0-2 'onnxruntime-core>=1.28.0-7' 'onnxruntime-core<1.28.1'
refused_tag "cuda12: a lower bound on a core of pkgver 1.28.0, not the recipe's" \
    "depend onnxruntime-core>=1\.28\.0-7: the recipe \(onnxruntime-split/PKGBUILD\) is pkgver 1\.29\.0, so the ABI pkgrel of 1\.28\.0's core is unknown" \
    "$bad" cuda12-v1.28.0-2
bad=$(mktemp -d "$tmp/c12-otherver.XXXXXX")
cuda12 "$bad" 1.28.0-1 onnxruntime-core=1.28.0
refused_tag "cuda12: the bare pin on a core of pkgver 1.28.0, not the recipe's" \
    "depend onnxruntime-core=1\.28\.0: the recipe \(onnxruntime-split/PKGBUILD\) is pkgver 1\.29\.0, so the ABI pkgrel of 1\.28\.0's core is unknown" \
    "$bad" cuda12-v1.28.0-1
# 1.29.0-2, built with the training ops: the pin pair on the main line's
# cores from 1.29.0-7 (onnxruntime-split/PKGBUILD's _ortf_core_dep).
c12p=$tmp/cuda12-pair
cuda12 "$c12p" 1.29.0-2 'onnxruntime-core>=1.29.0-7' 'onnxruntime-core<1.29.1'
aur12p=$tmp/aur12-pair
gen --base-url "file://$c12p/" --out "$aur12p" "$c12p" cuda12-v1.29.0-2
srcinfo=$aur12p/.SRCINFO
if ((status == 0)) && srcinfo_has "$srcinfo" "${T}pkgrel = 2" \
    "${T}depends = onnxruntime-core>=1.29.0-7" "${T}depends = onnxruntime-core<1.29.1" \
    "${T}provides = onnxruntime-provider-cuda12-legacy=1.29.0-2"; then
    ok "cuda12 1.29.0-2: the pin pair on the main line's core is accepted and carried through"
else
    fail "cuda12 pin pair (exit $status): $out"
fi
# The pair's exemption is for this pkgver's core only.
for pins in 'onnxruntime-core>=1.28.0-7|onnxruntime-core<1.29.1' 'onnxruntime-core>=1.29.0-x|onnxruntime-core<1.29.1' \
    'onnxruntime-core>=1.29.0-7|onnxruntime-core<1.29.0' 'onnxruntime-core>=1.29.0-7|onnxruntime-core<1.28'; do
    bad=$(mktemp -d "$tmp/c12-badpair.XXXXXX")
    IFS='|' read -r lo hi <<<"$pins"
    cuda12 "$bad" 1.29.0-2 "$lo" "$hi"
    refused_tag "cuda12: the pins [$lo $hi]" 'is met by no -bin package' "$bad" cuda12-v1.29.0-2
done
# The pair comes whole, and its upper bound is exclusive: one bound alone
# admits the cores of another provider bridge, and <=V would be read as V.
bad=$(mktemp -d "$tmp/c12-hionly.XXXXXX")
cuda12 "$bad" 1.29.0-2 'onnxruntime-core<1.29.1'
refused_tag "cuda12: the upper bound alone" 'depends on onnxruntime-core with an upper bound but no lower one' "$bad" cuda12-v1.29.0-2
bad=$(mktemp -d "$tmp/c12-loonly.XXXXXX")
cuda12 "$bad" 1.29.0-2 'onnxruntime-core>=1.29.0-7'
refused_tag "cuda12: the lower bound alone" 'depends on onnxruntime-core with a lower bound but no upper one' "$bad" cuda12-v1.29.0-2
for hi in 'onnxruntime-core<=1.29.1' 'onnxruntime-core<=1.29.0'; do
    bad=$(mktemp -d "$tmp/c12-le.XXXXXX")
    cuda12 "$bad" 1.29.0-2 'onnxruntime-core>=1.29.0-7' "$hi"
    refused_tag "cuda12: an inclusive upper bound ($hi)" "depend ${hi//./\\.}: the upper bound of the pin pair is exclusive" \
        "$bad" cuda12-v1.29.0-2
done
bad=$(mktemp -d "$tmp/c12-leonly.XXXXXX")
cuda12 "$bad" 1.29.0-2 'onnxruntime-core<=1.29.1'
refused_tag "cuda12: an inclusive upper bound alone" 'the upper bound of the pin pair is exclusive' "$bad" cuda12-v1.29.0-2
# The lower bound's pkgrel: at least the recipe's _ortf_core_abi_pkgrel
# (read from onnxruntime-split/PKGBUILD, not repeated in gen-bin.sh); a
# lower one admits training-off cores of another provider bridge.
abi=$(sed -n 's/^_ortf_core_abi_pkgrel=//p' "$here/../../onnxruntime-split/PKGBUILD")
if [[ $abi =~ ^[1-9][0-9]*$ ]] && ! grep -qE '^[^#]*abi_rel=[0-9]' "$tool"; then
    ok "cuda12: gen-bin.sh reads the ABI pkgrel ($abi) from the recipe, no number of its own"
else
    fail "the ABI pkgrel: recipe '$abi'; gen-bin.sh: $(grep -n 'abi_rel=' "$tool")"
fi
bad=$(mktemp -d "$tmp/c12-lowabi.XXXXXX")
cuda12 "$bad" 1.29.0-2 "onnxruntime-core>=1.29.0-$((abi - 1))" 'onnxruntime-core<1.29.1'
refused_tag "cuda12: a lower bound below the ABI pkgrel ($((abi - 1)))" "below the core's ABI pkgrel $abi" "$bad" cuda12-v1.29.0-2
hi12=$(mktemp -d "$tmp/c12-hiabi.XXXXXX")
cuda12 "$hi12" 1.29.0-2 "onnxruntime-core>=1.29.0-$((abi + 1))" 'onnxruntime-core<1.29.1'
gen --out "$(mktemp -d "$tmp/aur12-hi.XXXXXX")" "$hi12" cuda12-v1.29.0-2
if ((status == 0)); then ok "cuda12: a lower bound above the ABI pkgrel ($((abi + 1))) is accepted"; else
    fail "cuda12: a lower bound of $((abi + 1)) (exit $status): $out"; fi
bad=$tmp/c12-mixed
cuda12 "$bad" 1.29.0-1 onnxruntime-core=1.29.0
core "$bad" 1.29.0-1
refused_tag "a set mixing the two pkgbases" "pkgbase is 'onnxruntime-split', not onnxruntime-split-cuda12" "$bad" cuda12-v1.29.0-1
refused_tag "the cuda12 package under a v tag" "pkgbase is 'onnxruntime-split-cuda12', not onnxruntime-split" "$c12" v1.29.0-2
bad=$tmp/c12-unknown
srcbase=onnxruntime-other
pack "$bad" onnxruntime-provider-cuda12-legacy 1.29.0-1 <<<'depend = onnxruntime-core=1.29.0'
srcbase=onnxruntime-split
refused_tag "an unknown pkgbase" "pkgbase is 'onnxruntime-other', not onnxruntime-split-cuda12" "$bad" cuda12-v1.29.0-1
gen "$good" 1.29.0-5
if ((status == 2)) && has 'not a tag'; then ok "a tag without v: exit 2"; else fail "bad tag (exit $status): $out"; fi
# The MIGraphX line is source-only: a migraphx- tag is refused, nothing written.
bad=$(mktemp -d "$tmp/mgxtag.XXXXXX")
srcbase=onnxruntime-split-migraphx
pack "$bad" onnxruntime-provider-migraphx 1.29.0-1 <<<$'depend = onnxruntime-core>=1.29.0-7\ndepend = onnxruntime-core<1.29.1'
pack "$bad" python-onnxruntime-provider-migraphx 1.29.0-1 <<<'depend = onnxruntime-provider-migraphx=1.29.0-1'
srcbase=onnxruntime-split
dest=$(mktemp -d "$tmp/aur-mgx.XXXXXX")
gen --out "$dest" "$bad" migraphx-v1.29.0-1
if ((status == 2)) && has 'the MIGraphX line \(onnxruntime-split-migraphx\) is not shipped as -bin' && [[ ! -e $dest/PKGBUILD ]]; then
    ok "a migraphx- tag (the MIGraphX line, source-only): exit 2, nothing written"
else
    fail "a migraphx- tag (exit $status): $out"
fi
# Its packages under a v tag: refused by pkgbase.
refused_tag "the MIGraphX line's packages under a v tag" "pkgbase is 'onnxruntime-split-migraphx', not onnxruntime-split" "$bad" v1.29.0-1

echo "$((count - fails))/$count passed"
((fails == 0))

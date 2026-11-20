#!/usr/bin/env bash
# Generates the PKGBUILD and .SRCINFO of onnxruntime-split-bin, the prebuilt
# counterpart of onnxruntime-split, from the packages of a release; or, for
# a cuda12- tag, of onnxruntime-split-cuda12-bin, the counterpart of
# onnxruntime-split-cuda12 (the CUDA 12 legacy provider).
#
#   scripts/gen-bin.sh [--base-url URL] [--out DIR] [--pkgrel N] <package dir> <tag>
#   e.g. scripts/gen-bin.sh ~/release v1.29.0-7
#        scripts/gen-bin.sh ~/release-cuda12 cuda12-v1.29.0-2
#
# <package dir> holds the release assets: every *.pkg.tar.zst in it (the
# packages of the rtx build plus the CUDA package of the `all` build, all
# from pkgbase onnxruntime-split; *-debug-* packages are skipped). <tag> is
# the release tag, v<pkgver>-<pkgrel>, or cuda12-v<pkgver>-<pkgrel> for the
# CUDA 12 line (its own tag scheme; its assets are all from pkgbase
# onnxruntime-split-cuda12, and a set mixing the two pkgbases is refused).
# One -bin pkgbase per source pkgbase. --base-url is where the assets are
# downloaded from (default
# https://github.com/rsd/omarchy-onnxruntime/releases/download/<tag>);
# --out is the directory written (default aur/onnxruntime-split-bin at the
# top of this repository, the AUR git clone; aur/onnxruntime-split-cuda12-bin
# for a cuda12- tag). --pkgrel N (a positive
# integer; default the tag's pkgrel) sets only the generated PKGBUILD's
# pkgrel, to bump a fix to that file alone without a new release: the
# assets' version (the tag's pkgver-pkgrel) still drives the version check,
# the <name>=<pkgver>-<pkgrel> provides and the dependency check.
#
# Each asset's .PKGINFO (bsdtar) must name pkgbase onnxruntime-split (or
# onnxruntime-split-cuda12 for a cuda12- tag; no other), the
# tag's pkgver-pkgrel, arch x86_64 and a pkgname no other asset has, and
# the file name must be <pkgname>-<pkgver>-<pkgrel>-x86_64.pkg.tar.zst. An
# asset holding anything but .PKGINFO, .BUILDINFO, .MTREE and usr/ (an
# .INSTALL script, say), or with backup files, is refused: the -bin
# package would silently lose it. So is onnxruntime-provider-migraphx: it
# never ran on AMD hardware and is not shipped prebuilt (its line,
# onnxruntime-split-migraphx, is source-only), with its Python add-on,
# python-onnxruntime-provider-migraphx; a migraphx- tag, that line's, is
# refused as a usage error (exit 2); and so is every -nccloff
# package (a build without NCCL, ORTF_NCCL=0: local builds only). Every other package of a
# release is taken as it is, whatever its name: onnxruntime-provider-dnnl
# (oneDNN, linked in statically) gets its onnxruntime-provider-dnnl-bin by
# the same convention as the others.
#
# For every asset <name>, package <name>-bin installs the asset's usr/
# unchanged (its licence directory renamed to usr/share/licenses/<name>-bin)
# with the asset's pkgdesc (" (prebuilt)" appended), depends, optdepends and replaces as they are
# (bounds included; for a cuda12- tag, the pkgdesc is cuda12_pkgdesc
# below instead, also the pkgbase's), provides <name>=<pkgver>-<pkgrel> plus the asset's
# provides, and conflicts with <name> plus the asset's conflicts: the -bin
# convention of onnxruntime-split/PKGBUILD. Every depend on an
# onnxruntime-* or python-onnxruntime* name must be met by what the -bin
# packages provide (as pacman compares versions); the generator fails
# naming the one that is not, except, for a cuda12- tag, the pins on the
# main line's core, which the main line's onnxruntime-core or
# onnxruntime-core-bin meets (the cuda12 line has no core): the pair
# onnxruntime-core>=<pkgver>-<N> and onnxruntime-core<<V>, V a version
# above pkgver (since 1.29.0-2, built with the training ops for the cores
# from 1.29.0-7), or onnxruntime-core=<pkgver>. N must be at least
# _ortf_core_abi_pkgrel of onnxruntime-split/PKGBUILD, read from the
# recipe, whose pkgver must be the tag's: a lower N admits cores of another
# provider bridge (exit 1). The bare onnxruntime-core=<pkgver>, which every
# core pkgrel of pkgver meets, is accepted only when that ABI pkgrel is 1:
# so cuda12-v1.29.0-1 (training off, bare pin) is refused, as it does not
# fit the cores from 1.29.0-7. The pair comes whole: a lower bound without
# the upper one, or the reverse, is refused, as is an inclusive upper
# bound (onnxruntime-core<=V, which pins a core to V itself) (exit 1).
# For a cuda12- tag, a depend on the bare nccl is refused too (exit 1):
# that line's NCCL is the AUR's nccl-cuda12.9, by name, which the -bin
# package carries over as it is.
# The licence is the assets' (per package when
# they differ). noextract lists every asset: extracted, they would all land
# in the same $srcdir and their usr/ trees would merge.
#
# Writes <out>/PKGBUILD and <out>/.SRCINFO (makepkg --printsrcinfo in
# <out>), and <package dir>/SHA256SUMS (sha256sum format, sorted by file
# name); prints the `gh release create` command that publishes the assets
# (it does not run it). Nothing else is written: scratch files go to mktemp.
# Exit status: 0 on success, 1 when an asset is refused, 2 on a usage error
# or a missing tool.
set -euo pipefail
shopt -s nullglob

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; }
die() { # die <status> <message>
    echo "gen-bin: $2" >&2
    exit "$1"
}

top=$(cd "$(dirname "$0")/.." && pwd)
base_url='' out='' bin_rel='' rel_given=0
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --base-url) (($# >= 2)) || die 2 "--base-url needs a URL"; base_url=$2; shift 2 ;;
        --base-url=*) base_url=${1#*=}; shift ;;
        --out) (($# >= 2)) || die 2 "--out needs a directory"; out=$2; shift 2 ;;
        --out=*) out=${1#*=}; shift ;;
        --pkgrel) (($# >= 2)) || die 2 "--pkgrel needs a number"; bin_rel=$2 rel_given=1; shift 2 ;;
        --pkgrel=*) bin_rel=${1#*=} rel_given=1; shift ;;
        --) shift; break ;;
        -*) die 2 "unknown option: $1 (see --help)" ;;
        *) break ;;
    esac
done
if (($# != 2)); then
    usage >&2
    exit 2
fi
pkgdir_in=$1 tag=$2
[[ -d $pkgdir_in ]] || die 2 "$pkgdir_in: not a directory"
pkgdir_in=$(cd "$pkgdir_in" && pwd)
# The MIGraphX line (pkgbase onnxruntime-split-migraphx) is source-only:
# no tag of its own, no -bin pkgbase.
if [[ $tag == migraphx-* ]]; then
    die 2 "$tag: the MIGraphX line (onnxruntime-split-migraphx) is not shipped as -bin (it never ran on AMD hardware): no migraphx- tag"
fi
if [[ ! $tag =~ ^(cuda12-)?v([^-]+)-([0-9]+(\.[0-9]+)?)$ ]]; then
    die 2 "$tag: not a tag v<pkgver>-<pkgrel> or cuda12-v<pkgver>-<pkgrel> (e.g. v1.29.0-6)"
fi
# The tag scheme names the line, and with it the source pkgbase.
srcbase=onnxruntime-split
[[ -z ${BASH_REMATCH[1]} ]] || srcbase=onnxruntime-split-cuda12
# The CUDA 12 line's pkgdesc, onnxruntime-split-cuda12's (its .SRCINFO;
# scripts/tests/gen-bin-test.sh checks they agree). Its released 1.29.0-1
# asset carries an older text, which cannot change: its -bin package, and
# the -bin pkgbase (namcap wants a pkgdesc there), take this one instead.
cuda12_pkgdesc='The CUDA execution provider of ONNX Runtime, CUDA 12 build for Pascal, Volta and newer NVIDIA GPUs, for onnxruntime-core'
pkgver=${BASH_REMATCH[2]} pkgrel=${BASH_REMATCH[3]}
version=$pkgver-$pkgrel
if ((rel_given)); then
    [[ $bin_rel =~ ^[1-9][0-9]*$ ]] || die 2 "--pkgrel $bin_rel: not a positive integer"
else
    bin_rel=$pkgrel
fi
base_url=${base_url:-https://github.com/rsd/omarchy-onnxruntime/releases/download/$tag}
base_url=${base_url%/}
out=${out:-$top/aur/$srcbase-bin}

for tool in bsdtar sha256sum makepkg vercmp; do
    command -v "$tool" >/dev/null 2>&1 || die 2 "$tool not found"
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The assets, by index: file name, pkgname, pkgdesc, and the .PKGINFO
# lists (newline-separated): license, depend, optdepend, provides,
# conflict, replaces.
declare -a files=() names=() descs=() licenses=() deps=() optdeps=() provs=() confs=() repls=()
declare -A seen=()
# field <.PKGINFO> <key>: every value of <key>, one per line.
field() { sed -n "s/^$2 = //p" "$1"; }
for path in "$pkgdir_in"/*.pkg.tar.zst; do
    file=${path##*/}
    if [[ $file == *-debug-* ]]; then
        echo "gen-bin: $file: skipped (debug package)" >&2
        continue
    fi
    info=$tmp/PKGINFO
    if ! bsdtar -xOf "$path" .PKGINFO >"$info" 2>"$tmp/err" || [[ ! -s $info ]]; then
        die 1 "$file: not a package file (no .PKGINFO: $(head -n 1 "$tmp/err"))"
    fi
    name=$(field "$info" pkgname | head -n 1)
    [[ -n $name ]] || die 1 "$file: no pkgname in its .PKGINFO"
    base=$(field "$info" pkgbase | head -n 1)
    [[ $base == "$srcbase" ]] || die 1 "$file: pkgbase is '$base', not $srcbase (the tag $tag takes only $srcbase packages)"
    if [[ $name == onnxruntime-provider-migraphx* || $name == python-onnxruntime-provider-migraphx* ]]; then
        die 1 "$file: $name is not shipped as -bin (the MIGraphX provider never ran on AMD hardware); remove it from $pkgdir_in"
    fi
    # A build without NCCL (ORTF_NCCL=0): a local build's, never released.
    if [[ $name == *-nccloff ]]; then
        die 1 "$file: $name is of a build without NCCL (ORTF_NCCL=0), which is not shipped as -bin; remove it from $pkgdir_in"
    fi
    v=$(field "$info" pkgver | head -n 1)
    [[ $v == "$version" ]] || die 1 "$file: version $v, but the tag $tag says $version"
    a=$(field "$info" arch | head -n 1)
    [[ $a == x86_64 ]] || die 1 "$file: arch $a, not x86_64"
    [[ -z ${seen[$name]:-} ]] || die 1 "$file: pkgname $name is also that of ${seen[$name]}"
    [[ $file == "$name-$version-x86_64.pkg.tar.zst" ]] ||
        die 1 "$file: the file name is not $name-$version-x86_64.pkg.tar.zst"
    seen[$name]=$file
    [[ -z $(field "$info" backup) ]] || die 1 "$file: has backup files, which the -bin package would not keep (not supported)"
    if ! bsdtar -tf "$path" >"$tmp/list" 2>"$tmp/err"; then
        die 1 "$file: cannot list it: $(head -n 1 "$tmp/err")"
    fi
    while IFS= read -r entry; do
        case $entry in
            .PKGINFO | .BUILDINFO | .MTREE | usr) ;;
            .INSTALL) die 1 "$file: has an .INSTALL script, which the -bin package would not run (not supported)" ;;
            *) die 1 "$file: unexpected top-level entry '$entry' (only .PKGINFO, .BUILDINFO, .MTREE and usr/)" ;;
        esac
    done < <(sed 's|^\./||; s|/.*||' "$tmp/list" | sort -u)
    files+=("$file") names+=("$name")
    if [[ $srcbase == onnxruntime-split-cuda12 ]]; then
        descs+=("$cuda12_pkgdesc (prebuilt)")
    else
        descs+=("$(field "$info" pkgdesc | head -n 1) (prebuilt)")
    fi
    licenses+=("$(field "$info" license)")
    deps+=("$(field "$info" depend)")
    optdeps+=("$(field "$info" optdepend)")
    provs+=("$name=$version"$'\n'"$(field "$info" provides)")
    confs+=("$name"$'\n'"$(field "$info" conflict)")
    repls+=("$(field "$info" replaces)")
done
((${#files[@]} > 0)) || die 1 "no *.pkg.tar.zst in $pkgdir_in"
n=${#files[@]}

# satisfies <dependency> <provision>: whether a provision <name>[=<version>]
# meets <name>[<op><version>], as pacman compares (scripts/check-deps.sh).
# A provision without a version meets only a dependency without one.
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
# The first pkgrel of the main line's onnxruntime-core that the other lines'
# providers are built for: _ortf_core_abi_pkgrel of onnxruntime-split/PKGBUILD,
# read from the recipe (not repeated here), and the recipe's pkgver it
# belongs to.
recipe=$top/onnxruntime-split/PKGBUILD
abi_rel=$(sed -n 's/^_ortf_core_abi_pkgrel=\([1-9][0-9]*\)$/\1/p' "$recipe" 2>/dev/null || true)
abi_ver=$(sed -n 's/^pkgver=\([^[:space:]]*\)$/\1/p' "$recipe" 2>/dev/null || true)
if [[ $srcbase == onnxruntime-split-cuda12 && ( -z $abi_rel || -z $abi_ver ) ]]; then
    die 2 "$recipe: no _ortf_core_abi_pkgrel=<N> or pkgver= line (the lowest core pkgrel a cuda12 pin may name)"
fi
# core_pin <dependency>: whether it is one of the cuda12 line's pins on the
# main line's onnxruntime-core of this pkgver: onnxruntime-core>=<pkgver>-<N>
# (N a pkgrel, at least the recipe's _ortf_core_abi_pkgrel: a lower one
# would admit cores of another provider bridge), onnxruntime-core<<V>, V
# above pkgver as pacman compares (1.29.1 for 1.29.0), or the bare
# onnxruntime-core=<pkgver>, which every pkgrel of pkgver meets: only when
# the ABI pkgrel is 1, so that all of them share one provider bridge (not
# for 1.29.0, whose cores from 7 have the training ops: the training-off
# cuda12-v1.29.0-1 is refused). Both rules need the pkgver to be the
# recipe's, whose ABI pkgrel is the only one known.
core_pin() {
    local n
    case $1 in
        "onnxruntime-core=$pkgver")
            [[ $pkgver == "$abi_ver" ]] ||
                die 1 "depend $1: the recipe (onnxruntime-split/PKGBUILD) is pkgver $abi_ver, so the ABI pkgrel of $pkgver's core is unknown"
            [[ $abi_rel == 1 ]] ||
                die 1 "depend $1: every pkgrel of $pkgver meets it, but the core's ABI pkgrel is $abi_rel (_ortf_core_abi_pkgrel, onnxruntime-split/PKGBUILD): it admits cores of another provider bridge"
            ;;
        "onnxruntime-core>=$pkgver-"*)
            n=${1#"onnxruntime-core>=$pkgver-"}
            [[ $n =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || return 1
            [[ $pkgver == "$abi_ver" ]] ||
                die 1 "depend $1: the recipe (onnxruntime-split/PKGBUILD) is pkgver $abi_ver, so the ABI pkgrel of $pkgver's core is unknown"
            (($(vercmp "$n" "$abi_rel") >= 0)) ||
                die 1 "depend $1: below the core's ABI pkgrel $abi_rel (_ortf_core_abi_pkgrel, onnxruntime-split/PKGBUILD): it admits cores of another provider bridge"
            ;;
        # Refused by pin_pair; never taken for the exclusive bound.
        'onnxruntime-core<='*) return 1 ;;
        'onnxruntime-core<'*) (($(vercmp "${1#onnxruntime-core<}" "$pkgver") > 0)) ;;
        *) return 1 ;;
    esac
}
# pin_pair <file> <depends, newline-separated>: the cuda12 line's pins on
# the core come as the whole pair, onnxruntime-core>=... with
# onnxruntime-core<...: either alone admits cores of another pkgver's
# provider bridge (a lower bound alone, every later release; an upper
# bound alone, every earlier pkgrel). The upper bound is exclusive:
# onnxruntime-core<=V is refused.
pin_pair() {
    local lo=0 hi=0 d
    while IFS= read -r d; do
        case $d in
            'onnxruntime-core>='*) lo=1 ;;
            'onnxruntime-core<='*)
                die 1 "$1: depend $d: the upper bound of the pin pair is exclusive, onnxruntime-core<<next release> (_ortf_core_next, onnxruntime-split/PKGBUILD)"
                ;;
            'onnxruntime-core<'*) hi=1 ;;
        esac
    done <<<"$2"
    local what='an upper bound but no lower one'
    ((hi)) || what='a lower bound but no upper one'
    ((lo == hi)) ||
        die 1 "$1: depends on onnxruntime-core with $what: the pin is the pair onnxruntime-core>=<pkgver>-<N> and onnxruntime-core<<next release>"
}
# What the -bin packages provide: their own names and their provides.
offered=()
for ((i = 0; i < n; i++)); do
    offered+=("${names[i]}-bin=$version")
    mapfile -t list < <(grep . <<<"${provs[i]}")
    offered+=("${list[@]}")
done
for ((i = 0; i < n; i++)); do
    mapfile -t list < <(grep . <<<"${deps[i]}" || true)
    [[ $srcbase != onnxruntime-split-cuda12 ]] || pin_pair "${files[i]}" "${deps[i]}"
    for d in "${list[@]}"; do
        # The CUDA 12 line's NCCL is nccl-cuda12.9 by name: the bare nccl
        # would let pacman install Arch's CUDA 13 build.
        if [[ $srcbase == onnxruntime-split-cuda12 && ${d%%[<>=]*} == nccl ]]; then
            die 1 "${files[i]}: depend $d: the cuda12 line's NCCL is nccl-cuda12.9 by name (onnxruntime-split/PKGBUILD, _ortf_nccl_pkg), not the bare nccl, which pacman resolves to Arch's CUDA 13 build"
        fi
        [[ $d == onnxruntime-* || $d == python-onnxruntime* ]] || continue
        # The CUDA 12 line has no core of its own: its pins are met by the
        # main line's onnxruntime-core or onnxruntime-core-bin.
        if [[ $srcbase == onnxruntime-split-cuda12 ]] && core_pin "$d"; then continue; fi
        met=0
        for p in "${offered[@]}"; do
            if satisfies "$d" "$p"; then
                met=1
                break
            fi
        done
        ((met)) || die 1 "${files[i]}: depend $d is met by no -bin package (they provide: ${offered[*]})"
    done
done

# q <value>: the value single-quoted for bash.
q() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
# array <indent> <name> <newline-separated values>: a bash array, one value
# per line; nothing when there are none.
array() {
    local v
    mapfile -t list < <(grep . <<<"$3" || true)
    ((${#list[@]} > 0)) || return 0
    printf '%s%s=(\n' "$1" "$2"
    for v in "${list[@]}"; do printf '%s    %s\n' "$1" "$(q "$v")"; done
    printf '%s)\n' "$1"
}
same_license=1
for ((i = 1; i < n; i++)); do
    [[ ${licenses[i]} == "${licenses[0]}" ]] || same_license=0
done
sums='' sources='' noextract=''
for ((i = 0; i < n; i++)); do
    sum=$(sha256sum -- "$pkgdir_in/${files[i]}")
    sums+=${sum%% *}$'\n'
    sources+="${files[i]}::$base_url/${files[i]}"$'\n'
    noextract+=${files[i]}$'\n'
done

pkgbuild=$tmp/PKGBUILD
{
    echo '# Maintainer: Raul Dias <raul@dias.com.br>'
    echo '#'
    echo "# Generated by scripts/gen-bin.sh of https://github.com/rsd/omarchy-onnxruntime"
    echo "# from the packages of its release $tag: do not edit it by hand, run the"
    echo '# generator again. Each <name>-bin package installs the files of the release'
    echo "# asset <name> (built by $srcbase) as they are."
    echo
    echo "pkgbase=$srcbase-bin"
    [[ $srcbase != onnxruntime-split-cuda12 ]] || echo "pkgdesc=$(q "$cuda12_pkgdesc (prebuilt)")"
    array '' pkgname "$(printf '%s-bin\n' "${names[@]}")"
    echo "pkgver=$(q "$pkgver")"
    echo "pkgrel=$(q "$bin_rel")"
    echo 'arch=(x86_64)'
    echo "url='https://github.com/rsd/omarchy-onnxruntime'"
    ((same_license == 0)) || array '' license "${licenses[0]}"
    echo 'options=(!strip !debug !lto)'
    array '' source "$sources"
    array '' sha256sums "$sums"
    echo "# Extracted, every asset would land in the same \$srcdir and their usr/"
    echo '# trees would merge: each package() unpacks its own.'
    array '' noextract "$noextract"
    for ((i = 0; i < n; i++)); do
        echo
        echo "package_${names[i]}-bin() {"
        echo "    pkgdesc=$(q "${descs[i]}")"
        ((same_license == 1)) || array '    ' license "${licenses[i]}"
        array '    ' depends "${deps[i]}"
        array '    ' optdepends "${optdeps[i]}"
        array '    ' provides "${provs[i]}"
        array '    ' conflicts "${confs[i]}"
        array '    ' replaces "${repls[i]}"
        echo
        echo "    bsdtar -xpf \"\$srcdir\"/$(q "${files[i]}") -C \"\$pkgdir\" usr"
        echo "    if [[ -d \$pkgdir/usr/share/licenses/${names[i]} ]]; then"
        echo "        mv \"\$pkgdir/usr/share/licenses/${names[i]}\" \"\$pkgdir/usr/share/licenses/${names[i]}-bin\""
        echo '    fi'
        echo '}'
    done
} >"$pkgbuild"
bash -n "$pkgbuild" || die 1 "the generated PKGBUILD does not parse (bug): $pkgbuild"

mkdir -p "$out"
out=$(cd "$out" && pwd)
cp "$pkgbuild" "$out/PKGBUILD"
if ! (cd "$out" && makepkg --printsrcinfo >"$tmp/SRCINFO" 2>"$tmp/err"); then
    cat "$tmp/err" >&2
    die 1 "makepkg --printsrcinfo failed in $out"
fi
cp "$tmp/SRCINFO" "$out/.SRCINFO"
(cd "$pkgdir_in" && sha256sum -- "${files[@]}" | LC_ALL=C sort -k 2) >"$tmp/SHA256SUMS"
cp "$tmp/SHA256SUMS" "$pkgdir_in/SHA256SUMS"

echo "gen-bin: wrote $out/PKGBUILD, $out/.SRCINFO and $pkgdir_in/SHA256SUMS ($n packages)"
echo "Publish the release with:"
printf '  gh release create %q' "$tag"
for f in "${files[@]}"; do printf ' %q' "$pkgdir_in/$f"; done
printf ' %q\n' "$pkgdir_in/SHA256SUMS"

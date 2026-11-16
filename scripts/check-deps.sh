#!/usr/bin/env bash
# Checks the library dependencies of built packages against this system.
#
#   scripts/check-deps.sh <package file>...
#   e.g. scripts/check-deps.sh onnxruntime-split/*-1.29.0-6-x86_64.pkg.tar.zst
#
# For each package file (unpacked with bsdtar into a temporary directory)
# and each ELF shared library in it (a regular file named *.so or *.so.*),
# it lists every DT_NEEDED soname with
#   - the path it resolves to: through the library's RUNPATH (or RPATH;
#     $ORIGIN is the library's installed directory), a library of one of
#     the package files given (installed under /usr/lib), else ldconfig's
#     cache (ldconfig -p) or /usr/lib;
#   - the package that owns that path (one of the package files given, or
#     on this system: pacman -Qo, read-only) and its version;
#   - whether a depend of the package's .PKGINFO covers that owner, by its
#     name or by a name it provides.
# Then every depend of the package: the installed package (or package file
# given) that is or provides it, its version, and whether that version is
# inside the depend's bounds (vercmp, as pacman compares).
#
# Flags, each a STATUS in the tables:
#   GAP         a needed library whose owner no depend covers, or that no
#               package owns
#   UNRESOLVED  a needed soname that resolves to no file on this system
#   BOUND       an installed version outside a depend's bounds: the vendor
#               package (cuda, cudnn, TensorRT, openvino, migraphx,
#               hip-runtime-amd, python) was
#               upgraded past what the package was built for, and a
#               rebuild is due
#   MISSING     a depend no installed package and no package file given
#               meets
# Limit: it sees only DT_NEEDED entries. A library a package dlopen()s at run
# time (cuDNN, which the CUDA provider opens, libcuda, libnvrtc, libcufft) is
# invisible to it: those are covered by the recipe's recorded bounds (cudnn by
# the build-time cuDNN major, cuda and nvidia-utils by depends), which the
# DEPEND table checks as any depend.
# Exit status: 0 when nothing is flagged, 1 when anything is, 2 on a usage
# error or a package file that cannot be read. Read-only: it queries
# pacman and ldconfig and writes only under mktemp.
set -uo pipefail
shopt -s nullglob

if (($# == 0)) || [[ $1 == -h || $1 == --help ]]; then
    sed -n '2,/^set -uo/{/^set -uo/d;s/^# \{0,1\}//;p}' "$0"
    (($# > 0)) || exit 2
    exit 0
fi
for tool in bsdtar readelf pacman vercmp; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "check-deps: $tool not found" >&2
        exit 2
    fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The packages: root directory, name, version (pkgver-pkgrel), depends and
# provides (newline-separated), by index.
declare -a roots=() names=() versions=() depends=() provides=() files=()
i=0
for file in "$@"; do
    root=$tmp/$i
    mkdir -p "$root"
    if ! bsdtar -xf "$file" -C "$root" 2>"$tmp/err" || [[ ! -f $root/.PKGINFO ]]; then
        echo "check-deps: $file: not a package file ($(head -n 1 "$tmp/err"))" >&2
        exit 2
    fi
    roots[i]=$root files[i]=$file
    names[i]=$(sed -n 's/^pkgname = //p' "$root/.PKGINFO" | head -n 1)
    versions[i]=$(sed -n 's/^pkgver = //p' "$root/.PKGINFO" | head -n 1)
    depends[i]=$(sed -n 's/^depend = //p' "$root/.PKGINFO")
    provides[i]=$(sed -n 's/^provides = //p' "$root/.PKGINFO")
    i=$((i + 1))
done

# ldconfig's cache: soname -> path, the x86-64 entries (the first wins).
declare -A cache=()
while IFS= read -r line; do
    if [[ $line =~ ^[[:space:]]+([^[:space:]]+)\ \(([^\)]*)\)\ =\>\ (.+)$ ]]; then
        [[ ${BASH_REMATCH[2]} == *x86-64* ]] || continue
        [[ -n ${cache[${BASH_REMATCH[1]}]:-} ]] || cache[${BASH_REMATCH[1]}]=${BASH_REMATCH[3]}
    fi
done < <(ldconfig -p 2>/dev/null)

# satisfies <dependency> <provision>: whether a package or provision
# <name>[=<version>] meets <name>[<op><version>], as pacman compares. A
# provision without a version meets only a dependency without one.
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

# The installed packages queried so far: name -> version, and name -> its
# provides (space-separated).
declare -A sys_version=() sys_provides=()
# query <name>: sets q_name, q_version and q_provides to the installed
# package that is or provides <name> (pacman -Qi); fails when there is none.
query() {
    local info line key='' val
    q_name='' q_version='' q_provides=''
    info=$(LC_ALL=C pacman -Qi -- "$1" 2>/dev/null) || return 1
    while IFS= read -r line; do
        if [[ $line =~ ^([A-Z][A-Za-z\ ]*[a-z])\ *:\ ?(.*)$ ]]; then
            key=${BASH_REMATCH[1]} val=${BASH_REMATCH[2]}
        elif [[ $line == ' '* ]]; then
            val=$line
        else
            key='' val=''
        fi
        case $key in
            Name) q_name=$val ;;
            Version) q_version=$val ;;
            Provides) [[ $val == None ]] || q_provides+=" $val" ;;
        esac
    done <<<"$info"
    [[ -n $q_name ]] || return 1
    sys_version[$q_name]=$q_version
    sys_provides[$q_name]=$q_provides
}

# offers <package> <version> <provides>: prints what a package
# offers to a dependency, one per line: <package>=<version> and each
# provision.
offers() {
    local p
    local -a words
    printf '%s=%s\n' "$1" "$2"
    read -ra words <<<"${3//$'\n'/ }"
    for p in "${words[@]}"; do printf '%s\n' "$p"; done
}

gaps=0 unresolved=0 bounds=0 missing=0
row() { printf '  %-44s %-26s %-44s %-28s %-24s %s\n' "$@"; }
drow() { printf '  %-44s %-34s %-28s %s\n' "$@"; }

for i in "${!roots[@]}"; do
    root=${roots[i]} self=${names[i]}
    echo "== $self ${versions[i]} (${files[i]##*/})"
    mapfile -t deplist <<<"${depends[i]}"
    depnames=()
    for d in "${deplist[@]}"; do [[ -z $d ]] || depnames+=("${d%%[<>=]*}"); done

    mapfile -t libs < <(cd "$root" && find . -type f \( -name '*.so' -o -name '*.so.*' \) | LC_ALL=C sort)
    elf=0
    for lib in "${libs[@]}"; do
        lib=${lib#./}
        [[ $(od -An -tx1 -N4 -- "$root/$lib" | tr -d ' \n') == 7f454c46 ]] || continue
        if ((elf == 0)); then
            row LIBRARY NEEDED RESOLVED OWNER VERSION STATUS
            elf=1
        fi
        dyn=$(readelf -d -- "$root/$lib" 2>/dev/null)
        mapfile -t needed < <(sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p' <<<"$dyn")
        runpath=$(sed -n 's/.*(RUNPATH).*\[\(.*\)\]$/\1/p' <<<"$dyn")
        [[ -n $runpath ]] || runpath=$(sed -n 's/.*(RPATH).*\[\(.*\)\]$/\1/p' <<<"$dyn")
        origin=/$(dirname -- "$lib")
        runpath=${runpath//\$\{ORIGIN\}/$origin}
        runpath=${runpath//\$ORIGIN/$origin}
        IFS=: read -ra rdirs <<<"$runpath"
        for soname in "${needed[@]}"; do
            path='' owner='' version='' in=-1
            # The library's RUNPATH, in the packages given, then here.
            for r in "${rdirs[@]}"; do
                [[ -n $r && -z $path ]] || continue
                for j in "${!roots[@]}"; do
                    if [[ -z $path && -e ${roots[j]}$r/$soname ]]; then path=$r/$soname in=$j; fi
                done
                [[ -n $path || ! -e $r/$soname ]] || path=$r/$soname
            done
            # The packages given install into the default library path.
            for j in "${!roots[@]}"; do
                if [[ -z $path && -e ${roots[j]}/usr/lib/$soname ]]; then path=/usr/lib/$soname in=$j; fi
            done
            [[ -n $path ]] || path=${cache[$soname]:-}
            [[ -n $path || ! -e /usr/lib/$soname ]] || path=/usr/lib/$soname
            if [[ -z $path ]]; then
                row "$lib" "$soname" '(not found)' '' '' UNRESOLVED
                unresolved=$((unresolved + 1))
                continue
            fi
            if ((in >= 0)); then
                owner=${names[in]} version=${versions[in]} owner_provides=${provides[in]}
            elif owner=$(pacman -Qqo -- "$path" 2>/dev/null) && [[ -n $owner ]]; then
                if [[ -z ${sys_version[$owner]+x} ]]; then query "$owner" || true; fi
                version=${sys_version[$owner]:-} owner_provides=${sys_provides[$owner]:-}
            else
                row "$lib" "$soname" "$path" '(no package)' '' GAP
                gaps=$((gaps + 1))
                continue
            fi
            covered=''
            if [[ $owner == "$self" ]]; then
                covered='ok (this package)'
            else
                read -ra owner_names <<<"${owner_provides//$'\n'/ }"
                for n in "$owner" "${owner_names[@]%%=*}"; do
                    [[ -z $covered && " ${depnames[*]} " == *" $n "* ]] && covered="ok ($n)"
                done
            fi
            if [[ -n $covered ]]; then
                row "$lib" "$soname" "$path" "$owner" "$version" "$covered"
            else
                row "$lib" "$soname" "$path" "$owner" "$version" "GAP (no depend on $owner)"
                gaps=$((gaps + 1))
            fi
        done
    done
    ((elf == 1)) || echo "  (no ELF shared library)"

    # Every depend: what meets it, and whether its version is in bounds.
    drow DEPEND 'MET BY' VERSION STATUS
    for d in "${deplist[@]}"; do
        [[ -n $d ]] || continue
        n=${d%%[<>=]*} by='' at=''
        # One of the packages given, first.
        for j in "${!roots[@]}"; do
            ((j != i)) && [[ -z $by ]] || continue
            while IFS= read -r o; do
                if [[ -n $o && -z $by && ${o%%=*} == "$n" ]]; then
                    by=${names[j]} at=${o#"$n"} at=${at#=}
                fi
            done < <(offers "${names[j]}" "${versions[j]}" "${provides[j]}")
        done
        if [[ -z $by ]] && query "$n"; then
            by=$q_name
            if [[ $q_name == "$n" ]]; then
                at=$q_version
            else
                at=''
                read -ra words <<<"$q_provides"
                for o in "${words[@]}"; do
                    if [[ $o == "$n="* ]]; then at=${o#*=}; fi
                done
            fi
        fi
        if [[ -z $by ]]; then
            drow "$d" '(nothing installed)' '' MISSING
            missing=$((missing + 1))
        elif [[ $d == "$n" ]]; then
            drow "$d" "$by" "${at:-(unversioned)}" ok
        elif [[ -n $at ]] && satisfies "$d" "$n=$at"; then
            drow "$d" "$by" "$at" ok
        else
            drow "$d" "$by" "${at:-(unversioned)}" "BOUND (outside $d: rebuild due)"
            bounds=$((bounds + 1))
        fi
    done
    echo
done

flags=$((gaps + unresolved + bounds + missing))
if ((flags > 0)); then
    echo "flagged: $gaps GAP, $unresolved UNRESOLVED, $bounds BOUND, $missing MISSING"
    exit 1
fi
echo "no gaps, unresolved libraries or bound violations"

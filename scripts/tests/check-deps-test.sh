#!/usr/bin/env bash
# Tests for scripts/check-deps.sh on fixture packages built under mktemp:
# small shared libraries compiled with gcc (one links this system's zlib,
# one a library no package installs, one finds that library through its
# RUNPATH), each packed with a .PKGINFO by bsdtar. The cases: every needed
# library covered by a depend passes; a missing depend is a GAP; a bound
# that excludes the installed zlib is a BOUND; an unfindable soname is
# UNRESOLVED; a depend nothing meets is MISSING; a library of another
# package file given, and a depend on it, are checked against that file; a
# library covered by a depend on a name its owner provides; a depend met by a
# virtual provision; a non-ELF *.so skipped; $ORIGIN in a RUNPATH; a
# needed libdnnl is a DNNL flag whoever owns it, the package's own copy
# included (oneDNN must be linked statically); a needed libwebgpu_dawn or
# libdawn* is a DAWN flag the same way (Dawn must be linked statically).
# It queries the real pacman and ldconfig (read-only) and needs gcc, zlib
# installed and pacman's vercmp; without them it says so and skips.
# Usage: scripts/tests/check-deps-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../check-deps.sh

for t in gcc bsdtar readelf pacman vercmp ldconfig; do
    if ! command -v "$t" >/dev/null 2>&1; then
        echo "skip: $t not found"
        exit 0
    fi
done
if ! zlib_owner=$(pacman -Qqo /usr/lib/libz.so.1 2>/dev/null); then
    echo "skip: /usr/lib/libz.so.1 belongs to no installed package"
    exit 0
fi
zlib_version=$(pacman -Q "$zlib_owner" | cut -d' ' -f2)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

# The libraries: libfxz.so.1 links zlib; libfxdep.so.1 is installed by no
# package (it stays in $tmp/dep); libfxb.so.1 links it without a RUNPATH,
# libfxr.so.1 with $tmp/dep as its RUNPATH.
mkdir -p "$tmp/src" "$tmp/dep"
printf 'const char *zlibVersion(void);\nconst char *fxz(void) { return zlibVersion(); }\n' >"$tmp/src/fxz.c"
printf 'int fxdep(void) { return 1; }\n' >"$tmp/src/fxdep.c"
printf 'int fxdep(void);\nint fxb(void) { return fxdep(); }\n' >"$tmp/src/fxb.c"
build() { # build <output> <soname> <source> <gcc arguments...>
    local out=$1 soname=$2 src=$3
    shift 3
    gcc -shared -fPIC -o "$out" -Wl,-soname,"$soname" "$src" "$@"
}
if ! build "$tmp/libfxz.so.1" libfxz.so.1 "$tmp/src/fxz.c" -lz 2>"$tmp/gcc.err"; then
    echo "skip: gcc cannot link a shared library against zlib: $(head -n 3 "$tmp/gcc.err")"
    exit 0
fi
build "$tmp/dep/libfxdep.so.1" libfxdep.so.1 "$tmp/src/fxdep.c"
build "$tmp/libfxb.so.1" libfxb.so.1 "$tmp/src/fxb.c" -L"$tmp/dep" -l:libfxdep.so.1
build "$tmp/libfxr.so.1" libfxr.so.1 "$tmp/src/fxb.c" -L"$tmp/dep" -l:libfxdep.so.1 -Wl,--enable-new-dtags,-rpath,"$tmp/dep"

# shellcheck disable=SC2016 # $ORIGIN is for the linker, not the shell
build "$tmp/libfxo.so.1" libfxo.so.1 "$tmp/src/fxb.c" -L"$tmp/dep" -l:libfxdep.so.1 '-Wl,--enable-new-dtags,-rpath,$ORIGIN'
# A linker script, as libc.so or libm.so are in a development package: a
# *.so that is not an ELF file.
printf '/* GNU ld script */\nINPUT(-lz)\n' >"$tmp/libfxtext.so"

# pack <name> <version> <library...> -- <depend...>: builds
# $tmp/<name>-<version>-x86_64.pkg.tar.gz holding the libraries in
# /usr/lib/$pack_sub (/usr/lib when unset) and a .PKGINFO with those depends; prints its path. Each in a
# root of its own (pack runs in a command substitution).
pack() {
    local name=$1 version=$2 root d
    shift 2
    root=$(mktemp -d "$tmp/root.XXXXXX")
    mkdir -p "$root/usr/lib/${pack_sub:-}"
    while (($#)) && [[ $1 != -- ]]; do
        cp "$tmp/$1" "$root/usr/lib/${pack_sub:-}"
        shift
    done
    (($# == 0)) || shift
    {
        printf 'pkgname = %s\npkgbase = %s\npkgver = %s\narch = x86_64\n' "$name" "$name" "$version"
        for d; do printf 'depend = %s\n' "$d"; done
    } >"$root/.PKGINFO"
    (cd "$root" && bsdtar -czf "$tmp/$name-$version-x86_64.pkg.tar.gz" .PKGINFO usr)
    printf '%s\n' "$tmp/$name-$version-x86_64.pkg.tar.gz"
}

# check <package file...>: runs the tool; sets status and out.
check() {
    status=0
    out=$("$tool" "$@" 2>&1) || status=$?
}
# has <regex>: some line of out matches.
has() { grep -qE -- "$1" <<<"$out"; }

covered=$(pack fx-zlib-user 1-1 libfxz.so.1 -- glibc "$zlib_owner")
check "$covered"
if ((status == 0)) && has "libfxz\.so\.1 +libz\.so\.1 +/usr/lib/libz\.so\.1 +$zlib_owner .*ok \($zlib_owner\)" &&
    has 'libc\.so\.6 .* glibc .*ok \(glibc\)' && has 'no gaps, unresolved libraries, bound violations, shared libdnnl or shared Dawn'; then
    ok "every needed library covered by a depend: exit 0"
else
    fail "covered (exit $status): $out"
fi

gap=$(pack fx-no-zlib 1-1 libfxz.so.1 -- glibc)
check "$gap"
if ((status == 1)) && has "libz\.so\.1 .* GAP \(no depend on $zlib_owner\)" && has 'flagged: 1 GAP, 0 UNRESOLVED, 0 BOUND, 0 MISSING'; then
    ok "a missing depend is a GAP: exit 1"
else
    fail "missing depend (exit $status): $out"
fi

bound=$(pack fx-old-zlib 1-1 libfxz.so.1 -- glibc "$zlib_owner<1:1.0" "$zlib_owner>=0")
check "$bound"
if ((status == 1)) && has "^  $zlib_owner<1:1\.0 +$zlib_owner +$zlib_version +BOUND" &&
    has "^  $zlib_owner>=0 +$zlib_owner +$zlib_version +ok" && has 'flagged: 0 GAP, 0 UNRESOLVED, 1 BOUND, 0 MISSING'; then
    ok "a bound that excludes the installed zlib ($zlib_version) is a BOUND: exit 1"
else
    fail "bound (exit $status): $out"
fi

unresolved=$(pack fx-unresolved 1-1 libfxb.so.1 -- glibc)
check "$unresolved"
if ((status == 1)) && has 'libfxb\.so\.1 +libfxdep\.so\.1 +\(not found\) .*UNRESOLVED' &&
    has 'flagged: 0 GAP, 1 UNRESOLVED'; then
    ok "a soname nothing provides is UNRESOLVED"
else
    fail "unresolved (exit $status): $out"
fi

runpath=$(pack fx-runpath 1-1 libfxr.so.1 -- glibc)
check "$runpath"
if ((status == 1)) && has "libfxr\.so\.1 +libfxdep\.so\.1 +$tmp/dep/libfxdep\.so\.1 +\(no package\) .*GAP" &&
    has 'flagged: 1 GAP, 0 UNRESOLVED'; then
    ok "a soname found through the RUNPATH, owned by no package, is a GAP"
else
    fail "runpath (exit $status): $out"
fi

missing=$(pack fx-missing 1-1 libfxz.so.1 -- glibc "$zlib_owner" fx-no-such-package-anywhere)
check "$missing"
if ((status == 1)) && has '^  fx-no-such-package-anywhere +\(nothing installed\) +MISSING' &&
    has 'flagged: 0 GAP, 0 UNRESOLVED, 0 BOUND, 1 MISSING'; then
    ok "a depend nothing meets is MISSING"
else
    fail "missing (exit $status): $out"
fi

# The owner's provides: zlib provides libz.so=1-64, so a depend on libz.so
# covers libz.so.1 without one on zlib itself.
if zlib_provides=$(pacman -Qi -- "$zlib_owner" | sed -n 's/^Provides *: //p') && [[ $zlib_provides == *libz.so=* ]]; then
    viaprov=$(pack fx-prov-user 1-1 libfxz.so.1 -- glibc libz.so)
    check "$viaprov"
    if ((status == 0)) && has "libz\.so\.1 +/usr/lib/libz\.so\.1 +$zlib_owner .*ok \(libz\.so\)" &&
        has '^  libz\.so +[^ ]+ +[^ ]+ +ok'; then
        ok "a library covered through a name its owner provides, the depend met by a provision"
    else
        fail "owner's provides (exit $status): $out"
    fi
else
    echo "skip: $zlib_owner provides no libz.so=<version>"
fi
# A depend met by a virtual provision (bash provides sh, unversioned).
if sh_owner=$(pacman -Qi -- sh 2>/dev/null | sed -n 's/^Name *: //p') && [[ -n $sh_owner && $sh_owner != sh ]]; then
    virtual=$(pack fx-virtual 1-1 libfxz.so.1 -- glibc "$zlib_owner" sh)
    check "$virtual"
    if ((status == 0)) && has "^  sh +$sh_owner +\(unversioned\) +ok"; then
        ok "a depend met by a virtual provision ($sh_owner provides sh)"
    else
        fail "virtual provision (exit $status): $out"
    fi
else
    echo "skip: no installed package provides sh"
fi

# A *.so that is no ELF file is skipped: no row for it, no failure.
textso=$(pack fx-text-so 1-1 libfxz.so.1 libfxtext.so -- glibc "$zlib_owner")
check "$textso"
if ((status == 0)) && ! has 'libfxtext' && has 'libfxz\.so\.1 .*ok'; then
    ok "a non-ELF *.so (a linker script) is skipped"
else
    fail "non-ELF .so (exit $status): $out"
fi
textonly=$(pack fx-text-only 1-1 libfxtext.so -- glibc)
check "$textonly"
if ((status == 0)) && has '\(no ELF shared library\)'; then ok "a package of only a non-ELF *.so has no ELF shared library"; else
    fail "text only (exit $status): $out"; fi

# $ORIGIN in a RUNPATH is the library's installed directory: libfxo.so.1 and
# the library it links are in /usr/lib/fx, which is no default library path.
cp "$tmp/dep/libfxdep.so.1" "$tmp/libfxdep.so.1"
pack_sub=fx
origin=$(pack fx-origin 1-1 libfxo.so.1 libfxdep.so.1 -- glibc)
pack_sub=
check "$origin"
if ((status == 0)) && has 'libfxo\.so\.1 +libfxdep\.so\.1 +/usr/lib/fx/libfxdep\.so\.1 +fx-origin +1-1 +ok \(this package\)'; then
    ok "\$ORIGIN in a RUNPATH expands to the library's installed directory"
else
    fail "origin (exit $status): $out"
fi

# Two package files: fx-dep holds libfxdep.so.1, which fx-user's library
# links; fx-user depends on fx-dep at its version, or at a later one.
dep=$(pack fx-dep 2-1 libfxdep.so.1 -- glibc)
user=$(pack fx-user 1-1 libfxb.so.1 -- glibc 'fx-dep=2-1')
check "$dep" "$user"
if ((status == 0)) && has 'libfxb\.so\.1 +libfxdep\.so\.1 +/usr/lib/libfxdep\.so\.1 +fx-dep +2-1 +ok \(fx-dep\)' &&
    has '^  fx-dep=2-1 +fx-dep +2-1 +ok'; then
    ok "a library and a depend met by another package file given pass"
else
    fail "two packages (exit $status): $out"
fi
user=$(pack fx-user-new 1-1 libfxb.so.1 -- glibc 'fx-dep>=3')
check "$dep" "$user"
if ((status == 1)) && has '^  fx-dep>=3 +fx-dep +2-1 +BOUND'; then
    ok "a package file given outside a depend's bounds is a BOUND"
else
    fail "two packages, bound (exit $status): $out"
fi

# A shared libdnnl: a stand-in libdnnl.so.3 (oneDNN's soname) shipped in the
# package itself, and a library that links it. Resolved and owned (this
# package), it is still flagged: the oneDNN provider links oneDNN
# statically, and a libdnnl.so.3 in /usr/lib would be Arch's onednn's file.
printf 'int dnnl_version(void) { return 3; }\n' >"$tmp/src/fxdnnl.c"
printf 'int dnnl_version(void);\nint fxp(void) { return dnnl_version(); }\n' >"$tmp/src/fxp.c"
build "$tmp/libdnnl.so.3" libdnnl.so.3 "$tmp/src/fxdnnl.c"
build "$tmp/libfxp.so" libfxp.so "$tmp/src/fxp.c" -L"$tmp" -l:libdnnl.so.3
shared_dnnl=$(pack fx-provider-dnnl 1-1 libfxp.so libdnnl.so.3 -- glibc)
check "$shared_dnnl"
if ((status == 1)) && has 'libfxp\.so +libdnnl\.so\.3 .*DNNL \(oneDNN must be linked statically\)' &&
    has 'flagged: 0 GAP, 0 UNRESOLVED, 0 BOUND, 0 MISSING, 1 DNNL'; then
    ok "a needed libdnnl, even the package's own, is a DNNL flag: exit 1"
else
    fail "shared libdnnl (exit $status): $out"
fi
static_dnnl=$(pack fx-provider-static 1-1 libfxdep.so.1 -- glibc)
check "$static_dnnl"
if ((status == 0)) && ! has 'DNNL'; then ok "no libdnnl needed: no DNNL flag"; else
    fail "no libdnnl (exit $status): $out"; fi

# A shared Dawn, the same way: stand-ins libwebgpu_dawn.so (the soname of
# Dawn's monolithic shared library) and libdawn_proc.so shipped in the
# provider's own package and needed by its library, each flagged: the
# WebGPU provider links Dawn statically.
printf 'int wgpuCreateInstance(void) { return 1; }\n' >"$tmp/src/fxdawn.c"
printf 'int dawnProcSetProcs(void) { return 1; }\n' >"$tmp/src/fxdawnproc.c"
printf 'int wgpuCreateInstance(void);\nint dawnProcSetProcs(void);\nint fxw(void) { return wgpuCreateInstance() + dawnProcSetProcs(); }\n' >"$tmp/src/fxw.c"
build "$tmp/libwebgpu_dawn.so" libwebgpu_dawn.so "$tmp/src/fxdawn.c"
build "$tmp/libdawn_proc.so" libdawn_proc.so "$tmp/src/fxdawnproc.c"
build "$tmp/libfxw.so" libfxw.so "$tmp/src/fxw.c" -L"$tmp" -l:libwebgpu_dawn.so -l:libdawn_proc.so
shared_dawn=$(pack fx-provider-webgpu 1-1 libfxw.so libwebgpu_dawn.so libdawn_proc.so -- glibc)
check "$shared_dawn"
if ((status == 1)) && has 'libfxw\.so +libwebgpu_dawn\.so .*DAWN \(Dawn must be linked statically\)' &&
    has 'libfxw\.so +libdawn_proc\.so .*DAWN \(Dawn must be linked statically\)' &&
    has 'flagged: 0 GAP, 0 UNRESOLVED, 0 BOUND, 0 MISSING, 0 DNNL, 2 DAWN'; then
    ok "a needed libwebgpu_dawn or libdawn*, even the package's own, is a DAWN flag: exit 1"
else
    fail "shared Dawn (exit $status): $out"
fi
check "$static_dnnl"
if ((status == 0)) && ! has 'DAWN \('; then ok "no Dawn library needed: no DAWN flag"; else
    fail "no Dawn library (exit $status): $out"; fi

check
if ((status == 2)) && has 'scripts/check-deps.sh <package file>'; then ok "no arguments: usage, exit 2"; else
    fail "no arguments (exit $status): $out"; fi
printf 'not a package\n' >"$tmp/bogus.pkg.tar.gz"
check "$tmp/bogus.pkg.tar.gz"
if ((status == 2)) && has 'not a package file'; then ok "a file that is no package: exit 2"; else
    fail "bogus file (exit $status): $out"; fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"

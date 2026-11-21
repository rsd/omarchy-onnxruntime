#!/usr/bin/env bash
# Makes the source archive of ONNX Runtime that the mirror release hosts, the
# same bytes on every run.
#
#   scripts/make-source-archive.sh [--repo GIT_DIR] [--out DIR] [--pkgbuild FILE]
#
# github.com/microsoft/onnxruntime has returned 404 since 2026-11-17, so the
# exact v<pkgver> source is hosted as an asset of the GitHub release
# source-v<pkgver> of rsd/omarchy-onnxruntime. This script builds that asset.
#
# It reads pkgver and _ortf_commit (the top-level `pkgver=` and
# `_ortf_commit=` lines; the file is never sourced) from the PKGBUILD, checks
# that the tag v<pkgver> in the repository resolves (peeled, ^{commit}) to
# _ortf_commit, and refuses otherwise. It then writes into DIR
#   onnxruntime-<pkgver>.tar.gz         git archive of the commit, gzip -n -9
#   onnxruntime-<pkgver>.tar.gz.sha256  sha256sum format, bare file name
#   PROVENANCE.md                       what it is and how to verify it
# and prints the sha256, b2sum and size and the `gh release create` command
# (it never runs it). The archive is exactly
#   git -C REPO archive --format=tar --prefix=onnxruntime-<pkgver>/ <commit> | gzip -n -9
# (the commit hash, not the tag; the external gzip -n leaves no name or mtime
# in the gzip header). It is written to a temporary file in DIR and renamed.
#
# --repo GIT_DIR  a clone or bare mirror of microsoft/onnxruntime
#                 (default: onnxruntime-split/onnxruntime in this repository)
# --out DIR       where the files go, created when missing
#                 (default: the current directory)
# --pkgbuild FILE the PKGBUILD to read (default: onnxruntime-split/PKGBUILD in
#                 this repository; the tests point it at a fixture)
#
# Exit status: 0 done, 1 the tag does not resolve to _ortf_commit (nothing is
# written), 2 usage error, missing tool, unreadable PKGBUILD or git error.
set -euo pipefail

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; }
die() { # die <status> <message>
    echo "make-source-archive: $2" >&2
    exit "$1"
}

top=$(cd "$(dirname "$0")/.." && pwd)
repo=$top/onnxruntime-split/onnxruntime
out=.
pkgbuild=$top/onnxruntime-split/PKGBUILD
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --repo) (($# >= 2)) || die 2 "--repo needs a directory"; repo=$2; shift 2 ;;
        --repo=*) repo=${1#*=}; shift ;;
        --out) (($# >= 2)) || die 2 "--out needs a directory"; out=$2; shift 2 ;;
        --out=*) out=${1#*=}; shift ;;
        --pkgbuild) (($# >= 2)) || die 2 "--pkgbuild needs a file"; pkgbuild=$2; shift 2 ;;
        --pkgbuild=*) pkgbuild=${1#*=}; shift ;;
        *) die 2 "unknown argument: $1 (see --help)" ;;
    esac
done

for tool in git gzip sha256sum b2sum; do
    command -v "$tool" >/dev/null 2>&1 || die 2 "$tool not found"
done
[[ -r $pkgbuild ]] || die 2 "$pkgbuild: cannot read it"
# value <name>: the first top-level `name=value` line, quotes dropped.
value() { sed -n "s/^$1=//p" "$pkgbuild" | head -n 1 | tr -d "\"'"; }
pkgver=$(value pkgver)
commit=$(value _ortf_commit)
[[ $pkgver =~ ^[0-9][0-9A-Za-z._]*$ ]] || die 2 "$pkgbuild: no usable pkgver= line"
[[ $commit =~ ^[0-9a-f]{40}$ ]] || die 2 "$pkgbuild: no usable _ortf_commit= line (40 hex digits)"
[[ -d $repo ]] || die 2 "$repo: no such directory"
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die 2 "$repo: not a git repository"

tag=v$pkgver
tagged=$(git -C "$repo" rev-parse --verify --quiet "refs/tags/$tag^{commit}") || tagged=none
if [[ $tagged != "$commit" ]]; then
    die 1 "tag $tag in $repo is $tagged, but _ortf_commit is $commit: refusing"
fi

case $(git -C "$repo" cat-file -t "refs/tags/$tag") in
    tag) tagtype=annotated ;;
    *) tagtype=lightweight ;;
esac

name=onnxruntime-$pkgver.tar.gz
mkdir -p "$out"
out=$(cd "$out" && pwd)
tmp=$(mktemp "$out/.$name.XXXXXX")
tmpmd=$(mktemp "$out/.PROVENANCE.XXXXXX")
trap 'rm -f "$tmp" "$tmpmd" "${tmpsha:-}"' EXIT

git -C "$repo" archive --format=tar --prefix="onnxruntime-$pkgver/" "$commit" | gzip -n -9 >"$tmp" ||
    die 2 "git archive or gzip failed"
sha=$(sha256sum "$tmp" | cut -d' ' -f1)
b2=$(b2sum "$tmp" | cut -d' ' -f1)
size=$(stat -c %s "$tmp")
tarsha=$(gzip -dc "$tmp" | sha256sum | cut -d' ' -f1)
# makepkg's checksum of the git source: the tag's tree, no prefix.
tagb2=$(git -C "$repo" -c core.abbrev=no archive --format tar "$tag" | b2sum | cut -d' ' -f1)
gitver=$(git --version)
gzver=$(gzip --version | head -n 1)
script=scripts/make-source-archive.sh
short=${commit:0:7}

cat >"$tmpmd" <<MD
# ONNX Runtime $pkgver source (mirror)

The exact source of ONNX Runtime $pkgver, hosted here because
https://github.com/microsoft/onnxruntime has returned 404 since 2026-11-17.
The build packages in this repository fetch it from this release when
\`ORTF_SOURCE\` selects the mirror.

| | |
|---|---|
| Upstream repository | https://github.com/microsoft/onnxruntime |
| Tag | \`$tag\` ($tagtype) |
| Commit | \`$commit\` |
| File | \`$name\` |
| Size | $size bytes |
| sha256 | \`$sha\` |
| b2sum | \`$b2\` |
| sha256 of the uncompressed tar stream | \`$tarsha\` |

## How it was made

\`\`\`
git -C REPO archive --format=tar --prefix=onnxruntime-$pkgver/ $commit | gzip -n -9 > $name
\`\`\`

By \`$script\` of this repository, with $gitver and $gzver. It archives the
commit hash, not the tag, and uses the external \`gzip -n\` so the gzip header
holds no file name and no time. The same inputs with these tool versions give
the same bytes. The tar stream is the stable, gzip-independent part: another
gzip build may compress differently, but the tar stream under it is the same.

## Submodules

The submodules of $tag (onnx, libprotobuf-mutator, emsdk) are not needed by the
build (onnx comes from a dependency zip). \`git archive\` leaves them as empty
directories, exactly as makepkg's checkout of the git source does, so they
are empty here too.

## Verify it against upstream

Use any clone or mirror of microsoft/onnxruntime that has the commit, run in
it, with $name and its .sha256 in the current directory:

1. The file matches its checksum: \`sha256sum -c $name.sha256\`.
2. The tar stream equals what git produces from the commit (the check that
   does not depend on gzip):
   \`\`\`
   git archive --format=tar --prefix=onnxruntime-$pkgver/ $commit | sha256sum
   gzip -dc $name | sha256sum
   \`\`\`
   Both print \`$tarsha\`.
3. git archive embeds the commit in the pax header:
   \`gzip -dc $name | git get-tar-commit-id\` prints \`$commit\`.
4. The link to makepkg's checksum of the tag. The git source of the PKGBUILD
   (and of Arch's onnxruntime package for this tag) is pinned by
   \`git -c core.abbrev=no archive --format tar $tag | b2sum\`, which is
   \`$tagb2\`
   (computed when this file was made). That is the same tree as this archive
   without the \`onnxruntime-$pkgver/\` prefix: \`git archive --format=tar $commit | b2sum\`
   gives the same value.
5. A tree-level check: extract this archive and a \`git archive $commit | tar -x\`
   (or a \`git worktree\` of the commit) into two directories and run
   \`diff -r\` on them (the prefix directory against the other tree).

Commit $short is what \`_ortf_commit\` in onnxruntime-split/PKGBUILD pins;
\`scripts/check-upstream.sh\` tells whether upstream is back and still has it.
MD

mv -f "$tmp" "$out/$name"
tmpsha=$(mktemp "$out/.$name.sha256.XXXXXX")
printf '%s  %s\n' "$sha" "$name" >"$tmpsha"
mv -f "$tmpsha" "$out/$name.sha256"
mv -f "$tmpmd" "$out/PROVENANCE.md"
chmod 644 "$out/$name" "$out/PROVENANCE.md" "$out/$name.sha256"

echo "file:   $out/$name"
echo "sha256: $sha"
echo "b2sum:  $b2"
echo "size:   $size bytes"
echo "publish (not run):"
echo "  cd '$out' && gh release create source-v$pkgver --repo rsd/omarchy-onnxruntime --latest=false --title 'ONNX Runtime $pkgver source (mirror)' --notes-file PROVENANCE.md $name $name.sha256"

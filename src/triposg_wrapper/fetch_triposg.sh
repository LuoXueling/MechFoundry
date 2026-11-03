#!/usr/bin/env bash
# Materialize TripoSG's source in src/dependency/TripoSG: the pinned upstream commit
# plus this repository's patch, checked file by file against a sha256 manifest.
#
#   bash src/triposg_wrapper/fetch_triposg.sh [TARGET_DIR]     (default src/dependency/TripoSG)
#
# Why a patch and not a vendored copy: until 2026-11 the 38 upstream files were committed
# here, though only one of them differs from upstream (patches/triposg.patch: the VAE's
# `from torch_cluster import fps`, which upstream commented out in 88cfe71 while the
# encoder still calls fps(), so encoding a surface into a latent raised NameError).
# Fetching upstream and applying the patch reproduces the vendored tree bit for bit
# (patches/triposg.sha256 is the sha256 of every file of that tree), and keeps
# src/dependency/ wholly third-party and git-ignored like the other checkouts.
#
# Safe to rerun.  It never deletes anything and never touches untracked files, so
# pretrained_weights/ (and stray outputs such as output.glb) survive every path:
#   * every manifest file matches            -> "already set up", nothing done;
#   * no TripoSG source present (no folder, or one holding only pretrained_weights/,
#     __pycache__/ or outputs, e.g. after the weights were downloaded first, or after
#     the vendored copy was removed from git) -> git init in place, shallow fetch of
#     the pinned commit, checkout, patch;
#   * a git checkout with no local edits to tracked files (any commit) -> check out the
#     pinned commit, patch;
#   * anything else (a hand-edited checkout, an older copy without .git) -> refuse and
#     say what differs; moving it aside is left to the user.
# The upstream .git is kept: it is small (one shallow commit) and `git -C
# src/dependency/TripoSG diff` then shows exactly our patch.
set -euo pipefail

URL=https://github.com/VAST-AI-Research/TripoSG.git
# Upstream main as of 2025-04-18 ("Update README.md"), the commit the vendored copy was
# taken from: of all 23 upstream commits it is the only one whose tree has exactly our
# 38 files, and it differs from ours in one file (8 diff lines).
PIN=fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PATCH="$HERE/patches/triposg.patch"
MANIFEST="$HERE/patches/triposg.sha256"
TARGET="${1:-${TRIPOSG_DIR:-$REPO/src/dependency/TripoSG}}"
# core.autocrlf would rewrite line endings on checkout (and then the checksums fail),
# so every git call here pins it off, whatever the user's global config says.
G=(git -c core.autocrlf=false -c core.eol=lf -c advice.detachedHead=false -C "$TARGET")

verify() {
    [ -d "$TARGET" ] && (cd "$TARGET" && sha256sum --quiet --strict -c "$MANIFEST") >/dev/null 2>&1
}

differences() {
    (cd "$TARGET" && sha256sum --quiet --strict -c "$MANIFEST" 2>&1 | sed 's/^/    /' | head -20) || true
}

refuse() {
    echo "fetch_triposg: $1" >&2
    echo "  files that differ from the pinned, patched TripoSG:" >&2
    differences >&2
    if [ -d "$TARGET/pretrained_weights" ]; then
        echo "  Nothing was changed.  To start over while keeping the weights:" >&2
        echo "    mv '$TARGET' '$TARGET.old' && mkdir '$TARGET' && mv '$TARGET.old/pretrained_weights' '$TARGET/'" >&2
    else
        echo "  Nothing was changed.  To start over: mv '$TARGET' '$TARGET.old'" >&2
    fi
    echo "  then rerun this script (and delete '$TARGET.old' once you no longer need it)." >&2
    exit 1
}

checkout_pin() {
    if [ "$("${G[@]}" rev-parse -q --verify 'HEAD^{commit}' 2>/dev/null || true)" != "$PIN" ]; then
        "${G[@]}" cat-file -e "$PIN^{commit}" 2>/dev/null || "${G[@]}" fetch -q --depth 1 "$URL" "$PIN"
        # Refuses (and changes nothing) if an untracked file sits where upstream has one.
        "${G[@]}" checkout -q --detach "$PIN"
    fi
}

apply_patch() {
    "${G[@]}" apply --check "$PATCH" || refuse "the patch does not apply to $TARGET at $PIN."
    "${G[@]}" apply "$PATCH"
}

if verify; then
    echo "fetch_triposg: $TARGET already set up (all files match the manifest)."
    exit 0
fi

if [ -e "$TARGET/.git" ]; then
    if [ -n "$("${G[@]}" status --porcelain --untracked-files=no)" ]; then
        refuse "$TARGET is a git checkout with local changes to tracked files."
    fi
    echo "fetch_triposg: checking out $PIN in the existing checkout $TARGET"
    checkout_pin
    apply_patch
else
    present=0
    while read -r _ f; do
        if [ -e "$TARGET/$f" ]; then present=1; break; fi
    done < "$MANIFEST"
    [ "$present" = 0 ] || refuse "$TARGET holds TripoSG files but is neither a git checkout nor the pinned, patched copy."
    echo "fetch_triposg: fetching $URL at $PIN into $TARGET"
    mkdir -p "$TARGET"
    git -c init.defaultBranch=main init -q "$TARGET"
    "${G[@]}" config core.autocrlf false
    checkout_pin
    apply_patch
fi

if verify; then
    echo "fetch_triposg: done; all $(wc -l < "$MANIFEST") files match the manifest."
else
    echo "fetch_triposg: $TARGET does not match the manifest after patching:" >&2
    differences >&2
    exit 1
fi

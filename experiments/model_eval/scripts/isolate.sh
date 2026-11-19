# Layer 1 of the evaluator isolation, shared by run_fable.sh and run_astra.sh.
# Sourced, not run.  Needs PACKET, OUT, CLI_HOME and HIDE (from config.py runner-env) and
# fills the array BWRAP with a bubblewrap prefix:
#
#  - the whole filesystem read-only;
#  - $HOME an empty tmpfs.  It hides everything an evaluator could learn the labels from: ~/.claude
#    (Claude transcripts, the Fable evaluators' sessions, auto-memory), ~/.claude.json and its
#    backup (project history), ~/.codex (Codex sessions, the Astra evaluators' rollouts), other
#    checkouts (~/MechFoundry), shell history and caches.  The paper's Astra grading hid ~/.claude,
#    ~/.codex and its work dir one by one; hiding the whole home is the same idea without a list that
#    misses the next new folder.  Bound back read-only: only what runs the CLIs and the
#    evaluator's own scripts (the claude and codex binaries, the anaconda python3 on PATH);
#    the rest of the tmpfs is writable scratch that vanishes with the namespace;
#  - the repo (this checkout and, from a worktree, the main checkout: results/ with the ratings
#    and mappings, records/, the other packets, the batch folders) and /tmp: empty tmpfs.  The
#    packet and the output dir sit under output/ in the repo and are bound back below;
#  - the packet: a fresh tmpfs with index.json, figures/ and runs/ bound into it read-only.  The
#    cwd is then writable (Codex's own sandbox must create its protected mount points .git,
#    .agents, ... there, which a read-only bind of the whole packet refuses, and bubblewrap 0.6
#    has no tmp-overlay) but nothing written beside the packet's content survives;
#  - the output dir: bound read-write, the one place a write survives (with the private CLI home
#    the runner binds itself);
#  - a clean environment (HOME, PATH, USER, LANG, TERM only): a runner started from a shell, or
#    from inside another agent's session, would otherwise hand on that session's variables
#    (CLAUDECODE, API keys, FEM_OPERATOR_*, ...).
# Binds come after the tmpfs that covers them, so bubblewrap creates their mount points inside it.
H=$HOME
BWRAP=(bwrap --clearenv --setenv HOME "$H" --setenv PATH "$PATH" --setenv USER "${USER:-$(id -un)}"
       --setenv LANG "${LANG:-C.UTF-8}" --setenv TERM "${TERM:-dumb}"
       --ro-bind / / --dev-bind /dev /dev --bind /proc /proc --tmpfs "$H")
for keep in "$H/.local/bin" "$H/.local/share/claude" "$H/.codex/packages" "$H/anaconda3"; do
  [ -e "$keep" ] && BWRAP+=(--ro-bind "$keep" "$keep")
done
for d in $HIDE /tmp; do
  [ -e "$d" ] && BWRAP+=(--tmpfs "$d")
done
BWRAP+=(--tmpfs "$PACKET"
        --ro-bind "$PACKET/index.json" "$PACKET/index.json"
        --ro-bind "$PACKET/figures" "$PACKET/figures"
        --ro-bind "$PACKET/runs" "$PACKET/runs"
        --bind "$OUT" "$OUT"
        --bind "$CLI_HOME" "$CLI_HOME"
        --chdir "$PACKET" --die-with-parent)

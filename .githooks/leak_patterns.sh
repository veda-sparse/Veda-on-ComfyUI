# Shared by the hooks in this directory (sourced, not executed).
#
# LEAK_PATTERNS matches secrets and machine-local information that must never
# be committed (AGENTS.md 1.3). Per-machine patterns (host names, user names,
# ssh aliases) go in .git/leak-patterns, one extended regex per line; that
# file lives inside .git and is never committed. Loopback (127.0.0.1) is
# allowed: tests serve files from it.

LEAK_PATTERNS='sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|/(Users|home|mnt|root|private/var)/|[A-Za-z]:\\\\Users\\\\|(^|[^0-9.])([0-9]{1,3}\.){3}[0-9]{1,3}([^0-9.]|$)|\.(internal|lan|corp|intra)([^A-Za-z0-9_]|$)'

# Prints the lines of file $1 that match a leak pattern, prefixed with $2.
leak_scan() {
    extra="$(git rev-parse --git-dir)/leak-patterns"
    grep -nE "$LEAK_PATTERNS" "$1" | grep -v '127\.0\.0\.1' | sed "s|^|$2:|"
    if [ -s "$extra" ]; then
        grep -v '^[[:space:]]*$' "$extra" > "$1.extra"
        grep -niE -f "$1.extra" "$1" | sed "s|^|$2:|"
        rm -f "$1.extra"
    fi
}

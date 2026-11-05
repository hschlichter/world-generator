#!/bin/sh
# Runs the Eldermark generator with the project's OpenUSD build.
#   generator/run.sh <command> [options]      (see generator/README.md)
# `validate` runs in its own memory-capped systemd scope (STREAMWORLD_MEM_MAX, default 16G, no swap) so
# a runaway check is killed on its own instead of systemd-oomd killing the whole terminal it was started from.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export PYTHONPATH="$ROOT/openusd_build/lib/python3.14/site-packages:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/openusd_build/bin:$PATH"
if [ "$1" = validate ] && [ -z "$STREAMWORLD_NO_SCOPE" ] && command -v systemd-run >/dev/null 2>&1 \
        && systemd-run --user --scope -q true 2>/dev/null; then
    exec systemd-run --user --scope -q -p MemoryMax="${STREAMWORLD_MEM_MAX:-16G}" -p MemorySwapMax=0 \
        python3 -m streamworld "$@"
fi
exec python3 -m streamworld "$@"

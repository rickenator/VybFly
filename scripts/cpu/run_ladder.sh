#!/usr/bin/env bash
# run_ladder.sh - run the fastest exact Vyb CPU ladder engine for a ladder graph.
#
#   VYBFLY_LADDER_DIR=results/energy/curves/s10 VYBFLY_LADDER_STIM=stim_s10.bin \
#   VYBFLY_THREADS=16 VYBFLY_EPOCHS=8 VYB=vyb scripts/cpu/run_ladder.sh [vyb flags...]
#
# All engines give the same digests, so the choice only affects speed. With VYBFLY_ENGINE unset
# (or "auto"), the choice is by edge count: ladder_prop below VYBFLY_DESC_MIN_EDGES (default 10M;
# 1x = 2.7M and 2x = 5.4M edges), ladder_desc at or above it (5x = 13.9M, 10x = 27.8M). This
# matches the measured crossover on an i9-10850K at 16 threads. Set VYBFLY_ENGINE=ladder_prop,
# ladder_seg or ladder_desc to force one. Extra arguments go to vyb before the source file.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dir=${VYBFLY_LADDER_DIR:-results/energy/curves/s1}
eng=${VYBFLY_ENGINE:-auto}
if [ "$eng" = auto ]; then
    idx="$dir/bin/csr.out.indices.i32"
    if [ ! -f "$idx" ]; then
        echo "run_ladder: no CSR at $idx" >&2
        exit 2
    fi
    ne=$(( $(stat -c %s "$idx") / 4 ))
    if [ "$ne" -lt "${VYBFLY_DESC_MIN_EDGES:-10000000}" ]; then eng=ladder_prop; else eng=ladder_desc; fi
fi
case "$eng" in
    ladder_prop|ladder_seg|ladder_desc) ;;
    *) echo "run_ladder: unknown engine '$eng'" >&2; exit 2 ;;
esac
echo "run_ladder: engine=$eng" >&2
exec "${VYB:-vyb}" "$@" "$here/../../src/vyb/$eng.vyb"

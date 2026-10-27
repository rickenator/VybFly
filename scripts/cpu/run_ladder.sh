#!/usr/bin/env bash
# run_ladder.sh - run the fastest exact Vyb ladder engine for a ladder graph.
#
#   VYBFLY_LADDER_DIR=results/energy/curves/s10 VYBFLY_LADDER_STIM=stim_s10.bin \
#   VYBFLY_THREADS=16 VYBFLY_EPOCHS=8 VYB=vyb scripts/cpu/run_ladder.sh [vyb flags...]
#
# All engines give the same digests, so the choice only affects speed. With VYBFLY_ENGINE unset
# (or "auto"), the choice is by edge count: ladder_prop below VYBFLY_DESC_MIN_EDGES (default 10M;
# 1x = 2.7M and 2x = 5.4M edges), ladder_desc at or above it (5x = 13.9M, 10x = 27.8M). This
# matches the measured crossover on an i9-10850K at 16 threads. On aarch64 (VYBFLY_ARCH, default
# `uname -m`) the default threshold is unbounded, i.e. always ladder_prop: on a DGX Spark (GB10,
# 20 Arm cores) ladder_prop is also faster at 5x and 10x (10x 7.52 s vs ladder_desc 11.06 s). Set
# VYBFLY_ENGINE=ladder_prop, ladder_seg or ladder_desc to force one. Extra arguments go to vyb
# before the source file.
#
# VYBFLY_ENGINE=gpu (or ladder_gpu) runs the CUDA engine src/vyb/ladder_gpu.vyb, which loads
# the committed PTX src/vyb_kernels/ladder_gpu_kernel.ptx (override with VYBFLY_GPU_PTX). It is
# never picked automatically: it needs an NVIDIA GPU and driver.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dir=${VYBFLY_LADDER_DIR:-results/energy/curves/s1}
eng=${VYBFLY_ENGINE:-auto}
if [ "$eng" = gpu ]; then eng=ladder_gpu; fi
if [ "$eng" = auto ]; then
    idx="$dir/bin/csr.out.indices.i32"
    if [ ! -f "$idx" ]; then
        echo "run_ladder: no CSR at $idx" >&2
        exit 2
    fi
    ne=$(( $(stat -c %s "$idx") / 4 ))
    min=10000000
    case "${VYBFLY_ARCH:-$(uname -m)}" in aarch64|arm64) min=9223372036854775807 ;; esac
    if [ "$ne" -lt "${VYBFLY_DESC_MIN_EDGES:-$min}" ]; then eng=ladder_prop; else eng=ladder_desc; fi
fi
case "$eng" in
    ladder_prop|ladder_seg|ladder_desc) ;;
    ladder_gpu) export VYBFLY_GPU_PTX="${VYBFLY_GPU_PTX:-$(cd "$here/../.." && pwd)/src/vyb_kernels/ladder_gpu_kernel.ptx}" ;;
    *) echo "run_ladder: unknown engine '$eng'" >&2; exit 2 ;;
esac
echo "run_ladder: engine=$eng" >&2
exec "${VYB:-vyb}" "$@" "$here/../../src/vyb/$eng.vyb"

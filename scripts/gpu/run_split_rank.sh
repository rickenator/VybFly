#!/bin/bash
# One rank of the two-machine Vyb GPU split (src/vyb/ladder_gpu_split.vyb).
# usage: VYBFLY_SPLIT_RANK=0|1 [VYBFLY_SPLIT_PEER=<rank-0 ip>] [VYBFLY_SPLIT_PORT=47011] \
#        scripts/gpu/run_split_rank.sh <ladder-dir> <stim.bin>
# Start rank 0 first; rank 1 retries its connect for about a minute. Refuses to run beside
# another GPU process, so it never shares a GPU with someone else's job.
set -euo pipefail
DIR=${1:?ladder dir}; STIM=${2:?stimulus file}
VYB=${VYB:-vyb}
STDLIB=${VYB_STDLIB:-$(dirname "$(dirname "$(command -v "$VYB")")")/stdlib}
cd "$(dirname "$0")/../.."
if command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "run_split_rank: another GPU process is running; not starting" >&2; exit 3
fi
export VYBFLY_LADDER_DIR=$DIR VYBFLY_LADDER_STIM=$STIM
export VYBFLY_EPOCHS=${VYBFLY_EPOCHS:-8} VYBFLY_SPLIT_PARTS=${VYBFLY_SPLIT_PARTS:-2}
exec "$VYB" --module-path "$STDLIB" src/vyb/ladder_gpu_split.vyb

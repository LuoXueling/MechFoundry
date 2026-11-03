#!/bin/bash
# Run clean-ups one after another on one GPU, each under a hard timeout.
#
#   bash run_queue.sh <gpu> <timeout_s> "<run.py options>" <object> [<object> ...]
#
# Start it in the background (nohup ... &); it logs each object to
# output/geometry_cleanup/_logs/<object><tag>.log and a line per object to
# _logs/queue_gpu<gpu>.txt.  It refuses to start an object while another compute
# process holds the GPU (one optimization per GPU): it waits at most an hour for the
# GPU to free up and otherwise SKIPS the object, with a "skip" line in the queue log.
set -u
GPU=$1; TMO=$2; OPTS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=/home/xl2254/implicit_optimize/output/geometry_cleanup/_logs
mkdir -p "$LOGS"
source /home/xl2254/anaconda3/etc/profile.d/conda.sh
conda activate mechfoundry-warp117
export PYTHONPATH=$(cd "$HERE/../.." && pwd)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TAG=$(echo "$OPTS" | sed -n 's/.*--tag \([^ ]*\).*/_\1/p')
for N in "$@"; do
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for _ in $(seq 720); do
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
    [ "$BUSY" -eq 0 ] && break
    sleep 5
  done
  if [ "$BUSY" -ne 0 ]; then
    echo "$(date +%T) skip $N: GPU $GPU still busy after an hour" >> "$LOGS/queue_gpu$GPU.txt"
    continue
  fi
  echo "$(date +%T) start $N $OPTS (busy=$BUSY)" >> "$LOGS/queue_gpu$GPU.txt"
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 "$TMO" python "$HERE/run.py" "$N" $OPTS \
      > "$LOGS/$N$TAG.log" 2>&1
  echo "$(date +%T) end $N exit=$?" >> "$LOGS/queue_gpu$GPU.txt"
done
echo "$(date +%T) QUEUE DONE" >> "$LOGS/queue_gpu$GPU.txt"

#!/bin/bash
# Usage: monitor.sh <PID>
# Polls VmHWM (peak RSS) and VmRSS (current RSS) every 1s from /proc
# Also shows cgroup memory usage for container-wide visibility
PID=$1
while kill -0 "$PID" 2>/dev/null; do
  awk '/VmRSS|VmHWM/ {printf "%s %s %s  ", $1, $2, $3}' /proc/$PID/status
  if [ -f /sys/fs/cgroup/memory.current ]; then
    CGROUP_MB=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))
    CGROUP_MAX_MB=$(( $(cat /sys/fs/cgroup/memory.max) / 1048576 ))
    printf "cgroup: %sMB/%sMB" "$CGROUP_MB" "$CGROUP_MAX_MB"
  fi
  echo ""
  sleep 1
done
echo "Process $PID exited"

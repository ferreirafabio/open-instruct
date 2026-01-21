#!/usr/bin/env bash
# Monitor RAM usage for a SLURM job
# Usage: ./monitor_job_ram.sh <job_id> [interval_seconds]

JOB_ID="${1:-}"
INTERVAL="${2:-10}"

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id> [interval_seconds]"
    echo "Example: $0 27012572_1 5"
    exit 1
fi

echo "Monitoring RAM for job $JOB_ID every ${INTERVAL}s (Ctrl+C to stop)"
echo "=============================================="

while true; do
    # Get job info
    JOB_INFO=$(squeue -j "$JOB_ID" -h -o "%T %N" 2>/dev/null)

    if [ -z "$JOB_INFO" ]; then
        echo "$(date '+%H:%M:%S') Job $JOB_ID not found or completed"
        break
    fi

    STATE=$(echo "$JOB_INFO" | awk '{print $1}')
    NODE=$(echo "$JOB_INFO" | awk '{print $2}')

    if [ "$STATE" == "PENDING" ]; then
        echo "$(date '+%H:%M:%S') Job pending..."
    elif [ "$STATE" == "RUNNING" ]; then
        # Get memory stats
        MEM_STATS=$(sstat -j "$JOB_ID" --format=MaxRSS,AvgRSS -n 2>/dev/null | tail -1)
        MAX_RSS=$(echo "$MEM_STATS" | awk '{print $1}')
        AVG_RSS=$(echo "$MEM_STATS" | awk '{print $2}')

        echo "$(date '+%H:%M:%S') Node: $NODE | MaxRSS: $MAX_RSS | AvgRSS: $AVG_RSS"
    else
        echo "$(date '+%H:%M:%S') Job state: $STATE"
    fi

    sleep "$INTERVAL"
done

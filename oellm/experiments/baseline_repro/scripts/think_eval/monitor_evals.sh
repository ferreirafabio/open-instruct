#!/bin/bash
# Monitor Think SFT v2 evaluation jobs, detect failures, and auto-resubmit
#
# Usage:
#   bash oellm/experiments/think_v2_checkpoint_eval/monitor_evals.sh          # one-shot status
#   bash oellm/experiments/think_v2_checkpoint_eval/monitor_evals.sh --watch  # loop every 5 min
#   bash oellm/experiments/think_v2_checkpoint_eval/monitor_evals.sh --resubmit  # resubmit failed jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/run_single_eval.sh"
LOG_DIR="$SCRIPT_DIR/logs"
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
RESULTS_ROOT="$PROJECT_ROOT/oellm/evaluations/benchmarks/OpenJury/results"

WATCH=false
RESUBMIT=false
INTERVAL=300  # seconds between checks in watch mode

for arg in "$@"; do
    case $arg in
        --watch) WATCH=true ;;
        --resubmit) RESUBMIT=true ;;
        --interval=*) INTERVAL="${arg#*=}" ;;
    esac
done

# All expected evaluations: step, eval_mode, swap_mode
STEPS=(500 1000 2000 3000 4000 5000 7000 8000 11000 13000 15000 17000 19000 21000 24000 27000 31000 34000 38000 42856)
# step42856 already has winrate-both and rubric results
EXPECTED_JOBS=()
for STEP in "${STEPS[@]}"; do
    if [ "$STEP" -eq 42856 ]; then
        EXPECTED_JOBS+=("42856:winrate:fixed")
    else
        EXPECTED_JOBS+=("${STEP}:winrate:fixed" "${STEP}:winrate:both" "${STEP}:rubric:fixed")
    fi
done

check_result_exists() {
    local step=$1 eval_mode=$2 swap_mode=$3
    # Check if result directory with matching pattern exists and has results JSON
    local pattern="think-v2-curve-${eval_mode}-${swap_mode}-step${step}-"
    for d in "$RESULTS_ROOT"/${pattern}*/; do
        [ -d "$d" ] || continue
        if ls "$d"/*/results-*.json 1>/dev/null 2>&1; then
            # Check we have results for both alpaca-eval and arena-hard
            local count
            count=$(find "$d" -name "results-*.json" | wc -l)
            if [ "$count" -ge 2 ]; then
                return 0  # complete
            fi
        fi
    done

    # Also check step42856 horeka results
    if [ "$step" -eq 42856 ]; then
        if [ "$eval_mode" == "winrate" ] && [ "$swap_mode" == "both" ]; then
            [ -d "$RESULTS_ROOT/horeka-winrate-Olmo-3-7B-Think-SFT-20260224_141545" ] && return 0
        fi
        if [ "$eval_mode" == "rubric" ]; then
            [ -d "$RESULTS_ROOT/horeka-rubric-Olmo-3-7B-Think-SFT-20260224_161039" ] && return 0
        fi
    fi

    return 1  # not found
}

find_slurm_job() {
    local step=$1 eval_mode=$2 swap_mode=$3
    # Map to job name prefix: v2-{e}{s}-s{step} where e=w/r, s=f/b
    local e_char="${eval_mode:0:1}"
    local s_char="${swap_mode:0:1}"
    local job_pattern="v2-${e_char}${s_char}-s${step}"

    # Check squeue for running/pending
    local state
    state=$(squeue -u "$USER" --format="%j %T" --noheader 2>/dev/null | grep "^${job_pattern} " | awk '{print $2}' | head -1)
    if [ -n "$state" ]; then
        echo "$state"
        return
    fi

    # Check sacct for recent completed/failed (last 48h)
    state=$(sacct -S "$(date -d '48 hours ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)" -u "$USER" --format="JobName%30,State%12" --noheader 2>/dev/null | grep "^ *${job_pattern} " | tail -1 | awk '{print $2}')
    if [ -n "$state" ]; then
        echo "$state"
        return
    fi

    echo "NONE"
}

diagnose_failure() {
    local step=$1 eval_mode=$2 swap_mode=$3
    local e_char="${eval_mode:0:1}"
    local s_char="${swap_mode:0:1}"
    local job_pattern="v2-${e_char}${s_char}-s${step}"

    # Find the log file for this job
    local job_id
    job_id=$(sacct -S "$(date -d '48 hours ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)" -u "$USER" --format="JobID,JobName%30,State%12" --noheader 2>/dev/null | grep "${job_pattern}" | grep "FAILED" | tail -1 | awk '{print $1}')

    if [ -z "$job_id" ]; then
        echo "unknown (no job found)"
        return
    fi

    local log_file="$LOG_DIR/eval_${job_id}.log"
    if [ ! -f "$log_file" ]; then
        echo "log missing (eval_${job_id}.log deleted?)"
        return
    fi

    # Check common errors
    if grep -q "max_position_embeddings" "$log_file" 2>/dev/null; then
        echo "config error: max_position_embeddings is None"
    elif grep -q "invalid choice.*rubric" "$log_file" 2>/dev/null; then
        echo "invalid swap_mode: rubric"
    elif grep -q "oom_kill\|OOM\|CUDA out of memory" "$log_file" 2>/dev/null; then
        echo "OOM"
    elif grep -q "Traceback" "$log_file" 2>/dev/null; then
        local err
        err=$(tail -1 "$log_file" | head -c 80)
        echo "error: $err"
    else
        echo "unknown (check eval_${job_id}.log)"
    fi
}

run_check() {
    local n_complete=0 n_running=0 n_pending=0 n_failed=0 n_missing=0
    local failed_jobs=()

    echo "=============================================="
    echo "Think SFT v2 Eval Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="
    echo ""

    printf "  %-8s %-10s %-8s %-12s %s\n" "STEP" "EVAL_MODE" "SWAP" "STATUS" "DETAIL"
    printf "  %s\n" "------------------------------------------------------"

    for job_spec in "${EXPECTED_JOBS[@]}"; do
        IFS=: read -r step eval_mode swap_mode <<< "$job_spec"

        # First check if results already exist
        if check_result_exists "$step" "$eval_mode" "$swap_mode"; then
            printf "  %-8s %-10s %-8s %-12s\n" "$step" "$eval_mode" "$swap_mode" "COMPLETE"
            n_complete=$((n_complete + 1))
            continue
        fi

        # Check SLURM job state
        local state
        state=$(find_slurm_job "$step" "$eval_mode" "$swap_mode")

        case "$state" in
            RUNNING)
                printf "  %-8s %-10s %-8s %-12s\n" "$step" "$eval_mode" "$swap_mode" "RUNNING"
                n_running=$((n_running + 1))
                ;;
            PENDING)
                printf "  %-8s %-10s %-8s %-12s\n" "$step" "$eval_mode" "$swap_mode" "PENDING"
                n_pending=$((n_pending + 1))
                ;;
            FAILED)
                local diagnosis
                diagnosis=$(diagnose_failure "$step" "$eval_mode" "$swap_mode")
                printf "  %-8s %-10s %-8s %-12s %s\n" "$step" "$eval_mode" "$swap_mode" "FAILED" "$diagnosis"
                n_failed=$((n_failed + 1))
                failed_jobs+=("$job_spec")
                ;;
            COMPLETED)
                # Job completed but no results found — partial completion?
                printf "  %-8s %-10s %-8s %-12s %s\n" "$step" "$eval_mode" "$swap_mode" "PARTIAL?" "job done but results missing"
                n_failed=$((n_failed + 1))
                failed_jobs+=("$job_spec")
                ;;
            NONE)
                printf "  %-8s %-10s %-8s %-12s\n" "$step" "$eval_mode" "$swap_mode" "NOT_SUBMITTED"
                n_missing=$((n_missing + 1))
                failed_jobs+=("$job_spec")
                ;;
            *)
                printf "  %-8s %-10s %-8s %-12s\n" "$step" "$eval_mode" "$swap_mode" "$state"
                ;;
        esac
    done

    echo ""
    echo "  Summary: $n_complete complete, $n_running running, $n_pending pending, $n_failed failed, $n_missing not submitted"
    echo "  Total expected: ${#EXPECTED_JOBS[@]}"
    echo ""

    # Resubmit failed/missing jobs if requested
    if [ "$RESUBMIT" = true ] && [ ${#failed_jobs[@]} -gt 0 ]; then
        echo "  Resubmitting ${#failed_jobs[@]} failed/missing jobs..."
        for job_spec in "${failed_jobs[@]}"; do
            IFS=: read -r step eval_mode swap_mode <<< "$job_spec"
            local e_char="${eval_mode:0:1}"
            local s_char="${swap_mode:0:1}"
            local job_id
            job_id=$(sbatch --job-name="v2-${e_char}${s_char}-s${step}" "$EVAL_SCRIPT" "$step" "$eval_mode" "$swap_mode" | awk '{print $NF}')
            echo "    Resubmitted: step${step} ${eval_mode} ${swap_mode} -> job ${job_id}"
        done
        echo ""
    fi

    # Return non-zero if not all complete
    if [ $((n_complete)) -eq ${#EXPECTED_JOBS[@]} ]; then
        echo "  All evaluations complete!"
        return 0
    fi
    return 1
}

if [ "$WATCH" = true ]; then
    echo "Watching every ${INTERVAL}s (Ctrl+C to stop)..."
    echo ""
    while true; do
        clear
        run_check || true
        echo "  Next check in ${INTERVAL}s..."
        sleep "$INTERVAL"
    done
else
    run_check || true
fi

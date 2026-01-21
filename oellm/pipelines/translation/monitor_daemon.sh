#!/bin/bash
# Lightweight background monitor for translation progress
#
# Usage:
#   ./monitor_daemon.sh start    # Start monitoring in background
#   ./monitor_daemon.sh stop     # Stop monitoring
#   ./monitor_daemon.sh status   # Check if running
#   ./monitor_daemon.sh once     # Run once and exit
#
# Output: STATUS.md updated every 2 minutes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
PID_FILE="$SCRIPT_DIR/.monitor.pid"
LOG_FILE="$SCRIPT_DIR/monitor_daemon.log"
STATUS_FILE="$SCRIPT_DIR/STATUS.md"
INTERVAL=120  # seconds between updates

run_monitor() {
    while true; do
        # Update status file
        cd "$PROJECT_ROOT"
        python oellm/pipelines/translation/monitor_status.py --output "$STATUS_FILE" 2>/dev/null

        # Log update time
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status updated" >> "$LOG_FILE"

        # Keep log file small (last 100 lines)
        if [ -f "$LOG_FILE" ] && [ $(wc -l < "$LOG_FILE") -gt 100 ]; then
            tail -50 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
        fi

        sleep $INTERVAL
    done
}

start_daemon() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Monitor already running (PID $pid)"
            return 1
        fi
    fi

    # Run in background
    nohup bash -c "$(declare -f run_monitor); run_monitor" > /dev/null 2>&1 &
    echo $! > "$PID_FILE"
    echo "Monitor started (PID $!)"
    echo "Status file: $STATUS_FILE"
    echo "Log file: $LOG_FILE"
}

stop_daemon() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "$PID_FILE"
            echo "Monitor stopped (PID $pid)"
        else
            rm -f "$PID_FILE"
            echo "Monitor was not running (stale PID file removed)"
        fi
    else
        echo "Monitor is not running"
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Monitor is running (PID $pid)"
            echo "Last update: $(stat -c %y "$STATUS_FILE" 2>/dev/null | cut -d. -f1)"
            return 0
        fi
    fi
    echo "Monitor is not running"
    return 1
}

run_once() {
    cd "$PROJECT_ROOT"
    python oellm/pipelines/translation/monitor_status.py --output "$STATUS_FILE"
    echo "Status updated: $STATUS_FILE"
}

case "${1:-once}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        check_status
        ;;
    once)
        run_once
        ;;
    *)
        echo "Usage: $0 {start|stop|status|once}"
        exit 1
        ;;
esac

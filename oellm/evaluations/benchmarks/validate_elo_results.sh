#!/bin/bash
# Validate Elo evaluation results by checking judge response counts.
# A valid run should have ~2000+ judge responses (200 battles/lang × 10-12 langs × ~1 judge call each).
# Corrupt cached runs have <100 judge responses and produce garbage Elo ratings (~200 instead of ~700+).
#
# Usage:
#   bash oellm/evaluations/benchmarks/validate_elo_results.sh oellm/evaluations/logs/elo_q35_*.log
#   bash oellm/evaluations/benchmarks/validate_elo_results.sh oellm/evaluations/logs/elo_q35_noen_*.log

MIN_JUDGE_RESPONSES=1000  # Minimum expected judge responses for a valid run

echo "Validating Elo results..."
echo "Min judge responses threshold: $MIN_JUDGE_RESPONSES"
echo ""
echo "Log | Judge Responses | Elo | Status"
echo "----|-----------------|-----|-------"

for LOG in "$@"; do
    BASENAME=$(basename "$LOG")
    JUDGE_COUNT=$(grep -c "200 OK" "$LOG" 2>/dev/null || echo 0)
    Elo=$(grep "<-----" "$LOG" 2>/dev/null | grep -oP '[-\d.]+ ± [\d.]+')
    COMPLETE=$(grep -ci "elo estimation complete" "$LOG" 2>/dev/null || echo 0)

    if [ "$COMPLETE" -eq 0 ]; then
        echo "$BASENAME | $JUDGE_COUNT | N/A | INCOMPLETE"
    elif [ "$JUDGE_COUNT" -lt "$MIN_JUDGE_RESPONSES" ]; then
        echo "$BASENAME | $JUDGE_COUNT | $Elo | ⚠ SUSPECT (too few judge responses)"
    else
        echo "$BASENAME | $JUDGE_COUNT | $Elo | ✓ VALID"
    fi
done

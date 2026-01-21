#!/bin/bash
# Visualize GPU fragmentation across SLURM nodes
# Usage: ./slurm_gpu_fragmentation.sh [partition]

# If partition provided as argument, use it directly
if [[ -n "$1" ]]; then
    PARTITION="$1"
else
    # Get list of GPU partitions (partitions with gres containing 'gpu')
    echo "Fetching available GPU partitions..."
    mapfile -t PARTITIONS < <(sinfo -o "%P %G" --noheader 2>/dev/null | grep -i gpu | awk '{print $1}' | tr -d '*' | sort -u)

    if [[ ${#PARTITIONS[@]} -eq 0 ]]; then
        echo "Error: No GPU partitions found."
        exit 1
    fi

    # Check if fzf is available for nicer selection
    if command -v fzf &> /dev/null; then
        echo ""
        PARTITION=$(printf '%s\n' "${PARTITIONS[@]}" | fzf --prompt="Select partition: " --height=~50% --reverse)
        if [[ -z "$PARTITION" ]]; then
            echo "No partition selected. Exiting."
            exit 0
        fi
    else
        # Fallback to bash select menu
        echo ""
        echo "Available GPU partitions:"
        echo "-------------------------"
        PS3="Select partition (enter number): "
        select PARTITION in "${PARTITIONS[@]}" "Quit"; do
            if [[ "$PARTITION" == "Quit" ]]; then
                echo "Exiting."
                exit 0
            elif [[ -n "$PARTITION" ]]; then
                break
            else
                echo "Invalid selection. Try again."
            fi
        done
    fi
fi

echo ""
echo "========================================"
echo "  GPU Fragmentation Report"
echo "  Partition: $PARTITION"
echo "  Time: $(date)"
echo "========================================"
echo ""

# Get node info with GPU details
echo "Per-Node GPU Status:"
echo "--------------------"
printf "%-20s %-12s %-10s %s\n" "NODE" "STATE" "USED/TOTAL" "VISUALIZATION"
echo ""

# Parse sinfo to get node-level GPU info
sinfo -p "$PARTITION" -N --noheader -O NodeHost,StateLong,Gres,GresUsed 2>/dev/null | while read -r node state gres gres_used; do
    # Extract total GPUs (e.g., "gpu:H200:8" -> 8)
    total_gpus=$(echo "$gres" | grep -oP 'gpu[^:]*:\K\d+' | head -1)
    total_gpus=${total_gpus:-0}

    # Extract used GPUs (e.g., "gpu:H200:3" -> 3)
    used_gpus=$(echo "$gres_used" | grep -oP 'gpu[^:]*:\K\d+' | head -1)
    used_gpus=${used_gpus:-0}

    # Calculate free GPUs
    free_gpus=$((total_gpus - used_gpus))

    # Create visual representation
    visual=""
    for ((i=0; i<used_gpus; i++)); do
        visual+="█"  # Used GPU
    done
    for ((i=0; i<free_gpus; i++)); do
        visual+="░"  # Free GPU
    done

    # Color based on state
    if [[ "$state" == *"idle"* ]]; then
        state_display="idle"
    elif [[ "$state" == *"mix"* ]]; then
        state_display="mixed"
    elif [[ "$state" == *"alloc"* ]]; then
        state_display="allocated"
    elif [[ "$state" == *"drain"* ]]; then
        state_display="drain"
    else
        state_display="$state"
    fi

    printf "%-20s %-12s %d/%-8d [%s]\n" "$node" "$state_display" "$used_gpus" "$total_gpus" "$visual"
done

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

# Calculate totals
totals=$(sinfo -p "$PARTITION" -N --noheader -O Gres,GresUsed 2>/dev/null | awk '
    BEGIN { total=0; used=0 }
    {
        # Extract total GPUs
        match($1, /gpu[^:]*:([0-9]+)/, t)
        if (t[1] != "") total += t[1]

        # Extract used GPUs
        match($2, /gpu[^:]*:([0-9]+)/, u)
        if (u[1] != "") used += u[1]
    }
    END { print total, used }
')

total_gpus=$(echo "$totals" | awk '{print $1}')
used_gpus=$(echo "$totals" | awk '{print $2}')
free_gpus=$((total_gpus - used_gpus))

# Count nodes by state
idle_nodes=$(sinfo -p "$PARTITION" -N --noheader -t idle 2>/dev/null | wc -l)
mixed_nodes=$(sinfo -p "$PARTITION" -N --noheader -t mix 2>/dev/null | wc -l)
allocated_nodes=$(sinfo -p "$PARTITION" -N --noheader -t alloc 2>/dev/null | wc -l)
total_nodes=$((idle_nodes + mixed_nodes + allocated_nodes))

echo ""
if [[ $total_gpus -gt 0 ]]; then
    echo "Total GPUs:     $total_gpus"
    echo "Used GPUs:      $used_gpus"
    echo "Free GPUs:      $free_gpus"
    echo "Utilization:    $(awk "BEGIN {printf \"%.1f\", ($used_gpus/$total_gpus)*100}")%"
else
    echo "No GPU information available for this partition."
fi
echo ""
echo "Node States:"
echo "  Idle (all free):      $idle_nodes"
echo "  Mixed (fragmented):   $mixed_nodes"
echo "  Allocated (all used): $allocated_nodes"
echo ""

# Show largest contiguous free blocks
echo "========================================"
echo "  Largest Contiguous GPU Blocks"
echo "========================================"
echo ""
echo "These are the largest single-node GPU allocations you can request:"
echo ""

sinfo -p "$PARTITION" -N --noheader -O NodeHost,StateLong,Gres,GresUsed 2>/dev/null | while read -r node state gres gres_used; do
    total_gpus=$(echo "$gres" | grep -oP 'gpu[^:]*:\K\d+' | head -1)
    total_gpus=${total_gpus:-0}
    used_gpus=$(echo "$gres_used" | grep -oP 'gpu[^:]*:\K\d+' | head -1)
    used_gpus=${used_gpus:-0}
    free_gpus=$((total_gpus - used_gpus))

    if [[ $free_gpus -gt 0 ]]; then
        echo "  $node: $free_gpus GPUs free"
    fi
done | sort -t: -k2 -rn | head -10

echo ""
echo "Legend: █ = Used GPU, ░ = Free GPU"
echo ""

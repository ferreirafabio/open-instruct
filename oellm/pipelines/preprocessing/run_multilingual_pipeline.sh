#!/bin/bash
# Full multilingual preprocessing pipeline orchestrator
# Chains all steps with SLURM dependencies for maximum parallelism:
#
#   Step 1: Profile (4 nodes)  ──┐
#   Step 2: Preprocess (4 nodes) ├──▶ Step 3: Split (4 nodes) ──▶ Step 4: Assemble ──▶ Step 5: Tokenize
#   (profile runs in parallel    │
#    with preprocess — profile   │
#    is informational only)      │
#
# Usage:
#   bash oellm/pipelines/preprocessing/run_multilingual_pipeline.sh
#   CONFIG=oellm/configs/multilingual_trackA_90en.yaml bash oellm/pipelines/preprocessing/run_multilingual_pipeline.sh

set -euo pipefail

PROJECT_ROOT="/work/dlclarge2/ferreira-oellm/open-instruct"
SCRIPT_DIR="$PROJECT_ROOT/oellm/pipelines/preprocessing"
CONFIG="${CONFIG:-$PROJECT_ROOT/oellm/configs/multilingual_trackA_90en.yaml}"

echo "=============================================="
echo "MULTILINGUAL PREPROCESSING PIPELINE"
echo "=============================================="
echo "Config: $CONFIG"
echo ""

# Step 1 + 2: Profile and Preprocess in parallel (independent)
echo "--- Step 1: Profiling (4 nodes, informational) ---"
PROFILE_JOB=$(sbatch --parsable "$SCRIPT_DIR/profile_multilingual_slurm.sh")
echo "  Profile job: $PROFILE_JOB"

echo "--- Step 2: Preprocessing (4 nodes) ---"
PREPROCESS_JOB=$(sbatch --parsable "$SCRIPT_DIR/preprocess_multilingual_slurm.sh")
echo "  Preprocess job: $PREPROCESS_JOB"

# Step 1b: Merge profiles after profiling completes
echo "--- Step 1b: Merge profiles (after profiling) ---"
MERGE_JOB=$(sbatch --parsable --dependency=afterok:$PROFILE_JOB "$SCRIPT_DIR/merge_profiles.sh")
echo "  Merge job: $MERGE_JOB (depends on $PROFILE_JOB)"

# Step 3: Split by language after preprocessing completes
echo "--- Step 3: Split by language (4 nodes, after preprocessing) ---"
SPLIT_JOB=$(sbatch --parsable --dependency=afterok:$PREPROCESS_JOB "$SCRIPT_DIR/split_multilingual_slurm.sh")
echo "  Split job: $SPLIT_JOB (depends on $PREPROCESS_JOB)"

# Step 4: Assemble mixture after splitting completes
echo "--- Step 4: Assemble mixture (after splitting) ---"
ASSEMBLE_JOB=$(sbatch --parsable \
    --dependency=afterok:$SPLIT_JOB \
    --partition=alldlc2_cpu-epyc9655 \
    --cpus-per-task=16 \
    --mem=64G \
    --time=1:00:00 \
    --job-name=assemble-mixture \
    --output="$SCRIPT_DIR/logs/assemble_%j.log" \
    --wrap="
        set -euo pipefail
        $PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/oellm/pipelines/preprocessing/assemble_mixture.py \
            --config $CONFIG \
            --by-language-dir $PROJECT_ROOT/data/datasets_multilingual_sft/by_language \
            --output-dir $PROJECT_ROOT/data/datasets_multilingual_sft/assembled
    ")
echo "  Assemble job: $ASSEMBLE_JOB (depends on $SPLIT_JOB)"

# Step 5: Tokenize after assembly completes
CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
# Extract experiment name from YAML
EXPERIMENT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['name'])")

echo "--- Step 5: Tokenize (GPU, after assembly) ---"
TOKENIZE_JOB=$(sbatch --parsable \
    --dependency=afterok:$ASSEMBLE_JOB \
    --partition=alldlc2_gpu-h200 \
    --gpus=1 \
    --time=2:00:00 \
    --job-name=tokenize-multilingual \
    --output="$SCRIPT_DIR/logs/tokenize_%j.log" \
    --wrap="
        set -euo pipefail
        export HF_HOME=$PROJECT_ROOT/models/huggingface
        export PYTHONPATH=/work/dlclarge2/ferreira-oellm/OLMo-core/src:\${PYTHONPATH:-}
        source $PROJECT_ROOT/.venv/bin/activate
        python $PROJECT_ROOT/scripts/data/convert_sft_data_for_olmocore.py \
            --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
            --dataset_mixer_list $PROJECT_ROOT/data/datasets_multilingual_sft/assembled/${EXPERIMENT_NAME}.parquet 1.0 \
            --output_dir $PROJECT_ROOT/data/datasets_multilingual_sft/tokenized/${EXPERIMENT_NAME} \
            --chat_template_name olmo \
            --max_seq_length 32768
    ")
echo "  Tokenize job: $TOKENIZE_JOB (depends on $ASSEMBLE_JOB)"

echo ""
echo "=============================================="
echo "PIPELINE SUBMITTED"
echo "=============================================="
echo ""
echo "Job chain:"
echo "  Profile:    $PROFILE_JOB (4 tasks)"
echo "  Merge:      $MERGE_JOB (after profile)"
echo "  Preprocess: $PREPROCESS_JOB (4 tasks, parallel with profile)"
echo "  Split:      $SPLIT_JOB (4 tasks, after preprocess)"
echo "  Assemble:   $ASSEMBLE_JOB (after split)"
echo "  Tokenize:   $TOKENIZE_JOB (after assemble)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep -E 'profile|preprocess|split|assemble|tokenize'"
echo ""
echo "After tokenize completes, train with:"
echo "  EXPERIMENT=$EXPERIMENT_NAME sbatch oellm/train/train_multilingual_sft_slurm.sh"

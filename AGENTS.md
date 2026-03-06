# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.

# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Feel free to ask questions whenever you encounter uncertainty.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`

# oellm Project Structure
  - `oellm/experiments/` - All experiments, each with `scripts/` and `results/` subdirs
    - `baseline_repro/` - Baseline reproduction (Think + Instruct SFT)
    - `dolci_distribution/` - Dolci language distribution analysis
    - `multilingual_eu/` - Multilingual EU fine-tuning (Track A/B)
    - `english_control/` - C0-100en English control experiment
  - `oellm/evaluations/` - Shared evaluation infrastructure (benchmarks, manual testing)
  - `oellm/dataset_selection/` - Dataset download and preprocessing
  - `oellm/translation/` - EU multilingual translation pipeline
  - `oellm/horeka/` - HoreKa HPC utilities (ssh, transfer, rsync)
  - Logs: per-experiment `logs/` dirs and `oellm/evaluations/logs/`

  # Key Paths
  - Project root: `/work/dlclarge2/ferreira-oellm/open-instruct`
  - Checkpoints: `checkpoints/ferreira/olmo3-7b-sft/`
  - Baselines: `models/baselines/`
  - HuggingFace cache: `models/huggingface/` (set HF_HOME)
  - Preprocessed data: `data/datasets_mixture_sft_preprocessed/`
  - Tokenized data: `data/datasets_mixture_sft_tokenized/`
  - Translated data: `data/datasets_eu24_sft_preprocessed/`

  # Experiment History

  ## dolci (baseline reproduction)
  - Path: `checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf/` and `dolci-instruct-sft-hf/`
  - Purpose: Reproduce the official OLMo-3-7B-Think-SFT and Instruct-SFT baselines

  ## dataset_mixtures (data scaling experiment)
  - Path: `data/datasets_mixture_sft_preprocessed/` (19 datasets, ~50GB total)
  - Purpose: Test if fine-tuning instruct checkpoints benefits from MORE data
  - Includes nvidia-Nemotron (42GB, ~6M samples) - the largest single dataset
  - Key insight: Original tokenization script OOM'd on 12.8M samples; fixed with chunked processing
  - Config: `oellm/dataset_selection/mixture_all.yaml`

  ## dolci language distribution analysis
  - Script: `oellm/experiments/dolci_distribution/scripts/detect_language_distribution.py`
  - Results: `oellm/experiments/dolci_distribution/results/`
  - Findings: Both Dolci-Instruct-SFT (2.15M) and Dolci-Think-SFT-7B (2.27M) are ~93-94% English
  - ~50 non-English languages detected in long tail
  - Run: `sbatch oellm/experiments/dolci_distribution/scripts/run_language_analysis.sh [instruct|think]` (64 array shards)
  - Merge: `sbatch --dependency=afterok:$JOB_ID oellm/experiments/dolci_distribution/scripts/merge_results.sh both`

  ## eu24 (multilingual fine-tuning) - IN PROGRESS
  - Path: `data/datasets_eu24_sft_preprocessed/`
  - Purpose: Fine-tune on translated data for 24 EU languages
  - Translation model: `facebook/nllb-200-distilled-600M` (chosen for 3.7x speed advantage)
    - 600M: 1,541 examples/min vs 3.3B: 422 examples/min
    - Quality trade-off acceptable for high/medium-resource EU languages
    - Can re-translate specific languages with 3.3B if evaluation shows issues
  - Data strategy: 10% sampling for fast iteration
  - Run: `SAMPLE=0.1 MODEL=facebook/nllb-200-distilled-600M sbatch oellm/translation/translate_slurm.sh`
  - Evaluation: m-arena-hard-EU benchmark (covers 12 of 24 EU languages)

  # HuggingFace Downloads
  IMPORTANT: All HuggingFace downloads must go to open-instruct, NOT home directory!
  Always set these environment variables in scripts:
  ```bash
  export HF_HOME="/work/dlclarge2/ferreira-oellm/open-instruct/models/huggingface"
  export HF_DATASETS_CACHE="/work/dlclarge2/ferreira-oellm/open-instruct/data/huggingface"
  ```

 # Training
  - OLMo-core expects a single pre-merged numpy dataset
  - Fine-tuning learning rate: start with 1e-5 (conservative: 5e-6, aggressive: 2e-5)
  - Convert checkpoints to HF format: `sbatch oellm/train/convert_to_hf.sh [instruct|think]`

  # Evaluation (OpenJury)
  - Run: `sbatch oellm/evaluations/benchmarks/run_evaluation.sh [instruct|think] [benchmark] [samples]`
  - Benchmarks: `alpaca-eval`, `arena-hard`, `m-arena-hard-EU` (European languages), `all`
  - Max tokens set to 32768 in run_evaluation.sh (max_out_tokens and truncate_all_input_chars)
  - Results: `oellm/evaluations/benchmarks/results/`

  # Relevant SLURM Partitions
  - GPU: `alldlc2_gpu-h200` # can also be used for CPU heavy tasks since it has fast CPUs
  - CPU: `bosch_cpu-cascadelake` # for non-gpu tasks, in case h200's are occupied
  - CPU: `alldlc2_cpu-epyc9655` # fast AMD EPYC CPUs, used for language analysis (48 cores/128G)

  # Model Baselines
  - Download instruct baseline: `sbatch oellm/evaluations/download_instruct_baseline.sh`
  - Download think baseline: `sbatch oellm/evaluations/download_think_baseline.sh`

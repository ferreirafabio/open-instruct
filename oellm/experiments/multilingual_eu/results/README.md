# Multilingual EU Fine-tuning Experiment

## Goal
Improve OLMo-3-7B EU language performance via continued training on multilingual data from pre-existing translated datasets (no new translations needed).

## Hypothesis
Adding 10-30% non-English EU data to continued SFT improves m-arena-hard-EU scores without degrading English (arena-hard) performance.

## Language Strategy

### m-arena-hard-EU benchmark (12 languages)

| Code | Language | Role |
|------|----------|------|
| en | English | Baseline (regression check) |
| de | German | Training language |
| es | Spanish | Training language |
| fr | French | Training language |
| it | Italian | Training language |
| pt | Portuguese | Training language |
| pl | Polish | Training language |
| nl | Dutch | Training language |
| cs | Czech | Training language |
| ro | Romanian | Held-out (zero-shot) — Latin script, Romance family, expects transfer from fr/es/it/pt |
| el | Greek | Held-out (zero-shot) — Greek script, tests cross-script generalization |
| uk | Ukrainian | In benchmark, not trained on |

- **8 training languages**: de, es, fr, it, pt, pl, nl, cs — equal distribution (EU% / 8 each)
- **2 held-out languages**: ro, el — zero-shot sanity check
- **2 other benchmark languages**: en (baseline), uk (not trained on)

## Datasets

| Dataset | Total | Format | EU Languages | Notes |
|---------|-------|--------|-------------|-------|
| `CohereLabs/fusion-synth-data-ufb` | 94.7k | SFT (prompt + completions) | de, es, fr, it, pt (5) | ~9.5k/lang, `Fusion` completion from 5 teachers |
| `allenai/WildChat-1M` | 838k (after toxic filter) | SFT (conversation) | 74 langs | Real GPT-3.5/4 chats, `toxic` flag filtered |
| `lmsys/lmsys-chat-1m` | 1M convs | SFT (conversation) | 154 langs | Real LLM chats, 25 models |
| `OpenAssistant/oasst2` | 135k msgs | Tree (message_id/parent_id) | 28 langs | Human-reviewed, best-ranked path extracted |

### Available EU Target Language Data (profiled 2026-03-05)

Source: `data/datasets_multilingual_sft/profiles/dataset_profiles.json`

| Language | Code | fusion-synth | WildChat | lmsys-chat | oasst2 | **Total** |
|----------|------|-------------|----------|-----------|--------|-----------|
| English | en | 9,593 | 478,498 | 777,453 | 64,513 | **1,330,057** |
| German | de | 8,943 | 16,865 | 15,317 | 6,145 | **47,270** |
| French | fr | 9,852 | 26,977 | 15,139 | 3,880 | **55,848** |
| Spanish | es | 7,648 | 20,447 | 23,758 | 28,199 | **80,052** |
| Italian | it | 9,782 | 5,120 | 14,362 | 943 | **30,207** |
| Portuguese | pt | 9,859 | 10,386 | 28,616 | 2,699 | **51,560** |
| Polish | pl | — | 3,667 | 3,348 | 435 | **7,450** |
| Dutch | nl | — | 1,730 | 1,080 | 72 | **2,882** |
| Czech | cs | — | 585 | 745 | 12 | **1,342** |
| *Romanian* | *ro* | *—* | *1,037* | *548* | *—* | ***1,585*** |
| *Greek* | *el* | *—* | *622* | *1,031* | *—* | ***1,653*** |

*Italic = held-out languages (not trained on, used for zero-shot evaluation)*

**Key observations:**
- Track A (94.7k): fusion-synth covers de/es/fr/it/pt well (~8-10k each), but has zero pl/nl/cs data — WildChat and lmsys-chat fill the gap
- Track B (500k): all 8 training languages have sufficient data across sources
- nl and cs are the scarcest (2.9k and 1.3k) — may need upsampling or accept lower counts
- Held-out languages (ro, el) have ~1.5k samples each — enough to verify zero-shot only

## File Structure

```
open-instruct/
├── data/datasets_multilingual_sft/              # All multilingual data
│   ├── profiles/                                # Step 1: dataset profiling output
│   │   ├── profile_fusion-synth.json            #   Per-dataset profiles (parallel-safe)
│   │   ├── profile_wildchat.json
│   │   ├── profile_lmsys-chat.json
│   │   ├── profile_oasst2.json
│   │   └── dataset_profiles.json                #   Merged profiles
│   ├── preprocessed/                            # Step 2: standardized messages format
│   │   ├── CohereLabs-fusion-synth-data-ufb.parquet
│   │   ├── allenai-WildChat-1M.parquet
│   │   ├── lmsys-lmsys-chat-1m-multilingual.parquet
│   │   └── OpenAssistant-oasst2.parquet
│   ├── by_language/                             # Step 3: split by language
│   │   ├── CohereLabs-fusion-synth-data-ufb/
│   │   │   ├── en.parquet
│   │   │   ├── de.parquet
│   │   │   └── ...
│   │   ├── allenai-WildChat-1M/
│   │   ├── lmsys-lmsys-chat-1m-multilingual/
│   │   └── OpenAssistant-oasst2/
│   ├── assembled/                               # Step 4: mixed per experiment config
│   │   ├── A1-90en.parquet
│   │   ├── A1-90en_metadata.json
│   │   └── ...
│   └── tokenized/                               # Step 5: OLMo-core numpy format
│       ├── A1-90en/
│       │   ├── token_ids_part_0.npy
│       │   └── labels_mask_part_0.npy
│       └── ...
│
├── oellm/
│   ├── configs/                                 # YAML mixture configs
│   │   ├── multilingual_trackA_90en.yaml        #   A1: 90/10 en/eu, 94.7k samples
│   │   ├── multilingual_trackA_80en.yaml        #   A2: 80/20
│   │   ├── multilingual_trackA_70en.yaml        #   A3: 70/30
│   │   ├── multilingual_trackB_90en.yaml        #   B1: 90/10, 500k samples
│   │   └── multilingual_trackB_80en.yaml        #   B2: 80/20, 500k samples
│   │
│   ├── pipelines/preprocessing/                 # Data processing scripts
│   │   ├── profile_multilingual_datasets.py     #   Profile datasets for per-language counts
│   │   ├── profile_multilingual_slurm.sh        #   SLURM: 4 parallel profile tasks
│   │   ├── merge_profiles.sh                    #   SLURM: merge per-dataset profiles
│   │   ├── preprocess_datasets.py               #   Transform to messages format (modified)
│   │   ├── preprocess_multilingual_slurm.sh     #   SLURM: 4 parallel preprocess tasks
│   │   ├── split_by_language.py                 #   Split parquets by language column
│   │   ├── split_multilingual_slurm.sh          #   SLURM: 4 parallel split tasks
│   │   ├── assemble_mixture.py                  #   Assemble mixture from YAML config
│   │   └── run_multilingual_pipeline.sh         #   Full pipeline orchestrator
│   │
│   ├── utils/
│   │   └── language_mixer.py                    #   LanguageMixer (extended for arbitrary langs)
│   │
│   ├── train/
│   │   └── train_multilingual_sft_slurm.sh      #   SLURM training script (8x H200)
│   │
│   ├── experiments/multilingual_eu/             #   This experiment
│   │   ├── README.md                            #   This file
│   │   ├── run_single_eval.sh                   #   Evaluate one checkpoint
│   │   ├── run_all_evals.sh                     #   Batch evaluate all checkpoints
│   │   ├── collect_results.py                   #   Aggregate JSON -> CSV
│   │   ├── plot_training_curve.py               #   Training curve plots
│   │   ├── results.csv                          #   Aggregated results (populated later)
│   │   ├── plots/                               #   Visualization outputs
│   │   └── logs/                                #   SLURM job logs
│   │
│   └── tests/
│       └── test_multilingual_preprocessing.py   #   33 tests (transforms, split, assembly, I/O)
│
├── checkpoints/ferreira/olmo3-7b-sft/
│   ├── dolci-instruct-sft-v2-horeka/            # Base checkpoint (OLMo-core format)
│   ├── dolci-instruct-eu-A1-90en/               # Training output
│   └── dolci-instruct-eu-A1-90en-hf/            # HF-converted checkpoint
```

## Pipeline Overview

```
┌───────────────────────────┐  ┌────────────────────────────┐
│ 1. Profile (4 CPU nodes)  │  │ 2. Preprocess (4 CPU nodes)│
│    Per-language counts     │  │    Transform to messages   │
│    (informational)         │  │    + language column       │
└───────────┬───────────────┘  └──────────┬─────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────────┐  ┌────────────────────────────┐
│ 1b. Merge profiles        │  │ 3. Split by lang           │
│     (1 CPU node)           │  │    (4 CPU nodes)           │
└───────────────────────────┘  └──────────┬─────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────────┐
                               │ 4. Assemble mixture         │
                               │    (1 CPU node)             │
                               │    YAML config -> parquet   │
                               └──────────┬─────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────────┐
                               │ 5. Tokenize (1 GPU node)   │
                               │    parquet -> numpy         │
                               └──────────┬─────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────────┐
                               │ 6. Train (8x H200 GPU)     │
                               │    OLMo-core SFT            │
                               └──────────┬─────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────────┐
                               │ 7. Convert to HF + Evaluate │
                               │    m-arena-hard-EU + arena- │
                               │    hard (English regression) │
                               └────────────────────────────┘
```

Steps 1 and 2 run in parallel (profile is informational only).
Steps 2→3→4→5 are chained via SLURM `--dependency=afterok`.

## SLURM Partitions
- **CPU tasks** (profile, preprocess, split, assemble): `alldlc2_cpu-epyc9655` (48 cores, 128G)
- **GPU tasks** (tokenize, train, evaluate): `alldlc2_gpu-h200`

## Experiment Matrix

### Track A: fusion-synth primary (~94.7k samples, ~1.7h train)
| Exp | En/EU | Total | Checkpoint save | Purpose |
|-----|-------|-------|-----------------|---------|
| A1  | 90/10 | 94.7k | every 50 steps | Baseline multilingual signal |
| A2  | 80/20 | 94.7k | every 50 steps | More EU data |
| A3  | 70/30 | 94.7k | every 50 steps | Aggressive EU |

### Track B: All 4 sources (~500k samples, ~9h train)
| Exp | En/EU | Total | Checkpoint save | Purpose |
|-----|-------|-------|-----------------|---------|
| B1  | 90/10 | 500k  | every 200 steps | Scale with more sources |
| B2  | 80/20 | 500k  | every 200 steps | Scale + more EU |

Start with A1→A2→A3. If promising, proceed to B1/B2.

### Training Hyperparameters (from OLMo-3 paper Table 47)
| Parameter | Value |
|-----------|-------|
| Base checkpoint | dolci-instruct-sft-v2-horeka (already SFT'd) |
| Learning rate | 8e-5 |
| Epochs | 2 |
| Sequence length | 32768 |
| Batch size | 1M tokens (seq_len * 32) |
| GPUs | 8x H200 |

## How to Run

### Option A: Full automated pipeline
```bash
# Runs all steps chained with SLURM dependencies
bash oellm/pipelines/preprocessing/run_multilingual_pipeline.sh

# Or with a specific config:
CONFIG=oellm/configs/multilingual_trackA_80en.yaml \
    bash oellm/pipelines/preprocessing/run_multilingual_pipeline.sh
```

### Option B: Step-by-step

#### 1. Profile datasets (4 CPU nodes in parallel)
```bash
JOB_ID=$(sbatch --parsable oellm/pipelines/preprocessing/profile_multilingual_slurm.sh)
sbatch --dependency=afterok:$JOB_ID oellm/pipelines/preprocessing/merge_profiles.sh
# Output: data/datasets_multilingual_sft/profiles/dataset_profiles.json
```

#### 2. Preprocess (4 CPU nodes in parallel)
```bash
sbatch oellm/pipelines/preprocessing/preprocess_multilingual_slurm.sh
# Output: data/datasets_multilingual_sft/preprocessed/*.parquet
```

#### 3. Split by language (4 CPU nodes in parallel, after step 2)
```bash
sbatch --dependency=afterok:$PREPROCESS_JOB \
    oellm/pipelines/preprocessing/split_multilingual_slurm.sh
# Output: data/datasets_multilingual_sft/by_language/<dataset>/<lang>.parquet
```

#### 4. Assemble mixture
```bash
python oellm/pipelines/preprocessing/assemble_mixture.py \
    --config oellm/configs/multilingual_trackA_90en.yaml
# Output: data/datasets_multilingual_sft/assembled/A1-90en.parquet
```

#### 5. Tokenize
```bash
python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --dataset_mixer_list data/datasets_multilingual_sft/assembled/A1-90en.parquet 1.0 \
    --output_dir data/datasets_multilingual_sft/tokenized/A1-90en \
    --chat_template_name olmo --max_seq_length 32768
```

#### 6. Train
```bash
EXPERIMENT=A1-90en sbatch oellm/train/train_multilingual_sft_slurm.sh

# Smoke test first:
TEST_RUN=true EXPERIMENT=A1-90en sbatch oellm/train/train_multilingual_sft_slurm.sh

# Track B (longer, save less often):
EXPERIMENT=B1-90en SAVE_INTERVAL=200 sbatch oellm/train/train_multilingual_sft_slurm.sh
```

#### 7. Convert + Evaluate
```bash
# Convert all intermediate checkpoints to HF format
# (adapt oellm/experiments/think_v2_checkpoint_eval/convert_checkpoints.sh)

# Evaluate
bash oellm/experiments/multilingual_eu/run_all_evals.sh A1-90en

# Collect and plot
python oellm/experiments/multilingual_eu/collect_results.py
python oellm/experiments/multilingual_eu/plot_training_curve.py
```

### Monitoring
```bash
squeue -u $USER | grep -E 'profile|preprocess|split|assemble|tokenize|multilingual'
```

## Output Deliverables (per experiment)
1. **Winrate table**: m-arena-hard-EU + arena-hard (English regression)
2. **Rubric table**: 4 criteria (instruction_following, naturalness, coherence, accuracy)
3. **Training curve plots**: winrate and rubric composite over training steps
4. **Results CSV**: `oellm/experiments/multilingual_eu/results.csv`

## Naming Conventions

### Checkpoints: `dolci-instruct-eu-{track}{id}-{ratio}`
```
checkpoints/ferreira/olmo3-7b-sft/
  dolci-instruct-eu-A1-90en/          # OLMo-core format (with intermediate steps)
  dolci-instruct-eu-A1-90en-hf/       # HF-converted final
  dolci-instruct-eu-A1-90en-hf/step50 # HF-converted intermediate
```

### Results CSV format
```csv
experiment,step,dataset,eval_mode,metric,value,baseline_value,num_battles
A1-90en,50,m-arena-hard-EU,winrate,winrate,0.52,0.50,12000
A1-90en,50,m-arena-hard-EU,rubric,rubric_composite,0.85,0.83,50000
```

## Progress

See **[experiment_progress.md](experiment_progress.md)** for the live logbook tracking all jobs, timings, and known issues.

## Results
*(populated as experiments complete)*

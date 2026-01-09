import argparse
import json
from pathlib import Path

import pandas as pd

"""
TODOs:
* load data from slurmpilot
* plot winrates
"""

DEFAULT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct/oellm/baseline_repro/evaluations/benchmarks/OpenJury/results")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    type=str,
    default=None,
    help="Path to results directory. Defaults to latest in OpenJury/results.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Scan all results directories instead of just the latest.",
)
args = parser.parse_args()

if args.results_dir:
    path = Path(args.results_dir).expanduser()
elif args.all:
    path = DEFAULT_ROOT
else:
    # Default: get latest results directory by modification time
    candidates = [d for d in DEFAULT_ROOT.iterdir() if d.is_dir()] if DEFAULT_ROOT.exists() else []
    if candidates:
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        path = candidates[0]
    else:
        path = DEFAULT_ROOT

result_rows = []
for result in path.rglob("*results-*.json"):
    # Accept any results-*.json file found under the results directory
    if "results" in str(result):
        print(result)
        with open(result, "r") as f:
            res = json.load(f)
            res["winrate"] = float(
                (res["num_wins"] + 0.5 * res["num_ties"])
                / (res["num_ties"] + res["num_wins"] + res["num_losses"])
            )

            result_rows.append(res)

df = pd.DataFrame(result_rows)
print(pd.DataFrame(result_rows).head().to_string())

df_pivot = df.pivot_table(index="model_B", columns="dataset", values="winrate")
# Include parent/name for more context, resolve symlinks
def resolve_short_name(x):
    path_str = x.split("VLLM/", 1)[-1] if "VLLM/" in x else x
    p = Path(path_str)
    if p.exists():
        try:
            p = p.resolve()
        except Exception:
            pass
    parts = p.parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else p.name

df_pivot.index = [resolve_short_name(x) for x in df_pivot.index]

# idx = [
#     "Llama-3.2-1B",
#     "Qwen2.5-1.5B-Instruct",
#     "openeurollm-datamix-2b-en-80pct-SFT-tulu3",
#     "openeurollm-datamix-2b-en-80pct-DPO-HelpSteer3",
# ]
#
# df_pivot = df_pivot.loc[idx, :]
# Use resolved model_A name for baseline
if not df.empty:
    baseline = resolve_short_name(df["model_A"].iloc[0])
else:
    baseline = "instruct-baseline"
df_pivot.loc[baseline] = 0.5
df_pivot["Average"] = df_pivot.mean(axis=1)
df_pivot.sort_values(by="Average", inplace=True)
print((1 - df_pivot).to_string(float_format="%.2f"))

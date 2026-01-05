import json
from pathlib import Path

import pandas as pd

"""
TODOs:
* load data from slurmpilot
* plot winrates
"""

path = Path(
    "/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/benchmarks/OpenJury/results"
).expanduser()

result_rows = []
for result in path.rglob("*results-*.json"):
    if result.parent.parent.name == "results":
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
df_pivot.index = [x.split("/")[-1] for x in df_pivot.index]

# idx = [
#     "Llama-3.2-1B",
#     "Qwen2.5-1.5B-Instruct",
#     "openeurollm-datamix-2b-en-80pct-SFT-tulu3",
#     "openeurollm-datamix-2b-en-80pct-DPO-HelpSteer3",
# ]
#
# df_pivot = df_pivot.loc[idx, :]
baseline = "instruct-baseline"
df_pivot.loc[baseline] = 0.5
df_pivot["Average"] = df_pivot.mean(axis=1)
df_pivot.sort_values(by="Average", inplace=True)
print((1 - df_pivot).to_string(float_format="%.2f"))

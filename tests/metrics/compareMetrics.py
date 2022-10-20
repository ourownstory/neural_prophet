import os
import pandas as pd
import numpy as np


def read_json(path, metrics_path, branch):
    try:
        df = pd.read_json(os.path.join(metrics_path, path), orient="index")
        df = df.reset_index()
        df.rename(columns={0: branch, "index": "Metric"}, inplace=True)
        return df
    except:
        return None


# Load all metrics paths
metrics_path = os.path.join(os.getcwd(), "tests", "metrics")
metrics_path_main = os.path.join(os.getcwd(), "tests", "metrics-main")
metrics_files = [f for f in os.listdir(metrics_path) if ".json" in f]
all_metrics = pd.DataFrame()

# Loop through all metrics files
for f in metrics_files:
    current = read_json(f, metrics_path, "current")
    main = read_json(f, metrics_path_main, "main")

    if main is not None:
        df = pd.merge(current, main, on=["Metric"], how="left")
        df["diff"] = np.where(
            not df["main"] == 0 and not df["current"] == 0, (df["main"] - df["current"]) / df["main"] * 100 * -1, 0
        )
        df = df.round(5)
        df["diff"] = np.where(df["diff"] == 0, 0.0, df["diff"])
        df["emoji"] = np.where(
            df["diff"] >= 0.03, np.where(df["diff"] >= 0.7, ":x:", ":warning:"), ":white_check_mark:"
        )
        df["diff"] = df["diff"].round(2).astype(str) + "%" + " " + df["emoji"]
    else:
        df = current.copy()
        df["main"] = "-"
        df["diff"] = "-"
        df = df.round(4)

    df["Benchmark"] = f.split(".")[0]
    df = df[["Benchmark", "Metric", "current", "main", "diff"]]
    all_metrics = pd.concat([all_metrics, df])

print(str(all_metrics.to_markdown(tablefmt="github", index=False)))

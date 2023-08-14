import sys
import os
import shutil
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from MultiLevelClustering import MultiLevelClustering, MetricGroup
sys.path.append("../")
from common.utils import load_pickle
from common.evaluation import evaluator

# Test cases:
# distance_name: l2 or dtw
# use_partition: True or False
# metric_names: 1, all(11) metrics

params = {
    "distance_name": "l2",
    "nrows": None,
    "plot_fig": False,
    "use_partition": True,
    "normalize": True,
    "label": "label"
}


metric_names = [
    "cpu_util",
    "network_vm_pps_out",
    "disk_write_requests_rate",
    "disk_read_requests_rate",
    "network_vm_bandwidth_out",
    "network_outgoing_bytes_aggregate_rate",
    "network_incoming_bytes_aggregate_rate",
    "network_vm_bandwidth_in",
    "disk_write_bytes_rate",
    "disk_read_bytes_rate",
    "network_vm_pps_in",
]
metric_hash = hashlib.md5(str(metric_names).encode("utf-8")).hexdigest()[0:8]


if __name__ == "__main__":
    metric_infile = f"../data/anonymized_metric.pkl"
    function_label_file = f"../data/anonymized_label.csv"

    df = pd.read_csv(function_label_file, index_col=None)
    metric_format = metric_infile[metric_infile.rfind("."):]
    threshold = 0.1
    metric_dict = load_pickle(metric_infile)
    print(f"Processing {len(metric_dict)} VMs.")
    vm2partition_file = Path(f"../outdir/threshold_{threshold}.pkl")

    partition_list = None
    if params["use_partition"]:
        partition_list = load_pickle(vm2partition_file)

    exp_str = "_".join([f"{k}={v}" for k,v in params.items()]) + f"_[{len(metric_names)}]metrics"
    outdir_root = Path(f"../outdir/{exp_str}")
    exp_records = []
    print(f"Start exp: {exp_str}")

    metric_group = MetricGroup(metric_names, metric_dict, metric_format, params["nrows"])

    for clustering_threshold in [0.002]:
        print(f"Using threshold: {clustering_threshold}.")
        HAC = MultiLevelClustering(
            metric_names=metric_names,
            clustering_threshold=clustering_threshold,
            outdir_root=outdir_root,
            distance_name=params["distance_name"],
            partitions=partition_list,
            normalize=params["normalize"],
            plot_fig=params["plot_fig"],
            nrows=params["nrows"],
        )
        clustering_results, clustering_time = HAC.fit_transform(metric_group)
        if function_label_file:
            homo, comp, v_measure = evaluator(metric_hash+"_"+str(clustering_threshold), outdir_root, function_label_file, params["label"]).evaluate_metrics(clustering_results)
            exp_records.append(
                {
                    "partition": params["use_partition"],
                    "distance": params["distance_name"],
                    "metric_names": len(metric_names),
                    "threshold": clustering_threshold,
                    "homo": homo,
                    "comp": comp,
                    "v_measure": v_measure,
                    "clustering_time": clustering_time
                }
            )

    if function_label_file:
        pd.DataFrame(exp_records).to_csv(f"{outdir_root}/exp_records_{exp_str}.csv", index=False)

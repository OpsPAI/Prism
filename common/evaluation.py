import os
import pandas as pd
from sklearn.metrics.cluster import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score
)


def evaluate_homo_comp(clustering_results, outpath=None, saved=True):
    homo = homogeneity_score(
        clustering_results["Function"], clustering_results["ClusterId"]
    )
    comp = completeness_score(
        clustering_results["Function"], clustering_results["ClusterId"]
    )
    v_measure = v_measure_score(
        clustering_results["Function"], clustering_results["ClusterId"]
    )
    if outpath is not None and saved == True:
        clustering_results.to_csv(outpath, index=False)
    return homo, comp, v_measure


class evaluator:
    def __init__(self, metric_hash, outdir_root, function_label_file, label_column="label", saved=True) -> None:
        self.metric_hash = metric_hash
        self.outdir_root = outdir_root
        self.saved = saved
        if str(function_label_file).split('.')[-1] == 'csv':
            function_label_df = pd.read_csv(function_label_file)
        else:
            function_label_df = pd.read_excel(function_label_file)
        self.vm2function = dict(
            zip(function_label_df["vmid"], function_label_df[label_column])
        )

    def evaluate_metrics(self, vm2cluster_mapping):
        clustering_results = []  # VMID, predicted cluster id, label
        for vmid, class_label in vm2cluster_mapping.items():
            if self.vm2function.get(vmid, None) is not None:
                row = {
                    "VMID": vmid,
                    "ClusterId": class_label,
                    "Function": self.vm2function.get(vmid, None),
                }
                clustering_results.append(row)
        clustering_results = pd.DataFrame(clustering_results)
        print(
            "{}/{} VMs miss labels.".format(
                clustering_results["Function"].isna().sum(), clustering_results.shape[0]
            )
        )
        print("{} clusters generated.".format(clustering_results["ClusterId"].nunique()))
        if not os.path.exists(os.path.join(self.outdir_root, "evaluation")):
            os.makedirs(os.path.join(self.outdir_root, "evaluation"))
        h, c, v = evaluate_homo_comp(
            clustering_results,
            os.path.join(
                self.outdir_root, f"evaluation/clustering_results_{self.metric_hash}.csv"
            ),
            self.saved
        )
        print(
            f"homogeneity_score: {100*h:.2f}%, completeness_score: {100*c:.2f}%, v_measure_score: {100*v:.2f}%",
        )
        return h, c, v

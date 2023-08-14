import os
import sys

import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from collections import defaultdict
from dtaidistance import dtw
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pathlib import Path
from matplotlib import pyplot as plt
from itertools import chain
sys.path.append("../")
from common.utils import (
    save_pickle,
    load_pickle,
    load_dict_from_hdf5,
    padding,
)


class MultiLevelClustering:
    def __init__(
        self,
        metric_names,
        clustering_threshold,
        outdir_root,
        distance_name="l2",
        partitions=None,
        normalize=True,
        plot_fig=False,
        nrows=None,
    ) -> None:

        self.metric_names = metric_names
        self.clustering_threshold = clustering_threshold
        self.dist_func = (
            dtw.distance_matrix_fast if distance_name == "dtw" else cosine_distances
            # dtw.distance_matrix_fast
        )
        self.partitions = partitions
        self.outdir_root = Path(outdir_root)
        self.normalize = normalize
        self.plot_fig = plot_fig
        self.nrows = nrows

        self.cache_dir = outdir_root / "cache_dir"
        self.plot_dir = outdir_root / f"th={clustering_threshold:.3f}" / "plot"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fit_transform(self, metric_group):
        s1 = time.time()
        vm2cluster_mapping = Clustering(
            metric_group,
            self.clustering_threshold,
            dist_func=self.dist_func,
            partitions=self.partitions,
            cache_path=self.cache_dir / "distance_matrix.pkl",
            normalize=True,
        ).fit_transform(metric_group.matrix_dict)
        s2 = time.time()
        clustering_time = s2 - s1
        print(f"Clustering done, time taken: [{clustering_time:.3f} seconds]")

        dump_path = self.outdir_root / "vm2cluster_mapping.pkl"
        save_pickle(vm2cluster_mapping, dump_path)
        print(f"Saving clustering mapping pickle to {dump_path}.")
        return vm2cluster_mapping, clustering_time

    def plot_clustering_results(self, metric_group, vm2cluster_mapping):
        # for each cluster
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        clustering_results = pd.DataFrame(list(vm2cluster_mapping.items()), columns=['VMID', 'ClusterId'])
        print("LEN: ", len(clustering_results))
        if self.plot_fig:
            print("Plotting clusters..")
            for cid, vmlist in tqdm(
                dict(
                    clustering_results.groupby("ClusterId")["VMID"].apply(list)
                ).items()
            ):
                vmidx_list = [metric_group.vm2idx[vmid] for vmid in vmlist]
                fig, ax = plt.subplots(
                    len(self.metric_names),
                    1,
                    figsize=(10, 5 if len(self.metric_names) == 1 else 40),
                )
                if len(self.metric_names) == 1:
                    ax = [ax]

                # for each metric
                for idx, metric_name in enumerate(self.metric_names):
                    data = metric_group.matrix_dict[metric_name][vmidx_list]
                    # plot a subplot
                    num_instance_per_class = len(data)
                    for didx, row in enumerate(data):
                        if didx == 0:
                            ax[idx].plot(row, label=self.metric_names[idx])
                        else:
                            ax[idx].plot(row)
                    ax[idx].legend()

                if num_instance_per_class > 5:
                    plt.title(f"Cluster: {cid} #Instance: {num_instance_per_class}")
                    plt.savefig(
                        os.path.join(
                            self.plot_dir, f"c{cid}-{num_instance_per_class}.png"
                        )
                    )
                    plt.close()
            print("Plotting done.")


class Clustering:
    """
    for each partition ->
        for each metric
            compute distance matrix
        merge_distance
        clustering
    """
    def __init__(
        self,
        metric_group,
        clustering_threshold,
        dist_func,
        partitions=None,
        cache_path="",
        normalize=True,
    ):
        self.metric_group = metric_group
        self.clustering_threshold = clustering_threshold
        self.cache_path = cache_path
        self.dist_func = dist_func

        if partitions is not None:
            self.partitions = partitions  # {k1: group1, k2: group2}
        else:
            self.partitions = {"all": [vmid for vmid in metric_group.vm2idx]}
        self.normalize = normalize
        self.clustering_mapping = {}

    def fit_transform(self, metric_matrix_dict):
        distance_matrix_dict = self.__compute_distance_matrix(metric_matrix_dict)

        print("Start clustering..")
        for part_id, distance_matrix_part in tqdm(distance_matrix_dict.items()):
            distance_matrix_part_merged = self.__merge_multi_dist_matrix(
                distance_matrix_part
            )
            part_vm_list = self.partitions[part_id]
            if len(part_vm_list) > 1:
                clustering_part = self.compute_cluster(distance_matrix_part_merged)
                for idx, lb in enumerate(clustering_part.labels_):
                    class_label = f"{part_id} | class-{lb}"
                    self.clustering_mapping[part_vm_list[idx]] = class_label
            else:
                self.clustering_mapping[part_vm_list[0]] = f"{part_id} | class-0"
        print("Clustering done.")
        return self.clustering_mapping

    def compute_cluster(self, dist_matrix):
        clustering = AgglomerativeClustering(
            metric="precomputed",
            distance_threshold=self.clustering_threshold,
            n_clusters=None,
            linkage="average",
        ).fit(dist_matrix)
        return clustering

    def __merge_multi_dist_matrix(self, matrix_dict):
        dist_tensor = np.array([v for _, v in matrix_dict.items()])
        dist_merged = dist_tensor.min(axis=0)
        # dist_argmin = dist_tensor.argmin(axis=0)
        return dist_merged

    def __compute_distance_matrix(self, metric_matrix_dict):
        print("Computing distance matrix..")
        if os.path.isfile(self.cache_path):
            print(f"Loading from {self.cache_path}")
            distance_matrix_dict = load_pickle(self.cache_path)
        else:
            distance_matrix_dict = defaultdict(dict)  # {part: metric: distance_matrix}
            for part_id, vm_list in tqdm(self.partitions.items()):
                vm_part_indice = []
                for vm in vm_list:
                    if vm in self.metric_group.vm2idx:
                        vm_part_indice.append(self.metric_group.vm2idx[vm])
                #vm_part_indice = [self.metric_group.vm2idx[vm] for vm in vm_list]
                for metric_name, metric_matrix in metric_matrix_dict.items():
                    #print(metric_name)
                    metric_matrix_metric = metric_matrix[vm_part_indice]
                    if self.normalize:
                        metric_matrix_metric = (
                            MinMaxScaler().fit_transform(metric_matrix_metric.T).T
                        )
                    weighted_matrix = None
                    if weighted_matrix is not None:
                        weights = np.zeros((len, len))
                        for i in range(len):
                            for j in range(len):
                                weights[i, j] = max(row_variances[i], row_variances[j])
                        distance_matrix = self.dist_func(metric_matrix_metric)
                        distance_matrix = np.multiply(distance_matrix, weights)
                    else:
                        distance_matrix = self.dist_func(metric_matrix_metric)
                    distance_matrix_dict[part_id][metric_name] = distance_matrix
            save_pickle(distance_matrix_dict, self.cache_path)
        print("Computing distance matrix done.")
        return distance_matrix_dict


class MetricGroup:
    def __init__(self, metric_names, metric_dict, file_format="pkl", nrows=None) -> None:
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        self.metric_names = metric_names
        self.file_format = file_format
        self.nrows = nrows
        self.num_vms = len(metric_dict)
        self.vm2idx = {}
        self.idx2vm = {}
        self.matrix_dict = self.__pack(metric_dict)

    def __pack(self, metric_dict):
        """
        Pack all vms's metric to the same matrix for each metric
        """
        matrix_dict = defaultdict(list)
        for idx, (vmid, value_dict) in enumerate(metric_dict.items()):
            if self.nrows and idx > self.nrows:
                break
            self.vm2idx[vmid] = idx
            self.idx2vm[idx] = vmid
            for metric in self.metric_names:
                if metric in value_dict:
                    if self.file_format == "hdf5":
                        value_tuple_list = value_dict[metric]
                        value_tuple_list = sorted(value_tuple_list, key=lambda x: x[0])
                        values = list(zip(*value_tuple_list))[1]
                    else:
                        values = value_dict[metric]
                    matrix_dict[metric].append(values)
        matrix_dict = {k: padding(v) for k, v in matrix_dict.items()}
        return matrix_dict

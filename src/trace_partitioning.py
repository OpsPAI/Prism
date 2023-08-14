import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from datasketch import MinHash, MinHashLSHForest, MinHashLSH
from matplotlib import pyplot as plt
import ipaddress
sys.path.append("../")
from common.evaluation import evaluator
from common.utils import load_pickle, save_pickle


class LocalitySearch():
    def __init__(self, num_perm=50):
        self.num_perm = num_perm
        self.model = None 

    def build_search_db(self, feature_dict, dbtype="threshold", threshold=None):
        db = dict()
        for key, item_set in feature_dict.items():
            db[key] = self.__build_minhash(item_set)

        if dbtype == "topk":
            self.model = MinHashLSHForest(num_perm=self.num_perm)
            for k, v in db.items():
                self.model.add(k, v)
            self.model.index()
        elif dbtype == "threshold":
            assert threshold is not None, "must set threshold when dbtype=threshold."
            self.model = MinHashLSH(threshold=threshold, num_perm=self.num_perm)
            for k, v in db.items():
                self.model.insert(k, v)

    def __build_minhash(self, aset):
        m = MinHash(num_perm=self.num_perm)
        for d in aset:
            m.update(str(d).encode('utf8'))
        return m

    def query_threshold(self, query):
        query = self.__build_minhash(query)
        results = self.model.query(query)
        return results


def get_partitions(vm2partition):
    vm_partitions = defaultdict(list)
    for vmid, cluster_id in vm2partition.items():
        vm_partitions[cluster_id].append(vmid)
    num_partitions = len(vm_partitions)
    print(f"Prepartition done, {num_partitions} partitions obtained.")
    return vm_partitions


def is_internal_ip(ip):
    try:
        ip = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return ip.is_private or ip.is_reserved or ip.is_link_local or ip.is_loopback


def get_vm2feats(row, metadata, vm2feats):
    srcip, dstip = row[0], row[1]
    if srcip in metadata:
        vmid = metadata[srcip]["vmid"]
        vm2feats[vmid].add(dstip)

    if dstip in metadata:
        vmid = metadata[dstip]["vmid"]
        vm2feats[vmid].add(srcip)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
                self.size[root_y] += self.size[root_x]
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                self.size[root_x] += self.size[root_y]


def trace_partition(vm2feats, threshold):
    vm2id = {vm : id for id, vm in enumerate(vm2feats.keys())}
    build_begin = time.time()
    LS = LocalitySearch()
    LS.build_search_db(vm2feats, dbtype="threshold", threshold=threshold)
    build_end = time.time()
    build_time = build_end - build_begin
    print(f"Build Time: {build_time:.3f}s")

    partition_idx = 0
    vm2partition = dict()
    
    start = time.time()
    if partition_algorithm == "simple":
        for key, feats in tqdm(vm2feats.items()):
            if key in vm2partition: continue
            neighbor_keys = LS.query_threshold(feats)
            neighbor_keys = [item for item in neighbor_keys if (item != key and item not in vm2partition)]
            if len(neighbor_keys) > 0:
                vm2partition[key] = partition_idx
                for vm in neighbor_keys:
                    vm2partition[vm] = partition_idx
                partition_idx += 1
            else:
                vm2partition[key] = -1
    elif partition_algorithm == "union_set":
            Union_Find = UnionFind(len(vm2id))
            for key, feats in tqdm(vm2feats.items()):
                id1 = vm2id[key]
                neighbor_keys = LS.query_threshold(feats)
                for item in neighbor_keys:
                    id2 = vm2id[item]
                    if Union_Find.find(id1) != Union_Find.find(id2):
                        Union_Find.union(id1, id2)
            for vm, idx in vm2id.items():
                root = Union_Find.find(idx)
                if Union_Find.size[root] == 1:
                    vm2partition[vm] = -1
                else:
                    vm2partition[vm] = root
    end = time.time()
    print(f"{end-start:.2f} s")
    return vm2partition


if __name__ == "__main__":
    date = "dataset1"
    trace_path  = Path(f"../data/anonymized_trace.csv")
    metadata_path = Path(f"../data/anonymized_metadata.pkl")
    vm2feats_path = Path(f"../outdir/vm2feats.pkl")
    partition_algorithm = "simple"
    threshold = 0.1
    if os.path.exists(vm2feats_path):
        vm2feats = load_pickle(Path(vm2feats_path))
        print("Load vm2feats data done.")
    else:
        trace_df = pd.read_csv(trace_path, nrows=None)
        metadata = load_pickle(metadata_path)
        print("Reading data done.")
        
        vm2feats = defaultdict(set)
        tqdm.pandas()
        trace_df.progress_apply(lambda x: get_vm2feats(x, metadata, vm2feats), axis=1, raw=True)
        save_pickle(vm2feats, Path(vm2feats_path))
    print("Total {} VMs.".format(len(vm2feats)))
    
    partition_list_outpath = Path(f"../outdir/threshold_{threshold}.pkl")

    vm2partition = trace_partition(vm2feats, threshold)

    partition_list = get_partitions(vm2partition)
    save_pickle(partition_list, partition_list_outpath)

    function_label_file = f"../data/anonymized_label.csv"
    outdir_root = Path(f"../outdir/partition_list/")
    evaluator(f"threshold={threshold}", outdir_root, function_label_file, "label", True).evaluate_metrics(vm2partition)

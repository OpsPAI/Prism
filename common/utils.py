import os
import _pickle as pickle
import datetime
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
import base64


def encode_str(input_str):
    encoded_bytes = base64.b64encode(input_str.encode('utf-8'))
    encoded_str = str(encoded_bytes, 'utf-8')
    return encoded_str


def decode_str(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (list, np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def save_matrix(matrix, path, title=""):
    np.savetxt(path, matrix, fmt="%.4f", delimiter=",")


def plot_metric_groups(metric_matrix, plt_title, plt_save_path):
    os.makedirs(os.path.dirname(plt_save_path), exist_ok=True)
    plt.figure()
    for row in metric_matrix:
        plt.plot(row)
    plt.title(plt_title)
    plt.savefig(plt_save_path)
    plt.close()


def save_pickle(obj, pklpath, p=2):
    pklpath = Path(pklpath)
    pklpath.parent.mkdir(parents=True, exist_ok=True)
    with open(pklpath, "wb") as fw:
        pickle.dump(obj, fw, protocol=p)


def load_pickle(pklpath):
    with open(pklpath, "rb") as fr:
        return pickle.load(fr)


def padding(insts, padding_token=0):
    maxlen = max([len(item) for item in insts])
    padded = np.array([list(inst) + [inst[-1]] * (maxlen - len(inst)) for inst in insts])
    return padded

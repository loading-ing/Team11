import argparse
import pickle
import gguf
from gguf.constants import GGMLQuantizationType
from gguf.gguf_writer import GGUFWriter
import torch
from pathlib import Path
import os
import struct
import numpy as np
import re
from typing import List

def load_activation_weights(models_base: Path):
    # TODO: might need a specification file to indicate which models to load.
    # But for now, let's assume it is a plain directory of activation_{0, ... , n_layers - 1}.pt
    *_, files = next(os.walk(models_base))
    activation_files = [f for f in files if re.match(r"activation_\d+.pt", f)]
    activation_files.sort()
    return [torch.load(models_base / f) for f in activation_files]

def append_gpu_idx(gguf: GGUFWriter, i_layer: int, activation, select_count) -> None:
    _, indices = torch.topk(activation, k=int(select_count))
    gpu_idx = torch.zeros_like(activation)
    gpu_idx[indices] = 1
    gpu_idx = gpu_idx.numpy().astype(np.int32)
    key = f"blk.{i_layer}.gpu_idx"
    print(
        f"{key} => {key} {gpu_idx.shape} {gpu_idx.dtype} {gpu_idx.nbytes/1024/1024} MiB"
    )
    gguf.add_tensor(
        name=key,
        tensor=gpu_idx,
        raw_shape=gpu_idx.shape[::-1],
        raw_dtype=GGMLQuantizationType.I32,
    )

    indices = indices.numpy().astype(np.int32)
    gpu_bucket = np.sort(indices)
    key = f"blk.{i_layer}.gpu_bucket"
    print(
        f"{key} => {key} {gpu_bucket.shape} {gpu_bucket.dtype} {gpu_bucket.nbytes/1024/1024} MiB"
    )
    gguf.add_tensor(
        name=key,
        tensor=gpu_bucket,
        raw_shape=gpu_bucket.shape[::-1],
        raw_dtype=GGMLQuantizationType.I32,
    )

def my_chosek(activation,k):
    """
    实现一个算法，在输入的激活向量中选择k个元素，并返回它们的索引。设选取的元素形成N(N<=k)个连通块，对于第i个连通块Ni设其权值为Qi(Qi等于Ni中所有元素的平均值)。
    设QA为所有连通块权值的平均值，求使得QA最大的情况下，选取的k个元素的索引。
    """
    pass

def choosekn(activation, k, N):
    """
    实现一个算法，在输入的激活向量中选择k个元素，并且选取的元素形成N个连通块，使得所选取的元素和最大，返回k个元素的索引。

    """
    n = len(activation)
    if k > n or N > n - (k - 1):
        return None  # 不合法的参数

    # 创建一个二维数组dp，dp[i][j]表示前i个元素中能形成j个连通块的最大和
    dp = [[0] * (N + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][1] = dp[i - 1][1] + activation[i - 1]
        for j in range(2, min(i - 1, N) + 1):  # 最多N个连通块，且至少需要两个元素
            # 选择当前元素作为新的连通块开始，或者加入前一个连通块
            dp[i][j] = max(dp[i - 1][j] + activation[i - 1], dp[i - 1][j - 1])

    # 回溯找到最大和的解
    indices = []
    current_sum = dp[n][N]
    i, j = n, N
    while i > 0 and j > 0:
        if dp[i - 1][j - 1] >= dp[i - 1][j]:
            # 前一个连通块包含当前元素
            indices.append(i - 1)
            j -= 1
        else:
            # 新的连通块开始
            indices.append(i - 1)
        i -= 1

    return indices[::-1]


def export_split(activations_path: str, output_path: str, solved_list: List[ int ], vram_capacity: int):
    predictors = load_activation_weights(Path(activations_path)) # predictor => activation acount
    gguf_out = GGUFWriter(output_path, "generic.gpu_index")
    for i, (activation, selected_count) in enumerate(zip(predictors, solved_list)):
        append_gpu_idx(gguf_out, i, activation, selected_count)

    # set kvs
    gguf_out.add_block_count(len(predictors))
    # TODO: better to save the actual capacity that split neurons require
    gguf_out.add_uint64(gguf.Keys.Split.VRAM_CAPACITY, vram_capacity)

    gguf_out.write_header_to_file()
    gguf_out.write_kv_data_to_file()
    gguf_out.write_tensors_to_file()
    gguf_out.close()

    # post-process: write another unique file header to distinguish from the origianl GGUF file
    with open(output_path, "r+b") as fout:
        POWERINFER_MAGIC = int.from_bytes(b"PWRI", "little")
        fout.write(struct.pack("<I", POWERINFER_MAGIC))
        fout.write(struct.pack("<I", 3))

    print(f"exported GPU index to {output_path}")


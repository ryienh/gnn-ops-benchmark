# Scatter ops

import torch
from torch_scatter import scatter
import torch.utils.benchmark as benchmark

import wandb

from graph_benchmark.datasets.fakeDatasets import FakeDataset

wandb.init(project="scatter-op-benchmark")


# CONFIGS HERE
length_ = 400096000
reduce_f_ = 8
src_dims = (2, length_)
idx_dims = src_dims
max_idx = int(length_ / reduce_f_)


assert src_dims == idx_dims

# dimensions: src size, idx size, src sparsity
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")


src = torch.rand(src_dims).to(device).to(torch.float32)
idx = torch.randint(high=max_idx, size=idx_dims).to(device).to(torch.int64)

# DEBUG
# src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).cuda()
# idx = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]).cuda()


"""
Begin scatter ops
"""


def op_scatter_max(src, idx):
    """Computes max scatter operation"""
    out, argmax = scatter(src, idx, dim=-1, reduce="max")
    return out, argmax


def op_scatter_min(src, idx):
    """Computes min scatter operation"""
    out, argmax = scatter(src, idx, dim=-1, reduce="min")
    return out, argmax


def op_scatter_mean(src, idx):
    """Computes mean scatter operation"""
    out, argmax = scatter(src, idx, dim=-1, reduce="mean")
    return out, argmax


def op_scatter_sum(src, idx):
    """Computes sum scatter operation"""
    out, argmax = scatter(src, idx, dim=-1, reduce="sum")
    return out, argmax


"""
Benchmark logic
"""
t0 = benchmark.Timer(
    stmt="op_scatter_max(src, idx)",
    setup="from __main__ import op_scatter_max",
    globals={"src": src, "idx": idx},
)

t1 = benchmark.Timer(
    stmt="op_scatter_min(src, idx)",
    setup="from __main__ import op_scatter_min",
    globals={"src": src, "idx": idx},
)

t2 = benchmark.Timer(
    stmt="op_scatter_mean(src, idx)",
    setup="from __main__ import op_scatter_mean",
    globals={"src": src, "idx": idx},
)

t3 = benchmark.Timer(
    stmt="op_scatter_sum(src, idx)",
    setup="from __main__ import op_scatter_sum",
    globals={"src": src, "idx": idx},
)

# print(t0.timeit(100))
# print(t1.timeit(100))

m0 = t0.blocked_autorange()
m1 = t1.blocked_autorange()
m2 = t2.blocked_autorange()
m3 = t3.blocked_autorange()

print(m0)
print(m1)
print(m2)
print(m3)

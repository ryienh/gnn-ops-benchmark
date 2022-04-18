# Scatter ops

import torch
from torch_scatter import (
    scatter_max,
    scatter_add,
    scatter_min,
    scatter_mean,
    segment_coo,
    segment_csr,
)
import torch.utils.benchmark as benchmark
import pandas as pd

import wandb

from graph_benchmark.datasets.fakeDatasets import FakeDataset

from pdb import set_trace as bp

# wandb.init(project="scatter-op-benchmark")

"""
Begin scatter ops
"""


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def op_scatter_max(src, idx):
    """Computes max scatter operation"""

    out = scatter_max(src, idx)
    return out


# def op_scatter_add_native(src, idx):
#     """Computes max scatter operation"""
#     temp = torch.zeros_like(src)
#     temp.scatter_(dim=0, index=idx, src=src, reduce="add")
#     return temp


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
op_name = "scatter_max_new"
native_exists = True
length_ = 1600384000
length__ = 40000
reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
sparsities = [0, 0.5, 0.9, 0.99]
tshapes = [
    (int(length_),),
    (int(length__), int(length__)),
]
# tshapes = [(int(length__), int(length__))]
# tshapes = [(int(length_), int(length_))]


# assert src_dims == idx_dims

# create data (list of dicts) for csv creation
data = []

# cuda stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")

torch.cuda.empty_cache()
counter = 0
# bp()
# define inputs over hyperparams
for reduce_f in reduce_f_:
    for sparsity in sparsities:
        for src_dims in tshapes:

            torch.cuda.empty_cache()

            # print_util_info()
            # bp()

            idx_dims = src_dims

            max_idx = int(src_dims[0] / reduce_f)

            # print(src_dims[2])
            src = torch.rand(
                size=src_dims, device="cuda", dtype=torch.float32, requires_grad=False
            )
            idx = torch.randint(
                high=max_idx,
                size=idx_dims,
                device="cuda",
                dtype=torch.int64,
                requires_grad=False,
            )

            # randomly drop values to create sparsity
            src = torch.nn.functional.dropout(
                src, p=sparsity, training=False, inplace=False
            )

            # begin benchmark logic
            t0 = benchmark.Timer(
                stmt="op_scatter_max(src, idx)",
                setup="from __main__ import op_scatter_max",
                globals={"src": src, "idx": idx},
            )
            m0 = t0.blocked_autorange()

            bm_val = m0.median

            del m0
            del t0

            # t1 = benchmark.Timer(
            #     stmt="op_scatter_add_native(src, idx)",
            #     setup="from __main__ import op_scatter_add_native",
            #     globals={"src": src, "idx": idx},
            # )

            # m1 = t1.blocked_autorange()
            # bm_val_native = m1.median

            print_util_info()

            shape_name = "LS" if len(src_dims) == 1 else "square"
            params_str = str(reduce_f) + " " + shape_name

            # output_value = (
            #     bm_val if not native_exists else combine_vals(bm_val, bm_val_native)
            # )

            data.append([params_str, str(src_dims), sparsity, bm_val])

            del src
            del idx
            # if native_exists:
            #     del t1
            #     del m1

            # print_util_info()

            # torch.cuda.empty_cache()

            print(f"done with {counter}")
            counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Reduce factor, shape",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"data/{op_name}")


"""
Benchmark logic
"""
# t0 = benchmark.Timer(
#     stmt="op_coalesce(idx, val, n, m)",
#     setup="from __main__ import op_coalesce",
#     globals={"idx": idx, "val": val, "n": n, "m": m},
# )

# t1 = benchmark.Timer(
#     stmt="op_scatter_min(src, idx)",
#     setup="from __main__ import op_scatter_min",
#     globals={"src": src, "idx": idx},
# )

# t2 = benchmark.Timer(
#     stmt="op_scatter_mean(src, idx)",
#     setup="from __main__ import op_scatter_mean",
#     globals={"src": src, "idx": idx},
# )

# t3 = benchmark.Timer(
#     stmt="op_scatter_sum(src, idx)",
#     setup="from __main__ import op_scatter_sum",
#     globals={"src": src, "idx": idx},
# )

# print(t0.timeit(100))
# print(t1.timeit(100))

# m0 = t0.blocked_autorange()
# m1 = t1.blocked_autorange()
# m2 = t2.blocked_autorange()
# m3 = t3.blocked_autorange()

# print(m0)
# print(m1)
# print(m2)
# print(m3)

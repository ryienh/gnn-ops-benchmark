import sys
import math
import numpy as np
import torch
from torch_scatter import (
    scatter_max,
)
import torch.utils.benchmark as benchmark
import pandas as pd

sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


def op_scatter_max(src, idx, dim):
    """Computes max scatter operation"""
    out = scatter_max(src, idx, dim)
    return out


# Configurable hyperparams here
setup_seed(42)
op_name = "scatter_max"
native_exists = True
# length_ = 512
# length__ = 256
reduce_f_ = [1, 2, 4, 8]
# sparsities = [0, 0.5, 0.9, 0.99]
# tshapes = [
#     (int(length_),),
#     (int(length__), int(length__)),
# ]
lengths_ = np.linspace(1_500_000, 45_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0]
num_bm_runs = 100


tshapes = [(length_, length_) for length_ in lengths_]

# create data (list of dicts) for csv creation
data = []

# cuda stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")

torch.cuda.empty_cache()
counter = 0

# define inputs over hyperparams
for reduce_f in reduce_f_:
    for sparsity in sparsities:
        for src_dims in tshapes:
            for dim in [0, 1]:

                torch.cuda.empty_cache()

                idx_dims = src_dims

                max_idx = int(src_dims[0] / reduce_f)

                # print(src_dims[2])
                src = torch.rand(
                    size=src_dims,
                    device="cuda",
                    dtype=torch.float16,
                    requires_grad=False,
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
                    src, p=sparsity, training=True, inplace=False
                )

                total_elts = idx.numel() + src.numel()
                input_mem = (
                    idx.element_size() * idx.numel() + src.element_size() * src.numel()
                ) / 1000000

                # begin benchmark logic
                t0 = benchmark.Timer(
                    stmt="op_scatter_max(src, idx, dim)",
                    setup="from __main__ import op_scatter_max",
                    globals={"src": src, "idx": idx, "dim": dim},
                )

                bm = t0.timeit(num_bm_runs)
                bm_val = bm.median
                bm_iqr = bm.iqr

                del t0
                del bm

                mem = get_reserved_in_mb()

                print_util_info()

                shape_name = "LS" if len(src_dims) == 1 else "square"
                params_str = str(reduce_f) + " " + shape_name + " " + str(dim)
                bm_pyg_data = str(bm_val) + "(" + str(bm_iqr) + ")"

                data.append(
                    [
                        params_str,
                        str(src_dims),
                        sparsity,
                        total_elts,
                        input_mem,
                        mem,
                        bm_pyg_data,
                    ]
                )

                del src
                del idx

                # print_util_info()

                # torch.cuda.empty_cache()

                print(f"done with {counter}")
                counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Reduce factor, shape, dim",
    "Input size",
    "Sparsity",
    "Total elements",
    "Input memory",
    "Total Memory",
    "GPU clock time py geo (IQR)",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

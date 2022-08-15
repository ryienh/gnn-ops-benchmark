import sys
import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


@torch.jit.script
def fused_gelu(input, dim: int, index):
    out = torch.index_select(input, dim, index).sum()
    return out


def gelu(input, dim, index):
    out = torch.index_select(input, dim, index).sum()
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
op_name = "fused_index_select_reduce"
# native_exists = True
# length_ = int(1600384000 * 1.6)
# length__ = int(2048)
# length___ = int(512)
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
# sparsities = [0, 0.5, 0.9, 0.99]


# tshapes = [
#     # (length_,),
#     (length__, length__),
#     (length___, length___, length___),
# ]


lengths_ = np.linspace(7_500_000, 200_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0]


tshapes = [(length_, length_) for length_ in lengths_]
num_bm_runs = 100


# create data (list of dicts) for csv creation
data = []

# cuda stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")

torch.cuda.empty_cache()
counter = 0

# define inputs over hyperparams
for sparsity in sparsities:
    for tshape in tshapes:
        for dim in [0, 1, 2]:
            for idx_reduce_factor in [1, 2, 4, 8]:

                torch.cuda.empty_cache()

                if dim >= len(tshape):
                    continue

                print(f"DEBUG: Current input has dims {len(tshape)}")

                input = torch.rand(
                    size=tshape,
                    device="cuda",
                    dtype=torch.float16,
                    requires_grad=False,
                )

                # randomly drop values to create sparsity
                input = torch.nn.functional.dropout(
                    input, p=sparsity, training=True, inplace=False
                )

                # get index based on reduction factor
                index = torch.randint(
                    low=0,
                    high=input.shape[dim],
                    size=(int(input.shape[dim] / idx_reduce_factor),),
                    device="cuda",
                    requires_grad=False,
                )

                # begin benchmark logic
                t0 = benchmark.Timer(
                    stmt="fused_gelu(input, dim, index)",
                    setup="from __main__ import fused_gelu",
                    globals={"input": input, "dim": dim, "index": index},
                )

                total_elts = input.numel() + index.numel()
                input_mem = (
                    input.element_size() * input.numel()
                    + index.element_size() * index.numel()
                ) / 1000000
                print(f"SIZE TOTAL (THEORETICAL): {input_mem} mb")

                bm = t0.timeit(num_bm_runs)
                bm_val = bm.median
                bm_iqr = bm.iqr

                del t0

                mem = get_reserved_in_mb()

                # begin benchmark logic
                t1 = benchmark.Timer(
                    stmt="gelu(input, dim, index)",
                    setup="from __main__ import gelu",
                    globals={"input": input, "dim": dim, "index": index},
                )

                bm_no_fuse = t1.timeit(num_bm_runs)
                bm_val_no_fuse = bm_no_fuse.median
                bm_iqr_no_fuse = bm_no_fuse.iqr

                del t1

                print_util_info()

                input_dims_str = str(len(tshape))
                sort_dim_str = str(dim)
                idx_reduce_factor_str = str(idx_reduce_factor)

                params_str = (
                    input_dims_str + "; " + sort_dim_str + "; " + idx_reduce_factor_str
                )

                formatted_input_dims = str(tshape)

                bm_val = str(bm_val) + "(" + str(bm_iqr) + ")"
                bm_val_no_fuse = str(bm_val_no_fuse) + "(" + str(bm_iqr_no_fuse) + ")"

                data.append(
                    [
                        params_str,
                        formatted_input_dims,
                        sparsity,
                        total_elts,
                        input_mem,
                        mem,
                        bm_val,
                        bm_val_no_fuse,
                    ]
                )

                del input
                del index

                print(f"done with {counter}")
                counter += 1


df = pd.DataFrame(data)
df.columns = [
    "Input dims, index dim, reduce factor (RF)",
    "Input size",
    "Sparsity",
    "Total elements",
    "Input memory",
    "Total Memory",
    "GPU clock time (fuse) (IQR)",
    "GPU clock time (no fuse) (IQR)",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

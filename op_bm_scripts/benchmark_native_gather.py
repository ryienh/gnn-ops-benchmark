from filecmp import clear_cache
import sys
import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import pandas as pd


sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


def op_native_gather(input, dim, index):
    """Computes gather operation"""
    out = torch.gather(input, dim, index)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
op_name = "native_gather"
# native_exists = True
# length_ = int(1024)
# length__ = int(512)
# length___ = int(256)
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
# sparsities = [0, 0.5, 0.9, 0.99]
# sparsities = [0, 0.9]


# tshapes = [
#     (length_,),
#     (length__, length__),
#     (length___, length___, length___),
# ]

lengths_ = np.linspace(1_500_000, 40_000_000, num=10).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
# length__ = int(math.sqrt(3495200))
# length___ = int(math.sqrt(3495200))
sparsities = [0]
num_bm_runs = 1


tshapes = [
    (length_, length_)
    for length_ in lengths_
    # [(length_, length_), (length_, 1), (1, length_)],
    # [(length___, 1), (length___, length___), (length___, 1)],
]

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

            index = torch.randint(
                low=0,
                high=input.shape[dim],
                size=input.shape,
                dtype=torch.int64,
                device="cuda",
                requires_grad=False,
            )

            total_elts = input.numel() + index.numel()
            input_mem = (
                input.element_size() * input.numel()
                + index.element_size() * index.numel()
            ) / 1000000
            print(f"SIZE TOTAL (THEORETICAL): {input_mem} mb")

            # randomly drop values to create sparsity
            input = torch.nn.functional.dropout(
                input, p=sparsity, training=True, inplace=False
            )

            # begin benchmark logic
            t0 = benchmark.Timer(
                stmt="op_native_gather(input, dim, index)",
                setup="from __main__ import op_native_gather",
                globals={"input": input, "dim": dim, "index": index},
            )

            bm = t0.timeit(num_bm_runs)
            bm_val = bm.median
            bm_iqr = bm.iqr

            del t0

            mem = get_reserved_in_mb()

            print_util_info()

            input_dims_str = str(len(tshape))
            sort_dim_str = str(dim)

            params_str = input_dims_str + "; " + sort_dim_str

            formatted_input_dims = str(tshape)

            bm_val = str(bm_val) + "(" + str(bm_iqr) + ")"

            data.append(
                [
                    params_str,
                    formatted_input_dims,
                    sparsity,
                    total_elts,
                    input_mem,
                    mem,
                    bm_val,
                ]
            )

            del input
            del index

            print(f"done with {counter}")
            counter += 1


# df = pd.DataFrame(data)
# df.columns = [
#     "Input dims",
#     "Input size",
#     "Sparsity",
#     "Total elements",
#     "Input memory",
#     "Total Memory",
#     "GPU clock time (IQR)",
# ]
# df.to_csv(f"mem_prof_data/{op_name}.csv")

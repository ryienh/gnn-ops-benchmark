# Scatter ops

import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd

from pdb import set_trace as bp


"""
Begin scatter ops
"""
import random
import numpy


def setup_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def op_native_gather(input, dim, index):
    """Computes max scatter operation"""
    out = torch.gather(input, dim, index)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
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

lengths_ = np.linspace(1_500_000, 100_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
# length__ = int(math.sqrt(3495200))
# length___ = int(math.sqrt(3495200))
sparsities = [0]


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
# bp()
# define inputs over hyperparams
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("bm"):
        for sparsity in sparsities:
            for tshape in tshapes:
                for dim in [0, 1, 2]:

                    torch.cuda.empty_cache()

                    # print_util_info()
                    # bp()

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

                    print(
                        f"SIZE TOTAL: {(input.element_size()*input.numel() + index.element_size()*index.numel())/1000000} mb"
                    )

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

                    bm_val = t0.timeit(100).median

                    del t0

                    print_util_info()

                    input_dims_str = str(len(tshape))
                    sort_dim_str = str(dim)

                    params_str = input_dims_str + "; " + sort_dim_str

                    formatted_input_dims = str(tshape)

                    data.append([params_str, formatted_input_dims, sparsity, bm_val])

                    del input
                    del index

                    print(f"done with {counter}")
                    counter += 1


# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


df = pd.DataFrame(data)
df.columns = [
    "Input dims, index dim",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

# Scatter ops

import torch
import torch.utils.benchmark as benchmark
import pandas as pd

import matplotlib.pyplot as plt

from pdb import set_trace as bp

import random
import numpy


"""
Begin scatter ops
"""


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


def op_native_sort(input, dim, stable):
    """Computes max scatter operation"""
    out, idx = torch.sort(input=input, dim=dim, stable=stable)
    return out, idx


# def op_scatter_add_native(src, idx):
#     """Computes max scatter operation"""
#     temp = torch.zeros_like(src)
#     temp.scatter_(dim=0, index=idx, src=src, reduce="add")
#     return temp


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
setup_seed(42)

op_name = "native_sort"
# native_exists = True
length_ = int(1600384000 * 0.6)
length__ = int(40000 * 0.705)
length___ = int(2000 * 0.45)
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
sparsities = [0, 0.5, 0.9, 0.99]


tshapes = [
    (length_,),
    (length__, length__),
    (length___, length___, length___),
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
for sparsity in sparsities:
    for tshape in tshapes:
        for dim in [0, 1, 2]:
            for stable in [True, False]:

                torch.cuda.empty_cache()

                # print_util_info()
                # bp()

                if dim >= len(tshape):
                    continue

                print(f"DEBUG: Current input has dims {len(tshape)}")

                input = torch.rand(
                    size=tshape, device="cuda", dtype=torch.float32, requires_grad=False
                )

                # randomly drop values to create sparsity
                input = torch.nn.functional.dropout(
                    input, p=sparsity, training=True, inplace=False
                )

                print(
                    f"for sparsity: {sparsity}, percent non-zero is: {torch.count_nonzero(input) / input.numel()}"
                )

                # begin benchmark logic
                t0 = benchmark.Timer(
                    stmt="op_native_sort(input, dim, stable)",
                    setup="from __main__ import op_native_sort",
                    globals={"input": input, "dim": dim, "stable": stable},
                )
                m0 = t0.blocked_autorange(min_run_time=1)

                bm_val = m0.median
                print(
                    f"debug blocked autorange: median is {m0.median}, iqr is {m0.iqr}, count is {len(m0.times)}"
                )

                del m0
                del t0

                print_util_info()

                input_dims_str = str(len(tshape))
                sort_dim_str = str(dim)
                stable_str = str(stable)

                params_str = input_dims_str + "; " + sort_dim_str + "; " + stable_str

                formatted_input_dims = str(tshape)

                data.append([params_str, formatted_input_dims, sparsity, bm_val])

                del input

                print(f"done with {counter}")
                counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input dims, sort dim, stable",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

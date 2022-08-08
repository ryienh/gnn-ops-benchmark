# Scatter ops

import math
import numpy as np
import torch
from torch_scatter import (
    scatter_add,
)
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


def op_scatter_add(src, idx):
    """Computes max scatter operation"""

    out = scatter_add(src, idx)
    return out


def op_native_scatter_add_(src, idx):
    temp = torch.zeros_like(src)
    temp.scatter_add_(-1, idx, src)
    return None


# Configurable hyperparams here
setup_seed(42)
op_name = "scatter_add"
native_exists = True
# length_ = 512
# length__ = 256
reduce_f_ = [1, 2, 4, 8]
# sparsities = [0, 0.5, 0.9, 0.99]
# tshapes = [
#     (int(length_),),
#     (int(length__), int(length__)),
# ]
lengths_ = np.linspace(1_500_000, 100_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0]


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
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("bm"):
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

                    print(
                        f"SIZE TOTAL: {(src.element_size()*src.numel() + idx.element_size()*idx.numel())/1000000} mb"
                    )

                    # print(
                    #     f"for sparsity: {sparsity}, percent non-zero is: {torch.count_nonzero(src) / src.numel()}"
                    # )

                    # begin benchmark logic
                    t0 = benchmark.Timer(
                        stmt="op_scatter_add(src, idx)",
                        setup="from __main__ import op_scatter_add",
                        globals={"src": src, "idx": idx},
                    )

                    m = t0.timeit(100)
                    bm_val = m.median
                    bm_iqr = m.iqr

                    del t0
                    del m

                    t1 = benchmark.Timer(
                        stmt="op_native_scatter_add_(src, idx)",
                        setup="from __main__ import op_native_scatter_add_",
                        globals={"src": src, "idx": idx},
                    )

                    bm_val_native = t1.timeit(100).median

                    del t1

                    print_util_info()

                    shape_name = "LS" if len(src_dims) == 1 else "square"
                    params_str = str(reduce_f) + " " + shape_name
                    bm_data = (
                        "%g" % round(bm_val, 3)
                        + " ("
                        + "%g" % round(bm_val_native, 3)
                        + ")"
                    )

                    data.append([params_str, str(src_dims), sparsity, bm_data])

                    del src
                    del idx

                    # print_util_info()

                    # torch.cuda.empty_cache()

                    print(f"done with {counter}")
                    counter += 1


# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


df = pd.DataFrame(data)
df.columns = [
    "Reduce factor, shape",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

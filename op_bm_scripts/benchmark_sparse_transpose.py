import sys
import math
import random
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


def op_native_transpose(matA):
    """only swap 0 and 1 dims as that what torch_sparse.transpose supports."""
    out = torch.transpose(matA, 0, 1)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
setup_seed(42)
op_name = "sparse_transpose"
num_bm_runs = 100

lengths_ = np.linspace(8_000_000, 250_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
tshapes = [(length_, length_) for length_ in lengths_]
sparsities = [0.995]

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

        torch.cuda.empty_cache()

        print(f"DEBUG: Current input has dims {tshape}")

        matA = torch.rand(
            size=tshape,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )

        torch.cuda.empty_cache()

        # randomly drop values to create sparsity

        matA = torch.nn.functional.dropout(
            matA, p=sparsity, training=True, inplace=False
        )
        sparsity = (torch.numel(matA) - torch.count_nonzero(matA)) / torch.numel(matA)
        sparsity = float(sparsity)
        print(f"Sparsity is {sparsity}")

        torch.cuda.empty_cache()

        total_elts = matA.numel()
        input_mem = (+matA.element_size() * matA.numel()) / 1000000
        print(f"SIZE TOTAL (THEORETICAL): {input_mem} mb")

        print("Begin benchmark logic (native)")
        t0 = benchmark.Timer(
            stmt="op_native_transpose(matA)",
            setup="from __main__ import op_native_transpose",
            globals={
                "matA": matA,
            },
        )

        bm = t0.timeit(num_bm_runs)
        bm_native = bm.median
        bm_iqr = bm.iqr

        del t0

        mem = get_reserved_in_mb()

        print_util_info()

        input_dims_str = "LS" if tshape[1] == 1 else "Square"

        params_str = input_dims_str

        formatted_input_dims = str(tshape)

        formatted_sparsities = str(sparsity)

        bms = str(bm_native) + "(" + str(bm_iqr) + ")"

        data.append(
            [
                params_str,
                formatted_input_dims,
                formatted_sparsities,
                total_elts,
                input_mem,
                mem,
                bms,
            ]
        )

        del matA

        print(f"done with {counter}")
        counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input Shape",
    "Input size",
    "Sparsities (matA)",
    "Total elements",
    "Input memory",
    "Total memory",
    "GPU clock time (IQR)",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

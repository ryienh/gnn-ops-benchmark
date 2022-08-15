import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.utils.benchmark as benchmark


sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


def op_native_matmul(input, other):
    """Computes matmul operation"""
    out = torch.matmul(input=input, other=other)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
setup_seed(42)
op_name = "native_matmul"
lengths_ = np.linspace(
    2_500_000, 66_666_667, num=100
).tolist()  # _small used 20,000,000
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0]
num_bm_runs = 100


tshapes = [
    [
        (length_, length_),
        (length_, length_),
    ]
    for length_ in lengths_
]

# create data (list of dicts) for csv creation
data = []

# cuda stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")

torch.cuda.empty_cache()
counter = 0
for sparsity_input in sparsities:
    for sparsity_A in sparsities:
        for tshape in tshapes:

            torch.cuda.empty_cache()

            print(f"DEBUG: Current input has dims {tshape[0]}, {tshape[1]}")

            input = torch.rand(
                size=tshape[0],
                device="cuda",
                dtype=torch.float16,
                requires_grad=False,
            )

            other = torch.rand(
                size=tshape[1],
                device="cuda",
                dtype=torch.float16,
                requires_grad=False,
            )

            total_elts = input.numel() + other.numel()
            input_mem = (
                input.element_size() * input.numel()
                + other.element_size() * other.numel()
            ) / 1000000
            print(f"SIZE TOTAL (THEORETICAL): {input_mem} mb")

            # randomly drop values to create sparsity
            input = torch.nn.functional.dropout(
                input, p=sparsity_input, training=True, inplace=False
            )

            other = torch.nn.functional.dropout(
                other, p=sparsity_A, training=True, inplace=False
            )

            # begin benchmark logic
            t0 = benchmark.Timer(
                stmt="op_native_matmul(input, other)",
                setup="from __main__ import op_native_matmul",
                globals={"input": input, "other": other},
            )

            bm = t0.timeit(num_bm_runs)
            bm_val = bm.median
            bm_iqr = bm.iqr

            del t0

            mem = get_reserved_in_mb()

            input_dims_str = str(len(tshape))

            params_str = input_dims_str

            formatted_input_dims = str(tshape)

            formatted_sparsities = str(sparsity_input) + " ; " + str(sparsity_A)

            data.append(
                [
                    params_str,
                    formatted_input_dims,
                    formatted_sparsities,
                    total_elts,
                    input_mem,
                    mem,
                    str(bm_val) + "(" + str(bm_iqr) + ")",
                ]
            )

            del input
            del other

            print(f"done with {counter}")
            counter += 1


df = pd.DataFrame(data)
df.columns = [
    "Input dims",
    "Input size",
    "Sparsities (input, other)",
    "Total elements",
    "Input memory",
    "Total Memory",
    "GPU clock time (IQR)",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")  # _small

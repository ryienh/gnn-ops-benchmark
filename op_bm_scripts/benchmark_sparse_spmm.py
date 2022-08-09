import sys
import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME
from graph_benchmark.benchmark.util import *


def op_native_smm(matA, matB):
    out = torch.sparse.mm(matA, matB)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
setup_seed(42)
op_name = "sparse_spmm"
# native_exists = True
# length_ = int(4000 * 8 * 2.2)
# length__ = int(4000 * 0.25 * 1.7)
# length___ = int(4000 * 6.5 * 1.15)
# sparsities = [0.5, 0.9, 0.99]

lengths_ = np.linspace(2_000_000, 50_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0.999]
num_bm_runs = 100


tshapes = [[(length_, length_), (length_, length_)] for length_ in lengths_]


# tshapes = [
#     [
#         (length__, length__),
#         (length__, length__),
#     ],
#     [(length_, 1), (1, length_)],
#     [(length___, length___), (length___, 1)],
# ]

# create data (list of dicts) for csv creation
data = []

# cuda stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("Benchmarking only supported for CUDA")

torch.cuda.empty_cache()
counter = 0
# define inputs over hyperparams
for sparsity_A in sparsities:
    for sparsity_B in sparsities:
        for tshape in tshapes:

            torch.cuda.empty_cache()

            # if dim >= len(tshape):
            #     continue

            print(f"DEBUG: Current input has dims {tshape[0]}, {tshape[1]}")

            matA = torch.rand(
                size=tshape[0],
                device="cuda",
                dtype=torch.float32,
                requires_grad=False,
            )

            matB = torch.rand(
                size=tshape[1],
                device="cuda",
                dtype=torch.float32,
                requires_grad=False,
            )

            torch.cuda.empty_cache()

            # randomly drop values to create sparsity

            matA = torch.nn.functional.dropout(
                matA, p=sparsity_A, training=True, inplace=False
            )

            matB = torch.nn.functional.dropout(
                matB, p=sparsity_B, training=True, inplace=False
            )

            matA = matA.to_sparse()

            torch.cuda.empty_cache()

            total_elts = matA.numel() + matB.numel()
            input_mem = (
                matA.element_size() * matA.numel() + matB.element_size() * matB.numel()
            ) / 1000000

            print("Begin benchmark logic (native)")
            t0 = benchmark.Timer(
                stmt="op_native_smm(matA, matB)",
                setup="from __main__ import op_native_smm",
                globals={
                    "matA": matA,
                    "matB": matB,
                },
            )

            bm = t0.timeit(num_bm_runs)
            bm_val = bm.median
            bm_iqr = bm.iqr

            del t0

            mem = get_reserved_in_mb()

            print_util_info()

            input_dims_str = str(len(tshape))

            params_str = input_dims_str

            formatted_input_dims = str(tshape)

            formatted_sparsities = str(sparsity_A) + " ; " + str(sparsity_B)

            vals = str(bm_val) + " (" + str(bm_iqr) + ")"

            data.append(
                [
                    params_str,
                    formatted_input_dims,
                    formatted_sparsities,
                    total_elts,
                    input_mem,
                    mem,
                    vals,
                ]
            )

            del matA
            del matB

            print(f"done with {counter}")
            counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input dims",
    "Input size",
    "Sparsities (matA, matB)",
    "Total elements",
    "Input memory",
    "Total Memory",
    "GPU clock time (IQR)",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

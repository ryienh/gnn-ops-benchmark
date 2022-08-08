# Scatter ops
import sys
import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from pdb import set_trace as bp

from torch_sparse import spmm

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


def op_sparse_spmm(indexA, valueA, m, n, matB):
    val = spmm(index=indexA, value=valueA, m=m, n=n, matrix=matB)
    return val


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

lengths_ = np.linspace(2_000_000, 10_000_000, num=100).tolist()
lengths_ = [int(math.sqrt(x)) for x in lengths_]
sparsities = [0.9]


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
# bp()
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

            torch.cuda.empty_cache()

            m = matA.shape[0]
            n = matA.shape[1]

            torch.cuda.empty_cache()

            print("Converting to COO")
            matA = matA.to_sparse()
            torch.cuda.empty_cache()

            print("Converted to COO")

            indexA = matA.indices()
            valueA = matA.values()

            torch.cuda.empty_cache()

            # begin benchmark logic
            t0 = benchmark.Timer(
                stmt="op_sparse_spmm(indexA, valueA, m, n, matB)",
                setup="from __main__ import op_sparse_spmm",
                globals={
                    "indexA": indexA,
                    "valueA": valueA,
                    "m": m,
                    "n": n,
                    "matB": matB,
                },
            )

            print(
                f"SIZE TOTAL: {(matA.element_size()*matA.numel() + matB.element_size()*matB.numel())/1000000} mb"
            )

            bm_val = t0.timeit(2).median

            del t0

            torch.cuda.empty_cache()

            print("Begin benchmark logic (native)")
            t1 = benchmark.Timer(
                stmt="op_native_smm(matA, matB)",
                setup="from __main__ import op_native_smm",
                globals={
                    "matA": matA,
                    "matB": matB,
                },
            )

            bm_native = t1.timeit(2).median

            del t1

            print_util_info()

            input_dims_str = str(len(tshape))

            params_str = input_dims_str

            formatted_input_dims = str(tshape)

            formatted_sparsities = str(sparsity_A) + " ; " + str(sparsity_B)

            vals = str(bm_val) + " (" + str(bm_native) + ")"

            data.append([params_str, formatted_input_dims, formatted_sparsities, vals])

            del matA
            del matB
            del indexA
            del valueA

            print(f"done with {counter}")
            counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input dims",
    "Input size (>95% mem util)*",
    "Sparsities (input, matA, matB)",
    "GPU clock time",
]
df.to_csv(f"mem_prof_data/{op_name}.csv")

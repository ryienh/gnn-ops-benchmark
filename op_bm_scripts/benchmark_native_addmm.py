# Scatter ops

import torch
import torch.utils.benchmark as benchmark
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


def op_native_addmm(input, mat1, mat2):
    """Computes max scatter operation"""
    out = torch.addmm(input=input, mat1=mat1, mat2=mat2)
    return out


# def op_scatter_add_native(src, idx):
#     """Computes max scatter operation"""
#     temp = torch.zeros_like(src)
#     temp.scatter_(dim=0, index=idx, src=src, reduce="add")
#     return temp


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
setup_seed(42)
op_name = "native_addmm"
# native_exists = True
# FIXME: fix sparse conversion st that full mem capacity can be used
# length_ = int(40000 * 1.6)
# length__ = int(40000 * 1.2)
# length___ = int(40000 * 1.6)
length_ = int(4)
length__ = int(4)
length___ = int(4)
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
sparsities = [0, 0.5, 0.9, 0.99]


tshapes = [
    [
        (length__, length__),
        (length__, length__),
        (length__, length__),
    ],
    [(length_, length_), (length_, 1), (1, length_)],
    [(length___, 1), (length___, length___), (length___, 1)],
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
for sparsity_input in sparsities:
    for sparsity_A in sparsities:
        for sparsity_B in sparsities:
            for tshape in tshapes:

                torch.cuda.empty_cache()

                # if dim >= len(tshape):
                #     continue

                print(
                    f"DEBUG: Current input has dims {tshape[0]}, {tshape[1]}, {tshape[2]}"
                )

                input = torch.rand(
                    size=tshape[0],
                    device="cuda",
                    dtype=torch.float32,
                    requires_grad=False,
                )

                matA = torch.rand(
                    size=tshape[1],
                    device="cuda",
                    dtype=torch.float32,
                    requires_grad=False,
                )

                matB = torch.rand(
                    size=tshape[2],
                    device="cuda",
                    dtype=torch.float32,
                    requires_grad=False,
                )

                # randomly drop values to create sparsity
                input = torch.nn.functional.dropout(
                    input, p=sparsity_input, training=True, inplace=False
                )

                matA = torch.nn.functional.dropout(
                    matA, p=sparsity_A, training=True, inplace=False
                )

                matB = torch.nn.functional.dropout(
                    matB, p=sparsity_B, training=True, inplace=False
                )

                matB = matB.to_sparse()

                # begin benchmark logic
                t0 = benchmark.Timer(
                    stmt="op_native_addmm(input, matA, matB)",
                    setup="from __main__ import op_native_addmm",
                    globals={"input": input, "matA": matA, "matB": matB},
                )
                m0 = t0.blocked_autorange()

                bm_val = m0.median

                del m0
                del t0

                print_util_info()

                input_dims_str = str(len(tshape))

                params_str = input_dims_str

                formatted_input_dims = str(tshape)

                formatted_sparsities = (
                    str(sparsity_input)
                    + " ; "
                    + str(sparsity_A)
                    + " ; "
                    + str(sparsity_B)
                )

                data.append(
                    [params_str, formatted_input_dims, formatted_sparsities, bm_val]
                )

                del input
                del matA
                del matB

                print(f"done with {counter}")
                counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input dims",
    "Input size (>95% mem util)*",
    "Sparsities (input, matA, matB)",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

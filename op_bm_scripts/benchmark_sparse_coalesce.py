# Scatter ops

import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from torch_sparse import coalesce

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


def op_sparse_coalesce(index, value, m, n):
    val = coalesce(index=index, value=value, m=m, n=n)
    return val


def op_native_coalesce(mat):
    mat.coalesce()
    return None


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
setup_seed(42)
op_name = "sparse_coalesce"
# native_exists = True
length_ = int(4000 * 250000 * 0.12)
length__ = int(4000 * 3)
length___ = int(4000 * 0.20)
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
sparsities = [0.5, 0.9, 0.99]


tshapes = [(length_, 1), (length__, length__)]

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
        for reduce_factor in [1, 2, 4, 8]:

            torch.cuda.empty_cache()

            # if dim >= len(tshape):
            #     continue

            print(f"DEBUG: Current input has dims {tshape}")
            print("util at beg of loop: ")
            print_util_info()
            print()

            mat = torch.rand(
                size=tshape,
                device="cuda",
                dtype=torch.float32,
                requires_grad=False,
            )

            torch.cuda.empty_cache()

            # randomly drop values to create sparsity

            mat = torch.nn.functional.dropout(
                mat, p=sparsity, training=True, inplace=False
            )

            torch.cuda.empty_cache()

            m = mat.shape[0] * reduce_factor
            n = mat.shape[1] * reduce_factor

            torch.cuda.empty_cache()

            # print("Converting to COO")
            mat = mat.to_sparse()
            torch.cuda.empty_cache()

            # print("Converted to COO")

            index_ = mat.indices()
            value_ = mat.values()
            torch.cuda.empty_cache()

            # print(f"Reduce factor is {reduce_factor}")

            if reduce_factor == 1:
                index = index_
                value = value_

            if reduce_factor == 2:
                index = torch.cat((index_, index_), dim=1)
                value = torch.cat((value_, value_))

                index = index.index_select(
                    1, torch.randperm(index.shape[1], device="cuda")
                )

            if reduce_factor == 4:
                index = torch.cat((index_, index_, index_, index_), dim=1)
                value = torch.cat((value_, value_, value_, value_))

                index = index.index_select(
                    1, torch.randperm(index.shape[1], device="cuda")
                )

            if reduce_factor == 8:
                index = torch.cat(
                    (index_, index_, index_, index_, index_, index_, index_, index_),
                    dim=1,
                )
                value = torch.cat(
                    (value_, value_, value_, value_, value_, value_, value_, value_)
                )
                index = index.index_select(
                    1, torch.randperm(index.shape[1], device="cuda")
                )

            del mat
            torch.cuda.empty_cache()
            mat = torch.sparse_coo_tensor(index, value, (m, n))

            del index_
            del value_
            torch.cuda.empty_cache()

            # print(f"DEBUGGGGG: is coalesced: {mat.is_coalesced()}")

            # begin benchmark logic
            t0 = benchmark.Timer(
                stmt="op_sparse_coalesce(index, value, m, n)",
                setup="from __main__ import op_sparse_coalesce",
                globals={
                    "index": index,
                    "value": value,
                    "m": m,
                    "n": n,
                },
            )
            m0 = t0.blocked_autorange()

            bm_val = m0.median

            del m0
            del t0

            torch.cuda.empty_cache()

            print("Begin benchmark logic (native)")
            t1 = benchmark.Timer(
                stmt="op_native_coalesce(mat)",
                setup="from __main__ import op_native_coalesce",
                globals={
                    "mat": mat,
                },
            )
            m1 = t1.blocked_autorange()

            bm_native = m1.median

            del m1
            del t1

            print("util info at end of loop")
            print_util_info()
            print()

            input_dims_str = str(reduce_factor)

            params_str = input_dims_str

            formatted_input_dims = str(tshape)

            formatted_sparsities = str(sparsity)

            vals = str(bm_val) + " (" + str(bm_native) + ")"

            data.append([params_str, formatted_input_dims, formatted_sparsities, vals])

            del mat
            del index
            del value

            print(f"done with {counter}")
            counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Reduce factor",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

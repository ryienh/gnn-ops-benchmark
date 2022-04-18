# Scatter ops

import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from pdb import set_trace as bp


"""
Begin scatter ops
"""


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def op_native_index_select(input, dim, index):
    """Computes max scatter operation"""
    out = torch.index_select(input=input, dim=dim, index=index)
    return out


# def op_scatter_add_native(src, idx):
#     """Computes max scatter operation"""
#     temp = torch.zeros_like(src)
#     temp.scatter_(dim=0, index=idx, src=src, reduce="add")
#     return temp


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
# scatter_mean required 0.82 reduction factor
op_name = "native_index_select"
# native_exists = True
length_ = int(1600384000 * 1.6)
length__ = int(40000 * 1.65)
length___ = int(2000 * 0.82)
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
            for idx_reduce_factor in [1, 2, 4, 8]:

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

                # get index based on reduction factor
                index = torch.randint(
                    low=0,
                    high=input.shape[dim],
                    size=(int(input.shape[dim] / idx_reduce_factor),),
                    device="cuda",
                    requires_grad=False,
                )

                # begin benchmark logic
                t0 = benchmark.Timer(
                    stmt="op_native_index_select(input, dim, index)",
                    setup="from __main__ import op_native_index_select",
                    globals={"input": input, "dim": dim, "index": index},
                )
                m0 = t0.blocked_autorange()

                bm_val = m0.median

                del m0
                del t0

                print_util_info()

                input_dims_str = str(len(tshape))
                sort_dim_str = str(dim)
                idx_reduce_factor_str = str(idx_reduce_factor)

                params_str = (
                    input_dims_str + "; " + sort_dim_str + "; " + idx_reduce_factor_str
                )

                formatted_input_dims = str(tshape)

                data.append([params_str, formatted_input_dims, sparsity, bm_val])

                del input
                del index

                print(f"done with {counter}")
                counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input dims, index dim, reduce factor (RF)",
    "Input size (>95% mem util)*",
    "Sparsity",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

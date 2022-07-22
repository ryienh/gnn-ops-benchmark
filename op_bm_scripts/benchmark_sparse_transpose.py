# Scatter ops
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from torch_sparse import transpose


from graph_benchmark.benchmark.OpBenchmark import make_sparse

import matplotlib.pyplot as plt

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


def op_sparse_transpose(indexA, valueA, m, n):
    val = transpose(index=indexA, value=valueA, m=m, n=n)
    return val


def op_native_transpose(matA):
    """only swap 0 and 1 dims as that what torch_sparse.transpose supports."""
    out = torch.transpose(matA, 0, 1)
    return out


# Configurable hyperparams here
# dimensions: src size, idx size, src sparsity
setup_seed(42)
op_name = "sparse_transpose"
# native_exists = True
# length_ = int(50000 * 0.4)
length_ = 100
# length__ = int(5000000 * 20)
length__ = 10
# reduce_f_ = [1, 2, 4, 8]
# idx_dims = src_dims
sparsities = [0.5, 0.9, 0.99]


tshapes = [
    (length_, length_),
    # (length__, 1),
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

        torch.cuda.empty_cache()

        # if dim >= len(tshape):
        #     continue

        print(f"DEBUG: Current input has dims {tshape}")

        matA = torch.rand(
            size=tshape,
            device="cuda",
            dtype=torch.float32,
            requires_grad=False,
        )

        torch.cuda.empty_cache()

        # randomly drop values to create sparsity

        matA = make_sparse(matA, sparsity)
        print("DEBUG")
        print(matA.shape)
        sparsity = (torch.numel(matA) - torch.count_nonzero(matA)) / torch.numel(matA)
        print(f"Sparsity is {sparsity}")
        plt.imshow(matA.to("cpu").numpy())
        plt.colorbar()
        plt.savefig(f"debug_figs/{sparsity}.png")
        # print(matA)
        # matA = torch.nn.functional.dropout(
        #     matA, p=sparsity, training=True, inplace=False
        # )

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
            stmt="op_sparse_transpose(indexA, valueA, m, n)",
            setup="from __main__ import op_sparse_transpose",
            globals={
                "indexA": indexA,
                "valueA": valueA,
                "m": m,
                "n": n,
            },
        )

        # blocked autorange
        # m0 = t0.blocked_autorange()
        # new way
        bm_val = t0.timeit(100).median

        # bm_val = t0.median

        # del m0
        del t0

        torch.cuda.empty_cache()

        print("Begin benchmark logic (native)")
        t1 = benchmark.Timer(
            stmt="op_native_transpose(matA)",
            setup="from __main__ import op_native_transpose",
            globals={
                "matA": matA,
            },
        )
        m1 = t1.blocked_autorange()

        bm_native = m1.median

        del m1
        del t1

        print_util_info()

        input_dims_str = "LS" if tshape[1] == 1 else "Square"

        params_str = input_dims_str

        formatted_input_dims = str(tshape)

        formatted_sparsities = str(sparsity)

        vals = str(bm_val) + " (" + str(bm_native) + ")"

        data.append([params_str, formatted_input_dims, formatted_sparsities, vals])

        del matA
        del indexA
        del valueA

        print(f"done with {counter}")
        counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input Shape",
    "Input size (>95% mem util)*",
    "Sparsities (input, matA, matB)",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

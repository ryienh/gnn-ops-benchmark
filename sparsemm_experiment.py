# Scatter ops

import torch
import torch.utils.benchmark as benchmark
import pandas as pd

import matplotlib.pyplot as plt

from torch_sparse import spspmm

from pdb import set_trace as bp
import numpy as np


"""
Begin scatter ops
"""


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def op_sparse_spspmm(indexA, valueA, indexB, valueB, m, k, n):
    idx, val = spspmm(indexA, valueA, indexB, valueB, m, k, n)
    return idx, val


def op_native_sparsemm(matA, matB):
    out = torch.sparse.mm(matA, matB)
    return out


def op_native_dense(matA, matB):
    out = torch.mm(matA, matB)
    return out


# baseline
length__ = int(1000)


print("Making matricies")
matA = torch.rand(
    size=(length__, length__),
    device="cuda",
    dtype=torch.float32,
    requires_grad=False,
)

matB = torch.rand(
    size=(length__, length__),
    device="cuda",
    dtype=torch.float32,
    requires_grad=False,
)
tbaseline = benchmark.Timer(
    stmt="op_native_dense(matA, matB)",
    setup="from __main__ import op_native_dense",
    globals={
        "matA": matA,
        "matB": matB,
    },
)
mbaseline = tbaseline.blocked_autorange()
bm_val_baseline = mbaseline.median
del mbaseline
del tbaseline


N = 100
sparsities = np.logspace(np.log10(0.1), np.log10(1.0), num=100).tolist()
print(sparsities)


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
for sparsityA in reversed(sparsities):

    torch.cuda.empty_cache()

    # print_util_info()
    # bp()

    print("Making matricies")
    matA = torch.rand(
        size=(length__, length__),
        device="cuda",
        dtype=torch.float32,
        requires_grad=False,
    )

    matB = torch.rand(
        size=(length__, length__),
        device="cuda",
        dtype=torch.float32,
        requires_grad=False,
    )

    # randomly drop values to create sparsity
    matA = torch.nn.functional.dropout(matA, p=sparsityA, training=True, inplace=False)

    matB = torch.nn.functional.dropout(matB, p=sparsityA, training=True, inplace=False)

    print("DEBUGGGGG")
    print(f"{torch.count_nonzero(matA)}/{length__**2}")
    print(f"{torch.count_nonzero(matB)}/{length__**2}")

    # extract earlier stuff
    m = matA.shape[0]
    k = matA.shape[1]
    assert k == matB.shape[0]
    n = matB.shape[1]
    # convert to COO tensors
    print("Converting to COO")
    matA = matA.to_sparse()
    matB = matB.to_sparse()
    print("Converted to COO")

    # extract info for pytorch geo call
    indexA = matA.indices()
    valueA = matA.values()
    indexB = matB.indices()
    valueB = matB.values()

    # print("DEBUGGGGG")
    # print(len(indexA))
    # print(len(valueA))
    # print(len(indexB))
    # print(len(valueB))
    # print(valueA)

    # begin benchmark logic
    print("Begin benchmark logic (non-native)")
    t0 = benchmark.Timer(
        stmt="op_sparse_spspmm(indexA, valueA, indexB, valueB, m, k, n)",
        setup="from __main__ import op_sparse_spspmm",
        globals={
            "indexA": indexA,
            "valueA": valueA,
            "indexB": indexB,
            "valueB": valueB,
            "m": m,
            "k": k,
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
        stmt="op_native_sparsemm(matA, matB)",
        setup="from __main__ import op_native_sparsemm",
        globals={
            "matA": matA,
            "matB": matB,
        },
    )
    m1 = t1.blocked_autorange()

    print("Done benchmarking")

    bm_val_native = m1.median

    del m1
    del t1

    # t1 = benchmark.Timer(
    #     stmt="op_scatter_add_native(src, idx)",
    #     setup="from __main__ import op_scatter_add_native",
    #     globals={"src": src, "idx": idx},
    # )

    # m1 = t1.blocked_autorange()
    # bm_val_native = m1.median

    print_util_info()

    data.append([sparsityA, bm_val, bm_val_native])

    del matA
    del matB
    del indexA
    del valueA
    del indexB
    del valueB

    print(f"done with {counter}")
    counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Sparsity",
    "Time non-native",
    "Time native",
]
df.to_csv("data/experiment.csv")

# plot
sparsities = df[["Sparsity"]].values
non_native = df[["Time non-native"]].values
native = df[["Time native"]].values

print(len(sparsities))
print(len(non_native))
print(len(native))

plt.plot(sparsities, non_native, label="torch.sparse")
plt.plot(sparsities, native, label="native pytorch")
plt.plot(sparsities, [bm_val_baseline] * N, label="dense baseline")
plt.xlabel("Sparsity (log %)")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("experiment0log.png")

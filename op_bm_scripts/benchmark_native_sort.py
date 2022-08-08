"""
Profiles native_sort op
"""
import sys
import os

import torch
import torch.utils.benchmark as benchmark

sys.path.insert(0, "/home/rhosseini/gnn-kernel-benchmark")  # FIXME

from graph_benchmark.benchmark.util import (
    setup_seed,
    print_util_info,
    setup_cuda,
    empty_cache,
    print_sparsity_info,
    print_bm_stats,
    print_input_dims,
)

from graph_benchmark.benchmark.DataWriter import DataWriter


# Begin bm script body

# Define op for benchmarking
def op_native_sort(input, dim, stable):
    out, idx = torch.sort(input=input, dim=dim, stable=stable)
    return out, idx


setup_seed(42)

# Define parameters to sweep over
op_name = "native_sort"
length_ = int(1600384000 * 0.5)
length__ = int(40000 * 0.5)
length___ = int(2000 * 0.4)
sparsities = [0, 0.5, 0.9, 0.99]
tshapes = [
    (length_,),
    (length__, length__),
    (length___, length___, length___),
]

# create data (list of dicts) for csv creation, set up cuda, empty cache
dw = DataWriter(op_name=op_name, param_names="Input dims, sort dim, stable")
device = setup_cuda()
empty_cache()

counter = 0

# define inputs over hyperparams
for sparsity in sparsities:

    # torch.cuda.nvtx.range_push(f"sparsity: {sparsity}")
    # with torch.autograd.profiler.emit_nvtx():

    for tshape in tshapes:

        # torch.cuda.nvtx.range_push(f"tshape: {tshape}")

        for dim in [0, 1, 2]:

            # torch.cuda.nvtx.range_push(f"dim: {dim}")

            for stable in [True, False]:

                empty_cache()

                # print_util_info()
                # bp()

                if dim >= len(tshape):
                    continue

                torch.cuda.nvtx.range_push(
                    f"sparsity: {sparsity}, tshape: {tshape}, dim: {dim}, stable: {stable}"
                )

                with torch.autograd.profiler.emit_nvtx():

                    # sanity check
                    print_input_dims(tshape)

                    input = torch.rand(
                        size=tshape,
                        device="cuda",
                        dtype=torch.float32,
                        requires_grad=False,
                    )

                    # randomly drop values to create sparsity
                    input = torch.nn.functional.dropout(
                        input, p=sparsity, training=True, inplace=False
                    )

                    print_sparsity_info(sparsity=sparsity, input=input)

                    # begin benchmark logic
                    t0 = benchmark.Timer(
                        stmt="op_native_sort(input, dim, stable)",
                        setup="from __main__ import op_native_sort",
                        globals={
                            "input": input,
                            "dim": dim,
                            "stable": stable,
                        },
                    )
                    m0 = t0.blocked_autorange(min_run_time=1)

                    bm_val = m0.median

                    print_bm_stats(m0)

                    del m0
                    del t0

                    print_util_info()

                    # convert inputs to string
                    input_dims_str = str(len(tshape))
                    sort_dim_str = str(dim)
                    stable_str = str(stable)

                    dw.add_entry(
                        params_lst=[input_dims_str, sort_dim_str, stable_str],
                        tshape=tshape,
                        sparsity=sparsity,
                        bm_val=bm_val,
                    )

                    del input

                    print(f"done with {counter}")
                    counter += 1


dw.write_data(path=os.path.join("./", "datatest"))

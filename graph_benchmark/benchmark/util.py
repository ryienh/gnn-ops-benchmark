"""
TODO: documentation

"""
import random
import numpy as np
import pandas as pd
import torch


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def get_reserved_in_mb():
    return torch.cuda.memory_reserved() / 1000000


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def setup_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise Exception("Benchmarking only supported for CUDA")
    return device


def empty_cache():
    torch.cuda.empty_cache()


def print_sparsity_info(sparsity, input, verbose=True):
    if verbose:
        print(
            f"Sparsity info: {sparsity}, percent non-zero is: {torch.count_nonzero(input) / input.numel()}"
        )


def print_bm_stats(m0, verbose=True):
    if verbose:
        print(
            f"Benchmark blocked autorange stats: median is {m0.median}, iqr is {m0.iqr}, count is {len(m0.times)}"
        )


def print_input_dims(tshape):
    print(f"DEBUG: Current input has dims {len(tshape)}")

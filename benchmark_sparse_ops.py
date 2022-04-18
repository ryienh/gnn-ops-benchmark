# Scatter ops
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from graph_benchmark.datasets.fakeDatasets import FakeDataset
from torch_geometric.loader import DataLoader

from torch_spline_conv import spline_conv


"""
Begin scatter ops
"""


def print_util_info():
    print("GPU INFO:")
    print("\t Memory allocated: ", (torch.cuda.memory_allocated() / 4e10))
    print("\t Memory reserved: ", (torch.cuda.memory_reserved() / 4e10))


def combine_vals(bm_val, bm_val_native):
    return str(bm_val) + " (" + str(bm_val_native) + ")"


def op_spline_conv_spline_conv(
    x, edge_index, pseudo, weight, kernel_size, is_open_spline
):
    """Computes max scatter operation"""
    out = spline_conv(
        x,
        edge_index,
        pseudo,
        weight,
        kernel_size,
        is_open_spline,
        degree=1,
        norm=True,
        root_weight=None,
        bias=None,
    )
    return out


# Configurable hyperparams here
op_name = "spline_conv_spline_conv"


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
avg_degree = [0.0, 0.1, 0.5, 1, 2, 5, 10]
for sparsity in avg_degree:

    kernel_size = torch.tensor([5, 5])

    dataset = FakeDataset(
        num_graphs=1,
        avg_num_nodes=10000,
        avg_degree=2,
        num_channels=25,
        edge_dim=2,
        num_classes=2,
        task="auto",
        is_undirected=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    data = next(loader)

    x, edge_index, batch = (
        data.x,
        data.edge_index,
        data.batch,
    )

    pseudo = torch.rand((edge_index[0].shape, edge_index[1].shape), dtype=torch.float)
    weight = torch.rand(
        (25, edge_index[1].shape, edge_index[0].shape), dtype=torch.float
    )

    torch.cuda.empty_cache()

    # print_util_info()
    # bp()

    # begin benchmark logic
    t0 = benchmark.Timer(
        stmt="op_spline_conv_spline_conv(x, edge_index, pseudo, weight, kernel_size, is_open_spline)",
        setup="from __main__ import op_spline_conv_spline_conv",
        globals={
            "x": x,
            "edge_index": edge_index,
            "pseudo": pseudo,
            "weight": weight,
            "kernel_size": kernel_size,
            "is_open_spline": True,
        },
    )
    m0 = t0.blocked_autorange()

    bm_val = m0.median

    del m0
    del t0

    print_util_info()

    input_dims_str = len(tshape)
    sort_dim_str = str(dim)
    # idx_reduce_factor_str = str(reduce_f)

    formatted_input_dims = str(x.shape)

    sparsity_str = str(sparsity)

    # output_value = (
    #     bm_val if not native_exists else combine_vals(bm_val, bm_val_native)
    # )

    data.append([formatted_input_dims, sparsity_str, bm_val])

    del pseudo
    del weight
    del pseudo
    del x
    del edge_index
    del batch
    del loader
    del dataset

    # print_util_info()

    # torch.cuda.empty_cache()

    print(f"done with {counter}")
    counter += 1

df = pd.DataFrame(data)
df.columns = [
    "Input size (>95% mem util)*",
    "Average node degree",
    "GPU clock time",
]
df.to_csv(f"new_data/{op_name}.csv")

import torch
import torch_geometric
import warnings
import tqdm

from torch_geometric.nn import FiLMConv, GINConv, CGConv, PNAConv, SAGEConv
from torch_geometric.profile import (
    profileit,
    get_stats_summary,
    get_model_size,
    get_data_size,
)
from torch_geometric.datasets import QM9, GNNBenchmarkDataset, GEDDataset, IMDB
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


# hyperparams
class Config:
    n = 100
    batch_size = 512


def get_degree_hist(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


@profileit()
def run_single_inst(model, x, edge_index):
    _ = model(x, edge_index)


def inference(model, loader):

    stats = []

    for idx, data in tqdm.tqdm(enumerate(loader), total=Config.n):

        x = data.x.cuda()
        edge_index = data.edge_index.cuda()

        # total compute only
        stat = run_single_inst(model, x, edge_index)[1]
        stats.append(stat)

        if idx == Config.n:
            break

    return stats


# get dataloader w shuffle
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # qm9
    dataset = QM9(root="/tmp/QM9")
    dataset_name = "QM9"

    # filmconv
    model_name = "FiLMConv"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = FiLMConv(in_channels=11, out_channels=2048).cuda()
    stats = inference(model, loader)
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{get_stats_summary(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()

    # GIN
    model_name = "GIN"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = GINConv(torch.nn.Linear(11, 2048)).cuda()
    stats = inference(model, loader)
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{get_stats_summary(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()

    # CGConv
    model_name = "CGConv"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = CGConv(11, 0).cuda()
    stats = inference(model, loader)
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{get_stats_summary(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()

    # MNIST
    dataset = GNNBenchmarkDataset(root="/tmp/MNIST", name="MNIST")
    dataset_name = "MNIST"

    # pnaconv
    model_name = "PNA"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = PNAConv(
        in_channels=1,
        out_channels=2048,
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "attenuation"],
        deg=get_degree_hist(dataset),
    ).cuda()
    stats = inference(model, loader)
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{get_stats_summary(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )

    # IMDBMULTI

    #
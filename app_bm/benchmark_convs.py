from asyncio import transports
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
from torch_geometric.datasets import QM9, GNNBenchmarkDataset, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
import numpy as np

# hyperparams
class Config:
    n = 300
    batch_size = 1


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


# def run_single_inst(model, x, edge_index):
#     starter.record()
#     _ = model(x, edge_index)
#     ender.record()
#     torch.cuda.synchronize()
#     curr_time = starter.elapsed_time(ender)
#     return curr_time


def inference(model, loader, start, end):

    stats = []
    warmup_cnt = 0
    for idx, data in tqdm.tqdm(enumerate(loader), total=Config.n):

        with torch.no_grad():

            x = data.x.to(torch.float16).cuda()
            edge_index = data.edge_index.cuda()

            # total compute only
            if warmup_cnt < 10:
                _ = model(x, edge_index)
            else:
                start.record()
                _ = model(x, edge_index)
                end.record()
                torch.cuda.synchronize()
                stat = start.elapsed_time(end)
                stats.append(stat)

            if idx == Config.n:
                break

        warmup_cnt += 1

    return stats


def inference_with_profs(model, loader):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/tensorboard/{model_name}_{dataset_name}_inf"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof_inf:

        with torch.no_grad():

            for idx, data in tqdm.tqdm(enumerate(loader), total=len(loader)):

                x = data.x.cuda()
                edge_index = data.edge_index.cuda()

                out = model(x, edge_index)

                prof_inf.step()


# with warnings.catch_warnings():
#     warnings.simplefilter(action="ignore", category=FutureWarning)

#     # qm9
#     dataset = QM9(root="/tmp/QM9")
#     dataset_name = "QM9"

#     # filmconv
#     model_name = "FiLMConv"
#     loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
#     model = FiLMConv(in_channels=11, out_channels=2048).cuda()
#     stats = inference_with_profs(model, loader)


# get dataloader w shuffle
with warnings.catch_warnings():

    torch.cuda.empty_cache()
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # qm9
    dataset = QM9(root="/tmp/QM9")
    dataset_name = "QM9"

    # filmconv
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    model_name = "FiLMConv"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = FiLMConv(in_channels=11, out_channels=2048).to(torch.float16).cuda()
    stats = np.array(inference(model, loader, starter, ender))
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{np.mean(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()

    del starter, ender

    # GIN
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    model_name = "GIN"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = GINConv(torch.nn.Linear(11, 2048)).to(torch.float16).cuda()
    stats = inference(model, loader, starter, ender)
    stats = np.array(stats)
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{np.mean(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()

    del starter, ender

    # CGConv
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    model_name = "CGConv"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = CGConv(11, 0).to(torch.float16).cuda()
    stats = np.array(inference(model, loader, starter, ender))
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{np.mean(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()
    del starter, ender

    # MNIST
    dataset = GNNBenchmarkDataset(root="/tmp/MNIST", name="MNIST")
    dataset_name = "MNIST"

    # pnaconv
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    model_name = "PNA"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = (
        PNAConv(
            in_channels=1,
            out_channels=2048,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=get_degree_hist(dataset),
        )
        .to(torch.float16)
        .cuda()
    )
    stats = np.array(inference(model, loader, starter, ender))
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{np.mean(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()
    del starter, ender

    # IMDBMULTI
    dataset = TUDataset(
        root="/tmp/IMDB-MULTI", name="IMDB-MULTI", transform=OneHotDegree(88)
    )
    dataset_name = "IMDB-MULTI"

    # SAGEConv
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    model_name = "GraphSAGE"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = SAGEConv(-1, 2048).to(torch.float16).cuda()
    stats = np.array(inference(model, loader, starter, ender))
    print(f"Statistics for model {model_name} and dataset {dataset_name}")
    print(f"\t{np.mean(stats)}")
    print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    print(
        f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    )
    print()
    del starter, ender
    #####
    # cache test
    # more tests

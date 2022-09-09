from asyncio import transports
import torch
import torch_geometric
import warnings
import tqdm
import numpy as np

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

from torch_geometric.data import Dataset, Data


# hyperparams
class Config:
    n = 100
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


@profileit()
def run_single_inst(model, x, edge_index):
    _ = model(x, edge_index)


def inference(model, loader):

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
                stat = run_single_inst(model, x, edge_index)[1]
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
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # qm9
    dataset = QM9(root="/tmp/QM9")
    dataset_name = "QM9"

    # # filmconv
    # model_name = "FiLMConv"
    # loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    # model = FiLMConv(in_channels=11, out_channels=2048).to(torch.float16).cuda()
    # stats = inference(model, loader)
    # print(f"Statistics for model {model_name} and dataset {dataset_name}")
    # print(f"\t{get_stats_summary(stats)}")
    # print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    # print(
    #     f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    # )
    # print()

    # CGConv
    model_name = "CGConv"
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = CGConv(11, 0).to(torch.float16).cuda()

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    x_qm9 = torch.rand(29, 11).to(torch.float16)
    edge_index_qm9 = torch.rand(2, 56).to(torch.long)
    y_qm9 = torch.tensor(1)

    data = Data(x=x_qm9, edge_index=edge_index_qm9)
    data.to(DEVICE)
    # GPU-WARM-UP
    for rep, data in tqdm.tqdm(enumerate(loader), total=Config.n):

        with torch.no_grad():

            x = data.x.to(torch.float16).cuda()
            edge_index = data.edge_index.cuda()

            # total compute only
            if rep < 10:
                _ = model(x, edge_index)
            else:
                starter.record()
                _ = model(x, edge_index)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

            if rep >= 299:
                break

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
    # stats = inference(model, loader)
    # print(f"Statistics for model {model_name} and dataset {dataset_name}")
    # print(f"\t{get_stats_summary(stats)}")
    # print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    # print(
    #     f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    # )
    # print()

    # # MNIST
    # dataset = GNNBenchmarkDataset(root="/tmp/MNIST", name="MNIST")
    # dataset_name = "MNIST"

    # # pnaconv
    # model_name = "PNA"
    # loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    # model = (
    #     PNAConv(
    #         in_channels=1,
    #         out_channels=2048,
    #         aggregators=["mean", "min", "max", "std"],
    #         scalers=["identity", "amplification", "attenuation"],
    #         deg=get_degree_hist(dataset),
    #     )
    #     .to(torch.float16)
    #     .cuda()
    # )
    # stats = inference(model, loader)
    # print(f"Statistics for model {model_name} and dataset {dataset_name}")
    # print(f"\t{get_stats_summary(stats)}")
    # print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    # print(
    #     f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    # )
    # print()

    # # IMDBMULTI
    # dataset = TUDataset(
    #     root="/tmp/IMDB-MULTI", name="IMDB-MULTI", transform=OneHotDegree(88)
    # )
    # dataset_name = "IMDB-MULTI"

    # # SAGEConv
    # model_name = "GraphSAGE"
    # loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    # model = SAGEConv(-1, 2048).to(torch.float16).cuda()
    # stats = inference(model, loader)
    # print(f"Statistics for model {model_name} and dataset {dataset_name}")
    # print(f"\t{get_stats_summary(stats)}")
    # print(f"\tModel actual disk size in mb: {get_model_size(model) * 1e-6}")
    # print(
    #     f"\tData example theoretical data usage in mb: {get_data_size(next(iter(loader))) * 1e-6}"
    # )
    # print()
    # #

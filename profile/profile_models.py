"""
Script to profile CUDA kernel use for several GNN models

    Models should be contained in ../models dir
    Workflow is as follows:
        1. Models are profiled by pytorch profiler
        2. Profile data is optionally visualized using Tensorboard
        3. Profile information is saved to file
        4. Profile infomation is then explored in one or more .ipynb in ./notebooks

    TODO: 
    1. add type suggestions
    2. fix hardcodes
    3. fix data encapsulation
    4. remove config dependencies from KernelProfiler
"""

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "./models")  # FIXME: make modules
sys.path.insert(1, "./datasets")  # FIXME: make modules

import torch
from ptg_models import AttentiveFPREG, GraphUNetREG, SchNetREG
from fakeDatasets import FakeDataset
from torch.profiler import profile, record_function, ProfilerActivity
from torch_geometric.loader import DataLoader

# from util import custom_table
import pandas as pd
import tqdm
import os
import click


class KernelProfiler:
    def __init__(self, config_path):
        self.__config_path = config_path

        self.models = self._config("models")
        self.datasets = self._config("datasets")
        self.batchsize = self._config("batch_size")
        self.num_graphs = self._config("num_graphs")

    def _config(self, attr):
        # FIXME: fix hardcodes
        with open(self.__config_path) as f:
            d = eval(f.read())
        node = d
        for part in attr.split("."):
            node = node[part]
        return node

    def _init_model_from_str(self, model, loader):

        in_dims = next(iter(loader)).x.shape[1]

        if model == "AttentiveFPREG":
            return AttentiveFPREG(input_dim=in_dims)
        elif model == "GraphUNetREG":
            return GraphUNetREG(input_dim=in_dims)
        elif model == "SchNetREG":
            return SchNetREG()
        else:
            raise NotImplementedError(f"Model {model} not yet implemented.")

    def _get_data_loaders_from_str(self, dataset):
        if dataset == "FakeDataset":
            dataset = FakeDataset(num_graphs=self.num_graphs)
        else:
            raise NotImplementedError(f"Dataset {dataset} not yet implemented")

        # create loader
        tr_loader = DataLoader(
            dataset[: int(len(dataset) * 0.8)], batch_size=self.batchsize, shuffle=True
        )
        va_loader = DataLoader(
            dataset[int(len(dataset) * 0.8) :], batch_size=self.batchsize, shuffle=False
        )

        return tr_loader, va_loader

    def _write_prof_tables_to_file(self, prof_train, prof_inf, model, dataset):
        with open(
            os.path.join("./", "data", "profile", f"{model}_{dataset}_train"), "w"
        ) as out:
            out.write(
                prof_train.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total"
                )
            )

        with open(
            os.path.join("./", "data", "profile", f"{model}_{dataset}_inf"), "w"
        ) as out:
            out.write(
                prof_inf.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total"
                )
            )

    def profile_model(self, model_name, dataset):
        """
        Profiles a given model and dataset.

        Requires CUDA. Records traces for tensorboard visualization, and writes summary to file.
        """

        # set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise Exception(f"Profiler requires CUDA but device is {device}")

        # init dataset
        tr_loader, va_loader = self._get_data_loaders_from_str(dataset)

        # init model
        model = (
            self._init_model_from_str(model_name, tr_loader)
            .to(device)
            .to(torch.float32)
        )  # FIXME: fix precision hardcode

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # call profiler
        # 1. train mode
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self._config("profiler.wait"),
                warmup=self._config("profiler.warmup"),
                active=self._config("profiler.active"),
                repeat=self._config("profiler.repeat"),
            ),  # FIXME: fix hardcodes
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./log/tensorboard/{model_name}_{dataset}_train"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof_train:
            with record_function("model_train"):
                # single forward pass
                model = model.train()

                for X in tqdm.tqdm(tr_loader):
                    X = X.to(device)

                    optimizer.zero_grad()

                    prediction = model(X)
                    prediction = torch.squeeze(prediction)

                    loss = model.loss(
                        prediction, X.y.to(torch.float32)
                    )  # FIXME: fix precision hardcode

                    loss.backward()
                    optimizer.step()

                    prof_train.step()

        # 2. inference mode
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self._config("profiler.wait"),
                warmup=self._config("profiler.warmup"),
                active=self._config("profiler.active"),
                repeat=self._config("profiler.repeat"),
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./log/tensorboard/{model_name}_{dataset}_inf"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof_inf:
            with record_function("model_inf"):
                # single forward pass
                model = model.eval()

                with torch.no_grad():

                    for X in tqdm.tqdm(va_loader):
                        X = X.to(device)

                        prediction = model(X)
                        prediction = torch.squeeze(prediction)
                        loss = model.loss(prediction, X.y)

                        prof_inf.step()

        # print top kernels to console
        if self._config("verbose"):

            print(
                prof_train.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=20
                )
            )

            print(
                prof_inf.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=20
                )
            )

        # write summaries to file
        self._write_prof_tables_to_file(prof_train, prof_inf, model_name, dataset)

    def profile_models(self):
        """
        Profile list of models and datasets
        """

        if self._config("verbose"):
            print("Profiling models: ")

            print(self.models)
            print("On datasets:")
            print(self.datasets)

        for model in self.models:
            for dataset in self.datasets:

                if self._config("verbose"):
                    print(f"Beginning profiling on {model} with {dataset}")

                self.profile_model(
                    model,
                    dataset,
                )


@click.command()
@click.option(
    "--config",
    default="./config.json",
    prompt="Enter relative path to config file.",
    help="Relative path to config file. Must be in json format.",
)
def main(config):

    # init Profiler
    my_profiler = KernelProfiler(config)
    my_profiler.profile_models()


if __name__ == "__main__":
    main()

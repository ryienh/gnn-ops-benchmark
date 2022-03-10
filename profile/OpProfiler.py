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
    4. remove config dependencies from OpProfiler
"""

# import sys

# # insert at 1, 0 is the script path (or '' in REPL)
# # sys.path.insert(1, "./models")  # FIXME: make modules
# sys.path.insert(1, "./datasets")  # FIXME: make modules

import torch
from ..models.ptg_models import AttentiveFPREG, GraphUNetREG, SchNetREG
from ..datasets.fakeDatasets import FakeDataset
from torch_geometric.loader import DataLoader

import tqdm
import os


class OpProfiler:
    def __init__(self, config_path):
        self.__config_path = config_path

        self.models = self._config("models")  # list of dics, once for each model
        self.model_names = [model["name"] for model in self.models]
        self.datasets = self._config("datasets")

    def _config(self, attr):
        # FIXME: fix hardcodes
        with open(self.__config_path) as f:
            d = eval(f.read())
        node = d
        for part in attr.split("."):
            node = node[part]
        return node

    def _init_model_from_str(self, idx, loader):

        in_dims = next(iter(loader)).x.shape[1]

        if self.model_names[idx] == "AttentiveFPREG":
            # unpack config args
            try:
                hidden_dim = self.models[idx]["hidden_dim"]
                dropout = self.models[idx]["dropout"]
                num_conv_layers = self.models[idx]["num_conv_layers"]
                num_out_channels = self.models[idx]["num_out_channels"]
                edge_dim = self.models[idx]["edge_dim"]
                num_timesteps = self.models[idx]["num_timesteps"]

            except KeyError:
                print(
                    f"Make sure config file contains all relevant params for model {self.model_names[idx]}"
                )
                return

            # init model
            return AttentiveFPREG(
                input_dim=in_dims,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_conv_layers=num_conv_layers,
                num_out_channels=num_out_channels,
                edge_dim=edge_dim,
                num_timesteps=num_timesteps,
            )
        elif self.model_names[idx] == "GraphUNetREG":
            return GraphUNetREG(input_dim=in_dims)
        elif self.model_names[idx] == "SchNetREG":
            return SchNetREG()
        else:
            raise NotImplementedError(
                f"Model {self.model_names[idx]} not yet implemented."
            )

    def _get_data_loaders_from_str(self, model_idx, dataset):

        try:
            batch_size = self.models[model_idx]["batch_size"]
        except KeyError:
            print(
                f"Model {self.model_names[model_idx]} does not have a batch_size value. Please update config file and try again."
            )
            return
        try:
            num_graphs = self.models[model_idx]["num_graphs"]
        except KeyError:
            print(
                f"Model {self.model_names[model_idx]} does not have a num_graphs value. Please update config file and try again."
            )
            return

        if dataset == "FakeDataset":
            dataset = FakeDataset(num_graphs=num_graphs)
        else:
            raise NotImplementedError(f"Dataset {dataset} not yet implemented")

        # create loader
        tr_loader = DataLoader(
            dataset[: int(len(dataset) * 0.8)],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        va_loader = DataLoader(
            dataset[int(len(dataset) * 0.8) :],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
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

    def profile_model(self, idx, dataset):
        """
        Profiles a given model and dataset.

        Requires CUDA. Records traces for tensorboard visualization, and writes summary to file.
        """

        model_name = self.model_names[idx]

        # set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise Exception(f"Profiler requires CUDA but device is {device}")

        # init dataset
        tr_loader, va_loader = self._get_data_loaders_from_str(idx, dataset)

        # init model
        model = (
            self._init_model_from_str(idx, tr_loader).to(device).to(torch.float32)
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
            # with record_function("model_train"):
            # single forward pass + backprop
            model = model.train()

            for X in tqdm.tqdm(tr_loader):
                X = X.to(device)

                optimizer.zero_grad()

                prediction = model(X)
                prediction = prediction.view(-1)

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
            # with record_function("model_inf"):
            # single forward pass
            model = model.eval()

            with torch.no_grad():

                for X in tqdm.tqdm(va_loader):
                    X = X.to(device)

                    prediction = model(X)
                    prediction = prediction.view(-1)
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
        self._write_prof_tables_to_file(
            prof_train, prof_inf, self.model_names[idx], dataset
        )

    def profile_models(self):
        """
        Profile list of models and datasets
        """

        if self._config("verbose"):
            print("Profiling models: ")

            print(self.model_names)
            print("On datasets:")
            print(self.datasets)

        for idx, model in enumerate(self.model_names):
            for dataset in self.datasets:

                if self._config("verbose"):
                    print(f"Beginning profiling on {model} with {dataset}")

                self.profile_model(
                    idx,
                    dataset,
                )
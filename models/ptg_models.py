"""

Graph convolutional network models for kernel profiling.

All models in this file are invoked using the pytorch geometric API

For questions or comments, contact rhosseini@anl.gov

TODO:
1. fix model defs to pure GNN

"""
import torch
from torch_geometric.nn import global_mean_pool, AttentiveFP, GraphUNet, SchNet


"""
Begin model defns.

Models selected from lit. review and include commonly used operations for scientific applications.
Each model impl includes single mean pooling with a single lienar layer (where not defined by model)
in order to avoid obstucting GNN kernel ops

"""


"""
From Xiong et al Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism

https://pubs.acs.org/doi/pdf/10.1021/acs.jmedchem.9b00959
"""


class AttentiveFPREG(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=250,
        dropout=0.0,
        num_conv_layers=10,
        num_out_channels=2,
        edge_dim=1,  # FIXME: fix hardcode
        num_timesteps=2,
    ):

        super(AttentiveFPREG, self).__init__()

        # call attentive model
        self.attentive_fp = AttentiveFP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=num_out_channels,
            edge_dim=edge_dim,
            num_layers=num_conv_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        # fully connected layers
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(num_out_channels, int(num_out_channels / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(num_out_channels / 2), 1),
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )
        edge_attr = torch.ones((edge_index.shape[1], 1)).cuda()
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        # call model
        x = self.attentive_fp(x, edge_index, edge_attr, batch)

        # MLP
        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)


"""
From Gao and Ji Graph U-Nets

https://arxiv.org/abs/1905.05178
"""


class GraphUNetREG(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_out_channels=10,
        depth=10,
    ):

        super(GraphUNetREG, self).__init__()

        # call attentive model
        self.attentive_fp = GraphUNet(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=num_out_channels,
            depth=depth,
        )
        # fully connected layers
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(num_out_channels, int(num_out_channels / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(num_out_channels / 2), 1),
        )

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        # call model
        x = self.attentive_fp(x, edge_index, batch)

        # pooling
        x = global_mean_pool(x, batch)

        # MLP
        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)


"""
From Schutt SchNet: A continuous-filter convolutional neural network for modeling quantum interactions

https://arxiv.org/abs/1706.08566
"""


class SchNetREG(torch.nn.Module):
    def __init__(
        self,
        num_out_channels=10,
    ):

        super(SchNetREG, self).__init__()

        # call attentive model
        self.attentive_fp = (
            SchNet()
        )  # all defaults right now -- add explicit hyperparams during benchmark
        # fully connected layers
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(num_out_channels, int(num_out_channels / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(num_out_channels / 2), 1),
        )

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        # call model
        x = self.attentive_fp(x, edge_index, batch)

        # # pooling
        # x = global_mean_pool(x, batch)

        # MLP
        # x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)

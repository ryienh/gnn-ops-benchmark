"""

Graph convolutional network models for kernel profiling.

All models in this file are invoked using the pytorch geometric API

For questions or comments, contact rhosseini@anl.gov

TODO:
1. fix model defs to pure GNN

"""
import torch
from torch_geometric.nn import global_mean_pool, AttentiveFP, GraphUNet, GATv2Conv


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
        hidden_dim,
        dropout,
        num_conv_layers,
        num_out_channels,
        edge_dim,
        num_timesteps,
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
            torch.nn.Linear(num_out_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )
        # FIXME: fix hardcodes
        x = x.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()

        edge_attr = torch.ones((edge_index.shape[1], 1)).cuda()
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        # call model
        x = self.attentive_fp(x, edge_index, edge_attr, batch)

        # # pooling
        # x = global_mean_pool(x, batch)

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
    def __init__(self, input_dim, hidden_dim, num_out_channels, depth):

        super(GraphUNetREG, self).__init__()

        # call attentive model
        self.graph_unet = GraphUNet(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=num_out_channels,
            depth=depth,
        )
        # fully connected layers
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(num_out_channels, 1),
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
        x = self.graph_unet(x, edge_index, batch)

        # # pooling
        # x = global_mean_pool(x, batch)

        # MLP
        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)


"""
GATv2 Operator from Brody et al

https://arxiv.org/abs/2105.14491
"""


class GATv2REG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, heads):

        super(GATv2REG, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, heads))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim, heads):
        return GATv2Conv(
            in_channels=input_dim, out_channels=hidden_dim, heads=heads, concat=False
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        # pooling
        x = global_mean_pool(x, batch)

        # MLP
        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)

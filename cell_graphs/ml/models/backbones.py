from typing import Literal

import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP


class MessagePassingLayer(nn.Module):
    """Single message passing layer with graph convolution, activation, dropout and batch norm."""

    GraphConv = Literal["GCNConv", "GINConv", "SAGEConv"]
    Norm = Literal["BatchNorm", "GraphNorm"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv: GraphConv,
        norm: Norm,
        dropout: float = 0.0,
    ) -> None:
        """Args:
        ----
            in_channels: Number of input dimensions.
            out_channels: Number of output dimensions.
            conv: Message passing layer to use. Can be one of `GCNConv`, `GINConv` or `SAGEConv`.
            norm: Normalization to use. Can be either `GraphNorm` or `BatchNorm`.
            dropout: Dropout ratio for the message passing layer.
        """
        super().__init__()

        if conv == "GCNConv":
            self.conv = gnn.conv.GCNConv(in_channels, out_channels)
        elif conv == "GINConv":
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, out_channels),
            )
            self.conv = gnn.conv.GINConv(gin_mlp)
        elif conv == "SAGEConv":
            self.conv = gnn.conv.SAGEConv(in_channels, out_channels)
        else:
            raise NotImplementedError

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        if norm == "BatchNorm":
            self.norm = gnn.norm.BatchNorm(out_channels)
        elif norm == "GraphNorm":
            self.norm = gnn.norm.GraphNorm(out_channels)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch) -> Batch:
        """Run a forward pass of the message passing layer.

        Applies steps in the following order:
        * Message passing
        * Activation
        * Dropout
        * Skip-connection
        * Normalization

        Args:
        ----
            batch: Batch of graphs to apply message passing on.

        Returns:
        -------
        Batch of graphs with updated node representations.
        """
        h = batch.x
        h_in = h

        h_conv = self.conv(h, batch.edge_index)
        h_conv = self.act(h_conv)
        h_conv = self.dropout(h_conv)
        h_conv = h_conv + h_in
        h_conv = (
            self.norm(h_conv)
            if isinstance(self.norm, gnn.norm.BatchNorm)
            else self.norm(h_conv, batch.batch)
        )

        batch.x = h_conv
        return batch


class PoolingLayer(nn.Module):
    """Graph pooling layer."""

    GraphPooling = Literal["SAGPooling", "TopKPooling"]

    def __init__(self, channels: int, pooling_ratio: float, pooling: GraphPooling) -> None:
        """Args:
        ----
            channels: Number of input and output dimensions.
            pooling_ratio: Share of nodes to drop from the graph.
            pooling: Graph pooling variant. Can be either `TopKPooling` or `SAGPooling`.
        """
        super().__init__()

        if pooling == "TopKPooling":
            self.pool = gnn.pool.TopKPooling(channels, ratio=pooling_ratio)
        elif pooling == "SAGPooling":
            self.pool = gnn.pool.SAGPooling(channels, ratio=pooling_ratio)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch) -> Batch:
        """Run a forward pass of the graph pooling operation on a batch.

        Args:
        ----
            batch: Batch to apply graph pooling on.

        Returns:
        -------
        Pooled graph batch.
        """
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, _, _ = self.pool(
            batch.x, batch.edge_index, getattr(batch, "edge_attr", None), batch.batch
        )
        return batch


class SurvivalGNN(nn.Module):
    """Modular GNN class for survival prediction on cell graphs."""

    GraphConv = Literal["GCNConv", "GINConv", "SAGEConv"]
    GraphPooling = Literal["SAGPooling", "TopKPooling"]
    Norm = Literal["BatchNorm", "GraphNorm"]

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        conv: GraphConv,
        pooling: GraphPooling | None = None,
        norm: Norm = "BatchNorm",
        dropout: float = 0.0,
        pooling_ratio: float = 0.5,
        num_pre_layers: int = 1,
        num_post_layers: int = 2,
    ) -> None:
        """Args:
        ----
            input_dim: Number of input dimensions.
            embedding_dim: Number of graph embedding (output) dimensions.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            conv: Message passing layer.
            pooling: Graph pooling layer.
            norm: Normalization layer.
            dropout: Dropout ratio to apply.
            pooling_ratio: Share of graph nodes to drop during graph pooling.
            num_pre_layers: Number of MLP layers to encode node features before message passing.
            num_post_layers: Number of MLP layers to encode node features after message passing.
        """
        super().__init__()

        self.node_enc = MLP(
            in_channels=input_dim, out_channels=hidden_dim, num_layers=num_pre_layers
        )

        blocks = nn.ModuleList()

        for _ in range(num_layers):
            layers = nn.ModuleList()

            layers.append(
                MessagePassingLayer(
                    hidden_dim,
                    hidden_dim,
                    conv,
                    norm,
                    dropout=dropout,
                )
            )

            if pooling is not None:
                layers.append(PoolingLayer(hidden_dim, pooling_ratio, pooling))

            blocks.append(nn.Sequential(*layers))

        self.num_layers = num_layers
        self.blocks = blocks
        self.head = MLP(
            in_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_post_layers,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run a forward pass of the full survival GNN.

        Args:
        ----
            batch: Input batch of graphs.

        Returns:
        -------
        Graph-level representations for the input graph batch.
        """
        batch.x = self.node_enc(batch.x)

        h_list = []

        for block in self.blocks:
            batch = block(batch)

            h_max = gnn.global_max_pool(batch.x, batch.batch)
            h_mean = gnn.global_mean_pool(batch.x, batch.batch)

            h_list.append(torch.cat([h_max, h_mean], dim=1))

        h_sum = sum(h_list)

        return self.head(h_sum)

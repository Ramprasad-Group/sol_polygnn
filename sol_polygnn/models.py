from sol_polygnn.std_module import StandardModule
import torch
import torch.nn.functional as F
import random
import numpy as np
import sol_trainer as st

import sol_polygnn.layers as layers

random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

# ##########################
# Multi-task models
# #########################
class polyGNN(StandardModule):
    """
    Multi-task GNN model for polymers.
    """

    def __init__(
        self,
        node_size,
        edge_size,
        selector_dim,
        hps,
        graph_feats_dim,
        node_feats_dim,
        normalize_embedding=True,
        debug=False,
        output_dim=1,
    ):
        super().__init__(hps)

        self.node_size = node_size
        self.edge_size = edge_size
        self.selector_dim = selector_dim
        self.graph_feats_dim = graph_feats_dim
        self.node_feats_dim = node_feats_dim
        self.normalize_embedding = normalize_embedding
        self.debug = debug
        self.output_dim = output_dim

        self.mpnn = layers.MtConcat_PolyMpnn(
            node_size,
            edge_size,
            selector_dim,
            self.hps,
            normalize_embedding,
            debug,
        )

        # set up linear blocks
        self.final_mlp = st.layers.Mlp(
            input_dim=self.mpnn.readout_dim + self.selector_dim + self.graph_feats_dim,
            output_dim=32,
            hps=self.hps,
            debug=False,
        )
        self.out_layer = st.layers.my_output(size_in=32, size_out=self.output_dim)

    def forward(self, data):
        data.yhat = self.mpnn(data.x, data.edge_index, data.edge_weight, data.batch)
        data.yhat = F.leaky_relu(data.yhat)
        data.yhat = torch.cat((data.yhat, data.graph_feats, data.selector), dim=1)
        data.yhat = self.final_mlp(data.yhat)  # hidden layers
        data.yhat = self.out_layer(data.yhat)  # output layer
        return data

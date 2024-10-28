import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import graclus, max_pool, global_mean_pool

from sgcn.src.graph_conv import SpatialGraphConv, SpatialGraphConvEquiv
from sgcn.src.normalized_cut_2d import normalized_cut_2d


class SGCN(torch.nn.Module):
    def __init__(self, dim_coor, out_dim, input_features,
                 layers_num, model_dim, out_channels_1, use_bias=True,
                 use_cluster_pooling=False, dropout=0):
        super(SGCN, self).__init__()
        self.layers_num = layers_num
        self.use_cluster_pooling = use_cluster_pooling

        self.conv_layers = [SpatialGraphConv(coors=dim_coor,
                                             in_channels=input_features,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout,
                                             use_bias=use_bias)] + \
                           [SpatialGraphConv(coors=dim_coor,
                                             in_channels=model_dim,
                                             out_channels=model_dim,
                                             hidden_size=out_channels_1,
                                             dropout=dropout,
                                             use_bias=use_bias) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc1 = torch.nn.Linear(model_dim, out_dim, bias=use_bias)

    def forward(self, data):
        for i in range(self.layers_num):
            data.x = self.conv_layers[i](data.x, data.pos, data.edge_index)

            if self.use_cluster_pooling:
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = global_mean_pool(data.x, data.batch)
        x = self.fc1(data.x)

        return F.log_softmax(x, dim=1)


class QGRN_Equiv_SGCN(torch.nn.Module):
    def __init__(self, dim_coor, out_dim, input_features,
                 layers_num, model_dim, out_channels_1,
                 hidden_sf=2, out_sf=1, hidden_size=7, dropout=0, use_bias=True):
        super(QGRN_Equiv_SGCN, self).__init__()
        self.layers_num = layers_num

        self.conv_layers =  [ SpatialGraphConvEquiv(coors=dim_coor,
                                                    in_channels=input_features,
                                                    out_channels=hidden_sf * model_dim,
                                                    hidden_size=hidden_size,
                                                    dropout=dropout,
                                                    use_bias=use_bias) ] + \
                            [ SpatialGraphConvEquiv(coors=dim_coor,
                                                    in_channels=hidden_sf * model_dim,
                                                    out_channels=hidden_sf * model_dim,
                                                    hidden_size=hidden_size+1,
                                                    dropout=dropout,
                                                    use_bias=use_bias) for _ in range(layers_num - 2) ] + \
                            [ SpatialGraphConvEquiv(coors=dim_coor,
                                                    in_channels=hidden_sf * model_dim,
                                                    out_channels=out_sf * out_channels_1,
                                                    hidden_size=hidden_size,
                                                    dropout=dropout,
                                                    use_bias=use_bias) ]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc1 = torch.nn.Linear(out_sf * out_channels_1, out_dim, bias=use_bias)

    def forward(self, data):
        for i in range(self.layers_num):
            data.x = self.conv_layers[i](data.x, data.pos, data.edge_index)
        data.x = global_mean_pool(data.x, data.batch)
        x = self.fc1(data.x)
        return F.log_softmax(x, dim=1)

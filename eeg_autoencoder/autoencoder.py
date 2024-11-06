from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F

# import QGRL code
from qgrn.qgrl import QGRL
from sgcn.src.graph_conv import SpatialGraphConvEquiv

class Autoencoder(nn.Module):
    """
    Autoencoder class for dimensionality reduction using an autoencoder model.
    """

    def __init__(self,
                 num_nodes,
                 num_factors, 
                 in_channels,
                 hidden_channels, 
                 latent_channels, 
                 layers_num,
                 conv_type="QGRL",
                 num_sub_kernels=1,
                 device="cpu",
                 **kwargs):
        super(Autoencoder, self).__init__()
        self.device = device
        self.layers_num = layers_num
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.num_sub_kernels = num_sub_kernels

        # Define Conv Layers for QGCN
        self.conv_layers_e, self.conv_layers_d = self._get_conv_layers(
                                                    conv_type, **kwargs)

        # create a module list
        self.conv_layers_e = nn.ModuleList(self.conv_layers_e)
        fc_channels = self.latent_channels*self.num_nodes
        self.fc_e = nn.Linear(fc_channels, num_factors).to(self.device)
        self.fc_d = nn.Linear(num_factors, fc_channels).to(self.device)
        self.conv_layers_d = nn.ModuleList(self.conv_layers_d)


    def _get_conv_layers(self,
                         conv_type,
                         num_sub_kernels=1,
                         use_shared_bias=True,
                         edge_attr_dim = -1, 
                         pos_descr_dim = -1,
                         quant_net_depth = 2,
                         quant_net_expansion = 2,
                         apply_mixture_net = True, 
                         mixture_net_depth = 2,
                         mixture_net_expansion = 2,
                         hidden_size = 7,
                         aggr="add", 
                         device="cpu"):
        
        if conv_type == "QGRL":
            e_layers = [ QGRL(in_channels = self.in_channels,
                              out_channels = self.hidden_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device) ] + \
                       [ QGRL(in_channels = self.hidden_channels, 
                              out_channels = self.hidden_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device)
                        for _ in range(self.layers_num - 2) ] + \
                       [ QGRL(in_channels = self.hidden_channels, 
                              out_channels = self.latent_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device) ]
            
            d_layers = [ QGRL(in_channels = self.latent_channels, 
                              out_channels = self.hidden_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device) ] + \
                       [ QGRL(in_channels = self.hidden_channels, 
                              out_channels = self.hidden_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device) 
                        for _ in range(self.layers_num - 2) ] + \
                       [ QGRL(in_channels = self.hidden_channels, 
                              out_channels = self.in_channels,
                              num_sub_kernels= num_sub_kernels,
                              use_shared_bias = use_shared_bias,
                              edge_attr_dim = edge_attr_dim, 
                              pos_descr_dim = pos_descr_dim,
                              quant_net_depth = quant_net_depth,
                              quant_net_expansion = quant_net_expansion,
                              apply_mixture_net = apply_mixture_net, 
                              mixture_net_depth = mixture_net_depth,
                              mixture_net_expansion = mixture_net_expansion,
                              aggr = aggr,
                              device = device) ]
        elif conv_type == "SGCN":
            e_layers =  [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.in_channels,
                                            out_channels=self.hidden_channels,
                                            hidden_size=hidden_size) ] + \
                        [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.hidden_channels,
                                            out_channels=self.hidden_channels,
                                            hidden_size=hidden_size+1)
                                    for _ in range(self.layers_num - 2) ] + \
                        [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.hidden_channels,
                                            out_channels=self.latent_channels,
                                            hidden_size=hidden_size) ]
            d_layers =  [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.latent_channels,
                                            out_channels=self.hidden_channels,
                                            hidden_size=hidden_size) ] + \
                        [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.hidden_channels,
                                            out_channels=self.hidden_channels,
                                            hidden_size=hidden_size+1)
                                    for _ in range(self.layers_num - 2) ] + \
                        [ SpatialGraphConvEquiv(coors=pos_descr_dim,
                                            in_channels=self.hidden_channels,
                                            out_channels=self.in_channels,
                                            hidden_size=hidden_size) ]
        else:
            raise ValueError(f"Invalid convolution type: {conv_type}")
            
        return e_layers, d_layers


    def forward(self, data):
        z = self.encoder(data)
        data_reconstructed = self.decoder(z)
        return data_reconstructed


    def encoder(self, data):
        # check pos and edge_index
        if not hasattr(self, "pos"):
            self.pos = data.pos
        else:
            assert self.pos.eq(data.pos).all()
        if not hasattr(self, "edge_index"):
            self.edge_index = data.edge_index
        else:
            assert self.edge_index.eq(data.edge_index).all()

        x = data.x
        for i in range(self.layers_num):
            layer_inputs = {"x": x, "pos": self.pos, "edge_index": self.edge_index}
            # if hasattr(data, "edge_attr"): layer_inputs.update({ "edge_attr": data.edge_attr })
            x = self.conv_layers_e[i](**layer_inputs)
        
        # flatten nodes/channels dims, retaining batch dim, then apply linear layer
        x = x.view(-1, self.latent_channels*self.num_nodes)
        return self.fc_e(x)
    

    def decoder(self, x, beta=10):
        h = self.fc_d(x)
        h = h.view(-1, self.latent_channels)  # Reshape to combine batch/node dimensions
        for i in range(self.layers_num):
            layer_inputs = { "x": h, "pos": self.pos, "edge_index": self.edge_index }
            h = self.conv_layers_d[i](**layer_inputs)
        h = F.softplus(h, beta=beta)
        # data = Data(x=h, pos=self.pos, edge_index=self.edge_index)
        return h#.view(-1, self.num_nodes, self.latent_channels) # separate batches/nodes
    

class SAE(Autoencoder):

    def __init__(self,
                 num_nodes,
                 in_channels,
                 num_factors, 
                 hidden_channels, 
                 latent_channels, 
                 layers_num,
                 class_hidden_units,
                 conv_type="QGRL",
                 num_sub_kernels=1,
                 use_shared_bias=True,
                 edge_attr_dim = -1, 
                 pos_descr_dim = -1,
                 quant_net_depth = 2,
                 quant_net_expansion = 2,
                 apply_mixture_net = True, 
                 mixture_net_depth = 2,
                 mixture_net_expansion = 2,
                 aggr="add", 
                 device="cpu",
                 **kwargs):
        super().__init__(conv_type=conv_type,
                         num_nodes=num_nodes, 
                         in_channels=in_channels, 
                         num_factors=num_factors,
                         hidden_channels=hidden_channels,
                         latent_channels=latent_channels,
                         layers_num=layers_num,
                         num_sub_kernels=num_sub_kernels,
                         use_shared_bias=use_shared_bias,
                         edge_attr_dim=edge_attr_dim,
                         pos_descr_dim=pos_descr_dim,
                         quant_net_depth=quant_net_depth,
                         quant_net_expansion=quant_net_expansion,
                         apply_mixture_net=apply_mixture_net,
                         mixture_net_depth=mixture_net_depth,
                         mixture_net_expansion=mixture_net_expansion,
                         aggr=aggr,
                         device=device,
                         **kwargs)

        self.fc_c1 = nn.Linear(num_factors, class_hidden_units).to(self.device)
        self.fc_c2 = nn.Linear(class_hidden_units, 1).to(self.device)
        self.dropout = nn.Dropout(0.2)
        # self.fc_c2 = nn.Linear(num_factors, 1).to(self.device)

    def forward(self, data):
        z = self.encoder(data)
        data_reconstructed = self.decoder(z)
        class_est = self.classifier(z)
        return data_reconstructed, class_est
    
    def classifier(self, z):
        h = F.relu(self.fc_c1(z))
        h = self.dropout(h)
        # h = z
        return self.fc_c2(h)


def kl_div_loss(x_est, x):
    """
    Compute the KL divergence loss between the estimated data and the original data.
    """
    kl_loss = nn.KLDivLoss(reduction="mean")
    # loss = kl_loss(data_est.x, data.x)
    loss = kl_loss(x_est.log(), x) - x.mean() + x_est.mean()
    return loss
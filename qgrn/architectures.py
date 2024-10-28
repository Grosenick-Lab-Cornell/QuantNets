import torch
from torch_geometric.nn import global_mean_pool
from qgrn.qgrl import QGRL


class QGRN(torch.nn.Module):
    def __init__(self,
                 out_dim, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 layers_num,
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
                 device="cpu"):
        super(QGRN, self).__init__()
        self.device = device
        self.layers_num = layers_num
        self.num_sub_kernels = num_sub_kernels

        # Define Conv Layers for QGCN
        self.conv_layers =  [ QGRL(
                                in_channels = in_channels, 
                                out_channels = hidden_channels,
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
                            [ QGRL(
                                in_channels = hidden_channels, 
                                out_channels = hidden_channels,
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
                                for _ in range(layers_num - 2) ] + \
                            [ QGRL(
                                in_channels = hidden_channels, 
                                out_channels = out_channels,
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

        # create a module list ...
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc1 = torch.nn.Linear(out_channels, out_dim).to(self.device)

    def forward(self, data):
      for i in range(self.layers_num):
        layer_inputs = { "x": data.x, "pos": data.pos, "edge_index": data.edge_index }
        if hasattr(data, "edge_attr"): layer_inputs.update({ "edge_attr": data.edge_attr })
        data.x = self.conv_layers[i](**layer_inputs)
      data.x = global_mean_pool(data.x, data.batch)
      x = self.fc1(data.x)
      return x

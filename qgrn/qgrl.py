import torch
import numpy as np
from torch import nn
from typing import Optional
from torch import Tensor
from torch_geometric.nn import MessagePassing


"""
QuantNet Activation:
    Wraps around the softmax and gumbel-softmax activations for QGRNs
"""
class QuantizedActivation(torch.nn.Module):
    __supported_activations = ["softmax", "gumbel_softmax"]

    def __init__(self, activation: str = "gumbel_softmax", tau: int = 0.001, hard_sampling: bool = True, hard_sampling_tau_sf: float = 0.1) -> Tensor:
        super(QuantizedActivation, self).__init__()
        self.hard_sampling = hard_sampling
        self.activation = activation.strip().lower()
        self.tau = (tau * hard_sampling_tau_sf) if (self.hard_sampling and (self.activation == "softmax")) else tau

        # Error handling for class
        assert self.activation in QuantizedActivation.__supported_activations, f"Target Activation - {activation} not in supported activations {QuantizedActivation.__supported_activations}"

    # Pickling requires actual function names 
    def custom_activation(self, x: Tensor):
        if self.activation == "softmax":
            return torch.nn.functional.softmax(x.squeeze(dim=-1) / self.tau, dim=-1).unsqueeze(dim=-1)
        else: # NOTE: gumbel softmax is default; all else being equal
            return torch.nn.functional.gumbel_softmax(x.squeeze(dim=-1), tau=self.tau, hard=self.hard_sampling, dim=-1).unsqueeze(dim=-1)

    def forward(self, x: Tensor):
        """
        x: [num_nodes, node_features] 
           This is the output (logits) 
           which we feed through this custom 
           activation layer
        """
        # Return output from custom activation
        return self.custom_activation(x)


"""
A pass through class that allows us to skip certain model complexities
"""
class PassThrough(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


"""
MixtureNet - Simple MLP for combining message-passed features
"""
class SimpleResConn(torch.nn.Module):
    def __init__(self, channels: int, apply_inner_resd_lyr: bool = True, enable_activations = True, normalize = True):
        super().__init__()
        self.channels              = channels
        self.apply_inner_resd_lyr  = apply_inner_resd_lyr
        self.enable_activations    = enable_activations
        self.normalize             = normalize
        self.pass_through          = PassThrough()

        if self.apply_inner_resd_lyr:
            # Define a conv1D layer to extract features in same hyper dimension as input for QGRL convolution layer
            self.glob_feat_extract_net = torch.nn.Sequential(
                                                            torch.nn.Conv1d(self.channels, self.channels, kernel_size=1, stride=1),
                                                            torch.nn.BatchNorm1d(self.channels) if self.normalize else self.pass_through,
                                                            torch.nn.ReLU() if self.enable_activations else self.pass_through
                                                        )
        else:
            self.glob_feat_extract_net = lambda x: 0 * x

    def forward(self, in_a: Tensor) -> Tensor:
        return in_a + self.glob_feat_extract_net(in_a.unsqueeze(dim=-1)).squeeze(dim=-1)


"""
MixtureNet - Simple MLP for combining message-passed features
"""
class MixtureNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mixture_net_depth: int, mixture_net_expansion: int, apply_mixture_net: bool = True, enable_activations = True, normalize = True):
        super().__init__()
        self.in_channels           = in_channels
        self.out_channels          = out_channels
        self.apply_mixture_net     = apply_mixture_net
        self.mixture_net_depth     = mixture_net_depth
        self.mixture_net_expansion = mixture_net_expansion
        self.enable_activations    = enable_activations
        self.normalize             = normalize
        self.pass_through          = PassThrough()

        # An N-layer network to mix cluster-agnostic messages and cluster-specific messages
        if self.apply_mixture_net:
            mixture_net_channels = [ self.in_channels ] + [ (i * self.out_channels * self.mixture_net_expansion) for i in range(1, self.mixture_net_depth) ] + [ self.out_channels ] 
            mixture_net_layers = []
            for ch_idx in range(len(mixture_net_channels) - 1):
                in_channels  = mixture_net_channels[ch_idx]
                out_channels = mixture_net_channels[ch_idx + 1]
                mixture_net_layers.extend([
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                    torch.nn.BatchNorm1d(out_channels) if self.normalize else self.pass_through,
                    torch.nn.ReLU() if self.enable_activations else self.pass_through
                ])
            self.aggr_net = torch.nn.Sequential(*mixture_net_layers)
        else:
            self.aggr_net = lambda x: x

    def forward(self, in_a: Tensor, in_b: Tensor) -> Tensor:
        return self.aggr_net((in_a + in_b).unsqueeze(dim=-1)).squeeze(dim=-1)

    
"""
QGRL Layer for QGRN
"""
class QGRL(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_sub_kernels = 3,
                 use_shared_bias = True,
                 edge_attr_dim = -1, 
                 pos_descr_dim = -1,
                 quant_net_depth = 2,
                 quant_net_expansion = 2,
                 quant_net_in_features = -1,
                 apply_mixture_net = True, 
                 mixture_net_depth = 2,
                 mixture_net_expansion = 2,
                 apply_inner_resd_lyr = True,
                 enable_activations = True,
                 normalize = True,
                 aggr="add", 
                 device="cpu"):
        """
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        edge_attr_dim - dimension of edge attribute features
        normalize      - whether or not to enable intermediate normalizations
        enable_activations  - whether or not to enable non-linear activation functions
        num_sub_kernels: This is the number of subkernels we pre-define for the convolution
                     These num kernels are simply centroids within neighborhoods mapped
                     out onto a 2D cartesian plane. Nodes are mapped to these fixed 
                     centroids (each with their own kernels) and are convolved.
        use_shared_bias: (bool) whether to constrain all subkernels to use a single bias term
                         as is in the CNN convolution kernel definition
        pos_descr_dim: dimension of the positional descriptors to be used
        quant_net_depth: depth of the MLP that learns the neighborhood-kernel quantization
        quant_net_expansion: defines expansion factor for the high dimensional space in which
                             to learn relative feature distances
        quant_net_in_features: defines the input features, i.e., compressed positional descriptors dimension to use
                               resolution is if -1 use either pos_descr_dim or in_channels as is, else compress / expand to target dimension
        apply_mixture_net: (bool) whether or not to combine convolution-like messages post-aggregation
                           with old messages from all nodes in the distinct manner indicated in the paper 
        mixture_net_depth: the depth of the network combining features from different learning paths
        mixture_net_expansion: the expansion (& simultaneously shrinkage factor) for the mixture net
        apply_inner_resd_lyr: determines whether inner residual layer should be applied
        aggr: how to aggregate node messages
        device: where model lives/runs e.g., cpu or cuda* (gpu)
        """
        super(QGRL, self).__init__(aggr=aggr)

        # Assert that model initialization is correct
        # Assert that num_sub_kernels override is expected for this model
        assert not (num_sub_kernels <= 0), "num_sub_kernels, if specified, must be >= 1"
        if pos_descr_dim != -1: assert not (pos_descr_dim <= 0), "positional descriptors must be >= 1"

        # save reference to channels params ...
        self.aggr = aggr
        self.device = device
        self.normalize = normalize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.pos_descr_dim = pos_descr_dim
        self.use_shared_bias = use_shared_bias
        self.num_sub_kernels = num_sub_kernels
        self.pos_descr_exists = pos_descr_dim != -1
        self.edge_attr_exists = edge_attr_dim != -1
        self.enable_activations = enable_activations
        self.apply_inner_resd_lyr = apply_inner_resd_lyr
        self.pass_through       = PassThrough()

        # Mixture MLP Depth
        self.apply_mixture_net = apply_mixture_net
        self.mixture_net_depth = mixture_net_depth
        self.mixture_net_expansion = mixture_net_expansion

        # Quantizer Net Params
        self.quant_net_depth = quant_net_depth
        self.quant_net_expansion = quant_net_expansion
        self.quant_net_in_features = quant_net_in_features

        # Define norm. and activation for update() stage of message passing
        self.conv1d = torch.nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, stride=1)
        self.relu   = torch.nn.ReLU() if self.enable_activations else self.pass_through
        self.bn1d   = torch.nn.BatchNorm1d(self.out_channels).to(self.device) if self.normalize else self.pass_through

        # Define subkernels for convolution-like operation in node neighborhoods
        self.nodes_feats_group_conv1d = self.init_sub_kernels_via_group_conv1d(self.in_channels, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
        if self.use_shared_bias: self.nodes_shared_bias = torch.nn.Parameter(torch.rand(1, self.out_channels), requires_grad=True).to(self.device)

        # Define subkernels for convolution-like operation on edge_attr if they exist
        if self.edge_attr_exists:
            assert isinstance(edge_attr_dim, int), "Num. Edge Attributes/Features (edge_attr_dim) is expected to be an int"
            self.edge_feats_group_conv1d = self.init_sub_kernels_via_group_conv1d(self.edge_attr_dim, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
            if self.use_shared_bias: self.edges_shared_bias = torch.nn.Parameter(torch.rand(1, self.out_channels), requires_grad=True).to(self.device)
            # Define the norm. and activation for output of sub_kernels for edges
            self.edge_attr_relu    = torch.nn.ReLU() if self.enable_activations else self.pass_through
            self.edge_attr_bn1d    = torch.nn.BatchNorm1d(self.out_channels).to(self.device) if self.normalize else self.pass_through

        # Learning Quantization via MLPs
        quant_net_in_channels = [ self.pos_descr_dim if self.pos_descr_exists else self.in_channels ] # Default: Learn from Node Features
        if quant_net_in_features != -1: quant_net_in_channels.append(quant_net_in_features)
        quant_net_out_channels = self.num_sub_kernels # Outputs distribution over num subkernels (from which we select best kernel mapping)
        quant_net_left_channels = quant_net_in_channels + [ (i * quant_net_in_channels[-1] * self.quant_net_expansion) for i in range(1, self.quant_net_depth) ]
        quant_net_left_layers = []
        for ch_idx in range(len(quant_net_left_channels) - 1):
            in_channels  = quant_net_left_channels[ch_idx]
            out_channels = quant_net_left_channels[ch_idx + 1]
            quant_net_left_layers.extend([
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                torch.nn.BatchNorm1d(out_channels) if self.normalize else self.pass_through,
                torch.nn.ReLU() if self.enable_activations else self.pass_through
            ])
        self.quant_net_left = torch.nn.Sequential(*quant_net_left_layers).to(self.device)

        # Define the layer that learns the quantization
        quant_net_right_channels = [ (i * quant_net_in_channels[-1] * self.quant_net_expansion) for i in range(self.quant_net_depth - 1, 0, -1) ] + [ quant_net_out_channels ]
        quant_net_right_layers = []
        for ch_idx in range(len(quant_net_right_channels) - 1):
            in_channels  = quant_net_right_channels[ch_idx]
            out_channels = quant_net_right_channels[ch_idx + 1]
            if ch_idx == len(quant_net_right_channels) - 2:
                # Last Layer is a mix of custom layers
                quant_net_right_layers.extend([
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                    QuantizedActivation(activation="gumbel_softmax", tau=0.01, hard_sampling=True)
                ])
                continue
            # Intermediate/Hidden Layers
            quant_net_right_layers.extend([
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                torch.nn.BatchNorm1d(out_channels) if self.normalize else self.pass_through,
                torch.nn.ReLU() if self.enable_activations else self.pass_through
            ])
        self.quant_net_right = torch.nn.Sequential(*quant_net_right_layers).to(self.device)

        # Layers for learning from prior timne step's message-passed and resolved features
        # Define a conv1D layer that learns features across all nodes in all neighborhoods the same
        self.glob_feat_net = torch.nn.Sequential(
                                                    torch.nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1),
                                                    torch.nn.BatchNorm1d(self.out_channels) if self.normalize else self.pass_through,
                                                    torch.nn.ReLU() if self.enable_activations else self.pass_through
                                                ).to(self.device)

        # Inner Residual Layer for processing features
        self.inner_residual_net = SimpleResConn(channels=self.out_channels, 
                                                apply_inner_resd_lyr=self.apply_inner_resd_lyr, 
                                                enable_activations=self.enable_activations, 
                                                normalize=self.normalize).to(self.device)

        # Mixture Net: combines the current stage of message-passed and aggregated results together with previous node messages (from immediate past message-passing stage)
        self.mixture_net = MixtureNet(in_channels=self.out_channels, 
                                      out_channels=self.out_channels, 
                                      mixture_net_depth=self.mixture_net_depth, 
                                      mixture_net_expansion=self.mixture_net_expansion, 
                                      apply_mixture_net=self.apply_mixture_net, 
                                      enable_activations=self.enable_activations, 
                                      normalize=self.normalize).to(self.device)

    def forward(self, x, edge_index, pos:Optional[Tensor]=None, edge_attr:Optional[Tensor]=None):
        # Information from neighborhood -> afforded us by the QGRL subkernels that learn only on a per neighborhood basis
        out_feats = self.propagate(edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr, aggr=self.aggr)  # [N, out_channels, label_dim] -- # Propagate the call through next stages of message passing framework

        # Information from entire graph -> afforded us by the conv1D/Lin layer that learns on all the data (conv1D/LinLayer -> BN -> Relu)
        glob_out_feats = self.inner_residual_net(self.glob_feat_net(x.unsqueeze(dim=-1)).squeeze(dim=-1))

        # Apply Mixture Net
        out_feats = self.mixture_net(out_feats, glob_out_feats)

        # return out_feats
        return out_feats


    def message(self, x_i, x_j, pos_i: Optional[Tensor], pos_j: Optional[Tensor], edge_attr: Optional[Tensor]):
        """
        x_i [num_edges, node_features]
        x_j [num_edges, node_features]
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        edge_attr [num_edges, edge_attr_dim]
        """
        # Learn the quantized neighborhoods from node attributes
        quant_source_feats, quant_target_feats = (pos_j, pos_i) if self.pos_descr_exists else (x_j, x_i)
        receptive_field_masks = self.quant_net_right( self.quant_net_left(quant_source_feats.unsqueeze(dim=-1)) - self.quant_net_left(quant_target_feats.unsqueeze(dim=-1)) )

        # Apply group conv for node features
        out_node_features_transformed = self.apply_group_conv1d(x=x_j,
                                                                group_conv1d=self.nodes_feats_group_conv1d, 
                                                                receptive_field_masks=receptive_field_masks, 
                                                                use_shared_bias=self.use_shared_bias, 
                                                                shared_bias=self.nodes_shared_bias)

        # Conditionally handle edge attributes
        if self.edge_attr_exists:
            # Extract edge feature vectors
            out_edge_features_transformed = self.apply_group_conv1d(x=edge_attr,
                                                                    group_conv1d=self.edge_feats_group_conv1d, 
                                                                    receptive_field_masks=receptive_field_masks, 
                                                                    use_shared_bias=self.use_shared_bias, 
                                                                    shared_bias=self.edges_shared_bias)
            # Compute scalars and apply them to node features
            out_edge_scalars = self.edge_attr_relu(self.edge_attr_bn1d(out_edge_features_transformed))
            out_node_features_transformed = out_node_features_transformed * out_edge_scalars

        # Return output [dim -> (num_edges, out_features) ]
        return out_node_features_transformed # size => (num_edges, out_features)
    
    # @torch.jit.script
    # def apply_group_conv1d(self, group_conv1d, x, receptive_field_masks, use_shared_bias, shared_bias) -> Tensor:
    def apply_group_conv1d(self, group_conv1d: torch.nn.Conv1d, x: Tensor, receptive_field_masks: Tensor, use_shared_bias: bool, shared_bias: torch.nn.Parameter) -> Tensor:
        """
        x [num_edges, in_features]
        """
        # Reshape into broadcast-friendly shape
        N, S, E  = receptive_field_masks.shape
        receptive_field_masks_reshaped = receptive_field_masks.reshape((N, E, S)) # [num_edges, num_subkernels, 1] -> [num_edges, 1, num_subkernels]

        # Create the group conv input for both nodes and edges group convolutions
        x_unsq = x.unsqueeze(dim=-1) # [num_edges, in_features, 1]
        x_group_conv_input = x_unsq * receptive_field_masks_reshaped # [num_edges, in_features, 1] * [num_edges, 1, num_subkernels] => [num_edges, in_features, num_subkernels]
        x_group_conv_input_collapsed = x_group_conv_input.reshape((x_group_conv_input.shape[0], -1, 1)) # [num_edges, in_features * num_subkernels, 1]

        # Feed-forward through group convs
        out_group_conv1d = group_conv1d(x_group_conv_input_collapsed) # [num_edges, out_features * num_subkernels, 1]

        # Apply subkernel filter masks again to extract out subkernel specific outputs
        N, G, _ = out_group_conv1d.shape
        out_group_conv1d_expnd = out_group_conv1d.reshape((N, G // self.num_sub_kernels, self.num_sub_kernels)) # [num_edges, out_features, num_subkernels]
        out_filtered_group_conv1d_expnd = out_group_conv1d_expnd * receptive_field_masks_reshaped # [num_edges, out_features, num_subkernels] * [num_edges, 1, num_subkernels] => [num_edges, out_features, num_subkernels]
        out_features_transformed = out_filtered_group_conv1d_expnd.sum(dim=-1) + use_shared_bias * shared_bias # [num_edges, out_features, num_subkernels] -> [num_edges, out_features]

        # Return output [dim -> (num_edges, out_features) ]
        return out_features_transformed # [num_edges, out_features]


    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        return self.relu(self.bn1d(self.conv1d(aggr_out.unsqueeze(dim=-1)).squeeze(dim=-1)))


    def init_sub_kernels_via_group_conv1d(self, in_channels: int, out_channels: int,  num_subkernels: int, use_bias: bool = True):
        """
        Assists with initializing new MLPs/weights in our adaptable kernel for the convolution
        """
        assert num_subkernels >= 1, "num subkernels in the convolution must be > 0"

        # Define group conv.: Optimizes on the for-loop through individual linear/conv1d subkernel convolutions
        return torch.nn.Conv1d(in_channels=(in_channels * num_subkernels), out_channels=(out_channels * num_subkernels), groups=num_subkernels, kernel_size=1, stride=1, bias=use_bias).to(self.device)

import torch
from torch_geometric.nn import global_mean_pool
from qgcn.qncl import QNCL


class QGCN(torch.nn.Module):
    def __init__(self, dim_coor, out_dim, in_channels,
                hidden_channels, out_channels, layers_num, dropout=0, num_kernels=1, 
                use_bias=True, use_batchNorm=True, use_learnable_batchNorm=True, use_maxPool=False,
                is_dataset_homogenous=True, apply_spatial_scalars=False, upper_bound_kernel_len=30, 
                self_loops_included=False, initializer_model=None, device="cpu"):
        super(QGCN, self).__init__()
        self.device = device
        self.layers_num = layers_num
        self.num_kernels = num_kernels
        self.self_loops_included = self_loops_included
        self.use_learnable_batchNorm = use_learnable_batchNorm

        ################################################################################################################
        # Initialize the model init params
        self.initializer_named_params_dict = None
        self.initializer_model_exists = not isinstance(initializer_model, type(None))
        if self.initializer_model_exists:
            params_dict = { param_name: param_value for param_name, param_value in initializer_model.named_parameters()}
            self.initializer_named_params_dict = params_dict
        #################################################################################################################

        # Define Conv Layers for QGCN
        self.conv_layers =  [ QNCL(
                                    coors=dim_coor, 
                                    in_channels=in_channels,
                                    out_channels=hidden_channels,
                                    use_bias=use_bias,
                                    use_maxPool=use_maxPool,
                                    use_batchNorm=use_batchNorm,
                                    use_learnable_batchNorm=use_learnable_batchNorm,
                                    dropout=dropout,
                                    num_kernels=num_kernels,
                                    aggr="add",
                                    init="kaiming",
                                    is_dataset_homogenous=is_dataset_homogenous,
                                    apply_spatial_scalars=apply_spatial_scalars, 
                                    upper_bound_kernel_len=upper_bound_kernel_len, 
                                    self_loops_included=self_loops_included,
                                    device=self.device) ] + \
                            [ QNCL(
                                    coors=dim_coor,
                                    in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    use_bias=use_bias,
                                    use_maxPool=use_maxPool,
                                    use_batchNorm=use_batchNorm,
                                    use_learnable_batchNorm=use_learnable_batchNorm,
                                    dropout=dropout,
                                    num_kernels=num_kernels,
                                    aggr="add",
                                    init="kaiming",
                                    is_dataset_homogenous=is_dataset_homogenous,
                                    apply_spatial_scalars=apply_spatial_scalars, 
                                    upper_bound_kernel_len=upper_bound_kernel_len,
                                    self_loops_included=self_loops_included,
                                    device=self.device) for _ in range(layers_num - 2) ] + \
                            [ QNCL(
                                    coors=dim_coor,
                                    in_channels=hidden_channels,
                                    out_channels=out_channels,
                                    use_bias=use_bias,
                                    use_maxPool=use_maxPool,
                                    use_batchNorm=use_batchNorm,
                                    use_learnable_batchNorm=use_learnable_batchNorm,
                                    dropout=dropout,
                                    num_kernels=num_kernels,
                                    aggr="add",
                                    init="kaiming",
                                    is_dataset_homogenous=is_dataset_homogenous,
                                    apply_spatial_scalars=apply_spatial_scalars, 
                                    upper_bound_kernel_len=upper_bound_kernel_len,
                                    self_loops_included=self_loops_included,
                                    device=self.device) ]

        # create a module list ...
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc1 = torch.nn.Linear(out_channels, out_dim, bias=use_bias).to(self.device)
        # loop through all the conv layers and other layer components like MLPs etc.
        # and inject the layer index and initializers to match QGCN with CNN
        # NOTE: We do the below to enforce that both CNN and QGCN are initialized exactly the same
        ########################################################################################################################
        if self.initializer_model_exists:
            for index, layer in enumerate(self.conv_layers):
                layer.set_initializer_params(layer_index=index, initializer_named_params_dict=self.initializer_named_params_dict)
            # initialize weights and biases based on conv2d model initializer
            self.initialize_fc_weights_and_biases() 
        #########################################################################################################################


    def forward(self, data):
      for i in range(self.layers_num):
          data.x = self.conv_layers[i](data.x, data.pos, data.edge_index)
      data.x = global_mean_pool(data.x, data.batch)
      x = self.fc1(data.x)
      return x
    
            
    """
    Helper function to initialize fc weights & biases with values from initializer CNN model
    """  
    def initialize_fc_weights_and_biases(self):
        if isinstance(self.initializer_named_params_dict, type(None)):
            return
        
        # get the weight keys
        fc_weight_key = f'fc.weight'
        fc_weight_key_exists = fc_weight_key in self.initializer_named_params_dict
        fc_bias_key = f'fc.bias'
        fc_bias_key_exists = fc_bias_key in self.initializer_named_params_dict
        
        # init fully connected layer weights etc.
        if fc_weight_key_exists:
            fc_weight_param = self.initializer_named_params_dict[fc_weight_key]
            self.fc1.weight.data = fc_weight_param.clone().detach().data
        if fc_bias_key_exists:
            fc_bias_param = self.initializer_named_params_dict[fc_bias_key]
            self.fc1.bias.data   = fc_bias_param.clone().detach().data

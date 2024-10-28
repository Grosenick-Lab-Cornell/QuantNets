import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math


class QNCL(MessagePassing):
    def __init__(self, coors, in_channels, out_channels, use_bias=True, 
                 use_batchNorm=False, use_learnable_batchNorm=True, 
                 use_maxPool=False, dropout=0.3, num_kernels=-1, 
                 is_dataset_homogenous=True, self_loops_included=False,
                 apply_spatial_scalars=False, upper_bound_kernel_len=9,
                 aggr="add", init="kaiming", device="cpu"):
        """
        coors - dimension of positional descriptors (e.g. 2 for 2D images)
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        dropout - dropout rate after the layer
        internal_max_kernel_len: This is the maximum number of kernels we can pre-define
                                 ahead of refinement as we explore the edge_indices 
                                 to extract key graph data properties.
                                 We do this because PyTorch isn't happy with dynamically defining
                                 kernels on the fly inside forward() so instead we define more than
                                 enough and refine them using info from the forward() function
        """
        super(QNCL, self).__init__(aggr=aggr)

        # Assert that model initialization is correct
        # Assert that num_kernels override is expected for this model
        assert not (num_kernels <= 0),            "num_kernels, if specified, must be >= 1"
        assert not (upper_bound_kernel_len <= 0), "upper_bound_kernel_len, if specified, must be >= 1"
        err_msg = f"num_kernels must be <= upper_bound_kernel_len: {upper_bound_kernel_len} \n"
        err_msg += "Please note: you can modify upper_bound_kernel_len, if you anticipate your local neighborhood to be complex"
        assert (num_kernels <= upper_bound_kernel_len), err_msg

        # save reference to channels params ...
        self.device = device
        self.coors = coors
        self.aggr = aggr
        self.init = init
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_maxPool = use_maxPool
        self.use_batchNorm = use_batchNorm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.user_defined_kernel_len = num_kernels
        self.is_dataset_homogenous = is_dataset_homogenous
        self.apply_spatial_scalars = apply_spatial_scalars
        self.use_learnable_batchNorm = use_learnable_batchNorm
        self.self_loops_included = self_loops_included

        # Define the initializer variables: Assists with initializing QGCN kernels == CNN kernels
        #######################################################################################
        self.layer_index = -1
        self.initializer_named_params_dict = None
        #######################################################################################

        # To make iso comparison between QGCN and SGCN
        if apply_spatial_scalars:
            hidden_dim = math.ceil(self.out_channels / 2)
            self.lin_in_head = torch.nn.Linear(self.coors, hidden_dim, bias=self.use_bias).to(self.device)
            self.lin_in_bn1d = torch.nn.BatchNorm1d(hidden_dim, affine=use_learnable_batchNorm).to(self.device)
            self.lin_in_tail = torch.nn.Linear(hidden_dim, self.out_channels, bias=self.use_bias).to(self.device)
            self.lin_out     = torch.nn.Linear(self.out_channels, self.out_channels, bias=self.use_bias).to(self.device)
 
        # Define bn layer for aggregation stage ...
        self.relu = torch.nn.ReLU()
        self.bn1d = torch.nn.BatchNorm1d(self.out_channels, affine=use_learnable_batchNorm).to(self.device)
        
        # Define kernel related parameters
        self.kernels = torch.nn.ModuleList([])
        self.kernel_weight_mask_map = {}
        self.max_kernel_len = -1
        self.upper_bound_kernel_len = upper_bound_kernel_len
        self.full_backward_hooks_handlers = []
        
        # Initialize enough kernels so that in refinement stage, 
        # we only either remove or ignore them
        for kernel_idx in range(upper_bound_kernel_len):
            self.init_kernel_weight_layer(kernel_idx=kernel_idx)

            
    def forward(self, x, pos, edge_index):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        pos - node position matrix [num_nodes, coors]
        edge_index - graph connectivity [2, num_edges]
        """
        # Generate the new kernels or masks for existing kernels
        self.handle_kernel_weights_and_masks_update(edge_index, pos)
        # Propagate the call through next stages of message passing framework
        res =  self.propagate(edge_index=edge_index, x=x, pos=pos, aggr=self.aggr)  # [N, out_channels, label_dim]
        return res
        

    """
    NOTE: We don't use the relative position magnitudes/displacement from origin for anything
    For comparison with CNN, this is fine but for the general model proposed, we'd also benefit
    by learning a representation from relative position (num_edges x num_coordinates) to (num_edges x out_channels)
    similar to what SGCN does with relative positions. This ensures that in comparing with SGCN, we are almost iso
    w.r.t to how SGCN uses their relative spatial information (project relative pos - which has mag and dir/angle information)
    into a 1-D dimension to scale the node features. In this regard, SGCN is implicitly using both 'mag. and dir'
    so we adopt a similar approach for QGCN for the generalized graph data for iso-comparison
    """
    def message(self, pos_i, pos_j, x_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """

        # Compute transformed node messages
        num_edges, node_features_len = x_j.shape
        x_j_transformed = torch.zeros((num_edges, self.out_channels)).to(self.device)
        for i in range(self.max_kernel_len):
            # Get the mask of the specific kernel
            kernel_mask = self.kernel_weight_mask_map[i][:len(x_j)]
            # Get the corresponding kernel too
            kernel_lin_layer = self.kernels[i]
            # apply masking and message preparation
            x_j_filtered = x_j[kernel_mask].unsqueeze(dim=-1)
            x_j_filtered_transformed = kernel_lin_layer(x_j_filtered)
            x_j_transformed[kernel_mask] = x_j_filtered_transformed.squeeze()

        # Generate and apply spatial scalars if user allowed
        if self.apply_spatial_scalars:
            relative_pos = pos_j - pos_i
            spatial_scalars = relative_pos
            for _, layer in enumerate([self.lin_in_head, self.relu, self.lin_in_bn1d, self.lin_in_tail, self.relu]):
                spatial_scalars = layer(spatial_scalars)
            # scale the transformed features by the spatial properties
            assert x_j_transformed.shape == spatial_scalars.shape, "spatial_scalars must match dimension of x_j_transformed"
            x_j_transformed = torch.mul(x_j_transformed, spatial_scalars) # Below: [n_edges, out_channels] * [n_edges, out_channels]
        
        # return the transformed node features
        return x_j_transformed # size => (num_edges, out_channels) => (num_edges, node_features)
        

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        if self.apply_spatial_scalars:
            aggr_out = self.relu(self.lin_out(aggr_out))
        if self.use_batchNorm:
            aggr_out = self.bn1d(aggr_out)
        aggr_out = self.relu(aggr_out)
        return aggr_out
    

    """
    Backward hook for scaling the biases to match CNN
    Please refer to paper for why this exists
    Goal: Establish equivalence between CNN and QGCN 
    """
    def _backward_hook(self, module, grad_input, grad_output):
        grad_input_index_0_item = grad_input[0]
        grad_input_index_0_item = None if isinstance(grad_input_index_0_item, type(None)) else (grad_input_index_0_item/self.max_kernel_len)
        grad_input_reassigned = (grad_input_index_0_item, *grad_input[1:])
        return grad_input_reassigned


    """
    To remove registered backward hook handlers
    Goal: Establishing equivalence between CNN and QGCN 
    """
    def __del__(self):
        if hasattr(self, "full_backward_hooks_handlers"):
            for handler in self.full_backward_hooks_handlers:
                handler.remove()


    """
    Initializes the dictionaries containing the initialization values from the initializer model
    Goal: To establish equivalence between CNN and QGCN
    """
    # Helper function: exposed to parent model to initialize weights&biases to values of initializer model
    def set_initializer_params(self, layer_index=-1, initializer_named_params_dict=None):
        # override the initialization of the kernel weights and biases etc.
        if layer_index == -1 or isinstance(initializer_named_params_dict, type(None)):
            return
        # initialize weights based on initializer model's params
        self.layer_index = layer_index
        self.initializer_named_params_dict=initializer_named_params_dict
        self._override_non_conv_layers_initializations()


    """
    Overrides conv layer initialization params
    This is triggered by 'forward()' of this class when kernels need to be initialized
    Goal: To establish equivalence between CNN and QGCN
    """
    def _override_conv_layer_initialization(self, kernel_idx):
        # Early exit if no initializer model exists
        if isinstance(self.initializer_named_params_dict, type(None)):
            return
        # override initializations
        conv2d_weight_key = f'conv_layers.{self.layer_index}.conv2d.weight'
        conv2d_weight_key_exists = conv2d_weight_key in self.initializer_named_params_dict
        conv2d_bias_key = f'conv_layers.{self.layer_index}.conv2d.bias'
        conv2d_bias_key_exists = conv2d_bias_key in self.initializer_named_params_dict
        if conv2d_weight_key_exists:
            conv2d_weight_param = self.initializer_named_params_dict[conv2d_weight_key]
            M, N, Kh, Kw = conv2d_weight_param.shape
        if conv2d_bias_key_exists:
            conv2d_bias_param = self.initializer_named_params_dict[conv2d_bias_key]
        # initialize the specific kernel weight in stack of weights
        # loop through the kernels and initialize
        for i in range(Kh):
            found = False
            for j in range(Kw):
                if (i * Kh + j) == kernel_idx:
                    kernel = self.kernels[kernel_idx]
                    mlp_weight_shape = kernel.weight.shape
                    if conv2d_weight_key_exists:
                        kernel.weight.data = conv2d_weight_param.clone().detach().data[:, :, i, j].reshape(mlp_weight_shape)
                    if conv2d_bias_key_exists:
                        kernel.bias.data   = conv2d_bias_param.clone().detach().data * ( 1/(Kh * Kw) )
                    found = True
            # exit once kernel is located and updated
            if found:
                break 
        # initialize bias tensor of the specific QGCN kernel
        # NB: kernel will refer to the kernel found above
        if found and conv2d_bias_key_exists:
            kernel.bias.data   = conv2d_bias_param.clone().detach().data * ( 1/(Kh * Kw) )


    """
    Overrides non conv layer initialization params
    This is triggered by 'set_initializer_params' from parent model of this class
    Goal: To establish equivalence between CNN and QGCN
    """
    def _override_non_conv_layers_initializations(self):
        bn2d_weight_key = f'conv_layers.{self.layer_index}.bn2d.weight'
        bn2d_weight_key_exists = bn2d_weight_key in self.initializer_named_params_dict
        bn2d_bias_key = f'conv_layers.{self.layer_index}.bn2d.bias'
        bn2d_bias_key_exists = bn2d_bias_key in self.initializer_named_params_dict
        if bn2d_weight_key_exists:
            bn2d_weight_param = self.initializer_named_params_dict[bn2d_weight_key]
        if bn2d_bias_key_exists:
            bn2d_bias_param = self.initializer_named_params_dict[bn2d_bias_key]
        # initialize the bn2d weight and biases
        if bn2d_weight_key_exists:
            if hasattr(self, "bn2d"):
                self.bn2d.weight.data = bn2d_weight_param.clone().detach().data
            if hasattr(self, "bn1d"):
                self.bn1d.weight.data = bn2d_weight_param.clone().detach().data
        if bn2d_bias_key_exists:
            if hasattr(self, "bn2d"):
                self.bn2d.bias.data   = bn2d_bias_param.clone().detach().data
            if hasattr(self, "bn1d"):
                self.bn1d.bias.data   = bn2d_bias_param.clone().detach().data


    """
    Retrieves the angular distance in degrees for some point (x,y) wrt origin (0,0)
    Return value is always in [0, 360] in degrees
    """
    def get_point_angle(self, x, y):
        res_in_radians = math.atan2(y,x)
        if (res_in_radians < 0):
            res_in_radians += 2 * math.pi
        res_in_deg = math.degrees(res_in_radians)
        if res_in_deg < 0:
            res_in_deg + 360
        return res_in_deg


    """
    Retrieves kernel properties, i.e., mapping of kernels to angular ranges 
    and the filter masks for the kernels. This is a computationally expensive op.
    """
    def get_kernel_properties(self, edge_index, pos, impose_module_max_kernel_len=False):
        nodes_to_degree_mapping = {}
        target_nodes_index_tracker = {}
        max_node_degree = 0
        self_loop_exists = False
        for i in range(0, len(edge_index[0])):
            source_node = edge_index[1][i].item()            
            target_node = edge_index[0][i].item()
            if not self_loop_exists  and (source_node - target_node) == 0:
                self_loop_exists = True
            if target_node not in nodes_to_degree_mapping:
                target_nodes_index_tracker[target_node] = { "curr_index": 0 }
                nodes_to_degree_mapping[target_node] = { source_node: 0 }
            else:
                target_nodes_index_tracker[target_node]["curr_index"] += 1
                nodes_to_degree_mapping[target_node][source_node] = target_nodes_index_tracker[target_node]["curr_index"]
            # update max node degree
            if (target_nodes_index_tracker[target_node]["curr_index"] + 1) > max_node_degree:
                max_node_degree = target_nodes_index_tracker[target_node]["curr_index"] + 1

        # compute the relative direction of each source node to target node
        target_to_source_node_dir_mapping = {}
        for target_node, target_node_config in nodes_to_degree_mapping.items():
            target_node_pos = pos[target_node]
            target_to_source_node_dir_mapping[target_node] = {}
            for source_node in target_node_config.keys():
                source_node_pos = pos[source_node]
                if target_node != source_node:
                    # get relative position of source node to target node
                    rel_pos = (source_node_pos - target_node_pos) # comes as (y, x)
                    # adjust the signs of y components so angles/directions are intuitive
                    rel_pos[0] = -rel_pos[0]
                    # get the angle/dir of the source node relative to target node
                    source_dir = self.get_point_angle(rel_pos[1], rel_pos[0]) # function takes (x, y)
                    target_to_source_node_dir_mapping[target_node][source_node] = source_dir

        # function to assist with assigning source nodes with kernel index
        def assign_target_source_nodes_to_kernel_idx(target_node, source_dir_map, kernel_angle_range, target_source_kernel_idx_mapping, flag_dup=True):
            kernel_ids_assigned = set()
            for source_node, source_dir in source_dir_map.items():
                failed = False
                for kernel_idx, angle_range in kernel_angle_range.items():
                    if angle_range[0] <= source_dir < angle_range[1]:
                        if flag_dup and kernel_idx in kernel_ids_assigned:
                            failed = True
                            break
                        target_source_kernel_idx_mapping[target_node][source_node] = kernel_idx
                        kernel_ids_assigned.add(kernel_idx)
                if failed:
                    return False, target_source_kernel_idx_mapping
            return True, target_source_kernel_idx_mapping

        # function to assign kernels to source nodes
        def assign_kernel_idx_to_all_source_nodes(target_to_source_node_dir_mapping, max_angular_split, flag_dup=True):
            max_angle = (360 / max_angular_split) if max_angular_split >= 1 else 360
            kernel_angle_range = { i: [i*max_angle, (i+1)*max_angle] for i in range(max_angular_split)}
            target_source_kernel_idx_mapping = {}
            mapping_successful = True
            for target_node, source_dir_map in target_to_source_node_dir_mapping.items():
                target_source_kernel_idx_mapping[target_node] = {}
                # get the source direction and bucketize
                status, updated_map = assign_target_source_nodes_to_kernel_idx(target_node, source_dir_map, kernel_angle_range, target_source_kernel_idx_mapping, flag_dup=flag_dup)
                if status:
                    target_source_kernel_idx_mapping = updated_map
                else:
                    mapping_successful = False
                    break
            return mapping_successful, kernel_angle_range, target_source_kernel_idx_mapping

        # Create mapping of kernel weight layer to angle range beginning with max_node_degree and finding 
        # the minimum degree such that in each bucket kernel bucket, we have unique members
        # It also checks to see if user imposed a fixed number of kernels
        if self.user_defined_kernel_len != -1:
            # Get the target node to source node kernel bin mapping
            max_angular_split = (self.user_defined_kernel_len - 1) if self_loop_exists else self.user_defined_kernel_len # add self loops for assigning kernels - var defines how to split cartesian space
            _, _, target_source_kernel_idx_mapping = assign_kernel_idx_to_all_source_nodes(target_to_source_node_dir_mapping, max_angular_split=self.user_defined_kernel_len, flag_dup=False)
        else:
            # Takes precedence over dynamic assignment of max kernel length
            if impose_module_max_kernel_len:
                # Get the target node to source node kernel bin mapping
                max_angular_split = (self.max_kernel_len - 1) if self_loop_exists else self.max_kernel_len # add self loops for assigning kernels - var defines how to split cartesian space
                _, _, target_source_kernel_idx_mapping = assign_kernel_idx_to_all_source_nodes(target_to_source_node_dir_mapping, max_angular_split=self.user_defined_kernel_len, flag_dup=False)
            else:
                # Naive algorithm to find min kernel length/bin size such that in every neighborhood
                # each source node is mapped to one specific kernel bin
                max_angular_split = (max_node_degree - 1) if self_loop_exists else max_node_degree # add self loops for assigning kernels
                max_angular_split = 1 if max_angular_split <= 0 else max_angular_split # minimum is 1 splits
                upper_bound_angular_split_divs = self.upper_bound_kernel_len # upper bound for bins
                while True:
                    mapping_successful, kernel_angle_range, target_source_kernel_idx_mapping = assign_kernel_idx_to_all_source_nodes(target_to_source_node_dir_mapping, max_angular_split, flag_dup=True)
                    # exit condition
                    if mapping_successful or max_angular_split == upper_bound_angular_split_divs:
                        break
                    max_angular_split += 1
                # If mapping was unsuccessful, assign mapping with max degree 9 as default behavior)
                if not mapping_successful:
                    max_angular_split = (self.upper_bound_kernel_len - 1) if self_loop_exists else self.upper_bound_kernel_len # add self loops for assigning kernels
                    _, _, target_source_kernel_idx_mapping = assign_kernel_idx_to_all_source_nodes(target_to_source_node_dir_mapping, max_angular_split, flag_dup=False)

        # Create the new tensor for kernel weight index masking and return that
        edge_index_source_masking = torch.zeros_like(edge_index[1])
        for i in range(len(edge_index[1])):
            source_node = edge_index[1][i].item()           
            target_node = edge_index[0][i].item()
            if source_node == target_node:
                # max_angular_split will map to the next kernel idx
                edge_index_source_masking[i] = max_angular_split
            else:
                edge_index_source_masking[i] = target_source_kernel_idx_mapping[target_node][source_node]
        
        # Get the max kernel length and return that
        max_kernel_len = edge_index_source_masking.max().item() + 1
        
        # return the results
        return max_kernel_len, edge_index_source_masking


    """
    Handles initializing kernels dynamically
    Notice the below:
    1. Function is expected to be called inside forward() function
    2. If dataset is homogenous -> spatial features &/ node neighborhoods adapt
        Then this function will only be called once for the 1st epoch and never again
        - This is because we cache the masks for the kernels
    3. If dataset is heterogenous -> spatial features &/ node neighborhoods adapt
        Then in each forward pass, we have to potentially generate new masks even for graphs with fixed node degrees
        Algorithm currently is very naive so this will incur a huge loss in performance during training
    """
    def handle_kernel_weights_and_masks_update(self, edge_index, pos):
        # Set control variables
        no_kernel_mapping_exists = len(self.kernel_weight_mask_map) == 0
        larger_graph_subset_available = not no_kernel_mapping_exists and len(edge_index[0]) > len(self.kernel_weight_mask_map[list(self.kernel_weight_mask_map.keys())[0]])
        # print("no_kernel_mapping_exists or larger_graph_subset_available or (not self.is_dataset_homogenous): ", no_kernel_mapping_exists or larger_graph_subset_available or (not self.is_dataset_homogenous))
        if no_kernel_mapping_exists or larger_graph_subset_available or (not self.is_dataset_homogenous):
            # search for what the optimal kernel length is for this batch of data
            # Function below takes into account whether user specified a fixed kernel length
            max_kernel_len, edge_index_mask_source = self.get_kernel_properties(edge_index, pos)
            if max_kernel_len >= self.max_kernel_len:
                self.max_kernel_len = max_kernel_len
                self.edge_index_mask_source = edge_index_mask_source
                # Update kernel masks
                for kernel_idx in range(self.max_kernel_len):
                    self.kernel_weight_mask_map[kernel_idx] = self.edge_index_mask_source == kernel_idx # Create the mask map entry
            else:
                # Update the kernel to edge_index mapping
                _, self.edge_index_mask_source = self.get_kernel_properties(edge_index, pos, impose_module_max_kernel_len=True)
                for kernel_idx in range(self.max_kernel_len):
                    self.kernel_weight_mask_map[kernel_idx] = self.edge_index_mask_source == kernel_idx # Create the mask map entry

    
    """
    Assists with initializing new MLPs/weights in our adaptable kernel for the convolution
    """
    def init_kernel_weight_layer(self, kernel_idx):
        lin_layer = torch.nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=self.use_bias)
        bh_handler = lin_layer.register_full_backward_hook(self._backward_hook, prepend=True)
        self.full_backward_hooks_handlers.append(bh_handler)
        self.kernels.append(lin_layer.to(self.device))
        self._override_conv_layer_initialization(kernel_idx=kernel_idx)
        
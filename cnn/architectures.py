import torch
from torch.nn import Dropout
import torch.nn.functional as F
from torch import nn

class GraphEquivCNNSubModule(torch.nn.Module):
  def __init__(self, in_channels=3, out_channels=1, use_batchNorm=True,
               use_learnable_batchNorm=True, use_maxPool=True, use_bias=True,
               kernel_size=3, stride=1, padding=1, dropout=0):
    super(GraphEquivCNNSubModule, self).__init__()
    self.use_batchNorm = use_batchNorm
    self.use_maxPool = use_maxPool
    self.use_bias = use_bias
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.use_learnable_batchNorm = use_learnable_batchNorm

    # define a similar "[spatial]graphconv" layer
    # define the individual layers -> conv2d->relu->dropout
    self.conv2d = torch.nn.Conv2d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, 
                                  padding=padding, 
                                  bias=use_bias)
    torch.nn.init.kaiming_uniform_(self.conv2d.weight, mode='fan_in', nonlinearity='relu')

    self.bn2d = torch.nn.BatchNorm2d(out_channels, affine=use_learnable_batchNorm)
    self.maxpool2d = torch.nn.MaxPool2d(kernel_size=kernel_size-1, stride=stride, padding=1) # (size * size) -- MODIFIED
    self.relu2d = torch.nn.ReLU()
    self.dropout2d = torch.nn.Dropout(dropout, inplace=False)
    # build the module list ...
    spatial_graph_conv_layer_equiv = []
    spatial_graph_conv_layer_equiv.append(self.conv2d)
    if self.use_batchNorm:
      spatial_graph_conv_layer_equiv.append(self.bn2d)
    spatial_graph_conv_layer_equiv.append(self.relu2d)
    if use_maxPool:
      spatial_graph_conv_layer_equiv.append(self.maxpool2d)
    # add the dropout ...
    spatial_graph_conv_layer_equiv.append(self.dropout2d)
    # COMMENT ON dropout before maxpool or the other below:
    # https://stackoverflow.com/questions/59634780/correct-order-for-spatialdropout2d-batchnormalization-and-activation-function#:~:text=Dropout%20vs%20MaxPooling%20The%20problem%20of%20using%20a,result%20in%20the%20second%20maximum%2C%20not%20in%20zero.?adlt=strict&toWww=1&redig=D78D46C8EC01428D97D04BF61E8B1C20
    # final modulel list ...
    self.module_list = torch.nn.ModuleList(spatial_graph_conv_layer_equiv)
    
  # return kernel size details
  def get_kernel_size_as_tuple(self):
    if isinstance(self.kernel_size, int):
      return self.kernel_size, self.kernel_size
    elif isinstance(self.kernel_size, list) or isinstance(self.kernel_size, tuple):
      return self.kernel_size[0], self.kernel_size[1]
    else:
      assert False, f"{self.kernel_size} of type {type(self.kernel_size)} not supported in interface"

  def get_in_out_channels(self):
    return self.in_channels, self.out_channels
    
  def forward(self, batch_data):
    # the convolution layers ...
    x = batch_data
    for i, layer in enumerate(self.module_list):
        x = layer(x)
    return x


class CNN(torch.nn.Module):
  def __init__(self, out_dim=10, in_channels=3, out_channels=1, hidden_channels=5,
               use_batchNorm=True, use_learnable_batchNorm=True, use_maxPool=False, use_bias=True, 
               kernel_size=3, stride=1, padding=1, dropout=0, layers_num=1):
      super(CNN, self).__init__()
      assert layers_num >= 1
      if layers_num == 1:
        hidden_channels = out_channels
      self.use_batchNorm = use_batchNorm
      self.use_maxPool = use_maxPool
      self.use_bias = use_bias

      # define a similar "[spatial]graphconv" layer
      # define the individual layers -> conv2d->relu->dropout->max_pooling
      self.conv_layers = [ GraphEquivCNNSubModule(in_channels=in_channels, 
                            out_channels=hidden_channels, 
                            use_batchNorm=use_batchNorm,
                            use_learnable_batchNorm=use_learnable_batchNorm,
                            use_maxPool=use_maxPool, 
                            use_bias=use_bias, 
                            kernel_size=kernel_size,
                            stride=stride, 
                            padding=padding, 
                            dropout=dropout) ] + \
                         [ GraphEquivCNNSubModule(in_channels=hidden_channels, 
                            out_channels=hidden_channels, 
                            use_batchNorm=use_batchNorm,
                            use_learnable_batchNorm=use_learnable_batchNorm,
                            use_maxPool=use_maxPool, 
                            use_bias=use_bias, 
                            kernel_size=kernel_size,
                            stride=stride, 
                            padding=padding, 
                            dropout=dropout) for _ in range(layers_num - 2) ] + \
                         [ GraphEquivCNNSubModule(in_channels=hidden_channels, 
                            out_channels=out_channels,
                            use_batchNorm=use_batchNorm,
                            use_learnable_batchNorm=use_learnable_batchNorm,
                            use_maxPool=use_maxPool, 
                            use_bias=use_bias, 
                            kernel_size=kernel_size,
                            stride=stride, 
                            padding=padding, 
                            dropout=dropout) ]
      # create a module list ...
      self.conv_layers = torch.nn.ModuleList(self.conv_layers)
      # we use global average pooling ...
      self.glob_avg_pooling = torch.nn.AdaptiveAvgPool2d(1)
      # Final FC layer ...
      # print(out_channels, out_dim)
      self.fc = torch.nn.Linear(out_channels, out_dim, bias=use_bias)
      ###################################### END ###############################
        
  # getters for extracting model layers
  def get_model_conv_layers(self):
    return self.conv_layers
  
  def forward(self, batch_data):
    x = batch_data # placeholder ...
    for i, layer in enumerate(self.conv_layers):
      x = layer(x)
    assert not isinstance(x, type(None))
    # x = self.glob_avg_pooling(x)
    x = torch.nn.AdaptiveAvgPool2d(1)(x)
    x = x.squeeze()
    x = self.fc(x)
    return x

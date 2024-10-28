from copy import deepcopy
import os
import sys
import pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

######################################################################
#                                                                    #  
#       Process Raw MNIST Data and Convert to Graph                  #
#                                                                    #
######################################################################
# Thanks to Tomasz (SGCN Paper Author) for sharing code (function: image_to_graph) with us
def image_to_graph(sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Data:
    """
    Convert an image with a missing mask to a graph.

    Args:
        sample: pair of image and its mask.

    Returns:
        A graph corresponding to the input image.
    """
    sample_shape = sample[0].shape
    image_size = sample_shape[1]
    img = sample[0].clone()
    mask = sample[1]
    y = sample[2]
    img[mask == 1] = np.nan
    if len(sample_shape) == 2:
        x = img.view(-1, 1)
        x_train = img.view(image_size, image_size, 1).numpy()
    else:
        n_channels = sample_shape[0]
        x = img.reshape(n_channels, -1).T
        x_train = x.reshape(image_size, image_size, n_channels).numpy()

    pos_grid = np.zeros(shape=(image_size * image_size, 2))

    for i in range(1, image_size + 1):
        pos_grid[image_size * (i - 1): image_size * i, 0] = i
        pos_grid[image_size * (i - 1): image_size * i, 1] = range(1, image_size + 1)

    pos1 = []
    pos1_index = []
    pos1_val_index = []
    pos_array = np.zeros(shape=(image_size, image_size))
    ###################
    ind = 0
    ind_val = -1
    for i in pos_grid:
        if (i[0] < image_size + 1) and (i[1] < image_size + 1):
            ind_val = ind_val + 1
            if not np.any(np.isnan(x_train[int(i[0] - 1), int(i[1] - 1)])):
                pos_array[int(i[0] - 1), int(i[1] - 1)] = int(ind)
                pos1.append(i)
                pos1_index.append(ind)
                pos1_val_index.append(ind_val)
                ind = ind + 1

    pos = torch.tensor(np.array(pos1, dtype=np.float64), dtype=torch.float)
    edge_index = list()

    for i in range(0, image_size):
        for j in range(0, image_size - 1):
            """"
            horizontal edges:
            (0*28, 0*28+1), (0*28+1, 0*28+2), ...(0*28+26, 0*28+27)
            (1*28, 1*28+1) ... (1*28+26,1*28+27)
            (2*28, 2*28+1) ... (2*28+26, 2*28+27)
            ...
            (27*28, 27*28+1) .... (27*28+26, 27*28+27)
            """
            if (not np.any(np.isnan(x_train[i, j])) and not np.any(np.isnan(x_train[i, j + 1]))):
                edge_index.append([pos_array[i, j], pos_array[i, j + 1]])
                edge_index.append([pos_array[i, j + 1], pos_array[i, j]])

            """
            vertical edges:
            (0*28, 1*28), (1*28, 2*28), ...(26*28, 27*28)
            (0*28+1, 1*28+1), (1*28+1, 2*28+1)... (26*28+1, 27*28+1)
            ...
            (0*28+27, 1*28+27), (1*28+27, 2*28+27).. (26*28+27, 27*28+27)
            """
            if (not np.any(np.isnan(x_train[j, i])) and not np.any(np.isnan(x_train[j + 1, i]))):
                edge_index.append([pos_array[j, i], pos_array[j + 1, i]])
                edge_index.append([pos_array[j + 1, i], pos_array[j, i]])

            """
            diagonal edges I:
            (0*28, 1*28+1), (0*28+1, 1*28+2) ... (0*28+26, 1*28+27)
            ...
            (26*28, 27*28+1), (26*28+1, 27*28+2) ... (26*28+26, 27*28+27)

            """
            if i < image_size - 1:
                if (not np.any(np.isnan(x_train[i + 1, j + 1])) and not np.any(np.isnan(x_train[i, j]))):
                    edge_index.append([pos_array[i + 1, j + 1], pos_array[i, j]])
                    edge_index.append([pos_array[i, j], pos_array[i + 1, j + 1]])
            """
            diag II:

            (0*28+1, 1*28), (0*28+2, 1*28+1) .... (0*28+27, 1*28+26)
            (1*28+1, 2*28, 1*28+2, 2*28+1) ... (1*28+27, 2*28+26)
            ...
            (26*28+1, 27*28) ... (26*28+27, 27*28+26)
            """
            if i < image_size - 1:
                if (not np.any(np.isnan(x_train[i + 1, j])) and not np.any(np.isnan(x_train[i, j + 1]))):
                    edge_index.append([pos_array[i + 1, j], pos_array[i, j + 1]])
                    edge_index.append([pos_array[i, j + 1], pos_array[i + 1, j]])

    ##############################
    edge_index = torch.LongTensor(edge_index)

    edge_index = edge_index.t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(pos1))
    ##############################

    data_sample = Data(x=x[pos1_val_index], pos=pos, edge_index=edge_index, y=y)
    return data_sample


def get_geometric_dataset_from_raw_pytorch_custom_dataset(data): # pixel_lb_threshold unused
  geometric_dataset = []
  raw_img_x_data, raw_img_y_data = [], []
  for i, geo_data in enumerate(data):
    # create new copy of data
    new_data_cpy = geo_data.clone()
    geometric_dataset.append(new_data_cpy)
  return raw_img_x_data, raw_img_y_data, geometric_dataset


def get_geometric_dataset_from_raw_pytorch_standard_dataset(data,
                                                            add_self_loops=True, 
                                                            edge_feature_type="average", 
                                                            normalize=True,
                                                            num_samples=1,
                                                            pixel_lb_threshold=0): # pixel_lb_threshold unused
  assert add_self_loops, "Tomasz's code by default includes self-loops: hence add_self_loops == True always"
  assert normalize, "Pytorch MNIST is default normalized, hence normalize == True always"
  lb_threshold = pixel_lb_threshold
  if 1 < lb_threshold <= 255:
    lb_threshold /= 255 # adjust accordingly ...
  geometric_dataset = []
  x_data, y_data = [], []
  for i, (img, label) in enumerate(data):
    if i == num_samples:
      break
    # print("torch img tensor shape: ", img.shape)
    img, mask, label = img, torch.zeros_like(img), torch.Tensor([label]).type(torch.LongTensor)
    # mask[img < lb_threshold] = 1 # FIXME: Disabled this to enforce that GraphData == ImageData for MNIST
    sample = [img, mask, label]
    outData = image_to_graph(sample)
    # add it to our list ...
    geometric_dataset.append(outData)
    x_data.append(img)
    y_data.append(label)
  # return the result ...
  x_data = torch.stack(x_data)
  y_data = torch.stack(y_data).type(torch.LongTensor)
  # print("x_data:", x_data.shape, " y_data:", y_data.shape)
  return x_data, y_data, geometric_dataset


# FOR CUSTOM DATASETS: saves dataset as pickle ...
def cache_raw_pytorch_custom_dataset(train_data, test_data, curr_dir_path="", dataset_name=None):
  assert curr_dir_path != None
  assert dataset_name != None
    
  # create the dataset name ...
  full_path = os.path.join(curr_dir_path, dataset_name + ".pkl")
  if os.path.exists(full_path):
    print("Custom Dataset Processing:")
    print("A file with name: {} already exists".format(dataset_name))
    print("Please provide a unique name or delete that old dataset and rerun")
    quit()
    
  # get the dataset to save ...
  _, _, geometric_gcn_train_dataset  = get_geometric_dataset_from_raw_pytorch_custom_dataset(train_data)
  _, _, geometric_gcn_test_dataset   = get_geometric_dataset_from_raw_pytorch_custom_dataset(test_data)
  _, _, geometric_sgcn_train_dataset = get_geometric_dataset_from_raw_pytorch_custom_dataset(train_data)
  _, _, geometric_sgcn_test_dataset  = get_geometric_dataset_from_raw_pytorch_custom_dataset(test_data)

  # create the object to store ...
  # save data into struct ...
  struct = {
      "raw": {
          "x_train_data": [],
          "y_train_data": [],
          "x_test_data" : [],
          "y_test_data" : []
      },
      "geometric": {
          "gcn_train_data": geometric_gcn_train_dataset,
          "gcn_test_data" : geometric_gcn_test_dataset,
          "sgcn_train_data": geometric_sgcn_train_dataset,
          "sgcn_test_data" : geometric_sgcn_test_dataset,
      }
  }

  # use pickle to save the dataset ...
  with open(full_path, "wb") as f:
    pickle.dump(struct, f)

  # inform user of success ...
  print("Successfully dumped the graph data for {}".format(dataset_name))
    

# FOR STANDARD DATASETS: saves dataset as pickle ...
def cache_raw_pytorch_standard_dataset(train_data, 
                                       test_data,
                                       pixel_lb_threshold=0,
                                       add_self_loops=True,
                                       normalize=True,
                                       edge_feature_type=None, # unused
                                       train_num_samples=None,
                                       test_num_samples=None,
                                       curr_dir_path="",
                                       dataset_name="dataset_selfloops_False_edgeft_average_norm_True"):
  assert curr_dir_path != None
  # create the dataset name ...
  full_path = os.path.join(curr_dir_path, dataset_name + ".pkl")
  if os.path.exists(full_path):
    print("Tomasz code processing")
    print("A file with name: {} already exists".format(dataset_name))
    print("Please provide a unique name or delete that old dataset and rerun")
    quit()
  # get the dataset to save ...
  train_samples_count = len(train_data) if train_num_samples is None else train_num_samples
  x_train_data, y_train_data, geometric_gcn_train_dataset = get_geometric_dataset_from_raw_pytorch_standard_dataset( 
                                                                                          train_data,
                                                                                          normalize=normalize,
                                                                                          add_self_loops=add_self_loops,
                                                                                          edge_feature_type=edge_feature_type,
                                                                                          num_samples=train_samples_count,
                                                                                          pixel_lb_threshold=pixel_lb_threshold )
  geometric_sgcn_train_dataset = deepcopy(geometric_gcn_train_dataset)
  # _, _, geometric_sgcn_train_dataset                      = get_geometric_dataset_from_raw_pytorch_standard_dataset( 
  #                                                                                          train_data,
  #                                                                                          normalize=normalize,
  #                                                                                          add_self_loops=add_self_loops,
  #                                                                                          edge_feature_type=edge_feature_type,
  #                                                                                          num_samples=train_samples_count,
  #                                                                                          pixel_lb_threshold=pixel_lb_threshold )
  
  # get num samples to save ...
  test_samples_count = len(test_data) if test_num_samples is None else test_num_samples
  x_test_data, y_test_data, geometric_gcn_test_dataset   = get_geometric_dataset_from_raw_pytorch_standard_dataset( 
                                                                                         test_data,
                                                                                         normalize=normalize,
                                                                                         add_self_loops=add_self_loops,
                                                                                         edge_feature_type=edge_feature_type,
                                                                                         num_samples=test_samples_count,
                                                                                         pixel_lb_threshold=pixel_lb_threshold )
  geometric_sgcn_test_dataset = deepcopy(geometric_gcn_test_dataset)
  # _, _, geometric_sgcn_test_dataset                      = get_geometric_dataset_from_raw_pytorch_standard_dataset( 
  #                                                                                         test_data,
  #                                                                                         normalize=normalize,
  #                                                                                         add_self_loops=add_self_loops,
  #                                                                                         edge_feature_type=edge_feature_type,
  #                                                                                         num_samples=test_samples_count,
  #                                                                                         pixel_lb_threshold=pixel_lb_threshold )
  # create the object to store ...
  # save data into struct ...
  struct = {
      "raw": {
          "x_train_data": x_train_data.clone(),
          "y_train_data": y_train_data.clone(),
          "x_test_data" : x_test_data.clone(),
          "y_test_data" : y_test_data.clone()
      },
      "geometric": {
          "gcn_train_data": geometric_gcn_train_dataset,
          "gcn_test_data" : geometric_gcn_test_dataset,
          "sgcn_train_data": geometric_sgcn_train_dataset,
          "sgcn_test_data" : geometric_sgcn_test_dataset,
      }
  }

  # use pickle to save the dataset ...
  with open(full_path, "wb") as f:
    pickle.dump(struct, f)

  # inform user of success ...
  print("Successfully dumped the raw and graph data for {}".format(dataset_name))


"""
Primary methods to generate graph representations of standard and custom datasets
"""
def create_raw_pytorch_graph_dataset_from_custom_dataset(dataset_name="navier_stokes", cls_type="BINARY", difficulty="EASY", grouped=True, behavior="INTERPOLATION", cls_group="ALL"):
    """
    behavior: INTERPOLATE / EXTRAPOLATE - determines how to split the train and tes
              INTERPOLATE will draw test samples from the same random set for train
              EXTRAPOLATE will dram test samples from future time steps of the same Re time series data
    cls_group: - only used by denary mode
               ALL - data comes from the dedicated denary dataset
               BINARY - means space out evenly data from both laminar and turbulent to get 10 classes
               LAMINAR   - creates denary dataset from only laminar dataset 
               TURBULENT - creates denary dataset from only turbulent dataset 
    """
    import math
    import torch
    import random
    random.seed(100)
    
    # We use behavior input to enable/disable the shuffling behavior
    allow_shuffling = False
    if (behavior == "INTERPOLATION"):
        allow_shuffling = True
    elif (behavior == "EXTRAPOLATION"):
        allow_shuffling = False
    # Other options can be enabled here    
    
    # load the navier data
    if cls_type == "BINARY": # For BINARY data
        data_cls_type_id = cls_type
        points_per_re = 1500
        all_re_points = [20, 120] # Binary classification will use extreme points
        # grouped is indicated, we mix all laminar / turbulent Re data points together into 2 sets
        # In grouped, we ignore Re=[20, 120] for convenience so that the logic inside get_dataset_by_split works nicely :)
        if grouped:
            laminar_re_min, laminar_re_max = 20, 40 # -> total 21 Re numbers [ignore first point Re=20]
            turbulent_re_min, turbulent_re_max = 100, 120 # -> 21 Re numbers [ignore last point Re=120]
            step = 5 if difficulty in ["ALLEASE", "MIDEASE"] else 1
            all_re_points = [*list(range(laminar_re_min, laminar_re_max+1, step)), *list(range(turbulent_re_min, turbulent_re_max+1, step))]
    elif cls_type == "DENARY": # For DENARY data only
        # Use the 'cls_group' to determine where data should come from
        data_cls_type_id = cls_type
        if cls_group == "ALL":
            points_per_re, denary_re_start, denary_re_end = 2000, 30, 120
            all_re_points = list(range(denary_re_start, denary_re_end+1, 10))
        elif cls_group == "LAMINAR":
            data_cls_type_id = "BINARY"
            points_per_re, laminar_re_min, laminar_re_max = 1500, 20, 40 # -> total 21 Re numbers [ignore last point Re=40]
            all_re_points = list(range(laminar_re_min, laminar_re_max, 2)) # [ignore last point Re=40]
            re_to_class_id_mapping = { re: idx for idx, re in enumerate(all_re_points) }
        elif cls_group == "TURBULENT":
            data_cls_type_id = "BINARY"
            points_per_re, turbulent_re_min, turbulent_re_max = 1500, 100, 120 # -> 21 Re numbers [ignore last point Re=120]
            all_re_points = list(range(turbulent_re_min, turbulent_re_max, 2)) # [ignore last point Re=120]
            re_to_class_id_mapping = { re: idx for idx, re in enumerate(all_re_points) }
        elif cls_group == "BINARY":
            data_cls_type_id = "BINARY"
            points_per_re = 2000 if difficulty in ["MIDEASE", "MIDEASER"] else 1500
            laminar_re_min, laminar_re_max = 20, 40 # -> total 21 Re numbers [ignore first point Re=20]
            turbulent_re_min, turbulent_re_max = 100, 120 # -> 21 Re numbers [ignore last point Re=120]
            all_re_points = [*list(range(laminar_re_min, laminar_re_max+1, 5)), *list(range(turbulent_re_min, turbulent_re_max+1, 5))]
            re_to_class_id_mapping = { re: idx for idx, re in enumerate(all_re_points) }
        else:
            assert False, f"cls_group: {cls_group}, not supported :("
        
    # define full path to data dir
    dataset_full_path_to_dir = f"/athena/grosenicklab/scratch/iso4003/Weill-Cornell-GCN/Dataset/Graph/Navier_Stokes/{data_cls_type_id.capitalize()}"
  
    # Retrieve the data set of interest
    def get_dataset_by_split(num_train=None, num_test=None, num_train_sf=0.9):
        train_dataset, test_dataset = [], []
        num_train_per_re = (num_train // len(all_re_points)) if num_train != None else num_train
        num_test_per_re  = (num_test // len(all_re_points)) if num_test != None else num_test
        if (num_train_per_re == 0) or (num_test_per_re == 0):
            return False, [], []
        # loop through and get dataset for training
        for re in all_re_points:
            diff_id = "EASY"
            if difficulty in ["MIDEASE", "MIDEASER"]:
                diff_id = "EASY"
            elif difficulty in ["ALLEASE", "ALLEASER"]:
                diff_id = "ALL"
            dataset_name = f"navier_stokes_{data_cls_type_id.lower()}_classification_Re_{re}_nppre_{points_per_re}_diff_{diff_id.lower()}.pt"
            full_path_name = f"{dataset_full_path_to_dir}/{dataset_name}"
            print(f"Accessing data: {dataset_name}")
            dataset = torch.load(full_path_name)
            dataset_points = dataset[re]
            if allow_shuffling: # Predictive dataset doesn't shuffle; refer to function parameters
                random.shuffle(dataset_points)
            train_data_len, test_data_len = num_train_per_re, num_test_per_re
            if isinstance(num_train, type(None)) or isinstance(num_test, type(None)):
                train_data_len = int( num_train_sf * len(dataset_points) )
                test_data_len = int( (1 - num_train_sf) * len(dataset_points) )
            # assert that the requested data can be serviced
            if (train_data_len + test_data_len) > len(dataset_points):
                return False, [], []
            # randomly shuffle indices for test set
            test_set_idx = [i for i in range(train_data_len, len(dataset_points)+1)][:test_data_len]
            random.shuffle(test_set_idx) # randmly shuffled idx
            test_set_idx = test_set_idx[:test_data_len] # select only required num of points
            # partition data according to split
            for i in range(len(dataset_points)):
                # Update class labelling if data source is not consistent with cls_type
                if cls_type == "DENARY" and data_cls_type_id == "BINARY": # update the class labelling
                    dataset_points[i].y = torch.Tensor([ re_to_class_id_mapping[re] ]).type(torch.LongTensor)
                # Add datapoint to the right set
                if i < train_data_len:
                    train_dataset.append(dataset_points[i])
                elif train_data_len <= i < (train_data_len + test_data_len):
                    # depending on difificulty level, pull data from different regions
                    test_data_idx = (train_data_len - i) - 1 # Corresponds to diff = "HARD"
                    if difficulty in ["MIDEASER", "ALLEASER"]:
                        test_data_idx = i
                    elif difficulty in ["MIDEASE", "ALLEASE"]:
                        test_data_idx = test_set_idx[ i - train_data_len ]
                    elif difficulty == "EASY":
                        test_data_idx = i
                    elif difficulty == "MEDIUM":
                        test_data_idx = i + ( (len(dataset_points) - train_data_len - test_data_len) // 2 )
                    elif difficulty == "HARD":
                        test_data_idx = (train_data_len - i) - 1
                    # add test data
                    test_dataset.append(dataset_points[test_data_idx])
                else: # We don't need any more data so break
                    break
        # Shuffle the training and test sets
        # This shuffling is fine - should hurt any of the dataset modes
        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
        # return results
        return True, train_dataset, test_dataset
        
    # define buckets -> NB: total datasets
    if cls_type == "DENARY":
        train_num_samples = [10, 100, 200, 1000, 10000, 20000, None]# None defaults to full sample set ...
        test_num_samples  = [2, 20, 40, 200, 1000, 2000, None] # None defaults to full sample set ...
    elif cls_type == "BINARY":
        train_num_samples = [100, 1000, 10000, 20000, None]# None defaults to full sample set ...
        test_num_samples  = [20, 200, 1000, 5000, None] # None defaults to full sample set ...
    
    # loop and create datasets ...
    for num_train, num_test in zip(train_num_samples, test_num_samples):
        valid_data_split_requested, train_dataset, test_dataset = get_dataset_by_split(num_train=num_train, num_test=num_test)
        if not valid_data_split_requested:
            continue
        # Assert request was serviced properly
        if num_train != None:
            assert len(train_dataset) == num_train, "Requested datasize not honored for training data"
        else:
            num_train = len(train_dataset)
        # Assert request was serviced properly
        if num_test != None:
            assert len(test_dataset) == num_test, "Requested datasize not honored for testing data"
        else:
            num_test = len(test_dataset)
        # call helper function to package data into the struct/dict that Experiment.py requires
        print("generating data for: train=",num_train, " and test=", num_test)
        dir_name = "train_{}_test_{}".format(num_train, num_test)
        full_path = os.path.join(root_dir, "Dataset", "RawGraph", dir_name)
        if not os.path.exists(full_path):
          os.mkdir(full_path)
        # generate dataset name
        grouped_key = "_grouped" if grouped else ""
        dataset_name_prefix = f"navier_stokes_{cls_type.lower()}{grouped_key}_{behavior.lower()}_dataset"
        dataset_name_postfix = f"group_{cls_group.lower()}_diff_{difficulty.lower()}_selfloops_False_edgeft_None_norm_True"
        dataset_name = f"{dataset_name_prefix}_{dataset_name_postfix}"
        # create and serialize the dataset ...
        cache_raw_pytorch_custom_dataset(train_dataset, test_dataset, curr_dir_path=full_path, dataset_name=dataset_name)

def create_raw_pytorch_graph_dataset_from_standard_dataset(dataset_name = "mnist"):
  from torchvision import transforms
  from torchvision.datasets import MNIST, CIFAR10, CIFAR100, Flowers102, FashionMNIST

  # load the raw mnist train and test dataset
  if dataset_name == "mnist":
    # 1, 28, 28 -> 10 classes
    train_dataset = MNIST(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=True, transform=transforms.ToTensor())
    test_dataset  = MNIST(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=False, transform=transforms.ToTensor())
  elif dataset_name == "fashionmnist":
    # 1, 28, 28 -> 10 classes
    train_dataset = FashionMNIST(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=True, transform=transforms.ToTensor())
    test_dataset  = FashionMNIST(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=False, transform=transforms.ToTensor())
  elif dataset_name == "cifar10":
    # 3, 32, 32 -> 10 classes
    train_dataset = CIFAR10(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=True, transform=transforms.ToTensor())
    test_dataset  = CIFAR10(os.path.join(root_dir, "Dataset", "Raw"), download=True, train=False, transform=transforms.ToTensor())

  assert not isinstance(train_dataset, type(None))
  assert not isinstance(test_dataset, type(None))
  
  # load meta data for experiments ...
  interested_datasets_meta = [{
                                 "add_self_loops": True,
                                 "edge_feature_type": None,
                                 "normalize": True,
                                 "dataset_name": f"{dataset_name}_dataset_selfloops_True_edgeft_None_norm_True"
                             }]
  # loop through options and create datasets ...
  # train_num_samples = [10, 100, 1000, 2000, 5000, 10000, None] # None defaults to full sample set ...
  # test_num_samples = [2, 20, 200, 500, 1000, 1000, None] # None defaults to full sample set ...
  train_num_samples = [len(train_dataset)]
  test_num_samples = [len(test_dataset)]
  # loop and create datasets ...
  for num_train, num_test in zip(train_num_samples, test_num_samples):
    # create the directory here if doesn't exist ...
    print("generating data for: train=",num_train, " and test=", num_test)
    dir_name = "train_{}_test_{}".format(num_train, num_test)
    assert dir_name != ""
    dataset_partial_path = os.path.join(root_dir, "Dataset")
    if not os.path.exists(dataset_partial_path):
      os.mkdir(dataset_partial_path)
    dataset_rawgraph_partial_path = os.path.join(dataset_partial_path, "RawGraph")
    if not os.path.exists(dataset_rawgraph_partial_path):
      os.mkdir(dataset_rawgraph_partial_path)
    full_path = os.path.join(dataset_rawgraph_partial_path, dir_name)
    if not os.path.exists(full_path):
      os.mkdir(full_path)
    # create and serialize the dataset ...
    for meta_data in interested_datasets_meta:
      # cache the python3 dataset ...
      cache_raw_pytorch_standard_dataset(train_dataset,
                                         test_dataset,
                                         pixel_lb_threshold=0,
                                         add_self_loops=meta_data["add_self_loops"],
                                         normalize=meta_data["normalize"],
                                         edge_feature_type=meta_data["edge_feature_type"],
                                         train_num_samples=num_train,
                                         test_num_samples=num_test,
                                         curr_dir_path=full_path,
                                         dataset_name=meta_data["dataset_name"])


import argparse
parser = argparse.ArgumentParser(description='Generate Graph Data')
parser.add_argument('-d', '--dataset_name', required=True, help='PyTorch Dataset Name required for data generation')
parser.add_argument('-c', '--cls_type',     required=False, default="BINARY", help='Classification problem type')
parser.add_argument('-s', '--difficulty',   required=False, default="DENARY", help='Dataset Difficulty level')
parser.add_argument('-g', '--grouped',      required=False, action='store_true', help='Dataset Difficulty level')
parser.add_argument('-b', '--behavior',     required=False, default="INTERPOLATION", help='Predict test interpolations or new time steps')
parser.add_argument('-cg', '--cls_group',     required=False, default="ALL", help='Type of data: LAMINAR/TURBULENT/ALL')

args = parser.parse_args()
dataset_name = args.dataset_name.lower()
cls_type = args.cls_type
difficulty = args.difficulty
grouped = args.grouped
behavior = args.behavior
cls_group = args.cls_group

# Add parent directory to sys.path
def get_root_dir(base: str = "."):
    if any([ os.path.isdir(os.path.join(base, child_dir)) and child_dir == "qgrn" for child_dir in os.listdir(base) ]):
        return base
    return get_root_dir(base=str(Path(base).parent.resolve()))

# Get and add root directory to script
root_dir = get_root_dir(); sys.path.extend([root_dir])

# Create the Graph Dataset
if dataset_name in ["mnist", "fashionmnist", "cifar10"]:
    print(f"Generating Graph Data for PyTorch Standard Dataset={dataset_name}")
    create_raw_pytorch_graph_dataset_from_standard_dataset(dataset_name=dataset_name)
elif dataset_name in ["navier_stokes"]:
    print(f"Generating Graph Data for PyTorch Custom Dataset={dataset_name}, classification_type={cls_type}, difficulty={difficulty}, grouped={grouped}, behavior={behavior}, cls_group={cls_group}")
    create_raw_pytorch_graph_dataset_from_custom_dataset(dataset_name=dataset_name, cls_type=cls_type, difficulty=difficulty, grouped=grouped, behavior=behavior, cls_group=cls_group)
